# Data downloaded from: https://damit.cuni.cz/projects/damit/exports

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import trimesh

from hac.constants import POSITION_IDS
from hac.mesh import normalize_mesh_to_z_bounds
from hac.simulate import generate_lightcurves_from_mesh


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HPC_ROOT = Path("/work3/projects/HAC-DTU")

DAMIT_PATH = Path(
    os.getenv(
        "DAMIT_PATH",
        HPC_ROOT / "damit-20260809T000301Z",
    )
)

SIMULATED_DATA_PATH = Path(
    os.getenv(
        "SIMULATED_DATA_PATH",
        HPC_ROOT / "simulated_data",
    )
)


# ---------------------------------------------------------------------------
# DAMIT shape loading
# ---------------------------------------------------------------------------


def load_damit_shape(
    path: str | Path,
) -> trimesh.Trimesh:
    """
    Load a DAMIT shape.txt file as a trimesh.Trimesh.

    DAMIT shape format:

        n_vertices n_faces
        x y z
        x y z
        ...
        i j k
        i j k
        ...

    The first ``n_vertices`` rows after the header contain vertex
    coordinates. The following ``n_faces`` rows contain triangular
    face indices.

    DAMIT face indices are 1-based, while NumPy/trimesh face indices
    are 0-based.

    Parameters
    ----------
    path
        Path to a DAMIT shape.txt file.

    Returns
    -------
    trimesh.Trimesh
        Raw DAMIT mesh without geometry processing or normalization.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"DAMIT shape file does not exist: {path}"
        )

    with path.open("r") as f:
        header = f.readline().split()

    if len(header) != 2:
        raise ValueError(
            f"Invalid DAMIT shape header in {path}: "
            f"expected 'n_vertices n_faces', got {header}"
        )

    n_vertices, n_faces = map(int, header)

    # Every row after the header contains exactly three numbers, so the
    # vertex and face sections can be read together in one operation.
    data = np.loadtxt(
        path,
        skiprows=1,
    )

    expected_rows = n_vertices + n_faces

    if data.shape != (expected_rows, 3):
        raise ValueError(
            f"Invalid DAMIT shape file {path}: "
            f"header declares {n_vertices} vertices and {n_faces} faces "
            f"({expected_rows} rows), but loaded array has shape {data.shape}"
        )

    vertices = np.asarray(
        data[:n_vertices],
        dtype=np.float64,
    )

    faces_raw = data[n_vertices:]

    # np.loadtxt reads the whole file body as floating point because the
    # vertex rows contain floats. Verify that the face section contains
    # exact integer values before converting it.
    if not np.all(
        faces_raw == np.floor(faces_raw)
    ):
        raise ValueError(
            f"Non-integer face indices found in DAMIT shape file: {path}"
        )

    faces = np.asarray(
        faces_raw,
        dtype=np.int64,
    )

    # DAMIT uses 1-based vertex numbering.
    faces -= 1

    if faces.size:
        if faces.min() < 0:
            raise ValueError(
                f"Invalid face index below 1 in DAMIT shape file: {path}"
            )

        if faces.max() >= n_vertices:
            raise ValueError(
                f"Face references vertex {faces.max() + 1}, "
                f"but {path} only contains {n_vertices} vertices"
            )

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DamitDataset:
    """
    DAMIT asteroid shape dataset with generated HAC-style lightcurves.

    Original DAMIT data and locally generated simulation data are kept
    separate.

    Original data::

        <DAMIT_PATH>/
            files/
                asteroid_<asteroid_id>/
                    model_<model_id>/
                        shape.txt
            tables/
                asteroid_models.csv

    Generated data::

        ./simulated_data/
            damit/
                asteroid_<asteroid_id>/
                    model_<model_id>/
                        cam_<cam_idx>_intensity.npy
                        cam_<cam_idx>_binary.npy

    Only models with quality flags listed in ``QUALITY_FLAGS`` are included.

    Every mesh is normalized isotropically using
    ``normalize_mesh_to_z_bounds`` before simulation, so its z bounds are
    approximately [-1, 1] while its original proportions are preserved.

    Lightcurves are generated for every camera position in ``POSITION_IDS``.

    Accessing an item:

    1. loads the original DAMIT shape.txt;
    2. converts it to a trimesh.Trimesh;
    3. normalizes it isotropically to z in [-1, 1];
    4. loads existing simulated lightcurves if present;
    5. otherwise generates the missing lightcurves;
    6. returns identifiers, normalized mesh, and lightcurves.

    Parameters
    ----------
    root
        Root directory of the extracted DAMIT export.

    simulated_data_root
        Root directory for locally generated simulation data.
    """

    QUALITY_FLAGS = [4.0]

    def __init__(
        self,
        root: str | Path = DAMIT_PATH,
        simulated_data_root: str | Path = SIMULATED_DATA_PATH,
    ):
        self.root = Path(root)

        self.simulated_data_root = (
            Path(simulated_data_root)
            / "damit"
        )

        self.table_path = (
            self.root
            / "tables"
            / "asteroid_models.csv"
        )

        if not self.table_path.is_file():
            raise FileNotFoundError(
                f"DAMIT asteroid model table does not exist: "
                f"{self.table_path}"
            )

        self.models = self._load_model_table()

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------

    def _load_model_table(
        self,
    ) -> pd.DataFrame:
        """
        Load asteroid_models.csv and select the DAMIT models used by
        this dataset.
        """
        df = pd.read_csv(
            self.table_path,
        )

        required_columns = {
            "id",
            "asteroid_id",
            "quality_flag",
        }

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{self.table_path} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        models = df.loc[
            df["quality_flag"].isin(
                self.QUALITY_FLAGS
            )
        ].copy()

        models = models.dropna(
            subset=[
                "id",
                "asteroid_id",
            ]
        )

        models["id"] = models[
            "id"
        ].astype(
            np.int64
        )

        models["asteroid_id"] = models[
            "asteroid_id"
        ].astype(
            np.int64
        )

        return models.reset_index(
            drop=True
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.models
        )

    # ------------------------------------------------------------------
    # Original DAMIT paths
    # ------------------------------------------------------------------

    def get_shape_path(
        self,
        model_id: int,
        asteroid_id: int,
    ) -> Path:
        """
        Return the original DAMIT shape.txt path for a model.
        """
        return (
            self.root
            / "files"
            / f"asteroid_{asteroid_id}"
            / f"model_{model_id}"
            / "shape.txt"
        )

    # ------------------------------------------------------------------
    # Generated-data paths
    # ------------------------------------------------------------------

    def get_simulated_asteroid_dir(
        self,
        asteroid_id: int,
    ) -> Path:
        """
        Return the generated-data directory for one asteroid.
        """
        return (
            self.simulated_data_root
            / f"asteroid_{asteroid_id}"
        )

    def get_simulated_model_dir(
        self,
        asteroid_id: int,
        model_id: int,
    ) -> Path:
        """
        Return the generated-data directory for one asteroid model.
        """
        return (
            self.get_simulated_asteroid_dir(
                asteroid_id
            )
            / f"model_{model_id}"
        )

    def get_lightcurve_paths(
        self,
        asteroid_id: int,
        model_id: int,
        cam_idx: str,
    ) -> dict[str, Path]:
        """
        Return generated lightcurve paths for one asteroid model and camera.
        """
        model_dir = self.get_simulated_model_dir(
            asteroid_id=asteroid_id,
            model_id=model_id,
        )

        return {
            "intensity": (
                model_dir
                / f"cam_{cam_idx}_intensity.npy"
            ),
            "binary": (
                model_dir
                / f"cam_{cam_idx}_binary.npy"
            ),
        }

    # ------------------------------------------------------------------
    # Mesh loading
    # ------------------------------------------------------------------

    def load_raw_mesh(
        self,
        model_id: int,
        asteroid_id: int,
    ) -> trimesh.Trimesh:
        """
        Load the original, unnormalized DAMIT shape mesh.
        """
        shape_path = self.get_shape_path(
            model_id=model_id,
            asteroid_id=asteroid_id,
        )

        return load_damit_shape(
            shape_path
        )

    def load_mesh(
        self,
        model_id: int,
        asteroid_id: int,
    ) -> trimesh.Trimesh:
        """
        Load and normalize one DAMIT shape mesh.

        The normalization is isotropic and maps the mesh's z bounds to
        approximately [-1, 1].
        """
        mesh = self.load_raw_mesh(
            model_id=model_id,
            asteroid_id=asteroid_id,
        )

        return normalize_mesh_to_z_bounds(
            mesh
        )

    # ------------------------------------------------------------------
    # Simulated lightcurves
    # ------------------------------------------------------------------

    def lightcurves_exist(
        self,
        asteroid_id: int,
        model_id: int,
        cam_idx: str,
    ) -> bool:
        """
        Return True if both generated lightcurve files exist for a
        model/camera pair.
        """
        paths = self.get_lightcurve_paths(
            asteroid_id=asteroid_id,
            model_id=model_id,
            cam_idx=cam_idx,
        )

        return all(
            path.is_file()
            for path in paths.values()
        )

    def load_lightcurves(
        self,
        asteroid_id: int,
        model_id: int,
        cam_idx: str,
    ) -> dict[str, np.ndarray]:
        """
        Load previously generated lightcurves for one model/camera pair.
        """
        paths = self.get_lightcurve_paths(
            asteroid_id=asteroid_id,
            model_id=model_id,
            cam_idx=cam_idx,
        )

        if not self.lightcurves_exist(
            asteroid_id=asteroid_id,
            model_id=model_id,
            cam_idx=cam_idx,
        ):
            raise FileNotFoundError(
                f"Simulated lightcurves do not exist for "
                f"DAMIT asteroid {asteroid_id}, "
                f"model {model_id}, "
                f"camera {cam_idx}"
            )

        return {
            "intensity": np.load(
                paths["intensity"]
            ),
            "binary": np.load(
                paths["binary"]
            ),
        }

    def load_or_generate_lightcurves(
        self,
        asteroid_id: int,
        model_id: int,
        cam_idx: str,
        mesh: trimesh.Trimesh,
    ) -> dict[str, np.ndarray]:
        """
        Load generated lightcurves if they exist, otherwise generate them.
        """
        if self.lightcurves_exist(
            asteroid_id=asteroid_id,
            model_id=model_id,
            cam_idx=cam_idx,
        ):
            return self.load_lightcurves(
                asteroid_id=asteroid_id,
                model_id=model_id,
                cam_idx=cam_idx,
            )

        model_dir = self.get_simulated_model_dir(
            asteroid_id=asteroid_id,
            model_id=model_id,
        )

        return generate_lightcurves_from_mesh(
            mesh=mesh,
            cam_idx=cam_idx,
            out_dir=model_dir,
        )

    # ------------------------------------------------------------------
    # Item access
    # ------------------------------------------------------------------

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.models.iloc[
            index
        ]

        model_id = int(
            row["id"]
        )

        asteroid_id = int(
            row["asteroid_id"]
        )

        quality_flag = float(
            row["quality_flag"]
        )

        shape_path = self.get_shape_path(
            model_id=model_id,
            asteroid_id=asteroid_id,
        )

        mesh = self.load_mesh(
            model_id=model_id,
            asteroid_id=asteroid_id,
        )

        lightcurves = {
            cam_idx: self.load_or_generate_lightcurves(
                asteroid_id=asteroid_id,
                model_id=model_id,
                cam_idx=cam_idx,
                mesh=mesh,
            )
            for cam_idx in POSITION_IDS
        }

        return {
            "model_id": model_id,
            "asteroid_id": asteroid_id,
            "quality_flag": quality_flag,
            "shape_path": shape_path,
            "mesh": mesh,
            "lightcurves": lightcurves,
        }

    # ------------------------------------------------------------------
    # Validation / convenience
    # ------------------------------------------------------------------

    def validate_shapes(
        self,
    ) -> pd.DataFrame:
        """
        Return selected DAMIT models whose shape.txt file is missing.

        An empty DataFrame means every selected model has a shape file.
        """
        missing = []

        for row in self.models.itertuples(
            index=False
        ):
            model_id = int(
                row.id
            )

            asteroid_id = int(
                row.asteroid_id
            )

            shape_path = self.get_shape_path(
                model_id=model_id,
                asteroid_id=asteroid_id,
            )

            if not shape_path.is_file():
                missing.append(
                    {
                        "model_id": model_id,
                        "asteroid_id": asteroid_id,
                        "shape_path": shape_path,
                    }
                )

        return pd.DataFrame(
            missing,
            columns=[
                "model_id",
                "asteroid_id",
                "shape_path",
            ],
        )

    def is_generated(
        self,
        index: int,
    ) -> bool:
        """
        Return True if all challenge-position lightcurves for an item
        have already been generated.
        """
        row = self.models.iloc[
            index
        ]

        model_id = int(
            row["id"]
        )

        asteroid_id = int(
            row["asteroid_id"]
        )

        return all(
            self.lightcurves_exist(
                asteroid_id=asteroid_id,
                model_id=model_id,
                cam_idx=cam_idx,
            )
            for cam_idx in POSITION_IDS
        )


if __name__ == "__main__":
    dataset = DamitDataset()

    print(
        f"DAMIT root:          "
        f"{dataset.root}"
    )

    print(
        f"Simulated data root: "
        f"{dataset.simulated_data_root}"
    )

    print(
        f"Quality flags:       "
        f"{dataset.QUALITY_FLAGS}"
    )

    print(
        f"Camera positions:    "
        f"{len(POSITION_IDS)}"
    )

    print(
        f"Number of models:    "
        f"{len(dataset)}"
    )

    missing = dataset.validate_shapes()

    if len(missing):
        print()
        print(
            "Missing shape files:"
        )
        print(
            missing
        )
    else:
        print(
            "All selected shape files exist."
        )

    if len(dataset):
        print()

        print(
            f"First item already generated: "
            f"{dataset.is_generated(0)}"
        )

        print()
        print(
            "Loading first sample..."
        )

        sample = dataset[0]

        print(
            f"Model ID:     "
            f"{sample['model_id']}"
        )

        print(
            f"Asteroid ID:  "
            f"{sample['asteroid_id']}"
        )

        print(
            f"Quality flag: "
            f"{sample['quality_flag']}"
        )

        print(
            f"Shape:        "
            f"{sample['shape_path']}"
        )

        print(
            f"Mesh:         "
            f"{sample['mesh']}"
        )

        print(
            f"Lightcurves:  "
            f"{list(sample['lightcurves'])}"
        )