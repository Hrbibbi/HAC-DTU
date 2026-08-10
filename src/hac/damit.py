# Data downloaded from: https://damit.cuni.cz/projects/damit/exports

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
import trimesh

from hac.constants import POSITION_IDS
from hac.mesh import normalize_mesh_to_z_bounds
from hac.simulate import (
    generate_lightcurves_from_mesh,
    simulate_rotation_frames,
)


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

    n_vertices, n_faces = map(
        int,
        header,
    )

    data = np.loadtxt(
        path,
        skiprows=1,
    )

    expected_rows = (
        n_vertices
        + n_faces
    )

    if data.shape != (
        expected_rows,
        3,
    ):
        raise ValueError(
            f"Invalid DAMIT shape file {path}: "
            f"header declares {n_vertices} vertices and {n_faces} faces "
            f"({expected_rows} rows), but loaded array has shape {data.shape}"
        )

    vertices = np.asarray(
        data[:n_vertices],
        dtype=np.float64,
    )

    faces_raw = data[
        n_vertices:
    ]

    if not np.all(
        faces_raw
        == np.floor(faces_raw)
    ):
        raise ValueError(
            f"Non-integer face indices found in DAMIT shape file: {path}"
        )

    faces = np.asarray(
        faces_raw,
        dtype=np.int64,
    )

    # DAMIT uses 1-based indexing.
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

    Original DAMIT data::

        <DAMIT_PATH>/
            files/
                asteroid_<asteroid_id>/
                    model_<model_id>/
                        shape.txt
            tables/
                asteroid_models.csv

    Generated data::

        <SIMULATED_DATA_PATH>/
            damit/
                asteroid_<asteroid_id>/
                    model_<model_id>/
                        cam_<cam_idx>_intensity.npy
                        cam_<cam_idx>_binary.npy

    Every mesh is normalized isotropically to z in [-1, 1].

    One camera distance is calculated for each normalized shape and reused
    for every camera position and every rotation angle.

    The camera distance is based on an origin-centered bounding sphere,
    making it independent of viewing direction.
    """

    QUALITY_FLAGS = [4.0]

    # Fraction of the full image kept empty on EACH edge.
    #
    # 0.05 means:
    #
    #     5% margin | 90% available image | 5% margin
    #
    # The bounding sphere therefore occupies at most the central 90%
    # of the image dimension.
    CAMERA_MARGIN = 0.10

    def __init__(
        self,
        root: str | Path = DAMIT_PATH,
        simulated_data_root: str | Path = SIMULATED_DATA_PATH,
    ):
        self.root = Path(
            root
        )

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

        self.models = (
            self._load_model_table()
        )

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
            self.table_path
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

        models["id"] = (
            models["id"]
            .astype(np.int64)
        )

        models["asteroid_id"] = (
            models["asteroid_id"]
            .astype(np.int64)
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
        model_dir = (
            self.get_simulated_model_dir(
                asteroid_id=asteroid_id,
                model_id=model_id,
            )
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
        shape_path = (
            self.get_shape_path(
                model_id=model_id,
                asteroid_id=asteroid_id,
            )
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

        Normalization is isotropic and maps the z bounds to [-1, 1].
        """
        mesh = self.load_raw_mesh(
            model_id=model_id,
            asteroid_id=asteroid_id,
        )

        return normalize_mesh_to_z_bounds(
            mesh
        )

    # ------------------------------------------------------------------
    # Camera distance
    # ------------------------------------------------------------------

    def get_camera_distance(
        self,
        mesh: trimesh.Trimesh,
        image_size: tuple[int, int] = (1024, 1024),
    ) -> float:
        """
        Calculate one camera distance for a normalized DAMIT shape.

        The same distance is used for every camera position and rotation.

        The mesh is enclosed in an origin-centered bounding sphere. Since
        the projected size of a sphere is independent of viewing direction,
        this gives a conservative camera distance without testing every
        camera and rotation.

        The PyVista/VTK camera's existing field of view is used unchanged.

        Parameters
        ----------
        mesh
            Normalized DAMIT mesh.

        image_size
            Rendering size as ``(width, height)``. This only matters for
            non-square images because the limiting horizontal/vertical
            field of view may differ.

        Returns
        -------
        float
            Camera distance from the origin.
        """
        if not 0 <= self.CAMERA_MARGIN < 0.5:
            raise ValueError(
                "CAMERA_MARGIN must satisfy "
                "0 <= CAMERA_MARGIN < 0.5"
            )

        vertices = np.asarray(
            mesh.vertices,
            dtype=float,
        )

        if len(vertices) == 0:
            raise ValueError(
                "Cannot determine camera distance for an empty mesh"
            )

        # Radius of an origin-centered sphere containing the whole mesh.
        radius = float(
            np.linalg.norm(
                vertices,
                axis=1,
            ).max()
        )

        width, height = image_size

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid image_size: {image_size}"
            )

        aspect = (
            width
            / height
        )

        # Read the normal VTK/PyVista perspective FOV.
        #
        # We do not modify it.
        camera = pv.Camera()

        vertical_fov = np.deg2rad(
            camera.view_angle
        )

        half_vertical_fov = (
            vertical_fov
            / 2.0
        )

        # Convert the vertical FOV into the corresponding horizontal FOV.
        half_horizontal_fov = np.arctan(
            np.tan(
                half_vertical_fov
            )
            * aspect
        )

        # The smaller field of view is the limiting image dimension.
        half_fov = min(
            half_vertical_fov,
            half_horizontal_fov,
        )

        # Fraction of the image half-width/height that the bounding sphere
        # is allowed to occupy.
        #
        # CAMERA_MARGIN = 0.05 -> 90% of full image -> 90% of half-image.
        fill_fraction = (
            1.0
            - 2.0 * self.CAMERA_MARGIN
        )

        # Perspective image coordinates scale with tan(angle).
        #
        # If alpha is the camera half-FOV, the sphere's apparent angular
        # radius beta should satisfy:
        #
        #     tan(beta) / tan(alpha) <= fill_fraction
        #
        # Therefore:
        #
        #     beta = atan(fill_fraction * tan(alpha))
        #
        max_angular_radius = np.arctan(
            fill_fraction
            * np.tan(half_fov)
        )

        # For a sphere of radius r viewed from distance d:
        #
        #     sin(beta) = r / d
        #
        # therefore:
        #
        #     d = r / sin(beta)
        #
        cam_dist = (
            radius
            / np.sin(
                max_angular_radius
            )
        )

        return float(
            cam_dist
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
        Return True if both generated lightcurve files exist.
        """
        paths = (
            self.get_lightcurve_paths(
                asteroid_id=asteroid_id,
                model_id=model_id,
                cam_idx=cam_idx,
            )
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
        paths = (
            self.get_lightcurve_paths(
                asteroid_id=asteroid_id,
                model_id=model_id,
                cam_idx=cam_idx,
            )
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
        cam_dist: float,
    ) -> dict[str, np.ndarray]:
        """
        Load generated lightcurves if present, otherwise generate them.

        ``cam_dist`` is calculated once for the complete DAMIT shape and
        reused for every camera.
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

        model_dir = (
            self.get_simulated_model_dir(
                asteroid_id=asteroid_id,
                model_id=model_id,
            )
        )

        return generate_lightcurves_from_mesh(
            mesh=mesh,
            cam_idx=cam_idx,
            out_dir=model_dir,
            cam_dist=cam_dist,
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

        shape_path = (
            self.get_shape_path(
                model_id=model_id,
                asteroid_id=asteroid_id,
            )
        )

        mesh = self.load_mesh(
            model_id=model_id,
            asteroid_id=asteroid_id,
        )

        # Calculate exactly once for this normalized shape.
        #
        # The same distance is then used for every camera position.
        cam_dist = self.get_camera_distance(
            mesh
        )

        lightcurves = {
            cam_idx: self.load_or_generate_lightcurves(
                asteroid_id=asteroid_id,
                model_id=model_id,
                cam_idx=cam_idx,
                mesh=mesh,
                cam_dist=cam_dist,
            )
            for cam_idx in POSITION_IDS
        }

        return {
            "model_id": model_id,
            "asteroid_id": asteroid_id,
            "quality_flag": quality_flag,
            "shape_path": shape_path,
            "mesh": mesh,
            "camera_distance": cam_dist,
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

            shape_path = (
                self.get_shape_path(
                    model_id=model_id,
                    asteroid_id=asteroid_id,
                )
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
        Return True if all camera-position lightcurves are generated.
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

    def get_mesh_diagnostics(
        self,
        index: int,
        image_size: tuple[int, int] = (1024, 1024),
    ) -> dict[str, Any]:
        """
        Return geometric diagnostics for a normalized DAMIT mesh.

        No lightcurves are loaded or generated.
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

        mesh = self.load_mesh(
            model_id=model_id,
            asteroid_id=asteroid_id,
        )

        vertices = np.asarray(
            mesh.vertices,
            dtype=float,
        )

        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        cylindrical_radii = np.sqrt(
            x**2
            + y**2
        )

        spherical_radii = np.sqrt(
            x**2
            + y**2
            + z**2
        )

        cam_dist = self.get_camera_distance(
            mesh,
            image_size=image_size,
        )

        return {
            "index": index,
            "asteroid_id": asteroid_id,
            "model_id": model_id,
            "n_vertices": len(
                mesh.vertices
            ),
            "n_faces": len(
                mesh.faces
            ),
            "x_bounds": (
                float(x.min()),
                float(x.max()),
            ),
            "y_bounds": (
                float(y.min()),
                float(y.max()),
            ),
            "z_bounds": (
                float(z.min()),
                float(z.max()),
            ),
            "max_xy_radius": float(
                cylindrical_radii.max()
            ),
            "max_3d_radius": float(
                spherical_radii.max()
            ),
            "camera_margin": self.CAMERA_MARGIN,
            "camera_distance": cam_dist,
        }

    def render_validation_frames(
        self,
        index: int,
        cam_idx: str = "0_hor1",
        rot_angles=np.arange(360),
        image_size: tuple[int, int] = (1024, 1024),
    ) -> np.ndarray:
        """
        Render validation frames for a normalized DAMIT shape.

        No lightcurves are generated or saved.

        The same fixed camera distance that would be used during normal
        DAMIT simulation is used for every requested frame.

        Parameters
        ----------
        index
            DAMIT dataset index.

        cam_idx
            Camera position ID.

        rot_angles
            Rotation angles in degrees.

        image_size
            Render size as ``(width, height)``.

        Returns
        -------
        np.ndarray
            Grayscale uint8 array with shape
            ``(n_frames, height, width)``.
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

        mesh = self.load_mesh(
            model_id=model_id,
            asteroid_id=asteroid_id,
        )

        cam_dist = self.get_camera_distance(
            mesh,
            image_size=image_size,
        )

        return simulate_rotation_frames(
            mesh=mesh,
            cam_idx=cam_idx,
            rot_angles=rot_angles,
            show_edges=False,
            cam_dist=cam_dist,
            focal_point=(0, 0, 0),
            image_size=image_size,
            light_dist=1000,
            verbose=False,
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
        f"Camera margin:       "
        f"{dataset.CAMERA_MARGIN:.1%}"
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

        diagnostics = (
            dataset.get_mesh_diagnostics(
                0
            )
        )

        print(
            "First item diagnostics:"
        )

        print(
            diagnostics
        )