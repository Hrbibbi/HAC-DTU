from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hac.constants import BASE_RADII, CHALLENGE_DATA_DIR

DataOrigin = Literal["real", "simulated"]
CurveType = Literal["binary", "intensity"]


def get_model_folder(model_num: int) -> Path:
    """
    Returns a Path object to the folder containing the data for the specified model number.
    """
    _model_folder_suffix = "public" if model_num <= 3 else "secret"
    model_folder_str = f"AsteroidModel{model_num:02d}_shape_{_model_folder_suffix}"

    model_dir = CHALLENGE_DATA_DIR / model_folder_str
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model data directory not found at {model_dir}. "
            f"Expected challenge data under {CHALLENGE_DATA_DIR}. Run ./bin/get_challenge_data.sh to download it."
        )

    return model_dir


def get_model_stl_path(model_num: int) -> Path:
    """
    Returns a Path object to the STL file containing the shape model for the specified model number.
    """
    if model_num > 3:
        raise ValueError(
            f"Model number {model_num} is a secret model and cannot be accessed."
        )

    model_dir = get_model_folder(model_num)

    stl_file_str = f"asteroid{model_num:d}.stl"
    stl_file_path = model_dir / stl_file_str
    if not stl_file_path.is_file():
        raise FileNotFoundError(
            f"STL file not found at {stl_file_path}. "
            f"Expected challenge data under {CHALLENGE_DATA_DIR}. Run ./bin/get_challenge_data.sh to download it."
        )

    return stl_file_path


@dataclass(frozen=True, slots=True)
class Dataset:
    """
    A class to represent a dataset of light curves for a specific model.

    See: https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/
    """

    model_num: int  # The model number (1-10)
    origin: DataOrigin  # Camera setup (real) or Blender simulation.
    curve_type: CurveType  # Binary (0/1) or intensity (0-255).

    data: pd.DataFrame  # The light curve data, with standardized index and columns
    base_radius: float

    def melt_data_to_long(self) -> pd.DataFrame:
        """
        Melt the data from wide format to long format for use with seaborn.
        """
        long = self.data.reset_index().melt(
            id_vars="frame", var_name="camera", value_name="value"
        )

        long[["cam_angle", "cam_pos"]] = long["camera"].str.split("_", n=1, expand=True)
        long["cam_angle"] = long["cam_angle"].astype(int)

        long = long[["frame", "cam_angle", "cam_pos", "value"]]
        return long


def load_data(
    model_num: int,
    origin: DataOrigin,
    curve_type: CurveType,
) -> Dataset:
    model_dir = get_model_folder(model_num)

    lightcurve_data_folder_str = f"Asteroid{model_num:d}_lightcurve_data"
    lightcurve_data_dir = model_dir / lightcurve_data_folder_str
    if not lightcurve_data_dir.is_dir():
        raise FileNotFoundError(
            f"Light curve data directory not found at {lightcurve_data_dir}. "
            f"Expected challenge data under {CHALLENGE_DATA_DIR}. Run ./bin/get_challenge_data.sh to download it."
        )

    data_file_str = f"Asteroid0{model_num}_lightcurve_{curve_type}"
    if origin == "simulated":
        data_file_str += "_blender"

    data_file_str += ".txt"

    data_file_path = lightcurve_data_dir / data_file_str
    if not data_file_path.is_file():
        raise FileNotFoundError(
            f"Data file not found at {data_file_path}. "
            f"Expected challenge data under {CHALLENGE_DATA_DIR}. Run ./bin/get_challenge_data.sh to download it."
        )

    # Load the data into a Dataframe
    data = pd.read_csv(data_file_path, header=None, index_col=0)

    # Standardize index and columns according to https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/
    data.index.name = "frame"
    data = data.rename(
        columns={
            1: "0_hor1",
            2: "0_hor2",
            3: "0_top",
            4: "0_bottom",
            5: "45_hor1",
            6: "45_hor2",
            7: "45_top",
            8: "45_bottom",
            9: "90_hor1",
            10: "90_hor2",
            11: "90_top",
            12: "90_bottom",
            13: "135_hor1",
            14: "135_hor2",
            15: "135_top",
            16: "135_bottom",
            17: "225_hor1",
            18: "225_hor2",
            19: "225_top",
            20: "225_bottom",
            21: "270_hor1",
            22: "270_hor2",
            23: "270_top",
            24: "270_bottom",
            25: "315_hor1",
            26: "315_hor2",
            27: "315_top",
            28: "315_bottom",
        }
    )

    return Dataset(
        model_num=model_num,
        origin=origin,
        curve_type=curve_type,
        data=data,
        base_radius=BASE_RADII[model_num - 1],
    )


@dataclass(frozen=True, slots=True)
class PreparedLightcurves:
    """Mean-normalized targets in the same observer order as ``ForwardModel``."""

    values: NDArray[np.float32]
    observer_names: tuple[str, ...]
    group_slices: dict[str, slice]
    n_phases: int  # number of frames kept; render this many phases to match


def _subsample_indices(n_native: int, n_phases: int) -> NDArray[np.intp]:
    """
    Pick ``n_phases`` native frame indices without averaging or interpolation.
    """
    if not 1 <= n_phases <= n_native:
        raise ValueError(
            f"n_phases must be in [1, {n_native}] (native frame count); "
            f"got {n_phases}."
        )
    return np.floor(
        np.arange(n_phases, dtype=np.float64) * n_native / n_phases
    ).astype(np.intp)


def prepare_lightcurve_targets(
    model_num: int,
    data_source: DataOrigin,
    curve_type: CurveType,
    n_phases: int | None = None,
    observer_names: Sequence[str] | None = None,
) -> PreparedLightcurves:
    """
    Load and mean-normalize one source's HAC curves.
    """
    dataset = load_data(
        model_num=model_num,
        origin=data_source,
        curve_type=curve_type,
    )
    selected_names = (
        tuple(dataset.data.columns)
        if observer_names is None
        else tuple(observer_names)
    )
    if not selected_names:
        raise ValueError("observer_names must contain at least one observer.")
    missing = [name for name in selected_names if name not in dataset.data.columns]
    if missing:
        raise ValueError(
            f"Observers {missing} are missing from model {model_num} "
            f"{data_source} data."
        )

    block = np.stack(
        [dataset.data[name].to_numpy() for name in selected_names]
    ).astype(np.float32)
    if not np.all(np.isfinite(block)):
        raise ValueError(
            f"Model {model_num} {data_source} {curve_type} curves contain "
            "non-finite samples."
        )
    if n_phases is not None:
        block = block[:, _subsample_indices(block.shape[1], n_phases)]

    means = np.mean(block, axis=1, keepdims=True)
    if not np.all(np.isfinite(means)) or np.any(means == 0.0):
        raise ValueError(
            f"Cannot mean-normalize model {model_num} {data_source} "
            f"{curve_type} curves."
        )
    block /= means

    return PreparedLightcurves(
        values=block,
        observer_names=tuple(selected_names),
        group_slices={data_source: slice(0, len(block))},
        n_phases=block.shape[1],
    )
