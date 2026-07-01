import os
from pathlib import Path
from collections.abc import Sequence

PROJECT_DIR = Path(__file__).resolve().parents[2]
DSDIR_ENV_VAR = "DSDIR"


def _resolve_challenge_data_dir() -> Path:
    dsdir = os.environ.get(DSDIR_ENV_VAR)
    if not dsdir:
        return PROJECT_DIR / "challenge_data"

    challenge_data_dir = Path(dsdir).expanduser()
    if challenge_data_dir.name != "challenge_data":
        challenge_data_dir = challenge_data_dir / "challenge_data"

    return challenge_data_dir.resolve()


CHALLENGE_DATA_DIR = _resolve_challenge_data_dir()

# Maps position id to camera angle alpha in degrees
# See: https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/
CAMERA_ANGLE_ALPHA: dict[str, float] = {
    "0_hor1": 0.0,
    "0_hor2": 0.0,
    "0_top": 21.0,
    "0_bottom": -21.0,
    "45_hor1": 0.0,
    "45_hor2": 0.0,
    "45_top": 26.0,
    "45_bottom": -24.0,
    "90_hor1": 0.0,
    "90_hor2": 0.0,
    "90_top": 26.0,
    "90_bottom": -24.0,
    "135_hor1": 0.0,
    "135_hor2": 0.0,
    "135_top": 26.0,
    "135_bottom": -24.0,
    "225_hor1": 0.0,
    "225_hor2": 0.0,
    "225_top": 24.0,
    "225_bottom": -26.0,
    "270_hor1": 0.0,
    "270_hor2": 0.0,
    "270_top": 24.0,
    "270_bottom": -26.0,
    "315_hor1": 0.0,
    "315_hor2": 0.0,
    "315_top": 24.0,
    "315_bottom": -26.0,
}

BASE_RADII = [1.12, 1.42, 0.88, 1.475, 1.22, 0.925, 1.205, 1.24, 0.67, 3.95]

class NamedAngleSequence(Sequence):
    def __init__(self, names: list[str], values: list[float | int]):
        self.names = names
        self.values = values
        self.by_name = dict(zip(names, values))

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.by_name[key]
        return self.values[key]

    def __len__(self):
        return len(self.values)


POSITION_IDS = list(CAMERA_ANGLE_ALPHA.keys())

XY_ANGLES = NamedAngleSequence(
    POSITION_IDS,
    [int(position_id.split("_", 1)[0]) for position_id in POSITION_IDS],
)

ELEV_ANGLES = NamedAngleSequence(
    POSITION_IDS,
    [CAMERA_ANGLE_ALPHA[position_id] for position_id in POSITION_IDS],
)

INTENSITY_CURVE_THRESHOLD = 25.5