import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DSDIR_ENV_VAR = "DSDIR"


def _resolve_challenge_data_dir() -> Path:
    dsdir = os.environ.get(DSDIR_ENV_VAR)
    if dsdir and not dsdir.endswith("challenge_data"):
        dsdir = os.path.join(dsdir, "challenge_data")
    if not dsdir:
        raise RuntimeError(
            f"Environment variable {DSDIR_ENV_VAR} is not set. "
            "Set it to the dataset root directory before importing hac, "
            'for example: export DSDIR="/path/to/challenge_data".'
        )

    challenge_data_dir = Path(dsdir).expanduser().resolve()
    if not challenge_data_dir.is_dir():
        raise FileNotFoundError(
            f"{DSDIR_ENV_VAR} points to '{challenge_data_dir}', but that directory does not exist. "
            "Create it or run ./bin/get_challenge_data.sh after setting DSDIR."
        )

    return challenge_data_dir


CHALLENGE_DATA_DIR = _resolve_challenge_data_dir()
