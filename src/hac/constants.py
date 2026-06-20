import os
from pathlib import Path


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
