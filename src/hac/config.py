from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_DIR = Path(__file__).resolve().parents[2]


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_DIR / ".env",
        env_prefix="HAC_",
        extra="ignore",
    )

    store_temp_obj_files_in_tempdir: bool = False


config = Config()
