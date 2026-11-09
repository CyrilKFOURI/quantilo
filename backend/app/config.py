"""Application settings, read from environment variables and `.env`."""

import os
import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_home() -> Path:
    if sys.platform == "win32" and (local_app_data := os.environ.get("LOCALAPPDATA")):
        base, name, legacy = Path(local_app_data), "Quantilo", "DataExplorer"
    else:
        base, name, legacy = Path.home() / ".local" / "share", "quantilo", "data-explorer"
    # The app was called Data Explorer before 043: an existing workspace keeps its datasets.
    if not (base / name).exists() and (base / legacy).exists():
        return base / legacy
    return base / name


def adopt_legacy_environment() -> None:
    """Variables named before the rename (DATA_EXPLORER_HOME…) count unless the new one is set."""
    for key, value in list(os.environ.items()):
        if key.startswith("DATA_EXPLORER_"):
            os.environ.setdefault("QUANTILO_" + key.removeprefix("DATA_EXPLORER_"), value)


adopt_legacy_environment()


def default_samples_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "samples"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QUANTILO_",
        env_file=("../.env", ".env"),
        extra="ignore",
    )

    home: Path = Field(default_factory=default_home)
    static_dir: Path | None = None
    # When set (Docker: /data), browsing and imports are restricted to this folder.
    data_dir: Path | None = None
    samples_dir: Path = Field(default_factory=default_samples_dir)
    max_upload_mb: int = 20480
    # "Keep the previous version: Under …" (Settings > Storage, 031); 200 MB in the browser build.
    keep_version_limit_mb: int = 2048
    # The browser-only build (lite/boot.py): recipes skip their machine-learning steps (030).
    lite: bool = False
    # Host names the API answers to (protection against DNS rebinding).
    allowed_hosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1", "::1"])
