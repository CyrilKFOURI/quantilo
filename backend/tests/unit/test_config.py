"""The rename to Quantilo (043) keeps an existing workspace and the old variable names."""

import os
import sys
from pathlib import Path

import pytest

from app import config


@pytest.fixture
def base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if sys.platform == "win32":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        return tmp_path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path / ".local" / "share"


def test_a_new_install_uses_the_quantilo_folder(base: Path) -> None:
    assert config.default_home().name.lower() == "quantilo"


def test_the_workspace_from_before_the_rename_keeps_its_datasets(base: Path) -> None:
    legacy = base / ("DataExplorer" if sys.platform == "win32" else "data-explorer")
    legacy.mkdir(parents=True)

    assert config.default_home() == legacy

    (base / ("Quantilo" if sys.platform == "win32" else "quantilo")).mkdir()
    assert config.default_home().name.lower() == "quantilo"


def test_old_variable_names_still_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_EXPLORER_MAX_UPLOAD_MB", "7")
    monkeypatch.delenv("QUANTILO_MAX_UPLOAD_MB", raising=False)
    monkeypatch.setenv("DATA_EXPLORER_PORT", "9000")  # the new name wins when both are set
    monkeypatch.setenv("QUANTILO_PORT", "9001")

    config.adopt_legacy_environment()

    assert config.Settings().max_upload_mb == 7
    assert os.environ["QUANTILO_PORT"] == "9001"
