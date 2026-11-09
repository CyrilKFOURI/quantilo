from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["db"] == "ok"
    assert body["version"]


def test_startup_creates_workspace(client: TestClient, settings: Settings) -> None:
    assert (settings.home / "app.db").is_file()
    assert (settings.home / "datasets").is_dir()
    assert (settings.home / "cache").is_dir()


def test_home_can_come_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("QUANTILO_HOME", str(tmp_path / "from-env"))

    assert Settings().home == tmp_path / "from-env"
