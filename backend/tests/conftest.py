from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.core.jobs import InlineJobRunner
from app.main import create_app


@pytest.fixture(autouse=True)
def allow_test_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """TestClient sends `Host: testserver`; production only allows localhost names."""
    monkeypatch.setenv("QUANTILO_ALLOWED_HOSTS", '["localhost", "127.0.0.1", "testserver"]')


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(home=tmp_path / "workspace")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings, job_runner=InlineJobRunner())) as test_client:
        yield test_client
