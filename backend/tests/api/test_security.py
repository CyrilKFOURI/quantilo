import sys
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.paths import to_local_path

from app.api.security import hostname
from app.config import Settings
from app.core.jobs import InlineJobRunner
from app.main import create_app


@pytest.fixture
def local_client(tmp_path: Path) -> TestClient:
    settings = Settings(home=tmp_path / "ws", allowed_hosts=["localhost", "127.0.0.1", "::1"])
    return TestClient(
        create_app(settings, job_runner=InlineJobRunner()), base_url="http://localhost:8000"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("localhost:8000", "localhost"),
        ("127.0.0.1", "127.0.0.1"),
        ("[::1]:8000", "::1"),
        ("http://localhost:5173", "localhost"),
        ("https://evil.example", "evil.example"),
    ],
)
def test_hostname(value: str, expected: str) -> None:
    assert hostname(value) == expected


def test_local_requests_are_allowed(local_client: TestClient) -> None:
    with local_client:
        response = local_client.get("/api/v1/health", headers={"Sec-Fetch-Site": "same-origin"})

    assert response.status_code == 200


def test_dns_rebinding_host_is_rejected(local_client: TestClient) -> None:
    with local_client:
        response = local_client.get("/api/v1/browse", headers={"Host": "attacker.example:8000"})

    assert response.status_code == 403


def test_cross_site_requests_are_rejected(local_client: TestClient) -> None:
    with local_client:
        by_fetch_metadata = local_client.get(
            "/api/v1/browse", headers={"Sec-Fetch-Site": "cross-site"}
        )
        by_origin = local_client.post(
            "/api/v1/sources/preview",
            json={"path": "x"},
            headers={"Origin": "https://evil.example"},
        )

    assert by_fetch_metadata.status_code == 403
    assert by_origin.status_code == 403


NETWORK_PATHS = [
    "\\\\attacker\\share\\data.csv",
    "//attacker/share",
    "/\\attacker\\share",
    "\\/attacker/share",
    "\\\\?\\UNC\\attacker\\share",
    "\\\\.\\UNC\\attacker\\share",
    "//?/UNC/attacker/share",
    "\\\\attacker@SSL\\DavWWWRoot\\x.csv",
]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path syntax")
@pytest.mark.parametrize("path", NETWORK_PATHS)
def test_network_paths_are_rejected_before_any_file_access(path: str) -> None:
    with pytest.raises(IngestionError, match="Only local paths"):
        to_local_path(path)


def test_local_paths_are_accepted(tmp_path: Path) -> None:
    assert to_local_path(str(tmp_path)) == tmp_path
    assert to_local_path(str(tmp_path / "a" / ".." / "b")) == tmp_path / "b"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path syntax")
@pytest.mark.parametrize("path", NETWORK_PATHS[:3])
def test_network_paths_are_rejected_by_every_endpoint(local_client: TestClient, path: str) -> None:
    with local_client:
        # Pre-encoded like a browser would send it (the test client normalizes backslashes).
        browse = local_client.get(f"/api/v1/browse?path={quote(path, safe='')}")
        preview = local_client.post("/api/v1/sources/preview", json={"path": path})
        imported = local_client.post("/api/v1/datasets", json={"path": path})

    for name, response in (("browse", browse), ("preview", preview), ("import", imported)):
        assert response.status_code == 422, name
        assert "Only local paths" in response.json()["error"]["message"], name


def test_every_response_carries_the_security_headers(local_client: TestClient) -> None:
    with local_client:
        allowed = local_client.get("/api/v1/health", headers={"Sec-Fetch-Site": "same-origin"})
        refused = local_client.get("/api/v1/health", headers={"Sec-Fetch-Site": "cross-site"})

    for response in (allowed, refused):
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Content-Security-Policy"] == "frame-ancestors 'none'"
        assert response.headers["Referrer-Policy"] == "no-referrer"
