"""089: join the open dataset with another one; 090: a log file imported; 091: the embeddings
option and its status."""

from pathlib import Path

from fastapi.testclient import TestClient

from tests.api.test_model import _shop
from tests.api.test_profiling import import_path
from tests.api.test_projects import create, move
from tests.factories import write_csv


def _ids(client: TestClient, tmp_path: Path) -> dict[str, str]:
    return {name: import_path(client, path) for name, path in _shop(tmp_path).items()}


def test_suggest_then_preview_then_create(client: TestClient, tmp_path: Path) -> None:
    ids = _ids(client, tmp_path)
    project = create(client, "Shop")
    move(client, [ids["sales"], ids["customers"]], project["id"])
    base = f"/api/v1/datasets/{ids['sales']}/join"

    found = client.post(f"{base}/suggest", json={"other_id": ids["customers"]})
    assert found.status_code == 200, found.text
    body = found.json()
    assert body["same_project"] is True
    key = body["keys"][0]
    assert key["left_columns"] == ["customer_id"] and key["mode"] == "lookup"
    assert {c["name"] for c in body["other_columns"] if c["default_pick"]} == {"country", "segment"}

    request = {
        "other_id": ids["customers"],
        "left_columns": key["left_columns"],
        "right_columns": key["right_columns"],
    }
    preview = client.post(f"{base}/preview", json=request).json()
    assert preview["rows_before"] == preview["rows_after"] == 120
    assert preview["join_type"] == "left" and preview["check"]["orphan_rows"] == 2
    assert preview["added"] == ["customers_country", "customers_segment"]
    assert len(preview["rows"]) == 20 and not preview["refused"]
    assert preview["suggested_name"] == "sales · with customers"

    created = client.post(base, json={**request, "name": "sales + customers"})
    assert created.status_code == 201, created.text
    dataset_id = created.json()["dataset_id"]
    dataset = client.get(f"/api/v1/datasets/{dataset_id}").json()
    assert dataset["status"] == "ready" and dataset["row_count"] == 120
    projects = client.get("/api/v1/projects").json()
    assert dataset_id in next(p for p in projects if p["id"] == project["id"])["dataset_ids"]
    columns = {c["name"]: c for c in client.get(f"/api/v1/datasets/{dataset_id}/columns").json()}
    assert "customers_country" in columns


def test_from_the_parent_side(client: TestClient, tmp_path: Path) -> None:
    ids = _ids(client, tmp_path)
    base = f"/api/v1/datasets/{ids['customers']}/join"
    request = {
        "other_id": ids["sales"],
        "left_columns": ["customer_id"],
        "right_columns": ["customer_id"],
    }
    preview = client.post(f"{base}/preview", json=request).json()
    assert preview["mode"] == "expand" and preview["rows_after"] == 120
    lookup = client.post(f"{base}/preview", json={**request, "mode": "lookup"}).json()
    assert lookup["refused"] and lookup["refused"][0]["code"] == "R-JN-04"
    refused = client.post(base, json={**request, "mode": "lookup"})
    assert refused.status_code == 422
    summary = client.post(base, json={**request, "mode": "summarise"})
    assert summary.status_code == 201, summary.text
    dataset = client.get(f"/api/v1/datasets/{summary.json()['dataset_id']}").json()
    assert dataset["row_count"] == 30


def test_a_dataset_is_not_joined_with_itself(client: TestClient, tmp_path: Path) -> None:
    ids = _ids(client, tmp_path)
    response = client.post(
        f"/api/v1/datasets/{ids['sales']}/join/suggest", json={"other_id": ids["sales"]}
    )
    assert response.status_code == 422


def test_a_log_file_is_imported_with_its_reading(client: TestClient, tmp_path: Path) -> None:
    lines = [
        f'10.0.0.{i % 4} - - [10/Oct/2026:13:55:{i % 60:02d} +0000] "GET /p/{i} HTTP/1.1" '
        f'{200 if i % 4 else 500} {100 + i} "-" "curl/8"'
        for i in range(50)
    ]
    path = write_csv(tmp_path / "access.log", "\n".join(lines) + "\n")
    preview = client.post("/api/v1/sources/preview", json={"path": str(path)})
    assert preview.status_code == 200, preview.text
    assert preview.json()["format"] == "log"
    assert any("R-LOG-01" in w for w in preview.json()["warnings"])
    dataset_id = import_path(client, path)
    dataset = client.get(f"/api/v1/datasets/{dataset_id}").json()
    assert dataset["row_count"] == 50
    names = [c["name"] for c in client.get(f"/api/v1/datasets/{dataset_id}/columns").json()]
    assert {"time", "status", "template"} <= set(names)


def test_embeddings_status_names_the_missing_extra(client: TestClient) -> None:
    response = client.get("/api/v1/modules/ml/text-embeddings")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"].endswith("potion-multilingual-128M")
    if not body["installed"]:
        assert not body["available"] and "embed" in body["reason"]
        started = client.post("/api/v1/modules/ml/text-embeddings/download").json()
        assert started["download"] == "idle" and started["error"]
