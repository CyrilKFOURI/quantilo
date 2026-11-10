"""Projects (087): zero or one project per dataset, cleaned versions follow their original."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.core import db
from tests.api.test_profiling import import_path
from tests.factories import write_csv

BASE = "/api/v1/projects"


def dataset(client: TestClient, tmp_path: Path, name: str) -> str:
    """A distinct small CSV each time (the same file twice would be reused)."""
    return import_path(client, write_csv(tmp_path / f"{name}.csv", f"id,{name}\n1,2\n3,4\n"))


def create(client: TestClient, name: str) -> dict[str, Any]:
    response = client.post(BASE, json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()


def move(client: TestClient, ids: list[str], project: str | None) -> list[dict[str, Any]]:
    response = client.post(f"{BASE}/move", json={"dataset_ids": ids, "project_id": project})
    assert response.status_code == 200, response.text
    return response.json()


def members(client: TestClient) -> dict[str, list[str]]:
    return {p["name"]: sorted(p["dataset_ids"]) for p in client.get(BASE).json()}


def link_version(settings: Settings, child: str, parent: str, seq: int = 1) -> None:
    """A cleaned version (065) without running the cleaning: only its lineage row."""
    connection = db.connect(settings.home / "app.db")
    try:
        connection.execute(
            "INSERT INTO clean_versions (dataset_id, parent_id, seq, steps, origin, parent_stamp, "
            "report, created_at) VALUES (?, ?, ?, '{\"steps\": []}', 'user', '', '{}', '')",
            (child, parent, seq),
        )
        connection.commit()
    finally:
        connection.close()


def test_create_rename_delete(client: TestClient) -> None:
    assert client.get(BASE).json() == []
    churn = create(client, "  Churn study ")
    assert churn["name"] == "Churn study" and churn["dataset_ids"] == []
    create(client, "ab tests")
    # by name, case ignored (R-PRJ-09)
    assert [p["name"] for p in client.get(BASE).json()] == ["ab tests", "Churn study"]

    renamed = client.patch(f"{BASE}/{churn['id']}", json={"name": "Churn"})
    assert renamed.status_code == 200 and renamed.json()["name"] == "Churn"
    assert client.delete(f"{BASE}/{churn['id']}").status_code == 204
    assert [p["name"] for p in client.get(BASE).json()] == ["ab tests"]
    assert client.delete(f"{BASE}/{churn['id']}").status_code == 404
    assert client.patch(f"{BASE}/nope", json={"name": "x"}).status_code == 404


def test_names_are_unique_case_ignored(client: TestClient) -> None:
    sales = create(client, "Sales")
    other = create(client, "Other")
    duplicate = client.post(BASE, json={"name": "sales"})
    assert duplicate.status_code == 409 and duplicate.json()["error"]["code"] == "project_exists"
    clash = client.patch(f"{BASE}/{other['id']}", json={"name": "SALES "})
    assert clash.status_code == 409
    # renaming to itself with another case is fine
    assert client.patch(f"{BASE}/{sales['id']}", json={"name": "SALES"}).json()["name"] == "SALES"
    assert client.post(BASE, json={"name": "   "}).status_code == 422
    assert client.post(BASE, json={"name": "x" * 81}).status_code == 422


def test_move_between_projects_and_out(client: TestClient, tmp_path: Path) -> None:
    a, b = create(client, "A"), create(client, "B")
    first, second = dataset(client, tmp_path, "first"), dataset(client, tmp_path, "second")

    projects = move(client, [first, second], a["id"])
    assert {p["name"]: len(p["dataset_ids"]) for p in projects} == {"A": 2, "B": 0}
    # zero or one project (R-PRJ-02): moving to B takes it out of A
    move(client, [first], b["id"])
    assert members(client) == {"A": [second], "B": [first]}
    move(client, [first], None)
    assert members(client) == {"A": [second], "B": []}

    missing = client.post(f"{BASE}/move", json={"dataset_ids": ["nope"], "project_id": a["id"]})
    assert missing.status_code == 404
    unknown = client.post(f"{BASE}/move", json={"dataset_ids": [first], "project_id": "nope"})
    assert unknown.status_code == 404
    empty = client.post(f"{BASE}/move", json={"dataset_ids": [], "project_id": None})
    assert empty.status_code == 422


def test_deleting_a_project_keeps_its_datasets(client: TestClient, tmp_path: Path) -> None:
    a = create(client, "A")
    ids = [dataset(client, tmp_path, name) for name in ("x", "y", "z")]
    move(client, ids, a["id"])
    assert client.delete(f"{BASE}/{a['id']}").status_code == 204
    kept = {d["id"]: d["status"] for d in client.get("/api/v1/datasets").json()}
    assert all(kept[i] == "ready" for i in ids)  # R-PRJ-03
    assert client.get(f"/api/v1/datasets/{ids[0]}/head").status_code == 200


def test_cleaned_versions_follow_their_original(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    a, b = create(client, "A"), create(client, "B")
    original, other = dataset(client, tmp_path, "orig"), dataset(client, tmp_path, "other")
    v1, v2 = dataset(client, tmp_path, "v1"), dataset(client, tmp_path, "v2")
    move(client, [v1], b["id"])  # its own row, before it becomes a version
    link_version(settings, v1, original, 1)
    link_version(settings, v2, original, 2)
    # R-PRJ-07: a dataset that becomes a version loses its own row
    assert members(client) == {"A": [], "B": []}

    move(client, [original], a["id"])
    assert members(client) == {"A": sorted([original, v1, v2]), "B": []}
    # moving a version moves its family (R-PRJ-04)
    move(client, [v2], b["id"])
    assert members(client) == {"A": [], "B": sorted([original, v1, v2])}
    move(client, [other], b["id"])

    # R-PRJ-06: deleting the original keeps its versions in its project
    assert client.delete(f"/api/v1/datasets/{original}").status_code == 204
    assert members(client) == {"A": [], "B": sorted([other, v1, v2])}
    # now ordinary datasets: each moves on its own
    move(client, [v1], a["id"])
    assert members(client) == {"A": [v1], "B": sorted([other, v2])}


def test_deleting_a_dataset_removes_it_from_its_project(client: TestClient, tmp_path: Path) -> None:
    a = create(client, "A")
    first = dataset(client, tmp_path, "first")
    move(client, [first], a["id"])
    client.delete(f"/api/v1/datasets/{first}")
    assert members(client) == {"A": []}


def test_import_into_a_project(client: TestClient, tmp_path: Path) -> None:
    a = create(client, "A")
    path = write_csv(tmp_path / "sales.csv", "id,amount\n1,2\n")
    response = client.post("/api/v1/datasets", json={"path": str(path), "project_id": a["id"]})
    assert response.status_code == 201, response.text
    created = response.json()["dataset"]["id"]
    assert members(client) == {"A": [created]}
    # an unknown project is refused before anything is imported (R-PRJ-05)
    other = write_csv(tmp_path / "other.csv", "id,amount\n1,2\n")
    refused = client.post("/api/v1/datasets", json={"path": str(other), "project_id": "nope"})
    assert refused.status_code == 404
    assert len(client.get("/api/v1/datasets").json()) == 1
    # without a project, nothing changes (R-PRJ-01): DatasetOut has no project field
    plain = import_path(client, other)
    assert "project" not in client.get(f"/api/v1/datasets/{plain}").json()
    assert members(client) == {"A": [created]}


@pytest.mark.parametrize("name", ["Q3 report", "Étude churn"])
def test_unicode_and_spaces_in_names(client: TestClient, name: str) -> None:
    assert create(client, name)["name"] == name
