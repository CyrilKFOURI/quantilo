"""C1 through the API: an item in several projects, "Move to…" out of one project, the project's
images and the tables that name them, inactive relations, keys compared without case, filters
that propagate, a generated date table, and a join that says it is out of date and is rebuilt."""

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from tests.api.test_model import _accept_all, _model, _project
from tests.api.test_profiling import import_path
from tests.api.test_projects import BASE, create, dataset, members, move
from tests.factories import write_csv


def test_a_dataset_in_several_projects(client: TestClient, tmp_path: Path) -> None:
    one = dataset(client, tmp_path, "one")
    two = dataset(client, tmp_path, "two")
    a, b = create(client, "A")["id"], create(client, "B")["id"]
    move(client, [one, two], a)
    added = client.post(f"{BASE}/{b}/add", json={"dataset_ids": [one]})
    assert added.status_code == 200, added.text
    assert members(client) == {"A": sorted([one, two]), "B": [one]}
    # Move to… from B only: A keeps it
    response = client.post(
        f"{BASE}/move", json={"dataset_ids": [one], "project_id": None, "from_project_id": b}
    )
    assert response.status_code == 200 and members(client) == {"A": sorted([one, two]), "B": []}
    client.post(f"{BASE}/{b}/add", json={"dataset_ids": [one, two]})
    assert client.post(f"{BASE}/{a}/remove", json={"dataset_ids": [two]}).status_code == 200
    assert members(client) == {"A": [one], "B": sorted([one, two])}
    assert client.get(f"{BASE}/{a}").json()["dataset_ids"] == [one]


def test_tables_that_name_the_images_of_the_project(client: TestClient, tmp_path: Path) -> None:
    images = tmp_path / "images.parquet"
    pd.DataFrame(
        {
            "image_id": [f"i{n}" for n in range(6)],
            "file_name": [f"P{n}.png" for n in range(6)],
            "rel_path": [f"scans/P{n}.png" for n in range(6)],
        }
    ).to_parquet(images)
    image_set = import_path(client, images)
    clinic = import_path(
        client,
        write_csv(
            tmp_path / "clinic.csv",
            "patient,scan\n" + "".join(f"patient {n},scans/P{n}.png\n" for n in range(6)),
        ),
    )
    project = create(client, "Breast cancer")["id"]
    move(client, [image_set, clinic], project)
    # an image set is a dataset of format "images" (106); this table stands in for one
    from app.core import db

    connection = db.connect(client.app.state.workspace.db_path)  # type: ignore[attr-defined]
    connection.execute("UPDATE datasets SET format = 'images' WHERE id = ?", (image_set,))
    connection.commit()
    connection.close()
    found = client.get(f"{BASE}/{project}/images").json()
    assert found["image_sets"] == [image_set] and found["tables"] == [clinic]
    [ref] = found["references"]
    assert (ref["column"], ref["by"], ref["found"], ref["rule"]) == ("scan", "path", 6, "R-PRJ-11")
    assert ref["applied"] is False


def test_inactive_relation_normalized_keys_filters_and_dates(
    client: TestClient, tmp_path: Path
) -> None:
    project_id, ids = _project(client, tmp_path)
    model = _accept_all(client, project_id, _model(client, project_id))
    base = f"/api/v1/projects/{project_id}/model"
    accepted = [r for r in model["relations"] if r["status"] == "accepted"]
    to_products = next(r for r in accepted if r["parent_id"] == ids["products"])
    assert to_products["active"] is True and to_products["match"] == "exact"
    patched = client.patch(f"{base}/relations/{to_products['id']}", json={"active": False})
    assert patched.status_code == 200, patched.text
    preview = client.post(f"{base}/preview", json={}).json()
    assert any(r["code"] == "R-REL-13" for r in preview["skipped"])
    assert not any(c["table_id"] == ids["products"] for c in preview["columns"])
    filtered = client.post(
        f"{base}/preview",
        json={
            "filters": [
                {"table_id": ids["customers"], "column": "country", "op": "eq", "values": ["FR"]}
            ]
        },
    ).json()
    assert 0 < filtered["rows_after"] < 120 and filtered["filtered_out"] > 0
    # keys compared without case: accepted with `normalized`
    lower = client.patch(f"{base}/relations/{to_products['id']}", json={"match": "normalized"})
    assert (
        next(r for r in lower.json()["relations"] if r["id"] == to_products["id"])["match"]
        == "normalized"
    )
    # a generated date table, related to its table
    orders = import_path(
        client,
        write_csv(tmp_path / "orders.csv", "order_id,day\n1,2024-01-30\n2,2024-02-02\n"),
    )
    move(client, [orders], project_id)
    made = client.post(f"{base}/dates", json={"dataset_id": orders, "column": "day"})
    assert made.status_code == 201, made.text
    dates_id = made.json()["dataset_id"]
    assert client.get(f"/api/v1/datasets/{dates_id}").json()["row_count"] == 4
    model = _model(client, project_id)
    assert any(t["id"] == dates_id for t in model["tables"])  # a dimension, not hidden
    relation = next(r for r in model["relations"] if r["parent_id"] == dates_id)
    assert relation["status"] == "accepted" and relation["child_columns"] == ["day"]
    assert any(b["dataset_id"] == dates_id and b["kind"] == "dates" for b in model["builds"])


def test_a_join_says_it_is_out_of_date_and_is_rebuilt(client: TestClient, tmp_path: Path) -> None:
    _, ids = _project(client, tmp_path)
    base = f"/api/v1/datasets/{ids['sales']}/join"
    request = {
        "other_id": ids["customers"],
        "left_columns": ["customer_id"],
        "right_columns": ["customer_id"],
    }
    joined = client.post(base, json=request).json()["dataset_id"]
    lineage = client.get(f"/api/v1/datasets/{joined}/join").json()
    assert lineage["out_of_date"] == [] and lineage["can_rebuild"] is True
    assert {s["id"] for s in lineage["sources"]} == {ids["sales"], ids["customers"]}
    sales = tmp_path / "sales.csv"
    sales.write_text(sales.read_text() + "121,C001,1,1,9.50\n")
    assert client.post(f"/api/v1/datasets/{ids['sales']}/reimport").status_code in (200, 202)
    lineage = client.get(f"/api/v1/datasets/{joined}/join").json()
    assert lineage["out_of_date"] == ["sales was reimported"]
    rebuilt = client.post(f"/api/v1/datasets/{joined}/join/rebuild")
    assert rebuilt.status_code == 200, rebuilt.text
    assert client.get(f"/api/v1/datasets/{joined}").json()["row_count"] == 121
    assert client.get(f"/api/v1/datasets/{joined}/join").json()["out_of_date"] == []
    assert client.get(f"/api/v1/datasets/{ids['sales']}/join").status_code == 404
