"""082, 080b and 085 through the API: the type by the first bytes, the tables of one file imported
as one source, its origin, its deletion."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests.factories import write_csv, write_excel, write_parquet

ORDERS = [
    {"id": 1, "status": "shipped", "items": [{"sku": "A", "qty": 2}, {"sku": "B", "qty": 1}]},
    {"id": 2, "status": "open", "items": [{"sku": "C", "qty": 5}]},
]


def side_by_side(path: Path) -> Path:
    rows: list[list[object]] = [["Product", "Units", "Price", None, "Region", "Target"]]
    for i in range(8):
        rows.append([f"P{i}", i * 3, 1.5 * i] + ([None, f"R{i}", 100 + i] if i < 3 else []))
    return write_excel(path, {"Two": rows})


def preview(client: TestClient, path: Path, **options: object) -> dict:
    response = client.post("/api/v1/sources/preview", json={"path": str(path), "options": options})
    assert response.status_code == 200, response.text
    return response.json()


def test_preview_says_what_the_bytes_are(client: TestClient, tmp_path: Path) -> None:
    """R-DET-01: a Parquet file named .csv is read as Parquet, and the preview says so."""
    path = write_parquet(tmp_path / "export.csv", rows=7)
    body = preview(client, path)
    assert body["format"] == "parquet" and body["row_estimate"] == 7
    assert body["detected"]["kind"] == "table" and body["detected"]["mismatch"] is True
    assert any("R-DET-01" in note for note in body["warnings"])


def test_unsupported_content_is_named(client: TestClient, tmp_path: Path) -> None:
    path = tmp_path / "report.csv"
    path.write_bytes(b"%PDF-1.7\n")
    response = client.post("/api/v1/sources/preview", json={"path": str(path)})
    assert response.status_code == 422
    assert "PDF detected" in response.json()["error"]["message"]


def test_upload_refuses_what_it_recognises_as_unreadable(client: TestClient) -> None:
    response = client.put(
        "/api/v1/uploads", params={"filename": "data.csv"}, content=b"\x1f\x8b\x08\x00zz"
    )
    assert response.status_code == 415 and "Gzip" in response.json()["error"]["message"]
    response = client.put(
        "/api/v1/uploads", params={"filename": "orders.json"}, content=json.dumps(ORDERS).encode()
    )
    assert response.status_code == 201


def test_workbook_tables_become_one_source(client: TestClient, tmp_path: Path) -> None:
    path = side_by_side(tmp_path / "stock.xlsx")
    body = preview(client, path)
    ids = [table["id"] for table in body["tables"]]
    assert ids == ["Two!A1:C9", "Two!E1:F4"]
    assert body["tables"][0]["area"]["header_rows"] == [1]

    original = client.post("/api/v1/sources/original", json={"path": str(path), "sheet": "Two"})
    assert original.status_code == 200 and original.json()["cells"][0][0] == "Product"

    response = client.post(
        "/api/v1/sources/import",
        json={"path": str(path), "tables": [{"id": ids[0]}, {"id": ids[1], "name": "Targets"}]},
    )
    assert response.status_code == 201, response.text
    result = response.json()
    first, second = (item["dataset"] for item in result["imports"])
    assert (first["status"], first["row_count"], first["col_count"]) == ("ready", 8, 3)
    assert (second["name"], second["row_count"]) == ("Targets", 3)
    assert result["source"]["kind"] == "workbook"
    assert result["source"]["dataset_ids"] == [first["id"], second["id"]]

    origin = client.get(f"/api/v1/datasets/{second['id']}/origin").json()
    assert origin["source"]["id"] == result["source"]["id"]
    assert origin["locator"] == "Two!E1:F4"
    assert origin["layout"]["level"] == "high" and origin["layout"]["reasons"]

    # reimport reads the same table again
    assert client.post(f"/api/v1/datasets/{second['id']}/reimport").status_code == 202
    assert client.get(f"/api/v1/datasets/{second['id']}").json()["row_count"] == 3

    # deleting a table keeps the source; deleting the source deletes its tables
    assert client.delete(f"/api/v1/datasets/{first['id']}").status_code == 204
    source_id = result["source"]["id"]
    assert client.get(f"/api/v1/sources/{source_id}").json()["dataset_ids"] == [second["id"]]
    assert client.delete(f"/api/v1/sources/{source_id}").status_code == 204
    assert client.get(f"/api/v1/datasets/{second['id']}").status_code == 404
    assert client.get(f"/api/v1/sources/{source_id}").status_code == 404


def test_document_tables_keep_their_links(client: TestClient, tmp_path: Path) -> None:
    path = write_csv(tmp_path / "orders.json", json.dumps(ORDERS))
    body = preview(client, path)
    assert body["format"] == "json" and body["table"] == "orders"
    assert {t["id"] for t in body["tables"]} == {"orders", "orders.items"}
    response = client.post(
        "/api/v1/sources/import",
        json={"path": str(path), "tables": [{"id": "orders"}, {"id": "orders.items"}]},
    )
    assert response.status_code == 201, response.text
    orders, items = (item["dataset"] for item in response.json()["imports"])
    assert (orders["row_count"], items["row_count"]) == (2, 3)
    assert items["name"] == "orders - orders.items" and items["format"] == "json"
    links = client.get(f"/api/v1/datasets/{items['id']}/origin").json()["links"]
    assert links == [
        {
            "parent_id": orders["id"],
            "child_id": items["id"],
            "parent_key": "_id",
            "child_key": "_parent_id",
            "rule": links[0]["rule"],
        }
    ]
    assert client.get("/api/v1/sources").json()[0]["kind"] == "json"


def test_a_single_import_has_no_source(client: TestClient, tmp_path: Path) -> None:
    path = write_csv(tmp_path / "a.csv", "x,y\n1,2\n")
    dataset = client.post("/api/v1/datasets", json={"path": str(path)}).json()["dataset"]
    origin = client.get(f"/api/v1/datasets/{dataset['id']}/origin").json()
    assert origin == {"source": None, "locator": None, "layout": None, "links": []}


def test_a_csv_is_not_a_source_of_several_tables(client: TestClient, tmp_path: Path) -> None:
    path = write_csv(tmp_path / "a.csv", "x,y\n1,2\n")
    response = client.post(
        "/api/v1/sources/import", json={"path": str(path), "tables": [{"id": "A1:B2"}]}
    )
    assert response.status_code == 422


def test_tables_go_in_the_open_project(client: TestClient, tmp_path: Path) -> None:
    """R-PRJ-05: an import from a project puts every table of the source in it."""
    project = client.post("/api/v1/projects", json={"name": "Stock"}).json()
    path = side_by_side(tmp_path / "stock.xlsx")
    response = client.post(
        "/api/v1/sources/import",
        json={
            "path": str(path),
            "tables": [{"id": "Two!A1:C9"}, {"id": "Two!E1:F4"}],
            "project_id": project["id"],
        },
    )
    ids = response.json()["source"]["dataset_ids"]
    projects = client.get("/api/v1/projects").json()
    assert sorted(next(p for p in projects if p["id"] == project["id"])["dataset_ids"]) == sorted(
        ids
    )
