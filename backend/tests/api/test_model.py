"""088: the model of a project — detection, decisions, layout, analysis datasets, lineage."""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings
from tests.api.test_profiling import import_path
from tests.api.test_projects import create, link_version, move
from tests.factories import write_csv


def _csv(rows: list[str]) -> str:
    return "\n".join(rows) + "\n"


def _shop(tmp_path: Path) -> dict[str, Path]:
    customers = ["customer_id,country,segment"] + [
        f"C{i:03d},{['FR', 'DE', 'ES'][i % 3]},{['retail', 'pro'][i % 2]}" for i in range(1, 31)
    ]
    products = ["ProductKey,name,family"] + [
        f"{i},P{i},{['a', 'b', 'c'][i % 3]}" for i in range(1, 21)
    ]

    def customer(i: int) -> str:
        return "C999" if i <= 2 else f"C{(i % 30) + 1:03d}"  # two orphans

    sales = ["sale_id,customer_id,product_id,quantity,amount"] + [
        f"{i},{customer(i)},{(i % 20) + 1},{i % 7 + 1},{i * 1.5:.2f}" for i in range(1, 121)
    ]
    return {
        "sales": write_csv(tmp_path / "sales.csv", _csv(sales)),
        "customers": write_csv(tmp_path / "customers.csv", _csv(customers)),
        "products": write_csv(tmp_path / "products.csv", _csv(products)),
    }


def _project(client: TestClient, tmp_path: Path) -> tuple[str, dict[str, str]]:
    ids = {name: import_path(client, path) for name, path in _shop(tmp_path).items()}
    project = create(client, "Shop")
    move(client, list(ids.values()), project["id"])
    return project["id"], ids


def _model(client: TestClient, project_id: str) -> dict[str, Any]:
    response = client.get(f"/api/v1/projects/{project_id}/model")
    assert response.status_code == 200, response.text
    return response.json()


def _accept_all(client: TestClient, project_id: str, model: dict[str, Any]) -> dict[str, Any]:
    for relation in model["relations"]:
        if relation["status"] != "suggested":
            continue
        response = client.post(
            f"/api/v1/projects/{project_id}/model/relations",
            json={
                key: relation[key]
                for key in ("child_id", "child_columns", "parent_id", "parent_columns")
            },
        )
        assert response.status_code == 200, response.text
        model = response.json()
    return model


def test_detects_the_star_of_a_project(client: TestClient, tmp_path: Path) -> None:
    project_id, ids = _project(client, tmp_path)
    model = _model(client, project_id)
    pairs = {
        (r["child_id"], tuple(r["child_columns"]), r["parent_id"], tuple(r["parent_columns"]))
        for r in model["relations"]
    }
    assert pairs == {
        (ids["sales"], ("customer_id",), ids["customers"], ("customer_id",)),
        (ids["sales"], ("product_id",), ids["products"], ("ProductKey",)),
    }
    assert all(r["status"] == "suggested" for r in model["relations"])
    by_parent = {r["parent_id"]: r for r in model["relations"]}
    customers = by_parent[ids["customers"]]
    assert customers["check"]["orphan_rows"] == 2
    assert (
        customers["join_type"] == "left" and customers["check"]["reasons"][0]["code"] == "R-JOIN-01"
    )
    assert model["facts"] == [ids["sales"]] and model["shape"] == "star"
    tables = {t["id"]: t for t in model["tables"]}
    assert (
        tables[ids["sales"]]["role"] == "fact" and tables[ids["customers"]]["role"] == "dimension"
    )
    assert tables[ids["customers"]]["keys"][0]["columns"] == ["customer_id"]


def test_accept_reject_edit_delete(client: TestClient, tmp_path: Path) -> None:
    project_id, ids = _project(client, tmp_path)
    model = _model(client, project_id)
    products = next(r for r in model["relations"] if r["parent_id"] == ids["products"])
    customers = next(r for r in model["relations"] if r["parent_id"] == ids["customers"])
    base = f"/api/v1/projects/{project_id}/model/relations"
    rejected = client.post(
        base,
        json={
            **{
                k: products[k] for k in ("child_id", "child_columns", "parent_id", "parent_columns")
            },
            "status": "rejected",
        },
    ).json()
    # R-REL-11: rejected stays rejected
    assert not any(r["parent_id"] == ids["products"] for r in rejected["relations"])
    accepted = client.post(
        base,
        json={
            k: customers[k] for k in ("child_id", "child_columns", "parent_id", "parent_columns")
        },
    ).json()
    relation = next(r for r in accepted["relations"] if r["status"] == "accepted")
    assert relation["join_type"] == "left"
    changed = client.patch(f"{base}/{relation['id']}", json={"join_type": "inner"}).json()
    inner = next(r for r in changed["relations"] if r["id"] == relation["id"])
    assert inner["join_type"] == "inner"
    assert inner["check"]["warnings"][0]["code"] == "R-JOIN-01"  # 2 sales rows dropped
    back = client.delete(f"{base}/{relation['id']}").json()
    assert (
        next(r for r in back["relations"] if r["parent_id"] == ids["customers"])["status"]
        == "suggested"
    )
    assert client.delete(f"{base}/{relation['id']}").status_code == 404


def test_refuses_what_is_not_a_table_of_the_project(client: TestClient, tmp_path: Path) -> None:
    project_id, ids = _project(client, tmp_path)
    stranger = import_path(client, write_csv(tmp_path / "other.csv", "customer_id,x\nC001,1\n"))
    body = {
        "child_id": ids["sales"],
        "child_columns": ["customer_id"],
        "parent_id": stranger,
        "parent_columns": ["customer_id"],
    }
    base = f"/api/v1/projects/{project_id}/model"
    assert client.post(f"{base}/relations", json=body).status_code == 409
    bad = {**body, "parent_id": ids["customers"], "child_columns": ["amount"]}
    response = client.post(f"{base}/relations", json=bad)
    assert response.status_code == 422 and response.json()["error"]["code"] == "incompatible"
    assert client.get("/api/v1/projects/nope/model").status_code == 404


def test_build_the_analysis_dataset(client: TestClient, tmp_path: Path) -> None:
    project_id, ids = _project(client, tmp_path)
    _accept_all(client, project_id, _model(client, project_id))
    base = f"/api/v1/projects/{project_id}/model"
    preview = client.post(f"{base}/preview", json={}).json()
    assert preview["fact_id"] == ids["sales"]
    assert (preview["rows_before"], preview["rows_after"]) == (120, 120)
    names = [column["name"] for column in preview["columns"]]
    assert "customers_country" in names and "products_family" in names
    assert "customers_customer_id" not in names  # the key is not repeated (R-BLD-01)
    assert preview["name"] == "sales · analysis"
    picked = client.post(f"{base}/preview", json={"picks": {ids["customers"]: ["country"]}}).json()
    assert [c["name"] for c in picked["columns"] if c["table_id"] == ids["customers"]] == [
        "customers_country"
    ]

    created = client.post(f"{base}/build", json={})
    assert created.status_code == 201, created.text
    dataset_id = created.json()["dataset_id"]
    dataset = client.get(f"/api/v1/datasets/{dataset_id}").json()
    assert dataset["status"] == "ready" and dataset["row_count"] == 120
    projects = client.get("/api/v1/projects").json()
    assert dataset_id in projects[0]["dataset_ids"]  # filed in the project (R-BLD-04)
    columns = {c["name"]: c for c in client.get(f"/api/v1/datasets/{dataset_id}/columns").json()}
    assert columns["customer_id"]["role"] == "customer_key"
    model = _model(client, project_id)
    assert dataset_id not in {t["id"] for t in model["tables"]}  # not a table of the model
    (build,) = model["builds"]
    assert build["dataset_id"] == dataset_id and build["out_of_date"] == []


def test_out_of_date_and_rebuild(client: TestClient, tmp_path: Path) -> None:
    project_id, ids = _project(client, tmp_path)
    _accept_all(client, project_id, _model(client, project_id))
    base = f"/api/v1/projects/{project_id}/model"
    dataset_id = client.post(f"{base}/build", json={}).json()["dataset_id"]
    sales = tmp_path / "sales.csv"
    sales.write_text(sales.read_text() + "121,C001,1,1,9.50\n")
    assert client.post(f"/api/v1/datasets/{ids['sales']}/reimport").status_code in (200, 202)
    (build,) = _model(client, project_id)["builds"]
    assert build["out_of_date"] == ["sales was reimported"]
    rebuilt = client.post(f"{base}/builds/{dataset_id}/rebuild")
    assert rebuilt.status_code == 200, rebuilt.text
    assert client.get(f"/api/v1/datasets/{dataset_id}").json()["row_count"] == 121
    (build,) = _model(client, project_id)["builds"]
    assert build["out_of_date"] == []


def test_aggregate_a_child_to_its_parent(client: TestClient, tmp_path: Path) -> None:
    project_id, ids = _project(client, tmp_path)
    model = _accept_all(client, project_id, _model(client, project_id))
    relation = next(r for r in model["relations"] if r["parent_id"] == ids["customers"])
    response = client.post(
        f"/api/v1/projects/{project_id}/model/aggregate", json={"relation_id": relation["id"]}
    )
    assert response.status_code == 201, response.text
    dataset_id = response.json()["dataset_id"]
    dataset = client.get(f"/api/v1/datasets/{dataset_id}").json()
    assert dataset["row_count"] == 30
    names = [c["name"] for c in client.get(f"/api/v1/datasets/{dataset_id}/columns").json()]
    assert names[:3] == ["customer_id", "country", "segment"]
    assert {"sales_count", "sales_amount_sum", "sales_amount_mean"} <= set(names)


def test_layout_and_versions(client: TestClient, settings: Settings, tmp_path: Path) -> None:
    project_id, ids = _project(client, tmp_path)
    version = import_path(
        client, write_csv(tmp_path / "customers_v1.csv", (tmp_path / "customers.csv").read_text())
    )
    link_version(settings, version, ids["customers"])
    base = f"/api/v1/projects/{project_id}/model"
    model = _model(client, project_id)
    customers = next(t for t in model["tables"] if t["root_id"] == ids["customers"])
    assert [v["seq"] for v in customers["versions"]] == [0, 1]
    # S13: one card per family, never a relation with its own version
    assert len(model["tables"]) == 3
    saved = client.put(
        f"{base}/layout",
        json={
            "positions": {ids["sales"]: {"x": 10, "y": 20}},
            "versions": {ids["customers"]: version},
        },
    ).json()
    assert saved["positions"][ids["sales"]] == {"x": 10, "y": 20}
    assert any(t["id"] == version and t["root_id"] == ids["customers"] for t in saved["tables"])
    wrong = client.put(f"{base}/layout", json={"versions": {ids["customers"]: ids["sales"]}})
    assert wrong.status_code == 409
