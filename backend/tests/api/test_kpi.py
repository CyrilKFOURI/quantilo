import json

import pytest
from fastapi.testclient import TestClient

from app.config import default_samples_dir
from tests.api.test_profiling import import_path

BASE = "/api/v1/modules/kpi/datasets"


@pytest.fixture
def sales(client: TestClient) -> str:
    return import_path(client, default_samples_dir() / "sales_timeseries.csv")


def test_opens_on_suggestions_ready_to_compute(client: TestClient, sales: str) -> None:
    found = client.get(f"{BASE}/{sales}/suggestions").json()

    names = [k["name"] for k in found["kpis"]]
    # 048: best score first, each with its reasons
    assert names[0] == "Total revenue"
    assert {"Total units", "Average unit price", "Rows"} <= set(names)
    scores = [k["score"] for k in found["kpis"]]
    assert scores == sorted(scores, reverse=True)
    assert found["kpis"][0]["reasons"][0].startswith("R-SUG-02 · revenue is an amount")
    assert found["splits"][0] == "store"
    assert found["hierarchies"] == []
    result = client.post(f"{BASE}/{sales}/compute", json=found["kpis"][0]["definition"]).json()
    assert result["name"] == "Total revenue by month"
    assert result["resolved"]["grain"]["rule"] == "R-TIME-01"
    assert result["measures"][0]["chart"]["value"] in ("bars", "area")
    assert result["tiles"][0]["value"] is not None and result["resolved"]["why"]
    # 054: signals always listed, the started period at an equal date on the tiles
    assert isinstance(result["signals"], list) and "to_date" in result["tiles"][0]


def test_split_by_store_and_page_filters(client: TestClient, sales: str) -> None:
    definition = {
        "measures": [{"id": "m1", "agg": "sum", "column": "units"}],
        "time": {"column": "date", "grain": "quarter"},
        "groups": [{"column": "store"}],
    }
    result = client.post(f"{BASE}/{sales}/compute", json=definition).json()
    assert result["measures"][0]["chart"]["value"] == "stacked_bars"
    assert result["resolved"]["layout"]["value"] == "periods_columns"
    stores = result["splits"][0]["levels"]

    paris = json.dumps([{"column": "store", "op": "in", "values": ["Paris"], "negate": True}])
    filtered = client.post(
        f"{BASE}/{sales}/compute", params={"filters": paris}, json=definition
    ).json()
    assert "Paris" not in filtered["splits"][0]["levels"] and "Paris" in stores
    assert filtered["rows"] < result["rows"]
    alone = {**definition, "time": {**definition["time"], "page_filters": False}}
    ignored = client.post(f"{BASE}/{sales}/compute", params={"filters": paris}, json=alone).json()
    assert ignored["rows"] == result["rows"]  # the KPI only sees its own period


def test_errors_say_which_field_to_fix(client: TestClient, sales: str) -> None:
    response = client.post(
        f"{BASE}/{sales}/compute",
        json={"measures": [{"id": "m1", "agg": "sum", "column": "store"}]},
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "wrong_type" and error["field"] == "measures.m1.column"


def test_saved_kpis_round_trip(client: TestClient, sales: str) -> None:
    definition = {
        "measures": [{"id": "m1", "agg": "mean", "column": "revenue", "label": "Avg revenue"}],
        "time": {"column": "date", "grain": "month", "period": {"kind": "last", "n": 6}},
        "view": {"chart": "bars", "show_as": "change_pct"},
    }
    created = client.post(
        f"{BASE}/{sales}/definitions", json={"name": "Revenue level", "definition": definition}
    )
    assert created.status_code == 201
    kpi = created.json()
    assert kpi["definition"]["view"]["chart"] == "bars"
    assert kpi["definition"]["time"]["period"]["n"] == 6

    first = client.post(f"{BASE}/{sales}/compute", json=kpi["definition"]).json()
    again = client.post(
        f"{BASE}/{sales}/compute",
        json=client.get(f"{BASE}/{sales}/definitions/{kpi['id']}").json()["definition"],
    ).json()
    assert first == again

    copy = client.post(f"{BASE}/{sales}/definitions/{kpi['id']}/duplicate").json()
    assert copy["name"] == "Revenue level (copy)"
    renamed = client.put(f"{BASE}/{sales}/definitions/{copy['id']}", json={"name": "Copy"}).json()
    assert renamed["name"] == "Copy" and renamed["definition"] == kpi["definition"]
    assert len(client.get(f"{BASE}/{sales}/definitions").json()) == 2
    assert client.delete(f"{BASE}/{sales}/definitions/{copy['id']}").status_code == 204
    assert client.delete(f"{BASE}/{sales}/definitions/{copy['id']}").status_code == 404


@pytest.mark.slow
def test_five_million_rows(client: TestClient, tmp_path) -> None:
    import time

    import duckdb

    path = tmp_path / "big.parquet"
    duckdb.execute(
        "COPY (SELECT range AS id, random() * 100 AS a, "
        "['north', 'south', 'east', 'west'][1 + range % 4] AS zone, "
        "DATE '2015-01-01' + CAST(range % 3650 AS INTEGER) AS day FROM range(5000000)) "
        f"TO '{path.as_posix()}' (FORMAT parquet)"
    )
    dataset_id = import_path(client, path)
    definition = {
        "measures": [{"id": "m1", "agg": "sum", "column": "a"}],
        "time": {"column": "day"},
        "groups": [{"column": "zone"}],
    }
    start = time.perf_counter()
    response = client.post(f"{BASE}/{dataset_id}/compute", json=definition)
    seconds = time.perf_counter() - start
    assert response.status_code == 200, response.text
    assert seconds < 2, seconds
