import json

import pytest
from fastapi.testclient import TestClient

from app.config import default_samples_dir
from tests.api.test_profiling import import_path

FR = [{"column": "country", "op": "in", "values": ["FR"]}]


@pytest.fixture
def traps(client: TestClient) -> str:
    return import_path(client, default_samples_dir() / "typing_traps.csv")


def get(client: TestClient, path: str, filters: list[dict] | None = None, **params):
    if filters is not None:
        params["filters"] = json.dumps(filters)
    return client.get(f"/api/v1{path}", params=params)


def test_overview_counts_filtered_rows(client: TestClient, traps: str) -> None:
    full = get(client, f"/datasets/{traps}/overview").json()
    france = get(client, f"/datasets/{traps}/overview", FR).json()

    assert (full["rows"], full["total_rows"]) == (600, 600)
    assert 0 < france["rows"] < 600 and france["total_rows"] == 600
    assert france["column_stats"]["country"]["distinct"] == 1


def test_charts_and_variable_pages_follow_filters(client: TestClient, traps: str) -> None:
    chart = get(client, f"/datasets/{traps}/columns/chart", FR, column="country").json()
    profile = get(client, f"/datasets/{traps}/variable", FR, column="price_eur").json()
    overview = get(client, f"/datasets/{traps}/overview", FR).json()

    assert [bar["label"] for bar in chart["bars"]] == ["FR"]
    assert profile["numeric"]["describe"]["rows"] == overview["rows"]


def test_and_between_filters(client: TestClient, traps: str) -> None:
    both = [*FR, {"column": "age", "op": "gte", "value": 60}]

    france = get(client, f"/datasets/{traps}/overview", FR).json()["rows"]
    older = get(client, f"/datasets/{traps}/overview", both).json()["rows"]

    assert 0 < older < france


def test_column_summaries_are_filtered_but_types_unchanged(client: TestClient, traps: str) -> None:
    summaries = get(client, f"/datasets/{traps}/columns/summaries", FR).json()
    columns = client.get(f"/api/v1/datasets/{traps}/columns").json()

    assert summaries["country"]["distinct_count"] == 1
    assert next(c for c in columns if c["name"] == "country")["type"] == "categorical_nominal"


def test_the_editor_hints_come_with_the_columns_and_last_days_counts_back(
    client: TestClient,
) -> None:
    sales = import_path(client, default_samples_dir() / "sales_timeseries.csv")
    columns = {c["name"]: c for c in client.get(f"/api/v1/datasets/{sales}/columns").json()}

    # 100, R-FLT-09: a number's range and a date's span are in the summary computed at import
    units = columns["units"]["summary"]["numeric"]
    assert units["min"] <= units["median"] <= units["max"]
    last = columns["date"]["summary"]["date"]["max"][:10]
    assert columns["store"]["summary"]["numeric"] is None

    week = [{"column": "date", "op": "last_days", "value": 7, "high": last}]
    rows = get(client, f"/datasets/{sales}/overview", week).json()["rows"]
    stores = len(get(client, f"/datasets/{sales}/columns/chart", column="store").json()["bars"])
    assert rows == 7 * stores  # one row per store and day
    bad = [{"column": "date", "op": "last_days", "value": 0, "high": last}]
    assert get(client, f"/datasets/{sales}/overview", bad).status_code == 422


def test_invalid_filter_is_a_clear_422(client: TestClient, traps: str) -> None:
    bad = [{"column": "nope", "op": "eq", "value": 1}]

    response = get(client, f"/datasets/{traps}/overview", bad)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_filter"
    assert "Unknown column" in response.json()["error"]["message"]


def test_saved_filters(client: TestClient, traps: str) -> None:
    base = f"/api/v1/datasets/{traps}/saved-filters"

    created = client.post(base, json={"name": "France", "conditions": FR})
    replaced = client.post(base, json={"name": "France", "conditions": [*FR, *FR]})
    listed = client.get(base).json()

    assert created.status_code == 201
    assert replaced.json()["id"] == created.json()["id"]
    assert len(listed) == 1 and len(listed[0]["conditions"]) == 2
    assert client.delete(f"{base}/{created.json()['id']}").status_code == 204
    assert client.get(base).json() == []


def test_saved_filter_is_validated(client: TestClient, traps: str) -> None:
    response = client.post(
        f"/api/v1/datasets/{traps}/saved-filters",
        json={"name": "Bad", "conditions": [{"column": "nope", "op": "is_null"}]},
    )

    assert response.status_code == 422


@pytest.mark.slow
def test_filter_cost_on_five_million_rows(client: TestClient, tmp_path) -> None:
    import time

    import duckdb

    path = tmp_path / "big.parquet"
    duckdb.execute(
        "COPY (SELECT range AS id, ['FR', 'BE', 'ES'][1 + range % 3] AS country, "
        "random() * 100 AS amount FROM range(5000000)) "
        f"TO '{path.as_posix()}' (FORMAT parquet)"
    )
    dataset_id = import_path(client, path)
    filters = [{"column": "country", "op": "in", "values": ["FR"]}]

    start = time.perf_counter()
    get(client, f"/datasets/{dataset_id}/overview")
    plain = time.perf_counter() - start
    start = time.perf_counter()
    filtered = get(client, f"/datasets/{dataset_id}/overview", filters)
    narrowed = time.perf_counter() - start

    assert filtered.json()["rows"] == 1666667
    assert narrowed - plain < 1, (plain, narrowed)


def test_unique_values_and_duplicates_everywhere(client: TestClient, traps: str) -> None:
    one_per_city = [{"column": "city", "op": "unique", "keep": "last", "order_by": "signup_date"}]
    repeated = [{"column": "city", "op": "duplicated", "negate": True}]

    overview = get(client, f"/datasets/{traps}/overview", one_per_city).json()
    chart = get(client, f"/datasets/{traps}/columns/chart", one_per_city, column="city").json()
    rows = get(client, f"/datasets/{traps}/rows", one_per_city, limit=500).json()
    lonely = get(client, f"/datasets/{traps}/overview", repeated).json()
    cities = overview["column_stats"]["city"]["distinct"]

    assert overview["rows"] == cities == rows["total"] < 600
    assert all(bar["count"] == 1 for bar in chart["bars"])
    assert lonely["rows"] < 600
    # Combined with a value filter: one row per French city.
    french = get(client, f"/datasets/{traps}/overview", [*FR, *one_per_city]).json()
    assert french["rows"] == french["column_stats"]["city"]["distinct"] < cities


def test_unique_values_can_be_saved_and_are_validated(client: TestClient, traps: str) -> None:
    bad = [{"column": "city", "op": "unique", "columns": ["nope"]}]

    response = get(client, f"/datasets/{traps}/overview", bad)
    saved = client.post(
        f"/api/v1/datasets/{traps}/saved-filters",
        json={"name": "one per city", "conditions": [{"column": "city", "op": "unique"}]},
    )

    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_filter"
    assert saved.status_code in (200, 201), saved.text
