import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import default_samples_dir
from tests.api.test_profiling import import_path


@pytest.fixture
def traps(client: TestClient) -> str:
    return import_path(client, default_samples_dir() / "typing_traps.csv")


def get(client: TestClient, dataset_id: str, **params):
    return client.get(f"/api/v1/datasets/{dataset_id}/compare", params=params)


def test_each_combination_through_the_api(client: TestClient, traps: str) -> None:
    kinds = {
        ("age", "income"): "scatter",
        ("segment", "price_eur"): "groups",
        ("country", "segment"): "contingency",
        ("signup_date", "price_eur"): "timeline",
        ("signup_date", "country"): "date_counts",
    }
    for (x, y), kind in kinds.items():
        response = get(client, traps, x=x, y=y)
        assert response.status_code == 200, (x, y, response.text)
        assert response.json()["kind"] == kind


def test_filters_narrow_the_comparison(client: TestClient, traps: str) -> None:
    full = get(client, traps, x="age", y="income").json()
    france = get(
        client,
        traps,
        x="age",
        y="income",
        filters=json.dumps([{"column": "country", "op": "in", "values": ["FR"]}]),
    ).json()

    assert france["total"] < full["total"] == 600
    assert france["rows"] <= france["total"]


def test_not_comparable_is_a_422(client: TestClient, traps: str) -> None:
    response = get(client, traps, x="age", y="comment")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "not_comparable"
    assert get(client, traps, x="age", y="nope").status_code == 422


def test_time_series_options(client: TestClient, traps: str) -> None:
    result = get(
        client, traps, x="signup_date", y="price_eur", period="month", agg="median", window=4
    ).json()

    assert result["time"]["period"] == "month" and result["time"]["aggregate"] == "median"
    assert result["time"]["window"] == 4 and not result["time"]["period_auto"]
    assert result["tests"][0]["key"] == "mann_kendall"
    assert get(client, traps, x="signup_date", y="price_eur", agg="mode").status_code == 422
    assert get(client, traps, x="signup_date", y="price_eur", window=1).status_code == 422


@pytest.mark.slow
def test_five_million_rows(client: TestClient, tmp_path: Path) -> None:
    import duckdb

    path = tmp_path / "big.parquet"
    duckdb.execute(
        "COPY (SELECT range AS id, random() * 100 AS a, random() * 50 AS b, "
        "['north', 'south', 'east', 'west'][1 + range % 4] AS zone, "
        "DATE '2015-01-01' + CAST(range % 3650 AS INTEGER) AS day FROM range(5000000)) "
        f"TO '{path.as_posix()}' (FORMAT parquet)"
    )
    dataset_id = import_path(client, path)

    for x, y in (("a", "b"), ("zone", "a")):
        start = time.perf_counter()
        response = get(client, dataset_id, x=x, y=y)
        seconds = time.perf_counter() - start
        assert response.status_code == 200, response.text
        assert seconds < 3, (x, y, seconds)

    start = time.perf_counter()
    response = get(client, dataset_id, x="day", y="a", color="zone")
    seconds = time.perf_counter() - start
    assert response.status_code == 200, response.text
    assert response.json()["kind"] == "timeline"
    assert seconds < 2, ("day", "a", seconds)


def test_whole_numbers_with_few_values_read_as_categories(client: TestClient, traps: str) -> None:
    url = f"/api/v1/datasets/{traps}/compare"
    grouped = client.get(url, params={"x": "nb_children", "y": "income"}).json()
    assert grouped["kind"] == "groups"
    assert grouped["categories"][0]["column"] == "nb_children"
    labels = [g["label"] for g in grouped["groups"]]
    assert labels == sorted(labels, key=float) and all("." not in label for label in labels)
    numbers = client.get(
        url, params={"x": "nb_children", "y": "income", "numbers": "nb_children"}
    ).json()
    assert numbers["kind"] == "scatter" and numbers["categories"] == []
    colored = client.get(url, params={"x": "segment", "y": "income", "color": "is_active"}).json()
    assert {g["color"] for g in colored["groups"]} == set(colored["color_levels"])
    many = client.get(url, params={"x": "segment", "y": "income", "color": "city"}).json()
    assert many["layout"]["axis"] == "city" and many["layout"]["swapped"]
    assert len(many["groups"]) <= 15


def test_survival_and_transitions_routes(client: TestClient, tmp_path: Path) -> None:
    lines = ["customer_id,month,status,tenure,Churn,Contract"]
    for i in range(60):
        lines.append(
            f"c{i % 20},2024-{1 + i // 20:02d}-01,{['trial', 'paid', 'churned'][min(i // 20, 2)]},"
            f"{i % 37},{'Yes' if i % 3 == 0 else 'No'},{'Two year' if i % 2 else 'Monthly'}"
        )
    path = tmp_path / "subs.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dataset = import_path(client, path)
    found = client.get(f"/api/v1/datasets/{dataset}/survival").json()
    assert found["setup"]["duration"] == "tenure" and found["setup"]["event"] == "Churn"
    assert found["setup"]["reason"].startswith("R-SRV-01")
    assert found["curves"][0]["times"][0] == 0
    grouped = client.get(
        f"/api/v1/datasets/{dataset}/survival", params={"group": "Contract"}
    ).json()
    assert len(grouped["curves"]) == 2 and grouped["log_rank"]["df"] == 1
    detected = client.get(
        f"/api/v1/datasets/{dataset}/transitions", params={"detect_only": True}
    ).json()
    assert detected["setup"]["status"] == "status" and detected["chain"] is None
    chain = client.get(f"/api/v1/datasets/{dataset}/transitions").json()["chain"]
    assert chain["absorbing"] == ["churned"]
    bad = client.get(
        f"/api/v1/datasets/{dataset}/survival", params={"duration": "nope", "event": "Churn"}
    )
    assert bad.status_code == 422
