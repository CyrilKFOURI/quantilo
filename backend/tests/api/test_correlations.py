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
    return client.get(f"/api/v1/datasets/{dataset_id}/correlations", params=params)


def test_matrix_and_insights_of_a_dataset(client: TestClient, traps: str) -> None:
    body = get(client, traps).json()

    matrix = body["matrix"]
    names = [c["name"] for c in matrix["columns"]]
    assert {"age", "income", "country"} <= set(names)
    assert "customer_id" in {e["name"] for e in matrix["excluded"]}
    assert len(matrix["values"]) == len(names) and len(matrix["clustered"]) == len(names)
    assert matrix["rows"] == matrix["total"] == 600 and not matrix["sampled"]
    assert body["insights"] and all(i["link"]["x"] for i in body["insights"])
    # city determines country: a redundancy between categories.
    assert any(
        i["kind"] == "redundant" and set(i["columns"]) == {"country", "city"}
        for i in body["insights"]
    )
    assert any(i["kind"] == "skewed" and i["columns"] == ["price_eur"] for i in body["insights"])


def test_method_columns_and_filters(client: TestClient, traps: str) -> None:
    body = get(
        client,
        traps,
        method="spearman",
        columns="age,income,segment",
        filters=json.dumps([{"column": "country", "op": "in", "values": ["FR"]}]),
    ).json()

    matrix = body["matrix"]
    assert matrix["method"] == "spearman"
    assert [c["name"] for c in matrix["columns"]] == ["age", "segment", "income"]  # file order
    assert matrix["total"] < 600


def test_unknown_method_is_rejected(client: TestClient, traps: str) -> None:
    assert get(client, traps, method="kendall").status_code == 422


@pytest.mark.slow
def test_sixty_columns_on_five_million_rows(client: TestClient, tmp_path: Path) -> None:
    import duckdb

    numbers = ", ".join(f"random() * {i + 1} AS n{i}" for i in range(50))
    categories = ", ".join(
        f"['a', 'b', 'c', 'd'][1 + (range * {2 * i + 1} + {i}) % 4] AS c{i}" for i in range(10)
    )
    path = tmp_path / "wide.parquet"
    duckdb.execute(
        f"COPY (SELECT {numbers}, {categories} FROM range(5000000)) "
        f"TO '{path.as_posix()}' (FORMAT parquet)"
    )
    dataset_id = import_path(client, path)

    start = time.perf_counter()
    response = get(client, dataset_id)
    seconds = time.perf_counter() - start

    assert response.status_code == 200, response.text
    assert len(response.json()["matrix"]["columns"]) == 60
    assert seconds < 10, seconds


@pytest.fixture
def latent(client: TestClient, tmp_path: Path) -> str:
    """Six numbers on two latent factors (three each) and a category."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 800
    f, g = rng.normal(size=n), rng.normal(size=n)
    frame = pd.DataFrame({f"a{i}": f + rng.normal(0, 0.5, n) for i in range(1, 4)})
    for i in range(1, 4):
        frame[f"b{i}"] = g + rng.normal(0, 0.5, n)
    frame["city"] = rng.choice(["Paris", "Lyon", "Nice"], n)
    path = tmp_path / "latent.csv"
    frame.to_csv(path, index=False)
    return import_path(client, path)


def test_structure_is_proposed_then_computed_on_request(client: TestClient, latent: str) -> None:
    rule = get(client, latent).json()["structure"]
    assert rule["proposed"] and rule["numeric"] == 6 and rule["strong"] == 6
    assert rule["reason"].startswith("6 numeric columns, 6 pairs")

    base = f"/api/v1/datasets/{latent}/correlations"
    found = client.get(f"{base}/structure").json()
    assert found["retained"] == 2 and found["columns"] == ["a1", "a2", "a3", "b1", "b2", "b3"]
    assert found["conclusion"].startswith("2 components explain")
    assert len(found["biplot"]["x"]) == 800 and found["parallel_rows"] == 800

    factors = client.get(f"{base}/factors").json()
    assert factors["k"] == 2 and factors["rotation"] == "varimax"
    assert factors["kmo"] > 0.6 and factors["bartlett_p"] < 0.001
    assert client.get(f"{base}/factors", params={"k": 1}).json()["k"] == 1
    narrow = client.get(f"{base}/structure", params={"columns": "a1,a2,a3"}).json()
    assert narrow["columns"] == ["a1", "a2", "a3"]


def test_structure_without_numbers_is_refused(client: TestClient, traps: str) -> None:
    base = f"/api/v1/datasets/{traps}/correlations"
    found = client.get(f"{base}/structure", params={"columns": "country,city"}).json()
    assert found["reason"] == "Needs at least 3 usable numeric columns (0 here)."
    assert client.get(f"{base}/factors", params={"k": 0}).status_code == 422
