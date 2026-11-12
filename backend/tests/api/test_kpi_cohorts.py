"""060 (cohorts, funnel) and 061 (target, kind of dataset) through the API."""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.api.test_profiling import import_path

BASE = "/api/v1/modules/kpi/datasets"
STAGES = ["Lead", "Qualified", "Proposal", "Won"]


@pytest.fixture
def deals(client: TestClient, tmp_path: Path) -> str:
    rng = random.Random(60)
    path = tmp_path / "deals.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "customer_id", "order_date", "stage", "sales", "budget"])
        for index in range(1_200):
            day = date(2023, 1, 1) + timedelta(days=rng.randrange(0, 365))
            stage = rng.choice([*STAGES, "Lost"])
            sales = round(rng.uniform(10, 500), 2)
            writer.writerow(
                [
                    f"O{index}",
                    f"C{rng.randrange(150)}",
                    day.isoformat(),
                    stage,
                    sales,
                    round(sales * rng.uniform(0.8, 1.3), 2),
                ]
            )
    return import_path(client, path)


def test_suggestions_propose_cohorts_a_funnel_and_a_kind(client: TestClient, deals: str) -> None:
    found = client.get(f"{BASE}/{deals}/suggestions").json()
    kinds = {a["kind"]: a for a in found["analyses"]}
    assert kinds["cohort"]["rule"] == "R-COH-01"
    assert kinds["cohort"]["customer"] == "customer_id"
    assert kinds["funnel"]["stages"] == STAGES and kinds["funnel"]["key"] == "order_id"
    assert found["archetype"]["id"] == "sales"
    assert found["archetype"]["message"].startswith("Looks like sales data · ")
    names = {k["name"] for k in found["kpis"]}
    assert set(found["archetype"]["kpis"]) <= names
    assert any(k["rule"] == "R-SUG-31" for k in found["kpis"])


def test_cohort_and_funnel_are_computed_on_demand(client: TestClient, deals: str) -> None:
    cohort = client.post(
        f"{BASE}/{deals}/cohort", json={"customer": "customer_id", "date": "order_date"}
    )
    assert cohort.status_code == 200, cohort.text
    body = cohort.json()
    assert body["grain"] == "month" and body["cohorts"][0]["retained"][0] == 1.0
    assert sum(p["new"] for p in body["periods"]) == body["customers"]
    funnel = client.post(
        f"{BASE}/{deals}/funnel", json={"column": "stage", "stages": STAGES, "key": "order_id"}
    ).json()
    reached = [s["reached"] for s in funnel["stages"]]
    assert reached == sorted(reached, reverse=True) and funnel["outside_labels"] == ["Lost"]
    missing = client.post(f"{BASE}/{deals}/cohort", json={"customer": "nope", "date": "order_date"})
    assert missing.status_code == 422


def test_a_target_reads_on_the_tile(client: TestClient, deals: str) -> None:
    definition = {
        "measures": [{"id": "m1", "agg": "sum", "column": "sales"}],
        "time": {"column": "order_date", "grain": "month"},
        "goal": {"column": "budget"},
    }
    tile = client.post(f"{BASE}/{deals}/compute", json=definition).json()["tiles"][0]
    assert tile["goal"]["source"] == "column" and 0.5 < tile["goal"]["attained"] < 1.5
    assert tile["goal"]["reason"].startswith("R-GOAL-01")
