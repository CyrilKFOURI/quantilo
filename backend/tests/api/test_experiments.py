"""034 Experiments module: candidates, analysis through the filters, baseline, sample size."""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import default_samples_dir
from tests.api.test_profiling import import_path

BASE = "/api/v1/modules/experiments"


@pytest.fixture
def ab(client: TestClient) -> str:
    return import_path(client, default_samples_dir() / "ab_test.csv")


def test_candidates_open_on_a_ready_design(client: TestClient, ab: str) -> None:
    found = client.get(f"{BASE}/datasets/{ab}/candidates").json()

    assert found["variants"][0] == {
        "column": "variant",
        "levels": ["control", "new_checkout"],
        "counts": found["variants"][0]["counts"],
        "control": "control",
    }
    kinds = {m["column"]: m["kind"] for m in found["metrics"]}
    assert kinds["converted"] == "rate" and kinds["revenue"] == "mean"
    assert kinds["orders"] == "count"
    assert found["units"][0]["column"] == "user_id"
    assert found["times"] == ["date"] and "device" in found["segments"]
    suggested = found["suggested"]
    assert suggested["variant"] == "variant" and suggested["unit"] == "user_id"
    assert suggested["metrics"][0]["column"] == "converted"


def test_analyze_by_user_with_cuped_segments_and_time(client: TestClient, ab: str) -> None:
    design = {
        "variant": "variant",
        "metrics": [
            {"column": "converted"},
            {"column": "revenue", "pre": "spend_before"},
            {"column": "revenue", "denominator": "orders"},
        ],
        "unit": "user_id",
        "segment": "device",
        "time": "date",
    }
    result = client.post(f"{BASE}/datasets/{ab}/analyze", json=design).json()

    assert result["units"] == 2_000 and result["rows"] > result["units"]
    assert result["design"]["control"] == "control"
    assert result["correction"] == "p adjusted for 3 comparisons (Holm)"
    converted, revenue, ratio = result["metrics"]
    assert converted["kind"] == "rate" and converted["primary"]
    assert converted["comparisons"][0]["verdict"] == "better"
    assert converted["comparisons"][0]["sentence"].startswith("new_checkout reaches")
    assert revenue["cuped"]["sentence"].startswith("CUPED reduced the variance")
    assert ratio["kind"] == "ratio" and ratio["label"] == "revenue ÷ orders"
    assert [s["level"] for s in result["segments"]["levels"]] == ["mobile", "desktop", "tablet"]
    assert result["cumulative"]["grain"] == "day"


def test_filters_narrow_the_experiment(client: TestClient, ab: str) -> None:
    design = {"variant": "variant", "metrics": [{"column": "converted"}], "unit": "user_id"}
    mobile = json.dumps([{"column": "device", "op": "in", "values": ["mobile"]}])

    everyone = client.post(f"{BASE}/datasets/{ab}/analyze", json=design).json()
    filtered = client.post(
        f"{BASE}/datasets/{ab}/analyze", params={"filters": mobile}, json=design
    ).json()

    assert filtered["units"] < everyone["units"]


def test_mistakes_say_what_to_fix(client: TestClient, ab: str) -> None:
    wrong = client.post(
        f"{BASE}/datasets/{ab}/analyze",
        json={"variant": "user_id", "metrics": [{"column": "converted"}]},
    )
    assert wrong.status_code == 422 and "2 to 5 variants" in wrong.json()["error"]["message"]
    text = client.post(
        f"{BASE}/datasets/{ab}/analyze",
        json={"variant": "variant", "metrics": [{"column": "device"}]},
    )
    assert (
        text.status_code == 422 and "a metric must be a number" in text.json()["error"]["message"]
    )
    unknown = client.post(
        f"{BASE}/datasets/{ab}/analyze",
        json={"variant": "variant", "metrics": [{"column": "convrted"}]},
    )
    assert unknown.status_code == 404 and "converted" in unknown.json()["error"]["message"]


def test_baseline_then_sample_size(client: TestClient, ab: str) -> None:
    baseline = client.get(f"{BASE}/datasets/{ab}/baseline", params={"column": "converted"}).json()
    assert baseline["kind"] == "rate" and 0 < baseline["value"] < 1

    plan = client.post(
        f"{BASE}/sample-size",
        json={"baseline": round(baseline["value"], 3), "effect": 0.1, "daily": 500},
    ).json()
    assert plan["control"] > 1_000 and plan["total"] == 2 * plan["control"]
    assert plan["days"] == -(-plan["total"] // 500)
    assert len(plan["curve"]) == 41

    refused = client.post(f"{BASE}/sample-size", json={"baseline": 10, "effect": 0.1})
    assert refused.status_code == 422 and "between 0 and 1" in refused.json()["error"]["message"]
