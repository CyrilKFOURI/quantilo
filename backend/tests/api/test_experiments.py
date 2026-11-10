"""034 Experiments module: candidates, analysis through the filters, baseline, sample size."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
        "planned": True,  # R-EXP-01 (094): its name says a variant
        "why": "Its name says a variant (variant).",
    }
    assert found["note"] is None
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


def test_bandit_view_in_the_analysis_and_the_plan_simulation(client: TestClient, ab: str) -> None:
    design = {"variant": "variant", "metrics": [{"column": "converted"}], "unit": "user_id"}
    result = client.post(f"{BASE}/datasets/{ab}/analyze", json=design).json()

    bandit = result["bandit"]
    assert [arm["variant"] for arm in bandit["arms"]] == ["control", "new_checkout"]
    assert sum(arm["p_best"] for arm in bandit["arms"]) == pytest.approx(1.0)
    assert bandit["replay"] is None and "date" in bandit["replay_reason"]
    dated = client.post(f"{BASE}/datasets/{ab}/analyze", json={**design, "time": "date"}).json()[
        "bandit"
    ]
    assert dated["replay"]["periods"] and dated["replay"]["sentence"].startswith("Over")

    plan = {"baseline": 0.2, "effect": 0.1, "daily": 500}
    simulated = client.post(f"{BASE}/bandit-simulation", json=plan)
    assert simulated.status_code == 200
    body = simulated.json()
    assert body["best"] == "B" and body["gained"] > 0 and body["guardrail"]
    assert client.post(f"{BASE}/bandit-simulation", json=plan).json() == body
    refused = client.post(f"{BASE}/bandit-simulation", json={"baseline": 2, "effect": 0.1})
    assert refused.status_code == 422


# --- 055: policies ------------------------------------------------------------------------------


def decisions_log(path: Path, n: int = 2_000) -> Path:
    """A log of decisions: sms is chosen more often for young customers, and helps them."""
    rng = np.random.default_rng(55)
    age = rng.integers(18, 70, n)
    young = age < 35
    p_sms = np.where(young, 0.6, 0.2)
    offer = np.where(rng.random(n) < p_sms, "sms", "email")
    reward = rng.random(n) < 0.2 + np.where((offer == "sms") & young, 0.2, 0.0)
    pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "offer": offer,
            "propensity": np.where(offer == "sms", p_sms, 1 - p_sms).round(3),
            "converted": reward.astype(int),
            "age": age,
            "region": rng.choice(["north", "south"], n),
        }
    ).to_csv(path, index=False)
    return path


def test_policy_on_the_sample_test(client: TestClient, ab: str) -> None:
    design = {"variant": "variant", "metrics": [{"column": "converted"}], "unit": "user_id"}
    analysis = client.post(f"{BASE}/datasets/{ab}/analyze", json=design).json()
    offer = analysis["policy_offer"]
    assert offer["covariates"] == ["device", "orders", "revenue", "spend_before"]
    if not offer["available"]:
        assert "200 units" in offer["reason"] or "at least 2" in offer["reason"]
        return
    body = {"design": design, "covariates": ["device", "spend_before"]}
    learnt = client.post(f"{BASE}/datasets/{ab}/policy", json=body)
    assert learnt.status_code == 200, learnt.text
    result = learnt.json()
    assert result["rule"].startswith("Give") and result["caveat"].startswith("Hypothesis")
    assert [v["key"] for v in result["values"]] == ["rule", "control", "new_checkout"]
    assert {c["column"] for c in result["covariates"] if c["used"]} == {"device", "spend_before"}
    assert result["qini"] == []  # "Advanced > More": computed only when asked
    with_qini = client.post(f"{BASE}/datasets/{ab}/policy", json={**body, "qini": True}).json()
    assert with_qini["rule"] == result["rule"] and with_qini["qini"][0]["variant"] == "new_checkout"
    refused = client.post(
        f"{BASE}/datasets/{ab}/policy", json={"design": design, "covariates": ["date", "device"]}
    )
    message = refused.json()["error"]["message"]
    assert refused.status_code == 422 and "cannot describe the units" in message


def test_logged_decisions_detected_and_evaluated(client: TestClient, tmp_path: Path) -> None:
    dataset = import_path(client, decisions_log(tmp_path / "decisions.csv"))
    found = client.get(f"{BASE}/datasets/{dataset}/decisions").json()
    assert found["detected"] and found["action"] == "offer" and found["reward"] == "converted"
    assert found["propensity"] == "propensity" and found["actions"] == ["email", "sms"]

    started = client.post(
        "/api/v1/modules/ml/datasets/" + dataset + "/runs", params={"target": "offer"}
    )
    job = started.json()
    for _ in range(600):
        job = client.get(f"/api/v1/modules/ml/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "done", job
    body = {
        "action": "offer",
        "reward": "converted",
        "propensity": "propensity",
        "models": [job["run_id"]],
        "learn": True,
    }
    result = client.post(f"{BASE}/datasets/{dataset}/evaluate-policy", json=body)
    assert result.status_code == 200, result.text
    evaluation = result.json()
    keys = [p["key"] for p in evaluation["policies"]]
    assert keys[:3] == ["logged", "always:email", "always:sms"] and "model:0" in keys
    assert "rule" in keys and evaluation["rule"].startswith("Choose")
    assert evaluation["diagnostics"]["propensity"] == "column"
    assert {e["estimator"] for e in evaluation["policies"][1]["estimates"]} == {
        "dm",
        "ips",
        "snips",
        "dr",
    }
    estimated = client.post(
        f"{BASE}/datasets/{dataset}/evaluate-policy",
        json={"action": "offer", "reward": "converted"},
    ).json()
    assert estimated["warnings"][0].startswith("Observational")


def test_ab_test_is_not_logged_decisions(client: TestClient, ab: str) -> None:
    found = client.get(f"{BASE}/datasets/{ab}/decisions").json()
    assert not found["detected"]
