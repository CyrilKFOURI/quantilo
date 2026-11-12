"""125 — interpretability under Explain, Advanced: ALE (cached), the surrogate and its
fidelity, the monotonic check, counterfactuals for one row."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.api.test_ml import BASE, wait
from tests.api.test_profiling import import_path


@pytest.fixture
def churn(client: TestClient, tmp_path: Path) -> str:
    rng = np.random.default_rng(0)
    n = 1_200
    tenure = rng.integers(0, 72, n)
    frame = pd.DataFrame(
        {
            "tenure": tenure,
            "months_left": 72 - tenure + rng.integers(0, 3, n),  # strongly linked to tenure
            "monthly": rng.uniform(20, 120, n).round(2),
            "contract": rng.choice(["month", "year"], n),
            "age": rng.integers(18, 90, n),
        }
    )
    score = (
        -0.06 * tenure + 0.03 * frame["monthly"] + np.where(frame["contract"] == "month", 1.5, -1)
    )
    frame["churn"] = np.where(score + rng.normal(0, 0.5, n) > 0.5, "yes", "no")
    path = tmp_path / "churn.csv"
    frame.to_csv(path, index=False)
    return import_path(client, path)


def train(client: TestClient, dataset: str, model: str, settings: str | None = None) -> str:
    params = {"target": "churn", "model": model, "validation": '{"test_size": 0.3}'}
    if settings:
        params["settings"] = settings
    job = wait(client, client.post(f"{BASE}/datasets/{dataset}/runs", params=params).json())
    assert job["status"] == "done", job
    return job["run_id"]


def explained(client: TestClient, dataset: str, run_id: str) -> dict:
    url = f"{BASE}/datasets/{dataset}/runs/{run_id}/explanation"
    for _ in range(1_500):
        body = client.get(url).json()
        if body["status"] != "running":
            return body
        time.sleep(0.2)
    raise AssertionError("explanation never finished")


def test_a_forest_gets_treeshap_and_the_advanced_views(client: TestClient, churn: str) -> None:
    run_id = train(client, churn, "random_forest", '{"n_estimators": 20, "max_depth": 6}')
    body = explained(client, churn, run_id)
    assert body["status"] == "done", body
    assert body["explanation"]["shap"]["method"] == "TreeSHAP (scikit-learn trees, exact)"
    tenure = next(d for d in body["explanation"]["dependence"] if d["column"] == "tenure")
    assert tenure["extrapolates"].startswith("R-XAI-01")

    url = f"{BASE}/datasets/{churn}/runs/{run_id}/explain"
    ale = client.get(f"{url}/ale", params={"column": "tenure", "bins": 10})
    assert ale.status_code == 200, ale.text
    found = ale.json()
    assert found["kind"] == "number" and len(found["grid"]) <= 11
    assert found["rule"] and found["rule"].startswith("R-XAI-01")  # the rule chose ALE
    assert client.get(f"{url}/ale", params={"column": "tenure", "bins": 10}).json() == found
    unordered = client.get(f"{url}/ale", params={"column": "contract"})
    assert unordered.status_code == 422 and "order" in unordered.json()["error"]["message"]

    tree = client.get(f"{url}/surrogate", params={"depth": 2}).json()
    assert tree["measure"] == "agreement" and 0 <= tree["fidelity"] <= 1
    assert tree["nodes"][0]["left_rule"]
    assert client.get(f"{url}/monotonic").json() == []

    first = client.get(f"{BASE}/datasets/{churn}/runs/{run_id}").json()["predictions"][0]
    row = first["row"]
    answer = client.post(f"{url}/counterfactual", json={"row": row})
    assert answer.status_code == 200, answer.text
    what = answer.json()
    assert what["target_class"] in what["classes"] and len(what["changes"]) <= 3
    age = next(c for c in what["columns"] if c["column"] == "age")
    assert age["editable"] is False and "age" in age["locked"]
    if what["reached"]:
        assert what["plausibility"]
    narrowed = client.post(
        f"{url}/counterfactual", json={"row": row, "editable": ["monthly"], "max_changes": 1}
    ).json()
    assert all(c["column"] == "monthly" for c in narrowed["changes"])
    locked = client.post(f"{url}/counterfactual", json={"row": row, "editable": []}).json()
    assert "unlock" in locked["sentence"] and not locked["changes"]
    assert client.post(f"{url}/counterfactual", json={"row": 10**6}).status_code == 422


def test_monotonic_constraints_are_checked(client: TestClient, churn: str) -> None:
    run_id = train(
        client, churn, "hist_gradient_boosting", '{"monotonic": "tenure:-1", "max_iter": 30}'
    )
    found = client.get(f"{BASE}/datasets/{churn}/runs/{run_id}/explain/monotonic").json()
    assert [m["column"] for m in found] == ["tenure"]
    assert found[0]["sign"] == -1 and found[0]["violations"] == 0
