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


def test_the_scorecard_tab_compares_readable_models_with_the_run(
    client: TestClient, churn: str
) -> None:
    """121, 122: FasterRisk's card and the sparse tree, cached; the optional extra greyed."""
    run_id = train(client, churn, "random_forest", '{"n_estimators": 20, "max_depth": 6}')
    url = f"{BASE}/datasets/{churn}/runs/{run_id}/readable"
    answer = client.get(f"{url}/scorecard", params={"conditions": 3, "points": 5})
    assert answer.status_code == 200, answer.text
    card = answer.json()
    assert 1 <= len(card["best"]["lines"]) <= 3
    assert all(abs(line["points"]) <= 5 for line in card["best"]["lines"])
    assert card["gap"]["sentence"] and "R-XAI-07" in card["gap"]["sentence"]
    assert client.get(f"{url}/scorecard", params={"conditions": 3, "points": 5}).json() == card
    tree = client.get(f"{url}/sparse-tree", params={"depth": 3}).json()
    assert tree["leaves"] <= 12 and "R-XAI-09" in tree["sentence"]

    every = {a["key"]: a for a in client.get(f"{BASE}/interpret/available").json()}
    assert set(every) == {"gosdt", "treefarms", "pacmap", "trimap"}
    if not every["gosdt"]["available"]:
        refused = client.get(f"{url}/optimal-tree")
        assert refused.status_code == 409 and "R-XAI-10" in refused.json()["error"]["message"]


def test_a_number_gets_no_scorecard(client: TestClient, churn: str) -> None:
    params = {"target": "monthly", "model": "linear_regression"}
    job = wait(client, client.post(f"{BASE}/datasets/{churn}/runs", params=params).json())
    url = f"{BASE}/datasets/{churn}/runs/{job['run_id']}/readable/scorecard"
    refused = client.get(url)
    assert refused.status_code == 422 and "R-XAI-05" in refused.json()["error"]["message"]
    # 122 bis (R-XAI-19): but a sparse tree, judged by R² against the run
    tree = client.get(f"{BASE}/datasets/{churn}/runs/{job['run_id']}/readable/sparse-tree")
    assert tree.status_code == 200, tree.text
    assert tree.json()["measure"] == "R²" and "R-XAI-19" in tree.json()["gap"]["sentence"]


def test_monotone_points_and_the_two_column_ale(client: TestClient, churn: str) -> None:
    """121 bis (R-XAI-16) and 125 bis (R-XAI-20), each cached next to the run."""
    run_id = train(client, churn, "random_forest", '{"n_estimators": 20, "max_depth": 6}')
    url = f"{BASE}/datasets/{churn}/runs/{run_id}"
    card = client.get(f"{url}/readable/scorecard", params={"conditions": 4, "monotone": True})
    assert card.status_code == 200, card.text
    assert card.json()["monotone"] is True and card.json()["shape"] == []
    surface = client.get(f"{url}/explain/ale-2d", params={"first": "tenure", "second": "monthly"})
    assert surface.status_code == 200, surface.text
    body = surface.json()
    assert len(body["values"]) == len(body["first_edges"]) and "R-XAI-20" in body["sentence"]
    again = client.get(f"{url}/explain/ale-2d", params={"first": "tenure", "second": "monthly"})
    assert again.json() == body
    refused = client.get(f"{url}/explain/ale-2d", params={"first": "tenure", "second": "contract"})
    assert refused.status_code == 422 and "not a number" in refused.json()["error"]["message"]


def test_a_knn_gets_the_advanced_more_views_each_cached(client: TestClient, churn: str) -> None:
    """126: SHAP by permutations for a model without exact SHAP, Friedman's H, one row's anchor,
    the prototypes and criticisms of each class."""
    run_id = train(client, churn, "knn")
    body = explained(client, churn, run_id)
    assert body["status"] == "done" and body["explanation"]["shap"] is None  # no exact SHAP
    url = f"{BASE}/datasets/{churn}/runs/{run_id}/explain"

    shap = client.get(f"{url}/permutation-shap", params={"pairs": 2, "background": 10})
    assert shap.status_code == 200, shap.text
    found = shap.json()
    assert found["shap"]["method"].startswith("Permutation sampling")
    assert found["additivity"] < 1e-9 and found["orders"] == 4
    assert len(found["half_width"]) == len(found["shap"]["rows"])
    again = client.get(f"{url}/permutation-shap", params={"pairs": 2, "background": 10}).json()
    assert again == found  # cached beside the run

    pairs = client.get(f"{url}/interactions", params={"columns": 3})
    assert pairs.status_code == 200, pairs.text
    h = pairs.json()
    assert len(h["columns"]) == 3 and len(h["pairs"]) == 3
    assert all(0 <= p["h"] <= 1 for p in h["pairs"]) and h["sentence"]

    row = client.get(f"{BASE}/datasets/{churn}/runs/{run_id}").json()["predictions"][0]["row"]
    rule = client.get(f"{url}/anchor", params={"row": row, "threshold": 0.9, "conditions": 3})
    assert rule.status_code == 200, rule.text
    anchor = rule.json()
    assert anchor["rule"].startswith("IF ") and 0 < len(anchor["conditions"]) <= 3
    assert anchor["threshold"] == 0.9 and 0 <= anchor["coverage"] <= 1
    missing = client.get(f"{url}/anchor", params={"row": 10**8})
    assert missing.status_code == 422 and "not among" in missing.json()["error"]["message"]

    examples = client.get(f"{url}/prototypes", params={"prototypes": 2, "criticisms": 1})
    assert examples.status_code == 200, examples.text
    groups = examples.json()["groups"]
    assert len(groups) == 2  # one per class (the app reads yes/no as a yes/no column)
    assert all(len(g["prototypes"]) == 2 and len(g["criticisms"]) == 1 for g in groups)
