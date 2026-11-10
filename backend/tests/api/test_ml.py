import json
import time

import pytest
from fastapi.testclient import TestClient
from quantilo.ml import browser

from app.config import default_samples_dir
from tests.api.test_profiling import import_path

BASE = "/api/v1/modules/ml"


@pytest.fixture
def traps(client: TestClient) -> str:
    return import_path(client, default_samples_dir() / "typing_traps.csv")


def wait(client: TestClient, job: dict) -> dict:
    for _ in range(300):
        job = client.get(f"{BASE}/jobs/{job['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.1)
    return job


def test_targets_say_what_each_column_can_be(client: TestClient, traps: str) -> None:
    result = client.get(f"{BASE}/datasets/{traps}/targets").json()
    by_name = {c["name"]: c for c in result["candidates"]}

    assert result["rows"] == 600
    assert by_name["income"]["task"] == "regression"
    assert by_name["segment"]["task"] == "multiclass"
    assert by_name["signup_date"]["task"] is None and by_name["signup_date"]["reason"]
    assert result["candidates"][0]["task"] is not None  # predictable columns first


def test_train_a_regression_and_read_it_back(client: TestClient, traps: str) -> None:
    started = client.post(f"{BASE}/datasets/{traps}/runs", params={"target": "income"})
    assert started.status_code == 202, started.text
    job = wait(client, started.json())
    assert job["status"] == "done", job

    run = client.get(f"{BASE}/datasets/{traps}/runs/{job['run_id']}").json()
    assert run["model"] == "linear_regression" and run["task"] == "regression"
    assert run["regression"]["baseline_mae"] is not None and run["primary"] == "r2"
    assert run["train_rows"] + run["test_rows"] <= 600
    assert any(d["rules"] == ["R-SEL-01"] for d in run["preparation"]["decisions"])
    assert run["coefficients"] and run["predictions"]

    listed = client.get(f"{BASE}/datasets/{traps}/runs", params={"target": "income"}).json()
    assert listed[0]["id"] == job["run_id"] and listed[0]["metric"] == "r2"

    csv = client.get(f"{BASE}/datasets/{traps}/runs/{job['run_id']}/predictions.csv")
    assert csv.status_code == 200 and csv.text.startswith("row,actual,predicted")


def test_train_a_classification_with_settings(client: TestClient, traps: str) -> None:
    job = client.post(
        f"{BASE}/datasets/{traps}/runs",
        params={"target": "segment", "settings": '{"C": 0.5}'},
    ).json()
    job = wait(client, job)
    assert job["status"] == "done", job

    run = client.get(f"{BASE}/datasets/{traps}/runs/{job['run_id']}").json()
    assert run["model"] == "logistic_regression" and run["task"] == "multiclass"
    assert run["settings"]["C"] == 0.5
    assert run["classification"]["confusion"] and run["classification"]["report"]


def test_catalog_and_errors(client: TestClient, traps: str) -> None:
    found = client.get(f"{BASE}/catalog", params={"task": "binary", "minority": 0.1}).json()
    keys = [m["key"] for m in found["models"]]
    assert "logistic_regression" in keys and "linear_regression" not in keys
    logistic = found["models"][keys.index("logistic_regression")]
    weight = next(s for s in logistic["settings"] if s["key"] == "class_weight")
    assert weight["default"] == "balanced" and weight["why"]
    assert found["default_metric"] == "roc_auc" and found["recommended"]
    assert any(r["setting"] == "solver" for r in logistic["rules"])

    assert (
        client.post(f"{BASE}/datasets/{traps}/runs", params={"target": "signup_date"}).status_code
        == 422
    )
    assert (
        client.post(f"{BASE}/datasets/{traps}/runs", params={"target": "nope"}).status_code == 422
    )
    bad = client.post(
        f"{BASE}/datasets/{traps}/runs", params={"target": "income", "settings": "[1]"}
    )
    assert bad.status_code == 422
    failed = wait(
        client,
        client.post(
            f"{BASE}/datasets/{traps}/runs",
            params={"target": "income", "settings": '{"alpha": -3}'},
        ).json(),
    )
    assert failed["status"] == "failed" and "between" in failed["error"]
    assert client.get(f"{BASE}/datasets/{traps}/runs/nope").status_code == 404


def test_predictions_csv_neutralises_formulas() -> None:
    from app.modules.ml.service import _cell

    assert _cell("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert _cell("@cmd") == "'@cmd" and _cell("+1+1") == "'+1+1"
    assert _cell("\tx") == "'\tx"
    assert _cell("-12.5") == "-12.5" and _cell("yes") == "yes"


def test_preparation_preview_and_training_with_choices(client: TestClient, traps: str) -> None:
    choices = {
        "columns": {"age": {"scaling": "minmax"}},
        "outliers": "clip_iqr",
        "selection": [{"method": "variance", "value": 0.0}],
    }
    shown = client.post(
        f"{BASE}/datasets/{traps}/preparation", params={"target": "income"}, json=choices
    )
    assert shown.status_code == 200, shown.text
    body = shown.json()
    assert len(body["rows"]) == 20 and body["columns"]
    assert (
        body["preparation"]["custom"]
        and body["preparation"]["selection"][0]["method"] == "variance"
    )
    age = next(d for d in body["preparation"]["decisions"] if d["column"] == "age")
    assert age["scaling"] == "minmax" and age["manual"] == ["scaling"] and age["options"]["scaling"]

    job = wait(
        client,
        client.post(
            f"{BASE}/datasets/{traps}/runs", params={"target": "income"}, json=choices
        ).json(),
    )
    assert job["status"] == "done", job
    run = client.get(f"{BASE}/datasets/{traps}/runs/{job['run_id']}").json()
    assert run["preparation"]["choices"]["outliers"] == "clip_iqr"

    bad = client.post(
        f"{BASE}/datasets/{traps}/preparation",
        params={"target": "income"},
        json={"outliers": "explode"},
    )
    assert bad.status_code == 422


def test_a_number_predicted_as_classes(client: TestClient, traps: str) -> None:
    shape = client.get(f"{BASE}/datasets/{traps}/targets/income/distribution").json()
    assert shape["rows"] > 0 and len(shape["quantiles"]) == 1001
    low, high = shape["quantiles"][333], shape["quantiles"][667]
    classes = json.dumps(
        {"method": "quantile", "edges": [low, high], "names": ["Low", "Medium", "High"]}
    )

    job = wait(
        client,
        client.post(
            f"{BASE}/datasets/{traps}/runs", params={"target": "income", "classes": classes}
        ).json(),
    )
    assert job["status"] == "done", job
    run = client.get(f"{BASE}/datasets/{traps}/runs/{job['run_id']}").json()
    assert run["task"] == "multiclass" and run["classes"]["names"] == ["Low", "Medium", "High"]
    assert run["classification"]["labels"] == ["Low", "Medium", "High"]
    listed = client.get(f"{BASE}/datasets/{traps}/runs", params={"target": "income"}).json()
    assert listed[0]["classes"]["edges"] == [low, high]

    bad = json.dumps({"edges": [2, 1], "names": ["a", "b", "c"]})
    assert (
        client.post(
            f"{BASE}/datasets/{traps}/runs", params={"target": "income", "classes": bad}
        ).status_code
        == 422
    )
    assert client.get(f"{BASE}/datasets/{traps}/targets/segment/distribution").status_code == 422


def test_explanation_is_computed_in_the_background_then_cached(
    client: TestClient, traps: str
) -> None:
    job = wait(
        client, client.post(f"{BASE}/datasets/{traps}/runs", params={"target": "income"}).json()
    )
    url = f"{BASE}/datasets/{traps}/runs/{job['run_id']}/explanation"

    state = client.get(url).json()
    for _ in range(300):
        if state["status"] != "running":
            break
        time.sleep(0.1)
        state = client.get(url).json()

    assert state["status"] == "done", state
    found = state["explanation"]
    assert found["permutation"]["items"] and found["shap"]["columns"]
    assert client.get(url).json()["status"] == "done"  # from the cache


def _until(client: TestClient, url: str, done: tuple[str, ...]) -> dict:
    state = client.get(url).json()
    for _ in range(600):
        if state["status"] in done:
            break
        time.sleep(0.1)
        state = client.get(url).json()
    return state


def test_compare_models_trains_each_and_isolates_failures(client: TestClient, traps: str) -> None:
    started = client.post(
        f"{BASE}/datasets/{traps}/compare",
        params={"target": "income"},
        json={"models": ["linear_regression", "decision_tree", "catboost"]},
    )
    assert started.status_code == 202, started.text
    job = _until(client, f"{BASE}/compare-jobs/{started.json()['id']}", ("done",))

    items = {i["key"]: i for i in job["items"]}
    assert items["linear_regression"]["status"] == "done" and items["decision_tree"]["run_id"]
    assert items["catboost"]["status"] == "failed" and "not installed" in items["catboost"]["error"]
    runs = client.get(f"{BASE}/datasets/{traps}/runs", params={"target": "income"}).json()
    compared = [r for r in runs if r["id"] in {i["run_id"] for i in job["items"]}]
    assert len(compared) == 2 and compared[0]["validation"] == compared[1]["validation"]


def test_tune_a_model_in_the_background(client: TestClient, traps: str) -> None:
    started = client.post(
        f"{BASE}/datasets/{traps}/tune",
        params={"target": "income"},
        json={"model": "linear_regression", "trials": 5, "budget_seconds": 60},
    )
    assert started.status_code == 202, started.text
    job = _until(client, f"{BASE}/tune-jobs/{started.json()['id']}", ("done", "stopped", "failed"))

    assert job["status"] == "done", job
    progress = job["progress"]
    assert (
        len(progress["trials"]) == 5
        and progress["best_params"]
        and progress["default_value"] is not None
    )
    assert {p["key"] for p in job["space"]} >= {"alpha"}

    bad = client.post(
        f"{BASE}/datasets/{traps}/tune",
        params={"target": "income"},
        json={
            "model": "linear_regression",
            "space": [{"key": "alpha", "kind": "float", "low": 5, "high": 1}],
        },
    )
    failed = _until(client, f"{BASE}/tune-jobs/{bad.json()['id']}", ("done", "failed"))
    assert failed["status"] == "failed" and "lower bound" in failed["error"]


def test_estimate_and_cancel(client: TestClient, traps: str) -> None:
    found = client.post(
        f"{BASE}/datasets/{traps}/estimate", params={"target": "income", "model": "random_forest"}
    )
    assert found.status_code == 200, found.text
    assert found.json()["seconds"] > 0 and found.json()["fits"] == 1

    job = client.post(
        f"{BASE}/datasets/{traps}/runs",
        params={"target": "income", "validation": json.dumps({"method": "kfold", "folds": 10})},
    ).json()
    client.delete(f"{BASE}/jobs/{job['id']}")
    final = wait(client, job)
    assert final["status"] in ("cancelled", "done")  # a very fast training may finish first
    if final["status"] == "cancelled":
        assert final["run_id"] is None


def test_the_browser_version_offers_scikit_learn_models_and_says_what_it_holds_back(
    client: TestClient, traps: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """102: in the browser (simulated), the heavy models and tuning are left to the desktop app;
    a scikit-learn model trains, on one core and at most 50 000 rows."""
    monkeypatch.setattr(browser, "IN_BROWSER", True)
    found = client.get(f"{BASE}/catalog", params={"task": "regression"}).json()
    assert found["browser"] == {"max_rows": 50_000, "auto_minutes": 5.0, "tuning": False}
    models = {m["key"]: m for m in found["models"]}
    for key in ("xgboost", "lightgbm", "catboost"):
        assert models[key]["desktop_only"] and not models[key]["installed"]
    assert models["random_forest"]["installed"] and not models["random_forest"]["desktop_only"]
    assert not {"xgboost", "lightgbm", "catboost"} & set(found["compare"])

    trained = wait(
        client,
        client.post(
            f"{BASE}/datasets/{traps}/runs",
            params={
                "target": "income",
                "model": "random_forest",
                "validation": json.dumps({"max_rows": None}),
            },
        ).json(),
    )
    assert trained["status"] == "done", trained
    run = client.get(f"{BASE}/datasets/{traps}/runs/{trained['run_id']}").json()
    assert run["validation"]["max_rows"] == 50_000

    heavy = wait(
        client,
        client.post(
            f"{BASE}/datasets/{traps}/runs", params={"target": "income", "model": "xgboost"}
        ).json(),
    )
    assert heavy["status"] == "failed"
    assert heavy["error"] == "XGBoost is available in the desktop app."
    tune = client.post(
        f"{BASE}/datasets/{traps}/tune",
        params={"target": "income"},
        json={"model": "linear_regression", "trials": 5},
    )
    assert tune.status_code == 422 and "desktop app" in tune.text


def test_the_desktop_app_holds_nothing_back(client: TestClient) -> None:
    found = client.get(f"{BASE}/catalog", params={"task": "regression"}).json()
    assert found["browser"] is None
    assert not any(m["desktop_only"] for m in found["models"])
