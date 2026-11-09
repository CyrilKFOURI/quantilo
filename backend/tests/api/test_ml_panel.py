"""035 — panel data through the API, the agent's tools and the library (same results)."""

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import quantilo as dx
from fastapi.testclient import TestClient

from tests.api.test_agent_actions import run_tool
from tests.api.test_ml import BASE, wait
from tests.api.test_profiling import import_path

PANEL = {
    "method": "panel_walk_forward",
    "period_column": "era",
    "entity_column": "id",
    "folds": 3,
    "embargo": 2,
}


def panel_csv(path: Path, periods: int = 30, entities: int = 30) -> Path:
    """ids by eras: a target four eras ahead (overlapping moves) plus one real signal."""
    rng = np.random.default_rng(11)
    moves = rng.normal(0, 1, (entities, periods + 4))
    rows = []
    for t in range(periods):
        for e in range(entities):
            signal = rng.normal()
            rows.append(
                {
                    "era": f"{t + 1:04d}",
                    "id": f"id{e:03d}",
                    "signal": signal,
                    "noise": rng.normal(),
                    "target": moves[e, t + 1 : t + 5].sum() + 0.8 * signal,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def panel_path(tmp_path: Path) -> Path:
    return panel_csv(tmp_path / "panel.csv")


@pytest.fixture
def panel_id(client: TestClient, panel_path: Path) -> str:
    return import_path(client, panel_path)


def train(client: TestClient, dataset_id: str, validation: dict[str, Any]) -> dict[str, Any]:
    started = client.post(
        f"{BASE}/datasets/{dataset_id}/runs",
        params={"target": "target", "validation": json.dumps(validation)},
    )
    assert started.status_code == 202, started.text
    job = wait(client, started.json())
    assert job["status"] == "done", job
    return client.get(f"{BASE}/datasets/{dataset_id}/runs/{job['run_id']}").json()


def test_the_panel_is_recognised_with_the_embargo_its_target_calls_for(
    client: TestClient, panel_id: str
) -> None:
    check = client.get(f"{BASE}/datasets/{panel_id}/panel", params={"target": "target"}).json()
    assert check["panel"] and check["period"] == "era" and check["entity"] == "id"
    assert check["periods"] == 30 and check["per_period"] == 30
    assert check["message"].startswith("This looks like panel data: 30 eras")
    assert check["embargo"] == 4 and "embargo of 4" in check["embargo_reason"]
    declared = client.get(
        f"{BASE}/datasets/{panel_id}/panel", params={"period": "era", "entity": "nope"}
    )
    assert declared.status_code == 422


def test_walk_forward_run_scores_every_period_and_neutralizes(
    client: TestClient, panel_id: str
) -> None:
    run = train(client, panel_id, PANEL)
    panel = run["panel"]
    assert run["primary"] == "spearman" and run["score"] == panel["value"]
    assert [f["fold"] for f in panel["folds"]] == [1, 2, 3]
    assert all(f["test_start"] - f["train_end"] > 2 for f in panel["folds"])
    assert len(panel["periods"]) == sum(f["test_periods"] for f in panel["folds"])
    assert panel["suggested_embargo"] == 4
    assert any(a["kind"] == "embargo" for a in panel["alerts"])  # 2 < 4: said in the run
    assert "out_of_fold" not in run

    csv = client.get(f"{BASE}/datasets/{panel_id}/runs/{run['id']}/periods.csv")
    assert csv.status_code == 200 and csv.text.startswith("era,fold,rows,spearman")
    assert len(csv.text.strip().splitlines()) == len(panel["periods"]) + 1

    done = client.post(
        f"{BASE}/datasets/{panel_id}/runs/{run['id']}/neutralize",
        json={"columns": ["signal"], "proportion": 1.0},
    )
    assert done.status_code == 200, done.text
    found = done.json()
    assert abs(found["after"][0]["mean"]) < 0.05 and found["before"][0]["mean"] > 0.3
    assert len(found["periods"]) == len(panel["periods"])
    refused = client.post(
        f"{BASE}/datasets/{panel_id}/runs/{run['id']}/neutralize",
        json={"columns": ["id"], "proportion": 1.0},
    )
    assert refused.status_code == 422

    plain = train(client, panel_id, {"method": "split"})
    assert plain["panel"] is None
    missing = client.post(
        f"{BASE}/datasets/{panel_id}/runs/{plain['id']}/neutralize", json={"columns": ["signal"]}
    )
    assert missing.status_code == 422


def test_the_periods_csv_neutralises_formulas_in_the_header(
    client: TestClient, tmp_path: Path
) -> None:
    path = panel_csv(tmp_path / "formula.csv")
    frame = pd.read_csv(path, dtype={"era": str}).rename(columns={"era": "=1+1"})
    frame.to_csv(path, index=False)
    dataset_id = import_path(client, path)
    run = train(client, dataset_id, {**PANEL, "period_column": "=1+1"})
    assert run["panel"]["period_column"] == "=1+1"
    csv = client.get(f"{BASE}/datasets/{dataset_id}/runs/{run['id']}/periods.csv")
    assert csv.status_code == 200
    header = csv.text.splitlines()[0]
    assert header.startswith("'=1+1,fold,rows,spearman")  # a label, never a formula


def test_a_panel_run_is_explained_on_its_rows(client: TestClient, panel_id: str) -> None:
    run = train(client, panel_id, PANEL)
    state: dict[str, Any] = {"status": "running"}
    for _ in range(300):
        state = client.get(f"{BASE}/datasets/{panel_id}/runs/{run['id']}/explanation").json()
        if state["status"] != "running":
            break
        time.sleep(0.1)
    assert state["status"] == "done", state


def test_compare_and_tune_use_the_panel_criterion(client: TestClient, panel_id: str) -> None:
    validation = json.dumps({**PANEL, "criterion": "worst"})
    started = client.post(
        f"{BASE}/datasets/{panel_id}/compare",
        params={"target": "target", "validation": validation},
        json={"models": ["linear_regression", "decision_tree"]},
    ).json()
    job = started
    for _ in range(600):
        job = client.get(f"{BASE}/compare-jobs/{started['id']}").json()
        if job["status"] == "done":
            break
        time.sleep(0.1)
    assert job["status"] == "done" and job["metric"] == "spearman"
    for item in job["items"]:
        run = client.get(f"{BASE}/datasets/{panel_id}/runs/{item['run_id']}").json()
        assert item["score"] == run["panel"]["summary"]["worst"]


def test_the_agent_checks_the_panel_and_trains_walk_forward(
    client: TestClient, panel_id: str
) -> None:
    checked = run_tool(client, panel_id, "check_panel", {"target": "target"})
    validation = checked.content["validation"]
    assert validation == {
        "method": "panel_walk_forward",
        "period_column": "era",
        "entity_column": "id",
        "embargo": 4,
    }
    trained = run_tool(
        client,
        panel_id,
        "train_model",
        {"target": "target", "validation": {**validation, "folds": 3}},
    )
    panel = trained.content["panel"]
    assert panel["period"] == "era" and panel["embargo"] == 4 and panel["folds"] == 3
    assert "mean" in panel["summary"]
    playbook = run_tool(client, panel_id, "get_playbook", {"topic": "panel"})
    assert "walk-forward" in playbook.content


def same(api: Any, library: Any) -> None:
    if isinstance(api, float) or isinstance(library, float):
        assert math.isclose(api, library, rel_tol=1e-9, abs_tol=1e-12), (api, library)
    else:
        assert api == library


def test_the_library_gives_the_same_periods_as_the_app(
    client: TestClient, panel_id: str, panel_path: Path, tmp_path: Path
) -> None:
    run = train(client, panel_id, PANEL)
    model = dx.open(panel_path, cache_dir=tmp_path / "cache").ml("target", validation=PANEL).train()
    library = dx.to_dict(model.panel)
    for a, b in zip(run["panel"]["periods"], library["periods"], strict=True):
        for key in ("period", "rank", "fold", "rows", "score", "pearson", "spearman"):
            same(a[key], b[key])
    for key in ("mean", "std", "sharpe", "share_good", "worst", "longest_bad_streak"):
        same(run["panel"]["summary"][key], library["summary"][key])
    same(run["score"], model.score)
