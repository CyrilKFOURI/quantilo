"""038 — the Auto through the API, the agent's tool and the library (same ranking)."""

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

from app.core.tools import ToolError
from tests.api.test_agent_actions import agent, run_tool
from tests.api.test_ml import BASE
from tests.api.test_profiling import import_path

KFOLD = {"method": "kfold", "folds": 5}
QUICK = {"minutes": 2, "models": ["decision_tree"], "trials": 2}


def data_csv(path: Path, rows: int = 600) -> Path:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (rows, 4))
    frame = pd.DataFrame(x, columns=["a", "b", "c", "d"])
    frame["shop"] = rng.choice(["north", "south", "east"], rows)
    frame["price"] = 3 * x[:, 0] - 2 * x[:, 1] + np.sin(2 * x[:, 2]) + rng.normal(0, 0.5, rows)
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return data_csv(tmp_path / "prices.csv")


@pytest.fixture
def dataset_id(client: TestClient, path: Path) -> str:
    return import_path(client, path)


def start(
    client: TestClient, dataset_id: str, body: dict[str, Any], validation: dict | None = None
) -> dict[str, Any]:
    params = {"target": "price"} | ({"validation": json.dumps(validation)} if validation else {})
    started = client.post(f"{BASE}/datasets/{dataset_id}/auto", params=params, json=body)
    assert started.status_code == 202, started.text
    return started.json()


def until(client: TestClient, job_id: str, done: Any) -> dict[str, Any]:
    state = client.get(f"{BASE}/auto-jobs/{job_id}").json()
    for _ in range(1_800):
        if done(state):
            break
        time.sleep(0.1)
        state = client.get(f"{BASE}/auto-jobs/{job_id}").json()
    return state


def finished(state: dict[str, Any]) -> bool:
    return state["status"] != "running"


def test_the_auto_runs_in_steps_and_saves_the_kept_model_as_a_run(
    client: TestClient, dataset_id: str
) -> None:
    job = until(client, start(client, dataset_id, QUICK)["id"], finished)
    assert job["status"] == "done", job
    progress = job["progress"]
    assert [s["key"] for s in progress["steps"]] == [
        "baselines",
        "families",
        "reduction",
        "tuning",
        "ensemble",
    ]
    assert all(s["status"] in ("done", "skipped") for s in progress["steps"])
    assert "5 folds of the training rows" in progress["compared_on"]  # the default split
    ids = {e["id"] for e in progress["entries"]}
    assert {"constant", "linear_regression", "decision_tree"} <= ids
    assert progress["log"] and progress["finished"] and progress["best"]
    run = client.get(f"{BASE}/datasets/{dataset_id}/runs/{job['run_id']}").json()
    assert run["auto"]["best"] == progress["best"]  # the journal is kept with the run
    assert run["validation"]["method"] == "split" and run["test_rows"] == 120
    assert run["auto"]["test_score"] == run["score"]
    runs = client.get(f"{BASE}/datasets/{dataset_id}/runs").json()
    assert runs[0]["id"] == job["run_id"] and runs[0]["label"]


def test_stopping_the_auto_keeps_the_best_model_so_far(client: TestClient, dataset_id: str) -> None:
    body = {"minutes": 2, "models": ["random_forest", "decision_tree"]}
    job_id = start(client, dataset_id, body, KFOLD)["id"]
    seen = until(
        client,
        job_id,
        lambda s: finished(s) or (s["progress"] or {}).get("step") == "tuning",
    )
    assert seen["status"] == "running", seen
    stopped = client.delete(f"{BASE}/auto-jobs/{job_id}")
    assert stopped.status_code == 200
    job = until(client, job_id, finished)
    assert job["status"] == "stopped" and job["run_id"], job
    run = client.get(f"{BASE}/datasets/{dataset_id}/runs/{job['run_id']}").json()
    best = next(e for e in job["progress"]["entries"] if e["id"] == job["progress"]["best"])
    assert run["model"] == best["model"] and run["score"] == best["score"]
    assert run["auto"]["stopped"]


def test_bad_requests_are_refused(client: TestClient, dataset_id: str) -> None:
    unknown = client.post(
        f"{BASE}/datasets/{dataset_id}/auto", params={"target": "price"}, json={"models": ["x"]}
    )
    assert unknown.status_code == 422 and "Unknown model" in unknown.text
    budget = client.post(
        f"{BASE}/datasets/{dataset_id}/auto", params={"target": "price"}, json={"minutes": 0}
    )
    assert budget.status_code == 422
    assert client.get(f"{BASE}/auto-jobs/nope").status_code == 422


def test_the_library_gives_the_same_ranking_as_the_app(
    client: TestClient, dataset_id: str, path: Path, tmp_path: Path
) -> None:
    body = {"minutes": 5, "models": ["decision_tree", "knn"], "trials": 3}
    job = until(client, start(client, dataset_id, body, KFOLD)["id"], finished)
    assert job["status"] == "done", job
    library = dx.open(path, cache_dir=tmp_path / "cache").ml("price", validation=KFOLD)
    found = library.auto(minutes=5, models=["decision_tree", "knn"], trials=3)
    app = job["progress"]
    lib = dx.to_dict(found.progress)
    sign = 1 if app["greater_is_better"] else -1

    def ranked(entries: list[dict[str, Any]]) -> list[tuple[str, float]]:
        done = [e for e in entries if e["status"] == "done" and e["score"] is not None]
        done.sort(key=lambda e: -sign * e["score"])
        return [(e["id"], e["score"]) for e in done]

    a, b = ranked(app["entries"]), ranked(lib["entries"])
    assert [i for i, _ in a] == [i for i, _ in b]
    for (_, x), (_, y) in zip(a, b, strict=True):
        assert math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)
    assert app["best"] == lib["best"] and app["trials"] == lib["trials"]


def test_the_agent_runs_the_auto_with_the_budget_it_was_given(
    client: TestClient, dataset_id: str
) -> None:
    tool = agent(client).registry.get("auto_ml")
    assert tool is not None and tool.kind == "compute"
    assert "ASK THE USER" in tool.description
    with pytest.raises(ToolError):
        tool.parse({"target": "price"})  # no budget: the agent must ask for one
    out = run_tool(client, dataset_id, "auto_ml", {"target": "price", "minutes": 0.25})
    assert out.content["run_id"] and out.content["leaderboard"]
    assert out.link.params["run"] == out.content["run_id"]
