"""036 — where a model is wrong: held-out predictions kept with the run, the analysis and its
cache, two runs compared, metrics per group, the agent tool."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.tools import ToolError
from tests.api.test_agent_actions import run_tool
from tests.api.test_ml import BASE, wait
from tests.api.test_profiling import import_path


@pytest.fixture
def shop(client: TestClient, tmp_path: Path) -> str:
    rng = np.random.default_rng(0)
    n = 2_000
    frame = pd.DataFrame(
        {
            "region": rng.choice(["North", "South", "East", "West"], n),
            "age": rng.uniform(18, 70, n).round(1),
            "plan": rng.choice(["basic", "plus", "pro"], n),
        }
    )
    noisy = (frame["region"] == "South") & (frame["age"] < 25)
    frame["spend"] = (
        frame["age"]
        + 30 * (frame["plan"] == "pro")
        + rng.normal(0, 8, n) * np.where(noisy, 5.0, 1.0)
    ).round(2)
    path = tmp_path / "shop.csv"
    frame.to_csv(path, index=False)
    return import_path(client, path)


def train(client: TestClient, dataset: str, model: str | None = None) -> str:
    params = {"target": "spend", "validation": '{"test_size": 0.4}'}
    job = client.post(
        f"{BASE}/datasets/{dataset}/runs", params=params | ({"model": model} if model else {})
    ).json()
    job = wait(client, job)
    assert job["status"] == "done", job
    return job["run_id"]


def test_the_errors_of_a_run_are_found_and_kept(client: TestClient, shop: str) -> None:
    run_id = train(client, shop)
    url = f"{BASE}/datasets/{shop}/runs/{run_id}/errors"

    found = client.get(url)

    assert found.status_code == 200, found.text
    body = found.json()
    assert body["source"] == "test rows" and body["rows"] == 800
    assert "region = South" in body["slices"][0]["where"]
    first = next(
        s for s in body["slices"] if {c["column"] for c in s["conditions"]} == {"region", "age"}
    )
    assert first["sentence"].startswith("When ")
    assert body["by_column"] and len(body["worst"]) == 50
    workspace = client.app.state.ml._folder(shop)  # type: ignore[attr-defined]
    assert (workspace / f"{run_id}.heldout.npz").is_file()
    assert (workspace / f"{run_id}.errors-absolute_error.json").is_file()
    assert client.get(url).json() == body  # from the cache
    assert client.get(url, params={"measure": "log_loss"}).status_code == 422

    # the subgroup's filter opens the same rows on the other pages
    rows = client.get(
        f"/api/v1/datasets/{shop}/rows",
        params={"filters": json.dumps(first["conditions"]), "limit": 1},
    ).json()
    assert rows["total"] >= first["rows"]  # every row of the group, training rows included

    # the worst predictions have their waterfall in the explanation
    state: dict[str, Any] = {"status": "running"}
    for _ in range(300):
        state = client.get(f"{BASE}/datasets/{shop}/runs/{run_id}/explanation").json()
        if state["status"] != "running":
            break
        time.sleep(0.1)
    assert state["status"] == "done", state
    explained = set(state["explanation"]["shap"]["rows"])
    assert {w["row"] for w in body["worst"]} <= explained


def test_two_runs_and_the_groups_of_a_column(client: TestClient, shop: str) -> None:
    first, second = train(client, shop), train(client, shop, "decision_tree")
    base = f"{BASE}/datasets/{shop}/runs/{first}/errors"

    compared = client.get(f"{base}/compare/{second}")
    assert compared.status_code == 200, compared.text
    body = compared.json()
    assert body["rows"] == 800 and "better" in body and "worse" in body

    groups = client.get(f"{base}/fairness", params={"column": "region"})
    assert groups.status_code == 200, groups.text
    assert {g["group"] for g in groups.json()["groups"]} == {"North", "South", "East", "West"}
    assert client.get(f"{base}/fairness", params={"column": "nope"}).status_code == 422


def test_a_run_without_held_out_predictions_says_so(client: TestClient, shop: str) -> None:
    run_id = train(client, shop)
    folder = client.app.state.ml._folder(shop)  # type: ignore[attr-defined]
    (folder / f"{run_id}.heldout.npz").unlink()

    found = client.get(f"{BASE}/datasets/{shop}/runs/{run_id}/errors")

    assert found.status_code == 422 and "train it again" in found.json()["error"]["message"]


def test_the_agent_reads_where_a_model_is_wrong(client: TestClient, shop: str) -> None:
    run_id = train(client, shop)

    out = run_tool(client, shop, "model_errors", {"run_id": run_id, "sensitive": "region"})

    assert out.summary.startswith("When ")
    assert out.content["subgroups"] and out.content["groups"]["column"] == "region"
    assert out.link is not None and out.link.params["run"] == run_id
    with pytest.raises(ToolError):
        run_tool(client, shop, "model_errors", {"run_id": "nope"})
