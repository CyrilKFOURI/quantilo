"""Recipes (030) through the API: save an analysis, make a recipe, replay it on next month's file,
the report, the Python export, import / export, cancellation, the agent's tools."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from quantilo.ml import browser

from app.config import Settings
from app.core.jobs import InlineJobRunner
from app.main import create_app
from tests.api.test_profiling import import_path

API = "/api/v1"
BASE = f"{API}/modules/recipes"
FORECAST = {
    "time": "date",
    "value": "montant",
    "period": "month",
    "aggregate": "sum",
    "models": ["naive", "seasonal_naive"],
    "horizon": 3,
}


def sales(seed: int, months: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", periods=months * 30, freq="D")
    n = len(days)
    frame = pd.DataFrame(
        {
            "date": days.strftime("%Y-%m-%d"),
            "pays": rng.choice(["FR", "BE", "DE"], n),
            "indice": rng.integers(1, 6, n),
            "nb_articles": rng.integers(1, 6, n),
            "montant": rng.gamma(2.0, 40.0, n).round(2),
        }
    )
    frame["churn"] = np.where(frame["montant"] + rng.normal(0, 30, n) > 90, "yes", "no")
    return frame


def write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def until(client: TestClient, url: str, done: Any = lambda body: True, tries: int = 600) -> Any:
    body = None
    for _ in range(tries):
        body = client.get(url).json()
        if done(body):
            return body
        time.sleep(0.1)
    raise AssertionError(f"Still waiting on {url}: {body}")


def finished(body: dict[str, Any]) -> bool:
    return body["status"] != "running"


def prepare(client: TestClient, dataset: str, model: str = "lightgbm") -> None:
    """October's work: 2 types corrected, 1 filter, 2 KPIs, 1 forecast kept, 1 model."""
    changes = [
        {"name": "indice", "type": "categorical_nominal"},
        {"name": "nb_articles", "type": "numeric_continuous"},
    ]
    assert client.patch(f"{API}/datasets/{dataset}/columns", json={"changes": changes}).is_success
    conditions = [{"column": "pays", "op": "in", "values": ["FR", "BE"]}]
    saved = client.post(
        f"{API}/datasets/{dataset}/saved-filters", json={"name": "FR+BE", "conditions": conditions}
    )
    assert saved.status_code == 201
    kpis = f"{API}/modules/kpi/datasets/{dataset}/definitions"
    revenue = {"measures": [{"id": "m1", "agg": "sum", "column": "montant"}]}
    orders = {
        "measures": [{"id": "m1", "agg": "count"}],
        "time": {"column": "date", "grain": "month"},
    }
    assert client.post(kpis, json={"name": "Revenue", "definition": revenue}).status_code == 201
    assert client.post(kpis, json={"name": "Orders", "definition": orders}).status_code == 201
    analyses = f"{API}/modules/timeseries/datasets/{dataset}/analyses"
    kept = client.post(analyses, json={"name": "Monthly sales", "config": FORECAST})
    assert kept.status_code == 201
    job = client.post(
        f"{API}/modules/ml/datasets/{dataset}/runs", params={"target": "churn", "model": model}
    ).json()
    job = until(client, f"{API}/modules/ml/jobs/{job['id']}", finished)
    assert job["status"] == "done", job


def make_recipe(client: TestClient, dataset: str, **body: Any) -> dict[str, Any]:
    draft = client.get(f"{BASE}/datasets/{dataset}/draft").json()
    active = next(s["id"] for s in draft["steps"] if s["kind"] == "filter")
    keep = [s["id"] for s in draft["steps"] if s["kind"] != "output"]
    created = client.post(
        f"{BASE}/recipes",
        json={
            "name": "Ventes mensuelles",
            "dataset_id": dataset,
            "keep": keep,
            "active_filter": active,
        }
        | body,
    )
    assert created.status_code == 201, created.text
    recipe = created.json()
    until(
        client,
        f"{BASE}/recipes/{recipe['id']}",
        lambda d: d["runs"] and all(r["status"] != "running" for r in d["runs"]),
    )
    return recipe


def replay(client: TestClient, recipe: str, dataset: str, mapping: Any = None) -> dict[str, Any]:
    started = client.post(
        f"{BASE}/recipes/{recipe}/runs", json={"dataset_id": dataset, "mapping": mapping}
    )
    assert started.status_code == 202, started.text
    return until(client, f"{BASE}/recipe-runs/{started.json()['id']}", finished)


@pytest.fixture
def october(client: TestClient, tmp_path: Path) -> str:
    dataset = import_path(client, write(sales(1), tmp_path / "ventes_octobre.csv"))
    prepare(client, dataset)
    return dataset


def test_save_an_analysis_on_the_time_series_page(client: TestClient, tmp_path: Path) -> None:
    dataset = import_path(client, write(sales(1), tmp_path / "a.csv"))
    url = f"{API}/modules/timeseries/datasets/{dataset}/analyses"

    saved = client.post(url, json={"name": "Monthly sales", "config": FORECAST}).json()

    listed = client.get(url).json()
    assert [a["name"] for a in listed] == ["Monthly sales"]
    assert listed[0]["config"]["horizon"] == 3 and listed[0]["origin"] is None
    bad = client.post(url, json={"name": "x", "config": {**FORECAST, "colour": 1}})
    assert bad.status_code == 422
    assert client.delete(f"{url}/{saved['id']}").status_code == 204
    assert client.get(url).json() == []


def test_replay_on_november_with_the_same_columns(
    client: TestClient, october: str, tmp_path: Path
) -> None:
    recipe = make_recipe(client, october)
    detail = client.get(f"{BASE}/recipes/{recipe['id']}").json()
    assert [s["kind"] for s in detail["step_views"]] == [
        "column", "column", "filter", "kpi", "kpi", "timeseries", "ml"
    ]  # fmt: skip
    assert "Treat indice as categorical" in detail["step_views"][0]["sentence"]
    baseline = detail["runs"][0]
    assert baseline["kind"] == "baseline" and baseline["status"] == "done"
    november = import_path(client, write(sales(2), tmp_path / "ventes_novembre.csv"))

    plan = client.post(f"{BASE}/recipes/{recipe['id']}/plan", json={"dataset_id": november}).json()
    assert {m["status"] for m in plan["matching"]["columns"]} == {"same"}
    assert all(s["skip"] is None for s in plan["steps"])

    run = replay(client, recipe["id"], november)

    assert [s["status"] for s in run["report"]["steps"]] == ["done"] * 7, run["report"]["steps"]
    assert run["status"] == "done"
    compared = {c["kind"]: c for c in run["compared"]}
    assert sorted(c["kind"] for c in run["compared"]) == ["forecast", "kpi", "kpi", "model"]
    assert all(c["previous"] is not None and c["value"] is not None for c in run["compared"])
    assert compared["model"]["detail"] == "roc_auc"
    assert run["changes"]["headline"].startswith("Compared with ventes_octobre")
    assert run["changes"]["vs"] == october
    links = {s["kind"]: s["link"] for s in run["report"]["steps"]}
    assert links["kpi"]["page"] == "kpi" and links["ml"]["page"] == "ml"
    # what the run added, marked with the recipe's name; nothing overwritten
    kpis = client.get(f"{API}/modules/kpi/datasets/{november}/definitions").json()
    assert sorted(k["name"] for k in kpis) == [
        "Orders · recipe Ventes mensuelles",
        "Revenue · recipe Ventes mensuelles",
    ]
    analyses = client.get(f"{API}/modules/timeseries/datasets/{november}/analyses").json()
    assert analyses[0]["origin"] == "Ventes mensuelles"
    columns = {c["name"]: c for c in client.get(f"{API}/datasets/{november}/columns").json()}
    assert columns["indice"]["type"] == "categorical_nominal"
    # the history draws each figure over the runs
    history = client.get(f"{BASE}/recipes/{recipe['id']}").json()["history"]
    assert {len(line["points"]) for line in history} == {2}
    csv_text = client.get(f"{BASE}/recipes/{recipe['id']}/history.csv").text
    assert csv_text.splitlines()[0] == "run,date,dataset,figure,detail,value"
    # a column a recipe reads that disappears is Breaking (031)
    used = client.app.state.versions._used(november)  # type: ignore[attr-defined]
    assert "Recipe “Ventes mensuelles”" in used["montant"]


def test_renamed_typo_and_missing_columns(client: TestClient, october: str, tmp_path: Path) -> None:
    recipe = make_recipe(client, october)
    frame = sales(3).rename(columns={"montant": "Montant", "pays": "pyas"}).drop(columns="indice")
    november = import_path(client, write(frame, tmp_path / "renamed.csv"))

    plan = client.post(f"{BASE}/recipes/{recipe['id']}/plan", json={"dataset_id": november}).json()

    found = {m["recipe"]: m for m in plan["matching"]["columns"]}
    assert (found["montant"]["status"], found["montant"]["column"]) == ("matched", "Montant")
    assert (found["pays"]["status"], found["pays"]["column"]) == ("proposed", "pyas")
    assert found["indice"]["status"] == "missing"
    skipped = {s["kind"]: s["skip"] for s in plan["steps"] if s["skip"]}
    assert "pays" in skipped["filter"]
    run = replay(client, recipe["id"], november)  # the proposal was not confirmed
    statuses = [s["status"] for s in run["report"]["steps"]]
    assert statuses[0] == "skipped" and statuses[1] == "done"
    assert statuses[2:6] == ["skipped"] * 4
    accepted = replay(client, recipe["id"], november, {"pays": "pyas"})
    kpi_steps = [s for s in accepted["report"]["steps"] if s["kind"] == "kpi"]
    assert [s["status"] for s in kpi_steps] == ["done", "done"]
    bad = client.post(
        f"{BASE}/recipes/{recipe['id']}/plan",
        json={"dataset_id": november, "mapping": {"pays": "nowhere"}},
    )
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "bad_mapping"


def test_the_exported_script_gives_the_same_key_figures(client: TestClient, tmp_path: Path) -> None:
    october = import_path(client, write(sales(1), tmp_path / "oct.csv"))
    prepare(client, october, model="logistic_regression")
    recipe = make_recipe(client, october)
    path = write(sales(4), tmp_path / "nov.csv")
    run = replay(client, recipe["id"], import_path(client, path))
    script = client.get(f"{BASE}/recipes/{recipe['id']}/python")
    assert "attachment" in script.headers["content-disposition"]
    (tmp_path / "recipe.py").write_text(script.text, encoding="utf-8")

    done = subprocess.run(
        [sys.executable, str(tmp_path / "recipe.py"), str(path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env={**os.environ, "QUANTILO_CACHE": str(tmp_path / "py")},
    )
    printed = json.loads(done.stdout[done.stdout.index("{") :])

    figures = {f["id"]: f["value"] for f in run["figures"]}
    assert set(printed) == set(figures)
    for key, value in figures.items():
        assert printed[key]["value"] == pytest.approx(value), key


def test_export_and_import_a_recipe_file(client: TestClient, october: str) -> None:
    recipe = make_recipe(client, october)
    exported = client.get(f"{BASE}/recipes/{recipe['id']}/export")
    assert exported.headers["content-disposition"].endswith("Ventes_mensuelles.recipe.json")

    imported = client.post(f"{BASE}/recipes/import", json={"content": exported.text})
    assert imported.status_code == 201
    assert imported.json()["steps"] == recipe["steps"]

    raw = json.loads(exported.text)
    raw["steps"][1]["colour"] = "red"
    refused = client.post(f"{BASE}/recipes/import", json={"content": json.dumps(raw)})
    assert refused.status_code == 422
    assert refused.json()["error"]["field"] == "steps[1].colour"
    raw = json.loads(exported.text) | {"schema": 9}
    refused = client.post(f"{BASE}/recipes/import", json={"content": json.dumps(raw)})
    assert refused.json()["error"]["field"] == "schema"
    assert "version 9" in refused.json()["error"]["message"]


def test_editing_makes_a_new_version(client: TestClient, october: str) -> None:
    recipe = make_recipe(client, october)
    edited = client.patch(
        f"{BASE}/recipes/{recipe['id']}",
        json={"name": "Sales", "enabled": {"s1": False}, "remove": ["s7"]},
    ).json()
    assert (edited["name"], edited["version"], edited["steps"]) == ("Sales", 2, 6)
    assert edited["step_views"][0]["enabled"] is False
    assert edited["runs"][0]["version"] == 1  # the baseline kept the version it used


def test_compatible_recipes_for_a_new_file(
    client: TestClient, october: str, tmp_path: Path
) -> None:
    recipe = make_recipe(client, october)
    november = import_path(client, write(sales(5), tmp_path / "nov.csv"))
    other = import_path(client, write(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), tmp_path / "o.csv"))

    found = client.get(f"{BASE}/datasets/{november}/compatible").json()

    assert [(c["recipe_id"], c["matched"], c["total"]) for c in found] == [(recipe["id"], 6, 6)]
    assert client.get(f"{BASE}/datasets/{other}/compatible").json() == []
    assert client.get(f"{BASE}/datasets/{october}/compatible").json() == []


def test_cancelling_a_run(client: TestClient, october: str, tmp_path: Path) -> None:
    recipe = make_recipe(client, october)
    november = import_path(client, write(sales(6), tmp_path / "nov.csv"))
    service = client.app.state.recipes  # type: ignore[attr-defined]
    original = service._save
    calls = {"n": 0}

    def save(run_id: str, report: Any, changes: Any = None) -> None:
        original(run_id, report, changes)
        calls["n"] += 1
        if calls["n"] == 4:  # after the first steps: the user clicks Cancel
            service.cancel(run_id)

    service._save = save
    run = replay(client, recipe["id"], november)

    statuses = [s["status"] for s in run["report"]["steps"]]
    assert run["status"] == "cancelled"
    assert "done" in statuses and statuses[-1] == "cancelled"
    first = statuses.index("cancelled")
    assert all(s == "done" for s in statuses[:first])
    assert all(s == "cancelled" for s in statuses[first:])


def test_machine_learning_runs_in_the_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """102: a scikit-learn model is trained again in the browser version too."""
    monkeypatch.setattr(browser, "IN_BROWSER", True)
    settings = Settings(home=tmp_path / "workspace", lite=True)
    with TestClient(create_app(settings, job_runner=InlineJobRunner())) as client:
        october = import_path(client, write(sales(1), tmp_path / "oct.csv"))
        prepare(client, october, model="logistic_regression")
        recipe = make_recipe(client, october)
        november = import_path(client, write(sales(2), tmp_path / "nov.csv"))
        run = replay(client, recipe["id"], november)

    ml = next(s for s in run["report"]["steps"] if s["kind"] == "ml")
    assert ml["status"] == "done", ml


def test_a_model_left_to_the_desktop_app_is_skipped_in_the_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-WEB-04: a recipe made on a computer with LightGBM, replayed in the browser version."""
    settings = Settings(home=tmp_path / "workspace", lite=True)
    with TestClient(create_app(settings, job_runner=InlineJobRunner())) as client:
        october = import_path(client, write(sales(1), tmp_path / "oct.csv"))
        prepare(client, october, model="lightgbm")
        recipe = make_recipe(client, october)
        monkeypatch.setattr(browser, "IN_BROWSER", True)
        november = import_path(client, write(sales(2), tmp_path / "nov.csv"))
        planned = client.post(f"{BASE}/recipes/{recipe['id']}/plan", json={"dataset_id": november})
        run = replay(client, recipe["id"], november)

    ml = next(s for s in run["report"]["steps"] if s["kind"] == "ml")
    assert ml["status"] == "skipped"
    assert ml["message"] == "LightGBM is available in the desktop app."
    assert run["status"] == "warning"
    assert planned.status_code == 200, planned.text
    step = next(s for s in planned.json()["steps"] if s["kind"] == "ml")
    assert step["skip"] == "LightGBM is available in the desktop app."


def test_agent_tools_list_and_run(client: TestClient, october: str, tmp_path: Path) -> None:
    from app.core.tools import ToolContext

    recipe = make_recipe(client, october)
    november = import_path(client, write(sales(7), tmp_path / "nov.csv"))
    tools = {t.name: t for t in client.app.state.agent_tools}  # type: ignore[attr-defined]

    listed = tools["list_recipes"].run(ToolContext(november), tools["list_recipes"].parse({}))
    assert listed.content["recipes"][0]["on_this_dataset"]["columns_found"] == "6 of 6"
    run_tool = tools["run_recipe"]
    assert run_tool.kind == "write"
    args = run_tool.parse({"recipe": "ventes mensuelles"})
    assert "Ventes mensuelles" in run_tool.confirmation(args) or "ventes" in run_tool.confirmation(
        args
    )

    done = run_tool.run(ToolContext(november), args)

    assert done.content["status"] == "done"
    assert len(done.content["key_figures"]) == 4
    assert done.link is not None and done.link.params["recipe"] == recipe["id"]
