"""026b: the tools the modules offer to the agent, run on real data, and confirmations."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import default_samples_dir
from app.core.tools import ToolContext, ToolError
from app.modules.agent.providers.vendors import VENDORS
from tests.api.test_profiling import import_path
from tests.unit.agent.fakes import ScriptedProvider, answer, calls


@pytest.fixture(autouse=True)
def no_key_in_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real key reaches a test: every vendor's variable is cleared."""
    for vendor in VENDORS:
        monkeypatch.delenv(vendor.secret, raising=False)
        monkeypatch.delenv(f"QUANTILO_{vendor.secret}", raising=False)


@pytest.fixture
def traps(client: TestClient) -> str:
    return import_path(client, default_samples_dir() / "typing_traps.csv")


@pytest.fixture
def sales(client: TestClient) -> str:
    return import_path(client, default_samples_dir() / "sales_timeseries.csv")


def agent(client: TestClient) -> Any:
    return client.app.state.agent  # type: ignore[attr-defined]


def run_tool(client: TestClient, dataset_id: str, name: str, args: dict[str, Any]) -> Any:
    tool = agent(client).registry.get(name)
    assert tool is not None, f"{name} is not registered"
    progress: list[str] = []
    ctx = ToolContext(dataset_id, progress=lambda message, share: progress.append(message))
    return tool.run(ctx, tool.parse(args))


def test_the_modules_tools_are_collected(client: TestClient) -> None:
    names = {t.name for t in agent(client).registry}

    assert {"train_model", "compare_models", "tune_model", "explain_model", "list_ml_runs"} <= names
    assert {"analyze_time_series", "forecast"} <= names
    assert {"suggest_kpis", "compute_kpi", "save_kpi"} <= names
    assert {"plan_synthetic", "generate_synthetic", "save_synthetic_dataset"} <= names
    assert {"analyze_experiment", "bandit_view", "sample_size"} <= names
    assert "export_report" in names
    kinds = {t.name: t.kind for t in agent(client).registry}
    assert kinds["save_kpi"] == kinds["save_synthetic_dataset"] == "write"
    assert kinds["train_model"] == "compute"


def test_train_explain_and_list_runs(client: TestClient, traps: str) -> None:
    trained = run_tool(client, traps, "train_model", {"target": "income"})
    assert trained.content["task"] == "regression" and trained.content["score"] is not None
    assert trained.link.page == "ml" and trained.link.params["run"] == trained.content["run_id"]

    explained = run_tool(client, traps, "explain_model", {"run_id": trained.content["run_id"]})
    assert explained.content["permutation"]["items"]

    runs = run_tool(client, traps, "list_ml_runs", {"target": "income"})
    assert runs.content["runs"][0]["run_id"] == trained.content["run_id"]

    with pytest.raises(ToolError, match="Unknown"):
        run_tool(client, traps, "train_model", {"target": "nope"})


def test_compare_models_ranks_by_the_metric(client: TestClient, traps: str) -> None:
    compared = run_tool(
        client,
        traps,
        "compare_models",
        {"target": "income", "models": ["linear_regression", "random_forest"]},
    )

    ranking = compared.content["ranking"]
    assert {r["model"] for r in ranking} == {"linear_regression", "random_forest"}
    scores = [r["score"] for r in ranking]
    expected = (
        sorted(scores) if compared.content["lower_is_better"] else sorted(scores, reverse=True)
    )
    assert scores == expected


def test_time_series_analysis_and_forecast(client: TestClient, sales: str) -> None:
    analysed = run_tool(client, sales, "analyze_time_series", {"time": "date", "value": "units"})
    assert analysed.content["trend"] and "stationarity" in analysed.content
    assert analysed.link.params == {"time": "date", "value": "units"}

    forecast = run_tool(
        client,
        sales,
        "forecast",
        {"time": "date", "value": "units", "period": "week", "horizon": 4},
    )
    assert forecast.summary == "Forecast ready" and forecast.content

    # 028 A4: drivers (a yes/no column and the calendar) feed the driver models
    driven = run_tool(
        client,
        sales,
        "forecast",
        {
            "time": "date",
            "value": "units",
            "period": "month",
            "horizon": 3,
            "models": ["naive", "tslm"],
            "drivers": [{"column": "promo", "known": True, "future": "last"}],
        },
    )
    tslm = next(r for r in driven.content["results"] if r["key"] == "tslm")
    assert tslm["uses_drivers"] and tslm["drivers"][0]["driver"] == "promo"
    with pytest.raises(ToolError):
        run_tool(client, sales, "forecast", {"time": "date", "models": ["nope"]})


def test_forecast_every_group_with_coherent_totals(client: TestClient, sales: str) -> None:
    """039: the forecasting tool gains every_group and reconcile."""
    found = run_tool(
        client,
        sales,
        "forecast",
        {
            "time": "date",
            "value": "units",
            "period": "month",
            "horizon": 3,
            "group": "store",
            "every_group": True,
            "reconcile": "bottom_up",
            "models": ["naive", "seasonal_naive"],
        },
    )

    stores = [r for r in found.content["rows"] if r["level"] == "store"]
    assert len(stores) == found.content["series"]["kept"] >= 2
    assert found.content["hierarchy"]["method"] == "bottom_up"
    assert found.content["hierarchy"]["gap"] < 1e-6
    assert found.link.params["group"] == "store"
    with pytest.raises(ToolError, match="group"):
        run_tool(client, sales, "forecast", {"time": "date", "every_group": True})


def test_kpis_are_suggested_and_computed(client: TestClient, sales: str) -> None:
    suggested = run_tool(client, sales, "suggest_kpis", {})
    first = suggested.content["kpis"][0]

    computed = run_tool(client, sales, "compute_kpi", {"definition": first["definition"]})

    assert computed.content["tiles"]
    with pytest.raises(ToolError):
        run_tool(
            client,
            sales,
            "compute_kpi",
            {"definition": {"measures": [{"id": "m1", "agg": "sum", "column": "nope"}]}},
        )


def test_report_tool_gives_a_download_link(client: TestClient, traps: str) -> None:
    exported = run_tool(client, traps, "export_report", {})

    assert exported.link.page == "report_file"
    file = client.get(f"/api/v1/reports/jobs/{exported.link.params['job']}/file")
    assert file.status_code == 200 and file.text.startswith("<!doctype html>")


def test_synthetic_plan_and_generation(client: TestClient, traps: str) -> None:
    plan = run_tool(client, traps, "plan_synthetic", {})
    assert plan.content["method"]

    generated = run_tool(
        client, traps, "generate_synthetic", {"method": "independent", "rows": 200}
    )
    assert generated.content["rows"] == 200 and generated.content["scores"]
    assert generated.link.params["run"] == generated.content["run_id"]


# --- confirmations ----------------------------------------------------------------------------


def _turn_with_confirmation(client: TestClient, dataset_id: str, approve: bool) -> tuple[list, Any]:
    definition = run_tool(client, dataset_id, "suggest_kpis", {}).content["kpis"][0]["definition"]
    service = agent(client)
    provider = ScriptedProvider(
        [calls(("save_kpi", {"name": "Units per week", "definition": definition})), answer("Done.")]
    )
    service.provider_factory = lambda *args: provider
    service.store.save_key("sk-ant-test-1234abcd")
    cid = client.post(f"/api/v1/modules/agent/datasets/{dataset_id}/conversations").json()["id"]

    run = service.send(cid, "Save it", [])
    seen: list = []
    for event in service.events(run, heartbeat=0.5):
        if event is None:
            continue
        seen.append(event)
        if event["type"] == "confirm":
            assert event["text"] == "Save the KPI “Units per week” in the KPI builder."
            response = client.post(
                f"/api/v1/modules/agent/conversations/{cid}/confirm/{event['id']}",
                json={"approved": approve},
            )
            assert response.status_code == 204
    return seen, client.get(f"/api/v1/modules/kpi/datasets/{dataset_id}/definitions").json()


def test_a_write_runs_only_after_the_user_confirms(client: TestClient, sales: str) -> None:
    seen, saved = _turn_with_confirmation(client, sales, approve=True)

    assert any(e["type"] == "confirm" for e in seen)
    assert [k["name"] for k in saved] == ["Units per week"]
    done = next(e for e in seen if e["type"] == "step_done")
    assert done["ok"] and done["link"]["params"]["kpi"] == saved[0]["id"]


def test_a_dismissed_write_does_nothing(client: TestClient, sales: str) -> None:
    seen, saved = _turn_with_confirmation(client, sales, approve=False)

    assert saved == []
    done = next(e for e in seen if e["type"] == "step_done")
    assert not done["ok"] and "did not allow" in done["summary"]


def test_an_unknown_confirmation_is_refused(client: TestClient, traps: str) -> None:
    cid = client.post(f"/api/v1/modules/agent/datasets/{traps}/conversations").json()["id"]
    response = client.post(
        f"/api/v1/modules/agent/conversations/{cid}/confirm/nope", json={"approved": True}
    )
    assert response.status_code == 404


def test_analyze_an_experiment_and_plan_the_next(client: TestClient) -> None:
    ab = import_path(client, default_samples_dir() / "ab_test.csv")

    analyzed = run_tool(
        client,
        ab,
        "analyze_experiment",
        {"variant": "variant", "metrics": ["converted", "revenue"], "unit": "user_id"},
    )
    first = analyzed.content["metrics"][0]["comparisons"][0]
    assert first["sentence"].startswith("new_checkout reaches") and first["verdict"] == "better"
    assert analyzed.content["correction"] == "p adjusted for 2 comparisons (Holm)"
    assert analyzed.link.page == "experiments" and analyzed.link.params["unit"] == "user_id"
    with pytest.raises(ToolError, match="2 to 5 variants"):
        run_tool(client, ab, "analyze_experiment", {"variant": "user_id", "metrics": ["converted"]})

    bandit = run_tool(
        client,
        ab,
        "bandit_view",
        {"variant": "variant", "metrics": ["converted"], "unit": "user_id", "time": "date"},
    )
    assert bandit.summary.startswith("P(new_checkout is best)")
    assert bandit.content["variants"][1]["p_best"] > 0.9
    assert "replay" in bandit.content and "guardrail" in bandit.content
    assert "reinforcement" not in str(bandit.content).lower()
    ratio = run_tool(
        client,
        ab,
        "bandit_view",
        {"variant": "variant", "metrics": [{"column": "revenue", "denominator": "orders"}]},
    )
    assert ratio.summary == "No bandit view" and "ratio" in ratio.content["why"]

    planned = run_tool(client, ab, "sample_size", {"baseline": 0.2, "effect": 0.1})
    assert planned.content["control"] > 1_000
    assert planned.summary.endswith("in total") and planned.link.params == {"tab": "plan"}
