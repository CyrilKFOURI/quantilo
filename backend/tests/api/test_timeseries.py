import json
import math
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import default_samples_dir
from tests.api.test_profiling import import_path

BASE = "/api/v1/modules/timeseries/datasets"


@pytest.fixture
def sales(client: TestClient) -> str:
    return import_path(client, default_samples_dir() / "sales_timeseries.csv")


def get(client: TestClient, dataset_id: str, section: str, **params):
    return client.get(f"{BASE}/{dataset_id}/{section}", params=params)


def test_overview_opens_already_configured(client: TestClient, sales: str) -> None:
    result = get(client, sales, "overview").json()
    setup = result["setup"]

    assert (setup["time"], setup["value"]) == ("date", "units")  # first quantity / amount
    assert setup["aggregate"] == "sum" and setup["period"] == "week" and setup["coarsened"]
    assert setup["season"] == 52 and setup["rows"] == setup["total"] == 3285
    assert len(result["periods"]) == setup["periods"] == len(result["analyzed"]["values"])
    assert result["window"] == 4 and len(result["analyzed"]["rolling_mean"]) == setup["periods"]
    assert [m["key"] for m in result["methods"]] == ["mann_kendall", "sen"]
    assert result["figures"]["start"] <= result["figures"]["end"]


def test_overview_choices_methods_and_groups(client: TestClient, sales: str) -> None:
    result = get(
        client,
        sales,
        "overview",
        value="revenue",
        period="month",
        agg="mean",
        group="store",
        series="Lyon",
        methods="hamed_rao,cox_stuart,linear_trend",
    ).json()

    assert result["setup"]["period"] == "month" and result["setup"]["aggregate"] == "mean"
    assert result["setup"]["series"] == "Lyon" and result["analyzed"]["label"] == "Lyon"
    assert "Lyon" in result["setup"]["groups"] and len(result["lines"]) >= 2
    assert [m["key"] for m in result["methods"]] == ["hamed_rao", "cox_stuart", "linear_trend"]


def test_seasonality_and_decomposition(client: TestClient, sales: str) -> None:
    season = get(client, sales, "seasonality", period="day", value="revenue").json()

    keys = [p["key"] for p in season["profiles"]]
    assert keys == ["weekday", "month"]
    assert season["overlay"]["unit"] == "week" and len(season["overlay"]["years"]) >= 2
    assert [m["key"] for m in season["methods"]] == ["kruskal_season", "periodogram"]

    parts = get(client, sales, "decomposition", value="revenue").json()
    assert parts["result"]["method"] == "stl" and parts["result"]["seasons"] == [52]
    assert len(parts["result"]["trend"]) == len(parts["periods"])
    assert parts["result"]["strength_seasonal"] is not None

    classical = get(client, sales, "decomposition", value="revenue", method="classical_additive")
    assert classical.json()["result"]["method"] == "classical_additive"


def test_gaps_and_season_override(client: TestClient, sales: str) -> None:
    result = get(client, sales, "decomposition", period="month", season=6).json()

    assert result["setup"]["season"] == 6 and result["result"]["seasons"] == [6]
    assert get(client, sales, "overview", gaps="interpolate").json()["setup"]["gaps"] == (
        "interpolate"
    )


def test_filters_apply(client: TestClient, sales: str) -> None:
    only = json.dumps([{"column": "store", "op": "in", "values": ["Paris"]}])

    result = get(client, sales, "overview", filters=only).json()

    assert result["setup"]["total"] < 3285


def test_errors_are_422(client: TestClient, sales: str) -> None:
    unknown = get(client, sales, "overview", methods="astrology")
    assert unknown.status_code == 422 and unknown.json()["error"]["code"] == "unknown_method"
    assert get(client, sales, "overview", gaps="guess").status_code == 422
    assert get(client, sales, "decomposition", method="x13").status_code == 422


def test_stationarity_and_autocorrelation(client: TestClient, sales: str) -> None:
    raw = get(client, sales, "stationarity", value="revenue").json()

    assert [m["key"] for m in raw["methods"]] == ["adf", "kpss", "phillips_perron"]
    assert raw["synthesis"]["verdict"] in (
        "stationary",
        "unit_root",
        "trend_or_difference",
        "inconclusive",
    )
    assert raw["suggestion"] is None or raw["suggestion"]["verdict"] == "stationary"
    assert len(raw["transformed"]) == len(raw["periods"]) == raw["setup"]["periods"] - 2

    diff = get(
        client, sales, "stationarity", value="revenue", transform="diff", methods="df_gls,rur"
    )
    body = diff.json()
    assert len(body["transformed"]) == body["setup"]["periods"] - 3  # 2 partial weeks, 1 difference
    assert [m["key"] for m in body["methods"]] == ["df_gls", "rur"]
    assert body["synthesis"] is not None  # ADF and KPSS run for the synthesis anyway

    memory = get(client, sales, "autocorrelation", value="revenue", pacf="burg").json()
    assert len(memory["correlogram"]["acf"]) == len(memory["correlogram"]["lags"])
    assert [m["key"] for m in memory["methods"]] == ["ljung_box", "durbin_watson"]
    assert get(client, sales, "autocorrelation", transform="cube").status_code == 422


def test_volatility_residuals_and_breaks(client: TestClient, sales: str) -> None:
    spread = get(
        client, sales, "volatility", value="revenue", methods="breusch_pagan,levene"
    ).json()
    assert spread["residuals_from"] == "STL remainder"
    # incomplete first and last weeks are left out of the analyses
    assert len(spread["residuals"]) == len(spread["rolling_std"]) == len(spread["periods"])
    assert len(spread["periods"]) == spread["setup"]["periods"] - 2
    assert [m["key"] for m in spread["methods"]] == ["breusch_pagan", "levene"]

    for source in ("residuals", "series", "returns"):
        shape = get(client, sales, "residuals", value="revenue", source=source).json()
        assert shape["distribution"]["source"] == source
        assert shape["distribution"]["fit"]["laws"] and shape["distribution"]["normality"]["tests"]

    found = get(
        client, sales, "breaks", value="revenue", sensitivity="high", anomaly="hampel"
    ).json()
    assert found["deseasonalized"] is True and found["breaks"]["search"] == "pelt"
    assert [m["key"] for m in found["methods"]] == ["pettitt"]
    for row in found["anomalies"]:
        assert row["start"] <= row["end"] and row["period"] in found["periods"]
    assert get(client, sales, "breaks", threshold=9).status_code == 422


def test_forecast_job(client: TestClient, sales: str) -> None:
    started = client.post(
        f"{BASE}/{sales}/forecast", params={"value": "revenue", "period": "month", "horizon": 6}
    )
    assert started.status_code == 202, started.text
    job = started.json()
    for _ in range(600):
        job = client.get(f"/api/v1/modules/timeseries/forecast-jobs/{job['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.1)
    assert job["status"] == "done", job
    result = job["result"]
    assert result["horizon"] == 6 and len(result["future"]) == 6
    assert result["future"][0] > result["periods"][-1]
    keys = [r["key"] for r in result["results"]]
    assert set(keys) == {"naive", "seasonal_naive", "ets_auto", "arima_auto"}
    assert result["best"] == keys[0] and result["reference"] == "seasonal_naive"
    best = result["results"][0]
    assert len(best["forecast"]) == 6 and best["scores"]["mase"] is not None

    again = client.post(
        f"{BASE}/{sales}/forecast", params={"value": "revenue", "period": "month", "horizon": 6}
    ).json()
    assert again["id"] == job["id"]  # the finished forecast is reused

    manual = client.post(
        f"{BASE}/{sales}/forecast",
        params={"value": "revenue", "period": "month", "methods": "sarima", "order": "1,1,9"},
    )
    assert manual.status_code == 422
    assert client.get("/api/v1/modules/timeseries/forecast-jobs/nope").status_code == 422


def wait(client: TestClient, job: dict) -> dict:
    for _ in range(1200):
        job = client.get(f"/api/v1/modules/timeseries/forecast-jobs/{job['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.1)
    return job


@pytest.fixture
def shop(client: TestClient, tmp_path: Path) -> str:
    """Monthly sales driven by the price; the last 4 rows are planned prices without sales."""
    lines = ["month,sales,price"]
    for i in range(64):
        price = 20 + (i % 7)
        sales = "" if i >= 60 else f"{400 - 10 * price + (i % 3):.1f}"
        lines.append(f"{2020 + i // 12}-{i % 12 + 1:02d}-01,{sales},{price}")
    path = tmp_path / "shop.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return import_path(client, path)


def test_forecast_with_drivers(client: TestClient, shop: str) -> None:
    params = {"time": "month", "value": "sales", "period": "month", "horizon": 4}
    candidates = get(client, shop, "drivers", **params).json()
    (price,) = candidates["candidates"]
    assert price["column"] == "price" and price["future_periods"] == 4 and price["suggested"]
    assert candidates["calendar"] == []

    body = {"drivers": [{"column": "price", "known": True}]}
    started = client.post(
        f"{BASE}/{shop}/forecast", params={**params, "methods": "naive,tslm"}, json=body
    )
    assert started.status_code == 202, started.text
    job = wait(client, started.json())
    assert job["status"] == "done", job
    result = job["result"]
    assert result["periods"][-1] == "2024-12" and result["future"][0] == "2025-01"
    (info,) = result["drivers"]
    assert info["source"] == "file" and info["future_values"] == [24.0, 25.0, 26.0, 20.0]
    tslm = next(r for r in result["results"] if r["key"] == "tslm")
    assert tslm["uses_drivers"] and tslm["drivers"][0]["verdict"] == "helps"
    assert tslm["drivers"][0]["coefficient"] == pytest.approx(-10, abs=0.5)
    assert tslm["mase_without_drivers"] > tslm["scores"]["mase"]
    assert result["best"] == "tslm"

    bad = [{"column": "sales"}], [{"column": "weekday", "kind": "calendar"}], [{"column": "nope"}]
    for drivers in bad:
        refused = client.post(f"{BASE}/{shop}/forecast", params=params, json={"drivers": drivers})
        assert refused.status_code == 422, refused.text


def test_ar_order_and_volatility_window(client: TestClient, sales: str) -> None:
    auto = get(client, sales, "autocorrelation", value="revenue").json()["ar"]
    assert auto["chosen_by"] == "AIC" and len(auto["orders"]) >= 5
    assert auto["order"] == min(auto["orders"], key=lambda o: o["aic"])["order"]

    fixed = get(client, sales, "autocorrelation", value="revenue", ar=3).json()["ar"]
    assert fixed["order"] == 3 and fixed["chosen_by"] == "you"
    assert [c["term"] for c in fixed["coefficients"]] == ["const", "lag 1", "lag 2", "lag 3"]

    spread = get(client, sales, "volatility", value="revenue", spread=10).json()
    assert spread["window"] == 10
    assert get(client, sales, "volatility", spread=1).status_code == 422


@pytest.mark.slow
def test_five_million_rows(client: TestClient, tmp_path: Path) -> None:
    import duckdb

    path = tmp_path / "big.parquet"
    duckdb.execute(
        "COPY (SELECT range AS id, random() * 100 AS amount, "
        "DATE '2015-01-01' + CAST(range % 3650 AS INTEGER) AS day FROM range(5000000)) "
        f"TO '{path.as_posix()}' (FORMAT parquet)"
    )
    dataset_id = import_path(client, path)

    start = time.perf_counter()
    response = get(client, dataset_id, "overview", time="day", value="amount")
    seconds = time.perf_counter() - start
    assert response.status_code == 200, response.text
    assert seconds < 2, seconds


# --- Every group, coherent totals, scenarios (039) ------------------------------------------------


@pytest.fixture
def stores(client: TestClient, tmp_path: Path) -> str:
    """30 stores in 3 regions, monthly sales over four years."""
    lines = ["date,montant,magasin,region"]
    for s in range(30):
        region = ("North", "South", "East")[s % 3]
        for m in range(48):
            amount = 100 + 5 * s + 20 * math.sin(2 * math.pi * m / 12) + (s * m) % 7
            lines.append(f"{2021 + m // 12}-{m % 12 + 1:02d}-05,{amount:.2f},S{s:02d},{region}")
    path = tmp_path / "stores.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return import_path(client, path)


def wait_groups(client: TestClient, job: dict) -> dict:
    for _ in range(1200):
        job = client.get(f"/api/v1/modules/timeseries/group-forecast-jobs/{job['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.1)
    return job


def test_forecast_every_group(client: TestClient, stores: str) -> None:
    """Spec 039: 30 stores → "Forecast every group" is offered and gives 30 forecasts; with
    store → region, the hierarchy is recognised and the totals add up at every date."""
    params = {"time": "date", "value": "montant", "group": "magasin", "period": "month"}
    setup = get(client, stores, "groups", **params).json()
    assert setup["groups"] == 30 and setup["levels"] == ["region"] and setup["summable"]
    assert setup["limit"] == 200 and not setup["browser"]

    started = client.post(
        f"{BASE}/{stores}/forecast-groups",
        params={**params, "horizon": 3, "methods": "naive,seasonal_naive", "reconcile": "auto"},
    )
    assert started.status_code == 202, started.text
    job = wait_groups(client, started.json())
    assert job["status"] == "done", job
    assert job["done"] == job["total"] == 34  # 30 stores, 3 regions, the total
    result = job["result"]
    leaves = [r for r in result["rows"] if r["level"] == "magasin"]
    assert len(leaves) == 30 and result["kept"] == 30 and result["pooled"] == 0
    hierarchy = result["hierarchy"]
    assert hierarchy["levels"] == ["Total", "region", "magasin"] and hierarchy["gap"] < 1e-6
    for region in (r for r in result["rows"] if r["level"] == "region"):
        children = [r["forecast"] for r in leaves if r["parent"] == region["label"]]
        for step in range(3):
            assert abs(sum(c[step] for c in children) - region["forecast"][step]) < 1e-6

    again = client.post(
        f"{BASE}/{stores}/forecast-groups",
        params={**params, "horizon": 3, "methods": "naive,seasonal_naive", "reconcile": "auto"},
    ).json()
    assert again["id"] == job["id"]  # reused

    csv = client.get(f"/api/v1/modules/timeseries/group-forecast-jobs/{job['id']}/forecast.csv")
    assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
    lines = csv.text.splitlines()
    assert lines[0] == "group,level,period,forecast,low80,high80,low95,high95,model"
    assert len(lines) == 1 + 34 * 3

    refused = client.post(
        f"{BASE}/{stores}/forecast-groups",
        params={**params, "reconcile": "bottom_up", "levels": "date"},
    )
    assert refused.status_code == 422
    no_group = client.post(f"{BASE}/{stores}/forecast-groups", params={"value": "montant"})
    assert no_group.status_code == 422 and no_group.json()["error"]["code"] == "needs_group"
    unknown = client.get("/api/v1/modules/timeseries/group-forecast-jobs/nope")
    assert unknown.status_code == 422


def test_scenario_route(client: TestClient, shop: str) -> None:
    """Spec 039: a price cut on one month moves only that month, upwards (sales fall with the
    price coefficient of about -10)."""
    params = {"time": "month", "value": "sales", "period": "month", "horizon": 4}
    body: dict = {"drivers": [{"column": "price", "known": True}]}
    base = client.post(f"{BASE}/{shop}/scenario", params=params, json=body)
    assert base.status_code == 200, base.text
    body["changes"] = [
        {"column": "price", "how": "percent", "amount": -10, "start": "2025-02", "end": "2025-02"}
    ]
    cut = client.post(f"{BASE}/{shop}/scenario", params=params, json=body).json()

    before, after = base.json()["forecast"], cut["forecast"]
    assert after[1] > before[1] + 10  # 10 % of 25 = 2.5 units, -10 per unit
    assert [after[i] for i in (0, 2, 3)] == pytest.approx([before[i] for i in (0, 2, 3)])
    assert cut["drivers"][0]["future_values"][1] == pytest.approx(22.5)

    wrong = client.post(f"{BASE}/{shop}/scenario", params={**params, "model": "naive"}, json=body)
    assert wrong.status_code == 422


def test_smoothing_route(client: TestClient, sales: str) -> None:
    only = get(client, sales, "smoothing", period="month").json()
    assert only["curves"] == [] and "hp" not in only["unavailable"]

    result = get(
        client,
        sales,
        "smoothing",
        period="month",
        methods="ewma,kalman_filtered,hp",
        ewma_alpha=0.4,
        kalman_model="llevel",
    ).json()
    curves = {c["key"]: c for c in result["curves"]}
    assert list(curves) == ["ewma", "kalman_filtered", "hp"]
    assert curves["ewma"]["params"][0] == {
        "key": "alpha",
        "value": 0.4,
        "auto": False,
        "reason": "set in the options",
    }
    assert curves["kalman_filtered"]["causal"] and not curves["hp"]["causal"]
    assert curves["hp"]["params"][0]["value"] == 129_600
    assert get(client, sales, "smoothing", methods="magic").status_code == 422
    assert get(client, sales, "smoothing", kalman_model="arima").status_code == 422


def test_kalman_gaps_anomalies_and_structural_model(client: TestClient, sales: str) -> None:
    filled = get(client, sales, "overview", gaps="kalman").json()
    assert filled["setup"]["gaps"] == "kalman"
    assert len(filled["analyzed"]["filled_low"]) == filled["setup"]["periods"]

    breaks = get(client, sales, "breaks", anomaly="kalman").json()
    assert breaks["anomaly_method"] == "kalman"
    assert breaks["residuals_from"].startswith("Kalman")

    job = client.post(
        f"{BASE}/{sales}/forecast", params={"methods": "naive,structural", "horizon": 4}
    ).json()
    done = wait(client, job)
    structural = next(m for m in done["result"]["results"] if m["key"] == "structural")
    assert structural["info"].startswith("UCM(") and structural["scores"] is not None
