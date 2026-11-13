"""The app and the Python library compute the same results: one engine, two front doors."""

import math
import time
from pathlib import Path
from typing import Any

import pytest
import quantilo as dx
from fastapi.testclient import TestClient

from app.config import default_samples_dir
from tests.api.test_profiling import import_path

TRAPS = default_samples_dir() / "typing_traps.csv"


def same(api: Any, library: Any, path: str = "") -> None:
    """Equal field by field; floats to 1e-9 (JSON round trip)."""
    if isinstance(api, dict):
        assert isinstance(library, dict), path
        assert set(api) == set(library), f"{path}: {set(api) ^ set(library)}"
        for key in api:
            same(api[key], library[key], f"{path}.{key}")
    elif isinstance(api, list):
        assert isinstance(library, list) and len(api) == len(library), path
        for i, (a, b) in enumerate(zip(api, library, strict=True)):
            same(a, b, f"{path}[{i}]")
    elif isinstance(api, float) or isinstance(library, float):
        assert api is not None and library is not None, f"{path}: {api} != {library}"
        assert math.isclose(api, library, rel_tol=1e-9, abs_tol=1e-12), (
            f"{path}: {api} != {library}"
        )
    else:
        assert api == library, f"{path}: {api!r} != {library!r}"


@pytest.fixture
def both(client: TestClient, tmp_path: Path) -> tuple[str, dx.Dataset]:
    return import_path(client, TRAPS), dx.open(TRAPS, cache_dir=tmp_path)


def test_types_are_the_same(client: TestClient, both: tuple[str, dx.Dataset]) -> None:
    dataset_id, ds = both
    columns = client.get(f"/api/v1/datasets/{dataset_id}/columns").json()
    app = {c["name"]: (c["type"], c["role"], c["parser"]) for c in columns}
    library = {c.name: (c.type, c.role, c.parser.model_dump()) for c in ds.types}
    assert app == library


def test_analyses_are_the_same(client: TestClient, both: tuple[str, dx.Dataset]) -> None:
    dataset_id, ds = both
    base = f"/api/v1/datasets/{dataset_id}"
    same(
        client.get(f"{base}/variable", params={"column": "price_eur"}).json(),
        dx.to_dict(ds.variable("price_eur")),
        "variable",
    )
    same(
        client.get(f"{base}/variable/fit", params={"column": "age"}).json(),
        dx.to_dict(ds.fit("age")),
        "fit",
    )
    same(
        client.get(f"{base}/variable/tail", params={"column": "income"}).json(),
        dx.to_dict(ds.tail("income")),
        "tail",
    )
    same(
        client.get(f"{base}/compare", params={"x": "income", "y": "segment"}).json(),
        dx.to_dict(ds.compare("income", "segment")),
        "compare",
    )
    same(client.get(f"{base}/correlations").json(), dx.to_dict(ds.correlations()), "correlations")
    same(
        client.get(f"{base}/correlations/structure").json(),
        dx.to_dict(ds.structure()),
        "structure",
    )
    same(client.get(f"{base}/correlations/factors").json(), dx.to_dict(ds.factors()), "factors")
    same(
        client.get(
            f"{base}/regression", params={"target": "price_eur", "columns": "age,income,segment"}
        ).json(),
        dx.to_dict(ds.regression("price_eur", ["age", "income", "segment"])),
        "regression",
    )


def test_filtered_analyses_are_the_same(client: TestClient, both: tuple[str, dx.Dataset]) -> None:
    dataset_id, ds = both
    condition = dx.Condition(column="country", op="in", values=["FR"])
    filters = f"[{condition.model_dump_json()}]"
    api = client.get(
        f"/api/v1/datasets/{dataset_id}/variable", params={"column": "age", "filters": filters}
    ).json()
    same(api, dx.to_dict(ds.where(condition).variable("age")), "filtered variable")


def test_version_changes_are_the_same(client: TestClient, tmp_path: Path) -> None:
    from tests.api.test_changes import orders, write

    october = write(tmp_path / "october.csv", orders())
    november = write(tmp_path / "november.csv", orders(factor=1.3, region="Outre-mer"))
    api = client.get(
        f"/api/v1/datasets/{import_path(client, november)}/changes",
        params={"vs": import_path(client, october)},
    ).json()
    previous = dx.open(october, cache_dir=tmp_path / "o")
    library = dx.to_dict(dx.open(november, cache_dir=tmp_path / "n").changes(previous))
    for side in ("before", "after"):  # when each photo was taken differs, nothing else
        assert api.pop(side)["rows"] == library.pop(side)["rows"]
    same(api, library, "changes")


def test_experiments_are_the_same(client: TestClient, tmp_path: Path) -> None:
    sample = default_samples_dir() / "ab_test.csv"
    dataset_id, ds = import_path(client, sample), dx.open(sample, cache_dir=tmp_path)
    base = f"/api/v1/modules/experiments/datasets/{dataset_id}"
    design = {
        "variant": "variant",
        "metrics": [{"column": "converted"}, {"column": "revenue", "pre": "spend_before"}],
        "unit": "user_id",
        "segment": "device",
        "time": "date",
    }
    library = ds.experiment(
        "variant",
        ["converted", {"column": "revenue", "pre": "spend_before"}],
        unit="user_id",
        segment="device",
        time="date",
    ).analyze()
    same(client.post(f"{base}/analyze", json=design).json(), dx.to_dict(library), "experiment")
    same(client.get(f"{base}/candidates").json(), dx.to_dict(ds.experiment_candidates()), "setup")
    plan = {"baseline": 0.2, "effect": 0.1, "daily": 400}
    same(
        client.post("/api/v1/modules/experiments/sample-size", json=plan).json(),
        dx.to_dict(dx.sample_size(0.2, 0.1, daily=400)),
        "sample size",
    )
    same(
        client.post("/api/v1/modules/experiments/bandit-simulation", json=plan).json(),
        dx.to_dict(dx.bandit_simulation(0.2, 0.1, daily=400)),
        "bandit simulation",
    )
    policy = {"variant": "variant", "metrics": [{"column": "converted"}], "unit": "user_id"}
    same(
        client.post(f"{base}/policy", json={"design": policy}).json(),
        dx.to_dict(ds.experiment("variant", ["converted"], unit="user_id").policy()),
        "policy",
    )
    same(
        client.get(f"{base}/decisions").json(),
        dx.to_dict(ds.logged_decisions()),
        "logged decisions",
    )


def _without_timings(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_timings(v) for k, v in value.items() if k != "seconds"}
    if isinstance(value, list):
        return [_without_timings(v) for v in value]
    return value


def test_forecast_with_drivers_is_the_same(client: TestClient, tmp_path: Path) -> None:
    """028 A4: the app's forecast with drivers equals `series.forecast(..., drivers=[...])`."""
    path = default_samples_dir() / "sales_timeseries.csv"
    dataset_id = import_path(client, path)
    ds = dx.open(path, cache_dir=tmp_path)
    params = {"time": "date", "value": "revenue", "period": "week", "horizon": 4}
    drivers: list[Any] = [
        {"column": "units"},
        {"column": "promo", "kind": "flag"},
        {"column": "month", "kind": "calendar"},
    ]
    started = client.post(
        f"/api/v1/modules/timeseries/datasets/{dataset_id}/forecast",
        params={**params, "methods": "naive,tslm"},
        json={"drivers": drivers},
    ).json()
    job = started
    for _ in range(1200):
        job = client.get(f"/api/v1/modules/timeseries/forecast-jobs/{started['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.1)
    assert job["status"] == "done", job

    library = ds.timeseries(time="date", value="revenue", period="week").forecast(
        horizon=4, models=["naive", "tslm"], drivers=drivers
    )

    same(_without_timings(job["result"]), _without_timings(dx.to_dict(library)), "forecast")
    candidates = client.get(
        f"/api/v1/modules/timeseries/datasets/{dataset_id}/drivers", params=params
    ).json()
    same(
        candidates,
        dx.to_dict(ds.timeseries(time="date", value="revenue", period="week").drivers(4)),
        "drivers",
    )


def test_forecast_with_holidays_and_tuning_is_the_same(client: TestClient, tmp_path: Path) -> None:
    """028 A5 / A7: `series.forecast(..., holidays="FR", tuning={...})` equals the app."""
    import datetime as dt

    start = dt.date(2024, 1, 1)
    lines = ["day,visits"] + [
        f"{start + dt.timedelta(days=i):%Y-%m-%d},{100 + (i % 7) * 5 + i * 0.1:.1f}"
        for i in range(200)
    ]
    path = tmp_path / "visits.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    dataset_id = import_path(client, path)
    ds = dx.open(path, cache_dir=tmp_path / "cache")
    params = {"time": "day", "value": "visits", "period": "day", "horizon": 7}
    tuning = {"changepoint_scale": 0.1, "time_limit": 120}
    started = client.post(
        f"/api/v1/modules/timeseries/datasets/{dataset_id}/forecast",
        params={**params, "methods": "naive,prophet_like"},
        json={"drivers": [{"column": "FR", "kind": "holidays", "known": True}], "tuning": tuning},
    ).json()
    job = started
    for _ in range(1200):
        job = client.get(f"/api/v1/modules/timeseries/forecast-jobs/{started['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.1)
    assert job["status"] == "done", job

    library = ds.timeseries(time="day", value="visits", period="day").forecast(
        horizon=7, models=["naive", "prophet_like"], holidays="FR", tuning=tuning
    )

    same(_without_timings(job["result"]), _without_timings(dx.to_dict(library)), "forecast")


def test_forecast_every_group_is_the_same(client: TestClient, tmp_path: Path) -> None:
    """039: the app's forecast of every group with MinT equals
    `ds.timeseries("date", "montant", group="magasin").forecast_groups(horizon=3,
    reconcile="mint_shrink")`, and the scenario route equals `TimeSeries.scenario`."""
    lines = ["date,montant,magasin,region"]
    for s in range(12):
        for m in range(40):
            amount = 80 + 6 * s + 15 * math.sin(2 * math.pi * m / 12) + (s * m) % 5
            lines.append(f"{2021 + m // 12}-{m % 12 + 1:02d}-07,{amount:.2f},S{s:02d},R{s % 3}")
    path = tmp_path / "stores.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    dataset_id = import_path(client, path)
    ds = dx.open(path, cache_dir=tmp_path / "cache")
    params = {"time": "date", "value": "montant", "group": "magasin"}
    base = "/api/v1/modules/timeseries"

    started = client.post(
        f"{base}/datasets/{dataset_id}/forecast-groups",
        params={**params, "horizon": 3, "reconcile": "mint_shrink"},
    ).json()
    job = started
    for _ in range(3000):
        job = client.get(f"{base}/group-forecast-jobs/{started['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.1)
    assert job["status"] == "done", job

    library = ds.timeseries("date", "montant", group="magasin").forecast_groups(
        horizon=3, reconcile="mint_shrink"
    )

    same(_without_timings(job["result"]), _without_timings(dx.to_dict(library)), "groups")
    assert library.hierarchy is not None and library.hierarchy.levels == [
        "Total",
        "region",
        "magasin",
    ]
    same(
        client.get(f"{base}/datasets/{dataset_id}/groups", params=params).json(),
        dx.to_dict(ds.timeseries("date", "montant", group="magasin").groups()),
        "group setup",
    )


def test_scenario_is_the_same(client: TestClient, tmp_path: Path) -> None:
    lines = ["month,sales,price"]
    for i in range(64):
        price = 20 + (i % 7)
        sales = "" if i >= 60 else f"{400 - 10 * price + (i % 3):.1f}"
        lines.append(f"{2020 + i // 12}-{i % 12 + 1:02d}-01,{sales},{price}")
    path = tmp_path / "shop.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    dataset_id = import_path(client, path)
    ds = dx.open(path, cache_dir=tmp_path / "cache")
    change = {"column": "price", "how": "value", "amount": 30, "start": "2025-03"}

    api = client.post(
        f"/api/v1/modules/timeseries/datasets/{dataset_id}/scenario",
        params={"time": "month", "value": "sales", "period": "month", "horizon": 4},
        json={"drivers": [{"column": "price", "known": True}], "changes": [change]},
    ).json()
    library = ds.timeseries(time="month", value="sales", period="month").scenario(
        [dx.Driver("price", known=True)], horizon=4, changes=[change]
    )

    same(api, dx.to_dict(library), "scenario")
