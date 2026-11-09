"""Drivers of a forecast (028 A4): known in advance or past only, calendar, flags, scenarios."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import quantilo as dx
from quantilo.timeseries import drivers as dv
from quantilo.timeseries import forecast as fc

RNG = np.random.default_rng(28)
N, H = 96, 12
MONTHS = [datetime(2017 + i // 12, i % 12 + 1, 1) for i in range(N + H)]

# A driver that generates the series: y = 40 + 3 x + noise, x wandering (not predictable).
X = np.cumsum(RNG.normal(0, 2, N + H)) + 50
Y = 40 + 3 * X[:N] + RNG.normal(0, 1, N)
NOISE = RNG.normal(0, 1, N + H)
SEASONAL = 100 + 10 * np.sin(2 * np.pi * np.arange(N) / 12) + RNG.normal(0, 1, N)


def raw(values: np.ndarray, future: bool = True) -> dv.RawDriver:
    return dv.RawDriver(
        [float(v) for v in values[:N]],
        [float(v) for v in values[N : N + H]] if future else [None] * H,
    )


def built(drivers: list[dv.Driver], columns: dict[str, dv.RawDriver]) -> dv.DriverSet:
    return dv.build(drivers, columns, MONTHS[:N], MONTHS[N:], "month")


@pytest.mark.parametrize("key", ["tslm", "gbm_lags", "sarimax"])
def test_a_driver_that_generates_the_series_improves_mase(key: str) -> None:
    drivers = built([dv.Driver("x", known=True)], {"x": raw(X)})
    model = fc.sarimax_model() if key == "sarimax" else fc.MODELS[key]

    result = dv.run(key, model, Y, H, 12, drivers)

    assert result.uses_drivers and result.scores and result.scores.mase is not None
    assert result.mase_without_drivers is not None
    assert result.scores.mase < 0.7 * result.mase_without_drivers
    (effect,) = result.drivers
    assert effect.driver == "x" and effect.verdict == "helps"
    assert "+ 1 driver" in (result.info or "")
    if key == "gbm_lags":
        assert effect.importance is not None and effect.importance > 0
    else:
        assert effect.coefficient == pytest.approx(3, abs=0.5)
        assert effect.p_value is not None and effect.p_value < 0.01


def test_pure_noise_does_not_help() -> None:
    drivers = built([dv.Driver("noise", known=True)], {"noise": raw(NOISE)})

    result = dv.run("tslm", fc.MODELS["tslm"], SEASONAL, H, 12, drivers)

    (effect,) = result.drivers
    assert effect.verdict != "helps"
    assert effect.p_value is not None and effect.p_value > 0.01


def test_each_driver_is_judged_without_the_others() -> None:
    drivers = built(
        [dv.Driver("x", known=True), dv.Driver("noise", known=True)],
        {"x": raw(X), "noise": raw(NOISE)},
    )

    result = dv.run("tslm", fc.MODELS["tslm"], Y, H, 12, drivers)

    useful, noise = result.drivers
    assert useful.verdict == "helps" and noise.verdict != "helps"
    assert useful.mase_without is not None and noise.mase_without is not None
    assert useful.mase_without > noise.mase_without  # dropping x hurts much more


def test_a_past_only_driver_is_lagged_by_the_horizon() -> None:
    drivers = built([dv.Driver("x")], {"x": raw(X, future=False)})
    (info,) = drivers.infos

    assert info.source == "lag" and info.lag == H and not info.known
    assert drivers.history[H:, 0] == pytest.approx(X[: N - H])
    assert drivers.future[:, 0] == pytest.approx(X[N - H : N])  # already observed values


def test_no_leakage_at_any_origin() -> None:
    """The model never sees a driver value from a period later than its training window."""
    drivers = built([dv.Driver("x")], {"x": raw(X, future=False)})
    seen: list[tuple[int, np.ndarray]] = []

    def spy(y, h, season, x=None, x_future=None):
        assert x is not None and x_future is not None
        seen.append((len(y), x_future[:, 0].copy()))
        return fc.naive(y, h, season)

    bound = dv.Bound(spy, drivers)
    for size in fc.origins(N, H, 12):
        bound(Y[:size], H, 12)
        fitted, ahead = seen[-1]
        assert fitted == size - H  # the first h periods have no lagged value: left out
        latest = max(int(np.flatnonzero(np.isclose(X, v))[0]) for v in ahead)
        assert latest < size


def test_a_contemporaneous_driver_cannot_cheat_when_past_only() -> None:
    """y equals the driver: known in advance it is perfect, past only it is not."""
    known = dv.run(
        "tslm", fc.MODELS["tslm"], X[:N], H, 12, built([dv.Driver("x", known=True)], {"x": raw(X)})
    )
    past = dv.run("tslm", fc.MODELS["tslm"], X[:N], H, 12, built([dv.Driver("x")], {"x": raw(X)}))

    assert known.scores and known.scores.mase is not None and known.scores.mase < 0.01
    assert past.scores and past.scores.mase is not None and past.scores.mase > 0.3


def test_future_values_from_the_file_constant_or_last() -> None:
    partial = raw(X)
    partial.future[-3:] = [None, None, None]
    drivers = built(
        [
            dv.Driver("a", known=True),
            dv.Driver("b", known=True, future="constant", value=7),
            dv.Driver("c", known=True, future="last"),
            dv.Driver("d", known=True),
        ],
        {"a": partial, "b": raw(X), "c": raw(X), "d": raw(X, future=False)},
    )
    a, b, c, d = drivers.infos

    assert a.source == "file" and a.future_values[:9] == pytest.approx(X[N : N + 9])
    assert a.future_values[9:] == pytest.approx([X[N + 8]] * 3)  # previous value carried
    assert b.source == "constant" and b.future_values == [7.0] * H
    assert c.source == "last" and c.future_values == pytest.approx([X[N - 1]] * H)
    assert d.source == "last"  # no future row in the file
    notes = " ".join(drivers.notes)
    assert "a: 3 of 12" in notes and "b: held at 7" in notes and "c: its last value" in notes
    assert "d: the file has no future values" in notes
    # in the validation, "last" drivers repeat their last value from each origin
    _, ahead = drivers.window(60, H)
    assert ahead[:, drivers.groups[2][0]] == pytest.approx([X[59]] * H)
    assert ahead[:, drivers.groups[0][0]] == pytest.approx(X[60:72])


def test_constant_needs_a_value() -> None:
    with pytest.raises(dv.DriverError):
        built([dv.Driver("b", known=True, future="constant")], {"b": raw(X)})


def test_calendar_variables() -> None:
    days = [datetime(2024, 1, 29) + timedelta(days=i) for i in range(5)]  # Mon 29 … Fri 2 Feb

    weekday = dv.calendar_columns("weekday", days, "day")
    month_end = dv.calendar_columns("month_end", days, "day")

    assert weekday.shape == (5, 6) and weekday[0].sum() == 0 and weekday[1, 0] == 1
    assert month_end[:, 0].tolist() == [0, 0, 1, 0, 0]  # 31 January
    assert dv.calendar_parts("day") == ["weekday", "month", "month_end"]
    assert dv.calendar_parts("month") == []
    with pytest.raises(dv.DriverError):
        dv.calendar_columns("weekday", MONTHS, "month")


def test_weekday_driver_on_daily_data() -> None:
    days = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(140 + 14)]
    effect = np.array([0, 0, 0, 0, 5, 20, 25])[[d.weekday() for d in days[:140]]]
    y = 50 + effect + RNG.normal(0, 1, 140)
    drivers = dv.build([dv.Driver("weekday", kind="calendar")], {}, days[:140], days[140:], "day")

    result = dv.run("tslm", fc.MODELS["tslm"], y, 14, 7, drivers)

    assert result.scores and result.scores.mase is not None
    assert result.info and "Linear trend" in result.info  # seasonal dummies would repeat it
    (weekday,) = result.drivers
    assert weekday.driver == "Day of week" and weekday.coefficient is None


def test_flags_without_future_rows_are_zero() -> None:
    flag = dv.RawDriver([1.0 if i % 12 == 11 else 0.0 for i in range(N)], [None] * H)

    drivers = built([dv.Driver("holiday", kind="flag")], {"holiday": flag})

    assert drivers.infos[0].known and drivers.future[:, 0].tolist() == [0.0] * H
    assert "holiday: 12 of 12 forecast periods have no value" in drivers.notes[0]


def test_scenario_changes_the_future_of_known_drivers() -> None:
    drivers = built([dv.Driver("x", known=True)], {"x": raw(X)})
    base = dv.forecast_with_drivers(Y, H, 12, fc.MODELS["tslm"], drivers)
    higher = [v + 10 for v in X[N : N + H]]

    changed = dv.forecast_with_drivers(Y, H, 12, fc.MODELS["tslm"], drivers, {"x": higher})

    assert changed.mean - base.mean == pytest.approx(np.full(H, 30), abs=5)
    with pytest.raises(dv.DriverError):
        drivers.with_future({"x": [1.0]})
    lagged = built([dv.Driver("x")], {"x": raw(X)})
    with pytest.raises(dv.DriverError):
        lagged.with_future({"x": list(higher)})


def test_correlation_of_changes_suggests_drivers() -> None:
    together = dv.candidate("x", False, Y, raw(X))
    apart = dv.candidate("noise", False, Y, raw(NOISE))

    assert together.suggested and together.correlation is not None and together.correlation > 0.9
    assert not apart.suggested and together.future_periods == H


def test_too_many_or_duplicated_drivers() -> None:
    with pytest.raises(dv.DriverError):
        built([dv.Driver("x", known=True)] * 2, {"x": raw(X)})
    with pytest.raises(dv.DriverError):
        built([dv.Driver(f"x{i}") for i in range(7)], {f"x{i}": raw(X) for i in range(7)})


# --- Through a dataset (the library, as the app) --------------------------------------------------


@pytest.fixture
def shop(tmp_path: Path) -> dx.Dataset:
    """Monthly sales driven by the price, with six future rows: the planned prices, no sales."""
    lines = ["month,sales,price,visits"]
    for i in range(N + 6):
        month = MONTHS[i].strftime("%Y-%m-%d")
        price = 20 + 3 * np.sin(i / 3) + (i % 5)
        visits = 1000 + 10 * i + RNG.normal(0, 30)
        sales = "" if i >= N else f"{500 - 12 * price + 0.1 * visits + RNG.normal(0, 3):.2f}"
        lines.append(f"{month},{sales},{price:.2f},{'' if i >= N else f'{visits:.0f}'}")
    path = tmp_path / "shop.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return dx.open(path, cache_dir=tmp_path / "cache")


def test_future_values_come_from_the_future_rows(shop: dx.Dataset) -> None:
    series = shop.timeseries(time="month", value="sales", period="month")

    found = series.forecast(
        horizon=6, models=["naive", "tslm"], drivers=[dx.Driver("price", known=True), "visits"]
    )

    assert len(found.periods) == N and found.future[0] == "2025-01"  # future rows: no sales
    price, visits = found.drivers
    assert price.source == "file" and price.future_values == pytest.approx(
        [round(20 + 3 * np.sin(i / 3) + (i % 5), 2) for i in range(N, N + 6)]
    )
    assert visits.source == "lag" and visits.lag == 6
    tslm = next(r for r in found.results if r.key == "tslm")
    assert tslm.uses_drivers and [e.driver for e in tslm.drivers] == ["price", "visits"]
    assert tslm.drivers[0].verdict == "helps"
    naive = next(r for r in found.results if r.key == "naive")
    assert not naive.uses_drivers and not naive.drivers


def test_driver_candidates_and_scenarios(shop: dx.Dataset) -> None:
    series = shop.timeseries(time="month", value="sales", period="month")

    candidates = series.drivers(horizon=6)

    by_name = {c.column: c for c in candidates.candidates}
    assert by_name["price"].future_periods == 6 and by_name["price"].suggested
    assert by_name["visits"].future_periods == 0
    assert candidates.calendar == []
    base = series.scenario([dx.Driver("price", known=True)], horizon=6)
    cheaper = series.scenario(
        [dx.Driver("price", known=True)],
        {"price": [v - 2 for v in base.drivers[0].future_values if v is not None]},
        horizon=6,
    )
    assert np.mean(cheaper.forecast) > np.mean(base.forecast) + 15  # sales fall 12 per unit
    with pytest.raises(dx.ExplorerError):
        series.forecast(horizon=6, drivers=["sales"])  # the series itself


def test_sarimax_reuses_the_orders_of_the_automatic_arima() -> None:
    plain = fc.arima_auto(Y[:60], H, 12)
    searched = len(fc._orders)

    without = fc.sarimax_model()(Y[:60], H, 12)

    assert without.info == f"{plain.info} · no drivers"
    assert len(fc._orders) == searched  # no second search
    assert fc.arima_auto(Y[:60], H, 12).mean == pytest.approx(plain.mean)
