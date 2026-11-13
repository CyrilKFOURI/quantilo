"""028 A6: the Prophet-like structural model (piecewise trend + Fourier + drivers) and STLF."""

import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantilo.timeseries import drivers as dv
from quantilo.timeseries import forecast as fc
from quantilo.timeseries import prophet_like as pl
from quantilo.timeseries.forecasting import ForecastJobs

RNG = np.random.default_rng(6)
DAYS = 3 * 365
D = np.arange(DAYS)
# Daily data with a weekly and a yearly cycle on a slowly growing level.
DAILY = (
    200
    + 0.03 * D
    + 12 * np.sin(2 * np.pi * D / 7)
    + 25 * np.sin(2 * np.pi * D / 365.25)
    + RNG.normal(0, 3, DAYS)
)
# Monthly data whose trend turns down after 70 % of the history.
M = np.arange(120)
BROKEN = np.where(M < 84, 50 + 1.0 * M, 134 - 1.5 * (M - 84)) + 5 * np.sin(2 * np.pi * M / 12)
BROKEN = BROKEN + RNG.normal(0, 1, 120)


def test_cycles_follow_the_period_and_the_length_of_the_history() -> None:
    used, skipped = pl.cycles("day", 7, DAYS)
    assert [(c.name, c.order) for c in used] == [("weekly", 3), ("yearly", 10)]
    assert skipped == []

    used, skipped = pl.cycles("day", 7, 500)
    assert [c.name for c in used] == ["weekly"] and skipped == ["yearly needs 2 cycles"]

    used, _ = pl.cycles("hour", 24, 1_000)
    assert [(c.name, c.length) for c in used] == [("daily", 24), ("weekly", 168)]

    used, _ = pl.cycles("month", 12, 30)
    assert [(c.name, c.order) for c in used] == [("yearly", 6)]  # capped at half the cycle

    assert pl.cycles("year", None, 40) == ([], [])


def test_the_period_is_inferred_from_the_season_and_a_custom_season_is_added() -> None:
    assert [c.name for c in pl.cycles(None, 7, DAYS)[0]] == ["weekly", "yearly"]
    used, _ = pl.cycles("day", 5, 200)  # five working days, set by hand
    assert [(c.name, c.length, c.order) for c in used] == [
        ("weekly", 7, 3),
        ("cycle of 5", 5, 2),
    ]


def test_changepoint_candidates_cover_the_first_80_percent() -> None:
    points = pl.changepoint_positions(200)
    assert len(points) == 25 and points[0] > 0 and points[-1] == 159

    short = pl.changepoint_positions(12)
    assert 0 < len(short) < 25 and short[-1] < 12 * 0.8


def test_prophet_like_follows_a_trend_that_bends() -> None:
    linear = fc.evaluate(BROKEN, fc.MODELS["tslm"], 12, 12)
    prophet = fc.evaluate(BROKEN, fc.prophet_model("month"), 12, 12)

    assert prophet.mase is not None and linear.mase is not None
    assert prophet.mase < 0.6 * linear.mase
    prediction = fc.prophet_model("month")(BROKEN, 12, 12)
    assert prediction.mean[-1] < prediction.mean[0]  # the new, falling slope goes on
    assert "changepoint" in (prediction.info or "") and "yearly (K=6)" in (prediction.info or "")
    assert prediction.approximate  # its final intervals are conformal


@pytest.mark.parametrize("key", ["prophet_like", "stlf"])
def test_models_with_several_seasons_beat_the_seasonal_naive(key: str) -> None:
    model = fc.PERIOD_MODELS[key]("day")
    started = time.monotonic()
    result = fc.run_model(key, model, DAILY, 30, 7)
    reference = fc.evaluate(DAILY, fc.seasonal_naive, 30, 7)

    assert result.error is None and result.scores and result.scores.mase is not None
    assert reference.mase is not None and result.scores.mase < reference.mase
    assert "conformal intervals" in (result.info or "")
    assert time.monotonic() - started < 30
    if key == "stlf":
        assert (result.info or "").startswith("MSTL (7, 365) + ETS(")
    else:
        assert "weekly (K=3) + yearly (K=10)" in (result.info or "")


def test_stlf_with_one_season_and_without_any() -> None:
    prediction = fc.stlf_model("month")(BROKEN, 12, 12)
    assert (prediction.info or "").startswith("STL (12) + ETS(")
    assert len(prediction.mean) == 12

    with pytest.raises(fc.ModelError, match="two full cycles"):
        fc.stlf_model("year")(BROKEN[:40], 3, None)


def test_prophet_like_takes_a_driver_like_the_linear_regression() -> None:
    n, h = 96, 12
    months = [datetime(2017 + i // 12, i % 12 + 1, 1) for i in range(n + h)]
    x = np.cumsum(RNG.normal(0, 2, n + h)) + 50
    y = 40 + 3 * x[:n] + RNG.normal(0, 1, n)
    drivers = dv.build(
        [dv.Driver("x", known=True)],
        {"x": dv.RawDriver([float(v) for v in x[:n]], [float(v) for v in x[n:]])},
        months[:n],
        months[n:],
        "month",
    )

    result = dv.run("prophet_like", fc.MODELS["prophet_like"], y, h, 12, drivers)

    assert result.scores and result.scores.mase is not None and result.mase_without_drivers
    assert result.scores.mase < 0.7 * result.mase_without_drivers
    (effect,) = result.drivers
    assert effect.verdict == "helps" and effect.p_value is None
    assert effect.coefficient == pytest.approx(3, abs=0.5)


def test_a_holiday_flag_gets_its_own_effect() -> None:
    flag = np.zeros(400)
    flag[::37] = 1
    y = 100 + 5 * np.sin(2 * np.pi * np.arange(400) / 7) + 40 * flag + RNG.normal(0, 1, 400)

    prediction = fc.prophet_model("day")(y[:370], 30, 7, flag[:370, None], flag[370:, None])

    assert prediction.coef is not None and prediction.coef[0] == pytest.approx(40, abs=3)
    assert prediction.mean[np.flatnonzero(flag[370:])[0]] > prediction.mean.mean() + 30


def test_the_forecast_lists_the_cycles_the_history_covers() -> None:
    jobs = ForecastJobs(workers=1)
    values: list[float | None] = [float(v) for v in DAILY]
    periods = [(datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(DAYS)]
    job = jobs.start(("c",), values, periods, "day", 7, 14, {"naive": fc.naive})
    for _ in range(200):
        if job.status != "running":
            break
        time.sleep(0.05)
    assert job.result is not None and job.result.cycles == ["weekly", "yearly"]
