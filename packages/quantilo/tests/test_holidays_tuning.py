"""028 A5 (public holidays by country), A7 (tuning options) and the Prophet-like components."""

import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import quantilo as dx
from quantilo.timeseries import drivers as dv
from quantilo.timeseries import forecast as fc
from quantilo.timeseries import public_holidays as ph

RNG = np.random.default_rng(5)
START = datetime(2021, 1, 4)  # a Monday
DAYS = 2 * 365 + 112  # the horizon holds May 1, May 8 and Ascension 2023
H = 30
MOMENTS = [START + timedelta(days=i) for i in range(DAYS + H)]
FR = ph.calendar("FR", None, range(2021, 2024))
FLAG = np.array([float(m.date() in FR) for m in MOMENTS])
# Daily sales: weekly cycle, and 60 units fewer on French public holidays (shops closed).
SALES = (
    300 + 30 * np.sin(2 * np.pi * np.arange(DAYS + H) / 7) - 60 * FLAG + RNG.normal(0, 4, DAYS + H)
)


def holiday_set(separate_too: bool = True) -> dv.DriverSet:
    return dv.build([dv.Driver("FR", "holidays", True)], {}, MOMENTS[:DAYS], MOMENTS[DAYS:], "day")


# --- The calendar ---------------------------------------------------------------------------------


def test_the_french_calendar_and_its_codes() -> None:
    assert date(2022, 7, 14) in FR and date(2022, 7, 13) not in FR
    assert "National Day" in FR[date(2022, 7, 14)]  # English names when the calendar has them
    assert ph.country_code("France") == "FR"
    assert ph.country_code("fra") == "FR" and ph.country_code("fr") == "FR"
    assert ph.country_code("USA") == "US" and ph.country_code("United Kingdom") == "GB"
    assert ph.country_code("Wakanda") is None and ph.country_code(12) is None
    assert ph.parse("us-ca") == ("US", "CA") and ph.parse("FR") == ("FR", None)
    codes = {c.code for c in ph.countries()}
    assert {"FR", "US", "DE", "JP"} <= codes
    us = next(c for c in ph.countries() if c.code == "US")
    assert us.name == "United States" and any(s.code == "CA" for s in us.subdivisions)


def test_a_subdivision_has_its_own_holidays() -> None:
    moselle = ph.calendar("FR", "57", range(2022, 2023))
    assert date(2022, 12, 26) in moselle and date(2022, 12, 26) not in FR  # Saint Stephen's Day
    with pytest.raises(ph.HolidayError, match="no subdivision"):
        ph.calendar("FR", "ZZ", range(2022, 2023))
    with pytest.raises(ph.HolidayError, match="does not know"):
        ph.calendar("XX", None, range(2022, 2023))


def test_flags_per_period() -> None:
    days = [datetime(2022, 7, 13), datetime(2022, 7, 14)]
    past, ahead, names, _ = ph.flags("FR", None, days, [datetime(2022, 12, 25)], "day")
    assert past[:, 0].tolist() == [0.0, 1.0] and ahead[:, 0].tolist() == [1.0]
    assert names == ["Public holidays (FR)"]
    hours = [datetime(2022, 7, 14, 9), datetime(2022, 7, 15, 9)]
    past, _, _, _ = ph.flags("FR", None, hours, [], "hour")
    assert past[:, 0].tolist() == [1.0, 0.0]
    # The week of Monday 2022-04-11 has none; the week of Easter Monday 2022-04-18 has one.
    weeks = [datetime(2022, 4, 11), datetime(2022, 4, 18)]
    past, _, _, _ = ph.flags("FR", None, weeks, [], "week")
    assert past[:, 0].tolist() == pytest.approx([0.0, 1 / 7])
    with pytest.raises(ph.HolidayError, match="hourly, daily or weekly"):
        ph.flags("FR", None, [datetime(2022, 1, 1)], [], "month")


def test_separate_columns_one_per_holiday_seen_in_the_history() -> None:
    past, ahead, names, _ = ph.flags("FR", None, MOMENTS[:DAYS], MOMENTS[DAYS:], "day", True)
    assert len(names) == 11 and "National Day" in names  # the 11 French public holidays
    assert past.shape == (DAYS, 11) and ahead.shape == (H, 11)
    assert past.sum(axis=1).max() == 1  # one holiday a day at most here


def test_years_outside_the_calendar_are_said() -> None:
    _, _, _, notes = ph.flags("FR", None, [datetime(2101, 1, 1)], [], "day")
    assert notes and "covers 1803 to 2100" in notes[0]


# --- As a driver ----------------------------------------------------------------------------------


def test_the_holidays_driver_is_known_in_advance_and_expanded() -> None:
    drivers = holiday_set()
    (info,) = drivers.infos
    assert info.kind == "holidays" and info.known and info.label == "Public holidays (FR)"
    assert drivers.history.shape == (DAYS, 1)
    assert drivers.future[:, 0].tolist() == FLAG[DAYS:].tolist()  # the calendar covers it
    assert info.future_values == FLAG[DAYS:].tolist()
    assert drivers.expanded is not None and drivers.expanded.history.shape[1] == 11
    with pytest.raises(dv.DriverError, match="calendar variable"):
        drivers.with_future({"FR": [0.0] * H})
    with pytest.raises(dv.DriverError, match="One public holiday calendar"):
        dv.build(
            [dv.Driver("FR", "holidays"), dv.Driver("DE", "holidays")],
            {},
            MOMENTS[:DAYS],
            MOMENTS[DAYS:],
            "day",
        )


@pytest.mark.parametrize("key", ["tslm", "prophet_like"])
def test_the_holidays_driver_helps_when_the_series_closes_on_holidays(key: str) -> None:
    model = fc.prophet_model("day") if key == "prophet_like" else fc.MODELS[key]

    result = dv.run(key, model, SALES[:DAYS], H, 7, holiday_set())

    (effect,) = result.drivers
    assert effect.driver == "Public holidays (FR)" and effect.verdict == "helps"
    if key == "tslm":
        assert effect.coefficient == pytest.approx(-60, abs=5)  # one flag, one coefficient
    else:
        assert effect.coefficient is None  # one effect per holiday: no single number
    holiday = int(np.flatnonzero(FLAG[DAYS:])[0])
    assert result.forecast[holiday] < np.mean(result.forecast) - 40


def test_prophet_like_components_add_up_to_its_forecast() -> None:
    result = dv.run("prophet_like", fc.prophet_model("day"), SALES[:DAYS], H, 7, holiday_set())

    kinds = [(c.kind, c.name) for c in result.components]
    assert kinds[0] == ("trend", "Trend") and ("season", "weekly") in kinds
    assert ("holidays", "Public holidays (FR)") in kinds
    assert result.components_start == 0
    assert all(len(c.values) == DAYS + H for c in result.components)
    total = np.sum([c.values for c in result.components], axis=0)
    assert total[DAYS:] == pytest.approx(result.forecast, rel=1e-6)
    holidays = next(c for c in result.components if c.kind == "holidays")
    assert min(holidays.values) == pytest.approx(-60, abs=12)


# --- Tuning (A7) ----------------------------------------------------------------------------------


def test_tuning_is_checked() -> None:
    with pytest.raises(ValueError, match="max_p goes from 0 to 5"):
        fc.Tuning(max_p=9)
    with pytest.raises(ValueError, match="time_limit"):
        fc.Tuning(time_limit=0)
    assert fc.Tuning() == fc.DEFAULT_TUNING


def test_arima_bounds_and_criterion() -> None:
    y = np.cumsum(RNG.normal(0, 1, 120)) + 0.5 * np.sin(np.arange(120))
    tight = fc.Tuning(max_p=0, max_q=0, max_d=1, criterion="bic")

    order, _, label, _ = fc.arima_search(y, None, tight)

    assert order[0] == 0 and order[2] == 0 and order[1] <= 1
    assert label.endswith("by BIC")
    # The orders are remembered per search: the default search is not confused with this one.
    assert fc.arima_orders(y, None, tight)[2] == label
    assert "by BIC" not in fc.arima_orders(y, None)[2]


def test_gradient_boosting_grid_is_chosen_on_an_inner_hold_out() -> None:
    assert fc.lag_grid("fixed", 12) == [(12, 15)]
    assert len(fc.lag_grid("small", 12)) == 4 and len(fc.lag_grid("wide", 12)) == 9
    y = 100 + 10 * np.sin(2 * np.pi * np.arange(120) / 12) + RNG.normal(0, 1, 120)

    prediction = fc.lag_model("gbm", "small")(y, 12, 12)

    assert prediction.info is not None and "grid small: best of 4 on the last 12" in prediction.info
    assert np.mean(np.abs(prediction.mean - y[-12:])) < 5  # the cycle is followed


def test_prophet_like_changepoints_and_prior() -> None:
    m = np.arange(120)
    y = np.where(m < 84, 50 + 1.0 * m, 134 - 1.5 * (m - 84)) + RNG.normal(0, 1, 120)

    none = fc.prophet_model("month", fc.Tuning(changepoints=0))(y, 12, 12)
    loose = fc.prophet_model("month", fc.Tuning(changepoint_scale=0.5))(y, 12, 12)

    assert none.info is not None and none.info.startswith("Piecewise trend (0 changepoints)")
    assert loose.mean[-1] < y[-1]  # the downturn is followed with changepoints, not without
    assert none.mean[-1] > loose.mean[-1]


def test_a_model_over_its_time_limit_stops_and_says_so() -> None:
    def slow(y: np.ndarray, h: int, season: int | None) -> fc.Prediction:
        time.sleep(0.6)
        return fc.naive(y, h, season)

    result = fc.run_model("slow", slow, SALES[:200], 7, 7, limit=1)

    assert result.timed_out and result.scores is None
    assert result.error is not None and result.error.startswith("Stopped at the time limit (1 s)")
    fast = fc.run_model("naive", fc.naive, SALES[:200], 7, 7, limit=1)
    assert not fast.timed_out and fast.scores is not None


# --- Through a dataset (the library, as the app) --------------------------------------------------


@pytest.fixture
def shop(tmp_path: Path) -> dx.Dataset:
    lines = ["day,sales,country"]
    for i in range(DAYS):
        lines.append(f"{MOMENTS[i]:%Y-%m-%d},{SALES[i]:.2f},France")
    path = tmp_path / "shop.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return dx.open(path, cache_dir=tmp_path / "cache")


def test_library_holidays_and_tuning(shop: dx.Dataset) -> None:
    series = shop.timeseries(time="day", value="sales", period="day")

    offer = series.drivers(horizon=H).holidays
    assert offer is not None and offer.usable and offer.guess.code == "FR"
    assert "The column country holds a single value, France" in offer.guess.reason
    found = series.forecast(
        horizon=H,
        models=["naive", "tslm", "prophet_like"],
        holidays="FR",
        tuning={"changepoints": 10, "time_limit": 120},
    )

    assert [d.label for d in found.drivers] == ["Public holidays (FR)"]
    tslm = next(r for r in found.results if r.key == "tslm")
    assert tslm.drivers[0].verdict == "helps"
    prophet = next(r for r in found.results if r.key == "prophet_like")
    assert any(c.kind == "holidays" for c in prophet.components)
    monthly = shop.timeseries(time="day", value="sales", period="month").drivers().holidays
    assert monthly is not None and not monthly.usable
    assert monthly.reason is not None and "hourly, daily or weekly" in monthly.reason
