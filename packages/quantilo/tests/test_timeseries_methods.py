from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy import stats
from statsmodels.tsa.seasonal import STL

from quantilo.timeseries import methods
from quantilo.timeseries import series as ts
from quantilo.timeseries.decomposition import DecompositionError, decompose, strength
from quantilo.timeseries.seasonality import profiles, starts, year_overlay

RNG = np.random.default_rng(11)
WEEKLY = 10 + np.tile([0, 0, 0, 0, 1, 6, 4], 30) + RNG.normal(0, 0.5, 210)  # strong Sat/Sun
NOISE = RNG.normal(0, 1, 300)
TRENDING = list(0.3 * np.arange(120) + RNG.normal(0, 1, 120))


# --- gaps ---------------------------------------------------------------------------------------


def test_fill_gaps() -> None:
    values = [1.0, None, 3.0, None]
    assert ts.fill_gaps(values, "leave") == values
    assert ts.fill_gaps(values, "interpolate") == [1.0, 2.0, 3.0, None]
    assert ts.fill_gaps(values, "zero") == [1.0, 0.0, 3.0, 0.0]
    assert ts.fill_gaps(values, "carry") == [1.0, 1.0, 3.0, 3.0]


def test_complete_fills_inside_and_edges() -> None:
    series, filled = ts.complete([None, 2.0, None, 4.0, None])
    assert list(series) == [2.0, 2.0, 3.0, 4.0, 4.0] and filled == 3


def test_season_length() -> None:
    assert [ts.season_length(p) for p in ("hour", "day", "week", "month", "quarter", "year")] == [
        24,
        7,
        52,
        12,
        4,
        None,
    ]


# --- trend --------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(methods.TREND))
def test_every_trend_method_finds_a_clear_trend(key: str) -> None:
    result = methods.TREND[key](TRENDING, 12)

    assert result.key == key and result.significant is True, result


@pytest.mark.parametrize("key", ["mann_kendall", "hamed_rao", "cox_stuart", "spearman_time"])
def test_trend_tests_stay_quiet_on_noise(key: str) -> None:
    result = methods.TREND[key](list(NOISE), 12)

    assert result.p_value is not None and result.p_value > 0.01


def test_linear_trend_matches_scipy() -> None:
    result = methods.linear_trend(TRENDING)
    fit = stats.linregress(np.arange(120), TRENDING)

    assert result.value == pytest.approx(fit.slope)  # type: ignore[attr-defined]
    assert result.p_value == pytest.approx(fit.pvalue)  # type: ignore[attr-defined]
    assert result.low is not None and result.low < result.value < result.high  # type: ignore[operator]


def test_hamed_rao_inflates_the_variance_of_an_autocorrelated_series() -> None:
    walk: list[float | None] = [float(v) for v in np.cumsum(RNG.normal(0, 1, 200))]

    plain = methods.mann_kendall_test(walk)
    corrected = methods.hamed_rao(walk)

    assert corrected.statistic is not None and plain.statistic is not None
    assert abs(corrected.statistic) < abs(plain.statistic)


def test_seasonal_mann_kendall_ignores_the_seasonal_pattern() -> None:
    flat_seasonal = list(np.tile([0.0, 5.0, 10.0, 5.0], 25) + RNG.normal(0, 0.3, 100))

    result = methods.seasonal_mann_kendall(flat_seasonal, 4)

    assert result.p_value is not None and result.p_value > 0.01


def test_cox_stuart_counts_rises() -> None:
    result = methods.cox_stuart([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    assert result.statistic == 4 and result.value == 1.0


def test_methods_explain_why_they_did_not_run() -> None:
    result = methods.hamed_rao([1.0, 2.0])

    assert result.p_value is None and result.note


# --- seasonality --------------------------------------------------------------------------------


def test_kruskal_finds_the_weekly_pattern_and_not_in_noise() -> None:
    found = methods.kruskal_seasons(WEEKLY - WEEKLY.mean(), 7)
    quiet = methods.kruskal_seasons(NOISE[:210], 7)

    assert found.significant is True and found.value is not None and found.value > 0.5
    assert quiet.significant is False


def test_periodogram_finds_the_cycle_length() -> None:
    result = methods.periodogram_peak(WEEKLY, 7)

    assert result.value == pytest.approx(7, abs=0.2)


@pytest.mark.parametrize("key", ["friedman", "qs", "fisher_g", "seasonal_acf"])
def test_seasonality_tests_detect_a_cycle(key: str) -> None:
    method, _ = methods.SEASONALITY[key]

    assert method(WEEKLY - WEEKLY.mean(), 7).significant is True


def test_fisher_g_is_quiet_on_noise() -> None:
    assert methods.fisher_g(NOISE, None).significant is False


def test_profiles_by_weekday_point_at_the_weekend() -> None:
    periods = [(datetime(2024, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(210)]
    moments = starts(periods, "day")

    weekday = profiles("day", moments, WEEKLY, WEEKLY - WEEKLY.mean())[0]

    assert weekday.key == "weekday" and weekday.strongest == "Sat"
    assert weekday.test.significant is True
    assert len(weekday.values) == 7 and all(len(v) == 30 for v in weekday.values)


def test_month_profile_needs_a_year_and_overlay_two_years() -> None:
    periods = [(datetime(2024, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(210)]
    moments = starts(periods, "day")

    assert [p.key for p in profiles("day", moments, WEEKLY, WEEKLY)] == ["weekday"]
    assert year_overlay("day", moments, list(WEEKLY)) is None
    months = [f"{2022 + i // 12}-{i % 12 + 1:02d}" for i in range(30)]
    overlay = year_overlay("month", starts(months, "month"), [float(i) for i in range(30)])
    assert overlay is not None and [y.year for y in overlay.years] == [2022, 2023, 2024]
    assert overlay.years[0].x == list(range(1, 13))


# --- decomposition ------------------------------------------------------------------------------


def synthetic(n: int = 156, season: int = 52) -> np.ndarray:
    t = np.arange(n)
    return 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / season) + RNG.normal(0, 1, n)


def test_stl_matches_statsmodels_and_measures_strengths() -> None:
    series = synthetic()

    result = decompose(series, "stl", 52, "week")
    robust = not result.notes  # three cycles: constant seasonal smoother (seasonal_deg=0)
    expected = STL(series, period=52, robust=robust, seasonal_deg=0).fit()

    assert result.trend == pytest.approx(list(expected.trend))
    assert result.strength_trend is not None and result.strength_trend > 0.9
    assert result.strength_seasonal is not None and result.strength_seasonal > 0.8


def test_noise_has_weak_strengths() -> None:
    result = decompose(NOISE[:104], "stl", 52, "week")

    assert result.strength_trend is not None and result.strength_trend < 0.5
    assert result.strength_seasonal is not None and result.strength_seasonal < 0.5


@pytest.mark.parametrize("method", ["mstl", "classical_additive", "classical_multiplicative"])
def test_other_decompositions(method: str) -> None:
    result = decompose(synthetic(), method, 52, "week")  # type: ignore[arg-type]

    assert len(result.trend) == 156 and len(result.seasonal) == 1
    assert result.strength_seasonal is not None


def test_mstl_adds_the_yearly_cycle_to_daily_data() -> None:
    t = np.arange(800)
    series = (
        50 + 5 * np.sin(2 * np.pi * t / 7) + 8 * np.sin(2 * np.pi * t / 365) + RNG.normal(0, 1, 800)
    )

    result = decompose(series, "mstl", 7, "day")

    assert result.seasons == [7, 365] and len(result.seasonal) == 2


def test_decomposition_needs_two_cycles() -> None:
    with pytest.raises(DecompositionError, match="two full cycles"):
        decompose(NOISE[:60], "stl", 52, "week")
    with pytest.raises(DecompositionError, match="above 0"):
        decompose(NOISE[:120], "classical_multiplicative", 12, "month")


def test_strength_formula() -> None:
    component = np.array([1.0, -1.0, 1.0, -1.0])
    assert strength(component, np.zeros(4)) == 1.0
    assert strength(np.zeros(4), component) == 0.0


def test_overlay_drops_edge_years_with_a_single_point() -> None:
    weeks = [(datetime(2021, 12, 27) + timedelta(weeks=i)).strftime("%Y-%m-%d") for i in range(158)]

    overlay = year_overlay("week", starts(weeks, "week"), [float(i) for i in range(158)])

    assert overlay is not None and [y.year for y in overlay.years] == [2022, 2023, 2024]
