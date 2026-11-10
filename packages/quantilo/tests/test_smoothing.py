"""Smoothing methods and Kalman filtering of a series (064)."""

import math

import numpy as np
import pytest

from quantilo.timeseries import series as ts
from quantilo.timeseries import smoothing as sm


def noisy_level(n: int = 120, seed: int = 1) -> list[float | None]:
    rng = np.random.default_rng(seed)
    level = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return [float(v) for v in level + rng.normal(0, 3, n)]


def seasonal(n: int = 96, season: int = 12, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 50 + 0.2 * t + 10 * np.sin(2 * np.pi * t / season) + rng.normal(0, 1, n)


# --- EWMA ----------------------------------------------------------------------------------------


def test_ewma_alpha_by_likelihood_matches_ses() -> None:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    values = noisy_level()
    curve = sm.smooth(values, ["ewma"], "day", 7)[0]
    alpha = next(p for p in curve.params if p.key == "alpha")
    reference = SimpleExpSmoothing(
        np.array(values), initialization_method="known", initial_level=values[0]
    ).fit(optimized=True)
    assert alpha.auto
    assert float(alpha.value) == pytest.approx(reference.params["smoothing_level"], abs=0.02)
    assert curve.causal


def test_ewma_is_causal_and_carries_the_level_over_gaps() -> None:
    values: list[float | None] = [1.0, 2.0, None, None, 5.0, 6.0]
    curve = sm.smooth(values, ["ewma"], "day", None, sm.SmoothParams(ewma_alpha=0.5))[0]
    assert curve.values[:4] == [1.0, 1.5, 1.5, 1.5]
    assert curve.values[4] == pytest.approx(3.25)
    # changing a later value never moves an earlier point
    later = sm.smooth([*values[:5], 100.0], ["ewma"], "day", None, sm.SmoothParams(ewma_alpha=0.5))
    assert later[0].values[:5] == curve.values[:5]


# --- LOWESS, Savitzky-Golay, HP ------------------------------------------------------------------


def test_lowess_frac_rule_and_resistance_to_outliers() -> None:
    values = [float(v) for v in seasonal(96)]
    values[40] = 500.0
    curve = sm.smooth(values, ["lowess"], "month", 12)[0]
    frac = next(p for p in curve.params if p.key == "frac")
    assert float(frac.value) == pytest.approx(24 / 96)
    assert not curve.causal
    assert curve.values[40] is not None and curve.values[40] < 100  # the outlier is down-weighted


def test_lowess_skips_gaps_but_draws_every_period() -> None:
    values: list[float | None] = [float(v) for v in seasonal(60)]
    values[10] = None
    curve = sm.smooth(values, ["lowess"], "month", 12)[0]
    assert all(v is not None for v in curve.values)


def test_savgol_window_rule() -> None:
    from scipy.signal import savgol_filter

    values = [float(v) for v in seasonal(200, 7)]
    curve = sm.smooth(values, ["savgol"], "day", 7)[0]
    window = next(p for p in curve.params if p.key == "window")
    assert int(window.value) == 11  # 200 / 20 = 10 -> nearest odd above
    expected = np.asarray(savgol_filter(np.array(values), 11, 2, mode="interp"), dtype=float)
    assert np.allclose([v for v in curve.values if v is not None], expected)


def test_hp_lambda_by_period_and_unavailable_on_days() -> None:
    from statsmodels.tsa.filters.hp_filter import hpfilter

    values = [float(v) for v in seasonal(48, 4)]
    curve = sm.smooth(values, ["hp"], "quarter", 4)[0]
    lam = next(p for p in curve.params if p.key == "lambda")
    assert float(lam.value) == 1600
    _, trend = hpfilter(np.array(values), 1600)
    assert np.allclose([v for v in curve.values if v is not None], trend)
    assert any("Hamilton" in note for note in curve.notes)
    daily = sm.smooth(values, ["hp"], "day", 7)[0]
    assert daily.error and "monthly" in daily.error
    assert sm.unavailable("day")["hp"]
    assert "hp" not in sm.unavailable("month")


# --- Kalman --------------------------------------------------------------------------------------


def test_kalman_chooses_a_seasonal_form_for_a_seasonal_series() -> None:
    fit = sm.kalman(seasonal(96), 12)
    assert fit.form.endswith("seasonal")
    assert "seasonal (12)" in fit.label


def test_kalman_chooses_a_local_level_for_a_noisy_level() -> None:
    fit = sm.kalman(np.array(noisy_level(), dtype=float), None)
    assert fit.form in ("llevel", "lltrend")
    assert fit.sigmas["noise"] > 0


def test_kalman_filtered_is_causal_and_smoothed_is_not() -> None:
    values = noisy_level(80)
    changed = [*values[:-1], 300.0]
    params = sm.SmoothParams(kalman_model="llevel")
    a = sm.smooth(values, ["kalman_filtered", "kalman_smoothed"], "day", None, params)
    b = sm.smooth(changed, ["kalman_filtered", "kalman_smoothed"], "day", None, params)
    assert a[0].causal and not a[1].causal
    assert b[1].values[-2] is not None and a[1].values[-2] is not None
    assert b[1].values[-2] > a[1].values[-2]  # the smoother saw the future jump
    for v, lo, hi in zip(a[1].values, a[1].low, a[1].high, strict=True):
        assert v is None or lo is None or hi is None or lo <= v <= hi


def test_kalman_filtered_does_not_look_ahead_with_fixed_parameters() -> None:
    values = np.array(noisy_level(80), dtype=float)
    fit = sm.kalman(values, None, "llevel")
    changed = values.copy()
    changed[-1] = 300.0
    again = sm.kalman(changed, None, "llevel", params=fit.params)
    assert np.allclose(fit.level_filtered[:-1], again.level_filtered[:-1])
    assert not np.allclose(fit.level_smoothed[:-1], again.level_smoothed[:-1])


def test_kalman_fill_gives_estimates_with_bands_at_missing_dates() -> None:
    y = seasonal(96)
    values: list[float | None] = [float(v) for v in y]
    for i in (20, 21, 50):
        values[i] = None
    filled = sm.kalman_fill(values, 12)
    assert filled.values[0] == values[0]
    for i in (20, 21, 50):
        v, lo, hi = filled.values[i], filled.low[i], filled.high[i]
        assert v is not None and lo is not None and hi is not None
        assert lo < v < hi
        assert abs(v - y[i]) < 4 * (hi - lo) / 3.92 + 1
    assert filled.low[5] is None  # observed periods keep no band
    linear = ts.fill_gaps(values, "interpolate")
    rmse_k = math.sqrt(np.mean([(filled.values[i] - y[i]) ** 2 for i in (20, 21, 50)]))  # type: ignore[operator]
    rmse_l = math.sqrt(np.mean([(linear[i] - y[i]) ** 2 for i in (20, 21, 50)]))  # type: ignore[operator]
    assert rmse_k < rmse_l * 1.5


def test_fill_gaps_kalman() -> None:
    values: list[float | None] = [float(v) for v in seasonal(48, 4)]
    values[10] = None
    filled = ts.fill_gaps(values, "kalman", 4)
    assert all(v is not None for v in filled)
    assert filled[11] == values[11]


def test_kalman_parameters_are_fitted_on_the_last_periods_of_a_long_series() -> None:
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(0, 1, 1500)) + rng.normal(0, 2, 1500)
    fit = sm.kalman(y, None, "llevel")
    assert fit.fitted_on == sm.MAX_FIT
    assert len(fit.level_smoothed) == 1500


def test_one_step_predictions_flag_a_spike() -> None:
    values = np.array(noisy_level(100), dtype=float)
    values[60] += 40
    fit = sm.kalman(values, None)
    scores = sm.innovation_scores(fit)
    assert int(np.argmax(scores)) == 60


# --- Suggestions ---------------------------------------------------------------------------------


def test_suggestions_say_why() -> None:
    values: list[float | None] = [float(v) for v in seasonal(96)]
    values[30] = None
    found = {s.key: s.reason for s in sm.suggest(values, "month", 12)}
    assert "kalman_smoothed" in found and "1 of 96 periods" in found["kalman_smoothed"]
    assert "hp" not in found


def test_ewma_suggested_for_noise_without_season() -> None:
    rng = np.random.default_rng(4)
    values = [float(v) for v in 10 + rng.normal(0, 1, 60)]
    found = {s.key for s in sm.suggest(values, "day", None)}
    assert "ewma" in found


def test_lowess_suggested_when_outliers_exist() -> None:
    values = [float(v) for v in seasonal(96)]
    values[40] = 400.0
    assert "lowess" in {s.key for s in sm.suggest(values, "month", 12)}
    assert "lowess" not in {
        s.key for s in sm.suggest([float(v) for v in seasonal(96)], "month", 12)
    }


def test_savgol_suggested_for_dense_regular_signals() -> None:
    values = [float(v) for v in seasonal(200, 24)]
    assert "savgol" in {s.key for s in sm.suggest(values, "hour", 24)}
    assert "savgol" not in {s.key for s in sm.suggest(values, "month", 12)}


def test_moving_average_curve_matches_rolling() -> None:
    values = noisy_level(40)
    curve = sm.smooth(values, ["moving_average"], "day", 7, sm.SmoothParams(window=5))[0]
    assert curve.values == ts.rolling(values, 5, "trailing")[0]
    assert curve.causal
    centered = sm.smooth(values, ["moving_average"], "day", 7, sm.SmoothParams(5, "centered"))[0]
    assert not centered.causal


# --- Forecast and anomalies (L4) -----------------------------------------------------------------


def test_structural_forecast_names_its_form_and_widens() -> None:
    from quantilo.timeseries import forecast as fc

    prediction = fc.structural(seasonal(96), 12, 12)
    assert prediction.info is not None and prediction.info.startswith("UCM(")
    widths = prediction.high95 - prediction.low95
    assert widths[-1] > widths[0]
    assert "structural" in fc.MODELS


def test_kalman_anomalies_follow_a_drifting_level() -> None:
    from quantilo.timeseries import breaks as br

    rng = np.random.default_rng(8)
    level = 100 + np.cumsum(rng.normal(0, 1.5, 200))  # slow drift that a flat z-score misreads
    series = level + rng.normal(0, 1, 200)
    series[120] += 15
    found = br.anomalies(series - series.mean(), series, "kalman", 3.5)
    assert found and found[0].index == 120
    assert abs(found[0].expected - level[120]) < 6
    assert len(found) <= 4


def test_lowess_not_suggested_when_far_points_are_the_rule() -> None:
    rng = np.random.default_rng(6)
    walk = [float(v) for v in 100 + np.cumsum(rng.standard_t(2, 400))]  # fat-tailed moves
    reasons = {s.key for s in sm.suggest(walk, "day", 7)}
    assert "lowess" not in reasons
