import numpy as np
import pytest

from quantilo.timeseries import forecast as fc

RNG = np.random.default_rng(12)
T = np.arange(120)
SEASONAL = 100 + 10 * np.sin(2 * np.pi * T / 12) + 0.3 * T + RNG.normal(0, 1, 120)
AR1 = np.zeros(300)
for t in range(1, 300):
    AR1[t] = 0.7 * AR1[t - 1] + RNG.normal()
AR1 += 50


# The AI models (028 phase B) need the forecast-ai extra: tests/test_forecast_ai.py.
@pytest.mark.parametrize("key", [k for k in fc.MODELS if k != "sarima" and k not in fc.AI_MODELS])
def test_every_model_forecasts_with_ordered_intervals(key: str) -> None:
    prediction = fc.MODELS[key](SEASONAL, 12, 12)

    assert len(prediction.mean) == 12
    assert np.all(prediction.low95 <= prediction.low80 + 1e-9)
    assert np.all(prediction.low80 <= prediction.mean + 1e-9)
    assert np.all(prediction.mean <= prediction.high80 + 1e-9)
    assert np.all(prediction.high80 <= prediction.high95 + 1e-9)


def test_naive_intervals_widen_with_the_horizon() -> None:
    prediction = fc.naive(AR1, 5, None)
    widths = prediction.high95 - prediction.low95

    assert np.all(np.diff(widths) > 0)
    assert prediction.mean[0] == AR1[-1]


def test_seasonal_naive_intervals_account_for_growth() -> None:
    growing = 100 * 1.15 ** (T / 12) + 10 * np.sin(2 * np.pi * T / 12) + RNG.normal(0, 1, 120)

    scores = fc.evaluate(growing, fc.seasonal_naive, 12, 12)

    assert scores.coverage95 is not None and scores.coverage95 >= 0.5


def test_seasonal_naive_repeats_the_last_cycle() -> None:
    prediction = fc.seasonal_naive(SEASONAL, 24, 12)

    assert prediction.mean[:12] == pytest.approx(SEASONAL[-12:])
    assert prediction.mean[12:] == pytest.approx(SEASONAL[-12:])


def test_seasonal_models_beat_naive_on_a_seasonal_series() -> None:
    naive = fc.evaluate(SEASONAL, fc.naive, 12, 12)
    seasonal = fc.evaluate(SEASONAL, fc.seasonal_naive, 12, 12)
    ets = fc.evaluate(SEASONAL, fc.ets_auto, 12, 12)

    assert seasonal.mase is not None and naive.mase is not None and ets.mase is not None
    assert seasonal.mase < naive.mase and ets.mase < naive.mase


def test_mase_of_the_reference_is_about_one_on_a_random_walk() -> None:
    walk = np.cumsum(RNG.normal(0, 1, 400)) + 100

    scores = fc.evaluate(walk, fc.naive, 1, None)

    assert scores.mase == pytest.approx(1, abs=0.35)


def test_arima_finds_an_ar1() -> None:
    prediction = fc.arima_auto(AR1, 6, None)

    assert prediction.info is not None and prediction.info.startswith("ARIMA(1,0,")


def test_auto_ets_picks_a_seasonal_form() -> None:
    prediction = fc.ets_auto(SEASONAL, 12, 12)

    assert prediction.info is not None
    assert prediction.info.endswith((",A)", ",M)"))


def test_intervals_cover_about_their_level_on_simulated_data() -> None:
    noise = RNG.normal(10, 2, 600)

    scores = fc.evaluate(noise, fc.mean_model, 20, None)

    assert scores.coverage95 is not None and 0.85 <= scores.coverage95 <= 1


def test_origins_and_short_series() -> None:
    assert fc.origins(120, 12, 12) == [84, 90, 96, 102, 108]
    assert fc.origins(36, 9, 12) == [15, 19, 23, 27]
    assert fc.default_horizon("month", 36) == 9 and fc.default_horizon("day", 1000) == 30
    with pytest.raises(fc.ModelError, match="cycle"):
        fc.seasonal_naive(SEASONAL[:10], 3, 12)


def test_run_model_reports_failures_and_residual_checks() -> None:
    failed = fc.run_model("holt_winters", fc.holt_winters, SEASONAL[:20], 3, 12)
    fitted = fc.run_model("ets_auto", fc.ets_auto, SEASONAL, 12, 12)

    assert failed.error and failed.scores is None
    assert fitted.scores is not None and fitted.residual_check is not None
    assert len(fitted.forecast) == 12


def test_reference_improvement() -> None:
    results = [
        fc.run_model(key, fc.MODELS[key], SEASONAL, 12, 12)
        for key in ("naive", "seasonal_naive", "ets_auto")
    ]

    reference = fc.compare_to_reference(results, 12)

    assert reference == "seasonal_naive"
    assert next(r for r in results if r.key == "seasonal_naive").vs_reference == pytest.approx(0)


def test_manual_sarima() -> None:
    model = fc.sarima_manual((1, 1, 1), (0, 1, 1))

    prediction = model(SEASONAL, 6, 12)

    assert prediction.info == "ARIMA(1,1,1)(0,1,1)[12]"


def test_jobs_cancel_between_models_and_reuse_finished_ones() -> None:
    import threading
    import time

    from quantilo.timeseries.forecasting import ForecastJobs, future_periods

    gate = threading.Event()

    def slow(y, h, season):
        gate.wait(5)
        return fc.naive(y, h, season)

    jobs = ForecastJobs(workers=1)
    values: list[float | None] = [float(v) for v in AR1[:60]]
    periods = [f"{2000 + i // 12}-{i % 12 + 1:02d}" for i in range(60)]
    job = jobs.start(
        ("a",), values, periods, "month", None, 3, {"naive": slow, "mean": fc.mean_model}
    )
    for _ in range(200):  # cancel once the first model is really running
        if job.current == "naive":
            break
        time.sleep(0.01)
    jobs.cancel(job.id)
    gate.set()
    for _ in range(100):
        if job.status != "running":
            break
        time.sleep(0.05)
    assert job.status == "cancelled" and job.done == 1

    finished = jobs.start(("b",), values, periods, "month", None, 3, {"naive": fc.naive})
    for _ in range(100):
        if finished.status != "running":
            break
        time.sleep(0.05)
    assert finished.status == "done"
    assert jobs.start(("b",), values, periods, "month", None, 3, {"naive": fc.naive}) is finished
    assert future_periods("2004-12", "month", 2) == ["2005-01", "2005-02"]


@pytest.mark.parametrize("key", ["tslm", "gbm_lags", "mlp_lags"])
def test_regression_and_lag_models_beat_naive_on_a_seasonal_series(key: str) -> None:
    naive = fc.evaluate(SEASONAL, fc.naive, 12, 12)
    model = fc.evaluate(SEASONAL, fc.MODELS[key], 12, 12)

    assert model.mase is not None and naive.mase is not None and model.mase < naive.mase


def test_tslm_recovers_the_trend() -> None:
    prediction = fc.linear_trend_season(SEASONAL, 12, 12)

    assert prediction.info == "Trend + seasonal dummies"
    assert prediction.mean[-1] - prediction.mean[0] == pytest.approx(0.3 * 11, abs=6)


# --- 028: intermittent demand, WAPE, backtests, conformal intervals, combination ---------------

SPARSE = np.array([0, 3, 0, 0, 6, 0], dtype=float)


def test_croston_and_sba_by_hand() -> None:
    # sizes 3, 6 and intervals 2, 3: z = 3 + 0.1 (6 - 3) = 3.3, p = 2 + 0.1 (3 - 2) = 2.1
    assert fc.croston(SPARSE, 2, None).mean[0] == pytest.approx(3.3 / 2.1)
    assert fc.sba(SPARSE, 2, None).mean[0] == pytest.approx(0.95 * 3.3 / 2.1)


def test_tsb_by_hand() -> None:
    # probability 1 → 0.9 → 0.81 → 0.829 (demand) → 0.7461; size 3 → 3.3
    assert fc.tsb(SPARSE, 1, None).mean[0] == pytest.approx(0.7461 * 3.3)


def test_croston_type_models_refuse_negative_or_all_zero_series() -> None:
    with pytest.raises(fc.ModelError):
        fc.croston(np.zeros(20), 3, None)
    with pytest.raises(fc.ModelError):
        fc.tsb(np.array([0, -1, 2, 0, 3.0]), 3, None)


def test_demand_pattern_classification() -> None:
    rng = np.random.default_rng(3)
    lumpy = np.where(rng.random(200) < 0.2, rng.exponential(50, 200), 0)
    smooth = rng.normal(100, 5, 200)

    pattern = fc.demand_pattern(lumpy)
    assert pattern is not None and pattern.intermittent
    assert pattern.adi > 1.32 and pattern.kind in ("intermittent", "lumpy")
    assert fc.demand_pattern(smooth) is None


def test_intermittent_models_beat_naive_on_sparse_demand() -> None:
    rng = np.random.default_rng(5)
    demand = np.where(rng.random(150) < 0.25, rng.poisson(8, 150), 0).astype(float)
    naive = fc.evaluate(demand, fc.naive, 10, None)
    for model in (fc.croston, fc.sba, fc.tsb, fc.adida):
        scores = fc.evaluate(demand, model, 10, None)
        assert scores.mae is not None and naive.mae is not None and scores.mae < naive.mae
        assert scores.mape is None and scores.wape is not None  # zeros: MAPE is meaningless


def test_wape_is_total_error_over_total_value() -> None:
    y = np.array([10.0] * 20 + [20.0] * 5)
    scores = fc.evaluate(y, fc.naive, 5, None)  # last origin forecasts 10 for five 20s
    assert scores.wape is not None and scores.wape > 0


def test_run_model_keeps_one_backtest_per_origin() -> None:
    result = fc.run_model("ets_auto", fc.ets_auto, SEASONAL, 12, 12)

    assert len(result.backtests) == len(fc.origins(len(SEASONAL), 12, 12))
    assert all(len(b.forecast) == 12 for b in result.backtests)


def test_lag_models_get_conformal_intervals() -> None:
    result = fc.run_model("gbm_lags", fc.MODELS["gbm_lags"], SEASONAL, 12, 12)

    assert result.info is not None and "conformal" in result.info
    widths = np.array(result.high95) - np.array(result.low95)
    assert np.all(widths > 0) and np.all(np.diff(widths) >= 0)


def test_combination_scores_without_hindsight() -> None:
    keys = ["naive", "seasonal_naive", "ets_auto", "tslm"]
    results = [fc.run_model(k, fc.MODELS[k], SEASONAL, 12, 12) for k in keys]
    combined = fc.combine(results, SEASONAL, 12, 12)

    assert combined.scores is not None and combined.scores.mase is not None
    assert combined.info is not None and combined.info.startswith("Mean of")
    worst = max(r.scores.mase for r in results if r.scores and r.scores.mase)
    assert combined.scores.mase < worst
    assert len(combined.forecast) == 12
    assert combined.backtests[0].low95[0] is None  # no earlier error at the first origin


def test_combination_needs_two_scored_models() -> None:
    only = [fc.run_model("naive", fc.naive, SEASONAL, 12, 12)]
    assert fc.combine(only, SEASONAL, 12, 12).error is not None


def test_arima_bands_come_from_the_residuals_when_the_native_ones_are_not_numbers() -> None:
    """Linux's BLAS gave NaN forecast variances on SEASONAL where Windows did not."""
    fit = fc._sarimax(SEASONAL, (1, 1, 1), (0, 0, 0, 0))
    native = fit.get_forecast

    class Broken:
        def __init__(self, h, exog=None):
            self.inner = native(h, exog=exog)
            self.predicted_mean = self.inner.predicted_mean

        def conf_int(self, alpha):
            return np.full_like(np.asarray(self.inner.conf_int(alpha=alpha)), np.nan)

    fit.get_forecast = Broken
    prediction = fc._arima_prediction(fit, 12, "ARIMA(1,1,1)")
    assert np.isfinite(prediction.low95).all() and np.isfinite(prediction.high95).all()
    assert np.all(prediction.low95 < prediction.low80)
    assert np.all(prediction.low80 < prediction.mean)
    widths = prediction.high95 - prediction.low95
    assert np.all(np.diff(widths) > 0)  # wider further out
