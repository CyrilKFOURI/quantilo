import numpy as np
import pytest
from statsmodels.tsa.stattools import adfuller

from quantilo.timeseries import autocorrelation as ac
from quantilo.timeseries import stationarity as st

RNG = np.random.default_rng(21)
NOISE = np.random.default_rng(30).normal(0, 1, 400)  # a draw without chance autocorrelation
WALK = np.cumsum(RNG.normal(0, 1, 400)) + 50
AR2 = np.zeros(600)
for t in range(2, 600):
    AR2[t] = 0.6 * AR2[t - 1] - 0.3 * AR2[t - 2] + RNG.normal()
SEASONAL = 10 + np.tile([0, 1, 2, 3, 6, 9, 4], 60) + RNG.normal(0, 0.5, 420)


# --- transformations ---------------------------------------------------------------------------


def test_transformations() -> None:
    series = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    assert list(st.transform(series, "diff", None)) == [1, 2, 4, 8]
    assert list(st.transform(series, "seasonal_diff", 2)) == [3, 6, 12]
    assert st.transform(series, "log_diff", None) == pytest.approx([np.log(2)] * 4)
    assert list(st.transform(series, "diff2", None)) == [1, 2, 4]
    assert list(st.transform(series, "seasonal_diff_diff", 2)) == [3, 6]
    with pytest.raises(st.TransformError, match="above 0"):
        st.transform(np.array([1.0, -1.0, 2.0]), "log", None)


# --- unit-root tests ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["adf", "phillips_perron", "df_gls", "zivot_andrews"])
def test_unit_root_tests_reject_for_noise_not_for_a_walk(key: str) -> None:
    method = st.STATIONARITY[key]

    assert method(NOISE, "c").significant is True
    assert method(WALK, "c").significant is False


def test_kpss_and_rur_reject_stationarity_for_a_walk() -> None:
    assert st.kpss_test(WALK).significant is True
    assert st.kpss_test(NOISE).significant is False
    assert st.rur(WALK).p_value is not None


def test_phillips_perron_without_lags_equals_dickey_fuller() -> None:
    result = st.phillips_perron(WALK, "c", lags=0)
    expected = adfuller(WALK, maxlag=0, autolag=None, regression="c")[0]  # type: ignore[arg-type]

    assert result.statistic == pytest.approx(expected, rel=1e-6)


def test_variance_ratio_flags_mean_reversion() -> None:
    walk = st.variance_ratio(WALK)
    reverting = st.variance_ratio(NOISE)  # levels that revert: changes are negatively linked

    assert walk.significant is False and walk.value == pytest.approx(1, abs=0.25)
    assert reverting.significant is True and reverting.value is not None and reverting.value < 1


def test_synthesis_table() -> None:
    walk = st.judge(WALK)
    noise = st.judge(NOISE)

    assert walk is not None and walk.verdict == "unit_root"
    assert noise is not None and noise.verdict == "stationary"


def test_suggestion_differences_a_walk_and_leaves_noise() -> None:
    assert st.suggest(WALK, None) == st.Suggestion("diff", "stationary")
    assert st.suggest(NOISE, None) == st.Suggestion("none", "stationary")


def test_suggestion_is_the_transformation_it_names() -> None:
    twice = np.cumsum(np.cumsum(np.random.default_rng(5).normal(0, 1, 400)))  # two differences

    found = st.suggest(twice, None)

    assert found is not None and found.transform == "diff2"
    verdict = st.judge(st.transform(twice, found.transform, None))
    assert verdict is not None and verdict.verdict == "stationary"


def test_too_short_series_explain() -> None:
    assert st.adf(NOISE[:10]).note and st.adf(NOISE[:10]).p_value is None


# --- autocorrelation ---------------------------------------------------------------------------


def test_white_noise_has_no_memory() -> None:
    gram = ac.correlogram(NOISE, None)

    assert gram.readings == ["white_noise"] and gram.band == pytest.approx(1.96 / 20)
    assert ac.ljung_box(NOISE, None).significant is False


def test_ar2_pacf_cuts_off_at_two() -> None:
    gram = ac.correlogram(AR2, None)

    assert gram.ar_order == 2 and "pacf_cutoff" in gram.readings


def test_walk_decays_slowly() -> None:
    assert "slow_decay" in ac.correlogram(WALK, None).readings


def test_seasonal_spikes() -> None:
    assert "seasonal_spikes" in ac.correlogram(SEASONAL, 7).readings


@pytest.mark.parametrize("key", list(ac.AUTOCORRELATION))
def test_every_method_detects_memory_in_a_walk(key: str) -> None:
    result = ac.AUTOCORRELATION[key](WALK, 12)

    assert result.significant is True, result


@pytest.mark.parametrize("key", ["ljung_box", "box_pierce", "runs", "turning_points", "bartels"])
def test_randomness_tests_stay_quiet_on_noise(key: str) -> None:
    assert ac.AUTOCORRELATION[key](NOISE, 12).significant is False


def test_portmanteau_lags() -> None:
    assert ac.portmanteau_lags(200, 12) == [1, 12, 24]
    assert ac.portmanteau_lags(50, 12) == [1]
    assert ac.portmanteau_lags(200, None) == [1, 10]
