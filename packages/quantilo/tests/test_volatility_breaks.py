import numpy as np
import pytest
from scipy import stats

from quantilo.timeseries import breaks as br
from quantilo.timeseries import residuals as rs
from quantilo.timeseries import volatility as vo

RNG = np.random.default_rng(8)
STEADY = np.random.default_rng(30).normal(0, 1, 300)
GROWING = RNG.normal(0, 1, 300) * np.linspace(0.5, 4, 300)
ARCH = np.zeros(600)
for t in range(1, 600):
    ARCH[t] = RNG.normal() * np.sqrt(0.2 + 0.7 * ARCH[t - 1] ** 2)
STEPS = np.concatenate([np.full(60, 10.0), np.full(60, 18.0), np.full(60, 12.0)]) + RNG.normal(
    0, 1, 180
)


# --- volatility ---------------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(vo.VOLATILITY))
def test_steady_noise_has_no_heteroscedasticity(key: str) -> None:
    assert vo.VOLATILITY[key](STEADY).significant is False


@pytest.mark.parametrize(
    "key",
    [
        "breusch_pagan",
        "white",
        "goldfeld_quandt",
        "levene",
        "brown_forsythe",
        "bartlett",
        "fligner",
    ],
)
def test_growing_variance_is_detected(key: str) -> None:
    assert vo.VOLATILITY[key](GROWING).significant is True


@pytest.mark.parametrize("key", ["arch_lm", "mcleod_li"])
def test_arch_effects_are_detected(key: str) -> None:
    assert vo.VOLATILITY[key](ARCH).significant is True


def test_levene_matches_scipy() -> None:
    result = vo.levene(GROWING)
    expected = stats.levene(*np.array_split(GROWING, 4), center="mean")

    assert result.statistic == pytest.approx(expected.statistic)


def test_summary_says_which_way_and_suggests_a_log() -> None:
    level = np.linspace(10, 100, 300)
    series = level + RNG.normal(0, 1, 300) * level * 0.1  # spread follows the level
    spread = vo.rolling_spread(series - level, 20)

    found = vo.summary(series, spread, level)

    assert found.rising is True and found.log_helps is True


def test_rolling_spread() -> None:
    values = vo.rolling_spread(np.array([1.0, 3.0, 5.0, 7.0]), 2)

    assert values[0] is None and values[1] == pytest.approx(np.std([1, 3], ddof=1))


# --- residuals ----------------------------------------------------------------------------------


def test_remainder_uses_stl_when_two_cycles_fit() -> None:
    t = np.arange(120)
    series = 50 + 0.2 * t + 5 * np.sin(2 * np.pi * t / 12) + RNG.normal(0, 1, 120)

    stl = rs.remainder(series, 12)
    line = rs.remainder(series[:20], 12)

    assert stl.method == "STL remainder" and np.std(stl.values) < 2
    assert line.method == "straight-line residuals"
    assert stl.values + stl.fitted == pytest.approx(series)


def test_degenerate_robust_stl_falls_back_to_the_ordinary_one() -> None:
    # three yearly cycles of weekly data: the robust STL leaves a remainder of exact zeros
    t = np.arange(156)
    series = 40_000 + 120 * t + 15_000 * np.sin(2 * np.pi * t / 52) + RNG.normal(0, 800, 156)
    series[[30, 90]] += 9_000

    found = rs.remainder(series, 52)
    mad = np.median(np.abs(found.values - np.median(found.values)))

    assert mad > 1  # a usable scale for robust z-scores
    assert len(br.anomalies(found.values, series)) < 20


def test_distribution_reuses_the_variable_analysis() -> None:
    found = rs.distribution(STEADY, "series", None)

    assert found.describe.count == 300 and found.describe.mean == pytest.approx(STEADY.mean())
    assert {t.key for t in found.normality.tests} >= {"shapiro", "jarque_bera"}
    assert found.qq is not None and found.density is not None
    assert sum(b["count"] for b in found.bins) == 300
    assert found.fit.laws and found.fit.laws[0].key in ("normal", "student_t", "logistic")


def test_returns_and_their_guard() -> None:
    values, _ = rs.values_for(np.array([100.0, 110.0, 99.0]), "returns", None)

    assert values == pytest.approx([0.1, -0.1])
    with pytest.raises(rs.DistributionError, match="never 0"):
        rs.values_for(np.array([0.0, 1.0, 2.0]), "returns", None)


# --- breaks -------------------------------------------------------------------------------------


@pytest.mark.parametrize("search", ["pelt", "binseg"])
def test_level_shifts_are_found(search: str) -> None:
    found = br.find_breaks(STEPS, search)  # type: ignore[arg-type]

    assert len(found.points) == 2
    assert abs(found.points[0] - 60) <= 2 and abs(found.points[1] - 120) <= 2
    assert [round(s.mean) for s in found.segments] == [10, 18, 12]


def test_no_break_in_noise_and_sensitivity() -> None:
    assert br.find_breaks(STEADY).points == []
    assert len(br.find_breaks(STEPS, sensitivity="low").points) <= 2


def test_mean_and_variance_cost_finds_a_variance_change() -> None:
    series = np.concatenate([RNG.normal(0, 1, 150), RNG.normal(0, 5, 150)])

    found = br.find_breaks(series, cost="mean_var")

    assert any(abs(p - 150) <= 10 for p in found.points)


def test_trend_cost_does_not_cut_a_steady_trend_into_steps() -> None:
    steady = 0.5 * np.arange(200) + RNG.normal(0, 2, 200)

    as_means = br.find_breaks(steady, cost="mean")
    as_trend = br.find_breaks(steady, cost="trend")

    assert len(as_means.points) >= 3  # the staircase
    assert as_trend.points == [] and as_trend.segments[0].slope == pytest.approx(0.5, abs=0.05)


def test_trend_cost_finds_a_change_of_slope() -> None:
    t = np.arange(200, dtype=float)
    kinked = np.where(t < 100, 0.2 * t, 20 + 1.5 * (t - 100)) + RNG.normal(0, 1, 200)

    found = br.find_breaks(kinked, cost="trend")

    assert len(found.points) == 1 and abs(found.points[0] - 100) <= 5
    assert found.segments[1].slope == pytest.approx(1.5, abs=0.1)


@pytest.mark.parametrize("key", list(br.BREAK_TESTS))
def test_single_break_tests(key: str) -> None:
    shifted = np.concatenate([np.zeros(80), np.full(80, 2.0)]) + RNG.normal(0, 1, 160)

    assert br.BREAK_TESTS[key](shifted).significant is True
    assert br.BREAK_TESTS[key](STEADY[:160]).significant is False


def test_pettitt_locates_the_change() -> None:
    shifted = np.concatenate([np.zeros(50), np.full(50, 3.0)]) + RNG.normal(0, 0.5, 100)

    found = br.pettitt(shifted)

    assert found.value is not None and abs(found.value - 51) <= 2


# --- anomalies ----------------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(br.ANOMALIES))
def test_injected_spikes_are_flagged(method: str) -> None:  # every method, learned ones included
    resid = np.random.default_rng(4).normal(0, 1, 300)
    resid[[50, 200]] += 12
    series = resid + 100

    found = br.anomalies(resid, series, method, threshold=3.5 if method != "iqr" else 3)  # type: ignore[arg-type]

    assert {50, 200} <= {a.index for a in found}
    assert found[0].expected == pytest.approx(100, abs=1)
    assert len(found) <= 6


def test_gesd_flags_only_real_outliers() -> None:
    clean_noise = np.random.default_rng(4).normal(0, 1, 200)
    spiked = clean_noise.copy()
    spiked[[20, 120, 170]] += [9, -8, 10]

    assert {a.index for a in br.anomalies(spiked, spiked + 50, "gesd")} == {20, 120, 170}
    assert br.anomalies(clean_noise, clean_noise, "gesd") == []


@pytest.mark.parametrize("method", ["isolation_forest", "lof", "autoencoder"])
def test_learned_methods_stay_quiet_on_noise(method: str) -> None:
    noise = np.random.default_rng(10).normal(0, 1, 300)

    assert len(br.anomalies(noise, noise, method)) <= 3  # type: ignore[arg-type]  # ≤ 1 % false alarms
