"""056: the right tail (POT / GPD, return levels, Hill, mean excess, VaR / ES) and the robust
center and spread of a numeric variable."""

import time

import numpy as np
import pytest

from quantilo.variable.analyze import analyze
from quantilo.variable.numeric import describe
from quantilo.variable.robust import robust
from quantilo.variable.tails import (
    analyze_tail,
    fit_gpd,
    hill,
    level_at,
    linearity,
    mean_excess,
    tail_rule,
)

from .test_variable import source_of


def _tail(values):
    source = source_of([float(v) for v in values])
    summary = describe(source)
    return analyze_tail(source, summary, tail_rule(source, summary))


def pareto(size: int = 20_000, alpha: float = 2.0, seed: int = 3) -> np.ndarray:
    """Pareto(α, xₘ = 1): tail index ξ = 1 / α."""
    return np.random.default_rng(seed).pareto(alpha, size) + 1


# --- Fit and formulas -----------------------------------------------------------------------


def test_gpd_fit_recovers_the_shape() -> None:
    from scipy import stats

    y = np.asarray(stats.genpareto.rvs(0.3, scale=2.0, size=5_000, random_state=1), dtype=float)
    xi, sigma = fit_gpd(y)
    assert xi == pytest.approx(0.3, abs=0.06)
    assert sigma == pytest.approx(2.0, rel=0.1)


def test_return_level_formula_matches_the_gpd_quantile() -> None:
    from scipy import stats

    u, xi, sigma, rate = 10.0, 0.4, 3.0, 0.05
    # exceeded once in 1,000 rows: P(X > x) = 0.001 = rate · P(Y > x − u)
    expected = u + stats.genpareto.ppf(1 - 0.001 / rate, xi, scale=sigma)
    assert level_at(u, xi, sigma, rate, 1_000) == pytest.approx(expected, rel=1e-9)
    assert level_at(u, 0.0, sigma, rate, 1_000) == pytest.approx(
        u + sigma * np.log(1_000 * rate), rel=1e-9
    )


def test_hill_estimates_the_pareto_index() -> None:
    curve = hill(np.sort(pareto(alpha=2.0)))
    assert curve is not None
    middle = [xi for k, xi in zip(curve.x, curve.y, strict=True) if 200 <= k <= 1_000]
    assert np.median(middle) == pytest.approx(0.5, abs=0.08)
    assert all(lo <= xi <= hi for lo, xi, hi in zip(curve.low, curve.y, curve.high, strict=True))


def test_mean_excess_is_linear_for_a_pareto_and_counts_fall() -> None:
    ordered = np.sort(pareto())
    curve = mean_excess(ordered, float(np.median(ordered)))
    assert len(curve.x) > 20
    assert curve.count == sorted(curve.count, reverse=True)
    assert min(curve.count) >= 20
    r2 = linearity(curve, float(np.quantile(ordered, 0.95)))
    assert r2 is not None and r2 > 0.7


# --- Acceptance scenarios -------------------------------------------------------------------


def test_pareto_tail_shape_and_return_level() -> None:
    tail = _tail(pareto(alpha=2.0))

    assert tail.rule.fires
    assert tail.reason is None
    assert tail.threshold is not None and tail.threshold.level == 0.95
    assert tail.threshold.exceedances >= 950
    assert tail.xi is not None and 0.4 <= tail.xi <= 0.6
    assert tail.xi_low is not None and tail.xi_high is not None
    assert tail.xi_low < 0.5 < tail.xi_high
    thousand = next(r for r in tail.return_levels if r.period == 1_000)
    true = 1_000 ** (1 / 2)  # Pareto(2): P(X > x) = x^-2
    assert thousand.value == pytest.approx(true, rel=0.2)
    assert thousand.low is not None and thousand.low < thousand.value
    assert thousand.high is None or thousand.high > thousand.value
    assert not thousand.beyond_data
    assert thousand.empirical is not None
    ten_thousand = next(r for r in tail.return_levels if r.period == 10_000)
    assert ten_thousand.value > thousand.value
    assert tail.conclusion.startswith("Heavy right tail (ξ = 0.")
    assert "× the 99th percentile" in tail.conclusion
    assert tail.hill is not None and tail.mean_excess is not None
    assert tail.curve is not None and tail.observed is not None
    assert len(tail.stability) == 3


def test_exponential_tail_is_not_different_from_zero() -> None:
    values = np.random.default_rng(5).exponential(3.0, 20_000)
    tail = _tail(values)
    assert tail.xi is not None and abs(tail.xi) < 0.1
    assert tail.xi_low is not None and tail.xi_high is not None
    assert tail.xi_low < 0 < tail.xi_high
    assert tail.conclusion.startswith("Exponential-like tail")
    # a light tail with a kurtosis of about 6: the rule fires on the kurtosis only
    assert tail.rule.fires


def test_uniform_tail_is_bounded_near_its_maximum() -> None:
    tail = _tail(np.random.default_rng(6).uniform(0, 10, 20_000))
    assert tail.xi is not None and tail.xi < 0
    assert tail.endpoint is not None and tail.endpoint == pytest.approx(10, abs=0.3)
    assert tail.conclusion.startswith("Bounded tail")
    assert not tail.rule.fires  # no high kurtosis, no heavy shape


def test_normal_column_does_not_fire() -> None:
    source = source_of([float(v) for v in np.random.default_rng(1).normal(100, 10, 5_000)])
    rule = tail_rule(source, describe(source))
    assert not rule.fires
    assert rule.reason.startswith("Light tail")


def test_too_few_values_do_not_fire() -> None:
    source = source_of([float(v) for v in pareto(400)])
    rule = tail_rule(source, describe(source))
    assert not rule.fires
    assert "at least 500 values" in rule.reason


def test_many_negative_values_do_not_fire() -> None:
    values = pareto(5_000)
    values[:500] = -values[:500]
    source = source_of([float(v) for v in values])
    rule = tail_rule(source, describe(source))
    assert not rule.fires
    assert "right tail of positive values" in rule.reason


def test_a_few_negative_values_keep_the_tail() -> None:
    values = pareto(5_000)
    values[:100] = -1.0
    tail = _tail(values)
    assert tail.rule.fires and tail.xi is not None
    assert any("negative" in note for note in tail.notes)


def test_small_column_lowers_the_threshold_to_keep_50_exceedances() -> None:
    tail = _tail(pareto(600))
    assert tail.threshold is not None
    assert tail.threshold.exceedances >= 50
    assert 0.90 <= tail.threshold.level <= 0.95


def test_var_and_expected_shortfall_read_every_row() -> None:
    values = pareto(10_000)
    tail = _tail(values)
    by_level = {r.level: r for r in tail.risk}
    var = float(np.quantile(values, 0.95))
    assert by_level[0.95].var == pytest.approx(var, rel=1e-9)
    assert by_level[0.95].es == pytest.approx(values[values >= var].mean(), rel=1e-9)
    assert by_level[0.99].es > by_level[0.99].var > by_level[0.95].var


def test_large_column_is_sampled_deterministically_and_fast() -> None:
    values = pareto(300_000, seed=11)
    started = time.perf_counter()
    first = _tail(values)
    seconds = time.perf_counter() - started
    second = _tail(values)
    assert first.sampled and first.sample == 200_000 and first.n == 300_000
    assert first.xi == second.xi
    assert any("sample of 200,000" in note for note in first.notes)
    assert seconds < 10


def test_profile_carries_the_rule_and_the_robust_block() -> None:
    profile = analyze(source_of([float(v) for v in pareto(5_000)]))
    assert profile.numeric is not None
    assert profile.numeric.tail_rule is not None and profile.numeric.tail_rule.fires
    assert profile.numeric.robust is not None and profile.numeric.robust.fires


# --- Robust center and spread (R-ROB-1) -----------------------------------------------------


def test_robust_block_shows_with_many_outliers() -> None:
    rng = np.random.default_rng(2)
    values = np.r_[rng.normal(100, 10, 9_500), np.full(500, 1_000.0)]
    source = source_of([float(v) for v in values])
    found = robust(source, describe(source))
    assert found is not None and found.fires
    assert found.outlier_share == pytest.approx(0.05, abs=0.005)
    assert found.reason.startswith("5.4 % of values are outliers by the IQR rule (above 1 %)")
    assert found.huber_center == pytest.approx(100, abs=2)
    assert found.huber_scale == pytest.approx(10, rel=0.2)
    assert found.mad_sigma == pytest.approx(10, rel=0.1)
    assert found.mean_shift is not None and found.mean_shift > 0.3


def test_robust_block_stays_hidden_on_a_normal_column() -> None:
    source = source_of([float(v) for v in np.random.default_rng(3).normal(0, 1, 5_000)])
    found = robust(source, describe(source))
    assert found is not None and not found.fires


def test_robust_block_without_spread_keeps_the_median() -> None:
    values = [0.0] * 6_000 + [float(v) for v in np.random.default_rng(4).exponential(5, 4_000)]
    source = source_of(values)
    found = robust(source, describe(source))
    assert found is not None
    assert found.huber_center is None
    assert found.note is not None and "MAD is 0" in found.note
