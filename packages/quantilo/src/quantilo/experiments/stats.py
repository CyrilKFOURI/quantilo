"""The tests of an experiment: one variant against the control, for each kind of metric.

Every estimate is "variant − control" with its 95 % interval (or 1 − α), the standard errors of
each side (for the lift's interval and the power), and the p-value of the test that suits the
metric. The reference implementations are statsmodels and SciPy (tests/test_experiments.py).
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest
from statsmodels.stats.rates import confint_poisson_2indep, test_poisson_2indep

from quantilo.compare.stats import GroupStats, mann_whitney, welch_t
from quantilo.variable.source import clean

BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CELLS = 4_000_000  # resampled values held in memory at once
POWER_TARGET = 0.8


@dataclass
class Estimate:
    test: str  # z_proportions, welch_t, poisson, delta_ratio
    control: float
    variant: float
    se_control: float  # standard error of each side's value
    se_variant: float
    diff: float  # variant − control
    low: float | None  # interval of the difference
    high: float | None
    statistic: float | None
    p_value: float | None

    @property
    def se(self) -> float:
        return math.hypot(self.se_control, self.se_variant)


def _quiet[T](compute: Callable[[], T]) -> T:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute()


def z_value(alpha: float) -> float:
    return float(stats.norm.ppf(1 - alpha / 2))


# --- Rates (0/1) --------------------------------------------------------------------------------


def two_proportions(
    successes_control: int, n_control: int, successes_variant: int, n_variant: int, alpha: float
) -> Estimate:
    """z-test of two proportions (pooled, as statsmodels' `proportions_ztest`) and Newcomb's
    interval of the difference (`confint_proportions_2indep(method="newcomb")`)."""
    pc, pv = successes_control / n_control, successes_variant / n_variant
    se_c = math.sqrt(pc * (1 - pc) / n_control)
    se_v = math.sqrt(pv * (1 - pv) / n_variant)
    statistic = p_value = None
    if 0 < successes_control + successes_variant < n_control + n_variant:
        z, p = _quiet(
            lambda: proportions_ztest(
                [successes_variant, successes_control], [n_variant, n_control]
            )
        )
        statistic, p_value = clean(z), clean(p)
    low, high = _quiet(
        lambda: confint_proportions_2indep(
            successes_variant,
            n_variant,
            successes_control,
            n_control,
            method="newcomb",
            alpha=alpha,
        )
    )
    return Estimate(
        "z_proportions", pc, pv, se_c, se_v, pv - pc, clean(low), clean(high), statistic, p_value
    )


# --- Means --------------------------------------------------------------------------------------


def group_stats(values: np.ndarray) -> GroupStats:
    return GroupStats(len(values), float(values.mean()), float(values.var(ddof=1)))


def welch(control: GroupStats, variant: GroupStats, alpha: float) -> Estimate:
    """Welch's t (compare.stats.welch_t) and its interval, with Welch–Satterthwaite's df."""
    test = welch_t(variant, control)
    se_c = math.sqrt(control.var / control.n)
    se_v = math.sqrt(variant.var / variant.n)
    diff = variant.mean - control.mean
    low = high = None
    se2 = se_c**2 + se_v**2
    if se2 > 0:
        df = se2**2 / (se_c**4 / (control.n - 1) + se_v**4 / (variant.n - 1))
        margin = float(stats.t.ppf(1 - alpha / 2, df)) * math.sqrt(se2)
        low, high = diff - margin, diff + margin
    return Estimate(
        "welch_t",
        control.mean,
        variant.mean,
        se_c,
        se_v,
        diff,
        low,
        high,
        test.statistic,
        test.p_value,
    )


def rank_check(control: np.ndarray, variant: np.ndarray) -> float | None:
    """Mann–Whitney's p-value: do the values of one side tend to be larger?"""
    return mann_whitney(variant, control).p_value


def _resampled_means(values: np.ndarray, rng: np.random.Generator, draws: int) -> np.ndarray:
    chunk = max(1, BOOTSTRAP_CELLS // max(len(values), 1))
    means = []
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        picks = rng.integers(0, len(values), size=(size, len(values)))
        means.append(values[picks].mean(axis=1))
    return np.concatenate(means)


def bootstrap_diff(
    control: np.ndarray, variant: np.ndarray, alpha: float, draws: int = BOOTSTRAP_DRAWS
) -> tuple[float, float]:
    """Percentile interval of the difference of means, seeded (same data, same interval)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs = _resampled_means(variant, rng, draws) - _resampled_means(control, rng, draws)
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


# --- Counts per unit ----------------------------------------------------------------------------


def dispersion(values: np.ndarray) -> float | None:
    """Variance ÷ mean: about 1 for Poisson counts."""
    mean = float(values.mean())
    return float(values.var(ddof=1)) / mean if mean > 0 and len(values) > 1 else None


def poisson(control: np.ndarray, variant: np.ndarray, alpha: float) -> Estimate:
    """Difference of two Poisson rates per unit (statsmodels' score test and interval)."""
    total_c, total_v = float(control.sum()), float(variant.sum())
    n_c, n_v = len(control), len(variant)
    rate_c, rate_v = total_c / n_c, total_v / n_v
    result = _quiet(
        lambda: test_poisson_2indep(total_v, n_v, total_c, n_c, compare="diff", method="score")
    )
    low, high = _quiet(
        lambda: confint_poisson_2indep(
            total_v, n_v, total_c, n_c, compare="diff", method="score", alpha=alpha
        )
    )
    return Estimate(
        "poisson",
        rate_c,
        rate_v,
        math.sqrt(rate_c / n_c),
        math.sqrt(rate_v / n_v),
        rate_v - rate_c,
        clean(low),
        clean(high),
        clean(result.statistic),
        clean(result.pvalue),
    )


# --- Ratios of sums -----------------------------------------------------------------------------


def ratio_se(numerator: np.ndarray, denominator: np.ndarray) -> tuple[float, float]:
    """Σ numerator ÷ Σ denominator over units, and its standard error by the delta method."""
    n = len(numerator)
    mean_y, mean_x = float(numerator.mean()), float(denominator.mean())
    ratio = mean_y / mean_x
    if n < 2:
        return ratio, math.nan
    var_y = float(numerator.var(ddof=1))
    var_x = float(denominator.var(ddof=1))
    cov = float(np.cov(numerator, denominator, ddof=1)[0, 1])
    variance = (var_y - 2 * ratio * cov + ratio**2 * var_x) / (mean_x**2 * n)
    return ratio, math.sqrt(max(variance, 0.0))


def delta_ratio(
    control: tuple[np.ndarray, np.ndarray], variant: tuple[np.ndarray, np.ndarray], alpha: float
) -> Estimate:
    """z-test of two ratios of sums (revenue ÷ orders), each unit counting once."""
    rc, se_c = ratio_se(*control)
    rv, se_v = ratio_se(*variant)
    diff = rv - rc
    se = math.hypot(se_c, se_v)
    low = high = statistic = p_value = None
    if math.isfinite(se) and se > 0:
        margin = z_value(alpha) * se
        low, high = diff - margin, diff + margin
        statistic = diff / se
        p_value = float(2 * stats.norm.sf(abs(statistic)))
    return Estimate("delta_ratio", rc, rv, se_c, se_v, diff, low, high, statistic, p_value)


def bootstrap_ratio(
    control: tuple[np.ndarray, np.ndarray],
    variant: tuple[np.ndarray, np.ndarray],
    alpha: float,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[float, float]:
    """Percentile interval of the difference of two ratios of sums, units resampled whole."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    def ratios(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        n = len(numerator)
        chunk = max(1, BOOTSTRAP_CELLS // max(n, 1))
        found = []
        for start in range(0, draws, chunk):
            picks = rng.integers(0, n, size=(min(chunk, draws - start), n))
            found.append(numerator[picks].sum(axis=1) / denominator[picks].sum(axis=1))
        return np.concatenate(found)

    diffs = ratios(*variant) - ratios(*control)
    low, high = np.nanquantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


# --- Lift, power, corrections -------------------------------------------------------------------


def lift(estimate: Estimate, alpha: float) -> tuple[float | None, float | None, float | None]:
    """(variant − control) ÷ control and its interval, by the delta method on the ratio of the
    two independent values."""
    c, v = estimate.control, estimate.variant
    if c == 0:
        return None, None, None
    relative = v / c - 1
    se = math.sqrt(estimate.se_variant**2 / c**2 + v**2 * estimate.se_control**2 / c**4)
    margin = z_value(alpha) * se
    return relative, relative - margin, relative + margin


def power_and_mde(
    estimate: Estimate, n_control: int, n_variant: int, alpha: float
) -> tuple[float | None, float | None]:
    """Power reached for the observed difference, and the smallest difference detectable 80 % of
    the time with these groups (statsmodels' `NormalIndPower` on the standardised effect)."""
    se = estimate.se
    if not math.isfinite(se) or se <= 0 or n_control < 2 or n_variant < 2:
        return None, None
    # the spread of one unit that gives this standard error with these group sizes
    sigma = se / math.sqrt(1 / n_control + 1 / n_variant)
    ratio = n_variant / n_control
    analysis = NormalIndPower()
    power = _quiet(
        lambda: analysis.power(
            abs(estimate.diff) / sigma,
            nobs1=n_control,
            alpha=alpha,
            ratio=ratio,  # type: ignore[arg-type]  (statsmodels' stubs say int)
        )
    )
    effect = _quiet(
        lambda: analysis.solve_power(
            effect_size=None, nobs1=n_control, alpha=alpha, power=POWER_TARGET, ratio=ratio
        )
    )
    return clean(power), clean(float(effect) * sigma)


def holm(p_values: list[float | None]) -> list[float | None]:
    """Holm's adjustment over every comparison that has a p-value."""
    known = [(i, p) for i, p in enumerate(p_values) if p is not None]
    adjusted: list[float | None] = [None] * len(p_values)
    if not known:
        return adjusted
    corrected = multipletests([p for _, p in known], method="holm")[1]
    for (i, _), value in zip(known, corrected, strict=True):
        adjusted[i] = float(value)
    return adjusted


def srm(observed: list[int], shares: list[float]) -> tuple[float | None, float | None]:
    """χ² goodness of fit of the units per variant against the planned split."""
    total = sum(observed)
    if total == 0 or len(observed) < 2:
        return None, None
    weight = sum(shares)
    expected = [total * share / weight for share in shares]
    result = stats.chisquare(observed, expected)
    return clean(result.statistic), clean(result.pvalue)


@dataclass
class Adjusted:
    values: np.ndarray
    theta: float
    correlation: float
    reduction: float  # share of the metric's variance removed


def cuped(metric: np.ndarray, pre: np.ndarray) -> Adjusted | None:
    """CUPED: the metric minus θ × (pre − its mean), θ = cov(metric, pre) ÷ var(pre), pooled over
    every variant (the assignment is random, so pre does not depend on it)."""
    var_pre = float(pre.var(ddof=1)) if len(pre) > 1 else 0.0
    var_metric = float(metric.var(ddof=1)) if len(metric) > 1 else 0.0
    if var_pre == 0 or var_metric == 0:
        return None
    theta = float(np.cov(metric, pre, ddof=1)[0, 1]) / var_pre
    adjusted = metric - theta * (pre - pre.mean())
    correlation = float(np.corrcoef(metric, pre)[0, 1])
    reduction = 1 - float(adjusted.var(ddof=1)) / var_metric
    return Adjusted(adjusted, theta, correlation, reduction)


def skewness(values: np.ndarray) -> float | None:
    if len(values) < 3 or float(np.ptp(values)) == 0:
        return None
    return clean(_quiet(lambda: stats.skew(values)))
