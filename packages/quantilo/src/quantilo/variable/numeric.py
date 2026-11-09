"""Numeric variable: descriptive statistics (every row), normality tests and curves (sample)."""

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from quantilo.variable.source import VariableSource, clean

TEST_SAMPLE = 5_000  # Shapiro-Wilk is only reliable up to 5,000 values
CURVE_SAMPLE = 20_000
POINTS = 200
EXACT_QUANTILES = 200_000  # beyond, curve quantiles come from a t-digest (error well under 1%)
MIN_TEST_VALUES = 8
EXACT_DISTINCT = 200_000
PERCENTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass
class Describe:
    rows: int
    count: int  # finite values
    missing: int
    infinite: int
    zeros: int
    negatives: int
    distinct: int
    distinct_approx: bool  # HyperLogLog estimate (about ±2%) beyond EXACT_DISTINCT values
    mean: float | None
    std: float | None
    cv: float | None
    min: float | None
    p1: float | None
    p5: float | None
    p10: float | None
    q1: float | None
    median: float | None
    q3: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    max: float | None
    iqr: float | None
    mad: float | None
    skewness: float | None
    kurtosis: float | None  # excess kurtosis (normal = 0)
    sum: float | None


@dataclass
class TestResult:
    key: str  # shapiro, dagostino, anderson, jarque_bera, lilliefors
    statistic: float | None
    p_value: float | None
    n: int
    note: str | None = None  # why the test was skipped, or how its p-value is obtained


@dataclass
class Normality:
    n: int  # finite values in the column
    sample: int  # values the tests ran on
    tests: list[TestResult] = field(default_factory=list)
    reason: str | None = None  # set when no test applies


@dataclass
class Curve:
    x: list[float]
    y: list[float]


@dataclass
class QQ:
    """Sample quantiles against the standard normal's, with the line through Q1 and Q3."""

    theoretical: list[float]
    sample: list[float]
    slope: float
    intercept: float


def describe(source: VariableSource) -> Describe:
    values, numbers = source.values, source.numbers
    rows, valid, infinite, estimate = source.one(
        f"SELECT count(*), count(t), "
        f"count(*) FILTER (WHERE t IS NOT NULL AND NOT isfinite(CAST(t AS DOUBLE))), "
        f"approx_count_distinct(t) FROM {values}"
    )
    approximate = valid > EXACT_DISTINCT
    if approximate:
        distinct = estimate
    else:
        (distinct,) = source.one(f"SELECT count(DISTINCT t) FROM {values}")
    probabilities = ", ".join(str(p) for p in PERCENTILES)
    row = source.one(
        f"SELECT count(x), count(*) FILTER (WHERE x = 0), count(*) FILTER (WHERE x < 0), "
        f"avg(x), stddev_samp(x), min(x), max(x), mad(x), skewness(x), kurtosis(x), sum(x), "
        f"quantile_cont(x, [{probabilities}]) FROM {numbers}"
    )
    count, zeros, negatives, mean, std, low, high, mad, skew, kurt, total, quantiles = row
    q = [clean(v) for v in quantiles] if quantiles else [None] * len(PERCENTILES)
    mean, std = clean(mean), clean(std)
    q1, q3 = q[3], q[5]
    return Describe(
        rows=int(rows),
        count=int(count),
        missing=int(rows - valid),
        infinite=int(infinite),
        zeros=int(zeros),
        negatives=int(negatives),
        distinct=int(distinct),
        distinct_approx=approximate,
        mean=mean,
        std=std,
        cv=std / abs(mean) if std is not None and mean else None,
        min=clean(low),
        p1=q[0],
        p5=q[1],
        p10=q[2],
        q1=q1,
        median=q[4],
        q3=q3,
        p90=q[6],
        p95=q[7],
        p99=q[8],
        max=clean(high),
        iqr=q3 - q1 if q1 is not None and q3 is not None else None,
        mad=clean(mad),
        skewness=clean(skew),
        kurtosis=clean(kurt),
        sum=clean(total),
    )


def _run(key: str, sample: np.ndarray, compute) -> TestResult:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, p_value, note = compute(sample)
    except (ValueError, FloatingPointError) as error:
        return TestResult(key, None, None, len(sample), note=str(error))
    return TestResult(key, clean(statistic), clean(p_value), len(sample), note)


def _shapiro(x: np.ndarray):
    result = stats.shapiro(x)
    return result.statistic, result.pvalue, None


def _dagostino(x: np.ndarray):
    if len(x) < 20:
        raise ValueError("Needs at least 20 values.")
    result = stats.normaltest(x)
    return result.statistic, result.pvalue, None


def _anderson(x: np.ndarray):
    result: Any = stats.anderson(x, "norm", method="interpolate")
    return result.statistic, result.pvalue, "p-value interpolated from tables (0.01 to 0.15)."


def _jarque_bera(x: np.ndarray):
    result: Any = stats.jarque_bera(x)
    return result.statistic, result.pvalue, None


def _lilliefors(x: np.ndarray):
    # Kolmogorov-Smirnov with mean and std estimated from the data: Monte Carlo p-value.
    result = stats.goodness_of_fit(
        stats.norm, x, statistic="ks", n_mc_samples=999, rng=np.random.default_rng(0)
    )
    return result.statistic, result.pvalue, "p-value by Monte Carlo (999 draws)."


TESTS = (
    ("shapiro", _shapiro),
    ("dagostino", _dagostino),
    ("anderson", _anderson),
    ("jarque_bera", _jarque_bera),
    ("lilliefors", _lilliefors),
)


def normality(source: VariableSource, summary: Describe) -> Normality:
    result = Normality(n=summary.count, sample=min(summary.count, TEST_SAMPLE))
    if summary.count < MIN_TEST_VALUES:
        result.reason = f"Needs at least {MIN_TEST_VALUES} values."
        return result
    if summary.min == summary.max:
        result.reason = "The variable is constant."
        return result
    sample = source.sample(TEST_SAMPLE)
    result.tests = [_run(key, sample, compute) for key, compute in TESTS]
    return result


def kde(source: VariableSource, summary: Describe) -> Curve | None:
    low, high = summary.min, summary.max
    if summary.count < 2 or low is None or high is None or low == high:
        return None
    sample = source.sample(CURVE_SAMPLE)
    try:
        density = stats.gaussian_kde(sample)  # Scott's rule
    except (np.linalg.LinAlgError, ValueError):
        return None
    grid = np.linspace(low, high, POINTS)
    return Curve([float(v) for v in grid], [float(v) for v in density(grid)])


def probabilities() -> list[float]:
    return [(i + 0.5) / POINTS for i in range(POINTS)]


def quantiles(source: VariableSource, levels: list[float], count: int) -> list[float]:
    """Quantiles over every row, for the ECDF and the QQ-plot: exact up to EXACT_QUANTILES."""
    listed = ", ".join(f"{p:.6f}" for p in levels)
    function = "quantile_cont" if count <= EXACT_QUANTILES else "approx_quantile"
    (values,) = source.one(f"SELECT {function}(x, [{listed}]) FROM {source.numbers}")
    return [float(v) for v in values or []]


def ecdf(source: VariableSource, summary: Describe) -> Curve | None:
    if summary.count == 0:
        return None
    levels = [i / (POINTS - 1) for i in range(POINTS)]
    return Curve(quantiles(source, levels, summary.count), levels)


def qq(source: VariableSource, summary: Describe) -> QQ | None:
    if summary.count < 3 or summary.q1 is None or summary.q3 is None:
        return None
    levels = probabilities()
    theoretical = [float(v) for v in stats.norm.ppf(levels)]
    z1, z3 = float(stats.norm.ppf(0.25)), float(stats.norm.ppf(0.75))
    slope = (summary.q3 - summary.q1) / (z3 - z1)
    sample = quantiles(source, levels, summary.count)
    return QQ(theoretical, sample, slope, summary.q1 - slope * z1)
