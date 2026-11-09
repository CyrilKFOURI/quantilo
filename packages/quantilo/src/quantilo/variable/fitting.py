"""Which law describes the variable best, and which transformation makes it closer to normal.

Fitted by maximum likelihood on a seeded sample; ranked by AIC. Transformations are previews:
the data itself is never modified.
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from quantilo.charts.prepare import Bin
from quantilo.variable.numeric import TEST_SAMPLE, probabilities
from quantilo.variable.source import VariableSource, clean

FIT_SAMPLE = 10_000
POINTS = 200
TRANSFORM_BINS = 30


@dataclass
class Law:
    key: str
    label: str
    discrete: bool
    params: dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    ks: float  # largest gap between the empirical and fitted CDFs
    density: list[float]  # pdf on Fit.grid (pmf on integers for a discrete law)
    quantiles: list[float]  # at numeric.probabilities(), for a QQ-plot against this law


@dataclass
class Transform:
    key: str  # log, sqrt, boxcox, yeojohnson
    lmbda: float | None
    skewness_before: float | None
    skewness_after: float | None
    shapiro_before: float | None
    shapiro_after: float | None
    bins: list[Bin]  # histogram of the transformed sample


@dataclass
class Fit:
    n: int  # sample size the laws were fitted on
    grid: list[float]
    sample_quantiles: list[float]
    laws: list[Law] = field(default_factory=list)
    transforms: list[Transform] = field(default_factory=list)
    reason: str | None = None


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    dist: stats.rv_continuous
    params: tuple[str, ...]
    fixed: dict[str, float]
    applies: Callable[[np.ndarray], bool]


def positive(x: np.ndarray) -> bool:
    return bool(x.min() > 0)


def non_negative(x: np.ndarray) -> bool:
    return bool(x.min() >= 0)


def unit_interval(x: np.ndarray) -> bool:
    return bool(x.min() > 0 and x.max() < 1)


def always(_: np.ndarray) -> bool:
    return True


CONTINUOUS = (
    Candidate("normal", "Normal", stats.norm, ("loc", "scale"), {}, always),
    Candidate("lognormal", "Log-normal", stats.lognorm, ("s", "scale"), {"floc": 0}, positive),
    Candidate("exponential", "Exponential", stats.expon, ("scale",), {"floc": 0}, non_negative),
    Candidate("gamma", "Gamma", stats.gamma, ("a", "scale"), {"floc": 0}, positive),
    Candidate("weibull", "Weibull", stats.weibull_min, ("c", "scale"), {"floc": 0}, positive),
    Candidate("beta", "Beta", stats.beta, ("a", "b"), {"floc": 0, "fscale": 1}, unit_interval),
    Candidate("uniform", "Uniform", stats.uniform, ("loc", "scale"), {}, always),
    Candidate("student_t", "Student's t", stats.t, ("df", "loc", "scale"), {}, always),
    Candidate("pareto", "Pareto", stats.pareto, ("b", "scale"), {"floc": 0}, positive),
    Candidate("logistic", "Logistic", stats.logistic, ("loc", "scale"), {}, always),
)


def _all_params(dist: stats.rv_continuous) -> list[str]:
    shapes = [s.strip() for s in (dist.shapes or "").split(",") if s.strip()]
    return [*shapes, "loc", "scale"]


def _criteria(log_likelihood: float, k: int, n: int) -> tuple[float, float]:
    return 2 * k - 2 * log_likelihood, k * math.log(n) - 2 * log_likelihood


def _fit_continuous(candidate: Candidate, x: np.ndarray, grid: np.ndarray) -> Law | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = candidate.dist.fit(x, **candidate.fixed)
        frozen: Any = candidate.dist(*fitted)
        log_likelihood = float(np.sum(frozen.logpdf(x)))
        if not math.isfinite(log_likelihood):
            return None
        ks = float(stats.kstest(x, frozen.cdf).statistic)
        density = frozen.pdf(grid)
        quantiles = frozen.ppf(probabilities())
    names = dict(zip(_all_params(candidate.dist), fitted, strict=True))
    aic, bic = _criteria(log_likelihood, len(candidate.params), len(x))
    return Law(
        candidate.key,
        candidate.label,
        False,
        {name: float(names[name]) for name in candidate.params},
        log_likelihood,
        aic,
        bic,
        ks,
        [clean(v) or 0.0 for v in density],
        [clean(v) or 0.0 for v in quantiles],
    )


def _discrete_law(
    key: str, label: str, frozen, params: dict[str, float], x: np.ndarray, grid: np.ndarray
) -> Law | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_likelihood = float(np.sum(frozen.logpmf(x)))
        if not math.isfinite(log_likelihood):
            return None
        support = np.arange(int(x.min()), int(x.max()) + 1)
        empirical = np.searchsorted(np.sort(x), support, side="right") / len(x)
        ks = float(np.max(np.abs(empirical - frozen.cdf(support))))
        density = frozen.pmf(np.round(grid))
        quantiles = frozen.ppf(probabilities())
    aic, bic = _criteria(log_likelihood, len(params), len(x))
    return Law(
        key,
        label,
        True,
        params,
        log_likelihood,
        aic,
        bic,
        ks,
        [clean(v) or 0.0 for v in density],
        [clean(v) or 0.0 for v in quantiles],
    )


def _fit_discrete(x: np.ndarray, grid: np.ndarray) -> list[Law]:
    mean, var = float(x.mean()), float(x.var(ddof=1)) if len(x) > 1 else 0.0
    laws: list[Law | None] = []
    if mean > 0:
        laws.append(_discrete_law("poisson", "Poisson", stats.poisson(mean), {"mu": mean}, x, grid))
        p = 1 / (mean + 1)  # geometric counting failures before the first success
        laws.append(
            _discrete_law("geometric", "Geometric", stats.geom(p, loc=-1), {"p": p}, x, grid)
        )
    if var > mean > 0:  # overdispersion: method of moments
        p = mean / var
        n = mean * mean / (var - mean)
        laws.append(
            _discrete_law(
                "negative_binomial",
                "Negative binomial",
                stats.nbinom(n, p),
                {"n": n, "p": p},
                x,
                grid,
            )
        )
    return [law for law in laws if law is not None]


def _skew(x: np.ndarray) -> float | None:
    return clean(stats.skew(x, bias=False)) if len(x) > 2 and np.ptp(x) > 0 else None


def _shapiro(x: np.ndarray) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return clean(stats.shapiro(x[:TEST_SAMPLE]).pvalue)


def _bins(x: np.ndarray) -> list[Bin]:
    counts, edges = np.histogram(x, bins=TRANSFORM_BINS)
    return [
        Bin(start=float(edges[i]), end=float(edges[i + 1]), count=int(counts[i]))
        for i in range(len(counts))
    ]


def transforms(x: np.ndarray) -> list[Transform]:
    before_skew, before_shapiro = _skew(x), _shapiro(x)
    options: list[tuple[str, np.ndarray, float | None]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if x.min() > 0:
            options.append(("log", np.log(x), None))
        if x.min() >= 0:
            options.append(("sqrt", np.sqrt(x), None))
        if x.min() > 0 and np.ptp(x) > 0:
            boxcox: Any = stats.boxcox(x)
            options.append(("boxcox", boxcox[0], float(boxcox[1])))
        if np.ptp(x) > 0:
            yeojohnson: Any = stats.yeojohnson(x)
            options.append(("yeojohnson", yeojohnson[0], float(yeojohnson[1])))
    return [
        Transform(
            key,
            lmbda,
            before_skew,
            _skew(after),
            before_shapiro,
            _shapiro(after),
            _bins(after),
        )
        for key, after, lmbda in options
        if np.all(np.isfinite(after))
    ]


def fit(source: VariableSource, discrete: bool) -> Fit:
    x = source.sample(FIT_SAMPLE)
    if len(x) < 8 or np.ptp(x) == 0:
        reason = "The variable is constant." if len(x) >= 8 else "Needs at least 8 values."
        return Fit(len(x), [], [], reason=reason)
    grid = np.linspace(x.min(), x.max(), POINTS)
    laws: list[Law] = []
    integers = bool(np.all(x == np.round(x)) and x.min() >= 0)
    if discrete and integers:
        laws.extend(_fit_discrete(x, grid))
    for candidate in CONTINUOUS:
        if not candidate.applies(x):
            continue
        try:
            law = _fit_continuous(candidate, x, grid)
        except (ValueError, RuntimeError, FloatingPointError):
            continue
        if law is not None:
            laws.append(law)
    laws.sort(key=lambda law: law.aic)
    sorted_x = np.sort(x)
    sample_quantiles = np.quantile(sorted_x, probabilities())
    return Fit(
        len(x),
        [float(v) for v in grid],
        [float(v) for v in sample_quantiles],
        laws,
        transforms(x),
    )
