"""The right tail of a numeric variable (056): peaks over a threshold fitted by a generalised
Pareto law (GPD), return levels ("1 in 1,000 rows exceed X") with profile-likelihood intervals,
Hill and mean-excess plots, Value at Risk and Expected Shortfall.

Rules R-TAIL-1 to R-TAIL-7 of docs/features/056-queues-robustesse/spec.md. Everything runs in
numpy / scipy on at most TAIL_SAMPLE values (Pyodide-compatible); VaR and ES read every row."""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize, stats

from quantilo.variable.numeric import Describe
from quantilo.variable.source import VariableSource, clean

MIN_VALUES = 500
MIN_EXCEEDANCES = 50
NEGATIVE_SHARE = 0.05
KURTOSIS = 3.0  # excess kurtosis above which the tail is worth a look
HEAVY_XI = 0.1  # without a high kurtosis: GPD shape from which the tail counts as heavy
LEVEL = 0.95  # ★ threshold
LOWEST_LEVEL = 0.90
CANDIDATES = (0.95, 0.975, 0.90)  # tried in this order when the mean excess is not linear
LINEAR_R2 = 0.7
MEAN_EXCESS_POINTS = 60
MEAN_EXCESS_LAST = 20  # the mean-excess curve stops where 20 values remain above
HILL_POINTS = 100
HILL_MIN_K = 10
HILL_MAX_K = 20_000
HILL_MAX_SHARE = 0.10
TAIL_SAMPLE = 200_000
RULE_SAMPLE = 20_000  # the profile's sample, already read
PERIODS = (100, 1_000, 10_000)
RISK_LEVELS = (0.95, 0.99)
CHI2 = float(stats.chi2.ppf(0.95, 1))  # 3.84
XI_BOUNDS = (-0.95, 3.0)
UNBOUNDED = 4_096  # profile search stops at this many times the return level's excess
ZERO_XI = 1e-6
CURVE_POINTS = 60


@dataclass
class TailRule:
    """Whether the Tail tab is shown, and why (R-TAIL-1)."""

    fires: bool
    reason: str
    kurtosis: float | None = None
    xi: float | None = None  # GPD shape on the profile's sample, when the kurtosis is not enough


@dataclass
class Threshold:
    value: float
    level: float  # quantile level (0.95 = the 95th percentile)
    exceedances: int
    linear_r2: float | None  # weighted R² of the mean excess above the threshold
    why: str


@dataclass
class ShapeAt:
    """ξ̂ at another threshold, to see whether it is stable."""

    level: float
    threshold: float
    exceedances: int
    xi: float | None


@dataclass
class ReturnLevel:
    period: int  # exceeded once every `period` rows on average
    value: float
    low: float | None
    high: float | None  # None: the profile likelihood does not close (too few extreme values)
    beyond_data: bool  # period larger than the number of values: an extrapolation
    empirical: float | None  # quantile 1 - 1/period of the values, when within the data


@dataclass
class Risk:
    level: float  # 0.95, 0.99
    var: float  # the quantile at `level`
    es: float  # mean of the values at or above it ("average of the worst 5 %")
    rows: int


@dataclass
class Series:
    x: list[float]
    y: list[float]
    low: list[float] = field(default_factory=list)
    high: list[float] = field(default_factory=list)
    count: list[int] = field(default_factory=list)


@dataclass
class Tail:
    n: int  # finite values in the column
    sample: int  # values the fit read
    sampled: bool
    rule: TailRule
    reason: str | None = None  # set when no analysis is possible
    threshold: Threshold | None = None
    xi: float | None = None
    xi_low: float | None = None
    xi_high: float | None = None
    sigma: float | None = None
    rate: float | None = None  # share of values above the threshold (ζ)
    endpoint: float | None = None  # ξ < 0: the value the tail stops at
    stability: list[ShapeAt] = field(default_factory=list)
    return_levels: list[ReturnLevel] = field(default_factory=list)
    risk: list[Risk] = field(default_factory=list)
    hill: Series | None = None  # k → ξ̂ (Hill) with its band
    mean_excess: Series | None = None  # threshold → mean excess, with counts
    curve: Series | None = None  # return-level plot: GPD curve (x = period)
    observed: Series | None = None  # return-level plot: the largest values (x = period)
    conclusion: str = ""
    notes: list[str] = field(default_factory=list)


# --- GPD -----------------------------------------------------------------------------------


def nll(y: np.ndarray, xi: float, sigma: float) -> float:
    """Negative log-likelihood of a GPD(ξ, σ) for the excesses y ≥ 0."""
    if sigma <= 0:
        return math.inf
    if abs(xi) < ZERO_XI:
        return float(len(y) * math.log(sigma) + y.sum() / sigma)
    z = 1 + xi * y / sigma
    if np.any(z <= 0):
        return math.inf
    return float(len(y) * math.log(sigma) + (1 + 1 / xi) * np.log(z).sum())


def fit_gpd(y: np.ndarray) -> tuple[float, float]:
    """Maximum likelihood (ξ, σ) of a GPD with location 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xi, _loc, sigma = stats.genpareto.fit(y, floc=0)
    xi, sigma = float(xi), float(sigma)
    # scipy may stop on a boundary; a profile over ξ from its answer settles it.
    best = _profile_xi(y, xi)
    if best[0] < nll(y, xi, sigma) - 1e-9:
        return best[1], best[2]
    return xi, sigma


def _minimize(
    function: Callable[[float], float], bounds: tuple[float, float]
) -> tuple[float, float]:
    """Bounded scalar minimisation: (argument, value)."""
    found: Any = optimize.minimize_scalar(function, bounds=bounds, method="bounded")
    return float(found.x), float(found.fun)


def _best_sigma(y: np.ndarray, xi: float) -> tuple[float, float]:
    """min over σ of the NLL at fixed ξ: (NLL, σ)."""
    top = float(y.max()) if len(y) else 1.0
    mean = float(y.mean()) if len(y) else 1.0
    low = math.log(max(mean, 1e-300)) - 12
    if xi < 0:
        low = max(low, math.log(-xi * top) + 1e-9)  # support: y < -σ/ξ
    high = math.log(max(mean, 1e-300)) + 12
    at, value = _minimize(lambda s: nll(y, xi, math.exp(s)), (low, high))
    return value, math.exp(at)


def _profile_xi(y: np.ndarray, start: float) -> tuple[float, float, float]:
    shape, _value = _minimize(lambda xi: _best_sigma(y, xi)[0], XI_BOUNDS)
    value, sigma = _best_sigma(y, shape)
    return value, shape, sigma


def xi_interval(y: np.ndarray, xi: float, best: float) -> tuple[float | None, float | None]:
    """95 % profile-likelihood interval of ξ."""

    def gap(value: float) -> float:
        return 2 * (_best_sigma(y, value)[0] - best) - CHI2

    return _root(gap, xi, XI_BOUNDS[0]), _root(gap, xi, XI_BOUNDS[1])


def _root(gap: Callable[[float], float], centre: float, edge: float) -> float | None:
    try:
        if not gap(edge) > 0:
            return None
        found: Any = optimize.brentq(gap, *sorted((centre, edge)), xtol=1e-6)
        return float(found)
    except (ValueError, RuntimeError):
        return None


def level_at(u: float, xi: float, sigma: float, rate: float, period: float) -> float:
    """x_m = u + σ/ξ ((mζ)^ξ − 1): exceeded once every m rows on average."""
    grow = period * rate
    if abs(xi) < ZERO_XI:
        return u + sigma * math.log(grow)
    return u + sigma / xi * (grow**xi - 1)


def _sigma_for(excess: float, xi: float, grow: float) -> float:
    """σ such that the return level of the period is u + excess."""
    if abs(xi) < ZERO_XI:
        return excess / math.log(grow)
    return excess * xi / (grow**xi - 1)


def level_interval(
    y: np.ndarray, u: float, xi: float, sigma: float, rate: float, period: int, best: float
) -> tuple[float | None, float | None]:
    """95 % profile-likelihood interval of a return level: σ rewritten through x_m."""
    grow = period * rate
    excess = level_at(u, xi, sigma, rate, period) - u
    if excess <= 0:
        return None, None

    def profile(level_excess: float) -> float:
        return _minimize(
            lambda shape: nll(y, shape, _sigma_for(level_excess, shape, grow)), XI_BOUNDS
        )[1]

    def gap(level_excess: float) -> float:
        return 2 * (profile(level_excess) - best) - CHI2

    low = _root(gap, excess, excess * 1e-6)
    high = None
    edge = excess
    while edge < excess * UNBOUNDED:
        edge *= 4
        if gap(edge) > 0:
            high = _root(gap, excess, edge)
            break
    return (u + low if low is not None else None), (u + high if high is not None else None)


# --- Plots ---------------------------------------------------------------------------------


def mean_excess(ordered: np.ndarray, start: float) -> Series:
    """e(u) = mean(X − u | X > u) for thresholds from `start` to where 20 values remain above.
    `ordered`: the values, sorted ascending."""
    n = len(ordered)
    if n <= MEAN_EXCESS_LAST:
        return Series([], [])
    last = float(ordered[n - MEAN_EXCESS_LAST - 1])
    if last <= start:
        return Series([], [])
    grid = np.unique(np.quantile(ordered[ordered >= start], np.linspace(0, 1, MEAN_EXCESS_POINTS)))
    grid = grid[grid <= last]
    tail_sums = np.concatenate([np.cumsum(ordered[::-1])[::-1], [0.0]])
    xs, ys, counts = [], [], []
    for u in grid:
        first = int(np.searchsorted(ordered, u, side="right"))
        above = n - first
        if above < MEAN_EXCESS_LAST:
            continue
        xs.append(float(u))
        ys.append(float(tail_sums[first] / above - u))
        counts.append(above)
    return Series(xs, ys, count=counts)


def linearity(curve: Series, start: float) -> float | None:
    """Weighted R² of a straight line through the mean excess at and above `start`."""
    x = np.asarray(curve.x)
    keep = x >= start
    if keep.sum() < 5:
        return None
    x, y = x[keep], np.asarray(curve.y)[keep]
    w = np.asarray(curve.count, dtype=float)[keep]
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    sxx = np.average((x - mx) ** 2, weights=w)
    syy = np.average((y - my) ** 2, weights=w)
    if sxx == 0 or syy == 0:
        return 1.0
    sxy = np.average((x - mx) * (y - my), weights=w)
    return float(sxy**2 / (sxx * syy))


def hill(ordered: np.ndarray) -> Series | None:
    """Hill estimates ξ̂_k = mean(log X_(i) / X_(k)) over the k largest positive values."""
    positive = ordered[ordered > 0]
    top = min(int(len(ordered) * HILL_MAX_SHARE), HILL_MAX_K, len(positive) - 1)
    if top < HILL_MIN_K:
        return None
    logs = np.log(positive[::-1][: top + 1])  # descending
    ks = np.unique(np.geomspace(HILL_MIN_K, top, HILL_POINTS).astype(int))
    cumulative = np.cumsum(logs)
    xi = cumulative[ks - 1] / ks - logs[ks]
    band = 1.96 * xi / np.sqrt(ks)
    return Series(
        [float(k) for k in ks],
        [float(v) for v in xi],
        [float(v) for v in xi - band],
        [float(v) for v in xi + band],
    )


# --- Rule and analysis ---------------------------------------------------------------------


def _threshold_at(ordered: np.ndarray, level: float) -> tuple[float, int]:
    """The `level` quantile, lowered (never under LOWEST_LEVEL) until MIN_EXCEEDANCES remain."""
    n = len(ordered)
    u = float(np.quantile(ordered, level))
    above = n - int(np.searchsorted(ordered, u, side="right"))
    if above < MIN_EXCEEDANCES and level == LEVEL:
        lowest = float(np.quantile(ordered, LOWEST_LEVEL))
        wanted = float(ordered[max(n - MIN_EXCEEDANCES - 1, 0)])
        u = max(wanted, lowest)
        above = n - int(np.searchsorted(ordered, u, side="right"))
    return u, above


def _blocked(n: int, negatives: int, u: float, above: int) -> str | None:
    if n < MIN_VALUES:
        return f"Needs at least {MIN_VALUES} values ({n:,} here)."
    if negatives > NEGATIVE_SHARE * n:
        return (
            f"The tail analysis reads the right tail of positive values: "
            f"{100 * negatives / n:.1f} % of values are negative (more than 5 %)."
        )
    if u <= 0:
        return "The 95th percentile is not above 0: no right tail of positive values."
    if above < MIN_EXCEEDANCES:
        return (
            f"Only {above} values above the threshold (at least {MIN_EXCEEDANCES} needed): "
            "too many equal values in the tail."
        )
    return None


def tail_rule(source: VariableSource, summary: Describe) -> TailRule:
    """R-TAIL-1, on the profile's sample: cheap enough to decide whether the tab is shown."""
    kurtosis = summary.kurtosis
    ordered = np.sort(source.sample(RULE_SAMPLE))
    u, above = _threshold_at(ordered, LEVEL) if len(ordered) else (0.0, 0)
    blocked = _blocked(summary.count, summary.negatives, u, above)
    if blocked:
        return TailRule(False, blocked, kurtosis)
    if summary.count > len(ordered):
        excess = f"{above:,} of {len(ordered):,} sampled values above the 95th percentile"
    else:
        excess = f"{above:,} values above the threshold"
    if kurtosis is not None and kurtosis > KURTOSIS:
        return TailRule(
            True, f"Heavy tail: excess kurtosis {kurtosis:,.1f} (above 3), {excess}.", kurtosis
        )
    xi, _sigma = fit_gpd(ordered[ordered > u] - u)
    if xi >= HEAVY_XI:
        return TailRule(
            True,
            f"Heavy tail: the shape above the threshold is ξ = {xi:.2f} (0.1 or more), {excess}.",
            kurtosis,
            xi,
        )
    shown = "—" if kurtosis is None else f"{kurtosis:.1f}"
    return TailRule(
        False,
        f"Light tail: excess kurtosis {shown} (3 or less) and shape ξ = {xi:.2f} (under 0.1).",
        kurtosis,
        xi,
    )


def _choose_threshold(ordered: np.ndarray) -> tuple[Threshold, Series]:
    """R-TAIL-2: the 95th percentile, unless its mean excess is not linear."""
    start = float(np.quantile(ordered, 0.5))
    curve = mean_excess(ordered, start)
    tried = []
    for level in CANDIDATES:
        u, above = _threshold_at(ordered, level)
        if above < MIN_EXCEEDANCES:
            continue
        r2 = linearity(curve, u)
        tried.append((level, u, above, r2))
        if r2 is None or r2 >= LINEAR_R2:
            break
    passing = [t for t in tried if t[3] is None or t[3] >= LINEAR_R2]
    level, u, above, r2 = passing[0] if passing else tried[0]
    name = f"{_ordinal(level)} percentile"
    shown = "" if r2 is None else f" (R² {r2:.2f})"
    if not passing:
        why = (
            f"{name}: the mean excess is not linear above any candidate threshold{shown}, "
            "so the GPD is an approximation: read the return levels with caution."
        )
    elif level == LEVEL:
        why = f"{name}: the mean excess is roughly linear above it{shown}."
    else:
        why = (
            f"{name}: the mean excess is not linear above the 95th percentile, "
            f"it is above this one{shown}."
        )
    return Threshold(u, level, above, r2, why), curve


def _ordinal(level: float) -> str:
    number = f"{100 * level:g}"
    return f"{number}th"


def _risk(source: VariableSource, summary: Describe) -> list[Risk]:
    """R-TAIL-5 on every row: VaR = the summary's exact quantile, ES = mean at or above it."""
    found = []
    for level, var in zip(RISK_LEVELS, (summary.p95, summary.p99), strict=True):
        if var is None:
            continue
        es, rows = source.one(
            f"SELECT avg(x), count(*) FROM {source.numbers} WHERE x >= {float(var)!r}"
        )
        if es is not None:
            found.append(Risk(level, float(var), float(es), int(rows)))
    return found


def _conclusion(tail: Tail, p99: float | None) -> str:
    xi, low, high = tail.xi, tail.xi_low, tail.xi_high
    assert xi is not None
    thousand = next((r for r in tail.return_levels if r.period == 1_000), None)
    ratio = ""
    if thousand is not None and p99:
        ratio = f"the 1-in-1,000 value is {thousand.value / p99:.1f}× the 99th percentile"
    if low is not None and low > 0:
        text = f"Heavy right tail (ξ = {xi:.2f})"
        if ratio:
            text += f": {ratio}"
        if xi >= 1:
            text += "; so heavy that even the mean is unstable: prefer medians and quantiles"
        elif xi >= 0.5:
            text += (
                "; so heavy that the variance may not exist: prefer medians and quantiles to "
                "the mean"
            )
        return text + "."
    if high is not None and high < 0 and tail.endpoint is not None:
        return f"Bounded tail (ξ = {xi:.2f}): values seem to stop near {tail.endpoint:,.4g}."
    text = f"Exponential-like tail (ξ = {xi:.2f}, not different from 0)"
    return text + (f": {ratio}." if ratio else ".")


def analyze_tail(
    source: VariableSource, summary: Describe, rule: TailRule, discrete: bool = False
) -> Tail:
    n = summary.count
    values = source.sample(TAIL_SAMPLE)
    ordered = np.sort(values)
    tail = Tail(n, len(ordered), n > len(ordered), rule)
    u, above = _threshold_at(ordered, LEVEL) if len(ordered) else (0.0, 0)
    blocked = _blocked(n, summary.negatives, u, above)
    if blocked:
        tail.reason = blocked
        return tail
    threshold, curve = _choose_threshold(ordered)
    tail.threshold, tail.mean_excess = threshold, curve
    u = threshold.value
    y = ordered[ordered > u] - u
    xi, sigma = fit_gpd(y)
    best = nll(y, xi, sigma)
    tail.xi, tail.sigma = xi, sigma
    tail.xi_low, tail.xi_high = xi_interval(y, xi, best)
    rate = len(y) / len(ordered)
    tail.rate = rate
    if xi < 0:
        tail.endpoint = u - sigma / xi
    for level in CANDIDATES:
        other, count = _threshold_at(ordered, level)
        shape = fit_gpd(ordered[ordered > other] - other)[0] if count >= MIN_EXCEEDANCES else None
        tail.stability.append(ShapeAt(level, other, count, clean(shape)))
    tail.stability.sort(key=lambda s: s.level)
    tail.notes.extend(_stability_notes(tail))
    for period in PERIODS:
        value = level_at(u, xi, sigma, rate, period)
        low, high = level_interval(y, u, xi, sigma, rate, period, best)
        within = period <= n
        empirical = float(np.quantile(ordered, 1 - 1 / period)) if period <= len(ordered) else None
        tail.return_levels.append(ReturnLevel(period, value, low, high, not within, empirical))
    tail.risk = _risk(source, summary)
    tail.hill = hill(ordered)
    tail.curve, tail.observed = _return_plot(ordered, u, xi, sigma, rate)
    tail.conclusion = _conclusion(tail, summary.p99)
    if tail.sampled:
        tail.notes.append(
            f"Fitted on a reproducible sample of {len(ordered):,} of {n:,} values; VaR and "
            "Expected Shortfall use every row."
        )
    if discrete:
        tail.notes.append(
            "Whole numbers: the GPD treats them as continuous; values equal to the threshold "
            "count below it."
        )
    if summary.negatives:
        tail.notes.append(
            f"{summary.negatives:,} negative values sit in the left part and do not change "
            "the right tail."
        )
    return tail


def _stability_notes(tail: Tail) -> list[str]:
    """R-TAIL-3: ξ̂ at the 97.5th percentile outside the interval at the threshold means the
    furthest return levels lean one way."""
    higher = next((s for s in tail.stability if s.level == 0.975), None)
    if higher is None or higher.xi is None or tail.threshold is None:
        return []
    if tail.threshold.level >= higher.level:
        return []
    if tail.xi_high is not None and higher.xi > tail.xi_high:
        return [
            f"The shape grows further out (ξ = {higher.xi:.2f} above the 97.5th percentile): "
            "the 1-in-10,000 level may be underestimated."
        ]
    if tail.xi_low is not None and higher.xi < tail.xi_low:
        return [
            f"The shape shrinks further out (ξ = {higher.xi:.2f} above the 97.5th percentile): "
            "the 1-in-10,000 level may be overestimated."
        ]
    return []


def _return_plot(
    ordered: np.ndarray, u: float, xi: float, sigma: float, rate: float
) -> tuple[Series, Series]:
    """The GPD's return levels from 1 / ζ rows to 10 × the largest period, and the largest
    values at their empirical period (n / rank)."""
    n = len(ordered)
    periods = np.geomspace(1 / rate, PERIODS[-1] * 10, CURVE_POINTS)
    curve = Series(
        [float(p) for p in periods], [level_at(u, xi, sigma, rate, float(p)) for p in periods]
    )
    above = ordered[ordered > u][::-1]
    ranks = np.unique(np.geomspace(1, len(above), min(len(above), 200)).astype(int))
    observed = Series([float(n / r) for r in ranks], [float(above[r - 1]) for r in ranks])
    return curve, observed
