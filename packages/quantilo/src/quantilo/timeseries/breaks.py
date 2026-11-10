"""Where does the series change regime, and which periods are atypical?

Change-point search (PELT, binary segmentation) is written on numpy; the homogeneity tests
(Pettitt, Buishand, SNHT) and the CUSUM / Chow tests give a p-value for a single break.
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import breaks_cusumolsresid

from quantilo.timeseries.methods import ALPHA, MethodResult
from quantilo.variable.source import clean

Search = Literal["pelt", "binseg"]
SEARCHES: tuple[Search, ...] = ("pelt", "binseg")
Cost = Literal["mean", "mean_var", "trend"]  # trend: a straight line per segment
Sensitivity = Literal["low", "medium", "high"]
PENALTY_FACTOR: dict[Sensitivity, float] = {"high": 0.5, "medium": 1.0, "low": 2.0}
Anomaly = Literal[
    "robust_z",
    "iqr",
    "rolling_z",
    "hampel",
    "gesd",
    "isolation_forest",
    "lof",
    "autoencoder",
    "kalman",
]
ANOMALIES: tuple[Anomaly, ...] = (
    "robust_z",
    "iqr",
    "rolling_z",
    "hampel",
    "gesd",
    "isolation_forest",
    "lof",
    "autoencoder",
    "kalman",  # one-step-ahead errors of a structural model (064)
)
GESD_ALPHA = 0.05
GESD_MAX_SHARE = 0.1  # at most 10 % of the periods tested as outliers
MAX_ANOMALIES = 50
SIMULATIONS = 2_000  # Buishand and SNHT p-values by simulation (seeded)


@dataclass
class Segment:
    start: int  # index of the first period
    end: int  # index after the last period
    mean: float
    slope: float = 0.0  # change per period inside the segment (trend cost)
    first: float = 0.0  # fitted value at the first and last period of the segment
    last: float = 0.0


@dataclass
class Breaks:
    search: Search
    cost: Cost
    points: list[int]  # index of the first period of each new segment
    segments: list[Segment] = field(default_factory=list)
    cost_auto: bool = False  # the cost was chosen from the data (trend when the series trends)


@dataclass
class AnomalyPoint:
    index: int
    value: float
    expected: float
    score: float  # how unusual, in the method's own unit (robust z, IQR multiples…)


# --- Change-point search -------------------------------------------------------------------------


def _noise_variance(series: np.ndarray) -> float:
    """Robust variance of the noise from first differences (insensitive to level shifts)."""
    mad = float(np.median(np.abs(np.diff(series) - np.median(np.diff(series)))))
    sigma = mad / (0.6745 * math.sqrt(2))
    return sigma**2 if sigma > 0 else float(np.var(series)) or 1.0


class _Sums:
    """Prefix sums giving any segment's mean or least-squares line in constant time."""

    def __init__(self, series: np.ndarray) -> None:
        t = np.arange(len(series), dtype=float)
        self.y = np.concatenate([[0.0], np.cumsum(series)])
        self.yy = np.concatenate([[0.0], np.cumsum(series**2)])
        self.t = np.concatenate([[0.0], np.cumsum(t)])
        self.tt = np.concatenate([[0.0], np.cumsum(t**2)])
        self.ty = np.concatenate([[0.0], np.cumsum(t * series)])

    def line(self, start: int, end: int) -> tuple[float, float, float]:
        """Slope, intercept (in absolute period index) and residual sum of squares."""
        n = end - start
        sy, syy = self.y[end] - self.y[start], self.yy[end] - self.yy[start]
        st, stt = self.t[end] - self.t[start], self.tt[end] - self.tt[start]
        sty = self.ty[end] - self.ty[start]
        spread = n * stt - st * st
        slope = (n * sty - st * sy) / spread if spread > 0 else 0.0
        intercept = (sy - slope * st) / n
        sse = syy - sy * sy / n - slope * (sty - st * sy / n)
        return slope, intercept, max(sse, 0.0)


def _segment_cost(sums: _Sums, start: int, end: int, cost: Cost) -> float:
    n = end - start
    if cost == "trend":
        return sums.line(start, end)[2]
    total = sums.y[end] - sums.y[start]
    sse = sums.yy[end] - sums.yy[start] - total * total / n
    if cost == "mean":
        return sse
    return n * math.log(max(sse / n, 1e-12))


def pelt(series: np.ndarray, penalty: float, min_size: int, cost: Cost = "mean") -> list[int]:
    """Pruned Exact Linear Time (Killick et al. 2012): the segmentation minimising the total cost
    plus a penalty per change, with segments of at least `min_size` periods."""
    n = len(series)
    sums = _Sums(series)
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    candidates = [0]
    for end in range(min_size, n + 1):
        usable = [s for s in candidates if end - s >= min_size]
        totals = [best[s] + _segment_cost(sums, s, end, cost) for s in usable]
        index = int(np.argmin(totals))
        best[end], last[end] = totals[index] + penalty, usable[index]
        kept = {s for s, total in zip(usable, totals, strict=True) if total <= best[end]}
        candidates = [s for s in candidates if s in kept or end - s < min_size] + [end]
    points, end = [], n
    while end > 0:
        start = int(last[end])
        if start > 0:
            points.append(start)
        end = start
    return sorted(points)


def binary_segmentation(
    series: np.ndarray, penalty: float, min_size: int, cost: Cost = "mean"
) -> list[int]:
    """Greedy: split where it lowers the cost most, as long as the gain beats the penalty."""
    sums = _Sums(series)
    points: list[int] = []
    stack = [(0, len(series))]
    while stack:
        start, end = stack.pop()
        if end - start < 2 * min_size:
            continue
        whole = _segment_cost(sums, start, end, cost)
        gains = [
            (
                whole - _segment_cost(sums, start, k, cost) - _segment_cost(sums, k, end, cost),
                k,
            )
            for k in range(start + min_size, end - min_size + 1)
        ]
        gain, split = max(gains)
        if gain > penalty:
            points.append(split)
            stack += [(start, split), (split, end)]
    return sorted(points)


def find_breaks(
    series: np.ndarray,
    search: Search = "pelt",
    cost: Cost = "mean",
    sensitivity: Sensitivity = "medium",
    season: int | None = None,
) -> Breaks:
    n = len(series)
    min_size = max(5, min(season or 5, n // 10))
    variance = _noise_variance(series)
    factor = PENALTY_FACTOR[sensitivity]
    if cost == "mean":
        scaled, penalty = series / math.sqrt(variance), factor * 2 * math.log(n)
    elif cost == "trend":  # two parameters per segment (level and slope)
        scaled, penalty = series / math.sqrt(variance), factor * 3 * math.log(n)
    else:
        scaled, penalty = series, factor * 3 * math.log(n)
    finder = pelt if search == "pelt" else binary_segmentation
    points = finder(scaled, penalty, min_size, cost) if n >= 2 * min_size else []
    edges = [0, *points, n]
    sums = _Sums(series)
    segments = []
    for a, b in pairwise(edges):
        slope, intercept, _ = sums.line(a, b) if cost == "trend" else (0.0, 0.0, 0.0)
        mean = float(series[a:b].mean())
        first, last = (
            (intercept + slope * a, intercept + slope * (b - 1))
            if cost == "trend"
            else (mean, mean)
        )
        segments.append(Segment(a, b, mean, float(slope), float(first), float(last)))
    return Breaks(search, cost, points, segments)


# --- Single-break tests --------------------------------------------------------------------------


def _not_run(key: str, n: int, reason: str) -> MethodResult:
    return MethodResult(key, None, None, n, None, note=reason)


def pettitt(series: np.ndarray) -> MethodResult:
    """Rank-based test for one change of level at an unknown date."""
    n = len(series)
    if n < 10:
        return _not_run("pettitt", n, "Needs at least 10 periods.")
    ranks = stats.rankdata(series)
    u = np.array([2 * ranks[: k + 1].sum() - (k + 1) * (n + 1) for k in range(n - 1)])
    k = int(np.argmax(np.abs(u)))
    statistic = float(abs(u[k]))
    p = min(1.0, 2 * math.exp(-6 * statistic**2 / (n**3 + n**2)))
    return MethodResult(
        "pettitt",
        statistic,
        p,
        n,
        bool(p < ALPHA),
        value=k + 2,
        note=f"Most likely change after period {k + 1}.",
    )


def _simulated_p(statistic: float, n: int, compute: Callable[[np.ndarray], float]) -> float:
    rng = np.random.default_rng(0)
    simulated = np.array([compute(rng.normal(size=n)) for _ in range(SIMULATIONS)])
    return float((np.sum(simulated >= statistic) + 1) / (SIMULATIONS + 1))


def _buishand_statistic(x: np.ndarray) -> float:
    deviations = np.cumsum(x - x.mean())
    sd = x.std()
    return (
        float(np.ptp(np.concatenate([[0.0], deviations])) / (sd * math.sqrt(len(x)))) if sd else 0.0
    )


def buishand(series: np.ndarray) -> MethodResult:
    n = len(series)
    if n < 10 or np.ptp(series) == 0:
        return _not_run("buishand", n, "Needs at least 10 periods that vary.")
    statistic = _buishand_statistic(series)
    deviations = np.cumsum(series - series.mean())
    k = int(np.argmax(np.abs(deviations)))
    p = _simulated_p(statistic, n, _buishand_statistic)
    return MethodResult(
        "buishand",
        statistic,
        p,
        n,
        bool(p < ALPHA),
        value=k + 2,
        note=f"Most likely change after period {k + 1}.",
    )


def _snht_statistic(x: np.ndarray) -> float:
    sd = x.std()
    if not sd:
        return 0.0
    z = (x - x.mean()) / sd
    n = len(z)
    k = np.arange(1, n)
    left = np.cumsum(z)[:-1] / k
    right = (z.sum() - np.cumsum(z)[:-1]) / (n - k)
    return float(np.max(k * left**2 + (n - k) * right**2))


def snht(series: np.ndarray) -> MethodResult:
    """Standard normal homogeneity test (Alexandersson)."""
    n = len(series)
    if n < 10 or np.ptp(series) == 0:
        return _not_run("snht", n, "Needs at least 10 periods that vary.")
    statistic = _snht_statistic(series)
    p = _simulated_p(statistic, n, _snht_statistic)
    return MethodResult("snht", statistic, p, n, bool(p < ALPHA))


def cusum(series: np.ndarray) -> MethodResult:
    """CUSUM of the residuals of a constant: are the parameters stable over time?"""
    n = len(series)
    if n < 10 or np.ptp(series) == 0:
        return _not_run("cusum", n, "Needs at least 10 periods that vary.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = OLS(series, np.ones(n)).fit()
        statistic, p, _ = breaks_cusumolsresid(fit.resid, ddof=1)
    return MethodResult("cusum", clean(statistic), clean(p), n, bool(p < ALPHA))


def chow(series: np.ndarray, at: int | None = None) -> MethodResult:
    """F test of a different mean before and after one date (the strongest break by default)."""
    n = len(series)
    if n < 10:
        return _not_run("chow", n, "Needs at least 10 periods.")
    if at is None:
        found = find_breaks(series, "binseg", "mean", "high")
        at = found.points[0] if found.points else n // 2
    if at < 3 or n - at < 3:
        return _not_run("chow", n, "The break is too close to an end.")
    whole = float(((series - series.mean()) ** 2).sum())
    parts = sum(float(((part - part.mean()) ** 2).sum()) for part in (series[:at], series[at:]))
    k = 1
    f = ((whole - parts) / k) / (parts / (n - 2 * k)) if parts > 0 else 0.0
    p = float(stats.f.sf(f, k, n - 2 * k))
    return MethodResult(
        "chow", f, p, n, bool(p < ALPHA), value=at + 1, note=f"Tested at period {at + 1}."
    )


BREAK_TESTS: dict[str, Callable[[np.ndarray], MethodResult]] = {
    "pettitt": pettitt,
    "cusum": cusum,
    "chow": chow,
    "buishand": buishand,
    "snht": snht,
}
DEFAULT_BREAK_TESTS = ("pettitt",)


# --- Anomalies -----------------------------------------------------------------------------------


def _robust_z(scores: np.ndarray) -> np.ndarray:
    """How far each score sits above the typical one, in robust standard deviations."""
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    return (scores - median) / (1.4826 * mad) if mad else np.zeros_like(scores)


def _windows(values: np.ndarray, width: int) -> np.ndarray:
    """Sliding windows of `width` values, one per period (edges padded with the nearest value)."""
    half = width // 2
    padded = np.concatenate(
        [np.full(half, values[0]), values, np.full(width - half - 1, values[-1])]
    )
    return np.lib.stride_tricks.sliding_window_view(padded, width)


def gesd_scores(resid: np.ndarray, alpha: float = GESD_ALPHA) -> np.ndarray:
    """Generalized ESD test (Rosner 1983). Returns R_i / λ_i for the points it removes: the
    outliers are the first k removed, k being the largest i with R_i > λ_i (score > 1)."""
    from scipy import stats as sstats

    n = len(resid)
    values = resid.astype(float).copy()
    index = np.arange(n)
    removed: list[tuple[int, float]] = []
    for i in range(1, max(1, int(GESD_MAX_SHARE * n)) + 1):
        sd = values.std(ddof=1)
        if len(values) < 3 or sd == 0:
            break
        deviation = np.abs(values - values.mean())
        j = int(np.argmax(deviation))
        r = float(deviation[j] / sd)
        m = n - i + 1
        p = 1 - alpha / (2 * m)
        t = float(sstats.t.ppf(p, m - 2))
        lam = (m - 1) * t / np.sqrt((m - 2 + t * t) * m)
        removed.append((int(index[j]), r / lam))
        values, index = np.delete(values, j), np.delete(index, j)
    last = max((k for k, (_, ratio) in enumerate(removed) if ratio > 1), default=-1)
    scores = np.zeros(n)
    for k, (position, ratio) in enumerate(removed[: last + 1]):
        scores[position] = max(ratio, 1.0 + 1e-9) if k <= last else ratio
    return scores


def _learned_scores(
    resid: np.ndarray, method: str, season: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Isolation forest, local outlier factor or an autoencoder on sliding windows of the
    residuals; returns each period's raw outlier score and how much the period itself stands out
    in its window (0 to 1)."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.neural_network import MLPRegressor

    scale = resid.std() or 1.0
    standard = (resid - resid.mean()) / scale
    width = max(3, min(season or 7, 13, len(resid) // 4))
    windows = _windows(standard, width)
    center = width // 2
    # A window is unusual when one of its values is: the score goes to the period at its centre,
    # weighted by how much that period stands out in its window (otherwise every window that
    # merely contains a spike would flag its neighbours too).
    share = np.abs(windows[:, center]) / np.maximum(np.abs(windows).max(axis=1), 1e-12)
    if method == "isolation_forest":
        forest = IsolationForest(n_estimators=200, random_state=42).fit(windows)
        return -forest.score_samples(windows), share
    if method == "lof":
        lof = LocalOutlierFactor(n_neighbors=min(20, len(windows) - 1)).fit(windows)
        return -lof.negative_outlier_factor_, share
    # masked autoencoder: a small network rebuilds each period from its neighbours only (the
    # centre of the window is hidden); a period it cannot rebuild does not follow the pattern
    neighbours = np.delete(windows, center, axis=1)
    encoder = MLPRegressor(
        hidden_layer_sizes=(max(2, width // 2),), max_iter=2_000, random_state=42, alpha=1e-3
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoder.fit(neighbours, windows[:, center])
    error = encoder.predict(neighbours) - windows[:, center]  # signed, close to normal
    return error, np.ones(len(error))  # the masked centre is already the period's own error


def _kalman_fit(series: np.ndarray, season: int | None):
    from quantilo.timeseries import smoothing as sm

    try:
        return sm.kalman(series, season)
    except sm.KalmanError:
        return None


def kalman_source(series: np.ndarray, season: int | None) -> str:
    """How the Kalman anomalies were scored, for the line under the method."""
    fit = _kalman_fit(series, season)
    return f"Kalman one-step errors ({fit.label})" if fit else "Kalman: model could not be fitted"


def kalman_anomalies(
    series: np.ndarray, threshold: float, season: int | None
) -> list[AnomalyPoint]:
    """Periods whose one-step-ahead prediction error, divided by its predicted standard
    deviation, is beyond `threshold` robust z-scores (064). Causal; empty periods are skipped."""
    from quantilo.timeseries import smoothing as sm

    fit = _kalman_fit(series, season)
    if fit is None:
        return []
    scores = sm.innovation_scores(fit)
    flagged = [i for i in np.argsort(-scores) if scores[i] > threshold][:MAX_ANOMALIES]
    return [
        AnomalyPoint(int(i), float(series[i]), float(fit.predicted[i]), float(scores[i]))
        for i in flagged
    ]


def anomalies(
    resid: np.ndarray,
    series: np.ndarray,
    method: Anomaly = "robust_z",
    threshold: float = 3.5,
    season: int | None = None,
) -> list[AnomalyPoint]:
    """Periods whose residual is unusual; at most MAX_ANOMALIES, the strongest first.

    Statistical rules score in their own unit (robust z, IQRs, …); the generalized ESD test
    flags points with a score above 1 at alpha = 5 %; learned methods (isolation forest, LOF,
    autoencoder) are scored as robust z-scores of their outlier score.
    """
    n = len(resid)
    scores = np.zeros(n)
    if method == "kalman":
        return kalman_anomalies(series, threshold, season)
    if method == "gesd":
        scores = gesd_scores(resid)
        flagged = [i for i in np.argsort(-scores) if scores[i] > 1][:MAX_ANOMALIES]
        return [
            AnomalyPoint(int(i), float(series[i]), float(series[i] - resid[i]), float(scores[i]))
            for i in flagged
        ]
    if method in ("isolation_forest", "lof", "autoencoder"):
        # outlier scores are skewed: robust z on a log scale, then weighted by how much the
        # period itself stands out in its window
        raw, share = _learned_scores(resid, method, season)
        if method == "autoencoder":  # robust z of the signed rebuilding error, like a residual
            scores = np.abs(_robust_z(raw))
        else:
            scores = _robust_z(np.log(np.maximum(raw, 1e-12))) * share
    elif method == "robust_z":
        mad = float(np.median(np.abs(resid - np.median(resid))))
        scores = np.abs(resid - np.median(resid)) / (1.4826 * mad) if mad else scores
    elif method == "iqr":
        q1, q3 = np.percentile(resid, [25, 75])
        iqr = q3 - q1
        if iqr:
            below, above = (q1 - resid) / iqr, (resid - q3) / iqr
            scores = np.maximum(np.maximum(below, above), 0)  # in IQRs beyond the box
    else:
        window = max(5, min(25, n // 10))
        half = window // 2
        for i in range(n):
            chunk = resid[max(0, i - half) : i + half + 1]
            if method == "rolling_z":
                sd = chunk.std()
                scores[i] = abs(resid[i] - chunk.mean()) / sd if sd else 0
            else:  # hampel: robust rolling median and MAD
                median = np.median(chunk)
                mad = np.median(np.abs(chunk - median))
                scores[i] = abs(resid[i] - median) / (1.4826 * mad) if mad else 0
    flagged = [i for i in np.argsort(-scores) if scores[i] > threshold][:MAX_ANOMALIES]
    return [
        AnomalyPoint(int(i), float(series[i]), float(series[i] - resid[i]), float(scores[i]))
        for i in flagged
    ]
