"""Trend and seasonality methods of the Time series page (catalogue: docs/TIMESERIES_METHODS.md).

Every method takes a series in period order (gaps as None where allowed) and returns a
`MethodResult`. The default methods of each section are listed in `DEFAULT_*`.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import signal, stats

from quantilo.timeseries.trend import mann_kendall
from quantilo.variable.source import clean

ALPHA = 0.05


@dataclass
class MethodResult:
    key: str
    statistic: float | None
    p_value: float | None
    n: int
    significant: bool | None  # at ALPHA; None when the method is not a test
    value: float | None = None  # what the method estimates (slope, tau, dominant period…)
    low: float | None = None  # 95% interval of `value`, when there is one
    high: float | None = None
    note: str | None = None


def _known(values: list[float | None]) -> tuple[np.ndarray, np.ndarray]:
    """Positions and values of the periods that have a value."""
    positions = np.array([i for i, v in enumerate(values) if v is not None], dtype=float)
    data = np.array([v for v in values if v is not None], dtype=float)
    return positions, data


def _not_run(key: str, n: int, reason: str) -> MethodResult:
    return MethodResult(key, None, None, n, None, note=reason)


def _test(key: str, statistic: float | None, p: float | None, n: int, **extra) -> MethodResult:
    p = clean(p)
    return MethodResult(key, clean(statistic), p, n, None if p is None else p < ALPHA, **extra)


# --- Trend ---------------------------------------------------------------------------------------


def mann_kendall_test(values: list[float | None], season: int | None = None) -> MethodResult:
    _, data = _known(values)
    if len(data) < 3 or np.ptp(data) == 0:
        return _not_run("mann_kendall", len(data), "Needs at least 3 periods that vary.")
    _, z, p, tau = mann_kendall(data)
    return _test("mann_kendall", z, p, len(data), value=tau)


def sen_slope(values: list[float | None], season: int | None = None) -> MethodResult:
    positions, data = _known(values)
    if len(data) < 3:
        return _not_run("sen", len(data), "Needs at least 3 periods.")
    fit = np.asarray(stats.theilslopes(data, positions, alpha=0.95), dtype=float)
    slope, low, high = float(fit[0]), float(fit[2]), float(fit[3])
    excludes_zero = low > 0 or high < 0
    return MethodResult("sen", None, None, len(data), excludes_zero, slope, low, high)


def hamed_rao(values: list[float | None], season: int | None = None) -> MethodResult:
    """Mann-Kendall with its variance inflated by the autocorrelation of the detrended ranks."""
    positions, data = _known(values)
    n = len(data)
    if n < 10 or np.ptp(data) == 0:
        return _not_run("hamed_rao", n, "Needs at least 10 periods that vary.")
    s, _, _, tau = mann_kendall(data)
    fit = np.asarray(stats.theilslopes(data, positions), dtype=float)
    ranks = stats.rankdata(data - fit[0] * positions)
    centred = ranks - ranks.mean()
    denominator = float((centred**2).sum())
    _, counts = np.unique(data, return_counts=True)
    ties = counts[counts > 1]
    variance = (n * (n - 1) * (2 * n + 5) - float((ties * (ties - 1) * (2 * ties + 5)).sum())) / 18
    bound = 1.96 / math.sqrt(n)
    correction = 0.0
    for k in range(1, n - 2):
        r = float((centred[:-k] * centred[k:]).sum()) / denominator
        if abs(r) > bound:
            correction += (n - k) * (n - k - 1) * (n - k - 2) * r
    factor = max(1 + 2 / (n * (n - 1) * (n - 2)) * correction, 1e-9)
    variance *= factor
    corrected = s - 1 if s > 0 else s + 1
    z = 0.0 if s == 0 or variance <= 0 else corrected / math.sqrt(variance)
    p = float(2 * stats.norm.sf(abs(z)))
    note = f"Variance multiplied by {factor:.2f} for autocorrelation."
    return _test("hamed_rao", z, p, n, value=tau, note=note)


def seasonal_mann_kendall(values: list[float | None], season: int | None = None) -> MethodResult:
    """Mann-Kendall within each position of the cycle (each month…), summed: a trend that
    the seasonal pattern cannot fake."""
    if not season or season < 2:
        return _not_run("seasonal_mk", 0, "Needs a seasonal period.")
    s_total, variance, pairs, n = 0, 0.0, 0, 0
    for offset in range(season):
        _, data = _known(values[offset::season])
        m = len(data)
        if m < 2:
            continue
        signs = np.sign(data[None, :] - data[:, None])
        s_total += int(np.triu(signs, k=1).sum())
        _, counts = np.unique(data, return_counts=True)
        ties = counts[counts > 1]
        variance += (
            m * (m - 1) * (2 * m + 5) - float((ties * (ties - 1) * (2 * ties + 5)).sum())
        ) / 18
        pairs += m * (m - 1) // 2
        n += m
    if pairs == 0 or variance <= 0:
        return _not_run("seasonal_mk", n, "Needs at least two full cycles.")
    corrected = s_total - 1 if s_total > 0 else s_total + 1
    z = 0.0 if s_total == 0 else corrected / math.sqrt(variance)
    p = float(2 * stats.norm.sf(abs(z)))
    return _test("seasonal_mk", z, p, n, value=s_total / pairs)


def cox_stuart(values: list[float | None], season: int | None = None) -> MethodResult:
    """Sign test between each value of the first half and its counterpart in the second."""
    _, data = _known(values)
    n = len(data)
    half = n // 2
    if half < 3:
        return _not_run("cox_stuart", n, "Needs at least 6 periods.")
    differences = data[n - half :] - data[:half]
    differences = differences[differences != 0]
    if len(differences) == 0:
        return _not_run("cox_stuart", n, "Both halves are identical.")
    rises = int((differences > 0).sum())
    p = float(stats.binomtest(rises, len(differences), 0.5).pvalue)
    return _test("cox_stuart", rises, p, n, value=rises / len(differences))


def spearman_time(values: list[float | None], season: int | None = None) -> MethodResult:
    positions, data = _known(values)
    if len(data) < 3 or np.ptp(data) == 0:
        return _not_run("spearman_time", len(data), "Needs at least 3 periods that vary.")
    rho, p = (float(x) for x in np.asarray(stats.spearmanr(positions, data), dtype=float)[:2])
    return _test("spearman_time", rho, p, len(data), value=rho)


def linear_trend(values: list[float | None], season: int | None = None) -> MethodResult:
    positions, data = _known(values)
    if len(data) < 3 or np.ptp(data) == 0:
        return _not_run("linear_trend", len(data), "Needs at least 3 periods that vary.")
    fit = stats.linregress(positions, data)
    slope, error = float(fit.slope), float(fit.stderr)  # type: ignore[attr-defined]
    t_value = slope / error if error > 0 else None
    margin = float(stats.t.ppf(0.975, len(data) - 2)) * error
    note = f"R² = {float(fit.rvalue) ** 2:.3f}"  # type: ignore[attr-defined]
    return _test(
        "linear_trend",
        t_value,
        float(fit.pvalue),  # type: ignore[attr-defined]
        len(data),
        value=slope,
        low=slope - margin,
        high=slope + margin,
        note=note,
    )


Method = Callable[[list[float | None], int | None], MethodResult]
TREND: dict[str, Method] = {
    "mann_kendall": mann_kendall_test,
    "sen": sen_slope,
    "hamed_rao": hamed_rao,
    "seasonal_mk": seasonal_mann_kendall,
    "cox_stuart": cox_stuart,
    "spearman_time": spearman_time,
    "linear_trend": linear_trend,
}
DEFAULT_TREND = ("mann_kendall", "sen")


# --- Seasonality (on a series without holes) -----------------------------------------------------


def kruskal_seasons(detrended: np.ndarray, season: int | None) -> MethodResult:
    """Do the positions of the cycle (Mondays, Tuesdays…) differ once the trend is removed?"""
    if not season or season < 2 or len(detrended) < 2 * season:
        return _not_run("kruskal_season", len(detrended), "Needs at least two full cycles.")
    groups = [detrended[offset::season] for offset in range(season)]
    if all(np.ptp(g) == 0 for g in groups):
        return _not_run("kruskal_season", len(detrended), "The values don't vary.")
    h, p = (float(x) for x in np.asarray(stats.kruskal(*groups), dtype=float)[:2])
    n, k = len(detrended), season
    epsilon = max(0.0, (h - k + 1) / (n - k)) if n > k else None
    return _test("kruskal_season", h, p, n, value=epsilon)


def friedman_seasons(detrended: np.ndarray, season: int | None) -> MethodResult:
    """Kruskal-Wallis paired by cycle: each full cycle is a block."""
    if not season or season < 3:
        return _not_run("friedman", len(detrended), "Needs a cycle of at least 3 periods.")
    cycles = len(detrended) // season
    if cycles < 2:
        return _not_run("friedman", len(detrended), "Needs at least two full cycles.")
    table = detrended[: cycles * season].reshape(cycles, season)
    statistic, p = (
        float(x) for x in np.asarray(stats.friedmanchisquare(*table.T), dtype=float)[:2]
    )
    return _test("friedman", statistic, p, cycles * season, note=f"{cycles} full cycles.")


def _autocorrelation(values: np.ndarray, lag: int) -> float | None:
    if lag >= len(values) or np.ptp(values) == 0:
        return None
    centred = values - values.mean()
    return float((centred[:-lag] * centred[lag:]).sum() / (centred**2).sum())


def qs_test(series: np.ndarray, season: int | None) -> MethodResult:
    """Positive autocorrelation at one and two cycles of the differenced series (X-13's QS)."""
    if not season or season < 2:
        return _not_run("qs", len(series), "Needs a seasonal period.")
    differenced = np.diff(series)
    n = len(differenced)
    if n < 2 * season + 2:
        return _not_run("qs", n, "Needs more than two full cycles.")
    total = 0.0
    for lag in (season, 2 * season):
        r = _autocorrelation(differenced, lag) or 0.0
        total += max(0.0, r) ** 2 / (n - lag)
    statistic = n * (n + 2) * total
    return _test("qs", statistic, float(stats.chi2.sf(statistic, 2)), n)


def periodogram_peak(series: np.ndarray, season: int | None) -> MethodResult:
    """The cycle length carrying the most variance once a straight trend is removed."""
    n = len(series)
    if n < 8:
        return _not_run("periodogram", n, "Needs at least 8 periods.")
    frequencies, power = signal.periodogram(signal.detrend(series), scaling="spectrum")
    frequencies, power = frequencies[1:], power[1:]
    if power.sum() <= 0:
        return _not_run("periodogram", n, "The values don't vary.")
    order = np.argsort(power)[::-1][:3]
    peaks = [f"{1 / frequencies[i]:.1f} periods ({power[i] / power.sum():.0%})" for i in order]
    best = int(order[0])
    return MethodResult(
        "periodogram",
        float(power[best] / power.sum()),
        None,
        n,
        None,
        value=float(1 / frequencies[best]),
        note="Strongest cycles: " + ", ".join(peaks) + ".",
    )


def fisher_g(series: np.ndarray, season: int | None) -> MethodResult:
    """Is the strongest periodogram peak more than white noise would produce?"""
    n = len(series)
    q = (n - 1) // 2
    if q < 3:
        return _not_run("fisher_g", n, "Needs at least 7 periods.")
    _, power = signal.periodogram(series - series.mean())
    ordinates = power[1 : q + 1]
    if ordinates.sum() <= 0:
        return _not_run("fisher_g", n, "The values don't vary.")
    g = float(ordinates.max() / ordinates.sum())
    p = 0.0
    for j in range(1, int(1 / g) + 1):
        p += (-1) ** (j - 1) * math.comb(q, j) * (1 - j * g) ** (q - 1)
    return _test("fisher_g", g, min(1.0, max(0.0, p)), n)


def seasonal_acf(series: np.ndarray, season: int | None) -> MethodResult:
    if not season or season < 2:
        return _not_run("seasonal_acf", len(series), "Needs a seasonal period.")
    r = _autocorrelation(series, season)
    n = len(series)
    if r is None:
        return _not_run("seasonal_acf", n, "Needs more than one full cycle.")
    p = float(2 * stats.norm.sf(abs(r) * math.sqrt(n)))
    return _test("seasonal_acf", r, p, n, value=r)


SeasonMethod = Callable[[np.ndarray, int | None], MethodResult]
# Methods marked `detrended` receive the series with its trend removed.
SEASONALITY: dict[str, tuple[SeasonMethod, bool]] = {
    "kruskal_season": (kruskal_seasons, True),
    "periodogram": (periodogram_peak, False),
    "friedman": (friedman_seasons, True),
    "qs": (qs_test, False),
    "fisher_g": (fisher_g, True),
    "seasonal_acf": (seasonal_acf, True),
}
DEFAULT_SEASONALITY = ("kruskal_season", "periodogram")
