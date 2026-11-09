"""Does the variability change over time? Heteroscedasticity and ARCH effects on the residuals."""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch,
    het_breuschpagan,
    het_goldfeldquandt,
    het_white,
)

from quantilo.timeseries.methods import ALPHA, MethodResult
from quantilo.variable.source import clean

WINDOWS = 4  # equal time windows compared by Levene and its relatives
MIN_PERIODS = 20


@dataclass
class VolatilitySummary:
    rising: bool | None  # direction of the rolling standard deviation over time
    log_helps: bool  # variability grows with the level and every value is above 0


def _quiet(compute: Callable):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute()


def _not_run(key: str, n: int, reason: str) -> MethodResult:
    return MethodResult(key, None, None, n, None, note=reason)


def _time(n: int, squared: bool = False) -> np.ndarray:
    t = np.arange(n, dtype=float) / max(n - 1, 1)
    columns = [np.ones(n), t] + ([t**2] if squared else [])
    return np.column_stack(columns)


def _enough(key: str, resid: np.ndarray) -> MethodResult | None:
    if len(resid) < MIN_PERIODS or np.ptp(resid) == 0:
        return _not_run(key, len(resid), f"Needs at least {MIN_PERIODS} periods that vary.")
    return None


def breusch_pagan(resid: np.ndarray) -> MethodResult:
    if problem := _enough("breusch_pagan", resid):
        return problem
    lm, p, _, _ = _quiet(lambda: het_breuschpagan(resid, _time(len(resid))))
    return MethodResult("breusch_pagan", clean(lm), clean(p), len(resid), bool(p < ALPHA))


def white(resid: np.ndarray) -> MethodResult:
    if problem := _enough("white", resid):
        return problem
    lm, p, _, _ = _quiet(lambda: het_white(resid, _time(len(resid))))
    return MethodResult("white", clean(lm), clean(p), len(resid), bool(p < ALPHA))


def goldfeld_quandt(resid: np.ndarray) -> MethodResult:
    """First part against last part, the central 20 % left out."""
    if problem := _enough("goldfeld_quandt", resid):
        return problem
    f, p, _ = _quiet(
        lambda: het_goldfeldquandt(resid, _time(len(resid)), drop=0.2, alternative="two-sided")
    )
    return MethodResult(
        "goldfeld_quandt", clean(f), clean(p), len(resid), bool(p < ALPHA), value=clean(f)
    )


def arch_lm(resid: np.ndarray) -> MethodResult:
    """Engle: are large shocks followed by large shocks?"""
    if problem := _enough("arch_lm", resid):
        return problem
    lags = max(1, min(12, len(resid) // 10))
    lm, p, _, _ = _quiet(lambda: het_arch(resid, nlags=lags))
    return MethodResult(
        "arch_lm", clean(lm), clean(p), len(resid), bool(p < ALPHA), note=f"{lags} lags."
    )


def mcleod_li(resid: np.ndarray) -> MethodResult:
    """Ljung-Box on the squared residuals."""
    if problem := _enough("mcleod_li", resid):
        return problem
    lags = max(1, min(10, len(resid) // 5))
    table = _quiet(lambda: acorr_ljungbox((resid - resid.mean()) ** 2, lags=[lags]))
    statistic, p = float(table["lb_stat"].iloc[0]), float(table["lb_pvalue"].iloc[0])
    return MethodResult(
        "mcleod_li",
        clean(statistic),
        clean(p),
        len(resid),
        bool(p < ALPHA),
        note=f"Up to lag {lags}.",
    )


def _windows(resid: np.ndarray) -> list[np.ndarray]:
    return [chunk for chunk in np.array_split(resid, WINDOWS) if len(chunk) >= 3]


def _between_windows(key: str, resid: np.ndarray, test: Callable) -> MethodResult:
    if problem := _enough(key, resid):
        return problem
    statistic, p = (
        float(x) for x in np.asarray(_quiet(lambda: test(*_windows(resid))), dtype=float)[:2]
    )
    return MethodResult(
        key,
        clean(statistic),
        clean(p),
        len(resid),
        bool(p < ALPHA),
        note=f"{WINDOWS} equal time windows.",
    )


def levene(resid: np.ndarray) -> MethodResult:
    return _between_windows("levene", resid, lambda *g: stats.levene(*g, center="mean"))


def brown_forsythe(resid: np.ndarray) -> MethodResult:
    return _between_windows("brown_forsythe", resid, lambda *g: stats.levene(*g, center="median"))


def bartlett(resid: np.ndarray) -> MethodResult:
    return _between_windows("bartlett", resid, stats.bartlett)


def fligner(resid: np.ndarray) -> MethodResult:
    return _between_windows("fligner", resid, stats.fligner)


VOLATILITY: dict[str, Callable[[np.ndarray], MethodResult]] = {
    "breusch_pagan": breusch_pagan,
    "arch_lm": arch_lm,
    "white": white,
    "goldfeld_quandt": goldfeld_quandt,
    "mcleod_li": mcleod_li,
    "levene": levene,
    "brown_forsythe": brown_forsythe,
    "bartlett": bartlett,
    "fligner": fligner,
}
DEFAULT_VOLATILITY = ("breusch_pagan", "arch_lm")


def summary(
    series: np.ndarray, rolling_std: list[float | None], fitted: np.ndarray
) -> VolatilitySummary:
    """Which way the variability goes, and whether it follows the level (then a log helps)."""
    pairs = [(i, s) for i, s in enumerate(rolling_std) if s is not None]
    if len(pairs) < 3:
        return VolatilitySummary(None, False)
    positions = np.array([i for i, _ in pairs], dtype=float)
    spreads = np.array([s for _, s in pairs], dtype=float)
    slope = float(np.polyfit(positions, spreads, 1)[0])
    levels = fitted[positions.astype(int)]
    linked = (
        np.ptp(levels) > 0
        and np.ptp(spreads) > 0
        and float(np.corrcoef(levels, spreads)[0, 1]) > 0.5
    )
    return VolatilitySummary(slope > 0, bool(linked and (series > 0).all()))


def rolling_spread(resid: np.ndarray, window: int) -> list[float | None]:
    """Rolling standard deviation of the residuals (trailing, half a window required)."""
    needed = math.ceil(window / 2)
    out: list[float | None] = []
    for i in range(len(resid)):
        chunk = resid[max(0, i - window + 1) : i + 1]
        out.append(float(chunk.std(ddof=1)) if len(chunk) >= max(2, needed) else None)
    return out
