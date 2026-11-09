"""Split a series into trend, seasonality and remainder (STL, MSTL or classical)."""

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from statsmodels.tsa.seasonal import MSTL, STL, seasonal_decompose

Method = Literal["stl", "mstl", "classical_additive", "classical_multiplicative"]
METHODS: tuple[Method, ...] = ("stl", "mstl", "classical_additive", "classical_multiplicative")


@dataclass
class Decomposition:
    method: Method
    seasons: list[int]  # cycle lengths, in periods (several with MSTL)
    trend: list[float | None]
    seasonal: list[list[float | None]]  # one component per cycle length
    resid: list[float | None]
    strength_trend: float | None  # Hyndman: 0 (none) … 1 (dominant)
    strength_seasonal: float | None
    notes: list[str] = field(default_factory=list)


class DecompositionError(ValueError):
    pass


@dataclass
class StlFit:
    trend: np.ndarray
    seasonal: np.ndarray
    resid: np.ndarray
    robust: bool


def _degenerate(resid: np.ndarray, series: np.ndarray) -> bool:
    """A remainder made mostly of exact zeros, or with no robust spread at all."""
    scale = float(np.std(series)) or 1.0
    flat = float(np.mean(np.abs(resid) < 1e-9 * scale))
    mad = float(np.median(np.abs(resid - np.median(resid))))
    return flat >= 0.2 or mad <= 1e-6 * scale


def stl(series: np.ndarray, season: int) -> StlFit:
    """STL tuned for short series.

    With fewer than 7 cycles, each position of the cycle has only a few values: a linear
    seasonal smoother passes through them and swallows the noise, so a constant one is used.
    The robust fit is kept unless it degenerates (its weights can leave a remainder of exact
    zeros); then the ordinary fit is used.
    """
    degree = 0 if len(series) // season < 7 else 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = STL(series, period=season, robust=True, seasonal_deg=degree).fit()
        robust = not _degenerate(np.asarray(fit.resid, dtype=float), series)
        if not robust:
            fit = STL(series, period=season, robust=False, seasonal_deg=degree).fit()
    return StlFit(
        np.asarray(fit.trend, dtype=float),
        np.asarray(fit.seasonal, dtype=float),
        np.asarray(fit.resid, dtype=float),
        robust,
    )


NON_ROBUST_NOTE = "Too few cycles for a robust STL: the ordinary STL is used."


def _listed(values) -> list[float | None]:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(values, dtype=float)]


def strength(component: np.ndarray, resid: np.ndarray) -> float | None:
    """max(0, 1 - Var(R) / Var(component + R)), over the periods where both exist."""
    keep = np.isfinite(component) & np.isfinite(resid)
    if keep.sum() < 3:
        return None
    total = float(np.var(component[keep] + resid[keep]))
    if total == 0:
        return None
    return max(0.0, 1 - float(np.var(resid[keep])) / total)


def secondary_seasons(period: str, n: int, season: int) -> list[int]:
    """Cycle lengths for MSTL: the main one, plus the next calendar cycle when it fits twice."""
    longer = {"hour": 168, "day": 365}.get(period)
    seasons = [season]
    if longer and longer > season and n >= 2 * longer:
        seasons.append(longer)
    return seasons


def decompose(series: np.ndarray, method: Method, season: int, period: str) -> Decomposition:
    n = len(series)
    if season < 2 or n < 2 * season:
        raise DecompositionError(
            f"Decomposition needs at least two full cycles: {2 * season} periods, you have {n}."
        )
    notes: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "stl":
            found = stl(series, season)
            seasons = [season]
            trend, seasonal, resid = found.trend, [found.seasonal], found.resid
            if not found.robust:
                notes.append(NON_ROBUST_NOTE)
        elif method == "mstl":
            seasons = secondary_seasons(period, n, season)
            if len(seasons) == 1:
                notes.append("Only one cycle fits twice in the data: MSTL equals STL here.")
            fit = MSTL(series, periods=seasons).fit()
            components = np.asarray(fit.seasonal, dtype=float)
            seasonal = [components] if components.ndim == 1 else list(components.T)
            trend, resid = fit.trend, fit.resid
        else:
            model = "multiplicative" if method == "classical_multiplicative" else "additive"
            if model == "multiplicative" and (series <= 0).any():
                raise DecompositionError("A multiplicative decomposition needs values above 0.")
            fit = seasonal_decompose(series, model=model, period=season, two_sided=True)
            seasons = [season]
            trend, seasonal, resid = fit.trend, [fit.seasonal], fit.resid
            notes.append("Moving averages leave half a cycle without trend at each end.")
    trend = np.asarray(trend, dtype=float)
    resid = np.asarray(resid, dtype=float)
    seasonal_sum = np.sum([np.asarray(s, dtype=float) for s in seasonal], axis=0)
    if method == "classical_multiplicative":
        notes.append(
            "Seasonal and remainder are factors (1 = no effect); strengths use their logs."
        )
        log_resid = np.log(np.where(resid > 0, resid, np.nan))
        f_trend = strength(np.log(np.where(trend > 0, trend, np.nan)), log_resid)
        f_season = strength(np.log(np.where(seasonal_sum > 0, seasonal_sum, np.nan)), log_resid)
    else:
        f_trend, f_season = strength(trend, resid), strength(seasonal_sum, resid)
    return Decomposition(
        method,
        seasons,
        _listed(trend),
        [_listed(s) for s in seasonal],
        _listed(resid),
        f_trend,
        f_season,
        notes,
    )


def trend_component(series: np.ndarray, season: int | None) -> tuple[np.ndarray, str]:
    """A trend to remove before comparing seasons: STL's when two cycles fit, else a line."""
    if season and season >= 2 and len(series) >= 2 * season:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            found = stl(series, season)
            return found.trend, "STL trend" if found.robust else "STL trend, non-robust"
    positions = np.arange(len(series), dtype=float)
    slope, intercept = np.polyfit(positions, series, 1) if len(series) > 1 else (0.0, series[0])
    return slope * positions + intercept, "straight line"
