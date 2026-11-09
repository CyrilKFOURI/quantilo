"""How far a column's distribution moved between two versions: PSI, normalised Wasserstein
distance, Kolmogorov–Smirnov and χ².

The numeric measures work on quantile functions (the snapshots keep 101 quantiles per column, see
`snapshot.py`), so two versions compare without reading their rows again. Read from 1 % steps,
the KS statistic is exact to about ±0.01 and the Wasserstein distance to the trapezoid rule.

PSI (population stability index) = Σ (a − e) · ln(a / e) over shared bins, e the share in the
old version, a in the new one. An empty bin would make it infinite: shares are floored at
ε = 1e-4 before the sum (the usual smoothing). Reading (Siddiqi, *Credit Risk Scorecards*, 2006):
below 0.1 stable, 0.1 to 0.25 moderate shift, above 0.25 strong shift.
"""

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy import stats

PSI_EPSILON = 1e-4
PSI_MODERATE = 0.1
PSI_STRONG = 0.25
PSI_SOURCE = "PSI < 0.1 stable · 0.1–0.25 moderate · > 0.25 strong (Siddiqi, 2006)"

Values = Sequence[float] | np.ndarray
Drift = Literal["stable", "moderate", "strong"]


def drift_of(psi: float | None) -> Drift | None:
    if psi is None:
        return None
    if psi >= PSI_STRONG:
        return "strong"
    return "moderate" if psi >= PSI_MODERATE else "stable"


def psi(expected: Values, actual: Values, epsilon: float = PSI_EPSILON) -> float:
    """PSI of two sets of counts (or shares) over the same bins, empty bins floored at ε."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.shape != a.shape or e.sum() <= 0 or a.sum() <= 0:
        raise ValueError("PSI needs two non-empty sets of counts over the same bins.")
    e = np.maximum(e / e.sum(), epsilon)
    a = np.maximum(a / a.sum(), epsilon)
    return float(np.sum((a - e) * np.log(a / e)))


def cdf_at(quantiles: Values, levels: Values, x: np.ndarray) -> np.ndarray:
    """F(x) = share of values ≤ x, read from a quantile function (values at given levels).

    Tied quantiles (a value held by many rows) keep their highest level: the CDF jumps there.
    """
    q = np.asarray(quantiles, dtype=float)
    p = np.asarray(levels, dtype=float)
    values, last = np.unique(q[::-1], return_index=True)
    shares = p[::-1][last]  # the highest level of each distinct quantile
    shares = np.maximum.accumulate(shares)
    return np.interp(x, values, shares, left=0.0, right=1.0)


def bin_shares(quantiles: Values, levels: Values, edges: np.ndarray) -> np.ndarray:
    """Share of values in (−∞, e1], (e1, e2] … (ek, +∞): k inner edges give k + 1 bins."""
    inner = cdf_at(quantiles, levels, edges)
    cumulative = np.concatenate([[0.0], inner, [1.0]])
    return np.clip(np.diff(cumulative), 0.0, 1.0)


def decile_edges(quantiles: Values, levels: Values) -> np.ndarray:
    """The old version's deciles (10 % … 90 %), duplicates dropped (heavy ties merge bins)."""
    q = np.asarray(quantiles, dtype=float)
    p = np.asarray(levels, dtype=float)
    edges = np.interp(np.arange(1, 10) / 10, p, q)
    return np.unique(edges)


def psi_on_deciles(before: Values, after: Values, levels: Values) -> float:
    """PSI of a number, on the old version's deciles: each holds 10 % of the old rows."""
    edges = decile_edges(before, levels)
    return psi(bin_shares(before, levels, edges), bin_shares(after, levels, edges))


def wasserstein(before: Values, after: Values, levels: Values) -> float:
    """W₁ = ∫₀¹ |Q_before(p) − Q_after(p)| dp, from two quantile functions on the same levels."""
    gap = np.abs(np.asarray(before, dtype=float) - np.asarray(after, dtype=float))
    p = np.asarray(levels, dtype=float)
    return float(np.sum((gap[1:] + gap[:-1]) / 2 * np.diff(p)))


def normalised_wasserstein(before: Values, after: Values, levels: Values) -> float | None:
    """W₁ in units of the old version's interquartile range (its standard deviation when the
    IQR is 0); None for a constant column."""
    q = np.asarray(before, dtype=float)
    p = np.asarray(levels, dtype=float)
    q1, q3 = np.interp([0.25, 0.75], p, q)
    scale = float(q3 - q1)
    if scale <= 0:
        scale = float(np.std(q))
    if scale <= 0 or not math.isfinite(scale):
        return None
    return wasserstein(before, after, levels) / scale


def ks_from_quantiles(
    before: Values,
    after: Values,
    levels: Values,
    n_before: int,
    n_after: int,
) -> tuple[float, float | None]:
    """Two-sample Kolmogorov–Smirnov from quantile functions: D = sup |F_before − F_after| over
    both grids, p from the asymptotic Kolmogorov distribution with n·m / (n + m) rows."""
    grid = np.unique(np.concatenate([np.asarray(before, float), np.asarray(after, float)]))
    gap = np.abs(cdf_at(before, levels, grid) - cdf_at(after, levels, grid))
    statistic = float(np.max(gap)) if len(gap) else 0.0
    if n_before <= 0 or n_after <= 0:
        return statistic, None
    effective = n_before * n_after / (n_before + n_after)
    return statistic, float(stats.kstwobign.sf(statistic * math.sqrt(effective)))


def chi_square(before: Values, after: Values) -> tuple[float, float, int] | None:
    """χ² of homogeneity on a 2 × k table of counts (no continuity correction); None when fewer
    than two levels are seen."""
    table = np.array([before, after], dtype=float)
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2 or (table.sum(axis=1) == 0).any():
        return None
    result = stats.chi2_contingency(table, correction=False)
    return float(result.statistic), float(result.pvalue), int(result.dof)  # type: ignore[attr-defined]
