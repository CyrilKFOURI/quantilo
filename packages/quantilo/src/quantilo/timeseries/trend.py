"""Monotonic trend: Mann-Kendall test and Sen's slope, on a series with one value per period."""

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

MAX_POINTS = 5_000  # pairwise comparisons grow with n²: a regular sample beyond this


@dataclass
class Trend:
    s: int
    z: float
    p_value: float
    tau: float  # S over the number of pairs: -1 (always falls) … 1 (always rises)
    slope: float  # Sen's slope, in units of the value per period
    slope_low: float  # 95% confidence interval
    slope_high: float
    n: int
    lag1: float | None  # lag-1 autocorrelation: above 0.5 the p-value is optimistic
    sampled: bool = False


def mann_kendall(values: np.ndarray) -> tuple[int, float, float, float]:
    """S, Z (tie-corrected variance, continuity correction), two-sided p-value and tau."""
    n = len(values)
    signs = np.sign(values[None, :] - values[:, None])
    s = int(np.triu(signs, k=1).sum())
    _, counts = np.unique(values, return_counts=True)
    ties = counts[counts > 1]
    variance = (n * (n - 1) * (2 * n + 5) - float((ties * (ties - 1) * (2 * ties + 5)).sum())) / 18
    corrected = s - 1 if s > 0 else s + 1  # continuity correction
    z = 0.0 if variance <= 0 or s == 0 else corrected / math.sqrt(variance)
    p = float(2 * stats.norm.sf(abs(z)))
    return s, z, p, s / (n * (n - 1) / 2)


def lag1_autocorrelation(values: np.ndarray) -> float | None:
    if len(values) < 3 or np.ptp(values[:-1]) == 0 or np.ptp(values[1:]) == 0:
        return None
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def trend(values: list[float | None]) -> Trend | None:
    """Trend of a series in period order; gaps are left out (their position is kept)."""
    positions = np.array([i for i, v in enumerate(values) if v is not None], dtype=float)
    data = np.array([v for v in values if v is not None], dtype=float)
    sampled = len(data) > MAX_POINTS
    if sampled:
        keep = np.linspace(0, len(data) - 1, MAX_POINTS).round().astype(int)
        positions, data = positions[keep], data[keep]
    if len(data) < 3 or np.ptp(data) == 0:
        return None
    s, z, p, tau = mann_kendall(data)
    # slope, intercept, low slope, high slope
    fit = np.asarray(stats.theilslopes(data, positions, alpha=0.95), dtype=float)
    slope, low, high = float(fit[0]), float(fit[2]), float(fit[3])
    return Trend(
        s,
        z,
        p,
        tau,
        slope,
        low,
        high,
        len(data),
        lag1_autocorrelation(data),
        sampled,
    )
