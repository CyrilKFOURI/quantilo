"""Bootstrap intervals (051): seeded, bounded, and saying which method gave the interval and why.

Each group is resampled 2,000 times from its own seeded stream (independent groups need
independent draws). A group too large for the budget is resampled on a reproducible sample of m
values, and the spread around the centre is scaled by √(m/n) ("m out of n" bootstrap): the
sampling spread of a mean or a median shrinks as 1/√n. Percentile intervals are widened for small
samples (Hesterberg's expanded percentile), where the plain percentile interval is too narrow.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

DRAWS = 2_000
SEED = 42
BUDGET = 8_000_000  # resampled values per view
CHUNK = 2_000_000  # resampled values held in memory at once
MIN_VALUES = 5  # R-BS-01: below, resampling teaches nothing
SMALL = 30  # R-BS-01: fewer values than this → bootstrap the mean
SKEWED = 1.0  # R-BS-01: |skewness| beyond this → bootstrap the mean…
SKEW_LIMIT = 2_000  # …up to this many values; beyond, the mean is close to normal anyway

IntervalMethod = Literal["t", "bootstrap"]


@dataclass
class Replicates:
    means: np.ndarray
    medians: np.ndarray
    m: int  # values resampled
    n: int  # values in the group


def held_size(groups: int, available: int) -> int:
    """Values resampled per group so that the view stays within the budget (R-BS-05)."""
    return max(MIN_VALUES, min(available, BUDGET // (DRAWS * max(groups, 1))))


def skewness(values: np.ndarray) -> float | None:
    if len(values) < 3 or np.ptp(values) == 0:
        return None
    found = float(stats.skew(values, bias=False))
    return found if math.isfinite(found) else None


def mean_rule(n: int, values: np.ndarray) -> tuple[IntervalMethod, str]:
    """Bootstrap or Student t for a group's mean, and why (R-BS-01)."""
    if n < MIN_VALUES:
        return "t", f"{n} values: too few to resample, Student t kept"
    if n < SMALL:
        return "bootstrap", f"{n} values: too few for the t interval"
    skew = skewness(values)
    if skew is not None and abs(skew) > SKEWED:
        if n <= SKEW_LIMIT:
            return "bootstrap", f"skewness {skew:.1f}: the mean is far from normal"
        return "t", f"skewness {skew:.1f}, but {n:,} values: the mean is close to normal"
    return "t", f"{n:,} values, skewness within ±1"


def replicates(
    values: np.ndarray, n: int, stream: int, centers: tuple[float, float] | None = None
) -> Replicates:
    """2,000 resampled means and medians of `values` (seeded by `stream`), on the scale of n
    values and centred on the group's own mean and median when given."""
    m = len(values)
    rng = np.random.default_rng([SEED, stream])
    rows = max(1, CHUNK // max(m, 1))
    means, medians = [], []
    for start in range(0, DRAWS, rows):
        drawn = values[rng.integers(0, m, size=(min(rows, DRAWS - start), m))]
        means.append(drawn.mean(axis=1))
        medians.append(np.median(drawn, axis=1))
    mean_reps, median_reps = np.concatenate(means), np.concatenate(medians)
    mean, median = float(values.mean()), float(np.median(values))
    if m < n:
        scale = math.sqrt(m / n)
        mean_reps = mean + (mean_reps - mean) * scale
        median_reps = median + (median_reps - median) * scale
    if centers is not None:
        mean_reps = mean_reps + (centers[0] - mean)
        median_reps = median_reps + (centers[1] - median)
    return Replicates(mean_reps, median_reps, m, n)


def interval(reps: np.ndarray, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval, widened for small n (R-BS-04)."""
    tail = alpha / 2
    if n >= 2:
        tail = float(stats.norm.cdf(-math.sqrt(n / (n - 1)) * stats.t.ppf(1 - alpha / 2, n - 1)))
    low, high = np.quantile(reps, [tail, 1 - tail])
    return float(low), float(high)


def sample_note(m: int, n: int) -> str:
    """How a large group was resampled (R-BS-05), empty when every value was."""
    return f" (on a {m:,}-value sample, scaled to {n:,})" if m < n else ""
