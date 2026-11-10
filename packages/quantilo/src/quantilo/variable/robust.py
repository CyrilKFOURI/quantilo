"""Robust center and spread of a numeric variable (056, rule R-ROB-1): shown when the outliers
are numerous enough to pull the mean and the standard deviation."""

import warnings
from dataclasses import dataclass

import numpy as np
from statsmodels.robust.norms import HuberT, estimate_location
from statsmodels.robust.scale import Huber

from quantilo.variable.numeric import Describe
from quantilo.variable.outliers import bounds
from quantilo.variable.source import VariableSource, clean

OUTLIER_SHARE = 0.01  # a normal law leaves 0.7 % of values beyond Tukey's fences
MAD_TO_SIGMA = 1.4826  # 1 / Φ⁻¹(0.75): the MAD of a normal law times this is σ
IQR_TO_SIGMA = 1.349  # the IQR of a normal law is 1.349 σ
HUBER_C = 1.5
HUBER_ITERATIONS = 200  # statsmodels' 30 do not settle on a skewed amount column
SAMPLE = 20_000  # the profile's sample, already read
MIN_VALUES = 8


@dataclass
class Robust:
    fires: bool  # shown on the variable page
    reason: str
    outliers: int  # beyond 1.5 IQR, on every row
    outlier_share: float
    median: float | None
    mad_sigma: float | None  # MAD × 1.4826
    iqr_sigma: float | None  # IQR / 1.349
    huber_center: float | None
    huber_scale: float | None
    mean_shift: float | None  # (mean − Huber center) / |Huber center|
    sample: int  # values Huber's estimate read
    note: str | None = None


def _huber(values: np.ndarray, mad_sigma: float) -> tuple[float | None, float | None]:
    """Huber's joint center and scale ("proposal 2"); when it does not converge, the center
    alone with the scale held at the MAD."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            center, scale = Huber(c=HUBER_C, maxiter=HUBER_ITERATIONS)(values)
            return clean(center), clean(scale)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            pass
        try:
            center = estimate_location(
                values, mad_sigma, norm=HuberT(HUBER_C), maxiter=HUBER_ITERATIONS
            )
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return None, None
        return clean(center), None


def robust(source: VariableSource, summary: Describe) -> Robust | None:
    if summary.count < MIN_VALUES:
        return None
    low, high = bounds("iqr", summary)
    count = 0
    if low is not None and high is not None:
        (count,) = source.one(
            f"SELECT count(*) FROM {source.numbers} WHERE x < {low!r} OR x > {high!r}"
        )
    share = int(count) / summary.count
    fires = share > OUTLIER_SHARE
    percent = f"{100 * share:.1f} %"
    if fires:
        reason = (
            f"{percent} of values are outliers by the IQR rule (above 1 %): the mean and the "
            "standard deviation are pulled by them."
        )
    else:
        reason = f"{percent} of values are outliers by the IQR rule (1 % or less)."
    found = Robust(
        fires,
        reason,
        int(count),
        share,
        summary.median,
        summary.mad * MAD_TO_SIGMA if summary.mad is not None else None,
        summary.iqr / IQR_TO_SIGMA if summary.iqr is not None else None,
        None,
        None,
        None,
        0,
    )
    if not summary.mad:
        found.note = (
            "The MAD is 0 (more than half the values are equal): Huber's estimate needs a "
            "spread, the median is the robust center."
        )
        return found
    values = np.asarray(source.sample(SAMPLE), dtype=float)
    found.sample = len(values)
    center, scale = _huber(values, summary.mad * MAD_TO_SIGMA)
    if center is None:
        found.note = "Huber's estimate did not converge: the median is the robust center."
        return found
    if scale is None:
        found.note = (
            "Huber's center with the scale held at the MAD (the joint estimate did not converge)."
        )
        scale = summary.mad * MAD_TO_SIGMA
    found.huber_center, found.huber_scale = center, clean(scale)
    if found.huber_center and summary.mean is not None:
        found.mean_shift = (summary.mean - found.huber_center) / abs(found.huber_center)
    return found
