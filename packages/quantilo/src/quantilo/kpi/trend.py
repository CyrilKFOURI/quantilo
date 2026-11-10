"""Show as "Trend" (053, R-SHOW-02): a smoothed trend of each series, read for its shape.

Robust STL when two full cycles fit and no complete period is empty, LOWESS otherwise. The fit
uses the complete periods only: an incomplete one (the current month) gets no trend. Neither is
causal: the trend at a period reads the periods after it, so the last values move when new data
comes in, and the reason says so. Pure numpy and statsmodels, no thread: runs in the browser too.
"""

import math
import warnings
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from quantilo.kpi.definition import Grain
from quantilo.kpi.rules import Decision

SEASONS: dict[Grain, int] = {"day": 7, "week": 52, "month": 12, "quarter": 4}
MIN_PERIODS = 4
FRACTION = 2 / 3  # Cleveland's default span
CAVEAT = "it reads later periods, so the last ones may change"


class Series(Protocol):
    values: list[float | None]
    trend: list[float | None] | None


def _cycle(grain: Grain) -> str:
    return {"day": "7-day", "week": "52-week", "month": "12-month", "quarter": "4-quarter"}.get(
        grain, ""
    )


def _lowess(x: np.ndarray, y: np.ndarray, at: np.ndarray) -> np.ndarray:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = lowess(y, x, frac=FRACTION, it=1, xvals=at)
    return np.asarray(fitted, dtype=float)


def trend(
    values: Sequence[float | None], partial: Sequence[bool], grain: Grain
) -> tuple[list[float | None], Decision]:
    """The trend of one series (None where there is none) and the method's decision."""
    out: list[float | None] = [None] * len(values)
    complete = [i for i, p in enumerate(partial) if not p]
    known = [i for i in complete if values[i] is not None and math.isfinite(values[i])]  # type: ignore[arg-type]
    if len(known) < MIN_PERIODS:
        return out, Decision(
            "none",
            "R-SHOW-02",
            f"No trend: it needs at least {MIN_PERIODS} complete periods with a value.",
            False,
        )
    first, last = known[0], known[-1]
    span = [i for i in complete if first <= i <= last]
    season = SEASONS.get(grain)
    full = len(known) == len(span)
    if season and full and len(span) >= 2 * season:
        from quantilo.timeseries.decomposition import stl

        series = np.asarray([values[i] for i in span], dtype=float)
        fitted = stl(series, season).trend
        method, reason = (
            "stl",
            f"Trend: robust STL over {_cycle(grain)} cycles ({len(span) // season} cycles of "
            f"data); {CAVEAT} (R-SHOW-02).",
        )
    else:
        x = np.asarray(known, dtype=float)
        y = np.asarray([values[i] for i in known], dtype=float)
        fitted = _lowess(x, y, np.asarray(span, dtype=float))
        why = (
            f"fewer than two {_cycle(grain)} cycles"
            if season and len(span) < 2 * season
            else "empty periods in the range"
            if season and not full
            else "no seasonal cycle at this grain"
        )
        method, reason = "lowess", f"Trend: LOWESS ({why}); {CAVEAT} (R-SHOW-02)."
    for index, value in zip(span, fitted, strict=True):
        out[index] = float(value) if math.isfinite(value) else None
    return out, Decision(method, "R-SHOW-02", reason, False)


def attach(series: Sequence[Series], partial: Sequence[bool], grain: Grain) -> Decision:
    """Sets `trend` on every series; the decision is the total's (the first series)."""
    decisions = []
    for line in series:
        fitted, decision = trend(line.values, partial, grain)
        line.trend = fitted
        decisions.append(decision)
    return decisions[0] if decisions else trend([], [], grain)[1]
