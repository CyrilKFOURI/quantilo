"""Turn dated rows into a series: one aggregated value per period, then rolling statistics.

A file often holds several rows per date (the sales of ten stores on the same day); a rolling mean
over rows would mean nothing, so rows are always aggregated per period first.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Period = Literal["hour", "day", "week", "month", "quarter", "year"]
Aggregate = Literal["sum", "mean", "median", "min", "max", "count"]
Align = Literal["trailing", "centered"]

PERIODS: tuple[Period, ...] = ("hour", "day", "week", "month", "quarter", "year")
AGGREGATES: tuple[Aggregate, ...] = ("sum", "mean", "median", "min", "max", "count")
PERIOD_SECONDS: dict[Period, float] = {
    "hour": 3_600,
    "day": 86_400,
    "week": 7 * 86_400,
    "month": 30.44 * 86_400,
    "quarter": 91.31 * 86_400,
    "year": 365.25 * 86_400,
}
PER_YEAR: dict[Period, float] = {
    "hour": 8_766,
    "day": 365.25,
    "week": 52.18,
    "month": 12,
    "quarter": 4,
    "year": 1,
}
INTERVALS: dict[Period, str] = {
    "hour": "1 hour",
    "day": "1 day",
    "week": "7 days",
    "month": "1 month",
    "quarter": "3 months",
    "year": "1 year",
}
FORMATS: dict[Period, str] = {
    "hour": "%Y-%m-%d %H:00",
    "day": "%Y-%m-%d",
    "week": "%Y-%m-%d",
    "month": "%Y-%m",
    "quarter": "%Y-%m",
    "year": "%Y",
}
AGGREGATE_SQL: dict[Aggregate, str] = {
    "sum": "sum(v)",
    "mean": "avg(v)",
    "median": "median(v)",
    "min": "min(v)",
    "max": "max(v)",
    "count": "count(*)",
}
DEFAULT_WINDOW: dict[Period, int] = {
    "hour": 24,
    "day": 7,
    "week": 4,
    "month": 3,
    "quarter": 4,
    "year": 3,
}
# Amounts and volumes add up over a period; prices, ages or ratings are averaged.
SUM_ROLES = frozenset({"amount", "quantity"})
AUTO_MAX_PERIODS = 400  # an automatic period is coarsened beyond this many periods
MAX_PERIODS = 2_000  # a chosen period is coarsened beyond this many (points drawn per series)
MIN_PERIODS = 8  # fewer periods: no rolling statistics, no trend test


@dataclass
class Options:
    period: Period | Literal["auto"] = "auto"
    aggregate: Aggregate | Literal["auto"] = "auto"
    window: int | None = None
    align: Align = "trailing"


def default_aggregate(role: str | None) -> Aggregate:
    return "sum" if role in SUM_ROLES else "mean"


def typical_period(median_gap: float, has_time: bool) -> Period:
    """The period matching the usual gap between two consecutive distinct dates."""
    for period, limit in (("hour", 1.5 * 3_600), ("day", 1.5 * 86_400), ("week", 10 * 86_400)):
        if median_gap <= limit and (period != "hour" or has_time):
            return period  # type: ignore[return-value]
    if median_gap <= 45 * 86_400:
        return "month"
    if median_gap <= 120 * 86_400:
        return "quarter"
    return "year"


def choose_period(
    requested: Period | Literal["auto"], typical: Period, span: float
) -> tuple[Period, bool]:
    """The period to use, and whether it was coarsened to keep the series readable."""
    start = typical if requested == "auto" else requested
    limit = AUTO_MAX_PERIODS if requested == "auto" else MAX_PERIODS
    for period in PERIODS[PERIODS.index(start) :]:
        if span / PERIOD_SECONDS[period] <= limit:
            return period, period != start
    return "year", start != "year"


def default_window(period: Period, periods: int) -> int:
    """Rolling window in periods: a natural cycle, never more than a quarter of the series."""
    return max(2, min(DEFAULT_WINDOW[period], periods // 4))


def resolve_window(requested: int | None, period: Period, periods: int) -> int:
    """The window asked for (or the default), kept between 2 and the series length."""
    return max(2, min(requested or default_window(period, periods), periods))


def rolling(
    values: list[float | None], window: int, align: Align
) -> tuple[list[float | None], list[float | None]]:
    """Rolling mean and sample standard deviation; gaps are skipped, half a window is required.

    Matches pandas' `rolling(window, min_periods=ceil(window / 2), center=...)`.
    """
    data = np.array([np.nan if v is None else v for v in values], dtype=float)
    needed = math.ceil(window / 2)
    means: list[float | None] = []
    stds: list[float | None] = []
    for i in range(len(data)):
        if align == "centered":
            start = i - window // 2
            end = start + window
        else:
            start, end = i - window + 1, i + 1
        chunk = data[max(0, start) : max(0, end)]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) < needed:
            means.append(None)
            stds.append(None)
            continue
        means.append(float(chunk.mean()))
        stds.append(float(chunk.std(ddof=1)) if len(chunk) > 1 else None)
    return means, stds


Gaps = Literal["leave", "interpolate", "zero", "carry"]
GAPS: tuple[Gaps, ...] = ("leave", "interpolate", "zero", "carry")
# One full cycle, in periods: a week of days, a year of weeks or months…
SEASON: dict[Period, int] = {"hour": 24, "day": 7, "week": 52, "month": 12, "quarter": 4}


def season_length(period: Period) -> int | None:
    return SEASON.get(period)


def fill_gaps(values: list[float | None], how: Gaps) -> list[float | None]:
    """Empty periods left as they are, linearly interpolated, set to 0 or carried forward."""
    if how == "leave":
        return list(values)
    if how == "zero":
        return [0.0 if v is None else v for v in values]
    filled: list[float | None] = list(values)
    if how == "carry":
        last: float | None = None
        for i, v in enumerate(filled):
            if v is None:
                filled[i] = last
            else:
                last = v
        return filled
    known = [i for i, v in enumerate(values) if v is not None]
    if len(known) < 2:
        return filled
    points = np.array([values[i] for i in known], dtype=float)
    interpolated = np.interp(np.arange(known[0], known[-1] + 1), known, points)
    for offset, value in enumerate(interpolated):
        filled[known[0] + offset] = float(value)
    return filled


def complete(values: list[float | None]) -> tuple[np.ndarray, int]:
    """A series without holes for methods that need one: interpolated inside, nearest at the
    edges. Returns the values and how many periods were filled."""
    missing = sum(v is None for v in values)
    filled = fill_gaps(values, "interpolate")
    known = [v for v in filled if v is not None]
    if not known:
        return np.array([], dtype=float), missing
    first = next(v for v in filled if v is not None)
    last = next(v for v in reversed(filled) if v is not None)
    head = True
    out: list[float] = []
    for v in filled:
        if v is None:
            out.append(first if head else last)
        else:
            head = False
            out.append(v)
    return np.array(out, dtype=float), missing
