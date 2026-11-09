"""Seasonal profiles: the series by weekday, month, quarter or hour, and year by year."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from scipy import stats

from quantilo.timeseries.methods import ALPHA, MethodResult
from quantilo.variable.source import clean

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PARSE = {
    "hour": "%Y-%m-%d %H:00",
    "day": "%Y-%m-%d",
    "week": "%Y-%m-%d",
    "month": "%Y-%m",
    "quarter": "%Y-%m",
    "year": "%Y",
}
# Calendar profiles worth drawing for each period.
PROFILES = {
    "hour": ("hour", "weekday"),
    "day": ("weekday", "month"),
    "week": ("month",),
    "month": ("month",),
    "quarter": ("quarter",),
    "year": (),
}


@dataclass
class SeasonalProfile:
    key: str  # weekday, month, quarter, hour
    labels: list[str]
    values: list[list[float]]  # values of the series per label (for boxes)
    deviation: list[float | None]  # % above / below the average level, trend removed
    test: MethodResult  # Kruskal-Wallis between labels, trend removed
    strongest: str | None = None  # the label furthest from the average


@dataclass
class YearLine:
    year: int
    x: list[int]  # month, ISO week or quarter number
    y: list[float | None]


@dataclass
class YearOverlay:
    unit: str  # month, week, quarter
    years: list[YearLine] = field(default_factory=list)


def starts(periods: list[str], period: str) -> list[datetime]:
    return [datetime.strptime(p, PARSE[period]) for p in periods]


def period_end(start: datetime, period: str) -> datetime:
    """The last second of the period that begins at `start`."""
    if period == "hour":
        following = start + timedelta(hours=1)
    elif period == "day":
        following = start + timedelta(days=1)
    elif period == "week":
        following = start + timedelta(days=7)
    else:
        months = {"month": 1, "quarter": 3, "year": 12}[period]
        total = start.month - 1 + months
        following = start.replace(year=start.year + total // 12, month=total % 12 + 1)
    return following - timedelta(seconds=1)


def _key(moment: datetime, profile: str) -> int:
    if profile == "weekday":
        return moment.weekday()
    if profile == "month":
        return moment.month - 1
    if profile == "quarter":
        return (moment.month - 1) // 3
    return moment.hour


def _labels(profile: str) -> list[str]:
    if profile == "weekday":
        return WEEKDAYS
    if profile == "month":
        return MONTHS
    if profile == "quarter":
        return ["Q1", "Q2", "Q3", "Q4"]
    return [f"{h:02d}:00" for h in range(24)]


def profile(
    key: str,
    moments: list[datetime],
    values: np.ndarray,
    detrended: np.ndarray,
) -> SeasonalProfile:
    """Values and trend-free deviations per calendar label, with a Kruskal-Wallis test."""
    labels = _labels(key)
    raw: list[list[float]] = [[] for _ in labels]
    free: list[list[float]] = [[] for _ in labels]
    for moment, value, rest in zip(moments, values, detrended, strict=True):
        index = _key(moment, key)
        raw[index].append(float(value))
        free[index].append(float(rest))
    level = float(np.mean(np.abs(values))) or 1.0
    deviation = [clean(np.mean(group) / level * 100) if group else None for group in free]
    used = [np.array(group) for group in free if len(group) >= 2]
    n = sum(len(group) for group in used)
    if len(used) < 2 or all(np.ptp(group) == 0 for group in used):
        test = MethodResult("kruskal_season", None, None, n, None, note="Too few values per label.")
    else:
        h, p = (float(x) for x in np.asarray(stats.kruskal(*used), dtype=float)[:2])
        k = len(used)
        epsilon = max(0.0, (h - k + 1) / (n - k)) if n > k else None
        test = MethodResult(
            "kruskal_season", h, clean(p), n, clean(p) is not None and p < ALPHA, epsilon
        )
    known = [(abs(d), labels[i]) for i, d in enumerate(deviation) if d is not None]
    strongest = max(known)[1] if known else None
    present = [i for i, group in enumerate(raw) if group]
    return SeasonalProfile(
        key,
        [labels[i] for i in present],
        [raw[i] for i in present],
        [deviation[i] for i in present],
        test,
        strongest,
    )


def profiles(
    period: str, moments: list[datetime], values: np.ndarray, detrended: np.ndarray
) -> list[SeasonalProfile]:
    """The calendar profiles that fit the period and the span (a month profile needs a year)."""
    span_days = (moments[-1] - moments[0]).days if moments else 0
    found = []
    for key in PROFILES.get(period, ()):
        if key == "month" and span_days < 360:
            continue
        if key == "weekday" and period == "hour" and span_days < 14:
            continue
        found.append(profile(key, moments, values, detrended))
    return found


def year_overlay(
    period: str, moments: list[datetime], values: list[float | None]
) -> YearOverlay | None:
    """One line per year (by month, ISO week or quarter), when the data covers two years."""
    unit = {"day": "week", "week": "week", "month": "month", "quarter": "quarter"}.get(period)
    if unit is None or len({m.year for m in moments}) < 2:
        return None
    buckets: dict[int, dict[int, list[float]]] = {}
    for moment, value in zip(moments, values, strict=True):
        if value is None:
            continue
        if unit == "week":
            year, position = moment.isocalendar()[0], moment.isocalendar()[1]
        elif unit == "month":
            year, position = moment.year, moment.month
        else:
            year, position = moment.year, (moment.month - 1) // 3 + 1
        buckets.setdefault(year, {}).setdefault(position, []).append(value)
    overlay = YearOverlay(unit)
    positions_per_year = max(len(b) for b in buckets.values())
    for year in sorted(buckets):
        if len(buckets[year]) < max(2, positions_per_year // 10):
            continue  # a year with one or two points (an edge week) says nothing
        positions = sorted(buckets[year])
        overlay.years.append(
            YearLine(year, positions, [clean(np.mean(buckets[year][p])) for p in positions])
        )
    return overlay if len(overlay.years) >= 2 else None
