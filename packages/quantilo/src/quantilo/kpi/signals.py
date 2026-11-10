"""Signals of a KPI (054): what deserves attention or care, each with its rule and its reason.

Read from the already aggregated result: the total series of each measure (unusual periods,
level shifts, R-SIG-01 · 02), the counts behind each cell (low n, thin periods, empty measure,
R-SIG-03 · 04 · 05) and one fingerprint count for repeated order lines (R-SIG-06). The mix
effect and Simpson's paradox (R-SIG-07 · 08) are in `kpi/mix.py`. Pure numpy and statsmodels, no
thread: runs in the browser too.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from quantilo.kpi.definition import Grain

Kind = Literal[
    "anomaly",
    "level_shift",
    "low_n",
    "thin_period",
    "empty_measure",
    "duplicates",
    "mix",
    "simpson",
]
Severity = Literal["info", "warning"]
# Where it shows: the Why line, the chart's markers and the table (series), a sentence under the
# chart (change), under the tiles (measure), icons in the table and a summary (cells).
Scope = Literal["series", "change", "measure", "cells"]
SCOPES: tuple[Scope, ...] = ("series", "change", "measure", "cells")

MIN_PERIODS = 12  # R-SIG-01 · 02: complete periods with a value
ANOMALY_Z = 3.5
MAX_ANOMALIES = 5
ROWS_NOTE = 0.15  # an unusual sum with 15 % more (fewer) rows says so
MAX_SHIFTS = 2
SHIFT_JUMP = 0.05  # under the trend cost, a jump of 5 % or more between two lines is a shift
MATERIAL = 0.05  # an anomaly or a shift moves the series by 5 % of its typical level at least
BEND = 0.1  # a change of trend moves the end of the series by 10 % of its level at least
MAX_SHIFT_PERIODS = 400  # PELT is quadratic at worst: beyond, no level shift (spec 054)
SEASONS: dict[Grain, int] = {"day": 7, "week": 52, "month": 12, "quarter": 4}
LOW_N = 30  # R-SIG-03
THIN = 0.2  # R-SIG-04: share of the median period's rows
EMPTY = 0.2  # R-SIG-05
DUPLICATES = 0.01  # R-SIG-06
MAX_CELLS = 1_000  # cells listed by one signal


@dataclass
class SignalCell:
    period: str | None  # ISO date of the period's first day; None = the whole range
    g1: str | None = None  # None = every label of the split (a total)
    g2: str | None = None
    value: float | None = None  # what the rule read there: n, robust z…


@dataclass
class Signal:
    kind: Kind
    severity: Severity
    scope: Scope
    rule: str
    measure: str | None  # the measure's id (the ratio's too), None for every measure
    message: str
    cells: list[SignalCell] = field(default_factory=list)
    score: float | None = None  # the signal's strength in its own unit


class Period(Protocol):
    start: str
    label: str
    partial: bool


def number(value: float) -> str:
    """A number in a sentence: 3 significant digits, K, M or B beyond a thousand."""
    size = abs(value)
    for step, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if size >= step:
            return f"{value / step:.3g}{suffix}"
    return f"{value:.3g}"


def percent(value: float, decimals: int = 0) -> str:
    return f"{value * 100:.{decimals}f} %"


def ordered(signals: list[Signal]) -> list[Signal]:
    """Series first, then changes, measures and cells; the measures' order kept inside."""
    return sorted(signals, key=lambda s: SCOPES.index(s.scope))


# --- P12: the series --------------------------------------------------------------------------


def _complete(
    values: Sequence[float | None], periods: Sequence[Period]
) -> tuple[list[int], np.ndarray, bool]:
    """Indexes and values of the complete periods with a value, and whether they follow each
    other (no complete period without a value between the first and the last)."""
    complete = [i for i, p in enumerate(periods) if not p.partial]
    known = [
        i
        for i in complete
        if values[i] is not None and math.isfinite(values[i])  # type: ignore[arg-type]
    ]
    if not known:
        return [], np.array([]), False
    span = [i for i in complete if known[0] <= i <= known[-1]]
    series = np.asarray([values[i] for i in known], dtype=float)
    return known, series, len(span) == len(known)


def _moving_median(series: np.ndarray, width: int) -> np.ndarray:
    """Centred on full windows only; the edges keep the nearest full window's median (padding
    the series itself would bend a seasonal edge)."""
    width = min(width + 1 - width % 2, len(series) - (1 - len(series) % 2))  # odd, centred
    half = width // 2
    inner = np.median(np.lib.stride_tricks.sliding_window_view(series, width), axis=1)
    return np.pad(inner, half, mode="edge")


def decompose(series: np.ndarray, season: int | None) -> tuple[np.ndarray, np.ndarray]:
    """A robust remainder and the series without its seasonality. With two cycles: the trend is
    a moving median over one cycle and each position of the cycle has its median deviation as
    seasonality; otherwise the trend is a moving median over 5 periods and nothing is seasonal.
    Medians rather than STL: with 2 or 3 cycles, STL folds a lone spike into the seasonality
    of its month and the spike vanishes from the remainder (054, R-SIG-01)."""
    n = len(series)
    if season and season >= 2 and n >= 2 * season:
        trend = _moving_median(series, season)
        detrended = series - trend
        positions = np.arange(n) % season
        profile = np.array([np.median(detrended[positions == p]) for p in range(season)])
        seasonal = profile[positions] - float(np.median(profile))
        return detrended - seasonal + float(np.median(profile)), series - seasonal
    return series - _moving_median(series, 5), series


def _rows_note(rows: Sequence[int] | None, known: list[int], index: int, side: str) -> str:
    """ ", with 25 % more rows than the median period" when the rows explain the value."""
    if not rows:
        return ""
    median = float(np.median([rows[i] for i in known]))
    if median <= 0:
        return ""
    gap = rows[index] / median - 1
    if abs(gap) < ROWS_NOTE or (gap > 0) != (side == "high"):
        return ""
    if rows[index] == 0:
        return ", with no rows"
    return f", with {abs(gap) * 100:.0f} % {'more' if gap > 0 else 'fewer'} rows than usual"


def series_signals(
    measure: str,
    label: str | None,
    values: Sequence[float | None],
    periods: Sequence[Period],
    grain: Grain,
    rows: Sequence[int] | None = None,
) -> list[Signal]:
    """R-SIG-01 unusual periods and R-SIG-02 level shifts of one total series. `label` prefixes
    the messages when the KPI has several measures. With `rows` (a measure that adds up), an
    unusual period that also has many more or fewer rows than usual says so: a month of five
    weeks is high because it is longer."""
    known, series, contiguous = _complete(values, periods)
    if len(known) < MIN_PERIODS or np.ptp(series) == 0:
        return []
    from quantilo.timeseries import breaks
    from quantilo.timeseries import methods as ts_methods

    season = SEASONS.get(grain) if contiguous else None
    head = f"{label}: " if label else ""
    out: list[Signal] = []
    remainder, level = decompose(series, season)
    # A near-perfect fit makes any wiggle a high robust z: an anomaly must also be material.
    typical = float(np.median(np.abs(series))) or 1.0
    # A median leaves its own point at exactly 0: those points are left out of the robust scale
    # (otherwise the MAD shrinks and ordinary noise looks unusual), and are never unusual.
    free = np.flatnonzero(np.abs(remainder) > 1e-9 * typical)
    unusual = [
        point
        for point in breaks.anomalies(remainder[free], series[free], "robust_z", ANOMALY_Z, season)
        if abs(point.value - point.expected) >= MATERIAL * typical
    ]
    for point in unusual[:MAX_ANOMALIES]:
        period = periods[known[int(free[point.index])]]
        side = "high" if point.value > point.expected else "low"
        why = _rows_note(rows, known, known[int(free[point.index])], side)
        out.append(
            Signal(
                "anomaly",
                "info",
                "series",
                "R-SIG-01",
                measure,
                f"{head}{period.label} unusually {side}{why} "
                f"(robust z {point.score:.1f}, R-SIG-01)",
                [SignalCell(period.start, None, None, round(point.score, 2))],
                round(point.score, 2),
            )
        )
    # Without two cycles a seasonal peak looks like a shift: no shift is read then.
    cycle = SEASONS.get(grain)
    if len(known) > MAX_SHIFT_PERIODS or (cycle and (not season or len(known) < 2 * cycle)):
        return out
    trending = bool(ts_methods.mann_kendall_test([float(v) for v in level]).significant)
    cost: breaks.Cost = "trend" if trending else "mean"
    found = breaks.find_breaks(level, "pelt", cost, "medium", season)
    shifts = list(zip(found.points, found.segments, found.segments[1:], strict=False))
    kept: list[Signal] = []
    for point, before, after in shifts:
        period = periods[known[point]]
        # Under the trend cost a break is a jump between two lines, or a bend of one line.
        old, new = (before.last, after.first) if cost == "trend" else (before.mean, after.mean)
        change = (new - old) / abs(old) if old else None
        jump = abs(new - old) / typical
        if cost == "trend" and jump < SHIFT_JUMP:
            bend = abs(after.slope - before.slope) * (after.end - after.start) / typical
            if bend < BEND:
                continue
            text = f"change of trend since {period.label} (R-SIG-02)"
        elif jump < MATERIAL:
            continue
        else:
            size = f"{change * 100:+.0f} %" if change is not None else number(new)
            text = f"level shift since {period.label} ({size}, R-SIG-02)"
        kept.append(
            Signal(
                "level_shift",
                "info",
                "series",
                "R-SIG-02",
                measure,
                head + text,
                [SignalCell(period.start, None, None, None if change is None else change)],
                None if change is None else round(change, 4),
            )
        )
    return out + kept[-MAX_SHIFTS:]


# --- P13: quality -----------------------------------------------------------------------------


def low_n(measure: str, label: str, cells: Sequence[SignalCell]) -> Signal | None:
    """R-SIG-03: cells of a mean, a rate or a ratio resting on fewer than 30 values."""
    low = [c for c in cells if c.value is not None and 0 < c.value < LOW_N]
    if not low:
        return None
    count = len(low)
    what = "cell" if count == 1 else "cells"
    verb = "rests" if count == 1 else "rest"
    return Signal(
        "low_n",
        "warning",
        "cells",
        "R-SIG-03",
        measure,
        f"{count:,} {what} of {label} {verb} on fewer than {LOW_N} values: read them with care "
        "(R-SIG-03).",
        low[:MAX_CELLS],
        float(count),
    )


def thin_periods(periods: Sequence[Period], rows: Sequence[int]) -> Signal | None:
    """R-SIG-04: complete periods under 20 % of the median complete period's rows. An empty
    period is not thin: it shows as empty already (no rows on a Sunday)."""
    complete = [i for i, p in enumerate(periods) if not p.partial and rows[i] > 0]
    if len(complete) < 3:
        return None
    median = float(np.median([rows[i] for i in complete]))
    thin = [i for i in complete if rows[i] < THIN * median]
    if not thin or median <= 0:
        return None
    names = [periods[i].label for i in thin]
    listed = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
    verb = "has" if len(thin) == 1 else "have"
    return Signal(
        "thin_period",
        "warning",
        "cells",
        "R-SIG-04",
        None,
        f"{listed} {verb} under {percent(THIN)} of the median period's rows ({median:,.0f}): "
        "a value can move a lot by chance there (R-SIG-04).",
        [SignalCell(periods[i].start, None, None, float(rows[i])) for i in thin][:MAX_CELLS],
        len(thin) / len(complete),
    )


def empty_measure(measure: str, label: str, filled: int, rows: int) -> Signal | None:
    """R-SIG-05: the measure's column is empty on more than 20 % of the rows."""
    if rows <= 0:
        return None
    empty = 1 - filled / rows
    if empty <= EMPTY:
        return None
    return Signal(
        "empty_measure",
        "warning",
        "measure",
        "R-SIG-05",
        measure,
        f"{label} has no value on {percent(empty)} of the rows: the KPI reads the other "
        f"{percent(1 - empty)} (R-SIG-05).",
        [],
        round(empty, 4),
    )


def duplicates(key: str, copies: int, rows: int) -> Signal | None:
    """R-SIG-06: rows identical to another row of the same order, counted twice by a sum."""
    if rows <= 0 or copies / rows <= DUPLICATES:
        return None
    share = copies / rows
    return Signal(
        "duplicates",
        "warning",
        "measure",
        "R-SIG-06",
        None,
        f"{copies:,} rows ({percent(share, 1)}) are exact copies of another row of the same "
        f"{key}: sums count them twice (R-SIG-06).",
        [],
        round(share, 4),
    )
