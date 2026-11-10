"""The current period to date, compared at an equal date (054, P9: R-TILE-03 · 04).

"Nov 1-10 vs Oct 1-10 and vs Nov 1-10 last year": the started period is measured from its
first day to the last date of the data, and compared with the same length of time from the start
of the previous period and of the same period last year (each cut at its own period's end). One
query on the KPI's rows, for the totals only.
"""

import calendar
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time, timedelta

import duckdb

from quantilo.kpi import query as q
from quantilo.kpi.definition import Grain

GRAINS: tuple[Grain, ...] = ("week", "month", "quarter", "year")
DAY = 86_400.0
DASH = chr(0x2013)  # an en dash between two days: "Nov 1-10"


@dataclass
class ToDate:
    period: str  # the started period ("Nov 2026")
    label: str  # "Nov 1-10"
    value: float | None
    previous_label: str  # "Oct 1-10"
    previous: float | None
    change: float | None
    last_year_label: str  # "Nov 1-10 last year"
    last_year: float | None
    change_yoy: float | None
    projected: float | None  # additive measures only (R-TILE-04)
    elapsed: float  # share of the period gone, 0 to 1


@dataclass
class Window:
    start: datetime
    end: datetime  # excluded


def _add_months(moment: datetime, months: int) -> datetime:
    month = moment.month - 1 + months
    year = moment.year + month // 12
    month = month % 12 + 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def shift(start: datetime, grain: Grain, steps: int) -> datetime:
    if grain == "day":
        return start + timedelta(days=steps)
    if grain == "week":
        return start + timedelta(days=7 * steps)
    return _add_months(start, steps * {"month": 1, "quarter": 3, "year": 12}[grain])


def span_label(window: Window) -> str:
    """ "Nov 1-10" inside one month, "Oct 1 - Nov 10" across months."""
    first, last = window.start, window.end - timedelta(microseconds=1)
    if (first.year, first.month) == (last.year, last.month):
        days = f"{first.day}" if first.day == last.day else f"{first.day}{DASH}{last.day}"
        return f"{first:%b} {days}"
    return f"{first:%b} {first.day} {DASH} {last:%b} {last.day}"


def windows(
    start: datetime, last: datetime, gap: float, grain: Grain, year_before: datetime
) -> tuple[Window, Window, Window, float]:
    """The started period's window, the previous one's and last year's, and the share gone.
    Dates without hours (a typical gap of a day or more) count their last step in full. Months,
    quarters and years end on the same date of the calendar (Nov 10 -> Oct 10, a leap year
    changes nothing); weeks on the same day of the week."""
    end = shift(start, grain, 1)
    cut = last + timedelta(seconds=gap) if gap >= DAY else last + timedelta(microseconds=1)
    cut = min(cut, end)
    elapsed = cut - start
    previous = shift(start, grain, -1)
    year_end = shift(year_before, grain, 1)
    if grain == "week":
        previous_cut, year_cut = previous + elapsed, year_before + elapsed
    else:
        previous_cut, year_cut = shift(cut, grain, -1), _add_months(cut, -12)
    return (
        Window(start, cut),
        Window(previous, min(max(previous_cut, previous), start)),
        Window(year_before, min(max(year_cut, year_before), year_end)),
        elapsed / (end - start),
    )


def _between(window: Window) -> str:
    return f"(t >= {q.timestamp(window.start)} AND t < {q.timestamp(window.end)})"


def values(
    connection: duckdb.DuckDBPyConnection,
    aggregates: list[str],
    spans: tuple[Window, Window, Window],
    data_first: datetime,
) -> list[tuple[float | None, ...] | None]:
    """Each window's aggregates; None for a window that starts before the data does."""
    # one select per window: at the year grain, the previous period is last year's
    selects = " UNION ALL ".join(
        f"SELECT {i} AS k, {', '.join(aggregates)}, count(*) FROM {q.TABLE} WHERE {_between(w)}"
        for i, w in enumerate(spans)
    )
    rows = [r for r in connection.execute(selects).fetchall() if r[-1]]
    found = {int(r[0]): tuple(None if v is None else float(v) for v in r[1:]) for r in rows}
    first = datetime.combine(data_first.date(), time()) if data_first else None
    return [
        None if first is not None and w.start < first else found.get(i, ())
        for i, w in enumerate(spans)
    ]


def change(value: float | None, before: float | None) -> float | None:
    if value is None or before is None or before == 0:
        return None
    return (value - before) / abs(before)


def to_date(
    period: str,
    spans: tuple[Window, Window, Window],
    elapsed: float,
    read: Callable[[int], float | None],
    additive: bool,
) -> ToDate:
    """One measure's figures; `read(i)` gives its value in window i (None without data)."""
    now, before, year = read(0), read(1), read(2)
    current, previous, last_year = spans
    return ToDate(
        period,
        span_label(current),
        now,
        span_label(previous),
        before,
        change(now, before),
        f"{span_label(last_year)} last year",
        year,
        change(now, year),
        now / elapsed if additive and now is not None and elapsed > 0 else None,
        round(elapsed, 4),
    )
