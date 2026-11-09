"""Parts of a date (year, month, day of week...) as virtual columns, and date granularity.

See docs/DATES.md. Every part is a SQL expression over the column's typed value, so it
always agrees with the column's parser.
"""

from dataclasses import dataclass
from typing import Any, Literal

import duckdb

DatePart = Literal[
    "year", "quarter", "month", "year_month", "week", "day_of_week", "day_of_month", "hour"
]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass(frozen=True)
class PartSpec:
    label: str
    sql: str  # template on {t}
    # Fixed category labels for cyclic parts, indexed from the SQL value (1-based or 0-based).
    labels: tuple[str, ...] | None = None
    first: int = 1


PARTS: dict[DatePart, PartSpec] = {
    "year": PartSpec("Year", "CAST(year({t}) AS INTEGER)"),
    "quarter": PartSpec("Quarter", "quarter({t})", ("Q1", "Q2", "Q3", "Q4")),
    "month": PartSpec("Month", "month({t})", tuple(MONTHS)),
    "year_month": PartSpec("Year-month", "strftime({t}, '%Y-%m')"),
    "week": PartSpec("Week", "weekofyear({t})"),
    "day_of_week": PartSpec("Day of week", "isodow({t})", tuple(WEEKDAYS)),
    "day_of_month": PartSpec("Day of month", "day({t})"),
    "hour": PartSpec("Hour", "hour(CAST({t} AS TIMESTAMP))", tuple(str(h) for h in range(24)), 0),
}

TIMELINE_STEPS = ("hour", "day", "week", "month", "quarter", "year")
STEP_SECONDS = {
    "hour": 3_600,
    "day": 86_400,
    "week": 7 * 86_400,
    "month": 30.44 * 86_400,
    "quarter": 91.3 * 86_400,
    "year": 365.25 * 86_400,
}
STEP_FORMATS = {
    "hour": "%Y-%m-%d %H:00",
    "day": "%Y-%m-%d",
    "week": "%Y-%m-%d",
    "month": "%Y-%m",
    "quarter": "%Y-%m",
    "year": "%Y",
}
MAX_PERIODS = 120
# True when a value has a time of day (not midnight).
HAS_TIME = "CAST(t AS TIMESTAMP) <> CAST(CAST(t AS DATE) AS TIMESTAMP)"


def part_sql(typed: str, part: DatePart) -> str:
    return PARTS[part].sql.format(t=typed)


def part_label(part: DatePart, value: Any) -> str:
    spec = PARTS[part]
    if spec.labels is not None and isinstance(value, int):
        index = value - spec.first
        if 0 <= index < len(spec.labels):
            return spec.labels[index]
    return str(value)


def classify_step(seconds: float | None) -> str:
    """Typical gap between consecutive distinct dates → human frequency."""
    if seconds is None:
        return "single"
    for name, limit in (
        ("sub-hourly", 3_000),
        ("hourly", 5_400),
        ("daily", 1.5 * 86_400),
        ("weekly", 8 * 86_400),
        ("monthly", 35 * 86_400),
        ("quarterly", 100 * 86_400),
        ("yearly", 400 * 86_400),
    ):
        if seconds <= limit:
            return name
    return "irregular"


def date_profile(
    connection: duckdb.DuckDBPyConnection, source: str, typed: str, year_only: bool = False
) -> dict[str, Any]:
    """Range, presence of a time of day, typical step and the parts worth offering."""
    low, high, timed = connection.execute(
        f"SELECT min(t), max(t), count(*) FILTER (WHERE {HAS_TIME}) "
        f"FROM (SELECT {typed} AS t FROM {source}) WHERE t IS NOT NULL"
    ).fetchone() or (None, None, 0)
    gap = connection.execute(
        "SELECT median(d) FROM (SELECT epoch(t) - lag(epoch(t)) OVER (ORDER BY t) AS d "
        f"FROM (SELECT DISTINCT {typed} AS t FROM {source} WHERE {typed} IS NOT NULL))"
    ).fetchone()
    has_time = bool(timed)
    if year_only:
        parts: list[DatePart] = ["year"]
    else:
        parts = ["year", "quarter", "month", "year_month", "week", "day_of_week", "day_of_month"]
        if has_time:
            parts.append("hour")
    return {
        "has_time": has_time,
        "min": str(low) if low is not None else None,
        "max": str(high) if high is not None else None,
        "step": classify_step(float(gap[0]) if gap and gap[0] is not None else None),
        "parts": parts,
    }


def auto_step(range_seconds: float, has_time: bool, year_only: bool = False) -> str:
    """Smallest timeline step keeping at most MAX_PERIODS periods."""
    if year_only:
        return "year"
    for step in TIMELINE_STEPS:
        if step == "hour" and not has_time:
            continue
        if range_seconds / STEP_SECONDS[step] <= MAX_PERIODS:
            return step
    return "year"
