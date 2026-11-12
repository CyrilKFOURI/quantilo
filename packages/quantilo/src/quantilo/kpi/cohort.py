"""Cohorts and funnels (060, P6 and P7): who comes back, and how far a status goes.

Retention by cohort: each customer belongs to the period (month or week) of their first row; a
cohort's retention at offset k is the share of its customers seen again k periods later. The same
pass gives new and returning customers per period. Funnel: the distinct keys (or rows) that
reached each stage of an ordered status, a row at a later stage having passed the earlier ones.

Both are proposed by rules on the columns (R-COH-01, R-FUN-01, cheap: names, roles and the column
summary), and computed only when the user opens them (onion rule). One DuckDB connection, no
thread: the browser version runs it as is.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Protocol

import duckdb

from quantilo.compare.analyze import label_sql
from quantilo.errors import ExplorerError
from quantilo.ingestion.sql import lit
from quantilo.typing.knowledge_base import load
from quantilo.typing.parsers import typed_sql
from quantilo.typing.types import AnalysisType, Parser

CohortGrain = Literal["month", "week"]
REPEATED = 0.9  # a key repeats when it has fewer distinct values than this share of rows
MIN_PERIODS = 3  # R-COH-02
MIN_RETURNING = 0.2  # R-COH-02: share of keys seen in two periods or more
MONTHS_FOR_MONTHLY = 4  # R-COH-03: fewer months of data: weekly cohorts
MAX_COHORTS = 12  # R-COH-04: the latest cohorts shown, the older ones said
MIN_COHORT = 5  # R-COH-04: fewer customers give a share that jumps from 0 to 100 %
MAX_OFFSETS = 12
MIN_STAGES = 3  # R-FUN-01
KEY_TYPES = frozenset({"identifier", "categorical_nominal", "numeric_discrete"})
FUNNEL_TYPES = frozenset({"categorical_nominal", "categorical_ordinal"})
NEVER_DATES = frozenset({"birth", "birthday", "dob", "born", "naissance"})


class CohortError(ExplorerError):
    def __init__(self, message: str, code: str = "cohort") -> None:
        super().__init__(message, code=code, status_code=422)


class Column(Protocol):
    name: str
    type: AnalysisType
    parser: Parser
    role: str | None
    summary: dict[str, Any] | None


class Named(Protocol):
    name: str


# --- Proposals (cheap: no row is read) -----------------------------------------------------------


@dataclass
class Analysis:
    """A cohort or funnel the data allows, with the rule and its reason (060)."""

    kind: Literal["cohort", "funnel"]
    rule: str
    name: str
    reason: str
    customer: str | None = None  # cohort: the key and the date
    date: str | None = None
    column: str | None = None  # funnel: the status, its stages and the key counted
    stages: list[str] = field(default_factory=list)
    key: str | None = None


def _summary(column: Column) -> dict[str, Any]:
    return column.summary or {}


def _distinct(column: Column) -> int:
    return int(_summary(column).get("distinct_count") or 0)


def _valid(column: Column) -> int:
    summary = _summary(column)
    return int(summary.get("rows") or 0) - int(summary.get("null_count") or 0)


def _repeats(column: Column) -> bool:
    valid = _valid(column)
    return column.summary is None or (valid > 0 and 2 <= _distinct(column) < REPEATED * valid)


def _keys(columns: Sequence[Column], role: str) -> list[Column]:
    return [c for c in columns if c.role == role and c.type in KEY_TYPES and _repeats(c)]


def _spelling(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").replace("-", " ").split())


def _levels(column: Column) -> list[str]:
    """The most frequent values, from the column summary (8 at most)."""
    return [str(bar.get("label")) for bar in _summary(column).get("histogram") or []]


def stages_of(levels: Sequence[str]) -> tuple[list[str], list[str]] | None:
    """R-FUN-01: the stages of a known funnel these values hold (3 at least, in the funnel's
    order), and the funnel's whole list of stage names; None when no funnel fits."""
    spelled = {_spelling(level): level for level in levels}
    best: tuple[list[str], list[str]] | None = None
    for funnel in load().funnels:
        found = []
        for stage in funnel:
            level = next((spelled[word] for word in stage if word in spelled), None)
            if level is not None:
                found.append(level)
        if len(found) >= MIN_STAGES and (best is None or len(found) > len(best[0])):
            best = (found, [stage[0] for stage in funnel])
    return best


def propose(columns: Sequence[Column], date: Named | None) -> list[Analysis]:
    """R-COH-01 and R-FUN-01: what the columns allow, best first; `date` is the KPI's date."""
    found: list[Analysis] = []
    customers = _keys(columns, "customer_key")
    if customers and date is not None and not set(_spelling(date.name).split()) & NEVER_DATES:
        customer = customers[0]
        counted = (
            f" ({_distinct(customer):,} values over {_valid(customer):,} rows)"
            if customer.summary
            else ""
        )
        found.append(
            Analysis(
                "cohort",
                "R-COH-01",
                "Retention by cohort",
                f"R-COH-01 · {customer.name} names the customer and repeats{counted}, "
                f"{date.name} dates each row: customers grouped by the period of their first "
                "row, and the share seen again in each later period",
                customer=customer.name,
                date=date.name,
            )
        )
    # the funnel counts orders (or deals) once each, repeated or not
    orders = [c for c in columns if c.role == "order_key" and c.type in KEY_TYPES]
    key = orders[0] if orders else customers[0] if customers else None
    statuses = [c for c in columns if c.type in FUNNEL_TYPES and 3 <= _distinct(c) <= 50]
    statuses.sort(key=lambda c: c.role != "status")
    for column in statuses:
        matched = stages_of(_levels(column))
        if matched is None:
            continue
        stages, _ = matched
        counted = f"distinct {key.name}" if key else "rows"
        found.append(
            Analysis(
                "funnel",
                "R-FUN-01",
                f"Funnel of {column.name}",
                f"R-FUN-01 · {column.name} holds the stages {' → '.join(stages)} of a known "
                f"funnel: the {counted} that reached each stage (a later stage has passed the "
                "earlier ones), and the conversion from one to the next",
                column=column.name,
                stages=stages,
                key=key.name if key else None,
            )
        )
        break
    return found


# --- Retention by cohort -------------------------------------------------------------------------


@dataclass
class Cohort:
    start: str  # ISO date of the cohort's period
    label: str
    size: int  # customers whose first row falls in this period
    retained: list[float | None]  # share seen again at offset 0, 1, …; None once not observed


@dataclass
class PeriodCount:
    start: str
    label: str
    new: int
    returning: int
    partial: bool


@dataclass
class CohortResult:
    customer: str
    date: str
    grain: CohortGrain
    grain_reason: str
    cohorts: list[Cohort]
    periods: list[PeriodCount]  # new and returning customers in each period
    customers: int
    returning_share: float  # customers seen in two periods or more
    reason: str
    summary: str  # one sentence above the chart (Datawrapper style)
    warnings: list[str] = field(default_factory=list)


def _label(start: datetime, grain: CohortGrain) -> str:
    return start.strftime("%b %Y") if grain == "month" else f"Wk of {start:%d %b %Y}"


def _next(start: datetime, grain: CohortGrain) -> datetime:
    if grain == "week":
        return start + timedelta(days=7)
    month = start.month % 12 + 1
    return start.replace(year=start.year + (start.month == 12), month=month)


def _offset(first: datetime, later: datetime, grain: CohortGrain) -> int:
    if grain == "week":
        return (later - first).days // 7
    return (later.year - first.year) * 12 + later.month - first.month


def _percent(value: float) -> str:
    return f"{value:.0%}"


def retention(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    customer: Column,
    date: Column,
    grain: CohortGrain | Literal["auto"] = "auto",
) -> CohortResult:
    """Cohorts by the period of each customer's first row (R-COH-01 to 05)."""
    if date.type != "datetime":
        raise CohortError(f"{date.name} is not a date.", "wrong_type")
    key = typed_sql(customer.name, customer.parser)
    moment = f"CAST({typed_sql(date.name, date.parser)} AS TIMESTAMP)"
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE kpi_cohort_rows AS "
        f"SELECT CAST({key} AS VARCHAR) AS k, {moment} AS t FROM {source}"
    )
    try:
        first, last, keys = connection.execute(
            "SELECT min(t), max(t), count(DISTINCT k) FROM kpi_cohort_rows "
            "WHERE k IS NOT NULL AND t IS NOT NULL"
        ).fetchone() or (None, None, 0)
        if first is None or last is None or not keys:
            raise CohortError(
                f"No row has both a {customer.name} and a {date.name} here.", "no_rows"
            )
        months = (last.year - first.year) * 12 + last.month - first.month + 1
        if grain == "auto":
            grain = "month" if months >= MONTHS_FOR_MONTHLY else "week"
            grain_reason = (
                f"R-COH-03 · Monthly cohorts: the data spans {months} months"
                if grain == "month"
                else f"R-COH-03 · Weekly cohorts: the data spans {months} months, fewer than "
                f"{MONTHS_FOR_MONTHLY}"
            )
        else:
            grain_reason = f"R-COH-03 · {grain.capitalize()}ly cohorts, as chosen"
        rows = connection.execute(
            f"WITH a AS (SELECT DISTINCT k, date_trunc('{grain}', t) AS p FROM kpi_cohort_rows "
            "WHERE k IS NOT NULL AND t IS NOT NULL), "
            "f AS (SELECT k, min(p) AS c FROM a GROUP BY k) "
            "SELECT f.c, a.p, count(*) FROM a JOIN f USING (k) GROUP BY ALL ORDER BY 1, 2"
        ).fetchall()
        (returning,) = connection.execute(
            f"SELECT count(*) FROM (SELECT k FROM kpi_cohort_rows WHERE k IS NOT NULL AND t IS "
            f"NOT NULL GROUP BY k HAVING count(DISTINCT date_trunc('{grain}', t)) >= 2)"
        ).fetchone() or (0,)
    finally:
        connection.execute("DROP TABLE IF EXISTS kpi_cohort_rows")

    starts = sorted({p for _, p, _ in rows})
    if len(starts) < MIN_PERIODS:
        raise CohortError(
            f"R-COH-02 · Cohorts need at least {MIN_PERIODS} {grain}s of data: "
            f"{len(starts)} here. Try weekly cohorts or more data.",
            "too_few_periods",
        )
    share = float(returning) / float(keys)
    if share < MIN_RETURNING:
        raise CohortError(
            f"R-COH-02 · Only {_percent(share)} of {customer.name} values come back in a "
            f"second {grain} (at least {_percent(MIN_RETURNING)} needed): retention would read "
            "almost nothing. Check that it names a customer.",
            "few_returning",
        )
    last_start = starts[-1]
    ends_early = (_next(last_start, grain) - last).total_seconds() > 86_400
    sizes: dict[datetime, int] = {}
    seen: dict[tuple[datetime, int], int] = {}
    new: dict[datetime, int] = {}
    back: dict[datetime, int] = {}
    for cohort, period, count in rows:
        offset = _offset(cohort, period, grain)
        seen[(cohort, offset)] = int(count)
        if offset == 0:
            sizes[cohort] = int(count)
            new[period] = new.get(period, 0) + int(count)
        else:
            back[period] = back.get(period, 0) + int(count)
    cohort_starts = sorted(sizes)
    # R-COH-04: the latest cohorts large enough to read a share on, the others said
    large = [start for start in cohort_starts if sizes[start] >= MIN_COHORT]
    shown = (large or cohort_starts)[-MAX_COHORTS:]
    width = min(MAX_OFFSETS, _offset(shown[0], last_start, grain) + 1)
    cohorts = []
    for start in shown:
        observed = _offset(start, last_start, grain)
        retained: list[float | None] = []
        for offset in range(width):
            if offset > observed:
                retained.append(None)
            else:
                retained.append(seen.get((start, offset), 0) / sizes[start])
        cohorts.append(
            Cohort(start.date().isoformat(), _label(start, grain), sizes[start], retained)
        )
    periods = [
        PeriodCount(
            start.date().isoformat(),
            _label(start, grain),
            new.get(start, 0),
            back.get(start, 0),
            ends_early and start == last_start,
        )
        for start in starts
    ]
    warnings = []
    small = len(cohort_starts) - len(large) if large else 0
    older = len(large or cohort_starts) - len(shown)
    if small or older:
        left = []
        if older:
            left.append(f"{older} older ones")
        if small:
            left.append(f"{small} of fewer than {MIN_COHORT} customers")
        warnings.append(
            f"R-COH-04 · The latest {len(shown)} cohorts of at least {MIN_COHORT} customers are "
            f"shown; {' and '.join(left)} are left out of the triangle but counted in the new "
            "and returning customers."
        )
    if ends_early:
        warnings.append(
            f"The last {grain} ({_label(last_start, grain)}) is incomplete: the data ends on "
            f"{last:%d %b %Y}, so its retention reads low."
        )
    # every cohort observed one period later, weighted by its size (R-COH-05)
    observed_next = [start for start in cohort_starts if start < last_start]
    arrived = sum(sizes[start] for start in observed_next)
    if arrived:
        back_next = sum(seen.get((start, 1), 0) for start in observed_next)
        summary = (
            f"{_percent(back_next / arrived)} of new customers come back the next {grain} "
            f"({arrived:,} customers in {len(observed_next)} cohorts)."
        )
    else:
        summary = f"{_percent(share)} of customers are seen in two {grain}s or more."
    reason = (
        f"R-COH-05 · {keys:,} {customer.name} values; {_percent(share)} are seen in two "
        f"{grain}s or more. A cohort is the {grain} of a customer's first row; offset k is k "
        f"{grain}s later; a customer counts once per {grain}"
    )
    return CohortResult(
        customer.name,
        date.name,
        grain,
        grain_reason,
        cohorts,
        periods,
        int(keys),
        round(share, 4),
        reason,
        summary,
        warnings,
    )


# --- Funnel --------------------------------------------------------------------------------------


@dataclass
class Stage:
    label: str
    reached: int
    of_first: float | None  # share of the first stage
    conversion: float | None  # share of the previous stage


@dataclass
class FunnelResult:
    column: str
    key: str | None
    stages: list[Stage]
    outside: int  # keys (or rows) whose values are outside the stages
    outside_labels: list[str]  # the most frequent of these values, 5 at most
    reason: str
    summary: str
    warnings: list[str] = field(default_factory=list)


def funnel(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    status: Column,
    stages: Sequence[str],
    key: Column | None = None,
) -> FunnelResult:
    """R-FUN-02 and 03: the keys (or rows) that reached each stage, and where they drop."""
    if status.type not in FUNNEL_TYPES and status.type != "boolean":
        raise CohortError(f"A funnel reads a category: {status.name} is not one.", "wrong_type")
    stages = list(dict.fromkeys(stages))
    if len(stages) < 2:
        raise CohortError("A funnel needs at least two stages.", "too_few_stages")
    label = label_sql(status)
    rank = " ".join(f"WHEN {lit(stage)} THEN {index}" for index, stage in enumerate(stages))
    entity = f"CAST({typed_sql(key.name, key.parser)} AS VARCHAR)" if key else "file_row_number"
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE kpi_funnel_rows AS "
        f"SELECT {entity} AS e, {label} AS s, CASE {label} {rank} END AS r FROM {source}"
    )
    try:
        counts_sql = ", ".join(f"count(*) FILTER (WHERE r >= {i})" for i in range(len(stages)))
        reached = (
            connection.execute(
                "WITH m AS (SELECT e, max(r) AS r FROM kpi_funnel_rows WHERE e IS NOT NULL "
                f"AND r IS NOT NULL GROUP BY e) SELECT {counts_sql} FROM m"
            ).fetchone()
            or ()
        )
        (outside,) = connection.execute(
            "SELECT count(*) FROM (SELECT e FROM kpi_funnel_rows WHERE e IS NOT NULL "
            "AND s IS NOT NULL GROUP BY e HAVING max(r) IS NULL)"
        ).fetchone() or (0,)
        labels = [
            str(row[0])
            for row in connection.execute(
                "SELECT s, count(*) AS n FROM kpi_funnel_rows WHERE r IS NULL AND s IS NOT NULL "
                "GROUP BY s ORDER BY n DESC, s LIMIT 5"
            ).fetchall()
        ]
    finally:
        connection.execute("DROP TABLE IF EXISTS kpi_funnel_rows")
    counts = [int(value or 0) for value in reached]
    if not counts or counts[0] == 0:
        raise CohortError(f"No row of {status.name} is at one of these stages.", "no_rows")
    out = []
    for index, (stage, count) in enumerate(zip(stages, counts, strict=True)):
        before = counts[index - 1] if index else None
        out.append(
            Stage(
                stage,
                count,
                count / counts[0],
                None if before is None else (count / before if before else None),
            )
        )
    counted = f"distinct {key.name}" if key else "rows"
    drops = [(1 - (s.conversion or 0.0), i) for i, s in enumerate(out) if s.conversion is not None]
    summary = f"{_percent(out[-1].of_first or 0.0)} of {counted} reach {stages[-1]}."
    if drops:
        drop, index = max(drops)
        summary += (
            f" The largest drop is {stages[index - 1]} → {stages[index]}: "
            f"{_percent(drop)} do not go further (R-FUN-03)."
        )
    warnings = []
    if outside:
        warnings.append(
            f"{outside:,} {counted} are only at other values ({', '.join(labels)}): left out of "
            "the funnel."
        )
    reason = (
        f"R-FUN-02 · {counted.capitalize()} counted once at the furthest stage they reached in "
        f"{status.name}; a later stage counts as having passed the earlier ones"
    )
    return FunnelResult(
        status.name, key.name if key else None, out, int(outside), labels, reason, summary, warnings
    )
