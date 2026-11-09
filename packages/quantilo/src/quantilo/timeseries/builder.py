"""Build series from dated rows in DuckDB: one aggregated value per period, per group.

Shared by Compare (a date against a number) and the Time series page, so both always agree.
The relation must expose `t` (timestamp) and `v` (number), and `c` (label) when grouped.
"""

from dataclasses import dataclass, field

import duckdb

from quantilo.ingestion.sql import lit
from quantilo.timeseries import series as ts
from quantilo.variable.source import clean

OTHERS = "Others"


@dataclass
class Values:
    label: str | None
    values: list[float | None]
    counts: list[int]  # rows aggregated in each period


@dataclass
class Aggregated:
    period: ts.Period
    aggregate: ts.Aggregate
    periods: list[str]  # formatted period starts, in order, without holes
    overall: Values  # every row, whatever the group
    groups: list[Values] = field(default_factory=list)  # empty when not grouped
    period_auto: bool = True
    aggregate_auto: bool = True
    coarsened: bool = False
    # The data starts after the first period begins (or ends before the last one ends).
    first_partial: bool = False
    last_partial: bool = False


def _one(connection: duckdb.DuckDBPyConnection, sql: str) -> tuple:
    return connection.execute(sql).fetchone() or ()


def aggregate(
    connection: duckdb.DuckDBPyConnection,
    relation: str,
    role: str | None,
    options: ts.Options,
    groups: list[str] | None = None,
) -> Aggregated:
    """Aggregate per period; `groups` = labels kept as their own series (others pooled)."""
    span, timed, first, last = _one(
        connection,
        "SELECT epoch(max(t)) - epoch(min(t)), "
        "count(*) FILTER (WHERE t <> CAST(CAST(t AS DATE) AS TIMESTAMP)), min(t), max(t) "
        f"FROM {relation}",
    )
    (gap,) = _one(
        connection,
        "SELECT median(g) FROM (SELECT epoch(d) - epoch(lag(d) OVER (ORDER BY d)) AS g "
        f"FROM (SELECT DISTINCT t AS d FROM {relation}))",
    )
    gap = float(gap or ts.PERIOD_SECONDS["day"])
    typical = ts.typical_period(gap, bool(timed))
    period, coarsened = ts.choose_period(options.period, typical, float(span or 0))
    chosen = ts.default_aggregate(role) if options.aggregate == "auto" else options.aggregate
    fmt, truncated, step = ts.FORMATS[period], f"date_trunc('{period}', t)", ts.INTERVALS[period]
    grid = [
        str(p)
        for (p,) in connection.execute(
            f"SELECT strftime(p, '{fmt}') FROM range("
            f"date_trunc('{period}', TIMESTAMP '{first}'), "
            f"date_trunc('{period}', TIMESTAMP '{last}') + INTERVAL '{step}', "
            f"INTERVAL '{step}') AS r(p) ORDER BY p"
        ).fetchall()
    ]
    empty = 0.0 if chosen in ("sum", "count") else None  # nothing sold is 0, not unknown
    measure = ts.AGGREGATE_SQL[chosen]

    def values(label: str | None, rows: dict[str, tuple[float | None, int]]) -> Values:
        return Values(
            label,
            [rows[p][0] if p in rows else empty for p in grid],
            [rows[p][1] if p in rows else 0 for p in grid],
        )

    overall = values(
        None,
        {
            str(p): (clean(v), int(k))
            for p, v, k in connection.execute(
                f"SELECT strftime({truncated}, '{fmt}'), {measure}, count(*) "
                f"FROM {relation} GROUP BY {truncated}"
            ).fetchall()
        },
    )
    grouped: list[Values] = []
    if groups is not None:
        listed = ", ".join(lit(label) for label in groups) or "NULL"
        bucket = f"CASE WHEN c IN ({listed}) THEN c ELSE '{OTHERS}' END"
        rows: dict[str, dict[str, tuple[float | None, int]]] = {}
        for g, p, v, k in connection.execute(
            f"SELECT {bucket} AS g, strftime({truncated}, '{fmt}'), {measure}, count(*) "
            f"FROM {relation} GROUP BY g, {truncated}"
        ).fetchall():
            rows.setdefault(str(g), {})[str(p)] = (clean(v), int(k))
        order = [*groups, OTHERS]
        grouped = [values(label, rows[label]) for label in order if label in rows]

    before, after = _one(
        connection,
        f"SELECT epoch(min(t)) - epoch(date_trunc('{period}', min(t))), "
        f"epoch(date_trunc('{period}', max(t)) + INTERVAL '{step}') - epoch(max(t)) "
        f"FROM {relation}",
    )
    return Aggregated(
        period,
        chosen,
        grid,
        overall,
        grouped,
        options.period == "auto",
        options.aggregate == "auto",
        coarsened,
        float(before) >= gap / 2,
        float(after) - gap >= gap / 2,
    )


def trend_input(built: Aggregated, values: list[float | None]) -> list[float | None]:
    """Values for a trend test: incomplete edge periods left out of sums and counts."""
    tested = list(values)
    if built.aggregate in ("sum", "count") and tested:
        if built.first_partial:
            tested[0] = None
        if built.last_partial:
            tested[-1] = None
    return tested
