"""The SQL of a KPI: one typed table of the rows, the top groups, then two aggregate queries.

The rows are read once into a temporary table (typed date, split labels, measure values); every
other query runs on it. Groups outside the top N are renamed "Other" *before* aggregating, so
"Other" is right even for a median or a distinct count, and `GROUPING SETS` give every total
in the same pass, recomputed on the rows (never a sum of cells for a non-additive measure).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import duckdb

from quantilo.compare.analyze import label_sql, number_sql
from quantilo.ingestion.sql import lit
from quantilo.kpi.definition import Agg, Grain, Measure
from quantilo.typing.parsers import typed_sql
from quantilo.typing.types import AnalysisType, Parser

TABLE = "kpi_rows"
OTHER = "Other"
EMPTY = "(empty)"
EXACT_MEDIAN_ROWS = 200_000  # CLAUDE.md: approx_quantile beyond
INTERVALS: dict[Grain, str] = {
    "day": "1 day",
    "week": "7 days",
    "month": "1 month",
    "quarter": "3 months",
    "year": "1 year",
}


class Column(Protocol):
    name: str
    type: AnalysisType
    parser: Parser


def value_sql(measure: Measure, column: Column | None, until: Column | None = None) -> str | None:
    if column is None:
        return None
    if measure.agg == "true_share":
        return typed_sql(column.name, column.parser)
    if measure.agg == "distinct":
        return typed_sql(column.name, column.parser)
    if measure.agg == "share":
        # R-MES-05: true when the label (as filters show it) is one of the values, NULL if empty
        label = label_sql(column)
        listed = ", ".join(lit(value) for value in measure.values)
        return f"(CASE WHEN {label} IS NULL THEN NULL ELSE {label} IN ({listed}) END)"
    if until is not None:
        # R-MES-07: a delay in days, hours included
        start = f"CAST({typed_sql(column.name, column.parser)} AS TIMESTAMP)"
        end = f"CAST({typed_sql(until.name, until.parser)} AS TIMESTAMP)"
        return f"((epoch({end}) - epoch({start})) / 86400.0)"
    number = number_sql(column)
    return f"(CASE WHEN isfinite({number}) THEN {number} END)"


def weight_sql(column: Column) -> str:
    """R-MES-06: a weight is a size; missing, infinite or ≤ 0 is no weight."""
    number = number_sql(column)
    return f"(CASE WHEN isfinite({number}) AND {number} > 0 THEN {number} END)"


def aggregate_sql(
    agg: Agg, value: str, approximate: bool, weight: str | None = None, key: str | None = None
) -> str:
    """`weight` (weighted mean) and `key` (share of distinct keys) name their own columns."""
    if agg == "count":
        return "count(*)"
    if agg == "sum":
        return f"sum({value})"
    if agg == "mean":
        return f"avg({value})"
    if agg == "median":
        return f"approx_quantile({value}, 0.5)" if approximate else f"median({value})"
    if agg == "min":
        return f"min({value})"
    if agg == "max":
        return f"max({value})"
    if agg == "distinct":
        return f"count(DISTINCT {value})"
    if agg == "weighted_mean" and weight is not None:
        both = f"CASE WHEN {value} IS NOT NULL THEN {weight} END"
        return f"(sum({value} * {weight}) / nullif(sum({both}), 0))"
    if agg == "share" and key is not None:
        hits = f"count(DISTINCT CASE WHEN {value} THEN {key} END)"
        known = f"count(DISTINCT CASE WHEN {value} IS NOT NULL THEN {key} END)"
        return f"({hits} / nullif({known}, 0))"
    return f"avg(CASE WHEN {value} THEN 1.0 WHEN NOT {value} THEN 0.0 END)"


def build_table(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    time: Column | None,
    splits: Sequence[Column],
    measures: list[tuple[Measure, Column | None]],
    extras: dict[str, Column] | None = None,
    order_key: Column | None = None,
) -> None:
    """Typed rows: t (timestamp), g1, g2 (labels, '(empty)' for a missing one), v1… (values),
    w1… (weights) and k1… (keys) of the measures that have one. `extras` gives the weight, key
    and end date columns by name. With `order_key`, `ok` is the order and `h` a fingerprint of
    the whole source row, to find rows repeated exactly (054, R-SIG-06)."""
    extras = extras or {}
    selected = []
    if time is not None:
        selected.append(f"CAST({typed_sql(time.name, time.parser)} AS TIMESTAMP) AS t")
    for index, split in enumerate(splits, start=1):
        selected.append(f"COALESCE({label_sql(split)}, {lit(EMPTY)}) AS g{index}")
    for index, (measure, column) in enumerate(measures, start=1):
        until = extras.get(measure.until) if measure.until else None
        value = value_sql(measure, column, until)
        if value is not None:
            selected.append(f"{value} AS v{index}")
        if measure.weight and measure.weight in extras:
            selected.append(f"{weight_sql(extras[measure.weight])} AS w{index}")
        if measure.key and measure.key in extras:
            key = extras[measure.key]
            selected.append(f"{typed_sql(key.name, key.parser)} AS k{index}")
    if order_key is not None:
        selected.append(f"{typed_sql(order_key.name, order_key.parser)} AS ok")
        selected.append("hash(kpi_source_row) AS h")
    columns = ", ".join(selected) or "1 AS one"
    connection.execute(
        f"CREATE OR REPLACE TEMP TABLE {TABLE} AS SELECT {columns} FROM {source} AS kpi_source_row"
    )


def duplicated(connection: duckdb.DuckDBPyConnection, where: str) -> tuple[int, int]:
    """Rows with an order key, and how many of them are exact copies of another one."""
    rows, distinct = one(
        connection,
        f"SELECT count(*), count(DISTINCT h) FROM {TABLE} WHERE ({where}) AND ok IS NOT NULL",
    )
    return int(rows or 0), int((rows or 0) - (distinct or 0))


def one(connection: duckdb.DuckDBPyConnection, sql: str) -> tuple:
    return connection.execute(sql).fetchone() or ()


def median_gap(connection: duckdb.DuckDBPyConnection, where: str) -> float | None:
    (gap,) = one(
        connection,
        "SELECT median(g) FROM (SELECT epoch(d) - epoch(lag(d) OVER (ORDER BY d)) AS g "
        f"FROM (SELECT DISTINCT t AS d FROM {TABLE} WHERE {where}))",
    )
    return float(gap) if gap is not None else None


def timestamp(value: datetime) -> str:
    return f"TIMESTAMP '{value.isoformat(sep=' ')}'"


def period_starts(
    connection: duckdb.DuckDBPyConnection, grain: Grain, first: datetime, last: datetime
) -> list[datetime]:
    """Every period start from the one holding `first` to the one holding `last`."""
    step = INTERVALS[grain]
    return [
        row[0]
        for row in connection.execute(
            f"SELECT p FROM range(date_trunc('{grain}', {timestamp(first)}), "
            f"date_trunc('{grain}', {timestamp(last)}) + INTERVAL '{step}', "
            f"INTERVAL '{step}') AS r(p) ORDER BY p"
        ).fetchall()
    ]


@dataclass
class Level:
    label: str
    value: float | None  # the main measure over the displayed range, for the order


def top_levels(
    connection: duckdb.DuckDBPyConnection,
    split: int,
    main: str,
    where: str,
    order: list[str] | None,
) -> list[Level]:
    """Every label of a split with the main measure, largest first (natural order if ordinal)."""
    rows = connection.execute(
        f"SELECT g{split}, {main} AS s FROM {TABLE} WHERE {where} GROUP BY g{split} "
        f"ORDER BY s DESC NULLS LAST, g{split}"
    ).fetchall()
    levels = [Level(str(label), None if value is None else float(value)) for label, value in rows]
    if order:
        rank = {label: index for index, label in enumerate(order)}
        levels.sort(key=lambda level: (level.label not in rank, rank.get(level.label, 0)))
    return levels


def fold_others(connection: duckdb.DuckDBPyConnection, split: int, kept: list[str]) -> None:
    listed = ", ".join(lit(label) for label in kept) or "NULL"
    connection.execute(
        f"UPDATE {TABLE} SET g{split} = {lit(OTHER)} WHERE g{split} NOT IN ({listed})"
    )


Cell = tuple[datetime | None, str | None, str | None]  # period, g1, g2 (None = total)


def aggregate(
    connection: duckdb.DuckDBPyConnection,
    aggregates: list[str],
    splits: int,
    grain: Grain | None,
    where: str,
) -> dict[Cell, tuple[float | None, ...]]:
    """Every cell and total: values of `aggregates` then the row count, per cell key."""
    dims = [f"g{i}" for i in range(1, splits + 1)]
    sets: list[list[str]] = [[]]
    for dim in dims:
        sets = [*sets, *[[*s, dim] for s in sets]]
    period = f"date_trunc('{grain}', t)" if grain else None
    if period:
        sets = [["p", *s] for s in sets]
    grouping = ", ".join("(" + ", ".join(s) + ")" for s in sets)
    keys = (["p"] if period else []) + dims
    selected = []
    if period:
        selected.append(f"{period} AS p")
    selected += [*dims, *aggregates, "count(*)"]
    sql = f"SELECT {', '.join(selected)} FROM {TABLE} WHERE {where}"
    if keys:
        sql += f" GROUP BY GROUPING SETS ({grouping})"
    cells: dict[Cell, tuple[float | None, ...]] = {}
    for row in connection.execute(sql).fetchall():
        offset = len(keys)
        key_values = dict(zip(keys, row[:offset], strict=True))
        key: Cell = (
            key_values.get("p"),
            key_values.get("g1"),
            key_values.get("g2"),
        )
        cells[key] = tuple(None if v is None else float(v) for v in row[offset:])
    return cells
