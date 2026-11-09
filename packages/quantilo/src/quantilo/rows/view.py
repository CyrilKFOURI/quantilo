"""The rows the Data page shows: filtered rows narrowed by a display mode and a search, sorted."""

import math
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Literal

import duckdb

from quantilo.errors import ExplorerError
from quantilo.ingestion.sql import ident, lit
from quantilo.typing.parsers import present_sql, typed_sql
from quantilo.typing.types import AnalysisType, Parser
from quantilo.variable.source import SEED

Mode = Literal["first", "sample", "duplicates", "missing"]
SAMPLE = 10_000
MAX_LIMIT = 500
ROW = "file_row_number"


@dataclass
class RowColumn:
    name: str
    type: AnalysisType
    parser: Parser
    excluded: bool = False


@dataclass
class RowsColumn:
    name: str
    type: AnalysisType
    excluded: bool


@dataclass
class RowsPage:
    columns: list[RowsColumn]
    # Row number in the file (from 1), then one value per column: typed, or {"invalid": raw}
    # when the stored value is present but unreadable in the column's type.
    rows: list[list[Any]]
    total: int  # rows in the view (filters, mode, search)
    offset: int


@dataclass
class View:
    """Narrowing of the filtered rows; `sort` is a column name or None (file order)."""

    mode: Mode = "first"
    search: str = ""
    sort: str | None = None
    desc: bool = False


class UnknownColumnError(ExplorerError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Unknown column {name!r}.", code="unknown_column", status_code=422)


def _search_sql(columns: list[RowColumn], search: str) -> str:
    """Case-insensitive substring in the text of any column (not across two columns)."""
    needle = lit(search.lower())
    tests = [
        f"contains(lower(CAST({ident(c.name)} AS VARCHAR)), {needle})"
        for c in columns
        if not c.excluded
    ]
    return "(" + " OR ".join(tests) + ")" if tests else "false"


def view_relation(source: str, columns: list[RowColumn], view: View) -> tuple[str, str]:
    """The view's rows (every stored column plus the row number) and the ORDER BY that lists them.

    `source` must carry `file_row_number`. The order is always explicit: on a large Parquet file
    read in parallel, rows without ORDER BY come back in any order (sorting by row number with
    LIMIT/OFFSET stays fast: ~0.1 s at offset 4 M on 5 M rows).
    """
    relation = source
    conditions: list[str] = []
    if view.search.strip():
        conditions.append(_search_sql(columns, view.search.strip()))
    if view.mode == "missing":
        kept = [c for c in columns if not c.excluded]
        missing = " OR ".join(f"{present_sql(c.name, c.parser)} IS NULL" for c in kept)
        conditions.append(f"({missing})" if missing else "false")
    if conditions:
        relation = f"(SELECT * FROM {relation} WHERE {' AND '.join(conditions)})"
    order = ROW
    if view.mode == "sample":
        relation = (
            f"(SELECT * FROM {relation} WHERE {ROW} IN (SELECT {ROW} FROM {relation} "
            f"ORDER BY hash({ROW} + {SEED}) LIMIT {SAMPLE}))"
        )
    elif view.mode == "duplicates":
        everything = ", ".join(ident(c.name) for c in columns)
        relation = (
            f"(SELECT * EXCLUDE (_copies) FROM (SELECT *, "
            f"count(*) OVER (PARTITION BY {everything}) AS _copies, "
            f"min({ROW}) OVER (PARTITION BY {everything}) AS _first FROM {relation}) "
            "WHERE _copies > 1)"
        )
        order = f"_first, {ROW}"  # copies sit together
    if view.sort is not None:
        column = next((c for c in columns if c.name == view.sort), None)
        if column is None:
            raise UnknownColumnError(view.sort)
        key = ident(column.name) if column.excluded else typed_sql(column.name, column.parser)
        order = f"{key} {'DESC' if view.desc else 'ASC'} NULLS LAST, {ROW}"
    return relation, order


def value_sql(column: RowColumn) -> list[str]:
    """Typed value, then (for parsed columns) the raw text when it is present but unreadable."""
    if column.excluded or column.parser.kind == "raw":
        return [ident(column.name)]
    typed = typed_sql(column.name, column.parser)
    raw = present_sql(column.name, column.parser)
    return [typed, f"CASE WHEN ({typed}) IS NULL THEN CAST({raw} AS VARCHAR) END"]


def jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    return value


def count(connection: duckdb.DuckDBPyConnection, relation: str) -> int:
    row = connection.execute(f"SELECT count(*) FROM {relation}").fetchone()
    return int(row[0]) if row else 0


def page(
    connection: duckdb.DuckDBPyConnection,
    relation: str,
    order: str,
    columns: list[RowColumn],
    offset: int,
    limit: int,
) -> list[list[Any]]:
    parts = [value_sql(c) for c in columns]
    select = ", ".join([f"{ROW} + 1", *(sql for part in parts for sql in part)])
    # Two steps: rank only the row numbers (narrow), then read the full rows of the page.
    window = f"LIMIT {int(limit)} OFFSET {int(offset)}"
    picked = f"SELECT {ROW} FROM {relation} ORDER BY {order} {window}"
    fetched = connection.execute(
        f"SELECT {select} FROM {relation} WHERE {ROW} IN ({picked}) ORDER BY {order}"
    ).fetchall()
    rows: list[list[Any]] = []
    for record in fetched:
        row: list[Any] = [int(record[0])]
        position = 1
        for part in parts:
            value = record[position]
            if len(part) == 2 and value is None and record[position + 1] is not None:
                row.append({"invalid": record[position + 1]})  # unreadable in this type
            else:
                row.append(jsonable(value))
            position += len(part)
        rows.append(row)
    return rows
