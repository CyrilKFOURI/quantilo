"""Files out of the Data page: the current view (CSV, Parquet, Excel) and a per-column summary."""

import csv
from pathlib import Path
from typing import Any, Literal

import duckdb
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell

from quantilo.errors import ExplorerError
from quantilo.ingestion.sql import ident, lit
from quantilo.rows.view import RowColumn
from quantilo.typing.parsers import present_sql, typed_sql

Format = Literal["csv", "parquet", "xlsx"]
EXCEL_ROWS = 1_048_575  # a sheet holds 1,048,576 rows, the header takes one
EXACT_QUANTILES = 200_000  # values: exact quantiles and distinct counts up to here
CHUNK = 10_000
NUMERIC = ("numeric_continuous", "numeric_discrete")


class TooManyRowsError(ExplorerError):
    def __init__(self, rows: int) -> None:
        super().__init__(
            f"{rows:,} rows do not fit in an Excel sheet ({EXCEL_ROWS:,} at most): "
            "export as CSV or Parquet instead.",
            code="too_many_rows",
            status_code=422,
        )


def export_select(relation: str, order: str, columns: list[RowColumn]) -> str:
    """Typed values (numbers as numbers, dates as dates) under the original column names."""
    parts = [
        f"{ident(c.name) if c.excluded else typed_sql(c.name, c.parser)} AS {ident(c.name)}"
        for c in columns
    ]
    return f"SELECT {', '.join(parts) or '1 AS empty'} FROM {relation} ORDER BY {order}"


def write_view(
    connection: duckdb.DuckDBPyConnection,
    relation: str,
    order: str,
    columns: list[RowColumn],
    file_format: Format,
    target: Path,
    rows: int,
) -> None:
    query = export_select(relation, order, columns)
    if file_format == "csv":
        connection.execute(f"COPY ({query}) TO {lit(target.as_posix())} (HEADER, DELIMITER ',')")
        return
    if file_format == "parquet":
        connection.execute(f"COPY ({query}) TO {lit(target.as_posix())} (FORMAT parquet)")
        return
    if rows > EXCEL_ROWS:
        raise TooManyRowsError(rows)
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet("data")
    sheet.append([c.name for c in columns])
    result = connection.execute(query)
    while chunk := result.fetchmany(CHUNK):
        for record in chunk:
            sheet.append([_cell(sheet, value) for value in record])
    workbook.save(target)


def _cell(sheet: Any, value: Any) -> Any:
    if getattr(value, "tzinfo", None) is not None:
        return value.replace(tzinfo=None)  # Excel has no time zones
    if isinstance(value, str) and value.startswith("="):
        # openpyxl writes "=..." as a formula, which Excel would run on opening: keep it text.
        cell = WriteOnlyCell(sheet, value)
        cell.data_type = "s"
        return cell
    return value


SUMMARY_FIELDS = [
    "column",
    "type",
    "rows",
    "missing",
    "missing_pct",
    "unreadable",
    "distinct",
    "min",
    "q1",
    "median",
    "q3",
    "max",
    "mean",
    "std",
    "top_value",
    "top_share_pct",
]


def column_summary(
    connection: duckdb.DuckDBPyConnection, source: str, column: RowColumn
) -> dict[str, Any]:
    typed = typed_sql(column.name, column.parser)
    present = present_sql(column.name, column.parser)
    rows, stored, valid = connection.execute(
        f"SELECT count(*), count(r), count(t) "
        f"FROM (SELECT {typed} AS t, {present} AS r FROM {source})"
    ).fetchone() or (0, 0, 0)
    # Exact up to 200,000 values; beyond, the approximation (a few % off) keeps it fast.
    distinct_sql = "count(DISTINCT t)" if valid <= EXACT_QUANTILES else "approx_count_distinct(t)"
    distinct_row = connection.execute(
        f"SELECT {distinct_sql} FROM (SELECT {typed} AS t FROM {source})"
    ).fetchone()
    distinct = int(distinct_row[0]) if distinct_row else 0
    summary: dict[str, Any] = {
        "column": column.name,
        "type": column.type,
        "rows": rows,
        "missing": rows - valid,
        "missing_pct": round(100 * (rows - valid) / rows, 2) if rows else None,
        "unreadable": stored - valid,
        "distinct": distinct,
    }
    if not valid:
        return summary
    if column.type in NUMERIC:
        quantile = "quantile_cont" if valid <= EXACT_QUANTILES else "approx_quantile"
        stats = connection.execute(
            f"SELECT min(t), {quantile}(t, 0.25), {quantile}(t, 0.5), {quantile}(t, 0.75), "
            f"max(t), avg(t), stddev_samp(t) FROM (SELECT CAST({typed} AS DOUBLE) AS t "
            f"FROM {source}) WHERE t IS NOT NULL AND isfinite(t)"
        ).fetchone()
        if stats:
            keys = ("min", "q1", "median", "q3", "max", "mean", "std")
            summary.update(dict(zip(keys, stats, strict=True)))
    elif column.type == "datetime":
        bounds = connection.execute(f"SELECT min({typed}), max({typed}) FROM {source}").fetchone()
        if bounds:
            summary.update(min=bounds[0], max=bounds[1])
    if column.type not in ("numeric_continuous", "datetime"):
        top = connection.execute(
            f"SELECT CAST(t AS VARCHAR), count(*) AS c FROM (SELECT {typed} AS t FROM {source}) "
            "WHERE t IS NOT NULL GROUP BY t ORDER BY c DESC, 1 LIMIT 1"
        ).fetchone()
        if top:
            summary.update(top_value=top[0], top_share_pct=round(100 * top[1] / valid, 2))
    return summary


def write_summary(
    connection: duckdb.DuckDBPyConnection, source: str, columns: list[RowColumn], target: Path
) -> None:
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for column in columns:
            if not column.excluded:
                writer.writerow(column_summary(connection, source, column))
