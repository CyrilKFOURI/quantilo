"""A generated date table (C1, R-BLD-08): one row per day of a fact table's dates.

Power BI's "Mark as date table": the days from the first to the last date of the column, gaps
included (a day without a sale is still a day), with the parts an analysis groups by. Written as a
dataset of the project and related to the fact by the date (`date`, one row per day: n:1).
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

from quantilo.ingestion.sql import lit
from quantilo.model.build import ModelError
from quantilo.model.keys import TableInput
from quantilo.typing.parsers import typed_sql

MAX_DAYS = 200 * 366  # R-BLD-08: two centuries at most (a date typed 1900 or 9999 is a mistake)
COLUMNS = (
    "date",
    "year",
    "quarter",
    "month",
    "month_name",
    "year_month",
    "week",
    "weekday",
    "weekday_name",
    "weekend",
    "day_of_year",
)


@dataclass
class DateTable:
    rows: int
    first: date
    last: date


def date_range(
    connection: duckdb.DuckDBPyConnection, table: TableInput, column: str
) -> tuple[date, date]:
    view = next((v for v in table.columns if v.name == column), None)
    if view is None:
        raise ModelError(f"`{column}` is not a column of {table.name}.", code="unknown_column")
    if view.type != "datetime" and view.parser.kind != "date":
        raise ModelError(f"`{column}` is not a date column (R-BLD-08).", code="not_a_date")
    typed = f"TRY_CAST({typed_sql(view.name, view.parser)} AS DATE)"
    row = connection.execute(
        f"SELECT min({typed}), max({typed}) FROM read_parquet({lit(table.storage_path)})"
    ).fetchone()
    if not row or row[0] is None:
        raise ModelError(f"`{column}` has no date to start from.", code="not_a_date")
    first, last = row[0], row[1]
    if (last - first).days + 1 > MAX_DAYS:
        raise ModelError(
            f"`{column}` spans {first} to {last}: more than two centuries, likely a typing slip. "
            "Fix the outlying dates first (R-BLD-08).",
            code="too_long",
        )
    return first, last


def date_table_sql(first: date, last: date) -> str:
    return (
        "SELECT CAST(d AS DATE) AS date, year(d) AS year, quarter(d) AS quarter, "
        "month(d) AS month, monthname(d) AS month_name, strftime(d, '%Y-%m') AS year_month, "
        "weekofyear(d) AS week, isodow(d) AS weekday, dayname(d) AS weekday_name, "
        "isodow(d) >= 6 AS weekend, dayofyear(d) AS day_of_year "
        f"FROM range(TIMESTAMP {lit(first.isoformat())}, TIMESTAMP {lit(last.isoformat())} "
        "+ INTERVAL 1 DAY, INTERVAL 1 DAY) AS t(d) ORDER BY d"
    )


def write_date_table(
    connection: duckdb.DuckDBPyConnection, table: TableInput, column: str, target: Path
) -> DateTable:
    first, last = date_range(connection, table, column)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"COPY ({date_table_sql(first, last)}) TO {lit(str(target))} "
        "(FORMAT parquet, COMPRESSION zstd)"
    )
    return DateTable((last - first).days + 1, first, last)
