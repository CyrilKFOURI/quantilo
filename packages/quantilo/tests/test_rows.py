import csv
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest
from openpyxl import load_workbook

from quantilo.rows.export import TooManyRowsError, column_summary, write_view
from quantilo.rows.view import (
    SAMPLE,
    RowColumn,
    UnknownColumnError,
    View,
    count,
    page,
    view_relation,
)
from quantilo.typing.types import Parser

RAW = Parser(kind="raw")
NUMBER = Parser(kind="number", decimal=",")
DATE = Parser(kind="date", format="%d/%m/%Y")

COLUMNS = [
    RowColumn("city", "categorical_nominal", Parser(kind="text")),
    RowColumn("price", "numeric_continuous", NUMBER),
    RowColumn("day", "datetime", DATE),
    RowColumn("note", "text", RAW, excluded=True),
]


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    table = pa.table(
        {
            "file_row_number": pa.array(range(6), pa.int64()),
            "city": ["Paris", "Lyon", "Paris", "Nice", "Lyon", "Paris"],
            "price": ["10,5", "7", "10,5", "n/a", "abc", "3"],
            "day": ["01/02/2024", "05/02/2024", "01/02/2024", "", "09/02/2024", "02/02/2024"],
            "note": ["a", "b", "a", "c", "d", "e"],
        }
    )
    con = duckdb.connect()
    con.register("t", table)
    con.execute("CREATE TABLE data AS SELECT * FROM t")
    return con


def rows_of(con, view: View, offset: int = 0, limit: int = 100):
    relation, order = view_relation("data", COLUMNS, view)
    return count(con, relation), page(con, relation, order, COLUMNS, offset, limit)


def test_first_rows_are_typed_and_numbered(connection) -> None:
    total, rows = rows_of(connection, View())

    assert total == 6
    assert rows[0][0] == 1 and rows[0][1] == "Paris" and rows[0][2] == 10.5
    assert rows[0][3].startswith("2024-02-01")
    assert [row[0] for row in rows] == [1, 2, 3, 4, 5, 6]  # file order


def test_unreadable_values_come_back_raw(connection) -> None:
    _, rows = rows_of(connection, View())

    assert rows[4][2] == {"invalid": "abc"}  # not a number: shown as stored, flagged
    assert rows[3][2] is None  # "n/a" is a missing value, not an unreadable one
    assert rows[3][3] is None


def test_pages_and_limits(connection) -> None:
    total, rows = rows_of(connection, View(), offset=2, limit=2)

    assert total == 6 and [row[0] for row in rows] == [3, 4]


def test_sort_by_typed_value_with_empty_last(connection) -> None:
    _, ascending = rows_of(connection, View(sort="price"))
    _, descending = rows_of(connection, View(sort="price", desc=True))

    assert [row[0] for row in ascending] == [6, 2, 1, 3, 4, 5]  # 3, 7, 10.5, 10.5, then empty
    assert [row[0] for row in descending][:4] == [1, 3, 2, 6]
    with pytest.raises(UnknownColumnError):
        rows_of(connection, View(sort="nope"))


def test_search_is_case_insensitive_and_skips_ignored_columns(connection) -> None:
    total, rows = rows_of(connection, View(search="pAR"))
    hidden, _ = rows_of(connection, View(search="e"))  # "e" only in the ignored note… and Nice

    assert total == 3 and {row[1] for row in rows} == {"Paris"}
    assert hidden == 1


def test_duplicates_sit_together(connection) -> None:
    total, rows = rows_of(connection, View(mode="duplicates"))

    assert total == 2 and [row[0] for row in rows] == [1, 3]


def test_missing_mode(connection) -> None:
    _, rows = rows_of(connection, View(mode="missing"))

    assert [row[0] for row in rows] == [4]  # "n/a" and "" are missing; "abc" is present


def test_sample_is_reproducible_and_in_file_order() -> None:
    con = duckdb.connect()
    con.execute("CREATE TABLE big AS SELECT range AS file_row_number, range AS v FROM range(30000)")
    columns = [RowColumn("v", "numeric_discrete", RAW)]
    relation, order = view_relation("big", columns, View(mode="sample"))

    first = page(con, relation, order, columns, 0, 50)
    again = page(con, relation, order, columns, 0, 50)

    assert count(con, relation) == SAMPLE
    assert first == again and first == sorted(first) and first[0][0] > 0


def test_export_formats(connection, tmp_path: Path) -> None:
    relation, order = view_relation("data", COLUMNS, View(sort="price"))
    chosen = COLUMNS[:2]

    write_view(connection, relation, order, chosen, "csv", tmp_path / "a.csv", 0)
    write_view(connection, relation, order, chosen, "parquet", tmp_path / "a.parquet", 0)
    write_view(connection, relation, order, chosen, "xlsx", tmp_path / "a.xlsx", 6)

    with (tmp_path / "a.csv").open(encoding="utf-8") as handle:
        lines = list(csv.reader(handle))
    assert lines[0] == ["city", "price"] and lines[1] == ["Paris", "3.0"]
    parquet = duckdb.execute(f"SELECT * FROM '{(tmp_path / 'a.parquet').as_posix()}'").fetchall()
    assert parquet[0] == ("Paris", 3.0) and len(parquet) == 6
    sheet = load_workbook(tmp_path / "a.xlsx").active
    assert sheet is not None
    assert [cell.value for cell in sheet[2]] == ["Paris", 3]


def test_excel_refuses_too_many_rows(connection, tmp_path: Path) -> None:
    relation, order = view_relation("data", COLUMNS, View())

    with pytest.raises(TooManyRowsError):
        write_view(connection, relation, order, COLUMNS, "xlsx", tmp_path / "a.xlsx", 2_000_000)


def test_column_summary(connection) -> None:
    price = column_summary(connection, "data", COLUMNS[1])
    city = column_summary(connection, "data", COLUMNS[0])

    assert (price["rows"], price["missing"], price["unreadable"]) == (6, 2, 1)
    assert price["min"] == 3 and price["max"] == 10.5 and price["median"] == pytest.approx(8.75)
    assert city["top_value"] == "Paris" and city["top_share_pct"] == 50
    assert city["distinct"] == 3 and price["distinct"] == 3


def test_excel_keeps_formula_like_text_as_text(tmp_path: Path) -> None:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE f AS SELECT 0::BIGINT AS file_row_number, '=HYPERLINK(\"x\")' AS note"
    )
    columns = [RowColumn("note", "text", Parser(kind="text"))]
    relation, order = view_relation("f", columns, View())

    write_view(con, relation, order, columns, "xlsx", tmp_path / "f.xlsx", 1)

    sheet = load_workbook(tmp_path / "f.xlsx").active
    assert sheet is not None
    cell = sheet["A2"]
    assert cell.data_type == "s" and cell.value == '=HYPERLINK("x")'
