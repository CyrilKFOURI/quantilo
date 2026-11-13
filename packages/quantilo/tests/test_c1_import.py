"""C1, import: a running balance marked derived (K5), tables of the same schema stacked (S1)."""

import datetime as dt
from pathlib import Path

import duckdb
from openpyxl import Workbook

from quantilo.ingestion.layout import detect_layout
from quantilo.ingestion.options import ImportOptions
from quantilo.ingestion.preview import stack_groups
from quantilo.ingestion.reader import SourceReader
from quantilo.ingestion.sources import list_source_files


def bank(path: Path, in_out: bool = False) -> Path:
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Bank"
    header = (
        ["Date", "Label", "Money in", "Money out", "Balance"]
        if in_out
        else ["Date", "Label", "Amount", "Balance"]
    )
    ws.append(header)
    balance = 1000.0
    for i in range(1, 25):
        amount = float((i * 37) % 200 - 90)
        balance += amount
        day = dt.date(2024, 1, 1) + dt.timedelta(days=i)
        if in_out:
            ws.append([day, f"op {i}", max(amount, 0) or None, max(-amount, 0) or None, balance])
        else:
            ws.append([day, f"op {i}", amount, balance])
    wb.save(path)
    return path


def test_a_running_balance_is_marked_derived(tmp_path: Path) -> None:
    for in_out in (False, True):
        table = detect_layout(bank(tmp_path / f"b{in_out}.xlsx", in_out)).tables[0]
        derived = {c.name for c in table.columns if c.role == "derived"}
        assert derived == {"Balance"}, in_out
        [hit] = [h for h in table.reasons if h.rule == "K5"]
        assert "running balance" in hit.message and "Balance" in hit.message


def test_no_running_balance_in_unrelated_numbers(tmp_path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.append(["Store", "Sales", "Visitors"])
    for i in range(30):
        ws.append([f"s{i}", (i * 53) % 97, (i * 31) % 89])
    wb.save(tmp_path / "s.xlsx")
    table = detect_layout(tmp_path / "s.xlsx").tables[0]
    assert not [h for h in table.reasons if h.rule == "K5"]


def months(path: Path) -> Path:
    wb = Workbook()
    first = wb.active
    assert first is not None
    for index, month in enumerate(("Jan", "Feb", "Mar")):
        ws = first if index == 0 else wb.create_sheet()
        ws.title = month
        ws.append(["Product", "Units", "Price"])
        for i in range(5):
            ws.append([f"p{i}", i + index, 2.5])
    other = wb.create_sheet("Notes")
    other.append(["Who", "Said"])
    other.append(["Ann", "hello"])
    wb.save(path)
    return path


def test_tables_with_the_same_columns_are_stacked(tmp_path: Path) -> None:
    path = months(tmp_path / "m.xlsx")
    [stack] = stack_groups(path)
    assert stack.rule == "S1" and [t.split("!")[0] for t in stack.tables] == ["Jan", "Feb", "Mar"]
    assert stack.rows == 15 and stack.columns == ["Product", "Units", "Price"]
    connection = duckdb.connect()
    reader = SourceReader(connection, ImportOptions(stack=stack.tables))
    sql = reader.combined_sql(list_source_files(path))
    rows = connection.execute(
        f"SELECT source_sheet, count(*) FROM ({sql}) GROUP BY 1 ORDER BY 1"
    ).fetchall()
    assert rows == [("Feb", 5), ("Jan", 5), ("Mar", 5)]
    connection.close()


# --- R-LOG-11: a line pattern the user declares ------------------------------------------------

APP_LOG = """2024-03-01 10:00:01 | web-1 | ERROR | payment 41 failed
2024-03-01 10:00:02 | web-2 | INFO | payment 42 done
    at com.shop.Pay(Pay.java:12)
not a log line
2024-03-01 10:00:03 | web-1 | WARN | slow request 900 ms
"""


def test_a_declared_pattern_reads_the_lines(tmp_path: Path) -> None:
    import pytest

    from quantilo.ingestion.errors import IngestionError
    from quantilo.ingestion.logs import read_log, user_format

    path = tmp_path / "app.log"
    path.write_text(APP_LOG, encoding="utf-8")
    table, reading = read_log(path, pattern="{time} | {host} | {level} | {message}")
    rows = table.to_pylist()
    assert reading.rule == "R-LOG-11" and reading.entries == 4 and reading.unmatched == 1
    assert rows[0]["host"] == "web-1" and rows[0]["level"] == "ERROR"
    assert rows[0]["time"] == "2024-03-01 10:00:01"  # text: the import types it
    assert "Pay.java" in rows[1]["message"]  # the indented line glued (R-LOG-07)
    assert rows[2]["message"] == "not a log line" and rows[2]["host"] is None
    assert "3 of 4 entries follow it" in reading.notes[0]
    regex, _ = read_log(path, pattern=r"(?P<stamp>\S+ \S+) \| (?P<node>\S+) .*")
    assert (
        regex.column_names[:3] == ["line_no", "stamp", "node"] and "message" in regex.column_names
    )
    for bad in ("no fields", "{a} {a}", "(?P<x>[)"):
        with pytest.raises(IngestionError):
            user_format(bad)
