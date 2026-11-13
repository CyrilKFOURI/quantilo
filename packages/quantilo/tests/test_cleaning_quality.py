"""N37: each column's quality (empty, unreadable for its type, the rest) from the Basic profile,
with no other read of the file."""

import csv
from pathlib import Path
from typing import Any

import duckdb

import quantilo as dx
from quantilo.cleaning.suggest import suggest, suggest_and_measure


def write(tmp_path: Path, rows: list[list[Any]]) -> dx.Dataset:
    path = tmp_path / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return dx.open(path, cache_dir=tmp_path / "cache")


def test_quality_counts_empty_and_unreadable_cells(tmp_path: Path) -> None:
    rows: list[list[Any]] = [["amount", "city", "score"]]
    for index in range(80):
        amount = (
            f"{index % 12}.5"
            if index < 74
            else ["12 kg", "12 kg", "about 5", "", "N/A", "8.5"][index - 74]
        )
        city = ["Paris", "Lyon", "", "Nice"][index % 4]
        rows.append([amount, city, index if index % 10 else ""])
    # a value the type cannot read keeps a column as text (R-TYP-01): the user calls it a number
    ds = write(tmp_path, rows).retype("amount", "numeric_continuous")
    found, quality = suggest_and_measure(duckdb.connect(), str(ds.path), ds.types)
    by = {q.column: q for q in quality}
    assert [q.column for q in quality] == ["amount", "city", "score"]
    amount = by["amount"]
    assert amount.rows == 80 and amount.reading == "number"
    assert amount.missing == 2  # the blank and 'N/A'
    assert amount.unreadable == 3
    assert [(v.value, v.rows) for v in amount.unreadable_values] == [("12 kg", 2), ("about 5", 1)]
    city = by["city"]
    assert city.missing == 20 and city.unreadable == 0 and city.reading is None
    score = by["score"]
    assert score.missing == 8 and score.unreadable == 0
    # the suggestions are those `suggest` makes alone
    alone = suggest(duckdb.connect(), str(ds.path), ds.types)
    assert [s.model_dump() for s in found] == [s.model_dump() for s in alone]


def test_quality_reads_no_row_again(tmp_path: Path) -> None:
    """Measured on the profile's distinct values: one query more, and none reads the file."""
    rows: list[list[Any]] = [["amount", "label"]]
    rows += [[f"{i % 9},5" if i % 7 else "n.c.", f"x{i % 5}"] for i in range(60)]
    ds = write(tmp_path, rows).retype("amount", "numeric_continuous")

    class Counting:
        def __init__(self) -> None:
            self.connection = duckdb.connect()
            self.queries: list[str] = []

        def execute(self, sql: str, *args: Any) -> Any:
            self.queries.append(sql)
            return self.connection.execute(sql, *args)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.connection, name)

    alone, measured = Counting(), Counting()
    suggest(alone, str(ds.path), ds.types)  # type: ignore[arg-type]
    suggest_and_measure(measured, str(ds.path), ds.types)  # type: ignore[arg-type]
    (extra,) = [sql for sql in measured.queries if "list(v ORDER BY c DESC, v)" in sql]
    assert "read_parquet" not in extra

    def reads(c: Counting) -> int:
        return sum("read_parquet" in sql for sql in c.queries)

    assert reads(measured) == reads(alone)
