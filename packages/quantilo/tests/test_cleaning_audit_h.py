"""Audit of 2026-11-12, wave 3 agent H (performance): N43 (Basic profiled in a few scans on a wide
file) and N44 (Advanced bounded on a wide file).

N43 changes how the rules read the data, never what they find: each test compares the new reading
with the one it replaces (the same counts, the same rows), on a small synthetic file. The real
files' suggestions are compared before / after by the bench (`scripts/clean_bench.py --real`)."""

import csv
import random
from pathlib import Path
from typing import Any

import duckdb
import pytest

import quantilo as dx
from quantilo.cleaning.rules import advanced as adv
from quantilo.cleaning.rules import basic, shapes
from quantilo.cleaning.suggest import suggest
from quantilo.ingestion.sql import lit
from quantilo.profiling import aggregates


def write(tmp_path: Path, rows: list[list[Any]], name: str = "data") -> dx.Dataset:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return dx.open(path, cache_dir=tmp_path / f"cache-{name}")


class Counting:
    """A connection that counts the queries it runs."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection
        self.queries = 0

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.queries += 1
        return self.connection.execute(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.connection, name)


def wide_numbers(tmp_path: Path, columns: int = 40) -> dx.Dataset:
    """A few columns of text and many of numbers stored as text, with a decimal comma, percents,
    blanks and a word (législatives, 76 columns of '69,55%')."""
    rng = random.Random(43)
    header = ["Département", "Libellé", *(f"Score {i}" for i in range(columns))]
    rows: list[list[Any]] = [header]
    for row in range(120):
        values = []
        for i in range(columns):
            roll = rng.random()
            if roll < 0.05:
                values.append("")
            elif i % 7 == 0 and roll < 0.1:
                values.append("n/a")
            else:
                values.append(f"{rng.uniform(0, 100):.2f}".replace(".", ",") + "%")
        rows.append([f"{row % 95 + 1:02d}", f"Dép {row}", *values])
    return write(tmp_path, rows, "wide")


def ids(found: list[Any]) -> list[tuple[str, str, int, list[Any]]]:
    return [(s.id, s.reason, s.cells, [e.model_dump() for e in s.examples]) for s in found]


# --- N43: Basic -----------------------------------------------------------------------------------


def test_the_values_table_is_dropped(tmp_path: Path) -> None:
    ds = wide_numbers(tmp_path, 8)
    connection = duckdb.connect()
    suggest(connection, str(ds.path), ds.types)
    tables = connection.execute("SELECT table_name FROM duckdb_tables()").fetchall()
    assert tables == []
    connection.close()


def test_numbers_counted_on_distinct_values_as_on_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B-07 on the values table finds what the query on the rows finds, reasons and examples
    included."""
    ds = wide_numbers(tmp_path, 12)
    batched = suggest(duckdb.connect(), str(ds.path), ds.types)
    assert sum(1 for s in batched if s.rule == "B-07") == 12
    monkeypatch.setattr(basic, "_number_counts", lambda context, columns: {})
    rows = suggest(duckdb.connect(), str(ds.path), ds.types)
    assert ids(batched) == ids(rows)


def test_queries_do_not_grow_with_the_number_columns(tmp_path: Path) -> None:
    """Before N43: a query per text column for its values, three per number column (B-07)."""
    found = {}
    for columns in (10, 40):
        ds = wide_numbers(tmp_path / str(columns), columns)
        counting = Counting(duckdb.connect())
        suggest(counting, str(ds.path), ds.types)  # type: ignore[arg-type]
        found[columns] = counting.queries
    # one query per number column remains (B-07's examples, in the file's order)
    assert found[40] - found[10] <= 30 + 5
    assert found[40] < 40 * 3


def test_blank_and_empty_agree(tmp_path: Path) -> None:
    rows: list[list[Any]] = [["n", "d", "t", "b"]]
    rows += [["1.5", "2024-01-02", " x ", "true"], ["", "", "  ", ""], ["0", "", "", "false"]]
    ds = write(tmp_path, rows)
    connection = duckdb.connect()
    context = basic.profile(connection, f"read_parquet({lit(str(ds.path))})", ds.types)
    for column in context.columns:
        blank, empty = (
            connection.execute(
                f"SELECT list({e} ORDER BY n NULLS FIRST) FROM {context.source}"
            ).fetchone()
            for e in (basic._blank(column), basic._empty(column))
        )
        assert blank == empty, column.name
    context.close()


def test_never_named() -> None:
    column = basic.Column("population", 0, "DOUBLE", None, 1, 1, None, None)
    assert basic._never_named(column)
    for name, dtype in (("2020", "BIGINT"), ("nan", "DOUBLE"), ("true", "BOOLEAN")):
        assert not basic._never_named(basic.Column(name, 0, dtype, None, 1, 1, None, None))
    assert not basic._never_named(basic.Column("name", 0, "VARCHAR", None, 1, 1, None, None))


def test_repeated_header_still_found_with_number_columns(tmp_path: Path) -> None:
    """A number column named '2020' can hold its name: B-15 still tests every column; a column
    named 'population' cannot, and no row can then repeat the names."""
    path = tmp_path / "header.parquet"
    connection = duckdb.connect()
    connection.execute(
        "COPY (SELECT * FROM (VALUES ('Paris', 1, 5), ('city', 2020, 6), ('Lyon', 3, 7)) "
        f't(city, "2020", population)) TO {lit(str(path))} (FORMAT parquet)'
    )
    source = f"read_parquet({lit(str(path))})"
    context = basic.profile(connection, source, [])
    context.columns = context.columns[:2]  # city, 2020
    basic.b15_empty_rows(context)
    assert [s.id for s in context.found] == ["B-15:header"]
    context = basic.profile(connection, source, [])
    basic.b15_empty_rows(context)
    assert context.found == []
    context.close()


def test_plain_values_skip_the_text_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = random.Random(7)
    alphabet = "ab Z9&;\t\u00a0\u200b\u00e9\u00c3\u00a9"
    values = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 6))) for _ in range(5000)]
    counted = [(v, i % 7 + 1) for i, v in enumerate(dict.fromkeys(values))]
    assert any(basic._plain(v) for v, _ in counted) and not all(basic._plain(v) for v, _ in counted)
    fast = basic.text_stages(counted)
    monkeypatch.setattr(basic, "_plain", lambda value: False)
    assert basic.text_stages(counted) == fast


def test_code_column_as_with_count_distinct(tmp_path: Path) -> None:
    rows: list[list[Any]] = [["country", "iso_code", "code", "value"]]
    for i in range(60):
        rows.append([f"Country {i % 12}", f"C{i % 12:02d}", f"K{i % 5}", i])
    rows.append(["World", "OWID_WRL", "K1", 1])
    ds = write(tmp_path, rows)
    source = f"read_parquet({lit(str(ds.path))})"
    connection = duckdb.connect()
    entity = next(v for v in ds.types if v.name == "country")
    found = aggregates._code_column(connection, source, entity, ds.types)
    assert found is not None and found[0].name == "iso_code"
    label, code = aggregates._label(entity), aggregates._label(found[0])
    rows_old = connection.execute(
        f"SELECT {label}, max({code}), count(DISTINCT {code}) FROM {source} "
        f"WHERE {label} IS NOT NULL GROUP BY 1"
    ).fetchall()
    assert all(k <= 1 for _, _, k in rows_old)
    assert found[1] == {str(lab): str(c) for lab, c, _ in rows_old if c is not None}


# --- N44: Advanced --------------------------------------------------------------------------------


def test_shape_as_before() -> None:
    def slow(value: str) -> str:
        out = []
        for char in value:
            if "A" <= char <= "Z":
                out.append("A")
            elif "a" <= char <= "z":
                out.append("a")
            elif char.isdigit():
                out.append("9")
            else:
                out.append(char)
        return "".join(out)

    rng = random.Random(44)
    alphabet = [chr(i) for i in range(32, 400)] + ["²", "٣", "\t"]
    for _ in range(20_000):
        value = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 9)))
        assert shapes.shape(value) == slow(value)


def near_rows(tmp_path: Path) -> dx.Dataset:
    rows: list[list[Any]] = [["order_date", "customer_id", "city", "amount", "note"]]
    rng = random.Random(15)
    for i in range(400):
        rows.append(
            [f"2024-01-{i % 28 + 1:02d}", f"C{i:04d}", rng.choice(["Paris", "Lyon"]), i, ""]
        )
    rows.append(["2024-01-05", "C0004", "paris", 4, "again"])  # near copy of row 5
    rows.append(["2024-01-05", "C0004", "PARIS ", 4, ""])
    rows.append(rows[10])  # an exact copy
    rows.append(["2024-01-11", "C0010", "lyon", 10, ""])
    return write(tmp_path, rows, "near")


def test_near_pairs_as_the_windows(tmp_path: Path) -> None:
    ds = near_rows(tmp_path)
    connection = duckdb.connect()
    context, rows_source = adv.profile(connection, str(ds.path), ds.types)
    advanced = adv.Advanced(context, rows_source, [])
    keys = [c for c in context.columns if c.name not in ("note",) and c.nonnull]
    empty = " AND ".join(basic._blank(c) for c in context.columns)
    normalized = ", ".join(adv._normalized(c) for c in keys)
    every = ", ".join(f'"{c.name}"' for c in context.columns)
    windows = connection.execute(
        f"SELECT file_row_number, first_row FROM (SELECT file_row_number, "
        f"min(file_row_number) OVER (PARTITION BY row({normalized})) AS first_row, "
        f"min(file_row_number) OVER (PARTITION BY row({every})) AS first_copy "
        f"FROM {rows_source} WHERE NOT ({empty})) "
        f"WHERE file_row_number <> first_row AND first_copy = file_row_number "
        f"ORDER BY file_row_number LIMIT 3"
    ).fetchall()
    rows, near, near_groups = adv._groups(advanced, normalized, empty)
    exact, exact_groups = adv._groups(advanced, every, empty)[1:]
    counted = connection.execute(
        f"SELECT count(*), count(DISTINCT row({normalized})), count(DISTINCT row({every})) "
        f"FROM {rows_source} WHERE NOT ({empty})"
    ).fetchone()
    assert (rows, near, exact) == counted
    assert adv._near_pairs(near_groups, exact_groups) == [(int(r), int(f)) for r, f in windows]
    assert windows
    context.close()


def test_near_duplicates_still_found(tmp_path: Path) -> None:
    found = {s.id: s for s in near_rows(tmp_path).clean().advanced()}
    near = found["A-15:rows:near"]
    assert near.rows >= 2
    assert (near.examples[0].before or "").startswith("row 40")


def many_categories(tmp_path: Path, columns: int) -> dx.Dataset:
    rng = random.Random(60)
    header = [f"Category {i}" for i in range(columns)]
    rows: list[list[Any]] = [header]
    for _ in range(300):
        rows.append([rng.choice(["Paris", "Lyon", "Nice", "Lille"]) for _ in range(columns)])
    rows[1][columns - 1] = "Pariss"  # a typo in the last, most varied column
    return write(tmp_path, rows, f"categories{columns}")


def test_spellings_bounded_on_a_wide_file(tmp_path: Path) -> None:
    """N44: past 60 category text columns, A-01 to A-03 check the 60 with the fewest distinct
    values per row; a note names the others."""
    ds = many_categories(tmp_path, adv.SPELLING_COLUMNS + 3)
    found = {s.id: s for s in ds.clean().advanced()}
    note = found["A-00:rows:budget"]
    assert note.mode == "I" and note.step is None and note.layer == "advanced"
    assert note.title.startswith("3 columns not checked (too many columns)")
    assert "open them one by one" in note.title
    assert [e.before for e in note.examples] == [
        f"Category {i}" for i in range(adv.SPELLING_COLUMNS, adv.SPELLING_COLUMNS + 3)
    ] or len(note.examples) == 3
    checked = {c for s in found.values() if s.rule in ("A-01", "A-02", "A-03") for c in s.columns}
    assert not checked & {e.before for e in note.examples}


def test_no_note_under_the_bounds(tmp_path: Path) -> None:
    found = many_categories(tmp_path, 8).clean().advanced()
    assert not any(s.rule == "A-00" for s in found)


def test_time_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adv, "TIME_BUDGET", 0.0)
    ds = many_categories(tmp_path, 4)
    found = {s.id: s for s in ds.clean().advanced()}
    note = found["A-00:rows:budget"]
    assert "time limit" in note.title
    assert "A-15" in note.reason and "A-01 to A-03" in note.reason
    assert [s.rule for s in found.values()] == ["A-00"]
