"""Cleaning (065): the trap file's suggestions (100 % recall), each step's SQL, the preview, the
cleaned child with its rowmap, and `ds.clean()`."""

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

import quantilo as dx
from quantilo.cleaning import CleanError, compile_steps, parse_steps
from quantilo.cleaning.apply import apply_steps
from quantilo.cleaning.preview import preview
from quantilo.cleaning.suggest import with_suggestion, without_suggestion
from quantilo.cleaning.text import fix_encoding, normalize_spaces, remove_invisible, unescape_html

SAMPLES = Path(__file__).resolve().parents[3] / "samples"


def parquet(tmp_path: Path, csv: str, name: str = "data") -> Path:
    source = tmp_path / f"{name}.csv"
    source.write_text(csv, encoding="utf-8")
    target = tmp_path / f"{name}.parquet"
    duckdb.execute(
        f"COPY (SELECT * FROM read_csv('{source.as_posix()}')) TO '{target.as_posix()}' "
        "(FORMAT parquet)"
    )
    return target


def run(path: Path, *steps: dict[str, Any]) -> tuple[list[str], list[tuple[Any, ...]]]:
    """The table after the steps: column names, rows in the file's order."""
    parsed = parse_steps([{"id": f"s{i + 1}", **step} for i, step in enumerate(steps)]).steps
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(path), parsed, strict=True)
        names = compiled.final.names
        rows = connection.execute(
            f"{compiled.sql(-1, compiled.output_columns())} ORDER BY __quantilo_row"
        ).fetchall()
    finally:
        connection.close()
    return names, rows


def column(path: Path, name: str, *steps: dict[str, Any]) -> list[Any]:
    names, rows = run(path, *steps)
    index = names.index(name)
    return [row[index] for row in rows]


# --- The trap file (M0) ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def traps(tmp_path_factory: pytest.TempPathFactory) -> dx.Dataset:
    cache = tmp_path_factory.mktemp("traps")
    return dx.open(SAMPLES / "dirty_traps.csv", cache_dir=cache)


def test_trap_file_every_expected_suggestion_with_its_mode(traps: dx.Dataset) -> None:
    expected = json.loads((SAMPLES / "dirty_traps.expected.json").read_text(encoding="utf-8"))
    found = {s.id: s for s in traps.clean().suggestions}
    missing = [e["id"] for e in expected["expected"] if e["id"] not in found]
    assert not missing, f"not found: {missing}"  # 100 % recall
    wrong = [
        (e["id"], found[e["id"]].mode)
        for e in expected["expected"]
        if found[e["id"]].mode != e["mode"]
    ]
    assert not wrong
    assert set(found) == {e["id"] for e in expected["expected"]}  # nothing else on this file
    touched = {c for s in found.values() for c in s.columns}
    assert not touched & set(expected["controls"])


def test_trap_suggestions_say_why_with_their_numbers(traps: dx.Dataset) -> None:
    found = {s.id: s for s in traps.clean().suggestions}
    assert "NEW YORK" in [e.before for e in found["B-03:city"].examples]
    assert found["B-03:city"].step.map["NEW YORK"] == "New York"  # type: ignore[union-attr]
    assert "6 rows are exact copies" in found["B-14:rows"].reason
    assert "cannot be read" in found["B-07:weight"].reason
    assert found["B-05:brand"].step.map["CafÃ© Noir"] == "Café Noir"  # type: ignore[union-attr]
    for suggestion in found.values():
        assert suggestion.reason and suggestion.title
        assert suggestion.step is not None
        assert suggestion.step.suggestion == suggestion.id
        assert suggestion.step.origin == ("auto" if suggestion.mode == "A" else "suggested")


def test_trap_file_cleaned_child(traps: dx.Dataset, tmp_path: Path) -> None:
    before = hashlib.sha256(traps.path.read_bytes()).hexdigest()
    cleaning = traps.clean()
    child = cleaning.apply(cache_dir=tmp_path)
    assert hashlib.sha256(traps.path.read_bytes()).hexdigest() == before  # C-01
    assert len(child) == len(traps) - 7  # an empty row and six copies
    types = {c.name: (c.storage_dtype, c.type) for c in child.types}
    assert types["price"] == ("DOUBLE", "numeric_continuous")
    assert types["signup_date"] == ("DATE", "datetime")
    assert types["is_member"] == ("BOOLEAN", "boolean")
    assert types["zip_code"][0] == "VARCHAR"
    report = cleaning.report
    assert report is not None and report.rows_removed == 7 and report.cells_changed > 1000
    assert {c.column for c in report.type_changes} >= {"signup_date", "is_member"}
    rows = duckdb.execute(
        f"SELECT city, zip_code, customer_id FROM read_parquet('{child.path.as_posix()}')"
    ).fetchall()
    assert {r[0] for r in rows} == {"New York", "Paris", "London", "Berlin", "Madrid"}
    assert all(len(r[1]) == 5 for r in rows)
    assert not any(r[2].endswith(".0") for r in rows)


def test_suggestions_on_a_repeated_header(tmp_path: Path) -> None:
    text = "name,city\n" + "".join(f"n{i},Paris\n" for i in range(30)) + "name,city\nz,Lyon\n"
    ds = dx.from_pandas(
        __import__("pandas").read_csv(__import__("io").StringIO(text), dtype=str),
        cache_dir=tmp_path,
    )
    found = {s.id: s for s in ds.clean().suggestions}
    assert found["B-15:header"].rows == 1 and found["B-15:header"].mode == "A"


# --- Text helpers ------------------------------------------------------------------------------


def test_text_normalizations() -> None:
    assert normalize_spaces("  a\u00a0\t b  ") == "a b"
    assert remove_invisible("Mug\u200b\ufeff") == "Mug"
    assert fix_encoding("CafÃ©") == "Café"
    assert fix_encoding("Café") is None  # already right: left alone
    assert unescape_html("Tom &amp; Jerry") == "Tom & Jerry"
    assert unescape_html("AT&T") is None


def test_sql_and_python_normalize_the_same(tmp_path: Path) -> None:
    values = ["  a\u00a0\t b  ", "x\u3000y", "\u2003z"]
    path = tmp_path / "t.parquet"
    duckdb.execute("CREATE TABLE t AS SELECT unnest(?) AS v", [values])
    duckdb.execute(f"COPY t TO '{path.as_posix()}' (FORMAT parquet)")
    out = column(path, "v", {"op": "normalize_spaces", "columns": ["v"]})
    assert out == [normalize_spaces(v) for v in values]


# --- Steps -------------------------------------------------------------------------------------


@pytest.fixture
def table(tmp_path: Path) -> Path:
    return parquet(
        tmp_path,
        "id,name,amount,day,flag,zip\n"
        '1,  alice  smith ,"1,200.50",2024-01-05,yes,5207\n'
        "2,BOB,-3,05/02/2024,N,75001\n"
        "3,,n/a,,,\n"
        "4,carol,12,45000,1,123456\n",
    )


def test_text_steps(table: Path) -> None:
    assert (
        column(table, "name", {"op": "normalize_spaces", "columns": ["name"]})[0] == "alice smith"
    )
    assert (
        column(table, "name", {"op": "change_case", "columns": ["name"], "case": "upper"})[2]
        is None
    )
    titled = column(
        table,
        "name",
        {"op": "normalize_spaces", "columns": ["name"]},
        {"op": "change_case", "columns": ["name"], "case": "title"},
    )
    assert titled[:2] == ["Alice Smith", "Bob"]
    replaced = column(
        table,
        "name",
        {
            "op": "replace_values",
            "column": "name",
            "find": "bob",
            "replace": "Robert",
            "case_sensitive": False,
        },
    )
    assert replaced[1] == "Robert"
    contains = column(
        table,
        "name",
        {
            "op": "replace_values",
            "column": "name",
            "find": "ar",
            "replace": "AR",
            "match": "contains",
        },
    )
    assert contains[3] == "cARol"
    regex = column(
        table,
        "name",
        {
            "op": "replace_values",
            "column": "name",
            "find": "^c(.*)$",
            "replace": "C\\1",
            "match": "regex",
        },
    )
    assert regex[3] == "Carol"
    mapped = column(
        table, "name", {"op": "map_values", "column": "name", "map": {"BOB": "Bob", "carol": None}}
    )
    assert mapped[1:] == ["Bob", None, None]


def test_value_steps(table: Path) -> None:
    filled = column(table, "name", {"op": "fill", "column": "name", "direction": "down"})
    assert filled[2] == "BOB"
    up = column(table, "name", {"op": "fill", "column": "name", "direction": "up"})
    assert up[2] == "carol"
    assert column(table, "name", {"op": "fill_empty", "column": "name", "value": "?"})[2] == "?"
    tokens = column(
        table,
        "amount",
        {"op": "null_values", "column": "amount", "values": ["N/A"], "match": "token"},
    )
    assert tokens[2] is None and tokens[1] == "-3"
    numbers = column(table, "amount", {"op": "change_type", "column": "amount", "to": "number"})
    assert numbers == [1200.5, -3.0, None, 12.0]
    outside = column(
        table,
        "amount",
        {"op": "change_type", "column": "amount", "to": "number"},
        {"op": "null_outside", "column": "amount", "min": 0},
    )
    assert outside[1] is None and outside[0] == 1200.5
    padded = column(table, "zip", {"op": "pad_zeros", "column": "zip", "length": 5})
    assert padded == ["05207", "75001", None, "123456"]  # a longer value is never cut


def test_type_steps(table: Path) -> None:
    days = column(
        table,
        "day",
        {
            "op": "change_type",
            "column": "day",
            "to": "date",
            "formats": ["%Y-%m-%d", "%d/%m/%Y"],
            "excel_serial": True,
        },
    )
    assert [str(d) if d else None for d in days] == ["2024-01-05", "2024-02-05", None, "2023-03-15"]
    flags = column(table, "flag", {"op": "change_type", "column": "flag", "to": "boolean"})
    assert flags == [True, False, None, True]
    ids = column(table, "id", {"op": "change_type", "column": "id", "to": "text"})
    assert ids == ["1", "2", "3", "4"]
    integers = column(table, "amount", {"op": "change_type", "column": "amount", "to": "integer"})
    assert integers == [None, -3, None, 12]  # 1,200.50 is not a whole number


def test_text_of_an_integral_double(tmp_path: Path) -> None:
    path = parquet(tmp_path, "customer\n17850.0\n13047.0\n")
    assert column(path, "customer", {"op": "change_type", "column": "customer", "to": "text"}) == [
        "17850",
        "13047",
    ]


def test_column_steps(table: Path) -> None:
    names, _ = run(table, {"op": "rename_column", "column": "name", "to": "full name"})
    assert names[1] == "full name"
    names, _ = run(table, {"op": "drop_columns", "columns": ["zip", "flag"]})
    assert names == ["id", "name", "amount", "day"]
    names, rows = run(table, {"op": "duplicate_column", "column": "name", "to": "name 2"})
    assert names[2] == "name 2" and rows[1][2] == "BOB"


def test_row_steps(tmp_path: Path) -> None:
    path = parquet(tmp_path, "k,v,w\na,1,\na,1,\nb,2,x\na,3,y\n,,\nk,v,w\n")
    assert len(run(path, {"op": "drop_empty_rows"})[1]) == 5
    assert len(run(path, {"op": "drop_header_rows"})[1]) == 5
    assert len(run(path, {"op": "drop_duplicates"})[1]) == 5
    last = run(path, {"op": "drop_duplicates", "keys": ["k"], "keep": "last"})[1]
    assert [r[:2] for r in last] == [("b", "2"), ("a", "3"), (None, None), ("k", "v")]
    complete = run(path, {"op": "drop_duplicates", "keys": ["k"], "keep": "most_complete"})[1]
    assert ("a", "3", "y") in complete
    kept = run(
        path,
        {
            "op": "filter_rows",
            "conditions": [{"column": "k", "op": "eq", "value": "a"}],
            "action": "keep",
        },
    )[1]
    assert len(kept) == 3
    numbers = parquet(tmp_path, "x\n1\n5\n10\n", "numbers")
    removed = run(
        numbers, {"op": "filter_rows", "conditions": [{"column": "x", "op": "gt", "value": "4"}]}
    )[1]
    assert removed == [(1,)]  # compared as numbers: '10' > '4'
    any_of = run(
        numbers,
        {
            "op": "filter_rows",
            "combine": "any",
            "action": "keep",
            "conditions": [
                {"column": "x", "op": "in", "values": ["1", "10"]},
                {"column": "x", "op": "empty"},
            ],
        },
    )[1]
    assert any_of == [(1,), (10,)]


def test_a_step_that_cannot_run_is_reported_and_skipped(table: Path) -> None:
    steps = parse_steps(
        [
            {"id": "s1", "op": "rename_column", "column": "name", "to": "who"},
            {"id": "s2", "op": "normalize_spaces", "columns": ["name"]},  # renamed just before
            {"id": "s3", "op": "change_case", "columns": ["who"], "case": "lower"},
            {"id": "s4", "op": "null_outside", "column": "name", "max": 3, "enabled": False},
        ]
    ).steps
    connection = duckdb.connect()
    compiled = compile_steps(connection, str(table), steps)
    assert set(compiled.problems) == {"s2"} and "does not exist" in compiled.problems["s2"]
    assert [s.step.id for s in compiled.stages[1:]] == ["s1", "s3"]  # s4 is off
    with pytest.raises(CleanError, match="Step s2"):
        compile_steps(connection, str(table), steps, strict=True)


def test_steps_are_validated() -> None:
    with pytest.raises(CleanError, match="Unknown field"):
        parse_steps([{"id": "s1", "op": "drop_empty_rows", "sql": "DROP TABLE x"}])
    with pytest.raises(CleanError):
        parse_steps([{"id": "s1", "op": "run_python", "code": "print()"}])
    with pytest.raises(CleanError, match="same id"):
        parse_steps([{"id": "s1", "op": "drop_empty_rows"}, {"id": "s1", "op": "drop_header_rows"}])
    with pytest.raises(CleanError):
        parse_steps({"schema": 2, "steps": []})
    steps = parse_steps('{"schema": 1, "steps": [{"id": "s1", "op": "drop_empty_rows"}]}')
    assert json.loads(steps.to_json())["schema"] == 1


def test_values_reach_sql_as_literals(table: Path) -> None:
    attack = "x'); DROP TABLE t; --"
    out = column(table, "name", {"op": "fill_empty", "column": "name", "value": attack})
    assert out[2] == attack


# --- Preview and the child ---------------------------------------------------------------------


def test_preview_counts_on_every_row_and_shows_the_first_touched(tmp_path: Path) -> None:
    path = parquet(tmp_path, "a,b\n" + " x ,1\n" * 60 + "y,1\n" + "y,1\n")
    steps = parse_steps(
        [
            {"id": "s1", "op": "normalize_spaces", "columns": ["a"]},
            {"id": "s2", "op": "drop_duplicates"},
        ]
    ).steps
    connection = duckdb.connect()
    compiled = compile_steps(connection, str(path), steps)
    whole = preview(connection, compiled)
    assert (whole.rows_before, whole.rows_after, whole.rows_removed) == (62, 2, 60)
    assert whole.cells_changed == 1  # the row kept of the 60 " x "
    first = preview(connection, compiled, "s1")
    assert first.cells_changed == 60 and first.rows_removed == 0
    assert (
        len(first.sample) == 50
        and first.sample[0].before == [" x "]
        and first.sample[0].after == ["x"]
    )
    assert first.sample[0].changed == [True]
    second = preview(connection, compiled, "s2")
    assert second.rows_removed == 60 and second.sample[0].removed
    with pytest.raises(CleanError):
        preview(connection, compiled, "s9")


def test_apply_writes_the_child_and_its_rowmap(tmp_path: Path) -> None:
    path = parquet(tmp_path, "a\n" + "".join(f"{i % 3}\n" for i in range(9)))
    before = path.read_bytes()
    steps = parse_steps(
        [
            {
                "id": "s1",
                "op": "filter_rows",
                "conditions": [{"column": "a", "op": "eq", "value": "0"}],
            },
            {"id": "s2", "op": "drop_duplicates"},
        ]
    ).steps
    connection = duckdb.connect()
    target, rowmap = tmp_path / "child.parquet", tmp_path / "rowmap.parquet"
    report = apply_steps(connection, str(path), steps, target, rowmap)
    assert path.read_bytes() == before
    assert duckdb.execute(f"SELECT a FROM '{target.as_posix()}'").fetchall() == [(1,), (2,)]
    mapped = duckdb.execute(
        f"SELECT child_row, parent_row, step FROM '{rowmap.as_posix()}' ORDER BY parent_row"
    ).fetchall()
    assert mapped[:3] == [(None, 0, "s1"), (0, 1, None), (1, 2, None)]
    assert sum(1 for r in mapped if r[2] == "s2") == 4
    assert report.rows_removed == 7 and [s.rows_removed for s in report.steps] == [3, 4]
    assert not list(tmp_path.glob("*.partial"))


# --- Library -----------------------------------------------------------------------------------


def test_library_clean_use_skip_add(traps: dx.Dataset) -> None:
    cleaning = traps.clean()
    automatic = {s.suggestion for s in cleaning.steps}
    assert "B-03:city" in automatic and "B-19:country" not in automatic
    cleaning.use("B-19:country").skip("B-03:city")
    ids = [s.suggestion for s in cleaning.steps]
    assert ids[-1] == "B-19:country" and "B-03:city" not in ids
    cleaning.add(op="rename_column", column="city", to="town")
    assert cleaning.steps[-1].origin == "manual"
    assert len({s.id for s in cleaning.steps}) == len(cleaning.steps)
    assert json.loads(cleaning.to_json())["steps"][-1]["to"] == "town"
    assert "Cleaning" in repr(cleaning)


def test_suggestion_inserted_at_its_place_with_a_free_id(traps: dx.Dataset) -> None:
    cleaning = traps.clean()
    found = {s.id: s for s in cleaning.suggestions}
    steps = without_suggestion(cleaning.steps, "B-02:product")
    manual = parse_steps([{"id": found["B-02:product"].step.id, "op": "drop_empty_rows"}]).steps  # type: ignore[union-attr]
    steps = [*steps, *manual]
    back = with_suggestion(steps, found["B-02:product"], cleaning.suggestions)
    position = [s.suggestion for s in back].index("B-02:product")
    assert position == 1  # after B-15, before B-01, as C-06 orders them
    assert len({s.id for s in back}) == len(back)


def test_child_keeps_the_users_corrections(tmp_path: Path) -> None:
    ds = dx.from_pandas(
        __import__("pandas").DataFrame({"code": ["a ", "b", "c"] * 10, "n": list(range(30))}),
        cache_dir=tmp_path,
    )
    ds.set_role("n", "quantity").exclude("code")
    child = ds.clean().apply(cache_dir=tmp_path)
    assert child.column("n").role == "quantity"
    assert child.column("code").excluded
