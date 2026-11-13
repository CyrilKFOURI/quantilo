"""Clean, audit of 2026-11-12: N05 (don't suggest again), N31 (declared checks), N45 (cleaning as
SQL and Python), N46 (the cleaning journal)."""

import csv
import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

import quantilo as dx
from quantilo.cleaning import dismiss, parse_recipe
from quantilo.cleaning.checks import failing_rows
from quantilo.cleaning.suggestion import CleanSuggestion


def write(tmp_path: Path, rows: list[dict[str, Any]], name: str) -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dx.open(path, cache_dir=tmp_path / "cache")


def orders(count: int = 60) -> list[dict[str, Any]]:
    return [
        {
            "order_id": f"O{i:04d}",
            "city": " Paris" if i % 3 == 0 else "Lyon",
            "brand": "Acme " if i % 4 == 0 else "Acme",
            "notes": "",
            "amount": f"{10 + i % 7}.50",
        }
        for i in range(count)
    ]


def suggestion(rule: str, sid: str, columns: list[str]) -> CleanSuggestion:
    return CleanSuggestion(id=sid, rule=rule, mode="A", title="t", reason="r", columns=columns)


# --- N05 ---------------------------------------------------------------------------------------


def test_what_a_dismissal_hides() -> None:
    city = suggestion("B-01", "B-01:city", ["city"])
    brand = suggestion("B-01", "B-01:brand", ["brand"])
    empty = suggestion("B-19", "B-19:notes", ["notes"])
    one = dismiss.Dismissal(scope="suggestion", rule="B-01", suggestion="B-01:city")
    rule = dismiss.Dismissal(scope="rule", rule="B-01", column="city")
    column = dismiss.Dismissal(scope="column", rule="B-01", column="brand", suggestion="x")
    assert [dismiss.matches(one, s) for s in (city, brand, empty)] == [True, False, False]
    assert [dismiss.matches(rule, s) for s in (city, brand, empty)] == [True, True, False]
    assert rule.column is None and column.suggestion is None  # the reach says what is kept
    assert [dismiss.matches(column, s) for s in (city, brand, empty)] == [False, True, False]
    shown, hidden = dismiss.split([city, brand, empty], [one, column])
    assert [s.id for s in shown] == ["B-19:notes"]
    assert [(s.id, i) for s, i in hidden] == [("B-01:city", 0), ("B-01:brand", 1)]
    assert "every dataset" in dismiss.reason(dismiss.Dismissal(scope="everywhere", rule="B-19"))
    with pytest.raises(ValueError, match="needs its id"):
        dismiss.Dismissal(scope="suggestion", rule="B-01")


def test_the_recipe_carries_them_and_the_replay_keeps_them_hidden(tmp_path: Path) -> None:
    september = write(tmp_path, orders(), "september")
    cleaning = september.clean()
    assert any(s.suggestion == "B-01:city" for s in cleaning.steps)
    cleaning.dismiss("B-01:city").dismiss("B-19:notes", scope="rule", note="always empty")
    assert not any(s.suggestion == "B-01:city" for s in cleaning.steps)  # out of the steps
    assert {s.id for s, _ in cleaning.hidden()} == {"B-01:city", "B-19:notes"}
    recipe = cleaning.recipe("September")
    raw = json.loads(recipe.to_json())
    assert raw["schema"] == 2 and [d["scope"] for d in raw["dismissed"]] == ["suggestion", "rule"]
    assert parse_recipe(raw).dismissed == recipe.dismissed

    october = write(tmp_path, orders(45), "october")
    other = october.clean()
    report = other.replay(recipe)
    assert report.hidden == 2 and len(other.dismissed) == 2
    assert not {"B-01:city", "B-19:notes"} & {s.id for s in report.new_problems}
    plain = september.clean()
    assert json.loads(plain.recipe("plain").to_json())["schema"] == 1  # nothing new: schema 1


# --- N31 ---------------------------------------------------------------------------------------


def tickets(count: int = 120, wrong: int = 9) -> list[dict[str, Any]]:
    """Service requests: `closed` before `created` on `wrong` rows (nyc_311), one city spelled
    'Pariss', one repeated id, totals that are their parts but one."""
    rows = []
    for i in range(count):
        day = 1 + i % 20
        closed = day + 2 if i >= wrong else day - 1
        rows.append(
            {
                "ticket": f"T{i:04d}" if i != 7 else "T0006",
                "created": f"2024-03-{day:02d}",
                "closed": f"2024-03-{closed:02d}" if closed > 0 else "2024-02-28",
                "city": "Pariss" if i == 50 else ("Paris" if i % 2 else "Lyon"),
                "country": "France",
                "status": "open" if i % 3 else "closed",
                "zip": f"{75000 + i % 20:05d}" if i != 60 else "75A01",
                "a": str(i % 7),
                "b": str(i % 5),
                "total": str(i % 7 + i % 5 + (3 if i == 33 else 0)),
                "notes": "" if i in (4, 5) else "ok",
            }
        )
    return rows


def test_declared_checks_pass_fail_and_say_why(tmp_path: Path) -> None:
    ds = write(tmp_path, tickets(), "tickets")
    cleaning = ds.clean()
    cleaning.check("compare", "created", "closed", op="le")
    cleaning.check("unique", "ticket")
    cleaning.check("in_list", "city", values=["Paris", "Lyon"])
    cleaning.check("not_empty", "notes")
    cleaning.check("between", "b", min="0", max="4")
    cleaning.check("pattern", "zip", pattern=r"[0-9]{5}")
    cleaning.check("sum", "total", "a", "b")
    cleaning.check("determines", "city", "country")
    cleaning.check(
        "condition", conditions=[{"column": "status", "op": "eq", "value": "lost"}], expect="none"
    )
    cleaning.check("not_empty", "nope")
    report = cleaning.verify()
    by_id = {r.id: r for r in report.results}
    assert by_id["c1"].status == "fail" and by_id["c1"].failing == 9  # nyc_311: 9 rows
    assert by_id["c1"].examples == [0, 1, 2, 3, 4] and by_id["c1"].description == "created ≤ closed"
    assert (by_id["c2"].failing, by_id["c3"].failing, by_id["c4"].failing) == (2, 1, 2)
    assert by_id["c5"].status == "pass" and by_id["c8"].status == "pass"
    assert (by_id["c6"].failing, by_id["c7"].failing, by_id["c9"].status) == (1, 1, "pass")
    assert by_id["c10"].status == "error" and "not a column" in (by_id["c10"].message or "")
    assert (report.passed, report.failed, report.errors) == (3, 6, 1)
    assert report.failing_rows == len({0, 1, 2, 3, 4, 5, 6, 7, 8, 50, 60, 33})  # 4 and 5 included
    again = ds.clean()
    again.checks = cleaning.checks
    assert again.verify().results == report.results  # the same on each run


def test_the_failing_rows_and_the_recipe_runs_them_where_it_is_replayed(tmp_path: Path) -> None:

    ds = write(tmp_path, tickets(), "march")
    cleaning = ds.clean().add(op="normalize_spaces", columns=["city"])
    cleaning.check("compare", "created", "closed")
    connection = duckdb.connect()
    found = failing_rows(connection, str(ds.path), ds.types, cleaning.checks[0])
    assert found.failing == 9 and found.columns[:2] == ["created", "closed"]
    assert found.rows[0].row == 0 and found.rows[0].values[:2] == ["2024-03-01", "2024-02-28"]
    recipe = cleaning.recipe("March")
    assert json.loads(recipe.to_json())["schema"] == 2
    renamed = [
        {("Created" if k == "created" else k): v for k, v in r.items()} for r in tickets(60, 4)
    ]
    april = write(tmp_path, renamed, "april")
    replayed = april.clean()
    report = replayed.replay(recipe)
    assert report.checks[0].columns == ["Created", "closed"]  # paired like the steps
    assert report.check_report is not None and report.check_report.results[0].failing == 4
    assert replayed.checks == report.checks


# --- N45 ---------------------------------------------------------------------------------------


def same_rows(a: Path, b: Path) -> None:
    """Two Parquet files hold the same columns, types and rows, in the same order."""
    connection = duckdb.connect()

    def read(path: Path) -> tuple[list[Any], list[Any]]:
        source = f"read_parquet('{path.as_posix()}', file_row_number = true)"
        described = connection.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
        rows = connection.execute(
            f"SELECT * EXCLUDE (file_row_number) FROM {source} ORDER BY file_row_number"
        ).fetchall()
        return [(d[0], d[1]) for d in described if d[0] != "file_row_number"], rows

    assert read(a) == read(b)


def test_the_sql_and_python_exports_make_the_same_rows(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    from quantilo.cleaning.apply import apply_steps
    from quantilo.cleaning.export import ExportInfo, to_python, to_sql

    ds = write(tmp_path, tickets(), "orders")
    cleaning = ds.clean()
    cleaning.add(op="normalize_spaces", columns=["city"])
    cleaning.add(op="map_values", column="city", map={"Pariss": "Paris"})
    cleaning.add(op="filter_rows", conditions=[{"column": "notes", "op": "empty"}])
    cleaning.add(op="rename_column", column="status", to="state", enabled=False)
    cleaning.add(op="change_type", column="total", to="integer")
    connection = duckdb.connect()
    saved = tmp_path / "saved.parquet"
    apply_steps(connection, str(ds.path), cleaning.steps, saved, tmp_path / "rowmap.parquet")
    info = ExportInfo("orders · Cleaned v1", "orders", titles={"s2": "Merge 'Pariss' into Paris"})

    sql = to_sql(connection, str(ds.path), cleaning.steps, info, target=str(tmp_path / "a.parquet"))
    assert "-- t2: s2 · applied" not in sql and "-- t2: s2 · added by the user · Merge" in sql
    assert "-- s4 · added by the user · rename_column on status: turned off, not run." in sql
    assert "4 added by hand; 1 turned off" in sql.replace("5 steps: 0 applied automatically", "")
    duckdb.connect().execute(sql)
    same_rows(tmp_path / "a.parquet", saved)

    script = tmp_path / "cleaning.py"
    script.write_text(to_python(str(ds.path), cleaning.steps, info), encoding="utf-8")
    env = {**os.environ, "QUANTILO_CACHE": str(tmp_path / "py")}
    done = subprocess.run(
        [sys.executable, str(script), str(ds.path), str(tmp_path / "b.parquet")],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    same_rows(tmp_path / "b.parquet", saved)


def test_names_and_values_reach_the_code_as_literals(tmp_path: Path) -> None:
    import ast

    from quantilo.cleaning.export import ExportInfo, to_python, to_sql

    evil = "x'); DROP TABLE t; --"
    lines = f"{evil}\nimport os"  # a line break in a comment would start code
    ds = write(tmp_path, [{"a": evil, evil: "1"}, {"a": "b", evil: "2"}], "evil")
    cleaning = ds.clean().add(op="replace_values", column="a", find="b", replace=lines)
    info = ExportInfo(name='Bad """\nname', made_on=lines, titles={"s1": lines})
    target = str(tmp_path / "e.parquet")
    sql = to_sql(duckdb.connect(), str(ds.path), cleaning.steps, info, target=target)
    duckdb.connect().execute(sql)  # runs: every name and value is quoted
    written = duckdb.connect().execute(f"SELECT * FROM '{(tmp_path / 'e.parquet').as_posix()}'")
    assert lines in {value for row in written.fetchall() for value in row}  # a value, as given
    header = [line for line in sql.splitlines() if line.startswith("--")]
    assert any('Bad """ name' in line for line in header)  # the line break became a space
    code = to_python(str(ds.path), cleaning.steps, info)
    imports = [n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Import | ast.ImportFrom)]
    assert sorted(a.name for n in imports for a in n.names) == ["quantilo", "shutil", "sys"]


def test_a_name_never_declares_the_script_encoding(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    from quantilo.cleaning.export import ENCODING_LINE, ExportInfo, to_python, to_sql

    name = "Sales coding=utf-7 +AAo-import os"
    ds = write(tmp_path, orders(10), "sales")
    cleaning = ds.clean().add(op="normalize_spaces", columns=["city"])
    info = ExportInfo(name=name, made_on=name, titles={"s1": name})
    code = to_python(str(ds.path), cleaning.steps, info)
    lines = code.splitlines()
    assert lines[0] == ENCODING_LINE
    comments = [line for line in lines[1:] if line.lstrip().startswith("#")]
    assert not any("coding=" in line or "coding:" in line for line in comments)
    compile(code.encode("utf-8"), "cleaning.py", "exec")  # read as UTF-8, as written
    script = tmp_path / "cleaning.py"
    script.write_bytes(code.encode("utf-8"))
    env = {**os.environ, "QUANTILO_CACHE": str(tmp_path / "py")}
    target = str(tmp_path / "out.parquet")
    done = subprocess.run(
        [sys.executable, str(script), str(ds.path), target],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    sql = to_sql(duckdb.connect(), str(ds.path), cleaning.steps, info, target=target)
    assert not any("coding=" in line for line in sql.splitlines() if line.startswith("--"))


# --- N46 -------------------------------------------------------------------------------------


def test_the_journal_diffs_the_steps_and_writes_a_log() -> None:
    from quantilo.cleaning import model
    from quantilo.cleaning.journal import (
        CleanJournal,
        Known,
        diff,
        journal_steps,
        to_html,
        to_markdown,
    )
    from quantilo.cleaning.report import CleanReport, StepReport

    def step(raw: dict[str, Any]) -> Any:
        return model.parse_steps(json.dumps({"steps": [raw]})).steps[0]

    trim = step(
        {
            "id": "s1",
            "op": "normalize_spaces",
            "columns": ["city"],
            "origin": "auto",
            "rule": "B-01",
            "suggestion": "B-01:city",
        }
    )
    drop = step({"id": "s2", "op": "drop_empty_rows", "origin": "suggested", "rule": "B-15"})
    name = step({"id": "s3", "op": "rename_column", "column": "city", "to": "town"})
    known = {"B-01:city": Known("Trim extra spaces in city", 52, 0, (" Paris", "Paris"))}
    entries = diff(
        [trim, drop], [name, drop.model_copy(update={"enabled": False})], "t", "user", known
    )
    assert [(e.action, e.step_id) for e in entries] == [
        ("added", "s3"),
        ("turned_off", "s2"),
        ("removed", "s1"),
    ]
    assert entries[2].detail == "Trim extra spaces in city" and entries[2].cells == 52
    moved = diff([trim, drop], [drop, trim], "t", "user")
    assert [e.action for e in moved] == ["moved"]

    report = CleanReport(
        10,
        8,
        2,
        52,
        3,
        3,
        {"city": 52},
        [StepReport("s2", "drop_empty_rows", "B-15", "suggested", 2)],
    )
    found = journal_steps([trim, drop], report, known)
    journal = CleanJournal(
        "orders <b> · Cleaned v1", "d2", "orders", "2026-11-13", "user", found, report, entries
    )
    text = to_markdown(journal)
    assert (
        "| s1 | B-01 | applied automatically (a reliable fix) | Trim extra spaces in city | 52 |"
        in text
    )
    assert "2 rows by s2" in text and "you removed s1" in text
    page = to_html(journal)
    assert "&lt;b&gt;" in page and "<b>" not in page
