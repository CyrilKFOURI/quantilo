"""Clean, C1: the rest of the M5 batch. Dates (a shift, a time zone), the first or last rows, the
first row as header, split into rows, and the structure steps (unpivot, pivot, group by, append,
merge) that reshape the table (C-59: rows numbered again, no row map, an honest preview)."""

import csv
from pathlib import Path
from typing import Any

import duckdb
import pytest

import quantilo as dx
from quantilo.cleaning import compile_steps, model
from quantilo.cleaning.preview import preview
from quantilo.cleaning.replay import made_by_steps, renamed_step, step_columns


def write(tmp_path: Path, rows: list[list[Any]], name: str = "data") -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return dx.open(path, cache_dir=tmp_path / f"cache-{name}")


def steps(*raw: dict[str, Any]) -> list[Any]:
    return model.parse_steps([{"id": f"s{i}", **r} for i, r in enumerate(raw, 1)]).steps


def run(
    ds: dx.Dataset, todo: list[Any], sql: str, sources: dict[str, str] | None = None
) -> list[tuple[Any, ...]]:
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(ds.path), todo, strict=True, sources=sources)
        return connection.execute(sql.replace("{t}", f"({compiled.sql()})")).fetchall()
    finally:
        connection.close()


def compiled_of(ds: dx.Dataset, todo: list[Any], sources: dict[str, str] | None = None) -> Any:
    connection = duckdb.connect()
    return connection, compile_steps(connection, str(ds.path), todo, sources=sources)


SALES = [
    ["region", "month", "amount", "when"],
    ["North", "Jan", "10", "2024-03-30 12:00:00"],
    ["North", "Feb", "20", "2024-03-31 12:00:00"],
    ["South", "Jan", "5", "2024-07-01 12:00:00"],
    ["South", "Feb", "7", "2024-12-01 08:30:00"],
    ["South", "Feb", "3", "2024-12-02 08:30:00"],
]


def test_dates_shift_and_change_time_zone(tmp_path: Path) -> None:
    ds = write(tmp_path, SALES)
    shifted = run(
        ds,
        steps({"op": "shift_date", "column": "when", "amount": -2, "unit": "hours"}),
        "SELECT strftime(\"when\", '%H:%M') FROM {t} ORDER BY __quantilo_row",
    )
    assert [r[0] for r in shifted][:2] == ["10:00", "10:00"]
    zoned = run(
        ds,
        steps(
            {
                "op": "convert_timezone",
                "column": "when",
                "from_zone": "Europe/Paris",
                "to_zone": "UTC",
            }
        ),
        "SELECT strftime(\"when\", '%H:%M') FROM {t} ORDER BY __quantilo_row",
    )
    # 30 March: winter time (UTC+1); 31 March: summer time (UTC+2), each zone's own rules
    assert [r[0] for r in zoned] == ["11:00", "10:00", "10:00", "07:30", "07:30"]
    connection, compiled = compiled_of(
        ds, steps({"op": "convert_timezone", "column": "when", "to_zone": "Mars/Olympus"})
    )
    assert "Unknown time zone 'Mars/Olympus'" in compiled.problems["s1"]
    connection, compiled = compiled_of(
        ds, steps({"op": "shift_date", "column": "region", "amount": 1})
    )
    assert "not a date" in compiled.problems["s1"]
    connection.close()


def test_first_or_last_rows_kept_or_removed(tmp_path: Path) -> None:
    ds = write(tmp_path, SALES)
    for how, expected in (
        ("first", [10, 20]),
        ("last", [7, 3]),
        ("remove_first", [5, 7, 3]),
        ("remove_last", [10, 20, 5]),
    ):
        rows = run(
            ds,
            steps({"op": "keep_rows", "how": how, "count": 2}),
            "SELECT amount FROM {t} ORDER BY __quantilo_row",
        )
        assert [r[0] for r in rows] == expected, how


def test_first_row_promoted_to_header(tmp_path: Path) -> None:
    ds = write(
        tmp_path, [["c1", "c2", "c3"], ["Name", "", "Name"], ["Ann", "3", "x"], ["Bob", "4", "y"]]
    )
    connection, compiled = compiled_of(ds, steps({"op": "promote_header"}))
    assert compiled.final.names == ["Name", "c2", "Name 2"]
    found = preview(connection, compiled)
    assert found.rows_removed == 1 and found.reshaped is None
    connection.close()


def test_split_into_rows_keeps_empty_cells(tmp_path: Path) -> None:
    ds = write(tmp_path, [["order", "tags"], ["A", "red, blue"], ["B", ""], ["C", "green"]])
    todo = steps({"op": "split_rows", "column": "tags", "delimiter": ","})
    rows = run(ds, todo, 'SELECT "order", tags, __quantilo_row FROM {t} ORDER BY __quantilo_row')
    assert rows == [("A", "red", 0), ("A", "blue", 1), ("B", None, 2), ("C", "green", 3)]
    connection, compiled = compiled_of(ds, todo)
    shown = preview(connection, compiled)
    assert shown.reshaped == "s1" and (shown.rows_before, shown.rows_after) == (3, 4)
    assert shown.cells_changed == 0 and len(shown.sample) == 4
    connection.close()


def test_unpivot_pivot_and_group_by(tmp_path: Path) -> None:
    wide = write(tmp_path, [["id", "q1", "q2"], ["a", "1", "2"], ["b", "3", ""]], "wide")
    long = run(
        wide,
        steps(
            {"op": "unpivot", "columns": ["q1", "q2"], "names_to": "quarter", "values_to": "sales"}
        ),
        "SELECT id, quarter, sales FROM {t} ORDER BY __quantilo_row",
    )
    assert long == [("a", "q1", 1), ("a", "q2", 2), ("b", "q1", 3), ("b", "q2", None)]
    dropped = run(
        wide,
        steps({"op": "unpivot", "columns": ["q1", "q2"], "drop_empty": True}),
        "SELECT count(*) FROM {t}",
    )
    assert dropped == [(3,)]
    ds = write(tmp_path, SALES)
    pivoted = run(
        ds,
        steps(
            {"op": "drop_columns", "columns": ["when"]},
            {"op": "pivot", "column": "month", "value_column": "amount", "aggregate": "sum"},
        ),
        "SELECT region, Jan, Feb FROM {t} ORDER BY __quantilo_row",
    )
    assert pivoted == [("North", 10, 20), ("South", 5, 10)]
    grouped = run(
        ds,
        steps(
            {
                "op": "group_by",
                "keys": ["region"],
                "aggregations": [
                    {"fn": "count", "to": "rows"},
                    {"column": "amount", "fn": "sum", "to": "total"},
                    {"column": "month", "fn": "last", "to": "last month"},
                ],
            }
        ),
        'SELECT region, "rows", total, "last month" FROM {t} ORDER BY __quantilo_row',
    )
    assert grouped == [("North", 2, 30, "Feb"), ("South", 3, 15, "Feb")]
    connection, compiled = compiled_of(
        ds,
        steps(
            {
                "op": "group_by",
                "keys": ["region"],
                "aggregations": [{"column": "month", "fn": "mean", "to": "m"}],
            }
        ),
    )
    assert "not a number column" in compiled.problems["s1"]
    connection.close()


def test_append_and_merge_other_datasets(tmp_path: Path) -> None:
    ds = write(tmp_path, SALES)
    more = write(tmp_path, [["region", "amount", "note"], ["West", "9", "new"]], "more")
    regions = write(tmp_path, [["code", "manager"], ["North", "Ann"], ["South", "Bob"]], "regions")
    twice = write(tmp_path, [["code", "manager"], ["North", "Ann"], ["North", "Al"]], "twice")
    sources = {"more": str(more.path), "regions": str(regions.path), "twice": str(twice.path)}
    appended = run(
        ds,
        steps(
            {
                "op": "append_rows",
                "dataset": "more",
                "label": "more",
                "source_column": "from",
                "this_label": "sales",
            }
        ),
        'SELECT region, amount, note, "from" FROM {t} ORDER BY __quantilo_row',
        sources,
    )
    assert len(appended) == 6 and appended[-1] == ("West", 9, "new", "more")
    assert appended[0][2:] == (None, "sales")
    merged = run(
        ds,
        steps(
            {
                "op": "merge_dataset",
                "dataset": "regions",
                "keys": ["region"],
                "other_keys": ["code"],
                "add": ["manager"],
            }
        ),
        "SELECT region, manager FROM {t} ORDER BY __quantilo_row",
        sources,
    )
    assert [r[1] for r in merged] == ["Ann", "Ann", "Bob", "Bob", "Bob"]
    connection, compiled = compiled_of(
        ds,
        steps(
            {
                "op": "merge_dataset",
                "dataset": "twice",
                "label": "twice",
                "keys": ["region"],
                "other_keys": ["code"],
            }
        ),
        sources,
    )
    assert "rows would multiply" in compiled.problems["s1"]
    connection, compiled = compiled_of(ds, steps({"op": "append_rows", "dataset": "gone"}), sources)
    assert "not available" in compiled.problems["s1"]
    # a merge keeps the rows one by one: no reshape, the row map stays valid
    connection, compiled = compiled_of(
        ds,
        steps(
            {
                "op": "merge_dataset",
                "dataset": "regions",
                "keys": ["region"],
                "other_keys": ["code"],
            }
        ),
        sources,
    )
    assert compiled.reshaped() is None
    connection.close()


def test_a_reshape_numbers_rows_again_and_saves_no_row_map(tmp_path: Path) -> None:
    ds = write(tmp_path, SALES)
    cleaning = ds.clean(steps=[])
    cleaning.add(op="group_by", keys=["region"], aggregations=[{"fn": "count", "to": "n"}])
    cleaning.add(op="drop_rows", rows=[0])
    connection = duckdb.connect()
    compiled = compile_steps(connection, str(ds.path), cleaning.steps)
    assert "reshaped the rows" in compiled.problems["s2"]
    connection.close()
    cleaning.steps = cleaning.steps[:1]
    child = cleaning.apply(cache_dir=tmp_path / "out")
    assert len(child) == 2 and cleaning.rowmap is None
    assert cleaning.report is not None and cleaning.report.reshaped_by == "s1"
    kept = ds.clean(steps=[]).add(op="keep_rows", how="first", count=2)
    kept.apply(cache_dir=tmp_path / "out2")
    assert kept.rowmap is not None and kept.rowmap.is_file()


def test_replay_follows_the_columns_of_structure_steps() -> None:
    [group, pivot] = steps(
        {
            "op": "group_by",
            "keys": ["city"],
            "aggregations": [{"column": "price", "fn": "sum", "to": "total"}],
        },
        {"op": "pivot", "column": "month", "value_column": "total", "headers": ["Jan"]},
    )
    assert step_columns(group) == ["city", "price"]
    moved = renamed_step(group, {"city": "town", "price": "amount"})
    assert moved.keys == ["town"] and moved.aggregations[0].column == "amount"
    assert step_columns(pivot) == ["month", "total"]
    assert {"total", "Jan"} <= made_by_steps([group, pivot])


def test_new_ops_have_an_effect_and_a_kind() -> None:
    for op in (
        "shift_date",
        "convert_timezone",
        "keep_rows",
        "promote_header",
        "split_rows",
        "unpivot",
        "pivot",
        "group_by",
        "append_rows",
        "merge_dataset",
    ):
        assert op in model.OPS and op in model.OP_IMPACT
    assert set(model.OPS) >= model.RESHAPE_OPS
    with pytest.raises(model.CleanError, match="two different"):
        steps({"op": "convert_timezone", "column": "a", "from_zone": "UTC", "to_zone": "UTC"})
    with pytest.raises(model.CleanError, match="as many keys"):
        steps({"op": "merge_dataset", "dataset": "x", "keys": ["a"], "other_keys": ["a", "b"]})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("07:40", "07:40"),
        ("7h40", "07:40"),
        ("0740", "07:40"),
        ("740", "07:40"),
        ("5", "00:05"),
        ("23:59:00", "23:59"),
        ("25:00", None),
        ("noon", None),
    ],
)
def test_time_of_day_joins_the_assembled_date(
    tmp_path: Path, value: str, expected: str | None
) -> None:
    ds = write(tmp_path, [["an", "mois", "jour", "hrmn"], ["2024", "3", "9", value]])
    rows = run(
        ds,
        steps(
            {
                "op": "assemble_date",
                "year": "an",
                "month": "mois",
                "day": "jour",
                "time": "hrmn",
                "to": "date",
            }
        ),
        "SELECT strftime(\"date\", '%H:%M') FROM {t}",
    )
    assert rows == [(expected,)]


def test_b20_reads_an_hour_column_written_as_a_time(tmp_path: Path) -> None:
    ds = write(
        tmp_path,
        [
            ["annee", "mois", "jour", "heure"],
            *[["2024", "3", str(d), f"{d:02d}:15"] for d in range(1, 21)],
        ],
    )
    found = next(s for s in ds.clean().suggestions if s.rule == "B-20")
    step = found.step
    assert isinstance(step, model.AssembleDate) and step.time == "heure" and step.hour is None
