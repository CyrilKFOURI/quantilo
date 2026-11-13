"""Clean N35: the review of duplicates by group. The groups of a removal of duplicates (B-14,
B-23, A-15 by their keys, A-16 listed by row), and the step the review writes: a row chosen in
its group, a group said not duplicates, the kept row's empty cells filled from its group; replayed
on another file and exported as SQL."""

import csv
import random
from pathlib import Path
from typing import Any

import duckdb
import pytest

import quantilo as dx
from quantilo.cleaning import compile_steps, model
from quantilo.cleaning.duplicates import duplicate_groups
from quantilo.cleaning.export import ExportInfo, to_sql
from quantilo.cleaning.replay import REVIEW_ELSEWHERE, ROWS_ELSEWHERE

ROWS = [
    # id, name, city, updated
    {"id": "A1", "name": "Ada", "city": "Paris", "updated": "2024-01-01"},
    {"id": "A1", "name": "Ada", "city": "", "updated": "2024-03-01"},
    {"id": "B2", "name": "Alan", "city": "Lyon", "updated": "2024-02-01"},
    {"id": "C3", "name": "Grace", "city": "", "updated": ""},
    {"id": "C3", "name": "Grace H", "city": "Nice", "updated": "2024-05-01"},
    {"id": "C3", "name": "Grace", "city": "Nice", "updated": "2024-04-01"},
    {"id": "D4", "name": "Linus", "city": "Oslo", "updated": "2024-01-09"},
]


def write(tmp_path: Path, rows: list[dict[str, Any]], name: str = "data") -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dx.open(path, cache_dir=tmp_path / f"cache-{name}")


def steps(*raw: dict[str, Any]) -> list[Any]:
    return model.parse_steps([{"id": f"s{i}", **r} for i, r in enumerate(raw, 1)]).steps


def run(ds: dx.Dataset, found: list[Any]) -> list[tuple[Any, ...]]:
    """The cleaned rows: (row number, id, name, city)."""
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(ds.path), found, strict=True)
        return connection.execute(
            f"SELECT __quantilo_row, id, name, city FROM ({compiled.sql()}) ORDER BY 1"
        ).fetchall()
    finally:
        connection.close()


def groups_of(ds: dx.Dataset, found: list[Any], step_id: str, **page: int) -> Any:
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(ds.path), found)
        return duplicate_groups(connection, compiled, step_id, **page)
    finally:
        connection.close()


KEYED = {"op": "drop_duplicates", "keys": ["id"], "keep": "first"}


def test_the_rule_alone_is_unchanged(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    assert [r[0] for r in run(ds, steps(KEYED))] == [0, 2, 3, 6]


def test_groups_side_by_side_with_each_rule_survivor(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    found = groups_of(ds, steps({**KEYED, "keep": "latest", "order_by": "updated"}), "s1")
    assert (found.total_groups, found.total_rows, found.removed) == (2, 5, 3)
    assert found.columns == ["id", "name", "city", "updated"] and found.dates == ["updated"]
    first, second = found.groups
    assert [r.row for r in first.rows] == [0, 1] and first.order == [1, 0]  # the latest first
    assert first.first == 0 and first.most_complete == 0 and first.latest == {"updated": 1}
    assert first.differing == [2, 3]  # city and updated differ, not id and name
    assert first.rows[1].cells[2] is None  # an empty cell is empty, not ''
    assert second.most_complete == 4 and second.latest == {"updated": 4} and second.order[0] == 4
    assert (first.size, second.size) == (2, 3)
    paged = groups_of(ds, steps(KEYED), "s1", offset=1, limit=1)
    assert [g.first for g in paged.groups] == [3] and paged.offset == 1


def test_a_chosen_row_is_kept_instead_of_the_rules(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    found = run(ds, steps({**KEYED, "survivors": [5]}))
    assert [r[0] for r in found] == [0, 2, 5, 6]


def test_a_group_said_not_duplicates_keeps_every_row(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    found = run(ds, steps({**KEYED, "not_duplicates": [[3, 4, 5]]}))
    assert [r[0] for r in found] == [0, 2, 3, 4, 5, 6]


def test_merge_fills_the_kept_rows_empty_cells_from_its_group(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    # C3: the first row (3) has no city; merged, it takes Nice from row 4 (the next by the rule)
    merged = run(ds, steps({**KEYED, "merge": True}))
    assert merged == [
        (0, "A1", "Ada", "Paris"),
        (2, "B2", "Alan", "Lyon"),
        (3, "C3", "Grace", "Nice"),
        (6, "D4", "Linus", "Oslo"),
    ]
    # a group said not duplicates is left as it is, merge or not
    apart = run(ds, steps({**KEYED, "merge": True, "not_duplicates": [[3, 4, 5]]}))
    assert (3, "C3", "Grace", None) in apart


def test_listed_groups_a16(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    listed = {"op": "drop_duplicates", "groups": [[2, 6]], "keep": "most_complete"}
    assert [r[0] for r in run(ds, steps(listed))] == [0, 1, 2, 3, 4, 5]
    assert [r[0] for r in run(ds, steps({**listed, "survivors": [6]}))] == [0, 1, 3, 4, 5, 6]
    assert len(run(ds, steps({**listed, "not_duplicates": [[2, 6]]}))) == 7
    found = groups_of(ds, steps(listed), "s1")
    assert found.keep == "listed" and found.total_groups == 1 and found.groups[0].order == [2, 6]


def test_the_model_refuses_what_cannot_be_a_review() -> None:
    for bad in (
        {"op": "drop_duplicates", "groups": [[1]]},
        {"op": "drop_duplicates", "groups": [[1, 2], [2, 3]]},
        {"op": "drop_duplicates", "survivors": [1, 1]},
    ):
        with pytest.raises(model.CleanError):
            steps(bad)
    assert model.names_rows(steps({"op": "drop_duplicates", "groups": [[1, 2]]})[0])
    assert not model.names_rows(steps(KEYED)[0])
    assert model.reviewed(steps({**KEYED, "survivors": [1]})[0])


def test_replay_applies_the_rule_alone_and_says_so(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    other = write(tmp_path, [dict(r) for r in reversed(ROWS)], "other")
    found = steps(
        {**KEYED, "survivors": [5], "merge": True}, {"op": "drop_duplicates", "groups": [[0, 6]]}
    )
    report = other.clean().replay(ds.clean(found).recipe())
    by = {s.id: s for s in report.steps}
    assert by["s1"].status == "applies" and by["s1"].message == REVIEW_ELSEWHERE
    assert by["s2"].status == "skipped" and by["s2"].message == ROWS_ELSEWHERE
    adapted = {s.id: s for s in report.adapted}
    assert adapted["s1"].survivors is None and adapted["s1"].merge is True  # type: ignore[union-attr]
    assert adapted["s2"].enabled is False


def test_the_exported_sql_gives_the_reviewed_rows(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    found = steps({**KEYED, "survivors": [5], "not_duplicates": [[0, 1]], "merge": True})
    target = (tmp_path / "out.parquet").as_posix()
    connection = duckdb.connect()
    try:
        script = to_sql(connection, str(ds.path), found, ExportInfo(name="t"), target=target)
        connection.execute(script)
        rows = connection.execute(f"SELECT id, name, city FROM read_parquet('{target}')").fetchall()
    finally:
        connection.close()
    assert rows == [
        ("A1", "Ada", "Paris"),
        ("A1", "Ada", None),
        ("B2", "Alan", "Lyon"),
        ("C3", "Grace", "Nice"),
        ("D4", "Linus", "Oslo"),
    ]


def test_a16_lists_its_groups_the_kept_row_first(tmp_path: Path) -> None:
    from datetime import date, timedelta

    rng = random.Random(6)
    names = [
        f"{a} {b}"
        for a in ("Ada", "Alan", "Grace", "Linus", "Barbara")
        for b in ("Smith", "Jones", "Martin", "Dubois")
    ]
    rows = []
    for i in range(600):
        customer = rng.randrange(150)
        rows.append(
            {
                "order_id": f"O-{i:05d}",
                "customer_id": f"C{customer:04d}",
                "name": names[customer % len(names)],
                "day": (date(2024, 1, 1) + timedelta(days=rng.randrange(300))).isoformat(),
                "product": rng.choice(["Mouse", "Keyboard", "Screen", "Cable", "Lamp"]),
                "quantity": rng.randint(1, 5),
                "total": round(rng.uniform(5, 500), 2),
            }
        )
    for index, source in enumerate(rows[100:108]):
        rows.append(dict(source, order_id=f"O-9{index:04d}", name=source["name"][:-1]))
    ds = write(tmp_path, rows)
    found = {s.id: s for s in ds.clean().advanced()}["A-16:rows:linkage"]
    step = found.step
    assert step is not None and step.op == "drop_duplicates" and step.groups  # type: ignore[union-attr]
    assert sum(len(g) - 1 for g in step.groups) == found.rows == 8  # type: ignore[union-attr]
    shown = groups_of(ds, [step], step.id)
    assert shown.keep == "listed" and shown.removed == 8
    assert all(g.order[0] == listed[0] for g, listed in zip(shown.groups, step.groups, strict=True))  # type: ignore[union-attr]


def test_a_large_group_shows_its_first_rows_and_the_rules_row(tmp_path: Path) -> None:
    rows = [{"id": "Z", "n": str(i), "updated": f"2024-01-{1 + i % 28:02d}"} for i in range(60)]
    ds = write(tmp_path, rows)
    step = {"op": "drop_duplicates", "keys": ["id"], "keep": "last"}
    [group] = groups_of(ds, steps(step), "s1").groups
    assert group.size == 60 and len(group.rows) == 51 and group.order[0] == 59
