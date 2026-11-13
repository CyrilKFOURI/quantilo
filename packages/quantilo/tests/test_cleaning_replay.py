"""Replay a cleaning on another file (101, M5): the recipe, its strict reading, and the report of
what applies, what no longer applies and why, and what is new (S9, S10)."""

import csv
import json
from pathlib import Path
from typing import Any

import pytest

import quantilo as dx
from quantilo.cleaning import model, parse_recipe
from quantilo.cleaning.replay import FORMAT, ROWS_ELSEWHERE


def write(tmp_path: Path, rows: list[dict[str, Any]], name: str) -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dx.open(path, cache_dir=tmp_path / "cache")


def month(
    cities: list[str], count: int = 120, rename: bool = False, drop: bool = False
) -> list[dict[str, Any]]:
    rows = []
    for i in range(count):
        row = {
            "order_id": f"O{i:04d}",
            "city": cities[i % len(cities)],
            "brand": " Acme " if i % 3 == 0 else "Acme",
            "amount": f"{10 + i % 7}.50",
        }
        rows.append(row)
    if rename:
        rows = [{("City" if k == "city" else k): v for k, v in r.items()} for r in rows]
    if drop:
        rows = [{k: v for k, v in r.items() if k != "brand"} for r in rows]
    return rows


SEPTEMBER = ["Paris", "Lyon", "PARIS", "Lille", "paris"]
OCTOBER = ["Paris", "Lyon", "PARIS", "Lille", "Nantes", "Pariss"]


@pytest.fixture
def recipe(tmp_path: Path) -> Any:
    september = write(tmp_path, month(SEPTEMBER), "september")
    cleaning = september.clean()
    cleaning.use("B-03:city")
    cleaning.add(op="set_cells", column="amount", cells={3: "99"})
    cleaning.add(op="rename_column", column="amount", to="total")
    cleaning.add(op="change_case", columns=["total"], case="upper")  # a column a step made
    return cleaning.recipe("September cleaning")


def test_the_recipe_keeps_columns_steps_and_known_values(recipe: Any) -> None:
    raw = json.loads(recipe.to_json())
    assert raw["format"] == FORMAT and raw["schema"] == 1 and raw["made_on"] == "september"
    assert {c["name"] for c in raw["columns"]} == {"order_id", "city", "brand", "amount"}
    merge = next(s for s in recipe.steps if s.op == "map_values")
    assert set(recipe.known[merge.id]) == {"Paris", "Lyon", "PARIS", "Lille", "paris"}
    again = parse_recipe(recipe.to_json())
    assert again.steps == recipe.steps
    with pytest.raises(model.CleanError, match="Unknown field `extra`"):
        parse_recipe({**raw, "extra": 1})
    with pytest.raises(model.CleanError, match="Not a cleaning recipe"):
        parse_recipe({**raw, "format": "quantilo/recipe"})
    with pytest.raises(model.CleanError, match="steps"):
        parse_recipe({**raw, "steps": [{"id": "s1", "op": "eval"}]})


def test_replay_reports_what_applies_and_what_no_longer_does(recipe: Any, tmp_path: Path) -> None:
    october = write(tmp_path, month(OCTOBER, rename=True, drop=True), "october")
    cleaning = october.clean()
    report = cleaning.replay(recipe)
    by_op = {s.op: s for s in report.steps}
    assert {c.recipe: c.status for c in report.columns} == {
        "order_id": "same", "city": "matched", "brand": "missing", "amount": "same",
    }  # fmt: skip
    merge = by_op["map_values"]
    assert merge.status == "applies" and merge.cells > 0 and merge.columns == ["City"]
    assert set(merge.new_values) == {"Nantes", "Pariss"} and merge.new_count == 2  # C-42
    assert "does not know" in (merge.message or "")
    spaces = by_op["normalize_spaces"]  # B-01 on brand, ticked by default in September
    assert spaces.rule == "B-01" and spaces.status == "skipped"  # C-41
    assert "brand" in (spaces.message or "")
    cells = by_op["set_cells"]
    assert cells.status == "skipped" and cells.message == ROWS_ELSEWHERE
    assert by_op["rename_column"].status == "applies"
    assert by_op["change_case"].status == "unchanged"  # a column made by a step is no miss
    assert report.applies == 3 and report.skipped == 2
    adapted = {s.op: s for s in report.adapted}  # S10: the draft, skipped steps turned off
    assert adapted["map_values"].column == "City" and adapted["map_values"].enabled  # type: ignore[union-attr]
    assert not adapted["set_cells"].enabled and not adapted["normalize_spaces"].enabled
    assert cleaning.steps == report.adapted
    assert not cleaning.preview().problems  # the draft runs as it is


def test_replay_lists_new_problems_and_a_type_it_cannot_read(tmp_path: Path) -> None:
    september = write(
        tmp_path, [{"code": "12", "name": "a"}, {"code": "13", "name": "b"}] * 30, "s"
    )
    recipe = september.clean().add(op="null_outside", column="code", max=12.5).recipe()
    spaced = [{"code": "x1", "name": " a"}, {"code": "x2", "name": "b "}] * 30
    october = write(tmp_path, spaced, "o")
    report = october.clean().replay(recipe)
    [step] = report.steps
    assert step.status == "skipped" and "Cannot run on this file" in (step.message or "")
    assert any(s.rule == "B-01" for s in report.new_problems)  # spaces: new, not applied
    assert not any(s.enabled for s in report.adapted)
