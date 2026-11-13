"""Clean's Expert layer (101, M6, *More checks*): the two row-number steps, each method on a small
table (what it must find, what it must leave alone), cancellation, the 096 contract (impact, risk,
never ticked) and the messy sample's expectations."""

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import pytest

import quantilo as dx
from quantilo.cleaning import CleanCancelled, compile_steps, default_steps, expert, model
from quantilo.cleaning.expert import data as expert_data

SAMPLES = Path(__file__).resolve().parents[3] / "samples"


def write(tmp_path: Path, rows: list[dict[str, Any]], name: str = "data") -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dx.open(path, cache_dir=tmp_path / "cache")


def values(ds: dx.Dataset, steps: list[Any], columns: str) -> list[tuple[Any, ...]]:
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(ds.path), steps, strict=True)
        query = f"SELECT {columns} FROM ({compiled.sql()}) ORDER BY __quantilo_row"
        return connection.execute(query).fetchall()
    finally:
        connection.close()


# --- The two steps (C-33) ------------------------------------------------------------------------


def test_set_cells_and_drop_rows_follow_the_row_numbers(tmp_path: Path) -> None:
    rows = [{"id": str(i), "city": f"c{i}", "amount": str(i * 10)} for i in range(6)]
    rows[0]["city"] = ""
    ds = write(tmp_path, rows)
    steps = [
        model.FilterRows(
            id="s1", conditions=[model.RowCondition(column="city", op="empty")], action="remove"
        ),
        model.SetCells(id="s2", column="city", cells={3: "Paris", 5: None}),
        model.SetCells(id="s3", column="amount", cells={2: "75", 4: "x"}),
        model.DropRows(id="s4", rows=[1]),
    ]
    got = values(ds, steps, "id, city, amount")
    assert [r[0] for r in got] == [2, 3, 4, 5]  # row 0 filtered, row 1 dropped
    assert dict((r[0], r[1]) for r in got) == {2: "c2", 3: "Paris", 4: "c4", 5: None}
    amounts = dict((r[0], r[2]) for r in got)
    assert amounts[2] == 75 and amounts[4] is None  # cast to the column's type; unreadable: empty
    assert model.OP_IMPACT["set_cells"] == "values" and model.OP_IMPACT["drop_rows"] == "rows"
    assert {"set_cells", "drop_rows", "label_rows"} == model.ROW_NUMBER_OPS
    raw = {"id": "s1", "op": "set_cells", "column": "a", "cells": {"3": "x"}}
    parsed = model.parse_steps({"steps": [raw]})
    assert parsed.steps[0].cells == {3: "x"}  # type: ignore[union-attr]
    with pytest.raises(model.CleanError):
        model.parse_steps([{"id": "s1", "op": "drop_rows", "rows": []}])


# --- The list of methods (C-29) ------------------------------------------------------------------


def orders(count: int = 400, seed: int = 1) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    products = {"Chair": ("Furniture", 199.0, 14.0), "Desk": ("Furniture", 449.0, 28.0),
                "Mouse": ("Accessories", 24.9, 0.1), "Cable": ("Accessories", 11.5, 0.05),
                "Lamp": ("Lighting", 45.0, 1.5)}  # fmt: skip
    rows = []
    for i in range(count):
        product = rng.choice(list(products))
        category, price, weight = products[product]
        quantity = rng.choice([1, 1, 1, 2, 3])
        rows.append(
            {
                "order_id": f"O{i:05d}",
                "product": product,
                "category": category,
                "unit_price": f"{price:.2f}",
                "weight": f"{weight:.2f}",
                "quantity": str(quantity),
                "total": f"{price * quantity:.2f}",
                "channel": rng.choice(["Web", "Store", "Phone"]),
            }
        )
    return rows


def test_methods_say_what_applies_without_reading_rows(tmp_path: Path) -> None:
    ds = write(tmp_path, orders())
    found = {m.id: m for m in expert.methods(ds.types)}
    assert list(found) == list(expert.METHOD_IDS)
    assert found["unusual_rows"].available
    assert "category" in found["suspicious_labels"].columns
    assert "order_id" not in found["suspicious_labels"].columns  # an identifier is no label
    assert not found["fill_empty"].available and "empty cells" in (
        found["fill_empty"].why_not or ""
    )
    assert "unit_price" in found["cell_errors"].columns


# --- E-06 -----------------------------------------------------------------------------------------


def test_unusual_rows_are_found_and_only_suggested(tmp_path: Path) -> None:
    rng = random.Random(6)
    rows = [
        {"a": f"{rng.gauss(50, 5):.2f}", "b": f"{rng.gauss(10, 1):.2f}", "kind": rng.choice("xyz")}
        for _ in range(1_000)
    ]
    planted = list(range(100, 1_000, 90))[:10]
    for index in planted:
        rows[index]["a"] = f"{50 + 50 * 5 * rng.choice([-1, 1]):.2f}"
    ds = write(tmp_path, rows)
    result = expert.run(duckdb.connect(), str(ds.path), ds.types, "unusual_rows")
    by_id = {s.id: s for s in result.suggestions}
    mark, remove = by_id["E-06:mark"], by_id["E-06:remove"]
    assert set(planted) <= set(remove.step.rows)  # type: ignore[union-attr]
    assert mark.step.op == "label_rows" and mark.impact == "columns"  # type: ignore[union-attr]
    assert remove.step.op == "drop_rows" and remove.impact == "rows"  # type: ignore[union-attr]
    assert "Isolation Forest" in remove.reason and "may be right" in remove.reason
    assert remove.examples[0].before.startswith("row ")  # type: ignore[union-attr]


# --- E-03 -----------------------------------------------------------------------------------------


def test_suspicious_labels_flag_the_contradicted_rows(tmp_path: Path) -> None:
    rows = orders(600)
    wrong = [i for i, r in enumerate(rows) if r["product"] == "Chair"][:6]
    for index in wrong:
        rows[index]["category"] = "Accessories"
    for index in range(0, 600, 50):
        rows[index]["category"] = rows[index]["category"].lower()  # C-34: case is no class
    ds = write(tmp_path, rows)
    connection = duckdb.connect()
    result = expert.run(
        connection, str(ds.path), ds.types, "suspicious_labels",
        expert.ExpertOptions(column="category"),
    )  # fmt: skip
    by_id = {s.id: s for s in result.suggestions}
    relabel = by_id["E-03:category:relabel"]
    cells = relabel.step.cells  # type: ignore[union-attr]
    assert len(set(wrong) & set(cells)) >= 5
    assert {cells[i] for i in set(wrong) & set(cells)} == {"Furniture"}  # the frequent spelling
    assert len(cells) <= 8  # the lowercase labels are not flagged
    assert by_id["E-03:category:remove"].impact == "rows"
    quiet = expert.run(
        connection, str(ds.path), ds.types, "suspicious_labels",
        expert.ExpertOptions(column="channel"),
    )  # fmt: skip
    assert quiet.suggestions == [] and "do not tell channel" in (quiet.note or "")  # C-35
    asked = expert.run(connection, str(ds.path), ds.types, "suspicious_labels")
    assert asked.suggestions == [] and "Pick" in (asked.note or "")


# --- E-07 -----------------------------------------------------------------------------------------


def test_fill_empty_cells_with_a_band(tmp_path: Path) -> None:
    rng = random.Random(7)
    rows = []
    for _ in range(500):
        x = rng.uniform(0, 100)
        y = 2 * x + rng.gauss(0, 1)
        rows.append({"x": f"{x:.2f}", "y": f"{y:.2f}", "n": str(round(x / 10))})
    empty = list(range(0, 500, 25))
    for index in empty:
        rows[index]["y"] = ""
        rows[index]["n"] = ""
    ds = write(tmp_path, rows)
    connection = duckdb.connect()
    for imputer in ("knn", "iterative"):
        result = expert.run(
            connection, str(ds.path), ds.types, "fill_empty", expert.ExpertOptions(imputer=imputer)
        )
        by_id = {s.id: s for s in result.suggestions}
        filled = by_id[f"E-07:y:{imputer}"].step.cells  # type: ignore[union-attr]
        assert sorted(filled) == empty
        for index in empty:
            truth = 2 * float(rows[index]["x"])
            assert abs(float(filled[index] or "nan") - truth) < 12, (imputer, index)
        integers = by_id[f"E-07:n:{imputer}"].step.cells  # type: ignore[union-attr]
        assert all(float(v or "nan").is_integer() for v in integers.values())  # whole numbers
        example = by_id[f"E-07:y:{imputer}"].examples[0]
        assert (example.before or "").endswith("empty") and "(" in (example.after or "")
        assert "For the analysis only" in by_id[f"E-07:y:{imputer}"].reason  # C-36


# --- E-01 -----------------------------------------------------------------------------------------


def test_cell_errors_ask_for_marks_then_learn(tmp_path: Path) -> None:
    rows = orders(500)
    cents = [i for i in range(500) if i % 23 == 5]
    for index in cents:
        rows[index]["unit_price"] = str(round(float(rows[index]["unit_price"]) * 100))
    ds = write(tmp_path, rows)
    connection = duckdb.connect()
    path, views = str(ds.path), ds.types
    first = expert.run(
        connection, path, views, "cell_errors", expert.ExpertOptions(column="unit_price")
    )
    assert first.suggestions == [] and 2 <= len(first.to_mark) <= 20  # S7
    one_kind = expert.ExpertOptions(
        column="unit_price", marks=[expert.ExpertMark(row=cents[0], error=True)]
    )
    lonely = expert.run(connection, path, views, "cell_errors", one_kind)
    assert lonely.suggestions == [] and "OK" in (lonely.note or "")
    fine = [i for i in range(500) if i not in cents][:3]
    marks = [
        expert.ExpertMark(row=cents[0], error=True),
        expert.ExpertMark(row=cents[1], error=True),
    ]
    marks += [expert.ExpertMark(row=i, error=False) for i in fine]
    learnt = expert.run(
        connection,
        path,
        views,
        "cell_errors",
        expert.ExpertOptions(column="unit_price", marks=marks),
    )
    [fix] = learnt.suggestions
    cells = fix.step.cells  # type: ignore[union-attr]
    assert len(set(cents) & set(cells)) >= 0.8 * len(cents)
    assert not set(fine) & set(cells)  # a cell marked OK is never changed (C-37)
    price = {"Chair": "199", "Desk": "449", "Mouse": "24.9", "Cable": "11.5", "Lamp": "45"}
    for index in set(cents) & set(cells):
        assert cells[index] == price[rows[index]["product"]]  # from product, which determines it
    assert "product" in fix.reason


# --- Cancellation and the 096 contract -----------------------------------------------------------


def test_a_run_stops_when_cancelled(tmp_path: Path) -> None:
    rows = orders()
    for row in rows[::20]:
        row["weight"] = ""
    ds = write(tmp_path, rows)
    for method in expert.METHOD_IDS:
        options = expert.ExpertOptions(
            column="category" if method != "cell_errors" else "unit_price"
        )
        with pytest.raises(CleanCancelled):
            expert.run(duckdb.connect(), str(ds.path), ds.types, method, options, lambda: True)


@pytest.fixture(scope="module")
def messy(tmp_path_factory: pytest.TempPathFactory) -> tuple[dx.Dataset, Any, dict[str, Any]]:
    ds = dx.open(SAMPLES / "messy_orders.csv", cache_dir=tmp_path_factory.mktemp("messy"))
    cleaning = ds.clean()
    results: dict[str, Any] = {}
    results["channel"] = cleaning.expert("suspicious_labels", column="channel")
    assert results["channel"].suggestions == []
    results["unusual"] = cleaning.expert("unusual_rows")
    results["labels"] = cleaning.expert("suspicious_labels", column="category")
    results["iterative"] = cleaning.expert("fill_empty", imputer="iterative")
    knn = cleaning.expert("fill_empty", imputer="knn")  # replaces the iterative run (same check)
    results["knn"] = knn
    cleaning.suggestions += results["iterative"].suggestions  # both kept, to check both below
    return ds, cleaning, results


def test_messy_sample_expert_expectations(messy: tuple[dx.Dataset, Any, dict[str, Any]]) -> None:
    _, cleaning, results = messy
    expected = json.loads((SAMPLES / "messy_orders.expected.json").read_text(encoding="utf-8"))
    with (SAMPLES / "messy_orders.csv").open(encoding="utf-8") as handle:
        rows = dict(enumerate(csv.DictReader(handle)))  # the file's row numbers, from 0
    # E-01 with marks: 2 prices in cents marked Error, 3 prices in euros marked OK
    cents = [i for i, r in rows.items() if r["unit_price"] and float(r["unit_price"]) > 1_000]
    euros = [i for i, r in rows.items() if r["unit_price"] and float(r["unit_price"]) < 500][:3]
    marks = [{"row": cents[0], "error": True}, {"row": cents[1], "error": True}]
    marks += [{"row": i, "error": False} for i in euros]
    cleaning.expert("cell_errors", column="unit_price", marks=marks)
    found = {s.id: s for s in cleaning.suggestions if s.layer == "expert"}
    for wanted in expected["expert"]:
        assert wanted["id"] in found, wanted
        if "rows" in wanted:
            suggestion = found[wanted["id"]]
            assert max(suggestion.rows, suggestion.cells) == wanted["rows"]
    flagged = set(found["E-06:remove"].step.rows)  # type: ignore[union-attr]
    shares = {column: Counter(r[column] for r in rows.values()) for column in ("city", "product")}
    planted = {
        i for i, r in rows.items()
        if (r["unit_price"] and float(r["unit_price"]) > 1_000)  # cents (A-05)
        or (r["weight_kg"] and float(r["weight_kg"]) >= 50)  # grams (A-05)
        or r["rating"] == "99"  # a code for "no rating" (B-09)
        or any(shares[c][r[c]] < 0.01 * len(rows) for c in shares)  # a rare spelling (A-03)
    }  # fmt: skip
    assert len(flagged & planted) / len(flagged) >= 0.6  # most unusual rows carry a planted trap
    chairs = found["E-03:category:relabel"].step.cells  # type: ignore[union-attr]
    assert set(chairs.values()) == {"Furniture"}
    assert all(rows[i]["product"] == "Office Chair" for i in chairs)
    fixed = found["E-01:unit_price"].step.cells  # type: ignore[union-attr]
    assert len(set(cents) & set(fixed)) >= 0.85 * len(cents) and not set(euros) & set(fixed)
    for quiet in expected["expert_quiet"]:
        assert results["channel"].suggestions == [] and quiet["options"]["column"] == "channel"
    for suggestion in found.values():  # controls: no Expert step writes them (none was chosen)
        step = suggestion.step
        named = {getattr(step, "column", None), getattr(step, "to", None)}
        assert not named & set(expected["controls"])


def test_every_expert_suggestion_is_suggested_only(
    messy: tuple[dx.Dataset, Any, dict[str, Any]],
) -> None:
    _, cleaning, _ = messy
    found = [s for s in cleaning.suggestions if s.layer == "expert"]
    assert found
    for suggestion in found:  # C-30, 096
        assert (suggestion.mode, suggestion.risk) == ("S", "review")
        assert suggestion.impact in model.IMPACTS
        assert suggestion.rule in expert_data.ORDER and suggestion.order >= 400 * 10_000  # C-38
    assert not {s.suggestion for s in default_steps(cleaning.suggestions)} & {s.id for s in found}
    ids = [s.step.id for s in cleaning.suggestions if s.step is not None]
    assert len(ids) == len(set(ids))
    assert not any(s.suggestion in {f.id for f in found} for s in cleaning.steps)  # none ticked
    before = len(found)
    cleaning.expert("unusual_rows")  # a second run replaces the first's suggestions
    assert len([s for s in cleaning.suggestions if s.layer == "expert"]) == before
