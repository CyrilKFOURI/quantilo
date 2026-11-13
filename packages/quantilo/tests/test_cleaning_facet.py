"""N40: a facet of a column of the draft (values and rows, histogram, blank, unreadable) and the
rows a selection picks."""

import csv
from pathlib import Path
from typing import Any

import duckdb
import pytest

import quantilo as dx
from quantilo.cleaning import CleanError, compile_steps, parse_steps
from quantilo.cleaning.facet import BINS, FacetRequest, FacetSelection, facet


def write(tmp_path: Path, rows: list[list[Any]]) -> dx.Dataset:
    path = tmp_path / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return dx.open(path, cache_dir=tmp_path / "cache")


def ratings(tmp_path: Path) -> dx.Dataset:
    rows: list[list[Any]] = [["title", "rating", "score", "amount"]]
    labels = ["Not Rated"] * 5 + ["Unrated"] * 3 + ["PG"] * 8 + [""] * 2 + ["R"] * 2
    for index, label in enumerate(labels):
        amount = f"{index}.5" if index % 6 else "12 kg"
        rows.append([f"film {index}", label, index * 10, amount])
    return write(tmp_path, rows)


def run(ds: dx.Dataset, request: FacetRequest, *steps: dict[str, Any], view: str | None = None):
    parsed = parse_steps([{"id": f"s{i + 1}", **step} for i, step in enumerate(steps)]).steps
    connection = duckdb.connect()
    compiled = compile_steps(connection, str(ds.path), parsed)
    typed = next((v for v in ds.types if v.name == view), None) if view else None
    return facet(connection, compiled, request, typed)


def test_text_facet_lists_values_with_their_rows(tmp_path: Path) -> None:
    ds = ratings(tmp_path)
    found = run(ds, FacetRequest(column="rating"))
    assert found.kind == "text" and found.rows == 20 and found.blank == 2
    assert [(v.value, v.rows) for v in found.values][:3] == [
        ("PG", 8),
        ("Not Rated", 5),
        ("Unrated", 3),
    ]
    assert found.listed_from == 4 and found.matched is None  # NULL (the blanks) is not a value
    by_name = run(ds, FacetRequest(column="rating", sort="name"))
    assert [v.value for v in by_name.values] == ["Not Rated", "PG", "R", "Unrated"]
    searched = run(ds, FacetRequest(column="rating", search="rated"))
    assert [v.value for v in searched.values] == [
        "Not Rated",
        "Unrated",
    ] and searched.listed_from == 2


def test_a_selection_picks_rows_of_the_draft(tmp_path: Path) -> None:
    ds = ratings(tmp_path)
    picked = run(
        ds, FacetRequest(column="rating", selection=FacetSelection(values=["Unrated"], blank=True))
    )
    assert picked.matched == 5
    assert picked.shown[0] == "rating" and "__quantilo_row" not in picked.shown
    assert [r.row for r in picked.sample] == [5, 6, 7, 16, 17]
    # read on the draft: once 'Unrated' is merged, it is gone
    merged = run(
        ds,
        FacetRequest(column="rating", selection=FacetSelection(values=["Unrated"])),
        {"op": "map_values", "column": "rating", "map": {"Unrated": "Not Rated"}},
    )
    assert ("Not Rated", 8) in [(v.value, v.rows) for v in merged.values]
    assert merged.matched == 0 and merged.sample == []


def test_number_facet_is_a_histogram_with_a_range(tmp_path: Path) -> None:
    ds = ratings(tmp_path)
    found = run(ds, FacetRequest(column="score"))
    assert found.kind == "number" and (found.low, found.high) == (0.0, 190.0)
    assert len(found.bins) == BINS and sum(b.rows for b in found.bins) == 20
    assert found.bins[-1].high == 190.0
    ranged = run(ds, FacetRequest(column="score", selection=FacetSelection(low=50, high=100)))
    assert ranged.matched == 6
    with pytest.raises(CleanError):
        run(ds, FacetRequest(column="rating", selection=FacetSelection(low=1)))


def test_unreadable_cells_of_a_text_read_as_numbers(tmp_path: Path) -> None:
    ds = ratings(tmp_path).retype("amount", "numeric_continuous")
    found = run(ds, FacetRequest(column="amount"), view="amount")
    assert found.reading == "number" and found.unreadable == 4
    assert [(v.value, v.rows) for v in found.unreadable_values] == [("12 kg", 4)]
    picked = run(
        ds, FacetRequest(column="amount", selection=FacetSelection(unreadable=True)), view="amount"
    )
    assert picked.matched == 4


def test_a_column_no_step_leaves_is_refused(tmp_path: Path) -> None:
    ds = ratings(tmp_path)
    with pytest.raises(CleanError):
        run(ds, FacetRequest(column="rating"), {"op": "drop_columns", "columns": ["rating"]})
