"""099 — Pinned rows: what distinguishes them from the other rows (R-PIN-06 to R-PIN-14), the
row keys found again after a re-import (R-PIN-02), and the `label_rows` cleaning step (R-PIN-15)."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo.cleaning import CleanError, parse_steps
from quantilo.cleaning.model import OP_IMPACT
from quantilo.segments import pinned
from quantilo.segments.engine import SegmentsEngine
from tests.test_cleaning import column, parquet, run


def shop(n: int = 1_000, seed: int = 0) -> tuple[pd.DataFrame, list[int]]:
    """Orders where the 20 pinned ones are ten times larger and nearly all on the web."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "order_id": [f"O{i:05d}" for i in range(n)],
            "amount": rng.gamma(4, 50, n).round(2),
            "items": rng.integers(1, 6, n),
            "channel": rng.choice(["store", "web", "phone"], n, p=[0.5, 0.3, 0.2]),
        }
    )
    chosen = sorted(rng.choice(n, 20, replace=False).tolist())
    frame.loc[chosen, "amount"] = rng.gamma(4, 500, 20).round(2)
    frame.loc[chosen, "channel"] = ["web"] * 18 + ["store"] * 2
    return frame, chosen


def split(frame: pd.DataFrame, rows: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    inside = frame.index.isin(rows)
    return cast(pd.DataFrame, frame[inside]), cast(pd.DataFrame, frame[~inside])


# --- The comparison (pure) -------------------------------------------------------------------


def test_the_columns_that_set_pinned_rows_apart_come_first() -> None:
    frame, chosen = shop()
    mine, rest = split(frame, list(chosen))
    found = pinned.compare(mine, rest, ["amount", "items"], ["channel"], label="big")

    assert found.pinned == 20 and found.others == 980 and not found.sampled
    assert [t.column for t in found.traits[:2]] == ["amount", "channel"]
    amount, channel = found.traits[0], found.traits[1]
    assert amount.rule == "R-PIN-07" and amount.effect > 2
    assert "the others' average" in amount.text
    assert channel.rule == "R-PIN-08" and channel.value == "web"
    assert channel.text.startswith("channel: 90 % of pinned rows are web, vs ")
    assert amount.p_value is not None and amount.p_value < 0.05 and not amount.chance
    assert channel.p_value is not None and channel.p_value < 0.05
    assert all(t.column != "items" for t in found.traits)  # no gap on items
    assert found.reading.startswith("The 20 rows pinned “big” differ most on amount, channel")
    assert found.rules is not None and found.rules.recall >= 0.8
    assert "of the pinned rows" in found.rules.text
    assert found.warnings == []


def test_one_pinned_row_gives_the_reason_and_no_figure() -> None:
    frame, chosen = shop()
    mine, rest = split(frame, list(chosen[:1]))
    found = pinned.compare(mine, rest, ["amount"], ["channel"])

    assert found.traits == [] and found.rules is None
    assert found.reading == "Pin at least 2 rows to compare them with the others (1 found)."


def test_few_pinned_rows_are_hints_and_get_no_rules() -> None:
    frame, chosen = shop()
    mine, rest = split(frame, list(chosen[:4]))
    found = pinned.compare(mine, rest, ["amount"], ["channel"])

    assert found.warnings == ["4 pinned rows: read these differences as hints."]
    assert found.rules is None  # R-PIN-12: from 5 rows on


def test_empty_cells_among_pinned_rows_are_a_trait() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"price": rng.normal(100, 10, 500), "kind": rng.choice(["a", "b"], 500)})
    frame.loc[:5, "price"] = np.nan
    mine, rest = split(frame, list(range(6)))
    found = pinned.compare(mine, rest, ["price"], ["kind"])

    trait = next(t for t in found.traits if t.kind == "missing")
    assert trait.rule == "R-PIN-09" and trait.column == "price"
    assert trait.text == "price is empty in 100 % of pinned rows, vs 0 % of the others"
    assert trait.p_value is not None and trait.p_value < 0.05


def test_differences_of_chance_are_said_so() -> None:
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=300) for i in range(10)})
    mine, rest = split(frame, list([3, 50, 120]))
    found = pinned.compare(mine, rest, list(frame.columns), [])

    assert all(t.chance for t in found.traits)  # 3 random rows, 20 tests (Holm)
    if found.traits:
        assert found.reading.endswith("but every difference could be chance.")


def test_a_category_of_one_pinned_row_is_not_a_trait() -> None:
    frame = pd.DataFrame({"day": [f"d{i}" for i in range(300)], "kind": ["a", "b"] * 150})
    mine, rest = split(frame, [0, 1, 2, 3, 4])
    found = pinned.compare(mine, rest, [], ["day"])

    assert found.traits == []  # every day is unique: one pinned row each (R-PIN-08)


def test_holm_is_over_every_column_tested() -> None:
    frame, chosen = shop()
    mine, rest = split(frame, list(chosen))
    alone = pinned.compare(mine, rest, ["amount"], [])
    many = pinned.compare(mine, rest, ["amount", "items"], ["channel"])

    first = alone.traits[0].p_value
    later = next(t for t in many.traits if t.column == "amount").p_value
    assert first is not None and later is not None and later >= first


# --- On a dataset: the engine and the library --------------------------------------------------


def test_compare_rows_on_a_dataset(tmp_path: Path) -> None:
    frame, chosen = shop()
    ds = dx.from_pandas(frame, "shop", cache_dir=tmp_path / "shop")
    found = ds.compare_rows(chosen, label="big")

    assert found.pinned == 20 and found.others == 980
    assert "order_id" not in found.columns  # R-PIN-14: no identifier
    assert set(found.columns) == {"amount", "items", "channel"}
    assert [t.column for t in found.traits[:2]] == ["amount", "channel"]


def test_keys_are_found_again_after_the_rows_move(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"a": rng.normal(size=50).round(4), "b": rng.choice(["x", "y"], 50)})
    first = dx.from_pandas(frame, "first", cache_dir=tmp_path / "first")
    engine = first._state.service(SegmentsEngine)
    keys = engine.keys(first._id, [7, 30])
    assert all(k.startswith("row:") for k in keys.values())

    flipped = frame.iloc[::-1].reset_index(drop=True).copy()
    moved = dx.from_pandas(flipped, "moved", cache_dir=tmp_path / "moved")
    again = moved._state.service(SegmentsEngine)
    found = again.find_keys(moved._id, [keys[7], keys[30], "row:not-there"])
    assert found == {keys[7]: 42, keys[30]: 19}


def test_identifier_keys_are_found_by_their_column(tmp_path: Path) -> None:
    frame, _ = shop(100)
    ds = dx.from_pandas(frame, "shop", cache_dir=tmp_path / "shop")
    engine = ds._state.service(SegmentsEngine)
    keys = engine.keys(ds._id, [5])
    assert keys == {5: "order_id=O00005"}
    assert engine.find_keys(ds._id, ["order_id=O00042", "order_id=nope", "gone=1"]) == {
        "order_id=O00042": 42
    }


# --- The label_rows step (R-PIN-15) ---------------------------------------------------------


CSV = "city,n\nParis,1\nLyon,2\nNice,3\nLille,4\n"


def test_label_rows_adds_the_labels_by_row_number(tmp_path: Path) -> None:
    path = parquet(tmp_path, CSV)
    step = {"op": "label_rows", "to": "pin_label", "labels": {"0": "test", "2": "fraud"}}

    names, _ = run(path, step)
    assert names == ["city", "n", "pin_label"]
    assert column(path, "pin_label", step) == ["test", None, "fraud", None]
    other = {**step, "other": "none"}
    assert column(path, "pin_label", other) == ["test", "none", "fraud", "none"]


def test_label_rows_follows_the_rows_after_rows_are_removed(tmp_path: Path) -> None:
    path = parquet(tmp_path, CSV)
    remove = {
        "op": "filter_rows",
        "conditions": [{"column": "city", "op": "eq", "value": "Paris"}],
        "action": "remove",
    }
    step = {"op": "label_rows", "to": "pin_label", "labels": {"2": "fraud", "3": "test"}}

    assert column(path, "city", remove, step) == ["Lyon", "Nice", "Lille"]
    assert column(path, "pin_label", remove, step) == [None, "fraud", "test"]


def test_label_rows_refuses_an_existing_column_and_an_empty_label(tmp_path: Path) -> None:
    path = parquet(tmp_path, CSV)
    with pytest.raises(CleanError, match="already exists"):
        run(path, {"op": "label_rows", "to": "city", "labels": {"0": "test"}})
    with pytest.raises(CleanError, match="labels"):
        parse_steps([{"id": "s1", "op": "label_rows", "to": "x", "labels": {"0": ""}}])
    with pytest.raises(CleanError, match="labels"):
        parse_steps([{"id": "s1", "op": "label_rows", "to": "x", "labels": {}}])
    assert OP_IMPACT["label_rows"] == "columns"
