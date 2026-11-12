"""089: join two datasets (docs/features/089-jointures)."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

import quantilo as dx
from quantilo import model
from quantilo.model.keys import TableInput


def _table(tmp_path: Path, name: str, frame: pd.DataFrame) -> TableInput:
    table = model.table_of(dx.from_pandas(frame, name, cache_dir=tmp_path / name))
    table.id = name
    return table


@pytest.fixture
def shop(tmp_path: Path) -> dict[str, TableInput]:
    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(1, 31)],
            "country": ["FR", "DE", "US"] * 10,
            "segment": ["a", "b"] * 15,
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": range(1, 101),
            "customer_id": [f"C{(i % 30) + 1:03d}" for i in range(100)],
            "amount": [float(10 + i) for i in range(100)],
        }
    )
    orders.loc[0, "customer_id"] = "C999"  # an orphan
    return {
        "customers": _table(tmp_path, "customers", customers),
        "orders": _table(tmp_path, "orders", orders),
    }


@pytest.fixture
def document(tmp_path: Path) -> dict[str, TableInput]:
    """The tables of orders.json, as 083 splits them (`_id`, `_parent_id`)."""
    orders = pd.DataFrame({"_id": range(1, 6), "country": ["FR", "DE", "FR", "US", "DE"]})
    items = pd.DataFrame(
        {
            "_id": range(1, 13),
            "_parent_id": [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5],
            "_position": [0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1],
            "sku": [f"S{i}" for i in range(12)],
            "qty": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        }
    )
    return {
        "orders": _table(tmp_path, "orders", orders),
        "orders.items": _table(tmp_path, "orders.items", items),
    }


def test_suggest_finds_the_foreign_key_either_way(shop: dict[str, TableInput]) -> None:
    duck = duckdb.connect()
    found = model.suggest(duck, shop["orders"], shop["customers"])
    assert found[0].left_columns == ["customer_id"] and found[0].mode == "lookup"
    assert found[0].reasons[0].code == "R-JN-01"
    back = model.suggest(duck, shop["customers"], shop["orders"])
    assert back[0].mode == "expand" and back[0].cardinality == "1:n"


def test_lookup_keeps_every_left_row(shop: dict[str, TableInput]) -> None:
    duck = duckdb.connect()
    result = model.preview_pair(
        duck, shop["orders"], shop["customers"], ["customer_id"], ["customer_id"]
    )
    assert result.mode == "lookup" and result.join_type == "left"
    assert result.rows_before == result.rows_after == 100
    assert result.added == ["customers_country", "customers_segment"]
    assert result.check.orphan_rows == 1
    assert result.reasons[0].code == "R-JOIN-01"
    assert len(result.rows) == 20 and not result.refused


def test_inner_warns_of_dropped_rows(shop: dict[str, TableInput]) -> None:
    duck = duckdb.connect()
    result = model.preview_pair(
        duck,
        shop["orders"],
        shop["customers"],
        ["customer_id"],
        ["customer_id"],
        join_type="inner",
    )
    assert result.rows_after == 99
    assert any(w.code == "R-JOIN-01" and "dropped" in w.text for w in result.warnings)


def test_lookup_into_a_table_with_repeated_keys_is_refused(shop: dict[str, TableInput]) -> None:
    duck = duckdb.connect()
    result = model.preview_pair(
        duck, shop["customers"], shop["orders"], ["customer_id"], ["customer_id"], mode="lookup"
    )
    assert result.refused and result.refused[0].code == "R-JN-04"
    with pytest.raises(model.ModelError):
        model.write_pair(
            duck,
            shop["customers"],
            shop["orders"],
            ["customer_id"],
            ["customer_id"],
            Path("never.parquet"),
            mode="lookup",
        )


def test_from_the_parent_expand_is_recommended_and_summarise_counts(
    shop: dict[str, TableInput], tmp_path: Path
) -> None:
    duck = duckdb.connect()
    left, right = shop["customers"], shop["orders"]
    expanded = model.preview_pair(duck, left, right, ["customer_id"], ["customer_id"])
    assert expanded.mode == "expand"
    assert expanded.rows_after == 100 and expanded.reasons[0].code == "R-JN-05"
    summary = model.preview_pair(
        duck, left, right, ["customer_id"], ["customer_id"], mode="summarise"
    )
    assert summary.rows_after == 30
    assert "orders_count" in summary.added and "orders_amount_sum" in summary.added
    target = tmp_path / "out.parquet"
    model.write_pair(duck, left, right, ["customer_id"], ["customer_id"], target, mode="summarise")
    total = duck.execute(
        f"SELECT sum(orders_count) FROM read_parquet('{target.as_posix()}')"
    ).fetchone()
    assert total == (99,)  # the orphan order has no customer row


def test_document_tables_join_on_their_known_link(document: dict[str, TableInput]) -> None:
    duck = duckdb.connect()
    found = model.suggest(duck, document["orders.items"], document["orders"])
    known = [key for key in found if key.left_columns == ["_parent_id"]]
    assert known and known[0].mode == "lookup"
    assert any(reason.code == "R-JN-02" for reason in known[0].reasons)
    result = model.preview_pair(
        duck, document["orders.items"], document["orders"], ["_parent_id"], ["_id"]
    )
    assert result.rows_after == 12 and "orders_country" in result.added


def test_same_name_columns_are_offered_last(tmp_path: Path) -> None:
    left = _table(tmp_path, "a", pd.DataFrame({"code": ["x", "y", "z", "w"] * 5, "v": range(20)}))
    right = _table(
        tmp_path, "b", pd.DataFrame({"code": ["x", "y", "z", "q"], "label": ["X", "Y", "Z", "Q"]})
    )
    duck = duckdb.connect()
    found = model.suggest(duck, left, right)
    assert found and found[0].left_columns == ["code"]
    assert any(reason.code in ("R-JN-01", "R-JN-03") for reason in found[0].reasons)
