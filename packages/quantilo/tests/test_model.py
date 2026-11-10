"""088: candidate keys, relations, roles and the analysis table, on small synthetic stars."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo import model
from quantilo.model import names
from quantilo.model.keys import TableInput


def _tables(tmp_path: Path, frames: dict[str, pd.DataFrame]) -> dict[str, TableInput]:
    out = {}
    for name, frame in frames.items():
        table = model.table_of(dx.from_pandas(frame, name, cache_dir=tmp_path / name))
        table.id = name  # readable ids in the tests
        out[name] = table
    return out


def _star(tmp_path: Path) -> dict[str, TableInput]:
    rng = np.random.default_rng(7)
    rows = 200
    codes = [f"C{index:03d}" for index in range(1, 41)]
    customer = rng.choice(codes, rows).astype(object)
    customer[:3] = "C999"  # orphans
    sales = pd.DataFrame(
        {
            "sale_id": np.arange(1, rows + 1),
            "customer_id": customer,
            "ProductKey": rng.integers(1, 31, rows),
            "store": rng.integers(1, 6, rows),
            "priority": rng.integers(1, 6, rows),  # 1..5 like stores.id, no shared name
            "quantity": rng.integers(1, 10, rows),
            "amount": rng.normal(100, 20, rows).round(2),
            "sold_on": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "is_promo": rng.random(rows) > 0.5,
            "flag": rng.integers(0, 2, rows),
        }
    )
    customers = pd.DataFrame(
        {
            "Customer ID": codes,
            "country": rng.choice(["FR", "DE", "ES"], len(codes)),
            "segment": rng.choice(["retail", "pro"], len(codes)),
            "phone": [f"+33 1 23 45 {index:02d} {index:02d}" for index in range(len(codes))],
        }
    )
    products = pd.DataFrame(
        {
            "productID": np.arange(1, 31),
            "product_name": [f"Product {index}" for index in range(1, 31)],
            "category_id": rng.integers(1, 5, 30),
            "amount": rng.normal(50, 5, 30).round(2),
        }
    )
    categories = pd.DataFrame(
        {"category_id": [1, 2, 3, 4], "category": ["Food", "Toys", "Books", "Tools"]}
    )
    stores = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5], "city": ["Paris", "Lyon", "Nice", "Lille", "Metz"]}
    )
    return _tables(
        tmp_path,
        {
            "sales": sales,
            "customers": customers,
            "products": products,
            "categories": categories,
            "stores": stores,
        },
    )


def _pairs(found: model.Detection) -> set[tuple[str, str, str, str]]:
    return {
        (r.child_id, ",".join(r.child_columns), r.parent_id, ",".join(r.parent_columns))
        for r in found.relations
    }


# --- Names (R-REL-02) ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("child", "parent", "parent_table", "score"),
    [
        ("customer_id", "Customer ID", "customers", 1.0),
        ("customer_id", "CustomerKey", "customers", 1.0),
        ("customerID", "id", "customers", 1.0),
        ("category_id", "categoryID", "categories", 1.0),
        ("ship_city", "city", "cities", 0.8),
        ("shipVia", "shipperID", "shippers", 0.6),
        ("employeeID", "shipperID", "shippers", 0.0),
        ("priority", "id", "stores", 0.0),
    ],
)
def test_names_compare_like_a_reader(
    child: str, parent: str, parent_table: str, score: float
) -> None:
    assert names.similarity(child, "facts", parent, parent_table) == score


def test_words_and_singulars() -> None:
    assert names.words("CustomerKey") == ["customer", "key"]
    assert names.words("orderID") == ["order", "id"]
    assert names.words("Prénom client") == ["prenom", "client"]
    assert names.singular("categories") == "category"
    assert names.singular("address") == "address"
    assert names.base("id", "order_details") == "order detail"
    assert names.prefix("Sample - Superstore") == "Sample_Superstore"


# --- Keys (R-KEY) -------------------------------------------------------------------------------


def test_candidate_keys_and_exclusions(tmp_path: Path) -> None:
    tables = _star(tmp_path)
    connection = duckdb.connect()
    sales = model.profile_table(connection, tables["sales"])
    assert sales.primary_key is not None and sales.primary_key.columns == ["sale_id"]
    by_name = {column.name: column for column in sales.columns}
    assert by_name["is_promo"].excluded_by and by_name["is_promo"].excluded_by.code == "R-REL-06"
    assert by_name["amount"].kind is None  # decimals, or a measure role: never a key
    assert by_name["quantity"].excluded_by and by_name["quantity"].excluded_by.code == "R-KEY-02"
    stores = model.profile_table(connection, tables["stores"])
    assert stores.primary_key and stores.primary_key.columns == ["id"]
    assert any(reason.code == "R-KEY-04" for reason in stores.primary_key.reasons)


def test_a_key_of_two_columns_when_none_is_unique(tmp_path: Path) -> None:
    lines = pd.DataFrame(
        {
            "order_id": [1, 1, 2, 2, 3],
            "product_id": [10, 11, 10, 12, 11],
            "quantity": [1, 2, 3, 4, 5],
        }
    )
    tables = _tables(tmp_path, {"lines": lines})
    profile = model.profile_table(duckdb.connect(), tables["lines"])
    assert profile.primary_key and profile.primary_key.columns == ["order_id", "product_id"]
    assert any(reason.code == "R-KEY-03" for reason in profile.primary_key.reasons)


# --- Relations (R-REL) --------------------------------------------------------------------------


def test_detects_the_star_and_nothing_else(tmp_path: Path) -> None:
    tables = _star(tmp_path)
    found = model.detect_tables(list(tables.values()))
    assert _pairs(found) == {
        ("sales", "customer_id", "customers", "Customer ID"),
        ("sales", "ProductKey", "products", "productID"),
        ("sales", "store", "stores", "id"),
        ("products", "category_id", "categories", "category_id"),
    }
    by_parent = {r.parent_id: r for r in found.relations}
    assert all(r.cardinality == "n:1" for r in found.relations)
    customers = by_parent["customers"]
    assert 0.95 <= customers.inclusion < 1  # one orphan code among ~40
    assert customers.confidence == "medium"
    assert by_parent["categories"].confidence == "high"
    assert any(reason.code == "R-REL-04" for reason in customers.reasons)


def test_small_integers_need_a_shared_name(tmp_path: Path) -> None:
    """R-REL-05: `priority` (1-5) is inside `stores.id` (1-5) but shares no name."""
    found = model.detect_tables(list(_star(tmp_path).values()))
    assert not any(r.child_columns == ["priority"] for r in found.relations)
    assert not any(r.child_columns == ["flag"] for r in found.relations)
    assert not any("amount" in r.child_columns for r in found.relations)


def test_values_alone_link_specific_codes(tmp_path: Path) -> None:
    codes = [f"ZX{index:04d}" for index in range(60)]
    facts = pd.DataFrame({"ref_a": codes * 3, "v": range(180)})
    things = pd.DataFrame({"code": codes, "label": [f"t{index}" for index in range(60)]})
    found = model.detect_tables(
        list(_tables(tmp_path, {"facts": facts, "things": things}).values())
    )
    (relation,) = found.relations
    assert relation.name_score == 0
    assert relation.reasons[0].code == "R-REL-05"
    assert relation.confidence == "medium"


def test_text_and_integer_codes_meet(tmp_path: Path) -> None:
    facts = pd.DataFrame({"store_id": [str(1000 + i % 30) for i in range(90)], "v": range(90)})
    stores = pd.DataFrame(
        {"store_id": list(range(1000, 1030)), "name": [f"s{i}" for i in range(30)]}
    )
    found = model.detect_tables(
        list(_tables(tmp_path, {"facts": facts, "stores": stores}).values())
    )
    (relation,) = found.relations
    assert relation.inclusion == 1


def test_never_a_dataset_with_its_own_version(tmp_path: Path) -> None:
    frame = pd.DataFrame({"id": range(50), "x": range(50)})
    tables = _tables(tmp_path, {"a": frame, "b": frame.copy()})
    assert model.detect_tables(list(tables.values())).relations  # 1:1 between strangers
    found = model.detect_tables(list(tables.values()), families={"a": "a", "b": "a"})
    assert found.relations == []


def test_many_to_many_is_proposed_with_a_warning(tmp_path: Path) -> None:
    left = pd.DataFrame({"campaign_code": [f"K{i % 25}" for i in range(100)], "spend": range(100)})
    right = pd.DataFrame({"campaign_code": [f"K{i % 25}" for i in range(75)], "clicks": range(75)})
    found = model.detect_tables(list(_tables(tmp_path, {"spend": left, "clicks": right}).values()))
    (relation,) = found.relations
    assert relation.cardinality == "n:m"
    assert relation.confidence == "low"


# --- Roles (R-ROLE) -----------------------------------------------------------------------------


def test_fact_in_the_centre_of_a_snowflake(tmp_path: Path) -> None:
    found = model.detect_tables(list(_star(tmp_path).values()))
    roles = found.roles
    assert roles.facts == ["sales"]
    assert roles.tables["sales"].primary and roles.tables["sales"].depth == 0
    assert roles.tables["products"].role == "dimension" and roles.tables["products"].depth == 1
    assert roles.tables["categories"].depth == 2
    assert roles.shape == "snowflake"


# --- Joins and the analysis table (R-JOIN, R-BLD) -----------------------------------------------


def _links(found: model.Detection) -> list[model.Link]:
    return [
        model.Link(r.child_id, r.child_columns, r.parent_id, r.parent_columns)
        for r in found.relations
        if r.cardinality != "n:m"
    ]


def test_check_counts_orphans_and_recommends_left(tmp_path: Path) -> None:
    tables = _star(tmp_path)
    found = model.check(
        duckdb.connect(), tables["sales"], ["customer_id"], tables["customers"], ["Customer ID"]
    )
    assert found.orphan_rows == 3 and found.orphans[0].value == "C999"
    assert found.matched_rows == 197
    assert found.recommended == "left" and found.reasons[0].code == "R-JOIN-01"
    warned = model.warnings(found, "inner", "sales", "customers")
    assert warned and "3 sales rows would be dropped" in warned[0].text


def test_inner_only_when_complete_both_ways(tmp_path: Path) -> None:
    tables = _star(tmp_path)
    found = model.check(
        duckdb.connect(), tables["products"], ["category_id"], tables["categories"], ["category_id"]
    )
    used_all = found.unused_keys == 0
    assert found.recommended == ("inner" if used_all else "left")


def test_flat_table_keeps_every_fact_row(tmp_path: Path) -> None:
    tables = _star(tmp_path)
    found = model.detect_tables(list(tables.values()))
    plan = model.plan("sales", tables, _links(found))
    target = tmp_path / "flat.parquet"
    preview = model.write_flat(duckdb.connect(), plan, target)
    assert preview.rows_before == preview.rows_after == 200
    frame = duckdb.connect().execute(f"SELECT * FROM read_parquet('{target.as_posix()}')").df()
    assert len(frame) == 200
    columns = list(frame.columns)
    assert columns[:10] == list(tables["sales"].columns[i].name for i in range(10))
    assert "customers_country" in columns and "categories_category" in columns
    assert "customers_Customer ID" not in columns  # the key is not repeated
    assert "customers_phone" not in columns  # contact details are not picked by default
    assert "products_amount" in columns  # prefixed: no clash with sales.amount
    assert frame["sale_id"].tolist() == list(range(1, 201))  # the fact's order
    assert frame.loc[frame["customer_id"] == "C999", "customers_country"].isna().all()


def test_picks_and_a_dimension_reached_twice(tmp_path: Path) -> None:
    facts = pd.DataFrame(
        {"from_city": [1, 2, 3, 1] * 10, "to_city": [2, 3, 1, 3] * 10, "km": range(40)}
    )
    cities = pd.DataFrame({"city_id": [1, 2, 3], "name": ["A", "B", "C"]})
    tables = _tables(tmp_path, {"trips": facts, "cities": cities})
    links = [
        model.Link("trips", ["from_city"], "cities", ["city_id"]),
        model.Link("trips", ["to_city"], "cities", ["city_id"]),
    ]
    plan = model.plan("trips", tables, links, picks={"cities": ["name"]})
    names_out = [column.name for column in plan.columns]
    assert names_out == ["from_city", "to_city", "km", "from_city_name", "to_city_name"]
    preview = model.preview(duckdb.connect(), plan)
    assert preview.rows_after == 40


def test_duplicate_keys_refuse_the_build(tmp_path: Path) -> None:
    facts = pd.DataFrame({"product_id": [1, 2, 3] * 5, "v": range(15)})
    products = pd.DataFrame({"product_id": [1, 2, 3, 3], "name": ["a", "b", "c", "c2"]})
    tables = _tables(tmp_path, {"facts": facts, "products": products})
    plan = model.plan(
        "facts", tables, [model.Link("facts", ["product_id"], "products", ["product_id"])]
    )
    preview = model.preview(duckdb.connect(), plan)
    assert preview.refused and preview.refused[0].code == "R-BLD-03"
    with pytest.raises(model.ModelError):
        model.write_flat(duckdb.connect(), plan, tmp_path / "x.parquet")


def test_inner_join_drops_the_orphans(tmp_path: Path) -> None:
    tables = _star(tmp_path)
    link = model.Link("sales", ["customer_id"], "customers", ["Customer ID"], "inner")
    preview = model.preview(duckdb.connect(), model.plan("sales", tables, [link]))
    assert preview.rows_after == 197


def test_aggregate_a_child_to_its_parent(tmp_path: Path) -> None:
    tables = _star(tmp_path)
    link = model.Link("sales", ["customer_id"], "customers", ["Customer ID"])
    target = tmp_path / "agg.parquet"
    measures = [model.Measure("amount", "sum"), model.Measure("quantity", "mean")]
    rows, added = model.aggregate(
        duckdb.connect(), tables["sales"], link, tables["customers"], measures, target
    )
    assert rows == 40
    assert [column.name for column in added] == [
        "sales_count",
        "sales_amount_sum",
        "sales_quantity_mean",
    ]
    counts = (
        duckdb.connect()
        .execute(f"SELECT sum(sales_count) FROM read_parquet('{target.as_posix()}')")
        .fetchone()
    )
    assert counts and int(counts[0]) == 197  # the 3 orphans have no customer row
    defaults = model.default_measures(tables["sales"], {"customer_id"})
    assert model.Measure("amount", "sum") in defaults


def test_incompatible_columns_are_refused(tmp_path: Path) -> None:
    tables = _star(tmp_path)
    with pytest.raises(model.ModelError) as error:
        model.check(duckdb.connect(), tables["sales"], ["is_promo"], tables["stores"], ["id"])
    assert "R-REL-06" in error.value.message


def test_pieces_of_one_table_are_not_joined(tmp_path: Path) -> None:
    """R-REL-12: two years of the same export share their columns: stack, don't join."""
    a = pd.DataFrame(
        {
            "invoice_id": range(100),
            "customer_id": [f"K{i % 30}" for i in range(100)],
            "v": range(100),
        }
    )
    b = pd.DataFrame(
        {
            "invoice_id": range(100, 180),
            "customer_id": [f"K{i % 30}" for i in range(80)],
            "v": range(80),
        }
    )
    found = model.detect_tables(list(_tables(tmp_path, {"y2010": a, "y2011": b}).values()))
    assert found.relations == []
    assert [note.code for note in found.notes] == ["R-REL-12"]


def test_one_to_one_is_proposed_once(tmp_path: Path) -> None:
    """R-REL-09: two tables keyed by the same ids give one 1:1 relation, not two."""
    ids = [f"{1000 + i}-AB" for i in range(60)]
    a = pd.DataFrame({"customerID": ids, "tenure": range(60)})
    b = pd.DataFrame({"customerID": ids, "churn": ["Yes", "No"] * 30})
    found = model.detect_tables(list(_tables(tmp_path, {"profiles": a, "answers": b}).values()))
    (relation,) = found.relations
    assert relation.cardinality == "1:1"
