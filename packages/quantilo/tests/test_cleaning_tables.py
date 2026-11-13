"""N32 (audit 2026-11-12, § 4.4): orphan keys between the tables of a project (T-01, T-02)."""

from pathlib import Path

import duckdb
import pandas as pd

import quantilo as dx
from quantilo import model
from quantilo.cleaning import catalogue
from quantilo.cleaning import model as m
from quantilo.cleaning.rules.tables import TableLink, suggest_tables
from quantilo.cleaning.suggestion import CleanSuggestion

CUSTOMERS = [
    "ALFKI",
    "ANATR",
    "ANTON",
    "AROUT",
    "BERGS",
    "BLAUS",
    "BLONP",
    "BOLID",
    "BONAP",
    "BOTTM",
]


def _table(tmp_path: Path, name: str, frame: pd.DataFrame) -> model.TableInput:
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    dataset = dx.open(path, cache_dir=tmp_path / "cache")
    table = model.table_of(dataset)
    return model.TableInput(name, name, table.storage_path, table.columns)


def _northwind(tmp_path: Path, orders_customers: list[str], customers: list[str]) -> TableLink:
    orders = pd.DataFrame(
        {"orderID": range(10_248, 10_248 + len(orders_customers)), "customerID": orders_customers}
    )
    names = pd.DataFrame(
        {"customerID": customers, "companyName": [f"Company {c}" for c in customers]}
    )
    return TableLink(
        child=_table(tmp_path, "orders", orders),
        child_columns=["customerID"],
        parent=_table(tmp_path, "customers", names),
        parent_columns=["customerID"],
        project="Northwind",
    )


def _found(link: TableLink, dataset: str) -> dict[str, CleanSuggestion]:
    connection = duckdb.connect()
    try:
        return {s.id: s for s in suggest_tables(connection, dataset, [link])}
    finally:
        connection.close()


def test_no_orphan_no_suggestion(tmp_path: Path) -> None:
    link = _northwind(tmp_path, CUSTOMERS[:8] * 3, CUSTOMERS)  # 2 customers without orders
    assert _found(link, "orders") == {}
    assert _found(link, "customers") == {}


def test_three_customers_removed(tmp_path: Path) -> None:
    """The audit's test: Northwind without 3 customers → T-01 on 3 keys, rows kept."""
    link = _northwind(tmp_path, CUSTOMERS * 3, CUSTOMERS[3:])
    found = _found(link, "orders")
    assert list(found) == ["T-01:customerID:customers"]
    orphans = found["T-01:customerID:customers"]
    assert orphans.mode == "I" and orphans.step is None and orphans.layer == "advanced"
    assert orphans.catalogue == "T-01" and not orphans.auto and orphans.risk == "review"
    assert orphans.title == "9 rows of orders point to a customerID missing from customers"
    assert "'ALFKI' (3 rows)" in orphans.reason and "3 values in all" in orphans.reason
    assert "They are kept" in orphans.reason
    # the rows, as a Data filter (see them, pin them)
    assert orphans.where == [
        m.RowCondition(column="customerID", op="in", values=["ALFKI", "ANATR", "ANTON"])
    ]
    assert _found(link, "customers") == {}  # the dimension has no problem of its own


def test_a_typo_gets_a_fix(tmp_path: Path) -> None:
    """'ALFKJ' for 'ALFKI' → T-01 and a map_values to tick (R-REL-15), never applied."""
    link = _northwind(tmp_path, CUSTOMERS * 2 + ["ALFKJ", "ALFKJ", "ZZZZZ"], CUSTOMERS)
    found = _found(link, "orders")
    assert set(found) == {"T-01:customerID:customers", "T-01:customerID:customers:near"}
    fix = found["T-01:customerID:customers:near"]
    assert fix.mode == "S" and fix.catalogue == "T-01:near" and not fix.auto
    assert isinstance(fix.step, m.MapValues) and fix.step.map == {"ALFKJ": "ALFKI"}
    assert fix.step.origin == "suggested" and fix.step.rule == "T-01" and fix.cells == 2
    assert "'ALFKJ' (2 rows) → 'ALFKI'" in fix.reason and "R-REL-15" in fix.reason
    orphans = found["T-01:customerID:customers"]
    assert orphans.title.startswith("3 rows of orders")
    assert "one letter away" in orphans.reason
    assert catalogue.CATALOGUE["T-01:near"].mode == "decision"


def test_a_key_repeated_in_the_dimension(tmp_path: Path) -> None:
    link = _northwind(tmp_path, CUSTOMERS * 2, [*CUSTOMERS, "ALFKI"])
    found = _found(link, "customers")
    assert list(found) == ["T-02:customerID:orders"]
    repeated = found["T-02:customerID:orders"]
    assert repeated.mode == "I" and repeated.catalogue == "T-02"
    assert repeated.title == "1 key of customerID repeats in customers"
    assert "R-BLD-03" in repeated.reason
    assert repeated.where == [m.RowCondition(column="customerID", op="in", values=["ALFKI"])]
    assert _found(link, "orders") == {}  # every order finds its customer


def test_whole_number_keys(tmp_path: Path) -> None:
    child = _table(tmp_path, "details", pd.DataFrame({"productID": [1, 2, 3, 77, 78, 78]}))
    parent = _table(
        tmp_path, "products", pd.DataFrame({"productID": range(1, 78), "name": ["p"] * 77})
    )
    link = TableLink(child, ["productID"], parent, ["productID"])
    orphans = _found(link, "details")["T-01:productID:products"]
    assert orphans.title.startswith("2 rows of details")
    assert orphans.where == [m.RowCondition(column="productID", op="in", values=["78"])]
