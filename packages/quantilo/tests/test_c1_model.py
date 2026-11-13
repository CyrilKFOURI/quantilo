"""C1, project model: keys of three columns, inactive relations, keys compared without case or
accents, slips pointed at, filters that propagate from a dimension, a generated date table, and
columns that name the images of the project."""

from pathlib import Path

import duckdb
import pandas as pd

import quantilo as dx
from quantilo import model
from quantilo.model.build import Filter, Link, check, flat_sql, plan, preview
from quantilo.model.dates import write_date_table
from quantilo.model.images import ImageSetInput, image_references
from quantilo.model.keys import TableInput, profile_table


def table(tmp_path: Path, name: str, frame: pd.DataFrame) -> TableInput:
    found = model.table_of(dx.from_pandas(frame, name, cache_dir=tmp_path / name))
    found.id = name
    return found


def test_a_key_of_three_columns(tmp_path: Path) -> None:
    rows = [
        (store, day, shift) for store in ("S1", "S2") for day in ("mon", "tue") for shift in (1, 2)
    ]
    frame = pd.DataFrame(rows * 1, columns=["store_id", "day_code", "shift_no"])
    frame["hours"] = range(len(frame))
    found = table(tmp_path, "shifts", frame)
    connection = duckdb.connect()
    profile = profile_table(connection, found)
    primary = profile.primary_key
    assert primary is not None and primary.columns == ["store_id", "day_code", "shift_no"]
    assert any(r.code == "R-KEY-03b" for r in primary.reasons)
    connection.close()


def stores(tmp_path: Path) -> tuple[TableInput, TableInput]:
    sales = pd.DataFrame(
        {
            "sale_id": range(1, 13),
            "store_code": ["PARIS", "paris", "Lyon ", "LYON", "Nantes", "nantes"] * 2,
            "amount": [10.0, 20, 30, 40, 50, 60] * 2,
        }
    )
    shops = pd.DataFrame({"code": ["Paris", "Lyon", "Nante"], "region": ["IDF", "ARA", "PDL"]})
    return table(tmp_path, "sales", sales), table(tmp_path, "shops", shops)


def test_a_normalized_comparison_and_slips(tmp_path: Path) -> None:
    sales, shops = stores(tmp_path)
    connection = duckdb.connect()
    exact = check(connection, sales, ["store_code"], shops, ["code"])
    assert exact.matched_rows == 2 and exact.normalized_gain == 6  # 'Lyon ' trimmed already
    assert any(r.code == "R-REL-14" for r in exact.reasons)
    loose = check(connection, sales, ["store_code"], shops, ["code"], "normalized")
    assert loose.matched_rows == 8 and loose.orphan_rows == 4  # Nantes is a slip, not joined
    near = check(connection, sales, ["store_code"], shops, ["code"])
    assert near.near_misses == [("Nantes", "Nante")]  # a case only is R-REL-14, not a slip
    assert any(r.code == "R-REL-15" for r in near.reasons)
    connection.close()


def test_inactive_relations_and_propagated_filters(tmp_path: Path) -> None:
    sales, shops = stores(tmp_path)
    links = [Link("sales", ["store_code"], "shops", ["code"], match="normalized")]
    tables = {"sales": sales, "shops": shops}
    connection = duckdb.connect()
    built = plan("sales", tables, links)
    assert [c.name for c in built.columns][-1] == "shops_region"
    inactive = plan(
        "sales", tables, [Link("sales", ["store_code"], "shops", ["code"], active=False)]
    )
    assert not inactive.steps and any(r.code == "R-REL-13" for r in inactive.skipped)
    filtered = plan("sales", tables, links, filters=[Filter("shops", "region", "eq", ["IDF"])])
    rows = connection.execute(f"SELECT store_code FROM ({flat_sql(filtered)})").fetchall()
    assert len(rows) == 4 and {r[0].strip().lower() for r in rows} == {"paris"}
    shown = preview(connection, filtered)
    assert shown.rows_after == 4 and shown.filtered_out == 8 and not shown.refused
    on_fact = plan("sales", tables, links, filters=[Filter("sales", "amount", "gt", ["35"])])
    assert connection.execute(f"SELECT count(*) FROM ({flat_sql(on_fact)})").fetchone() == (6,)
    connection.close()


def test_a_generated_date_table(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"sold": pd.to_datetime(["2024-02-27", "2024-03-02", "2024-02-28"]), "n": [1, 2, 3]}
    )
    fact = table(tmp_path, "orders", frame)
    connection = duckdb.connect()
    target = tmp_path / "dates.parquet"
    made = write_date_table(connection, fact, "sold", target)
    assert made.rows == 5  # 27, 28, 29 February (2024 is a leap year), 1 and 2 March
    rows = connection.execute(
        "SELECT CAST(date AS VARCHAR), year_month, weekday, weekend "
        f"FROM read_parquet('{target}') ORDER BY date"
    ).fetchall()
    assert rows[2] == ("2024-02-29", "2024-02", 4, False) and rows[3][3] is False
    assert rows[-1] == ("2024-03-02", "2024-03", 6, True)
    connection.close()


def test_columns_that_name_the_images_of_the_project(tmp_path: Path) -> None:
    images = pd.DataFrame(
        {
            "image_id": [f"i{n}" for n in range(8)],
            "file_name": [f"P{n:03d}_left.png" for n in range(8)],
            "rel_path": [f"mammo/P{n:03d}_left.png" for n in range(8)],
        }
    )
    images_path = tmp_path / "images.parquet"
    images.to_parquet(images_path)
    clinic = pd.DataFrame(
        {
            "patient": [f"P{n:03d}" for n in range(8)],
            "scan": [f"Mammo\\P{n:03d}_LEFT.PNG" for n in range(7)] + ["../../etc/passwd"],
            "stem": [f"P{n:03d}_left" for n in range(8)],
        }
    )
    clinical = table(tmp_path, "clinic", clinic)
    connection = duckdb.connect()
    found = image_references(
        connection, clinical, [ImageSetInput("set", "Mammograms", str(images_path))]
    )
    by_column = {ref.column: ref for ref in found}
    assert by_column["scan"].by == "path" and by_column["scan"].found == 7
    assert by_column["scan"].refused == 1 and "never followed" in by_column["scan"].reason
    assert by_column["stem"].by == "stem"
    assert "patient" not in by_column
    connection.close()
