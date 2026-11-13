"""Clean N36 (A-20): one entity written several ways, numbered instead of merged. The rule's
clusters on the samples and the bench's owners, its guards (another number, legal form, initials,
opposite, first name), the step (`entity_column`: a new column, the names unchanged; a cluster
said apart or split), the review's clusters side by side, the replay on another file and the
exported SQL."""

import csv
import unicodedata
from pathlib import Path
from typing import Any

import duckdb
import pytest

import quantilo as dx
from quantilo.cleaning import compile_steps, model
from quantilo.cleaning.catalogue import CATALOGUE
from quantilo.cleaning.entities import entity_clusters
from quantilo.cleaning.export import ExportInfo, to_sql
from quantilo.cleaning.rules import entities as rule
from tests.clean_corpus.cases import OWNER_PAIRS, OWNER_SPELLINGS, _owners

SAMPLES = Path(__file__).resolve().parents[3] / "samples"

ROWS = [
    # name, zip, city
    {"name": "NYC Housing Authority", "zip": "10007", "city": "New York"},
    {"name": "N.Y.C. HOUSING AUTHORITY", "zip": "10007", "city": "NY"},
    {"name": "NYC Housing Authorty", "zip": "10 007", "city": "New York"},
    {"name": "NYC Housing Authority", "zip": "11201", "city": "Brooklyn"},
    {"name": "Acme Realty", "zip": "10007", "city": "New York"},
    {"name": "", "zip": "10007", "city": "New York"},
    {"name": "NYC HOUSING AUTHORITY", "zip": "10007", "city": "New York"},
]
CLUSTER = [
    {"anchor": "10007", "name": "nyc housing authority"},
    {"anchor": "10007", "name": "n.y.c. housing authority"},
    {"anchor": "10007", "name": "nyc housing authorty"},
]
STEP = {
    "op": "entity_column",
    "column": "name",
    "anchor": "zip",
    "anchor_kind": "postal_code",
    "to": "name_entity",
    "groups": [CLUSTER],
}


def fold(name: str) -> str:
    """A name as the step reads it: case, accents and runs of spaces folded."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    return " ".join(plain.lower().split())


def write(tmp_path: Path, rows: list[dict[str, Any]], name: str = "data") -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dx.open(path, cache_dir=tmp_path / f"cache-{name}")


def steps(*raw: dict[str, Any]) -> list[Any]:
    return model.parse_steps([{"id": f"s{i}", **r} for i, r in enumerate(raw, 1)]).steps


def numbers(ds: dx.Dataset, found: list[Any]) -> list[Any]:
    """The entity number of each row, in the file's order."""
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(ds.path), found, strict=True)
        return [
            r[0]
            for r in connection.execute(
                f"SELECT name_entity FROM ({compiled.sql()}) ORDER BY __quantilo_row"
            ).fetchall()
        ]
    finally:
        connection.close()


def a20(ds: dx.Dataset) -> list[Any]:
    return [s for s in ds.clean().advanced() if s.rule == "A-20"]


# --- The step ------------------------------------------------------------------------------------


def test_the_records_of_a_cluster_share_one_number_the_names_unchanged(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    # case and spaces folded (rows 0 and 6 are one record); the postcode without its space (row
    # 2); the same name at another postcode another entity; no name, no entity
    assert numbers(ds, steps(STEP)) == [1, 1, 1, 2, 3, None, 1]
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(ds.path), steps(STEP), strict=True)
        names = connection.execute(
            f"SELECT name FROM ({compiled.sql()}) ORDER BY __quantilo_row"
        ).fetchall()
    finally:
        connection.close()
    assert [n for (n,) in names] == [r["name"] or None for r in ROWS]


def test_without_clusters_each_record_is_its_own_entity(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    assert numbers(ds, steps({**STEP, "groups": []})) == [1, 2, 3, 4, 5, None, 1]


def test_a_cluster_said_apart_or_split(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    assert numbers(ds, steps({**STEP, "not_entities": [0]})) == [1, 2, 3, 4, 5, None, 1]
    # the misspelt record taken out: an entity of its own (with any other taken with it)
    assert numbers(ds, steps({**STEP, "split": {"0": [2]}})) == [1, 1, 2, 3, 4, None, 1]


def test_the_step_refuses_what_is_not_a_cluster() -> None:
    with pytest.raises(model.CleanError, match="two records"):
        steps({**STEP, "groups": [CLUSTER[:1]]})
    with pytest.raises(model.CleanError, match="two clusters"):
        steps({**STEP, "groups": [CLUSTER[:2], CLUSTER[1:]]})
    with pytest.raises(model.CleanError, match="said apart"):
        steps({**STEP, "not_entities": [3]})
    with pytest.raises(model.CleanError, match="taken out"):
        steps({**STEP, "split": {"0": [7]}})


def test_the_replay_numbers_the_rows_of_another_file(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    other = write(tmp_path, [ROWS[2], ROWS[4], ROWS[1]], "other")
    report = other.clean().replay(ds.clean(steps(STEP)).recipe())
    assert [s.status for s in report.steps] == ["applies"]
    assert numbers(other, report.adapted) == [1, 2, 1]  # value-based: no row number named


def test_the_exported_sql_gives_the_numbers(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    target = (tmp_path / "out.parquet").as_posix()
    connection = duckdb.connect()
    try:
        script = to_sql(connection, str(ds.path), steps(STEP), ExportInfo(name="t"), target=target)
        connection.execute(script)
        found = connection.execute(
            f"SELECT name, zip, name_entity FROM read_parquet('{target}') "
            "ORDER BY name_entity, name"
        ).fetchall()
    finally:
        connection.close()
    assert {e for *_, e in found if e is not None} == {1, 2, 3}
    assert sorted(n for n, _, e in found if e == 1) == sorted(
        [ROWS[i]["name"] for i in (0, 1, 2, 6)]
    )


# --- The review ----------------------------------------------------------------------------------


def test_the_clusters_side_by_side(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(ds.path), steps({**STEP, "not_entities": [0]}))
        found = entity_clusters(connection, compiled, "s1")
    finally:
        connection.close()
    assert (found.total_clusters, found.total_records, found.total_rows) == (1, 3, 4)
    assert found.columns == ["name", "zip", "city"] and found.to == "name_entity"
    (cluster,) = found.clusters  # the rule's, whatever the review said
    assert [r.rows for r in cluster.records] == [2, 1, 1]
    assert [r.first for r in cluster.records] == [0, 1, 2]
    assert cluster.records[2].spelling == "NYC Housing Authorty"
    assert cluster.records[1].cells == ["N.Y.C. HOUSING AUTHORITY", "10007", "NY"]
    assert cluster.differing == [0, 1, 2]  # names, the postcode's space, the city


def test_the_review_refuses_another_step(tmp_path: Path) -> None:
    ds = write(tmp_path, ROWS)
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(ds.path), steps({"op": "drop_duplicates"}))
        with pytest.raises(model.CleanError, match="does not number entities"):
            entity_clusters(connection, compiled, "s1")
    finally:
        connection.close()


# --- The rule ------------------------------------------------------------------------------------


def record(name: str, company: bool = True) -> rule.Record:
    found = rule.Record("10007", name.lower(), name, 1, 0)
    rule._prepare(found, company)
    return found


@pytest.mark.parametrize(
    ("a", "b", "company", "level"),
    [
        ("NYC HOUSING AUTHORITY", "N.Y.C. HOUSING AUTHORITY", True, 3),
        ("VORNADO OFFICE MANAGEMENT", "VORNADO OFFICE MANAGEMENT, LLC", True, 3),
        ("DEPT OF H P D", "Dept of HPD", True, 3),
        ("Mouse, Wireless", "Wireless Mouse", True, 3),
        ("LE HARVE OWNERS CORP.", "LE HAVRE OWNERS CORP.", True, 2),
        ("NYC HOUSING AUTHORITY", "NYC HOUSING AUTHORTY", True, 2),
        ("SWIG WEILER & ARNOW", "SWIG WEILER & ARRNOW", True, 2),
        ("Hugo Richard", "Hugo Rihard", False, 2),
        ("CONTINENTIAL DEVELOPMENT CORP", "CONTINENTAL DEVELOPEMENT CORP", True, 1),
        # another number, another legal form, other initials, an opposite: two entities
        ("509 MADISON AVENUE ASSOCIATES LP", "590 MADISON AVENUE ASSOCIATES LP", True, 0),
        ("1285 ASSOCIATES LP", "1285 ASSOCIATES LTD", True, 0),
        ("A.I.G. REALTY INC.", "A.E.G. REALTY INC.", True, 0),
        ("NYC HOUSING AUTHORITY", "NYU HOUSING AUTHORITY", True, 0),
        ("A.T.A. CONSTRUTION CORP.", "AIA CONSTRUCTION", True, 0),
        ("BUSHWICK NORTH", "BUSHWICK SOUTH", True, 0),
        # people: another first name, another surname; the order of the words kept
        ("Paul Martin", "Paula Martin", False, 0),
        ("Louis Petit", "Louise Petit", False, 0),
        ("Mario Rossi", "Maria Rossi", False, 0),
        ("Ann William", "Ann Williams", False, 0),
        ("Martin Paul", "Paul Martin", False, 0),
        ("Acme", "Acne", True, 0),  # too short for a slip
    ],
)
def test_names_compared(a: str, b: str, company: bool, level: int) -> None:
    assert rule.level(record(a, company), record(b, company), company) == level


def test_messy_orders_the_customers_typed_again_and_no_other(tmp_path: Path) -> None:
    """B1's A-16 trap: 10 orders typed again under a new number with a slip in the name, the same
    customer_id: each pair one entity, nothing else grouped (a name shared by two customers is
    two entities: another customer_id)."""
    path = SAMPLES / "messy_orders.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    number = {r["order_id"]: int(r["order_id"][4:]) for r in rows if r["order_id"][4:].isdigit()}
    typed = {
        r["customer_id"]: fold(r["customer_name"])
        for r in rows
        if 103_101 <= number.get(r["order_id"], 0) <= 103_110
    }
    spellings = {customer: set() for customer in typed}
    for r in rows:
        if r["customer_id"] in typed:
            spellings[r["customer_id"]].add(fold(r["customer_name"]))
    ds = dx.open(path, cache_dir=tmp_path / "cache")
    (found,) = a20(ds)
    step = found.step
    assert found.id == "A-20:customer_name:entity" and found.mode == "S"
    assert (step.anchor, step.anchor_kind, step.to) == (
        "customer_id",
        "key",
        "customer_name_entity",
    )
    assert {g[0].anchor: {r.name for r in g} for g in step.groups} == spellings
    assert all(len(names) == 2 for names in spellings.values())
    assert "the names are not changed" in found.reason and found.reason.endswith("Check your data.")
    assert CATALOGUE["A-20"].mode == "decision" and not found.auto


def test_owners_one_postcode_numbered_alike_owners_apart(tmp_path: Path) -> None:
    path = tmp_path / "owners.csv"
    path.write_text(_owners(), encoding="utf-8", newline="")
    ds = dx.open(path, cache_dir=tmp_path / "cache")
    (found,) = a20(ds)
    assert found.columns == ["owner_s_business_name"]
    assert found.step.anchor == "owner_s_zip_code"  # the owner's, not the site's zip_code
    clusters = [{r.name for r in g} for g in found.step.groups]
    expected = [
        {name.lower(), *(spelling.lower() for spelling in spellings)}
        for name, spellings in OWNER_SPELLINGS.items()
    ]
    assert sorted(map(sorted, clusters)) == sorted(map(sorted, expected))
    alike = {name.lower() for pair in OWNER_PAIRS for name in pair}
    assert not alike & {name for cluster in clusters for name in cluster}


def test_name_columns_and_their_anchors(tmp_path: Path) -> None:
    rows = [
        {
            "owner_s_business_name": f"OWNER {i % 30}",
            "owner_s_house__": f"{i % 40} MAIN",
            "gis_nta_name": f"PLACE {i % 25}",
            "permittee_s_phone__": f"212555{i % 50:04d}",
            "owner_s_zip_code": f"10{i % 35:03d}",
            "product": f"ITEM {i % 30}",
            "zip_code": f"11{i % 30:03d}",
        }
        for i in range(400)
    ]
    ds = write(tmp_path, rows)
    from quantilo.cleaning.rules.advanced import Advanced, profile

    connection = duckdb.connect()
    try:
        context, source = profile(connection, str(ds.path), ds.types)
        advanced = Advanced(context, source, [])
        names = [c.name for c in rule._name_columns(advanced)]
        assert sorted(names) == ["owner_s_business_name", "product"]
        owner = advanced.column("owner_s_business_name")
        product = advanced.column("product")
        assert owner is not None and product is not None
        assert [c.name for c, _ in rule._anchors(advanced, owner)] == [
            "owner_s_zip_code",
            "zip_code",
        ]  # never the permittee's phone
        assert rule._anchors(advanced, product) == []  # a product by its code only
    finally:
        connection.close()


def test_no_name_no_anchor_nothing(tmp_path: Path) -> None:
    ds = write(tmp_path, [{"city": f"C{i % 30}", "zip": f"{i % 40:05d}"} for i in range(200)])
    assert a20(ds) == []
