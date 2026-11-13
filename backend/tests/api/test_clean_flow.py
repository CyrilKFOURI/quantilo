"""Clean, audit of 2026-11-12: N05 (don't suggest again), N31 (declared checks), N45 (cleaning as
SQL and Python) and N46 (the cleaning journal)."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import duckdb
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, default_samples_dir
from tests.api.test_clean import BASE, save
from tests.api.test_profiling import import_path


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def spaced(rows: int = 30) -> str:
    """Extra spaces in `city` and `brand` (B-01), an empty column `notes` (B-19)."""
    lines = ["order_id,city,brand,notes,amount"]
    for i in range(rows):
        city = " Paris" if i % 3 == 0 else "Lyon"
        brand = "Acme " if i % 4 == 0 else "Acme"
        lines.append(f"O{i:03d},{city},{brand},,{10 + i % 5}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def traps(client: TestClient) -> str:
    return import_path(client, default_samples_dir() / "dirty_traps.csv")


@pytest.fixture
def shop(client: TestClient, tmp_path: Path) -> str:
    return import_path(client, write_csv(tmp_path / "shop.csv", spaced()))


def found(client: TestClient, dataset: str) -> dict[str, Any]:
    response = client.get(f"{BASE}/datasets/{dataset}/suggestions")
    assert response.status_code == 200, response.text
    return response.json()


def dismiss(client: TestClient, dataset: str, **body: Any) -> dict[str, Any]:
    response = client.post(f"{BASE}/datasets/{dataset}/dismissed", json=body)
    assert response.status_code == 201, response.text
    return response.json()


# --- N05: don't suggest again ---------------------------------------------------------------


def test_a_dismissed_suggestion_is_hidden_listed_and_never_applied(
    client: TestClient, shop: str
) -> None:
    before = {s["id"]: s for s in found(client, shop)["suggestions"]}
    assert {"B-01:city", "B-01:brand"} <= set(before) and before["B-01:city"]["auto"]
    draft = client.get(f"{BASE}/datasets/{shop}/draft").json()
    assert "B-01:city" in {s["suggestion"] for s in draft["steps"]}

    done = dismiss(
        client, shop, scope="suggestion", rule="B-01", suggestion="B-01:city", note="kept"
    )
    assert done["hidden"] == ["B-01:city"]
    assert done["dismissal"]["reason"].startswith("You dismissed this suggestion")
    after = found(client, shop)
    assert "B-01:city" not in {s["id"] for s in after["suggestions"]}
    assert [h["suggestion"]["id"] for h in after["hidden"]] == ["B-01:city"]
    assert after["hidden"][0]["dismissal"] == done["dismissal"]["id"]
    assert [d["note"] for d in after["dismissals"]] == ["kept"]
    # never applied for the user: the automatic draft leaves it out
    draft = client.get(f"{BASE}/datasets/{shop}/draft").json()
    assert "B-01:city" not in {s["suggestion"] for s in draft["steps"]}
    assert "B-01:brand" in {s["suggestion"] for s in draft["steps"]}
    # twice is kept once
    assert (
        dismiss(client, shop, scope="suggestion", rule="B-01", suggestion="B-01:city")["dismissal"][
            "id"
        ]
        == done["dismissal"]["id"]
    )

    restored = client.delete(f"{BASE}/dismissed/{done['dismissal']['id']}")
    assert restored.status_code == 204
    assert "B-01:city" in {s["id"] for s in found(client, shop)["suggestions"]}
    assert client.delete(f"{BASE}/dismissed/{done['dismissal']['id']}").status_code == 404


def test_a_rule_here_and_a_column_everywhere(client: TestClient, shop: str, tmp_path: Path) -> None:
    other = import_path(client, write_csv(tmp_path / "other.csv", spaced(20)))
    rule = dismiss(client, shop, scope="rule", rule="B-01")
    assert set(rule["hidden"]) == {"B-01:city", "B-01:brand"}
    assert not [s for s in found(client, shop)["suggestions"] if s["rule"] == "B-01"]
    assert [s for s in found(client, other)["suggestions"] if s["rule"] == "B-01"]  # here only

    column = dismiss(client, other, scope="column", rule="B-01", column="brand")
    assert column["dismissal"]["dataset_id"] is None
    for dataset in (shop, other):
        assert "B-01:brand" in {h["suggestion"]["id"] for h in found(client, dataset)["hidden"]}
    assert "B-01:city" in {s["id"] for s in found(client, other)["suggestions"]}
    bad = client.post(f"{BASE}/datasets/{shop}/dismissed", json={"scope": "column", "rule": "B-01"})
    assert bad.status_code == 422


def test_a_cleaned_version_shares_them_and_its_recipe_carries_them(
    client: TestClient, shop: str, tmp_path: Path
) -> None:
    dismiss(client, shop, scope="rule", rule="B-19", title="Remove the empty column notes")
    steps = client.get(f"{BASE}/datasets/{shop}/draft").json()["steps"]
    child = save(client, shop, steps)
    assert not [s for s in found(client, child)["suggestions"] if s["rule"] == "B-19"]

    recipe = client.get(f"{BASE}/versions/{child}/recipe").json()
    assert recipe["schema"] == 2
    assert [(d["scope"], d["rule"]) for d in recipe["dismissed"]] == [("rule", "B-19")]
    other = import_path(client, write_csv(tmp_path / "next.csv", spaced(24)))
    report = client.post(f"{BASE}/datasets/{other}/replay", json={"recipe": recipe}).json()
    assert report["hidden"] >= 1 and report["dismissed"][0]["rule"] == "B-19"
    assert not [s for s in report["new_problems"] if s["rule"] == "B-19"]


# --- N31: declared checks ------------------------------------------------------------------------


def tickets(rows: int = 80, wrong: int = 9) -> str:
    """Requests closed before they were opened on `wrong` rows (nyc_311), one id repeated."""
    lines = ["ticket,created,closed,status"]
    for i in range(rows):
        day = 1 + i % 20
        closed = f"2024-03-{day + 2:02d}" if i >= wrong else "2024-02-28"
        ticket = f"T{i:04d}" if i != 20 else "T0019"
        lines.append(f"{ticket},2024-03-{day:02d},{closed},{'open' if i % 3 else 'closed'}")
    return "\n".join(lines) + "\n"


DECLARED = [
    {"id": "c1", "kind": "compare", "columns": ["created", "closed"], "op": "le"},
    {"id": "c2", "kind": "unique", "columns": ["ticket"]},
    {"id": "c3", "kind": "in_list", "columns": ["status"], "values": ["open", "closed"]},
    {"id": "c4", "kind": "not_empty", "columns": ["gone"]},
]


def test_declared_checks_run_again_at_each_import_and_version(
    client: TestClient, tmp_path: Path
) -> None:
    source = write_csv(tmp_path / "tickets.csv", tickets())
    dataset = import_path(client, source)
    assert client.get(f"{BASE}/datasets/{dataset}/checks").json()["checks"] == []
    saved = client.put(f"{BASE}/datasets/{dataset}/checks", json={"checks": DECLARED}).json()
    results = {r["id"]: r for r in saved["report"]["results"]}
    assert (results["c1"]["status"], results["c1"]["failing"]) == ("fail", 9)
    assert results["c1"]["description"] == "created ≤ closed"
    assert (results["c2"]["failing"], results["c3"]["status"]) == (2, "pass")
    assert results["c4"]["status"] == "error" and "not a column" in results["c4"]["message"]
    assert saved["report"]["failing_rows"] == 11 and saved["trigger"] == "declared"
    again = client.get(f"{BASE}/datasets/{dataset}/checks").json()
    assert again["ran_at"] == saved["ran_at"]  # not run again: same data, same checks

    rows = client.get(f"{BASE}/datasets/{dataset}/checks/c1/rows").json()
    assert rows["failing"] == 9 and rows["columns"][:2] == ["created", "closed"]
    assert rows["rows"][0]["row"] == 0 and len(rows["rows"]) == 9
    assert client.get(f"{BASE}/datasets/{dataset}/checks/c9/rows").status_code == 404
    bad = [{"id": "c1", "kind": "unique", "columns": []}]
    assert client.put(f"{BASE}/datasets/{dataset}/checks", json={"checks": bad}).status_code == 422

    job = client.post(f"/api/v1/datasets/{dataset}/reimport").json()
    assert job["status"] == "done", job
    after = client.get(f"{BASE}/datasets/{dataset}/checks").json()
    assert after["trigger"] == "import" and after["report"]["failing_rows"] == 11  # still 9 + 2
    assert [run["trigger"] for run in after["history"]][:2] == ["import", "declared"]

    steps = [
        {
            "id": "s1",
            "op": "filter_rows",
            "conditions": [{"column": "closed", "op": "eq", "value": "2024-02-28"}],
            "action": "remove",
        }
    ]
    child = save(client, dataset, steps)
    on_child = client.get(f"{BASE}/datasets/{child}/checks").json()
    assert on_child["original_id"] == dataset and on_child["trigger"] == "version"
    by_id = {r["id"]: r for r in on_child["report"]["results"]}
    assert by_id["c1"]["status"] == "pass"  # the cleaning removed those rows
    family = client.get(f"{BASE}/datasets/{dataset}/versions").json()
    assert family["versions"][0]["checks"]["failed"] == 1  # the repeated id

    recipe = client.get(f"{BASE}/versions/{child}/recipe").json()
    assert recipe["schema"] == 2 and [c["id"] for c in recipe["checks"]] == ["c1", "c2", "c3", "c4"]
    other = import_path(client, write_csv(tmp_path / "april.csv", tickets(40, 4)))
    report = client.post(f"{BASE}/datasets/{other}/replay", json={"recipe": recipe}).json()
    assert report["check_report"]["results"][0]["failing"] == 4  # on the file as it is


# --- N45: the cleaning as SQL and Python ---------------------------------------------------------


def rows_of(path: Path) -> tuple[list[Any], list[Any]]:
    """A Parquet file's columns with their types, and its rows in file order."""
    connection = duckdb.connect()
    source = f"read_parquet('{path.as_posix()}', file_row_number = true)"
    described = connection.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
    rows = connection.execute(
        f"SELECT * EXCLUDE (file_row_number) FROM {source} ORDER BY file_row_number"
    ).fetchall()
    return [(d[0], d[1]) for d in described if d[0] != "file_row_number"], rows


def test_the_exported_sql_and_python_give_the_saved_version(
    client: TestClient, traps: str, settings: Settings, tmp_path: Path
) -> None:
    steps = client.get(f"{BASE}/datasets/{traps}/draft").json()["steps"]
    steps += [
        {"id": "s90", "op": "rename_column", "column": "city", "to": "town", "enabled": False}
    ]
    child = save(client, traps, steps)
    saved = rows_of(Path(settings.home) / "datasets" / child / "data.parquet")

    response = client.get(f"{BASE}/versions/{child}/sql")
    assert response.status_code == 200 and "attachment" in response.headers["content-disposition"]
    sql = response.text
    assert sql.startswith("-- Cleaning “dirty_traps · Cleaned v1”, exported from Quantilo")
    assert "applied automatically, a reliable fix ·" in sql
    assert "s90 · added by the user · rename_column on city: turned off, not run." in sql
    target = tmp_path / "cleaned.parquet"
    duckdb.connect().execute(sql.replace("TO 'cleaned.parquet'", f"TO '{target.as_posix()}'"))
    assert rows_of(target) == saved

    script = tmp_path / "cleaning.py"
    script.write_text(client.get(f"{BASE}/versions/{child}/python").text, encoding="utf-8")
    target.unlink()  # the script writes it again
    env = {**os.environ, "QUANTILO_CACHE": str(tmp_path / "cache")}
    done = subprocess.run(
        [sys.executable, str(script)],  # its defaults: the dataset's copy, cleaned.parquet
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert rows_of(tmp_path / "cleaned.parquet") == saved
    assert client.get(f"{BASE}/versions/{traps}/sql").status_code == 404


# --- N46: the cleaning journal -----------------------------------------------------------------


def test_the_journal_says_who_changed_what_and_goes_into_the_version(
    client: TestClient, traps: str
) -> None:
    draft = client.get(f"{BASE}/datasets/{traps}/draft").json()
    assert draft["automatic"] and draft["steps"]
    automatic = [s for s in draft["steps"] if s["origin"] == "auto"]
    kept = [s for s in draft["steps"] if s["id"] != automatic[0]["id"]]
    manual = {"id": "s99", "op": "rename_column", "column": "city", "to": "town"}
    client.put(f"{BASE}/datasets/{traps}/draft", json={"steps": [*kept, manual]})
    entries = client.get(f"{BASE}/datasets/{traps}/journal").json()
    applied = [e for e in entries if e["action"] == "applied"]
    assert len(applied) == len(automatic) and {e["actor"] for e in applied} == {"auto"}
    assert all(e["detail"] and e["cells"] is not None for e in applied)  # its title and cells
    changes = [(e["actor"], e["action"], e["step_id"]) for e in entries if e["actor"] == "user"]
    assert changes == [("user", "added", "s99"), ("user", "removed", automatic[0]["id"])]
    off = [{**s, "enabled": False} if s["id"] == "s99" else s for s in [*kept, manual]]
    client.put(f"{BASE}/datasets/{traps}/draft", json={"steps": off})
    last = client.get(f"{BASE}/datasets/{traps}/journal").json()[-1]
    assert (last["action"], last["step_id"], last["columns"]) == ("turned_off", "s99", ["city"])

    child = save(client, traps, [*kept, manual])
    assert client.get(f"{BASE}/datasets/{traps}/journal").json() == []  # it went into v1
    journal = client.get(f"{BASE}/versions/{child}/journal").json()
    assert journal["name"].endswith("Cleaned v1") and journal["made_from"] == "dirty_traps"
    assert journal["applied"] == {"auto": len(automatic) - 1, "suggested": 0, "manual": 1}
    assert journal["entries"][-1]["action"] == "saved" and journal["entries"][-1]["actor"] == "user"
    assert journal["report"]["rows_removed"] > 0
    removing = [s for s in journal["steps"] if s["rows_removed"]]
    assert removing and removing[0]["title"]  # rows removed, and why

    markdown = client.get(f"{BASE}/versions/{child}/journal/markdown")
    assert markdown.status_code == 200 and "attachment" in markdown.headers["content-disposition"]
    text = markdown.text
    assert text.startswith("# Cleaning log: dirty_traps · Cleaned v1")
    for part in ("## Steps", "## Cells changed per column", "## Rows removed", "## Journal"):
        assert part in text
    assert "applied automatically (a reliable fix)" in text and "you removed" in text
    page = client.get(f"{BASE}/versions/{child}/journal/html").text
    assert page.startswith("<!doctype html>") and "<script" not in page


def test_dismissals_and_checks_are_in_the_journal(client: TestClient, shop: str) -> None:
    dismiss(client, shop, scope="rule", rule="B-19")
    checks = [{"id": "c1", "kind": "unique", "columns": ["order_id"]}]
    client.put(f"{BASE}/datasets/{shop}/checks", json={"checks": checks})
    entries = client.get(f"{BASE}/datasets/{shop}/journal").json()
    assert [e["action"] for e in entries] == ["dismissed", "checks"]
    assert entries[1]["detail"] == "0 → 1 checks: order_id is unique"
