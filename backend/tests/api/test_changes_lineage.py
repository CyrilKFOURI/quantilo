"""Changes between an original and its cleaned versions (095): rows paired by the cleaning's row
map, said in the answer; an earlier import chosen with `version`."""

import csv
import io
import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import default_samples_dir
from tests.api.test_changes import orders, reimport, write
from tests.api.test_clean import BASE, save
from tests.api.test_profiling import import_path

API = "/api/v1/datasets"


@pytest.fixture
def traps(client: TestClient, tmp_path: Path) -> str:
    source = tmp_path / "dirty_traps.csv"
    shutil.copyfile(default_samples_dir() / "dirty_traps.csv", source)
    return import_path(client, source)


def draft(client: TestClient, dataset_id: str) -> list[dict]:
    return client.get(f"{BASE}/datasets/{dataset_id}/draft").json()["steps"]


def rows(client: TestClient, dataset_id: str, **params: str) -> dict:
    response = client.get(f"{API}/{dataset_id}/changes/rows", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_a_cleaned_version_against_its_original_pairs_rows_by_the_rowmap(
    client: TestClient, traps: str
) -> None:
    steps = draft(client, traps)
    v1 = save(client, traps, steps)
    removed = client.get(f"{BASE}/datasets/{v1}/versions").json()["versions"][0]["report"]

    found = rows(client, v1, vs=traps)
    assert found["alignment"] == "rowmap" and found["rowmap"] is True
    assert found["alignment_note"] is None
    result = found["result"]
    assert result["key"] == "row" and result["aligned"] is True
    assert result["removed"] == removed["rows_removed"] == 7
    assert result["added"] == 0 and result["rows_before"] == 407
    assert result["modified"] > 0
    assert all(key.isdigit() for key in result["removed_keys"])

    # From the original's side, the same pairing read backwards.
    back = rows(client, traps, vs=v1)["result"]
    assert (back["added"], back["removed"], back["modified"]) == (7, 0, result["modified"])

    # The trap file has no clean identifier (order_id repeats and has gaps): only the row map
    # pairs its rows. A key can still be asked for, and is refused with its reason.
    assert found["candidates"] == []
    keyed = client.get(f"{API}/{v1}/changes/rows", params={"vs": traps, "key": "order_id"})
    assert keyed.status_code == 422 and "missing values" in keyed.json()["error"]["message"]

    export = client.get(f"{API}/{v1}/changes/rows/export", params={"vs": traps})
    assert export.status_code == 200
    header = next(csv.reader(io.StringIO(export.text)))
    assert header == ["change", "row", "column", "before", "after"]


def test_two_cleaned_versions_pair_through_both_rowmaps(client: TestClient, traps: str) -> None:
    steps = draft(client, traps)
    v1 = save(client, traps, steps)
    v2 = save(client, traps, steps)
    found = rows(client, v2, vs=v1)
    assert found["alignment"] == "rowmap"
    assert (found["result"]["modified"], found["result"]["added"], found["result"]["removed"]) == (
        0,
        0,
        0,
    )
    summary = client.get(f"{API}/{v2}/changes", params={"vs": v1}).json()
    assert summary["level"] == "none"


def test_a_reimported_original_falls_back_to_a_key_and_says_why(
    client: TestClient, traps: str
) -> None:
    v1 = save(client, traps, draft(client, traps))
    source = Path(client.get(f"{API}/{traps}").json()["source_path"])
    stat = source.stat()
    os.utime(source, (stat.st_atime, stat.st_mtime + 10))
    reimport(client, traps)
    found = rows(client, v1, vs=traps)
    assert found["rowmap"] is False and found["alignment"] is None
    assert "reimported since Cleaned v1" in found["alignment_note"]
    assert found["available"] is False and "No column identifies" in found["reason"]


def test_unrelated_datasets_say_nothing_about_a_rowmap(client: TestClient, tmp_path: Path) -> None:
    first = import_path(client, write(tmp_path / "a.csv", orders()))
    second = import_path(client, write(tmp_path / "b.csv", orders(changed=5)))
    found = rows(client, second, vs=first)
    assert found["alignment"] == "key" and found["rowmap"] is False
    assert found["alignment_note"] is None


def test_an_earlier_import_through_version(client: TestClient, tmp_path: Path) -> None:
    source = write(tmp_path / "orders.csv", orders())
    dataset_id = import_path(client, source)
    write(source, orders(rows=320))
    reimport(client, dataset_id)
    write(source, orders(rows=350))
    reimport(client, dataset_id)
    previous = client.get(f"{API}/{dataset_id}/changes").json()
    assert previous["rows_change"] == 30
    first = client.get(f"{API}/{dataset_id}/changes", params={"version": 1}).json()
    assert first["rows_change"] == 50
    assert client.get(f"{API}/{dataset_id}/changes", params={"version": 3}).status_code == 404
    gone = rows(client, dataset_id, version="1")
    assert gone["available"] is False  # only the previous import's rows are kept


def test_the_agent_compares_a_cleaned_version_with_its_original(
    client: TestClient, traps: str
) -> None:
    from app.core.tools import ToolContext

    v1 = save(client, traps, draft(client, traps))
    tool = client.app.state.agent.registry.get("what_changed")  # type: ignore[attr-defined]
    assert tool is not None
    output = tool.run(ToolContext(v1), tool.parse({}))
    assert output.link.params == {"vs": traps}
    assert "7 rows (" in output.summary  # 407 rows, 7 removed by the cleaning
