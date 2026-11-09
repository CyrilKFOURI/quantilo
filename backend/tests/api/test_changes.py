"""What changed between two versions (031): photos at import, the previous copy kept at a
reimport, the comparison, rows, KPIs, relations, and comparing two datasets."""

import csv
import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.api.test_profiling import import_path

API = "/api/v1/datasets"
KPIS = "/api/v1/modules/kpi/datasets"
REGIONS = ["North", "South", "East", "West"]


def orders(
    rows: int = 300, factor: float = 1.0, changed: int = 0, region: str | None = None
) -> list[dict[str, object]]:
    found = []
    for i in range(rows):
        amount = round(10 + (i * 37 % 101) * 1.5, 2) * factor
        if i < changed:
            amount += 5
        found.append(
            {
                "order_id": f"o{i:04d}",
                "amount": round(amount, 2),
                "region": region if region and i % 10 == 0 else REGIONS[i % 4],
                "month": f"2026-{1 + i % 6:02d}-15",
                "units": i % 7 + 1,
            }
        )
    return found


def write(path: Path, rows: list[dict[str, object]], drop: str | None = None) -> Path:
    columns = [c for c in rows[0] if c != drop]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))  # a new version of the source
    return path


def reimport(client: TestClient, dataset_id: str) -> None:
    job = client.post(f"{API}/{dataset_id}/reimport").json()
    assert job["status"] == "done", job


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return write(tmp_path / "orders.csv", orders())


def test_a_first_import_has_no_previous_version(client: TestClient, source: Path) -> None:
    dataset_id = import_path(client, source)
    versions = client.get(f"{API}/{dataset_id}/versions").json()
    assert [(v["version"], v["current"], v["rows"]) for v in versions] == [(1, True, 300)]
    response = client.get(f"{API}/{dataset_id}/changes")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_previous_version"


def test_reimport_without_change(client: TestClient, source: Path) -> None:
    dataset_id = import_path(client, source)
    write(source, orders())
    reimport(client, dataset_id)
    changes = client.get(f"{API}/{dataset_id}/changes").json()
    assert changes["level"] == "none"
    assert changes["level_label"] == "No notable change"
    assert all((s["psi"] or 0) < 0.1 for s in changes["distributions"])
    versions = client.get(f"{API}/{dataset_id}/versions").json()
    assert [(v["version"], v["current"], v["kept"]) for v in versions] == [
        (2, True, True),
        (1, False, True),  # its Parquet copy is kept (under 2 GB)
    ]
    assert changes["headline"].startswith(f"Compared with {versions[1]['imported_at'][:10]}:")


def test_reimport_with_changes(client: TestClient, source: Path) -> None:
    dataset_id = import_path(client, source)
    write(source, orders(factor=1.3, region="Outre-mer"))
    reimport(client, dataset_id)
    changes = client.get(f"{API}/{dataset_id}/changes").json()
    psis = [s["psi"] for s in changes["distributions"]]
    assert psis == sorted(psis, reverse=True)  # strongest shift first
    first = next(s for s in changes["distributions"] if s["column"] == "amount")
    assert first["psi"] > 0.25 and first["drift"] == "strong"
    assert first["histogram"]["before"] and first["histogram"]["after"]
    region = next(s for s in changes["distributions"] if s["column"] == "region")
    assert [(a["value"], round(a["after"], 3)) for a in region["appeared"]] == [("Outre-mer", 0.1)]
    assert changes["level"] == "check"
    assert changes["counts"]["new_categories"] == 1


def test_a_column_used_by_a_kpi_that_disappears_is_breaking(
    client: TestClient, source: Path
) -> None:
    dataset_id = import_path(client, source)
    definition = {"measures": [{"id": "m1", "agg": "sum", "column": "units"}]}
    created = client.post(
        f"{KPIS}/{dataset_id}/definitions", json={"name": "Units sold", "definition": definition}
    )
    assert created.status_code == 201
    write(source, orders(), drop="units")
    reimport(client, dataset_id)
    changes = client.get(f"{API}/{dataset_id}/changes").json()
    assert changes["level"] == "breaking"
    removed = next(c for c in changes["schema"] if c["kind"] == "removed")
    assert removed["column"] == "units" and removed["used_by"] == ["KPI “Units sold”"]
    [kpi] = client.get(f"{API}/{dataset_id}/changes/kpis").json()
    assert kpi["name"] == "Units sold" and kpi["status"] == "broken"
    assert kpi["before"] is not None and kpi["problem"]


def test_kpis_before_and_after_with_common_periods(client: TestClient, source: Path) -> None:
    dataset_id = import_path(client, source)
    definition = {
        "measures": [{"id": "m1", "agg": "sum", "column": "amount"}],
        "time": {"column": "month", "grain": "month"},
    }
    client.post(
        f"{KPIS}/{dataset_id}/definitions", json={"name": "Sales", "definition": definition}
    )
    write(source, orders(changed=12))
    reimport(client, dataset_id)
    [kpi] = client.get(f"{API}/{dataset_id}/changes/kpis").json()
    assert kpi["status"] == "ok"
    assert kpi["after"] == pytest.approx(kpi["before"] + 12 * 5)
    assert len(kpi["common"]) == 6
    assert kpi["revised"] == 6  # the 12 changed rows fall in every month: past values revised


def test_rows_twelve_modified_and_the_export(client: TestClient, source: Path) -> None:
    dataset_id = import_path(client, source)
    write(source, orders(changed=12))
    reimport(client, dataset_id)
    answer = client.get(f"{API}/{dataset_id}/changes/rows").json()
    assert answer["available"] and answer["candidates"][0] == "order_id"
    result = answer["result"]
    assert (result["key"], result["modified"], result["added"], result["removed"]) == (
        "order_id",
        12,
        0,
        0,
    )
    assert result["columns"] == [{"column": "amount", "modified": 12}]
    exported = client.get(f"{API}/{dataset_id}/changes/rows/export", params={"key": "order_id"})
    assert exported.status_code == 200
    assert exported.headers["content-type"].startswith("text/csv")
    lines = list(csv.DictReader(io.StringIO(exported.text)))
    assert len(lines) == 12
    assert all(float(line["after"]) == pytest.approx(float(line["before"]) + 5) for line in lines)
    assert lines[0]["order_id"] == "o0000" and lines[0]["column"] == "amount"


def test_never_keep_explains_the_rows_tab(client: TestClient, source: Path) -> None:
    settings = client.get("/api/v1/settings").json()
    assert settings["versions"]["keep"] == "under_limit"
    settings["versions"]["keep"] = "never"
    assert client.put("/api/v1/settings", json=settings).status_code == 200
    dataset_id = import_path(client, source)
    write(source, orders(changed=12))
    reimport(client, dataset_id)
    answer = client.get(f"{API}/{dataset_id}/changes/rows").json()
    assert not answer["available"]
    assert "Keep the previous version" in answer["reason"]
    assert client.get(f"{API}/{dataset_id}/changes").status_code == 200
    relations = client.get(f"{API}/{dataset_id}/changes/relations").json()
    assert not relations["available"]
    export = client.get(f"{API}/{dataset_id}/changes/rows/export", params={"key": "order_id"})
    assert export.status_code == 409
    assert client.get("/api/v1/storage").json()["versions_bytes"] == 0


def test_only_the_previous_copy_is_kept(client: TestClient, source: Path) -> None:
    dataset_id = import_path(client, source)
    for factor in (1.1, 1.2):
        write(source, orders(factor=factor))
        reimport(client, dataset_id)
    versions = client.get(f"{API}/{dataset_id}/versions").json()
    assert [(v["version"], v["kept"]) for v in versions] == [(3, True), (2, True), (1, False)]
    storage = client.get("/api/v1/storage").json()
    assert storage["versions_bytes"] > 0
    assert storage["keep_limit_bytes"] == 2048 * 1024 * 1024
    assert client.get(f"{API}/{dataset_id}/changes/relations").json()["available"]


def test_compare_two_datasets(client: TestClient, tmp_path: Path) -> None:
    october = import_path(client, write(tmp_path / "october.csv", orders()))
    november = import_path(
        client, write(tmp_path / "november.csv", orders(factor=1.3), drop="units")
    )
    changes = client.get(f"{API}/{november}/changes", params={"vs": october}).json()
    assert changes["headline"].startswith("Compared with october:")
    assert changes["distributions"][0]["column"] == "amount"
    assert [c["kind"] for c in changes["schema"]] == ["removed"]
    focused = client.get(
        f"{API}/{november}/changes", params={"vs": october, "focus": "region"}
    ).json()
    assert focused["distributions"][0]["column"] == "region"
    rows = client.get(f"{API}/{november}/changes/rows", params={"vs": october}).json()
    assert rows["available"] and rows["result"]["modified"] == 300


def test_a_type_correction_is_seen_by_the_comparison(client: TestClient, source: Path) -> None:
    dataset_id = import_path(client, source)
    write(source, orders())
    reimport(client, dataset_id)
    response = client.patch(
        f"{API}/{dataset_id}/columns",
        json={"changes": [{"name": "units", "type": "categorical_nominal"}]},
    )
    assert response.status_code == 200, response.text
    changes = client.get(f"{API}/{dataset_id}/changes").json()
    retyped = next(c for c in changes["schema"] if c["kind"] == "retyped")
    assert (retyped["column"], retyped["after"]) == ("units", "categorical_nominal")


def test_the_agent_tool_what_changed(client: TestClient, tmp_path: Path) -> None:
    from app.core.tools import ToolContext, ToolError

    october = import_path(client, write(tmp_path / "october.csv", orders()))
    november = import_path(client, write(tmp_path / "november.csv", orders(factor=1.3)))
    registry = client.app.state.agent.registry  # type: ignore[attr-defined]
    tool = registry.get("what_changed")
    assert tool is not None and tool.kind == "read"
    output = tool.run(ToolContext(november), tool.parse({"versus": "october"}))
    assert output.summary.startswith("Check: Compared with october:")
    assert output.content["shifts"][0]["column"] == "amount"
    assert output.link.page == "changes" and output.link.params == {"vs": october}
    with pytest.raises(ToolError, match="no previous version"):
        tool.run(ToolContext(november), tool.parse({}))
    with pytest.raises(ToolError, match="No dataset called"):
        tool.run(ToolContext(november), tool.parse({"versus": "nope"}))
