"""b2: the agent's `describe_layout` tool (080, J2 of the table-detection study) reads how the open
dataset was read from its file: a table of a source, a whole sheet, a plain CSV."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.modules.agent.engine.tools import ToolContext
from tests.api.test_profiling import import_path
from tests.api.test_sources import side_by_side
from tests.factories import write_csv


def describe(client: TestClient, dataset_id: str):
    tool = client.app.state.agent.registry.get("describe_layout")  # type: ignore[attr-defined]
    return tool.run(ToolContext(dataset_id), tool.parse({}))


def test_a_table_of_a_workbook_says_its_place_rules_and_keys(
    client: TestClient, tmp_path: Path
) -> None:
    path = side_by_side(tmp_path / "stock.xlsx")
    response = client.post(
        "/api/v1/sources/import",
        json={
            "path": str(path),
            "tables": [{"id": "Two!A1:C9"}, {"id": "Two!E1:F4"}],
            "options": {"null_tokens": ["nil"]},
        },
    )
    assert response.status_code == 201, response.text
    dataset_id = response.json()["imports"][1]["dataset"]["id"]
    found = describe(client, dataset_id)
    table = found.content["table"]
    assert table["location"] == "Two!E1:F4" and table["level"] in ("high", "review", "low")
    assert table["header_rows"] == [1] and table["rules"]
    assert found.content["source"]["tables"] == 2 and found.content["file"] == "stock.xlsx"
    assert found.content["placeholders_chosen_by_the_user"] == {"null_tokens": ["nil"]}
    assert found.summary.startswith("Two!E1:F4:") and found.link is not None


def test_a_whole_sheet_lists_what_the_detection_sees(client: TestClient, tmp_path: Path) -> None:
    path = side_by_side(tmp_path / "stock.xlsx")
    dataset_id = import_path(client, path)
    found = describe(client, dataset_id)
    assert found.content["read"].startswith("whole sheet")
    seen = [t["location"] for t in found.content["tables_the_detection_sees"]]
    assert seen == ["Two!A1:C9", "Two!E1:F4"] and "Choose tables" in found.content["advice"]


def test_a_csv_has_no_layout(client: TestClient, tmp_path: Path) -> None:
    dataset_id = import_path(client, write_csv(tmp_path / "plain.csv", "a,b\n1,2\n3,4\n"))
    found = describe(client, dataset_id)
    assert "no layout detection" in found.content["read"] and found.content["format"] == "csv"
