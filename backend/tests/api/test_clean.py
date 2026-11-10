"""Clean module (065, M2): suggestions, draft, preview, cleaned versions as datasets, lineage,
settings, automatic mode, the flag, and the agent's tools."""

import hashlib
import time
from pathlib import Path
from typing import Any

import duckdb
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, default_samples_dir
from app.core.jobs import InlineJobRunner
from app.core.tools import ToolContext
from app.main import create_app
from tests.api.test_profiling import import_path

BASE = "/api/v1/modules/clean"


@pytest.fixture
def traps(client: TestClient) -> str:
    return import_path(client, default_samples_dir() / "dirty_traps.csv")


def wait(client: TestClient, job: dict[str, Any], seconds: float = 120) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    while job["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
        job = client.get(f"{BASE}/jobs/{job['id']}").json()
    return job


def save(client: TestClient, dataset: str, steps: list[dict[str, Any]]) -> str:
    response = client.post(f"{BASE}/datasets/{dataset}/versions", json={"steps": steps})
    assert response.status_code == 202, response.text
    job = wait(client, response.json())
    assert job["status"] == "done", job
    return job["created_dataset_id"]


def test_suggestions_and_the_automatic_draft(client: TestClient, traps: str) -> None:
    found = client.get(f"{BASE}/datasets/{traps}/suggestions").json()
    ids = {s["id"]: s for s in found["suggestions"]}
    assert ids["B-03:city"]["mode"] == "A" and ids["B-14:rows"]["rows"] == 6
    assert ids["B-03:city"]["reason"] and ids["B-03:city"]["step"]["op"] == "map_values"
    draft = client.get(f"{BASE}/datasets/{traps}/draft").json()
    assert draft["automatic"] is True
    automatic = {s["id"] for s in found["suggestions"] if s["mode"] == "A"}
    assert {s["suggestion"] for s in draft["steps"]} == automatic
    again = client.get(f"{BASE}/datasets/{traps}/suggestions").json()
    assert again == found  # cached until the next import or retyping


def test_draft_is_saved_and_reset(client: TestClient, traps: str) -> None:
    steps = [{"id": "s1", "op": "rename_column", "column": "city", "to": "town"}]
    saved = client.put(f"{BASE}/datasets/{traps}/draft", json={"steps": steps}).json()
    assert saved["automatic"] is False
    assert client.get(f"{BASE}/datasets/{traps}/draft").json()["steps"][0]["to"] == "town"
    bad = client.put(f"{BASE}/datasets/{traps}/draft", json={"steps": [{"id": "s1", "op": "eval"}]})
    assert bad.status_code == 422
    reset = client.delete(f"{BASE}/datasets/{traps}/draft").json()
    assert reset["automatic"] is True


def test_preview_of_the_draft_and_of_one_step(client: TestClient, traps: str) -> None:
    steps = client.get(f"{BASE}/datasets/{traps}/draft").json()["steps"]
    whole = client.post(f"{BASE}/datasets/{traps}/preview", json={"steps": steps}).json()
    assert whole["rows_before"] == 407 and whole["rows_removed"] == 7
    assert whole["cells_changed"] > 1000 and whole["sample"]
    city = next(s for s in steps if s["suggestion"] == "B-03:city")
    one = client.post(
        f"{BASE}/datasets/{traps}/preview", json={"steps": steps, "step_id": city["id"]}
    ).json()
    changed = {c["column"]: c["cells"] for c in one["columns"] if c["cells"]}
    assert list(changed) == ["city"]
    broken = [*steps, {"id": "s99", "op": "normalize_spaces", "columns": ["nope"]}]
    reported = client.post(f"{BASE}/datasets/{traps}/preview", json={"steps": broken}).json()
    assert "does not exist" in reported["problems"]["s99"]


def test_save_a_cleaned_version(client: TestClient, traps: str, settings: Settings) -> None:
    parent = client.get(f"/api/v1/datasets/{traps}").json()
    storage = Path(settings.home) / "datasets" / traps / "data.parquet"
    before = hashlib.sha256(storage.read_bytes()).hexdigest()
    changes = {"changes": [{"name": "region", "role": "city"}]}
    assert client.patch(f"/api/v1/datasets/{traps}/columns", json=changes).status_code == 200
    steps = client.get(f"{BASE}/datasets/{traps}/draft").json()["steps"]
    child = save(client, traps, steps)

    dataset = client.get(f"/api/v1/datasets/{child}").json()
    assert dataset["status"] == "ready" and dataset["name"] == f"{parent['name']} · Cleaned v1"
    assert dataset["row_count"] == 400 and dataset["source_state"] == "ok"
    assert set(dataset) == set(parent)  # DatasetOut unchanged (C-13)
    assert hashlib.sha256(storage.read_bytes()).hexdigest() == before  # C-01
    columns = {c["name"]: c for c in client.get(f"/api/v1/datasets/{child}/columns").json()}
    assert columns["signup_date"]["type"] == "datetime"  # detected again on the child
    assert columns["region"]["role"] == "city"  # the user's correction, copied (C-09)
    rowmap = Path(settings.home) / "datasets" / child / "clean" / "rowmap.parquet"
    assert duckdb.execute(f"SELECT count(*) FROM '{rowmap.as_posix()}'").fetchone() == (407,)

    family = client.get(f"{BASE}/datasets/{child}/versions").json()
    assert family["original_id"] == traps
    [version] = family["versions"]
    assert (version["seq"], version["origin"], version["out_of_date"]) == (1, "user", False)
    assert version["report"]["rows_removed"] == 7
    assert {c["column"] for c in version["report"]["type_changes"]} >= {"signup_date"}
    assert client.get(f"{BASE}/lineage").json() == [
        {
            "dataset_id": child,
            "parent_id": traps,
            "seq": 1,
            "origin": "user",
            "steps": len(steps),
            "created_at": version["created_at"],
            "out_of_date": False,
        }
    ]
    second = save(client, traps, steps[:2])
    assert client.get(f"/api/v1/datasets/{second}").json()["name"].endswith("Cleaned v2")
    refused = client.post(f"{BASE}/datasets/{child}/versions", json={"steps": steps[:1]})
    assert refused.status_code == 409  # Clean starts from the original

    retype = {"changes": [{"name": "age", "type": "categorical_nominal"}]}
    assert client.patch(f"/api/v1/datasets/{traps}/columns", json=retype).status_code == 200
    assert all(v["out_of_date"] for v in client.get(f"{BASE}/lineage").json())  # S7
    client.delete(f"/api/v1/datasets/{child}")
    assert [v["seq"] for v in client.get(f"{BASE}/lineage").json()] == [2]
    client.delete(f"/api/v1/datasets/{traps}")
    assert client.get(f"{BASE}/lineage").json() == []
    assert client.get(f"/api/v1/datasets/{second}").json()["status"] == "ready"  # still there


def test_a_broken_step_is_refused_at_save(client: TestClient, traps: str) -> None:
    steps = [{"id": "s1", "op": "normalize_spaces", "columns": ["nope"]}]
    job = wait(
        client, client.post(f"{BASE}/datasets/{traps}/versions", json={"steps": steps}).json()
    )
    assert job["status"] == "failed" and "s1" in job["error"]
    assert client.get(f"{BASE}/lineage").json() == []


def test_settings(client: TestClient) -> None:
    assert client.get(f"{BASE}/settings").json() == {"mode": "suggest"}  # decision 10
    assert client.put(f"{BASE}/settings", json={"mode": "off"}).json() == {"mode": "off"}
    assert client.get(f"{BASE}/settings").json() == {"mode": "off"}
    assert client.put(f"{BASE}/settings", json={"mode": "always"}).status_code == 422


def test_automatic_mode_makes_cleaned_v1_after_an_import(client: TestClient) -> None:
    client.put(f"{BASE}/settings", json={"mode": "auto"})
    parent = import_path(client, default_samples_dir() / "dirty_traps.csv")
    deadline = time.monotonic() + 60
    while not client.get(f"{BASE}/lineage").json() and time.monotonic() < deadline:
        time.sleep(0.1)
    [version] = client.get(f"{BASE}/lineage").json()
    assert (version["parent_id"], version["origin"], version["seq"]) == (parent, "auto", 1)
    # the cleaned version itself is not cleaned again, and its import made no other version
    time.sleep(0.5)
    assert len(client.get(f"{BASE}/lineage").json()) == 1


def test_turned_off(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTILO_CLEAN_ENABLED", "false")
    off = Settings(home=settings.home)
    with TestClient(create_app(off, job_runner=InlineJobRunner())) as client:
        assert client.get(f"{BASE}/lineage").status_code == 404
        names = [tool.name for tool in client.app.state.agent_tools]  # type: ignore[attr-defined]
        assert "cleaning_suggestions" not in names


def test_the_agent_tools(client: TestClient, traps: str) -> None:
    tools = {t.name: t for t in client.app.state.agent_tools}  # type: ignore[attr-defined]
    listed = tools["cleaning_suggestions"].run(
        ToolContext(traps), tools["cleaning_suggestions"].parse({})
    )
    assert "automatic" in listed.summary
    assert any(s["id"] == "B-14:rows" for s in listed.content["suggestions"])
    saver = tools["save_cleaned_version"]
    assert saver.kind == "write"
    saved = saver.run(ToolContext(traps), saver.parse({"add": ["B-19:country"]}))
    child = saved.content["dataset_id"]
    columns = [c["name"] for c in client.get(f"/api/v1/datasets/{child}/columns").json()]
    assert "country" not in columns
