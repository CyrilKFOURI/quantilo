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
    # N03: by default only the reliable fixes are applied, each step marked automatic
    automatic = {s["id"] for s in found["suggestions"] if s["auto"]}
    assert automatic and automatic < {s["id"] for s in found["suggestions"] if s["mode"] == "A"}
    assert {s["suggestion"] for s in draft["steps"]} == automatic
    assert {s["origin"] for s in draft["steps"]} == {"auto"}
    assert all(ids[i].get("measured") is not None for i in automatic)
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
    assert whole["cells_changed"] > 500 and whole["sample"]  # the reliable fixes only (N03)
    city = next(s for s in steps if s["suggestion"] == "B-03:city")
    one = client.post(
        f"{BASE}/datasets/{traps}/preview", json={"steps": steps, "step_id": city["id"]}
    ).json()
    changed = {c["column"]: c["cells"] for c in one["columns"] if c["cells"]}
    assert list(changed) == ["city"]
    broken = [*steps, {"id": "s99", "op": "normalize_spaces", "columns": ["nope"]}]
    reported = client.post(f"{BASE}/datasets/{traps}/preview", json={"steps": broken}).json()
    assert "does not exist" in reported["problems"]["s99"]


def test_duplicates_side_by_side_and_a_group_kept(client: TestClient, traps: str) -> None:
    """N35: the groups of B-14, and the step without one group removes fewer rows."""
    found = client.get(f"{BASE}/datasets/{traps}/suggestions").json()["suggestions"]
    dedupe = {**next(s for s in found if s["id"] == "B-14:rows")["step"], "id": "s1"}
    body = {"steps": [dedupe], "step_id": "s1", "limit": 50}
    groups = client.post(f"{BASE}/datasets/{traps}/duplicates", json=body).json()
    assert groups["removed"] == 6 and groups["keep"] == "first"
    first = groups["groups"][0]
    assert len(first["rows"]) >= 2 and first["order"][0] == first["first"]
    assert first["differing"] == []  # exact copies: nothing differs
    apart = {**dedupe, "not_duplicates": [[row["row"] for row in first["rows"]]]}
    preview = client.post(f"{BASE}/datasets/{traps}/preview", json={"steps": [apart]}).json()
    assert preview["rows_removed"] == 6 - (len(first["rows"]) - 1)
    other = {"steps": [{"id": "s1", "op": "rename_column", "column": "city", "to": "town"}]}
    refused = client.post(f"{BASE}/datasets/{traps}/duplicates", json={**other, "step_id": "s1"})
    assert refused.status_code == 422 and "does not remove duplicates" in refused.text


def test_entity_clusters_side_by_side(client: TestClient, traps: str) -> None:
    """N36: the clusters of an entity number (A-20's step), the rule's whatever the review says;
    another step refused."""
    cluster = [{"anchor": "FR", "name": "paris"}, {"anchor": "FR", "name": "madrid"}]
    step = {
        "id": "s1",
        "op": "entity_column",
        "column": "city",
        "anchor": "country",
        "to": "city_entity",
        "groups": [cluster],
        "not_entities": [0],
    }
    body = {"steps": [step], "step_id": "s1"}
    found = client.post(f"{BASE}/datasets/{traps}/entities", json=body).json()
    assert (found["total_clusters"], found["total_records"], found["to"]) == (1, 2, "city_entity")
    (shown,) = found["clusters"]
    assert [r["name"] for r in shown["records"]] == ["paris", "madrid"]
    assert all(r["rows"] > 0 and r["cells"] for r in shown["records"])
    preview = client.post(f"{BASE}/datasets/{traps}/preview", json={"steps": [step]}).json()
    assert preview["added"] == ["city_entity"] and preview["rows_removed"] == 0
    other = {"steps": [{"id": "s1", "op": "rename_column", "column": "city", "to": "town"}]}
    refused = client.post(f"{BASE}/datasets/{traps}/entities", json={**other, "step_id": "s1"})
    assert refused.status_code == 422 and "does not number entities" in refused.text


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
            "source_id": None,
            "source_seq": None,
        }
    ]
    second = save(client, traps, steps[:2])
    assert client.get(f"/api/v1/datasets/{second}").json()["name"].endswith("Cleaned v2")

    retype = {"changes": [{"name": "age", "type": "categorical_nominal"}]}
    assert client.patch(f"/api/v1/datasets/{traps}/columns", json=retype).status_code == 200
    assert all(v["out_of_date"] for v in client.get(f"{BASE}/lineage").json())  # S7
    client.delete(f"/api/v1/datasets/{child}")
    assert [v["seq"] for v in client.get(f"{BASE}/lineage").json()] == [2]
    client.delete(f"/api/v1/datasets/{traps}")
    assert client.get(f"{BASE}/lineage").json() == []
    assert client.get(f"/api/v1/datasets/{second}").json()["status"] == "ready"  # still there


def test_clean_a_cleaned_version(client: TestClient, traps: str) -> None:
    """096: Clean works on the selected version; v1 cleaned again gives v2, made from v1, in the
    same family under the original."""
    original = client.get(f"/api/v1/datasets/{traps}").json()
    first = save(client, traps, client.get(f"{BASE}/datasets/{traps}/draft").json()["steps"])
    found = client.get(f"{BASE}/datasets/{first}/suggestions").json()
    assert found["dataset_id"] == first  # computed on v1, not on the original
    steps = [{"id": "s1", "op": "rename_column", "column": "city", "to": "town"}]
    second = save(client, first, steps)
    child = client.get(f"/api/v1/datasets/{second}").json()
    assert child["name"] == f"{original['name']} · Cleaned v2"
    columns = {c["name"] for c in client.get(f"/api/v1/datasets/{second}/columns").json()}
    assert "town" in columns and "city" not in columns
    assert child["row_count"] == client.get(f"/api/v1/datasets/{first}").json()["row_count"]
    family = client.get(f"{BASE}/datasets/{second}/versions").json()
    assert family["original_id"] == traps
    assert [(v["seq"], v["source_seq"]) for v in family["versions"]] == [(1, None), (2, 1)]
    assert family["versions"][1]["source_id"] == first
    lineage = {item["dataset_id"]: item for item in client.get(f"{BASE}/lineage").json()}
    assert lineage[second]["parent_id"] == traps  # grouped under the original
    assert lineage[second]["source_id"] == first
    client.delete(f"/api/v1/datasets/{first}")  # the version it was made from
    [left] = client.get(f"{BASE}/datasets/{second}/versions").json()["versions"]
    assert (left["seq"], left["source_id"], left["out_of_date"]) == (2, None, False)


def test_a_broken_step_is_refused_at_save(client: TestClient, traps: str) -> None:
    steps = [{"id": "s1", "op": "normalize_spaces", "columns": ["nope"]}]
    job = wait(
        client, client.post(f"{BASE}/datasets/{traps}/versions", json={"steps": steps}).json()
    )
    assert job["status"] == "failed" and "s1" in job["error"]
    assert client.get(f"{BASE}/lineage").json() == []


def test_settings(client: TestClient) -> None:
    assert client.get(f"{BASE}/settings").json() == {"mode": "apply"}  # N03
    assert client.put(f"{BASE}/settings", json={"mode": "off"}).json() == {"mode": "off"}
    assert client.get(f"{BASE}/settings").json() == {"mode": "off"}
    assert client.put(f"{BASE}/settings", json={"mode": "always"}).status_code == 422


def test_suggest_only_ticks_nothing_and_nothing_is_saved_by_itself(
    client: TestClient, traps: str
) -> None:
    """N03: Suggest only (the automatic mode turned off) applies nothing; in the automatic mode
    opening Clean saves neither a draft nor a version: only the user's Save makes one."""
    client.get(f"{BASE}/datasets/{traps}/draft")
    assert client.get(f"{BASE}/lineage").json() == []
    client.put(f"{BASE}/settings", json={"mode": "suggest"})
    draft = client.get(f"{BASE}/datasets/{traps}/draft").json()
    assert (draft["steps"], draft["automatic"]) == ([], True)
    assert client.get(f"{BASE}/datasets/{traps}/suggestions").json()["suggestions"]
    assert client.delete(f"{BASE}/datasets/{traps}/draft").json()["steps"] == []
    client.put(f"{BASE}/settings", json={"mode": "apply"})
    assert client.get(f"{BASE}/datasets/{traps}/draft").json()["steps"]  # the default again


def test_automatic_mode_makes_cleaned_v1_after_an_import(client: TestClient) -> None:
    client.put(f"{BASE}/settings", json={"mode": "auto"})
    parent = import_path(client, default_samples_dir() / "dirty_traps.csv")
    deadline = time.monotonic() + 60
    while not client.get(f"{BASE}/lineage").json() and time.monotonic() < deadline:
        time.sleep(0.1)
    [version] = client.get(f"{BASE}/lineage").json()
    assert (version["parent_id"], version["origin"], version["seq"]) == (parent, "auto", 1)
    # N03: only the reliable fixes, each step said automatic (Versions, Changes)
    [saved] = client.get(f"{BASE}/datasets/{parent}/versions").json()["versions"]
    assert saved["steps"] and {s["origin"] for s in saved["steps"]} == {"auto"}
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


# --- 101: Advanced on demand, the Expert checks, replay ------------------------------------------


def wait_expert(client: TestClient, job: dict[str, Any], seconds: float = 120) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    while job["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
        job = client.get(f"{BASE}/expert-jobs/{job['id']}").json()
    return job


def test_advanced_layer_on_demand(client: TestClient, traps: str) -> None:
    found = client.get(f"{BASE}/datasets/{traps}/suggestions/advanced").json()
    assert found["dataset_id"] == traps
    assert {s["layer"] for s in found["suggestions"]} <= {"advanced"}
    assert {s["risk"] for s in found["suggestions"]} <= {"review"}
    again = client.get(f"{BASE}/datasets/{traps}/suggestions/advanced").json()
    assert again == found  # cached until the next import or retyping


def test_expert_checks_run_as_jobs_and_suggest_only(client: TestClient, traps: str) -> None:
    methods = {m["id"]: m for m in client.get(f"{BASE}/datasets/{traps}/expert").json()}
    assert list(methods) == ["unusual_rows", "suspicious_labels", "fill_empty", "cell_errors"]
    assert methods["unusual_rows"]["available"] is True
    started = client.post(f"{BASE}/datasets/{traps}/expert/unusual_rows", json={})
    assert started.status_code == 202, started.text
    job = wait_expert(client, started.json())
    assert job["status"] == "done", job
    result = job["result"]
    assert result["method"] == "unusual_rows" and result["rule"] == "E-06"
    for suggestion in result["suggestions"]:
        assert suggestion["layer"] == "expert"
        assert (suggestion["mode"], suggestion["risk"]) == ("S", "review")
    draft = client.get(f"{BASE}/datasets/{traps}/draft").json()
    found = {s["id"] for s in result["suggestions"]}
    assert not {s["suggestion"] for s in draft["steps"]} & found  # never ticked (C-30)
    bad = client.post(f"{BASE}/datasets/{traps}/expert/eval", json={})
    assert bad.status_code == 422
    refused = client.post(f"{BASE}/datasets/{traps}/expert/unusual_rows", json={"nope": 1})
    assert refused.status_code == 422


def test_an_expert_check_is_cancelled(
    client: TestClient, traps: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quantilo.cleaning import CleanCancelled

    from app.modules.clean import service as module

    def slow(*args: Any, **_kwargs: Any) -> Any:
        stopped = args[5]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if stopped():
                raise CleanCancelled
            time.sleep(0.02)
        raise AssertionError("not cancelled")

    monkeypatch.setattr(module.expert, "run", slow)
    job = client.post(f"{BASE}/datasets/{traps}/expert/unusual_rows", json={}).json()
    assert client.delete(f"{BASE}/expert-jobs/{job['id']}").status_code == 200
    job = wait_expert(client, job)
    assert job["status"] == "cancelled" and job["result"] is None
    assert client.get(f"{BASE}/expert-jobs/unknown").status_code == 404


def test_replay_a_version_and_its_recipe(client: TestClient, traps: str) -> None:
    steps = client.get(f"{BASE}/datasets/{traps}/draft").json()["steps"]
    steps.append({"id": "s900", "op": "drop_rows", "rows": [0]})
    child = save(client, traps, steps)
    recipe = client.get(f"{BASE}/versions/{child}/recipe").json()
    assert recipe["format"] == "quantilo/clean-recipe" and recipe["schema"] == 1
    assert len(recipe["steps"]) == len(steps) and recipe["columns"]
    report = client.post(f"{BASE}/datasets/{traps}/replay", json={"version_id": child}).json()
    by_id = {s["id"]: s for s in report["steps"]}
    assert by_id["s900"]["status"] == "skipped"  # row numbers of another file (C-41)
    assert report["applies"] >= len(steps) - 1
    assert next(s for s in report["adapted"] if s["id"] == "s900")["enabled"] is False
    from_file = client.post(f"{BASE}/datasets/{traps}/replay", json={"recipe": recipe}).json()
    assert from_file["steps"] == report["steps"]
    assert client.post(f"{BASE}/datasets/{traps}/replay", json={}).status_code == 400
    assert client.get(f"{BASE}/versions/{traps}/recipe").status_code == 404
    broken = {**recipe, "extra": 1}
    refused = client.post(f"{BASE}/datasets/{traps}/replay", json={"recipe": broken})
    assert refused.status_code == 422


def test_rebuild_from_original_saves_the_next_version(client: TestClient, traps: str) -> None:
    """B1 (C-47): the version's steps run again on the dataset it was made from, as the replay
    adapts them: its step by row number is kept, turned off (C-41)."""
    steps = client.get(f"{BASE}/datasets/{traps}/draft").json()["steps"]
    steps.append({"id": "s900", "op": "drop_rows", "rows": [0]})
    child = save(client, traps, steps)
    response = client.post(f"{BASE}/versions/{child}/rebuild")
    assert response.status_code == 202, response.text
    job = wait(client, response.json())
    assert job["status"] == "done", job
    family = client.get(f"{BASE}/datasets/{traps}/versions").json()
    rebuilt = next(v for v in family["versions"] if v["dataset_id"] == job["created_dataset_id"])
    assert rebuilt["seq"] == 2 and len(rebuilt["steps"]) == len(steps)
    assert [s["id"] for s in rebuilt["steps"] if not s["enabled"]] == ["s900"]
    assert client.post(f"{BASE}/versions/{traps}/rebuild").status_code == 404


def test_structure_steps_read_another_dataset_and_reshape(
    client: TestClient, traps: str, tmp_path: Path
) -> None:
    """C1 (C-57 to C-59): append another dataset by its id, group by; the saved version says it
    was reshaped and keeps no row map."""
    other = tmp_path / "more.csv"
    other.write_text("city,score\nOslo,1\nRome,2\n", encoding="utf-8")
    more = import_path(client, other)
    steps = [
        {"id": "s1", "op": "append_rows", "dataset": more, "label": "more"},
        {
            "id": "s2",
            "op": "group_by",
            "keys": ["city"],
            "aggregations": [{"fn": "count", "to": "n"}],
        },
    ]
    shown = client.post(f"{BASE}/datasets/{traps}/preview", json={"steps": steps}).json()
    assert shown["reshaped"] == "s1" and not shown["problems"], shown
    assert shown["rows_after"] < shown["rows_before"]
    gone = [{"id": "s1", "op": "append_rows", "dataset": "nope", "label": "gone"}]
    missing = client.post(f"{BASE}/datasets/{traps}/preview", json={"steps": gone}).json()
    assert "not available" in missing["problems"]["s1"]
    child = save(client, traps, steps)
    family = client.get(f"{BASE}/datasets/{child}/versions").json()
    version = next(v for v in family["versions"] if v["dataset_id"] == child)
    assert version["report"]["reshaped_by"] == "s1"


def test_column_quality_comes_with_the_basic_suggestions(client: TestClient, traps: str) -> None:
    """N37: each column's quality from the Basic profile, in the file's order (Basic only)."""
    found = client.get(f"{BASE}/datasets/{traps}/suggestions").json()
    quality = found["quality"]
    columns = [c["name"] for c in client.get(f"/api/v1/datasets/{traps}/columns").json()]
    assert [q["column"] for q in quality] == columns
    assert all(q["rows"] == found["rows"] and 0 <= q["missing"] <= q["rows"] for q in quality)
    assert any(q["missing"] > 0 for q in quality)
    assert client.get(f"{BASE}/datasets/{traps}/suggestions/advanced").json()["quality"] is None


def test_facet_of_a_column_of_the_draft(client: TestClient, traps: str) -> None:
    """N40: values and rows of a column after the draft's steps, and the rows a selection picks."""
    url = f"{BASE}/datasets/{traps}/facet"
    found = client.post(url, json={"column": "city"}).json()
    assert found["kind"] == "text" and found["values"] and found["matched"] is None
    top = found["values"][0]
    picked = client.post(
        url, json={"column": "city", "selection": {"values": [top["value"]]}}
    ).json()
    assert picked["matched"] == top["rows"] and picked["shown"][0] == "city"
    assert len(picked["sample"]) == min(50, top["rows"])
    merged = client.post(
        url,
        json={
            "column": "city",
            "steps": [
                {"id": "s1", "op": "map_values", "column": "city", "map": {top["value"]: "Zz"}}
            ],
        },
    ).json()
    assert ("Zz", top["rows"]) in [(v["value"], v["rows"]) for v in merged["values"]]
    missing = client.post(url, json={"column": "nope"})
    assert missing.status_code == 422
