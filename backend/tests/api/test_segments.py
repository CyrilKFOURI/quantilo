"""032 / 033 — the Segments module over HTTP: plan, jobs, runs, names, t-SNE, profiles, new
datasets, unusual rows, marks, the agent's tools, and the same results as the library."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import quantilo as dx
from fastapi.testclient import TestClient

from app.config import default_samples_dir
from app.core.tools import ToolContext
from tests.api.test_library_parity import same
from tests.api.test_profiling import import_path

BASE = "/api/v1/modules/segments"


def blobs_csv(path: Path, n: int = 300) -> Path:
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 3, n)
    centres = np.array([[0, 0], [8, 8], [0, 8]], dtype=float)
    points = centres[truth] + rng.normal(0, 1, (n, 2))
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "income": np.round(points[:, 0] * 1_000 + 20_000, 2),
            "age": np.round(points[:, 1] * 2 + 30, 1),
            "country": np.where(np.arange(n) % 2 == 0, "FR", "BE"),
        }
    )
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def blobs(client: TestClient, tmp_path: Path) -> str:
    return import_path(client, blobs_csv(tmp_path / "blobs.csv"))


@pytest.fixture
def traps(client: TestClient) -> str:
    return import_path(client, default_samples_dir() / "typing_traps.csv")


def wait(client: TestClient, job: dict[str, Any], seconds: float = 120) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    while job["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.1)
        job = client.get(f"{BASE}/jobs/{job['id']}").json()
    return job


NUMBERS = ["income", "age"]  # the three blobs; "country" would split each one in two


def segment(client: TestClient, dataset: str, **body: Any) -> dict[str, Any]:
    params = body.pop("params", {})
    body.setdefault("columns", NUMBERS)
    started = client.post(f"{BASE}/datasets/{dataset}/runs", json=body, params=params)
    assert started.status_code == 202, started.text
    job = wait(client, started.json())
    assert job["status"] == "done", job
    return client.get(f"{BASE}/datasets/{dataset}/runs/{job['run_id']}").json()


def test_plan_leaves_out_identifiers_and_dates(client: TestClient, traps: str) -> None:
    plan = client.post(f"{BASE}/datasets/{traps}/plan").json()
    by_name = {d["column"]: d for d in plan["preparation"]["decisions"]}

    assert not by_name["customer_id"]["use"]
    assert by_name["customer_id"]["why"][0].startswith("Identifier")
    assert by_name["signup_date"]["rules"] == ["R-DIST-03"]
    assert plan["rows"] == 600 and plan["method"] in ("kmeans", "kprototypes")
    assert {m["key"] for m in plan["methods"]} == {
        "kmeans",
        "kprototypes",
        "gmm",
        "ward",
        "hdbscan",
    }


def test_a_segmentation_is_saved_renamed_and_compared(client: TestClient, blobs: str) -> None:
    run = segment(client, blobs)
    result = run["result"]
    assert result["k"] == 3 and result["stability"]["stable"]
    assert sum(s["size"] for s in result["segments"]) == 300
    assert [d["column"] for d in result["preparation"]["decisions"] if d["use"]] == NUMBERS

    renamed = client.patch(
        f"{BASE}/datasets/{blobs}/runs/{run['id']}/names", json={"names": {"0": "Big ones"}}
    ).json()
    assert renamed["result"]["segments"][0]["name"] == "Big ones"
    bad = client.patch(
        f"{BASE}/datasets/{blobs}/runs/{run['id']}/names", json={"names": {"9": "x"}}
    )
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "unknown_segment"

    forced = segment(client, blobs, k=4, method="kmeans")
    assert forced["result"]["k"] == 4 and forced["result"]["choice"]["forced"]
    listed = client.get(f"{BASE}/datasets/{blobs}/runs").json()
    assert [r["id"] for r in listed] == [forced["id"], run["id"]]
    assert listed[1]["names"][0] == "Big ones"
    compared = client.get(
        f"{BASE}/datasets/{blobs}/runs/compare", params={"runs": f"{run['id']},{forced['id']}"}
    ).json()
    assert compared["agreement"][0][0] == pytest.approx(1.0)
    assert 0.5 < compared["agreement"][0][1] < 1

    csv = client.get(f"{BASE}/datasets/{blobs}/runs/{run['id']}/profiles.csv")
    assert csv.status_code == 200 and csv.text.startswith("column,overall,Big ones")
    assert client.delete(f"{BASE}/datasets/{blobs}/runs/{forced['id']}").status_code == 204
    assert len(client.get(f"{BASE}/datasets/{blobs}/runs").json()) == 1


def test_filters_apply_and_save_as_dataset(client: TestClient, blobs: str) -> None:
    condition = dx.Condition(column="country", op="in", values=["FR"])
    run = segment(client, blobs, params={"filters": f"[{condition.model_dump_json()}]"})
    assert run["result"]["rows"] == 150 and run["conditions"][0]["column"] == "country"

    made = client.post(f"{BASE}/datasets/{blobs}/runs/{run['id']}/dataset").json()["dataset_id"]
    dataset = client.get(f"/api/v1/datasets/{made}").json()
    assert dataset["status"] == "ready" and dataset["row_count"] == 150
    columns = {c["name"]: c["type"] for c in client.get(f"/api/v1/datasets/{made}/columns").json()}
    assert columns["segment"] == "categorical_nominal" and columns["customer_id"] == "identifier"
    compare = client.get(f"/api/v1/datasets/{made}/compare", params={"x": "income", "y": "segment"})
    assert compare.status_code == 200 and compare.json()["tests"]
    again = client.post(f"{BASE}/datasets/{blobs}/runs/{run['id']}/dataset").json()["dataset_id"]
    assert again == made
    assert client.get(f"{BASE}/datasets/{blobs}/runs").json()[0]["created_dataset_id"] == made


def test_tsne_job_and_errors(client: TestClient, blobs: str) -> None:
    run = segment(client, blobs)
    job = wait(client, client.post(f"{BASE}/datasets/{blobs}/runs/{run['id']}/tsne").json())
    assert job["status"] == "done" and job["kind"] == "tsne"
    shown = client.get(f"{BASE}/datasets/{blobs}/runs/{run['id']}").json()["result"]["tsne"]
    assert shown["projection"] == "tsne" and len(shown["x"]) == 300

    assert client.get(f"{BASE}/datasets/{blobs}/runs/0123456789ab").status_code == 404
    assert client.get(f"{BASE}/jobs/nope").status_code == 404
    bad = client.post(f"{BASE}/datasets/{blobs}/runs", json={"columns": ["nope"]})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "unknown_column"


def test_unusual_rows_marks_and_exports(client: TestClient, tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 800
    orders = rng.integers(1, 30, n).astype(float)
    frame = pd.DataFrame({"orders": orders, "revenue": orders * 100 + rng.normal(0, 40, n)})
    frame.loc[0, ["orders", "revenue"]] = [0, 3_012]
    frame.to_csv(tmp_path / "shop.csv", index=False)
    dataset = import_path(client, tmp_path / "shop.csv")

    job = wait(client, client.post(f"{BASE}/datasets/{dataset}/unusual", json={}).json())
    assert job["status"] == "done", job
    run = client.get(f"{BASE}/datasets/{dataset}/unusual/{job['run_id']}").json()
    result = run["result"]
    first = result["listed"][0]
    assert first["row"] == 0 and {r["column"] for r in first["reasons"][:2]} == {
        "orders",
        "revenue",
    }
    assert result["threshold"]["rows"] >= 1 and result["curve"]["ranks"][0] == 1

    card = client.get(f"{BASE}/datasets/{dataset}/unusual/{run['id']}/rows/0").json()
    assert {v["column"] for v in card} == {"orders", "revenue"}
    assert next(v for v in card if v["column"] == "orders")["percentile"] < 0.01

    marked = client.put(
        f"{BASE}/datasets/{dataset}/marks", json={"key": first["key"], "mark": "expected"}
    ).json()
    assert marked[0]["key"] == first["key"] and marked[0]["mark"] == "expected"
    shown = client.get(f"{BASE}/datasets/{dataset}/unusual/{run['id']}").json()
    assert shown["marks"] == {first["key"]: "expected"}
    csv = client.get(f"{BASE}/datasets/{dataset}/unusual/{run['id']}/export.csv").text
    assert csv.splitlines()[0].startswith("rank,row,score,mark,reasons,orders,revenue")
    assert ",expected," in csv.splitlines()[1]

    again = wait(client, client.post(f"{BASE}/datasets/{dataset}/unusual", json={}).json())
    hidden = client.get(f"{BASE}/datasets/{dataset}/unusual/{again['run_id']}").json()["result"]
    assert hidden["hidden_expected"] == 1 and 0 not in {r["row"] for r in hidden["listed"]}
    assert len(client.get(f"{BASE}/datasets/{dataset}/unusual").json()) == 2
    client.put(f"{BASE}/datasets/{dataset}/marks", json={"key": first["key"], "mark": None})
    assert client.get(f"{BASE}/datasets/{dataset}/marks").json() == []

    made = client.post(
        f"{BASE}/datasets/{dataset}/unusual/{run['id']}/dataset", json={"threshold": None}
    ).json()["dataset_id"]
    columns = {c["name"]: c["type"] for c in client.get(f"/api/v1/datasets/{made}/columns").json()}
    assert columns["unusual_score"] == "numeric_continuous" and columns["is_unusual"] == "boolean"


def test_distance_to_its_segment_needs_a_run(client: TestClient, blobs: str) -> None:
    bad = client.post(f"{BASE}/datasets/{blobs}/unusual", json={"method": "segment"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "no_segmentation"
    run = segment(client, blobs)
    job = wait(
        client,
        client.post(
            f"{BASE}/datasets/{blobs}/unusual",
            json={"method": "segment", "segments_run": run["id"]},
        ).json(),
    )
    found = client.get(f"{BASE}/datasets/{blobs}/unusual/{job['run_id']}").json()["result"]
    assert found["method"] == "segment" and found["listed"][0]["segment"]


def test_the_app_and_the_library_segment_alike(
    client: TestClient, blobs: str, tmp_path: Path
) -> None:
    run = segment(client, blobs)
    ds = dx.open(tmp_path / "blobs.csv", cache_dir=tmp_path / "cache")
    library = dx.to_dict(ds.segments(NUMBERS).fit())
    api = run["result"]
    for timed in (api, library):
        timed.pop("seconds")
    same(api, library, "segments")

    job = wait(client, client.post(f"{BASE}/datasets/{blobs}/unusual", json={}).json())
    found = client.get(f"{BASE}/datasets/{blobs}/unusual/{job['run_id']}").json()["result"]
    odd = dx.to_dict(ds.unusual_rows())
    for timed in (found, odd):
        timed.pop("seconds")
    same(found, odd, "unusual")


def agent(client: TestClient) -> Any:
    return client.app.state.agent  # type: ignore[attr-defined]


def test_the_agent_tools(client: TestClient, blobs: str) -> None:
    registry = agent(client).registry
    kinds = {t.name: t.kind for t in registry}
    assert kinds["find_segments"] == "compute" and kinds["save_segments"] == "write"
    assert kinds["find_unusual_rows"] == "compute"

    found = registry.get("find_segments")
    output = found.run(ToolContext(blobs), found.parse({"columns": NUMBERS}))
    assert output.content["segments_count"] == 3 and output.link.page == "segments"
    assert "customer_id" in output.content["columns_left_out"]
    saver = registry.get("save_segments")
    assert "new dataset" in saver.confirmation(saver.parse({"run_id": output.content["run_id"]}))
    saved = saver.run(ToolContext(blobs), saver.parse({"run_id": output.content["run_id"]}))
    assert saved.link.page == "dataset"

    unusual = registry.get("find_unusual_rows")
    odd = unusual.run(ToolContext(blobs), unusual.parse({}))
    assert odd.content["most_unusual"] and odd.link.params["section"] == "unusual"


def test_pacmap_and_trimap_are_refused_with_their_reason_without_the_extra(
    client: TestClient, blobs: str
) -> None:
    """122, R-XAI-10: the optional maps say why they cannot run here."""
    from quantilo.interpret import extras

    run = segment(client, blobs)
    for method in ("pacmap", "trimap"):
        if extras.check(method).available:
            continue
        refused = client.post(f"{BASE}/datasets/{blobs}/runs/{run['id']}/map/{method}")
        assert refused.status_code == 409 and "R-XAI-10" in refused.json()["error"]["message"]
    assert client.post(f"{BASE}/datasets/{blobs}/runs/{run['id']}/map/umap").status_code == 422


def test_describe_segments_with_a_tree(client: TestClient, blobs: str) -> None:
    """124: a sparse tree predicts each row's segment; its fidelity and each segment's rules."""
    run = segment(client, blobs)
    url = f"{BASE}/datasets/{blobs}/runs/{run['id']}/tree"
    found = client.post(url, json={})
    assert found.status_code == 200, found.text
    tree = found.json()
    assert tree["fidelity"] > 0.9 and tree["depth"] == 3 and tree["max_leaves"] == 8
    names = {s["name"] for s in run["result"]["segments"]}
    assert {s["name"] for s in tree["segments"]} == names
    assert all(s["rules"] for s in tree["segments"])
    assert tree["nodes"][0]["left_rule"].split()[0] in NUMBERS
    small = client.post(url, json={"depth": 1, "leaves": 2}).json()
    assert small["found_depth"] <= 1 and small["leaves"] <= 2 and small["warning"]
    assert client.post(url, json={"depth": 9}).status_code == 422
    # 124 bis: each rule as filter conditions (R-SGT-05); the optimal tree needs the extra
    first = tree["segments"][0]["filters"][0]
    assert first["rule"] and first["conditions"][0]["op"] in ("lte", "gt", "in")
    from quantilo.interpret import extras

    optimal = client.post(f"{url}/optimal", json={"segment": tree["segments"][0]["segment"]})
    if not extras.check("gosdt").available:
        assert optimal.status_code == 409 and "R-XAI-10" in optimal.json()["error"]["message"]
