"""037: a trained run used on new data — check, predictions, scores when the answer is there,
CSV export, Save as dataset, the model exported as a zip, and the agent's tools."""

import csv
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import quantilo as dx
from fastapi.testclient import TestClient

from app.core.tools import ToolContext, ToolError
from tests.api.test_profiling import import_path

BASE = "/api/v1/modules/ml"
REGIONS = ["North", "South", "East", "West"]


def month(seed: int, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "client_id": [f"C{seed}-{i:04d}" for i in range(n)],
            "region": rng.choice(REGIONS, n),
            "income": rng.normal(40_000, 8_000, n).round(2),
            "age": rng.integers(18, 70, n),
        }
    )
    lifts = {"North": 2.0, "South": -2.0, "East": 0.5, "West": -0.5}
    lift = np.array([lifts[r] for r in frame["region"]])
    score = lift + (frame["income"] - 40_000) / 4_000 + rng.normal(0, 0.5, n)
    frame["churn"] = np.where(score > 0, "leave", "stay")
    return frame


def write(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def wait(client: TestClient, url: str) -> dict[str, Any]:
    job: dict[str, Any] = {}
    for _ in range(600):
        job = client.get(url).json()
        if job["status"] != "running":
            break
        time.sleep(0.1)
    return job


@pytest.fixture
def trained(client: TestClient, tmp_path: Path) -> tuple[str, str, Path]:
    """(dataset id, run id, october.csv) of a churn model trained on October."""
    october = write(month(10), tmp_path / "october.csv")
    dataset = import_path(client, october)
    started = client.post(f"{BASE}/datasets/{dataset}/runs", params={"target": "churn"})
    job = wait(client, f"{BASE}/jobs/{started.json()['id']}")
    assert job["status"] == "done", job
    return dataset, job["run_id"], october


def predict(client: TestClient, dataset: str, run: str, source: str, keep: list[str]) -> dict:
    started = client.post(
        f"{BASE}/datasets/{dataset}/runs/{run}/predictions", json={"source": source, "keep": keep}
    )
    assert started.status_code == 202, started.text
    job = wait(client, f"{BASE}/predict-jobs/{started.json()['id']}")
    assert job["status"] == "done", job
    return client.get(f"{BASE}/datasets/{dataset}/predictions/{job['prediction_id']}").json()


def all_rows(client: TestClient, dataset: str, prediction: str) -> pd.DataFrame:
    page = client.get(
        f"{BASE}/datasets/{dataset}/predictions/{prediction}/rows", params={"limit": 500}
    ).json()
    rows = page["rows"]
    while len(rows) < page["total"]:
        page = client.get(
            f"{BASE}/datasets/{dataset}/predictions/{prediction}/rows",
            params={"limit": 500, "offset": len(rows)},
        ).json()
        rows += page["rows"]
    return pd.DataFrame(rows, columns=page["columns"])


def test_same_columns_are_ready_and_match_the_library(
    client: TestClient, trained: tuple[str, str, Path], tmp_path: Path
) -> None:
    dataset, run, october = trained
    november = write(month(11).drop(columns="churn"), tmp_path / "november.csv")
    source = import_path(client, november)

    checked = client.get(
        f"{BASE}/datasets/{dataset}/runs/{run}/check", params={"source": source}
    ).json()
    found = predict(client, dataset, run, source, ["client_id"])
    rows = all_rows(client, dataset, found["id"])

    assert checked["level"] == "ready", checked["message"]
    assert checked["rows"] == 600 and checked["target_present"] is False
    assert found["level"] == "ready" and found["rows"] == 600
    assert list(rows.columns[:3]) == ["row", "client_id", "predicted"]
    # the library, trained the same way on the same file, predicts the same
    model = dx.open(october, cache_dir=tmp_path / "lib").ml("churn").train()
    expected = model.predict(dx.open(november, cache_dir=tmp_path / "lib"))
    assert list(rows["predicted"]) == list(expected["predicted"])
    assert np.allclose(rows["p(leave)"].astype(float), expected["p(leave)"])
    listed = client.get(f"{BASE}/datasets/{dataset}/runs/{run}/predictions").json()
    assert listed[0]["id"] == found["id"] and listed[0]["source_name"] == "november"
    summary = found["summary"]
    assert summary["distribution"]["kind"] == "classes" and summary["scores"] is None


def test_a_missing_column_blocks(
    client: TestClient, trained: tuple[str, str, Path], tmp_path: Path
) -> None:
    dataset, run, _ = trained
    source = import_path(
        client, write(month(11).drop(columns=["churn", "income"]), tmp_path / "no_income.csv")
    )

    checked = client.get(
        f"{BASE}/datasets/{dataset}/runs/{run}/check", params={"source": source}
    ).json()
    refused = client.post(
        f"{BASE}/datasets/{dataset}/runs/{run}/predictions", json={"source": source}
    )

    assert checked["level"] == "blocked" and checked["missing"] == ["income"]
    assert "income" in checked["message"]
    assert refused.status_code == 422 and "income" in refused.json()["error"]["message"]


def test_unknown_categories_are_shown_and_rows_less_reliable(
    client: TestClient, trained: tuple[str, str, Path], tmp_path: Path
) -> None:
    dataset, run, _ = trained
    frame = month(11).drop(columns="churn")
    frame.loc[frame.index[:18], "region"] = "Corsica"  # 3 % of 600 rows
    source = import_path(client, write(frame, tmp_path / "corsica.csv"))

    found = predict(client, dataset, run, source, [])
    region = next(c for c in found["check"]["columns"] if c["column"] == "region")
    flagged = client.get(
        f"{BASE}/datasets/{dataset}/predictions/{found['id']}/rows",
        params={"less_reliable": True, "limit": 50},
    ).json()

    assert found["level"] == "check"
    assert region["unknown_rows"] == 18 and region["unknown_values"] == ["Corsica"]
    assert region["unknown_rule"] and "Corsica" in region["note"]
    assert found["less_reliable"] >= 18
    corsica = [r for r in flagged["rows"] if "region" in str(r[flagged["columns"].index("why")])]
    assert [r[0] for r in corsica] == list(range(1, 19))


def test_with_the_target_the_scores_sit_next_to_the_validation(
    client: TestClient, trained: tuple[str, str, Path], tmp_path: Path
) -> None:
    dataset, run, _ = trained
    source = import_path(client, write(month(11), tmp_path / "answered.csv"))

    found = predict(client, dataset, run, source, [])
    summary = found["summary"]
    run_json = client.get(f"{BASE}/datasets/{dataset}/runs/{run}").json()

    assert found["check"]["target_present"] is True
    assert summary["scored_rows"] == 600
    assert summary["scores"][summary["primary"]] is not None
    assert (
        summary["validation_scores"][summary["primary"]] == run_json["scores"][run_json["primary"]]
    )


def test_csv_export_neutralises_formulas_in_cells_and_headers(
    client: TestClient, trained: tuple[str, str, Path], tmp_path: Path
) -> None:
    dataset, run, _ = trained
    frame = month(11).drop(columns="churn")
    frame["=cmd"] = '=HYPERLINK("http://x")'
    frame.loc[0, "client_id"] = "@SUM(A1)"
    source = import_path(client, write(frame, tmp_path / "formulas.csv"))
    found = predict(client, dataset, run, source, ["client_id", "=cmd"])

    exported = client.get(f"{BASE}/datasets/{dataset}/predictions/{found['id']}/predictions.csv")
    lines = list(csv.reader(io.StringIO(exported.text)))

    assert exported.status_code == 200
    assert lines[0][:4] == ["row", "client_id", "'=cmd", "predicted"]
    assert lines[1][1] == "'@SUM(A1)" and lines[1][2].startswith("'=HYPERLINK")
    assert len(lines) == 601


def test_save_as_dataset_keeps_the_rows_and_adds_the_predictions(
    client: TestClient, trained: tuple[str, str, Path], tmp_path: Path
) -> None:
    dataset, run, _ = trained
    source = import_path(client, write(month(11).drop(columns="churn"), tmp_path / "save.csv"))
    found = predict(client, dataset, run, source, ["client_id"])

    created = client.post(f"{BASE}/datasets/{dataset}/predictions/{found['id']}/dataset")
    new_id = created.json()["dataset_id"]
    columns = client.get(f"/api/v1/datasets/{new_id}/columns").json()
    names = [c["name"] for c in columns]

    assert created.status_code == 201
    assert names[:4] == ["client_id", "region", "income", "age"]
    assert {"predicted", "p(leave)", "p(stay)", "less_reliable", "why"} <= set(names)
    assert client.get(f"/api/v1/datasets/{new_id}").json()["row_count"] == 600
    again = client.post(f"{BASE}/datasets/{dataset}/predictions/{found['id']}/dataset").json()
    assert again["dataset_id"] == new_id


def test_one_predicted_row_is_explained(
    client: TestClient, trained: tuple[str, str, Path], tmp_path: Path
) -> None:
    dataset, run, _ = trained
    source = import_path(client, write(month(11).drop(columns="churn"), tmp_path / "why.csv"))
    found = predict(client, dataset, run, source, [])

    explained = client.get(
        f"{BASE}/datasets/{dataset}/predictions/{found['id']}/rows/3/explanation"
    ).json()

    assert explained["row"] == 3 and explained["unavailable"] is None
    assert explained["steps"] and explained["scale"] == "log-odds"


def test_a_predicted_row_gets_the_scorecard_and_a_counterfactual(
    client: TestClient, trained: tuple[str, str, Path], tmp_path: Path
) -> None:
    """121 bis (R-XAI-18) and 125 bis (R-XAI-21) on a row of new data."""
    dataset, run, _ = trained
    source = import_path(client, write(month(12).drop(columns="churn"), tmp_path / "card.csv"))
    found = predict(client, dataset, run, source, [])
    url = f"{BASE}/datasets/{dataset}/predictions/{found['id']}/rows/2"

    card = client.get(f"{url}/scorecard")
    assert card.status_code == 200, card.text
    body = card.json()
    assert body["row"] == 2 and "R-XAI-18" in body["sentence"]
    assert body["total"] == sum(line["points"] for line in body["lines"])

    changed = client.post(f"{url}/counterfactual", json={"max_changes": 2})
    assert changed.status_code == 200, changed.text
    answer = changed.json()
    assert answer["row"] == 2 and len(answer["changes"]) <= 2
    assert "R-XAI-21" in answer["plausibility"]
    missing = client.post(
        f"{BASE}/datasets/{dataset}/predictions/{found['id']}/rows/99999/counterfactual", json={}
    )
    assert missing.status_code == 422


def test_the_exported_model_has_no_rows_and_its_script_predicts_like_the_app(
    client: TestClient, trained: tuple[str, str, Path], tmp_path: Path
) -> None:
    dataset, run, _ = trained
    november = write(month(11).drop(columns="churn"), tmp_path / "november.csv")
    source = import_path(client, november)
    found = predict(client, dataset, run, source, [])
    rows = all_rows(client, dataset, found["id"])

    exported = client.get(f"{BASE}/datasets/{dataset}/runs/{run}/model.zip")
    name = exported.headers["content-disposition"].split('filename="')[1].rstrip('"')
    folder = tmp_path / "elsewhere"
    folder.mkdir()
    (folder / name).write_bytes(exported.content)
    with zipfile.ZipFile(folder / name) as archive:
        names = set(archive.namelist())
        card = json.loads(archive.read("model_card.json"))
        (folder / "predict.py").write_bytes(archive.read("predict.py"))
    done = subprocess.run(
        [sys.executable, "predict.py", str(november), "out.csv"],
        cwd=folder,
        capture_output=True,
        text=True,
        env={**os.environ, "QUANTILO_CACHE": str(tmp_path / "script-cache")},
        timeout=300,
    )

    assert name == "churn_logistic_regression.zip"
    assert names == {"model.joblib", "model_card.json", "predict.py"}
    assert card["target"]["name"] == "churn" and card["pipeline_sha256"]
    assert "C10-0001" not in json.dumps(card)  # no row of the training data
    assert done.returncode == 0, done.stderr
    out = pd.read_csv(folder / "out.csv")
    assert list(out["predicted"]) == list(rows["predicted"])
    assert np.allclose(out["p(leave)"], rows["p(leave)"].astype(float))


def test_unknown_prediction_and_run(client: TestClient, trained: tuple[str, str, Path]) -> None:
    dataset, _, _ = trained
    assert client.get(f"{BASE}/datasets/{dataset}/predictions/nope").status_code == 404
    assert client.get(f"{BASE}/datasets/{dataset}/runs/nope/model.zip").status_code == 404
    assert client.get(f"{BASE}/predict-jobs/nope").status_code == 404  # not an id
    assert client.get(f"{BASE}/predict-jobs/0123456789ab").status_code == 422  # no longer followed


def test_the_agent_predicts_then_saves_after_confirmation(
    client: TestClient, trained: tuple[str, str, Path], tmp_path: Path
) -> None:
    dataset, run, _ = trained
    import_path(client, write(month(11), tmp_path / "december.csv"))
    registry = client.app.state.agent.registry  # type: ignore[attr-defined]
    tool, save = registry.get("predict"), registry.get("save_predictions")
    assert tool is not None and save is not None
    ctx = ToolContext(dataset)

    found = tool.run(ctx, tool.parse({"run_id": run, "data": "december"}))
    saved = save.run(ctx, save.parse({"prediction_id": found.content["prediction_id"]}))

    assert tool.kind == "compute" and save.kind == "write"
    assert found.content["rows"] == 600 and found.content["check"]["level"] == "ready"
    assert found.content["scores_on_this_data"] and found.content["validation_scores"]
    assert found.link.params["prediction"] == found.content["prediction_id"]
    assert saved.content["dataset_id"]
    with pytest.raises(ToolError, match="No ready dataset"):
        tool.run(ctx, tool.parse({"run_id": run, "data": "nope"}))


@pytest.mark.parametrize("bad", ["../x", r"..\x", "5fdf23a930fe/../../x", "C:/x", "ABCDEF123456"])
def test_ids_never_name_a_file_outside_the_workspace(
    client: TestClient, trained: tuple[str, str, Path], bad: str
) -> None:
    from quantilo.errors import ExplorerError

    dataset, run, _ = trained
    service = client.app.state.ml_predictions  # type: ignore[attr-defined]
    calls = [
        lambda: service.card(dataset, bad),
        lambda: service.export(dataset, bad),
        lambda: service.check(dataset, bad, dataset),
        lambda: service.check(dataset, run, bad),
        lambda: service.list(dataset, bad),
        lambda: service.start(bad, run, dataset),
        lambda: service.get(dataset, bad),
        lambda: service.rows(dataset, bad),
        lambda: list(service.csv(dataset, bad)),
        lambda: service.create_dataset(dataset, bad),
        lambda: service.explain_row(dataset, bad, 0),
        lambda: service.job(bad),
    ]
    for call in calls:
        with pytest.raises(ExplorerError) as refused:
            call()
        assert refused.value.status_code == 404

    tool = client.app.state.agent.registry.get("predict")  # type: ignore[attr-defined]
    with pytest.raises(ToolError, match="Unknown run"):
        tool.run(ToolContext(dataset), tool.parse({"run_id": bad}))


def test_an_encoded_path_in_the_url_is_refused(
    client: TestClient, trained: tuple[str, str, Path]
) -> None:
    dataset, _, _ = trained
    for url in (
        f"{BASE}/datasets/{dataset}/runs/..%2Fx/model.zip",
        f"{BASE}/datasets/{dataset}/runs/..%5Cx/card",
        f"{BASE}/datasets/{dataset}/predictions/..%2F..%2Fx/rows",
    ):
        assert client.get(url).status_code == 404, url


def test_the_check_shows_the_drift_against_the_training_data(
    client: TestClient, trained: tuple[str, str, Path], tmp_path: Path
) -> None:
    # 037 M4: 031's comparison, the model's columns only, a strong shift asks for a look
    dataset, run, _ = trained
    same = import_path(client, write(month(11).drop(columns="churn"), tmp_path / "same.csv"))
    shifted = month(12).drop(columns="churn")
    shifted["income"] = shifted["income"] * 1.5
    moved = import_path(client, write(shifted, tmp_path / "shifted.csv"))
    url = f"{BASE}/datasets/{dataset}/runs/{run}/check"

    calm = client.get(url, params={"source": same}).json()
    assert calm["drift"] is not None and calm["drift"]["shifted"] == []
    assert {c["column"] for c in calm["drift"]["columns"]} >= {"income", "region", "age"}
    found = client.get(url, params={"source": moved}).json()
    assert found["drift"]["shifted"] == ["income"]
    assert found["level"] == "check"
    assert any("income" in p and "PSI" in p for p in found["problems"])
