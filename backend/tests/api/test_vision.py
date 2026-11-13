"""112: vision models as a background job: the task rule, a run, its results, cancel, delete."""

import io
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fastapi.testclient import TestClient
from PIL import Image

BASE = "/api/v1/modules/vision"


def stripes(seed: int, vertical: bool) -> bytes:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:64, 0:64]
    wave = np.sin((xx if vertical else yy) / 3.0 + rng.uniform(0, 2 * np.pi))
    grey = 128 + 90 * wave + rng.normal(0, 25, (64, 64))
    rgb = np.clip(np.stack([grey, grey * 0.8, grey * 0.6], axis=2), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buffer, "PNG")
    return buffer.getvalue()


def stripe_folder(root: Path, per_class: int = 12) -> Path:
    for label, vertical in (("horizontal", False), ("vertical", True)):
        (root / label).mkdir(parents=True)
        for i in range(per_class):
            (root / label / f"{i}.png").write_bytes(
                stripes(i + (1000 if vertical else 0), vertical)
            )
    return root


def import_set(client: TestClient, path: Path) -> str:
    response = client.post("/api/v1/images/import", json={"path": str(path)})
    assert response.status_code == 201, response.text
    return response.json()["dataset"]["id"]


def wait(client: TestClient, job: dict, timeout: float = 300) -> dict:
    deadline = time.monotonic() + timeout
    while job["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.2)
        job = client.get(f"{BASE}/jobs/{job['id']}").json()
    return job


def test_task_rule_and_a_run(client: TestClient, tmp_path: Path) -> None:
    dataset_id = import_set(client, stripe_folder(tmp_path / "stripes"))
    task = client.get(f"{BASE}/datasets/{dataset_id}/task")
    assert task.status_code == 200, task.text
    body = task.json()
    assert body["choice"]["task"] == "classification" and body["choice"]["code"] == "R-VML-01"
    assert body["split"]["code"] == "R-VSP-02" and body["split"]["scored"] == 24
    assert body["models"][:2] == [
        "Logistic regression on classic descriptors",
        "Gradient boosting (HGB) on classic descriptors",
    ]
    forced = client.get(f"{BASE}/datasets/{dataset_id}/task", params={"task": "clusters"}).json()
    assert forced["choice"]["chosen_by"] == "user" and forced["choice"]["rule_code"] == "R-VML-01"
    impossible = client.get(f"{BASE}/datasets/{dataset_id}/task", params={"task": "anomaly"})
    assert impossible.status_code == 422

    started = client.post(f"{BASE}/datasets/{dataset_id}/runs", json={"deep": False})
    assert started.status_code == 202, started.text
    job = wait(client, started.json())
    assert job["status"] == "done", job
    assert job["progress"] == 1.0 and job["run_id"]
    run = client.get(f"{BASE}/datasets/{dataset_id}/runs/{job['run_id']}").json()
    result = run["result"]
    assert result["task"]["task"] == "classification" and result["deep"]["used"] is False
    assert [m["key"] for m in result["models"]] == ["logistic", "hgb"]
    default = next(m for m in result["models"] if m["default"])
    assert default["classification"]["n"] == 24 and default["classification"]["intervals"]
    summaries = client.get(f"{BASE}/datasets/{dataset_id}/runs").json()
    assert [s["id"] for s in summaries] == [job["run_id"]]
    assert summaries[0]["metric"] in ("accuracy", "balanced_accuracy")
    assert summaries[0]["scored"] == 24
    rows = client.get(f"{BASE}/datasets/{dataset_id}/runs/{job['run_id']}/predictions").json()
    assert len(rows) == 24 and {"row", "image_id", "fold", "truth", "logistic_pred"} <= set(rows[0])
    # the vectors are cached with the set
    settings_home = Path(client.app.state.workspace.root)  # type: ignore[attr-defined]
    assert (settings_home / "images" / dataset_id / "embeddings" / "classic.parquet").is_file()
    # 119: the map's space is kept apart from the vector cache (it was overwritten by it)
    space = settings_home / "images" / dataset_id / "embeddings" / "map-classic.parquet"
    assert b"quantilo.space" in (pq.read_schema(space).metadata or {})
    # R-VQ-16 (b2): the latest classification run, read on the Quality tab
    mislabelled = client.get(f"{BASE}/datasets/{dataset_id}/mislabelled").json()
    assert mislabelled["run_id"] == job["run_id"] and mislabelled["checked"] == 24
    assert mislabelled["parameters"] == {"given": 0.1, "other": 0.8}
    assert client.delete(f"{BASE}/datasets/{dataset_id}/runs/{job['run_id']}").status_code == 204
    assert client.get(f"{BASE}/datasets/{dataset_id}/runs").json() == []
    none = client.get(f"{BASE}/datasets/{dataset_id}/mislabelled").json()
    assert none["run_id"] is None and "Model tab" in none["message"]
    assert client.get(f"{BASE}/datasets/{dataset_id}/runs/{job['run_id']}").status_code == 404


def test_cancel_keeps_nothing(client: TestClient, tmp_path: Path) -> None:
    from app.modules.vision.service import VisionJob, VisionRunRequest

    dataset_id = import_set(client, stripe_folder(tmp_path / "stripes"))
    service = client.app.state.vision  # type: ignore[attr-defined]
    job = VisionJob("a" * 12, dataset_id, "running", "features")
    service._jobs[job.id] = job
    service._stops.add(job.id)
    service._run(job, service._record(dataset_id), VisionRunRequest(deep=False))
    assert job.status == "cancelled" and job.error == "Stopped: nothing was saved."
    assert service.runs(dataset_id) == []
    assert client.delete(f"{BASE}/jobs/{job.id}").json()["status"] == "cancelled"


def test_not_an_image_set_or_too_few_images(client: TestClient, tmp_path: Path) -> None:
    csv = tmp_path / "t.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    imported = client.post("/api/v1/datasets", json={"path": str(csv)})
    assert imported.status_code in (200, 201), imported.text
    table_id = imported.json()["dataset"]["id"]
    assert client.get(f"{BASE}/datasets/{table_id}/task").status_code == 409
    small = stripe_folder(tmp_path / "small", per_class=4)
    small_id = import_set(client, small)
    body = client.get(f"{BASE}/datasets/{small_id}/task").json()
    assert body["choice"]["code"] == "R-VML-07" and body["split"] is None
    refused = client.post(f"{BASE}/datasets/{small_id}/runs", json={})
    assert refused.status_code == 409 and "at least 20 images" in refused.json()["error"]["message"]
    assert client.get(f"{BASE}/jobs/{'b' * 12}").status_code == 404
    assert client.get(f"{BASE}/datasets/{small_id}/runs/../x").status_code == 404


def part(seed: int, defect: bool) -> tuple[bytes, bytes]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:96, 0:96]
    grey = np.full((96, 96), 60.0)
    grey[(yy - 48) ** 2 + (xx - 48) ** 2 < 34**2] = 200.0
    grey += rng.normal(0, 4, grey.shape)
    mask = np.zeros((96, 96), dtype=np.uint8)
    if defect:
        cy, cx = rng.integers(30, 66, 2)
        blot = (yy - cy) ** 2 + (xx - cx) ** 2 < 8**2
        grey[blot], mask[blot] = 20.0, 255
    out = []
    for array, mode in ((np.stack([grey] * 3, axis=2), "RGB"), (mask, "L")):
        buffer = io.BytesIO()
        Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode).save(buffer, "PNG")
        out.append(buffer.getvalue())
    return out[0], out[1]


def inspection_folder(root: Path) -> Path:
    """The MVTec AD layout: train/good, test/good, test/blot and ground_truth/blot masks."""
    for folder in ("train/good", "test/good", "test/blot", "ground_truth/blot"):
        (root / folder).mkdir(parents=True)
    for i in range(30):
        (root / "train/good" / f"{i:03d}.png").write_bytes(part(i, False)[0])
    for i in range(6):
        (root / "test/good" / f"{i:03d}.png").write_bytes(part(100 + i, False)[0])
        image, mask = part(200 + i, True)
        (root / "test/blot" / f"{i:03d}.png").write_bytes(image)
        (root / "ground_truth/blot" / f"{i:03d}_mask.png").write_bytes(mask)
    return root


def test_anomaly_run_keeps_its_maps(client: TestClient, tmp_path: Path) -> None:
    dataset_id = import_set(client, inspection_folder(tmp_path / "bottle"))
    body = client.get(f"{BASE}/datasets/{dataset_id}/task").json()
    assert body["choice"]["task"] == "anomaly"
    keys = {d["key"] for d in body["detectors"]}
    assert keys == {"patchcore", "padim", "autoencoder", "vae", "patchcore_ae"}
    assert {w["key"] for w in body["weights"]} == {"wide_resnet50_2", "resnet18"}
    started = client.post(
        f"{BASE}/datasets/{dataset_id}/runs", json={"deep": False, "detector": "none"}
    )
    job = wait(client, started.json())
    assert job["status"] == "done", job
    run = client.get(f"{BASE}/datasets/{dataset_id}/runs/{job['run_id']}").json()
    models = {m["key"]: m for m in run["result"]["models"]}
    assert models["ensemble"]["default"] and models["ensemble"]["map_source"] == "patch_pca"
    low, high = models["patch_pca"]["map_range"]
    assert high > low
    rows = client.get(f"{BASE}/datasets/{dataset_id}/runs/{job['run_id']}/predictions").json()
    row = next(r["row"] for r in rows if r["is_defect"])
    image = client.get(f"{BASE}/datasets/{dataset_id}/runs/{job['run_id']}/maps/patch_pca/{row}")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert image.headers["x-content-type-options"] == "nosniff"
    assert Image.open(io.BytesIO(image.content)).mode == "L"
    missing = f"{BASE}/datasets/{dataset_id}/runs/{job['run_id']}/maps/PATCH/{row}"
    assert client.get(missing).status_code == 404
    recon = f"{BASE}/datasets/{dataset_id}/runs/{job['run_id']}/reconstructions/patch_pca/{row}"
    assert client.get(recon).status_code == 404


def test_weights_are_listed_and_unknown_ones_refused(client: TestClient) -> None:
    listed = client.get(f"{BASE}/weights").json()
    assert {"dinov2_s14", "wide_resnet50_2", "resnet18"} <= {w["key"] for w in listed}
    assert all(w["size"] and w["licence"] for w in listed)
    assert client.post(f"{BASE}/weights/nope/download").status_code == 404


def test_a_regression_on_a_column_of_the_labels_file(client: TestClient, tmp_path: Path) -> None:
    """c3, R-VML-05: a numeric column of a labels file, offered then trained (R-VRG)."""
    root = stripe_folder(tmp_path / "parts")
    rows = ["file,label,weight_g"]
    for label in ("horizontal", "vertical"):
        for i in range(12):
            rows.append(
                f"{label}/{i}.png,{label},{(i * 7.5 + (3 if label == 'vertical' else 0)):.2f}"
            )
    labels = tmp_path / "labels.csv"
    labels.write_text("\n".join(rows) + "\n", encoding="utf-8")
    response = client.post(
        "/api/v1/images/import", json={"path": str(root), "options": {"labels": str(labels)}}
    )
    assert response.status_code == 201, response.text
    dataset_id = response.json()["dataset"]["id"]
    for _ in range(200):
        if client.get(f"/api/v1/datasets/{dataset_id}").json()["status"] == "ready":
            break
        time.sleep(0.1)
    task = client.get(f"{BASE}/datasets/{dataset_id}/task").json()
    assert task["choice"]["numeric_targets"] == ["weight_g"]
    assert "regression" in task["choice"]["alternatives"]
    forced = client.get(
        f"{BASE}/datasets/{dataset_id}/task", params={"task": "regression", "target": "weight_g"}
    ).json()
    assert forced["choice"]["code"] == "R-VML-05" and forced["split"]["code"] == "R-VSP-06"
    assert forced["models"] == ["Ridge regression on classic descriptors"]
    bad = client.get(
        f"{BASE}/datasets/{dataset_id}/task", params={"task": "regression", "target": "nope"}
    )
    assert bad.status_code == 422
    started = client.post(
        f"{BASE}/datasets/{dataset_id}/runs",
        json={"task": "regression", "target": "weight_g", "deep": False},
    )
    job = wait(client, started.json())
    assert job["status"] == "done", job
    run = client.get(f"{BASE}/datasets/{dataset_id}/runs/{job['run_id']}").json()
    ridge = run["result"]["models"][0]
    assert ridge["key"] == "ridge" and ridge["regression"]["n"] == 24
    summary = client.get(f"{BASE}/datasets/{dataset_id}/runs").json()[0]
    assert summary["metric"] == "r2" and summary["task"] == "regression"
    predictions = client.get(f"{BASE}/datasets/{dataset_id}/runs/{job['run_id']}/predictions")
    assert "ridge_pred" in predictions.json()[0]
