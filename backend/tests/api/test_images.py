"""106: an image set as a dataset (`format = 'images'`): preview, import job, report, files."""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings


def png(seed: int, size: tuple[int, int] = (96, 72)) -> bytes:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : size[1], 0 : size[0]]
    pixels = np.zeros((size[1], size[0], 3))
    for _ in range(5):
        cy, cx, r = rng.uniform(0, size[1]), rng.uniform(0, size[0]), rng.uniform(8, 30)
        pixels[(yy - cy) ** 2 + (xx - cx) ** 2 < r**2] = rng.uniform(0, 255, 3)
    buffer = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8), "RGB").save(buffer, "PNG")
    return buffer.getvalue()


def image_folder(root: Path) -> Path:
    for label in ("cat", "dog"):
        (root / label).mkdir(parents=True)
        for i in range(5):
            (root / label / f"{i}.png").write_bytes(png(i + (100 if label == "dog" else 0)))
    (root / "cat" / "copy.png").write_bytes(png(0))  # an exact copy of cat/0.png
    (root / "readme.txt").write_text("not an image")
    return root


def import_set(client: TestClient, path: Path, **options: object) -> dict:
    response = client.post("/api/v1/images/import", json={"path": str(path), "options": options})
    assert response.status_code == 201, response.text
    return response.json()


def test_table_preview_routes_a_folder_of_images(client: TestClient, tmp_path: Path) -> None:
    root = image_folder(tmp_path / "pets")
    preview = client.post("/api/v1/sources/preview", json={"path": str(root)}).json()
    assert preview["detected"]["kind"] == "image"
    assert preview["detected"]["rule"] == "R-VIS-SRC-01"


def test_preview_reads_labels_without_decoding(client: TestClient, tmp_path: Path) -> None:
    root = image_folder(tmp_path / "pets")
    body = client.post("/api/v1/images/preview", json={"path": str(root)}).json()
    assert body["source_kind"] == "folder" and body["n_images"] == 11
    assert body["labels"] == {"cat": 6, "dog": 5}
    codes = {rule["code"] for rule in body["rules"]}
    assert {"R-VIS-LAY-01", "R-VIS-ING-02"} <= codes
    assert all(rule["reason"] for rule in body["rules"])
    ignored = client.post(
        "/api/v1/images/preview",
        json={"path": str(root), "options": {"ignore_rules": ["R-VIS-LAY-01"]}},
    ).json()
    assert ignored["labels"] == {}
    assert not (tmp_path / "workspace" / "images").exists()


def test_import_is_a_dataset_with_quality(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    root = image_folder(tmp_path / "pets")
    response = import_set(client, root)
    dataset = response["dataset"]
    assert response["job"]["status"] == "done"
    fresh = client.get(f"/api/v1/datasets/{dataset['id']}").json()
    assert fresh["format"] == "images" and fresh["status"] == "ready"
    assert fresh["row_count"] == 11 and fresh["source_kind"] == "folder"
    assert fresh["source_state"] == "ok"
    table = pq.read_table(settings.home / "datasets" / dataset["id"] / "data.parquet")
    assert {"image_id", "label", "issues", "excluded", "duplicate_group"} <= set(table.column_names)
    # the tabular pages work on the metadata: columns were detected
    columns = client.get(f"/api/v1/datasets/{dataset['id']}/columns")
    assert columns.status_code == 200 and len(columns.json()) == table.num_columns
    report = client.get(f"/api/v1/datasets/{dataset['id']}/images/report").json()
    assert report["n_images"] == 11 and report["labels"] == {"cat": 6, "dog": 5}
    codes = [f["code"] for f in report["quality"]["findings"]]
    assert "R-VQ-09" in codes
    assert report["quality"]["excluded"] == 1
    # the same source and options again: the dataset is reused
    again = import_set(client, root)
    assert again["reused"] is True and again["dataset"]["id"] == dataset["id"]


def test_files_and_thumbnails_are_served_safely(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id = import_set(client, image_folder(tmp_path / "pets"))["dataset"]["id"]
    table = pq.read_table(settings.home / "datasets" / dataset_id / "data.parquet")
    image_id = table["image_id"][0].as_py()
    thumb = client.get(f"/api/v1/datasets/{dataset_id}/images/{image_id}/thumb")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/webp"
    assert thumb.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in thumb.headers["content-security-policy"]
    original = client.get(f"/api/v1/datasets/{dataset_id}/images/{image_id}/file")
    assert original.headers["content-type"] == "image/png"
    assert original.content.startswith(b"\x89PNG")
    for bad in ("../../app.db", "ZZZZ", "0" * 16):
        assert client.get(f"/api/v1/datasets/{dataset_id}/images/{bad}/file").status_code == 404
    assert client.get(f"/api/v1/datasets/{dataset_id}/images/{image_id}/mask").status_code == 404


def test_a_tampered_table_cannot_point_outside(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id = import_set(client, image_folder(tmp_path / "pets"))["dataset"]["id"]
    service = client.app.state.image_sets  # type: ignore[attr-defined]
    import pytest
    from quantilo.errors import NotFoundError

    for rel in ("../../app.db", "//server/share/x.png", "\\\\server\\share\\x.png"):
        with pytest.raises(NotFoundError):
            service._safe_path(dataset_id, rel)


def test_zip_upload_and_delete_removes_the_files(client: TestClient, settings: Settings) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for i in range(4):
            archive.writestr(f"a/{i}.png", png(i))
        archive.writestr("b/0.png", png(9))
    data = buffer.getvalue()
    upload = client.put("/api/v1/uploads", params={"filename": "set.zip"}, content=data)
    assert upload.status_code == 201
    dataset_id = import_set(client, Path(upload.json()["path"]))["dataset"]["id"]
    images_dir = settings.home / "images" / dataset_id
    assert any(images_dir.rglob("*.webp"))
    assert client.delete(f"/api/v1/datasets/{dataset_id}").status_code == 204
    assert not images_dir.exists()


def test_reimport_and_rules(client: TestClient, tmp_path: Path) -> None:
    root = image_folder(tmp_path / "pets")
    dataset_id = import_set(client, root, thresholds={"blur_sharpness": 50})["dataset"]["id"]
    job = client.post(f"/api/v1/datasets/{dataset_id}/reimport")
    assert job.status_code == 202
    assert client.get(f"/api/v1/jobs/{job.json()['id']}").json()["status"] == "done"
    rules = client.get("/api/v1/images/rules").json()
    assert [r["code"] for r in rules][:2] == ["R-VQ-01", "R-VQ-02"]
    assert rules[3]["parameters"]["blur_sharpness"] == 100.0
    bad = client.post(
        "/api/v1/images/import",
        json={"path": str(root), "options": {"thresholds": {"nope": 1}}},
    )
    assert bad.status_code == 422


def test_network_paths_and_tables_are_refused(client: TestClient, tmp_path: Path) -> None:
    unc = client.post("/api/v1/images/preview", json={"path": "\\\\server\\share\\photos"})
    # a network path only on Windows; elsewhere backslashes are a file name that does not exist
    assert unc.status_code in ((400, 422) if sys.platform == "win32" else (400, 404, 422))
    csv = tmp_path / "t.csv"
    csv.write_text("a\n1\n")
    imported = client.post("/api/v1/datasets", json={"path": str(csv)}).json()
    report = client.get(f"/api/v1/datasets/{imported['dataset']['id']}/images/report")
    assert report.status_code == 409
