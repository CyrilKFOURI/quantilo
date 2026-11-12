"""107 and 108: the gallery's index, and the user's decisions saved as a new version."""

from pathlib import Path

import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from app.config import Settings
from tests.api.test_images import image_folder, import_set


def _set(client: TestClient, tmp_path: Path) -> str:
    return import_set(client, image_folder(tmp_path / "pets"))["dataset"]["id"]


def test_index_lists_columns_in_row_order_without_paths(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id = _set(client, tmp_path)
    response = client.get(f"/api/v1/datasets/{dataset_id}/images/index")
    assert response.status_code == 200, response.text
    index = response.json()
    table = pq.read_table(settings.home / "datasets" / dataset_id / "data.parquet")
    assert index["count"] == 11
    assert index["image_id"] == table["image_id"].to_pylist()
    assert sorted(set(index["label"])) == ["cat", "dog"]
    assert all(len(index[name]) == 11 for name in ("brightness", "issues", "excluded", "has_mask"))
    assert index["rule_excluded"] == index["excluded"] and sum(index["excluded"]) == 1
    # R-107-04: never a disk path nor a raw hash (rel_path is the path inside the source)
    for hidden in ("stored_path", "thumb_path", "mask_path", "phash", "sha256", "pixel_sha"):
        assert hidden not in index
    assert all(value is None or value == round(value, 4) for value in index["sharpness"])


def test_index_refuses_a_table(client: TestClient, tmp_path: Path) -> None:
    csv = tmp_path / "t.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    created = client.post("/api/v1/datasets", json={"path": str(csv)}).json()
    response = client.get(f"/api/v1/datasets/{created['dataset']['id']}/images/index")
    assert response.status_code == 409


def test_decisions_are_a_draft_until_saved_as_a_version(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id = _set(client, tmp_path)
    base = f"/api/v1/datasets/{dataset_id}/images"
    index = client.get(f"{base}/index").json()
    copy = index["excluded"].index(True)  # the second exact copy, left out by R-VQ-09
    flagged = 0 if copy != 0 else 1
    # decisions go by rel_path: the two exact copies share their image_id
    keep_id, exclude_id = index["rel_path"][copy], index["rel_path"][flagged]
    assert index["image_id"].count(index["image_id"][copy]) == 2

    empty = client.get(f"{base}/decisions").json()
    assert empty["pending"] == 0 and empty["draft"] == {}
    assert client.post(f"{base}/decisions/apply").status_code == 409

    body = {"decisions": {keep_id: "keep", exclude_id: "exclude"}}
    draft = client.put(f"{base}/decisions", json=body).json()
    assert draft["pending"] == 2 and draft["applied"] == {}
    # nothing changes in the table before saving
    assert client.get(f"{base}/index").json()["excluded"] == index["excluded"]

    bad = client.put(f"{base}/decisions", json={"decisions": {"nope/x.png": "exclude"}})
    assert bad.status_code == 422
    assert client.put(f"{base}/decisions", json={"decisions": {keep_id: "drop"}}).status_code == 422

    before = client.get(f"/api/v1/datasets/{dataset_id}/versions")
    applied = client.post(f"{base}/decisions/apply")
    assert applied.status_code == 200, applied.text
    result = applied.json()
    assert result["excluded"] == 1 and result["decided"] == 2
    assert result["version"] is not None and result["version"] >= 2
    if before.status_code == 200:
        assert result["version"] > max((v["version"] for v in before.json()), default=0)

    after = client.get(f"{base}/index").json()
    assert after["excluded"][copy] is False  # keep wins over R-VQ-09
    assert after["excluded"][flagged] is True
    assert after["rule_excluded"] == index["excluded"]  # what the rules said is kept
    assert after["decision"][copy] == "keep"
    table = pq.read_table(settings.home / "datasets" / dataset_id / "data.parquet")
    assert {"decision", "rule_excluded"} <= set(table.column_names)
    summary = client.get(f"{base}/decisions").json()
    assert summary["pending"] == 0 and summary["applied"] == body["decisions"]

    # back to the rules for one image, then discard the draft
    reset = client.put(f"{base}/decisions", json={"decisions": {keep_id: None}}).json()
    assert reset["pending"] == 1 and keep_id not in reset["draft"]
    discarded = client.delete(f"{base}/decisions").json()
    assert discarded["pending"] == 0 and discarded["draft"] == body["decisions"]


def test_saved_decisions_survive_a_reimport(client: TestClient, tmp_path: Path) -> None:
    dataset_id = _set(client, tmp_path)
    base = f"/api/v1/datasets/{dataset_id}/images"
    index = client.get(f"{base}/index").json()
    target = index["rel_path"][0]
    client.put(f"{base}/decisions", json={"decisions": {target: "exclude"}})
    client.post(f"{base}/decisions/apply")
    response = client.post(f"/api/v1/datasets/{dataset_id}/reimport")
    assert response.status_code in (200, 201, 202), response.text
    again = client.get(f"{base}/index").json()
    position = again["rel_path"].index(target)
    assert again["excluded"][position] is True and again["decision"][position] == "exclude"
