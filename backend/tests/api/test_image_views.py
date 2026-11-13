"""109 and 110: the map of an image set, and the previews and recipe of the Prepare tab."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from tests.api.test_images import image_folder, import_set


def _set(client: TestClient, tmp_path: Path) -> str:
    return import_set(client, image_folder(tmp_path / "pets"))["dataset"]["id"]


def test_classic_map_is_built_after_the_import(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id = _set(client, tmp_path)
    # R-VMAP-10: built by the import (inline runner here), saved beside the images
    assert (settings.home / "images" / dataset_id / "embeddings" / "map-classic.parquet").is_file()
    body = client.get(f"/api/v1/datasets/{dataset_id}/images/map").json()
    assert body["state"] == "ready", body
    found = body["map"]
    index = client.get(f"/api/v1/datasets/{dataset_id}/images/index").json()
    assert len(found["clusters"]) == index["count"] == 11
    assert len(found["x"]) == len(found["rows"]) == found["points"] == 11
    assert found["k"] >= 2 and found["reading"]["labels"] == 2
    assert found["sentence"]
    tsne = client.get(f"/api/v1/datasets/{dataset_id}/images/map?projection=tsne&k=3").json()
    assert tsne["map"]["projection"] == "tsne" and tsne["map"]["k"] == 3


def test_map_build_states_and_methods(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id = _set(client, tmp_path)
    space = settings.home / "images" / dataset_id / "embeddings" / "map-classic.parquet"
    space.unlink()
    base = f"/api/v1/datasets/{dataset_id}/images/map"
    assert client.get(base).json()["state"] == "missing"
    built = client.post(base, json={"method": "classic"})
    assert built.status_code == 202 and built.json()["state"] == "ready"  # inline runner
    assert client.get(f"{base}?k=1").status_code == 422
    assert client.post(base, json={"method": "nothing"}).status_code == 422

    methods = client.get("/api/v1/images/map/methods").json()
    keys = [m["key"] for m in methods["methods"]]
    assert keys[0] == "classic" and "dinov2_s14" in keys
    deep = next(m for m in methods["methods"] if m["key"] == "dinov2_s14")
    assert deep["size"] == "88 MB" and deep["licence"] == "Apache-2.0"
    if not methods["installed"]:  # the extra is absent here: greyed with its reason
        assert not deep["available"] and "R-VDP-01" in deep["reason"]
        refused = client.post(base, json={"method": "dinov2_s14"})
        assert refused.status_code == 409


def test_map_refuses_a_table(client: TestClient, tmp_path: Path) -> None:
    csv = tmp_path / "t.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    created = client.post("/api/v1/datasets", json={"path": str(csv)}).json()
    response = client.get(f"/api/v1/datasets/{created['dataset']['id']}/images/map")
    assert response.status_code == 409


def test_prepare_recipe_default_save_and_bounds(client: TestClient, tmp_path: Path) -> None:
    dataset_id = _set(client, tmp_path)
    base = f"/api/v1/datasets/{dataset_id}/images/prepare"
    first = client.get(base).json()
    assert first["task"] == "classification" and not first["saved"]
    assert first["recipe"]["augment"] and first["recipe"]["size"] == 224
    assert 0 <= first["borders"]["share"] <= 1
    recipe = {**first["recipe"], "equalise": "clahe", "size": 256}
    saved = client.put(base, json={"recipe": recipe}).json()
    assert saved["saved"] and saved["recipe"]["equalise"] == "clahe"
    assert client.get(base).json()["recipe"]["size"] == 256
    bad = client.put(base, json={"recipe": {**recipe, "sigma": 99}})
    assert bad.status_code == 422 and "sigma" in bad.text


def test_preview_of_a_step(client: TestClient, tmp_path: Path) -> None:
    dataset_id = _set(client, tmp_path)
    index = client.get(f"/api/v1/datasets/{dataset_id}/images/index").json()
    image_id = index["image_id"][0]
    recipe = client.get(f"/api/v1/datasets/{dataset_id}/images/prepare").json()["recipe"]
    url = f"/api/v1/datasets/{dataset_id}/images/prepare/preview"
    body = {"image_id": image_id, "step": "equalise", "recipe": recipe}
    shown = client.post(url, json=body)
    assert shown.status_code == 200, shown.text
    out = shown.json()
    assert out["before"].startswith("data:image/png;base64,")
    assert out["before_size"] == out["after_size"] == [224, 224]
    assert not out["enabled"] and not out["geometric"]
    assert set(out["after_measures"]) >= {"brightness", "contrast", "sharpness"}
    again = client.post(url, json=body).json()
    assert again == out  # from the cache
    resized = client.post(url, json={**body, "step": "resize"}).json()
    assert resized["geometric"] and resized["after_size"] == [224, 224]
    assert client.post(url, json={**body, "step": "glow"}).status_code == 422
    assert client.post(url, json={**body, "image_id": "0" * 16}).status_code == 404


def test_semantic_near_copies_only_when_asked(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-VQ-17 (b2): nothing before the user asks; greyed with its reason without the extra;
    then the groups the hashes missed, kept with the table's stamp."""
    import numpy as np
    from quantilo.vision import classic_models, deep

    dataset_id = _set(client, tmp_path)
    base = f"/api/v1/datasets/{dataset_id}/images/semantic"
    unavailable = SimpleNamespace(
        available=False, reason="Needs the extra (R-VDP-01).", title="DINOv2"
    )
    monkeypatch.setattr(deep, "status", lambda: SimpleNamespace(backbone=lambda key: unavailable))
    monkeypatch.setattr(deep, "installed", lambda: False)
    body = client.get(base).json()
    assert body["state"] == "missing" and body["available"] is False and body["cosine"] == 0.95
    assert "R-VDP-01" in body["reason"]
    refused = client.post(base, json={})
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "vision_extra"

    ready = SimpleNamespace(available=True, reason=None, title="DINOv2 ViT-S/14")
    monkeypatch.setattr(deep, "status", lambda: SimpleNamespace(backbone=lambda key: ready))
    seen: dict = {}

    def fake(sources, image_ids, cache_path, **kwargs):  # rows 1 and 7: the same subject
        seen["cache"] = cache_path
        x = np.eye(len(sources), 16, dtype=np.float32)
        x[7] = x[1] + 0.01  # (rows 0 and 5 are already exact copies: R-VQ-09)
        return x, None

    monkeypatch.setattr(classic_models, "deep_features", fake)
    started = client.post(base, json={"confirm_long": False})
    assert started.status_code == 202, started.text
    found = started.json()  # inline runner: already done
    assert found["state"] == "ready" and found["available"] is True, found
    assert [g["rows"] for g in found["groups"]] == [[1, 7]] and found["compared"] == 11
    assert "1 group" in found["sentence"] and "DINOv2" in found["sentence"]
    assert seen["cache"].name == "dinov2_s14.parquet"  # the models' own vector cache
    assert client.get(base).json()["groups"] == found["groups"]


def test_pacmap_and_trimap_are_greyed_without_the_extra(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """c3, R-VMAP-12: listed with their availability; asked without the extra, 409 and why."""
    from quantilo.interpret import extras

    methods = client.get("/api/v1/images/map/methods").json()
    assert [p["key"] for p in methods["projections"]] == ["pacmap", "trimap"]
    monkeypatch.setattr(extras, "_installed", lambda package: False)
    greyed = client.get("/api/v1/images/map/methods").json()["projections"]
    assert not greyed[0]["available"] and "interpret" in greyed[0]["reason"]
    dataset_id = _set(client, tmp_path)
    response = client.get(
        f"/api/v1/datasets/{dataset_id}/images/map", params={"projection": "pacmap"}
    )
    assert response.status_code == 409 and "R-XAI-10" in response.text
