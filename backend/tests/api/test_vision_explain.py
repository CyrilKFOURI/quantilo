"""117: "Why?" on an image of a kept run, its maps, and the Explain tab (no torch needed)."""

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.api.test_vision import BASE, import_set, stripe_folder, wait


def run_of(client: TestClient, dataset_id: str) -> str:
    started = client.post(f"{BASE}/datasets/{dataset_id}/runs", json={"deep": False})
    assert started.status_code == 202, started.text
    job = wait(client, started.json())
    assert job["status"] == "done", job
    return job["run_id"]


def test_why_on_a_classified_image(client: TestClient, tmp_path: Path) -> None:
    dataset_id = import_set(client, stripe_folder(tmp_path / "stripes"))
    run_id = run_of(client, dataset_id)
    rows = client.get(f"{BASE}/datasets/{dataset_id}/runs/{run_id}/predictions").json()
    image_id = rows[0]["image_id"]
    url = f"{BASE}/datasets/{dataset_id}/runs/{run_id}/explain/images/{image_id}"
    response = client.get(url)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task"] == "classification" and body["prediction"]["tested"] is True
    probabilities = body["prediction"]["probabilities"]
    assert {p["label"] for p in probabilities} == {"horizontal", "vertical"}
    assert abs(sum(p["probability"] for p in probabilities) - 1) < 1e-6
    assert body["prediction"]["label"] == probabilities[0]["label"]
    # the run's own prediction for the image comes back (same model, same fold, same C)
    default = body["model"]["key"]
    assert rows[0][f"{default}_pred"] == body["prediction"]["label"]
    assert abs(rows[0][f"{default}_proba"] - body["prediction"]["probability"]) < 1e-4
    nearest = body["evidence"]["nearest"]
    assert len(nearest) == 5 and image_id not in {n["image_id"] for n in nearest}
    sims = [n["similarity"] for n in nearest]
    assert sims == sorted(sims, reverse=True)
    assert "R-VEX-01" in body["evidence"]["sentence"]
    assert body["evidence"]["prototype"]["label"] == body["prediction"]["label"]
    assert body["counterfactual"]["image"]["label"] != body["prediction"]["label"]
    methods = {m["key"]: m for m in body["methods"]}
    assert methods["occlusion"]["available"] and methods["occlusion"]["level"] == "advanced"
    assert not methods["grad_cam"]["available"] and "gradient" in methods["grad_cam"]["reason"]
    assert methods["sanity"]["level"] == "more"
    again = client.get(url).json()
    assert again["evidence"] == body["evidence"]  # from the cache next to the run

    occlusion = client.get(f"{url}/map", params={"method": "occlusion"})
    assert occlusion.status_code == 200, occlusion.text
    found = occlusion.json()
    assert found["rows"] == 8 and found["columns"] == 8 and found["blocky"]
    assert found["png"].startswith("data:image/png;base64,")
    assert found["low"] >= 0 and found["high"] >= found["low"]
    assert "post-hoc" in found["warning"] and not found["own"]
    other = "vertical" if body["prediction"]["label"] == "horizontal" else "horizontal"
    targeted = client.get(f"{url}/map", params={"method": "occlusion", "target": other}).json()
    assert targeted["target"] == other
    assert client.get(f"{url}/map", params={"method": "occlusion"}).json()["cached"] is True
    refused = client.get(f"{url}/map", params={"method": "grad_cam"})
    assert refused.status_code == 422
    assert client.get(f"{url}/map", params={"method": "nope"}).status_code == 422
    assert (
        client.get(f"{url}/map", params={"method": "occlusion", "target": "x"}).status_code == 404
    )

    overview = client.get(f"{BASE}/datasets/{dataset_id}/runs/{run_id}/explain")
    assert overview.status_code == 200, overview.text
    tab = overview.json()
    assert [c["label"] for c in tab["classes"]] == ["horizontal", "vertical"]
    assert all(len(c["prototypes"]) == 3 for c in tab["classes"])
    families = {f["key"] for f in tab["families"]}
    assert families == {"colour", "shape", "texture"}
    assert "R-VEX-13" in tab["families_sentence"]

    hgb = client.get(url, params={"model": "hgb"}).json()
    assert hgb["model"]["key"] == "hgb"
    assert client.get(url, params={"model": "nope"}).status_code == 404
    assert (
        client.get(f"{BASE}/datasets/{dataset_id}/runs/{run_id}/explain/images/x").status_code
        == 404
    )


def texture(seed: int, broken: bool = False) -> bytes:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:64, 0:64]
    grey = 128 + 40 * np.sin(xx / 3.0) + 40 * np.cos(yy / 4.0) + rng.normal(0, 4, (64, 64))
    rgb = np.clip(np.stack([grey] * 3, axis=2), 0, 255).astype(np.uint8)
    if broken:
        rgb[20:36, 24:40] = (250, 20, 20)
    buffer = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buffer, "PNG")
    return buffer.getvalue()


def mvtec_folder(root: Path) -> Path:
    for folder, count, broken in (
        ("train/good", 24, False),
        ("test/good", 5, False),
        ("test/crack", 5, True),
    ):
        (root / folder).mkdir(parents=True)
        for i in range(count):
            seed = len(folder) * 100 + i
            (root / folder / f"{i:03}.png").write_bytes(texture(seed, broken))
    return root


def test_gradient_maps_are_desktop_only_in_the_browser(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """119, R-VIS-WEB-03: in the browser version every gradient method gives the same reason."""
    from quantilo.ml import browser as ml_browser
    from quantilo.vision import browser

    dataset_id = import_set(client, stripe_folder(tmp_path / "stripes"))
    run_id = run_of(client, dataset_id)
    monkeypatch.setattr(ml_browser, "IN_BROWSER", True)
    image_id = client.get(f"{BASE}/datasets/{dataset_id}/runs/{run_id}/predictions").json()[0][
        "image_id"
    ]
    body = client.get(f"{BASE}/datasets/{dataset_id}/runs/{run_id}/explain/images/{image_id}")
    methods = {m["key"]: m for m in body.json()["methods"]}
    assert methods["occlusion"]["available"]
    for key in ("grad_cam", "rollout", "saliency", "smoothgrad", "integrated_gradients", "sanity"):
        assert not methods[key]["available"]
        assert methods[key]["reason"] == browser.DESKTOP_REASON


def test_why_on_an_anomaly(client: TestClient, tmp_path: Path) -> None:
    dataset_id = import_set(client, mvtec_folder(tmp_path / "part"))
    run_id = run_of(client, dataset_id)
    rows = client.get(f"{BASE}/datasets/{dataset_id}/runs/{run_id}/predictions").json()
    defect = next(r for r in rows if r["is_defect"])
    url = f"{BASE}/datasets/{dataset_id}/runs/{run_id}/explain/images/{defect['image_id']}"
    body = client.get(url).json()
    assert body["task"] == "anomaly", body
    assert body["prediction"]["score"] is not None
    nearest = body["evidence"]["nearest"]
    assert len(nearest) == 5 and all(n["distance"] is not None for n in nearest)
    assert "R-VAN-02" in body["evidence"]["sentence"]
    assert body["counterfactual"] is None
    methods = {m["key"]: m for m in body["methods"]}
    assert methods["patch_pca"]["own"] and methods["patch_pca"]["level"] == "first"
    found = client.get(f"{url}/map", params={"method": "patch_pca"})
    assert found.status_code == 200, found.text
    patch = found.json()
    assert patch["own"] and patch["warning"] == "" and patch["rows"] == 16
    # the painted square (x 24-40, y 20-36 of 64) is where the rebuild fails most
    assert "R-VAN-01" in patch["sentence"]
    tab = client.get(f"{BASE}/datasets/{dataset_id}/runs/{run_id}/explain").json()
    assert [c["label"] for c in tab["classes"]] == ["normal"] and tab["families"] is None
