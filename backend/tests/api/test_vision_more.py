"""118: concepts of a set, Advanced > More options, a run with a concept bottleneck and its
explanations (classic vectors: no torch needed)."""

from pathlib import Path

from fastapi.testclient import TestClient

from tests.api.test_vision import BASE, import_set, stripe_folder, wait


def test_concepts_options_and_a_bottleneck_run(client: TestClient, tmp_path: Path) -> None:
    dataset_id = import_set(client, stripe_folder(tmp_path / "stripes", per_class=14))
    url = f"{BASE}/datasets/{dataset_id}"
    task = client.get(f"{url}/task").json()
    more = {o["key"]: o for o in task["more"]}
    assert set(more) == {"finetune", "protopnet", "concept_bottleneck"}
    bottleneck = more["concept_bottleneck"]
    assert not bottleneck["available"] and "at least 2 concepts" in bottleneck["reason"]
    labels = {c["key"]: c for c in task["concepts"]}
    assert (
        labels["label:horizontal"]["source"] == "label"
        and not labels["label:horizontal"]["bottleneck"]
    )

    index = client.get(f"/api/v1/datasets/{dataset_id}/images/index").json()
    pairs = list(zip(index["image_id"], index["label"], strict=True))
    horizontal = [i for i, label in pairs if label == "horizontal"]
    every_other = [i for i, _ in pairs][::2]
    saved = client.post(f"{url}/concepts", json={"name": "lines across", "image_ids": horizontal})
    assert saved.status_code == 201, saved.text
    assert saved.json()["images"] == 14 and saved.json()["bottleneck"]
    assert (
        client.post(
            f"{url}/concepts", json={"name": "Lines across", "image_ids": horizontal}
        ).status_code
        == 409
    )
    other = client.post(f"{url}/concepts", json={"name": "every other", "image_ids": every_other})
    assert other.status_code == 201
    listed = client.get(f"{url}/concepts").json()
    assert {c["name"] for c in listed if c["source"] == "group"} == {"lines across", "every other"}

    task = client.get(f"{url}/task").json()
    bottleneck = next(o for o in task["more"] if o["key"] == "concept_bottleneck")
    assert bottleneck["available"] and bottleneck["estimate"]

    started = client.post(f"{url}/runs", json={"deep": False, "more": ["concept_bottleneck"]})
    assert started.status_code == 202, started.text
    job = wait(client, started.json())
    assert job["status"] == "done", job
    run_id = job["run_id"]
    run = client.get(f"{url}/runs/{run_id}").json()
    model = next(m for m in run["result"]["models"] if m["key"] == "concept_bottleneck")
    assert model["level"] == "more" and not model["default"]
    quality = {c["name"]: c for c in model["concepts"]}
    assert quality["lines across"]["metric"] == "roc_auc" and quality["lines across"]["value"] > 0.8

    predictions = client.get(f"{url}/runs/{run_id}/predictions").json()
    image_id = predictions[0]["image_id"]
    found = client.get(f"{url}/runs/{run_id}/explain/images/{image_id}/concepts")
    assert found.status_code == 200, found.text
    body = found.json()
    assert [c["name"] for c in body["concepts"]] == ["lines across", "every other"]
    assert body["head"]["classes"] == ["horizontal", "vertical"]
    assert body["concepts"][0]["truth"] in (0.0, 1.0)

    why = client.get(
        f"{url}/runs/{run_id}/explain/images/{image_id}", params={"model": "concept_bottleneck"}
    )
    assert why.status_code == 200, why.text
    why = why.json()
    assert abs(why["prediction"]["probability"] - predictions[0]["concept_bottleneck_proba"]) < 1e-4
    assert not any(m["available"] for m in why["methods"])
    assert client.get(f"{url}/runs/{run_id}/explain/prototypes").json() is None

    options = client.get(f"{url}/runs/{run_id}/explain/tcav/options").json()
    assert not options["available"] and "linear probe" in options["reason"]
    refused = client.get(
        f"{url}/runs/{run_id}/explain/tcav", params={"concept": "label:horizontal"}
    )
    assert refused.status_code == 409

    group_id = next(c["id"] for c in listed if c["name"] == "every other")
    assert client.delete(f"{url}/concepts/{group_id}").status_code == 204
    assert client.delete(f"{url}/concepts/{group_id}").status_code == 404
