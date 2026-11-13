"""AI forecasting models (028 B1): states, confirmed downloads as jobs, Settings > Models.

No network: the engine's download is replaced by a fake that writes the files."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from quantilo.timeseries import ai

from app.config import default_samples_dir
from tests.api.test_profiling import import_path
from tests.api.test_timeseries import wait

BASE = "/api/v1/modules/timeseries/forecast-ai"


@pytest.fixture(autouse=True)
def online(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_EXPLORER_OFFLINE", raising=False)
    monkeypatch.delenv("QUANTILO_OFFLINE", raising=False)
    monkeypatch.setattr(ai, "missing_packages", lambda key: [])


def wait_idle(client: TestClient, key: str) -> dict:
    for _ in range(200):
        found = next(m for m in client.get(BASE).json()["models"] if m["key"] == key)
        if found["download"] != "running":
            return found
        time.sleep(0.02)
    raise AssertionError("download still running")


def test_models_are_listed_with_their_state_size_and_source(client: TestClient) -> None:
    body = client.get(BASE).json()

    keys = [m["key"] for m in body["models"]]
    assert keys[:5] == list(ai.PRETRAINED) and keys[5:] == ["nhits", "nbeats"]
    small = body["models"][1]
    assert small["state"] == "needs_download" and small["size_text"] == "191 MB"
    assert small["source"].startswith("https://huggingface.co/amazon/chronos-bolt-small/tree/")
    # the weights live in the app's folder, never the global Hugging Face cache
    assert body["folder"].endswith(str(Path("workspace") / "models" / "forecast"))
    assert small["folder"].startswith(body["folder"])
    assert body["offline"] is False and "forecast-ai" in body["install"]


def test_a_confirmed_download_runs_in_the_background(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(key: str, *, confirm: bool = False, progress=None, cancelled=None, **_) -> Path:
        assert confirm  # the page asked first
        where = ai.folder(key)
        where.mkdir(parents=True)
        for file in ai.spec_of(key).files:
            with (where / file.name).open("wb") as out:
                out.truncate(file.size)
        if progress:
            progress(1, 1)
        return where

    monkeypatch.setattr(ai, "download", fake)

    started = client.post(f"{BASE}/chronos_bolt_tiny/download")
    assert started.status_code == 202
    assert wait_idle(client, "chronos_bolt_tiny")["state"] == "ready"

    deleted = client.delete(f"{BASE}/chronos_bolt_tiny").json()
    assert deleted["state"] == "needs_download" and not ai.folder("chronos_bolt_tiny").exists()


def test_a_failed_download_says_why(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(key: str, **_) -> Path:
        raise ai.AiError("SHA-256 does not match the manifest: file deleted.")

    monkeypatch.setattr(ai, "download", broken)
    client.post(f"{BASE}/chronos_2/download")

    found = wait_idle(client, "chronos_2")
    assert found["download"] == "failed" and "SHA-256" in found["error"]


def test_never_download_is_kept_and_refuses_downloads(client: TestClient) -> None:
    body = client.put(f"{BASE}/offline", json={"offline": True}).json()
    assert body["offline"] is True
    small = next(m for m in body["models"] if m["key"] == "chronos_bolt_small")
    assert small["state"] == "offline" and small["folder"] in small["reason"]

    refused = client.post(f"{BASE}/chronos_bolt_small/download")
    assert refused.status_code == 409 and "Never download" in refused.json()["error"]["message"]
    assert client.put(f"{BASE}/offline", json={"offline": False}).json()["offline"] is False


def test_the_environment_forces_never_download(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_EXPLORER_OFFLINE", "1")

    assert client.get(BASE).json()["offline_forced"] is True
    assert client.put(f"{BASE}/offline", json={"offline": False}).status_code == 409


def test_trained_networks_have_nothing_to_download_and_unknown_keys_are_404(
    client: TestClient,
) -> None:
    assert client.post(f"{BASE}/nhits/download").status_code == 400
    assert client.post(f"{BASE}/gpt/download").status_code == 404


def test_a_forecast_with_a_model_to_download_gives_the_reason_in_its_row(
    client: TestClient,
) -> None:
    sales = import_path(client, default_samples_dir() / "sales_timeseries.csv")
    job = client.post(
        f"/api/v1/modules/timeseries/datasets/{sales}/forecast",
        params={
            "value": "revenue",
            "period": "month",
            "horizon": 6,
            "methods": "naive,chronos_bolt_small",
        },
    ).json()
    job = wait(client, job)

    rows = {r["key"]: r for r in job["result"]["results"]}
    assert rows["naive"]["scores"] is not None
    assert rows["chronos_bolt_small"]["scores"] is None
    assert "Download it first" in rows["chronos_bolt_small"]["error"]
    # 36 months is long enough for the statistical models: no "short series" suggestion
    assert job["result"]["suggestions"] == []
