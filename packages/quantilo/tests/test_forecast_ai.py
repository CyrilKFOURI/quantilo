"""AI forecasting models (028 phase B): states, downloads, coded rules, pretrained adapters and
the trained networks.

No download here: files are served by a fake opener into a temporary models folder. The real
weights test runs only when Chronos-Bolt Tiny is already in the default folder (never in CI);
the torch tests are skipped without torch."""

import hashlib
import io
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from quantilo.timeseries import ai, neural
from quantilo.timeseries import forecast as fc
from quantilo.timeseries import groups as gr

REAL_MODELS = Path(os.environ.get("QUANTILO_MODELS_DIR") or ai.models_root())
RNG = np.random.default_rng(5)
T = np.arange(120)
SEASONAL = 100 + 10 * np.sin(2 * np.pi * T / 12) + 0.3 * T + RNG.normal(0, 1, 120)


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("QUANTILO_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("DATA_EXPLORER_OFFLINE", raising=False)
    monkeypatch.delenv("QUANTILO_OFFLINE", raising=False)
    ai.forget()
    yield tmp_path
    ai.forget()


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def fake_model(monkeypatch: pytest.MonkeyPatch, payloads: dict[str, bytes]) -> ai.ModelSpec:
    files = tuple(
        ai.FileSpec(name, len(data), hashlib.sha256(data).hexdigest())
        for name, data in payloads.items()
    )
    spec = ai.ModelSpec(
        "chronos_bolt_tiny",
        "Chronos-Bolt Tiny",
        "chronos_bolt",
        "chronos",
        "Apache-2.0",
        "9 M",
        "",
        "amazon/chronos-bolt-tiny",
        "a" * 40,
        files,
    )
    monkeypatch.setitem(ai.MANIFEST, spec.key, spec)
    monkeypatch.setattr(ai, "missing_packages", lambda key: [])
    return spec


def serving(payloads: dict[str, bytes], calls: list[str]):
    def opener(request, timeout):
        calls.append(request.full_url)
        name = request.full_url.rsplit("/", 1)[-1]
        return FakeResponse(payloads[name])

    return opener


# --- States (B1) ---------------------------------------------------------------------------------


def test_every_ai_model_is_listed_with_a_state_and_a_reason(models_dir: Path) -> None:
    found = {s.key: s for s in ai.statuses()}

    assert set(found) == set(fc.AI_MODELS)
    for key in fc.PRETRAINED_MODELS:
        spec = ai.spec_of(key)
        assert spec.revision and len(spec.revision) == 40 and spec.licence == "Apache-2.0"
        assert all(len(f.sha256) == 64 and f.size > 0 for f in spec.files)
        assert found[key].source.startswith("https://huggingface.co/")
        assert found[key].state != "ready" or ai.downloaded(key)
        if found[key].state != "ready":
            assert found[key].reason
    assert ai.spec_of("chronos_2").drivers and not ai.spec_of("chronos_bolt_small").drivers
    assert not found["nhits"].pretrained and found["nhits"].size == 0


def test_without_the_extra_the_reason_gives_the_install_command(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ai, "_installed", lambda package: package != "chronos")

    found = ai.status("chronos_bolt_small")

    assert found.state == "needs_extra" and found.missing == ["chronos-forecasting"]
    assert found.install == ai.INSTALL and "forecast-ai" in (found.reason or "")
    with pytest.raises(fc.Unavailable, match="forecast-ai"):
        fc.ai_model("chronos_bolt_small")(SEASONAL, 12, 12)


def test_the_browser_version_greys_every_ai_model(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "emscripten")

    assert {s.state for s in ai.statuses()} == {"browser"}
    with pytest.raises(ai.AiError, match="browser"):
        ai.download("chronos_bolt_tiny", confirm=True)


def test_an_unavailable_model_reports_its_reason_instead_of_breaking_the_forecast(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ai, "missing_packages", lambda key: [])

    result = fc.run_model("chronos_bolt_base", fc.ai_model("chronos_bolt_base"), SEASONAL, 12, 12)

    assert result.scores is None and "Download it first" in (result.error or "")
    assert ai.refusal("chronos_bolt_base", 120) == result.error


# --- Downloads (B1): only on confirmation, checked, resumed, never offline -----------------------


def test_nothing_is_fetched_without_confirmation(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {"config.json": b"{}", "model.safetensors": b"w" * 1000}
    fake_model(monkeypatch, payloads)
    calls: list[str] = []
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: pytest.fail("connected without confirmation")
    )

    with pytest.raises(ai.ConfirmationRequired, match="needs your confirmation"):
        ai.download("chronos_bolt_tiny", opener=serving(payloads, calls))

    assert calls == [] and not ai.downloaded("chronos_bolt_tiny")


def test_download_fetches_every_file_from_the_pinned_revision(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {"config.json": b'{"a": 1}', "model.safetensors": b"w" * 5000}
    spec = fake_model(monkeypatch, payloads)
    calls: list[str] = []
    seen: list[tuple[int, int]] = []

    where = ai.download(
        "chronos_bolt_tiny",
        confirm=True,
        opener=serving(payloads, calls),
        progress=lambda have, total: seen.append((have, total)),
    )

    assert where == models_dir / "forecast" / "chronos_bolt_tiny"
    assert (where / "model.safetensors").read_bytes() == payloads["model.safetensors"]
    assert all(f"/resolve/{spec.revision}/" in url for url in calls) and len(calls) == 2
    assert seen[-1] == (spec.size, spec.size)
    assert ai.status("chronos_bolt_tiny").state == "ready"
    # already there: no connection
    ai.download("chronos_bolt_tiny", opener=lambda r, t: pytest.fail("connected again"))
    assert ai.delete("chronos_bolt_tiny") and not where.exists()


def test_a_wrong_digest_is_refused_and_deleted(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {"config.json": b"{}", "model.safetensors": b"w" * 100}
    fake_model(monkeypatch, payloads)
    served = {**payloads, "model.safetensors": b"x" * 100}

    with pytest.raises(ai.AiError, match="SHA-256"):
        ai.download("chronos_bolt_tiny", confirm=True, opener=serving(served, []), pause=0)

    assert not ai.downloaded("chronos_bolt_tiny")
    assert not list((models_dir / "forecast").rglob("*.part"))


def test_a_cancelled_download_keeps_its_part_and_resumes_with_range(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {"model.safetensors": b"w" * 3_000_000}
    fake_model(monkeypatch, payloads)
    stop = {"now": True}

    with pytest.raises(ai.AiError, match="cancelled"):
        ai.download(
            "chronos_bolt_tiny",
            confirm=True,
            opener=serving(payloads, []),
            cancelled=lambda: stop["now"],
        )
    (part,) = list((models_dir / "forecast").rglob("*.part"))
    kept = part.stat().st_size
    assert 0 < kept < 3_000_000

    ranges: list[str | None] = []

    def resuming(request, timeout):
        ranges.append(request.get_header("Range"))
        start = int(request.get_header("Range", "bytes=0-")[6:-1])
        response = FakeResponse(payloads["model.safetensors"][start:])
        response.status = 206
        return response

    stop["now"] = False
    ai.download("chronos_bolt_tiny", confirm=True, opener=resuming)

    assert ranges == [f"bytes={kept}-"] and ai.downloaded("chronos_bolt_tiny")


def test_never_download_blocks_the_network_and_says_where_to_copy_the_folder(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {"config.json": b"{}", "model.safetensors": b"w" * 10}
    fake_model(monkeypatch, payloads)
    monkeypatch.setenv("DATA_EXPLORER_OFFLINE", "1")

    found = ai.status("chronos_bolt_tiny")
    assert found.state == "offline" and str(ai.folder("chronos_bolt_tiny")) in (found.reason or "")
    with pytest.raises(ai.AiError, match="Never download"):
        ai.download("chronos_bolt_tiny", confirm=True, opener=lambda r, t: pytest.fail("network"))

    monkeypatch.delenv("DATA_EXPLORER_OFFLINE")
    ai.set_offline(True)
    assert ai.offline() and not ai.offline_forced()
    ai.set_offline(False)
    assert not ai.offline()


# --- Coded rules (B2, B4) ------------------------------------------------------------------------


def test_a_short_series_suggests_a_pretrained_model_with_its_reason(models_dir: Path) -> None:
    (found,) = ai.suggestions(18, 12, ["naive", "ets_auto"])

    assert found.kind == "short_series" and found.models == ["chronos_bolt_small"]
    assert found.periods == 18 and found.threshold == 30 and "Only 18 periods" in found.reason
    assert ai.suggestions(18, 12, ["chronos_bolt_tiny"]) == []
    assert ai.suggestions(60, 12, ["naive"]) == []  # five cycles: long enough
    assert ai.suggestions(40, 24, ["naive"])[0].threshold == 48  # under two cycles


def test_a_long_series_suggests_n_hits_and_a_short_one_refuses_it(models_dir: Path) -> None:
    (found,) = ai.suggestions(600, 24, ["naive"])

    assert found.kind == "long_series" and found.models == ["nhits"]
    assert neural.refusal(499) and "500 periods" in (neural.refusal(499) or "")
    assert neural.refusal(500) is None


def test_every_group_needs_many_groups_or_long_series_for_a_network() -> None:
    assert "10 groups" in (gr.batch_refusal("nhits", 4, 120) or "")


# --- Pretrained adapters, with a fake model (needs torch) ----------------------------------------


class FakeBolt:
    """Deciles around the last value, spread growing with the horizon."""

    def __init__(self) -> None:
        self.seen: list = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
        import torch

        self.seen.append((inputs, kwargs))
        rows = []
        for x in inputs:
            target = x["target"] if isinstance(x, dict) else x
            last = float(np.asarray(target)[-1])
            steps = np.arange(1, prediction_length + 1)[:, None]
            z = np.array(
                [{0.025: -2, 0.1: -1.3, 0.5: 0, 0.9: 1.3, 0.975: 2}[q] for q in quantile_levels]
            )
            rows.append(last + steps * z)
        found = torch.tensor(np.array(rows), dtype=torch.float32)
        if kwargs:  # Chronos-2: one (variates, h, q) tensor per series
            return [row.unsqueeze(0) for row in found], None
        return found, found[..., len(quantile_levels) // 2]


@pytest.fixture
def fake_pipelines(models_dir: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeBolt]:
    pytest.importorskip("torch")
    fakes = {key: FakeBolt() for key in ("chronos_bolt_tiny", "chronos_2")}
    monkeypatch.setattr(ai, "require", ai.spec_of)
    monkeypatch.setattr(ai, "pipeline", lambda key, loader=None: fakes[key])
    return fakes


def test_chronos_bolt_scores_on_rolling_origins_with_widened_95_intervals(
    fake_pipelines: dict[str, FakeBolt],
) -> None:
    result = fc.run_model("chronos_bolt_tiny", fc.ai_model("chronos_bolt_tiny"), SEASONAL, 12, 12)

    assert result.scores and result.scores.mase is not None and len(result.backtests) == 5
    assert "zero-shot on 120 periods" in (result.info or "")
    low80, low95 = np.array(result.low80), np.array(result.low95)
    mean = np.array(result.forecast)
    assert np.allclose(mean - low95, (mean - low80) * fc.Z95 / fc.Z80)
    # each origin only saw the history before it
    sizes = [len(inputs[0]) for inputs, _ in fake_pipelines["chronos_bolt_tiny"].seen]
    assert sizes == [*fc.origins(120, 12, 12), 120]


def test_chronos_2_gets_known_in_advance_drivers_as_covariates(
    fake_pipelines: dict[str, FakeBolt],
) -> None:
    x = RNG.normal(size=(120, 2))
    future = RNG.normal(size=(12, 2))

    prediction = fc.ai_model("chronos_2")(SEASONAL, 12, 12, x, future)

    (item,), kwargs = fake_pipelines["chronos_2"].seen[-1]
    assert set(item) == {"target", "past_covariates", "future_covariates"}
    assert len(item["past_covariates"]) == 2 and len(item["future_covariates"]["driver_0"]) == 12
    assert kwargs["cross_learning"] is False
    assert np.all(prediction.low95 <= prediction.low80) and "chronos_2" in fc.DRIVER_MODELS


def test_chronos_2_forecasts_every_group_in_one_batch_learning_across_them(
    fake_pipelines: dict[str, FakeBolt],
) -> None:
    from quantilo.timeseries import pretrained

    values = np.vstack([SEASONAL, SEASONAL * 2, SEASONAL + 50])

    found = pretrained.forecast_many("chronos_2", values, 12, 12)

    assert len(found) == 3 and all(f.scores and f.scores.mase is not None for f in found)
    inputs, kwargs = fake_pipelines["chronos_2"].seen[-1]
    assert len(inputs) == 3 and kwargs["cross_learning"] is True


# --- Real weights (only when already downloaded on this machine) ---------------------------------


@pytest.mark.skipif(
    not (REAL_MODELS / "forecast" / "chronos_bolt_tiny" / "model.safetensors").is_file(),
    reason="Chronos-Bolt Tiny weights are not on this machine (never downloaded by the tests)",
)
def test_chronos_bolt_tiny_with_its_real_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("chronos")
    monkeypatch.setenv("QUANTILO_MODELS_DIR", str(REAL_MODELS))
    ai.forget()

    result = fc.run_model("chronos_bolt_tiny", fc.ai_model("chronos_bolt_tiny"), SEASONAL, 12, 12)
    naive = fc.run_model("seasonal_naive", fc.seasonal_naive, SEASONAL, 12, 12)

    assert result.error is None and result.scores and result.scores.mase is not None
    assert naive.scores and naive.scores.mase is not None
    assert result.scores.mase < 2 * naive.scores.mase
    assert result.scores.coverage80 is not None and result.scores.coverage80 >= 0.5
    ai.forget()


# --- Trained networks (needs torch) --------------------------------------------------------------


@pytest.mark.parametrize("kind", ["nhits", "nbeats"])
def test_a_network_learns_a_long_seasonal_series(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(neural, "STEPS", 150)
    t = np.arange(560)
    y = 50 + 10 * np.sin(2 * np.pi * t / 24) + RNG.normal(0, 0.5, 560)

    result = fc.run_model(kind, neural.model(kind), y, 24, 24)
    flat = fc.run_model("naive", fc.naive, y, 24, 24)

    assert result.error is None and result.scores and flat.scores
    assert result.scores.mase is not None and flat.scores.mase is not None
    assert result.scores.mase < flat.scores.mase  # it learnt the cycle
    assert "conformal intervals" in (result.info or "")


def test_the_time_limit_shortens_the_training() -> None:
    assert neural.budget(fc.Tuning(time_limit=12)) == 2.0
    assert neural.budget(fc.DEFAULT_TUNING) == neural.FIT_SECONDS


# --- The forecast job and the every-group forecast -----------------------------------------------


def test_a_forecast_job_gives_the_reasons_and_the_suggestion_of_a_short_series(
    models_dir: Path,
) -> None:
    import time

    from quantilo.timeseries.forecasting import ForecastJobs

    jobs = ForecastJobs(1)
    values: list[float | None] = [float(v) for v in SEASONAL[:24]]
    periods = [f"{2020 + i // 12}-{i % 12 + 1:02d}" for i in range(24)]
    models = {"naive": fc.naive, "nhits": fc.ai_model("nhits")}

    job = jobs.start(("t",), values, periods, "month", 12, 6, models)
    while job.status == "running":
        time.sleep(0.02)

    assert job.result is not None
    rows = {r.key: r for r in job.result.results}
    assert rows["naive"].scores and rows["nhits"].scores is None and rows["nhits"].error
    (found,) = job.result.suggestions
    assert found.kind == "short_series" and found.periods == 24


def test_every_group_runs_a_pretrained_model_once_on_all_series(
    fake_pipelines: dict[str, FakeBolt],
) -> None:
    nodes = [gr.Node(f"s{i}", "store", [float(v) for v in SEASONAL * (i + 1)]) for i in range(3)]
    job = gr.GroupJob("t", "running", 0, 3)
    meta = {"group": "store", "groups": 3, "kept": 3, "pooled": 0, "limit": 200}

    found = gr.forecast_groups(
        job, nodes, None, ["naive", "chronos_bolt_tiny", "nhits"], 12, 12, 12,
        [str(i) for i in range(120)], [f"f{i}" for i in range(12)], meta, lambda: False,
        workers=1,
    )  # fmt: skip

    assert found is not None
    batches = {b.model: b for b in found.batch_models}
    assert batches["chronos_bolt_tiny"].error is None and batches["chronos_bolt_tiny"].series == 3
    assert "10 groups" in (batches["nhits"].error or "")
    # one call per rolling origin plus the final one, every series in each call
    calls = fake_pipelines["chronos_bolt_tiny"].seen
    assert len(calls) == len(fc.origins(120, 12, 12)) + 1 and all(len(c[0]) == 3 for c in calls)
    assert all(row.best != "chronos_bolt_tiny" or row.mase is not None for row in found.rows)
