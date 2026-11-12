"""Deep vision (114): optional extra, weights manager, budget, embeddings, linear probe and kNN.

No download here: the weights tests serve bytes from a fake opener, the networks are tiny
(hand-made modules, or timm's `test_vit` / `test_resnet` with random weights)."""

import dataclasses
import hashlib
import io
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from quantilo.vision import deep
from quantilo.vision.deep import probe as probes
from quantilo.vision.deep import weights as w

SRC = Path(__file__).resolve().parents[1] / "src"


# ---------------------------------------------------------------------------------------------
# Without the extra (R-VDP-01)


def test_the_package_imports_and_greys_out_without_the_extra(tmp_path):
    """torch, torchvision, timm and safetensors made unimportable in a fresh interpreter."""
    code = textwrap.dedent(
        """
        import sys
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {"torch", "torchvision", "timm", "safetensors"}:
                    raise ImportError(name)
                return None
        sys.meta_path.insert(0, Block())
        import importlib.util
        real = importlib.util.find_spec
        def find_spec(name, package=None):
            if name.split(".")[0] in {"torch", "torchvision", "timm", "safetensors"}:
                return None
            return real(name, package)
        importlib.util.find_spec = find_spec
        import numpy as np
        import quantilo, quantilo.vision
        from quantilo.vision import deep
        from quantilo.vision.deep import backbones, budget, probe, weights
        status = deep.status()
        assert not status.available and not status.installed, status
        assert 'pip install "quantilo[vision]"' in status.reason, status.reason
        assert set(status.missing) == {"torch", "torchvision", "timm", "safetensors"}
        assert all(not b.available and b.reason == status.reason for b in status.backbones)
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(0, 1, (30, 8)), rng.normal(3, 1, (30, 8))])
        y = ["a"] * 30 + ["b"] * 30
        report = deep.probe_embeddings(x, y, bootstrap=False)
        assert report.linear.report.accuracy > 0.9
        try:
            backbones.load("dinov2_s14")
        except RuntimeError as error:
            assert "R-VDP-01" in str(error)
        else:
            raise AssertionError("load without the extra")
        assert "torch" not in sys.modules
        print("ok")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SRC), "QUANTILO_VISION_WEIGHTS": str(tmp_path)},
        timeout=240,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("ok")


def test_status_reasons(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANTILO_VISION_WEIGHTS", str(tmp_path))
    monkeypatch.setattr(deep, "missing", lambda: ["timm"])
    status = deep.status()
    assert not status.available
    assert status.reason == (
        'Needs the optional extra vision (pip install "quantilo[vision]"); missing: timm '
        "(R-VDP-01)."
    )
    monkeypatch.setattr(deep, "missing", lambda: [])
    status = deep.status()
    assert status.available and status.reason is None
    first = status.backbone("dinov2_s14")
    assert first.level == "first" and first.dims == 768 and not first.downloaded
    assert first.reason == (
        "Download the weights first: DINOv2 ViT-S/14, 88 MB (Apache-2.0). "
        "Needs a connection once (R-VDP-02)."
    )
    assert "ImageNet" in status.backbone("resnet50").note
    monkeypatch.setattr(sys, "platform", "emscripten")
    status = deep.status()
    assert not status.available and status.reason == deep.BROWSER


# ---------------------------------------------------------------------------------------------
# Weights manager (R-VDP-02, R-VDP-03)


def test_manifest_is_pinned_https_and_safetensors():
    for spec in w.MANIFEST.values():
        assert spec.url.startswith("https://huggingface.co/timm/")
        assert f"/resolve/{spec.revision}/" in spec.url and len(spec.revision) == 40
        assert len(spec.sha256) == 64 and int(spec.sha256, 16) >= 0
        assert spec.filename.endswith(".safetensors") and spec.size > 10_000_000
    assert w.MANIFEST["dinov2_s14"].size_text == "88 MB"
    assert w.MANIFEST["dinov2_b14"].size_text == "346 MB"
    assert w.size_text(1_500_000_000) == "1.5 GB"


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200, fail_after: int | None = None):
        super().__init__(data)
        self.status = status
        self.fail_after = fail_after
        self.sent = 0

    def read(self, size: int | None = -1) -> bytes:
        if self.fail_after is not None and self.sent >= self.fail_after:
            raise ConnectionResetError("reset by peer")
        chunk = super().read(min(size or 1 << 20, 1 << 16))
        self.sent += len(chunk)
        return chunk


def fake_spec(data: bytes, **changes) -> w.WeightSpec:
    fields = {
        "key": "tiny",
        "title": "Tiny net",
        "repo": "timm/tiny",
        "revision": "0" * 40,
        "filename": "model.safetensors",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "licence": "MIT",
    }
    fields.update(changes)
    return w.WeightSpec(**fields)


@pytest.fixture
def weights_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANTILO_VISION_WEIGHTS", str(tmp_path / "weights"))
    w._VERIFIED.clear()
    return tmp_path / "weights"


def test_download_needs_confirmation_and_never_connects_without_it(weights_dir):
    data = b"x" * 1000
    calls = []
    with pytest.raises(w.ConfirmationRequired) as raised:
        w.download(fake_spec(data), opener=lambda r, t: calls.append(r) or FakeResponse(data))
    assert calls == []
    assert "(MIT)" in str(raised.value) and "R-VDP-02" in str(raised.value)
    assert not weights_dir.exists() or not any(weights_dir.iterdir())


def test_download_checks_size_and_sha256(weights_dir):
    data = bytes(range(256)) * 300
    spec = fake_spec(data)
    seen = []
    file = w.download(
        spec,
        confirm=True,
        opener=lambda r, t: seen.append(r) or FakeResponse(data),
        progress=lambda have, total: None,
    )
    assert file.read_bytes() == data and file.name == spec.local_name
    assert seen[0].full_url == spec.url and seen[0].get_header("User-agent") == w.USER_AGENT
    assert w.downloaded(spec) and w.verify(spec) == file
    # already there: no second connection, even without confirmation
    assert w.download(spec, opener=lambda r, t: pytest.fail("connected again")) == file


def test_download_refuses_a_wrong_digest_and_deletes_the_file(weights_dir):
    data = b"weights" * 100
    spec = fake_spec(data, sha256="0" * 64)
    with pytest.raises(w.WeightsError, match="SHA-256"):
        w.download(spec, confirm=True, opener=lambda r, t: FakeResponse(data), pause=0)
    assert not w.path(spec).exists() and not list(weights_dir.glob("*.part"))


def test_download_stops_when_the_server_sends_too_much(weights_dir):
    data = b"a" * 5000
    spec = fake_spec(data[:100])
    with pytest.raises(w.WeightsError, match="more than"):
        w.download(spec, confirm=True, opener=lambda r, t: FakeResponse(data), pause=0)
    assert not list(weights_dir.glob("*"))


def test_download_resumes_with_range_after_a_reset(weights_dir):
    data = np.random.default_rng(0).bytes(300_000)
    spec = fake_spec(data)
    ranges = []

    def opener(request, timeout):
        asked = request.get_header("Range")
        ranges.append(asked)
        if asked is None:
            return FakeResponse(data, fail_after=100_000)
        start = int(asked.removeprefix("bytes=").rstrip("-"))
        return FakeResponse(data[start:], status=206)

    file = w.download(spec, confirm=True, opener=opener, pause=0)
    assert file.read_bytes() == data
    assert ranges[0] is None and ranges[1].startswith("bytes=") and ranges[1] != "bytes=0-"


class PlainHttp(w.WeightSpec):
    @property
    def url(self) -> str:
        return "http://example.com/model.safetensors"


def test_download_refuses_plain_http(weights_dir):
    plain = PlainHttp(**dataclasses.asdict(fake_spec(b"z")))
    with pytest.raises(w.WeightsError, match="HTTPS"):
        w.download(plain, confirm=True, opener=lambda r, t: pytest.fail("connected"))


def test_a_modified_file_is_refused_on_load(weights_dir):
    data = b"good" * 1000
    spec = fake_spec(data)
    w.download(spec, confirm=True, opener=lambda r, t: FakeResponse(data))
    file = w.path(spec)
    file.write_bytes(b"evil" * 1000)  # same size, other content
    w._VERIFIED.clear()
    with pytest.raises(w.WeightsError, match="does not match"):
        w.verify(spec)
    file.unlink()
    with pytest.raises(w.WeightsMissing, match="Download the weights first"):
        w.verify(spec)


def test_weights_folder_follows_the_app_workspace(monkeypatch, tmp_path):
    monkeypatch.delenv("QUANTILO_VISION_WEIGHTS", raising=False)
    monkeypatch.setenv("QUANTILO_EMBEDDINGS_CACHE", str(tmp_path / "cache" / "embeddings"))
    assert w.root() == tmp_path / "cache" / "models" / "vision"


# ---------------------------------------------------------------------------------------------
# Probes (no torch): R-VDP-06 to R-VDP-08


def clusters(n_per: int = 40, dims: int = 16, seed: int = 0, spread: float = 1.0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 4, (3, dims))
    x = np.vstack([c + rng.normal(0, spread, (n_per, dims)) for c in centres])
    y = np.repeat(["angular", "rust", "healthy"], n_per)
    return x, y


def test_probe_on_separable_clusters_uses_the_producer_split():
    x, y = clusters()
    splits = np.where(np.arange(len(y)) % 4 == 0, "test", "train")
    splits[1::8] = "val"
    report = probes.probe_embeddings(x, y, splits, backbone="toy", bootstrap=False)
    assert "producer's split" in report.split.reason and "train + val" in report.split.reason
    assert len(report.split.test) == 30
    assert report.linear.report.accuracy == pytest.approx(1.0)
    assert report.knn.report.accuracy == pytest.approx(1.0)
    assert report.linear.report.baseline.accuracy < 0.5
    assert report.linear.setting.startswith("C = ") and "R-VDP-07" in report.linear.notes[0]
    assert report.sentence.startswith("Linear probe on toy: accuracy 1.000")
    assert "majority baseline" in report.sentence


def test_probe_report_carries_intervals_and_metrics_of_111():
    x, y = clusters(spread=4.0, seed=1)
    report = probes.probe_embeddings(x, y, bootstrap=True)
    linear = report.linear.report
    assert {"accuracy", "balanced_accuracy"} <= set(linear.intervals)
    assert linear.log_loss is not None and linear.calibration is not None
    assert 0.4 < linear.accuracy <= 1.0
    assert "[" in report.sentence


def test_split_without_producer_keeps_exact_copies_together():
    _, y = clusters(n_per=30)
    groups = [f"g{i // 3}" for i in range(len(y))]  # three copies of each image
    split = probes.choose_split(y, None, groups, seed=0)
    assert "No producer's split" in split.reason and "pixel_sha" in split.reason
    train_groups = {groups[i] for i in split.train}
    assert not train_groups & {groups[i] for i in split.test}
    assert 0.15 <= len(split.test) / len(y) <= 0.25
    assert sorted(split.train + split.test) == list(range(len(y)))


def test_knn_matches_a_brute_force_vote():
    rng = np.random.default_rng(3)
    xtr = rng.normal(size=(25, 6))
    ytr = np.array(list("ab") * 12 + ["a"])
    xte = rng.normal(size=(7, 6))
    yte = np.array(list("abababa"))
    result = probes.knn_probe(
        xtr, ytr, xte, yte, k=5, train_rows=list(range(100, 125)), bootstrap=False
    )
    ntr = xtr / np.linalg.norm(xtr, axis=1, keepdims=True)
    nte = xte / np.linalg.norm(xte, axis=1, keepdims=True)
    sims = nte @ ntr.T
    for i in range(len(xte)):
        top = np.argsort(-sims[i])[:5]
        votes = {c: np.exp(sims[i, top][ytr[top] == c] / 0.07).sum() for c in "ab"}
        assert result.predictions[i] == max(votes, key=lambda c: votes[c])
        assert result.neighbours[i] == [100 + int(t) for t in top]


def test_probe_leaves_out_rows_without_vector_and_refuses_lonely_classes():
    x, y = clusters(n_per=20)
    x[:5] = np.nan
    report = probes.probe_embeddings(x, y, bootstrap=False)
    used = set(report.split.train) | set(report.split.test)
    assert not used & set(range(5))
    with pytest.raises(ValueError, match="Too few training images"):
        probes.linear_probe(x[5:], ["a"] * (len(x) - 6) + ["b"], x[5:], y[5:])
