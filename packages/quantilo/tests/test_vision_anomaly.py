"""Deep anomaly detectors (115, 116) without torch: split rules, status, manifest, imports.

The detectors themselves are tested in `test_vision_anomaly_torch.py` (tiny networks)."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from quantilo.vision import deep
from quantilo.vision.deep import anomaly
from quantilo.vision.deep import weights as w

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def weights_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANTILO_VISION_WEIGHTS", str(tmp_path / "weights"))
    w._VERIFIED.clear()
    return tmp_path / "weights"


def test_wide_resnet_is_in_the_manifest_with_its_fingerprint():
    spec = w.MANIFEST["wide_resnet50_2"]
    assert spec.url.startswith("https://huggingface.co/timm/wide_resnet50_2.tv_in1k/resolve/")
    assert len(spec.revision) == 40 and len(spec.sha256) == 64
    assert spec.size_text == "276 MB" and spec.licence == "BSD-3-Clause"
    assert "ImageNet" in spec.note


def test_detectors_are_greyed_until_their_weights_are_there(weights_dir):
    status = deep.status()
    if not status.available:
        pytest.skip("the extra is absent: covered by the subprocess test")
    patchcore = status.detector("patchcore")
    assert not patchcore.available and "Wide ResNet-50-2" in (patchcore.reason or "")
    assert "R-VDP-02" in (patchcore.reason or "")
    assert not status.detector("padim").available
    assert status.detector("autoencoder").available and status.detector("vae").available
    # ResNet-18 on disk (right size): PatchCore runs on the CPU, PaDiM runs
    weights_dir.mkdir(parents=True)
    spec = w.MANIFEST["resnet18"]
    with (weights_dir / spec.local_name).open("wb") as handle:
        handle.truncate(spec.size)
    status = deep.status()
    assert status.detector("patchcore").available and status.detector("padim").available


def test_the_detectors_import_and_grey_out_without_the_extra(tmp_path):
    code = textwrap.dedent(
        """
        import sys
        BLOCKED = {"torch", "torchvision", "timm", "safetensors"}
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(name)
                return None
        sys.meta_path.insert(0, Block())
        import importlib.util
        real = importlib.util.find_spec
        def find_spec(name, package=None):
            if name.split(".")[0] in BLOCKED:
                return None
            return real(name, package)
        importlib.util.find_spec = find_spec
        from quantilo.vision import deep
        from quantilo.vision.deep import anomaly, autoencoder, padim, patchcore
        status = deep.status()
        assert all(not d.available and d.reason == status.reason for d in status.detectors)
        assert deep.PatchCore is patchcore.PatchCore
        assert "torch" not in sys.modules
        print("ok")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SRC), "QUANTILO_VISION_WEIGHTS": str(tmp_path)},
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


# ---------------------------------------------------------------------------------------------
# R-VAD-01, R-VAD-02


def test_producer_split_learns_from_normal_training_images_only():
    splits = ["train"] * 6 + ["test"] * 4
    normal = [True] * 5 + [False] + [True, True, False, False]
    split = anomaly.split_rows(splits, normal, [True] * 10)
    assert split.code == "R-VAD-01" and "producer" in split.reason
    assert split.fit == [0, 1, 2, 3, 4] and split.test == [6, 7, 8, 9]
    assert split.holdout == []  # defects in the test: R-VTH-01, nothing held out
    assert any("left out of the model" in n for n in split.notes)


def test_without_a_split_a_fifth_of_the_normals_join_the_defects():
    normal = [True] * 20 + [False] * 5
    split = anomaly.split_rows(None, normal, [True] * 25)
    assert len(split.fit) == 16 and len(split.test) == 9
    assert set(range(20, 25)) <= set(split.test)
    assert not set(split.fit) & set(split.test)
    again = anomaly.split_rows(None, normal, [True] * 25)
    assert again.fit == split.fit  # seed 0


def test_no_defect_to_score_holds_normals_out_for_the_threshold():
    splits = ["train"] * 20 + ["test"] * 5
    split = anomaly.split_rows(splits, [True] * 25, [True] * 25)
    assert len(split.holdout) == 2 and len(split.fit) == 18
    assert not set(split.holdout) & set(split.fit)
    assert any("R-VAD-02" in n for n in split.notes)


def test_without_any_normal_flag_every_image_is_assumed_normal():
    split = anomaly.split_rows(None, [None] * 6, [True] * 5 + [False])
    assert split.code == "R-VML-03" and "assumes most images are normal" in split.reason
    assert split.fit == split.test == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="No normal image"):
        anomaly.split_rows(["train", "test"], [False, True], [True, True])


def test_images_and_masks_are_brought_to_the_side():
    from PIL import Image

    picture = Image.fromarray(np.full((30, 50, 3), 200, np.uint8))
    decoded = anomaly.decode_rgb(picture, 64)
    assert not isinstance(decoded, str)
    pixels, size = decoded
    assert pixels.shape == (64, 64, 3) and size == (50, 30)
    mask = np.zeros((30, 50), np.uint8)
    mask[10:20, 20:30] = 255
    resized = anomaly.load_mask(Image.fromarray(mask), 64)
    assert resized.dtype == bool and resized.shape == (64, 64) and 0 < resized.mean() < 0.2
    assert not anomaly.load_mask(None, 8).any()
    assert anomaly.decode_rgb(b"not an image", 64) == anomaly.decode_rgb(b"not an image", 64)
    assert isinstance(anomaly.decode_rgb(b"not an image", 64), str)
    peak = np.zeros((16, 16), np.float32)
    peak[8, 8] = 1
    blurred = anomaly.smooth(peak, 2.0)
    assert blurred.argmax() == peak.argmax() and blurred.max() < 1
    assert anomaly.resize_map(peak, (32, 24)).shape == (24, 32)
    assert anomaly.scale_box((0, 0, 32, 64), 64, (128, 256)) == (0, 0, 64, 256)
