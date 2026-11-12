"""Deep vision (114) with torch: budget, tiny backbones, embeddings, loading into timm.

Skipped without torch. No download: tiny hand-made modules, or timm's `test_vit` and
`test_resnet` with random weights served by a fake opener."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quantilo.vision import deep  # noqa: E402
from quantilo.vision.deep import backbones as bb  # noqa: E402
from quantilo.vision.deep import budget as budgets  # noqa: E402
from quantilo.vision.deep import embed as embedding  # noqa: E402
from quantilo.vision.deep import weights as w  # noqa: E402
from tests.test_vision_deep import FakeResponse, fake_spec  # noqa: E402
from tests.vision_fixtures import encode, image_folder, scene  # noqa: E402


@pytest.fixture
def weights_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANTILO_VISION_WEIGHTS", str(tmp_path / "weights"))
    w._VERIFIED.clear()
    return tmp_path / "weights"


class TinyViT(torch.nn.Module):
    """Patches of 8 px averaged into 4 numbers per token, one CLS token, like timm's ViTs."""

    num_prefix_tokens = 1

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.proj = torch.nn.Conv2d(3, 4, 8, stride=8)
        self.cls = torch.nn.Parameter(torch.randn(1, 1, 4))

    def forward_features(self, x):
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        return torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)


class TinyCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 6, 3, stride=4)

    def forward_features(self, x):
        return self.conv(x)


def tiny_spec(kind: bb.Kind = "vit", dims: int = 8, side: int = 32) -> bb.BackboneSpec:
    return bb.BackboneSpec(
        f"tiny_{kind}", f"Tiny {kind}", "none", kind, dims, "test", "advanced", 4, 4, side=side
    )


def tiny(kind: bb.Kind = "vit") -> bb.Backbone:
    return bb.Backbone(
        tiny_spec(kind, 8 if kind == "vit" else 6), TinyViT() if kind == "vit" else TinyCNN()
    )


def test_scaled_batch_and_cpu_plan():
    assert budgets.scaled_batch(64, None) == 64
    assert budgets.scaled_batch(64, 6144) == 64 and budgets.scaled_batch(64, 8192) == 64
    assert budgets.scaled_batch(64, 4096) == 32  # 42.7 -> 32
    assert budgets.scaled_batch(64, 2048) == 16 and budgets.scaled_batch(1, 1024) == 1
    spec = bb.BACKBONES["dinov2_s14"]
    plan = budgets.plan(spec, "cpu")
    assert plan.device == "cpu" and plan.batch == 16 and not plan.half and (plan.threads or 0) >= 1
    assert "R-VDP-04" in plan.sentence
    assert budgets.plan(spec, "auto", allow_gpu=False).device == "cpu"


def test_out_of_memory_halves_the_batch_then_gives_up_at_one():
    plan = budgets.Budget("cuda", "Fake GPU", 8, True)
    sizes = []

    def fit(chunk):
        sizes.append(len(chunk))
        if len(chunk) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return len(chunk)

    out = budgets.run_in_batches(fit, list(range(7)), plan)
    assert sum(out) == 7 and plan.batch == 2
    assert sizes == [7, 4, 2, 2, 2, 1]
    assert sum("halved" in note for note in plan.notes) == 2

    def never(chunk):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(budgets.GpuExhausted):
        budgets.run_in_batches(never, [1, 2], budgets.Budget("cuda", "Fake", 1, True))

    def other(chunk):
        raise RuntimeError("other")

    with pytest.raises(RuntimeError, match="other"):
        budgets.run_in_batches(other, [1], plan)


def test_estimate_and_long_run():
    e = budgets.estimate(0.5, 64, 1200, "NVIDIA GeForce RTX 4050 Laptop GPU")
    assert e.sentence == "≈ 9 s for 1,200 images on NVIDIA GeForce RTX 4050 Laptop GPU"
    assert not e.needs_confirmation
    long = budgets.estimate(60, 16, 50_000, "CPU")
    assert long.needs_confirmation and long.sentence.startswith("≈ 52.1 h")
    assert "R-VDP-05" in str(budgets.LongRun(long))
    assert budgets.duration_text(180) == "3 min"


def test_preprocess_resizes_crops_and_normalises():
    spec = bb.BACKBONES["resnet18"]
    image = scene(size=(400, 300))
    x = embedding.preprocess(image, spec)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    grey = embedding.preprocess(scene().convert("L").convert("RGB"), bb.BACKBONES["dinov2_s14"])
    assert grey.shape == (3, 224, 224)
    from PIL import Image

    white = embedding.preprocess(Image.new("RGB", (50, 80), (255, 255, 255)), spec)
    assert np.allclose(white[0], (1 - 0.485) / 0.229, atol=1e-5)


def test_tiny_backbones_pool_as_their_kind():
    vit, cnn = tiny("vit"), tiny("cnn")
    x = torch.randn(3, 3, 32, 32)
    tokens = vit.module.forward_features(x)
    expected = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=1)
    assert torch.allclose(vit.features(x), expected)
    assert vit.features(x).shape == (3, 8) and cnn.features(x).shape == (3, 6)
    assert not any(p.requires_grad for p in vit.module.parameters())
    assert vit.model_key.startswith("tiny_vit|custom|32px")


def test_embed_images_in_order_with_failures_and_cpu():
    images = [scene(i) for i in range(5)]
    sources = [images[0], encode(images[1]), b"not an image", encode(images[3], "JPEG"), images[4]]
    seen = []
    result = embedding.embed_images(
        sources, tiny("vit"), device="cpu", batch=2, progress=lambda d, n: seen.append(d)
    )
    assert result.vectors.shape == (5, 8) and result.ok.tolist() == [1, 1, 0, 1, 1]
    assert np.isnan(result.vectors[2]).all() and 2 in result.failures
    assert result.failures[2].startswith("R-VQ-01")
    assert seen[-1] == 5 and result.device == "CPU" and result.estimate is not None
    again = embedding.embed_images([images[0]], tiny("vit"), device="cpu")
    assert np.allclose(again.vectors[0], result.vectors[0], atol=1e-5)
    assert "CPU" in result.notes[0] and "1 left out" in result.sentence


def test_long_runs_ask_first(monkeypatch):
    monkeypatch.setattr(budgets, "LONG_RUN_SECONDS", -1)
    with pytest.raises(budgets.LongRun):
        embedding.embed_images([scene(0), scene(1)], tiny("cnn"), device="cpu", batch=1)
    out = embedding.embed_images([scene(0)], tiny("cnn"), device="cpu", confirm_long=True)
    assert out.ok.all()


def test_gpu_exhausted_finishes_on_the_cpu(monkeypatch):
    real = budgets.run_in_batches
    calls = {"n": 0}

    def flaky(fn, items, plan):
        calls["n"] += 1
        if calls["n"] == 1:
            raise budgets.GpuExhausted("full")
        return real(fn, items, plan)

    monkeypatch.setattr(budgets, "run_in_batches", flaky)
    out = embedding.embed_images([scene(0), scene(1)], tiny("vit"), device="cpu")
    assert out.ok.all() and any("finished on the CPU" in note for note in out.notes)


def test_cancel_stops_the_run():
    with pytest.raises(embedding.Cancelled):
        embedding.embed_images([scene(0)], tiny("vit"), device="cpu", cancel=lambda: True)


def test_embed_image_set_aligns_with_the_table(tmp_path):
    folder = image_folder(
        tmp_path / "set",
        {
            "cats/a.png": scene(1),
            "cats/b.png": scene(2),
            "dogs/c.png": scene(3),
            "dogs/d.png": scene(4),
            "dogs/broken.png": b"\x89PNG\r\n\x1a\n broken",
        },
    )
    from quantilo.vision import image_set

    # two images a class: R-VQ-14 (class too small) would leave them all out
    images = image_set(
        folder, workspace=tmp_path / "ws", dataset_id="pets", thumbnails=False, ignore_rules=("R-VQ-14",)
    )
    result = embedding.embed_image_set(images, tiny("cnn"), device="cpu")
    assert result.vectors.shape == (images.n_images, 6)
    decoded = images.table["decode_ok"].to_pylist()
    excluded = images.table["excluded"].to_pylist()  # 105: quality rules may leave images out
    assert result.ok.tolist() == [ok and not out for ok, out in zip(decoded, excluded, strict=True)]
    assert result.image_ids == images.table["image_id"].to_pylist()
    broken = decoded.index(False)
    assert "R-VQ-01" in result.failures[broken]
    coordinates = embedding.map_coordinates(result)
    assert np.isnan(coordinates.x[broken]) and np.isfinite(coordinates.x[result.ok]).all()
    assert "% of the variance" in coordinates.sentence


def test_load_never_downloads(monkeypatch, weights_dir):
    if deep.missing():
        pytest.skip("the vision extra is not installed")
    with pytest.raises(w.WeightsMissing, match="88 MB"):
        bb.load("dinov2_s14")
    assert not weights_dir.exists() or not any(weights_dir.iterdir())


@pytest.mark.parametrize(
    ("architecture", "kind", "dims"), [("test_vit", "vit", 128), ("test_resnet", "cnn", 96)]
)
def test_load_fills_a_timm_network_from_a_verified_safetensors_file(
    monkeypatch, weights_dir, architecture, kind, dims
):
    """A random 'pretrained' checkpoint (with its ImageNet head, position embedding at 160 px)
    loads into the network built at 224 px: head dropped, positions resampled, strict check."""
    if deep.missing():
        pytest.skip("the vision extra is not installed")
    import timm
    from safetensors.torch import save_file

    source = timm.create_model(architecture, pretrained=False)
    state = {k: v.contiguous() for k, v in source.state_dict().items()}
    blob = weights_dir / "staging.safetensors"
    blob.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(blob))
    data = blob.read_bytes()
    spec = fake_spec(data, key=f"fake_{kind}")
    monkeypatch.setitem(w.MANIFEST, spec.key, spec)
    w.download(spec, confirm=True, opener=lambda r, t: FakeResponse(data))
    backbone_spec = bb.BackboneSpec(
        f"fake_{kind}", "Fake", architecture, kind, dims, "test", "advanced", 4, 4, spec.key
    )
    net = bb.load(backbone_spec)
    first = "patch_embed.proj.weight" if kind == "vit" else "conv1.0.weight"
    if first not in state:
        first = next(k for k in state if k.endswith("weight"))
    assert torch.equal(net.module.state_dict()[first], state[first])
    out = net.features(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, dims)
    assert net.model_key.startswith(f"fake_{kind}|{spec.sha256[:12]}")
