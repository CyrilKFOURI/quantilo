"""Deep anomaly detectors with torch (115, 116): PatchCore, PaDiM, autoencoder and VAE.

Skipped without torch. No download: timm's `test_resnet` with random weights stands for
Wide ResNet-50-2, images are synthetic textures with a painted defect, everything on the CPU."""

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from quantilo.vision.deep import anomaly  # noqa: E402
from quantilo.vision.deep import budget as budgets  # noqa: E402
from quantilo.vision.deep import weights as w  # noqa: E402
from quantilo.vision.deep.autoencoder import ConvAutoencoder, error_map, ssim_map  # noqa: E402
from quantilo.vision.deep.backbones import Backbone, BackboneSpec  # noqa: E402
from quantilo.vision.deep.padim import PaDiM  # noqa: E402
from quantilo.vision.deep.patchcore import (  # noqa: E402
    PatchCore,
    PatchCoreResult,
    greedy_coreset,
    nearest,
    patch_embedding,
)
from tests.vision_fixtures import encode, image_folder  # noqa: E402

SIDE = 64
SIZE = (80, 64)  # (width, height) of the synthetic images: maps come back at this size
BOX = (32, 24, 48, 40)  # the painted defect, in the images' own pixels


def tiny() -> Backbone:
    torch.manual_seed(0)
    module = timm.create_model("test_resnet", pretrained=False, num_classes=0)
    spec = BackboneSpec(
        "tiny_resnet", "Tiny ResNet", "test_resnet", "cnn", 0, "test", "advanced", 4, 4, side=SIDE
    )
    return Backbone(spec, module)


def texture(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    h, w = SIZE[1], SIZE[0]
    yy, xx = np.mgrid[0:h, 0:w]
    base = 120 + 40 * np.sin(xx / 3.0) + 40 * np.cos(yy / 4.0)
    pixels = base[..., None] + rng.normal(0, 4, (h, w, 3))
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def defect(seed: int) -> Image.Image:
    pixels = np.asarray(texture(seed)).copy()
    x0, y0, x1, y1 = BOX
    pixels[y0:y1, x0:x1] = (255, 20, 20)
    return Image.fromarray(pixels, "RGB")


def mask() -> Image.Image:
    m = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    x0, y0, x1, y1 = BOX
    m[y0:y1, x0:x1] = 255
    return Image.fromarray(m, "L")


@pytest.fixture
def weights_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANTILO_VISION_WEIGHTS", str(tmp_path / "weights"))
    w._VERIFIED.clear()
    return tmp_path / "weights"


NORMALS = [texture(i) for i in range(8)]
TEST = [texture(100), texture(101), defect(200), defect(201)]
TRUTH = np.array([False, False, True, True])


# ---------------------------------------------------------------------------------------------
# Pieces


def test_greedy_coreset_spreads_its_points():
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 10, (5, 4))
    points = np.concatenate([c + rng.normal(0, 0.1, (200, 4)) for c in centres])
    z = torch.from_numpy(points).float()
    chosen = greedy_coreset(z, 5, seed=0)
    assert len(set((chosen // 200).tolist())) == 5  # one point in each cluster
    assert list(greedy_coreset(z[:3], 10)) == [0, 1, 2]
    assert (greedy_coreset(z, 5, seed=0) == chosen).all()


def test_nearest_by_blocks_matches_brute_force():
    g = torch.Generator().manual_seed(0)
    query = torch.randn(37, 6, generator=g)
    bank = torch.randn(53, 6, generator=g).half()
    dist, where = nearest(query, bank, block=8)
    full = torch.cdist(query, bank.float())
    assert torch.allclose(dist, full.min(dim=1).values, atol=1e-5)
    assert (where == full.argmin(dim=1)).all()


def test_patch_embedding_brings_stage_three_to_the_grid_of_stage_two():
    maps = [torch.randn(2, 4, 8, 8), torch.randn(2, 6, 4, 4)]
    emb = patch_embedding(maps, 3)
    assert emb.shape == (2, 10, 8, 8)


def test_ssim_of_an_image_with_itself_is_one():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2, 3, 32, 32, generator=g)
    assert torch.allclose(ssim_map(x, x), torch.ones_like(x), atol=1e-4)
    assert error_map(x, x).abs().max() < 1e-4 and error_map(x, x, "l2").max() == 0
    assert (error_map(x, 1 - x) > 0.05).float().mean() > 0.5


# ---------------------------------------------------------------------------------------------
# PatchCore (R-VAD-04 to R-VAD-06)


def test_patchcore_finds_the_defect_and_explains_it_by_a_normal_patch():
    model = PatchCore(tiny(), side=SIDE, device="cpu", coreset=0.25).fit(NORMALS)
    assert 0 < len(model.bank) <= model.patches and model.bank.dtype == torch.float16
    assert any("R-VAD-05" in n for n in model.notes)
    result = model.score(TEST)
    assert result.ok.all() and result.maps.shape == (4, SIDE, SIDE)
    assert result.scores[TRUTH].min() > result.scores[~TRUTH].max()
    full = result.full_map(2)
    assert full.shape == (SIZE[1], SIZE[0])
    y, x = np.unravel_index(int(np.argmax(full)), full.shape)
    x0, y0, x1, y1 = BOX
    assert x0 - 12 <= x <= x1 + 12 and y0 - 12 <= y <= y1 + 12
    top = result.top[2]
    assert top is not None and 0 <= top.normal_image < len(NORMALS)
    assert top.distance == pytest.approx(result.scores[2])
    bx0, by0, bx1, by1 = top.box
    assert bx0 < x1 and bx1 > x0 and by0 < y1 and by1 > y0  # the box covers the defect
    assert "R-VAD-06" in top.sentence
    elsewhere = result.explain(2, 2, 2)  # a corner of the same image: an ordinary area
    assert elsewhere.distance < top.distance
    part = result.subset([2, 0])
    assert part.top[0] is not None and part.top[0].image == 0 and part.scores[0] == result.scores[2]


def test_patchcore_says_when_it_caps_the_bank_and_the_patches():
    model = PatchCore(tiny(), side=SIDE, device="cpu", bank_mb=0.004, store_mb=0.05).fit(NORMALS)
    text = " ".join(model.notes)
    assert "Memory bank capped" in text and "random patches kept per image" in text
    assert len(model.bank) * model.bank.shape[1] * 2 <= 0.004 * 2**20


def test_patchcore_leaves_out_what_it_cannot_decode():
    model = PatchCore(tiny(), side=SIDE, device="cpu").fit([*NORMALS[:4], b"nope"])
    assert 4 in model.fit_failures
    result = model.score([TEST[0], b"broken"])
    assert result.ok.tolist() == [True, False] and np.isnan(result.scores[1])
    with pytest.raises(ValueError, match="not scored"):
        result.explain(1)


def test_patchcore_never_downloads_its_weights(weights_dir):
    with pytest.raises(w.WeightsMissing, match="Download the weights first"):
        PatchCore("resnet18", device="cpu").fit(NORMALS)
    assert not weights_dir.exists() or not any(weights_dir.iterdir())


def test_patchcore_asks_before_a_long_run(monkeypatch):
    monkeypatch.setattr(budgets, "LONG_RUN_SECONDS", 0)
    with pytest.raises(budgets.LongRun):
        PatchCore(tiny(), side=SIDE, device="cpu").fit(NORMALS)
    PatchCore(tiny(), side=SIDE, device="cpu", confirm_long=True).fit(NORMALS[:2])


# ---------------------------------------------------------------------------------------------
# PaDiM (R-VAD-07)


def test_padim_scores_the_defects_above_the_normals():
    model = PaDiM(tiny(), side=SIDE, device="cpu", dims=20).fit(NORMALS)
    assert len(model.index) == 20 and model.inverse.shape[1:] == (20, 20)
    result = model.score(TEST)
    assert result.scores[TRUTH].min() > result.scores[~TRUTH].max()
    assert "PaDiM" in result.title and result.maps.shape == (4, SIDE, SIDE)
    capped = PaDiM(tiny(), side=SIDE, device="cpu", covariance_mb=0.05).fit(NORMALS[:3])
    assert any("drawn instead" in n for n in capped.notes)
    with pytest.raises(ValueError, match="at least 2"):
        PaDiM(tiny(), side=SIDE, device="cpu").fit(NORMALS[:1])


# ---------------------------------------------------------------------------------------------
# Autoencoder and VAE (R-VAD-08)


def test_autoencoder_reconstructs_and_shows_before_and_after():
    model = ConvAutoencoder(side=32, epochs=4, batch=4, device="cpu", learning_rate=1e-3)
    model.fit(NORMALS)
    assert len(model.losses) == 4 and model.losses[-1] < model.losses[0]
    result = model.score(TEST)
    assert result.ok.all() and result.inputs.shape == (4, 32, 32, 3)
    assert result.reconstructions.dtype == np.uint8 and result.maps.shape == (4, 32, 32)
    view = result.before_after(2)
    assert view.input.size == view.reconstruction.size == view.error.size == (32, 32)
    assert view.error.mode == "L" and "R-VAD-08" in view.sentence
    assert all(b.startswith(b"\x89PNG") for b in view.pngs())
    assert result.subset([3]).inputs.shape == (1, 32, 32, 3)
    with pytest.raises(ValueError, match="divisible by 32"):
        ConvAutoencoder(side=48)


def test_vae_trains_and_reconstructs_from_its_mean():
    model = ConvAutoencoder(variational=True, side=32, epochs=2, batch=4, device="cpu")
    result = model.fit(NORMALS[:4]).score(TEST[:2])
    assert model.method == "vae" and result.title == "Variational autoencoder"
    again = model.score(TEST[:2])
    assert np.array_equal(result.reconstructions, again.reconstructions)  # mean, no sampling


def test_autoencoder_asks_before_a_long_training(monkeypatch):
    monkeypatch.setattr(budgets, "LONG_RUN_SECONDS", 0)
    with pytest.raises(budgets.LongRun):
        ConvAutoencoder(side=32, epochs=3, batch=4, device="cpu").fit(NORMALS[:4])


# ---------------------------------------------------------------------------------------------
# An imported image set (R-VAD-01, R-VAD-02, metrics of 111)


def test_anomaly_image_set_scores_the_test_images_with_pixel_metrics(tmp_path):
    from quantilo.vision.ingest import image_set

    layout: dict[str, Image.Image | bytes] = {
        f"train/good/{i:03d}.png": texture(i) for i in range(8)
    }
    layout.update({f"test/good/{i:03d}.png": texture(100 + i) for i in range(3)})
    layout.update({f"test/broken/{i:03d}.png": defect(200 + i) for i in range(3)})
    layout.update({f"ground_truth/broken/{i:03d}_mask.png": encode(mask()) for i in range(3)})
    root = image_folder(tmp_path / "set", layout)
    images = image_set(root, workspace=tmp_path / "ws", thumbnails=False, quality=False)
    model = PatchCore(tiny(), side=SIDE, device="cpu")
    run = anomaly.anomaly_image_set(images, model=model, bootstrap=False)
    assert run.split.code == "R-VAD-01" and len(run.split.fit) == 8 and len(run.rows) == 6
    assert run.report is not None and run.report.auroc == 1.0
    assert run.report.pixel is not None and run.report.pixel.auroc > 0.8
    assert run.threshold is not None and run.threshold.code == "R-VTH-01"
    assert set(run.report.auroc_by_type) == {"broken"}
    assert isinstance(run.result, PatchCoreResult)
    top = run.result.top[int(np.nanargmax(run.result.scores))]
    assert top is not None and top.normal_row in run.split.fit
    assert "image AUROC 1.000" in run.sentence
