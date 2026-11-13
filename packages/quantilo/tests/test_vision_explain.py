"""Post-hoc maps without torch (117): occlusion, rollout arithmetic, PNG maps, the linear head."""

import io
import math

import numpy as np
import pytest
from PIL import Image

from quantilo.vision import explain as ex
from quantilo.vision.deep.backbones import BACKBONES


def red_square(size: tuple[int, int] = (64, 48)) -> Image.Image:
    pixels = np.full((size[1], size[0], 3), 90, dtype=np.uint8)
    pixels[8:16, 40:48] = (255, 0, 0)  # the evidence: top right
    return Image.fromarray(pixels, "RGB")


def redness(images):
    """A 'model' whose class 1 probability is the share of strongly red pixels, times 10."""
    out = []
    for image in images:
        a = np.asarray(image, dtype=np.float64)
        red = ((a[..., 0] > 200) & (a[..., 1] < 60)).mean()
        p = min(1.0, red * 10)
        out.append([1 - p, p])
    return np.array(out)


def test_occlusion_finds_the_region_the_model_reads():
    image = red_square()
    found = ex.occlusion(image, redness, 1, classes=["plain", "red"], grid=8)
    assert found.values.shape == (8, 8)
    assert found.blocky and not found.own
    row, col = np.unravel_index(int(np.argmax(found.values)), found.values.shape)
    # the square spans x 40-48 of 64 (column 5-6 of 8) and y 8-16 of 48 (row 1-2)
    assert col in (5, 6) and row in (1, 2)
    assert found.values.min() >= 0
    assert found.target == "red"
    assert "R-VEX-05" in found.sentence
    assert found.extra["passes"] == 65
    assert "not why" in found.warning and "Adebayo" in found.warning


def test_occlusion_of_a_model_that_reads_nothing_is_flat():
    found = ex.occlusion(red_square(), lambda images: np.full((len(images), 2), 0.5), 0)
    assert np.allclose(found.values, 0)
    assert "No blurred cell" in found.sentence


def test_rollout_identity_attention_is_uniform_over_nothing_but_cls():
    tokens, heads = 5, 2
    eye = np.broadcast_to(np.eye(tokens), (heads, tokens, tokens))
    values = ex.rollout([eye, eye], prefix=1, grid=(2, 2))
    assert np.allclose(values, 0.0)  # the class token only attends to itself


def test_rollout_follows_attention_to_one_patch():
    tokens = 5
    a = np.zeros((1, tokens, tokens))
    a[0, :, 3] = 1.0  # everybody attends to token 3 (the third patch)
    values = ex.rollout([a], prefix=1, grid=(2, 2))
    assert values.shape == (2, 2)
    assert np.unravel_index(int(np.argmax(values)), (2, 2)) == (1, 0)
    with pytest.raises(ValueError):
        ex.rollout([a], prefix=1, grid=(3, 3))


def test_png_gray_and_uint8():
    values = np.array([[0.0, 0.5], [1.0, 2.0]])
    pixels, low, high = ex.to_uint8(values)
    assert (low, high) == (0.0, 2.0)
    assert pixels.tolist() == [[0, 64], [128, 255]]
    image = Image.open(io.BytesIO(ex.png_gray(values)))
    assert image.mode == "L" and image.size == (2, 2)
    flat, _, _ = ex.to_uint8(np.ones((3, 3)))
    assert (flat == 0).all()


def test_spearman():
    a = np.arange(16.0).reshape(4, 4)
    assert ex.spearman(a, a * 3 + 1) == pytest.approx(1.0)
    assert ex.spearman(a, -a) == pytest.approx(-1.0)
    assert math.isnan(ex.spearman(a, np.ones((4, 4))))
    with pytest.raises(ValueError):
        ex.spearman(a, np.ones(3))


def test_pool_averages_down():
    v = np.zeros((224, 224), dtype=np.float32)
    v[:112, :112] = 1.0
    small = ex.pool(v, 28)
    assert small.shape == (28, 28)
    assert small[:14, :14].mean() == pytest.approx(1.0)
    assert small[14:, 14:].mean() == pytest.approx(0.0)
    assert ex.pool(np.ones((8, 8)), 28).shape == (8, 8)


def test_crop_box_of_dinov2_and_resnet():
    square = ex.crop_box((500, 500), BACKBONES["dinov2_s14"])
    assert square == (0.0, 0.0, 1.0, 1.0)
    wide = ex.crop_box((400, 200), BACKBONES["dinov2_s14"])
    assert wide[1] == 0.0 and wide[3] == 1.0
    assert wide[0] == pytest.approx(0.25, abs=0.01) and wide[2] == pytest.approx(0.75, abs=0.01)
    resnet = ex.crop_box((256, 256), BACKBONES["resnet18"])  # 256 -> centre 224
    assert resnet[0] == pytest.approx(16 / 256, abs=0.01)


def test_linear_head_equals_the_scikit_learn_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    x = rng.normal(size=(90, 6))
    y = np.array(["a", "b", "c"] * 30)
    x[y == "b", 0] += 2
    x[y == "c", 1] += 2
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(x, y)
    head = ex.head_from_pipeline(model, ["a", "b", "c", "d"])
    p = head.proba(x[:7])
    assert p.shape == (7, 4)
    assert np.allclose(p[:, :3], model.predict_proba(x[:7]), atol=1e-6)
    assert np.allclose(p[:, 3], 0.0)
    binary = make_pipeline(StandardScaler(), LogisticRegression()).fit(x[:60], y[:60] == "b")
    two = ex.head_from_pipeline(binary, ["False", "True"])
    assert np.allclose(two.proba(x[:5]), binary.predict_proba(x[:5]), atol=1e-6)
    with pytest.raises(ValueError):
        ex.head_from_pipeline(binary, ["True", "False"])
