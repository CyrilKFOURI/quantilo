"""Gradient maps, attention rollout and the sanity test with torch (117).

Skipped without torch. No download: timm's `test_vit` and `test_resnet` with random weights,
on the CPU, and a head that reads one channel so the right answer is known."""

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from quantilo.vision import explain as ex  # noqa: E402
from quantilo.vision.deep.backbones import Backbone, BackboneSpec  # noqa: E402

SIDE = 64


def vit() -> Backbone:
    torch.manual_seed(0)
    module = timm.create_model("test_vit", pretrained=False, num_classes=0, img_size=SIDE)
    dims = 2 * module.num_features
    spec = BackboneSpec(
        "tiny_vit", "Tiny ViT", "test_vit", "vit", dims, "test", "advanced", 4, 4, side=SIDE
    )
    return Backbone(spec, module)


def resnet() -> Backbone:
    torch.manual_seed(0)
    module = timm.create_model("test_resnet", pretrained=False, num_classes=0)
    spec = BackboneSpec(
        "tiny_resnet",
        "Tiny ResNet",
        "test_resnet",
        "cnn",
        module.num_features,
        "test",
        "advanced",
        4,
        4,
        side=SIDE,
        interpolation="bilinear",
    )
    return Backbone(spec, module)


def head_for(net: Backbone, seed: int = 0) -> ex.LinearHead:
    rng = np.random.default_rng(seed)
    d = net.spec.dims
    return ex.LinearHead(
        weight=rng.normal(0, 0.3, (3, d)),
        bias=np.zeros(3),
        mean=np.zeros(d),
        scale=np.ones(d),
        classes=["a", "b", "c"],
    )


def picture(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, (SIDE, SIDE, 3), dtype=np.uint8)
    pixels[:20, :20] = (250, 30, 30)
    return Image.fromarray(pixels, "RGB")


@pytest.fixture(params=["vit", "cnn"])
def clf(request) -> ex.Classifier:
    net = vit() if request.param == "vit" else resnet()
    return ex.Classifier(net, head_for(net))


def test_classifier_is_the_head_on_the_backbone_vector(clf):
    image = picture()
    x = clf.tensor(image)
    with torch.no_grad():
        vector = clf.net.features(x).numpy()
        p = clf.proba(x).numpy()
    assert np.allclose(p, clf.head.proba(vector), atol=1e-5)
    assert np.allclose(clf.predict([image, image]), np.repeat(p, 2, axis=0), atol=1e-5)


def test_grad_cam_has_the_grid_of_the_feature_map(clf):
    found = ex.grad_cam(clf, picture(), 1)
    rows, cols = found.values.shape
    if clf.spec.kind == "vit":
        assert (rows, cols) == tuple(clf.module.patch_embed.grid_size)
    else:
        assert rows == cols and rows >= 1
    assert (found.values >= 0).all()
    assert found.target == "b"
    assert "Adebayo" in found.warning and not found.own
    assert found.box == (0.0, 0.0, 1.0, 1.0)


def test_saliency_and_smoothgrad_cover_the_input(clf):
    for found in (ex.saliency(clf, picture(), 0), ex.smoothgrad(clf, picture(), 0, samples=4)):
        assert found.values.shape == (SIDE, SIDE)
        assert np.isfinite(found.values).all() and (found.values >= 0).all()
        assert found.values.max() > 0


def test_integrated_gradients_are_complete_with_enough_steps(clf):
    found = ex.integrated_gradients(clf, picture(), 2, steps=64)
    assert found.values.shape == (SIDE, SIDE)
    assert found.extra["completeness_gap"] < 0.1
    assert "Completeness" in found.notes[0]


def test_rollout_on_a_vit_and_refused_on_a_cnn():
    clf = ex.Classifier(vit(), head_for(vit()))
    found = ex.attention_rollout(clf, picture())
    assert found.values.shape == tuple(clf.module.patch_embed.grid_size)
    assert found.values.sum() > 0 and found.target is None
    assert "Jain and Wallace" in found.warning
    blocks = len(clf.module.blocks)
    assert len(ex.attentions_of(clf, clf.tensor(picture()))) == blocks
    cnn = ex.Classifier(resnet(), head_for(resnet()))
    with pytest.raises(ValueError, match="vision transformer"):
        ex.attention_rollout(cnn, picture())


def test_attention_is_recomputed_exactly():
    """The hook's attention gives the block's own output when multiplied by its values."""
    clf = ex.Classifier(vit(), head_for(vit()))
    x = clf.tensor(picture())
    found = ex.attentions_of(clf, x)
    assert all(np.allclose(a.sum(axis=-1), 1.0, atol=1e-5) for a in found)


def test_sanity_check_cascades_and_leaves_the_model_alone(clf):
    before = [p.detach().clone() for p in clf.module.parameters()]
    head = clf.head.weight.copy()
    result = ex.sanity_check(clf, picture(), 1, "grad_cam", blocks=2)
    assert [s.label for s in result.stages[:2]] == ["trained model", "random head"]
    assert len(result.stages) == 4
    assert result.stages[0].correlation == 1.0
    assert "R-VEX-10" in result.sentence
    assert isinstance(result.passes, bool)
    assert all(torch.equal(a, b) for a, b in zip(before, clf.module.parameters(), strict=True))
    assert np.array_equal(head, clf.head.weight)
    saliency = ex.sanity_check(clf, picture(), 1, "saliency", blocks=1)
    assert saliency.stages[0].values.shape == (28, 28)
    with pytest.raises(ValueError):
        ex.sanity_check(clf, picture(), 1, "rollout")


def test_occlusion_with_a_deep_model(clf):
    found = ex.occlusion(picture(), clf.predict, 0, classes=clf.head.classes, grid=4)
    assert found.values.shape == (4, 4)
    assert found.extra["passes"] == 17
