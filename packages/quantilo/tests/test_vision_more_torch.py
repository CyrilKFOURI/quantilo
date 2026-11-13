"""118 with torch: fine-tuning, ProtoPNet-lite, TCAV and the More models of a run.

Skipped without torch. No download: timm's `test_vit` and `test_resnet` with random weights, on
the CPU, and small coloured images whose class is known."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from quantilo.vision import explain as ex  # noqa: E402
from quantilo.vision.deep import concepts as cb  # noqa: E402
from quantilo.vision.deep import finetune as ft  # noqa: E402
from quantilo.vision.deep import more  # noqa: E402
from quantilo.vision.deep import protopnet as pp  # noqa: E402
from quantilo.vision.deep import tcav as tc  # noqa: E402
from quantilo.vision.deep.backbones import Backbone, BackboneSpec  # noqa: E402

SIDE = 64


def vit(seed: int = 0) -> Backbone:
    torch.manual_seed(seed)
    module = timm.create_model("test_vit", pretrained=False, num_classes=0, img_size=SIDE)
    spec = BackboneSpec(
        "tiny_vit", "Tiny ViT", "test_vit", "vit", 2 * module.num_features, "test", "advanced",
        8, 8, side=SIDE,
    )  # fmt: skip
    return Backbone(spec, module)


def resnet(seed: int = 0) -> Backbone:
    torch.manual_seed(seed)
    module = timm.create_model("test_resnet", pretrained=False, num_classes=0)
    spec = BackboneSpec(
        "tiny_resnet", "Tiny ResNet", "test_resnet", "cnn", module.num_features, "test",
        "advanced", 8, 8, side=SIDE, crop=0.875, interpolation="bilinear",
    )  # fmt: skip
    return Backbone(spec, module)


def picture(red: bool, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 60, (80, 80, 3), dtype=np.uint8)
    channel = 0 if red else 2
    pixels[:, :, channel] = np.clip(pixels[:, :, channel].astype(int) + 180, 0, 255)
    return Image.fromarray(pixels, "RGB")


def images(n: int = 48) -> tuple[list[Image.Image], np.ndarray]:
    labels = np.array([i % 2 for i in range(n)])
    return [picture(bool(c), i) for i, c in enumerate(labels)], labels


def test_finetune_learns_the_top_and_keeps_the_rest_frozen():
    pics, y = images()
    net = vit()
    frozen_before = net.module.blocks[0].attn.qkv.weight.detach().clone()
    top_before = net.module.blocks[-1].attn.qkv.weight.detach().clone()
    bank = ft.ImageBank(pics, net.spec)
    rows = np.arange(len(pics))
    steps = []
    model = ft.finetune(
        net, bank, rows[:36], y[:36], rows[:36].astype(str), 2, allow_gpu=False,
        max_epochs=6, progress=lambda p, m: steps.append(m),
    )  # fmt: skip
    proba, vectors = model.predict(rows[36:].tolist())
    assert proba.shape == (12, 2) and np.allclose(proba.sum(axis=1), 1)
    assert vectors.shape == (12, net.spec.dims)
    assert np.mean(proba.argmax(axis=1) == y[36:]) >= 0.75  # red against blue
    assert torch.equal(net.module.blocks[0].attn.qkv.weight, frozen_before)  # R-VFT-01
    assert not torch.equal(net.module.blocks[-1].attn.qkv.weight, top_before)
    log = model.log
    assert log.precision == "fp32" and log.device == "CPU"
    assert 1 <= log.best_epoch <= log.epochs <= 6 and log.history
    assert "held out" in log.notes[0] and "R-VFT-04" in log.stopped
    assert steps and steps[0].startswith("Fine-tuning: epoch 1")
    weight, bias = model.head_arrays()
    scores = vectors.astype(np.float64) @ weight.T + bias
    z = np.exp(scores - scores.max(axis=1, keepdims=True))
    assert np.allclose(z / z.sum(axis=1, keepdims=True), proba, atol=1e-3)  # the kept head
    model.release()


def test_finetune_cnn_opens_its_last_stage():
    pics, y = images(32)
    net = resnet()
    frozen = net.module.layer1[0].conv1.weight.detach().clone()
    bank = ft.ImageBank(pics, net.spec)
    model = ft.finetune(
        net, bank, np.arange(32), y, np.arange(32).astype(str), 2, allow_gpu=False, max_epochs=2
    )
    assert torch.equal(net.module.layer1[0].conv1.weight, frozen)
    assert ft.trainable_blocks(net.module, "cnn")[0][0] == "stage 4"
    assert model.log.epochs >= 1


def test_finetune_long_run_asks_first(monkeypatch):
    from quantilo.vision.deep import budget

    pics, y = images(24)
    net = vit()
    bank = ft.ImageBank(pics, net.spec)
    monkeypatch.setattr(budget, "LONG_RUN_SECONDS", -1.0)
    monkeypatch.setattr(budget.Estimate, "needs_confirmation", property(lambda self: True))
    with pytest.raises(budget.LongRun):
        ft.finetune(net, bank, np.arange(24), y, np.arange(24).astype(str), 2, allow_gpu=False)


def test_protopnet_pushes_onto_real_patches_and_explains():
    pics, y = images(40)
    net = vit()
    bank = ft.ImageBank(pics, net.spec)
    rows = np.arange(40)
    features = pp.patch_features(net, bank, rows, allow_gpu=False)
    assert features.grid == (4, 4) and features.features.shape[:2] == (40, 16)
    labels = np.where(y == 1, "red", "blue")
    model, found = pp.fit_protopnet(
        features, rows[:30], labels[:30], rows[:30].astype(str), ["blue", "red"],
        allow_gpu=False, per_class=3, max_epochs=8, last_epochs=10,
    )  # fmt: skip
    assert len(found.prototypes) == 6
    for proto in found.prototypes:
        assert labels[proto.row] == proto.label  # R-VPP-04: a patch of its own class
        assert 0 <= proto.cell < 16 and proto.row < 30
    # after the push a prototype *is* its patch: distance 0 to it
    z = model.head.add_on(
        torch.from_numpy(features.take([found.prototypes[0].row]).astype(np.float32))
    )
    d = ((z[0, found.prototypes[0].cell] - model.head.prototypes[0]) ** 2).sum()
    assert float(d) < 1e-6
    proba, contributions = model.explain(features.take(rows[30:]))
    assert proba.shape == (10, 2)
    assert np.mean(np.array(["blue", "red"])[proba.argmax(axis=1)] == labels[30:]) >= 0.7
    first = contributions[0]
    assert len(first) == pp.TOP
    assert first[0].points >= first[1].points >= first[2].points
    assert all(c.points == pytest.approx(c.similarity * c.weight, abs=1e-3) for c in first)
    assert found.log.history and "R-VPP-03" in found.log.stopped


def test_tcav_directional_derivative_is_the_gradient_along_the_cav():
    net = vit()
    rng = np.random.default_rng(0)
    head = ex.LinearHead(
        rng.normal(0, 0.3, (2, net.spec.dims)), np.zeros(2), np.zeros(net.spec.dims),
        np.ones(net.spec.dims), ["a", "b"],
    )  # fmt: skip
    clf = ex.Classifier(net, head)
    pics, _ = images(4)
    layer = tc.default_layer(clf.module, "vit")
    assert layer == "block 4"  # 3/4 of 6 blocks (R-VTC-02)
    grads = tc.gradients(clf, pics[:2], layer, [1])[1]
    width = net.module.blocks[0].attn.qkv.in_features
    cav = rng.normal(size=width)
    cav /= np.linalg.norm(cav)
    block = net.module.blocks[3]
    eps = 1e-3

    def shifted(step: float) -> np.ndarray:
        handle = block.register_forward_hook(
            lambda _m, _a, out: out + step * torch.from_numpy(cav).float()
        )
        try:
            with torch.no_grad():
                x = torch.cat([clf.tensor(p) for p in pics[:2]])
                return clf.scores(x)[:, 1].double().numpy()
        finally:
            handle.remove()

    numeric = (shifted(eps) - shifted(-eps)) / (2 * eps)
    assert np.allclose(grads @ cav, numeric, rtol=0.05, atol=1e-3)


def test_learn_cav_points_to_the_concept():
    rng = np.random.default_rng(0)
    positive = rng.normal(size=(40, 8)) + np.r_[3.0, np.zeros(7)]
    negative = rng.normal(size=(40, 8))
    cav, accuracy = tc.learn_cav(positive, negative)
    assert accuracy >= 0.85 and cav[0] > 0.9 and np.linalg.norm(cav) == pytest.approx(1)
    assert tc.tcav_score(
        np.array([[1.0] + [0] * 7, [-1.0] + [0] * 7, [2.0] + [0] * 7]), cav
    ) == pytest.approx(2 / 3)


def test_tcav_end_to_end_reports_a_test_per_class():
    net = resnet()
    rng = np.random.default_rng(1)
    head = ex.LinearHead(
        rng.normal(0, 0.3, (2, net.spec.dims)), np.zeros(2), np.zeros(net.spec.dims),
        np.ones(net.spec.dims), ["blue", "red"],
    )  # fmt: skip
    clf = ex.Classifier(net, head)
    reds = [picture(True, 100 + i) for i in range(24)]
    blues = [picture(False, 200 + i) for i in range(40)]
    found = tc.tcav(
        clf, reds, blues, {"blue": blues[:10], "red": reds[:10]}, name="red", random_sets=4
    )
    assert found.layer == "stage 3" and found.layers[-1] == "stage 4"
    assert found.images == 24 and found.random_images == 24
    assert found.cav_accuracy > 0.8  # red against blue is easy to separate
    assert [c.label for c in found.classes] == ["blue", "red"]
    for c in found.classes:
        assert 0 <= c.score <= 1 and c.p_value is not None and c.sentence
    with pytest.raises(ValueError, match="at least 20 images"):
        tc.tcav(clf, reds[:5], blues, {"red": reds})


def test_run_more_trains_on_the_folds_and_keeps_what_explains(monkeypatch):
    from quantilo.vision import classic_models as cm
    from quantilo.vision.deep import backbones

    monkeypatch.setattr(backbones, "load", lambda key=None: vit())
    pics, y = images(48)
    labels = np.where(y == 1, "red", "blue").astype(object)
    groups = np.arange(48).astype(str)
    plan = SimpleNamespace(
        folds=[SimpleNamespace(train=list(range(36)), test=list(range(36, 48)))],
        scored=list(range(36, 48)),
    )
    table = pa.table(
        {"width": [80] * 48, "height": [80] * 48, "image_id": [str(i) for i in range(48)]}
    )
    rng = np.random.default_rng(0)
    deep_x = rng.normal(size=(48, 16)).astype(np.float32)
    deep_x[:, 0] += 3 * y  # the vectors know the colour
    concepts = [
        cb.Concept("column:red", "red", "binary", y.astype(float), "column", positive="True"),
        cb.Concept("column:noise", "noise", "numeric", rng.normal(size=48), "column"),
    ]
    reference = np.tile([0.5, 0.5], (12, 1))
    messages = []
    found = more.run_more(
        ["finetune", "protopnet", "concept_bottleneck"], table, plan, labels, groups,
        ["blue", "red"], pics, np.zeros((48, 4)), deep_x, lambda p, m: messages.append(m),
        concepts=concepts, reference=reference, allow_gpu=False, bootstrap=False,
    )  # fmt: skip
    assert [m.key for m in found.models] == ["finetune", "protopnet", "concept_bottleneck"]
    for m in found.models:
        assert m.level == "more" and not m.default
        assert m.classification is not None and "over the linear probe" in m.notes[0]
    assert found.columns["finetune_probs"] and len(found.columns["protopnet_pred"]) == 12
    assert set(found.arrays) >= {"finetune_f0_rows.npy", "finetune_f0_vectors.npy"}
    proto = found.files["protopnet.json"]
    assert proto["grid"] == [4, 4] and len(proto["folds"][0]["prototypes"]) == 20
    looks = proto["folds"][0]["images"]["36"]
    assert len(looks) == 3 and len(looks[0]["box"]) == 4
    bottleneck = found.files["concept_bottleneck.json"]
    assert [c["name"] for c in bottleneck["concepts"]] == ["red", "noise"]
    assert len(bottleneck["folds"][0]["predicted"]) == 12
    concept_scores = {c["name"]: c for c in found.models[2].concepts}
    assert concept_scores["red"]["value"] > 0.9 and concept_scores["noise"]["weak"]
    assert found.models[0].training and found.models[0].training[0]["epochs"] >= 1
    assert any("Fine-tuning" in m for m in messages)
    assert issubclass(cm.Cancelled, RuntimeError)
