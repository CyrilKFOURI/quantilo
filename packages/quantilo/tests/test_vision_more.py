"""118 without torch: concept bottleneck, the More options and their reasons, ProtoPNet's boxes,
the held-out rows of early stopping."""

import numpy as np
import pyarrow as pa
import pytest

from quantilo.vision.deep import concepts as cb
from quantilo.vision.deep import more
from quantilo.vision.deep.backbones import BACKBONES
from quantilo.vision.deep.finetune import square, validation_rows
from quantilo.vision.deep.protopnet import cell_box


def _data(n: int = 240, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 12))
    spots = (x[:, 0] + 0.2 * rng.normal(size=n) > 0).astype(float)  # a binary concept
    size = 3.0 * x[:, 1] + 10 + 0.1 * rng.normal(size=n)  # a numeric one
    labels = np.where(spots == 1, "rust", "healthy").astype(object)
    groups = np.arange(n).astype(str)
    return x, spots, size, labels, groups


def _concepts(spots, size):
    return [
        cb.Concept("column:spots", "spots", "binary", spots, "column", positive="True"),
        cb.Concept("column:size", "size", "numeric", size, "column"),
    ]


def test_column_concepts_keep_only_metadata_columns():
    table = pa.table(
        {
            "image_id": ["a", "b", "c", "d", "e", "f"],
            "brightness": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # made by the import: not a concept
            "label": ["x", "y", "x", "y", "x", "y"],
            "has_spots": [True, False, True, None, False, False],
            "leaf": ["old", "young", "young", "young", "young", "old"],
            "width_mm": [1.0, 2.5, 3.0, 4.2, 5.1, 6.0],
            "colour": ["red", "red", "green", "blue", "red", "green"],  # 3 text values: no
            "count": [1, 1, 2, 2, 1, 2],  # numeric, two values: binary
        }
    )
    found = {c.name: c for c in cb.column_concepts(table)}
    assert set(found) == {"has_spots", "leaf", "width_mm", "count"}
    spots = found["has_spots"]
    assert spots.kind == "binary" and spots.positive == "True"
    assert np.isnan(spots.values[3]) and spots.values.tolist()[:3] == [1.0, 0.0, 1.0]
    assert found["leaf"].positive == "old"  # the rarer value is the concept
    assert found["width_mm"].kind == "numeric"
    assert found["count"].kind == "binary"


def test_group_concept_marks_its_rows():
    c = cb.group_concept("group:g1", "yellow edges", [1, 3], 5)
    assert c.values.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert "others are taken as without it" in c.note


def test_bottleneck_reads_the_concepts_and_follows_an_intervention():
    x, spots, size, labels, groups = _data()
    concepts = _concepts(spots, size)
    train = np.arange(180)
    test = np.arange(180, 240)
    model = cb.fit_bottleneck(
        x[train], concepts, train, labels[train], groups[train], ["healthy", "rust"]
    )
    predicted = model.concepts(x[test])
    found = {q.name: q for q in cb.quality(model, concepts, predicted, test)}
    assert found["spots"].metric == "roc_auc" and (found["spots"].value or 0.0) > 0.9
    assert found["size"].metric == "r2" and (found["size"].value or 0.0) > 0.9
    assert not found["spots"].weak
    proba = model.head.proba(predicted)
    accuracy = np.mean(np.array(["healthy", "rust"])[proba.argmax(axis=1)] == labels[test])
    assert accuracy >= 0.8
    # R-VCB-05: setting the spots concept to 1 moves every image towards rust
    edited = predicted.copy()
    edited[:, 0] = 1.0
    assert (model.head.proba(edited)[:, 1] >= proba[:, 1] - 1e-9).all()
    assert found["spots"].weights["rust"] > 0 > found["spots"].weights["healthy"]


def test_bottleneck_needs_two_concepts():
    x, spots, size, labels, groups = _data(60)
    with pytest.raises(ValueError, match="at least 2 concepts"):
        cb.fit_bottleneck(x, _concepts(spots, size)[:1], np.arange(60), labels, groups, ["a"])


def test_options_say_why_and_estimate():
    found = {o.key: o for o in more.options("anomaly", 100, 1, 0, deep_ready=True)}
    assert not found["finetune"].available and "classification" in (found["finetune"].reason or "")
    found = {o.key: o for o in more.options("classification", 1000, 1, 1, deep_ready=False)}
    assert not found["protopnet"].available and found["protopnet"].reason
    bottleneck = found["concept_bottleneck"]
    assert not bottleneck.available and "at least 2 concepts" in (bottleneck.reason or "")
    found = {o.key: o for o in more.options("classification", 1000, 5, 3, deep_ready=False)}
    assert found["concept_bottleneck"].available
    assert (
        found["concept_bottleneck"].estimate and "5 folds" in found["concept_bottleneck"].estimate
    )


def test_cell_box_follows_the_centre_square():
    dino = BACKBONES["dinov2_s14"]  # crop 1.0: the whole centre square
    box = cell_box(0, (16, 16), (400, 200), dino)
    assert box == pytest.approx((100 / 400, 0.0, (100 + 200 / 16) / 400, 1 / 16))
    last = cell_box(255, (16, 16), (200, 200), dino)
    assert last == pytest.approx((15 / 16, 15 / 16, 1.0, 1.0))
    resnet = BACKBONES["resnet18"]  # crop 0.875: the centre 87.5 % of the square
    box = cell_box(0, (7, 7), (256, 256), resnet)
    assert box[0] == pytest.approx(0.0625) and box[2] == pytest.approx(0.0625 + 0.875 / 7)


def test_validation_rows_are_grouped_and_stratified():
    y = np.repeat([0, 1], 50)
    groups = np.repeat(np.arange(50), 2)  # copies come in pairs
    held = validation_rows(y, groups.astype(str), seed=0)
    assert 8 <= len(held) <= 25
    assert set(groups[held]).isdisjoint(set(np.delete(groups, held)))
    assert set(y[held]) == {0, 1}
    assert len(validation_rows(y[:10], groups[:10].astype(str))) == 0  # too few: none


def test_square_is_the_centre():
    from PIL import Image

    pixels = np.zeros((100, 300, 3), dtype=np.uint8)
    pixels[:, 100:200] = 255  # the centre square is white
    out = square(Image.fromarray(pixels), 32)
    assert out.shape == (32, 32, 3) and out.min() > 200
