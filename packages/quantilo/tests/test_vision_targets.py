"""c3: regression on images (R-VML-05, R-VRG) and multi-label models (R-VML-06, R-VMLB), with the
grouped split R-VSP-06."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image

from quantilo.vision import classic_models as cm
from quantilo.vision import target_models as tm
from quantilo.vision import targets as tg
from quantilo.vision.split import plan_split
from quantilo.vision.task import detect_task, usable_rows
from tests.test_vision_models import stripes, table


def tinted(seed: int, level: float) -> Image.Image:
    """Stripes whose brightness follows `level` (0 to 1): a number the pixels tell."""
    base = np.asarray(stripes(seed, seed % 2 == 0), dtype=float)
    return Image.fromarray(np.clip(base * (0.3 + 0.7 * level), 0, 255).astype(np.uint8), "RGB")


def regression_set(n: int = 60) -> tuple[pa.Table, list[Image.Image]]:
    rng = np.random.default_rng(3)
    levels = rng.uniform(0, 1, n)
    weight = (levels * 100 + rng.normal(0, 2, n)).round(2).tolist()
    data = table(n, label=["part"] * n, weight_g=weight, lot=[i % 3 for i in range(n)])
    return pa.table(data), [tinted(i, float(v)) for i, v in enumerate(levels)]


def test_numeric_columns_are_offered_never_chosen() -> None:
    data, _ = regression_set()
    choice = detect_task(data)
    assert choice.task == "anomaly"  # a single class: R-VML-02, not a regression
    assert choice.numeric_targets == ["weight_g"]  # lot has 3 values only
    assert "regression" in choice.alternatives
    forced = detect_task(data, "regression")
    assert forced.task == "regression" and forced.code == "R-VML-05"
    assert forced.target == "weight_g" and forced.n_target == 60
    assert forced.chosen_by == "user" and forced.rule_task == "anomaly"
    with pytest.raises(ValueError, match="R-VML-05"):
        detect_task(data, "regression", target="lot")


def test_regression_learns_a_number_from_the_pixels() -> None:
    data, images = regression_set()
    found = cm.evaluate(data, images, task="regression", deep=False, bootstrap=False)
    result = found.result
    assert result.split.code == "R-VSP-06" and len(result.split.folds) == 5
    model = result.model("ridge")
    assert model.default and model.regression is not None
    report = model.regression
    assert report.r2 > 0.5, report
    assert report.mae < report.baseline_mae
    assert "R²" in result.sentence and "weight_g" in result.sentence
    assert set(found.predictions.column_names) >= {"truth", "ridge_pred", "fold"}
    assert model.galleries["over"] or model.galleries["under"]


def test_regression_report_numbers() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    report = tm.regression_report(y, y + 0.5, float(y.mean()), bootstrap=True)
    assert report.mae == pytest.approx(0.5)
    assert report.rmse == pytest.approx(0.5)
    assert report.r2 == pytest.approx(1 - 2.5 / 82.5)
    assert report.baseline_r2 == pytest.approx(0.0)
    assert report.spearman == pytest.approx(1.0)
    assert "mae" in report.intervals


def multilabel_set(n: int = 60) -> tuple[pa.Table, list[Image.Image]]:
    """`stripe` (vertical stripes) and `bright` (a brighter picture) are independent labels."""
    labels, images = [], []
    for i in range(n):
        vertical, bright = i % 2 == 1, (i // 2) % 2 == 1
        names = [n for n, on in (("stripe", vertical), ("bright", bright)) if on] or ["plain"]
        labels.append("|".join(names))
        base = np.asarray(stripes(i, vertical), dtype=float) * (1.0 if bright else 0.45)
        images.append(Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB"))
    return pa.table(table(n, label=labels)), images


def test_pipes_make_the_rule_multilabel() -> None:
    data, _ = multilabel_set()
    choice = detect_task(data)
    assert choice.task == "multilabel" and choice.code == "R-VML-06"
    assert choice.label_names == ["bright", "plain", "stripe"]
    assert choice.label_source == "pipe"
    plan = plan_split(data, choice)
    assert plan.code == "R-VSP-06"


def test_multilabel_models_and_metrics() -> None:
    data, images = multilabel_set()
    found = cm.evaluate(data, images, deep=False, bootstrap=False)
    model = found.result.model("ovr_logistic")
    assert model.default and model.multilabel is not None
    report = model.multilabel
    assert report.labels == ["bright", "plain", "stripe"]
    assert report.macro_f1 > report.baseline_macro_f1
    assert 0 <= report.hamming_loss <= 1 and report.micro_f1 > 0.5
    stripe = next(s for s in report.per_label if s.label == "stripe")
    assert stripe.auroc is not None and stripe.auroc > 0.8
    assert "Macro F1" in found.result.sentence
    row = found.predictions.slice(0, 1).to_pylist()[0]
    assert isinstance(row["truth"], list) and len(row["ovr_logistic_probs"]) == 3


def test_true_false_columns_offer_multilabel() -> None:
    n = 40
    data = pa.table(
        table(
            n,
            label=["a" if i % 2 else "b" for i in range(n)],
            has_spot=[bool(i % 3 == 0) for i in range(n)],
            outdoor=["yes" if i % 4 < 2 else "no" for i in range(n)],
            decision=[None] * n,  # R-VED-04: never a target
            pinned=[bool(i % 2) for i in range(n)],
        )
    )
    choice = detect_task(data)
    assert choice.task == "classification"
    assert choice.binary_targets == ["has_spot", "outdoor"]
    assert "multilabel" in choice.alternatives
    forced = detect_task(data, "multilabel")
    assert forced.label_source == "columns" and forced.label_names == ["has_spot", "outdoor"]
    sets = tg.label_sets(data, usable_rows(data), [None] * n, forced.binary_targets)
    assert sets.y.shape == (n, 2) and sets.known.all()


def test_rare_labels_are_left_out() -> None:
    labels = ["a|b"] * 10 + ["a"] * 10 + ["c"] * 3 + ["b"] * 10
    data = pa.table(table(len(labels), label=labels))
    choice = detect_task(data)
    assert choice.task == "multilabel"
    assert "c" not in choice.label_names and "R-VMLB-02" in choice.message


# --- R-VAN-07: PatchCore + autoencoder ------------------------------------------------------------


def test_the_ensemble_runs_both_members_and_averages_their_ranks_and_maps() -> None:
    from tests.test_vision_models import FakeDetector, anomaly_set

    t, images, masks = anomaly_set()
    made: list[FakeDetector] = []

    def build() -> FakeDetector:
        made.append(FakeDetector())
        return made[-1]

    outcome = cm.evaluate(
        t, images, masks=masks, deep=False, bootstrap=False, detector="patchcore_ae",
        detector_model=build,
    )  # fmt: skip
    result = outcome.result
    keys = [m.key for m in result.models]
    assert keys == ["patch_pca", "knn", "ensemble", "patchcore", "autoencoder", "patchcore_ae"]
    assert len(made) == 2  # one per member, on the producer's single fold
    assert result.default == "patchcore_ae"
    ensemble = result.model("patchcore_ae")
    assert ensemble.level == "more" and "R-VAN-07" in ensemble.notes[0]
    assert ensemble.anomaly is not None and ensemble.anomaly.auroc > 0.95
    assert ensemble.anomaly.pixel is not None
    assert ensemble.map_source == "patchcore_ae"
    maps = outcome.extra["maps"]["patchcore_ae"]
    first = next(iter(maps.values()))
    assert first.shape == (cm.PIXEL_SIDE, cm.PIXEL_SIDE) and 0 <= first.min() <= first.max() <= 1
    assert "patchcore_ae_score" in outcome.predictions.column_names


def test_the_ensemble_needs_both_members(monkeypatch: pytest.MonkeyPatch) -> None:
    from quantilo.vision import deep

    monkeypatch.setattr(deep, "missing", lambda: ["timm"])
    row = deep.status().detector("patchcore_ae")
    assert not row.available and row.level == "more" and "R-VDP-01" in (row.reason or "")
    method, reason = cm.choose_detector("patchcore_ae", None, [])
    assert method is None and reason
