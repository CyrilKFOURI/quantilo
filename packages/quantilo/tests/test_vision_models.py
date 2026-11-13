"""112: the task rule (R-VML), grouped splits (R-VSP), classic models and the vector cache."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image

from quantilo.vision import classic_models as cm
from quantilo.vision import vectors
from quantilo.vision.split import groups_of, plan_split
from quantilo.vision.task import detect_task

# ---------------------------------------------------------------------------------------------
# Synthetic sets (64 x 64, made here)


def table(n: int, **columns: list[Any]) -> dict[str, list[Any]]:
    base: dict[str, list[Any]] = {
        "image_id": [f"{i:016x}" for i in range(n)],
        "decode_ok": [True] * n,
        "excluded": [False] * n,
        "label": [None] * n,
        "split": [None] * n,
        "is_normal": [None] * n,
        "mask_path": [None] * n,
        "duplicate_group": [None] * n,
        "pixel_sha": [f"sha{i}" for i in range(n)],
    }
    base.update(columns)
    return base


def stripes(seed: int, vertical: bool) -> Image.Image:
    """Class pictures: stripes one way or the other, a little noise, a random phase."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:64, 0:64]
    phase = rng.uniform(0, 2 * np.pi)
    wave = np.sin((xx if vertical else yy) / 3.0 + phase)
    grey = 128 + 90 * wave + rng.normal(0, 12, (64, 64))
    colour = [grey, grey * 0.8, grey * 0.6] if vertical else [grey * 0.6, grey * 0.8, grey]
    rgb = np.clip(np.stack(colour, axis=2), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def part(seed: int, defect: bool = False) -> tuple[Image.Image, Image.Image]:
    """An inspection picture (a centred disc) and its mask; a defect adds a dark blot."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:128, 0:128]
    grey = np.full((128, 128), 60.0)
    grey[(yy - 64) ** 2 + (xx - 64) ** 2 < 45**2] = 200.0
    grey += rng.normal(0, 4, grey.shape)
    mask = np.zeros((128, 128), dtype=np.uint8)
    if defect:
        cy, cx = rng.integers(40, 88, 2)
        blot = (yy - cy) ** 2 + (xx - cx) ** 2 < 10**2
        grey[blot] = 20.0
        mask[blot] = 255
    rgb = np.clip(np.stack([grey] * 3, axis=2), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB"), Image.fromarray(mask, "L")


def classification_set(n: int = 40) -> tuple[pa.Table, list[Image.Image]]:
    labels = ["vertical" if i % 2 else "horizontal" for i in range(n)]
    images = [stripes(i, i % 2 == 1) for i in range(n)]
    return pa.table(table(n, label=labels)), images


def anomaly_set(
    train: int = 30, test_normal: int = 10, test_defect: int = 10
) -> tuple[pa.Table, list[Image.Image], list[Image.Image | None]]:
    images, masks, labels, splits, normal = [], [], [], [], []
    for i in range(train + test_normal + test_defect):
        defect = i >= train + test_normal
        image, mask = part(i, defect)
        images.append(image)
        masks.append(mask if defect else None)
        labels.append("blot" if defect else "good")
        splits.append("train" if i < train else "test")
        normal.append(not defect)
    n = len(images)
    t = pa.table(
        table(
            n,
            label=labels,
            split=splits,
            is_normal=normal,
            mask_path=["m" if m is not None else None for m in masks],
        )
    )
    return t, images, masks


# ---------------------------------------------------------------------------------------------
# R-VML


def test_too_few_images_blocks_models() -> None:
    choice = detect_task(table(12, label=["a"] * 6 + ["b"] * 6))
    assert choice.task is None and choice.code == "R-VML-07"
    assert "at least 20 images" in (choice.blocked or "")


def test_classification_leaves_rare_classes_out() -> None:
    labels = ["cat"] * 12 + ["dog"] * 12 + ["bird"] * 3
    choice = detect_task(table(27, label=labels))
    assert choice.task == "classification" and choice.code == "R-VML-01"
    assert choice.classes == {"cat": 12, "dog": 12} and choice.left_out == {"bird": 3}
    assert "bird (3 images)" in choice.message and "R-VQ-14" in choice.message
    assert choice.chosen_by == "rule" and "clusters" in choice.alternatives


def test_unusable_rows_are_not_counted() -> None:
    data = table(30, label=["a"] * 15 + ["b"] * 15, excluded=[True] * 5 + [False] * 25)
    choice = detect_task(data)
    assert choice.n_usable == 25 and choice.classes == {"a": 10, "b": 15}


def test_normal_training_images_mean_anomalies_with_pixel_metrics() -> None:
    t, _, _ = anomaly_set()
    choice = detect_task(t)
    assert choice.task == "anomaly" and choice.code == "R-VML-02"
    assert choice.evaluable and choice.pixel and choice.n_defect == 10
    assert "R-VML-04" in choice.message


def test_a_single_class_means_anomalies_without_evaluation() -> None:
    choice = detect_task(table(25, label=["ok_part"] * 25))
    assert choice.task == "anomaly" and not choice.evaluable
    assert "most to least unusual" in choice.message


def test_no_label_means_clusters_with_anomalies_offered() -> None:
    choice = detect_task(table(25))
    assert choice.task == "clusters" and choice.code == "R-VML-03"
    assert "assumes most images are normal" in choice.message
    assert choice.alternatives == ["anomaly"]


def test_the_user_may_change_the_task_and_the_rule_stays() -> None:
    choice = detect_task(table(25), task="anomaly")
    assert choice.task == "anomaly" and choice.chosen_by == "user"
    assert choice.rule_task == "clusters" and choice.rule_code == "R-VML-03"
    assert "You chose this task" in choice.message


def test_an_impossible_task_is_refused() -> None:
    with pytest.raises(ValueError, match="R-VML-01"):
        detect_task(table(25), task="classification")
    with pytest.raises(ValueError, match="normal images"):
        detect_task(table(30, label=["a"] * 15 + ["b"] * 15), task="anomaly")


# ---------------------------------------------------------------------------------------------
# R-VSP


def test_producer_split_is_kept_and_leaks_counted() -> None:
    n = 30
    labels = ["a", "b"] * 15
    splits = ["train"] * 20 + ["test"] * 10
    groups: list[int | None] = [None] * n
    groups[0] = groups[25] = 7  # a copy in train and in test
    data = table(n, label=labels, split=splits, duplicate_group=groups)
    plan = plan_split(data, detect_task(data))
    assert plan.code == "R-VSP-01" and plan.kind == "producer"
    assert plan.folds[0].train == list(range(20)) and plan.scored == list(range(20, 30))
    assert plan.leaks == 1 and "R-VQ-11" in plan.notes[0]


def test_folds_keep_copies_together() -> None:
    n = 60
    labels = ["a", "b", "c"] * 20
    groups: list[int | None] = [None] * n
    for row in (0, 3, 6, 9):
        groups[row] = 1  # four near copies of class a
    data = table(n, label=labels, duplicate_group=groups)
    plan = plan_split(data, detect_task(data))
    assert plan.code == "R-VSP-02" and len(plan.folds) == 5
    assert plan.scored == list(range(n))  # every image is scored once
    fold = next(i for i, f in enumerate(plan.folds) if 0 in f.test)
    assert {0, 3, 6, 9} <= set(plan.folds[fold].test)
    for f in plan.folds:
        assert not set(f.train) & set(f.test)


def test_groups_fall_back_to_the_pixel_sha() -> None:
    data = table(3, pixel_sha=["x", "x", "y"], duplicate_group=[None, None, 4])
    assert groups_of(data).tolist() == ["sx", "sx", "g4"]


def test_anomaly_split_holds_normals_out_only_when_the_test_cannot_calibrate() -> None:
    t, _, _ = anomaly_set()
    plan = plan_split(t, detect_task(t))
    assert plan.code == "R-VSP-04" and plan.holdout == []
    assert plan.folds[0].train == list(range(30)) and len(plan.scored) == 20
    t2, _, _ = anomaly_set(train=30, test_normal=10, test_defect=1)
    plan2 = plan_split(t2, detect_task(t2))
    assert plan2.holdout and not set(plan2.holdout) & set(plan2.folds[0].train)
    assert any("R-VTH-02" in note for note in plan2.notes)


def test_anomalies_without_defects_are_scored_out_of_fold() -> None:
    data = table(25, label=["ok"] * 25)
    plan = plan_split(data, detect_task(data))
    assert plan.code == "R-VSP-05" and len(plan.folds) == 5
    assert plan.scored == list(range(25))


def test_anomalies_without_split_tie_groups_to_the_test() -> None:
    n = 40
    normal = [True] * 35 + [False] * 5
    groups: list[int | None] = [None] * n
    groups[2] = groups[36] = 3  # a normal image and a defect that are near copies
    data = table(n, label=["good"] * 35 + ["scratch"] * 5, is_normal=normal, duplicate_group=groups)
    plan = plan_split(data, detect_task(data, task="anomaly"))
    assert 2 in plan.scored and 2 not in plan.folds[0].train
    assert set(range(35, 40)) <= set(plan.scored)


# ---------------------------------------------------------------------------------------------
# Vector cache


def test_vectors_round_trip_and_key(tmp_path: Path) -> None:
    path = tmp_path / "embeddings" / "classic.parquet"
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    x[1] = np.nan
    vectors.save(path, "k1", ["a", "b", "c"], x)
    found = vectors.load(path, "k1")
    assert set(found) == {"a", "c"} and found["c"].tolist() == [8, 9, 10, 11]
    assert vectors.load(path, "k2") == {}
    assert vectors.load(tmp_path / "missing.parquet", "k1") == {}


def test_descriptors_are_cached_per_image(tmp_path: Path) -> None:
    images: list[cm.Source] = [stripes(i, i % 2 == 1) for i in range(3)]
    images.append(b"not an image")
    ids = ["a", "b", "c", "d"]
    path = tmp_path / "classic.parquet"
    x, info = cm.classic_features(images, ids, path)
    assert x.shape == (4, cm.CLASSIC_SIZE) and info.computed == 3 and info.failed == 1
    assert np.isnan(x[3]).all()
    again, info2 = cm.classic_features(images, ids, path)
    assert info2.computed == 0 and np.allclose(again[:3], x[:3])


# ---------------------------------------------------------------------------------------------
# Models


def test_classification_scores_every_model_out_of_fold() -> None:
    t, images = classification_set()
    outcome = cm.evaluate(t, images, deep=False, bootstrap=False)
    result = outcome.result
    assert result.task.task == "classification" and result.split.code == "R-VSP-02"
    assert [m.key for m in result.models] == ["logistic", "hgb"]
    assert sum(m.default for m in result.models) == 1
    default = result.model(result.default or "")
    assert default.selection == max(m.selection or 0 for m in result.models)
    assert any("R-VCL-04" in note for note in default.notes)
    for m in result.models:
        assert m.classification is not None and m.classification.n == 40
        assert m.classification.balanced_accuracy > 0.7
    assert default.classification is not None and default.classification.balanced_accuracy > 0.85
    assert result.deep.used is False and result.deep.reason
    preds = outcome.predictions
    assert preds.num_rows == 40 and {"row", "fold", "truth", "logistic_pred"} <= set(
        preds.column_names
    )
    assert result.sentence.startswith(default.title)
    # R-VQ-16 (b2): read on the default model's out-of-fold probabilities of every image
    found = result.mislabelled
    assert found is not None and found.code == "R-VQ-16" and found.details["checked"] == 40
    assert found.details["model"] == default.title
    assert any("R-VSP-02" in note for note in found.details["notes"])
    assert result.model_dump(mode="json")["mislabelled"]["code"] == "R-VQ-16"


def test_mislabelled_image_is_suggested() -> None:
    """A vertical image labelled horizontal: the model that never saw it disagrees (R-VQ-16)."""
    t, images = classification_set()
    labels = t["label"].to_pylist()
    wrong = labels.index("vertical")
    labels[wrong] = "horizontal"
    t = t.set_column(t.column_names.index("label"), "label", pa.array(labels))
    result = cm.evaluate(t, images, deep=False, bootstrap=False).result
    assert result.mislabelled is not None and wrong in result.mislabelled.rows
    image = next(i for i in result.mislabelled.details["images"] if i["row"] == wrong)
    assert image["suggested"] == "vertical" and image["label"] == "horizontal"


def test_deep_vectors_become_the_desktop_default(monkeypatch: pytest.MonkeyPatch) -> None:
    t, images = classification_set()
    rng = np.random.default_rng(0)
    labels = t["label"].to_pylist()

    def fake(sources: list[Any], *args: Any, **kwargs: Any) -> tuple[np.ndarray, cm.FeatureInfo]:
        x = rng.normal(size=(len(sources), 16)).astype(np.float32)
        x[:, 0] += [3.0 if v == "vertical" else -3.0 for v in labels]
        info = cm.FeatureInfo(
            key="dinov2_s14", title="fake", dims=16, computed=40, failed=0, seconds=0
        )
        return x, info

    monkeypatch.setattr(cm, "deep_option", lambda deep: cm.DeepOption(used=True))
    monkeypatch.setattr(cm, "deep_features", fake)
    result = cm.evaluate(t, images, bootstrap=False).result
    assert [m.key for m in result.models] == ["logistic", "hgb", "deep_linear", "deep_knn"]
    assert result.default == "deep_linear"
    assert "R-VCL-06" in result.model("deep_linear").notes[0]
    assert "pts over the best classic model" in result.sentence


def test_class_weights_when_imbalanced() -> None:
    n = 48
    labels = ["horizontal"] * 40 + ["vertical"] * 8
    images = [stripes(i, i >= 40) for i in range(n)]
    result = cm.evaluate(pa.table(table(n, label=labels)), images, deep=False, bootstrap=False)
    m = result.result.model("logistic")
    assert any("R-VCL-05" in note for note in m.notes)
    assert m.classification is not None and m.classification.main.metric == "balanced_accuracy"


def test_anomalies_find_the_blots_and_their_place() -> None:
    t, images, masks = anomaly_set()
    outcome = cm.evaluate(t, images, masks=masks, deep=False, bootstrap=False)
    result = outcome.result
    assert [m.key for m in result.models] == ["patch_pca", "knn", "ensemble"]
    assert result.default == "ensemble"
    patch = result.model("patch_pca").anomaly
    assert patch is not None and patch.auroc > 0.95
    assert patch.pixel is not None and patch.pixel.auroc > 0.9
    assert patch.threshold is not None and patch.threshold.code == "R-VTH-01"
    ensemble = result.model("ensemble")
    assert ensemble.anomaly is not None and ensemble.anomaly.auroc > 0.9
    assert ensemble.galleries["most_unusual"][0] >= 40  # a defect comes first
    assert outcome.predictions.num_rows == 20
    assert "ensemble_score" in outcome.predictions.column_names


def test_a_threshold_set_by_hand() -> None:
    t, images, _ = anomaly_set(train=25, test_normal=4, test_defect=4)
    result = cm.evaluate(t, images, deep=False, bootstrap=False, threshold=0.5).result
    report = result.model("ensemble").anomaly
    assert report is not None and report.threshold is not None
    assert report.threshold.code == "R-VTH-03"


def test_anomalies_without_defects_rank_and_set_a_threshold() -> None:
    n = 25
    images = [part(i)[0] for i in range(n)]
    data = pa.table(table(n, label=["part"] * n))
    result = cm.evaluate(data, images, deep=False, bootstrap=False).result
    assert result.split.code == "R-VSP-05"
    model = result.model("ensemble")
    assert model.anomaly is None and model.threshold is not None
    assert model.threshold.code == "R-VTH-02" and model.flagged is not None
    assert len(model.galleries["most_unusual"]) == min(cm.GALLERY, n)


def test_patch_errors_are_larger_on_a_defect() -> None:
    normals = [cm.luma_square(part(i)[0]) for i in range(20)]
    assert all(n is not None for n in normals)
    bank = np.concatenate([cm.patches_of(n) for n in normals if n is not None])
    model = cm.PatchPCA().fit(bank).calibrate([n for n in normals if n is not None])
    clean_luma = cm.luma_square(part(100)[0])
    blot_luma = cm.luma_square(part(101, defect=True)[0])
    assert clean_luma is not None and blot_luma is not None
    assert model.errors(blot_luma).max() > 3 * max(1.0, model.errors(clean_luma).max())


def test_clusters_find_two_kinds() -> None:
    n = 30
    images = [stripes(i, i % 2 == 1) for i in range(n)]
    labels = ["v" if i % 2 else "h" for i in range(n)]
    outcome = cm.evaluate(pa.table(table(n)), images, deep=False)
    clusters = outcome.result.clusters
    assert clusters is not None and clusters.k >= 2 and clusters.silhouette > 0
    assert outcome.result.models == [] and outcome.predictions.num_rows == n
    x, _ = cm.classic_features(images)
    found, assigned = cm.find_clusters(x, labels)
    assert found.adjusted_rand is not None and len(assigned) == n


def test_cancel_stops_the_run() -> None:
    t, images = classification_set()
    with pytest.raises(cm.Cancelled):
        cm.evaluate(t, images, deep=False, cancel=lambda: True)


def test_progress_goes_forward() -> None:
    t, images = classification_set()
    seen: list[float] = []
    cm.evaluate(t, images, deep=False, bootstrap=False, progress=lambda f, m: seen.append(f))
    assert seen and seen[-1] == 1.0 and all(b >= a for a, b in itertools.pairwise(seen))


def test_the_browser_version_holds_to_its_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from quantilo.ml import browser

    t, images = classification_set()
    monkeypatch.setattr(browser, "IN_BROWSER", True)
    monkeypatch.setattr(cm, "BROWSER_MAX_IMAGES", 30)
    with pytest.raises(ValueError, match="R-VCL-07"):
        cm.evaluate(t, images)
    assert cm.deep_option(None).used is False


def test_deep_option_says_why_without_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    from quantilo.vision import deep

    monkeypatch.setattr(deep, "missing", lambda: ["timm"])
    option = cm.deep_option(None)
    assert option.used is False and option.reason is not None and "R-VDP-01" in option.reason
    assert cm.deep_option(False).reason == "Deep vectors not asked for."


# ---------------------------------------------------------------------------------------------
# 113: a deep detector (115, 116) in the run, maps and full probabilities kept


class FakeDetector:
    """Stands for PatchCore / the autoencoder: the darker the disc, the more unusual."""

    title = "Fake detector"
    method = "autoencoder"

    def __init__(self) -> None:
        self.progress: Any = None
        self.cancel: Any = None
        self.notes = ["fitted"]
        self.fitted = 0

    def fit(self, sources: list[Any]) -> FakeDetector:
        self.fitted = len(sources)
        if self.progress is not None:
            self.progress(len(sources), len(sources))
        return self

    def score(self, sources: list[Any]) -> Any:
        from quantilo.vision.deep.anomaly import AnomalyResult

        grey = [np.asarray(s.convert("L").resize((256, 256)), dtype=np.float32) for s in sources]
        maps = np.stack([np.clip(120 - g, 0, None) for g in grey]).astype(np.float32)
        result = AnomalyResult(
            "autoencoder", self.title, maps.reshape(len(sources), -1).max(axis=1).astype(float),
            maps, np.ones(len(sources), dtype=bool), [(128, 128)] * len(sources), 256, "CPU",
            0.0, "fake",
        )  # fmt: skip
        result.reconstructions = np.stack(  # type: ignore[attr-defined]
            [np.asarray(s.convert("RGB").resize((256, 256))) for s in sources]
        )
        return result


def test_a_deep_detector_runs_on_the_same_folds_and_becomes_the_default() -> None:
    t, images, masks = anomaly_set()
    made: list[FakeDetector] = []

    def build() -> FakeDetector:
        made.append(FakeDetector())
        return made[-1]

    outcome = cm.evaluate(
        t, images, masks=masks, deep=False, bootstrap=False, detector="autoencoder",
        detector_model=build,
    )  # fmt: skip
    result = outcome.result
    assert [m.key for m in result.models] == ["patch_pca", "knn", "ensemble", "autoencoder"]
    assert result.default == "autoencoder" and result.sentence.startswith("Fake detector")
    deep = result.model("autoencoder")
    assert deep.family == "deep" and deep.anomaly is not None and deep.anomaly.auroc > 0.95
    assert deep.anomaly.pixel is not None and deep.anomaly.pixel.auroc > 0.9
    assert made[0].fitted == 30 and "fitted" in deep.notes
    assert deep.map_source == "autoencoder" and deep.reconstructions
    assert result.model("ensemble").map_source == "patch_pca"
    maps = outcome.extra["maps"]
    assert set(maps) == {"patch_pca", "autoencoder"} and len(maps["autoencoder"]) == 20
    assert next(iter(maps["autoencoder"].values())).shape == (cm.PIXEL_SIDE, cm.PIXEL_SIDE)
    recon = outcome.extra["reconstructions"]["autoencoder"]
    assert next(iter(recon.values())).shape == (256, 256, 3)
    assert "autoencoder_score" in outcome.predictions.column_names


def test_auto_detector_falls_back_with_its_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    from quantilo.vision import deep

    monkeypatch.setattr(deep, "missing", lambda: ["timm"])
    notes: list[str] = []
    assert cm.choose_detector(None, None, notes) == (None, cm.choose_detector("auto", None, [])[1])
    assert cm.choose_detector("none", None, notes)[0] is None
    method, reason = cm.choose_detector("padim", None, notes)
    assert method is None and reason and "R-VDP-01" in reason and "PaDiM" in notes[0]
    with pytest.raises(ValueError):
        cm.choose_detector("magic", None, [])


def test_classification_keeps_every_class_probability() -> None:
    t, images = classification_set()
    preds = cm.evaluate(t, images, deep=False, bootstrap=False).predictions
    probs = preds["logistic_probs"].to_pylist()
    assert len(probs[0]) == 2 and abs(sum(probs[0]) - 1) < 1e-4
