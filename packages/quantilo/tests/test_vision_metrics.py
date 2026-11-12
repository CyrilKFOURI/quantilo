"""111 - vision metrics, checked against scikit-learn, published values and brute force."""

import math
from typing import Any, cast

import numpy as np
import pytest
from sklearn import metrics as skm

from quantilo.vision import metrics as vm

RNG = np.random.default_rng(7)
ZERO: Any = 0  # zero_division (the stubs expect a string)
NONE: Any = None


def multiclass(n: int = 400, k: int = 4, seed: int = 1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    logits = rng.normal(size=(n, k)) + 2.0 * np.eye(k)[y]
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return y, proba, proba.argmax(axis=1)


# --- labels ------------------------------------------------------------------------------------


def test_label_metrics_match_sklearn():
    y, _, pred = multiclass()
    assert vm.accuracy(y, pred) == pytest.approx(skm.accuracy_score(y, pred))
    assert vm.balanced_accuracy(y, pred) == pytest.approx(skm.balanced_accuracy_score(y, pred))
    assert vm.f1_macro(y, pred) == pytest.approx(skm.f1_score(y, pred, average="macro"))
    assert vm.cohen_kappa(y, pred) == pytest.approx(skm.cohen_kappa_score(y, pred))
    np.testing.assert_array_equal(vm.confusion_matrix(y, pred), skm.confusion_matrix(y, pred))
    p, r, f, s = (
        np.asarray(v) for v in skm.precision_recall_fscore_support(y, pred, zero_division=ZERO)
    )
    for i, row in enumerate(vm.per_class(y, pred)):
        assert (row.precision, row.recall, row.f1, row.support) == pytest.approx(
            (p[i], r[i], f[i], s[i])
        )


def test_string_labels_and_a_class_never_predicted():
    y = np.array(["cat", "dog", "dog", "fox", "fox", "fox"])
    pred = np.array(["dog", "dog", "dog", "fox", "fox", "dog"])
    assert vm.f1_macro(y, pred) == pytest.approx(
        skm.f1_score(y, pred, average="macro", zero_division=ZERO)
    )
    assert vm.balanced_accuracy(y, pred) == pytest.approx(skm.balanced_accuracy_score(y, pred))
    np.testing.assert_array_equal(vm.confusion_matrix(y, pred), skm.confusion_matrix(y, pred))


def test_kappa_published_example():
    """Cohen's kappa, the textbook example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> 0.4."""
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert vm.cohen_kappa(a, b) == pytest.approx(0.4)


# --- scores ------------------------------------------------------------------------------------


def test_published_roc_and_ap_example():
    """scikit-learn's documented example: ROC AUC 0.75, average precision 0.8333."""
    y, s = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    assert vm.roc_auc(y, s) == pytest.approx(0.75)
    assert vm.average_precision(y, s) == pytest.approx(0.8333, abs=1e-4)


def test_hanley_mcneil_1982_ratings():
    """Hanley and McNeil (Radiology, 1982), 5-point CT ratings: area 0.893 (heavy ties)."""
    normal = [1] * 33 + [2] * 6 + [3] * 6 + [4] * 11 + [5] * 2
    abnormal = [1] * 3 + [2] * 2 + [3] * 2 + [4] * 11 + [5] * 33
    y = [0] * len(normal) + [1] * len(abnormal)
    assert vm.roc_auc(y, normal + abnormal) == pytest.approx(0.893, abs=5e-4)


@pytest.mark.parametrize("seed", range(5))
def test_binary_scores_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=300)
    s = np.round(rng.normal(size=300) + y, 1)  # rounded: many ties
    assert vm.roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s))
    assert vm.average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s))
    fpr, tpr, _ = vm.roc_curve(y, s)
    ref_fpr, ref_tpr, _ = skm.roc_curve(y, s, drop_intermediate=False)
    np.testing.assert_allclose(fpr, ref_fpr)
    np.testing.assert_allclose(tpr, ref_tpr)


def test_auc_undefined_without_both_classes():
    assert math.isnan(vm.roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert math.isnan(vm.average_precision([0, 0], [0.1, 0.2]))


def test_one_vs_rest_matches_sklearn():
    y, proba, _ = multiclass()
    macro, by_class = vm.roc_auc_ovr(y, proba)
    assert macro == pytest.approx(skm.roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    ref = np.asarray(skm.roc_auc_score(np.eye(4)[y], proba, average=NONE))
    assert [by_class[str(i)] for i in range(4)] == pytest.approx(list(ref))
    pr_macro, _ = vm.pr_auc_ovr(y, proba)
    assert pr_macro == pytest.approx(skm.average_precision_score(np.eye(4)[y], proba))


def test_binary_probabilities_use_the_second_label():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=200)
    p1 = np.clip(rng.normal(0.35 + 0.3 * y, 0.2), 0, 1)
    proba = np.column_stack([1 - p1, p1])
    assert vm.roc_auc_ovr(y, proba)[0] == pytest.approx(skm.roc_auc_score(y, p1))
    assert vm.log_loss(y, proba) == pytest.approx(skm.log_loss(y, proba))
    assert vm.brier(y, proba) == pytest.approx(skm.brier_score_loss(y, p1))


def test_probabilistic_losses_match_sklearn():
    y, proba, _ = multiclass()
    proba[0] = [1.0, 0.0, 0.0, 0.0]  # a confident zero: clipped like scikit-learn
    assert vm.log_loss(y, proba) == pytest.approx(skm.log_loss(y, proba))
    assert vm.brier(y, proba) == pytest.approx(skm.brier_score_loss(y, proba))
    for k in (1, 2, 3):
        assert vm.top_k_accuracy(y, proba, k) == pytest.approx(
            skm.top_k_accuracy_score(y, proba, k=k)
        )


def test_top_k_levels():
    assert vm.top_k_levels(3) == []
    assert vm.top_k_levels(5) == [3]
    assert vm.top_k_levels(12) == [3, 5]


# --- calibration -------------------------------------------------------------------------------


def test_ece_by_hand():
    """Two bins used: (0.6, 0.6667] holds 0.65 x2 (one right), (0.9333, 1] holds 0.95 x2 (both
    right): ECE = 0.5 * |0.5 - 0.65| + 0.5 * |1 - 0.95| = 0.1."""
    proba = np.array([[0.65, 0.35], [0.65, 0.35], [0.05, 0.95], [0.05, 0.95]])
    y = [0, 1, 1, 1]
    diagram = vm.reliability(y, proba)
    assert diagram.ece == pytest.approx(0.1)
    assert diagram.mce == pytest.approx(0.15)
    assert sum(diagram.counts) == 4 and len(diagram.edges) == 16


def test_ece_matches_a_direct_loop():
    y, proba, _ = multiclass(seed=5)
    conf, correct = proba.max(axis=1), proba.argmax(axis=1) == y
    expected = 0.0
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:], strict=True):
        inside = (conf > lo) & (conf <= hi)
        if inside.any():
            expected += inside.mean() * abs(correct[inside].mean() - conf[inside].mean())
    assert vm.expected_calibration_error(y, proba) == pytest.approx(expected)


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(11)
    n, k = 4000, 3
    logits = rng.normal(size=(n, k)) * 1.5
    true_p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in true_p])
    sharp = np.exp(3 * logits) / np.exp(3 * logits).sum(axis=1, keepdims=True)
    assert vm.fit_temperature(y, sharp) == pytest.approx(3.0, rel=0.1)
    found = vm.calibration(y, sharp)
    assert found.suggest_temperature and found.ece_after is not None
    assert found.ece_after < found.reliability.ece
    assert "R-VM-03" in found.reason
    assert not vm.calibration(y, true_p).suggest_temperature


# --- main metric, baseline, bootstrap, galleries -------------------------------------------------


def test_main_metric_rule():
    assert vm.main_metric([0] * 30 + [1] * 10).metric == "accuracy"  # ratio 3
    imbalanced = vm.main_metric([0] * 31 + [1] * 10)
    assert imbalanced.metric == "balanced_accuracy" and imbalanced.code == "R-VQ-13"


def test_majority_baseline_uses_training_labels():
    base = vm.majority_baseline(["a", "b", "b"], y_train=["a", "a", "b"])
    assert base.label == "a"
    assert base.accuracy == pytest.approx(1 / 3)
    assert base.balanced_accuracy == pytest.approx(0.5)


def test_bootstrap_is_seeded_and_groups_widen_it():
    y = np.repeat(RNG.integers(0, 2, size=40), 10)  # 40 objects photographed 10 times
    pred = np.where(RNG.random(400) < 0.8, y, 1 - y)
    groups = np.repeat(np.arange(40), 10)
    fn = lambda rows: vm.accuracy(y[rows], pred[rows])  # noqa: E731
    plain = vm.bootstrap_interval(fn, 400, draws=300)
    again = vm.bootstrap_interval(fn, 400, draws=300)
    assert plain is not None and again is not None and plain == again
    assert plain.low < vm.accuracy(y, pred) < plain.high
    grouped = vm.bootstrap_interval(fn, 400, groups=groups, draws=300)
    assert grouped is not None and grouped.by_group


def test_bootstrap_drops_degenerate_draws():
    y = np.array([0] * 19 + [1])
    s = np.linspace(0, 1, 20)
    found = vm.bootstrap_interval(lambda rows: vm.roc_auc(y[rows], s[rows]), 20, draws=200)
    assert found is not None and found.dropped > 0 and found.draws + found.dropped == 200


def test_galleries():
    y = np.array([0, 1, 1, 0])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.95, 0.05], [0.55, 0.45]])
    assert vm.most_confident_errors(y, proba) == [2]
    assert vm.most_uncertain(proba, k=1) == [3]


def test_classification_report():
    y, proba, _ = multiclass(n=300, k=5)
    report = vm.classification_report(y, proba=proba, draws=200)
    assert report.labels == ["0", "1", "2", "3", "4"]
    assert report.accuracy == pytest.approx(skm.accuracy_score(y, proba.argmax(axis=1)))
    assert report.roc_auc == pytest.approx(skm.roc_auc_score(y, proba, multi_class="ovr"))
    assert set(report.top_k) == {3}
    assert set(report.intervals) == {"accuracy", "balanced_accuracy", "f1_macro", "roc_auc"}
    assert report.calibration is not None
    assert report.model_dump_json()  # serialisable for the API


def test_report_notes_a_class_absent_from_the_test():
    y = np.array([0, 1, 0, 1, 0, 1])
    proba = np.full((6, 3), 1 / 3)
    proba[:, 0] += np.array([0.2, -0.1, 0.2, -0.1, 0.2, -0.1])
    proba[:, 1] -= np.array([0.2, -0.1, 0.2, -0.1, 0.2, -0.1])
    report = vm.classification_report(y, proba=proba, labels=[0, 1, 2], bootstrap=False)
    assert "2" not in report.roc_auc_by_class
    assert any("R-VM-05" in note for note in report.notes)


# --- anomalies: image level and thresholds --------------------------------------------------------


def test_at_threshold_counts():
    found = vm.at_threshold([0, 0, 1, 1, 1], [0.1, 0.6, 0.6, 0.9, 0.2], 0.6)
    assert (found.tp, found.fp, found.fn, found.tn) == (2, 1, 1, 1)
    assert found.f1 == pytest.approx(skm.f1_score([0, 0, 1, 1, 1], [0, 1, 1, 1, 0]))
    assert found.false_alarm_rate == pytest.approx(0.5)


def test_f1_max_threshold_matches_sklearn_curve():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=200)
    s = rng.normal(size=200) + 1.5 * y
    cut, f1 = vm.threshold_f1_max(y, s)
    p, r, _ = skm.precision_recall_curve(y, s)
    best = np.nanmax(2 * p * r / np.where(p + r == 0, np.nan, p + r))
    assert f1 == pytest.approx(best)
    assert vm.at_threshold(y, s, cut).f1 == pytest.approx(best)


def test_rule_vth_01_splits_the_test():
    y = np.array([0] * 20 + [1] * 10)
    s = np.r_[RNG.normal(0, 1, 20), RNG.normal(2, 1, 10)]
    choice = vm.choose_threshold(y, s, seed=0)
    assert choice.code == "R-VTH-01"
    calib, evaluation = set(choice.calibration), set(choice.evaluation)
    assert not calib & evaluation and len(calib | evaluation) == 30
    assert sum(y[list(calib)]) == 5 and sum(y[list(evaluation)]) == 5
    assert vm.choose_threshold(y, s, seed=0) == choice


def test_rule_vth_02_and_03():
    normals = np.arange(200) / 200
    choice = vm.choose_threshold([0, 0, 0], [0.1, 0.2, 0.3], holdout_normal_scores=normals)
    assert choice.code == "R-VTH-02"
    assert choice.threshold == pytest.approx(np.quantile(normals, 0.99))
    assert "rough" not in choice.message
    assert "rough" in vm.choose_threshold(holdout_normal_scores=normals[:20]).message
    assert vm.choose_threshold([0, 1], [0.1, 0.9], user_threshold=0.5).code == "R-VTH-03"
    with pytest.raises(ValueError):
        vm.choose_threshold([0, 0], [0.1, 0.2])


def test_auroc_by_type_and_histogram():
    kinds = ["good", "good", "crack", "crack", "scratch", "scratch"]
    s = [0.1, 0.2, 0.9, 0.8, 0.15, 0.3]
    by_type = vm.auroc_by_type(kinds, s)
    assert by_type == {"crack": 1.0, "scratch": 0.75}
    hist = vm.score_histogram([k != "good" for k in kinds], s, 0.5, bins=4)
    assert sum(hist.normal) == 2 and sum(hist.defect) == 4 and hist.threshold == 0.5


# --- anomalies: pixel level --------------------------------------------------------------------


def test_aupro_by_hand():
    gt = np.array([[1, 1, 0, 0]])
    assert vm.aupro(gt, np.array([[0.9, 0.8, 0.1, 0.2]])) == pytest.approx(1.0)
    # 0.9 finds half the region, then 0.8 is a false alarm (FPR 0.5): PRO stays 0.5 up to 30 %
    assert vm.aupro(gt, np.array([[0.9, 0.1, 0.8, 0.2]])) == pytest.approx(0.5)


def brute_aupro(gt: np.ndarray, am: np.ndarray, limit: float) -> float:
    """PRO by its definition at every threshold: mean overlap over regions, 8-connected."""
    from scipy import ndimage

    regions = []
    for image, g in enumerate(gt):
        lab, count = cast(tuple[np.ndarray, int], ndimage.label(g, structure=np.ones((3, 3))))
        regions += [(image, lab == i + 1) for i in range(count)]
    xs, ys = [0.0], [0.0]
    for t in np.sort(np.unique(am))[::-1]:
        flagged = am >= t
        xs.append(float(flagged[~gt].mean()))
        ys.append(float(np.mean([flagged[image][region].mean() for image, region in regions])))
    return vm._area_until(np.array(xs), np.array(ys), limit) / limit


@pytest.mark.parametrize("seed", range(4))
def test_aupro_matches_its_definition(seed):
    rng = np.random.default_rng(seed)
    gt = np.zeros((3, 12, 12), dtype=bool)
    gt[0, 2:5, 2:5] = True  # region of 9
    gt[0, 8:10, 7:11] = True  # region of 8
    gt[1, 5, 5] = True  # region of 1
    gt[1, 6, 6] = True  # diagonal neighbour: the same region (8-connected)
    am = np.round(rng.random(gt.shape) + 0.6 * gt, 2)
    for limit in (0.3, 1.0):
        assert vm.aupro(gt, am, limit) == pytest.approx(brute_aupro(gt, am, limit))
    assert vm.pro_curve(gt, am)[2] == 3


def test_pixel_report():
    gt = np.zeros((2, 8, 8), dtype=bool)
    gt[0, 2:4, 2:4] = True
    am = gt.astype(float) * 0.8 + 0.1
    found = vm.pixel_report(gt, am, threshold=0.5)
    assert found.auroc == pytest.approx(1.0) and found.aupro == pytest.approx(1.0)
    assert found.f1 == pytest.approx(1.0) and found.iou == pytest.approx(1.0)
    assert vm.pixel_auroc(gt, am) == pytest.approx(skm.roc_auc_score(gt.ravel(), am.ravel()))
    with pytest.raises(ValueError):
        vm.pixel_report(gt, am[:1])


def test_anomaly_report():
    rng = np.random.default_rng(4)
    kinds = np.array(["good"] * 30 + ["crack"] * 10 + ["hole"] * 10)
    y = kinds != "good"
    s = rng.normal(size=50) + 2 * y
    report = vm.anomaly_report(y, s, defect_types=kinds, draws=200)
    assert report.auroc == pytest.approx(skm.roc_auc_score(y, s))
    assert report.ap == pytest.approx(skm.average_precision_score(y, s))
    assert report.threshold is not None and report.threshold.code == "R-VTH-01"
    assert report.at_threshold is not None
    counts = report.at_threshold
    assert counts.tp + counts.fp + counts.fn + counts.tn == len(report.threshold.evaluation)
    assert set(report.auroc_by_type) == {"crack", "hole"}
    assert report.auroc_interval is not None
    assert report.model_dump_json()
