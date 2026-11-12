"""Vision metrics (111): image classification and anomaly detection, in numpy only.

No torch and no scikit-learn, so the same code runs in the browser version (Pyodide). scipy is
used only for the connected regions of AUPRO (`scipy.ndimage.label`). Every value is checked
against scikit-learn or a published value in `tests/test_vision_metrics.py`.

Conventions: class labels are any hashable values; probability columns follow `labels` (sorted
unique labels of y_true and y_pred by default, like scikit-learn). An image or a pixel is declared
defective when its score is >= the threshold.
"""

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel

ECE_BINS = 15
ECE_ALERT = 0.05  # R-VM-03: beyond, temperature scaling is proposed
IMBALANCE_RATIO = 3.0  # R-VQ-13: majority / minority above this -> balanced accuracy
BOOTSTRAP_DRAWS = 1_000  # R-VM-02
BOOTSTRAP_LEVEL = 0.95
SEED = 0
AUPRO_LIMIT = 0.3  # integrate the PRO curve up to 30 % false positive rate
NORMAL_QUANTILE = 0.99  # R-VTH-02
FEW_NORMALS = 100  # R-VTH-02: below, the 99 % quantile is rough
TEMPERATURE_BOUNDS = (0.05, 20.0)
TOP_K_FROM = {3: 5, 5: 10}  # R-VM-04: top-k shown from this many classes

Array = Any  # array-like accepted by np.asarray


# ---------------------------------------------------------------------------------------------
# Result models (serialisable for the API and the bench)


class ClassScores(BaseModel):
    label: str
    precision: float
    recall: float
    f1: float
    support: int


class Interval(BaseModel):
    low: float
    high: float
    level: float = BOOTSTRAP_LEVEL
    draws: int  # resamples kept (degenerate ones are dropped)
    dropped: int = 0
    by_group: bool = False


class MainMetric(BaseModel):
    metric: Literal["accuracy", "balanced_accuracy"]
    code: str
    reason: str


class Baseline(BaseModel):
    label: str
    accuracy: float
    balanced_accuracy: float
    reason: str = "R-VM-01: always predicts the most frequent class"


class Reliability(BaseModel):
    """Reliability diagram on equal-width bins of the top-class confidence."""

    edges: list[float]
    confidence: list[float | None]  # mean confidence per bin (None when the bin is empty)
    accuracy: list[float | None]
    counts: list[int]
    ece: float
    mce: float


class Calibration(BaseModel):
    reliability: Reliability
    suggest_temperature: bool
    temperature: float | None = None
    ece_after: float | None = None
    reason: str


class ClassificationReport(BaseModel):
    labels: list[str]
    n: int
    main: MainMetric
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    kappa: float
    per_class: list[ClassScores]
    confusion: list[list[int]]
    confusion_normalized: list[list[float]]
    baseline: Baseline
    roc_auc: float | None = None
    pr_auc: float | None = None
    roc_auc_by_class: dict[str, float] = {}
    pr_auc_by_class: dict[str, float] = {}
    log_loss: float | None = None
    brier: float | None = None
    calibration: Calibration | None = None
    top_k: dict[int, float] = {}
    intervals: dict[str, Interval] = {}
    notes: list[str] = []


class ThresholdScores(BaseModel):
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    false_alarm_rate: float


class ThresholdChoice(BaseModel):
    code: Literal["R-VTH-01", "R-VTH-02", "R-VTH-03"]
    threshold: float
    reason: str
    message: str  # the sentence shown next to the threshold
    calibration: list[int] = []  # R-VTH-01: test images the threshold was chosen on
    evaluation: list[int] = []  # R-VTH-01: test images the counts are computed on


class Histogram(BaseModel):
    edges: list[float]
    normal: list[int]
    defect: list[int]
    threshold: float | None = None


class PixelReport(BaseModel):
    auroc: float
    aupro: float
    limit: float = AUPRO_LIMIT
    threshold: float | None = None
    f1: float | None = None
    iou: float | None = None
    regions: int


class AnomalyReport(BaseModel):
    n_normal: int
    n_defect: int
    auroc: float
    ap: float
    auroc_interval: Interval | None = None
    auroc_by_type: dict[str, float] = {}
    threshold: ThresholdChoice | None = None
    at_threshold: ThresholdScores | None = None
    histogram: Histogram
    pixel: PixelReport | None = None
    notes: list[str] = []


# ---------------------------------------------------------------------------------------------
# Encoding


def _labels(y_true: np.ndarray, y_pred: np.ndarray | None, labels: Sequence[Any] | None) -> list:
    if labels is not None:
        return list(labels)
    values = y_true if y_pred is None else np.concatenate([y_true, y_pred])
    return sorted(np.unique(values).tolist())


def _encode(values: np.ndarray, labels: list) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    try:
        return np.array([index[v] for v in values.tolist()], dtype=np.int64)
    except KeyError as missing:
        raise ValueError(f"Label {missing} is not in the labels.") from None


def _binary(y: Array) -> np.ndarray:
    return np.asarray(y).astype(bool).ravel()


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


# ---------------------------------------------------------------------------------------------
# Classification from predicted labels


def confusion_matrix(
    y_true: Array, y_pred: Array, labels: Sequence[Any] | None = None
) -> np.ndarray:
    """Counts, rows = true class, columns = predicted class (scikit-learn's layout)."""
    t, p = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    order = _labels(t, p, labels)
    k = len(order)
    flat = _encode(t, order) * k + _encode(p, order)
    return np.bincount(flat, minlength=k * k).reshape(k, k)


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.zeros(cm.shape, dtype=float), where=totals > 0)


def accuracy(y_true: Array, y_pred: Array) -> float:
    return float(np.mean(np.asarray(y_true).ravel() == np.asarray(y_pred).ravel()))


def balanced_accuracy(y_true: Array, y_pred: Array, labels: Sequence[Any] | None = None) -> float:
    """Mean recall over the classes present in y_true."""
    cm = confusion_matrix(y_true, y_pred, labels)
    support = cm.sum(axis=1)
    present = support > 0
    return float(np.mean(np.diag(cm)[present] / support[present]))


def _prf(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    predicted, support = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    both = precision + recall
    f1 = np.divide(2 * precision * recall, both, out=np.zeros_like(tp), where=both > 0)
    return precision, recall, f1, support


def per_class(
    y_true: Array, y_pred: Array, labels: Sequence[Any] | None = None
) -> list[ClassScores]:
    """Precision, recall, F1 and support per class (0 when undefined, like zero_division=0)."""
    t, p = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    order = _labels(t, p, labels)
    precision, recall, f1, support = _prf(confusion_matrix(t, p, order))
    return [
        ClassScores(
            label=str(label),
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i]),
        )
        for i, label in enumerate(order)
    ]


def f1_macro(y_true: Array, y_pred: Array, labels: Sequence[Any] | None = None) -> float:
    return float(np.mean(_prf(confusion_matrix(y_true, y_pred, labels))[2]))


def cohen_kappa(y_true: Array, y_pred: Array, labels: Sequence[Any] | None = None) -> float:
    cm = confusion_matrix(y_true, y_pred, labels).astype(float)
    n = cm.sum()
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    off = 1 - np.eye(len(cm))
    denominator = float((off * expected).sum())
    return 1.0 - float((off * cm).sum()) / denominator if denominator else 0.0


def main_metric(y_true: Array) -> MainMetric:
    """R-VQ-13: balanced accuracy leads when the classes are imbalanced."""
    _, counts = np.unique(np.asarray(y_true).ravel(), return_counts=True)
    ratio = counts.max() / counts.min() if len(counts) else 1.0
    if ratio > IMBALANCE_RATIO:
        return MainMetric(
            metric="balanced_accuracy",
            code="R-VQ-13",
            reason=(
                f"Classes are imbalanced (ratio {ratio:.1f}): accuracy would hide the rare class."
            ),
        )
    return MainMetric(
        metric="accuracy",
        code="R-VQ-13",
        reason=f"Classes are balanced enough (ratio {ratio:.1f}, at most {IMBALANCE_RATIO:g}).",
    )


def majority_baseline(
    y_true: Array, y_train: Array | None = None, labels: Sequence[Any] | None = None
) -> Baseline:
    """R-VM-01: the most frequent class of the training labels (else of y_true), always."""
    t = np.asarray(y_true).ravel()
    source = t if y_train is None else np.asarray(y_train).ravel()
    values, counts = np.unique(source, return_counts=True)
    label = values[int(np.argmax(counts))]
    guess = np.array([label] * len(t))
    order = _labels(t, np.asarray([label]), labels)
    return Baseline(
        label=str(label),
        accuracy=accuracy(t, guess),
        balanced_accuracy=balanced_accuracy(t, guess, order),
    )


# ---------------------------------------------------------------------------------------------
# Scores: ROC and precision-recall


def _curve_counts(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False and true positives at each distinct threshold, highest threshold first."""
    order = np.argsort(-scores, kind="mergesort")
    s, t = scores[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    tps = np.cumsum(t, dtype=np.int64)[last]
    fps = (last + 1) - tps
    return fps, tps, s[last]


def roc_curve(y_true: Array, scores: Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds), starting at (0, 0) with an infinite threshold."""
    y, s = _binary(y_true), np.asarray(scores, dtype=float).ravel()
    fps, tps, thresholds = _curve_counts(y, s)
    fps, tps = np.r_[0, fps], np.r_[0, tps]
    fpr = fps / fps[-1] if fps[-1] else np.full(fps.shape, np.nan)
    tpr = tps / tps[-1] if tps[-1] else np.full(tps.shape, np.nan)
    return fpr, tpr, np.r_[np.inf, thresholds]


def roc_auc(y_true: Array, scores: Array) -> float:
    """Area under the ROC curve (ties count 1/2); nan without both classes."""
    y = _binary(y_true)
    if y.all() or not y.any():
        return math.nan
    fpr, tpr, _ = roc_curve(y, scores)
    return float(np.trapezoid(tpr, fpr))


def pr_curve(y_true: Array, scores: Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(precision, recall, thresholds), highest threshold first, without the (1, 0) end point."""
    y, s = _binary(y_true), np.asarray(scores, dtype=float).ravel()
    fps, tps, thresholds = _curve_counts(y, s)
    precision = tps / (tps + fps)
    recall = tps / tps[-1] if tps[-1] else np.full(tps.shape, np.nan)
    return precision, recall, thresholds


def average_precision(y_true: Array, scores: Array) -> float:
    """PR AUC as scikit-learn's average precision: sum of (R_n - R_n-1) * P_n."""
    y = _binary(y_true)
    if not y.any():
        return math.nan
    precision, recall, _ = pr_curve(y, scores)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


# ---------------------------------------------------------------------------------------------
# Probabilities


def _proba(proba: Array, k: int) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1 - p, p])
    if p.shape[1] != k:
        raise ValueError(f"{p.shape[1]} probability columns for {k} labels.")
    return p


def roc_auc_ovr(
    y_true: Array, proba: Array, labels: Sequence[Any] | None = None
) -> tuple[float, dict[str, float]]:
    """Binary: AUC of the second label. Several classes: macro one-vs-rest, and per class
    (R-VM-05: a class absent from y_true is left out)."""
    return _ovr(roc_auc, y_true, proba, labels)


def pr_auc_ovr(
    y_true: Array, proba: Array, labels: Sequence[Any] | None = None
) -> tuple[float, dict[str, float]]:
    return _ovr(average_precision, y_true, proba, labels)


def _ovr(
    score: Callable[[Array, Array], float],
    y_true: Array,
    proba: Array,
    labels: Sequence[Any] | None,
) -> tuple[float, dict[str, float]]:
    t = np.asarray(y_true).ravel()
    order = _labels(t, None, labels)
    p = _proba(proba, len(order))
    codes = _encode(t, order)
    if len(order) == 2:
        value = score(codes == 1, p[:, 1])
        return value, {str(order[1]): value}
    by_class = {str(label): score(codes == i, p[:, i]) for i, label in enumerate(order)}
    kept = [v for v in by_class.values() if not math.isnan(v)]
    return (float(np.mean(kept)) if kept else math.nan), {
        k: v for k, v in by_class.items() if not math.isnan(v)
    }


def log_loss(y_true: Array, proba: Array, labels: Sequence[Any] | None = None) -> float:
    t = np.asarray(y_true).ravel()
    order = _labels(t, None, labels)
    p = _proba(proba, len(order))
    eps = np.finfo(p.dtype).eps
    picked = np.clip(p, eps, 1 - eps)[np.arange(len(t)), _encode(t, order)]
    return float(-np.mean(np.log(picked)))


def brier(y_true: Array, proba: Array, labels: Sequence[Any] | None = None) -> float:
    """Mean of sum (p - y)^2 over the classes, halved for two classes (scikit-learn's choice:
    the binary Brier score is the mean of (p1 - y)^2)."""
    t = np.asarray(y_true).ravel()
    order = _labels(t, None, labels)
    p = _proba(proba, len(order))
    onehot = np.eye(len(order))[_encode(t, order)]
    value = float(np.mean(np.sum((onehot - p) ** 2, axis=1)))
    return value / 2 if len(order) == 2 else value


def top_k_accuracy(
    y_true: Array, proba: Array, k: int, labels: Sequence[Any] | None = None
) -> float:
    t = np.asarray(y_true).ravel()
    order = _labels(t, None, labels)
    p = _proba(proba, len(order))
    ranked = np.argsort(p, axis=1, kind="mergesort")[:, ::-1][:, :k]
    return float(np.mean((ranked == _encode(t, order)[:, None]).any(axis=1)))


def top_k_levels(n_classes: int) -> list[int]:
    """R-VM-04: top-3 from 5 classes, top-5 from 10."""
    return [k for k, start in TOP_K_FROM.items() if n_classes >= start]


def reliability(
    y_true: Array, proba: Array, labels: Sequence[Any] | None = None, bins: int = ECE_BINS
) -> Reliability:
    """Expected calibration error on `bins` equal-width bins (a, b] of the top-class confidence
    (Guo et al., 2017), with the points of the reliability diagram."""
    t = np.asarray(y_true).ravel()
    order = _labels(t, None, labels)
    p = _proba(proba, len(order))
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == _encode(t, order)
    which = np.clip(np.ceil(confidence * bins).astype(int) - 1, 0, bins - 1)
    counts = np.bincount(which, minlength=bins)
    conf_sum = np.bincount(which, weights=confidence, minlength=bins)
    acc_sum = np.bincount(which, weights=correct.astype(float), minlength=bins)
    filled = counts > 0
    mean_conf = np.divide(conf_sum, counts, out=np.zeros(bins), where=filled)
    mean_acc = np.divide(acc_sum, counts, out=np.zeros(bins), where=filled)
    gaps = np.abs(mean_acc - mean_conf)
    return Reliability(
        edges=np.linspace(0, 1, bins + 1).tolist(),
        confidence=[float(c) if f else None for c, f in zip(mean_conf, filled, strict=True)],
        accuracy=[float(a) if f else None for a, f in zip(mean_acc, filled, strict=True)],
        counts=counts.tolist(),
        ece=float(np.sum(gaps * counts) / len(t)),
        mce=float(gaps[filled].max()) if filled.any() else 0.0,
    )


def expected_calibration_error(
    y_true: Array, proba: Array, labels: Sequence[Any] | None = None, bins: int = ECE_BINS
) -> float:
    return reliability(y_true, proba, labels, bins).ece


def apply_temperature(proba: Array, temperature: float) -> np.ndarray:
    """softmax(log p / T): T > 1 softens over-confident probabilities, the ranking is kept."""
    logits = np.log(np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(
    y_true: Array,
    proba: Array,
    labels: Sequence[Any] | None = None,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
) -> float:
    """The temperature with the smallest log loss (golden-section search on log T)."""
    t = np.asarray(y_true).ravel()
    order = _labels(t, None, labels)
    p = _proba(proba, len(order))
    codes = _encode(t, order)

    def nll(log_t: float) -> float:
        q = apply_temperature(p, math.exp(log_t))
        return float(-np.mean(np.log(np.clip(q[np.arange(len(codes)), codes], 1e-300, 1))))

    a, b = math.log(bounds[0]), math.log(bounds[1])
    ratio = (math.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = nll(d)
    return math.exp((a + b) / 2)


def calibration(
    y_true: Array, proba: Array, labels: Sequence[Any] | None = None, bins: int = ECE_BINS
) -> Calibration:
    """R-VM-03: ECE and the reliability diagram; temperature scaling proposed beyond 0.05."""
    diagram = reliability(y_true, proba, labels, bins)
    if diagram.ece <= ECE_ALERT:
        return Calibration(
            reliability=diagram,
            suggest_temperature=False,
            reason=f"ECE {diagram.ece:.3f} (at most {ECE_ALERT:g}): probabilities are reliable.",
        )
    t = np.asarray(y_true).ravel()
    order = _labels(t, None, labels)
    temperature = fit_temperature(t, proba, order)
    after = reliability(t, apply_temperature(_proba(proba, len(order)), temperature), order, bins)
    return Calibration(
        reliability=diagram,
        suggest_temperature=True,
        temperature=temperature,
        ece_after=after.ece,
        reason=(
            f"R-VM-03: ECE {diagram.ece:.3f} is above {ECE_ALERT:g}: the probabilities are "
            f"{'over' if temperature > 1 else 'under'}-confident. Temperature scaling "
            f"(T = {temperature:.2f}) brings the ECE to {after.ece:.3f} without changing the "
            "predictions."
            + (
                " T reached its upper bound: the probabilities carry almost no information."
                if temperature > 0.99 * TEMPERATURE_BOUNDS[1]
                else ""
            )
        ),
    )


def most_confident_errors(
    y_true: Array, proba: Array, labels: Sequence[Any] | None = None, k: int = 24
) -> list[int]:
    """Indices of the wrong predictions made with the highest confidence (gallery)."""
    t = np.asarray(y_true).ravel()
    order = _labels(t, None, labels)
    p = _proba(proba, len(order))
    wrong = np.flatnonzero(p.argmax(axis=1) != _encode(t, order))
    ranked = wrong[np.argsort(-p[wrong].max(axis=1), kind="mergesort")]
    return ranked[:k].tolist()


def most_uncertain(proba: Array, k: int = 24) -> list[int]:
    """Indices with the smallest gap between the two most likely classes (gallery)."""
    p = np.asarray(proba, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1 - p, p])
    top2 = np.sort(p, axis=1)[:, -2:]
    return np.argsort(top2[:, 1] - top2[:, 0], kind="mergesort")[:k].tolist()


# ---------------------------------------------------------------------------------------------
# Bootstrap (R-VM-02)


def bootstrap_interval(
    statistic: Callable[[np.ndarray], float],
    n: int,
    groups: Array | None = None,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
    level: float = BOOTSTRAP_LEVEL,
) -> Interval | None:
    """Percentile interval of `statistic(rows)` over resampled rows, or over resampled groups
    (all the rows of a drawn group come together). Draws where the statistic is undefined (nan,
    e.g. a single class) are dropped and counted."""
    rng = np.random.default_rng(seed)
    if groups is None:
        members = None
        units = n
    else:
        codes = np.unique(np.asarray(groups).ravel(), return_inverse=True)[1]
        order = np.argsort(codes, kind="mergesort")
        starts = np.r_[0, np.cumsum(np.bincount(codes))]
        members = [order[starts[g] : starts[g + 1]] for g in range(len(starts) - 1)]
        units = len(members)
    if units < 2:
        return None
    values = []
    for _ in range(draws):
        picked = rng.integers(0, units, size=units)
        rows = picked if members is None else np.concatenate([members[g] for g in picked])
        value = statistic(rows)
        if not math.isnan(value):
            values.append(value)
    if not values:
        return None
    alpha = (1 - level) / 2
    low, high = np.quantile(values, [alpha, 1 - alpha])
    return Interval(
        low=float(low),
        high=float(high),
        level=level,
        draws=len(values),
        dropped=draws - len(values),
        by_group=groups is not None,
    )


# ---------------------------------------------------------------------------------------------
# Classification report


def classification_report(
    y_true: Array,
    y_pred: Array | None = None,
    proba: Array | None = None,
    labels: Sequence[Any] | None = None,
    y_train: Array | None = None,
    groups: Array | None = None,
    bootstrap: bool = True,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> ClassificationReport:
    """Every classification metric of § 7.1, the main one first (R-VQ-13). Predictions default to
    the most likely class when only probabilities are given."""
    t = np.asarray(y_true).ravel()
    if y_pred is None and proba is None:
        raise ValueError("Give predictions, probabilities or both.")
    order = _labels(t, None if y_pred is None else np.asarray(y_pred).ravel(), labels)
    p = None if proba is None else _proba(proba, len(order))
    if y_pred is None:
        assert p is not None
        pred = np.array([order[i] for i in p.argmax(axis=1)])
    else:
        pred = np.asarray(y_pred).ravel()
    cm = confusion_matrix(t, pred, order)
    precision, recall, f1, support = _prf(cm)
    report = ClassificationReport(
        labels=[str(v) for v in order],
        n=len(t),
        main=main_metric(t),
        accuracy=accuracy(t, pred),
        balanced_accuracy=balanced_accuracy(t, pred, order),
        f1_macro=float(np.mean(f1)),
        kappa=cohen_kappa(t, pred, order),
        per_class=[
            ClassScores(
                label=str(label),
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1=float(f1[i]),
                support=int(support[i]),
            )
            for i, label in enumerate(order)
        ],
        confusion=cm.tolist(),
        confusion_normalized=normalize_rows(cm).tolist(),
        baseline=majority_baseline(t, y_train, order),
    )
    if p is not None:
        report.roc_auc, report.roc_auc_by_class = roc_auc_ovr(t, p, order)
        report.pr_auc, report.pr_auc_by_class = pr_auc_ovr(t, p, order)
        missing = [str(v) for v in order if str(v) not in report.roc_auc_by_class]
        if len(order) > 2 and missing:
            report.notes.append(f"R-VM-05: no test image of {', '.join(missing)}: AUC left out.")
        report.log_loss = log_loss(t, p, order)
        report.brier = brier(t, p, order)
        report.calibration = calibration(t, p, order)
        report.top_k = {k: top_k_accuracy(t, p, k, order) for k in top_k_levels(len(order))}
    if bootstrap:
        report.intervals = _classification_intervals(t, pred, p, order, groups, draws, seed)
    return report


def _classification_intervals(
    t: np.ndarray,
    pred: np.ndarray,
    p: np.ndarray | None,
    order: list,
    groups: Array | None,
    draws: int,
    seed: int,
) -> dict[str, Interval]:
    codes, pcodes = _encode(t, order), _encode(pred, order)
    k = len(order)

    def cm_of(rows: np.ndarray) -> np.ndarray:
        return np.bincount(codes[rows] * k + pcodes[rows], minlength=k * k).reshape(k, k)

    def bal(rows: np.ndarray) -> float:
        cm = cm_of(rows)
        support = cm.sum(axis=1)
        present = support > 0
        return float(np.mean(np.diag(cm)[present] / support[present]))

    stats: dict[str, Callable[[np.ndarray], float]] = {
        "accuracy": lambda rows: float(np.mean(codes[rows] == pcodes[rows])),
        "balanced_accuracy": bal,
        "f1_macro": lambda rows: float(np.mean(_prf(cm_of(rows))[2])),
    }
    if p is not None:
        if k == 2:
            stats["roc_auc"] = lambda rows: roc_auc(codes[rows] == 1, p[rows, 1])
        else:

            def macro(rows: np.ndarray) -> float:
                values = [roc_auc(codes[rows] == i, p[rows, i]) for i in range(k)]
                kept = [v for v in values if not math.isnan(v)]
                return float(np.mean(kept)) if len(kept) == len(values) else math.nan

            stats["roc_auc"] = macro
    out: dict[str, Interval] = {}
    for offset, (name, fn) in enumerate(stats.items()):
        found = bootstrap_interval(fn, len(t), groups, draws, seed + offset)
        if found is not None:
            out[name] = found
    return out


# ---------------------------------------------------------------------------------------------
# Anomalies: image level


def at_threshold(is_defect: Array, scores: Array, threshold: float) -> ThresholdScores:
    y, s = _binary(is_defect), np.asarray(scores, dtype=float).ravel()
    flagged = s >= threshold
    tp = int(np.sum(flagged & y))
    fp = int(np.sum(flagged & ~y))
    fn = int(np.sum(~flagged & y))
    tn = int(np.sum(~flagged & ~y))
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return ThresholdScores(
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=_ratio(2 * precision * recall, precision + recall),
        false_alarm_rate=_ratio(fp, fp + tn),
    )


def auroc_by_type(defect_types: Array, scores: Array, normal: str = "good") -> dict[str, float]:
    """AUROC of each defect type against the normal images."""
    kinds = np.asarray(defect_types).astype(str).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    normals = kinds == normal
    out = {}
    for kind in sorted(set(kinds.tolist()) - {normal}):
        keep = normals | (kinds == kind)
        out[kind] = roc_auc(kinds[keep] == kind, s[keep])
    return out


def score_histogram(
    is_defect: Array, scores: Array, threshold: float | None = None, bins: int = 30
) -> Histogram:
    """Normal and defect scores on shared bins, with the threshold."""
    y, s = _binary(is_defect), np.asarray(scores, dtype=float).ravel()
    edges = np.histogram_bin_edges(s, bins=bins)
    return Histogram(
        edges=edges.tolist(),
        normal=np.histogram(s[~y], bins=edges)[0].tolist(),
        defect=np.histogram(s[y], bins=edges)[0].tolist(),
        threshold=threshold,
    )


# ---------------------------------------------------------------------------------------------
# Thresholds (R-VTH)


def threshold_f1_max(is_defect: Array, scores: Array) -> tuple[float, float]:
    """The threshold with the best F1 on these scores (e.g. a validation set), placed halfway
    between the last score kept and the next one, and that F1."""
    y, s = _binary(is_defect), np.asarray(scores, dtype=float).ravel()
    if not y.any():
        raise ValueError("F1 needs at least one defect.")
    precision, recall, thresholds = pr_curve(y, s)
    both = precision + recall
    f1 = np.divide(2 * precision * recall, both, out=np.zeros_like(both), where=both > 0)
    best = int(np.argmax(f1))
    cut = float(thresholds[best])
    if best + 1 < len(thresholds):
        cut = float((thresholds[best] + thresholds[best + 1]) / 2)
    return cut, float(f1[best])


def threshold_normal_quantile(normal_scores: Array, quantile: float = NORMAL_QUANTILE) -> float:
    s = np.asarray(normal_scores, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("No normal image held out.")
    return float(np.quantile(s, quantile))


def stratified_halves(is_defect: Array, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """R-VTH-01: two halves with the same share of defects (the first gets the extra image)."""
    y = _binary(is_defect)
    rng = np.random.default_rng(seed)
    first, second = [], []
    for value in (False, True):
        idx = rng.permutation(np.flatnonzero(y == value))
        cut = (len(idx) + 1) // 2
        first.append(idx[:cut])
        second.append(idx[cut:])
    return np.sort(np.concatenate(first)), np.sort(np.concatenate(second))


def choose_threshold(
    test_is_defect: Array | None = None,
    test_scores: Array | None = None,
    holdout_normal_scores: Array | None = None,
    user_threshold: float | None = None,
    seed: int = SEED,
) -> ThresholdChoice:
    """R-VTH-03 (a threshold set by hand) first, then R-VTH-01 (labelled defects), then R-VTH-02
    (held-out normals only)."""
    if user_threshold is not None:
        return ThresholdChoice(
            code="R-VTH-03",
            threshold=float(user_threshold),
            reason="The cost of a false alarm is a business call.",
            message=f"Threshold set by hand: {user_threshold:.4g} (R-VTH-03).",
        )
    if test_is_defect is not None and test_scores is not None:
        y = _binary(test_is_defect)
        if y.sum() >= 2 and (~y).sum() >= 2:
            s = np.asarray(test_scores, dtype=float).ravel()
            calib, evaluation = stratified_halves(y, seed)
            cut, f1 = threshold_f1_max(y[calib], s[calib])
            return ThresholdChoice(
                code="R-VTH-01",
                threshold=cut,
                reason="Choosing the threshold on the scored images would inflate F1.",
                message=(
                    f"Threshold {cut:.4g}: best F1 ({f1:.2f}) on half of the test images "
                    f"({len(calib)}, stratified, seed {seed}); the counts come from the other "
                    f"{len(evaluation)} (R-VTH-01)."
                ),
                calibration=calib.tolist(),
                evaluation=evaluation.tolist(),
            )
    if holdout_normal_scores is not None and np.asarray(holdout_normal_scores).size:
        normals = np.asarray(holdout_normal_scores, dtype=float).ravel()
        cut = threshold_normal_quantile(normals)
        rough = (
            f" Only {normals.size} held-out normal images: this quantile is rough."
            if normals.size < FEW_NORMALS
            else ""
        )
        return ThresholdChoice(
            code="R-VTH-02",
            threshold=cut,
            reason="No defect to calibrate on: the false alarm rate is fixed instead.",
            message=(
                f"Threshold {cut:.4g}: 99% quantile of {normals.size} held-out normal images, "
                f"about 1 false alarm in 100 normal images (R-VTH-02).{rough}"
            ),
        )
    raise ValueError("A threshold needs labelled defects, held-out normal images or a value.")


# ---------------------------------------------------------------------------------------------
# Anomalies: pixel level


def pixel_auroc(masks: Array, maps: Array) -> float:
    return roc_auc(np.asarray(masks).astype(bool).ravel(), np.asarray(maps, dtype=float).ravel())


def _stack(masks: Array, maps: Array) -> tuple[np.ndarray, np.ndarray]:
    gt = np.asarray(masks).astype(bool)
    am = np.asarray(maps, dtype=float)
    if gt.ndim == 2:
        gt, am = gt[None], am[None]
    if gt.shape != am.shape:
        raise ValueError(f"Masks {gt.shape} and maps {am.shape} differ in shape.")
    return gt, am


def pro_curve(masks: Array, maps: Array) -> tuple[np.ndarray, np.ndarray, int]:
    """(fpr, pro, regions), the official MVTec computation (Bergmann et al., IJCV 2021): defect
    regions are 8-connected per image; every defect pixel weighs 1 / (its region's size x number
    of regions), every normal pixel 1 / (number of normal pixels)."""
    from scipy import ndimage

    gt, am = _stack(masks, maps)
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1] = 1  # 8-connected within an image, never across images
    labelled = ndimage.label(gt, structure=structure)
    regions, count = cast(tuple[np.ndarray, int], labelled)
    ok = ~gt
    n_ok = int(ok.sum())
    fp_change = np.zeros(gt.shape, dtype=float)
    if n_ok:
        fp_change[ok] = 1.0 / n_ok
    pro_change = np.zeros(gt.shape, dtype=float)
    if count:
        sizes = np.bincount(regions.ravel(), minlength=count + 1).astype(float)
        weights = np.zeros(count + 1)
        weights[1:] = 1.0 / (sizes[1:] * count)
        pro_change = weights[regions]
    order = np.argsort(-am.ravel(), kind="mergesort")
    s = am.ravel()[order]
    fpr = np.cumsum(fp_change.ravel()[order])
    pro = np.cumsum(pro_change.ravel()[order])
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    fpr = np.clip(np.r_[0.0, fpr[last]], 0, 1)
    pro = np.clip(np.r_[0.0, pro[last]], 0, 1)
    return fpr, pro, int(count)


def _area_until(x: np.ndarray, y: np.ndarray, x_max: float) -> float:
    """Trapezoid area under y(x) for x <= x_max, interpolating y at x_max (MVTec's `trapezoid`)."""
    correction = 0.0
    if x_max not in x:
        ins = int(np.searchsorted(x, x_max, side="left"))
        if 0 < ins < len(x):
            y_at = y[ins - 1] + (y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
            correction = 0.5 * (y_at + y[ins - 1]) * (x_max - x[ins - 1])
    keep = x <= x_max
    x, y = x[keep], y[keep]
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction)


def aupro(masks: Array, maps: Array, limit: float = AUPRO_LIMIT) -> float:
    """Area under the PRO curve up to `limit` false positive rate, divided by `limit` (1 = every
    defect region fully found before any false alarm)."""
    fpr, pro, count = pro_curve(masks, maps)
    if not count:
        return math.nan
    return _area_until(fpr, pro, limit) / limit


def pixel_at_threshold(masks: Array, maps: Array, threshold: float) -> tuple[float, float]:
    """(F1, IoU) of the pixels at or above the threshold against the masks."""
    gt, am = _stack(masks, maps)
    flagged = am >= threshold
    tp = float(np.sum(flagged & gt))
    fp = float(np.sum(flagged & ~gt))
    fn = float(np.sum(~flagged & gt))
    return _ratio(2 * tp, 2 * tp + fp + fn), _ratio(tp, tp + fp + fn)


def pixel_report(
    masks: Array, maps: Array, threshold: float | None = None, limit: float = AUPRO_LIMIT
) -> PixelReport:
    gt, am = _stack(masks, maps)
    fpr, pro, count = pro_curve(gt, am)
    out = PixelReport(
        auroc=pixel_auroc(gt, am),
        aupro=_area_until(fpr, pro, limit) / limit if count else math.nan,
        limit=limit,
        regions=count,
    )
    if threshold is not None:
        out.threshold = float(threshold)
        out.f1, out.iou = pixel_at_threshold(gt, am, threshold)
    return out


# ---------------------------------------------------------------------------------------------
# Anomaly report


def anomaly_report(
    is_defect: Array,
    scores: Array,
    defect_types: Array | None = None,
    holdout_normal_scores: Array | None = None,
    user_threshold: float | None = None,
    masks: Array | None = None,
    maps: Array | None = None,
    pixel_threshold: float | None = None,
    groups: Array | None = None,
    bootstrap: bool = True,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> AnomalyReport:
    """Every anomaly metric of § 7.2 on the test images: AUROC (main) and AP, the threshold and
    its rule (R-VTH), counts at the threshold (on the evaluation half under R-VTH-01), AUROC per
    defect type, the score histogram and, with masks, the pixel metrics."""
    y, s = _binary(is_defect), np.asarray(scores, dtype=float).ravel()
    notes: list[str] = []
    choice: ThresholdChoice | None
    try:
        choice = choose_threshold(y, s, holdout_normal_scores, user_threshold, seed)
    except ValueError as error:
        choice = None
        notes.append(str(error))
    counts = None
    if choice is not None:
        rows = np.asarray(choice.evaluation, dtype=int) if choice.evaluation else np.arange(len(y))
        counts = at_threshold(y[rows], s[rows], choice.threshold)
    report = AnomalyReport(
        n_normal=int((~y).sum()),
        n_defect=int(y.sum()),
        auroc=roc_auc(y, s),
        ap=average_precision(y, s),
        threshold=choice,
        at_threshold=counts,
        histogram=score_histogram(y, s, None if choice is None else choice.threshold),
        notes=notes,
    )
    if defect_types is not None:
        report.auroc_by_type = auroc_by_type(defect_types, s)
    if bootstrap:
        report.auroc_interval = bootstrap_interval(
            lambda rows: roc_auc(y[rows], s[rows]), len(y), groups, draws, seed
        )
    if masks is not None and maps is not None:
        report.pixel = pixel_report(masks, maps, pixel_threshold)
    return report
