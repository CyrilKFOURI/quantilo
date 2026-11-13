"""Linear probe and cosine kNN on image vectors (114; study § 6.1, § 6.3, § 6.4).

numpy and scikit-learn only: no torch, so the probe also runs on the classic descriptors (103)
and in the browser version. Scored by `vision.metrics` (111): every metric of § 7.1, the main
one first, bootstrap intervals and the majority baseline.

- R-VDP-06: the producer's split when there is one (`train` and `val` train, `test` scores);
  otherwise 20 % of test images, stratified by label and grouped by `pixel_sha` (exact copies on
  the same side), seed 0.
- R-VDP-07: linear probe = standardisation + multinomial logistic regression, C chosen by an
  internal stratified cross-validation (at most 5 folds) on balanced accuracy.
- R-VDP-08: kNN = cosine similarity, k = 20 (at most the training images), votes weighted by
  exp(similarity / 0.07) (Wu et al. 2018, as in DINO and DINOv2); the 5 nearest training images
  of each test image are returned: a prediction explained by cases (§ 6.4)."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel

from quantilo.vision import metrics as vm

if TYPE_CHECKING:
    from quantilo.vision.deep.embed import Embeddings
    from quantilo.vision.ingest import ImageSet

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
FOLDS = 5
K_NEIGHBOURS = 20
TEMPERATURE = 0.07
SHOWN_NEIGHBOURS = 5
TEST_SHARE = 0.2
MIN_PER_CLASS = 2
SEED = 0
Array = Any  # array-like: list, numpy array, pyarrow column as a list
TRAIN_SPLITS = ("train", "val", "validation", "dev")
TEST_SPLITS = ("test",)


class ProbeSplit(BaseModel):
    train: list[int]
    test: list[int]
    code: str = "R-VDP-06"
    reason: str


class ProbeResult(BaseModel):
    model: Literal["linear", "knn"]
    title: str
    setting: str  # "C = 0.1" or "k = 20, cosine"
    report: vm.ClassificationReport
    predictions: list[str]
    neighbours: list[list[int]] = []  # kNN: nearest training rows (in the caller's numbering)
    notes: list[str] = []


class ProbeReport(BaseModel):
    backbone: str
    split: ProbeSplit
    linear: ProbeResult
    knn: ProbeResult
    sentence: str


def _score(report: vm.ClassificationReport) -> float:
    return (
        report.balanced_accuracy if report.main.metric == "balanced_accuracy" else report.accuracy
    )


def _interval_text(report: vm.ClassificationReport) -> str:
    key = report.main.metric
    value = _score(report)
    interval = report.intervals.get(key)
    if interval is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{interval.low:.3f}; {interval.high:.3f}]"


def choose_split(
    labels: Array,
    splits: Array | None = None,
    groups: Array | None = None,
    seed: int = SEED,
) -> ProbeSplit:
    """R-VDP-06 on the rows that have a label."""
    y = np.asarray([None if v is None else str(v) for v in labels], dtype=object)
    labelled = np.flatnonzero(np.array([v is not None and v != "" for v in y]))
    if splits is not None:
        s = np.asarray([None if v is None else str(v).lower() for v in splits], dtype=object)
        train = [int(i) for i in labelled if s[i] in TRAIN_SPLITS]
        test = [int(i) for i in labelled if s[i] in TEST_SPLITS]
        if train and test:
            used = sorted({str(s[i]) for i in train})
            return ProbeSplit(
                train=train,
                test=test,
                reason=(
                    f"The producer's split: trained on {' + '.join(used)} ({len(train):,} images), "
                    f"scored on test ({len(test):,}) (R-VDP-06)."
                ),
            )
    from sklearn.model_selection import StratifiedGroupKFold

    g = (
        np.asarray([str(v) for v in groups], dtype=object)[labelled]
        if groups is not None
        else np.arange(len(labelled))
    )
    folds = round(1 / TEST_SHARE)
    _, counts = np.unique(y[labelled].astype(str), return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("A probe needs at least 2 labels with 2 images each (R-VDP-06).")
    splitter = StratifiedGroupKFold(min(folds, int(counts.min())), shuffle=True, random_state=seed)
    train_at, test_at = next(splitter.split(labelled, y[labelled].astype(str), g))
    return ProbeSplit(
        train=[int(labelled[i]) for i in train_at],
        test=[int(labelled[i]) for i in test_at],
        reason=(
            f"No producer's split: {len(test_at):,} test images "
            f"({100 * len(test_at) / len(labelled):.0f} %) drawn stratified by label, exact "
            f"copies kept together (pixel_sha), seed {seed} (R-VDP-06)."
        ),
    )


def _check(x_train: np.ndarray, y_train: np.ndarray) -> None:
    labels, counts = np.unique(y_train, return_counts=True)
    if len(labels) < 2:
        raise ValueError("The probe needs at least 2 labels in the training images.")
    if counts.min() < MIN_PER_CLASS:
        rare = [str(v) for v, c in zip(labels, counts, strict=True) if c < MIN_PER_CLASS]
        raise ValueError(
            f"Too few training images for {', '.join(rare)} (fewer than {MIN_PER_CLASS})."
        )
    if not np.isfinite(x_train).all():
        raise ValueError("Some training vectors are missing (NaN): leave those images out.")


def linear_probe(
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    *,
    groups: Any = None,
    bootstrap: bool = True,
    seed: int = SEED,
) -> ProbeResult:
    """R-VDP-07: standardised logistic regression, C by internal cross-validation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    xtr = np.asarray(x_train, dtype=np.float64)
    ytr = np.asarray(y_train).astype(str)
    xte = np.asarray(x_test, dtype=np.float64)
    yte = np.asarray(y_test).astype(str)
    _check(xtr, ytr)
    _, counts = np.unique(ytr, return_counts=True)
    folds = int(min(FOLDS, counts.min()))
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    search = GridSearchCV(
        pipeline,
        {"logisticregression__C": list(C_GRID)},
        scoring="balanced_accuracy",
        cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
        refit=True,
    )
    search.fit(xtr, ytr)
    c = float(search.best_params_["logisticregression__C"])
    classes = [str(v) for v in search.classes_]
    if unseen := sorted(set(yte.tolist()) - set(classes)):
        raise ValueError(f"No training image for {', '.join(unseen)}: the probe cannot score it.")
    proba = search.predict_proba(xte)
    report = vm.classification_report(
        yte, proba=proba, labels=classes, y_train=ytr, groups=groups, bootstrap=bootstrap, seed=seed
    )
    notes = [
        f"C = {c:g} chosen among {', '.join(f'{v:g}' for v in C_GRID)} by {folds}-fold "
        f"cross-validation on the training images (balanced accuracy "
        f"{search.best_score_:.3f}) (R-VDP-07)."
    ]
    if c in (C_GRID[0], C_GRID[-1]):
        notes.append(f"C = {c:g} is at the edge of the grid: a wider grid might do better.")
    predictions = [classes[i] for i in proba.argmax(axis=1)]
    return ProbeResult(
        model="linear",
        title="Linear probe (logistic regression)",
        setting=f"C = {c:g}",
        report=report,
        predictions=predictions,
        notes=notes,
    )


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def knn_proba(
    x_train: np.ndarray,
    coded: np.ndarray,
    n_classes: int,
    x_test: np.ndarray,
    k: int = K_NEIGHBOURS,
    temperature: float = TEMPERATURE,
    block: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """R-VDP-08 on unit vectors: (probabilities, the 5 nearest training rows of each test row).
    `coded` holds each training row's class number."""
    kk = int(min(k, len(x_train)))
    proba = np.zeros((len(x_test), n_classes))
    nearest = np.zeros((len(x_test), min(SHOWN_NEIGHBOURS, kk)), dtype=int)
    for start in range(0, len(x_test), block):
        sims = x_test[start : start + block] @ x_train.T
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        top_sims = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_sims, axis=1)
        top = np.take_along_axis(top, order, axis=1)
        top_sims = np.take_along_axis(top_sims, order, axis=1)
        weights = np.exp((top_sims - top_sims[:, :1]) / temperature)  # stable softmax weights
        for c in range(n_classes):
            proba[start : start + len(top), c] = (weights * (coded[top] == c)).sum(axis=1)
        nearest[start : start + len(top)] = top[:, : nearest.shape[1]]
    proba /= proba.sum(axis=1, keepdims=True)
    return proba, nearest


def knn_probe(
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    *,
    k: int = K_NEIGHBOURS,
    temperature: float = TEMPERATURE,
    train_rows: Sequence[int] | None = None,
    groups: Any = None,
    bootstrap: bool = True,
    seed: int = SEED,
    block: int = 1024,
) -> ProbeResult:
    """R-VDP-08: weighted cosine kNN; `neighbours` are given in `train_rows` numbering."""
    xtr = _normalise(np.asarray(x_train, dtype=np.float64))
    ytr = np.asarray(y_train).astype(str)
    xte = _normalise(np.asarray(x_test, dtype=np.float64))
    yte = np.asarray(y_test).astype(str)
    _check(xtr, ytr)
    classes = sorted(set(ytr.tolist()) | set(yte.tolist()))
    index = {label: i for i, label in enumerate(classes)}
    coded = np.array([index[v] for v in ytr])
    kk = int(min(k, len(ytr)))
    proba, nearest = knn_proba(xtr, coded, len(classes), xte, kk, temperature, block)
    report = vm.classification_report(
        yte, proba=proba, labels=classes, y_train=ytr, groups=groups, bootstrap=bootstrap, seed=seed
    )
    mapping = np.asarray(train_rows) if train_rows is not None else np.arange(len(ytr))
    return ProbeResult(
        model="knn",
        title="Nearest neighbours (cosine)",
        setting=f"k = {kk}, cosine, temperature {temperature:g}",
        report=report,
        predictions=[classes[i] for i in proba.argmax(axis=1)],
        neighbours=mapping[nearest].tolist(),
        notes=[
            f"Each test image takes the labels of its {kk} most similar training images, "
            f"weighted by exp(similarity / {temperature:g}) (R-VDP-08)."
        ],
    )


def probe_embeddings(
    vectors: Array,
    labels: Array,
    splits: Array | None = None,
    groups: Array | None = None,
    *,
    backbone: str = "vectors",
    bootstrap: bool = True,
    seed: int = SEED,
) -> ProbeReport:
    """Both probes on one split (R-VDP-06). Rows without a vector (NaN) or a label are left out
    of training and scoring."""
    x = np.asarray(vectors, dtype=np.float64)
    finite = np.isfinite(x).all(axis=1)
    usable = [v if finite[i] else None for i, v in enumerate(labels)]
    split = choose_split(usable, splits, groups, seed)
    y = np.asarray([str(v) for v in labels], dtype=object)
    tr, te = np.asarray(split.train), np.asarray(split.test)
    g = None if groups is None else np.asarray([str(v) for v in groups], dtype=object)[te]
    common = {"groups": g, "bootstrap": bootstrap, "seed": seed}
    linear = linear_probe(x[tr], y[tr], x[te], y[te], **common)
    knn = knn_probe(x[tr], y[tr], x[te], y[te], train_rows=split.train, **common)
    metric = linear.report.main.metric.replace("_", " ")
    baseline = linear.report.baseline
    floor = baseline.balanced_accuracy if metric == "balanced accuracy" else baseline.accuracy
    sentence = (
        f"Linear probe on {backbone}: {metric} {_interval_text(linear.report)}; "
        f"kNN {_interval_text(knn.report)}; majority baseline {floor:.3f}."
    )
    return ProbeReport(backbone=backbone, split=split, linear=linear, knn=knn, sentence=sentence)


def probe_image_set(
    images: "ImageSet",
    embeddings: "Embeddings",
    *,
    bootstrap: bool = True,
    seed: int = SEED,
) -> ProbeReport:
    """Both probes on an imported image set and its vectors (`embed_image_set`): labels and the
    producer's split from the table, exact copies grouped by `pixel_sha`."""
    table = images.table
    return probe_embeddings(
        embeddings.vectors,
        table["label"].to_pylist(),
        table["split"].to_pylist(),
        [v or f"row{i}" for i, v in enumerate(table["pixel_sha"].to_pylist())],
        backbone=embeddings.title,
        bootstrap=bootstrap,
        seed=seed,
    )
