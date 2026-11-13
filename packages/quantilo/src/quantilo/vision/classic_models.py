"""Image models of the first plan (112; study §§ 6.1 to 6.3, 7, 10): classic descriptors everywhere,
the frozen DINOv2 probe on the desktop when the extra `vision` and its weights are there.

numpy, scipy and scikit-learn only, no thread, no process: the classic models run in the browser
version (Pyodide). torch is imported only for the deep vectors, and only when `deep.status()` says
they can be computed (R-VDP-01). Every metric comes from `vision.metrics` (111).

Classification (R-VML-01):
- R-VCL-01: classic descriptors (103: HSV 72 + HOG 1,764 + LBP 236) standardised, then a PCA
  fitted on the training images only keeps 95 % of the variance (a PCA to 50 cost the logistic
  15 points of balanced accuracy on Beans).
- R-VCL-02: logistic regression on those components, C among 0.001 to 10 by a grouped inner
  cross-validation on balanced accuracy. R-VCL-03: histogram gradient boosting on the first 50
  components, leaves of at least min(20, n / 10) images, seed 0.
- R-VCL-04: the better classic model by inner balanced accuracy, never by the test.
- R-VCL-05: `class_weight="balanced"` when the majority / minority ratio of training exceeds 3
  (R-VQ-13).
- R-VCL-06: with the extra and the DINOv2-S/14 weights, the linear probe on the frozen vectors
  (R-VDP-07) is the desktop default, the cosine kNN (R-VDP-08) beside it; the classic models stay
  shown as the floor.
- R-VCL-07: in the browser, at most 2,000 images, classic models only (R-VIS-WEB-01).

Anomalies (R-VML-02):
- R-VAN-01: PCA of patches: luma at 256 x 256, 16 x 16 patches, PCA of the training normals'
  patches (at most 100,000 drawn) keeping 95 % of the variance; the reconstruction error of each
  patch is read as a robust z against the normals' errors at the same place; image score = 99 %
  quantile of its patches; map = those z enlarged and smoothed.
- R-VAN-02: kNN on descriptors: standardisation + PCA 50 fitted on the normals, mean distance to
  the 5 nearest training normals.
- R-VAN-03: the ensemble (default) averages the percentile ranks of both scores.
- R-VAN-04: with the extra and the weights, the same kNN on the DINOv2 vectors, shown beside.
- R-VAN-05: pixel metrics on maps and masks at 128 x 128, on at most 1,000 test images.

Clusters (R-VML-03): R-VCU-01: k-means on the PCA-50 descriptors, k from 2 to 12 by silhouette.
"""

import math
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyarrow as pa
from PIL import Image
from pydantic import BaseModel

from quantilo.vision import browser as web
from quantilo.vision import metrics as vm
from quantilo.vision import vectors as cache
from quantilo.vision.decode import DecodeFailure, try_decode
from quantilo.vision.descriptors import CLASSIC_SIZE, classic_descriptor
from quantilo.vision.quality import QualityFinding, possibly_mislabelled
from quantilo.vision.split import SplitPlan, groups_of, normal_flags, plan_split
from quantilo.vision.target_models import ImageMultilabelReport, ImageRegressionReport
from quantilo.vision.task import Task, TaskChoice, column, detect_task, labels_of, usable_rows

if TYPE_CHECKING:
    from quantilo.vision.ingest import ImageSet

SEED = 0
DESCRIPTOR_KEY = f"classic-v1|hsv-hog-lbp|{CLASSIC_SIZE}|512px"
DECODE_SIDE = 512  # descriptors are measured on the 512-pixel working copy (103)
PCA_DIMS = 50  # R-VCL-03 (HGB), R-VAN-02
SPACE_VARIANCE = 0.95  # R-VCL-01: PCA 50 lost 15 points of balanced accuracy on Beans
SPACE_FIT_ROWS = 5_000
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)  # R-VCL-02
INNER_FOLDS = 5
HGB_INNER_FOLDS = 3
IMBALANCE = 3.0  # R-VCL-05 (R-VQ-13)
PATCH_SIDE = 256  # R-VAN-01
PATCH = 16
PATCH_VARIANCE = 0.95
PATCH_QUANTILE = 0.99
MAX_PATCHES = 100_000
CALIBRATION_IMAGES = 1_000  # normal images whose errors set each position's median and MAD
MAP_SIGMA = 4.0  # pixels at PATCH_SIDE
ANOMALY_K = 5  # R-VAN-02
PIXEL_SIDE = 128  # R-VAN-05
PIXEL_MAX_IMAGES = 1_000
CLUSTER_KS = range(2, 13)  # R-VCU-01
SILHOUETTE_SAMPLE = 2_000
BROWSER_MAX_IMAGES = web.MAX_IMAGES  # R-VCL-07 (R-VIS-WEB-01)
DEEP_BACKBONE = "dinov2_s14"
GALLERY = 24

Source = str | Path | bytes | Image.Image | None
Progress = Callable[[float, str], None]
Cancel = Callable[[], bool]


class Cancelled(RuntimeError):
    """Raised at a checkpoint when `cancel()` turned true: nothing is kept."""


# ---------------------------------------------------------------------------------------------
# Results


class VisionModelResult(BaseModel):
    key: str  # logistic, hgb, deep_linear, deep_knn, patch_pca, knn, ensemble, deep_knn_anomaly
    title: str
    family: Literal["classic", "deep"]
    setting: str
    default: bool = False
    selection: float | None = None  # inner cross-validated balanced accuracy (R-VCL-04)
    classification: vm.ClassificationReport | None = None
    anomaly: vm.AnomalyReport | None = None
    regression: ImageRegressionReport | None = None  # c3, R-VRG-02
    multilabel: ImageMultilabelReport | None = None  # c3, R-VMLB-02
    threshold: vm.ThresholdChoice | None = None  # anomalies without defects to score against
    flagged: int | None = None  # images at or above that threshold
    galleries: dict[str, list[int]] = {}  # table rows: errors, uncertain, unusual...
    seconds: float = 0.0
    notes: list[str] = []
    # 113: anomaly maps kept with the run (8-bit grey PNGs scaled on this range), the model whose
    # maps are shown (the ensemble shows the patch PCA's), and the autoencoders' reconstructions
    map_range: tuple[float, float] | None = None
    map_source: str | None = None
    reconstructions: bool = False
    # 118: Advanced > More models (never the default), what their training did (one log per
    # fold), the concepts of a bottleneck and their quality, the distinct prototypes kept
    level: Literal["first", "advanced", "more"] = "first"
    training: list[dict[str, Any]] | None = None
    concepts: list[dict[str, Any]] | None = None
    prototypes: int | None = None


class ClusterResult(BaseModel):
    k: int
    silhouette: float
    by_k: dict[int, float]
    sizes: dict[int, int]
    medoids: dict[int, int]  # cluster -> table row of its most typical image
    adjusted_rand: float | None = None
    sentence: str


class FeatureInfo(BaseModel):
    key: str  # classic, dinov2_s14
    title: str
    dims: int
    computed: int  # images computed in this run (the others came from the cache)
    failed: int
    seconds: float
    notes: list[str] = []


class DeepOption(BaseModel):
    used: bool
    backbone: str = DEEP_BACKBONE
    reason: str | None = None  # why it is not used (R-VDP-01, R-VDP-02), or None


class VisionResult(BaseModel):
    task: TaskChoice
    split: SplitPlan
    deep: DeepOption
    features: list[FeatureInfo]
    models: list[VisionModelResult] = []
    default: str | None = None
    clusters: ClusterResult | None = None
    sentence: str
    seconds: float
    seed: int = SEED
    notes: list[str] = []
    # R-VQ-16 (Advanced, Quality tab): images whose label the default model, out of fold, finds
    # unlikely (classification only)
    mislabelled: QualityFinding | None = None

    def model(self, key: str) -> VisionModelResult:
        return next(m for m in self.models if m.key == key)


@dataclass
class Outcome:
    """The result (JSON-ready) and one row per scored image (`predictions`)."""

    result: VisionResult
    predictions: pa.Table
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Small helpers


class _Steps:
    """Progress over the stages of a run: each stage owns a slice of [0, 1]."""

    def __init__(self, progress: Progress | None, cancel: Cancel | None) -> None:
        self.progress = progress
        self.cancel = cancel
        self.low, self.high = 0.0, 1.0

    def stage(self, low: float, high: float) -> None:
        self.low, self.high = low, high

    def check(self) -> None:
        if self.cancel is not None and self.cancel():
            raise Cancelled("Stopped: nothing was saved.")

    def __call__(self, share: float, message: str) -> None:
        self.check()
        if self.progress is not None:
            self.progress(self.low + (self.high - self.low) * min(1.0, max(0.0, share)), message)


@contextmanager
def _quiet() -> Iterator[None]:
    """scikit-learn's convergence and fold warnings are said by our notes instead."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def _decode(source: Source, side: int = DECODE_SIDE) -> Image.Image | None:
    if source is None:
        return None
    if isinstance(source, Image.Image):
        return source
    decoded = try_decode(source, reduce_to=side)
    return None if isinstance(decoded, DecodeFailure) else decoded.image


def _in_browser() -> bool:
    return web.active()


# ---------------------------------------------------------------------------------------------
# Features


def classic_features(
    sources: Sequence[Source],
    image_ids: Sequence[str] | None = None,
    cache_path: str | Path | None = None,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
) -> tuple[np.ndarray, FeatureInfo]:
    """Classic descriptors (103) of every source, in order: `(n, 2,072)` float32, NaN rows for the
    images that cannot be decoded. With `cache_path` and `image_ids`, images already computed are
    read from the cache and the new ones are added to it."""
    started = time.perf_counter()
    n = len(sources)
    out = np.full((n, CLASSIC_SIZE), np.nan, dtype=np.float32)
    known = cache.load(cache_path, DESCRIPTOR_KEY) if cache_path and image_ids else {}
    todo = []
    for i in range(n):
        hit = known.get(str(image_ids[i])) if image_ids is not None else None
        if hit is not None and hit.shape == (CLASSIC_SIZE,):
            out[i] = hit
        else:
            todo.append(i)
    failed = 0
    for done, i in enumerate(todo):
        if done % 25 == 0:
            if cancel is not None and cancel():
                raise Cancelled("Stopped: nothing was saved.")
            if progress is not None:
                progress(done / max(1, len(todo)), f"Descriptors {done:,} / {len(todo):,} images")
        image = _decode(sources[i])
        if image is None:
            failed += 1
            continue
        out[i] = classic_descriptor(image).vector
    if cache_path and image_ids is not None and todo:
        known_rows = {str(image_ids[i]): out[i] for i in range(n) if np.isfinite(out[i]).all()}
        for key, vec in cache.load(cache_path, DESCRIPTOR_KEY).items():
            known_rows.setdefault(key, vec)
        ids = list(known_rows)
        cache.save(cache_path, DESCRIPTOR_KEY, ids, np.stack([known_rows[k] for k in ids]))
    info = FeatureInfo(
        key="classic",
        title="Classic descriptors (colour histogram, HOG, LBP)",
        dims=CLASSIC_SIZE,
        computed=len(todo) - failed,
        failed=failed,
        seconds=round(time.perf_counter() - started, 2),
    )
    return out, info


def deep_option(deep: bool | None) -> DeepOption:
    """Whether the frozen DINOv2 vectors can be used here (R-VCL-06): the extra is installed and
    the weights are downloaded. Never downloads; cheap (no torch import)."""
    if deep is False:
        return DeepOption(used=False, reason="Deep vectors not asked for.")
    if _in_browser():
        from quantilo.vision.deep import BROWSER

        return DeepOption(used=False, reason=BROWSER)
    from quantilo.vision import deep as vision_deep

    status = vision_deep.status()
    backbone = status.backbone(DEEP_BACKBONE)
    if not backbone.available:
        return DeepOption(used=False, reason=backbone.reason)
    return DeepOption(used=True)


def deep_features(
    sources: Sequence[Source],
    image_ids: Sequence[str] | None = None,
    cache_path: str | Path | None = None,
    *,
    allow_gpu: bool = True,
    confirm_long: bool = False,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
) -> tuple[np.ndarray, FeatureInfo]:
    """DINOv2-S/14 vectors (114) of every source, NaN rows for failures, cached like the
    descriptors (the key names the network, its weights and its input size)."""
    from quantilo.vision.deep.backbones import load
    from quantilo.vision.deep.embed import Cancelled as EmbedCancelled
    from quantilo.vision.deep.embed import embed_images

    started = time.perf_counter()
    net = load(DEEP_BACKBONE)
    key = net.model_key
    n = len(sources)
    out = np.full((n, net.spec.dims), np.nan, dtype=np.float32)
    known = cache.load(cache_path, key) if cache_path and image_ids else {}
    todo = []
    for i in range(n):
        hit = known.get(str(image_ids[i])) if image_ids is not None else None
        if hit is not None and hit.shape == (net.spec.dims,):
            out[i] = hit
        elif sources[i] is not None:
            todo.append(i)
    notes: list[str] = []
    failed = 0
    if todo:

        def tick(done: int, total: int) -> None:
            if progress is not None:
                progress(done / max(1, total), f"{net.spec.title}: {done:,} / {total:,} images")

        try:
            found = embed_images(
                [sources[i] for i in todo],  # type: ignore[misc]
                net,
                allow_gpu=allow_gpu,
                confirm_long=confirm_long,
                progress=tick,
                cancel=cancel,
            )
        except EmbedCancelled as error:
            raise Cancelled("Stopped: nothing was saved.") from error
        out[todo] = found.vectors
        failed = len(found.failures)
        notes = [*found.notes, found.sentence] + ([found.estimate] if found.estimate else [])
        if cache_path and image_ids is not None:
            rows = {str(image_ids[i]): out[i] for i in range(n) if np.isfinite(out[i]).all()}
            for k, vec in cache.load(cache_path, key).items():
                rows.setdefault(k, vec)
            ids = list(rows)
            cache.save(cache_path, key, ids, np.stack([rows[k] for k in ids]))
    info = FeatureInfo(
        key=DEEP_BACKBONE,
        title=f"{net.spec.title} (frozen)",
        dims=net.spec.dims,
        computed=len(todo) - failed,
        failed=failed,
        seconds=round(time.perf_counter() - started, 2),
        notes=notes,
    )
    return out, info


# ---------------------------------------------------------------------------------------------
# Classification


def _class_weight(y: np.ndarray) -> str | None:
    """R-VCL-05 (R-VQ-13)."""
    _, counts = np.unique(y, return_counts=True)
    return "balanced" if counts.max() / counts.min() > IMBALANCE else None


def _inner(y: np.ndarray, groups: np.ndarray, folds: int, seed: int) -> Any:
    from sklearn.model_selection import StratifiedGroupKFold

    _, counts = np.unique(y, return_counts=True)
    k = int(min(folds, counts.min(), len(np.unique(groups))))
    if k < 2:
        raise ValueError(
            "Each class needs at least 2 training images, in 2 groups of copies (R-VML-01)."
        )
    return StratifiedGroupKFold(k, shuffle=True, random_state=seed)


def _pca_dims(n_train: int, n_features: int) -> int:
    return int(max(1, min(PCA_DIMS, n_features, n_train // 2)))


def descriptor_space(
    x_train: np.ndarray, seed: int = SEED, variance: float = SPACE_VARIANCE
) -> Any:
    """R-VCL-01: standardisation + PCA keeping `variance` of the variance, fitted on the training
    rows only (at most 5,000 of them, drawn with the seed). Components come sorted: the first 50
    are the HGB's input."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = x_train
    if len(rows) > SPACE_FIT_ROWS:
        rng = np.random.default_rng(seed)
        rows = rows[rng.choice(len(rows), SPACE_FIT_ROWS, replace=False)]
    pipeline = make_pipeline(StandardScaler(), PCA(variance, svd_solver="full", random_state=seed))
    with _quiet():
        pipeline.fit(rows)
    return pipeline


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    scale: bool = False,
    seed: int = SEED,
) -> tuple[Any, float, float]:
    """R-VCL-02 (R-VDP-07 with `scale`): (fitted model, C, inner balanced accuracy). `x` is the
    descriptor space (`descriptor_space`) or, with `scale`, raw vectors standardised here."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps: list[Any] = [StandardScaler()] if scale else []
    steps.append(LogisticRegression(max_iter=3000, class_weight=_class_weight(y)))
    search = GridSearchCV(
        make_pipeline(*steps),
        {"logisticregression__C": list(C_GRID)},
        scoring="balanced_accuracy",
        cv=_inner(y, groups, INNER_FOLDS, seed),
        refit=True,
    )
    with _quiet():
        search.fit(x, y, groups=groups)
    c = float(search.best_params_["logisticregression__C"])
    return search.best_estimator_, c, float(search.best_score_)


def fit_hgb(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int = SEED
) -> tuple[Any, float]:
    """R-VCL-03 on the first 50 components: (fitted model, inner balanced accuracy on 3 grouped
    folds)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    z = x[:, :PCA_DIMS]

    leaf = int(max(2, min(20, len(z) // 10)))  # the default 20 allows no split below 40 images

    def build() -> Any:
        return HistGradientBoostingClassifier(
            min_samples_leaf=leaf, random_state=seed, class_weight=_class_weight(y)
        )

    with _quiet():
        scores = cross_val_score(
            build(), z, y, groups=groups, cv=_inner(y, groups, HGB_INNER_FOLDS, seed),
            scoring="balanced_accuracy",
        )  # fmt: skip
        model = build().fit(z, y)
    return model, float(np.mean(scores))


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _proba_in(model: Any, x: np.ndarray, classes: list[str]) -> np.ndarray:
    """Probabilities in the order of `classes` (a class absent from a fold's training gets 0)."""
    raw = model.predict_proba(x)
    order = [str(c) for c in model.classes_]
    out = np.zeros((len(x), len(classes)))
    for j, c in enumerate(order):
        out[:, classes.index(c)] = raw[:, j]
    return out


@dataclass
class _Scored:
    proba: np.ndarray
    selection: list[float] = field(default_factory=list)
    settings: list[str] = field(default_factory=list)
    neighbours: np.ndarray | None = None
    seconds: float = 0.0


def classify(
    plan: SplitPlan,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    classic: np.ndarray,
    deep: np.ndarray | None,
    steps: _Steps,
    *,
    seed: int = SEED,
    bootstrap: bool = True,
) -> tuple[list[VisionModelResult], dict[str, Any]]:
    """Every classification model on every fold; out-of-fold probabilities, then one report per
    model on the scored images (`plan.scored`)."""
    scored = np.asarray(plan.scored, dtype=int)
    position = {int(r): i for i, r in enumerate(scored)}
    keys = ["logistic", "hgb"] + (["deep_linear", "deep_knn"] if deep is not None else [])
    found = {k: _Scored(np.zeros((len(scored), len(classes)))) for k in keys}
    if deep is not None:
        found["deep_knn"].neighbours = np.zeros((len(scored), 5), dtype=int)
    fold_of = np.zeros(len(scored), dtype=int)
    dims: list[int] = []
    n_steps = len(plan.folds) * len(keys)
    done = 0
    for f, fold in enumerate(plan.folds):
        train = np.asarray(fold.train, dtype=int)
        test = np.asarray(fold.test, dtype=int)
        train = train[np.isfinite(classic[train]).all(axis=1)]
        at = np.array([position[int(r)] for r in test], dtype=int)
        fold_of[at] = f
        ytr, gtr = labels[train].astype(str), groups[train].astype(str)
        space = descriptor_space(classic[train], seed)
        ztr, zte = space.transform(classic[train]), space.transform(np.nan_to_num(classic[test]))
        dims.append(int(ztr.shape[1]))
        for key in keys:
            steps(
                done / n_steps,
                f"Training {key.replace('_', ' ')} (fold {f + 1} / {len(plan.folds)})",
            )
            started = time.perf_counter()
            slot = found[key]
            if key == "logistic":
                model, c, inner = fit_logistic(ztr, ytr, gtr, seed=seed)
                slot.settings.append(f"C = {c:g}")
                slot.selection.append(inner)
                slot.proba[at] = _proba_in(model, zte, classes)
            elif key == "hgb":
                model, inner = fit_hgb(ztr, ytr, gtr, seed=seed)
                slot.selection.append(inner)
                slot.proba[at] = _proba_in(model, zte[:, :PCA_DIMS], classes)
            else:
                assert deep is not None
                ok = train[np.isfinite(deep[train]).all(axis=1)]
                xte = np.nan_to_num(deep[test])
                if key == "deep_linear":
                    model, c, inner = fit_logistic(
                        deep[ok], labels[ok].astype(str), groups[ok].astype(str), scale=True,
                        seed=seed,
                    )  # fmt: skip
                    slot.settings.append(f"C = {c:g}")
                    slot.selection.append(inner)
                    slot.proba[at] = _proba_in(model, xte, classes)
                else:
                    from quantilo.vision.deep.probe import K_NEIGHBOURS, TEMPERATURE, knn_proba

                    index = {c: i for i, c in enumerate(classes)}
                    coded = np.array([index[str(v)] for v in labels[ok]])
                    proba, nearest = knn_proba(
                        _normalise(deep[ok].astype(np.float64)),
                        coded,
                        len(classes),
                        _normalise(xte.astype(np.float64)),
                        K_NEIGHBOURS,
                        TEMPERATURE,
                    )
                    slot.proba[at] = proba
                    assert slot.neighbours is not None
                    width = nearest.shape[1]
                    slot.neighbours[at, :width] = ok[nearest]
                    slot.settings.append(f"k = {min(K_NEIGHBOURS, len(ok))}, cosine")
            slot.seconds += time.perf_counter() - started
            done += 1
    y = labels[scored].astype(str)
    g = groups[scored].astype(str)
    y_train = labels[np.asarray(sorted({r for f in plan.folds for r in f.train}), dtype=int)]
    results = []
    for key in keys:
        slot = found[key]
        report = vm.classification_report(
            y, proba=slot.proba, labels=classes, y_train=y_train.astype(str), groups=g,
            bootstrap=bootstrap, seed=seed,
        )  # fmt: skip
        errors = vm.most_confident_errors(y, slot.proba, classes, GALLERY)
        uncertain = vm.most_uncertain(slot.proba, GALLERY)
        results.append(
            VisionModelResult(
                key=key,
                title=_TITLES[key],
                family="deep" if key.startswith("deep") else "classic",
                setting=_setting(slot.settings),
                selection=float(np.mean(slot.selection)) if slot.selection else None,
                classification=report,
                galleries={
                    "confident_errors": [int(scored[i]) for i in errors],
                    "uncertain": [int(scored[i]) for i in uncertain],
                },
                seconds=round(slot.seconds, 2),
                notes=_class_notes(key, y_train.astype(str), dims),
            )
        )
    _pick_default(results)
    columns: dict[str, Any] = {"fold": fold_of.tolist(), "truth": y.tolist()}
    for key in keys:
        p = found[key].proba
        columns[f"{key}_pred"] = [classes[i] for i in p.argmax(axis=1)]
        columns[f"{key}_proba"] = p.max(axis=1).round(6).tolist()
        columns[f"{key}_probs"] = p.round(6).tolist()  # 113: ROC and PR curves per class
        neighbours = found[key].neighbours
        if neighbours is not None:
            columns[f"{key}_neighbours"] = neighbours.tolist()
    return results, columns


_TITLES = {
    "logistic": "Logistic regression on classic descriptors",
    "hgb": "Gradient boosting (HGB) on classic descriptors",
    "deep_linear": "Linear probe on DINOv2 ViT-S/14 (frozen)",
    "deep_knn": "Nearest neighbours on DINOv2 ViT-S/14 (cosine)",
    "patch_pca": "PCA of patches (reconstruction error)",
    "knn": "Nearest normal images (classic descriptors)",
    "ensemble": "Patch PCA + nearest normals (average rank)",
    "deep_knn_anomaly": "Nearest normal images (DINOv2 ViT-S/14)",
}


def _setting(settings: list[str]) -> str:
    counts = Counter(settings)
    if not counts:
        return "default settings"
    if len(counts) == 1:
        return settings[0]
    return ", ".join(f"{k} ({v} folds)" for k, v in counts.most_common())


def _class_notes(key: str, y_train: np.ndarray, dims: list[int]) -> list[str]:
    notes = []
    if key in ("logistic", "hgb", "deep_linear") and _class_weight(y_train):
        notes.append(
            "Classes weighted by their inverse frequency: the largest is more than 3 times the "
            "smallest (R-VCL-05, R-VQ-13)."
        )
    if key in ("logistic", "hgb"):
        kept = f"{min(dims)}-{max(dims)}" if dims and min(dims) != max(dims) else str(dims[0])
        used = f"its {kept} components" if key == "logistic" else f"its first {PCA_DIMS} components"
        notes.append(
            f"Classic descriptors standardised, then a PCA fitted on the training images only "
            f"keeps 95 % of their variance; the model reads {used} (R-VCL-01)."
        )
    if key == "logistic":
        notes.append(
            "C chosen by a cross-validation grouped by copies on the training images, on balanced "
            "accuracy (R-VCL-02)."
        )
    if key == "deep_knn":
        notes.append(
            "Each image takes the labels of its 20 most similar training images, weighted by "
            "exp(similarity / 0.07); its 5 nearest are kept to show (R-VDP-08)."
        )
    return notes


def _main(report: vm.ClassificationReport) -> float:
    if report.main.metric == "balanced_accuracy":
        return report.balanced_accuracy
    return report.accuracy


def _pick_default(results: list[VisionModelResult]) -> None:
    """R-VCL-04 then R-VCL-06: the deep probe when it ran, else the better classic model by its
    inner cross-validated score (never by the test)."""
    by_key = {m.key: m for m in results}
    classic = [by_key[k] for k in ("logistic", "hgb") if k in by_key]
    best = max(classic, key=lambda m: m.selection if m.selection is not None else -1.0)
    best.notes.insert(
        0,
        f"Best classic model by inner cross-validation (balanced accuracy {best.selection:.3f}), "
        "not by the test (R-VCL-04).",
    )
    chosen = by_key.get("deep_linear", best)
    chosen.default = True
    if chosen is not best:
        chosen.notes.insert(
            0, "Desktop default: a frozen DINOv2 network with a linear head (R-VCL-06)."
        )


# ---------------------------------------------------------------------------------------------
# Anomalies


def luma_square(source: Source, side: int = PATCH_SIDE) -> np.ndarray | None:
    """Luma (0-1) resized to `side` x `side` (R-VAN-01), or None when it cannot be decoded."""
    image = _decode(source)
    if image is None:
        return None
    grey = image.convert("L")
    smaller = max(grey.size) <= side
    resample = Image.Resampling.BILINEAR if smaller else Image.Resampling.BOX
    return np.asarray(grey.resize((side, side), resample), dtype=np.float32) / 255.0


def patches_of(luma: np.ndarray, patch: int = PATCH) -> np.ndarray:
    """(grid * grid, patch * patch) non-overlapping patches of a square luma image."""
    g = luma.shape[0] // patch
    return (
        luma[: g * patch, : g * patch]
        .reshape(g, patch, g, patch)
        .transpose(0, 2, 1, 3)
        .reshape(g * g, patch * patch)
    )


class PatchPCA:
    """R-VAN-01: a linear autoencoder of patches, fitted on normal images. Each patch's error is
    then read against the errors of the same position in the normal training images (median and
    MAD): an edge or a label is always hard to rebuild, only an unusually large error counts."""

    def __init__(self, variance: float = PATCH_VARIANCE, seed: int = SEED) -> None:
        self.variance = variance
        self.seed = seed
        self.mean: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.centre: np.ndarray | None = None  # per position: median error of the normals
        self.spread: np.ndarray | None = None  # per position: MAD x 1.4826

    def fit(self, patches: np.ndarray) -> "PatchPCA":
        """Principal components from the 256 x 256 covariance of the patches (an eigen
        decomposition: the same axes as an SVD of the patches, in a fraction of the time)."""
        x = np.asarray(patches, dtype=np.float64)
        mean = x.mean(axis=0)
        centred = x - mean
        values, axes = np.linalg.eigh(centred.T @ centred / max(1, len(x) - 1))
        order = np.argsort(values)[::-1]
        values, axes = np.clip(values[order], 0, None), axes[:, order]
        total = values.sum()
        share = np.cumsum(values) / total if total > 0 else np.ones_like(values)
        kept = int(min(len(values), np.searchsorted(share, self.variance) + 1))
        self.mean = mean.astype(np.float32)
        self.components = axes[:, :kept].T.astype(np.float32)
        return self

    @property
    def dims(self) -> int:
        return 0 if self.components is None else int(self.components.shape[0])

    def raw_errors(self, luma: np.ndarray) -> np.ndarray:
        """Mean squared reconstruction error of each patch, as a (grid, grid) map."""
        assert self.mean is not None and self.components is not None
        x = patches_of(luma) - self.mean
        rebuilt = (x @ self.components.T) @ self.components
        err = np.mean((x - rebuilt) ** 2, axis=1)
        g = round(math.sqrt(len(err)))
        return err.reshape(g, g)

    def calibrate(self, lumas: Sequence[np.ndarray]) -> "PatchPCA":
        """Median and MAD of the error at each position over normal images."""
        errors = np.stack([self.raw_errors(luma) for luma in lumas])
        self.centre = np.median(errors, axis=0)
        mad = np.median(np.abs(errors - self.centre), axis=0) * 1.4826
        self.spread = np.maximum(mad, 1e-3 * float(np.median(self.centre)) + 1e-9)
        return self

    def errors(self, luma: np.ndarray) -> np.ndarray:
        """Robust z of each patch's error against the normals at its position (raw errors before
        `calibrate`)."""
        raw = self.raw_errors(luma)
        if self.centre is None or self.spread is None:
            return raw
        return (raw - self.centre) / self.spread


def _pixel_map(grid: np.ndarray, side: int = PIXEL_SIDE) -> np.ndarray:
    """A patch-error grid enlarged to `side` x `side` (bilinear), smoothed (sigma 4 px at 256)."""
    from scipy import ndimage

    zoomed = ndimage.zoom(grid.astype(np.float64), side / grid.shape[0], order=1, mode="nearest")
    return ndimage.gaussian_filter(zoomed, MAP_SIGMA * side / PATCH_SIDE)


def _mask(source: Source, side: int = PIXEL_SIDE) -> np.ndarray | None:
    image = _decode(source, side * 4)
    if image is None:
        return None
    grey = image.convert("L").resize((side, side), Image.Resampling.NEAREST)
    return np.asarray(grey) > 127


def _ranks(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(values) / len(values)


def _knn_distance(x_train: np.ndarray, x_score: np.ndarray, k: int, cosine: bool) -> np.ndarray:
    """Mean distance of each scored row to its k nearest training rows (R-VAN-02)."""
    if cosine:
        a, b = _normalise(x_train), _normalise(x_score)
        out = np.empty(len(b))
        for start in range(0, len(b), 1024):
            d = 1.0 - b[start : start + 1024] @ a.T
            out[start : start + 1024] = np.sort(np.partition(d, k - 1, axis=1)[:, :k], axis=1).mean(
                1
            )
        return out
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(x_train)
    distances, _ = nn.kneighbors(x_score)
    return distances.mean(axis=1)


def _knn_space(x_train: np.ndarray, seed: int) -> Any:
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = make_pipeline(
        StandardScaler(), PCA(_pca_dims(len(x_train) * 2, x_train.shape[1]), random_state=seed)
    )
    return pipeline.fit(x_train)


DETECTORS = ("patchcore", "padim", "autoencoder", "vae")  # 115, 116 (R-VAN-06)
ENSEMBLE = "patchcore_ae"  # c3, R-VAN-07: Advanced > More
ENSEMBLE_MEMBERS = ("patchcore", "autoencoder")
_DETECTOR_NAMES = {
    "patchcore": "PatchCore",
    "padim": "PaDiM",
    "autoencoder": "Convolutional autoencoder",
    "vae": "Variational autoencoder",
    ENSEMBLE: "PatchCore + autoencoder (average rank)",
}
KEPT_MAPS = 1_000  # 113: anomaly maps (and reconstructions) kept with a run, at most
RECONSTRUCTION_SIDE = 256


@dataclass
class _Detected:
    """What a deep detector (115, 116) leaves behind over the folds: maps at PIXEL_SIDE,
    reconstructions (autoencoders) and, for PatchCore, the case behind each score."""

    key: str
    title: str = ""
    setting: str = ""
    notes: list[str] = field(default_factory=list)
    maps: dict[int, np.ndarray] = field(default_factory=dict)
    reconstructions: dict[int, np.ndarray] = field(default_factory=dict)
    cases: dict[int, dict[str, Any]] = field(default_factory=dict)


def _shrink(values: np.ndarray, side: int = PIXEL_SIDE) -> np.ndarray:
    """A square map brought to `side` x `side` (block mean when it divides, else bilinear)."""
    n = values.shape[0]
    if n == side:
        return values.astype(np.float32)
    if n % side == 0:
        k = n // side
        return values.reshape(side, k, side, k).mean(axis=(1, 3)).astype(np.float32)
    image = Image.fromarray(np.ascontiguousarray(values, dtype=np.float32), mode="F")
    return np.asarray(image.resize((side, side), Image.Resampling.BILINEAR), dtype=np.float32)


def _run_detector(
    found: _Detected,
    model: Any,
    fit_rows: np.ndarray,
    rows: np.ndarray,
    sources: Sequence[Source],
    steps: _Steps,
    low: float,
    high: float,
) -> np.ndarray:
    """R-VAN-06: fit a deep detector on one fold's normal training images and score its rows.
    Its progress fills [low, high] of the fold; a stop asked on the way raises `Cancelled`."""
    from quantilo.vision.deep import anomaly as deep_anomaly

    phase = {"what": "learning the normal images", "low": low, "high": (low + high) / 2}

    def progress(done: int, total: int) -> None:
        share = done / max(1, total)
        steps(
            phase["low"] + (phase["high"] - phase["low"]) * share,
            f"{found.title or found.key}: {phase['what']} {done:,} / {total:,}",
        )

    if not found.title:
        found.title = _DETECTOR_NAMES.get(found.key, found.key)
    model.progress = progress
    model.cancel = lambda: bool(steps.cancel is not None and steps.cancel())
    fit = [int(r) for r in fit_rows if sources[r] is not None]
    try:
        model.fit([sources[r] for r in fit])
        found.title = str(getattr(model, "title", found.title))
        phase.update(what="scoring", low=(low + high) / 2, high=high)
        result = model.score([sources[r] for r in rows])
    except deep_anomaly.Cancelled as error:
        raise Cancelled("Stopped: nothing was saved.") from error
    if not found.setting:
        found.setting = f"{result.device}, {result.side} px, fitted on {len(fit):,} normal images"
    for note in [*getattr(model, "notes", []), *result.notes]:
        if note not in found.notes:
            found.notes.append(note)
    if hasattr(result, "fit_rows"):  # PatchCore: the normal patch behind each score (§ 8)
        result.fit_rows = fit
    for j, r in enumerate(rows):
        if not result.ok[j]:
            continue
        if len(found.maps) < KEPT_MAPS:
            found.maps[int(r)] = _shrink(result.maps[j])
        if hasattr(result, "reconstructions") and len(found.reconstructions) < KEPT_MAPS:
            found.reconstructions[int(r)] = np.asarray(result.reconstructions[j], dtype=np.uint8)
        if hasattr(result, "explain"):
            case = result.explain(j)
            found.cases[int(r)] = {
                "box": list(case.box),
                "distance": float(case.distance),
                "normal_row": case.normal_row,
                "normal_box": list(case.normal_box),
                "sentence": case.sentence,
            }
    return np.asarray(result.scores, dtype=np.float64)


def choose_detector(
    detector: str | None, deep: bool | None, notes: list[str]
) -> tuple[str | None, str | None]:
    """R-VAN-06: which deep detector runs beside the classic ones, and why not when none does.
    `None` or "auto": PatchCore when its weights are here (the desktop default, 115), never when
    `deep` is False; "none": classic only; a named one when it can run, else its reason."""
    if detector == "none" or deep is False:
        return None, "Classic models only (asked)."
    if _in_browser():
        from quantilo.vision.deep import BROWSER

        return None, BROWSER
    if detector not in (None, "auto", *DETECTORS, ENSEMBLE):
        raise ValueError(
            f"Unknown detector {detector!r}: one of {', '.join((*DETECTORS, ENSEMBLE))}."
        )
    from quantilo.vision import deep as vision_deep

    wanted = "patchcore" if detector in (None, "auto") else detector
    status = vision_deep.status().detector(wanted)
    if status.available:
        return wanted, None
    if detector not in (None, "auto"):
        notes.append(f"{status.title} asked for but not available: {status.reason}")
    return None, status.reason


def _build_detector(method: str, seed: int, allow_gpu: bool, confirm_long: bool) -> Any:
    from quantilo.vision.deep import anomaly as deep_anomaly

    return deep_anomaly.detector(
        method,  # type: ignore[arg-type]
        seed=seed,
        allow_gpu=allow_gpu,
        confirm_long=confirm_long,
    )


def detect_anomalies(
    plan: SplitPlan,
    sources: Sequence[Source],
    normal: np.ndarray,
    types: np.ndarray,
    groups: np.ndarray,
    classic: np.ndarray,
    deep: np.ndarray | None,
    steps: _Steps,
    *,
    masks: Sequence[Source] | None = None,
    pixel: bool = False,
    threshold: float | None = None,
    seed: int = SEED,
    bootstrap: bool = True,
    detector: str | None = None,
    detector_model: Callable[[], Any] | None = None,
    allow_gpu: bool = True,
    confirm_long: bool = False,
) -> tuple[list[VisionModelResult], dict[str, Any]]:
    """R-VAN-01 to 06 on the plan's folds: every scored image (test, held-out normals) gets a
    score per model, then the metrics of § 7.2 (or, without defects, the R-VTH-02 threshold).
    `detector` (115, 116: patchcore, padim, autoencoder, vae) runs beside the classic models on
    the same folds and becomes the default; `detector_model()` builds it (tests)."""
    rng = np.random.default_rng(seed)
    holdout = np.asarray(plan.holdout, dtype=int)
    tested = np.asarray(plan.scored, dtype=int)
    scored = np.concatenate([tested, holdout]) if len(holdout) else tested
    position = {int(r): i for i, r in enumerate(scored)}
    keys = ["patch_pca", "knn"] + (["deep_knn_anomaly"] if deep is not None else [])
    # R-VAN-07: the ensemble runs both members, each shown, then their average rank
    members = list(ENSEMBLE_MEMBERS) if detector == ENSEMBLE else ([detector] if detector else [])
    keys += members
    found_by = {m: _Detected(m) for m in members}
    scores = {k: np.full(len(scored), np.nan) for k in keys}
    grids: dict[int, np.ndarray] = {}
    fold_of = np.zeros(len(scored), dtype=int)
    seconds = dict.fromkeys(keys, 0.0)
    pca_dims: list[int] = []
    n_folds = len(plan.folds)
    span = 0.4 if detector is not None else 1.0  # the deep detector takes the rest of a fold
    for f, fold in enumerate(plan.folds):
        train = np.asarray(fold.train, dtype=int)
        rows = np.asarray(fold.test, dtype=int)
        if f == 0 and len(holdout):
            rows = np.concatenate([rows, holdout])
        at = np.array([position[int(r)] for r in rows], dtype=int)
        fold_of[at] = f
        # R-VAN-01: patches of the training normals (at most MAX_PATCHES drawn), then the scores
        started = time.perf_counter()
        per_image = max(1, MAX_PATCHES // max(1, len(train)))
        calibrating = set(
            rng.choice(train, min(len(train), CALIBRATION_IMAGES), replace=False).tolist()
        )
        bank, kept = [], []
        for i, r in enumerate(train):
            if i % 10 == 0:
                steps(
                    (f + span * 0.4 * i / max(1, len(train))) / n_folds,
                    f"Patch PCA: learning normal patches (fold {f + 1} / {n_folds})",
                )
            luma = luma_square(sources[r])
            if luma is None:
                continue
            if int(r) in calibrating:
                kept.append(np.round(luma * 255).astype(np.uint8))
            p = patches_of(luma)
            if len(p) > per_image:
                p = p[rng.choice(len(p), per_image, replace=False)]
            bank.append(p)
        if not bank:
            raise ValueError("No training image could be decoded for the patch PCA.")
        model = PatchPCA(seed=seed).fit(np.concatenate(bank))
        del bank
        model.calibrate([k.astype(np.float32) / 255.0 for k in kept])
        pca_dims.append(model.dims)
        del kept
        for i, (r, a) in enumerate(zip(rows, at, strict=True)):
            if i % 10 == 0:
                steps(
                    (f + span * (0.4 + 0.4 * i / max(1, len(rows)))) / n_folds,
                    f"Patch PCA: scoring {i:,} / {len(rows):,} images",
                )
            luma = luma_square(sources[r])
            if luma is None:
                continue
            grid = model.errors(luma)
            grids[int(r)] = grid.astype(np.float32)
            scores["patch_pca"][a] = float(np.quantile(grid, PATCH_QUANTILE))
        seconds["patch_pca"] += time.perf_counter() - started
        # R-VAN-02 and R-VAN-04: nearest normal images
        steps((f + span * 0.85) / n_folds, "Nearest normal images")
        started = time.perf_counter()
        ok = train[np.isfinite(classic[train]).all(axis=1)]
        space = _knn_space(classic[ok], seed)
        good = np.isfinite(classic[rows]).all(axis=1)
        k = int(min(ANOMALY_K, len(ok)))
        values = np.full(len(rows), np.nan)
        values[good] = _knn_distance(
            space.transform(classic[ok]), space.transform(classic[rows][good]), k, False
        )
        scores["knn"][at] = values
        seconds["knn"] += time.perf_counter() - started
        if deep is not None:
            started = time.perf_counter()
            ok_deep = train[np.isfinite(deep[train]).all(axis=1)]
            good = np.isfinite(deep[rows]).all(axis=1)
            values = np.full(len(rows), np.nan)
            kd = int(min(ANOMALY_K, len(ok_deep)))
            values[good] = _knn_distance(
                deep[ok_deep].astype(np.float64), deep[rows][good].astype(np.float64), kd, True
            )
            scores["deep_knn_anomaly"][at] = values
            seconds["deep_knn_anomaly"] += time.perf_counter() - started
        width = (0.98 - span) / max(1, len(members))
        for m, member in enumerate(members):  # R-VAN-06: 115, 116; R-VAN-07: both members
            started = time.perf_counter()
            built = (
                detector_model()
                if detector_model is not None
                else _build_detector(member, seed, allow_gpu, confirm_long)
            )
            values = _run_detector(
                found_by[member], built, train, rows, sources, steps,
                (f + span + m * width) / n_folds, (f + span + (m + 1) * width) / n_folds,
            )  # fmt: skip
            scores[member][at] = values
            seconds[member] += time.perf_counter() - started
    # R-VAN-03: average percentile rank (images a method could not score take its median rank)
    ranks = []
    for key in ("patch_pca", "knn"):
        s = scores[key]
        r = np.full(len(s), 0.5)
        finite = np.isfinite(s)
        if finite.any():
            r[finite] = _ranks(s[finite])
        ranks.append(r)
    scores["ensemble"] = np.mean(ranks, axis=0)
    seconds["ensemble"] = 0.0
    keys = ["patch_pca", "knn", "ensemble"] + [k for k in keys if k.startswith("deep")]
    keys += members
    combined: dict[int, np.ndarray] = {}
    if detector == ENSEMBLE:  # R-VAN-07: ranks averaged, maps averaged on their own ranges
        member_ranks = []
        for member in members:
            s = scores[member]
            r = np.full(len(s), 0.5)
            finite = np.isfinite(s)
            if finite.any():
                r[finite] = _ranks(s[finite])
            member_ranks.append(r)
        scores[ENSEMBLE] = np.mean(member_ranks, axis=0)
        seconds[ENSEMBLE] = sum(seconds[m] for m in members)
        combined = _combined_maps([found_by[m].maps for m in members])
        keys.append(ENSEMBLE)
    steps(0.95, "Scoring")
    tested_at = np.arange(len(tested))
    held_at = np.arange(len(tested), len(scored))
    y = ~normal[tested]
    evaluable = bool(y.sum() >= 1 and (~y).sum() >= 1)
    pixel_data = _pixel_data(tested, y, grids, masks, rng) if pixel and evaluable else None
    deep_pixels = {
        name: _pixel_data(tested, y, maps, masks, rng, enlarge=False)
        for name, maps in [*((m, found_by[m].maps) for m in members), (ENSEMBLE, combined)]
        if maps and pixel and evaluable
    }
    # 113: the maps kept with the run, at PIXEL_SIDE (patch PCA: its grids enlarged)
    kept_rows = [int(r) for r in scored if int(r) in grids][:KEPT_MAPS]
    kept_maps: dict[str, dict[int, np.ndarray]] = {
        "patch_pca": {r: _pixel_map(grids[r]).astype(np.float32) for r in kept_rows}
    }
    for member in members:
        kept_maps[member] = found_by[member].maps
    if combined:
        kept_maps[ENSEMBLE] = combined
    results = []
    for key in keys:
        detected = found_by.get(key)
        s = np.nan_to_num(
            scores[key],
            nan=float(np.nanmedian(scores[key])) if np.isfinite(scores[key]).any() else 0.0,
        )
        is_detector = detected is not None and key == detected.key
        result = VisionModelResult(
            key=key,
            title=detected.title
            if is_detector and detected
            else _TITLES.get(key, _DETECTOR_NAMES.get(key, key)),
            family="deep"
            if key.startswith("deep") or is_detector or key == ENSEMBLE
            else "classic",
            setting=detected.setting
            if is_detector and detected
            else _anomaly_setting(key, pca_dims),
            default=key == (detector or "ensemble"),
            seconds=round(seconds[key], 2),
            notes=[*_anomaly_notes(key), *(detected.notes if is_detector and detected else [])],
            level="more" if key == ENSEMBLE else "first",
        )
        if key == ENSEMBLE:
            result.setting = "mean of the two percentile ranks; maps averaged on their ranges"
        source = "patch_pca" if key in ("patch_pca", "ensemble") else key
        if kept_maps.get(source):
            result.map_source = source
            result.map_range = _map_range(kept_maps[source])
        result.reconstructions = bool(is_detector and detected and detected.reconstructions)
        held = s[held_at] if len(held_at) else None
        if plan.code == "R-VSP-05":
            held = s[tested_at]  # every image is scored out of fold: they are the held-out normals
        if evaluable:
            maps = None
            gt = None
            if pixel_data is not None and key in ("patch_pca", "ensemble"):
                gt, maps = pixel_data
            if key in deep_pixels and deep_pixels[key] is not None:
                gt, maps = deep_pixels[key]  # type: ignore[misc]
            report = vm.anomaly_report(
                y,
                s[tested_at],
                defect_types=types[tested],
                holdout_normal_scores=held,
                user_threshold=threshold,
                masks=gt,
                maps=maps,
                pixel_threshold=None if maps is None else float(np.quantile(maps, 0.99)),
                groups=groups[tested].astype(str),
                bootstrap=bootstrap,
                seed=seed,
            )
            if maps is not None and key == "ensemble":
                report.notes.append(
                    "The map comes from the patch PCA (the nearest-image score has no map)."
                )
            result.anomaly = report
            order = np.argsort(-s[tested_at], kind="mergesort")
            result.galleries = {
                "most_unusual": [int(tested[i]) for i in order[:GALLERY]],
                "missed_defects": [int(tested[i]) for i in order[::-1] if y[i]][:GALLERY],
                "false_alarms": [int(tested[i]) for i in order if not y[i]][:GALLERY],
            }
        else:
            choice = vm.choose_threshold(holdout_normal_scores=held, user_threshold=threshold)
            result.threshold = choice
            result.flagged = int(np.sum(s[tested_at] >= choice.threshold))
            order = np.argsort(-s[tested_at], kind="mergesort")
            result.galleries = {"most_unusual": [int(tested[i]) for i in order[:GALLERY]]}
        results.append(result)
    if pixel and evaluable and pixel_data is None:
        results[0].notes.append(
            "No test image has a mask that could be read: pixel metrics left out."
        )
    columns: dict[str, Any] = {
        "fold": fold_of.tolist(),
        "truth": types[scored].tolist(),
        "is_defect": (~normal[scored]).tolist(),
        "holdout": [i >= len(tested) for i in range(len(scored))],
    }
    for key in keys:
        columns[f"{key}_score"] = [
            None if not math.isfinite(v) else round(float(v), 8) for v in scores[key]
        ]
    for found in found_by.values():
        if not found.cases:  # PatchCore: the case behind each score (§ 8)
            continue
        cases = [found.cases.get(int(r)) for r in scored]
        columns[f"{found.key}_box"] = [c["box"] if c else None for c in cases]
        columns[f"{found.key}_normal_row"] = [c["normal_row"] if c else None for c in cases]
        columns[f"{found.key}_normal_box"] = [c["normal_box"] if c else None for c in cases]
        columns[f"{found.key}_distance"] = [c["distance"] if c else None for c in cases]
    extra: dict[str, Any] = {"maps": kept_maps}
    recons = {f.key: f.reconstructions for f in found_by.values() if f.reconstructions}
    if recons:
        extra["reconstructions"] = recons
    return results, {"rows": scored, "columns": columns, "extra": extra}


def _combined_maps(parts: list[dict[int, np.ndarray]]) -> dict[int, np.ndarray]:
    """R-VAN-07: the members' maps of each image, each brought to 0-1 on its own range (0.5 % to
    99.5 %, as `_map_range`), then averaged; only the images every member has a map for."""
    if not parts or any(not p for p in parts):
        return {}
    ranges = [_map_range(p) for p in parts]
    rows = set.intersection(*(set(p) for p in parts))
    out: dict[int, np.ndarray] = {}
    for r in sorted(rows):
        scaled = [
            np.clip((p[r] - low) / (high - low), 0.0, 1.0)
            for p, (low, high) in zip(parts, ranges, strict=True)
        ]
        out[r] = np.mean(scaled, axis=0).astype(np.float32)
    return out


def _map_range(maps: dict[int, np.ndarray]) -> tuple[float, float]:
    """113: the range the kept maps are scaled on (0.5 % and 99.5 % of their values), so one grey
    level means the same in every image of the run."""
    values = np.concatenate([m.ravel()[::7] for m in maps.values()])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    low, high = (float(v) for v in np.quantile(values, [0.005, 0.995]))
    return (low, high if high > low else low + 1.0)


def _pixel_data(
    tested: np.ndarray,
    is_defect: np.ndarray,
    grids: dict[int, np.ndarray],
    masks: Sequence[Source] | None,
    rng: np.random.Generator,
    enlarge: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """R-VAN-05: masks and maps at 128 x 128 for the test images that have a map, and a mask when
    they are defects (normals: an empty mask); at most 1,000 images. `enlarge`: the maps are
    patch grids (patch PCA); otherwise they already are at 128 x 128 (deep detectors)."""
    if masks is None:
        return None
    chosen = [i for i, r in enumerate(tested) if int(r) in grids]
    if len(chosen) > PIXEL_MAX_IMAGES:
        chosen = sorted(rng.choice(chosen, PIXEL_MAX_IMAGES, replace=False).tolist())
    gts, maps = [], []
    for i in chosen:
        r = int(tested[i])
        if is_defect[i]:
            gt = _mask(masks[r]) if masks[r] is not None else None
            if gt is None:
                continue
        else:
            gt = np.zeros((PIXEL_SIDE, PIXEL_SIDE), dtype=bool)
        gts.append(gt)
        maps.append(_pixel_map(grids[r]) if enlarge else grids[r])
    if not any(g.any() for g in gts):
        return None
    return np.stack(gts), np.stack(maps)


def _anomaly_setting(key: str, dims: list[int]) -> str:
    if key == "patch_pca":
        kept = (
            f"{min(dims)}-{max(dims)}"
            if dims and min(dims) != max(dims)
            else str(dims[0] if dims else 0)
        )
        return (
            f"{PATCH}x{PATCH} patches at {PATCH_SIDE} px, {kept} components (95 % of variance), "
            "99 % quantile"
        )
    if key == "knn":
        return f"k = {ANOMALY_K}, PCA {PCA_DIMS}, Euclidean"
    if key == "deep_knn_anomaly":
        return f"k = {ANOMALY_K}, cosine"
    if key == "ensemble":
        return "mean of the two percentile ranks"
    return ""


def _anomaly_notes(key: str) -> list[str]:
    return {
        "patch_pca": [
            "A linear autoencoder: each 16 x 16 patch is rebuilt from the components of the "
            "normal patches; its error is compared with the error at the same place in the "
            "normal images (median and MAD), so only an unusually poor rebuild counts (R-VAN-01)."
        ],
        "knn": [
            "Mean distance to the 5 most similar normal training images, on the classic "
            "descriptors (R-VAN-02)."
        ],
        "ensemble": ["Both scores turned into percentile ranks and averaged (R-VAN-03)."],
        "deep_knn_anomaly": [
            "The same nearest-normal score on the frozen DINOv2 vectors, shown beside the "
            "default (R-VAN-04)."
        ],
        "patchcore": [
            "Each patch of the image is compared with a memory of normal patches from a frozen "
            "network; its distance to the closest one is how unusual it is, and that closest "
            "normal patch is shown (R-VAD-06, R-VAN-06)."
        ],
        "padim": [
            "A Gaussian of the normal patches at each place of the image; the Mahalanobis "
            "distance to it is how unusual a patch is (R-VAD-07, R-VAN-06)."
        ],
        "autoencoder": [
            "A network trained here on the normal images rebuilds each image; where it cannot "
            "(1 - SSIM) is unusual. Much weaker than PatchCore, but the rebuild shows what it "
            "expected (R-VAD-08, R-VAN-06)."
        ],
        "vae": [
            "The variational autoencoder: the same rebuild through a probabilistic code "
            "(R-VAD-08, R-VAN-06)."
        ],
        ENSEMBLE: [
            "PatchCore and the autoencoder trained on the same folds; their scores turned into "
            "percentile ranks and averaged, their maps brought to their own range and averaged "
            "(R-VAN-07). Each is shown on its own too."
        ],
    }[key]


# ---------------------------------------------------------------------------------------------
# Clusters


def find_clusters(
    x: np.ndarray, labels: Sequence[str | None] | None, *, seed: int = SEED
) -> tuple[ClusterResult, np.ndarray]:
    """R-VCU-01: (result, cluster of each row, -1 for rows without a vector)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    valid = np.isfinite(x).all(axis=1)
    data = _knn_space(x[valid], seed).transform(x[valid])
    rng = np.random.default_rng(seed)
    sample = (
        rng.choice(len(data), SILHOUETTE_SAMPLE, replace=False)
        if len(data) > SILHOUETTE_SAMPLE
        else np.arange(len(data))
    )
    by_k: dict[int, float] = {}
    fitted: dict[int, Any] = {}
    for k in CLUSTER_KS:
        if k >= len(data):
            break
        model = KMeans(k, n_init=4, random_state=seed).fit(data)  # pyright: ignore[reportArgumentType]
        fitted[k] = model
        assigned = np.asarray(model.labels_)
        by_k[k] = float(silhouette_score(data[sample], assigned[sample]))
    if not by_k:
        raise ValueError("Too few images to draw clusters.")
    best = max(by_k, key=lambda k: by_k[k])
    model = fitted[best]
    out = np.full(len(x), -1, dtype=int)
    out[valid] = model.labels_
    rows = np.flatnonzero(valid)
    medoids = {}
    for c in range(best):
        members = np.flatnonzero(model.labels_ == c)
        d = np.linalg.norm(data[members] - model.cluster_centers_[c], axis=1)
        medoids[c] = int(rows[members[int(np.argmin(d))]])
    sizes = {c: int(np.sum(model.labels_ == c)) for c in range(best)}
    sentence = (
        f"{best} clusters, the best silhouette from 2 to {max(by_k)} ({by_k[best]:.2f}) (R-VCU-01)."
    )
    ari = None
    if labels is not None:
        known = [i for i in range(len(rows)) if labels[rows[i]] is not None]
        if len({labels[rows[i]] for i in known}) >= 2:
            ari = float(
                adjusted_rand_score([str(labels[rows[i]]) for i in known], model.labels_[known])
            )
            sentence += (
                f" Clusters match labels: adjusted Rand {ari:.2f}."
                if ari >= 0.5
                else f" Labels overlap in this view (adjusted Rand {ari:.2f}): a linear model on "
                "these features will struggle."
            )
    if by_k[best] < 0.1:
        sentence += " The silhouette is low: read these clusters as tendencies."
    return (
        ClusterResult(
            k=best,
            silhouette=by_k[best],
            by_k=by_k,
            sizes=sizes,
            medoids=medoids,
            adjusted_rand=ari,
            sentence=sentence,
        ),
        out,
    )


# ---------------------------------------------------------------------------------------------
# One run


def evaluate(
    table: pa.Table,
    sources: Sequence[Source],
    *,
    masks: Sequence[Source] | None = None,
    task: Task | None = None,
    deep: bool | None = None,
    threshold: float | None = None,
    cache_dir: str | Path | None = None,
    allow_gpu: bool = True,
    confirm_long: bool = False,
    seed: int = SEED,
    bootstrap: bool = True,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
    detector: str | None = None,
    detector_model: Callable[[], Any] | None = None,
    more: Sequence[str] = (),
    concepts: Sequence[Any] = (),
    target: str | None = None,
) -> Outcome:
    """Detect the task (R-VML), split by groups (R-VSP), compute the vectors, train and score
    every model of the first plan. Anomalies: `detector` (None or "auto" = PatchCore when its
    weights are here, "none", or patchcore / padim / autoencoder / vae) runs beside the classic
    models and becomes the default (R-VAN-06). `sources[i]` is the image of table row `i` (a
    path, bytes or a PIL image); `masks[i]` its mask, if any. `deep`: None = when available
    (R-VCL-06), False = classic only, True = asked (its absence is said). `cancel()` is checked
    between images and models (`Cancelled`). 118: `more` names Advanced > More classification
    models (`vision.deep.more`: finetune, protopnet, concept_bottleneck), trained on the same
    folds after the first plan; `concepts` are the bottleneck's (`vision.deep.concepts`). c3:
    `task="regression"` with `target` (a numeric column, R-VML-05) or `task="multilabel"`
    (R-VML-06) train the models of `vision.target_models` on the same vectors."""
    started = time.perf_counter()
    steps = _Steps(progress, cancel)
    choice = detect_task(table, task, target)
    if choice.task is None or choice.blocked:
        raise ValueError(choice.blocked or choice.message)
    if _in_browser() and choice.n_usable > BROWSER_MAX_IMAGES:
        raise ValueError(
            f"The browser version trains on at most {BROWSER_MAX_IMAGES:,} images; this set has "
            f"{choice.n_usable:,}: use the desktop app (R-VCL-07)."
        )
    plan = plan_split(table, choice, seed)
    notes: list[str] = []
    option = deep_option(deep)
    if choice.task == "clusters" and option.used:
        option = DeepOption(used=False, reason="Clusters use the classic descriptors (R-VCU-01).")
    if deep is True and not option.used:
        notes.append(f"Deep vectors asked for but not available: {option.reason}")
    image_ids = [str(v) for v in column(table, "image_id")]
    cache_root = Path(cache_dir) if cache_dir is not None else None
    used = {r for f in plan.folds for r in (*f.train, *f.test)} | set(plan.holdout)
    srcs: list[Source] = [sources[i] if i in used else None for i in range(len(sources))]
    steps.stage(0.0, 0.45 if option.used else 0.6)
    classic, info = classic_features(
        srcs,
        image_ids,
        cache_root / "classic.parquet" if cache_root else None,
        steps,
        cancel,
    )
    # images never used keep NaN (they were not computed)
    features = [info]
    deep_x = None
    if option.used:
        steps.stage(0.45, 0.65)
        deep_x, deep_info = deep_features(
            srcs,
            image_ids,
            cache_root / f"{DEEP_BACKBONE}.parquet" if cache_root else None,
            allow_gpu=allow_gpu,
            confirm_long=confirm_long,
            progress=steps,
            cancel=cancel,
        )
        features.append(deep_info)
    steps.stage(0.65 if option.used else 0.6, 0.98)
    labels = np.asarray(labels_of(table), dtype=object)
    groups = groups_of(table)
    extra: dict[str, Any] = {}
    clusters = None
    models: list[VisionModelResult] = []
    if more and choice.task != "classification":
        notes.append("Advanced > More models are for classification only: not trained (R-VMO-01).")
    if choice.task == "classification":
        classes = sorted(choice.classes)
        if more:
            steps.stage(0.65 if option.used else 0.6, 0.72)
        models, cols = classify(
            plan, labels, groups, classes, classic, deep_x, steps, seed=seed, bootstrap=bootstrap
        )
        rows = np.asarray(plan.scored, dtype=int)
        if more:
            from quantilo.vision.deep.more import run_more

            steps.stage(0.72, 0.98)
            reference = cols.get("deep_linear_probs")
            advanced = run_more(
                more, table, plan, labels, groups, classes, srcs, classic, deep_x, steps,
                concepts=concepts, reference=None if reference is None else np.asarray(reference),
                allow_gpu=allow_gpu, confirm_long=confirm_long, seed=seed, bootstrap=bootstrap,
                cancel=cancel,
            )  # fmt: skip
            models += advanced.models
            cols.update(advanced.columns)
            notes += advanced.notes
            extra = {"files": advanced.files, "arrays": advanced.arrays}
    elif choice.task in ("regression", "multilabel"):
        from quantilo.vision import target_models as tm
        from quantilo.vision import targets as tg

        if choice.task == "regression":
            assert choice.target is not None
            y = tg.regression_values(table, choice.target)
            models, cols = tm.regress(
                plan, y, groups, classic, deep_x, steps, seed=seed, bootstrap=bootstrap
            )
        else:
            usable = usable_rows(table)
            binary = choice.binary_targets if choice.label_source == "columns" else []
            sets = tg.label_sets(table, usable, labels_of(table), binary)
            models, cols = tm.multilabel(
                plan, sets.y, sets.names, groups, classic, deep_x, steps, seed=seed,
                bootstrap=bootstrap,
            )  # fmt: skip
        rows = np.asarray(plan.scored, dtype=int)
    elif choice.task == "anomaly":
        if detector_model is not None:
            method: str | None = detector or "patchcore"
        else:
            method, why = choose_detector(detector, deep, notes)
            if method is None and why and detector not in ("none",) and deep is not False:
                notes.append(f"PatchCore not used: {why}")
        normal = normal_flags(table)
        types = np.asarray(
            ["good" if normal[i] else (labels[i] or "defect") for i in range(len(normal))],
            dtype=object,
        )
        models, found = detect_anomalies(
            plan, srcs, normal, types, groups, classic, deep_x, steps,
            masks=masks, pixel=choice.pixel, threshold=threshold, seed=seed, bootstrap=bootstrap,
            detector=method, detector_model=detector_model, allow_gpu=allow_gpu,
            confirm_long=confirm_long,
        )  # fmt: skip
        rows, cols = found["rows"], found["columns"]
        extra = found["extra"]
    else:
        steps(0.1, "Clusters")
        rows = np.asarray(plan.folds[0].train, dtype=int)
        clusters, assigned = find_clusters(classic[rows], list(labels[rows]), seed=seed)
        clusters.medoids = {c: int(rows[r]) for c, r in clusters.medoids.items()}
        cols = {"cluster": assigned.tolist(), "truth": [labels[r] for r in rows]}
    steps.stage(0.0, 1.0)
    steps(1.0, "Done")
    predictions = pa.table(
        {
            "row": pa.array(np.asarray(rows, dtype=np.int64)),
            "image_id": pa.array([image_ids[r] for r in rows], type=pa.string()),
            **cols,
        }
    )
    default = next((m.key for m in models if m.default), None)
    mislabelled = None
    if choice.task == "classification" and default is not None:
        mislabelled = _mislabelled(
            plan, labels, groups, sorted(choice.classes), np.asarray(rows), cols, models, seed
        )
    result = VisionResult(
        task=choice,
        split=plan,
        deep=option,
        features=features,
        models=models,
        default=default,
        clusters=clusters,
        sentence=_sentence(choice, models, clusters),
        seconds=round(time.perf_counter() - started, 2),
        seed=seed,
        notes=notes,
        mislabelled=mislabelled,
    )
    return Outcome(result, predictions, extra)


def _mislabelled(
    plan: SplitPlan,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    rows: np.ndarray,
    cols: dict[str, Any],
    models: list[VisionModelResult],
    seed: int,
) -> QualityFinding | None:
    """R-VQ-16 on the default model's out-of-fold probabilities of the scored images, after the
    temperature scaling of R-VM-06 when it was applied."""
    model = next(m for m in models if m.default)
    probs = cols.get(f"{model.key}_probs")
    if probs is None or model.classification is None:
        return None
    y = labels[rows].astype(str)
    proba = vm.calibrated_proba(
        y, np.asarray(probs, dtype=float), classes, model.classification.calibration,
        groups[rows].astype(str), seed,
    )  # fmt: skip
    applied = bool(model.classification.calibration and model.classification.calibration.applied)
    if plan.kind == "folds":
        scope = (
            f"Every labelled image was checked ({len(rows):,}): each was scored by a model that "
            "did not see it (R-VSP-02)."
        )
    else:
        scope = (
            f"Only the {len(rows):,} scored images were checked: the training images of a "
            f"{'producer' if plan.kind == 'producer' else 'held-out'} split have no out-of-fold "
            f"probability ({plan.code})."
        )
    return possibly_mislabelled(
        y.tolist(),
        proba,
        classes,
        rows.tolist(),
        model=model.title,
        calibrated=applied,
        scope=scope,
    )


def _sentence(
    choice: TaskChoice, models: list[VisionModelResult], clusters: ClusterResult | None
) -> str:
    if clusters is not None:
        return clusters.sentence
    default = next((m for m in models if m.default), None)
    if default is None:
        return choice.message
    if default.classification is not None:
        report = default.classification
        metric = report.main.metric.replace("_", " ")
        interval = report.intervals.get(report.main.metric)
        value = _main(report)
        text = f"{default.title}: {metric} {value:.3f}"
        if interval is not None:
            text += f" [{interval.low:.3f}; {interval.high:.3f}]"
        floor = (
            report.baseline.balanced_accuracy
            if report.main.metric == "balanced_accuracy"
            else report.baseline.accuracy
        )
        text += f"; majority baseline {floor:.3f}"
        classic = [m for m in models if m.family == "classic" and m.classification is not None]
        if default.family == "deep" and classic:
            best = max(classic, key=lambda m: m.selection or -1.0)
            assert best.classification is not None
            gain = value - _main(best.classification)
            text += f"; {100 * gain:+.1f} pts over the best classic model ({best.key})"
        return text + "."
    if default.regression is not None:
        return f"{default.title} ({choice.target}): {default.regression.sentence}"
    if default.multilabel is not None:
        return f"{default.title}: {default.multilabel.sentence}"
    if default.anomaly is not None:
        a = default.anomaly
        text = f"{default.title}: AUROC {a.auroc:.3f}"
        if a.auroc_interval is not None:
            text += f" [{a.auroc_interval.low:.3f}; {a.auroc_interval.high:.3f}]"
        text += f", AP {a.ap:.3f}"
        if a.pixel is not None:
            text += f"; pixel AUROC {a.pixel.auroc:.3f}, AUPRO 30 % {a.pixel.aupro:.3f}"
        return text + "."
    if default.threshold is not None:
        return (
            f"{default.title}: {_plural(default.flagged or 0, 'image')} at or above the threshold. "
            f"{default.threshold.message}"
        )
    return choice.message


def evaluate_image_set(images: "ImageSet", **options: Any) -> Outcome:
    """`evaluate` on an imported image set (104): its table, its stored files and masks, and its
    vector cache in `images/<id>/embeddings/`."""
    table = images.table
    stored = column(table, "stored_path")
    masks = column(table, "mask_path")
    sources: list[Source] = [images.store.path(v) if v else None for v in stored]
    mask_sources: list[Source] = [images.store.path(v) if v else None for v in masks]
    options.setdefault("cache_dir", images.store.root / "embeddings")
    return evaluate(table, sources, masks=mask_sources, **options)


__all__ = [
    "Cancelled",
    "ClusterResult",
    "DeepOption",
    "FeatureInfo",
    "Outcome",
    "PatchPCA",
    "VisionModelResult",
    "VisionResult",
    "choose_detector",
    "classic_features",
    "classify",
    "deep_features",
    "deep_option",
    "detect_anomalies",
    "evaluate",
    "evaluate_image_set",
    "find_clusters",
    "fit_hgb",
    "fit_logistic",
    "luma_square",
    "patches_of",
]
