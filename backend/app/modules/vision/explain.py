"""Explaining a kept vision run (117): "Why?" on one image, and the Explain tab.

The run keeps its report and its predictions, not its fitted models. The models are fitted again,
on demand, on the same training rows of the same fold, by the same rules (R-VEX-12): the C the run
chose is read from its setting, so the logistic and the probe come back identical, and the
evidence shown is the evidence of the model whose metrics the Results tab shows.

- First plane, computed when "Why?" opens (seconds): the prediction with every class's
  probability, the nearest training images in the space the model reads (R-VEX-01), the nearest
  prototype of the predicted class (R-VEX-02); for anomalies, the nearest normal images (the
  score itself, R-VAN-02) and the patch PCA's own map.
- Advanced, one request each: the counterfactual example (R-VEX-04), occlusion, Grad-CAM,
  attention rollout; PatchCore's nearest normal patch (fitted on the run's normal images once).
- Advanced > More, never before it is asked for: saliency, SmoothGrad, Integrated Gradients, the
  sanity test.

Results are cached: fitted models in memory (a few per process), every map and evidence on disk
next to the run (`runs/<run>/explain/`, deleted with it). One computation at a time (a lock):
memory and the GPU stay bounded.
"""

import base64
import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel
from quantilo.interpret import prototypes as pt
from quantilo.vision import classic_models as cm
from quantilo.vision import explain as ex
from quantilo.vision import vectors as cache
from quantilo.vision.descriptors import HOG_SIZE, HSV_SIZE, classic_descriptor
from quantilo.vision.split import groups_of
from quantilo.vision.task import column, labels_of

from app.core.errors import AppError
from app.modules.vision.service import VisionService, clean

Level = Literal["first", "advanced", "more"]
DETECTORS = ("patchcore", "padim", "autoencoder", "vae")  # 115, 116, run beside the classics (113)
MAP_METHODS = (
    "kept",
    "occlusion",
    "grad_cam",
    "rollout",
    "saliency",
    "smoothgrad",
    "integrated_gradients",
    "patch_pca",
    "patchcore",
)
DEEP_METHODS = ("grad_cam", "rollout", "saliency", "smoothgrad", "integrated_gradients")
MORE_MODELS = ("finetune", "protopnet", "concept_bottleneck")  # 118: Advanced > More
FAMILIES = (
    ("colour", "Colour histogram (HSV)", 0, HSV_SIZE),
    ("shape", "Shape and edges (HOG)", HSV_SIZE, HSV_SIZE + HOG_SIZE),
    ("texture", "Texture (LBP)", HSV_SIZE + HOG_SIZE, None),
)
PERMUTATIONS = 5
KEPT_MODELS = 6
VERSION = "117-4"  # cached files of another version are ignored


# --- API shapes ------------------------------------------------------------------------------


class WhyImage(BaseModel):
    row: int
    image_id: str
    label: str | None = None
    similarity: float | None = None  # cosine in the model's space
    distance: float | None = None  # when the model reads a distance (anomalies)
    share: float | None = None  # prototypes: share of the class it stands for
    witness: float | None = None  # criticisms


class WhyClassProbability(BaseModel):
    label: str
    probability: float


class WhyPrediction(BaseModel):
    label: str | None = None
    probability: float | None = None
    probabilities: list[WhyClassProbability] = []
    score: float | None = None  # anomalies (higher = more unusual)
    threshold: float | None = None
    flagged: bool | None = None
    truth: str | None = None
    correct: bool | None = None
    tested: bool  # False: the image was among those the model learnt from
    sentence: str


class WhyEvidence(BaseModel):
    space: str  # what the comparison reads, in words
    nearest: list[WhyImage]
    sentence: str
    prototype: WhyImage | None = None  # of the predicted class
    prototype_sentence: str | None = None


class WhyCounterfactual(BaseModel):
    image: WhyImage
    own_label: str | None
    probability: float | None  # the model's probability of the example's own class
    sentence: str


class WhyMethod(BaseModel):
    key: str
    title: str
    level: Level
    available: bool
    reason: str | None = None
    cost: str  # "instant", "about 1 s", "about 10 s"...
    own: bool = False  # the model's own computation (patch PCA, PatchCore)


class WhyModel(BaseModel):
    key: str
    title: str
    default: bool


class WhyExplanation(BaseModel):
    run_id: str
    image_id: str
    row: int
    task: str
    model: WhyModel
    models: list[WhyModel]
    fold: int
    prediction: WhyPrediction
    evidence: WhyEvidence
    counterfactual: WhyCounterfactual | None
    methods: list[WhyMethod]
    seconds: float
    notes: list[str] = []
    own_map: str | None = None  # the map method of the model's own computation, first plane
    patch: "WhyPatch | None" = None  # PatchCore: the case behind the score, kept by the run


class WhyPatch(BaseModel):
    box: tuple[float, float, float, float]  # in the image, relative
    normal: WhyImage
    normal_box: tuple[float, float, float, float]  # in that normal image, relative
    distance: float
    sentence: str


class WhyMap(BaseModel):
    method: str
    title: str
    unit: str
    sentence: str
    warning: str
    own: bool
    blocky: bool
    box: tuple[float, float, float, float]
    target: str | None
    rows: int
    columns: int
    low: float
    high: float
    png: str  # data:image/png;base64,... (8-bit grey: 0 = low, 255 = high)
    seconds: float
    cached: bool = False
    notes: list[str] = []
    extra: dict[str, float] = {}
    patch: WhyPatch | None = None


class WhySanityStage(BaseModel):
    label: str
    correlation: float | None
    png: str
    low: float
    high: float


class WhySanity(BaseModel):
    method: str
    title: str
    passes: bool
    sentence: str
    box: tuple[float, float, float, float]
    target: str | None
    rows: int
    columns: int
    stages: list[WhySanityStage]
    seconds: float
    cached: bool = False


class WhyClassPrototypes(BaseModel):
    label: str
    size: int
    prototypes: list[WhyImage]
    criticisms: list[WhyImage]
    notes: list[str] = []


class WhyFamily(BaseModel):
    key: str
    title: str
    drop: float  # balanced accuracy lost when the family is shuffled
    spread: float  # standard deviation over the shuffles
    columns: int


class ExplainOverview(BaseModel):
    run_id: str
    task: str
    model: WhyModel
    models: list[WhyModel]
    space: str
    classes: list[WhyClassPrototypes]
    sentence: str
    families: list[WhyFamily] | None = None
    families_sentence: str | None = None
    seconds: float
    notes: list[str] = []


# --- helpers ---------------------------------------------------------------------------------


def _data_url(values: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(ex.png_gray(values)).decode("ascii")


def _setting_c(setting: str) -> float | None:
    """The C every fold chose ("C = 0.1"); None when the folds chose different ones."""
    if not setting.startswith("C = ") or "(" in setting:
        return None
    try:
        return float(setting.removeprefix("C = "))
    except ValueError:
        return None


class _Lru(OrderedDict[Any, Any]):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def get_or(self, key: Any, build: Callable[[], Any]) -> Any:
        if key in self:
            self.move_to_end(key)
            return self[key]
        value = build()
        self[key] = value
        while len(self) > self.size:
            self.popitem(last=False)
        return value


@dataclass
class _Run:
    """A kept run, read once: its table, files, folds and predictions."""

    dataset_id: str
    run_id: str
    data: dict[str, Any]
    table: Any
    sources: list[Path | None]
    image_ids: list[str]
    labels: np.ndarray
    groups: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]
    predictions: dict[int, dict[str, Any]]
    cache_root: Path
    folder: Path

    @property
    def result(self) -> dict[str, Any]:
        return self.data["result"]

    @property
    def task(self) -> str:
        return str(self.result["task"]["task"])

    @property
    def classes(self) -> list[str]:
        return sorted(self.result["task"].get("classes") or {})

    @property
    def models(self) -> list[dict[str, Any]]:
        return list(self.result.get("models") or [])

    def model(self, key: str | None) -> dict[str, Any]:
        models = self.models
        if not models:
            raise AppError(
                "This run has no model to explain (clusters).", code="no_model", status_code=409
            )
        if self.task in ("regression", "multilabel"):  # c3
            raise AppError(
                "Why? explains classification and anomaly runs; a regression or multi-label run "
                "shows its largest errors in Results.",
                code="no_model",
                status_code=409,
            )
        if key is None:
            return next((m for m in models if m.get("default")), models[0])
        found = next((m for m in models if m["key"] == key), None)
        if found is None:
            raise AppError(f"This run has no model {key!r}.", code="not_found", status_code=404)
        return found

    def choices(self) -> list[WhyModel]:
        return [
            WhyModel(key=m["key"], title=m["title"], default=bool(m.get("default")))
            for m in self.models
        ]

    def row_of(self, image_id: str) -> int:
        try:
            return self.image_ids.index(image_id)
        except ValueError:
            raise AppError(
                "Image not found in this set.", code="not_found", status_code=404
            ) from None

    def fold_of(self, row: int) -> tuple[int, bool]:
        """(fold, tested): the fold whose test holds the row, else the first fold it trained."""
        for f, (_train, test) in enumerate(self.folds):
            if row in set(test.tolist()):
                return f, True
        return 0, False


# --- the service -----------------------------------------------------------------------------


class ExplainService:
    def __init__(self, vision: VisionService) -> None:
        self.vision = vision
        self._lock = threading.RLock()
        self._runs = _Lru(3)
        self._models = _Lru(KEPT_MODELS)
        self._vectors = _Lru(4)
        self._nets = _Lru(1)
        self._patchcore = _Lru(1)

    # --- reading a run ---------------------------------------------------------------------------

    def _run(self, dataset_id: str, run_id: str) -> _Run:
        data = self.vision.run(dataset_id, run_id)
        folder = self.vision._run_dir(dataset_id, run_id)
        record = self.vision._record(dataset_id)
        stamp = Path(record.storage_path).stat().st_mtime_ns

        def build() -> _Run:
            import pyarrow.parquet as pq

            table = self.vision._table(record)
            store = self.vision._store(record)
            sources = [store.path(v) if v else None for v in table["stored_path"].to_pylist()]
            ids = [str(v) for v in column(table, "image_id")]
            folds = [
                (np.asarray(f["train"], dtype=int), np.asarray(f["test"], dtype=int))
                for f in data["result"]["split"]["folds"]
            ]
            if folds and max(int(max(t.max(initial=0), s.max(initial=0))) for t, s in folds) >= len(
                ids
            ):
                raise AppError(
                    "The set changed since this run: train again.",
                    code="stale_run",
                    status_code=409,
                )
            rows = pq.read_table(folder / "predictions.parquet").to_pylist()
            predictions = {int(r["row"]): r for r in rows}
            for r, record_ in predictions.items():
                if r >= len(ids) or ids[r] != record_["image_id"]:
                    raise AppError(
                        "The set changed since this run: train again.",
                        code="stale_run",
                        status_code=409,
                    )
            return _Run(
                dataset_id,
                run_id,
                data,
                table,
                sources,
                ids,
                np.asarray(labels_of(table), dtype=object),
                groups_of(table),
                folds,
                predictions,
                store.root / "embeddings",
                folder,
            )

        return self._runs.get_or((dataset_id, run_id, stamp), build)

    def _decode(self, run: _Run, row: int, side: int = cm.DECODE_SIDE) -> Any:
        image = cm._decode(run.sources[row], side)
        if image is None:
            raise AppError("This image could not be decoded.", code="not_decoded", status_code=409)
        return image.convert("RGB")

    # --- vectors ---------------------------------------------------------------------------------

    def _classic(self, run: _Run, rows: np.ndarray) -> np.ndarray:
        """Descriptors of `rows` (cached by the run; the others computed here)."""
        known = self._vectors.get_or(
            (run.dataset_id, "classic"),
            lambda: cache.load(run.cache_root / "classic.parquet", cm.DESCRIPTOR_KEY),
        )
        out = np.full((len(rows), cm.CLASSIC_SIZE), np.nan, dtype=np.float32)
        for i, r in enumerate(rows.tolist()):
            hit = known.get(run.image_ids[r])
            if hit is None:
                image = cm._decode(run.sources[r])
                if image is None:
                    continue
                hit = classic_descriptor(image).vector
                known[run.image_ids[r]] = hit
            out[i] = hit
        return out

    def _net(self) -> Any:
        from quantilo.vision.deep.backbones import load

        return self._nets.get_or("dinov2_s14", lambda: load(cm.DEEP_BACKBONE))

    def _device(self) -> Any:
        from quantilo.synthetic.deep.device import pick

        return pick("auto", self.vision.compute()).device

    def _deep(self, run: _Run, rows: np.ndarray) -> np.ndarray:
        from quantilo.vision.deep.backbones import BACKBONES

        spec = BACKBONES[cm.DEEP_BACKBONE]
        weight = spec.weight_spec
        assert weight is not None
        key = f"{spec.key}|{weight.sha256[:12]}|{spec.side}px|crop={spec.crop}|{spec.pooling}"
        known = self._vectors.get_or(
            (run.dataset_id, key),
            lambda: cache.load(run.cache_root / f"{cm.DEEP_BACKBONE}.parquet", key),
        )
        out = np.full((len(rows), spec.dims), np.nan, dtype=np.float32)
        todo = []
        for i, r in enumerate(rows.tolist()):
            hit = known.get(run.image_ids[r])
            if hit is None:
                todo.append(i)
            else:
                out[i] = hit
        if todo:
            from quantilo.vision.deep.embed import embed_images

            sources = [run.sources[int(rows[i])] for i in todo]
            if any(s is None for s in sources):
                raise AppError("An image has no stored file.", code="not_decoded")
            found = embed_images(sources, self._net(), allow_gpu=self.vision.compute())  # type: ignore[arg-type]
            for j, i in enumerate(todo):
                out[i] = found.vectors[j]
                if np.isfinite(found.vectors[j]).all():
                    known[run.image_ids[int(rows[i])]] = found.vectors[j]
        return out

    # --- the run's models, fitted again ---------------------------------------------------------

    def _fitted(self, run: _Run, model: dict[str, Any], fold: int) -> dict[str, Any]:
        """The model of `fold`, fitted again on its training rows (R-VEX-12), with the space its
        evidence is read in: `{"space": callable rows -> vectors, "bank": rows, "proba": ...}`."""
        key = (run.dataset_id, run.run_id, model["key"], fold)
        return self._models.get_or(key, lambda: self._fit(run, model, fold))

    def _fit(self, run: _Run, model: dict[str, Any], fold: int) -> dict[str, Any]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        train, _test = run.folds[fold]
        name = model["key"]
        seed = int(run.result.get("seed", 0))
        notes: list[str] = []
        if name in MORE_MODELS:  # 118: their kept vectors, or DINOv2's to compare with
            from app.modules.vision import more_explain

            return more_explain.fit(self, run, model, fold)
        if name in ("logistic", "hgb"):
            x = self._classic(run, train)
            ok = train[np.isfinite(x).all(axis=1)]
            x = x[np.isfinite(x).all(axis=1)]
            y, g = run.labels[ok].astype(str), run.groups[ok].astype(str)
            space = cm.descriptor_space(x, seed)
            z = space.transform(x)
            if name == "logistic":
                c = _setting_c(model.get("setting", ""))
                if c is None:
                    fitted, c, _ = cm.fit_logistic(z, y, g, seed=seed)
                    notes.append("C chosen again by the grouped inner cross-validation.")
                else:
                    fitted = make_pipeline(
                        LogisticRegression(C=c, max_iter=3000, class_weight=cm._class_weight(y))
                    )
                    with cm._quiet():
                        fitted.fit(z, y)
                width = z.shape[1]
            else:
                fitted, _ = cm.fit_hgb(z, y, g, seed=seed)
                width = cm.PCA_DIMS
            classes = run.classes

            def vectors(rows: np.ndarray) -> np.ndarray:
                return space.transform(np.nan_to_num(self._classic(run, rows)))[:, :width]

            def proba_of_vectors(v: np.ndarray) -> np.ndarray:
                return cm._proba_in(fitted, v[:, :width], classes)

            def predict_images(images: list[Any]) -> np.ndarray:
                d = np.stack([classic_descriptor(i).vector for i in images])
                return proba_of_vectors(space.transform(d))

            dims = z.shape[1]
            return {
                "bank": ok,
                "bank_vectors": z[:, :width],
                "vectors": vectors,
                "proba": proba_of_vectors,
                "predict": predict_images,
                "transform": space.transform,
                "metric": "cosine",
                "space": f"the {min(width, dims)} PCA components of the classic descriptors the "
                "model reads (cosine)",
                "notes": notes,
            }
        if name in ("deep_linear", "deep_knn"):
            x = self._deep(run, train)
            good = np.isfinite(x).all(axis=1)
            ok, x = train[good], x[good]
            y = run.labels[ok].astype(str)
            classes = run.classes
            if name == "deep_linear":
                c = _setting_c(model.get("setting", ""))
                if c is None:
                    fitted, c, _ = cm.fit_logistic(
                        x, y, run.groups[ok].astype(str), scale=True, seed=seed
                    )
                    notes.append("C chosen again by the grouped inner cross-validation.")
                else:
                    fitted = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=c, max_iter=3000, class_weight=cm._class_weight(y)),
                    )
                    with cm._quiet():
                        fitted.fit(x, y)
                head = ex.head_from_pipeline(fitted, classes)
                scaler = fitted.steps[0][1]
                return {
                    "bank": ok,
                    "bank_vectors": scaler.transform(x),
                    "vectors": lambda rows: scaler.transform(np.nan_to_num(self._deep(run, rows))),
                    "proba": lambda v: head.proba(scaler.inverse_transform(v)),
                    "head": head,
                    "metric": "cosine",
                    "space": "the DINOv2 ViT-S/14 vectors the probe reads, standardised (cosine)",
                    "notes": notes,
                }
            from quantilo.vision.deep.probe import K_NEIGHBOURS, TEMPERATURE, knn_proba

            index = {c: i for i, c in enumerate(classes)}
            coded = np.array([index[v] for v in y])
            bank = cm._normalise(x.astype(np.float64))

            def knn(v: np.ndarray) -> np.ndarray:
                p, _ = knn_proba(
                    bank,
                    coded,
                    len(classes),
                    cm._normalise(v.astype(np.float64)),
                    K_NEIGHBOURS,
                    TEMPERATURE,
                )
                return p

            return {
                "bank": ok,
                "bank_vectors": x,
                "vectors": lambda rows: np.nan_to_num(self._deep(run, rows)),
                "proba": knn,
                "metric": "cosine",
                "space": "the DINOv2 ViT-S/14 vectors (cosine), the kNN's own neighbours",
                "notes": notes,
            }
        if name in ("knn", "ensemble", "patch_pca", *DETECTORS):
            x = self._classic(run, train)
            good = np.isfinite(x).all(axis=1)
            ok, x = train[good], x[good]
            space = cm._knn_space(x, seed)
            return {
                "bank": ok,
                "bank_vectors": space.transform(x),
                "vectors": lambda rows: space.transform(np.nan_to_num(self._classic(run, rows))),
                "metric": "euclidean",
                "space": (
                    "the classic descriptors, standardised and reduced to 50 components "
                    "(Euclidean distance: the nearest-normal score)"
                    if name not in DETECTORS
                    else "the classic descriptors, standardised and reduced to 50 components "
                    f"(Euclidean distance); {model['title']} itself compares patches, below"
                ),
                "notes": notes,
            }
        if name == "deep_knn_anomaly":
            x = self._deep(run, train)
            good = np.isfinite(x).all(axis=1)
            return {
                "bank": train[good],
                "bank_vectors": x[good],
                "vectors": lambda rows: np.nan_to_num(self._deep(run, rows)),
                "metric": "cosine",
                "space": "the DINOv2 ViT-S/14 vectors (cosine distance: the nearest-normal score)",
                "notes": notes,
            }
        raise AppError(
            f"No explanation for the model {name!r}.", code="not_supported", status_code=422
        )

    # --- disk cache ------------------------------------------------------------------------------

    def _cache_file(self, run: _Run, *parts: Any) -> Path:
        digest = hashlib.sha1("|".join([VERSION, *map(str, parts)]).encode()).hexdigest()[:24]
        return run.folder / "explain" / f"{digest}.json"

    def _cached[T: BaseModel](
        self, run: _Run, kind: type[T], parts: tuple[Any, ...], build: Callable[[], T]
    ) -> T:
        file = self._cache_file(run, *parts)
        if file.is_file():
            try:
                found = kind.model_validate_json(file.read_text(encoding="utf-8"))
                if hasattr(found, "cached"):
                    found.cached = True  # type: ignore[attr-defined]
                return found
            except ValueError:
                pass
        with self._lock:
            value = build()
        file.parent.mkdir(parents=True, exist_ok=True)
        partial = file.with_name(file.name + ".partial")
        partial.write_text(json.dumps(clean(value.model_dump(mode="json"))), encoding="utf-8")
        partial.replace(file)
        return value

    # --- "Why?" on one image ---------------------------------------------------------------------

    def _ref(self, run: _Run, row: int, **extra: Any) -> WhyImage:
        label = run.labels[row]
        return WhyImage(
            row=row,
            image_id=run.image_ids[row],
            label=None if label is None else str(label),
            **extra,
        )

    def _copies(self, run: _Run, row: int) -> list[int]:
        """The row and its copies (same group): never shown as its own evidence."""
        group = run.groups[row]
        return [int(i) for i in np.flatnonzero(run.groups == group)]

    def image(
        self, dataset_id: str, run_id: str, image_id: str, model_key: str | None = None
    ) -> WhyExplanation:
        with self._lock:
            run = self._run(dataset_id, run_id)
        model = run.model(model_key)
        row = run.row_of(image_id)
        return self._cached(
            run,
            WhyExplanation,
            ("image", image_id, model["key"]),
            lambda: self._image(run, model, row),
        )

    def _image(self, run: _Run, model: dict[str, Any], row: int) -> WhyExplanation:
        started = time.perf_counter()
        fold, tested = run.fold_of(row)
        fitted = self._fitted(run, model, fold)
        notes = list(fitted["notes"])
        bank: np.ndarray = fitted["bank"]
        query = fitted["vectors"](np.asarray([row]))[0]
        exclude = [i for i, r in enumerate(bank.tolist()) if r in set(self._copies(run, row))]
        record = run.predictions.get(row, {})
        truth = run.labels[row]
        truth_text = None if truth is None else str(truth)
        counterfactual = None
        if not tested:
            notes.append(
                "This image is one the model learnt from: its prediction is not a test, and its "
                "copies are left out of the evidence."
            )
        if run.task == "classification":
            classes = run.classes
            row_proba = fitted.get("row_proba")  # 118: a model whose predictions are kept
            kept_p = row_proba(row) if row_proba is not None else None
            if row_proba is not None and kept_p is None:
                raise AppError(
                    f"{model['title']} keeps its predictions for the scored images only: open an "
                    "image of the test.",
                    code="not_scored",
                    status_code=409,
                )
            p = kept_p if kept_p is not None else fitted["proba"](query[None])[0]
            order = np.argsort(-p, kind="mergesort")
            label = classes[int(order[0])]
            probabilities = [
                WhyClassProbability(label=classes[int(i)], probability=float(p[i])) for i in order
            ]
            correct = None if truth_text is None else truth_text == label
            verdict = (
                ""
                if correct is None
                else (
                    " It is right."
                    if correct
                    else f" It is wrong: the image is labelled {truth_text}."
                )
            )
            sentence = f"{model['title']} says {label} with probability {p[order[0]]:.2f}.{verdict}"
            prediction = WhyPrediction(
                label=label,
                probability=float(p[order[0]]),
                probabilities=probabilities,
                truth=truth_text,
                correct=correct,
                tested=tested,
                sentence=sentence,
            )
            bank_labels = [None if v is None else str(v) for v in run.labels[bank]]
            found = pt.nearest_examples(
                query,
                fitted["bank_vectors"],
                bank_labels,
                label=label,
                exclude=exclude,
                metric=fitted["metric"],
            )
            nearest = [
                self._ref(run, int(bank[n.index]), similarity=n.similarity)
                for n in found.neighbours
            ]
            proto, proto_sentence = self._nearest_prototype(run, model, fold, query, label)
            second = classes[int(order[1])] if len(order) > 1 else None
            cf = pt.counterfactual_example(
                query, fitted["bank_vectors"], bank_labels, label, target=second, exclude=exclude
            )
            if cf is not None:
                cf_row = int(bank[cf.index])
                cf_p = (
                    fitted["proba"](fitted["bank_vectors"][cf.index][None])[0]
                    if fitted.get("proba") is not None
                    else None
                )
                own = (
                    float(cf_p[classes.index(cf.label)])
                    if cf_p is not None and cf.label in classes
                    else None
                )
                counterfactual = WhyCounterfactual(
                    image=self._ref(run, cf_row, similarity=cf.similarity),
                    own_label=label,
                    probability=own,
                    sentence=(
                        f"The nearest training image labelled {cf.label}, the second most "
                        f"probable class (similarity {cf.similarity:.2f}); the model gives it "
                        f"{own:.2f} {cf.label}. What differs between the two is what would have "
                        "to change (R-VEX-04)."
                        if own is not None
                        else f"The nearest training image labelled {cf.label} (R-VEX-04)."
                    ),
                )
            evidence = WhyEvidence(
                space=fitted["space"],
                nearest=nearest,
                sentence=found.sentence,
                prototype=proto,
                prototype_sentence=proto_sentence,
            )
        else:
            prediction = self._anomaly_prediction(run, model, row, record, tested, truth_text)
            found = pt.nearest_examples(
                query,
                fitted["bank_vectors"],
                None,
                exclude=exclude,
                metric=fitted["metric"],
                noun="normal training image",
            )
            nearest = [
                self._ref(run, int(bank[n.index]), similarity=n.similarity, distance=n.distance)
                for n in found.neighbours
            ]
            if fitted["metric"] == "cosine":
                for ref in nearest:
                    ref.distance = None if ref.similarity is None else 1.0 - ref.similarity
            mean = float(np.mean([r.distance for r in nearest if r.distance is not None] or [0]))
            sentence = (
                f"The {len(nearest)} most similar normal training images, at a mean distance of "
                f"{mean:.3g}: that mean is the nearest-normal score itself (R-VAN-02)."
            )
            if model["key"] in DETECTORS:
                sentence = (
                    f"The {len(nearest)} most similar normal training images (mean distance "
                    f"{mean:.3g} in the classic descriptors), to compare with; the score comes "
                    "from the patches (R-VAN-02)."
                )
            evidence = WhyEvidence(space=fitted["space"], nearest=nearest, sentence=sentence)
        own_map, patch = self._own(run, model, row, record)
        return WhyExplanation(
            run_id=run.run_id,
            image_id=run.image_ids[row],
            row=row,
            task=run.task,
            model=WhyModel(
                key=model["key"], title=model["title"], default=bool(model.get("default"))
            ),
            models=run.choices(),
            fold=fold,
            prediction=prediction,
            evidence=evidence,
            counterfactual=counterfactual,
            methods=self._methods(run, model),
            seconds=round(time.perf_counter() - started, 2),
            notes=notes,
            own_map=own_map,
            patch=patch,
        )

    def _kept_map(self, run: _Run, model: dict[str, Any], row: int) -> Path | None:
        """113: the map the run kept for this model and row, if any."""
        path = run.folder / "maps" / str(model["key"]) / f"{row}.png"
        return path if path.is_file() else None

    def _own(
        self, run: _Run, model: dict[str, Any], row: int, record: dict[str, Any]
    ) -> tuple[str | None, "WhyPatch | None"]:
        """The model's own map (first plane) and, for PatchCore, the case the run kept."""
        if run.task != "anomaly":
            return None, None
        key = str(model["key"])
        if key in DETECTORS:
            own = "kept" if self._kept_map(run, model, row) is not None else None
        else:
            own = "patch_pca"
        patch = None
        box = record.get(f"{key}_box")
        normal_row = record.get(f"{key}_normal_row")
        normal_box = record.get(f"{key}_normal_box")
        distance = record.get(f"{key}_distance")
        if box and normal_row is not None and normal_box and distance is not None:
            size = self._size(run, row)
            normal_size = self._size(run, int(normal_row))
            if size and normal_size:
                patch = WhyPatch(
                    box=_relative(box, size),
                    normal=self._ref(run, int(normal_row), distance=float(distance)),
                    normal_box=_relative(normal_box, normal_size),
                    distance=float(distance),
                    sentence=(
                        "The most unusual area looks least like the normal images: its nearest "
                        f"normal patch is at distance {float(distance):.3g}, in the training image "
                        "shown beside it (R-VAD-06). This is how the score was computed."
                    ),
                )
        return own, patch

    def _size(self, run: _Run, row: int) -> tuple[int, int] | None:
        width = column(run.table, "width")[row]
        height = column(run.table, "height")[row]
        return (int(width), int(height)) if width and height else None

    def _anomaly_prediction(
        self,
        run: _Run,
        model: dict[str, Any],
        row: int,
        record: dict[str, Any],
        tested: bool,
        truth: str | None,
    ) -> WhyPrediction:
        score = record.get(f"{model['key']}_score")
        report = model.get("anomaly") or {}
        threshold = (report.get("threshold") or {}).get("threshold") if report else None
        if threshold is None:
            threshold = (model.get("threshold") or {}).get("threshold")
        flagged = None if score is None or threshold is None else bool(score >= threshold)
        defect = record.get("is_defect")
        correct = None if flagged is None or defect is None else flagged == bool(defect)
        if score is None:
            sentence = "This image was not scored: it is one of the normal images the model learnt."
        else:
            verdict = (
                ""
                if flagged is None
                else (
                    " Above the threshold: flagged as unusual."
                    if flagged
                    else " Below the threshold: read as normal."
                )
            )
            sentence = f"{model['title']}: score {score:.3g}.{verdict}"
        return WhyPrediction(
            score=score,
            threshold=threshold,
            flagged=flagged,
            truth=truth,
            correct=correct,
            tested=tested and score is not None,
            sentence=sentence,
        )

    def _nearest_prototype(
        self, run: _Run, model: dict[str, Any], fold: int, query: np.ndarray, label: str
    ) -> tuple[WhyImage | None, str | None]:
        protos = self._prototypes(run, model, fold)
        mine = next((c for c in protos if c.label == label), None)
        if mine is None or not mine.prototypes:
            return None, None
        fitted = self._fitted(run, model, fold)
        bank = fitted["bank"].tolist()
        position = {r: i for i, r in enumerate(bank)}
        vectors = fitted["bank_vectors"][[position[p] for p in mine.prototypes]]
        sims = pt.cosine(query, vectors)[0]
        best = int(np.argmax(sims))
        ref = self._ref(
            run, mine.prototypes[best], similarity=float(sims[best]), share=mine.shares[best]
        )
        sentence = (
            f"The closest of the {len(mine.prototypes)} prototypes of {label} (k-medoids), which "
            f"stands for {mine.shares[best]:.0%} of the class's training images (similarity "
            f"{sims[best]:.2f}, R-VEX-02)."
        )
        return ref, sentence

    def _prototypes(self, run: _Run, model: dict[str, Any], fold: int) -> list[pt.ClassPrototypes]:
        def build() -> list[pt.ClassPrototypes]:
            fitted = self._fitted(run, model, fold)
            bank: np.ndarray = fitted["bank"]
            labels = (
                [None if v is None else str(v) for v in run.labels[bank]]
                if run.task == "classification"
                else ["normal"] * len(bank)
            )
            found = pt.class_prototypes(fitted["bank_vectors"], labels)
            for c in found:
                c.prototypes = [int(bank[i]) for i in c.prototypes]
                c.criticisms = [int(bank[i]) for i in c.criticisms]
            return found

        return self._models.get_or(
            (run.dataset_id, run.run_id, model["key"], fold, "protos"), build
        )

    def _methods(self, run: _Run, model: dict[str, Any]) -> list[WhyMethod]:
        from quantilo.vision import browser, deep

        name = model["key"]
        out: list[WhyMethod] = []
        if name in MORE_MODELS:  # 118: their own evidence is shown; maps explain the probe
            from app.modules.vision.more_explain import NOT_A_MAP

            for key, title, level, cost in (
                ("occlusion", "Occlusion", "advanced", "about 5 s (65 predictions)"),
                ("grad_cam", "Grad-CAM", "advanced", "about 1 s"),
                ("rollout", "Attention rollout", "advanced", "about 1 s"),
                ("saliency", "Vanilla saliency", "more", "about 1 s"),
                ("smoothgrad", "SmoothGrad", "more", "about 5 s (25 passes)"),
                ("integrated_gradients", "Integrated Gradients", "more", "about 5 s (32 steps)"),
                ("sanity", "Sanity test (random weights)", "more", "about 15 s"),
            ):
                out.append(
                    WhyMethod(
                        key=key,
                        title=title,
                        level=level,  # type: ignore[arg-type]
                        available=False,
                        reason=NOT_A_MAP,
                        cost=cost,
                    )
                )
            return out
        if run.task == "classification":
            out.append(
                WhyMethod(
                    key="occlusion",
                    title="Occlusion",
                    level="advanced",
                    available=True,
                    cost="about 5 s (65 predictions)",
                )
            )
            deep_reason = None
            if browser.active():  # 119, R-VIS-WEB-03: no torch in the page
                deep_reason = browser.DESKTOP_REASON
            elif name != "deep_linear":
                deep_reason = (
                    "Needs a differentiable model: the linear probe on DINOv2. The classic "
                    "descriptors (histograms, HOG, LBP) have no gradient: use occlusion."
                    if name in ("logistic", "hgb")
                    else "The kNN has no gradient: open the linear probe, or use occlusion."
                )
            else:
                backbone = deep.status().backbone(cm.DEEP_BACKBONE)
                if not backbone.available:
                    deep_reason = backbone.reason
            for key, title, level, cost in (
                ("grad_cam", "Grad-CAM", "advanced", "about 1 s"),
                ("rollout", "Attention rollout", "advanced", "about 1 s"),
                ("saliency", "Vanilla saliency", "more", "about 1 s"),
                ("smoothgrad", "SmoothGrad", "more", "about 5 s (25 passes)"),
                ("integrated_gradients", "Integrated Gradients", "more", "about 5 s (32 steps)"),
                ("sanity", "Sanity test (random weights)", "more", "about 15 s"),
            ):
                out.append(
                    WhyMethod(
                        key=key,
                        title=title,
                        level=level,  # type: ignore[arg-type]
                        available=deep_reason is None,
                        reason=deep_reason,
                        cost=cost,
                    )
                )
        elif name in DETECTORS:
            if self._has_maps(run, model):
                out.append(
                    WhyMethod(
                        key="kept",
                        title=f"{model['title']} map",
                        level="first",
                        available=True,
                        own=True,
                        cost="instant (kept with the run)",
                    )
                )
        else:
            out.append(
                WhyMethod(
                    key="patch_pca",
                    title="Patch PCA map",
                    level="first",
                    available=True,
                    own=True,
                    cost="a few seconds the first time",
                )
            )
            detector = deep.status().detector("patchcore")
            out.append(
                WhyMethod(
                    key="patchcore",
                    title="PatchCore: nearest normal patch",
                    level="advanced",
                    available=detector.available,
                    reason=detector.reason,
                    own=True,
                    cost="under a minute the first time, then instant",
                )
            )
        return out

    # --- maps ------------------------------------------------------------------------------------

    def map(
        self,
        dataset_id: str,
        run_id: str,
        image_id: str,
        method: str,
        model_key: str | None = None,
        target: str | None = None,
    ) -> WhyMap:
        if method not in MAP_METHODS:
            raise AppError(f"Unknown method {method!r}.", code="not_found", status_code=404)
        with self._lock:
            run = self._run(dataset_id, run_id)
        model = run.model(model_key)
        row = run.row_of(image_id)
        if run.task == "classification" and target is not None and target not in run.classes:
            raise AppError(f"No class {target!r}.", code="not_found", status_code=404)
        return self._cached(
            run,
            WhyMap,
            ("map", image_id, model["key"], method, target),
            lambda: self._map(run, model, row, method, target),
        )

    def _classifier(self, fitted: dict[str, Any]) -> ex.Classifier:
        head = fitted.get("head")
        if head is None:
            raise AppError(
                "This model has no gradient: use occlusion.", code="not_supported", status_code=422
            )
        return ex.Classifier(self._net(), head, self._device())

    def _target(self, run: _Run, fitted: dict[str, Any], row: int, target: str | None) -> int:
        if target is not None:
            return run.classes.index(target)
        p = fitted["proba"](fitted["vectors"](np.asarray([row])))[0]
        return int(np.argmax(p))

    def _map(
        self, run: _Run, model: dict[str, Any], row: int, method: str, target: str | None
    ) -> WhyMap:
        fold, _ = run.fold_of(row)
        if method == "kept":
            return self._kept(run, model, row)
        if method == "patch_pca":
            return self._patch_pca(run, fold, row)
        if method == "patchcore":
            return self._patchcore_map(run, fold, row)
        if run.task != "classification":
            raise AppError(
                "Post-hoc maps explain a classifier.", code="not_supported", status_code=422
            )
        if model["key"] in MORE_MODELS:
            from app.modules.vision.more_explain import NOT_A_MAP

            raise AppError(NOT_A_MAP, code="not_supported", status_code=422)
        fitted = self._fitted(run, model, fold)
        index = self._target(run, fitted, row, target)
        if method == "occlusion":
            if "predict" in fitted:
                predict = fitted["predict"]
                image = self._decode(run, row)
            else:
                clf_or_none = fitted.get("head")
                if clf_or_none is not None:
                    predict = self._classifier(fitted).predict
                else:  # the kNN: embed the blurred copies, then vote

                    def predict(images: Sequence[Any]) -> np.ndarray:
                        from quantilo.vision.deep.embed import embed_images

                        v = embed_images(images, self._net(), allow_gpu=self.vision.compute())
                        return fitted["proba"](np.nan_to_num(v.vectors))

                image = self._decode(run, row, 448)
            found = ex.occlusion(image, predict, index, classes=run.classes)
        else:
            clf = self._classifier(fitted)
            image = self._decode(run, row, 448)
            try:
                if method == "rollout":
                    found = ex.attention_rollout(clf, image)
                else:
                    found = ex.GRADIENT_METHODS[method](clf, image, index)
            finally:
                from quantilo.vision.deep.anomaly import free_gpu

                free_gpu()
        if method in ("saliency", "smoothgrad", "integrated_gradients"):
            found.values = ex.pool(found.values, 112)  # a lighter picture, same reading
        return self._map_out(found)

    def _map_out(self, found: ex.ExplanationMap, patch: WhyPatch | None = None) -> WhyMap:
        rows, cols = found.values.shape
        extra = {k: float(v) for k, v in found.extra.items() if isinstance(v, int | float)}
        return WhyMap(
            method=found.method,
            title=found.title,
            unit=found.unit,
            sentence=found.sentence,
            warning=found.warning,
            own=found.own,
            blocky=found.blocky,
            box=found.box,
            target=found.target,
            rows=rows,
            columns=cols,
            low=found.low,
            high=found.high,
            png=_data_url(found.values),
            seconds=found.seconds,
            notes=found.notes,
            extra=extra,
            patch=patch,
        )

    def _has_maps(self, run: _Run, model: dict[str, Any]) -> bool:
        folder = run.folder / "maps" / str(model["key"])
        return folder.is_dir()

    def _kept(self, run: _Run, model: dict[str, Any], row: int) -> WhyMap:
        """113: the detector's own map, as the run kept it (8-bit grey on the model's range)."""
        path = self._kept_map(run, model, row)
        if path is None:
            raise AppError(
                "The run kept no map for this image (only the scored images keep one).",
                code="not_found",
                status_code=404,
            )
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        low, high = model.get("map_range") or (0.0, 1.0)
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return WhyMap(
            method="kept",
            title=f"{model['title']} map",
            unit="anomaly score of each place",
            sentence=(
                f"The map {model['title']} computed when it scored this image: the score is its "
                "largest value. Grey levels share one range across the run, so maps compare."
            ),
            warning="",
            own=True,
            blocky=False,
            box=(0.0, 0.0, 1.0, 1.0),
            target=None,
            rows=height,
            columns=width,
            low=float(low),
            high=float(high),
            png=f"data:image/png;base64,{data}",
            seconds=0.0,
            notes=["This map is the model's own computation, not a post-hoc explanation."],
        )

    def _patch_pca(self, run: _Run, fold: int, row: int) -> WhyMap:
        """The patch PCA's own map (R-VAN-01), its model fitted again on the fold's normals by
        the run's rule (the same seed; patches drawn as the run drew them for its first fold)."""
        started = time.perf_counter()
        seed = int(run.result.get("seed", 0))

        def fit() -> Any:
            train, _ = run.folds[fold]
            rng = np.random.default_rng(seed)
            per_image = max(1, cm.MAX_PATCHES // max(1, len(train)))
            calibrating = set(
                rng.choice(train, min(len(train), cm.CALIBRATION_IMAGES), replace=False).tolist()
            )
            bank, kept = [], []
            for r in train.tolist():
                luma = cm.luma_square(run.sources[r])
                if luma is None:
                    continue
                if r in calibrating:
                    kept.append(luma)
                p = cm.patches_of(luma)
                if len(p) > per_image:
                    p = p[rng.choice(len(p), per_image, replace=False)]
                bank.append(p)
            if not bank:
                raise AppError("No training image could be decoded.", code="not_decoded")
            return cm.PatchPCA(seed=seed).fit(np.concatenate(bank)).calibrate(kept)

        model = self._models.get_or((run.dataset_id, run.run_id, "patch_pca", fold), fit)
        luma = cm.luma_square(run.sources[row])
        if luma is None:
            raise AppError("This image could not be decoded.", code="not_decoded")
        z = model.errors(luma)
        top = np.unravel_index(int(np.argmax(z)), z.shape)
        values = np.clip(z, 0.0, None).astype(np.float32)
        found = ex.ExplanationMap(
            method="patch_pca",
            title="Patch PCA map",
            values=values,
            unit="robust z of the rebuild error",
            sentence=(
                f"Each 16 x 16 patch rebuilt from the normal patches: the worst rebuilt, at row "
                f"{top[0] + 1}, column {top[1] + 1}, is {z.max():.1f} MADs above the normal "
                "images at that place. The score is the 99 % quantile of these (R-VAN-01)."
            ),
            warning="",
            own=True,
            seconds=round(time.perf_counter() - started, 2),
            notes=["This map is the model's own computation, not a post-hoc explanation."],
        )
        return self._map_out(found)

    def _patchcore_map(self, run: _Run, fold: int, row: int) -> WhyMap:
        """PatchCore (115) fitted on the fold's normal training images, kept for the next image:
        its map and the nearest normal patch (R-VAD-06)."""
        from quantilo.vision import deep
        from quantilo.vision.deep.patchcore import PatchCore

        detector = deep.status().detector("patchcore")
        if not detector.available:
            raise AppError(
                detector.reason or "PatchCore is not available here.",
                code="not_available",
                status_code=409,
            )
        started = time.perf_counter()
        train, _ = run.folds[fold]
        normal = train.tolist()

        def fit() -> Any:
            model = PatchCore(allow_gpu=self.vision.compute(), confirm_long=True)
            return model.fit([run.sources[r] for r in normal])  # type: ignore[misc]

        model = self._patchcore.get_or((run.dataset_id, run.run_id, fold), fit)
        try:
            result = model.score([run.sources[row]])  # type: ignore[list-item]
        finally:
            from quantilo.vision.deep.anomaly import free_gpu

            free_gpu()
        if not result.ok[0]:
            raise AppError("This image could not be decoded.", code="not_decoded")
        top = result.top[0]
        assert top is not None
        size = result.sizes[0] or (result.side, result.side)
        normal_row = normal[top.normal_image]
        normal_size = model.fit_sizes[top.normal_image] or (result.side, result.side)

        def rel(box: tuple[int, int, int, int], wh: tuple[int, int]) -> tuple[float, ...]:
            return (box[0] / wh[0], box[1] / wh[1], box[2] / wh[0], box[3] / wh[1])

        patch = WhyPatch(
            box=rel(top.box, size),  # type: ignore[arg-type]
            normal=self._ref(run, normal_row, distance=top.distance),
            normal_box=rel(top.normal_box, normal_size),  # type: ignore[arg-type]
            distance=top.distance,
            sentence=(
                f"The most unusual area looks least like the normal images: its nearest normal "
                f"patch, in training image {run.image_ids[normal_row][:8]}, is at distance "
                f"{top.distance:.3g} (R-VAD-06)."
            ),
        )
        found = ex.ExplanationMap(
            method="patchcore",
            title=model.title,
            values=result.patch_scores[0].astype(np.float32),
            unit="distance to the nearest normal patch",
            sentence=(
                f"{model.title}: score {result.scores[0]:.3g}, the largest patch distance; "
                f"{len(normal):,} normal images in the memory bank's source (R-VAD-06)."
            ),
            warning="",
            own=True,
            seconds=round(time.perf_counter() - started, 2),
            notes=[
                *model.notes,
                "This map is the model's own computation: the score is its largest cell.",
            ],
        )
        return self._map_out(found, patch)

    # --- sanity test -----------------------------------------------------------------------------

    def sanity(
        self,
        dataset_id: str,
        run_id: str,
        image_id: str,
        method: str = "grad_cam",
        model_key: str | None = None,
        target: str | None = None,
    ) -> WhySanity:
        if method not in ex.SANITY_METHODS:
            raise AppError(f"No sanity test for {method!r}.", code="not_found", status_code=404)
        with self._lock:
            run = self._run(dataset_id, run_id)
        model = run.model(model_key)
        row = run.row_of(image_id)

        def build() -> WhySanity:
            fold, _ = run.fold_of(row)
            fitted = self._fitted(run, model, fold)
            clf = self._classifier(fitted)
            index = self._target(run, fitted, row, target)
            result = ex.sanity_check(clf, self._decode(run, row, 448), index, method)
            rows, cols = result.stages[0].values.shape
            stages = [
                WhySanityStage(
                    label=s.label,
                    correlation=s.correlation if np.isfinite(s.correlation) else None,
                    png=_data_url(s.values),
                    low=float(np.min(s.values)),
                    high=float(np.max(s.values)),
                )
                for s in result.stages
            ]
            return WhySanity(
                method=result.method,
                title=result.title,
                passes=result.passes,
                sentence=result.sentence,
                box=result.box,
                target=result.target,
                rows=rows,
                columns=cols,
                stages=stages,
                seconds=result.seconds,
            )

        return self._cached(
            run, WhySanity, ("sanity", image_id, model["key"], method, target), build
        )

    # --- the Explain tab -------------------------------------------------------------------------

    def overview(
        self, dataset_id: str, run_id: str, model_key: str | None = None
    ) -> ExplainOverview:
        with self._lock:
            run = self._run(dataset_id, run_id)
        model = run.model(model_key)
        return self._cached(
            run, ExplainOverview, ("overview", model["key"]), lambda: self._overview(run, model)
        )

    def _overview(self, run: _Run, model: dict[str, Any]) -> ExplainOverview:
        started = time.perf_counter()
        fitted = self._fitted(run, model, 0)
        found = self._prototypes(run, model, 0)
        classes = [
            WhyClassPrototypes(
                label=c.label,
                size=c.size,
                prototypes=[
                    self._ref(run, r, share=s) for r, s in zip(c.prototypes, c.shares, strict=True)
                ],
                criticisms=[
                    self._ref(run, r, witness=w)
                    for r, w in zip(c.criticisms, c.witness, strict=True)
                ],
                notes=c.notes,
            )
            for c in found
        ]
        noun = "class" if run.task == "classification" else "set of normal images"
        sentence = (
            f"{len(found[0].prototypes) if found else 0} prototypes per {noun} (k-medoids) and the "
            "images they represent worst (MMD-critic criticisms), in the space the model reads "
            "(R-VEX-02, R-VEX-03)."
        )
        families, families_sentence = None, None
        if run.task == "classification" and any(m["key"] == "logistic" for m in run.models):
            families, families_sentence = self._families(run)
        return ExplainOverview(
            run_id=run.run_id,
            task=run.task,
            model=WhyModel(
                key=model["key"], title=model["title"], default=bool(model.get("default"))
            ),
            models=run.choices(),
            space=fitted["space"],
            classes=classes,
            sentence=sentence,
            families=families,
            families_sentence=families_sentence,
            seconds=round(time.perf_counter() - started, 2),
            notes=list(fitted["notes"]),
        )

    def _families(self, run: _Run) -> tuple[list[WhyFamily], str]:
        """R-VEX-13: permutation importance of the descriptor families for the classic logistic
        model, on the first fold's test images (each family's columns shuffled together across
        images, 5 times, seed 0; drop in balanced accuracy)."""
        from sklearn.metrics import balanced_accuracy_score

        model = run.model("logistic")
        fitted = self._fitted(run, model, 0)
        _train, test = run.folds[0]
        x = self._classic(run, test)
        good = np.isfinite(x).all(axis=1)
        x, rows = x[good].astype(np.float64), test[good]
        y = run.labels[rows].astype(str)
        classes = run.classes
        seed = int(run.result.get("seed", 0))
        rng = np.random.default_rng(seed)

        def score(d: np.ndarray) -> float:
            p = fitted["proba"](fitted["transform"](d))
            return float(balanced_accuracy_score(y, [classes[i] for i in p.argmax(axis=1)]))

        base = score(x)
        out = []
        for key, title, low, high in FAMILIES:
            drops = []
            for _ in range(PERMUTATIONS):
                shuffled = x.copy()
                shuffled[:, low:high] = x[rng.permutation(len(x))][:, low:high]
                drops.append(base - score(shuffled))
            columns = (high or cm.CLASSIC_SIZE) - low
            out.append(
                WhyFamily(
                    key=key,
                    title=title,
                    drop=float(np.mean(drops)),
                    spread=float(np.std(drops)),
                    columns=columns,
                )
            )
        out.sort(key=lambda f: -f.drop)
        lead = out[0]
        sentence = (
            f"The classic logistic model leans most on {lead.title[0].lower()}{lead.title[1:]}: "
            f"shuffling it across "
            f"the {len(rows):,} test images costs {lead.drop:.3f} of balanced accuracy "
            f"(base {base:.3f}, {PERMUTATIONS} shuffles, R-VEX-13)."
        )
        return out, sentence


def _relative(box: list[Any], size: tuple[int, int]) -> tuple[float, float, float, float]:
    w, h = size
    x0, y0, x1, y1 = (float(v) for v in box)
    return (x0 / w, y0 / h, x1 / w, y1 / h)


__all__ = ["ExplainOverview", "ExplainService", "WhyExplanation", "WhyMap", "WhySanity"]
