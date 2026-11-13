"""Vision models (112): the task rule, training and evaluation jobs, kept runs.

A run reads the image set's table (`format = 'images'`, 106), its stored files and masks, trains
every first-plan model of `quantilo.vision.classic_models` and keeps the result with the dataset:
`datasets/<id>/vision/runs/<run>/run.json` and `predictions.parquet` (deleted with the dataset, no
migration). Vectors are cached per image in `images/<id>/embeddings/` (deleted with the set).

One job at a time (a single worker, like ML): memory stays bounded. Progress and a message are kept
on the job; `cancel` stops it at the next checkpoint and nothing is kept.

113: the anomaly detector of 115 / 116 is chosen per run (`detector`, PatchCore by default when its
weights are here); the anomaly maps (8-bit grey PNG, scaled on the model's `map_range`) and the
autoencoders' reconstructions are kept in `maps/<model>/<row>.png` and `recon/<model>/<row>.png`;
weights are downloaded only on `POST /weights/{key}/download` (the page asks first, size shown).
"""

import json
import math
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from pydantic import BaseModel, Field
from quantilo.vision.classic_models import (
    Cancelled,
    DeepOption,
    VisionResult,
    choose_detector,
    deep_option,
    evaluate,
)
from quantilo.vision.deep.budget import LongRun
from quantilo.vision.split import plan_split
from quantilo.vision.store import ImageStore
from quantilo.vision.task import Task, TaskChoice, detect_task

from app.core.errors import AppError
from app.core.storage import Workspace
from app.services.datasets.repository import DatasetRecord
from app.services.datasets.service import DatasetService
from app.services.image_sets import FORMAT, ImageSetService

Stage = Literal["features", "embedding", "training", "saving"]
# c3, R-VAN-07: patchcore_ae, the ensemble of Advanced > More
Detector = Literal["auto", "patchcore", "padim", "autoencoder", "vae", "patchcore_ae", "none"]
More = Literal["finetune", "protopnet", "concept_bottleneck"]  # 118: Advanced > More
KEPT_JOBS = 32
RUN_ID = re.compile(r"^[0-9a-f]{12}$")
MODEL_KEY = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class VisionRunRequest(BaseModel):
    task: Task | None = None  # None: the task of the rule (R-VML)
    deep: bool | None = None  # None: DINOv2 when available (R-VCL-06); False: classic only
    threshold: float | None = None  # anomalies: a threshold set by hand (R-VTH-03)
    confirm_long: bool = False  # R-VDP-05: go on when the deep vectors take more than 15 min
    # anomalies (113, R-VAN-06): auto = PatchCore when its weights are here; none = classic only
    detector: Detector = "auto"
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    # 118 (classification): Advanced > More models, trained after the first plan on its folds;
    # the bottleneck's concepts by key (None: every concept the set has)
    more: list[More] = []
    concepts: list[str] | None = None
    # c3, R-VML-05: the numeric column a regression predicts (None: the first one offered)
    target: str | None = None


@dataclass
class VisionJob:
    id: str
    dataset_id: str
    status: Literal["running", "done", "failed", "cancelled"]
    stage: Stage
    progress: float = 0.0
    message: str | None = None
    run_id: str | None = None
    error: str | None = None
    # R-VDP-05: the estimate when a deep step would take more than 15 minutes (run again with
    # `confirm_long` to go on)
    long_run: str | None = None


class SplitSummary(BaseModel):
    kind: str
    code: str
    reason: str
    notes: list[str]
    folds: int
    train: int  # images learnt from (first fold)
    scored: int
    holdout: int
    leaks: int


class WeightOut(BaseModel):
    """A file of pretrained weights (114): downloaded only after the user confirms."""

    key: str
    title: str
    size: str
    licence: str
    note: str
    downloaded: bool
    download: Literal["idle", "running", "failed"] = "idle"
    progress: float = 0.0
    error: str | None = None


class DetectorOut(BaseModel):
    """An anomaly detector of 115 / 116 and whether it can run here (R-VDP-01, R-VDP-02)."""

    key: str
    title: str
    level: Literal["first", "advanced", "more"]
    available: bool
    reason: str | None
    note: str
    weights: list[str]  # keys of `weights` it may use (none for the autoencoders)


class TaskOut(BaseModel):
    choice: TaskChoice
    split: SplitSummary | None
    deep: DeepOption
    models: list[str]  # titles of the models a run would train
    detectors: list[DetectorOut] = []
    default_detector: str | None = None  # what `detector = auto` would run
    weights: list[WeightOut] = []
    installed: bool = False  # the optional extra vision is here (weights can be downloaded)
    install: str = ""  # the command that adds it
    more: list["MoreOptionOut"] = []  # 118: Advanced > More, each with its reason and estimate
    concepts: list["ConceptOut"] = []  # 118: concepts of the bottleneck and of TCAV


class VisionRunSummary(BaseModel):
    id: str
    created_at: str
    task: Task
    chosen_by: str
    default: str | None
    default_title: str | None
    metric: str | None  # "balanced_accuracy", "accuracy", "auroc" or None
    score: float | None
    models: list[str]
    deep: bool
    scored: int
    seconds: float
    sentence: str


class MislabelledImageOut(BaseModel):
    row: int
    label: str
    suggested: str
    suggested_proba: float
    label_proba: float


class MislabelledOut(BaseModel):
    """R-VQ-16 (Quality > Advanced, b2): what the latest classification run suggests. `run_id`
    is None when no classification run was kept yet (the reason says how to get one)."""

    run_id: str | None = None
    created_at: str | None = None
    model: str | None = None
    message: str
    reason: str
    count: int = 0
    checked: int = 0
    rows: list[int] = []
    images: list[MislabelledImageOut] = []
    notes: list[str] = []
    parameters: dict[str, float] = {}


class VisionRun(BaseModel):
    id: str
    dataset_id: str
    created_at: str
    request: VisionRunRequest
    result: VisionResult


def _now() -> str:
    return datetime.now(UTC).isoformat()


def clean(value: Any) -> Any:
    """NaN and infinities become null (JSON has none); nested lists and dicts are walked."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    return value


def _summary(choice: TaskChoice) -> list[str]:
    titles = {
        "classification": [
            "Logistic regression on classic descriptors",
            "Gradient boosting (HGB) on classic descriptors",
        ],
        "anomaly": [
            "PCA of patches (reconstruction error)",
            "Nearest normal images (classic descriptors)",
            "Patch PCA + nearest normals (average rank)",
        ],
        "clusters": ["k-means on classic descriptors (k by silhouette)"],
        "regression": ["Ridge regression on classic descriptors"],
        "multilabel": ["One logistic regression per label (classic descriptors)"],
    }
    return titles.get(choice.task or "", [])


class VisionService:
    def __init__(
        self,
        workspace: Workspace,
        datasets: DatasetService,
        image_sets: ImageSetService,
        compute: Callable[[], bool] = lambda: True,
    ) -> None:
        """`compute()` says whether the GPU may be used (Settings > Compute)."""
        self.workspace = workspace
        self.datasets = datasets
        self.image_sets = image_sets
        self.compute = compute
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision")
        self._jobs: OrderedDict[str, VisionJob] = OrderedDict()
        self._stops: set[str] = set()
        self._lock = threading.Lock()
        self._downloads: dict[str, dict[str, Any]] = {}

    # --- reading the set ------------------------------------------------------------------------

    def _record(self, dataset_id: str) -> DatasetRecord:
        record = self.datasets.get(dataset_id)
        if record.format != FORMAT:
            raise AppError(
                "Vision models need an image set: import a folder, a ZIP or a Parquet file of "
                "images.",
                code="not_images",
                status_code=409,
            )
        if record.status != "ready":
            raise AppError("This dataset is not ready yet.", code="not_ready", status_code=409)
        return record

    def _table(self, record: DatasetRecord) -> Any:
        path = Path(record.storage_path)
        if not path.is_file():
            raise AppError("The image table is missing: reimport the set.", code="not_ready")
        return pq.read_table(path)

    def _store(self, record: DatasetRecord) -> ImageStore:
        return ImageStore(self.workspace.root, record.id, Path(record.storage_path))

    def _runs_dir(self, dataset_id: str) -> Path:
        return self.workspace.dataset_dir(dataset_id) / "vision" / "runs"

    def _run_dir(self, dataset_id: str, run_id: str) -> Path:
        if not RUN_ID.match(run_id):
            raise AppError("Run not found.", code="not_found", status_code=404)
        return self._runs_dir(dataset_id) / run_id

    # --- task ------------------------------------------------------------------------------------

    def task(self, dataset_id: str, task: Task | None = None, target: str | None = None) -> TaskOut:
        record = self._record(dataset_id)
        table = self._table(record)
        try:
            choice = detect_task(table, task, target)
        except ValueError as error:
            raise AppError(str(error), code="task_impossible", status_code=422) from error
        split = None
        if choice.task is not None and not choice.blocked:
            plan = plan_split(table, choice)
            split = SplitSummary(
                kind=plan.kind,
                code=plan.code,
                reason=plan.reason,
                notes=plan.notes,
                folds=len(plan.folds),
                train=len(plan.folds[0].train),
                scored=len(plan.scored),
                holdout=len(plan.holdout),
                leaks=plan.leaks,
            )
        option = deep_option(None)
        models = _summary(choice)
        if option.used and choice.task == "classification":
            models += [
                "Linear probe on DINOv2 ViT-S/14 (frozen)",
                "Nearest neighbours on DINOv2 ViT-S/14 (cosine)",
            ]
        if option.used and choice.task == "anomaly":
            models.append("Nearest normal images (DINOv2 ViT-S/14)")
        if option.used and choice.task == "regression":
            models.append("Ridge regression on DINOv2 ViT-S/14 (frozen)")
        if option.used and choice.task == "multilabel":
            models.append("One logistic regression per label (DINOv2 ViT-S/14, frozen)")
        if choice.task == "clusters":
            option = DeepOption(
                used=False, reason="Clusters use the classic descriptors (R-VCU-01)."
            )
        detectors: list[DetectorOut] = []
        default_detector = None
        if choice.task == "anomaly":
            detectors = self.detectors()
            default_detector, _ = choose_detector(None, None, [])
            if default_detector is not None:
                models.append(next(d.title for d in detectors if d.key == default_detector))
        keys = {"classification": ["dinov2_s14"], "anomaly": ["wide_resnet50_2", "resnet18"]}
        concepts = concept_list(self, record, table)
        more_options = []
        if choice.task is not None:
            from quantilo.vision.deep import more

            more_options = [
                MoreOptionOut(**vars(o))
                for o in more.options(
                    choice.task,
                    split.train if split else 0,
                    split.folds if split else 1,
                    sum(1 for c in concepts if c.bottleneck),
                    allow_gpu=self.compute(),
                    deep_ready=option.used,
                )
            ]
        return TaskOut(
            choice=choice,
            split=split,
            deep=option,
            models=models,
            detectors=detectors,
            default_detector=default_detector,
            weights=self.weights(keys.get(choice.task or "", [])),
            installed=_installed(),
            install=_install_command(),
            more=more_options,
            concepts=concepts,
        )

    # --- deep options and their weights (114, 115, 116) ----------------------------------------

    def detectors(self) -> list[DetectorOut]:
        from quantilo.vision import deep

        found = deep.status()
        return [
            DetectorOut(
                key=d.key,
                title=d.title,
                level=d.level if d.level in ("first", "more") else "advanced",
                available=d.available,
                reason=d.reason,
                note=d.note,
                weights=list(d.weights),
            )
            for d in found.detectors
        ]

    def weights(self, keys: list[str] | None = None) -> list[WeightOut]:
        from quantilo.vision.deep import weights

        out = []
        for key in keys if keys is not None else list(weights.MANIFEST):
            spec = weights.MANIFEST[key]
            state = self._downloads.get(key, {})
            out.append(
                WeightOut(
                    key=key,
                    title=spec.title,
                    size=spec.size_text,
                    licence=spec.licence,
                    note=spec.note,
                    downloaded=weights.downloaded(spec),
                    download=state.get("status", "idle"),
                    progress=state.get("progress", 0.0),
                    error=state.get("error"),
                )
            )
        return out

    def download(self, key: str) -> WeightOut:
        """Fetch one weights file in the background, after the user confirmed (R-VDP-02): the
        page showed its size and licence first. Already there: nothing is fetched."""
        from quantilo.vision import deep
        from quantilo.vision.deep import weights

        if key not in weights.MANIFEST:
            raise AppError("Unknown weights.", code="not_found", status_code=404)
        if not deep.installed():
            raise AppError(
                deep.status().reason or "The optional extra vision is not installed.",
                code="extra_missing",
                status_code=409,
            )
        with self._lock:
            state = self._downloads.get(key, {})
            if state.get("status") != "running" and not weights.downloaded(key):
                self._downloads[key] = {"status": "running", "progress": 0.0, "error": None}
                threading.Thread(
                    target=self._download, args=(key,), daemon=True, name=f"weights-{key}"
                ).start()
        return self.weights([key])[0]

    def _download(self, key: str) -> None:
        from quantilo.vision.deep import weights

        def progress(have: int, total: int) -> None:
            self._downloads[key]["progress"] = round(have / max(1, total), 4)

        try:
            weights.download(key, confirm=True, progress=progress)
            self._downloads[key] = {"status": "idle", "progress": 1.0, "error": None}
        except Exception as error:  # network, size or SHA-256: said as it is
            self._downloads[key] = {"status": "failed", "progress": 0.0, "error": str(error)}

    # --- jobs ------------------------------------------------------------------------------------

    def start(self, dataset_id: str, request: VisionRunRequest) -> VisionJob:
        record = self._record(dataset_id)
        table = self._table(record)
        try:
            choice = detect_task(table, request.task, request.target)
        except ValueError as error:
            raise AppError(str(error), code="task_impossible", status_code=422) from error
        if choice.task is None or choice.blocked:
            raise AppError(choice.blocked or choice.message, code="too_few_images", status_code=409)
        job = VisionJob(uuid.uuid4().hex[:12], dataset_id, "running", "features")
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > KEPT_JOBS:
                self._jobs.popitem(last=False)
        self._executor.submit(self._run, job, record, request)
        return job

    def job(self, job_id: str) -> VisionJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise AppError(
                "This run is no longer followed: open the runs.",
                code="unknown_job",
                status_code=404,
            )
        return job

    def cancel(self, job_id: str) -> VisionJob:
        job = self.job(job_id)
        if job.status == "running":
            self._stops.add(job_id)
        return job

    def _run(self, job: VisionJob, record: DatasetRecord, request: VisionRunRequest) -> None:
        def stopped() -> bool:
            return job.id in self._stops

        def progress(share: float, message: str) -> None:
            job.progress = round(min(0.99, share), 4)
            job.message = message
            lowered = message.lower()
            if lowered.startswith("descriptors"):
                job.stage = "features"
            elif "dinov2" in lowered:
                job.stage = "embedding"
            else:
                job.stage = "training"

        folder: Path | None = None
        try:
            table = self._table(record)
            store = self._store(record)
            sources = [store.path(v) if v else None for v in table["stored_path"].to_pylist()]
            masks = [store.path(v) if v else None for v in table["mask_path"].to_pylist()]
            chosen = (
                engine_concepts(self, record, table, request.concepts)
                if "concept_bottleneck" in request.more
                else []
            )
            outcome = evaluate(
                table,
                sources,
                masks=masks,
                task=request.task,
                deep=request.deep,
                threshold=request.threshold,
                cache_dir=store.root / "embeddings",
                allow_gpu=self.compute(),
                confirm_long=request.confirm_long,
                seed=request.seed,
                progress=progress,
                cancel=stopped,
                detector=request.detector,
                more=list(request.more),
                concepts=chosen,
                target=request.target,
            )
            if stopped():
                raise Cancelled("Stopped: nothing was saved.")
            job.stage, job.message = "saving", "Saving the run"
            run = VisionRun(
                id=uuid.uuid4().hex[:12],
                dataset_id=record.id,
                created_at=_now(),
                request=request,
                result=outcome.result,
            )
            folder = self._run_dir(record.id, run.id)
            folder.mkdir(parents=True, exist_ok=True)
            pq.write_table(outcome.predictions, folder / "predictions.parquet")
            _write_images(folder, outcome.result, outcome.extra)
            partial = folder / "run.json.partial"
            partial.write_text(json.dumps(clean(run.model_dump(mode="json"))), encoding="utf-8")
            partial.replace(folder / "run.json")
            job.run_id, job.status, job.progress, job.message = run.id, "done", 1.0, "Done"
        except Cancelled:
            job.status, job.error = "cancelled", "Stopped: nothing was saved."
        except LongRun as error:
            job.status, job.error = "failed", str(error)
            job.long_run = error.estimate.sentence
        except AppError as error:
            job.status, job.error = "failed", error.message
        except Exception as error:  # LongRun, WeightsMissing, ValueError... said as they are
            job.status, job.error = "failed", str(error) or type(error).__name__
        finally:
            self._stops.discard(job.id)
            if job.status != "done" and folder is not None:
                shutil.rmtree(folder, ignore_errors=True)

    # --- kept runs --------------------------------------------------------------------------------

    def _load(self, dataset_id: str, run_id: str) -> dict[str, Any]:
        path = self._run_dir(dataset_id, run_id) / "run.json"
        if not path.is_file():
            raise AppError("Run not found.", code="not_found", status_code=404)
        return json.loads(path.read_text(encoding="utf-8"))

    def run(self, dataset_id: str, run_id: str) -> dict[str, Any]:
        """The run as stored (JSON-ready: NaN already null)."""
        self.datasets.get(dataset_id)
        return self._load(dataset_id, run_id)

    def runs(self, dataset_id: str) -> list[VisionRunSummary]:
        self.datasets.get(dataset_id)
        root = self._runs_dir(dataset_id)
        if not root.is_dir():
            return []
        out = []
        for folder in root.iterdir():
            if not (folder / "run.json").is_file() or not RUN_ID.match(folder.name):
                continue
            data = self._load(dataset_id, folder.name)
            out.append(_run_summary(data))
        return sorted(out, key=lambda s: s.created_at, reverse=True)

    def mislabelled(self, dataset_id: str) -> MislabelledOut:
        """R-VQ-16 from the latest kept classification run that carries it (reads files only)."""
        from quantilo.vision.quality import MODEL_RULES

        rule = MODEL_RULES["R-VQ-16"]
        for summary in self.runs(dataset_id):
            if summary.task != "classification":
                continue
            found = self._load(dataset_id, summary.id)["result"].get("mislabelled")
            if not found:
                continue
            details = found.get("details") or {}
            return MislabelledOut(
                run_id=summary.id,
                created_at=summary.created_at,
                model=details.get("model"),
                message=found["message"],
                reason=found["reason"],
                count=found["count"],
                checked=details.get("checked", 0),
                rows=found["rows"],
                images=details.get("images", []),
                notes=details.get("notes", []),
                parameters=found.get("parameters", {}),
            )
        return MislabelledOut(
            message=(
                "Train a classification model on the Model tab: R-VQ-16 reads its out-of-fold "
                "probabilities."
            ),
            reason=rule.reason,
        )

    def predictions(self, dataset_id: str, run_id: str) -> list[dict[str, Any]]:
        self.datasets.get(dataset_id)
        path = self._run_dir(dataset_id, run_id) / "predictions.parquet"
        if not path.is_file():
            raise AppError("Run not found.", code="not_found", status_code=404)
        return clean(pq.read_table(path).to_pylist())

    def image(self, dataset_id: str, run_id: str, kind: str, model: str, row: int) -> Path:
        """A kept anomaly map (`maps`) or reconstruction (`recon`) of one table row."""
        self.datasets.get(dataset_id)
        if kind not in ("maps", "recon") or not MODEL_KEY.match(model) or row < 0:
            raise AppError("Image not found.", code="not_found", status_code=404)
        path = self._run_dir(dataset_id, run_id) / kind / model / f"{row}.png"
        if not path.is_file():
            raise AppError("Image not found.", code="not_found", status_code=404)
        return path

    def delete(self, dataset_id: str, run_id: str) -> None:
        self.datasets.get(dataset_id)
        folder = self._run_dir(dataset_id, run_id)
        if not folder.is_dir():
            raise AppError("Run not found.", code="not_found", status_code=404)
        shutil.rmtree(folder, ignore_errors=True)

    def wait(self, job_id: str, timeout: float = 600.0) -> VisionJob:
        """Block until the job ends (scripts and the agent; the page polls `job`)."""
        deadline = time.monotonic() + timeout
        job = self.job(job_id)
        while job.status == "running" and time.monotonic() < deadline:
            time.sleep(0.1)
        return job


def _installed() -> bool:
    from quantilo.vision import deep

    return deep.installed()


def _install_command() -> str:
    from quantilo.vision import deep

    return deep.INSTALL


def _write_images(folder: Path, result: VisionResult, extra: dict[str, Any]) -> None:
    """113: each kept map as an 8-bit grey PNG on its model's `map_range` (the colours, the
    opacity and the threshold are applied by the page), each reconstruction as an RGB PNG.
    118: what the More models need to be explained, as JSON (`files`) and numpy (`arrays`)."""
    for name, data in (extra.get("files") or {}).items():
        if re.fullmatch(r"[a-z_]+\.json", name):
            (folder / name).write_text(json.dumps(clean(data)), encoding="utf-8")
    arrays = extra.get("arrays") or {}
    if arrays:
        (folder / "arrays").mkdir(exist_ok=True)
        for name, array in arrays.items():
            if re.fullmatch(r"[a-z0-9_]+\.npy", name):
                np.save(folder / "arrays" / name, np.asarray(array), allow_pickle=False)
    ranges = {m.map_source: m.map_range for m in result.models if m.map_source and m.map_range}
    for model, maps in (extra.get("maps") or {}).items():
        if model not in ranges or not MODEL_KEY.match(model):
            continue
        low, high = ranges[model]
        out = folder / "maps" / model
        out.mkdir(parents=True, exist_ok=True)
        for row, values in maps.items():
            grey = np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0.0, 1.0)
            Image.fromarray((grey * 255).round().astype(np.uint8), mode="L").save(
                out / f"{int(row)}.png", "PNG"
            )
    for model, images in (extra.get("reconstructions") or {}).items():
        if not MODEL_KEY.match(model):
            continue
        out = folder / "recon" / model
        out.mkdir(parents=True, exist_ok=True)
        for row, pixels in images.items():
            Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB").save(
                out / f"{int(row)}.png", "PNG"
            )


from app.modules.vision.concepts import (  # noqa: E402  (118; it reads VisionService)
    ConceptOut,
    MoreOptionOut,
    concept_list,
    engine_concepts,
)

TaskOut.model_rebuild()


def _run_summary(data: dict[str, Any]) -> VisionRunSummary:
    result = data["result"]
    models = result.get("models") or []
    default = next((m for m in models if m.get("default")), None)
    metric, score = None, None
    if default is not None and default.get("classification"):
        report = default["classification"]
        metric = report["main"]["metric"]
        score = report.get(metric)
    elif default is not None and default.get("anomaly"):
        metric, score = "auroc", default["anomaly"].get("auroc")
    elif default is not None and default.get("regression"):
        metric, score = "r2", default["regression"].get("r2")
    elif default is not None and default.get("multilabel"):
        metric, score = "macro_f1", default["multilabel"].get("macro_f1")
    return VisionRunSummary(
        id=data["id"],
        created_at=data["created_at"],
        task=result["task"]["task"],
        chosen_by=result["task"].get("chosen_by", "rule"),
        default=default["key"] if default else None,
        default_title=default["title"] if default else None,
        metric=metric,
        score=score,
        models=[m["key"] for m in models],
        deep=bool(result["deep"]["used"]),
        scored=len({r for f in result["split"]["folds"] for r in f["test"]}),
        seconds=float(result["seconds"]),
        sentence=result["sentence"],
    )
