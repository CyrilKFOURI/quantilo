"""Synthetic data page: plan, generation jobs, saved runs, on-demand evaluations, new dataset."""

import io
import json
import re
import shutil
import sqlite3
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
from pydantic import TypeAdapter
from quantilo.ingestion.options import ImportOptions
from quantilo.synthetic import compare as side_by_side
from quantilo.synthetic import plan
from quantilo.synthetic.engine import (
    Prepared,
    RunColumn,
    RunRequest,
    SyntheticEngine,
    SyntheticError,
    export_frame,
)
from quantilo.synthetic.methods import Cancelled
from quantilo.synthetic.plan import Method
from quantilo.synthetic.privacy import Disclosure, PrivacyError, disclosure
from quantilo.synthetic.report import Report
from quantilo.synthetic.utility import Utility, UtilityError, utility

from app.core import db
from app.core.errors import AppError
from app.core.storage import Workspace
from app.services.columns.service import ColumnChange, ColumnService

Stage = Literal["loading", "generating", "evaluating", "saving"]
KEPT_JOBS = 32
PREVIEW_ROWS = 100
FRAMES_CACHE = 2


@dataclass
class SyntheticJob:
    id: str
    status: Literal["running", "done", "failed", "cancelled"]
    stage: Stage
    run_id: str | None = None
    error: str | None = None
    progress: float | None = None  # deep methods: share of the epochs done
    message: str | None = None  # "Epoch 12 / 300 · loss …"


@dataclass
class SyntheticRun:
    id: str
    dataset_id: str
    created_at: str
    method: Method
    method_label: str
    settings: RunRequest
    rows: int
    seconds: float
    train_rows: int
    holdout_rows: int
    columns: list[RunColumn]
    report: Report
    created_dataset_id: str | None = None


@dataclass
class SyntheticRunSummary:
    id: str
    method: Method
    method_label: str
    rows: int
    seconds: float
    created_at: str
    scores: dict[str, float | None]
    created_dataset_id: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    device: str | None = None  # deep methods: the card or "CPU"


@dataclass
class SyntheticPreview:
    columns: list[str]
    rows: list[list[str | None]]
    total: int


@dataclass
class Origin:
    """A dataset created from a synthetic run."""

    source_id: str
    source_name: str
    run_id: str


RUN = TypeAdapter(SyntheticRun)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip() or "synthetic"


class SyntheticService(SyntheticEngine):
    """The engine, plus generation jobs, saved runs, on-demand evaluations and new datasets."""

    def __init__(
        self,
        columns: ColumnService,
        workspace: Workspace,
        compute: Callable[[], tuple[str | None, bool]] = lambda: (None, False),
    ) -> None:
        """`compute()` gives the ready card PyTorch sees (or None) and whether the app setting
        allows the GPU."""
        super().__init__(columns, compute)
        self.columns: ColumnService = columns
        self.workspace = workspace
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="synthetic")
        self._jobs: OrderedDict[str, SyntheticJob] = OrderedDict()
        self._stops: set[str] = set()
        self._lock = threading.Lock()
        self._frames: OrderedDict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = (
            OrderedDict()
        )
        self._pending: dict[str, list[ColumnChange]] = {}  # imported file → column types to copy

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def _folder(self, dataset_id: str, run_id: str) -> Path:
        return self.workspace.dataset_dir(dataset_id) / "synthetic" / run_id

    # --- Generation ----------------------------------------------------------------------------

    def start(self, dataset_id: str, request: RunRequest) -> SyntheticJob:
        prepared = self._prepare(dataset_id, request)
        job = SyntheticJob(uuid.uuid4().hex[:12], "running", "loading")
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > KEPT_JOBS:
                self._jobs.popitem(last=False)
        self._executor.submit(self._run, job, dataset_id, request, prepared)
        return job

    def job(self, job_id: str) -> SyntheticJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise SyntheticError(
                "This generation is no longer followed: open the runs.", "unknown_job", 404
            )
        return job

    def cancel(self, job_id: str) -> SyntheticJob:
        job = self.job(job_id)
        if job.status == "running":
            self._stops.add(job_id)
        return job

    def _run(
        self, job: SyntheticJob, dataset_id: str, request: RunRequest, prepared: Prepared
    ) -> None:
        def stopped() -> bool:
            return job.id in self._stops

        def progress(share: float, message: str) -> None:
            job.progress, job.message = share, message

        def stage(name: str) -> None:
            job.stage = cast(Stage, name)
            if name == "evaluating":
                job.progress, job.message = None, None

        try:
            made = self._generate(dataset_id, request, prepared, stopped, progress, stage)
            job.stage = "saving"
            run = SyntheticRun(
                uuid.uuid4().hex[:12],
                dataset_id,
                _now(),
                request.method,
                plan.METHODS[request.method][0],
                request,
                made.rows,
                made.seconds,
                len(made.train),
                len(made.holdout),
                made.columns,
                made.report,
            )
            self._save(run, made.train, made.holdout, made.data)
            job.run_id, job.status = run.id, "done"
        except Cancelled:
            job.status, job.error = "cancelled", "Stopped: nothing was saved."
        except (SyntheticError, ValueError) as error:
            job.status, job.error = "failed", str(error)
        except Exception as error:
            job.status, job.error = "failed", f"Generation failed: {error}"
        finally:
            self._stops.discard(job.id)

    def _save(
        self, run: SyntheticRun, train: pd.DataFrame, holdout: pd.DataFrame, data: pd.DataFrame
    ) -> None:
        folder = self._folder(run.dataset_id, run.id)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            self._export(data, run).to_parquet(folder / "data.parquet", index=False)
            train.to_parquet(folder / "train.parquet", index=False)
            holdout.to_parquet(folder / "holdout.parquet", index=False)
            (folder / "report.json").write_bytes(RUN.dump_json(run))
            with self._db() as connection, connection:
                connection.execute(
                    "INSERT INTO synthetic_runs (id, dataset_id, method, settings, rows, scores,"
                    " status, seconds, created_at, device)"
                    " VALUES (?, ?, ?, ?, ?, ?, 'done', ?, ?, ?)",
                    (
                        run.id,
                        run.dataset_id,
                        run.method,
                        run.settings.model_dump_json(),
                        run.rows,
                        json.dumps(self._scores(run)),
                        run.seconds,
                        run.created_at,
                        run.report.fitted.device,
                    ),
                )
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    @staticmethod
    def _scores(run: SyntheticRun) -> dict[str, float | None]:
        scores = run.report.scores
        return {
            "diagnostic": scores.diagnostic,
            "quality": scores.quality,
            "shapes": scores.shapes,
            "trends": scores.trends,
            "detection": scores.detection,
            "privacy": scores.privacy,
            "utility": scores.utility,
        }

    @staticmethod
    def _export(data: pd.DataFrame, run: SyntheticRun) -> pd.DataFrame:
        return export_frame(data, run.columns)

    # --- Runs ----------------------------------------------------------------------------------

    def runs(self, dataset_id: str) -> list[SyntheticRunSummary]:
        self.columns.datasets.get(dataset_id)
        with self._db() as connection:
            rows = connection.execute(
                "SELECT * FROM synthetic_runs WHERE dataset_id = ? ORDER BY created_at DESC",
                (dataset_id,),
            ).fetchall()
        return [self._summary(r) for r in rows]

    @staticmethod
    def _summary(row: sqlite3.Row) -> SyntheticRunSummary:
        method = row["method"]
        return SyntheticRunSummary(
            row["id"],
            method,
            plan.METHODS[method][0] if method in plan.METHODS else method,
            row["rows"],
            row["seconds"],
            row["created_at"],
            json.loads(row["scores"]) if row["scores"] else {},
            row["created_dataset_id"],
            json.loads(row["settings"]),
            row["device"],
        )

    def run(self, dataset_id: str, run_id: str) -> SyntheticRun:
        path = self._folder(dataset_id, run_id) / "report.json"
        if not path.is_file():
            raise SyntheticError("Unknown run.", "unknown_run", 404)
        run = RUN.validate_json(path.read_bytes())
        with self._db() as connection:
            row = connection.execute(
                "SELECT created_dataset_id FROM synthetic_runs WHERE id = ?", (run_id,)
            ).fetchone()
        run.created_dataset_id = row["created_dataset_id"] if row else None
        return run

    def compare(self, dataset_id: str, run_ids: list[str]) -> list[SyntheticRunSummary]:
        if not 2 <= len(run_ids) <= 5:
            raise SyntheticError("Pick 2 to 5 runs to compare.", "bad_compare")
        found = {r.id: r for r in self.runs(dataset_id)}
        missing = [i for i in run_ids if i not in found]
        if missing:
            raise SyntheticError(f"Unknown run(s): {', '.join(missing)}.", "unknown_run", 404)
        return [found[i] for i in run_ids]

    def delete(self, dataset_id: str, run_id: str) -> None:
        self.run(dataset_id, run_id)
        with self._db() as connection, connection:
            connection.execute("DELETE FROM synthetic_runs WHERE id = ?", (run_id,))
        self._frames.pop(run_id, None)
        shutil.rmtree(self._folder(dataset_id, run_id), ignore_errors=True)

    def _frames_of(
        self, dataset_id: str, run_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """(training rows, held-out rows, synthetic rows), typed as the evaluation saw them."""
        if run_id in self._frames:
            self._frames.move_to_end(run_id)
            return self._frames[run_id]
        run = self.run(dataset_id, run_id)
        folder = self._folder(dataset_id, run_id)
        train = pd.read_parquet(folder / "train.parquet")
        holdout = pd.read_parquet(folder / "holdout.parquet")
        data = pd.read_parquet(folder / "data.parquet")
        for column in run.columns:
            if column.type == "boolean" and column.treatment == "model":
                data[column.name] = data[column.name].map(
                    lambda v: None if pd.isna(v) else ("True" if v else "False")
                )
            elif column.type == "numeric_discrete" and column.treatment == "model":
                data[column.name] = data[column.name].astype(float)
        frames = (train, holdout, data)
        self._frames[run_id] = frames
        while len(self._frames) > FRAMES_CACHE:
            self._frames.popitem(last=False)
        return frames

    def preview(self, dataset_id: str, run_id: str) -> SyntheticPreview:
        _, _, data = self._frames_of(dataset_id, run_id)
        head = data.head(PREVIEW_ROWS)
        rows = [
            [None if pd.isna(v) else str(v).replace(" 00:00:00", "") for v in record]
            for record in head.itertuples(index=False)
        ]
        return SyntheticPreview(list(data.columns), rows, len(data))

    def download(self, dataset_id: str, run_id: str, kind: Literal["csv", "parquet"]) -> bytes:
        self.run(dataset_id, run_id)
        path = self._folder(dataset_id, run_id) / "data.parquet"
        if kind == "parquet":
            return path.read_bytes()
        buffer = io.StringIO()
        pd.read_parquet(path).to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8")

    # --- On-demand evaluations -----------------------------------------------------------------

    def _rewrite(self, run: SyntheticRun) -> None:
        (self._folder(run.dataset_id, run.id) / "report.json").write_bytes(RUN.dump_json(run))
        with self._db() as connection, connection:
            connection.execute(
                "UPDATE synthetic_runs SET scores = ? WHERE id = ?",
                (json.dumps(self._scores(run)), run.id),
            )

    def utility(self, dataset_id: str, run_id: str, target: str) -> Utility:
        run = self.run(dataset_id, run_id)
        train, holdout, data = self._frames_of(dataset_id, run_id)
        try:
            found = utility(train, data, holdout, run.report.learned, target, run.settings.seed)
        except UtilityError as error:
            raise SyntheticError(str(error), "bad_target") from None
        run.report.utility = found
        run.report.scores.utility = found.ratio
        self._rewrite(run)
        return found

    def disclosure(
        self, dataset_id: str, run_id: str, known: list[str], sensitive: str
    ) -> Disclosure:
        run = self.run(dataset_id, run_id)
        train, _, data = self._frames_of(dataset_id, run_id)
        try:
            found = disclosure(train, data, run.report.learned, known, sensitive, run.settings.seed)
        except PrivacyError as error:
            raise SyntheticError(str(error), "bad_disclosure") from None
        run.report.disclosure = found
        self._rewrite(run)
        return found

    def _learned(self, run: SyntheticRun, name: str):
        item = next((i for i in run.report.learned if i.name == name), None)
        if item is None:
            raise SyntheticError(
                f"{name!r} is not a modelled column of this run.", "unknown_column", 404
            )
        return item

    def column(
        self, dataset_id: str, run_id: str, column: str, bins: int | None, target: str | None
    ) -> side_by_side.ColumnDetail:
        run = self.run(dataset_id, run_id)
        item = self._learned(run, column)
        target_item = self._learned(run, target) if target else None
        train, _, data = self._frames_of(dataset_id, run_id)
        score = next((s.score for s in run.report.quality.shape_scores if s.column == column), None)
        real = train.sample(min(len(train), 20_000), random_state=run.settings.seed)
        fake = data.sample(min(len(data), 20_000), random_state=run.settings.seed)
        return side_by_side.column_detail(real, fake, item, score, bins, target_item)

    def pair(self, dataset_id: str, run_id: str, a: str, b: str) -> side_by_side.PairDetail:
        run = self.run(dataset_id, run_id)
        first, second = self._learned(run, a), self._learned(run, b)
        if a == b:
            raise SyntheticError("Pick two different columns.", "same_column")
        train, _, data = self._frames_of(dataset_id, run_id)
        return side_by_side.pair_detail(train, data, first, second, run.settings.seed)

    # --- New dataset ---------------------------------------------------------------------------

    def create_dataset(self, dataset_id: str, run_id: str) -> str:
        """Import the synthetic rows as a new dataset, with the source's column types."""
        run = self.run(dataset_id, run_id)
        datasets = self.columns.datasets
        if run.created_dataset_id:
            try:
                datasets.get(run.created_dataset_id)
                return run.created_dataset_id
            except AppError:
                pass
        source = datasets.get(dataset_id)
        name = f"{source.name} (synthetic)"
        target = datasets.new_upload_path(f"{_safe_name(name)}.parquet")
        shutil.copyfile(self._folder(dataset_id, run_id) / "data.parquet", target)
        views = {v.name: v for v in self.columns.columns(dataset_id)}
        self._pending[str(datasets.resolve(str(target)))] = [
            ColumnChange(
                name=c.name,
                type=views[c.name].type,
                role=views[c.name].role or "",
                order=views[c.name].order or [],
            )
            for c in run.columns
            if c.name in views
        ]
        record, _ = datasets.start_import(str(target), name, ImportOptions())
        with self._db() as connection, connection:
            connection.execute(
                "UPDATE synthetic_runs SET created_dataset_id = ? WHERE id = ?", (record.id, run_id)
            )
        return record.id

    def after_import(self, dataset_id: str, storage_path: str) -> None:
        """Import step: a dataset made from a run gets the source's types, roles and orders."""
        record = self.columns.datasets.get(dataset_id)
        changes = self._pending.pop(record.source_path, None)
        if changes:
            self.columns.apply(dataset_id, storage_path, changes)

    def origin(self, dataset_id: str) -> Origin | None:
        with self._db() as connection:
            row = connection.execute(
                "SELECT r.id, r.dataset_id, d.name FROM synthetic_runs r"
                " JOIN datasets d ON d.id = r.dataset_id WHERE r.created_dataset_id = ?",
                (dataset_id,),
            ).fetchone()
        return Origin(row["dataset_id"], row["name"], row["id"]) if row else None
