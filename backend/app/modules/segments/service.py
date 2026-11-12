"""Segments page: segmentation and unusual-rows jobs, saved runs, names, t-SNE, marks, new
datasets."""

import csv
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
from typing import Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, TypeAdapter
from pydantic import Field as PydanticField
from quantilo.filters.model import Condition
from quantilo.filters.scope import where_for
from quantilo.ingestion.options import ImportOptions
from quantilo.segments.engine import (
    Cancelled,
    Progress,
    RowValue,
    Segmentation,
    SegmentLabels,
    SegmentRequest,
    SegmentsEngine,
    SegmentsError,
    UnusualRequest,
    UnusualRows,
)
from quantilo.segments.export import with_columns
from quantilo.segments.facade import SEGMENT_COLUMN, free_name
from quantilo.text.distinctive import DistinctiveWords
from quantilo.text.service import TextError, TextService
from sklearn.metrics import adjusted_rand_score

from app.core import db
from app.core.errors import AppError
from app.core.storage import Workspace
from app.services.columns.service import ColumnChange, ColumnService

KEPT_JOBS = 32
FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")
Mark = Literal["expected", "error"]
# A job's body: given `stopped()` and `progress(stage, share, message)`, returns the run id.
Work = Callable[[Callable[[], bool], Progress], str]


@dataclass
class SegmentJob:
    id: str
    kind: Literal["segments", "tsne", "unusual"]
    status: Literal["running", "done", "failed", "cancelled"]
    stage: str
    progress: float | None = None
    message: str | None = None
    run_id: str | None = None
    error: str | None = None


@dataclass
class SegmentRun:
    id: str
    dataset_id: str
    created_at: str
    request: SegmentRequest
    conditions: list[Condition]
    result: Segmentation
    created_dataset_id: str | None = None


@dataclass
class SegmentRunSummary:
    id: str
    created_at: str
    method: str
    method_label: str
    k: int
    forced: bool
    rows: int
    seconds: float
    silhouette: float | None
    stability: float | None
    columns: list[str]
    names: list[str]
    filtered: bool
    created_dataset_id: str | None = None


@dataclass
class SegmentComparison:
    runs: list[SegmentRunSummary]
    # adjusted Rand index between each pair of runs, on the rows they share (1 on the diagonal)
    agreement: list[list[float | None]]


class UnusualStart(UnusualRequest):
    segments_run: str | None = PydanticField(default=None, max_length=64)


@dataclass
class UnusualRun:
    id: str
    dataset_id: str
    created_at: str
    request: UnusualStart
    conditions: list[Condition]
    result: UnusualRows
    marks: dict[str, Mark] = field(default_factory=dict)  # current marks of the listed rows


@dataclass
class UnusualRunSummary:
    id: str
    created_at: str
    method_label: str
    rows: int
    proposed: int
    seconds: float


@dataclass
class UnusualMark:
    key: str
    mark: Mark
    created_at: str


class MarkChange(BaseModel):
    key: str = PydanticField(min_length=1, max_length=2_000)
    mark: Mark | None  # None: no mark


RUN = TypeAdapter(SegmentRun)
UNUSUAL = TypeAdapter(UnusualRun)
CONDITIONS = TypeAdapter(list[Condition])


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _cell(value: object) -> str:
    """A value as a CSV cell; text a spreadsheet would run as a formula gets an apostrophe."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    text = str(value)
    try:
        float(text)
        return text
    except ValueError:
        return f"'{text}" if text.startswith(FORMULA_STARTS) else text


def _safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip() or "dataset"


class SegmentsService(SegmentsEngine):
    """The engine, plus jobs, saved runs, names, t-SNE maps, marks and new datasets."""

    def __init__(self, columns: ColumnService, workspace: Workspace) -> None:
        self._text = TextService(columns)  # 059: words of each segment
        super().__init__(columns)
        self.columns: ColumnService = columns
        self.workspace = workspace
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segments")
        self._jobs: OrderedDict[str, SegmentJob] = OrderedDict()
        self._stops: set[str] = set()
        self._lock = threading.Lock()
        self._pending: dict[str, list[ColumnChange]] = {}  # imported file → column types to set

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def _folder(self, dataset_id: str, run_id: str) -> Path:
        return self.workspace.dataset_dir(dataset_id) / "segments" / run_id

    def _unusual_folder(self, dataset_id: str, run_id: str | None = None) -> Path:
        base = self.workspace.dataset_dir(dataset_id) / "unusual"
        return base / run_id if run_id else base

    # --- Jobs ----------------------------------------------------------------------------------

    def _job(self, kind: Literal["segments", "tsne", "unusual"]) -> SegmentJob:
        job = SegmentJob(uuid.uuid4().hex[:12], kind, "running", "loading")
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > KEPT_JOBS:
                self._jobs.popitem(last=False)
        return job

    def job(self, job_id: str) -> SegmentJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise SegmentsError(
                "This job is no longer followed: open the runs.", "unknown_job", 404
            )
        return job

    def cancel(self, job_id: str) -> SegmentJob:
        """Stop at the next step; nothing is saved."""
        job = self.job(job_id)
        if job.status == "running":
            self._stops.add(job_id)
        return job

    def _follow(self, job: SegmentJob, work: Work) -> None:
        def progress(stage: str, share: float | None, message: str | None) -> None:
            job.stage, job.progress, job.message = stage, share, message

        try:
            job.run_id = work(lambda: job.id in self._stops, progress)
            job.status = "done"
        except Cancelled:
            job.status, job.error = "cancelled", "Stopped: nothing was saved."
        except (AppError, ValueError) as error:
            job.status = "failed"
            job.error = error.message if isinstance(error, AppError) else str(error)
        except Exception as error:
            job.status, job.error = "failed", f"Failed: {error}"
        finally:
            self._stops.discard(job.id)

    # --- Segmentation --------------------------------------------------------------------------

    def start(
        self, dataset_id: str, request: SegmentRequest, conditions: list[Condition]
    ) -> SegmentJob:
        # unknown dataset or filter: fail now
        fields, _ = self._fields(dataset_id, conditions, request)
        self.check(fields, request)
        job = self._job("segments")

        def work(stopped: Callable[[], bool], progress: Progress) -> str:
            fitted = self.segment(dataset_id, conditions, request, stopped, progress)
            if stopped():
                raise Cancelled
            progress("saving", None, None)
            run = SegmentRun(
                uuid.uuid4().hex[:12], dataset_id, _now(), request, conditions, fitted.result
            )
            self._save(run, fitted.labels, fitted.drawn)
            return run.id

        self._executor.submit(self._follow, job, work)
        return job

    def _save(self, run: SegmentRun, labels: pd.DataFrame, drawn: np.ndarray) -> None:
        folder = self._folder(run.dataset_id, run.id)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            labels.to_parquet(folder / "labels.parquet", index=False)
            np.save(folder / "drawn.npy", drawn)
            (folder / "run.json").write_bytes(RUN.dump_json(run))
            result = run.result
            with self._db() as connection, connection:
                connection.execute(
                    "INSERT INTO segment_runs (id, dataset_id, method, k, request, conditions,"
                    " scores, rows, seconds, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run.id,
                        run.dataset_id,
                        result.method,
                        result.k,
                        run.request.model_dump_json(),
                        CONDITIONS.dump_json(run.conditions).decode(),
                        json.dumps(
                            {
                                "silhouette": result.chosen.silhouette,
                                "calinski_harabasz": result.chosen.calinski_harabasz,
                                "davies_bouldin": result.chosen.davies_bouldin,
                                "stability": result.stability.ari if result.stability else None,
                            }
                        ),
                        result.rows,
                        result.seconds,
                        run.created_at,
                    ),
                )
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def runs(self, dataset_id: str) -> list[SegmentRunSummary]:
        self.columns.datasets.get(dataset_id)
        with self._db() as connection:
            rows = connection.execute(
                "SELECT id FROM segment_runs WHERE dataset_id = ? ORDER BY created_at DESC",
                (dataset_id,),
            ).fetchall()
        found = []
        for row in rows:
            try:
                found.append(self._summary(self.run(dataset_id, row["id"])))
            except SegmentsError:  # its files are gone
                continue
        return found

    @staticmethod
    def _summary(run: SegmentRun) -> SegmentRunSummary:
        result = run.result
        return SegmentRunSummary(
            run.id,
            run.created_at,
            result.method,
            result.method_label,
            result.k,
            result.choice.forced,
            result.rows,
            result.seconds,
            result.chosen.silhouette,
            result.stability.ari if result.stability else None,
            [d.column for d in result.preparation.used],
            [s.name for s in result.segments],
            bool(run.conditions),
            run.created_dataset_id,
        )

    def run(self, dataset_id: str, run_id: str) -> SegmentRun:
        path = self._folder(dataset_id, run_id) / "run.json"
        if not re.fullmatch(r"[0-9a-f]{12}", run_id) or not path.is_file():
            raise SegmentsError("Unknown run.", "unknown_run", 404)
        run = RUN.validate_json(path.read_bytes())
        with self._db() as connection:
            row = connection.execute(
                "SELECT created_dataset_id FROM segment_runs WHERE id = ?", (run_id,)
            ).fetchone()
        run.created_dataset_id = row["created_dataset_id"] if row else None
        return run

    def labels(self, dataset_id: str, run_id: str) -> pd.DataFrame:
        self.run(dataset_id, run_id)
        return pd.read_parquet(self._folder(dataset_id, run_id) / "labels.parquet")

    def delete(self, dataset_id: str, run_id: str) -> None:
        self.run(dataset_id, run_id)
        with self._db() as connection, connection:
            connection.execute("DELETE FROM segment_runs WHERE id = ?", (run_id,))
        shutil.rmtree(self._folder(dataset_id, run_id), ignore_errors=True)

    def compare(self, dataset_id: str, run_ids: list[str]) -> SegmentComparison:
        if not 2 <= len(run_ids) <= 5:
            raise SegmentsError("Pick 2 to 5 runs to compare.", "bad_compare")
        runs = [self.run(dataset_id, r) for r in run_ids]
        labels = [self.labels(dataset_id, r).set_index("row")["segment"] for r in run_ids]
        agreement: list[list[float | None]] = []
        for a in labels:
            line: list[float | None] = []
            for b in labels:
                shared = a.index.intersection(b.index)
                line.append(
                    float(adjusted_rand_score(a.loc[shared], b.loc[shared]))
                    if len(shared) > 1
                    else None
                )
            agreement.append(line)
        return SegmentComparison([self._summary(r) for r in runs], agreement)

    def rename(self, dataset_id: str, run_id: str, names: dict[int, str]) -> SegmentRun:
        run = self.run(dataset_id, run_id)
        known = {s.id for s in run.result.segments}
        unknown = [str(k) for k in names if k not in known]
        if unknown:
            raise SegmentsError(f"Unknown segment(s): {', '.join(unknown)}.", "unknown_segment")
        for segment in run.result.segments:
            given = names.get(segment.id, "").strip()
            if given:
                segment.name = given[:80]
        (self._folder(dataset_id, run_id) / "run.json").write_bytes(RUN.dump_json(run))
        return run

    def start_tsne(self, dataset_id: str, run_id: str) -> SegmentJob:
        run = self.run(dataset_id, run_id)
        if run.result.tsne is not None:
            job = self._job("tsne")
            job.status, job.run_id = "done", run_id
            return job
        job = self._job("tsne")

        def work(stopped: Callable[[], bool], progress: Progress) -> str:
            progress("projecting", None, None)
            drawn = np.load(self._folder(dataset_id, run_id) / "drawn.npy")
            found = self.tsne(run.result, drawn, run.request.seed)
            if stopped():
                raise Cancelled
            current = self.run(dataset_id, run_id)  # names may have changed meanwhile
            current.result.tsne = found
            (self._folder(dataset_id, run_id) / "run.json").write_bytes(RUN.dump_json(current))
            return run_id

        self._executor.submit(self._follow, job, work)
        return job

    def words(self, dataset_id: str, run_id: str, column: str) -> DistinctiveWords:
        """059, R-WRD-07: the words over-represented in each segment's texts."""
        run = self.run(dataset_id, run_id)
        names = {s.id: s.name for s in run.result.segments}
        labels = self.labels(dataset_id, run_id)
        groups = [names.get(int(v), str(v)) for v in labels["segment"]]
        order = [s.name for s in run.result.segments]
        try:
            return self._text.words_by_rows(
                dataset_id, column, labels["row"].astype(int).tolist(), groups, "segment", order
            )
        except TextError as error:
            raise SegmentsError(error.message, "not_text") from error

    def profiles_csv(self, dataset_id: str, run_id: str) -> bytes:
        """The profile table: one line per column, the whole then each segment."""
        result = self.run(dataset_id, run_id).result
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["column", "overall", *(_cell(s.name) for s in result.segments)])
        for row in result.profiles:
            cells = [row.overall, *row.cells]
            writer.writerow([_cell(row.column), *(_cell(c.value) for c in cells)])
        return buffer.getvalue().encode("utf-8")

    def create_dataset(self, dataset_id: str, run_id: str) -> str:
        """The run's rows (filtered as they were) with a `segment` column, as a new dataset."""
        run = self.run(dataset_id, run_id)
        if run.created_dataset_id:
            try:
                self.columns.datasets.get(run.created_dataset_id)
                return run.created_dataset_id
            except AppError:
                pass
        names = {s.id: s.name for s in run.result.segments}
        labels = self.labels(dataset_id, run_id)
        views = self.columns.columns(dataset_id)
        column = free_name([v.name for v in views], SEGMENT_COLUMN)
        extra = pd.DataFrame(
            {
                "row": labels["row"].to_numpy(dtype=np.int64),
                column: [names.get(int(v), str(v)) for v in labels["segment"]],
            }
        )
        source = self.columns.datasets.get(dataset_id)
        created = self._import(
            dataset_id,
            run.conditions,
            extra,
            f"{source.name} (segments)",
            [ColumnChange(name=column, type="categorical_nominal")],
        )
        with self._db() as connection, connection:
            connection.execute(
                "UPDATE segment_runs SET created_dataset_id = ? WHERE id = ?", (created, run_id)
            )
        return created

    def _import(
        self,
        dataset_id: str,
        conditions: list[Condition],
        extra: pd.DataFrame,
        name: str,
        added: list[ColumnChange],
    ) -> str:
        datasets = self.columns.datasets
        source = self.columns.ready(dataset_id)
        views = self.columns.columns(dataset_id)
        target = datasets.new_upload_path(f"{_safe_name(name)}.parquet")
        with_columns(source.storage_path, where_for(views, conditions), extra, target)
        self._pending[str(datasets.resolve(str(target)))] = [
            *(
                ColumnChange(
                    name=v.name,
                    type=v.type,
                    role=v.role or "",
                    order=v.order or [],
                    excluded=v.excluded,
                )
                for v in views
            ),
            *added,
        ]
        record, _ = datasets.start_import(str(target), name, ImportOptions())
        return record.id

    def after_import(self, dataset_id: str, storage_path: str) -> None:
        """Import step: a dataset made here gets the source's types, roles and orders."""
        record = self.columns.datasets.get(dataset_id)
        changes = self._pending.pop(record.source_path, None)
        if changes:
            self.columns.apply(dataset_id, storage_path, changes)

    # --- Unusual rows --------------------------------------------------------------------------

    def start_unusual(
        self, dataset_id: str, request: UnusualStart, conditions: list[Condition]
    ) -> SegmentJob:
        fields, _ = self._fields(dataset_id, conditions, request)
        self.check(fields, request)
        segments = None
        if request.segments_run:
            run = self.run(dataset_id, request.segments_run)
            segments = SegmentLabels(
                self.labels(dataset_id, run.id),
                {s.id: s.name for s in run.result.segments},
                run.request,
            )
        elif request.method == "segment":
            raise SegmentsError(
                "Distance to its segment needs a segmentation: run Segments first.",
                "no_segmentation",
            )
        hidden = {m.key for m in self.marks(dataset_id) if m.mark == "expected"}
        job = self._job("unusual")

        def work(stopped: Callable[[], bool], progress: Progress) -> str:
            scored = self.unusual(
                dataset_id, conditions, request, segments, hidden, stopped, progress
            )
            if stopped():
                raise Cancelled
            progress("saving", None, None)
            run = UnusualRun(
                uuid.uuid4().hex[:12], dataset_id, _now(), request, conditions, scored.result
            )
            folder = self._unusual_folder(dataset_id, run.id)
            folder.mkdir(parents=True, exist_ok=True)
            scored.scores.to_parquet(folder / "scores.parquet", index=False)
            (folder / "run.json").write_bytes(UNUSUAL.dump_json(run))
            return run.id

        self._executor.submit(self._follow, job, work)
        return job

    def unusual_runs(self, dataset_id: str) -> list[UnusualRunSummary]:
        self.columns.datasets.get(dataset_id)
        base = self._unusual_folder(dataset_id)
        found = []
        for path in base.glob("*/run.json") if base.is_dir() else []:
            run = UNUSUAL.validate_json(path.read_bytes())
            result = run.result
            found.append(
                UnusualRunSummary(
                    run.id,
                    run.created_at,
                    result.method_label,
                    result.rows,
                    result.threshold.rows,
                    result.seconds,
                )
            )
        return sorted(found, key=lambda r: r.created_at, reverse=True)

    def unusual_run(self, dataset_id: str, run_id: str) -> UnusualRun:
        path = self._unusual_folder(dataset_id, run_id) / "run.json"
        if not re.fullmatch(r"[0-9a-f]{12}", run_id) or not path.is_file():
            raise SegmentsError("Unknown run.", "unknown_run", 404)
        run = UNUSUAL.validate_json(path.read_bytes())
        marks: dict[str, Mark] = {m.key: m.mark for m in self.marks(dataset_id)}
        run.marks = {r.key: marks[r.key] for r in run.result.listed if r.key in marks}
        return run

    def row(self, dataset_id: str, run_id: str, row: int) -> list[RowValue]:
        run = self.unusual_run(dataset_id, run_id)
        return self.row_card(dataset_id, run.conditions, row)

    def _kept(self, run: UnusualRun, threshold: float | None) -> float:
        return run.result.threshold.value if threshold is None else threshold

    def unusual_csv(self, dataset_id: str, run_id: str, threshold: float | None) -> bytes:
        """The listed rows at or above the threshold: score, mark, reasons, every column."""
        run = self.unusual_run(dataset_id, run_id)
        cut = self._kept(run, threshold)
        columns = run.result.columns
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["rank", "row", "score", "mark", "reasons", *(_cell(c) for c in columns)])
        for item in run.result.listed:
            if item.score < cut:
                continue
            reasons = [item.combination] if item.combination else []
            reasons += [r.text for r in item.reasons]
            writer.writerow(
                [
                    item.rank,
                    item.row,
                    _cell(item.score),
                    run.marks.get(item.key, ""),
                    _cell("; ".join(reasons)),
                    *(_cell(item.values.get(c)) for c in columns),
                ]
            )
        return buffer.getvalue().encode("utf-8")

    def create_unusual_dataset(self, dataset_id: str, run_id: str, threshold: float | None) -> str:
        """The run's rows with `unusual_score` and `is_unusual` (at or above the threshold)."""
        run = self.unusual_run(dataset_id, run_id)
        cut = self._kept(run, threshold)
        scores = pd.read_parquet(self._unusual_folder(dataset_id, run_id) / "scores.parquet")
        names = [v.name for v in self.columns.columns(dataset_id)]
        score = free_name(names, "unusual_score")
        flag = free_name([*names, score], "is_unusual")
        extra = pd.DataFrame(
            {
                "row": scores["row"].to_numpy(dtype=np.int64),
                score: scores["score"].to_numpy(dtype=float),
                flag: scores["score"].to_numpy(dtype=float) >= cut,
            }
        )
        source = self.columns.datasets.get(dataset_id)
        return self._import(
            dataset_id,
            run.conditions,
            extra,
            f"{source.name} (unusual rows)",
            [
                ColumnChange(name=score, type="numeric_continuous"),
                ColumnChange(name=flag, type="boolean"),
            ],
        )

    # --- Marks ---------------------------------------------------------------------------------

    def marks(self, dataset_id: str) -> list[UnusualMark]:
        with self._db() as connection:
            rows = connection.execute(
                "SELECT row_key, mark, created_at FROM unusual_marks WHERE dataset_id = ?"
                " ORDER BY created_at",
                (dataset_id,),
            ).fetchall()
        return [UnusualMark(r["row_key"], cast(Mark, r["mark"]), r["created_at"]) for r in rows]

    def set_mark(self, dataset_id: str, change: MarkChange) -> list[UnusualMark]:
        self.columns.datasets.get(dataset_id)
        with self._db() as connection, connection:
            if change.mark is None:
                connection.execute(
                    "DELETE FROM unusual_marks WHERE dataset_id = ? AND row_key = ?",
                    (dataset_id, change.key),
                )
            else:
                connection.execute(
                    "INSERT INTO unusual_marks (dataset_id, row_key, mark, created_at)"
                    " VALUES (?, ?, ?, ?) ON CONFLICT (dataset_id, row_key)"
                    " DO UPDATE SET mark = excluded.mark, created_at = excluded.created_at",
                    (dataset_id, change.key, change.mark, _now()),
                )
        return self.marks(dataset_id)
