"""Use a trained run on new data (037): the check, prediction jobs in blocks of 100 000 rows
(cancellable), saved predictions (`ml_predictions`), their rows, the CSV export, "Save as
dataset", the explanation of one predicted row, and the model exported as a zip.

Every calculation is the engine's (`quantilo.ml.predict`): the library's
`model.check` / `model.predict` give the same results. The app never imports a model file: the
only pipelines it loads are the ones it wrote in its own workspace."""

import csv
import io
import re
import shutil
import sqlite3
import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import duckdb
import joblib
from pydantic import TypeAdapter
from quantilo.filters.scope import scoped_source, where_for
from quantilo.ingestion.options import ImportOptions
from quantilo.ingestion.sql import ident, lit
from quantilo.ml import predict as use
from quantilo.ml.engine import MlError

from app.core import db
from app.core.errors import AppError
from app.modules.ml.service import MlService, _cell
from app.services.columns.service import ColumnChange

KEPT_JOBS = 32
ID = re.compile(r"[0-9a-f]{12}")  # every id the app makes: uuid4().hex[:12]
MAX_PAGE = 500
CSV_BATCH = 10_000


class Cancelled(Exception):
    """The user stopped the prediction."""


@dataclass
class PredictionJob:
    id: str
    status: Literal["running", "done", "failed", "cancelled"]
    rows: int = 0  # predicted so far
    total: int = 0
    prediction_id: str | None = None
    error: str | None = None


@dataclass
class PredictionSummary:
    """A prediction made with a run, as the run lists it."""

    id: str
    run_id: str
    source_id: str | None  # None: the predicted dataset was deleted since
    source_name: str
    rows: int
    less_reliable: int
    level: str
    created_at: str
    created_dataset_id: str | None = None


@dataclass
class Prediction(PredictionSummary):
    keep: list[str] = field(default_factory=list)
    check: use.ModelCheck | None = None
    summary: use.PredictionReport | None = None


@dataclass
class PredictionRows:
    columns: list[str]  # "row" (1-based row of the predicted data) first
    rows: list[list[Any]]
    total: int  # rows matching (all, or the less reliable ones)
    offset: int


@dataclass
class Saved:
    """What the prediction keeps beside its rows (`<id>.json`)."""

    keep: list[str]
    check: use.ModelCheck
    summary: use.PredictionReport


CARD = TypeAdapter(use.ModelCard)
SAVED = TypeAdapter(Saved)


def _checked(value: str, what: str) -> str:
    """An id from a request or the agent, before it names a file: only the app's own format
    (12 hex digits), else "unknown" (404). No path can be built from anything else."""
    if not ID.fullmatch(value):
        raise MlError(f"Unknown {what}.", f"unknown_{what}", 404)
    return value


def _inside(path: Path, folder: Path) -> Path:
    """`path`, once resolved, must sit in `folder` (defence in depth after `_checked`)."""
    resolved = path.resolve()
    if not resolved.is_relative_to(folder.resolve()):
        raise MlError("Unknown file.", "unknown_file", 404)
    return resolved


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value)


def _json(value: Any) -> Any:
    """A cell of a page of rows: numbers and booleans as they are, the rest as text."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


class PredictionService:
    """Predictions of the ML module's runs on datasets of the workspace."""

    def __init__(self, ml: MlService, versions: Any = None) -> None:
        self.ml = ml
        self.versions = versions  # 031's VersionService: the drift against the training data
        self.columns = ml.columns
        self.workspace = ml.workspace
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-predict")
        self._jobs: OrderedDict[str, PredictionJob] = OrderedDict()
        self._stops: set[str] = set()
        self._lock = threading.Lock()
        self._pending: dict[str, list[ColumnChange]] = {}  # imported file → column types to copy

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def _folder(self, dataset_id: str) -> Path:
        return self.workspace.dataset_dir(_checked(dataset_id, "dataset")) / "ml" / "predictions"

    def _run_file(self, dataset_id: str, run_id: str, suffix: str) -> Path:
        """A file of the run (`.joblib`, `.card.json`), inside the run's folder only."""
        folder = self.ml._folder(_checked(dataset_id, "dataset"))
        return _inside(folder / f"{_checked(run_id, 'run')}{suffix}", folder)

    def _prediction_file(self, dataset_id: str, prediction_id: str, suffix: str = "") -> Path:
        """A prediction's rows folder or its `.json`, inside the predictions folder only."""
        folder = self._folder(dataset_id)
        return _inside(folder / f"{_checked(prediction_id, 'prediction')}{suffix}", folder)

    # --- The model -------------------------------------------------------------------------

    def _pipeline(self, dataset_id: str, run_id: str) -> Any:
        path = self._run_file(dataset_id, run_id, ".joblib")
        self.ml.run(dataset_id, run_id)  # an unknown run: 404
        # a file this app wrote itself in its own local workspace, never one received from outside
        return joblib.load(path)

    def card(self, dataset_id: str, run_id: str) -> use.ModelCard:
        """What the run's model expects, computed once from its training rows, then kept."""
        cached = self._run_file(dataset_id, run_id, ".card.json")
        if cached.is_file():
            return CARD.validate_json(cached.read_bytes())
        run = self.ml.run(dataset_id, run_id)
        try:
            found = use.card_from(
                self.ml,
                dataset_id,
                self._pipeline(dataset_id, run_id),
                run,
                self.ml._conditions(run_id),
                run.created_at,
            )
        except ValueError as error:
            raise MlError(str(error), "data_changed") from None
        cached.write_bytes(CARD.dump_json(found))
        return found

    def export(self, dataset_id: str, run_id: str) -> tuple[str, bytes]:
        """The run's model as a zip (pipeline, card, predict.py): no data rows."""
        card = self.card(dataset_id, run_id)
        name = use.zip_name(card)
        pipeline = self._run_file(dataset_id, run_id, ".joblib").read_bytes()
        return name, use.export_zip(card, pipeline, name)

    # --- Check and prediction jobs ---------------------------------------------------------

    def _source(self, source_id: str) -> tuple[str, list[Any], str]:
        """The predicted dataset: every row (the ML page's filters belong to the training data),
        its typed columns, its name."""
        record = self.columns.ready(_checked(source_id, "dataset"))
        views = self.columns.columns(source_id)
        return (
            scoped_source(record.storage_path, where_for(views, []), numbered=True),
            views,
            record.name,
        )

    def check(self, dataset_id: str, run_id: str, source_id: str) -> use.ModelCheck:
        card = self.card(dataset_id, run_id)
        source, views, _ = self._source(source_id)
        connection = duckdb.connect()
        try:
            checked = use.check(connection, source, card, views)
        finally:
            connection.close()
        return self._with_drift(checked, dataset_id, source_id)

    def _with_drift(
        self, checked: use.ModelCheck, dataset_id: str, source_id: str
    ) -> use.ModelCheck:
        """031's comparison of the new data with the training dataset, the model's most important
        columns first; skipped when the two are the same dataset or cannot be compared."""
        if self.versions is None or source_id == dataset_id:
            return checked
        focus = [c.column for c in checked.columns if c.important]
        try:
            changes = self.versions.changes(source_id, vs=dataset_id, focus=focus)
        except Exception:  # a drift summary never blocks a prediction
            return checked
        return use.with_drift(checked, changes)

    def start(
        self, dataset_id: str, run_id: str, source_id: str, keep: list[str] | None = None
    ) -> PredictionJob:
        """Check, then predict every row in the background; refused when the check blocks."""
        card = self.card(dataset_id, run_id)
        source, views, name = self._source(source_id)
        names = {v.name for v in views}
        unknown = [k for k in keep or [] if k not in names]
        if unknown:
            raise MlError(f"Unknown column(s) to keep: {', '.join(unknown)}.", "unknown_column")
        connection = duckdb.connect()
        try:
            checked = use.check(connection, source, card, views)
        finally:
            connection.close()
        checked = self._with_drift(checked, dataset_id, source_id)
        if checked.level == "blocked":
            raise MlError(checked.message, "blocked")
        job = PredictionJob(uuid.uuid4().hex[:12], "running", 0, checked.rows)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > KEPT_JOBS:
                self._jobs.popitem(last=False)
        kept = [k for k in keep or [] if k not in use.OUTPUTS]
        self._executor.submit(
            self._predict, job, dataset_id, run_id, source_id, name, source, views, kept, checked
        )
        return job

    def _predict(
        self,
        job: PredictionJob,
        dataset_id: str,
        run_id: str,
        source_id: str,
        name: str,
        source: str,
        views: list[Any],
        keep: list[str],
        checked: use.ModelCheck,
    ) -> None:
        prediction_id = uuid.uuid4().hex[:12]
        parts = self._prediction_file(dataset_id, prediction_id)
        try:
            card = self.card(dataset_id, run_id)
            pipeline = self._pipeline(dataset_id, run_id)
            parts.mkdir(parents=True, exist_ok=True)
            totals = use.Accumulator(card)
            connection = duckdb.connect()
            try:
                blocks = use.predict_blocks(
                    connection,
                    source,
                    pipeline,
                    card,
                    views,
                    keep,
                    should_stop=lambda: job.id in self._stops,
                )
                for number, block in enumerate(blocks, start=1):
                    totals.add(block.drop(columns="__row"))
                    writer = duckdb.connect()
                    try:
                        writer.register("block", block)
                        path = (parts / f"part-{number:05d}.parquet").as_posix()
                        writer.execute(f"COPY block TO {lit(path)} (FORMAT parquet)")
                    finally:
                        writer.close()
                    job.rows = totals.rows
            finally:
                connection.close()
            if job.id in self._stops:
                raise Cancelled
            summary = totals.summary()
            saved = Saved(keep, checked, summary)
            self._prediction_file(dataset_id, prediction_id, ".json").write_bytes(
                SAVED.dump_json(saved)
            )
            with self._db() as db_connection, db_connection:
                db_connection.execute(
                    "INSERT INTO ml_predictions (id, dataset_id, run_id, source_id, source_name,"
                    " rows, less_reliable, level, path, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        prediction_id,
                        dataset_id,
                        run_id,
                        source_id,
                        name,
                        summary.rows,
                        summary.less_reliable,
                        checked.level,
                        f"ml/predictions/{prediction_id}",
                        _now(),
                    ),
                )
            job.prediction_id, job.status = prediction_id, "done"
        except Cancelled:
            shutil.rmtree(parts, ignore_errors=True)
            job.status, job.error = "cancelled", "Stopped: nothing was saved."
        except (MlError, ValueError) as error:
            shutil.rmtree(parts, ignore_errors=True)
            job.status, job.error = "failed", str(error)
        except Exception as error:
            shutil.rmtree(parts, ignore_errors=True)
            job.status, job.error = "failed", f"Prediction failed: {error}"
        finally:
            self._stops.discard(job.id)

    def job(self, job_id: str) -> PredictionJob:
        job = self._jobs.get(_checked(job_id, "job"))
        if job is None:
            raise MlError("This prediction is no longer followed: open the run.", "unknown_job")
        return job

    def cancel(self, job_id: str) -> PredictionJob:
        """Stop after the current block; nothing is saved."""
        job = self.job(job_id)
        if job.status == "running":
            self._stops.add(job_id)
        return job

    # --- Saved predictions -----------------------------------------------------------------

    def list(self, dataset_id: str, run_id: str) -> list[PredictionSummary]:
        self._run_file(dataset_id, run_id, ".json")
        self.ml.run(dataset_id, run_id)
        with self._db() as connection:
            rows = connection.execute(
                "SELECT * FROM ml_predictions WHERE dataset_id = ? AND run_id = ?"
                " ORDER BY created_at DESC",
                (dataset_id, run_id),
            ).fetchall()
        return [self._summary(r) for r in rows]

    @staticmethod
    def _summary(row: sqlite3.Row) -> PredictionSummary:
        return PredictionSummary(
            row["id"],
            row["run_id"],
            row["source_id"],
            row["source_name"],
            row["rows"],
            row["less_reliable"],
            row["level"],
            row["created_at"],
            row["created_dataset_id"],
        )

    def _row(self, dataset_id: str, prediction_id: str) -> sqlite3.Row:
        _checked(dataset_id, "dataset")
        _checked(prediction_id, "prediction")
        with self._db() as connection:
            row = connection.execute(
                "SELECT * FROM ml_predictions WHERE id = ? AND dataset_id = ?",
                (prediction_id, dataset_id),
            ).fetchone()
        if row is None:
            raise MlError("Unknown prediction.", "unknown_prediction", 404)
        return row

    def get(self, dataset_id: str, prediction_id: str) -> Prediction:
        row = self._row(dataset_id, prediction_id)
        saved = SAVED.validate_json(
            self._prediction_file(dataset_id, prediction_id, ".json").read_bytes()
        )
        base = self._summary(row)
        return Prediction(**vars(base), keep=saved.keep, check=saved.check, summary=saved.summary)

    def _parts(self, dataset_id: str, prediction_id: str) -> str:
        self._row(dataset_id, prediction_id)
        folder = self._prediction_file(dataset_id, prediction_id)
        return f"read_parquet({lit((folder / '*.parquet').as_posix())})"

    def rows(
        self,
        dataset_id: str,
        prediction_id: str,
        offset: int = 0,
        limit: int = 100,
        less_reliable: bool = False,
    ) -> PredictionRows:
        """A page of the predicted rows, in the data's order (only the less reliable ones on
        demand)."""
        source = self._parts(dataset_id, prediction_id)
        where = "WHERE less_reliable" if less_reliable else ""
        limit = max(1, min(limit, MAX_PAGE))
        connection = duckdb.connect()
        try:
            (total,) = connection.execute(f"SELECT count(*) FROM {source} {where}").fetchone() or (
                0,
            )
            frame = connection.execute(
                f"SELECT __row + 1 AS row, * EXCLUDE (__row) FROM {source} {where} "
                f"ORDER BY __row LIMIT {limit} OFFSET {max(0, offset)}"
            ).df()
        finally:
            connection.close()
        return PredictionRows(
            [str(c) for c in frame.columns],
            [[_json(v) for v in values] for values in frame.itertuples(index=False)],
            int(total),
            offset,
        )

    def csv(self, dataset_id: str, prediction_id: str) -> Iterator[str]:
        """Every predicted row as CSV, streamed; every header and cell goes through `_cell`, so a
        spreadsheet never runs a value as a formula."""
        source = self._parts(dataset_id, prediction_id)
        connection = duckdb.connect()
        try:
            result = connection.execute(
                f"SELECT __row + 1 AS row, * EXCLUDE (__row) FROM {source} ORDER BY __row"
            )
            header = [d[0] for d in result.description or []]
            buffer = _CsvLine()
            yield buffer.line([_cell(str(name)) for name in header])
            while batch := result.fetchmany(CSV_BATCH):
                yield "".join(buffer.line([_cell(_text(v)) for v in values]) for values in batch)
        finally:
            connection.close()

    # --- Save as dataset -------------------------------------------------------------------

    def create_dataset(self, dataset_id: str, prediction_id: str) -> str:
        """Import the predicted data's rows with their predictions as a new dataset (with the
        predicted data's column types)."""
        row = self._row(dataset_id, prediction_id)
        datasets = self.columns.datasets
        if row["created_dataset_id"]:
            try:
                datasets.get(row["created_dataset_id"])
                return row["created_dataset_id"]
            except AppError:
                pass
        if not row["source_id"]:
            raise MlError("The predicted dataset was deleted since: predict again.", "no_source")
        record = self.columns.ready(row["source_id"])
        saved = self.get(dataset_id, prediction_id)
        views = self.columns.columns(record.id)
        names = {v.name for v in views}
        parts = self._parts(dataset_id, prediction_id)
        connection = duckdb.connect()
        try:
            produced = [
                str(d[0])
                for d in connection.execute(f"SELECT * FROM {parts} LIMIT 0").description or []
            ]
            added = [c for c in produced if c != "__row" and c not in saved.keep]
            picked = ", ".join(
                f"p.{ident(c)} AS {ident(c if c not in names else f'{c} (predicted)')}"
                for c in added
            )
            name = f"{record.name} (predictions)"
            target = datasets.new_upload_path(f"{_safe_name(name)}.parquet")
            connection.execute(
                f"COPY (SELECT s.* EXCLUDE (file_row_number), {picked} FROM "
                f"read_parquet({lit(record.storage_path)}, file_row_number = true) s "
                f"JOIN {parts} p ON s.file_row_number = p.__row ORDER BY s.file_row_number) "
                f"TO {lit(target.as_posix())} (FORMAT parquet)"
            )
        finally:
            connection.close()
        self._pending[str(datasets.resolve(str(target)))] = [
            ColumnChange(name=v.name, type=v.type, role=v.role or "", order=v.order or [])
            for v in views
        ]
        created, _ = datasets.start_import(str(target), name, ImportOptions())
        with self._db() as connection_db, connection_db:
            connection_db.execute(
                "UPDATE ml_predictions SET created_dataset_id = ? WHERE id = ?",
                (created.id, prediction_id),
            )
        return created.id

    def after_import(self, dataset_id: str, storage_path: str) -> None:
        """Import step: a dataset saved from predictions gets the predicted data's types."""
        record = self.columns.datasets.get(dataset_id)
        changes = self._pending.pop(record.source_path, None)
        if changes:
            self.columns.apply(dataset_id, storage_path, changes)

    # --- Explaining one predicted row ------------------------------------------------------

    def explain_row(self, dataset_id: str, prediction_id: str, row: int) -> use.RowExplanation:
        """Why the model predicts what it does for one predicted row (`row`: 0-based)."""
        found = self._row(dataset_id, prediction_id)
        if not found["source_id"]:
            raise MlError("The predicted dataset was deleted since.", "no_source")
        card = self.card(dataset_id, found["run_id"])
        source, views, _ = self._source(found["source_id"])
        pipeline = self._pipeline(dataset_id, found["run_id"])
        connection = duckdb.connect()
        try:
            return use.explain_row(connection, source, pipeline, card, views, row)
        finally:
            connection.close()


def _safe_name(name: str) -> str:
    return (
        "".join("_" if ch in '<>:"/\\|?*' or ord(ch) < 32 else ch for ch in name).strip()
        or "predictions"
    )


class _CsvLine:
    """One CSV line at a time (the csv module's quoting), for streaming."""

    def __init__(self) -> None:
        self._buffer = io.StringIO()
        self._writer = csv.writer(self._buffer)

    def line(self, cells: list[str]) -> str:
        self._buffer.seek(0)
        self._buffer.truncate()
        self._writer.writerow(cells)
        return self._buffer.getvalue()
