"""Clean (065): suggestions, the draft, the preview, and cleaned versions saved as datasets.

A cleaned version is written by the engine (`quantilo.cleaning`) from the parent's Parquet, which
is never written (C-01), then imported like Segments' "Save as dataset": `new_upload_path`,
`start_import`, and `after_import`, which copies the user's corrections, records the version
(`clean_versions`), moves its rowmap beside it and replaces the upload by a hard link.
"""

import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import duckdb
from pydantic import BaseModel, Field
from quantilo.catalog import ColumnChange, to_view
from quantilo.cleaning import model as m
from quantilo.cleaning.apply import CleanCancelled, apply_steps
from quantilo.cleaning.compile import compile_steps
from quantilo.cleaning.facade import corrections, type_changes
from quantilo.cleaning.preview import CleanPreview, preview
from quantilo.cleaning.report import CleanReport
from quantilo.cleaning.suggest import RULES_VERSION, default_steps, suggest
from quantilo.cleaning.suggestion import CleanSuggestion
from quantilo.ingestion.options import ImportOptions
from quantilo.ingestion.sources import list_source_files

from app.core import db
from app.core.errors import AppError
from app.core.storage import Workspace
from app.services.columns import repository as columns_repo
from app.services.columns.service import ColumnService
from app.services.datasets import repository as datasets_repo
from app.services.datasets.repository import DatasetRecord
from app.services.versions.service import RowLink

logger = logging.getLogger(__name__)

SETTINGS_KEY = "cleaning"
KEPT_JOBS = 50
READY_WAIT_SECONDS = 600  # the automatic mode waits for the import to finish
Mode = Literal["suggest", "auto", "off"]
# In the browser build (Pyodide) there are no threads: a save cannot be interrupted midway, and
# the automatic mode (which waits for the import to finish) is not offered.
THREADS = sys.platform != "emscripten"


class CleanSettings(BaseModel):
    """Settings > Cleaning. `suggest` (default for now, spec decision 10): suggestions computed
    when Clean opens; `off`: on demand; `auto`: a cleaned version made after each new import when
    an automatic rule changes a cell."""

    mode: Mode = "suggest"


class CleanSuggestions(BaseModel):
    dataset_id: str
    rows: int
    seconds: float
    suggestions: list[CleanSuggestion]


class CleanDraft(BaseModel):
    dataset_id: str
    steps: list[m.Step]
    automatic: bool  # no draft saved: the automatic suggestions' steps
    updated_at: str | None = None


class CleanPreviewRequest(BaseModel):
    steps: list[m.Step] = Field(default_factory=list, max_length=m.MAX_STEPS)
    step_id: str | None = Field(default=None, pattern=r"^s[0-9]{1,4}$")  # None: every step


class SaveRequest(BaseModel):
    steps: list[m.Step] = Field(min_length=1, max_length=m.MAX_STEPS)


@dataclass
class CleanJob:
    id: str
    status: Literal["running", "done", "failed", "cancelled"]
    stage: str
    dataset_id: str  # the parent
    progress: float | None = None
    created_dataset_id: str | None = None
    error: str | None = None


@dataclass
class CleanVersion:
    dataset_id: str
    parent_id: str
    seq: int
    name: str
    origin: str
    status: str  # the child dataset's: importing, ready, error
    created_at: str
    out_of_date: bool  # the parent was reimported or retyped since (S7)
    steps: list[m.Step] = field(default_factory=list)
    report: CleanReport | None = None


@dataclass
class CleanFamily:
    """A dataset, its original and every cleaned version of that original (v1 first)."""

    dataset_id: str
    original_id: str
    versions: list[CleanVersion]


@dataclass
class CleanLineageItem:
    dataset_id: str
    parent_id: str
    seq: int
    origin: str
    steps: int
    created_at: str
    out_of_date: bool


@dataclass
class _Pending:
    """A cleaned version being imported: what `after_import` records once it has its id."""

    parent_id: str
    seq: int
    steps: str
    origin: str
    stamp: str
    report: CleanReport
    rowmap: Path
    changes: list[ColumnChange]
    upload: Path


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stamp(record: DatasetRecord) -> str:
    return f"{record.imported_at}|{record.size_bytes}|{record.types_version}"


def _rows_state(stamp: str) -> str:
    """The part of a stamp that changes with the rows (a reimport), not with a type correction."""
    return "|".join(stamp.split("|")[:2])


def _safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip() or "dataset"


class CleanService:
    def __init__(self, columns: ColumnService, workspace: Workspace) -> None:
        self.columns = columns
        self.datasets = columns.datasets
        self.workspace = workspace
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clean")
        self._jobs: OrderedDict[str, CleanJob] = OrderedDict()
        self._stops: set[str] = set()
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}  # resolved upload path → the version

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def _folder(self, dataset_id: str) -> Path:
        return self.workspace.dataset_dir(dataset_id) / "clean"

    # --- Settings --------------------------------------------------------------------------------

    def settings(self) -> CleanSettings:
        with self._db() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE key = ?", (SETTINGS_KEY,)
            ).fetchone()
        return CleanSettings.model_validate_json(row["value"]) if row else CleanSettings()

    def update_settings(self, settings: CleanSettings) -> CleanSettings:
        with self._db() as connection, connection:
            connection.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (SETTINGS_KEY, settings.model_dump_json()),
            )
        return settings

    # --- Suggestions and draft -------------------------------------------------------------------

    def suggestions(self, dataset_id: str) -> CleanSuggestions:
        """Cached per import and types version (`clean/suggestions.json`)."""
        record = self.columns.ready(dataset_id)
        key = [RULES_VERSION, record.imported_at, record.types_version]
        cache = self._folder(dataset_id) / "suggestions.json"
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("key") == key:
                return CleanSuggestions.model_validate(saved["value"])
        except (OSError, ValueError):
            pass
        started = time.monotonic()
        views = self.columns.columns(dataset_id)
        connection = duckdb.connect()
        try:
            found = suggest(connection, record.storage_path, views)
        finally:
            connection.close()
        result = CleanSuggestions(
            dataset_id=dataset_id,
            rows=record.row_count or 0,
            seconds=round(time.monotonic() - started, 2),
            suggestions=found,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps({"key": key, "value": result.model_dump(mode="json")}), encoding="utf-8"
        )
        return result

    def draft(self, dataset_id: str, compute: bool = False) -> CleanDraft:
        """`compute`: the automatic steps even in the Off mode (Reset to automatic)."""
        self.columns.ready(dataset_id)
        with self._db() as connection:
            row = connection.execute(
                "SELECT steps, updated_at FROM clean_drafts WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
        if row is not None:
            try:
                steps = m.parse_steps(row["steps"]).steps
                return CleanDraft(
                    dataset_id=dataset_id,
                    steps=steps,
                    automatic=False,
                    updated_at=row["updated_at"],
                )
            except m.CleanError:
                logger.warning("Unreadable cleaning draft of %s: back to automatic", dataset_id)
        if not compute and self.settings().mode == "off":  # nothing computed until asked
            return CleanDraft(dataset_id=dataset_id, steps=[], automatic=True)
        steps = default_steps(self.suggestions(dataset_id).suggestions)
        return CleanDraft(dataset_id=dataset_id, steps=steps, automatic=True)

    def save_draft(self, dataset_id: str, steps: list[Any]) -> CleanDraft:
        self.columns.ready(dataset_id)
        body = m.Steps(steps=steps).to_json()  # two steps with one id: refused
        stamp = _now()
        with self._db() as connection, connection:
            connection.execute(
                "INSERT INTO clean_drafts (dataset_id, steps, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT (dataset_id) DO UPDATE SET steps = excluded.steps, "
                "updated_at = excluded.updated_at",
                (dataset_id, body, stamp),
            )
        return CleanDraft(dataset_id=dataset_id, steps=steps, automatic=False, updated_at=stamp)

    def reset_draft(self, dataset_id: str) -> CleanDraft:
        """Back to the automatic suggestions ("Reset to automatic")."""
        with self._db() as connection, connection:
            connection.execute("DELETE FROM clean_drafts WHERE dataset_id = ?", (dataset_id,))
        return self.draft(dataset_id, compute=True)

    def preview(self, dataset_id: str, request: CleanPreviewRequest) -> CleanPreview:
        record = self.columns.ready(dataset_id)
        m.Steps(steps=request.steps)
        connection = duckdb.connect()
        try:
            compiled = compile_steps(connection, record.storage_path, request.steps)
            return preview(connection, compiled, request.step_id)
        finally:
            connection.close()

    # --- Jobs ------------------------------------------------------------------------------------

    def job(self, job_id: str) -> CleanJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise AppError("This job is no longer followed.", code="unknown_job", status_code=404)
        return job

    def cancel(self, job_id: str) -> CleanJob:
        job = self.job(job_id)
        if job.status == "running":
            self._stops.add(job_id)
        return job

    def save(self, dataset_id: str, steps: list[Any], origin: str = "user") -> CleanJob:
        """Write a cleaned version in the background, then import it (S4)."""
        self.columns.ready(dataset_id)
        if self.parent_of(dataset_id) is not None:
            raise AppError(
                "This is a cleaned version: clean its original instead (Clean always starts "
                "from the original).",
                code="not_original",
                status_code=409,
            )
        m.Steps(steps=steps)
        job = CleanJob(uuid.uuid4().hex[:12], "running", "queued", dataset_id)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > KEPT_JOBS:
                self._jobs.popitem(last=False)
        self._executor.submit(self._run_save, job, steps, origin)
        return job

    def _run_save(self, job: CleanJob, steps: list[Any], origin: str) -> None:
        try:
            job.created_dataset_id = self._write(job, steps, origin)
            job.status, job.stage, job.progress = "done", "done", 1.0
        except CleanCancelled:
            job.status, job.error = "cancelled", "Stopped: nothing was saved."
        except AppError as error:
            job.status, job.error = "failed", error.message
        except Exception as error:
            logger.exception("Cleaning failed for dataset %s", job.dataset_id)
            job.status, job.error = "failed", f"Failed: {error}"
        finally:
            self._stops.discard(job.id)

    def _write(self, job: CleanJob, steps: list[Any], origin: str) -> str:
        parent = self.columns.ready(job.dataset_id)
        with self._db() as connection:
            row = connection.execute(
                "SELECT coalesce(max(seq), 0) FROM clean_versions WHERE parent_id = ?",
                (parent.id,),
            ).fetchone()
            records = columns_repo.list_columns(connection, parent.id)
        seq = int(row[0]) + 1
        name = f"{parent.name} · Cleaned v{seq}"
        target = self.datasets.new_upload_path(f"{_safe_name(name)}.parquet")
        rowmap = target.with_name("rowmap.parquet")
        job.stage, job.progress = "cleaning", 0.1
        connection = duckdb.connect()
        if THREADS:
            threading.Thread(target=self._watch, args=(job, connection), daemon=True).start()
        try:
            report = apply_steps(
                connection,
                parent.storage_path,
                steps,
                target,
                rowmap,
                lambda: job.id in self._stops,
            )
        except duckdb.InterruptException:
            shutil.rmtree(target.parent, ignore_errors=True)
            raise CleanCancelled from None
        except BaseException:
            shutil.rmtree(target.parent, ignore_errors=True)
            raise
        finally:
            job.stage = "importing"
            connection.close()
        if job.id in self._stops:
            shutil.rmtree(target.parent, ignore_errors=True)
            raise CleanCancelled
        job.progress = 0.7
        self._pending[str(self.datasets.resolve(str(target)))] = _Pending(
            parent_id=parent.id,
            seq=seq,
            steps=m.Steps(steps=steps).to_json(),
            origin=origin,
            stamp=_stamp(parent),
            report=report,
            rowmap=rowmap,
            changes=corrections(steps, records),
            upload=target,
        )
        record, _ = self.datasets.start_import(str(target), name, ImportOptions())
        # done once the version is a ready dataset, recorded in clean_versions (the page then
        # lists it at once)
        deadline = time.monotonic() + READY_WAIT_SECONDS
        while THREADS and time.monotonic() < deadline:
            child = self.datasets.get(record.id)
            if child.status == "error":
                raise AppError(child.error or "The cleaned version could not be imported.")
            if child.status == "ready":
                break
            time.sleep(0.1)
        return record.id

    def _watch(self, job: CleanJob, connection: duckdb.DuckDBPyConnection) -> None:
        """Interrupt DuckDB when the job is cancelled (the engine itself has no threads)."""
        while job.stage == "cleaning":
            if job.id in self._stops:
                with suppress(duckdb.Error):
                    connection.interrupt()
                return
            time.sleep(0.1)

    # --- After an import -------------------------------------------------------------------------

    def after_import(self, dataset_id: str, storage_path: str) -> None:
        """A cleaned version: record it. A new dataset in automatic mode: clean it (S9)."""
        record = self.datasets.get(dataset_id)
        pending = self._pending.pop(record.source_path, None)
        if pending is not None:
            self._record(dataset_id, storage_path, pending)
            return
        if not THREADS or self.settings().mode != "auto" or record.imported_at is not None:
            return  # a reimport keeps its versions (they show "Out of date")
        if self.parent_of(dataset_id) is not None:
            return
        self._executor.submit(self._auto, dataset_id)

    def _record(self, dataset_id: str, storage_path: str, pending: _Pending) -> None:
        with self._db() as connection:
            names = {r.name for r in columns_repo.list_columns(connection, dataset_id)}
        changes = [c for c in pending.changes if c.name in names]
        if changes:
            self.columns.apply(dataset_id, storage_path, changes)
        steps = m.parse_steps(pending.steps).steps
        with self._db() as connection:
            child = [to_view(r) for r in columns_repo.list_columns(connection, dataset_id)]
            parent = [to_view(r) for r in columns_repo.list_columns(connection, pending.parent_id)]
        pending.report.type_changes = type_changes(parent, child, steps)
        folder = self._folder(dataset_id)
        folder.mkdir(parents=True, exist_ok=True)
        if pending.rowmap.is_file():
            shutil.move(str(pending.rowmap), folder / "rowmap.parquet")
        with self._db() as connection, connection:
            connection.execute(
                "INSERT INTO clean_versions (dataset_id, parent_id, seq, steps, origin, "
                "parent_stamp, report, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    dataset_id,
                    pending.parent_id,
                    pending.seq,
                    pending.steps,
                    pending.origin,
                    pending.stamp,
                    json.dumps(pending.report.to_dict()),
                    _now(),
                ),
            )
        self._link_upload(dataset_id, storage_path, pending.upload)

    def _link_upload(self, dataset_id: str, storage_path: str, upload: Path) -> None:
        """The upload is a second copy of the child: a hard link to the imported Parquet keeps
        one on disk (study § 4), and the recorded file state follows it (no "source changed")."""
        linked = upload.with_name(upload.name + ".link")
        try:
            os.link(storage_path, linked)
            linked.replace(upload)
        except OSError:
            linked.unlink(missing_ok=True)
            return
        with self._db() as connection, connection:
            files = list_source_files(upload, None)
            datasets_repo.replace_files(connection, dataset_id, files)

    def _auto(self, dataset_id: str) -> None:
        """Automatic mode: wait for the import to finish, then save the automatic steps as
        "Cleaned v1" when one of them changes a cell."""
        deadline = time.monotonic() + READY_WAIT_SECONDS
        while time.monotonic() < deadline:
            try:
                status = self.datasets.get(dataset_id).status
            except AppError:
                return  # deleted meanwhile
            if status == "ready":
                break
            if status == "error":
                return
            time.sleep(0.2)
        else:
            return
        try:
            steps = default_steps(self.suggestions(dataset_id).suggestions)
            if not steps:
                return
            found = self.preview(dataset_id, CleanPreviewRequest(steps=steps))
            if not (found.cells_changed or found.rows_removed):
                return
            job = CleanJob(uuid.uuid4().hex[:12], "running", "queued", dataset_id)
            self._jobs[job.id] = job
            self._run_save(job, steps, "auto")
        except AppError:
            logger.exception("Automatic cleaning failed for dataset %s", dataset_id)

    # --- Versions --------------------------------------------------------------------------------

    def parent_of(self, dataset_id: str) -> str | None:
        with self._db() as connection:
            row = connection.execute(
                "SELECT parent_id FROM clean_versions WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
        return row["parent_id"] if row else None

    def _version(
        self, row: sqlite3.Row, stamps: dict[str, str], names: dict[str, Any]
    ) -> CleanVersion:
        child = names.get(row["dataset_id"])
        return CleanVersion(
            dataset_id=row["dataset_id"],
            parent_id=row["parent_id"],
            seq=int(row["seq"]),
            name=child.name if child else f"Cleaned v{row['seq']}",
            origin=row["origin"],
            status=child.status if child else "error",
            created_at=row["created_at"],
            out_of_date=stamps.get(row["parent_id"]) != row["parent_stamp"],
            steps=m.parse_steps(row["steps"]).steps,
            report=CleanReport.from_dict(json.loads(row["report"])),
        )

    def family(self, dataset_id: str) -> CleanFamily:
        """The original of `dataset_id` (itself when it is one) and its cleaned versions."""
        self.datasets.get(dataset_id)
        original = self.parent_of(dataset_id) or dataset_id
        records = {r.id: r for r in self.datasets.list_all()}
        stamps = {key: _stamp(r) for key, r in records.items()}
        with self._db() as connection:
            rows = connection.execute(
                "SELECT * FROM clean_versions WHERE parent_id = ? ORDER BY seq", (original,)
            ).fetchall()
        return CleanFamily(dataset_id, original, [self._version(r, stamps, records) for r in rows])

    def lineage(self) -> list[CleanLineageItem]:
        """Every cleaned version and its parent (for the home page and the top bar, C-13)."""
        stamps = {r.id: _stamp(r) for r in self.datasets.list_all()}
        with self._db() as connection:
            rows = connection.execute(
                "SELECT dataset_id, parent_id, seq, origin, steps, parent_stamp, created_at "
                "FROM clean_versions ORDER BY parent_id, seq"
            ).fetchall()
        return [
            CleanLineageItem(
                dataset_id=row["dataset_id"],
                parent_id=row["parent_id"],
                seq=int(row["seq"]),
                origin=row["origin"],
                steps=len(json.loads(row["steps"]).get("steps", [])),
                created_at=row["created_at"],
                out_of_date=stamps.get(row["parent_id"]) != row["parent_stamp"],
            )
            for row in rows
        ]

    def row_link(self, dataset_id: str) -> RowLink | None:
        """For Changes (095): a cleaned version's original and row map, to pair their rows. The
        row map holds while the original's rows are those it was made from (a type correction
        keeps them; a reimport does not)."""
        with self._db() as connection:
            row = connection.execute(
                "SELECT parent_id, seq, parent_stamp FROM clean_versions WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            parent = self.datasets.get(row["parent_id"])
        except AppError:
            return None
        made_from = _rows_state(row["parent_stamp"])
        path = self.rowmap(dataset_id)
        return RowLink(
            parent_id=row["parent_id"],
            name=f"Cleaned v{row['seq']}",
            rowmap=str(path) if path else None,
            made_from=made_from,
            stale=_rows_state(_stamp(parent)) != made_from,
        )

    def rowmap(self, dataset_id: str) -> Path | None:
        path = self._folder(dataset_id) / "rowmap.parquet"
        return path if path.is_file() else None
