"""Dataset lifecycle: preview a source, import it in the background, list, rename, delete."""

import json
import logging
import shutil
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Protocol

import duckdb
from quantilo.errors import NotFoundError
from quantilo.ingestion.browse import Listing, list_directory
from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.importer import ImportCancelled, import_to_parquet
from quantilo.ingestion.options import ImportOptions, SourceOptions
from quantilo.ingestion.original import Original, sheet_cells, text_head
from quantilo.ingestion.paths import is_within, to_local_path
from quantilo.ingestion.preview import (
    ColumnInfo,
    Preview,
    TableChoice,
    TableStack,
    detection_of,
    document_choices,
    preview_source,
    sheet_choices,
    stack_groups,
)
from quantilo.ingestion.sources import (
    DOCUMENT_FORMATS,
    SourceFile,
    detect_format,
    fingerprint,
    list_source_files,
)
from quantilo.ingestion.sql import lit

from app.config import Settings
from app.core import db
from app.core.errors import AppError
from app.core.jobs import JobRunner
from app.core.storage import Workspace
from app.services.datasets import repository as repo
from app.services.datasets import sources as source_repo
from app.services.datasets.repository import DatasetRecord, JobRecord, now
from app.services.datasets.sources import LinkRecord, SourceRecord

SourceState = Literal["ok", "changed", "missing"]


class FormatHandler(Protocol):
    """What a dataset of another kind than a table does at a reimport, a source check and a
    deletion (106: `format = 'images'`, an image set)."""

    def reimport(self, record: DatasetRecord) -> JobRecord: ...

    def source_state(self, record: DatasetRecord) -> SourceState: ...

    def deleted(self, record: DatasetRecord) -> None: ...


# 085: what a source of several tables is, by the format of its file
SOURCE_KINDS: dict[str, source_repo.SourceKind] = {
    "excel": "workbook",
    "json": "json",
    "jsonl": "jsonl",
    "xml": "xml",
}

logger = logging.getLogger(__name__)


class DatasetService:
    def __init__(self, workspace: Workspace, settings: Settings, jobs: JobRunner) -> None:
        self.workspace = workspace
        self.settings = settings
        self.jobs = jobs
        # Steps run after the Parquet copy is written (e.g. type detection), in order.
        self.after_import: list[Callable[[str, str], object]] = []
        # Steps run at a reimport, before the new copy replaces the current one (e.g.
        # keeping the previous version, 031).
        self.before_reimport: list[Callable[[str, str], object]] = []
        # One per running import: deleting the dataset sets it to stop the import.
        self._stops: dict[str, threading.Event] = {}
        # 106: datasets that are not tables, by `format` (image sets)
        self.handlers: dict[str, FormatHandler] = {}

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            yield connection
        finally:
            connection.close()

    # --- Sources ---------------------------------------------------------------------------

    def allowed_roots(self) -> list[Path] | None:
        if self.settings.data_dir is None:
            return None
        return [self.settings.data_dir, self.workspace.uploads_dir, self.settings.samples_dir]

    def resolve(self, raw_path: str) -> Path:
        """Validate a user path lexically, before any file system access (see `paths.py`)."""
        path = to_local_path(raw_path)
        roots = self.allowed_roots()
        if roots is not None and not is_within(path, [to_local_path(str(r)) for r in roots]):
            raise IngestionError("This path is outside the allowed data folder.")
        return path

    def browse(self, raw_path: str | None) -> Listing:
        roots = None if self.settings.data_dir is None else [self.settings.data_dir]
        return list_directory(self.resolve(raw_path) if raw_path else None, roots)

    def preview(self, raw_path: str, options: SourceOptions) -> Preview:
        return preview_source(self.resolve(raw_path), options)

    def stacks(self, raw_path: str) -> list[TableStack]:
        """S1 (C1): the tables of a workbook that share their columns, read on demand."""
        path = self.resolve(raw_path)
        if not path.is_file() or detect_format(path) != "excel":
            return []
        return stack_groups(path)

    def original(self, raw_path: str, sheet: str | None) -> Original:
        """The left pane of the import screen: a sheet's cells, or a document's first lines."""
        path = self.resolve(raw_path)
        if not path.is_file():
            raise IngestionError("Only a file has an original to show.")
        detected = detection_of(path)
        if detected.format == "excel":
            if not sheet:
                raise AppError("Choose a sheet.", code="sheet_required", status_code=422)
            return sheet_cells(path, sheet)
        if detected.kind in ("table", "document"):
            return text_head(path)
        raise IngestionError(f"{path.name}: {detected.reason}")

    # --- Sources (085): one file, several tables ---------------------------------------------

    def import_tables(
        self,
        raw_path: str,
        tables: list[tuple[str, str | None]],
        options: ImportOptions,
        name: str | None = None,
        stack: bool = False,
    ) -> tuple[SourceRecord, list[tuple[DatasetRecord, JobRecord | None]]]:
        """Import the chosen tables of one file (table id, dataset name) as one source: a
        workbook's tables (080b) or a document's linked tables (085), each a dataset. `stack`
        (S1, C1): tables of one workbook that share their columns make one dataset instead."""
        if not tables:
            raise AppError("Choose at least one table.", code="no_table", status_code=422)
        if stack:
            ids = [table_id for table_id, _ in tables]
            group = next((g for g in self.stacks(raw_path) if set(ids) <= set(g.tables)), None)
            if group is None or len(ids) < 2:
                raise AppError(
                    "These tables do not share their columns: they cannot be stacked (S1).",
                    code="not_stackable",
                    status_code=422,
                )
            ordered = [table_id for table_id in group.tables if table_id in set(ids)]
            options = options.model_copy(update={"stack": ordered, "table": None})
            tables = [(ordered[0], tables[0][1])]
        path = self.resolve(raw_path)
        if not path.is_file():
            raise IngestionError("A source of several tables is a single file.")
        files = list_source_files(path)
        source_file = files[0]
        kind = SOURCE_KINDS.get(source_file.format)
        if kind is None:
            raise AppError(
                "Only a workbook or a JSON, JSON Lines or XML document gives several tables.",
                code="not_multi_table",
                status_code=422,
            )
        choices = self._choices(path, source_file.format, options)
        detected = detection_of(path)
        source = SourceRecord(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            name=(name or "").strip() or path.stem,
            source_path=str(path),
            fingerprint=fingerprint(files, {}),
            size_bytes=source_file.size,
            created_at=now(),
            meta=json.dumps({"detected": detected.__dict__, "sheets": None}),
            plan=json.dumps(
                {
                    "tables": [table_id for table_id, _ in tables],
                    "options": options.model_dump(mode="json", exclude={"table"}),
                }
            ),
        )
        with self._db() as connection:
            source_repo.insert_source(connection, source)
        started: list[tuple[DatasetRecord, JobRecord | None]] = []
        by_table: dict[str, str] = {}
        for table_id, table_name in tables:
            choice = choices.get(table_id)
            locator = choice.location if choice else table_id
            layout = json.dumps(choice, default=lambda o: o.__dict__) if choice else None
            table_options = options.model_copy(update={"table": table_id})
            fallback = f"{source.name} - {choice.name.split(' - ', 1)[-1]}" if choice else None
            if options.stack:  # S1: one dataset of the stacked tables
                locator = " + ".join(options.stack)[:2_000]
                table_options = options
                fallback = f"{source.name} - stacked"
            record, job = self.start_import(
                raw_path,
                table_name or fallback or f"{source.name} - {table_id}",
                table_options,
                source=(source.id, locator, layout),
            )
            by_table[table_id] = record.id
            started.append((record, job))
        if source_file.format in DOCUMENT_FORMATS:
            _, links = document_choices(path, options)
            records = [
                LinkRecord(
                    by_table[link.parent],
                    by_table[link.child],
                    link.parent_key,
                    link.child_key,
                    link.rule,
                )
                for link in links
                if link.parent in by_table and link.child in by_table
            ]
            with self._db() as connection:
                source_repo.insert_links(connection, records)
        return source, started

    @staticmethod
    def _choices(path: Path, file_format: str, options: SourceOptions) -> dict[str, TableChoice]:
        if file_format == "excel":
            return {choice.id: choice for choice in sheet_choices(path)}
        return {choice.id: choice for choice in document_choices(path, options)[0]}

    def get_source(self, source_id: str) -> SourceRecord:
        with self._db() as connection:
            record = source_repo.get_source(connection, source_id)
        if record is None:
            raise NotFoundError("Source")
        return record

    def list_sources(self) -> list[tuple[SourceRecord, list[str]]]:
        with self._db() as connection:
            return [
                (record, source_repo.source_datasets(connection, record.id))
                for record in source_repo.list_sources(connection)
            ]

    def source_datasets(self, source_id: str) -> list[str]:
        with self._db() as connection:
            return source_repo.source_datasets(connection, source_id)

    def delete_source(self, source_id: str) -> None:
        """Its tables go with it; their cleaned versions stay as ordinary datasets (0024)."""
        self.get_source(source_id)
        for dataset_id in self.source_datasets(source_id):
            self.delete(dataset_id)
        with self._db() as connection:
            source_repo.delete_source(connection, source_id)

    def origin(
        self, dataset_id: str
    ) -> tuple[SourceRecord | None, list[str], DatasetRecord, list[LinkRecord]]:
        record = self.get(dataset_id)
        if record.source_id is None:
            return None, [], record, []
        with self._db() as connection:
            source = source_repo.get_source(connection, record.source_id)
            siblings = source_repo.source_datasets(connection, record.source_id)
            links = source_repo.links_of(connection, dataset_id)
        return source, siblings, record, links

    def samples(self) -> list[SourceFile]:
        directory = self.settings.samples_dir
        if not directory.is_dir():
            return []
        try:
            return list_source_files(directory)
        except IngestionError:
            return []

    # --- Import ----------------------------------------------------------------------------

    def start_import(
        self,
        raw_path: str,
        name: str | None,
        options: ImportOptions,
        source: tuple[str, str, str | None] | None = None,
    ) -> tuple[DatasetRecord, JobRecord | None]:
        """`source` = (source id, locator, layout JSON) for one table of a source (085)."""
        path = self.resolve(raw_path)
        files = list_source_files(path, options.files)
        source_fingerprint = fingerprint(files, options.model_dump(mode="json"))
        with self._db() as connection:
            existing = repo.find_ready_by_fingerprint(connection, source_fingerprint)
            if source is None and existing and Path(existing.storage_path).is_file():
                repo.update_dataset(connection, existing.id, last_opened_at=now())
                return self.get(existing.id), None

            dataset_id = uuid.uuid4().hex[:12]
            formats = {file.format for file in files}
            record = DatasetRecord(
                id=dataset_id,
                name=(name or "").strip() or (path.stem if path.is_file() else path.name),
                source_path=str(path),
                source_kind="file" if path.is_file() else "folder",
                format=formats.pop() if len(formats) == 1 else "mixed",
                options=options.model_dump_json(),
                fingerprint=source_fingerprint,
                storage_path=str(self.workspace.dataset_dir(dataset_id) / "data.parquet"),
                status="importing",
                created_at=now(),
                sample_pct=options.sample_pct,
                source_id=source[0] if source else None,
                source_locator=source[1] if source else None,
                layout=source[2] if source else None,
            )
            repo.insert_dataset(connection, record)
            repo.replace_files(connection, dataset_id, files)
            job = self._new_job(connection, dataset_id)
        self.jobs.submit(lambda: self._run_import(record.id, job.id, files, options))
        return record, job

    def reimport(self, dataset_id: str) -> JobRecord:
        record = self.get(dataset_id)
        if (handler := self.handlers.get(record.format)) is not None:
            return handler.reimport(record)
        options = ImportOptions.model_validate_json(record.options)
        path = Path(record.source_path)
        if not path.exists():
            raise IngestionError("The source no longer exists: it can't be reimported.")
        files = list_source_files(path, options.files)
        with self._db() as connection:
            repo.update_dataset(
                connection,
                dataset_id,
                status="importing",
                error=None,
                fingerprint=fingerprint(files, options.model_dump(mode="json")),
            )
            repo.replace_files(connection, dataset_id, files)
            job = self._new_job(connection, dataset_id)
        self.jobs.submit(lambda: self._run_import(dataset_id, job.id, files, options))
        return job

    def stop_event(self, dataset_id: str) -> threading.Event:
        """The event `delete()` sets to stop a running import (shared with the image sets, 106)."""
        return self._stops.setdefault(dataset_id, threading.Event())

    def release_stop(self, dataset_id: str) -> None:
        self._stops.pop(dataset_id, None)

    def new_job(self, connection: sqlite3.Connection, dataset_id: str) -> JobRecord:
        return self._new_job(connection, dataset_id)

    def _new_job(self, connection: sqlite3.Connection, dataset_id: str) -> JobRecord:
        job = JobRecord(
            id=uuid.uuid4().hex[:12],
            kind="import",
            dataset_id=dataset_id,
            status="pending",
            created_at=now(),
            message="Queued",
        )
        repo.insert_job(connection, job)
        return job

    def _run_import(
        self, dataset_id: str, job_id: str, files: list[SourceFile], options: ImportOptions
    ) -> None:
        stop = self._stops.setdefault(dataset_id, threading.Event())
        try:
            self._import(dataset_id, job_id, files, options, stop)
        finally:
            self._stops.pop(dataset_id, None)

    def _import(
        self,
        dataset_id: str,
        job_id: str,
        files: list[SourceFile],
        options: ImportOptions,
        stop: threading.Event,
    ) -> None:
        with self._db() as connection:
            repo.update_job(connection, job_id, status="running", message="Starting")

            def progress(fraction: float, message: str) -> None:
                repo.update_job(connection, job_id, progress=fraction, message=message)

            target = self.workspace.dataset_dir(dataset_id) / "data.parquet"
            if target.is_file():
                progress(0.01, "Keeping the previous version")
                for step in self.before_reimport:
                    try:
                        step(dataset_id, str(target))
                    except Exception:
                        # Not fatal: the reimport goes on, the comparison has less detail.
                        logger.exception("Pre-reimport step failed for dataset %s", dataset_id)
            try:
                result = import_to_parquet(files, options, target, progress, stop)
            except ImportCancelled:
                # delete() removed the dataset; on Windows its folder may have been locked.
                shutil.rmtree(self.workspace.dataset_dir(dataset_id), ignore_errors=True)
                return
            except Exception as error:
                message = (
                    error.message if isinstance(error, AppError) else f"Import failed: {error}"
                )
                repo.update_dataset(connection, dataset_id, status="error", error=message)
                repo.update_job(
                    connection, job_id, status="failed", message=message, finished_at=now()
                )
                return
            if stop.is_set():
                return
            progress(0.92, "Detecting column types")
            for step in self.after_import:
                try:
                    step(dataset_id, str(target))
                except Exception:
                    # Not fatal: the dataset is usable and the step runs again on demand.
                    logger.exception("Post-import step failed for dataset %s", dataset_id)
            stamp = now()
            repo.update_dataset(
                connection,
                dataset_id,
                status="ready",
                error=None,
                row_count=result.row_count,
                col_count=result.col_count,
                size_bytes=result.size_bytes,
                imported_at=stamp,
                last_opened_at=stamp,
            )
            repo.update_job(
                connection, job_id, status="done", progress=1, message="Done", finished_at=stamp
            )

    # --- Datasets --------------------------------------------------------------------------

    def get(self, dataset_id: str) -> DatasetRecord:
        with self._db() as connection:
            record = repo.get_dataset(connection, dataset_id)
        if record is None:
            raise NotFoundError("Dataset")
        return record

    def list_all(self) -> list[DatasetRecord]:
        with self._db() as connection:
            return repo.list_datasets(connection)

    def latest_job(self, dataset_id: str) -> JobRecord | None:
        with self._db() as connection:
            return repo.latest_job(connection, dataset_id)

    def job(self, job_id: str) -> JobRecord:
        with self._db() as connection:
            job = repo.get_job(connection, job_id)
        if job is None:
            raise NotFoundError("Job")
        return job

    def source_state(self, record: DatasetRecord) -> SourceState:
        if (handler := self.handlers.get(record.format)) is not None:
            return handler.source_state(record)
        with self._db() as connection:
            recorded = repo.list_files(connection, record.id)
        source = Path(record.source_path)
        if not source.exists():
            return "missing"
        for path, size, mtime in recorded:
            file = Path(path)
            if not file.exists():
                return "missing" if record.source_kind == "file" else "changed"
            stat = file.stat()
            if stat.st_size != size or stat.st_mtime != mtime:
                return "changed"
        if record.source_kind == "folder":
            options = ImportOptions.model_validate_json(record.options)
            try:
                current = {str(f.path) for f in list_source_files(source, options.files)}
            except IngestionError:
                return "changed"
            if current != {path for path, _, _ in recorded}:
                return "changed"
        return "ok"

    def rename(self, dataset_id: str, name: str) -> DatasetRecord:
        self.get(dataset_id)
        clean = name.strip()
        if not clean:
            raise AppError("The name can't be empty.", code="invalid_name", status_code=422)
        with self._db() as connection:
            repo.update_dataset(connection, dataset_id, name=clean)
        return self.get(dataset_id)

    def mark_opened(self, dataset_id: str) -> DatasetRecord:
        self.get(dataset_id)
        with self._db() as connection:
            repo.update_dataset(connection, dataset_id, last_opened_at=now())
        return self.get(dataset_id)

    def delete(self, dataset_id: str) -> None:
        """Also cancels the import of a dataset still importing."""
        record = self.get(dataset_id)
        if record.status == "importing":
            self._stops.setdefault(dataset_id, threading.Event()).set()
        with self._db() as connection:
            repo.delete_dataset(connection, dataset_id)
            if record.source_id:
                source_repo.delete_if_empty(connection, record.source_id)
            still_used = repo.count_datasets_with_source(connection, record.source_path) > 0
        shutil.rmtree(self.workspace.dataset_dir(dataset_id), ignore_errors=True)
        if (handler := self.handlers.get(record.format)) is not None:
            handler.deleted(record)
        source = Path(record.source_path)
        uploads = self.workspace.uploads_dir.resolve()
        if not still_used and source.is_relative_to(uploads):
            # Uploaded files are internal copies: remove the upload folder with its dataset.
            shutil.rmtree(uploads / source.relative_to(uploads).parts[0], ignore_errors=True)

    # --- Stored data -----------------------------------------------------------------------

    def _ready_storage(self, dataset_id: str) -> str:
        record = self.get(dataset_id)
        if record.status != "ready":
            raise AppError("This dataset is not ready yet.", code="not_ready", status_code=409)
        return f"read_parquet({lit(record.storage_path)})"

    def schema(self, dataset_id: str) -> list[ColumnInfo]:
        storage = self._ready_storage(dataset_id)
        connection = duckdb.connect()
        try:
            rows = connection.execute(f"DESCRIBE SELECT * FROM {storage}").fetchall()
        finally:
            connection.close()
        return [ColumnInfo(row[0], row[1]) for row in rows]

    def head(self, dataset_id: str, limit: int) -> list[list[str | None]]:
        storage = self._ready_storage(dataset_id)
        connection = duckdb.connect()
        try:
            rows = connection.execute(
                f"SELECT COLUMNS(*)::VARCHAR FROM {storage} LIMIT {int(limit)}"
            ).fetchall()
        finally:
            connection.close()
        return [list(row) for row in rows]

    # --- Uploads ---------------------------------------------------------------------------

    def new_upload_path(self, filename: str) -> Path:
        safe_name = Path(filename).name.strip() or "upload"
        directory = self.workspace.uploads_dir / uuid.uuid4().hex[:12]
        directory.mkdir(parents=True, exist_ok=True)
        return directory / safe_name

    def recover_interrupted(self) -> None:
        with self._db() as connection:
            repo.interrupt_unfinished(connection)
