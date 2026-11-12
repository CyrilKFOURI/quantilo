"""Image sets as datasets (106, study 2026-11-11 vision-et-xai § 3 and § 11).

An image set is an ordinary row of `datasets` with `format = 'images'`: its table (one row per
image, the columns of 104 and the quality of 105) is the dataset's `data.parquet`, so every tabular
page works on the metadata, projects (087) file it and its versions (031) are kept like any other.
No migration: the files live beside it, addressed by content, under `workspace/images/<id>/`.

- `preview`: what the import would read, nothing decoded (the rules R-VIS-* with their reasons).
- `start_import`: a background job (progress, cancelled by a deletion), then the usual post-import
  steps (type detection, project model, version snapshot); the report (reading rules, files left
  out, quality findings) is kept in `datasets/<id>/images/report.json`.
- `file`: an image, its thumbnail or its mask, by `image_id` only. The path comes from the table,
  is confined to the set's folder and must be a local drive path (never a network path); the
  content type is read from the first bytes, never from a name.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import duckdb
from quantilo.errors import NotFoundError
from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.images import sniff_image
from quantilo.ingestion.paths import is_within, to_local_path
from quantilo.ingestion.sql import lit

from app.core import db
from app.core.errors import AppError
from app.core.storage import Workspace
from app.services.datasets import repository as repo
from app.services.datasets.repository import DatasetRecord, JobRecord, now
from app.services.datasets.service import DatasetService, SourceState

logger = logging.getLogger(__name__)

FORMAT = "images"
IMAGE_ID = re.compile(r"^[0-9a-f]{16}$")
MEDIA_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "avif": "image/avif",
}
LIST_LIMIT = 500  # files left out and label conflicts kept in the report (the counts are exact)
FileKind = Literal["file", "thumb", "mask"]
_COLUMN = {"file": "stored_path", "thumb": "thumb_path", "mask": "mask_path"}


class ImportStopped(Exception):
    """Raised from the progress callback when the dataset was deleted during its import."""


@dataclass
class ImageImportOptions:
    """What an image-set import was asked (stored as the dataset's `options` JSON)."""

    labels: str | None = None
    label_column: str | None = None
    ignore_rules: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"kind": FORMAT, **asdict(self)}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ImageImportOptions:
        data = json.loads(text)
        return cls(
            labels=data.get("labels"),
            label_column=data.get("label_column"),
            ignore_rules=list(data.get("ignore_rules") or []),
            thresholds=dict(data.get("thresholds") or {}),
        )


def _stat(path: Path) -> tuple[int, float]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime


def _note(rule: Any) -> dict[str, Any]:
    return {"code": rule.code, "message": rule.message, "reason": rule.reason, "count": rule.count}


class ImageSetService:
    def __init__(self, workspace: Workspace, datasets: DatasetService) -> None:
        self.workspace = workspace
        self.datasets = datasets
        datasets.handlers[FORMAT] = self

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def images_dir(self, dataset_id: str) -> Path:
        return self.workspace.root / "images" / dataset_id

    def report_path(self, dataset_id: str) -> Path:
        return self.workspace.dataset_dir(dataset_id) / "images" / "report.json"

    # --- preview and import --------------------------------------------------------------------

    def _paths(self, raw_path: str, options: ImageImportOptions) -> tuple[Path, Path | None]:
        path = self.datasets.resolve(raw_path)
        labels = self.datasets.resolve(options.labels) if options.labels else None
        if not path.exists():
            raise NotFoundError("Source")
        return path, labels

    def preview(self, raw_path: str, options: ImageImportOptions) -> dict[str, Any]:
        from quantilo.vision.ingest import preview_image_set

        path, labels = self._paths(raw_path, options)
        found = preview_image_set(
            path,
            labels=labels,
            label_column=options.label_column,
            ignore_rules=options.ignore_rules,
        )
        return {
            "source_kind": found.source_kind,
            "suggested_name": path.stem if path.is_file() else path.name,
            "sentence": found.sentence,
            "n_images": found.n_images,
            "bytes": found.bytes,
            "formats": found.formats,
            "labels": found.labels,
            "splits": found.splits,
            "normal": found.normal,
            "masks": found.masks,
            "rules": [_note(r) for r in found.rules],
            "skipped": [asdict(s) for s in found.skipped[:LIST_LIMIT]],
            "skipped_count": len(found.skipped),
            "conflicts": len(found.conflicts),
            "sample": found.sample,
            "label_file": found.label_file,
        }

    def _fingerprint(self, path: Path, labels: Path | None, options: ImageImportOptions) -> str:
        parts: dict[str, Any] = {"path": str(path), "state": _stat(path)}
        parts["options"] = asdict(options)
        if labels is not None:
            parts["labels"] = [str(labels), *_stat(labels)]
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def start_import(
        self, raw_path: str, name: str | None, options: ImageImportOptions
    ) -> tuple[DatasetRecord, JobRecord | None]:
        from quantilo.vision.quality import thresholds_from

        path, labels = self._paths(raw_path, options)
        try:
            thresholds_from(options.thresholds)
        except ValueError as error:
            raise AppError(str(error), code="invalid_thresholds", status_code=422) from error
        if labels is not None:
            options = ImageImportOptions(
                str(labels), options.label_column, options.ignore_rules, options.thresholds
            )
        fingerprint = self._fingerprint(path, labels, options)
        with self._db() as connection:
            existing = repo.find_ready_by_fingerprint(connection, fingerprint)
            if existing and Path(existing.storage_path).is_file():
                repo.update_dataset(connection, existing.id, last_opened_at=now())
                return self.datasets.get(existing.id), None
            dataset_id = uuid.uuid4().hex[:12]
            record = DatasetRecord(
                id=dataset_id,
                name=(name or "").strip() or (path.stem if path.is_file() else path.name),
                source_path=str(path),
                source_kind="folder" if path.is_dir() else "file",
                format=FORMAT,
                options=options.to_json(),
                fingerprint=fingerprint,
                storage_path=str(self.workspace.dataset_dir(dataset_id) / "data.parquet"),
                status="importing",
                created_at=now(),
            )
            repo.insert_dataset(connection, record)
            self._record_source(connection, dataset_id, path)
            job = self.datasets.new_job(connection, dataset_id)
        self.datasets.jobs.submit(lambda: self._run(record.id, job.id))
        return record, job

    def _record_source(self, connection: sqlite3.Connection, dataset_id: str, path: Path) -> None:
        """Only the source itself (its size and date) is recorded, not 50,000 files: the source
        check of a listing stays one `stat`."""
        size, mtime = _stat(path)
        connection.execute("DELETE FROM dataset_files WHERE dataset_id = ?", (dataset_id,))
        connection.execute(
            "INSERT INTO dataset_files (dataset_id, path, size, mtime) VALUES (?, ?, ?, ?)",
            (dataset_id, str(path), size, mtime),
        )
        connection.commit()

    def _run(self, dataset_id: str, job_id: str) -> None:
        stop = self.datasets.stop_event(dataset_id)
        try:
            self._import(dataset_id, job_id, stop.is_set)
        finally:
            self.datasets.release_stop(dataset_id)

    def _import(self, dataset_id: str, job_id: str, stopped: Any) -> None:
        from quantilo.vision.ingest import image_set
        from quantilo.vision.quality import thresholds_from

        with self._db() as connection:
            record = repo.get_dataset(connection, dataset_id)
            if record is None:
                return
            options = ImageImportOptions.from_json(record.options)
            repo.update_job(connection, job_id, status="running", message="Starting")

            def progress(fraction: float, message: str) -> None:
                if stopped():
                    raise ImportStopped
                repo.update_job(connection, job_id, progress=0.02 + fraction * 0.9, message=message)

            target = Path(record.storage_path)
            if target.is_file():
                for step in self.datasets.before_reimport:
                    try:
                        step(dataset_id, str(target))
                    except Exception:
                        logger.exception("Pre-reimport step failed for image set %s", dataset_id)
            images_dir = self.images_dir(dataset_id)
            shutil.rmtree(images_dir, ignore_errors=True)  # a reimport stores the files again
            try:
                result = image_set(
                    record.source_path,
                    workspace=self.workspace.root,
                    dataset_id=dataset_id,
                    labels=options.labels,
                    label_column=options.label_column,
                    ignore_rules=options.ignore_rules,
                    quality_thresholds=thresholds_from(options.thresholds),
                    table_path=target,
                    progress=progress,
                )
            except ImportStopped:
                shutil.rmtree(images_dir, ignore_errors=True)
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
            if stopped():
                return
            self._write_report(dataset_id, result)
            repo.update_job(connection, job_id, progress=0.95, message="Detecting column types")
            for step in self.datasets.after_import:
                try:
                    step(dataset_id, str(target))
                except Exception:
                    logger.exception("Post-import step failed for image set %s", dataset_id)
            stamp = now()
            repo.update_dataset(
                connection,
                dataset_id,
                status="ready",
                error=None,
                row_count=result.n_images,
                col_count=result.table.num_columns,
                size_bytes=target.stat().st_size,
                imported_at=stamp,
                last_opened_at=stamp,
            )
            repo.update_job(
                connection, job_id, status="done", progress=1, message="Done", finished_at=stamp
            )

    def _write_report(self, dataset_id: str, result: Any) -> None:
        quality = result.quality.to_dict() if result.quality is not None else None
        report = {
            "sentence": result.sentence,
            "source_kind": result.source_kind,
            "n_images": result.n_images,
            "n_decoded": result.n_decoded,
            "labels": result.labels,
            "splits": result.splits,
            "rules": [_note(r) for r in result.rules],
            "skipped": [asdict(s) for s in result.skipped[:LIST_LIMIT]],
            "skipped_count": len(result.skipped),
            "conflicts": [asdict(c) for c in result.conflicts[:LIST_LIMIT]],
            "conflicts_count": len(result.conflicts),
            "quality": quality,
        }
        path = self.report_path(dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        partial.write_text(json.dumps(report), encoding="utf-8")
        partial.replace(path)

    # --- FormatHandler (reimport, source check, deletion) --------------------------------------

    def reimport(self, record: DatasetRecord) -> JobRecord:
        path = Path(record.source_path)
        if not path.exists():
            raise IngestionError("The source no longer exists: it can't be reimported.")
        with self._db() as connection:
            repo.update_dataset(connection, record.id, status="importing", error=None)
            self._record_source(connection, record.id, path)
            job = self.datasets.new_job(connection, record.id)
        self.datasets.jobs.submit(lambda: self._run(record.id, job.id))
        return job

    def source_state(self, record: DatasetRecord) -> SourceState:
        source = Path(record.source_path)
        if not source.exists():
            return "missing"
        with self._db() as connection:
            recorded = repo.list_files(connection, record.id)
        for _, size, mtime in recorded[:1]:
            if _stat(source) != (size, mtime):
                return "changed"
        return "ok"

    def deleted(self, record: DatasetRecord) -> None:
        shutil.rmtree(self.images_dir(record.id), ignore_errors=True)

    # --- reading -------------------------------------------------------------------------------

    def _ready(self, dataset_id: str) -> DatasetRecord:
        record = self.datasets.get(dataset_id)
        if record.format != FORMAT:
            raise AppError("This dataset is not an image set.", code="not_images", status_code=409)
        if record.status != "ready":
            raise AppError("This dataset is not ready yet.", code="not_ready", status_code=409)
        return record

    def report(self, dataset_id: str) -> dict[str, Any]:
        self._ready(dataset_id)
        path = self.report_path(dataset_id)
        if not path.is_file():
            raise NotFoundError("Image set report")
        return json.loads(path.read_text(encoding="utf-8"))

    def file(self, dataset_id: str, image_id: str, kind: FileKind) -> tuple[Path, str]:
        """(path, media type) of an image, its thumbnail or its mask."""
        if not IMAGE_ID.match(image_id):
            raise NotFoundError("Image")
        record = self._ready(dataset_id)
        column = _COLUMN[kind]
        connection = duckdb.connect()
        try:
            row = connection.execute(
                f"SELECT {column} FROM read_parquet({lit(record.storage_path)}) "
                f"WHERE image_id = {lit(image_id)} AND {column} IS NOT NULL LIMIT 1"
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise NotFoundError("Image")
        return self._safe_path(dataset_id, str(row[0]))

    def _safe_path(self, dataset_id: str, rel: str) -> tuple[Path, str]:
        """Lexical checks first (a local drive path inside the set's folder), so that a network
        path is refused before any file system call could make Windows contact a server; then
        the resolved path (links followed) must still be inside."""
        root = to_local_path(str(self.images_dir(dataset_id)))
        if re.match(r"^[\\/]{2}|^[A-Za-z]:", rel) or "\x00" in rel:
            raise NotFoundError("Image")
        try:
            candidate = to_local_path(str(root / rel))
        except IngestionError:
            raise NotFoundError("Image") from None
        if not is_within(candidate, [root]):
            raise NotFoundError("Image")
        target = candidate.resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            raise NotFoundError("Image")
        with target.open("rb") as handle:
            head = handle.read(64)
        media = MEDIA_TYPES.get(sniff_image(head) or "")
        if media is None:
            raise AppError("This file is not an image the browser can show.", code="unsupported",
                           status_code=415)  # fmt: skip
        return target, media


__all__ = ["FORMAT", "ImageImportOptions", "ImageSetService"]
