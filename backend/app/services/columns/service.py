"""Detect, list and correct the columns of a dataset."""

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import duckdb
from quantilo.catalog import (
    ColumnChange,
    ColumnView,
    apply_change,
    detection_record,
    effective,
    summarize_record,
    to_view,
)
from quantilo.ingestion.sql import lit
from quantilo.typing.derived import derived_columns
from quantilo.typing.engine import DEFAULT_THRESHOLDS, Thresholds, detect_columns
from quantilo.typing.knowledge_base import load
from quantilo.typing.types import AnalysisType, Parser

from app.core import db
from app.core.errors import AppError
from app.core.storage import Workspace
from app.services.columns import repository as repo
from app.services.datasets import repository as datasets_repo
from app.services.datasets.repository import DatasetRecord, now
from app.services.datasets.service import DatasetService, NotFoundError


class ColumnService:
    def __init__(self, workspace: Workspace, datasets: DatasetService) -> None:
        self.workspace = workspace
        self.datasets = datasets
        # Replaced by the settings service at start-up (see app.main).
        self.thresholds: Callable[[], Thresholds] = lambda: DEFAULT_THRESHOLDS

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def ready(self, dataset_id: str) -> DatasetRecord:
        record = self.datasets.get(dataset_id)
        if record.status != "ready":
            raise AppError("This dataset is not ready yet.", code="not_ready", status_code=409)
        return record

    # --- Detection -----------------------------------------------------------------------------

    def detect_storage(self, dataset_id: str, storage_path: str) -> list[ColumnView]:
        """Detect all columns of a stored dataset, keeping the user's corrections."""
        duck = duckdb.connect()
        try:
            detections = detect_columns(
                duck, f"read_parquet({lit(storage_path)})", self.thresholds()
            )
            with self._db() as sqlite_connection:
                previous = {r.name: r for r in repo.list_columns(sqlite_connection, dataset_id)}
                records = []
                for detection in detections:
                    record = detection_record(dataset_id, detection)
                    if old := previous.get(detection.name):
                        for field in repo.USER_FIELDS:
                            setattr(record, field, getattr(old, field))
                    record.summary = summarize_record(storage_path, record, duck)
                    records.append(record)
                repo.replace_columns(sqlite_connection, dataset_id, records)
        finally:
            duck.close()
        return [to_view(record) for record in records]

    def detect(self, dataset_id: str) -> list[ColumnView]:
        return self.detect_storage(dataset_id, self.ready(dataset_id).storage_path)

    def columns(self, dataset_id: str) -> list[ColumnView]:
        dataset = self.ready(dataset_id)
        with self._db() as connection:
            records = repo.list_columns(connection, dataset_id)
        if not records:
            return self.detect_storage(dataset_id, dataset.storage_path)
        return [to_view(record) for record in records]

    def derived(self, dataset_id: str) -> list[ColumnView]:
        """The derived columns the pickers offer (058, R-DER-01 to 06)."""
        dataset = self.ready(dataset_id)
        duck = duckdb.connect()
        try:
            source = f"read_parquet({lit(dataset.storage_path)})"
            return derived_columns(duck, source, self.columns(dataset_id))
        finally:
            duck.close()

    def keep_topics(self, dataset_id: str, column: str, topics: dict | None) -> None:
        """Keep (or forget, `None`) the topics of a text column in its summary (059, R-TOP-06):
        its Topic column becomes a derived column. Cached analyses are dropped (types_version)."""
        self.ready(dataset_id)
        with self._db() as connection:
            record = repo.get_column(connection, dataset_id, column)
            if record is None:
                raise NotFoundError(f"Column {column!r}")
            summary = json.loads(record.summary) if record.summary else {}
            if topics is None:
                summary.pop("topics", None)
            else:
                summary["topics"] = topics
            repo.update_column(connection, dataset_id, column, summary=json.dumps(summary))
            datasets_repo.update_dataset(
                connection,
                dataset_id,
                types_version=self._types_version(connection, dataset_id) + 1,
            )

    # --- Corrections ---------------------------------------------------------------------------

    def update(self, dataset_id: str, changes: list[ColumnChange]) -> list[ColumnView]:
        dataset = self.ready(dataset_id)
        return self.apply(dataset_id, dataset.storage_path, changes)

    def apply(
        self, dataset_id: str, storage_path: str, changes: list[ColumnChange]
    ) -> list[ColumnView]:
        """Save corrections; also usable during an import, before the dataset is ready."""
        valid_roles = {role.id for role in load().roles}
        types_changed = False
        with self._db() as connection:
            for change in changes:
                record = repo.get_column(connection, dataset_id, change.name)
                if record is None:
                    raise NotFoundError(f"Column {change.name!r}")
                before = effective(record)
                apply_change(record, change, valid_roles)
                after = effective(record)
                if before[0] != after[0] or before[2] != after[2]:
                    types_changed = True
                if before != after:
                    record.summary = summarize_record(storage_path, record)
                repo.update_column(
                    connection,
                    dataset_id,
                    record.name,
                    **{field: getattr(record, field) for field in (*repo.USER_FIELDS, "summary")},
                )
            if types_changed:
                datasets_repo.update_dataset(
                    connection,
                    dataset_id,
                    types_version=self._types_version(connection, dataset_id) + 1,
                )
            records = repo.list_columns(connection, dataset_id)
        return [to_view(record) for record in records]

    @staticmethod
    def _types_version(connection: sqlite3.Connection, dataset_id: str) -> int:
        row = connection.execute(
            "SELECT types_version FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
        return int(row[0]) if row else 0

    def conversion(self, dataset_id: str, column: str, target: AnalysisType) -> tuple[Parser, dict]:
        """What the column would look like in another type, without saving anything."""
        dataset = self.ready(dataset_id)
        with self._db() as connection:
            record = repo.get_column(connection, dataset_id, column)
        if record is None:
            raise NotFoundError(f"Column {column!r}")
        apply_change(record, ColumnChange(name=column, type=target), set())
        _type, _role, parser, _order = effective(record)
        return parser, json.loads(summarize_record(dataset.storage_path, record))

    def mark_reviewed(self, dataset_id: str) -> DatasetRecord:
        self.ready(dataset_id)
        with self._db() as connection:
            datasets_repo.update_dataset(connection, dataset_id, types_reviewed_at=now())
        return self.datasets.get(dataset_id)
