"""Saved filters: named lists of conditions per dataset, stored in SQLite."""

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from quantilo.filters.model import Condition, Conditions
from quantilo.filters.scope import where_for

from app.core import db
from app.core.errors import AppError
from app.core.storage import Workspace
from app.services.columns.service import ColumnService
from app.services.datasets.repository import now
from app.services.datasets.service import NotFoundError


@dataclass
class SavedFilter:
    id: str
    name: str
    conditions: list[Condition]
    created_at: str


def _saved(row: sqlite3.Row) -> SavedFilter:
    conditions = Conditions.validate_python(json.loads(row["conditions"]))
    return SavedFilter(row["id"], row["name"], conditions, row["created_at"])


class SavedFilterService:
    def __init__(self, workspace: Workspace, columns: ColumnService) -> None:
        self.workspace = workspace
        self.columns = columns

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def all(self, dataset_id: str) -> list[SavedFilter]:
        self.columns.datasets.get(dataset_id)
        with self._db() as connection:
            rows = connection.execute(
                "SELECT * FROM saved_filters WHERE dataset_id = ? ORDER BY name COLLATE NOCASE",
                (dataset_id,),
            ).fetchall()
        return [_saved(row) for row in rows]

    def save(self, dataset_id: str, name: str, conditions: list[Condition]) -> SavedFilter:
        """Create, or replace the conditions of the filter with the same name."""
        name = name.strip()
        if not name:
            raise AppError("Give the filter a name.", code="invalid_name", status_code=422)
        if not conditions:
            raise AppError("There is no filter to save.", code="empty_filter", status_code=422)
        where_for(self.columns.columns(dataset_id), conditions)  # validates every condition
        payload = json.dumps([c.model_dump() for c in conditions])
        with self._db() as connection, connection:
            existing = connection.execute(
                "SELECT id FROM saved_filters WHERE dataset_id = ? AND name = ?",
                (dataset_id, name),
            ).fetchone()
            if existing:
                connection.execute(
                    "UPDATE saved_filters SET conditions = ? WHERE id = ?",
                    (payload, existing["id"]),
                )
                filter_id = existing["id"]
            else:
                filter_id = uuid.uuid4().hex[:12]
                connection.execute(
                    "INSERT INTO saved_filters (id, dataset_id, name, conditions, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (filter_id, dataset_id, name, payload, now()),
                )
            row = connection.execute(
                "SELECT * FROM saved_filters WHERE id = ?", (filter_id,)
            ).fetchone()
        return _saved(row)

    def delete(self, dataset_id: str, filter_id: str) -> None:
        with self._db() as connection, connection:
            deleted = connection.execute(
                "DELETE FROM saved_filters WHERE id = ? AND dataset_id = ?",
                (filter_id, dataset_id),
            ).rowcount
        if not deleted:
            raise NotFoundError(f"Saved filter {filter_id!r}")
