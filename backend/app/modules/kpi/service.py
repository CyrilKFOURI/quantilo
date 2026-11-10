"""KPI builder: suggestions, computation through the page filters, and saved KPIs (SQLite)."""

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from quantilo.kpi.definition import Definition
from quantilo.kpi.service import KpiEngine

from app.core import db
from app.core.errors import AppError
from app.core.storage import Workspace
from app.services.columns.service import ColumnService
from app.services.datasets.repository import now
from app.services.datasets.service import NotFoundError


@dataclass
class SavedKpi:
    id: str
    name: str
    definition: Definition
    created_at: str
    updated_at: str


def _saved(row: sqlite3.Row) -> SavedKpi:
    return SavedKpi(
        row["id"],
        row["name"],
        Definition.model_validate(json.loads(row["definition"])),
        row["created_at"],
        row["updated_at"],
    )


def _name(name: str) -> str:
    name = name.strip()
    if not name:
        raise AppError("Give the KPI a name.", code="invalid_name", status_code=422)
    return name


class KpiService(KpiEngine):
    """The engine's suggestions and computation, plus the KPIs saved in the workspace."""

    def __init__(self, columns: ColumnService, workspace: Workspace) -> None:
        super().__init__(columns)
        self.columns: ColumnService = columns
        self.workspace = workspace

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            yield connection
        finally:
            connection.close()

    # --- Saved KPIs ------------------------------------------------------------------------------

    def all(self, dataset_id: str) -> list[SavedKpi]:
        self.columns.datasets.get(dataset_id)
        with self._db() as connection:
            rows = connection.execute(
                "SELECT * FROM kpi_definitions WHERE dataset_id = ? "
                "ORDER BY updated_at DESC, name COLLATE NOCASE",
                (dataset_id,),
            ).fetchall()
        return [_saved(row) for row in rows]

    def get(self, dataset_id: str, kpi_id: str) -> SavedKpi:
        with self._db() as connection:
            row = connection.execute(
                "SELECT * FROM kpi_definitions WHERE id = ? AND dataset_id = ?",
                (kpi_id, dataset_id),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"KPI {kpi_id!r}")
        return _saved(row)

    def create(self, dataset_id: str, name: str, definition: Definition) -> SavedKpi:
        self.columns.datasets.get(dataset_id)
        kpi_id = uuid.uuid4().hex[:12]
        stamp = now()
        with self._db() as connection, connection:
            connection.execute(
                "INSERT INTO kpi_definitions (id, dataset_id, name, definition, created_at, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    kpi_id,
                    dataset_id,
                    _name(name),
                    definition.model_dump_json(by_alias=True),
                    stamp,
                    stamp,
                ),
            )
        return self.get(dataset_id, kpi_id)

    def update(
        self, dataset_id: str, kpi_id: str, name: str | None, definition: Definition | None
    ) -> SavedKpi:
        current = self.get(dataset_id, kpi_id)
        with self._db() as connection, connection:
            connection.execute(
                "UPDATE kpi_definitions SET name = ?, definition = ?, updated_at = ? WHERE id = ?",
                (
                    _name(name) if name is not None else current.name,
                    (definition or current.definition).model_dump_json(by_alias=True),
                    now(),
                    kpi_id,
                ),
            )
        return self.get(dataset_id, kpi_id)

    def duplicate(self, dataset_id: str, kpi_id: str) -> SavedKpi:
        current = self.get(dataset_id, kpi_id)
        return self.create(dataset_id, f"{current.name} (copy)", current.definition)

    def delete(self, dataset_id: str, kpi_id: str) -> None:
        with self._db() as connection, connection:
            deleted = connection.execute(
                "DELETE FROM kpi_definitions WHERE id = ? AND dataset_id = ?",
                (kpi_id, dataset_id),
            ).rowcount
        if not deleted:
            raise NotFoundError(f"KPI {kpi_id!r}")

    # --- What changed between two versions (031) -------------------------------------------------

    def definitions(self, dataset_id: str) -> list[tuple[str, Definition]]:
        return [(kpi.name, kpi.definition) for kpi in self.all(dataset_id)]

    def usage(self, dataset_id: str) -> dict[str, list[str]]:
        """Column → the saved KPIs that read it: such a column disappearing breaks them."""
        used: dict[str, list[str]] = {}
        for kpi in self.all(dataset_id):
            definition = kpi.definition
            columns = [name for m in definition.measures for name in m.columns()]
            columns += [g.column for g in definition.groups]
            if definition.time is not None:
                columns.append(definition.time.column)
            for column in dict.fromkeys(columns):
                used.setdefault(column, []).append(f"KPI “{kpi.name}”")
        return used
