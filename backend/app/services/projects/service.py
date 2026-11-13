"""Projects (087, docs/features/087-projets/spec.md).

Only the root of a family is filed in `project_items`: the original of a cleaned version (065).
A version's project is its original's, computed here (R-PRJ-04); the two triggers of migration
0025 keep the rows right when an original is deleted (R-PRJ-06) or a dataset becomes a version
(R-PRJ-07)."""

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import duckdb
from quantilo.errors import NotFoundError
from quantilo.model.images import ImageSetInput, image_references
from quantilo.model.keys import TableInput

from app.core import db
from app.core.errors import AppError
from app.core.storage import Workspace
from app.schemas.projects import ImageRefOut, ProjectImagesOut, ProjectOut
from app.services.columns.service import ColumnService
from app.services.datasets.repository import now
from app.services.datasets.service import DatasetService

# Every dataset with the project of its root (itself, or the original of a cleaned version).
MEMBERS = (
    "SELECT item.project_id, dataset.id FROM datasets AS dataset "
    "LEFT JOIN clean_versions AS version ON version.dataset_id = dataset.id "
    "JOIN project_items AS item ON item.dataset_id = coalesce(version.parent_id, dataset.id) "
    "ORDER BY dataset.created_at, dataset.id"
)


class ProjectService:
    def __init__(
        self, workspace: Workspace, datasets: DatasetService, columns: ColumnService | None = None
    ) -> None:
        self.workspace = workspace
        self.datasets = datasets
        self.columns = columns  # C1: the columns of the tables read for image references

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def list_all(self) -> list[ProjectOut]:
        """By name, case ignored (R-PRJ-09)."""
        with self._db() as connection:
            rows = connection.execute(
                "SELECT * FROM projects ORDER BY name COLLATE NOCASE, created_at"
            ).fetchall()
            members: dict[str, list[str]] = {}
            for project_id, dataset_id in connection.execute(MEMBERS):
                members.setdefault(project_id, []).append(dataset_id)
        return [
            ProjectOut(
                id=row["id"],
                name=row["name"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                dataset_ids=members.get(row["id"], []),
            )
            for row in rows
        ]

    def get(self, project_id: str) -> ProjectOut:
        found = next((project for project in self.list_all() if project.id == project_id), None)
        if found is None:
            raise NotFoundError("Project")
        return found

    def create(self, name: str) -> ProjectOut:
        project_id = uuid.uuid4().hex[:12]
        stamp = now()
        with self._db() as connection:
            self._ensure_free(connection, name)
            connection.execute(
                "INSERT INTO projects (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (project_id, name, stamp, stamp),
            )
            connection.commit()
        return self.get(project_id)

    def rename(self, project_id: str, name: str) -> ProjectOut:
        self.get(project_id)
        with self._db() as connection:
            self._ensure_free(connection, name, project_id)
            connection.execute(
                "UPDATE projects SET name = ?, updated_at = ? WHERE id = ?",
                (name, now(), project_id),
            )
            connection.commit()
        return self.get(project_id)

    def delete(self, project_id: str) -> None:
        """Its datasets are kept and are in no project any more (R-PRJ-03)."""
        self.get(project_id)
        with self._db() as connection:
            connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            connection.commit()

    def move(
        self, dataset_ids: list[str], project_id: str | None, from_project_id: str | None = None
    ) -> list[ProjectOut]:
        """Put each dataset's family in `project_id`, or in no project (None). `from_project_id`
        (C1): out of that project only, its other projects kept; None: out of every other one."""
        if project_id is not None:
            self.get(project_id)
        if from_project_id is not None:
            self.get(from_project_id)
        for dataset_id in dataset_ids:
            self.datasets.get(dataset_id)  # 404 before anything changes
        stamp = now()
        with self._db() as connection:
            roots = {self._root(connection, dataset_id) for dataset_id in dataset_ids}
            for root in sorted(roots):
                if from_project_id is None:
                    connection.execute("DELETE FROM project_items WHERE dataset_id = ?", (root,))
                else:
                    connection.execute(
                        "DELETE FROM project_items WHERE dataset_id = ? AND project_id = ?",
                        (root, from_project_id),
                    )
                    connection.execute(
                        "DELETE FROM project_items WHERE dataset_id = ? AND project_id = ?",
                        (root, project_id),
                    )
                if project_id is not None:
                    connection.execute(
                        "INSERT INTO project_items (project_id, item_kind, dataset_id, added_at) "
                        "VALUES (?, 'dataset', ?, ?)",
                        (project_id, root, stamp),
                    )
            if project_id is not None:
                connection.execute(
                    "UPDATE projects SET updated_at = ? WHERE id = ?", (stamp, project_id)
                )
            connection.commit()
        return self.list_all()

    def add(self, dataset_ids: list[str], project_id: str) -> list[ProjectOut]:
        """C1: each family also in `project_id` (R-PRJ-02 lifted: an item in several projects)."""
        self.get(project_id)
        for dataset_id in dataset_ids:
            self.datasets.get(dataset_id)
        stamp = now()
        with self._db() as connection:
            roots = {self._root(connection, dataset_id) for dataset_id in dataset_ids}
            for root in sorted(roots):
                connection.execute(
                    "INSERT OR IGNORE INTO project_items (project_id, item_kind, dataset_id, "
                    "added_at) VALUES (?, 'dataset', ?, ?)",
                    (project_id, root, stamp),
                )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?", (stamp, project_id)
            )
            connection.commit()
        return self.list_all()

    def remove(self, dataset_ids: list[str], project_id: str) -> list[ProjectOut]:
        """C1: each family out of `project_id` only; its data and other projects are kept."""
        self.get(project_id)
        with self._db() as connection:
            roots = {self._root(connection, dataset_id) for dataset_id in dataset_ids}
            for root in sorted(roots):
                connection.execute(
                    "DELETE FROM project_items WHERE dataset_id = ? AND project_id = ?",
                    (root, project_id),
                )
            connection.commit()
        return self.list_all()

    def images(self, project_id: str) -> ProjectImagesOut:
        """C1 (R-PRJ-11): the project's tables and image sets, and the columns of its tables
        that name its images (read on the values, nothing changed)."""
        project = self.get(project_id)
        tables: list[str] = []
        sets: list[ImageSetInput] = []
        records = {}
        for dataset_id in project.dataset_ids:
            try:
                record = self.datasets.get(dataset_id)
            except NotFoundError:
                continue
            if record.status != "ready":
                continue
            records[dataset_id] = record
            if record.format == "images":
                sets.append(ImageSetInput(record.id, record.name, record.storage_path))
            else:
                tables.append(record.id)
        references: list[ImageRefOut] = []
        if sets and tables and self.columns is not None:
            connection = duckdb.connect()
            try:
                for dataset_id in tables:
                    record = records[dataset_id]
                    views = self.columns.columns(dataset_id)
                    table = TableInput(dataset_id, record.name, record.storage_path, views)
                    roles = {view.name: view.role for view in views}
                    for ref in image_references(connection, table, sets):
                        references.append(
                            ImageRefOut(
                                dataset_id=dataset_id,
                                column=ref.column,
                                image_set_id=ref.image_set_id,
                                by=ref.by,
                                found=ref.found,
                                distinct=ref.distinct,
                                refused=ref.refused,
                                examples=ref.examples,
                                rule=ref.rule,
                                reason=ref.reason,
                                applied=roles.get(ref.column) == "image_ref",
                            )
                        )
            finally:
                connection.close()
        return ProjectImagesOut(
            tables=tables, image_sets=[s.id for s in sets], references=references
        )

    @staticmethod
    def _root(connection: sqlite3.Connection, dataset_id: str) -> str:
        row = connection.execute(
            "SELECT parent_id FROM clean_versions WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()
        return row["parent_id"] if row else dataset_id

    @staticmethod
    def _ensure_free(connection: sqlite3.Connection, name: str, own_id: str | None = None) -> None:
        row = connection.execute(
            "SELECT id FROM projects WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchone()
        if row is not None and row["id"] != own_id:
            raise AppError(
                f"A project named {name!r} already exists: choose another name.",
                code="project_exists",
                status_code=409,
            )
