"""Versions of a dataset and what changed between two of them (031).

At every import the app takes a statistical photo of the new rows (a few kB, kept for every
version); at a reimport it keeps the previous Parquet copy too, unless the user said otherwise
(Settings > Storage), for the row by row comparison. Comparing two datasets imported separately
(October and November, a training file and a file to predict) uses the same code.

Entry points for the features that compare automatically (030 recipes, 037 predictions):
`changes(dataset_id)` (with its previous version) and `changes(dataset_id, vs=other_id)`.
"""

import logging
import os
import shutil
import sqlite3
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
from quantilo.catalog import ColumnView, to_view
from quantilo.changes.compare import Changes, compare_snapshots
from quantilo.changes.kpis import KpiChange, KpiValue, compare_kpi, compute_value
from quantilo.changes.relations import RelationChange, relation_changes
from quantilo.changes.rows import (
    RowChanges,
    key_candidates,
    row_changes,
    write_differences,
)
from quantilo.changes.snapshot import Snapshot, take_snapshot, views_of
from quantilo.errors import ExplorerError
from quantilo.ingestion.sql import lit
from quantilo.kpi.definition import Definition
from quantilo.rows.service import ExportFile

from app.core import db
from app.core.errors import AppError
from app.core.storage import Workspace
from app.services.columns import repository as columns_repo
from app.services.columns.service import ColumnService
from app.services.datasets.repository import now
from app.services.datasets.service import DatasetService
from app.services.settings.service import KeepPolicy
from app.services.versions import repository as repo
from app.services.versions.repository import VersionRecord

logger = logging.getLogger(__name__)
CACHE_SIZE = 32

Usage = Callable[[str], dict[str, list[str]]]  # dataset → column → what uses it
SavedKpis = Callable[[str], list[tuple[str, Definition]]]  # dataset → (name, definition)


@dataclass
class VersionSide:
    """One side of a comparison: its photo, its rows when they are at hand, its label."""

    snapshot: Snapshot
    storage_path: str | None
    views: list[ColumnView]
    label: str


@dataclass
class VersionInfo:
    version: int
    imported_at: str
    rows: int
    kept: bool  # its rows are at hand (row by row, KPIs, relations)
    kept_bytes: int
    current: bool


@dataclass
class RowsAnswer:
    available: bool
    reason: str | None
    candidates: list[str]
    result: RowChanges | None = None


@dataclass
class RelationsAnswer:
    available: bool
    reason: str | None
    pairs: list[RelationChange]


NOT_KEPT = (
    "The previous version's rows were not kept, only its statistics. To compare row by row, "
    "set Settings > Storage > Keep the previous version to Always or Under the limit, then "
    "reimport."
)


def storage_stamp(path: str) -> str:
    stat = os.stat(path)
    return f"{stat.st_size}:{stat.st_mtime_ns}"


class VersionService:
    def __init__(
        self,
        workspace: Workspace,
        datasets: DatasetService,
        columns: ColumnService,
        keep: Callable[[], KeepPolicy],
        limit_bytes: int,
    ) -> None:
        self.workspace = workspace
        self.datasets = datasets
        self.columns = columns
        self.keep = keep
        self.limit_bytes = limit_bytes
        # Filled by the modules (KPI builder; later recipes and models): what depends on a column
        # (a used column that disappears is Breaking) and the KPIs to compare.
        self.usages: list[Usage] = []
        self.saved_kpis: list[SavedKpis] = []
        self._cache: OrderedDict[tuple[Any, ...], Changes] = OrderedDict()

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def clear(self) -> None:
        self._cache.clear()

    # --- Photos at import ------------------------------------------------------------------------

    def _stored_views(self, dataset_id: str) -> list[ColumnView]:
        """The columns as saved, also while the dataset imports (not ready yet)."""
        with self._db() as connection:
            return [to_view(r) for r in columns_repo.list_columns(connection, dataset_id)]

    def _kpis(self, dataset_id: str) -> list[tuple[str, Definition]]:
        found: list[tuple[str, Definition]] = []
        for provider in self.saved_kpis:
            found += provider(dataset_id)
        return found

    def _photo(
        self, dataset_id: str, storage_path: str, views: list[ColumnView], types_version: int
    ) -> Snapshot:
        connection = duckdb.connect()
        try:
            source = f"read_parquet({lit(storage_path)})"
            snapshot = take_snapshot(connection, source, views, now(), types_version)
            for name, definition in self._kpis(dataset_id):
                try:
                    snapshot.kpis.append(compute_value(connection, source, views, name, definition))
                except (ExplorerError, ValueError, duckdb.Error):
                    continue  # a KPI that no longer computes is reported by the comparison
        finally:
            connection.close()
        return snapshot

    def _insert(self, dataset_id: str, storage_path: str, views: list[ColumnView]) -> VersionRecord:
        types_version = self.datasets.get(dataset_id).types_version
        snapshot = self._photo(dataset_id, storage_path, views, types_version)
        with self._db() as connection:
            last = repo.latest(connection, dataset_id)
            record = VersionRecord(
                dataset_id=dataset_id,
                version=(last.version if last else 0) + 1,
                imported_at=snapshot.taken_at,
                storage_stamp=storage_stamp(storage_path),
                row_count=snapshot.rows,
                snapshot=snapshot,
            )
            repo.insert(connection, record)
        return record

    def record(self, dataset_id: str, storage_path: str) -> None:
        """After an import (the columns are detected): the photo of the new version."""
        with self._db() as connection:
            last = repo.latest(connection, dataset_id)
        if last and last.storage_stamp == storage_stamp(storage_path):
            return
        self._insert(dataset_id, storage_path, self._stored_views(dataset_id))

    def keep_previous(self, dataset_id: str, storage_path: str) -> None:
        """Before a reimport replaces the rows: make sure the current version has its photo,
        then keep its Parquet copy if the setting says so (only the previous one is kept)."""
        with self._db() as connection:
            last = repo.latest(connection, dataset_id)
        if last is None or last.storage_stamp != storage_stamp(storage_path):
            last = self._insert(dataset_id, storage_path, self._stored_views(dataset_id))
        size = os.path.getsize(storage_path)
        policy = self.keep()
        keep = policy == "always" or (policy == "under_limit" and size <= self.limit_bytes)
        folder = self.workspace.dataset_dir(dataset_id) / "versions"
        with self._db() as connection:
            for older in repo.list_versions(connection, dataset_id):
                if older.parquet_path and older.version != last.version:
                    Path(older.parquet_path).unlink(missing_ok=True)
                    repo.set_kept(connection, dataset_id, older.version, None, 0)
            if not keep:
                return
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"v{last.version}.parquet"
            target.unlink(missing_ok=True)
            try:
                # The import writes a new file and renames it over the old one: a hard link
                # keeps the old rows without copying them.
                os.link(storage_path, target)
            except OSError:
                shutil.copyfile(storage_path, target)
            repo.set_kept(connection, dataset_id, last.version, str(target), size)

    # --- Reading ---------------------------------------------------------------------------------

    def _current(self, dataset_id: str) -> VersionRecord:
        """The photo of the rows the dataset holds now, in its current column types."""
        record = self.columns.ready(dataset_id)
        views = self.columns.columns(dataset_id)
        with self._db() as connection:
            last = repo.latest(connection, dataset_id)
        if last is None or last.storage_stamp != storage_stamp(record.storage_path):
            return self._insert(dataset_id, record.storage_path, views)  # imported before 031
        if last.snapshot.types_version != record.types_version:
            last.snapshot = self._photo(
                dataset_id, record.storage_path, views, record.types_version
            )
            with self._db() as connection:
                repo.update_snapshot(connection, dataset_id, last.version, last.snapshot)
        return last

    def versions(self, dataset_id: str) -> list[VersionInfo]:
        current = self._current(dataset_id)
        with self._db() as connection:
            found = repo.list_versions(connection, dataset_id)
        return [
            VersionInfo(
                v.version,
                v.imported_at,
                v.row_count,
                v.version == current.version
                or bool(v.parquet_path and Path(v.parquet_path).is_file()),
                v.kept_bytes,
                v.version == current.version,
            )
            for v in reversed(found)
        ]

    def _sides(self, dataset_id: str, vs: str | None) -> tuple[VersionSide, VersionSide]:
        record = self.datasets.get(dataset_id)
        current = self._current(dataset_id)
        after = VersionSide(
            current.snapshot, record.storage_path, self.columns.columns(dataset_id), record.name
        )
        if vs is not None and vs != dataset_id:
            other = self.datasets.get(vs)
            photo = self._current(vs)
            before = VersionSide(
                photo.snapshot, other.storage_path, self.columns.columns(vs), other.name
            )
            return before, after
        with self._db() as connection:
            older = [
                v for v in repo.list_versions(connection, dataset_id) if v.version < current.version
            ]
        if not older:
            raise AppError(
                "This dataset has no previous version yet: reimport it after its source changes, "
                "or compare it with another dataset from the home page.",
                code="no_previous_version",
                status_code=404,
            )
        last = older[-1]
        kept = (
            last.parquet_path if last.parquet_path and Path(last.parquet_path).is_file() else None
        )
        before = VersionSide(last.snapshot, kept, views_of(last.snapshot), last.imported_at[:10])
        after.label = "this version"
        return before, after

    def _used(self, dataset_id: str) -> dict[str, list[str]]:
        used: dict[str, list[str]] = {}
        for usage in self.usages:
            for column, users in usage(dataset_id).items():
                used.setdefault(column, []).extend(users)
        return used

    def changes(self, dataset_id: str, vs: str | None = None, focus: Sequence[str] = ()) -> Changes:
        """What changed in `dataset_id` since its previous version, or since dataset `vs`."""
        before, after = self._sides(dataset_id, vs)
        used = self._used(dataset_id)
        key = (
            dataset_id,
            vs,
            before.snapshot.taken_at,
            after.snapshot.taken_at,
            after.snapshot.types_version,
            before.label,
            tuple(sorted((k, tuple(v)) for k, v in used.items())),
            tuple(focus),
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        result = compare_snapshots(
            before.snapshot,
            after.snapshot,
            before_label=before.label,
            after_label=after.label,
            used=used,
            focus=focus,
        )
        self._cache[key] = result
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    def rows(self, dataset_id: str, vs: str | None = None, key: str | None = None) -> RowsAnswer:
        before, after = self._sides(dataset_id, vs)
        candidates = key_candidates(before.snapshot, after.snapshot)
        if before.storage_path is None or after.storage_path is None:
            return RowsAnswer(False, NOT_KEPT, candidates)
        chosen = key or next(iter(candidates), None)
        if chosen is None:
            reason = (
                "No column identifies a row in both versions (one value per row, none missing)."
            )
            return RowsAnswer(False, reason, candidates)
        connection = duckdb.connect()
        try:
            result = row_changes(
                connection,
                before.storage_path,
                before.views,
                after.storage_path,
                after.views,
                chosen,
            )
        finally:
            connection.close()
        return RowsAnswer(True, None, candidates, result)

    def export_rows(self, dataset_id: str, vs: str | None, key: str) -> ExportFile:
        before, after = self._sides(dataset_id, vs)
        if before.storage_path is None or after.storage_path is None:
            raise AppError(NOT_KEPT, code="previous_not_kept", status_code=409)
        folder = Path(tempfile.mkdtemp(prefix="changes-"))
        target = folder / "differences.csv"
        connection = duckdb.connect()
        try:
            write_differences(
                connection,
                before.storage_path,
                before.views,
                after.storage_path,
                after.views,
                key,
                target,
            )
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            connection.close()
        name = self.datasets.get(dataset_id).name
        return ExportFile(target, f"{name} - differences.csv")

    def kpis(self, dataset_id: str, vs: str | None = None) -> list[KpiChange]:
        """Each saved KPI on both versions: recomputed when the rows are at hand, otherwise read
        from the previous version's photo (KPIs saved since then cannot be compared)."""
        before, after = self._sides(dataset_id, vs)
        found: list[KpiChange] = []
        for name, definition in self._kpis(dataset_id):
            values: list[KpiValue | None] = []
            problems: list[str | None] = []
            for side in (before, after):
                if side.storage_path is None:
                    values.append(next((k for k in side.snapshot.kpis if k.name == name), None))
                    problems.append(
                        None
                        if values[-1]
                        else "Not in the previous version's photo, and its rows were not kept."
                    )
                    continue
                connection = duckdb.connect()
                try:
                    source = f"read_parquet({lit(side.storage_path)})"
                    values.append(compute_value(connection, source, side.views, name, definition))
                    problems.append(None)
                except (ExplorerError, ValueError, duckdb.Error) as error:
                    values.append(None)
                    problems.append(getattr(error, "message", None) or str(error))
                finally:
                    connection.close()
            problem = problems[1] or problems[0]
            found.append(compare_kpi(name, values[0], values[1], problem, values[1] is None))
        return found

    def relations(self, dataset_id: str, vs: str | None = None) -> RelationsAnswer:
        before, after = self._sides(dataset_id, vs)
        if before.storage_path is None or after.storage_path is None:
            return RelationsAnswer(False, NOT_KEPT, [])
        connection = duckdb.connect()
        try:
            pairs = relation_changes(
                connection, before.storage_path, before.views, after.storage_path, after.views
            )
        finally:
            connection.close()
        return RelationsAnswer(True, None, pairs)
