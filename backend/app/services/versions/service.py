"""Versions of a dataset and what changed between two of them (031).

At every import the app takes a statistical photo of the new rows (a few kB, kept for every
version); at a reimport it keeps the previous Parquet copy too, unless the user said otherwise
(Settings > Storage), for the row by row comparison. Comparing two datasets imported separately
(October and November, a training file and a file to predict) uses the same code.

Entry points for the features that compare automatically (030 recipes, 037 predictions):
`changes(dataset_id)` (with its previous version) and `changes(dataset_id, vs=other_id)`.

095: `version=n` compares with an earlier import than the previous one (from its photo), and the
rows of an original and its cleaned versions (065) are paired by the cleaning's row map when it
still applies. The link comes from `lineages`, filled by the Clean module (never imported here).
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
from typing import Any, Literal

import duckdb
from quantilo.catalog import ColumnView, to_view
from quantilo.changes.compare import Changes, compare_snapshots
from quantilo.changes.kpis import KpiChange, KpiValue, compare_kpi, compute_value
from quantilo.changes.relations import RelationChange, relation_changes
from quantilo.changes.rows import (
    RowAlignment,
    RowChanges,
    key_candidates,
    row_changes,
    rowmap_matches,
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
class RowLink:
    """A cleaned version's link to its original (065), to pair their rows (095)."""

    parent_id: str
    name: str  # "Cleaned v1"
    rowmap: str | None  # its rowmap.parquet (child row → original row), when it is there
    made_from: str  # the state of the original's rows it was made from
    stale: bool  # the original's rows were replaced since (a reimport)


Lineage = Callable[[str], RowLink | None]  # dataset → its link, when it is a cleaned version


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
    # How the rows were paired: by the cleaning's row map (exact) or on a key column (095).
    alignment: Literal["rowmap", "key"] | None = None
    rowmap: bool = False  # the cleaning's row map can pair these two versions
    alignment_note: str | None = None  # why two related versions are matched on a key


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
        # Filled by the Clean module (065): a cleaned version's link to its original (095).
        self.lineages: list[Lineage] = []
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

    def _sides(
        self, dataset_id: str, vs: str | None, version: int | None = None
    ) -> tuple[VersionSide, VersionSide]:
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
        if version is not None:
            chosen = next((v for v in older if v.version == version), None)
            if chosen is None:
                raise AppError(
                    f"This dataset has no earlier import number {version}.",
                    code="no_such_version",
                    status_code=404,
                )
            last = chosen
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

    def changes(
        self,
        dataset_id: str,
        vs: str | None = None,
        focus: Sequence[str] = (),
        version: int | None = None,
    ) -> Changes:
        """What changed in `dataset_id` since its previous version (or import `version`), or
        since dataset `vs`."""
        before, after = self._sides(dataset_id, vs, version)
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

    def _link(self, dataset_id: str) -> RowLink | None:
        for provider in self.lineages:
            found = provider(dataset_id)
            if found is not None:
                return found
        return None

    def original_of(self, dataset_id: str) -> str | None:
        """The original of a cleaned version (065), what it compares with by default (095)."""
        link = self._link(dataset_id)
        return link.parent_id if link else None

    def _alignment(
        self, dataset_id: str, vs: str | None, before: VersionSide, after: VersionSide
    ) -> tuple[RowAlignment | None, str | None]:
        """The cleaning's row map pairing `vs` (before) and `dataset_id` (after), or why it
        can't (None, None: the two datasets are not an original and its versions)."""
        if vs is None or vs == dataset_id or before.storage_path is None:
            return None, None
        old, new = self._link(vs), self._link(dataset_id)
        if new is not None and new.parent_id == vs:
            sides, maps = [(new, after)], (None, new.rowmap)
        elif old is not None and old.parent_id == dataset_id:
            sides, maps = [(old, before)], (old.rowmap, None)
        elif old is not None and new is not None and old.parent_id == new.parent_id:
            sides, maps = [(old, before), (new, after)], (old.rowmap, new.rowmap)
            if old.made_from != new.made_from:
                return None, (
                    f"{old.name} and {new.name} were made from different imports of the "
                    "original: their row maps don't pair their rows, so rows are matched on a key."
                )
        else:
            return None, None
        for link, side in sides:
            if len(sides) == 1 and link.stale:
                return None, (
                    f"The original was reimported since {link.name} was made: its row map no "
                    "longer pairs the right rows, so rows are matched on a key."
                )
            if link.rowmap is None or not Path(link.rowmap).is_file():
                return None, f"{link.name} has no row map: rows are matched on a key."
            connection = duckdb.connect()
            try:
                matches = rowmap_matches(connection, side.storage_path or "", link.rowmap)
            finally:
                connection.close()
            if not matches:
                return None, (
                    f"{link.name}'s rows no longer match its row map: rows are matched on a key."
                )
        return RowAlignment(*maps), None

    def rows(
        self,
        dataset_id: str,
        vs: str | None = None,
        key: str | None = None,
        version: int | None = None,
    ) -> RowsAnswer:
        """Rows added, removed and modified. Between an original and its cleaned versions, paired
        by the cleaning's row map unless a `key` is asked for (095)."""
        before, after = self._sides(dataset_id, vs, version)
        candidates = key_candidates(before.snapshot, after.snapshot)
        if before.storage_path is None or after.storage_path is None:
            return RowsAnswer(False, NOT_KEPT, candidates)
        alignment, note = self._alignment(dataset_id, vs, before, after)
        paired = alignment is not None
        chosen = key or next(iter(candidates), None)
        if alignment is None or key is not None:
            alignment = None
            if chosen is None:
                reason = (
                    "No column identifies a row in both versions (one value per row, none missing)."
                )
                return RowsAnswer(False, reason, candidates, rowmap=paired, alignment_note=note)
        connection = duckdb.connect()
        try:
            result = row_changes(
                connection,
                before.storage_path,
                before.views,
                after.storage_path,
                after.views,
                chosen or "",
                alignment,
            )
        finally:
            connection.close()
        how = "rowmap" if alignment is not None else "key"
        return RowsAnswer(True, None, candidates, result, how, paired, note)

    def export_rows(
        self, dataset_id: str, vs: str | None, key: str | None, version: int | None = None
    ) -> ExportFile:
        """Every difference as CSV; without `key`, paired as `rows` pairs them by default."""
        before, after = self._sides(dataset_id, vs, version)
        if before.storage_path is None or after.storage_path is None:
            raise AppError(NOT_KEPT, code="previous_not_kept", status_code=409)
        alignment = None
        if key is None:
            alignment, _note = self._alignment(dataset_id, vs, before, after)
            if alignment is None:
                key = next(iter(key_candidates(before.snapshot, after.snapshot)), None)
                if key is None:
                    raise AppError(
                        "No column identifies a row in both versions: pick a key.",
                        code="row_key",
                        status_code=422,
                    )
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
                key or "",
                target,
                alignment,
            )
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            connection.close()
        name = self.datasets.get(dataset_id).name
        return ExportFile(target, f"{name} - differences.csv")

    def kpis(
        self, dataset_id: str, vs: str | None = None, version: int | None = None
    ) -> list[KpiChange]:
        """Each saved KPI on both versions: recomputed when the rows are at hand, otherwise read
        from the previous version's photo (KPIs saved since then cannot be compared)."""
        before, after = self._sides(dataset_id, vs, version)
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

    def relations(
        self, dataset_id: str, vs: str | None = None, version: int | None = None
    ) -> RelationsAnswer:
        before, after = self._sides(dataset_id, vs, version)
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
