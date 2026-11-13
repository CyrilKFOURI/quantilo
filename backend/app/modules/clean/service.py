"""Clean (065): suggestions, the draft, the preview, and cleaned versions saved as datasets;
the Advanced layer on demand (097), the Expert checks as cancellable jobs and the replay of a
cleaning on another file (101).

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
from quantilo.cleaning import checks as ck
from quantilo.cleaning import dismiss as dm
from quantilo.cleaning import expert
from quantilo.cleaning import model as m
from quantilo.cleaning.apply import CleanCancelled, apply_steps
from quantilo.cleaning.compile import compile_steps
from quantilo.cleaning.duplicates import DuplicateGroups, duplicate_groups
from quantilo.cleaning.entities import EntityClusters, entity_clusters
from quantilo.cleaning.export import ExportInfo, to_python, to_sql
from quantilo.cleaning.facade import corrections, type_changes
from quantilo.cleaning.facet import Facet, FacetRequest, facet
from quantilo.cleaning.journal import (
    Actor,
    CleanJournal,
    JournalEntry,
    Known,
    counts,
    diff,
    journal_steps,
)
from quantilo.cleaning.journal import to_html as journal_html
from quantilo.cleaning.journal import to_markdown as journal_markdown
from quantilo.cleaning.preview import CleanPreview, preview
from quantilo.cleaning.quality import ColumnQuality
from quantilo.cleaning.replay import CleanRecipe, ReplayReport, recipe_of, replay
from quantilo.cleaning.report import CleanReport
from quantilo.cleaning.suggest import (
    RULES_VERSION,
    default_steps,
    suggest_advanced,
    suggest_and_measure,
)
from quantilo.cleaning.suggestion import CleanSuggestion
from quantilo.ingestion.options import ImportOptions
from quantilo.ingestion.sources import list_source_files
from quantilo.recipes.script import file_stem

from app.core import db
from app.core.errors import AppError
from app.core.storage import Workspace
from app.modules.clean import checks as checks_repo
from app.modules.clean import dismissed as dismissed_repo
from app.modules.clean import journal as journal_repo
from app.modules.clean.checks import CheckRun, CleanChecks, Trigger
from app.modules.clean.dismissed import CleanDismissal, Dismissed, HiddenSuggestion
from app.services.columns import repository as columns_repo
from app.services.columns.service import ColumnService
from app.services.datasets import repository as datasets_repo
from app.services.datasets.repository import DatasetRecord
from app.services.versions.service import RowLink

logger = logging.getLogger(__name__)

SETTINGS_KEY = "cleaning"
KEPT_JOBS = 50
READY_WAIT_SECONDS = 600  # the automatic mode waits for the import to finish
# N03: `apply` (the default) applies the reliable fixes in the draft; `auto` also saves them as a
# cleaned version after each new import (the user's explicit choice, never the default)
Mode = Literal["apply", "suggest", "off", "auto"]
APPLYING: tuple[Mode, ...] = ("apply", "auto")
# In the browser build (Pyodide) there are no threads: a save cannot be interrupted midway, and
# the automatic mode (which waits for the import to finish) is not offered.
THREADS = sys.platform != "emscripten"


class CleanSettings(BaseModel):
    """Settings > Cleaning. `apply` (default since N03, the owner's decision of 2026-11-12): when
    Clean opens, the reliable fixes (the catalogue's `auto` suggestions, right on 99 % or more of
    the cases measured) are ticked in the draft, shown and undoable; nothing is saved until the
    user saves. `suggest`: suggestions computed, none ticked. `off`: nothing computed until asked.
    `auto`: as `apply`, and a cleaned version of those fixes made after each new import when one
    changes a cell."""

    mode: Mode = "apply"


class CleanSuggestions(BaseModel):
    dataset_id: str
    rows: int
    seconds: float
    suggestions: list[CleanSuggestion]
    # N05: those "Don't suggest again" hides (never ticked, never applied), and the dismissals that
    # apply here (this family's, then every dataset's), listed under Dismissed
    hidden: list[HiddenSuggestion] = Field(default_factory=list)
    dismissals: list[CleanDismissal] = Field(default_factory=list)
    # N37: each column's quality, measured with the Basic profile (Basic only; None: not measured,
    # an Advanced layer or a cache written before N37)
    quality: list[ColumnQuality] | None = None


class CleanDraft(BaseModel):
    dataset_id: str
    steps: list[m.Step]
    automatic: bool  # no draft saved: the mode's default steps (the reliable fixes, or none)
    updated_at: str | None = None


class CleanPreviewRequest(BaseModel):
    steps: list[m.Step] = Field(default_factory=list, max_length=m.MAX_STEPS)
    step_id: str | None = Field(default=None, pattern=r"^s[0-9]{1,4}$")  # None: every step


class DuplicatesRequest(BaseModel):
    """N35: the groups of a removal of duplicates, as the draft (`steps`) leaves the rows before
    it: the first `limit` groups after `offset`."""

    steps: list[m.Step] = Field(default_factory=list, max_length=m.MAX_STEPS)
    step_id: str = Field(pattern=r"^s[0-9]{1,4}$")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=200)


class EntitiesRequest(BaseModel):
    """N36: the clusters of an entity number (A-20), as the draft (`steps`) leaves the rows
    before it: the first `limit` clusters after `offset`."""

    steps: list[m.Step] = Field(default_factory=list, max_length=m.MAX_STEPS)
    step_id: str = Field(pattern=r"^s[0-9]{1,4}$")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=200)


class CleanFacetRequest(FacetRequest):
    """N40: a facet of a column of the draft (`steps`, every one applied)."""

    steps: list[m.Step] = Field(default_factory=list, max_length=m.MAX_STEPS)


class SaveRequest(BaseModel):
    steps: list[m.Step] = Field(min_length=1, max_length=m.MAX_STEPS)


class ReplayRequest(BaseModel):
    """101: the cleaning to replay, a cleaned version's steps (`version_id`) or a recipe file."""

    version_id: str | None = Field(default=None, max_length=100)
    recipe: CleanRecipe | None = None
    accept_proposals: bool = False  # C-40: use the columns paired by a close name


@dataclass
class ExpertJob:
    """101: one run of a *More checks* method (C-29), cancellable (C-31)."""

    id: str
    status: Literal["running", "done", "failed", "cancelled"]
    stage: str
    dataset_id: str
    method: expert.MethodId
    error: str | None = None
    result: expert.ExpertResult | None = None


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
    out_of_date: bool  # the dataset it was made from was reimported or retyped since (S7)
    steps: list[m.Step] = field(default_factory=list)
    report: CleanReport | None = None
    # 096: the cleaned version it was made from (None: the original, or that version is gone)
    source_id: str | None = None
    source_seq: int | None = None
    checks: CheckRun | None = None  # N31: the last run of the family's checks on this version


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
    source_id: str | None = None  # 096: made from this cleaned version (parent_id: the original)
    source_seq: int | None = None


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
    source_id: str | None = None  # 096: a cleaned version cleaned again
    source_seq: int | None = None
    actor: Actor = "user"  # N46: who saved it
    detail: str | None = None  # N46: what the journal adds (a rebuild)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stamp(record: DatasetRecord) -> str:
    return f"{record.imported_at}|{record.size_bytes}|{record.types_version}"


def _rows_state(stamp: str) -> str:
    """The part of a stamp that changes with the rows (a reimport), not with a type correction."""
    return "|".join(stamp.split("|")[:2])


SOURCE = "source"  # 096: the report's key for the cleaned version a version was made from


def _source(report: dict[str, Any], stamps: dict[str, str]) -> tuple[str | None, int | None]:
    """The cleaned version a version was made from, popped from its report: its id while it
    exists, and its number."""
    source = report.pop(SOURCE, None)
    if not source:
        return None, None
    found = source["dataset_id"] if source["dataset_id"] in stamps else None
    return found, source.get("seq")


def _out_of_date(row: sqlite3.Row, report: dict[str, Any], stamps: dict[str, str]) -> bool:
    """S7: the dataset it was made from changed since; one deleted since compares nothing."""
    source = report.get(SOURCE)
    made_from = source["dataset_id"] if source else row["parent_id"]
    if made_from not in stamps:
        return False
    return stamps[made_from] != row["parent_stamp"]


def _data_stamp(storage_path: str, types_version: int | None) -> str:
    """N31: the stored data and its types, as a check run sees them (known during the import)."""
    try:
        stat = os.stat(storage_path)
    except OSError:
        return f"missing|{types_version}"
    return f"{stat.st_size}:{stat.st_mtime_ns}|{types_version}"


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
        # 101: the Expert checks run apart, so a long one never holds a save back
        self._expert = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clean-expert")
        self._expert_jobs: OrderedDict[str, ExpertJob] = OrderedDict()

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
        """The Basic suggestions shown, those dismissed apart (N05)."""
        return self._shown(dataset_id, self._basic(dataset_id))

    def _shown(self, dataset_id: str, found: CleanSuggestions) -> CleanSuggestions:
        """N05: the suggestions a dismissal hides moved to `hidden`."""
        dismissals = self.dismissals(dataset_id)
        if not dismissals:
            return found
        shown, hidden = dismissed_repo.split(found.suggestions, dismissals)
        return found.model_copy(
            update={"suggestions": shown, "hidden": hidden, "dismissals": dismissals}
        )

    def _basic(self, dataset_id: str) -> CleanSuggestions:
        """Cached per import and types version (`clean/suggestions.json`), every suggestion (the
        dismissed ones too: the step ids of Advanced follow them)."""
        record = self.columns.ready(dataset_id)
        key = [RULES_VERSION, record.imported_at, record.types_version]
        cache = self._folder(dataset_id) / "suggestions.json"
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            cached = CleanSuggestions.model_validate(saved["value"])
            # N37: a cache written before the quality was measured is made again (same suggestions)
            if saved.get("key") == key and cached.quality is not None:
                return cached
        except (OSError, ValueError):
            pass
        started = time.monotonic()
        views = self.columns.columns(dataset_id)
        connection = duckdb.connect()
        try:
            found, quality = suggest_and_measure(connection, record.storage_path, views)
        finally:
            connection.close()
        result = CleanSuggestions(
            dataset_id=dataset_id,
            rows=record.row_count or 0,
            seconds=round(time.monotonic() - started, 2),
            suggestions=found,
            quality=quality,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps({"key": key, "value": result.model_dump(mode="json")}), encoding="utf-8"
        )
        return result

    def advanced(self, dataset_id: str) -> CleanSuggestions:
        """097: the Advanced layer, computed when it is opened (C-16) and cached like Basic; those
        dismissed apart (N05)."""
        return self._shown(dataset_id, self._advanced(dataset_id))

    def _advanced(self, dataset_id: str) -> CleanSuggestions:
        record = self.columns.ready(dataset_id)
        key = [RULES_VERSION, record.imported_at, record.types_version]
        cache = self._folder(dataset_id) / "suggestions-advanced.json"
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("key") == key:
                return CleanSuggestions.model_validate(saved["value"])
        except (OSError, ValueError):
            pass
        basic = self._basic(dataset_id).suggestions
        started = time.monotonic()
        views = self.columns.columns(dataset_id)
        connection = duckdb.connect()
        try:
            found = suggest_advanced(connection, record.storage_path, views, basic)
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

    def draft(self, dataset_id: str) -> CleanDraft:
        """The saved draft, or else the mode's default (N03): the reliable fixes applied (`apply`,
        `auto`), or nothing ticked (`suggest`, `off`). Never saved by itself: the user's first
        change saves the draft, the user's Save makes a version."""
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
        if self.settings().mode not in APPLYING:  # suggestions only: nothing ticked for the user
            return CleanDraft(dataset_id=dataset_id, steps=[], automatic=True)
        steps = default_steps(self.suggestions(dataset_id).suggestions, auto_only=True)
        return CleanDraft(dataset_id=dataset_id, steps=steps, automatic=True)

    def save_draft(self, dataset_id: str, steps: list[Any]) -> CleanDraft:
        self.columns.ready(dataset_id)
        body = m.Steps(steps=steps).to_json()  # two steps with one id: refused
        stamp = _now()
        entries = self._draft_changes(dataset_id, steps, stamp)  # N46
        with self._db() as connection, connection:
            journal_repo.add(connection, dataset_id, entries)
            connection.execute(
                "INSERT INTO clean_drafts (dataset_id, steps, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT (dataset_id) DO UPDATE SET steps = excluded.steps, "
                "updated_at = excluded.updated_at",
                (dataset_id, body, stamp),
            )
        return CleanDraft(dataset_id=dataset_id, steps=steps, automatic=False, updated_at=stamp)

    def reset_draft(self, dataset_id: str) -> CleanDraft:
        """Back to the mode's default ("Reset"): the reliable fixes, or nothing ticked."""
        with self._db() as connection, connection:
            gone = connection.execute(
                "DELETE FROM clean_drafts WHERE dataset_id = ?", (dataset_id,)
            ).rowcount
            if gone:  # N46
                journal_repo.add(connection, dataset_id, [JournalEntry(_now(), "user", "reset")])
        return self.draft(dataset_id)

    # --- Journal (N46) ---------------------------------------------------------------------------

    def _known(self, dataset_id: str) -> dict[str, Known]:
        """What the suggestions found here say of a step made from one (title, cells, an example):
        the cached layers only, never a computation for the journal."""
        found: list[CleanSuggestion] = []
        for name in ("suggestions.json", "suggestions-advanced.json"):
            try:
                layer = self._cached(dataset_id, name)
            except AppError:
                return {}
            found += layer.suggestions if layer else []
        return {
            s.id: Known(
                s.title,
                s.cells,
                s.rows,
                (s.examples[0].before, s.examples[0].after) if s.examples else None,
            )
            for s in found
        }

    def _draft_changes(self, dataset_id: str, steps: list[Any], at: str) -> list[JournalEntry]:
        """The draft's change, step by step (N46). The first change of a draft Clean opened with
        the reliable fixes applied records them first, as applied automatically."""
        with self._db() as connection:
            row = connection.execute(
                "SELECT steps FROM clean_drafts WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
        known = self._known(dataset_id)
        entries: list[JournalEntry] = []
        if row is not None:
            try:
                before = m.parse_steps(row["steps"]).steps
            except m.CleanError:
                before = []
        else:
            before = self.draft(dataset_id).steps
            applied = [s for s in before if s.origin == "auto"]
            entries += diff([], applied, at, "auto", known)
            for entry in entries:
                entry.action = "applied"
        return entries + diff(before, steps, at, "user", known)

    def draft_journal(self, dataset_id: str) -> list[JournalEntry]:
        """The changes of this dataset's draft since its last saved version."""
        self.columns.ready(dataset_id)
        with self._db() as connection:
            return journal_repo.of_draft(connection, dataset_id)

    def _log(self, dataset_id: str, entry: JournalEntry) -> None:
        with self._db() as connection, connection:
            journal_repo.add(connection, dataset_id, [entry])

    def journal(self, version_id: str) -> CleanJournal:
        """A version's journal: its steps with where each came from, its report, and the
        changes of the draft that led to it."""
        with self._db() as connection:
            row = connection.execute(
                "SELECT * FROM clean_versions WHERE dataset_id = ?", (version_id,)
            ).fetchone()
            if row is None:
                raise AppError(
                    "This dataset is not a cleaned version.", code="not_a_version", status_code=404
                )
            entries = journal_repo.of_version(connection, version_id)
        report_raw = json.loads(row["report"])
        source = (report_raw.pop(SOURCE, None) or {}).get("dataset_id")
        made_from = source or row["parent_id"]
        steps = m.parse_steps(row["steps"]).steps
        report = CleanReport.from_dict(report_raw)
        known = self._known(made_from)
        names = {r.id: r.name for r in self.datasets.list_all()}
        found = journal_steps(steps, report, known)
        return CleanJournal(
            name=names.get(version_id, f"Cleaned v{row['seq']}"),
            dataset_id=version_id,
            made_from=names.get(made_from) or str(made_from),
            made_at=row["created_at"],
            origin=row["origin"],
            steps=found,
            report=report,
            entries=entries,
            applied=counts(found),
        )

    def journal_file(self, version_id: str, kind: Literal["markdown", "html"]) -> tuple[str, str]:
        found = self.journal(version_id)
        stem = file_stem(found.name)
        if kind == "html":
            return f"{stem}.cleaning-log.html", journal_html(found)
        return f"{stem}.cleaning-log.md", journal_markdown(found)

    def sources(self, steps: list[Any]) -> dict[str, str]:
        """C-57, C-58: the Parquet of the other datasets the steps append or merge, when they are
        ready; one that is gone is left out and its step says so (`problems`)."""
        found: dict[str, str] = {}
        for step in steps:
            if step.op not in m.OTHER_DATASET_OPS or step.dataset in found:
                continue
            try:
                record = self.datasets.get(step.dataset)
            except AppError:
                continue
            if record.status == "ready" and record.storage_path:
                found[step.dataset] = record.storage_path
        return found

    def preview(self, dataset_id: str, request: CleanPreviewRequest) -> CleanPreview:
        record = self.columns.ready(dataset_id)
        m.Steps(steps=request.steps)
        connection = duckdb.connect()
        try:
            compiled = compile_steps(
                connection, record.storage_path, request.steps, sources=self.sources(request.steps)
            )
            return preview(connection, compiled, request.step_id)
        finally:
            connection.close()

    def duplicates(self, dataset_id: str, request: DuplicatesRequest) -> DuplicateGroups:
        """N35: the groups of one removal of duplicates, side by side, for the review by group."""
        record = self.columns.ready(dataset_id)
        m.Steps(steps=request.steps)
        connection = duckdb.connect()
        try:
            compiled = compile_steps(
                connection, record.storage_path, request.steps, sources=self.sources(request.steps)
            )
            return duplicate_groups(
                connection, compiled, request.step_id, request.offset, request.limit
            )
        finally:
            connection.close()

    def entities(self, dataset_id: str, request: EntitiesRequest) -> EntityClusters:
        """N36: the clusters of one entity number, side by side, for the review by cluster."""
        record = self.columns.ready(dataset_id)
        m.Steps(steps=request.steps)
        connection = duckdb.connect()
        try:
            compiled = compile_steps(
                connection, record.storage_path, request.steps, sources=self.sources(request.steps)
            )
            return entity_clusters(
                connection, compiled, request.step_id, request.offset, request.limit
            )
        finally:
            connection.close()

    def facet(self, dataset_id: str, request: CleanFacetRequest) -> Facet:
        """N40: a column of the draft faceted (values, histogram, blank, unreadable) and the rows
        a selection picks; nothing is written."""
        record = self.columns.ready(dataset_id)
        m.Steps(steps=request.steps)
        views = {view.name: view for view in self.columns.columns(dataset_id)}
        connection = duckdb.connect()
        try:
            compiled = compile_steps(
                connection, record.storage_path, request.steps, sources=self.sources(request.steps)
            )
            return facet(connection, compiled, request, views.get(request.column))
        finally:
            connection.close()

    # --- Don't suggest again (N05) ---------------------------------------------------------------

    def root(self, dataset_id: str) -> str:
        """The original of a family: a cleaned version shares its dismissals and checks."""
        return self.parent_of(dataset_id) or dataset_id

    def dismissals(self, dataset_id: str) -> list[CleanDismissal]:
        with self._db() as connection:
            return dismissed_repo.applying(connection, self.root(dataset_id))

    def dismiss(self, dataset_id: str, body: dm.Dismissal) -> Dismissed:
        """Hide the suggestions `body` matches, here (and in every dataset for a column name or a
        rule everywhere); says which suggestions of this dataset it hides now."""
        self.columns.ready(dataset_id)
        with self._db() as connection, connection:
            dismissal = dismissed_repo.add(connection, self.root(dataset_id), body)
            entry = JournalEntry(
                _now(), "user", "dismissed", rule=body.rule, detail=dismissal.reason
            )
            entry.columns = [body.column] if body.column and body.scope != "rule" else []
            journal_repo.add(connection, dataset_id, [entry])  # N46
        found = list(self._basic(dataset_id).suggestions)
        advanced = self._cached(dataset_id, "suggestions-advanced.json")
        if advanced is not None:
            found += advanced.suggestions
        model = dismissal.model()
        return Dismissed(dismissal=dismissal, hidden=[s.id for s in found if dm.matches(model, s)])

    def restore(self, dismissal_id: str) -> CleanDismissal:
        """Show again what a dismissal hid (nothing is ticked again)."""
        with self._db() as connection, connection:
            gone = dismissed_repo.remove(connection, dismissal_id)
            if gone.dataset_id:  # N46: in the journal of the dataset it belonged to
                entry = JournalEntry(_now(), "user", "restored", rule=gone.rule, detail=gone.reason)
                journal_repo.add(connection, gone.dataset_id, [entry])
            return gone

    def _cached(self, dataset_id: str, name: str) -> CleanSuggestions | None:
        """A layer's suggestions when they are in the cache for this import, else None."""
        record = self.columns.ready(dataset_id)
        key = [RULES_VERSION, record.imported_at, record.types_version]
        try:
            saved = json.loads((self._folder(dataset_id) / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return CleanSuggestions.model_validate(saved["value"]) if saved.get("key") == key else None

    def _here(self, dataset_id: str) -> list[dm.Dismissal]:
        """The dismissals a recipe of this family carries (one suggestion, a rule here)."""
        return [d.model() for d in self.dismissals(dataset_id) if d.dataset_id is not None]

    # --- Declared checks (N31) ------------------------------------------------------------------

    def checks(self, dataset_id: str) -> CleanChecks:
        """The family's checks and what they find on this dataset: the last run when the data and
        the checks are those it ran on, else a run now (`opened`)."""
        record = self.columns.ready(dataset_id)
        root = self.root(dataset_id)
        with self._db() as connection:
            declared = checks_repo.declared(connection, root)
            rows = checks_repo.runs(connection, dataset_id)
        found = CleanChecks(
            dataset_id, root, declared, history=[checks_repo.summary(r) for r in rows[:10]]
        )
        if not declared:
            return found
        latest = rows[0] if rows else None
        stamp = _data_stamp(record.storage_path, record.types_version)
        if (
            latest is not None
            and latest["stamp"] == stamp
            and latest["checks_hash"] == checks_repo.digest(declared)
        ):
            found.report = checks_repo.report_of(latest["report"])
            found.ran_at, found.trigger = latest["ran_at"], latest["trigger"]
            return found
        views = self.columns.columns(dataset_id)
        self._verify(dataset_id, record.storage_path, record.types_version, views, "opened")
        return self.checks(dataset_id)

    def save_checks(self, dataset_id: str, checks: list[ck.CleanCheck]) -> CleanChecks:
        """Declare the family's checks (kept on its original) and run them here (`declared`)."""
        record = self.columns.ready(dataset_id)
        ck.Checks(checks=checks)  # two checks with one id: refused
        with self._db() as connection, connection:
            before = checks_repo.declared(connection, self.root(dataset_id))
            checks_repo.store(connection, self.root(dataset_id), checks)
            detail = f"{len(before)} → {len(checks)} checks: " + "; ".join(
                ck.describe(c) for c in checks[:5]
            )
            journal_repo.add(
                connection, dataset_id, [JournalEntry(_now(), "user", "checks", detail=detail)]
            )
        if checks:
            views = self.columns.columns(dataset_id)
            self._verify(dataset_id, record.storage_path, record.types_version, views, "declared")
        return self.checks(dataset_id)

    def check_rows(self, dataset_id: str, check_id: str) -> ck.FailingRows:
        """The first rows failing one check, its columns first (See the rows)."""
        record = self.columns.ready(dataset_id)
        with self._db() as connection:
            declared = checks_repo.declared(connection, self.root(dataset_id))
        check = next((c for c in declared if c.id == check_id), None)
        if check is None:
            raise AppError("This check no longer exists.", code="unknown_check", status_code=404)
        connection = duckdb.connect()
        try:
            return ck.failing_rows(
                connection, record.storage_path, self.columns.columns(dataset_id), check
            )
        except m.CleanError as error:
            raise AppError(error.message, code="check_cannot_run", status_code=409) from None
        finally:
            connection.close()

    def _verify(
        self,
        dataset_id: str,
        storage_path: str,
        types_version: int | None,
        views: list[Any],
        trigger: Trigger,
    ) -> ck.CheckReport | None:
        """Run the family's checks on this dataset and keep the run."""
        with self._db() as connection:
            declared = checks_repo.declared(connection, self.root(dataset_id))
        if not declared:
            return None
        connection = duckdb.connect()
        try:
            report = ck.run_checks(connection, storage_path, views, declared)
        finally:
            connection.close()
        stamp = _data_stamp(storage_path, types_version)
        with self._db() as connection, connection:
            checks_repo.record(connection, dataset_id, stamp, declared, trigger, report)
        return report

    def _replay_checks(self, dataset_id: str, storage_path: str, trigger: Trigger) -> None:
        """N31: the checks run again after a reimport or on a new version (never fatal)."""
        try:
            with self._db() as connection:
                views = [to_view(r) for r in columns_repo.list_columns(connection, dataset_id)]
            record = self.datasets.get(dataset_id)
            self._verify(dataset_id, storage_path, record.types_version, views, trigger)
        except Exception:
            logger.exception("Checks could not run on dataset %s", dataset_id)

    # --- Expert checks (101) ---------------------------------------------------------------------

    def expert_methods(self, dataset_id: str) -> list[expert.ExpertMethod]:
        """The methods of *More checks*, read from the columns only: no row is read (C-29)."""
        self.columns.ready(dataset_id)
        return expert.methods(self.columns.columns(dataset_id))

    def run_expert(
        self, dataset_id: str, method: expert.MethodId, options: expert.ExpertOptions
    ) -> ExpertJob:
        self.columns.ready(dataset_id)
        job = ExpertJob(uuid.uuid4().hex[:12], "running", "queued", dataset_id, method)
        with self._lock:
            self._expert_jobs[job.id] = job
            while len(self._expert_jobs) > KEPT_JOBS:
                self._expert_jobs.popitem(last=False)
        self._expert.submit(self._run_expert, job, options)
        return job

    def _run_expert(self, job: ExpertJob, options: expert.ExpertOptions) -> None:
        record = self.columns.ready(job.dataset_id)
        views = self.columns.columns(job.dataset_id)
        job.stage = "running"
        connection = duckdb.connect()
        if THREADS:
            threading.Thread(target=self._watch, args=(job, connection), daemon=True).start()
        try:
            job.result = expert.run(
                connection,
                record.storage_path,
                views,
                job.method,
                options,
                lambda: job.id in self._stops,
            )
            job.status = "done"
        except (CleanCancelled, duckdb.InterruptException):
            job.status, job.error = "cancelled", "Stopped: nothing was found."
        except AppError as error:
            job.status, job.error = "failed", error.message
        except Exception as error:
            logger.exception("Expert check %s failed for dataset %s", job.method, job.dataset_id)
            job.status, job.error = "failed", f"Failed: {error}"
        finally:
            job.stage = job.status
            connection.close()
            self._stops.discard(job.id)

    def expert_job(self, job_id: str) -> ExpertJob:
        job = self._expert_jobs.get(job_id)
        if job is None:
            raise AppError("This check is no longer followed.", code="unknown_job", status_code=404)
        return job

    def cancel_expert(self, job_id: str) -> ExpertJob:
        job = self.expert_job(job_id)
        if job.status == "running":
            self._stops.add(job_id)
        return job

    # --- Replay (101, M5) -------------------------------------------------------------------------

    def recipe(self, version_id: str) -> CleanRecipe:
        """A cleaned version's steps as a recipe (C-39), read on the dataset they ran on."""
        with self._db() as connection:
            row = connection.execute(
                "SELECT parent_id, seq, steps, report FROM clean_versions WHERE dataset_id = ?",
                (version_id,),
            ).fetchone()
        if row is None:
            raise AppError(
                "This dataset is not a cleaned version.", code="not_a_version", status_code=404
            )
        source = (json.loads(row["report"]).get(SOURCE) or {}).get("dataset_id")
        made_from = source or row["parent_id"]
        try:
            record = self.columns.ready(made_from)
        except AppError:
            raise AppError(
                "The dataset this version was made from is gone: its recipe cannot be read.",
                code="source_gone",
                status_code=409,
            ) from None
        steps = m.parse_steps(row["steps"]).steps
        original = self.datasets.get(row["parent_id"])
        connection = duckdb.connect()
        try:
            found = recipe_of(
                connection,
                record.storage_path,
                self.columns.columns(made_from),
                steps,
                f"{original.name} · Cleaned v{row['seq']}",
                record.name,
            )
        finally:
            connection.close()
        here = self._here(version_id)  # N05: kept hidden where it is replayed
        with self._db() as connection:  # N31: and the checks, run again where it is replayed
            declared = checks_repo.declared(connection, row["parent_id"])
        schema = 2 if here or declared else 1
        return found.model_copy(
            update={"dismissed": here, "checks": declared, "schema_version": schema}
        )

    # --- The cleaning as SQL and Python (N45) ---------------------------------------------------

    def export(self, version_id: str, kind: Literal["sql", "python"]) -> tuple[str, str]:
        """A cleaned version's steps as a DuckDB script or a Python script on the library, read on
        the dataset they ran on: run there, they give the version's rows exactly."""
        with self._db() as connection:
            row = connection.execute(
                "SELECT parent_id, seq, steps, report, parent_stamp FROM clean_versions "
                "WHERE dataset_id = ?",
                (version_id,),
            ).fetchone()
        if row is None:
            raise AppError(
                "This dataset is not a cleaned version.", code="not_a_version", status_code=404
            )
        source = (json.loads(row["report"]).get(SOURCE) or {}).get("dataset_id")
        made_from = source or row["parent_id"]
        try:
            record = self.columns.ready(made_from)
        except AppError:
            raise AppError(
                "The dataset this version was made from is gone: it cannot be exported.",
                code="source_gone",
                status_code=409,
            ) from None
        steps = m.parse_steps(row["steps"]).steps
        original = self.datasets.get(row["parent_id"])
        name = f"{original.name} · Cleaned v{row['seq']}"
        found = {s.id: s.title for s in self._basic(made_from).suggestions}
        advanced = self._cached(made_from, "suggestions-advanced.json")
        found |= {s.id: s.title for s in advanced.suggestions} if advanced else {}
        titles = {s.id: found[s.suggestion] for s in steps if s.suggestion in found}
        notes = []
        if _out_of_date(row, json.loads(row["report"]), {made_from: _stamp(record)}):  # S7
            notes.append(
                "Out of date: the dataset it was made from changed since; this script cleans it "
                "as it is now."
            )
        info = ExportInfo(name, record.name, titles, notes)
        sources = self.sources(steps)
        stem = file_stem(name)
        if kind == "python":
            return f"{stem}.py", to_python(record.storage_path, steps, info, sources)
        connection = duckdb.connect()
        try:
            return f"{stem}.sql", to_sql(connection, record.storage_path, steps, info, sources)
        finally:
            connection.close()

    def replay(self, dataset_id: str, request: ReplayRequest) -> ReplayReport:
        """What a cleaning does on this dataset, step by step (S9); nothing is saved (C-43)."""
        record = self.columns.ready(dataset_id)
        if request.recipe is not None:
            recipe = request.recipe
        elif request.version_id:
            recipe = self.recipe(request.version_id)
        else:
            raise AppError(
                "Give a cleaned version or a recipe to replay.", code="nothing_to_replay"
            )
        views = self.columns.columns(dataset_id)
        connection = duckdb.connect()
        try:
            return replay(
                connection,
                record.storage_path,
                views,
                recipe,
                request.accept_proposals,
                sources=self.sources(recipe.steps),
                dismissed=[d.model() for d in self.dismissals(dataset_id)],
            )
        finally:
            connection.close()

    # --- Rebuild from original (B1, M5) ----------------------------------------------------------

    def rebuild(self, version_id: str) -> CleanJob:
        """A cleaned version's steps run again on the dataset it was made from, as it is now (a
        reimport since), saved as the next version (C-47). The steps are those `replay` adapts
        (the dialog shows its report first): a step that holds row numbers, or that cannot run
        any more, stays in the new version's steps, turned off, with its reason (C-41)."""
        with self._db() as connection:
            row = connection.execute(
                "SELECT parent_id, seq, report FROM clean_versions WHERE dataset_id = ?",
                (version_id,),
            ).fetchone()
        if row is None:
            raise AppError(
                "This dataset is not a cleaned version.", code="not_a_version", status_code=404
            )
        source = (json.loads(row["report"]).get(SOURCE) or {}).get("dataset_id")
        made_from = source or row["parent_id"]
        try:
            self.columns.ready(made_from)
        except AppError:
            raise AppError(
                "The dataset this version was made from is gone: it cannot be rebuilt.",
                code="source_gone",
                status_code=409,
            ) from None
        report = self.replay(made_from, ReplayRequest(version_id=version_id))
        steps = report.adapted  # what no longer applies is turned off, with its reason (C-41)
        if not any(s.enabled for s in steps):
            raise AppError(
                f"No step of Cleaned v{row['seq']} applies to the data any more.",
                code="nothing_applies",
                status_code=409,
            )
        return self.save(
            made_from, steps, actor="replay", detail=f"rebuilt from Cleaned v{row['seq']}"
        )

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

    def save(
        self,
        dataset_id: str,
        steps: list[Any],
        origin: str = "user",
        actor: Actor = "user",
        detail: str | None = None,
    ) -> CleanJob:
        """Write a cleaned version in the background, then import it (S4). `actor` (N46): who
        saved it, for the journal."""
        self.columns.ready(dataset_id)
        m.Steps(steps=steps)
        job = CleanJob(uuid.uuid4().hex[:12], "running", "queued", dataset_id)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > KEPT_JOBS:
                self._jobs.popitem(last=False)
        self._executor.submit(self._run_save, job, steps, origin, actor, detail)
        return job

    def _run_save(
        self,
        job: CleanJob,
        steps: list[Any],
        origin: str,
        actor: Actor = "user",
        detail: str | None = None,
    ) -> None:
        try:
            job.created_dataset_id = self._write(job, steps, origin, actor, detail)
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

    def _write(
        self,
        job: CleanJob,
        steps: list[Any],
        origin: str,
        actor: Actor = "user",
        detail: str | None = None,
    ) -> str:
        """096: `job.dataset_id` may be a cleaned version: the steps run on it, and the new
        version joins the original's family (`parent_id`), made from it (`source`)."""
        parent = self.columns.ready(job.dataset_id)
        original_id = self.parent_of(parent.id) or parent.id
        original = self.datasets.get(original_id)
        with self._db() as connection:
            row = connection.execute(
                "SELECT coalesce(max(seq), 0) FROM clean_versions WHERE parent_id = ?",
                (original_id,),
            ).fetchone()
            source = connection.execute(
                "SELECT seq FROM clean_versions WHERE dataset_id = ?", (parent.id,)
            ).fetchone()
            records = columns_repo.list_columns(connection, parent.id)
        seq = int(row[0]) + 1
        name = f"{original.name} · Cleaned v{seq}"
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
                sources=self.sources(steps),
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
            parent_id=original_id,
            seq=seq,
            steps=m.Steps(steps=steps).to_json(),
            origin=origin,
            stamp=_stamp(parent),
            report=report,
            rowmap=rowmap,
            changes=corrections(steps, records),
            upload=target,
            source_id=parent.id if source else None,
            source_seq=int(source["seq"]) if source else None,
            actor=actor,
            detail=detail,
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

    def _watch(self, job: CleanJob | ExpertJob, connection: duckdb.DuckDBPyConnection) -> None:
        """Interrupt DuckDB when the job is cancelled (the engine itself has no threads)."""
        while job.stage in ("cleaning", "running"):
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
            self._replay_checks(dataset_id, storage_path, "version")  # N31
            return
        if record.imported_at is not None:  # N31: a reimport runs the family's checks again
            self._replay_checks(dataset_id, storage_path, "import")
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
            made_from = pending.source_id or pending.parent_id
            parent = [to_view(r) for r in columns_repo.list_columns(connection, made_from)]
        pending.report.type_changes = type_changes(parent, child, steps)
        report = pending.report.to_dict()
        if pending.source_id:  # 096: kept beside the report, no migration
            report[SOURCE] = {"dataset_id": pending.source_id, "seq": pending.source_seq}
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
                    json.dumps(report),
                    _now(),
                ),
            )
            # N46: the draft's changes go into this version, then the save itself
            journal_repo.close(connection, made_from, dataset_id)
            summary = pending.report.summary()
            saved = JournalEntry(
                _now(),
                pending.actor,
                "saved",
                detail=f"Cleaned v{pending.seq}, {len(steps)} steps: {summary}"
                + (f" ({pending.detail})" if pending.detail else ""),
                version_id=dataset_id,
            )
            journal_repo.add(connection, made_from, [saved])
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
        """Mode `auto`: wait for the import to finish, then save the reliable fixes (N03: the
        `auto` suggestions only) as "Cleaned v1" when one of them changes a cell."""
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
            steps = default_steps(self.suggestions(dataset_id).suggestions, auto_only=True)
            if not steps:
                return
            found = self.preview(dataset_id, CleanPreviewRequest(steps=steps))
            if not (found.cells_changed or found.rows_removed):
                return
            job = CleanJob(uuid.uuid4().hex[:12], "running", "queued", dataset_id)
            self._jobs[job.id] = job
            self._run_save(job, steps, "auto", "auto")
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
        report = json.loads(row["report"])
        out_of_date = _out_of_date(row, report, stamps)
        source_id, source_seq = _source(report, stamps)
        return CleanVersion(
            dataset_id=row["dataset_id"],
            parent_id=row["parent_id"],
            seq=int(row["seq"]),
            name=child.name if child else f"Cleaned v{row['seq']}",
            origin=row["origin"],
            status=child.status if child else "error",
            created_at=row["created_at"],
            out_of_date=out_of_date,
            steps=m.parse_steps(row["steps"]).steps,
            report=CleanReport.from_dict(report),
            source_id=source_id,
            source_seq=source_seq,
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
            runs = {  # N31: the last run of the checks on each version
                r["dataset_id"]: checks_repo.summary(r)
                for r in connection.execute(
                    "SELECT * FROM clean_check_runs WHERE id IN (SELECT max(id) FROM "
                    "clean_check_runs GROUP BY dataset_id)"
                ).fetchall()
            }
        versions = [self._version(r, stamps, records) for r in rows]
        for version in versions:
            version.checks = runs.get(version.dataset_id)
        return CleanFamily(dataset_id, original, versions)

    def lineage(self) -> list[CleanLineageItem]:
        """Every cleaned version and its parent (for the home page and the top bar, C-13)."""
        stamps = {r.id: _stamp(r) for r in self.datasets.list_all()}
        with self._db() as connection:
            rows = connection.execute(
                "SELECT dataset_id, parent_id, seq, origin, steps, parent_stamp, report, "
                "created_at FROM clean_versions ORDER BY parent_id, seq"
            ).fetchall()
        return [self._item(row, stamps) for row in rows]

    def _item(self, row: sqlite3.Row, stamps: dict[str, str]) -> CleanLineageItem:
        report = json.loads(row["report"])
        out_of_date = _out_of_date(row, report, stamps)
        source_id, source_seq = _source(report, stamps)
        return CleanLineageItem(
            dataset_id=row["dataset_id"],
            parent_id=row["parent_id"],
            seq=int(row["seq"]),
            origin=row["origin"],
            steps=len(json.loads(row["steps"]).get("steps", [])),
            created_at=row["created_at"],
            out_of_date=out_of_date,
            source_id=source_id,
            source_seq=source_seq,
        )

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
