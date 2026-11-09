"""Recipes (030): made from a dataset, kept with every version, replayed in the background on
another dataset, with a report, the key figures and what changed (031).

The engine (`quantilo.recipes`) holds the recipe, the matching, the runner and the
Python export; this service adds SQLite, background runs with cancellation and the app's target.
"""

import csv
import io
import json
import logging
import shutil
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from quantilo.catalog import ColumnView
from quantilo.errors import ExplorerError
from quantilo.recipes.build import (
    ModelSpec,
    build_recipe,
    edit_steps,
    parquet_categories,
    select_steps,
)
from quantilo.recipes.matching import Matching, match_columns
from quantilo.recipes.model import FilterStep, Recipe, ordered, parse
from quantilo.recipes.run import (
    FigureChange,
    KeyFigure,
    PlannedStep,
    RunReport,
    compare_figures,
    mapping_for,
    plan,
    run,
)
from quantilo.recipes.script import file_stem, to_python
from quantilo.recipes.steps import columns_read, sentence

from app.core import db
from app.core.errors import AppError
from app.core.storage import Workspace
from app.modules.recipes.target import AppTarget, BaselineTarget, Services

logger = logging.getLogger(__name__)

ML_IN_BROWSER = "Machine learning is not available in the browser version: it needs the full app."
COMPATIBLE_SHARE = 0.8  # columns matched on their name to suggest a recipe for a dataset
FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")
REPORT = TypeAdapter(RunReport)
FIGURES = TypeAdapter(list[KeyFigure])


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _cell(text: str) -> str:
    """A cell a spreadsheet would run as a formula gets a leading apostrophe; numbers (negative
    ones included) are left as they are."""
    try:
        float(text)
        return text
    except ValueError:
        return f"'{text}" if text.startswith(FORMULA_STARTS) else text


class RecipeNotFound(AppError):
    def __init__(self, what: str = "Recipe") -> None:
        super().__init__(f"{what} not found", code="not_found", status_code=404)


@dataclass
class RecipeStepView:
    id: str
    kind: str
    sentence: str
    enabled: bool = True


@dataclass
class RecipeRunSummary:
    id: str
    recipe_id: str
    version: int
    dataset_id: str | None
    dataset_name: str
    status: str
    created_at: str
    finished_at: str | None
    figures: list[KeyFigure] = field(default_factory=list)
    kind: str = "run"  # or "baseline": the figures of the dataset the recipe was made on


@dataclass
class RecipeSummary:
    id: str
    name: str
    version: int
    steps: int
    source_dataset_id: str | None
    source: str | None
    created_at: str
    updated_at: str
    last_run: RecipeRunSummary | None = None


@dataclass
class FigurePoint:
    run_id: str
    created_at: str
    dataset_name: str
    value: float | None


@dataclass
class FigureHistory:
    id: str
    label: str
    kind: str
    detail: str | None
    points: list[FigurePoint]


@dataclass
class RecipeDetail(RecipeSummary):
    recipe: Recipe | None = None
    step_views: list[RecipeStepView] = field(default_factory=list)
    runs: list[RecipeRunSummary] = field(default_factory=list)
    history: list[FigureHistory] = field(default_factory=list)


@dataclass
class RecipeDraft:
    """Everything a dataset keeps, as the steps of a recipe to tick."""

    recipe: Recipe
    steps: list[RecipeStepView]


@dataclass
class RecipeChanges:
    """What changed (031), at the top of a run's report."""

    level: str
    level_label: str
    headline: str
    vs: str | None  # the dataset compared with (None: the previous version of the same one)


@dataclass
class RecipeRunDetail(RecipeRunSummary):
    report: RunReport | None = None
    changes: RecipeChanges | None = None
    compared: list[FigureChange] = field(default_factory=list)
    previous_run_id: str | None = None
    recipe_name: str = ""
    error: str | None = None  # why the run itself failed (no report)


@dataclass
class RecipePlan:
    matching: Matching
    mapping: dict[str, str | None]
    steps: list[PlannedStep]


@dataclass
class CompatibleRecipe:
    recipe_id: str
    name: str
    matched: int
    total: int


@dataclass
class _Running:
    stop: threading.Event = field(default_factory=threading.Event)


class RecipeService:
    def __init__(self, workspace: Workspace, services: Services, lite: bool = False) -> None:
        self.workspace = workspace
        self.s = services
        self.skip_kinds = {"ml": ML_IN_BROWSER} if lite or services.ml is None else {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recipe")
        self._running: dict[str, _Running] = {}
        self._recovered = False
        self._lock = threading.Lock()

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            if not self._recovered:
                self._recover(connection)
            yield connection
        finally:
            connection.close()

    def _recover(self, connection: sqlite3.Connection) -> None:
        """Runs left running by a stopped app are failed (nothing runs yet at start-up)."""
        with self._lock:
            if self._recovered:
                return
            self._recovered = True
            running = list(self._running)
            marks = ",".join("?" * len(running))
            query = (
                "UPDATE recipe_runs SET status = 'failed', finished_at = ? WHERE status = 'running'"
            )
            if running:
                query += f" AND id NOT IN ({marks})"
            with connection:
                connection.execute(query, (_now(), *running))

    def folder(self, run_id: str) -> Path:
        return self.workspace.root / "recipes" / run_id

    # --- Recipes -------------------------------------------------------------------------------

    def _body(self, connection: sqlite3.Connection, recipe_id: str, version: int) -> Recipe:
        row = connection.execute(
            "SELECT body FROM recipe_versions WHERE recipe_id = ? AND version = ?",
            (recipe_id, version),
        ).fetchone()
        if row is None:
            raise RecipeNotFound()
        return Recipe.model_validate_json(row["body"])

    def _row(self, connection: sqlite3.Connection, recipe_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM recipes WHERE id = ?", (recipe_id,)).fetchone()
        if row is None:
            raise RecipeNotFound()
        return row

    def recipe(self, recipe_id: str) -> Recipe:
        with self._db() as connection:
            row = self._row(connection, recipe_id)
            return self._body(connection, recipe_id, row["version"])

    def _summary(self, connection: sqlite3.Connection, row: sqlite3.Row) -> RecipeSummary:
        body = self._body(connection, row["id"], row["version"])
        last = connection.execute(
            "SELECT * FROM recipe_runs WHERE recipe_id = ? ORDER BY created_at DESC LIMIT 1",
            (row["id"],),
        ).fetchone()
        return RecipeSummary(
            row["id"],
            row["name"],
            row["version"],
            len(body.steps),
            row["source_dataset_id"],
            body.source,
            row["created_at"],
            row["updated_at"],
            self._run_summary(last) if last else None,
        )

    def all(self) -> list[RecipeSummary]:
        with self._db() as connection:
            rows = connection.execute(
                "SELECT * FROM recipes ORDER BY updated_at DESC, name COLLATE NOCASE"
            ).fetchall()
            return [self._summary(connection, row) for row in rows]

    def detail(self, recipe_id: str) -> RecipeDetail:
        with self._db() as connection:
            row = self._row(connection, recipe_id)
            summary = self._summary(connection, row)
            body = self._body(connection, recipe_id, row["version"])
            runs = [
                self._run_summary(r)
                for r in connection.execute(
                    "SELECT * FROM recipe_runs WHERE recipe_id = ? ORDER BY created_at DESC",
                    (recipe_id,),
                ).fetchall()
            ]
        return RecipeDetail(
            **summary.__dict__,
            recipe=body,
            step_views=self._step_views(body),
            runs=runs,
            history=self._history(runs),
        )

    @staticmethod
    def _step_views(recipe: Recipe) -> list[RecipeStepView]:
        names = {s.id: s.name for s in recipe.steps if isinstance(s, FilterStep)}
        return [
            RecipeStepView(s.id, s.kind, sentence(s, names), s.enabled)
            for s in ordered(recipe.steps)
        ]

    def _insert(self, recipe: Recipe, source_dataset_id: str | None) -> RecipeSummary:
        recipe_id = uuid.uuid4().hex[:12]
        stamp = _now()
        with self._db() as connection, connection:
            connection.execute(
                "INSERT INTO recipes (id, name, version, source_dataset_id, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (recipe_id, recipe.name, recipe.version, source_dataset_id, stamp, stamp),
            )
            connection.execute(
                "INSERT INTO recipe_versions (recipe_id, version, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (recipe_id, recipe.version, recipe.to_json(None), stamp),
            )
            return self._summary(connection, self._row(connection, recipe_id))

    def draft(self, dataset_id: str) -> RecipeDraft:
        """Every correction, filter, KPI, analysis and model of the dataset as recipe steps."""
        record = self.s.columns.ready(dataset_id)
        views: list[ColumnView] = self.s.columns.columns(dataset_id)
        filters = [(f.name, f.conditions) for f in self.s.saved_filters.all(dataset_id)]
        kpis = self.s.kpi.definitions(dataset_id) if self.s.kpi is not None else []
        analyses = (
            [(a.name, a.config, True) for a in self.s.analyses.all(dataset_id)]
            if self.s.analyses is not None
            else []
        )
        recipe = build_recipe(
            record.name,
            views,
            filters=filters,
            kpis=kpis,
            analyses=analyses,
            models=self._models(dataset_id),
            report=True,
            csv=True,
            source=record.name,
            categories=parquet_categories(record.storage_path, views),
        )
        return RecipeDraft(recipe, self._step_views(recipe))

    def _models(self, dataset_id: str) -> list[ModelSpec]:
        """The latest run of each target and model."""
        if self.s.ml is None:
            return []
        latest: dict[tuple[str, str], Any] = {}
        for summary in self.s.ml.runs(dataset_id):  # newest first
            latest.setdefault((summary.target, summary.model), summary)
        found = []
        for summary in reversed(list(latest.values())):
            try:
                run_ = self.s.ml.run(dataset_id, summary.id)
            except ExplorerError:
                continue
            preparation = run_.preparation
            found.append(
                ModelSpec(
                    target=run_.target,
                    model=run_.model,
                    settings=dict(run_.settings),
                    preparation=preparation.choices if preparation.custom else None,
                    classes=run_.classes,
                    validation=run_.validation,
                    conditions=self.s.ml._conditions(summary.id),
                    inputs=[d.column for d in preparation.decisions if d.use],
                )
            )
        return found

    def create(
        self,
        dataset_id: str,
        name: str,
        keep: list[str] | None,
        active_filter: str | None,
        report: bool,
        csv_rows: bool,
    ) -> RecipeSummary:
        if not name.strip():
            raise AppError("Give the recipe a name.", code="invalid_name", status_code=422)
        draft = self.draft(dataset_id).recipe.model_copy(update={"name": name.strip()})
        recipe = select_steps(draft, keep, active_filter, report, csv_rows)
        if not recipe.steps:
            raise AppError("Tick at least one step.", code="empty_recipe", status_code=422)
        summary = self._insert(recipe, dataset_id)
        self.start(summary.id, dataset_id, None, baseline=True)
        return summary

    def import_recipe(self, content: str) -> RecipeSummary:
        recipe = parse(content)
        return self._insert(recipe.model_copy(update={"version": 1}), None)

    def edit(
        self,
        recipe_id: str,
        name: str | None,
        enabled: dict[str, bool] | None,
        remove: list[str],
    ) -> RecipeDetail:
        """A new version of the recipe (runs keep the version they used)."""
        current = self.recipe(recipe_id)
        unknown = [i for i in [*(enabled or {}), *remove] if i not in {s.id for s in current.steps}]
        if unknown:
            raise AppError(f"Unknown step {unknown[0]!r}.", code="unknown_step", status_code=422)
        updated = edit_steps(current, name, enabled, remove)
        stamp = _now()
        with self._db() as connection, connection:
            connection.execute(
                "INSERT INTO recipe_versions (recipe_id, version, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (recipe_id, updated.version, updated.to_json(None), stamp),
            )
            connection.execute(
                "UPDATE recipes SET name = ?, version = ?, updated_at = ? WHERE id = ?",
                (updated.name, updated.version, stamp, recipe_id),
            )
        return self.detail(recipe_id)

    def delete(self, recipe_id: str) -> None:
        with self._db() as connection:
            self._row(connection, recipe_id)
            runs = [
                r["id"]
                for r in connection.execute(
                    "SELECT id FROM recipe_runs WHERE recipe_id = ?", (recipe_id,)
                )
            ]
            for run_id in runs:
                if run_id in self._running:
                    raise AppError(
                        "This recipe is running: cancel the run first.",
                        code="recipe_running",
                        status_code=409,
                    )
            with connection:
                connection.execute("DELETE FROM recipes WHERE id = ?", (recipe_id,))
        for run_id in runs:
            shutil.rmtree(self.folder(run_id), ignore_errors=True)

    def export_json(self, recipe_id: str) -> tuple[str, str]:
        recipe = self.recipe(recipe_id)
        return f"{file_stem(recipe.name)}.recipe.json", recipe.to_json()

    def export_python(self, recipe_id: str) -> tuple[str, str]:
        recipe = self.recipe(recipe_id)
        path = f"{file_stem(recipe.source or 'data')}.csv"
        return f"{file_stem(recipe.name)}.py", to_python(recipe, path)

    # --- Matching -----------------------------------------------------------------------------

    def plan(
        self, recipe_id: str, dataset_id: str, mapping: dict[str, str | None] | None
    ) -> RecipePlan:
        recipe = self.recipe(recipe_id)
        views = self.s.columns.columns(self.s.columns.ready(dataset_id).id)
        matching, chosen = mapping_for(recipe, views, mapping)
        return RecipePlan(matching, chosen, plan(recipe, chosen, self.skip_kinds))

    def compatible(self, dataset_id: str) -> list[CompatibleRecipe]:
        """Recipes made elsewhere, not run on this dataset yet, that find ≥ 80 % of their
        columns in it by name."""
        views = self.s.columns.columns(self.s.columns.ready(dataset_id).id)
        actual = [(c.name, c.detected_type) for c in views]
        found = []
        with self._db() as connection:
            rows = connection.execute(
                "SELECT * FROM recipes WHERE (source_dataset_id IS NULL OR source_dataset_id != ?)"
                " AND id NOT IN (SELECT recipe_id FROM recipe_runs WHERE dataset_id = ?)",
                (dataset_id, dataset_id),
            ).fetchall()
            for row in rows:
                body = self._body(connection, row["id"], row["version"])
                if not body.columns:
                    continue
                matched = match_columns(body.columns, actual).matched
                if matched / len(body.columns) >= COMPATIBLE_SHARE:
                    found.append(
                        CompatibleRecipe(row["id"], row["name"], matched, len(body.columns))
                    )
        return sorted(found, key=lambda c: (-c.matched / c.total, c.name))

    def usage(self, dataset_id: str) -> dict[str, list[str]]:
        """Column → the recipes that read it on this dataset (031: a column gone is Breaking)."""
        used: dict[str, list[str]] = {}
        try:
            with self._db() as connection:
                rows = connection.execute(
                    "SELECT r.id, r.name, r.version, r.source_dataset_id FROM recipes r "
                    "WHERE r.source_dataset_id = ? OR r.id IN "
                    "(SELECT recipe_id FROM recipe_runs WHERE dataset_id = ?)",
                    (dataset_id, dataset_id),
                ).fetchall()
                for row in rows:
                    body = self._body(connection, row["id"], row["version"])
                    mapping: dict[str, str | None] = {}
                    if row["source_dataset_id"] != dataset_id:
                        last = connection.execute(
                            "SELECT report FROM recipe_runs WHERE recipe_id = ? AND dataset_id = ? "
                            "ORDER BY created_at DESC LIMIT 1",
                            (row["id"], dataset_id),
                        ).fetchone()
                        if last:
                            mapping = (
                                json.loads(last["report"]).get("report", {}).get("mapping", {})
                            )
                    for step in body.steps:
                        if not step.enabled:
                            continue
                        for column in columns_read(step):
                            name = mapping.get(column, column) if mapping else column
                            if name:
                                users = used.setdefault(name, [])
                                label = f"Recipe “{row['name']}”"
                                if label not in users:
                                    users.append(label)
        except sqlite3.OperationalError:  # tables not created yet (start-up)
            return {}
        return used

    # --- Runs ---------------------------------------------------------------------------------

    def _run_summary(self, row: sqlite3.Row) -> RecipeRunSummary:
        return RecipeRunSummary(
            row["id"],
            row["recipe_id"],
            row["version"],
            row["dataset_id"],
            row["dataset_name"],
            row["status"],
            row["created_at"],
            row["finished_at"],
            FIGURES.validate_json(row["figures"]),
            row["kind"],
        )

    def start(
        self,
        recipe_id: str,
        dataset_id: str,
        mapping: dict[str, str | None] | None,
        baseline: bool = False,
    ) -> RecipeRunSummary:
        """Replay the recipe in the background; `baseline`: only read the key figures of the
        dataset it was made on (nothing changed, models not trained again)."""
        with self._db() as connection:
            row = self._row(connection, recipe_id)
            recipe = self._body(connection, recipe_id, row["version"])
        dataset = self.s.columns.ready(dataset_id)
        _matching, chosen = mapping_for(recipe, self.s.columns.columns(dataset_id), mapping)
        run_id = uuid.uuid4().hex[:12]
        stamp = _now()
        with self._db() as connection, connection:
            connection.execute(
                "INSERT INTO recipe_runs (id, recipe_id, version, kind, dataset_id, dataset_name, "
                "status, report, figures, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'running', ?, '[]', ?)",
                (
                    run_id,
                    recipe_id,
                    recipe.version,
                    "baseline" if baseline else "run",
                    dataset_id,
                    dataset.name,
                    "{}",
                    stamp,
                ),
            )
        self._running[run_id] = _Running()
        self._executor.submit(self._run, run_id, recipe, dataset_id, chosen, baseline)
        return self.run_summary(run_id)

    def cancel(self, run_id: str) -> RecipeRunSummary:
        summary = self.run_summary(run_id)
        running = self._running.get(run_id)
        if running is not None:
            running.stop.set()
        return summary

    def _save(self, run_id: str, report: RunReport, changes: RecipeChanges | None = None) -> None:
        payload = {"report": REPORT.dump_python(report, mode="json")}
        if changes is not None:
            payload["changes"] = changes.__dict__
        status = report.status if report.finished_at else "running"
        with self._db() as connection, connection:
            connection.execute(
                "UPDATE recipe_runs SET status = ?, report = ?, figures = ?, finished_at = ? "
                "WHERE id = ?",
                (
                    status,
                    json.dumps(payload),
                    FIGURES.dump_json(report.figures).decode(),
                    report.finished_at,
                    run_id,
                ),
            )

    def _run(
        self,
        run_id: str,
        recipe: Recipe,
        dataset_id: str,
        mapping: dict[str, str | None],
        baseline: bool = False,
    ) -> None:
        state = self._running[run_id]
        try:
            kind = BaselineTarget if baseline else AppTarget
            target = kind(self.s, dataset_id, recipe.name, self.folder(run_id))
            report = run(
                recipe,
                target,
                mapping,
                skip_kinds=self.skip_kinds,
                cancelled=state.stop.is_set,
                on_update=lambda current: (
                    self._save(run_id, current) if current.finished_at is None else None
                ),
            )
            self._save(run_id, report, None if baseline else self._changes(run_id, dataset_id))
        except Exception as error:  # the run is marked failed, with the reason
            logger.exception("Recipe run %s failed", run_id)
            with self._db() as connection, connection:
                connection.execute(
                    "UPDATE recipe_runs SET status = 'failed', report = ?, finished_at = ? "
                    "WHERE id = ?",
                    (json.dumps({"error": str(error) or type(error).__name__}), _now(), run_id),
                )
        finally:
            self._running.pop(run_id, None)

    def _previous(self, connection: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
        row = connection.execute("SELECT * FROM recipe_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return connection.execute(
            "SELECT * FROM recipe_runs WHERE recipe_id = ? AND created_at < ? AND id != ? "
            "AND status IN ('done', 'warning', 'failed') ORDER BY created_at DESC LIMIT 1",
            (row["recipe_id"], row["created_at"], run_id),
        ).fetchone()

    def _changes(self, run_id: str, dataset_id: str) -> RecipeChanges | None:
        """Compared with the previous run's dataset, or the previous version of this one."""
        with self._db() as connection:
            previous = self._previous(connection, run_id)
        others = []
        if previous is not None and previous["dataset_id"] not in (None, dataset_id):
            others.append(previous["dataset_id"])
        others.append(None)
        for vs in others:
            try:
                found = self.s.versions.changes(dataset_id, vs)
            except ExplorerError:
                continue
            except Exception:  # the comparison is a bonus: never fail the run for it
                logger.exception("Comparing for recipe run %s failed", run_id)
                continue
            return RecipeChanges(found.level, found.level_label, found.headline, vs)
        return None

    def run_summary(self, run_id: str) -> RecipeRunSummary:
        with self._db() as connection:
            row = connection.execute("SELECT * FROM recipe_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise RecipeNotFound("Recipe run")
        return self._run_summary(row)

    def run_detail(self, run_id: str) -> RecipeRunDetail:
        with self._db() as connection:
            row = connection.execute("SELECT * FROM recipe_runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise RecipeNotFound("Recipe run")
            previous = self._previous(connection, run_id)
            name = connection.execute(
                "SELECT name FROM recipes WHERE id = ?", (row["recipe_id"],)
            ).fetchone()
        summary = self._run_summary(row)
        payload = json.loads(row["report"] or "{}")
        report = REPORT.validate_python(payload["report"]) if "report" in payload else None
        changes = RecipeChanges(**payload["changes"]) if payload.get("changes") else None
        before = FIGURES.validate_json(previous["figures"]) if previous else None
        detail = RecipeRunDetail(
            **summary.__dict__,
            report=report,
            changes=changes,
            compared=compare_figures(summary.figures, before),
            previous_run_id=previous["id"] if previous else None,
            recipe_name=name["name"] if name else "",
        )
        detail.error = payload.get("error")
        return detail

    def file(self, run_id: str, name: str) -> Path:
        detail = self.run_detail(run_id)
        names = {f for step in (detail.report.steps if detail.report else []) for f in step.files}
        if name not in names:
            raise RecipeNotFound("File")
        path = self.folder(run_id) / name
        if not path.is_file():
            raise RecipeNotFound("File")
        return path

    @staticmethod
    def _history(runs: list[RecipeRunSummary]) -> list[FigureHistory]:
        found: dict[str, FigureHistory] = {}
        for summary in reversed(runs):  # oldest first
            if summary.status in ("running", "cancelled"):
                continue
            for figure in summary.figures:
                line = found.setdefault(
                    figure.id,
                    FigureHistory(figure.id, figure.label, figure.kind, figure.detail, []),
                )
                line.label = figure.label
                line.points.append(
                    FigurePoint(summary.id, summary.created_at, summary.dataset_name, figure.value)
                )
        return list(found.values())

    def history_csv(self, recipe_id: str) -> tuple[str, str]:
        detail = self.detail(recipe_id)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["run", "date", "dataset", "figure", "detail", "value"])
        for line in detail.history:
            for point in line.points:
                writer.writerow(
                    [
                        point.run_id,
                        point.created_at,
                        _cell(point.dataset_name),
                        _cell(line.label),
                        _cell(line.detail or ""),
                        "" if point.value is None else repr(point.value),
                    ]
                )
        return f"{file_stem(detail.name)}_key_figures.csv", buffer.getvalue()
