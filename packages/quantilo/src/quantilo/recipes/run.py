"""Replay a recipe on a dataset, step by step, and say what happened.

The runner is the same for the app and the library: it pairs the columns, skips the steps whose
columns are missing or whose filter could not be applied, runs the others through a `Target`
(the app's services, or a library `Dataset`), and reads the key figures from their results.
A failing step does not stop the others; a cancelled run marks the remaining steps cancelled.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from quantilo.catalog import ColumnChange, ColumnView
from quantilo.changes.kpis import kpi_value
from quantilo.errors import ExplorerError
from quantilo.filters.model import Condition
from quantilo.kpi.compute import KpiResult
from quantilo.recipes.matching import Matching, match_columns
from quantilo.recipes.model import (
    ColumnStep,
    FilterStep,
    KpiStep,
    MlStep,
    OutputStep,
    Recipe,
    Step,
    TimeSeriesStep,
    ordered,
)
from quantilo.recipes.steps import TYPE_LABELS, categories_read, columns_read
from quantilo.recipes.steps import renamed as renamed_step
from quantilo.recipes.steps import sentence as step_sentence
from quantilo.timeseries.forecasting import SeriesForecast

UNREADABLE_LIMIT = 0.10  # a type correction leaving more values unreadable is not applied
StepStatus = Literal["waiting", "running", "done", "warning", "skipped", "failed", "cancelled"]
RunStatus = Literal["running", "done", "warning", "failed", "cancelled"]
FigureKind = Literal["kpi", "forecast", "model"]


@dataclass
class StepLink:
    """A page of the app the step filled (the library leaves it empty)."""

    page: str
    params: dict[str, str] = field(default_factory=dict)
    label: str = ""


@dataclass
class KeyFigure:
    id: str  # the step's id: figures of two runs of a recipe pair on it
    step: str
    kind: FigureKind
    label: str
    value: float | None
    detail: str | None = None  # the metric, the forecast period…
    higher_is_better: bool | None = None


@dataclass
class FigureChange:
    id: str
    label: str
    kind: FigureKind
    value: float | None
    previous: float | None
    change: float | None  # value - previous
    change_pct: float | None  # relative, 0.05 = +5 %
    detail: str | None = None
    higher_is_better: bool | None = None


@dataclass
class StepResult:
    id: str
    kind: str
    sentence: str
    status: StepStatus = "waiting"
    message: str | None = None
    warnings: list[str] = field(default_factory=list)
    link: StepLink | None = None
    files: list[str] = field(default_factory=list)  # output file names


@dataclass
class RunReport:
    recipe: str
    version: int
    dataset: str
    status: RunStatus
    steps: list[StepResult]
    matching: Matching
    mapping: dict[str, str | None]
    figures: list[KeyFigure] = field(default_factory=list)
    new_columns: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str | None = None


@dataclass
class TrainedLike:
    """What a key figure reads from a trained model (the app's run, the library's model)."""

    model: str
    primary: str
    score: float | None


class Target(Protocol):
    """Where a recipe is replayed: the app's services on one dataset, or a library `Dataset`."""

    name: str

    def columns(self) -> list[ColumnView]: ...

    def unreadable(self, column: str, analysis_type: str) -> float:
        """Share of the present values the type could not read (nothing saved)."""
        ...

    def correct(self, change: ColumnChange) -> StepLink | None: ...

    def categories(self, column: str, limit: int) -> list[str] | None:
        """The column's distinct values, or None when there are more than `limit`."""
        ...

    def count(self, conditions: list[Condition]) -> int: ...

    def save_filter(self, name: str, conditions: list[Condition]) -> StepLink | None: ...

    def kpi(
        self, step: KpiStep, conditions: list[Condition]
    ) -> tuple[KpiResult, StepLink | None]: ...

    def timeseries(
        self, step: TimeSeriesStep, conditions: list[Condition], cancelled: Callable[[], bool]
    ) -> tuple[SeriesForecast | None, StepLink | None]: ...

    def train(
        self, step: MlStep, conditions: list[Condition], cancelled: Callable[[], bool]
    ) -> tuple[Any, StepLink | None]: ...

    def output(self, step: OutputStep, conditions: list[Condition]) -> list[str]: ...


class StepCancelled(Exception):
    """Raised by a target when the run is cancelled in the middle of a step."""


class StepWarning(Exception):
    """A step that could not be done as asked, but whose failure is only a warning."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def unreadable_share(summary: dict[str, Any]) -> float:
    """Share of the present values a column's summary could not read in its type."""
    invalid = int(summary.get("invalid_count") or 0)
    valid = int(summary.get("rows") or 0) - int(summary.get("null_count") or 0)
    present = invalid + valid
    return invalid / present if present else 0.0


# --- Key figures -------------------------------------------------------------------------------


def kpi_figure(step: KpiStep, result: KpiResult) -> KeyFigure:
    measure = kpi_value(step.name, result).measure
    return KeyFigure(step.id, step.id, "kpi", step.name, result.value, measure or None)


def forecast_figure(step: TimeSeriesStep, forecast: SeriesForecast) -> KeyFigure | None:
    best = next((r for r in forecast.results if r.key == forecast.best), None)
    if best is None or not best.forecast:
        return None
    period = forecast.future[0] if forecast.future else None
    label = f"{step.analysis.value or 'Rows'}: next value"
    return KeyFigure(step.id, step.id, "forecast", label, float(best.forecast[0]), period)


def model_figure(step: MlStep, trained: Any) -> KeyFigure:
    from quantilo.ml.evaluate import LOWER_IS_BETTER
    from quantilo.recipes.steps import model_label

    score = getattr(trained, "score", None)
    primary = str(getattr(trained, "primary", ""))
    label = f"{model_label(getattr(trained, 'model', step.model))} → {step.target}"
    return KeyFigure(
        step.id,
        step.id,
        "model",
        label,
        None if score is None else float(score),
        primary or None,
        primary not in LOWER_IS_BETTER if primary else None,
    )


def compare_figures(
    current: Sequence[KeyFigure], previous: Sequence[KeyFigure] | None
) -> list[FigureChange]:
    """Each key figure next to the same figure of the previous run, with the change."""
    before = {figure.id: figure for figure in previous or []}
    found = []
    for figure in current:
        old = before.get(figure.id)
        value, past = figure.value, old.value if old else None
        change = None if value is None or past is None else value - past
        pct = None if change is None or not past else change / abs(past)
        found.append(
            FigureChange(
                figure.id,
                figure.label,
                figure.kind,
                value,
                past,
                change,
                pct,
                figure.detail,
                figure.higher_is_better,
            )
        )
    return found


# --- Planning -----------------------------------------------------------------------------------


@dataclass
class PlannedStep:
    id: str
    kind: str
    sentence: str
    enabled: bool
    skip: str | None  # why it will be skipped (None: it will run)


def mapping_for(
    recipe: Recipe,
    columns: Sequence[ColumnView],
    chosen: dict[str, str | None] | None = None,
    accept_proposals: bool = False,
) -> tuple[Matching, dict[str, str | None]]:
    matching = match_columns(recipe.columns, [(c.name, c.detected_type) for c in columns])
    mapping = matching.mapping(accept_proposals)
    names = {c.name for c in columns}
    for recipe_column, file_column in (chosen or {}).items():
        if recipe_column not in mapping:
            raise ExplorerError(
                f"The recipe has no column {recipe_column!r}.", code="bad_mapping", status_code=422
            )
        if file_column is not None and file_column not in names:
            raise ExplorerError(
                f"The file has no column {file_column!r}.", code="bad_mapping", status_code=422
            )
        mapping[recipe_column] = file_column
    taken = [c for c in mapping.values() if c is not None]
    twice = sorted({c for c in taken if taken.count(c) > 1})
    if twice:
        raise ExplorerError(
            f"Column {twice[0]!r} is paired with two recipe columns.",
            code="bad_mapping",
            status_code=422,
        )
    return matching, mapping


def _missing(step: Step, mapping: dict[str, str | None]) -> list[str]:
    # a column the recipe does not list (hand-written recipe) is looked up under its own name
    return [c for c in columns_read(step) if c in mapping and mapping[c] is None]


def plan(
    recipe: Recipe, mapping: dict[str, str | None], skip_kinds: dict[str, str] | None = None
) -> list[PlannedStep]:
    """Which steps will run, and why the others will be skipped, before running anything."""
    names = {s.id: s.name for s in recipe.steps if isinstance(s, FilterStep)}
    skipped_filters: set[str] = set()
    found = []
    for step in ordered(recipe.steps):
        reason = None
        if not step.enabled:
            reason = "Turned off in the recipe."
        elif step.kind in (skip_kinds or {}):
            reason = (skip_kinds or {})[step.kind]
        elif missing := _missing(step, mapping):
            reason = f"Uses {', '.join(missing)}, missing from the file."
        elif (used := getattr(step, "filter", None)) in skipped_filters:
            reason = (
                f"Uses the rows of the filter “{names.get(used, used)}”, which will be skipped."
            )
        if reason and isinstance(step, FilterStep):
            skipped_filters.add(step.id)
        found.append(
            PlannedStep(step.id, step.kind, step_sentence(step, names), step.enabled, reason)
        )
    return found


# --- Running ------------------------------------------------------------------------------------


def run_status(steps: Sequence[StepResult], cancelled: bool) -> RunStatus:
    if cancelled:
        return "cancelled"
    if any(s.status == "failed" for s in steps):
        return "failed"
    if any(
        s.status in ("warning", "skipped")
        for s in steps
        if s.message != "Turned off in the recipe."
    ):
        return "warning"
    return "done"


def run(
    recipe: Recipe,
    target: Target,
    mapping: dict[str, str | None] | None = None,
    *,
    accept_proposals: bool = False,
    skip_kinds: dict[str, str] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    on_update: Callable[[RunReport], None] | None = None,
) -> RunReport:
    """Replay `recipe` on `target`. `mapping` overrides the automatic matching (recipe column →
    file column, None = missing); `skip_kinds`: step kinds to skip, with the reason (machine
    learning in the browser); `on_update` is called after each step (progress)."""
    matching, chosen = mapping_for(recipe, target.columns(), mapping, accept_proposals)
    names = {s.id: s.name for s in recipe.steps if isinstance(s, FilterStep)}
    steps = ordered(recipe.steps)
    report = RunReport(
        recipe.name,
        recipe.version,
        target.name,
        "running",
        [StepResult(s.id, s.kind, step_sentence(s, names)) for s in steps],
        matching,
        chosen,
        new_columns=list(matching.new_columns),
        started_at=_now(),
    )
    expected = {column.name: column for column in recipe.columns}
    by_file = {file: name for name, file in chosen.items() if file}
    filters: dict[
        str, list[Condition] | None
    ] = {}  # filter step → its conditions (None: not applied)
    stopped = False

    def update() -> None:
        if on_update is not None:
            on_update(report)

    update()
    for step, result in zip(steps, report.steps, strict=True):
        if stopped or cancelled():
            stopped = True
            result.status, result.message = "cancelled", "Cancelled before this step."
            continue
        reason = None
        if not step.enabled:
            reason = "Turned off in the recipe."
        elif step.kind in (skip_kinds or {}):
            reason = (skip_kinds or {})[step.kind]
        elif missing := _missing(step, chosen):
            reason = f"Uses {', '.join(missing)}, missing from the file."
        elif (used := getattr(step, "filter", None)) is not None and filters.get(used) is None:
            reason = (
                f"Uses the rows of the filter “{names.get(used, used)}”, which was not applied."
            )
        if reason is not None:
            result.status, result.message = "skipped", reason
            if isinstance(step, FilterStep):
                filters[step.id] = None
            update()
            continue
        result.status = "running"
        update()
        live = renamed_step(step, chosen)
        conditions = filters.get(getattr(step, "filter", None) or "") or []
        try:
            figures, warnings, link, files = _execute(live, target, conditions, cancelled)
            if isinstance(live, FilterStep):
                filters[step.id] = live.conditions
            result.warnings += warnings
            result.link, result.files = link, files
            report.figures += figures
            result.status = "warning" if result.warnings else "done"
        except StepCancelled:
            stopped = True
            result.status, result.message = "cancelled", "Cancelled during this step."
        except StepWarning as warning:
            result.status, result.message = "warning", str(warning)
            if isinstance(step, FilterStep):
                filters[step.id] = None
        except ExplorerError as error:
            result.status, result.message = "failed", error.message
            if isinstance(step, FilterStep):
                filters[step.id] = None
        except Exception as error:  # a failing step never stops the others
            result.status, result.message = "failed", str(error) or type(error).__name__
            if isinstance(step, FilterStep):
                filters[step.id] = None
        if result.status in ("done", "warning"):
            new = _new_categories(live, step, target, expected, by_file)
            result.warnings += new
            if new:
                result.status = "warning"
        update()
    report.status = run_status(report.steps, stopped)
    report.finished_at = _now()
    update()
    return report


def _new_categories(
    live: Step, step: Step, target: Target, expected: dict[str, Any], by_file: dict[str, str]
) -> list[str]:
    found = []
    for column in dict.fromkeys(categories_read(live)):
        known = expected.get(by_file.get(column, column))
        if known is None or known.categories is None:
            continue
        values = target.categories(column, 1_000)
        if values is None:
            continue
        new = [v for v in values if v not in set(known.categories)]
        if new:
            sample = ", ".join(new[:3]) + ("…" if len(new) > 3 else "")
            found.append(
                f"{len(new)} new categor{'y' if len(new) == 1 else 'ies'} in {column}: {sample}"
            )
    return found


def _execute(
    step: Step, target: Target, conditions: list[Condition], cancelled: Callable[[], bool]
) -> tuple[list[KeyFigure], list[str], StepLink | None, list[str]]:
    """Run one step: (key figures, warnings, link, output files)."""
    if isinstance(step, ColumnStep):
        warnings: list[str] = []
        change = ColumnChange(
            name=step.column, role=step.role, order=step.order, excluded=step.excluded
        )
        if step.type is not None:
            share = target.unreadable(step.column, step.type)
            label = TYPE_LABELS.get(step.type, step.type)
            if share > UNREADABLE_LIMIT:
                problem = (
                    f"Not applied: {share:.0%} of the values of {step.column} can't be read as "
                    f"{label}."
                )
                if change.role is None and change.order is None and change.excluded is None:
                    raise StepWarning(problem)
                warnings.append(problem)
            else:
                change.type = step.type  # type: ignore[assignment]
        link = target.correct(change)
        return [], warnings, link, []
    if isinstance(step, FilterStep):
        link = target.save_filter(step.name, step.conditions)
        rows = target.count(step.conditions)
        warnings = ["No row matches this filter."] if rows == 0 else []
        return [], warnings, link, []
    if isinstance(step, KpiStep):
        result, link = target.kpi(step, conditions)
        figure = kpi_figure(step, result)
        warnings = list(result.warnings)
        if figure.value is None:
            warnings.append("The KPI has no value on these rows.")
        return [figure], warnings, link, []
    if isinstance(step, TimeSeriesStep):
        forecast, link = target.timeseries(step, conditions, cancelled)
        if forecast is None:
            return [], [], link, []
        figure = forecast_figure(step, forecast)
        if figure is None:
            return [], ["No forecast model could be fitted."], link, []
        return [figure], [], link, []
    if isinstance(step, MlStep):
        missing = [c for c in step.inputs if c not in {v.name for v in target.columns()}]
        trained, link = target.train(step, step.conditions, cancelled)
        warnings = []
        if missing:
            warnings.append(
                f"{len(missing)} column(s) the model used are missing: {', '.join(missing[:5])}"
                f"{'…' if len(missing) > 5 else ''}."
            )
        return [model_figure(step, trained)], warnings, link, []
    if isinstance(step, OutputStep):
        return [], [], None, target.output(step, conditions)
    raise ExplorerError(f"Unknown step {step.kind!r}.")
