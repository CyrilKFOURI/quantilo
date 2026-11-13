"""Replay a cleaning on another file (101, M5): the steps of a cleaned version, or a cleaning recipe
(`quantilo/clean-recipe`, C-39), run on the file of next month, with a report of what applies, what
no longer applies and why (C-41), and what is new there (C-42). Nothing is saved: the adapted
steps become a draft, read before any version is made (C-43)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

import duckdb
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from quantilo.catalog import ColumnView
from quantilo.cleaning import checks as ck
from quantilo.cleaning import dismiss
from quantilo.cleaning import formula as fx
from quantilo.cleaning import model as m
from quantilo.cleaning.apply import CleanCancelled
from quantilo.cleaning.compile import Compiled, as_text, compile_steps
from quantilo.cleaning.preview import compare
from quantilo.cleaning.suggest import suggest
from quantilo.cleaning.suggestion import CleanSuggestion
from quantilo.ingestion.sql import ident
from quantilo.recipes.matching import Status, match_columns
from quantilo.recipes.model import RecipeColumn
from quantilo.typing.types import AnalysisType

FORMAT = "quantilo/clean-recipe"
MAX_KNOWN = m.MAX_MAP  # values kept for a column a `map_values` step reads (C-39)
MAX_BYTES = 5_000_000
SHOWN_NEW = 10
COLUMN_FIELDS = ("column", "year", "month", "day", "hour", "minute", "time", "value_column")
COLUMN_FIELDS += ("order_by",)  # N22: the date a duplicate is kept by
COLUMN_FIELDS += ("anchor",)  # N36: what tells an entity apart (an entity number)
LIST_FIELDS = ("columns", "keys")
ROWS_ELSEWHERE = (
    "Holds row numbers of the file it was made on: it does not apply to another file. Make it "
    "again on this one if you need it."
)
# N35: a removal of duplicates whose groups were reviewed on the file it was made on
REVIEW_ELSEWHERE = (
    "Its groups were reviewed on the file it was made on (rows kept or chosen by number): here "
    "the rule alone applies. Review the groups again on this file if you need it."
)
StepStatus = Literal["applies", "unchanged", "skipped"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CleanRecipeColumn(Strict):
    name: Annotated[str, Field(min_length=1, max_length=500)]
    type: AnalysisType


class CleanRecipe(Strict):
    """A cleaning to replay: its steps, the columns of the dataset they were made on, and, for each
    `map_values` step, the values its column had there just before it (to flag new ones).
    Schema 2: also the suggestions dismissed on that dataset (N05), to keep them hidden, and the
    checks declared there (N31), run again where it is replayed; a recipe of schema 1 reads as one
    without them."""

    format: Literal["quantilo/clean-recipe"] = FORMAT
    schema_version: Literal[1, 2] = Field(default=1, alias="schema")
    name: Annotated[str, Field(min_length=1, max_length=300)]
    made_on: str | None = Field(default=None, max_length=500)
    made_at: str | None = Field(default=None, max_length=40)
    columns: list[CleanRecipeColumn] = Field(default_factory=list, max_length=5_000)
    steps: list[m.Step] = Field(min_length=1, max_length=m.MAX_STEPS)
    known: dict[str, list[Annotated[str, Field(max_length=2_000)]]] = Field(default_factory=dict)
    dismissed: list[dismiss.Dismissal] = Field(
        default_factory=list, max_length=dismiss.MAX_DISMISSED
    )
    checks: list[ck.CleanCheck] = Field(default_factory=list, max_length=ck.MAX_CHECKS)

    @model_validator(mode="after")
    def _schema(self) -> CleanRecipe:
        if self.dismissed or self.checks:  # what schema 1 cannot say
            self.schema_version = 2
        return self

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True, indent=1)


def parse_recipe(raw: str | bytes | dict[str, Any]) -> CleanRecipe:
    """A recipe read from its JSON, refused with the field named (C-39)."""
    if isinstance(raw, str | bytes):
        if len(raw) > MAX_BYTES:
            raise m.CleanError("This recipe is too large (5 MB at most).")
        try:
            raw = json.loads(raw)
        except ValueError as error:
            raise m.CleanError(f"Invalid recipe: not JSON ({error}).") from None
    if not isinstance(raw, dict):
        raise m.CleanError("Invalid recipe: expected a JSON object.")
    if raw.get("format") != FORMAT:
        raise m.CleanError(f"Not a cleaning recipe: `format` must be {FORMAT!r}.")
    if raw.get("schema", 1) not in (1, 2):
        raise m.CleanError(f"Unknown cleaning recipe schema version {raw.get('schema')!r}.")
    try:
        return CleanRecipe.model_validate(raw)
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        if first["type"] == "extra_forbidden":
            raise m.CleanError(f"Unknown field `{where}` in the recipe.") from None
        raise m.CleanError(f"Invalid recipe, field `{where}`: {first['msg']}.") from None


# --- Columns a step names ------------------------------------------------------------------------


def step_columns(step: Any) -> list[str]:
    """Every column a step reads, in its fields."""
    found: list[str] = []
    for name in COLUMN_FIELDS:
        value = getattr(step, name, None)
        if isinstance(value, str):
            found.append(value)
    for name in LIST_FIELDS:
        value = getattr(step, name, None)
        if isinstance(value, list):
            found += [v for v in value if isinstance(v, str)]
    for condition in getattr(step, "conditions", None) or []:
        found.append(condition.column)
    for text in (getattr(step, "formula", None), getattr(step, "when", None)):
        found += fx.columns_in(text)
    for case in getattr(step, "cases", None) or []:  # C-45: conditional steps
        found += [condition.column for condition in case.conditions]
        found += fx.columns_in(case.when) + fx.columns_in(case.formula)
    for aggregation in getattr(step, "aggregations", None) or []:  # C-56: group by
        if aggregation.column:
            found.append(aggregation.column)
    return list(dict.fromkeys(found))


def renamed_step(step: Any, names: dict[str, str]) -> Any:
    """The step with the recipe's column names replaced by the file's."""
    changes: dict[str, Any] = {}
    for name in COLUMN_FIELDS:
        value = getattr(step, name, None)
        if isinstance(value, str) and value in names:
            changes[name] = names[value]
    for name in LIST_FIELDS:
        value = getattr(step, name, None)
        if isinstance(value, list):
            changes[name] = [names.get(v, v) for v in value]
    conditions = getattr(step, "conditions", None)
    if conditions:
        changes["conditions"] = _renamed_conditions(conditions, names)
    for name in ("formula", "when"):
        text = getattr(step, name, None)
        if isinstance(text, str):
            changes[name] = fx.renamed(text, names)
    aggregations = getattr(step, "aggregations", None)
    if aggregations:
        changes["aggregations"] = [
            a.model_copy(update={"column": names.get(a.column, a.column) if a.column else None})
            for a in aggregations
        ]
    cases = getattr(step, "cases", None)
    if cases:
        changes["cases"] = [
            case.model_copy(
                update={
                    "conditions": _renamed_conditions(case.conditions, names),
                    "when": fx.renamed(case.when, names),
                    "formula": fx.renamed(case.formula, names),
                }
            )
            for case in cases
        ]
    return step.model_copy(update=changes, deep=True)


def _renamed_conditions(conditions: Sequence[Any], names: dict[str, str]) -> list[Any]:
    return [c.model_copy(update={"column": names.get(c.column, c.column)}) for c in conditions]


def made_by_steps(steps: Sequence[Any]) -> set[str]:
    """Columns the steps themselves add (a rename's new name, a duplicate, a date, labels, the
    parts of a split)."""
    found: set[str] = set()
    for step in steps:
        to = getattr(step, "to", None)
        if isinstance(to, str):
            found.add(to)
        elif isinstance(to, list):
            found.update(name for name in to if isinstance(name, str))
        # C1: the columns a structure step writes
        for name in ("names_to", "values_to", "source_column"):
            value = getattr(step, name, None)
            if isinstance(value, str):
                found.add(value)
        found.update(a.to for a in getattr(step, "aggregations", None) or [])
        if step.op == "merge_dataset":
            found.update(f"{step.prefix}{name}" for name in step.add or [])
        if step.op == "pivot":
            found.update(name or "(empty)" for name in step.headers)
    return found


# --- Recipe from steps ---------------------------------------------------------------------------


def _distinct(
    connection: duckdb.DuckDBPyConnection, compiled: Compiled, index: int, column: str
) -> list[str] | None:
    """The values of `column` at stage `index`, as text (None: more than MAX_KNOWN)."""
    stage = compiled.stages[index]
    text = as_text(column, stage.dtype(column))
    rows = connection.execute(
        f"{compiled.sql(index, f'DISTINCT {text} AS v')} WHERE {ident(column)} IS NOT NULL "
        f"LIMIT {MAX_KNOWN + 1}"
    ).fetchall()
    if len(rows) > MAX_KNOWN:
        return None
    return sorted(str(row[0]) for row in rows)


def _before(compiled: Compiled, step_id: str) -> int | None:
    index = compiled.stage_of(step_id)
    return None if index is None else index - 1


def recipe_of(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    views: Sequence[ColumnView],
    steps: Sequence[Any],
    name: str,
    made_on: str | None = None,
) -> CleanRecipe:
    """The recipe of `steps`, made on this dataset (its columns, and the values each `map_values`
    step met)."""
    compiled = compile_steps(connection, parquet_path, list(steps))
    known: dict[str, list[str]] = {}
    for step in steps:
        if step.op != "map_values" or not step.enabled:
            continue
        index = _before(compiled, step.id)
        if index is None:
            continue
        values = _distinct(connection, compiled, index, step.column)
        if values is not None:
            known[step.id] = values
    return CleanRecipe(
        name=name,
        made_on=made_on,
        made_at=datetime.now(UTC).isoformat(timespec="seconds"),
        columns=[
            CleanRecipeColumn(name=v.name, type=v.type)
            for v in sorted(views, key=lambda v: v.position)
        ],
        steps=[step.model_copy(deep=True) for step in steps],
        known=known,
    )


# --- Replay --------------------------------------------------------------------------------------


@dataclass
class ReplayStep:
    id: str
    op: str
    rule: str | None
    status: StepStatus
    message: str | None = None
    columns: list[str] = field(default_factory=list)  # the file's columns it works on
    cells: int = 0  # cells it changes on this file
    rows: int = 0  # rows it removes on this file
    new_values: list[str] = field(default_factory=list)  # C-42: values it does not know
    new_count: int = 0


@dataclass
class ReplayColumn:
    recipe: str
    column: str | None
    status: Status
    distance: int | None = None


@dataclass
class ReplayReport:
    recipe: str
    made_on: str | None
    steps: list[ReplayStep]
    columns: list[ReplayColumn]
    new_columns: list[str]  # in the file, unknown to the recipe
    new_problems: list[CleanSuggestion]  # C-42: Basic suggestions no step covers, not applied
    adapted: list[m.Step]  # the draft (C-43): the steps that no longer apply turned off
    applies: int = 0
    skipped: int = 0
    seconds: float = 0.0
    # N05: the recipe's dismissals (kept when it becomes the draft) and how many suggestions of
    # this file they hide (left out of `new_problems`)
    dismissed: list[dismiss.Dismissal] = field(default_factory=list)
    hidden: int = 0
    # N31: the recipe's checks, their columns paired with this file's, and what they find here
    checks: list[ck.CleanCheck] = field(default_factory=list)
    check_report: ck.CheckReport | None = None


def _covered(suggestion: CleanSuggestion, steps: Sequence[Any]) -> bool:
    for step in steps:
        if step.rule != suggestion.rule:
            continue
        if not suggestion.columns or set(suggestion.columns) & set(step_columns(step)):
            return True
    return False


def _new_values(
    connection: duckdb.DuckDBPyConnection, compiled: Compiled, step: Any, known: list[str]
) -> tuple[list[str], int]:
    index = _before(compiled, step.id)
    if index is None:
        return [], 0
    values = _distinct(connection, compiled, index, step.column)
    if values is None:
        return [], 0
    seen = set(known) | set(step.map) | {v for v in step.map.values() if v is not None}
    new = [v for v in values if v not in seen]
    return new[:SHOWN_NEW], len(new)


def replay(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    views: Sequence[ColumnView],
    recipe: CleanRecipe,
    accept_proposals: bool = False,
    stopped: Callable[[], bool] = lambda: False,
    sources: Mapping[str, str] | None = None,
    dismissed: Sequence[dismiss.Dismissal] = (),
) -> ReplayReport:
    """What `recipe` does on this file, step by step; `adapted` is the draft to review.
    `dismissed` (N05): this dataset's own dismissals, which hide new problems as the recipe's do."""
    started = time.monotonic()
    expected = [RecipeColumn(name=c.name, type=c.type) for c in recipe.columns]
    matching = match_columns(expected, [(v.name, v.type) for v in views])
    mapping = matching.mapping(accept_proposals)
    names = {old: new for old, new in mapping.items() if new is not None}
    known_columns = {c.name for c in recipe.columns}
    added = made_by_steps(recipe.steps)
    report: dict[str, ReplayStep] = {}
    adapted: list[Any] = []
    for step in recipe.steps:
        used = step_columns(step)
        entry = ReplayStep(step.id, step.op, step.rule, "skipped")
        live = renamed_step(step, names)
        entry.columns = step_columns(live)
        if not step.enabled:
            entry.message = "Turned off in the cleaning."
        elif m.names_rows(step):
            entry.message = ROWS_ELSEWHERE
            live = live.model_copy(update={"enabled": False})
        elif missing := [
            c for c in used if c in known_columns and c not in added and mapping.get(c) is None
        ]:
            proposed = {
                match.recipe: match.column
                for match in matching.columns
                if match.status == "proposed"
            }
            hint = "".join(
                f" (closest: '{proposed[c]}', not used unless you accept it)"
                for c in missing[:1]
                if proposed.get(c)
            )
            entry.message = f"Uses {', '.join(missing)}, missing from this file{hint}."
            live = live.model_copy(update={"enabled": False})
        else:
            entry.status = "applies"
            if m.reviewed(live):  # N35: the groups reviewed name the rows of the other file
                live = live.model_copy(update={"survivors": None, "not_duplicates": None})
                entry.message = REVIEW_ELSEWHERE
        report[step.id] = entry
        adapted.append(live)
    if stopped():
        raise CleanCancelled
    compiled = compile_steps(connection, parquet_path, adapted, sources=sources)
    for step_id, problem in compiled.problems.items():
        entry = report[step_id]
        entry.status, entry.message = "skipped", f"Cannot run on this file: {problem}"
    adapted = [
        s.model_copy(update={"enabled": False}) if s.id in compiled.problems else s for s in adapted
    ]
    for index, stage in enumerate(compiled.stages[1:], start=1):
        if stopped():
            raise CleanCancelled
        entry = report[stage.step.id]
        found = compare(connection, compiled, index - 1, index)
        entry.cells, entry.rows = found.cells_changed, found.rows_removed
        reshaped = any(
            c.before is not None and (c.before != c.column or c.dtype_before != c.dtype_after)
            for c in found.columns
        )  # a rename or a new type changes no shown value
        if not (entry.cells or entry.rows or found.added or found.dropped or reshaped):
            entry.status, entry.message = "unchanged", "Changes nothing on this file."
        known = recipe.known.get(stage.step.id)
        if stage.step.op == "map_values" and known is not None:
            entry.new_values, entry.new_count = _new_values(connection, compiled, stage.step, known)
            if entry.new_count:
                shown = ", ".join(f"'{v}'" for v in entry.new_values[:3])
                more = "…" if entry.new_count > 3 else ""
                entry.message = (
                    f"{entry.new_count:,} value{'s' if entry.new_count > 1 else ''} it does not "
                    f"know, left as they are: {shown}{more}"
                )
    if stopped():
        raise CleanCancelled
    found_now = suggest(connection, parquet_path, list(views))
    live_steps = [s for s in adapted if s.enabled]
    shown, hidden = dismiss.split(found_now, [*recipe.dismissed, *dismissed])
    checks = [renamed_check(check, names) for check in recipe.checks]
    check_report = ck.run_checks(connection, parquet_path, list(views), checks) if checks else None
    new_problems = [s for s in shown if s.step is not None and not _covered(s, live_steps)]
    steps = list(report.values())
    return ReplayReport(
        recipe=recipe.name,
        made_on=recipe.made_on,
        steps=steps,
        columns=[ReplayColumn(c.recipe, c.column, c.status, c.distance) for c in matching.columns],
        new_columns=list(matching.new_columns),
        new_problems=new_problems,
        adapted=adapted,
        applies=sum(1 for s in steps if s.status != "skipped"),
        skipped=sum(1 for s in steps if s.status == "skipped"),
        seconds=round(time.monotonic() - started, 2),
        dismissed=list(recipe.dismissed),
        hidden=len(hidden),
        checks=checks,
        check_report=check_report,
    )


def renamed_check(check: ck.CleanCheck, names: dict[str, str]) -> ck.CleanCheck:
    """N31: a recipe's check on this file's columns (paired as the steps are)."""
    return check.model_copy(
        update={
            "columns": [names.get(c, c) for c in check.columns],
            "conditions": _renamed_conditions(check.conditions, names),
        },
        deep=True,
    )
