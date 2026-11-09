"""A recipe made from what was kept on a dataset: its corrected columns, saved filters, KPIs,
time series analyses, models and the outputs wanted. Used by the app and the library alike."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import duckdb

from quantilo.catalog import ColumnView
from quantilo.filters.model import Condition
from quantilo.ingestion.sql import lit
from quantilo.kpi.definition import Definition
from quantilo.recipes.model import (
    MAX_CATEGORIES,
    ColumnStep,
    FilterStep,
    KpiStep,
    MlStep,
    OutputStep,
    Recipe,
    RecipeColumn,
    SeriesConfig,
    TimeSeriesStep,
)
from quantilo.recipes.steps import categories_read
from quantilo.typing.parsers import typed_sql

CATEGORICAL = ("categorical_nominal", "categorical_ordinal", "boolean")


@dataclass
class ModelSpec:
    """What a recipe keeps of a trained model to train it again."""

    target: str
    model: str | None
    settings: dict[str, Any] = field(default_factory=dict)
    preparation: Any = None  # ml.preparation.Choices
    classes: Any = None  # ml.classes.Classes
    validation: Any = None  # ml.evaluate.Validation
    conditions: list[Condition] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)


def column_step(step_id: str, view: ColumnView) -> ColumnStep | None:
    """The correction the user made to a column, or None when it is as detected."""
    changed_type = view.type if view.type != view.detected_type else None
    role = None
    if (view.role or None) != (view.detected_role or None):
        role = view.role or ""
    order = view.order if view.overridden and view.type == "categorical_ordinal" else None
    excluded = True if view.excluded else None
    if changed_type is None and role is None and not order and excluded is None:
        return None
    return ColumnStep(
        id=step_id,
        column=view.name,
        type=changed_type,
        role=role,
        order=order or None,
        excluded=excluded,
    )


def distinct_values(
    connection: duckdb.DuckDBPyConnection, source: str, view: ColumnView, limit: int
) -> list[str] | None:
    """The column's distinct values in its type (as text), or None when there are more."""
    typed = typed_sql(view.name, view.parser)
    rows = connection.execute(
        f"SELECT DISTINCT CAST({typed} AS VARCHAR) AS v FROM {source} "
        f"WHERE {typed} IS NOT NULL ORDER BY v LIMIT {int(limit) + 1}"
    ).fetchall()
    if len(rows) > limit:
        return None
    return [str(row[0]) for row in rows]


def parquet_categories(
    storage_path: str, views: Sequence[ColumnView]
) -> Callable[[str], list[str] | None]:
    """`categories(column)` read from a Parquet copy (≤ MAX_CATEGORIES values, else None)."""
    by_name = {view.name: view for view in views}

    def categories(column: str) -> list[str] | None:
        view = by_name.get(column)
        if view is None or view.type not in CATEGORICAL:
            return None
        connection = duckdb.connect()
        try:
            source = f"read_parquet({lit(storage_path)})"
            return distinct_values(connection, source, view, MAX_CATEGORIES)
        finally:
            connection.close()

    return categories


def build_recipe(
    name: str,
    columns: Sequence[ColumnView],
    *,
    filters: Sequence[tuple[str, list[Condition]]] = (),
    active_filter: str | None = None,
    kpis: Sequence[tuple[str, Definition]] = (),
    analyses: Sequence[tuple[str, SeriesConfig, bool]] = (),
    models: Sequence[ModelSpec] = (),
    report: bool = False,
    csv: bool = False,
    corrections: bool = True,
    source: str | None = None,
    categories: Callable[[str], list[str] | None] | None = None,
) -> Recipe:
    """Steps in execution order. `active_filter`: the name of the filter whose rows the KPIs,
    analyses and outputs use; `analyses`: (name, configuration, run the forecast)."""
    steps: list[Any] = []

    def next_id() -> str:
        return f"s{len(steps) + 1}"

    if corrections:
        for view in sorted(columns, key=lambda v: v.position):
            step = column_step(next_id(), view)
            if step is not None:
                steps.append(step)
    active_id = None
    for filter_name, conditions in filters:
        step_id = next_id()
        active = filter_name == active_filter
        active_id = step_id if active else active_id
        steps.append(FilterStep(id=step_id, name=filter_name, conditions=conditions, active=active))
    for kpi_name, definition in kpis:
        steps.append(KpiStep(id=next_id(), name=kpi_name, definition=definition, filter=active_id))
    for analysis_name, analysis, forecast in analyses:
        steps.append(
            TimeSeriesStep(
                id=next_id(),
                name=analysis_name,
                analysis=analysis,
                forecast=forecast,
                filter=active_id,
            )
        )
    for spec in models:
        values: dict[str, Any] = {
            "id": next_id(),
            "target": spec.target,
            "model": spec.model,
            "settings": spec.settings or {},
            "preparation": spec.preparation,
            "classes": spec.classes,
            "conditions": spec.conditions,
            "inputs": spec.inputs,
        }
        if spec.validation is not None:
            values["validation"] = spec.validation
        steps.append(MlStep.model_validate(values))
    if report or csv:
        steps.append(OutputStep(id=next_id(), report=report, csv=csv, filter=active_id))
    read = {c for step in steps for c in categories_read(step)}
    expected = []
    for view in sorted(columns, key=lambda v: v.position):
        known = categories(view.name) if categories and view.name in read else None
        expected.append(RecipeColumn(name=view.name, type=view.detected_type, categories=known))
    return Recipe(
        name=name.strip(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source=source,
        columns=expected,
        steps=steps,
    )


def select_steps(
    recipe: Recipe,
    keep: Sequence[str] | None,
    active_filter: str | None,
    report: bool = False,
    csv: bool = False,
) -> Recipe:
    """The steps ticked (`keep`: their ids, None = all), one filter made active (its id) for the
    KPIs, time series and outputs, and the output files wanted."""
    steps = [
        s
        for s in recipe.steps
        if not isinstance(s, OutputStep) and (keep is None or s.id in set(keep))
    ]
    filters = {s.id for s in steps if isinstance(s, FilterStep)}
    active = active_filter if active_filter in filters else None
    chosen: list[Any] = []
    for step in steps:
        if isinstance(step, FilterStep):
            step = step.model_copy(update={"active": step.id == active})
        elif isinstance(step, KpiStep | TimeSeriesStep):
            step = step.model_copy(update={"filter": active})
        chosen.append(step)
    if report or csv:
        used = {int(s.id[1:]) for s in recipe.steps}
        free = next(n for n in range(1, len(used) + 2) if n not in used)
        chosen.append(OutputStep(id=f"s{free}", report=report, csv=csv, filter=active))
    return Recipe.model_validate(
        {**recipe.model_dump(by_alias=True), "steps": [s.model_dump(by_alias=True) for s in chosen]}
    )


def edit_steps(
    recipe: Recipe,
    name: str | None = None,
    enabled: dict[str, bool] | None = None,
    remove: Sequence[str] = (),
) -> Recipe:
    """The next version of a recipe: renamed, steps turned on or off, steps deleted (a filter
    deleted takes its rows away from the steps that used them)."""
    gone = set(remove)
    steps = []
    for step in recipe.steps:
        if step.id in gone:
            continue
        update: dict[str, Any] = {}
        if enabled and step.id in enabled:
            update["enabled"] = enabled[step.id]
        if getattr(step, "filter", None) in gone:
            update["filter"] = None
        steps.append(step.model_copy(update=update).model_dump(by_alias=True))
    raw = recipe.model_dump(by_alias=True)
    raw.update(steps=steps, version=recipe.version + 1)
    if name is not None and name.strip():
        raw["name"] = name.strip()
    return Recipe.model_validate(raw)
