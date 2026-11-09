"""Recipes in Python: made from a library `Dataset`, replayed on another one.

recipe = dx.Recipe.from_dataset(october.where("pays", "in", ["FR", "BE"]), "Monthly sales",
                                kpis={"Revenue": {...}}, models=[model])
report = recipe.run(dx.open("november.csv"))
report.figures
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb

from quantilo.catalog import ColumnChange, apply_change, summarize_record
from quantilo.filters.model import Condition
from quantilo.ingestion.sql import lit
from quantilo.kpi.definition import Definition
from quantilo.recipes.build import (
    ModelSpec,
    build_recipe,
    distinct_values,
    parquet_categories,
)
from quantilo.recipes.model import (
    MlStep,
    OutputStep,
    Recipe,
    SeriesConfig,
    TimeSeriesStep,
)
from quantilo.recipes.run import RunReport, StepLink, unreadable_share
from quantilo.recipes.run import run as run_recipe
from quantilo.timeseries.drivers import Driver


def model_spec(model: Any) -> ModelSpec:
    """What a recipe keeps of a library `TrainedModel` (or of the app's run)."""
    result = getattr(model, "result", model)
    preparation = result.preparation
    return ModelSpec(
        target=getattr(model, "target", None) or result.target,
        model=result.model,
        settings=dict(result.settings),
        preparation=preparation.choices if preparation.custom else None,
        classes=result.classes,
        validation=result.validation,
        conditions=list(getattr(model, "_conditions", [])),
        inputs=[d.column for d in preparation.decisions if d.use],
    )


def from_dataset(
    ds: Any,
    name: str,
    *,
    kpis: dict[str, Definition | dict[str, Any]] | None = None,
    timeseries: dict[str, SeriesConfig | dict[str, Any]] | None = None,
    models: list[Any] | None = None,
    report: bool = False,
    csv: bool = False,
    corrections: bool = True,
    filter_name: str = "Filter",
) -> Recipe:
    """A recipe from a library `Dataset`: its corrected columns, its conditions (`ds.where`) as
    the active filter, and the `kpis`, `timeseries` analyses (forecast included), trained
    `models` and outputs given."""
    filters = [(filter_name, list(ds.conditions))] if ds.conditions else []
    return build_recipe(
        name,
        ds.types,
        filters=filters,
        active_filter=filter_name if filters else None,
        kpis=[
            (k, v if isinstance(v, Definition) else Definition.model_validate(v))
            for k, v in (kpis or {}).items()
        ],
        analyses=[
            (k, v if isinstance(v, SeriesConfig) else SeriesConfig.model_validate(v), True)
            for k, v in (timeseries or {}).items()
        ],
        models=[model_spec(m) for m in models or []],
        report=report,
        csv=csv,
        corrections=corrections,
        source=ds.name,
        categories=parquet_categories(str(ds.path), ds.types),
    )


class DatasetTarget:
    """A library `Dataset` as the place a recipe is replayed (corrections change it in place)."""

    def __init__(self, ds: Any, output_dir: str | Path | None = None) -> None:
        from quantilo.dataset import Dataset

        self._dataset_type = Dataset
        self.ds = ds.unfiltered()
        self.name = ds.name
        self.output_dir = Path(output_dir) if output_dir is not None else Path.cwd()

    def _on(self, conditions: list[Condition]) -> Any:
        return self._dataset_type(self.ds._state, tuple(conditions))

    def columns(self) -> list[Any]:
        return self.ds.types

    def unreadable(self, column: str, analysis_type: str) -> float:
        record = replace(self.ds._record(column))
        apply_change(record, ColumnChange(name=column, type=analysis_type), set())  # type: ignore[arg-type]
        return unreadable_share(json.loads(summarize_record(str(self.ds.path), record)))

    def correct(self, change: ColumnChange) -> StepLink | None:
        self.ds._change(change)
        return None

    def categories(self, column: str, limit: int) -> list[str] | None:
        connection = duckdb.connect()
        try:
            source = f"read_parquet({lit(str(self.ds.path))})"
            return distinct_values(connection, source, self.ds.column(column), limit)
        finally:
            connection.close()

    def count(self, conditions: list[Condition]) -> int:
        return len(self._on(conditions))

    def save_filter(self, name: str, conditions: list[Condition]) -> StepLink | None:
        return None

    def kpi(self, step: Any, conditions: list[Condition]) -> tuple[Any, StepLink | None]:
        return self._on(conditions).kpi(step.definition), None

    def timeseries(
        self, step: TimeSeriesStep, conditions: list[Condition], cancelled: Any
    ) -> tuple[Any, StepLink | None]:
        if not step.forecast:
            return None, None
        a = step.analysis
        series = self._on(conditions).timeseries(
            a.time,
            a.value,
            group=a.group,
            series=a.series,
            period=a.period,
            aggregate=a.aggregate,
            window=a.window,
            align=a.align,
            gaps=a.gaps,
            season=a.season,
        )
        forecast = series.forecast(
            horizon=a.horizon,
            models=a.models,
            order=a.order,
            seasonal_order=a.seasonal_order,
            drivers=[Driver(**d.model_dump()) for d in a.drivers] if a.drivers else None,
        )
        return forecast, None

    def train(
        self, step: MlStep, conditions: list[Condition], cancelled: Any
    ) -> tuple[Any, StepLink | None]:
        learner = self._on(conditions).ml(
            step.target, classes=step.classes, validation=step.validation
        )
        return learner.train(step.model, step.settings or None, step.preparation), None

    def output(self, step: OutputStep, conditions: list[Condition]) -> list[str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = self._on(conditions)
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.name) or "dataset"
        files = []
        if step.report:
            rows.report(self.output_dir / f"{stem}_report.html")
            files.append(f"{stem}_report.html")
        if step.csv:
            rows.export(self.output_dir / f"{stem}_rows.csv")
            files.append(f"{stem}_rows.csv")
        return files


def run(
    recipe: Recipe,
    ds: Any,
    mapping: dict[str, str | None] | None = None,
    *,
    accept_proposals: bool = False,
    output_dir: str | Path | None = None,
) -> RunReport:
    return run_recipe(
        recipe, DatasetTarget(ds, output_dir), mapping, accept_proposals=accept_proposals
    )
