"""The app as the place a recipe is replayed: each step goes through the service that owns it
(columns, saved filters, KPI builder, time series, machine learning, report, rows), and what it
adds to the dataset is marked with the recipe's name. Nothing already there is overwritten."""

import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
from quantilo.catalog import ColumnChange, ColumnView
from quantilo.errors import ExplorerError
from quantilo.filters.model import Condition
from quantilo.filters.scope import scoped_source, where_for
from quantilo.ingestion.sql import lit
from quantilo.kpi.compute import KpiResult
from quantilo.recipes.build import distinct_values
from quantilo.recipes.model import KpiStep, MlStep, OutputStep, TimeSeriesStep
from quantilo.recipes.run import StepCancelled, StepLink, unreadable_share
from quantilo.recipes.script import file_stem
from quantilo.rows.view import View
from quantilo.timeseries.drivers import Driver
from quantilo.timeseries.forecasting import SeriesForecast
from quantilo.timeseries.series import Options
from quantilo.timeseries.service import Request as SeriesRequest

POLL = 0.2  # seconds between two looks at a background job


@dataclass
class Services:
    """The app's services a recipe uses (a module never imports another one: they come from
    app.state)."""

    datasets: Any
    columns: Any
    saved_filters: Any
    reports: Any
    rows: Any
    versions: Any
    kpi: Any = None
    timeseries: Any = None
    analyses: Any = None
    ml: Any = None


def marked(name: str, recipe: str) -> str:
    """What a run adds carries the recipe's name: "Revenue · recipe Monthly sales"."""
    return f"{name} · recipe {recipe}"[:200]


def series_params(step: TimeSeriesStep) -> dict[str, str]:
    a = step.analysis
    found = {
        "time": a.time,
        "value": a.value,
        "group": a.group,
        "series": a.series,
        "period": None if a.period == "auto" else a.period,
        "agg": None if a.aggregate == "auto" else a.aggregate,
        "gaps": None if a.gaps == "leave" else a.gaps,
        "season": str(a.season) if a.season else None,
    }
    return {k: v for k, v in found.items() if v}


class AppTarget:
    keeps = True  # what a run adds is saved on the dataset (the baseline saves nothing)

    def __init__(self, services: Services, dataset_id: str, recipe: str, folder: Path) -> None:
        self.s = services
        self.dataset_id = dataset_id
        self.recipe = recipe
        self.folder = folder
        self.name = services.datasets.get(dataset_id).name

    def _wait(
        self, poll: Callable[[], Any], cancel: Callable[[], Any], cancelled: Callable[[], bool]
    ) -> Any:
        while True:
            job = poll()
            if job.status != "running":
                return job
            if cancelled():
                cancel()
                raise StepCancelled
            time.sleep(POLL)

    def columns(self) -> list[ColumnView]:
        return self.s.columns.columns(self.dataset_id)

    def unreadable(self, column: str, analysis_type: str) -> float:
        _parser, summary = self.s.columns.conversion(self.dataset_id, column, analysis_type)
        return unreadable_share(summary)

    def correct(self, change: ColumnChange) -> StepLink | None:
        self.s.columns.update(self.dataset_id, [change])
        return StepLink("columns", {}, "Open Columns")

    def _storage(self) -> str:
        return self.s.columns.ready(self.dataset_id).storage_path

    def categories(self, column: str, limit: int) -> list[str] | None:
        view = next((c for c in self.columns() if c.name == column), None)
        if view is None:
            return None
        connection = duckdb.connect()
        try:
            source = f"read_parquet({lit(self._storage())})"
            return distinct_values(connection, source, view, limit)
        finally:
            connection.close()

    def count(self, conditions: list[Condition]) -> int:
        source = scoped_source(self._storage(), where_for(self.columns(), conditions))
        connection = duckdb.connect()
        try:
            row = connection.execute(f"SELECT count(*) FROM {source}").fetchone()
        finally:
            connection.close()
        return int(row[0]) if row else 0

    def save_filter(self, name: str, conditions: list[Condition]) -> StepLink | None:
        saved = self.s.saved_filters.save(self.dataset_id, marked(name, self.recipe), conditions)
        return StepLink("data", {"saved": saved.id}, "Open Data")

    def kpi(self, step: KpiStep, conditions: list[Condition]) -> tuple[KpiResult, StepLink | None]:
        if self.s.kpi is None:
            raise ExplorerError("The KPI builder is not available.")
        result = self.s.kpi.compute(self.dataset_id, step.definition, conditions)
        name = marked(step.name, self.recipe)
        existing = next((k for k in self.s.kpi.all(self.dataset_id) if k.name == name), None)
        if existing is not None:  # this recipe's own KPI from an earlier run on this dataset
            saved = self.s.kpi.update(self.dataset_id, existing.id, None, step.definition)
        else:
            saved = self.s.kpi.create(self.dataset_id, name, step.definition)
        return result, StepLink("kpi", {"kpi": saved.id}, "Open the KPI")

    def timeseries(
        self, step: TimeSeriesStep, conditions: list[Condition], cancelled: Callable[[], bool]
    ) -> tuple[SeriesForecast | None, StepLink | None]:
        if self.s.timeseries is None:
            raise ExplorerError("The time series module is not available.")
        if self.s.analyses is not None and self.keeps:
            self.s.analyses.save(self.dataset_id, step.name, step.analysis, origin=self.recipe)
        link = StepLink("timeseries", series_params(step), "Open Time series")
        if not step.forecast:
            return None, link
        a = step.analysis
        request = SeriesRequest(
            a.time,
            a.value,
            a.group,
            a.series,
            Options(a.period, a.aggregate, a.window, a.align),
            a.gaps,
            a.season,
        )
        service = self.s.timeseries
        job = service.start_forecast(
            self.dataset_id,
            request,
            conditions,
            horizon=a.horizon,
            chosen=a.models,
            order=a.order,
            seasonal_order=a.seasonal_order,
            drivers=[Driver(**d.model_dump()) for d in a.drivers] if a.drivers else None,
        )
        job = self._wait(
            lambda: service.forecasts.get(job.id) or job,
            lambda: service.forecasts.cancel(job.id),
            cancelled,
        )
        if job.status == "cancelled":
            raise StepCancelled
        if job.result is None:
            raise ExplorerError(job.error or "The forecast failed.")
        return job.result, link

    def train(
        self, step: MlStep, conditions: list[Condition], cancelled: Callable[[], bool]
    ) -> tuple[Any, StepLink | None]:
        if self.s.ml is None:
            raise ExplorerError("Machine learning is not available.")
        ml = self.s.ml
        job = ml.start(
            self.dataset_id,
            step.target,
            step.model,
            step.settings or None,
            conditions,
            step.preparation,
            step.classes,
            step.validation,
        )
        job = self._wait(lambda: ml.job(job.id), lambda: ml.cancel(job.id), cancelled)
        if job.status == "cancelled":
            raise StepCancelled
        if job.status != "done" or job.run_id is None:
            raise ExplorerError(job.error or "The training failed.")
        run = ml.run(self.dataset_id, job.run_id)
        return run, StepLink("ml", {"target": step.target, "run": job.run_id}, "Open the model")

    def output(self, step: OutputStep, conditions: list[Condition]) -> list[str]:
        self.folder.mkdir(parents=True, exist_ok=True)
        stem = file_stem(self.name)
        files = []
        if step.report:
            content, _name = self.s.reports.build(self.dataset_id, conditions)
            (self.folder / f"{stem}_report.html").write_text(content, encoding="utf-8")
            files.append(f"{stem}_report.html")
        if step.csv:
            exported = self.s.rows.export(self.dataset_id, "csv", View(), None, conditions)
            try:
                shutil.move(exported.path, self.folder / f"{stem}_rows.csv")
            finally:
                exported.remove()
            files.append(f"{stem}_rows.csv")
        return files


class BaselineTarget(AppTarget):
    """The dataset a recipe was just made on: its key figures, computed without changing or
    adding anything (the models' scores are read from their runs, not trained again). The first
    run on a new file is compared with them."""

    keeps = False

    def unreadable(self, column: str, analysis_type: str) -> float:
        return 0.0

    def correct(self, change: ColumnChange) -> StepLink | None:
        return None

    def save_filter(self, name: str, conditions: list[Condition]) -> StepLink | None:
        return None

    def kpi(self, step: KpiStep, conditions: list[Condition]) -> tuple[KpiResult, StepLink | None]:
        if self.s.kpi is None:
            raise ExplorerError("The KPI builder is not available.")
        return self.s.kpi.compute(self.dataset_id, step.definition, conditions), None

    def timeseries(
        self, step: TimeSeriesStep, conditions: list[Condition], cancelled: Callable[[], bool]
    ) -> tuple[SeriesForecast | None, StepLink | None]:
        forecast, _link = super().timeseries(step, conditions, cancelled)
        return forecast, None

    def train(
        self, step: MlStep, conditions: list[Condition], cancelled: Callable[[], bool]
    ) -> tuple[Any, StepLink | None]:
        if self.s.ml is None:
            raise ExplorerError("Machine learning is not available.")
        for summary in self.s.ml.runs(self.dataset_id, step.target):  # newest first
            if summary.model == step.model or step.model is None:
                run = self.s.ml.run(self.dataset_id, summary.id)
                return run, StepLink("ml", {"target": step.target, "run": summary.id}, "Open")
        raise ExplorerError("The model's run is no longer on this dataset.")

    def output(self, step: OutputStep, conditions: list[Condition]) -> list[str]:
        return []
