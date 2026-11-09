"""Run every analysis the report needs and render it (R-REP-02): overview, one step per column,
correlations, relationships. Used by the app's report jobs and by `Dataset.report()`."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quantilo.catalog import Catalog
from quantilo.charts.prepare import ChartParams
from quantilo.compare.analyze import Comparison
from quantilo.compare.service import CompareService
from quantilo.correlations.matrix import Pair
from quantilo.correlations.service import Correlations, CorrelationService
from quantilo.filters.model import Condition
from quantilo.profiling.service import ProfilingService
from quantilo.report import html
from quantilo.report.build import ColumnResult, build_report, file_name, pairs_to_draw
from quantilo.variable.service import VariableService

NUMERIC = ("numeric_continuous", "numeric_discrete")

# (fraction done 0..1, message such as "Variables 12 / 40"); it may raise to stop the report.
Progress = Callable[[float, str], None]


@dataclass
class ReportServices:
    columns: Catalog
    profiling: ProfilingService
    variable: VariableService
    correlations: CorrelationService
    compare: CompareService


def generate(
    services: ReportServices,
    dataset_id: str,
    conditions: list[Condition],
    version: str,
    progress: Progress | None = None,
) -> tuple[str, str]:
    """(HTML, file name)."""
    dataset = services.columns.ready(dataset_id)
    views = services.columns.columns(dataset_id)
    used = [v for v in views if not v.excluded and v.type != "ignored"]
    excluded = [v.name for v in views if v.excluded or v.type == "ignored"]
    steps = len(used) + 3

    def advance(done: int, message: str) -> None:
        if progress is not None:
            progress(done / steps, message)

    overview = services.profiling.overview(dataset_id, conditions)
    results: list[ColumnResult] = []
    for i, view in enumerate(used):
        advance(1 + i, f"Variables {i + 1} / {len(used)}")
        result = ColumnResult(view)
        try:
            result.profile = services.variable.profile(dataset_id, view.name, conditions)
            result.chart = services.profiling.chart(
                dataset_id, view.name, ChartParams(), conditions
            )
            if view.type == "datetime" and result.chart.kind == "timeline":
                for part, title in (("month", "month"), ("day_of_week", "day of week")):
                    if part in result.chart.views:
                        extra = services.profiling.chart(
                            dataset_id, view.name, ChartParams(view=part), conditions
                        )
                        result.extra.append((title, extra))
            if view.type in NUMERIC:
                found = services.variable.outliers(dataset_id, view.name, "iqr", conditions)
                result.outliers = next((r for r in found.rules if r.method == "iqr"), None)
        except Exception as error:  # one column must not sink the report
            result.error = str(error) or type(error).__name__
        results.append(result)
    advance(steps - 1, "Correlations")
    correlations: Correlations | None
    try:
        correlations = services.correlations.correlations(dataset_id, "pearson", None, conditions)
    except Exception:  # the report goes on without the matrix
        correlations = None
    advance(steps - 1, "Relationships")
    comparisons: list[tuple[Pair, Comparison]] = []
    for pair in pairs_to_draw(correlations.matrix if correlations else None):
        try:
            comparisons.append(
                (pair, services.compare.compare(dataset_id, pair.a, pair.b, conditions=conditions))
            )
        except Exception:  # a pair that can't be drawn is left out
            continue
    report = build_report(
        title=dataset.name,
        source=Path(dataset.source_path).name if dataset.source_path else dataset.name,
        version=version,
        overview=overview,
        results=results,
        excluded=excluded,
        correlations=correlations,
        comparisons=comparisons,
        conditions=conditions,
        now=datetime.now(),
    )
    return html.render(report), file_name(dataset.name)
