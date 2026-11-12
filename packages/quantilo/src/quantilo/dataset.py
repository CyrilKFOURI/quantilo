"""A dataset opened in Python: the same import, typing and analyses as the app, without a server.

`open()` reads the source with the app's reader into a Parquet copy kept in a cache folder, then
detects the column types with the app's engine. A `Dataset` is its own catalog (in memory) for
the analysis services the app uses; `where()` gives a filtered view sharing that catalog.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import duckdb
import pandas as pd

from quantilo.catalog import (
    ColumnChange,
    ColumnRecord,
    ColumnView,
    apply_change,
    detection_record,
    summarize_record,
    to_view,
)
from quantilo.changes.compare import Changes, compare_snapshots
from quantilo.changes.kpis import KpiChange, compare_kpi, compute_value
from quantilo.changes.relations import RelationChange, relation_changes
from quantilo.changes.rows import (
    ROW_KEY,
    RowAlignment,
    RowChanges,
    RowKeyError,
    key_candidates,
    row_changes,
)
from quantilo.changes.snapshot import Snapshot, take_snapshot
from quantilo.charts.prepare import Chart, ChartParams
from quantilo.compare.analyze import Comparison
from quantilo.compare.durations import Survival, Transitions
from quantilo.compare.multiple import Correction
from quantilo.compare.service import CompareService
from quantilo.correlations.matrix import Method
from quantilo.correlations.service import Correlations, CorrelationService
from quantilo.correlations.structure import Factors, Structure
from quantilo.errors import ColumnNotFound, ExplorerError
from quantilo.experiments.analyze import ExperimentDesign
from quantilo.experiments.decisions import DecisionsRequest, LoggedDecisions
from quantilo.experiments.facade import Experiment, MetricLike, metric_of
from quantilo.experiments.service import ExperimentCandidates, ExperimentEngine, PolicyModel
from quantilo.filters.model import Condition, Operator
from quantilo.filters.scope import scoped_source, where_for
from quantilo.ingestion.importer import import_to_parquet
from quantilo.ingestion.normalize import copy_metadata, normalized_sql
from quantilo.ingestion.options import CsvOptions, ImportOptions
from quantilo.ingestion.sources import fingerprint, list_source_files
from quantilo.ingestion.sql import lit
from quantilo.kpi.cohort import CohortResult, FunnelResult
from quantilo.kpi.compute import KpiResult
from quantilo.kpi.definition import Definition
from quantilo.kpi.service import KpiEngine, Suggestions
from quantilo.profiling.overview import Overview
from quantilo.profiling.service import ProfilingService
from quantilo.regression.analyze import (
    FamilyAlternatives,
    InfluenceTable,
    MoreChecks,
    RegressionReport,
    StandardErrorTable,
    TermsTable,
)
from quantilo.regression.families import FamilyRequest
from quantilo.regression.iv import IvTable
from quantilo.regression.panel import PanelTable
from quantilo.regression.robust import QuantileProcess, RobustTable
from quantilo.regression.rules import RegressionSuggestion, SeRequest
from quantilo.regression.service import RegressionService
from quantilo.regression.service import Request as RegressionRequest
from quantilo.regression.service import Section as RegressionSection
from quantilo.regression.terms import PartialPlot
from quantilo.report.generate import ReportServices, generate
from quantilo.rows.export import Format
from quantilo.rows.service import RowService
from quantilo.rows.view import Mode, RowsPage, View
from quantilo.text.distinctive import DistinctiveWords
from quantilo.text.service import TextService, topics_summary
from quantilo.text.topics import Topics
from quantilo.timeseries.facade import TimeSeries
from quantilo.timeseries.series import Aggregate, Align, Gaps, Period
from quantilo.timeseries.series import Options as SeriesOptions
from quantilo.timeseries.service import Request as SeriesRequest
from quantilo.timeseries.service import TimeSeriesService
from quantilo.typing.derived import derived_columns, resolve
from quantilo.typing.engine import DEFAULT_THRESHOLDS, Thresholds, detect_columns
from quantilo.typing.knowledge_base import load
from quantilo.typing.types import AnalysisType
from quantilo.variable.analyze import VariableProfile
from quantilo.variable.fitting import Fit
from quantilo.variable.outliers import OutlierMethod, Outliers
from quantilo.variable.service import VariableService
from quantilo.variable.tails import Tail
from quantilo.version import __version__

if TYPE_CHECKING:  # machine learning and synthetic data load on first use (scikit-learn, Optuna)
    from quantilo.cleaning.facade import Cleaning
    from quantilo.experiments.offpolicy import PolicyEvaluation
    from quantilo.ml.classes import Classes
    from quantilo.ml.engine import Targets
    from quantilo.ml.evaluate import Validation
    from quantilo.ml.facade import MachineLearning
    from quantilo.segments.engine import MethodChoice
    from quantilo.segments.facade import FittedSegments, FoundUnusualRows, Segments
    from quantilo.synthetic.facade import Synthesizer

DETECTION_VERSION = 3  # bump when detection changes, to ignore cached detections


def default_cache_dir() -> Path:
    if env := os.environ.get("QUANTILO_CACHE"):
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "quantilo"


@dataclass
class DatasetInfo:
    """The dataset as the analyses see it (satisfies `catalog.DatasetRef`)."""

    id: str
    name: str
    source_path: str
    storage_path: str
    imported_at: str | None
    row_count: int | None
    size_bytes: int | None
    types_version: int = 0


@dataclass
class _State:
    """What every filtered view of one dataset shares: storage, columns, services (and their
    caches)."""

    info: DatasetInfo
    records: dict[str, ColumnRecord]
    _services: dict[str, Any] = field(default_factory=dict)

    # --- Catalog ----------------------------------------------------------------------------

    def ready(self, dataset_id: str) -> DatasetInfo:
        return self.info

    def columns(self, dataset_id: str) -> list[ColumnView]:
        return [to_view(record) for record in self.records.values()]

    def service(self, kind: type) -> Any:
        if kind.__name__ not in self._services:
            self._services[kind.__name__] = kind(self)
        return self._services[kind.__name__]


class Dataset:
    """A tabular dataset with typed columns. Build one with `open()`, `from_pandas()` or
    `from_arrow()`."""

    def __init__(self, state: _State, conditions: tuple[Condition, ...] = ()) -> None:
        self._state = state
        self._conditions = conditions

    # --- What it is -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._state.info.name

    @property
    def path(self) -> Path:
        """The Parquet copy the analyses read."""
        return Path(self._state.info.storage_path)

    @property
    def conditions(self) -> list[Condition]:
        return list(self._conditions)

    @property
    def types(self) -> list[ColumnView]:
        """Every column: effective type and role, how it is read, why, and its summary."""
        return self._state.columns(self._state.info.id)

    @property
    def columns(self) -> list[str]:
        return [record.name for record in self._state.records.values()]

    def column(self, name: str) -> ColumnView:
        return to_view(self._record(name))

    def __len__(self) -> int:
        if not self._conditions:
            return self._state.info.row_count or 0
        source = scoped_source(
            self._state.info.storage_path, where_for(self.types, self.conditions)
        )
        connection = duckdb.connect()
        try:
            row = connection.execute(f"SELECT count(*) FROM {source}").fetchone()
        finally:
            connection.close()
        return int(row[0]) if row else 0

    def __repr__(self) -> str:
        filters = f", {len(self._conditions)} filter(s)" if self._conditions else ""
        info = self._state.info
        return (
            f"<Dataset {info.name!r}: {info.row_count} rows × {len(self.columns)} columns{filters}>"
        )

    # --- Correcting the types ---------------------------------------------------------------

    def retype(self, column: str, analysis_type: AnalysisType) -> Dataset:
        """Read a column in another type (same rules as the app's Columns page)."""
        return self._change(ColumnChange(name=column, type=analysis_type))

    def set_role(self, column: str, role: str | None) -> Dataset:
        """A semantic role from the knowledge base (`None`: no role)."""
        return self._change(ColumnChange(name=column, role=role or ""))

    def set_order(self, column: str, order: list[str]) -> Dataset:
        """The order of an ordinal category's values (empty: back to the detected order)."""
        return self._change(ColumnChange(name=column, order=order))

    def exclude(self, *columns: str) -> Dataset:
        """Leave columns out of the analyses (overview, correlations, report)."""
        for column in columns:
            self._change(ColumnChange(name=column, excluded=True))
        return self

    def include(self, *columns: str) -> Dataset:
        for column in columns:
            self._change(ColumnChange(name=column, excluded=False))
        return self

    def reset_types(self, *columns: str) -> Dataset:
        """Back to the detected type, role and order (every column when none is given)."""
        for column in columns or tuple(self.columns):
            self._change(ColumnChange(name=column, reset=True))
        return self

    def _record(self, name: str) -> ColumnRecord:
        record = self._state.records.get(name)
        if record is None:
            raise ColumnNotFound(name, self.columns)
        return record

    def _known(self, name: str) -> None:
        """A stored column or a derived one (058, R-DER-03); else ColumnNotFound."""
        if name in self._state.records:
            return
        if resolve({view.name: view for view in self.types}, name) is None:
            raise ColumnNotFound(name, self.columns)

    @property
    def derived(self) -> list[ColumnView]:
        """The derived columns offered (058): the length of a text, an option of a list of
        choices... Each name works wherever a column name does (filters, compare, correlations,
        KPI, segments)."""
        connection = duckdb.connect()
        try:
            source = f"read_parquet({lit(self._state.info.storage_path)})"
            return derived_columns(connection, source, self.types)
        finally:
            connection.close()

    def _change(self, change: ColumnChange) -> Dataset:
        """Changes the dataset in place (every filtered view of it sees the change)."""
        record = self._record(change.name)
        before = (record.user_type, record.user_parser, record.user_role, record.user_order)
        updated = replace(record)
        apply_change(updated, change, {role.id for role in load().roles})
        after = (updated.user_type, updated.user_parser, updated.user_role, updated.user_order)
        if before != after:
            updated.summary = summarize_record(self._state.info.storage_path, updated)
        if (before[0], before[1]) != (after[0], after[1]):
            self._state.info.types_version += 1
        self._state.records[change.name] = updated
        return self

    # --- Filters ----------------------------------------------------------------------------

    def where(
        self,
        column: str | Condition,
        op: Operator = "eq",
        value: Any = None,
        *,
        negate: bool = False,
        **options: Any,
    ) -> Dataset:
        """A filtered view: `ds.where("Sex", "in", ["female"])`, `ds.where("Age", "between",
        (18, 30))`, `ds.where("Fare", "gt", 50)`. Conditions add up (AND); the values of an
        `in` combine with OR. Same operators as the app's filter bar."""
        if isinstance(column, Condition):
            condition = column
        else:
            self._known(column)
            fields: dict[str, Any] = {"column": column, "op": op, "negate": negate, **options}
            if op in ("in", "part_in"):
                values = value if isinstance(value, list | tuple | set) else [value]
                fields["values"] = [str(v) for v in values]
            elif op == "between":
                fields["low"], fields["high"] = value
            elif value is not None:
                fields["value"] = value
            condition = Condition(**fields)
        conditions = (*self._conditions, condition)
        where_for(self.types, list(conditions))  # an invalid filter fails now, not later
        return Dataset(self._state, conditions)

    def unfiltered(self) -> Dataset:
        return Dataset(self._state)

    # --- Analyses ---------------------------------------------------------------------------

    @property
    def _id(self) -> str:
        return self._state.info.id

    def overview(self) -> Overview:
        """Rows, columns by type, missing cells, duplicates and quality alerts."""
        return self._state.service(ProfilingService).overview(self._id, self.conditions)

    def variable(self, column: str) -> VariableProfile:
        """Descriptive statistics, normality tests and the profile matching the column's type."""
        self._known(column)
        return self._state.service(VariableService).profile(self._id, column, self.conditions)

    def fit(self, column: str) -> Fit:
        """Probability laws fitted to a numeric column, ranked by AIC."""
        self._known(column)
        return self._state.service(VariableService).fit(self._id, column, self.conditions)

    def outliers(self, column: str, method: OutlierMethod = "iqr") -> Outliers:
        """Outliers of a numeric column by every rule, with the rows of `method`."""
        self._known(column)
        return self._state.service(VariableService).outliers(
            self._id, column, method, self.conditions
        )

    def tail(self, column: str) -> Tail:
        """The right tail of a numeric column: a generalised Pareto law above a threshold,
        return levels ("1 in 1,000 rows exceed X") with intervals, Hill and mean-excess curves,
        VaR and Expected Shortfall at 95 and 99 %. `.rule` says whether the app shows it."""
        self._known(column)
        return self._state.service(VariableService).tail(self._id, column, self.conditions)

    def chart(
        self, column: str, view: str | None = None, *, top_n: int = 10, bins: int | None = None
    ) -> Chart:
        """The data of a column's chart, as on the Overview cards (`view`: one of `.views`)."""
        self._known(column)
        params = ChartParams(view=view, top_n=top_n, bins=bins)
        return self._state.service(ProfilingService).chart(
            self._id, column, params, self.conditions
        )

    def plot(
        self,
        column: str,
        view: str | None = None,
        *,
        theme: Literal["light", "dark"] = "light",
        title: str | None = None,
        height: int | None = None,
        **options: Any,
    ) -> Any:
        """A column drawn as in the app (a plotly Figure, the `plot` extra). `view`: the card's
        views (histogram, box, pie, timeline, month…), and for a number kde, ecdf, violin, qq or
        laws (the best fitted laws over the histogram); calendar for a date, words for text."""
        from quantilo.plots import PlotError, plot
        from quantilo.results import to_dict

        record = self.column(column)
        numeric_views = ("kde", "ecdf", "violin", "qq")
        if view in numeric_views or (view == "box" and record.type.startswith("numeric")):
            return plot(self.variable(column), view, theme=theme, title=title, height=height)
        if view in ("calendar", "words"):
            return plot(self.variable(column), view, theme=theme, title=title, height=height)
        if view == "laws":
            if not record.type.startswith("numeric"):
                raise PlotError(f"{column!r} is not numeric: no law to fit.")
            histogram = to_dict(self.chart(column, "histogram"))
            profile = self.variable(column)
            count = profile.numeric.describe.count if profile.numeric else 0
            return plot(
                self.fit(column),
                "laws",
                theme=theme,
                title=title,
                height=height,
                bins=histogram["bins"],
                count=count,
            )
        pie = view == "pie"
        chart = self.chart(column, None if pie else view)
        return plot(
            chart,
            "pie" if pie else None,
            theme=theme,
            title=title,
            height=height,
            analysis_type=record.type,
            **options,
        )

    def compare(
        self,
        x: str,
        y: str,
        z: str | None = None,
        *,
        color: str | None = None,
        options: SeriesOptions | None = None,
        numbers: list[str] | None = None,
        correction: Correction = "auto",
    ) -> Comparison:
        """Two or three variables: the visual and the tests follow their types; groups get
        bootstrap intervals by rule and pairwise differences adjusted by `correction`."""
        for name in (x, y, z, color):
            if name is not None:
                self._known(name)
        return self._state.service(CompareService).compare(
            self._id, x, y, z, color, self.conditions, options, numbers, correction
        )

    def survival(
        self,
        duration: str | None = None,
        event: str | None = None,
        *,
        start: str | None = None,
        end: str | None = None,
        group: str | None = None,
        event_value: str | None = None,
    ) -> Survival:
        """Kaplan-Meier curves with 95 % intervals, median survival and the log-rank test across
        `group`. Without columns, the reading a rule detects (R-SRV-01/02), if any."""
        for name in (duration, event, start, end, group):
            if name is not None:
                self._record(name)
        return self._state.service(CompareService).survival(
            self._id, self.conditions, duration, event, start, end, group, event_value
        )

    def transitions(
        self, entity: str | None = None, time: str | None = None, status: str | None = None
    ) -> Transitions:
        """Status transitions as a Markov chain: matrix, stationary shares, and the expected
        steps before an absorbing state. Without columns, the reading R-MKV-01 detects."""
        for name in (entity, time, status):
            if name is not None:
                self._record(name)
        return self._state.service(CompareService).transitions(
            self._id, self.conditions, entity, time, status
        )

    def correlations(
        self,
        method: Method = "pearson",
        columns: list[str] | None = None,
        correction: Correction = "auto",
    ) -> Correlations:
        """The mixed association matrix (r or ρ, Cramér's V, η), ranked pairs and insights;
        p-values adjusted over the pairs (`correction`: auto, bh, holm, bonferroni)."""
        for name in columns or []:
            self._known(name)
        return self._state.service(CorrelationService).correlations(
            self._id, method, columns, self.conditions, correction
        )

    def structure(self, columns: list[str] | None = None) -> Structure:
        """How many real dimensions the numeric columns carry: standardised PCA, components kept
        by Horn's parallel analysis, loadings, a sentence, a sampled biplot (078)."""
        for name in columns or []:
            self._record(name)
        return self._state.service(CorrelationService).structure(self._id, columns, self.conditions)

    def factors(self, columns: list[str] | None = None, k: int | None = None) -> Factors:
        """Factor analysis of the numeric columns with varimax rotation, communalities, KMO and
        Bartlett's test (078). `k=None`: as many factors as the parallel analysis keeps."""
        for name in columns or []:
            self._record(name)
        return self._state.service(CorrelationService).factors(
            self._id, columns, self.conditions, k
        )

    # --- Text (059) -------------------------------------------------------------------------

    def distinctive_words(self, column: str, by: str) -> DistinctiveWords:
        """The words over-represented in each group of `by` among the texts of `column`
        (R-WRD-01 to 06): what Compare shows for a text and a category."""
        found = self.compare(by, column).words
        assert found is not None
        return found

    def topics(self, column: str, k: int | None = None) -> Topics:
        """Topics of a free text column (TF-IDF then NMF, R-TOP-01 to 05), `k` of them or as
        many as R-TOP-03 says. `.agreement`: how often the column's rule matches the model."""
        self._record(column)
        return self._state.service(TextService).topics(self._id, column, k, self.conditions)

    def keep_topics(self, topics: Topics) -> Dataset:
        """Keep these topics on their column (R-TOP-06): its Topic column becomes a derived
        column, usable everywhere a column name is."""
        record = self._record(topics.column)
        summary = json.loads(record.summary) if record.summary else {}
        summary["topics"] = topics_summary(topics)
        self._state.records[topics.column] = replace(record, summary=json.dumps(summary))
        self._state.info.types_version += 1  # analyses cached on the old topics are stale
        return self

    def _regression_service(self) -> RegressionService:
        services = self._state._services
        if "RegressionService" not in services:
            correlations = self._state.service(CorrelationService)
            services["RegressionService"] = RegressionService(self._state, correlations)
        return services["RegressionService"]

    def regression(
        self,
        target: str | None = None,
        columns: list[str] | None = None,
        *,
        weights: str | None = None,
        cluster: str | None = None,
        se: SeRequest = "auto",
        focus: str | None = None,
        family: FamilyRequest = "auto",
        exposure: str | None = None,
        terms: list[str] | None = None,
        center: bool = False,
        coding: Literal["reference", "deviation"] = "reference",
    ) -> RegressionReport:
        """Linear regression with inference (067): OLS, WLS for a weight column, logistic for a
        yes/no target; standard errors chosen by rule (HC3, clustered for a panel or a
        `cluster` column, HAC for a dated series), adjusted p-values, diagnostics in sentences.
        Without `target` or `columns`, they are picked from the correlations (R-REG-20/21).
        `weights=""` turns automatic weights off; `se` forces classic, hc0…hc3, hac, cluster,
        cluster2 or dk. `family` (069) forces ols, logit, probit, poisson, negbin, gamma,
        log_ols, boxcox, fractional or mnlogit instead of the rules' choice (R-REG-01…07);
        `exposure` names the offset of a count model (`""`: none, default: a column of role
        exposure). `terms` (070): `spline:x`, `square:x`, `cube:x`, `log:x`, `interaction:a|b`,
        `trend:date`, `month:date`, `weekday:date`, `reference:column|level`; `center` centres
        the numbers at their mean; `coding="deviation"` reads each level against the average of
        the levels."""
        request = RegressionRequest(
            target,
            tuple(columns) if columns is not None else None,
            weights,
            cluster,
            se,
            focus,
            family,
            exposure,
            tuple(terms or ()),
            center,
            coding,
        )
        return self._regression_service().regression(self._id, request, self.conditions)

    def regression_suggestion(self, target: str | None = None) -> RegressionSuggestion:
        """The target and up to 5 columns the regression would pick, each with its reason."""
        return self._regression_service().suggestion(self._id, target, self.conditions)

    def regression_section(
        self,
        section: RegressionSection,
        target: str | None = None,
        columns: list[str] | None = None,
        *,
        weights: str | None = None,
        cluster: str | None = None,
        se: SeRequest = "auto",
        family: FamilyRequest = "auto",
        exposure: str | None = None,
        terms: list[str] | None = None,
        center: bool = False,
        coding: Literal["reference", "deviation"] = "reference",
    ) -> (
        StandardErrorTable
        | InfluenceTable
        | TermsTable
        | MoreChecks
        | PanelTable
        | FamilyAlternatives
        | QuantileProcess
    ):
        """An Advanced section of the same regression: standard_errors (every kind side by side),
        influence (the 20 most influential rows, DFBETAS), terms (references, left out, VIF),
        more (White's test, the fit without the influential rows), panel (071: fixed, first
        differences, between and random effects, Mundlak and Hausman), families (069: probit,
        Box-Cox, fractional and multinomial logit on the same rows), quantiles (072: the
        coefficients at τ = 0.1…0.9, Student-t errors)."""
        request = RegressionRequest(
            target,
            tuple(columns) if columns is not None else None,
            weights,
            cluster,
            se,
            family=family,
            exposure=exposure,
            terms=tuple(terms or ()),
            center=center,
            coding=coding,
        )
        return self._regression_service().section(self._id, section, request, self.conditions)

    def regression_robust(
        self,
        target: str | None = None,
        columns: list[str] | None = None,
        *,
        quantile: float = 0.5,
        weights: str | None = None,
        cluster: str | None = None,
        se: SeRequest = "auto",
        family: FamilyRequest = "auto",
        terms: list[str] | None = None,
        center: bool = False,
    ) -> RobustTable:
        """Robust and quantile regression (072): Huber and Tukey (RLM) and the `quantile`
        (0.1, 0.25, 0.5, 0.75 or 0.9) side by side with least squares, and the law and tail of the
        residuals."""
        request = RegressionRequest(
            target,
            tuple(columns) if columns is not None else None,
            weights,
            cluster,
            se,
            family=family,
            terms=tuple(terms or ()),
            center=center,
        )
        return self._regression_service().robust(self._id, request, quantile, self.conditions)

    def regression_partial(
        self,
        column: str,
        target: str | None = None,
        columns: list[str] | None = None,
        *,
        weights: str | None = None,
        terms: list[str] | None = None,
        center: bool = False,
    ) -> PartialPlot:
        """Added-variable and component + residual plots of one column of the regression (070):
        least squares only, 2 000 points at most."""
        request = RegressionRequest(
            target,
            tuple(columns) if columns is not None else None,
            weights,
            terms=tuple(terms or ()),
            center=center,
        )
        return self._regression_service().partial(self._id, request, column, self.conditions)

    def regression_iv(
        self,
        target: str | None = None,
        columns: list[str] | None = None,
        *,
        endogenous: list[str],
        instruments: list[str],
        weights: str | None = None,
        cluster: str | None = None,
        se: SeRequest = "auto",
    ) -> IvTable:
        """Two-stage least squares (071), only with the endogenous columns (among `columns`) and
        the instruments (outside them) that you name: standard errors by the same rule as
        `regression`, first-stage and effective F, Hansen J or Sargan, Wu-Hausman and
        Anderson-Rubin tests."""
        request = RegressionRequest(
            target, tuple(columns) if columns is not None else None, weights, cluster, se
        )
        return self._regression_service().instruments(
            self._id, request, tuple(endogenous), tuple(instruments), self.conditions
        )

    def rows(
        self,
        offset: int = 0,
        limit: int = 100,
        *,
        mode: Mode = "first",
        search: str = "",
        sort: str | None = None,
        desc: bool = False,
    ) -> RowsPage:
        """A page of typed rows (`mode`: first, sample, duplicates, missing)."""
        view = View(mode=mode, search=search, sort=sort, desc=desc)
        return self._state.service(RowService).rows(self._id, view, offset, limit, self.conditions)

    def to_pandas(self, columns: list[str] | None = None) -> pd.DataFrame:
        """The (filtered) rows as a pandas DataFrame, each column read in its type."""
        service: RowService = self._state.service(RowService)
        exported = service.export(self._id, "parquet", View(), columns, self.conditions)
        try:
            return pd.read_parquet(exported.path)
        finally:
            exported.remove()

    def export(
        self,
        path: str | os.PathLike[str],
        file_format: Format | None = None,
        *,
        columns: list[str] | None = None,
    ) -> Path:
        """Write the (filtered) rows to CSV, Parquet or Excel; the format follows the extension."""
        target = Path(path)
        chosen = file_format or _format_of(target)
        service: RowService = self._state.service(RowService)
        exported = service.export(self._id, chosen, View(), columns, self.conditions)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(exported.path, target)
        finally:
            exported.remove()
        return target

    def timeseries(
        self,
        time: str | None = None,
        value: str | None = None,
        *,
        group: str | None = None,
        series: str | None = None,
        period: Period | Literal["auto"] = "auto",
        aggregate: Aggregate | Literal["auto"] = "auto",
        window: int | None = None,
        align: Align = "trailing",
        gaps: Gaps = "leave",
        season: int | None = None,
        methods: list[str] | None = None,
    ) -> TimeSeries:
        """A value over time; unset choices follow the app's rules (`.setup()` shows them).
        `value=None` counts rows per period."""
        for name in (time, value, group):
            if name is not None:
                self._record(name)
        options = SeriesOptions(period=period, aggregate=aggregate, window=window, align=align)
        request = SeriesRequest(time, value, group, series, options, gaps, season)
        service = self._state.service(TimeSeriesService)
        return TimeSeries(service, self._id, request, self.conditions, methods)

    def kpi_suggestions(self) -> Suggestions:
        """KPIs suggested from the columns, and the best columns to split them by."""
        return self._state.service(KpiEngine).suggestions(self._id)

    def kpi(self, definition: Definition | dict[str, Any]) -> KpiResult:
        """Compute a KPI (a suggested `definition`, or one written as a dict): the grain, chart,
        totals and comparisons are chosen by the app's rules unless set."""
        chosen = (
            definition
            if isinstance(definition, Definition)
            else Definition.model_validate(definition)
        )
        return self._state.service(KpiEngine).compute(self._id, chosen, self.conditions)

    def kpi_cohort(
        self, customer: str, date: str, grain: Literal["auto", "month", "week"] = "auto"
    ) -> CohortResult:
        """Retention by cohort (060): customers grouped by the period of their first row, the
        share seen again in each later period, and new vs returning customers per period."""
        engine = self._state.service(KpiEngine)
        return engine.cohort(self._id, customer, date, grain, self.conditions)

    def kpi_funnel(self, column: str, stages: list[str], key: str | None = None) -> FunnelResult:
        """The funnel of an ordered status (060): the distinct `key` values (or rows) that reached
        each stage, a later stage counting as having passed the earlier ones."""
        engine = self._state.service(KpiEngine)
        return engine.funnel(self._id, column, stages, key, self.conditions)

    def experiment_candidates(self) -> ExperimentCandidates:
        """The columns that can describe an experiment: variants (2 to 5 values) with the control
        guessed, metrics with their kind, units that repeat, dates and subgroups."""
        return self._state.service(ExperimentEngine).candidates(self._id, self.conditions)

    def experiment(
        self,
        variant: str,
        metrics: MetricLike | list[MetricLike],
        *,
        unit: str | None = None,
        control: str | None = None,
        primary: str | None = None,
        expected: dict[str, float] | None = None,
        segment: str | None = None,
        time: str | None = None,
        alpha: float = 0.05,
    ) -> Experiment:
        """An A/B test: `variant` names each unit's group, `metrics` are column names (or
        `{"column": "revenue", "denominator": "orders"}` for a ratio, `"pre": "revenue_before"`
        for CUPED). `unit`: the column naming a user when one has several rows; `expected`: the
        planned split (equal by default); `.analyze()` runs it."""
        chosen = [metric_of(m) for m in (metrics if isinstance(metrics, list) else [metrics])]
        for name in (variant, unit, segment, time):
            if name is not None:
                self._record(name)
        for metric in chosen:
            for name in (metric.column, metric.denominator, metric.pre):
                if name is not None:
                    self._record(name)
        design = ExperimentDesign(
            variant=variant,
            metrics=chosen,
            primary=primary,
            control=control,
            unit=unit,
            expected=expected,
            segment=segment,
            time=time,
            alpha=alpha,
        )
        engine = self._state.service(ExperimentEngine)
        return Experiment(engine, self._id, design, self.conditions)

    def logged_decisions(self) -> LoggedDecisions:
        """Does this dataset look like logged decisions (055)? An action column (2 to 10
        actions), its reward, columns describing the units, optionally the propensity."""
        return self._state.service(ExperimentEngine).decisions(self._id, self.conditions)

    def evaluate_policy(
        self,
        action: str,
        reward: str,
        *,
        propensity: str | None = None,
        covariates: list[str] | None = None,
        models: list[Any] | None = None,
        learn: bool = False,
        alpha: float = 0.05,
    ) -> PolicyEvaluation:
        """Evaluate policies from logged decisions (055): the logged one, always each action,
        the rule learnt from the log when `learn` (slower) and each of `models` (trained or
        reloaded with `dx.load_model`: one that
        predicts the action, or the reward from the action), with the DM, IPS, SNIPS and DR
        estimators and their intervals, the diagnostics first. Without `propensity`, it is
        estimated: valid only if no hidden factor drove the decisions."""
        for name in (action, reward, propensity, *(covariates or [])):
            if name is not None:
                self._record(name)
        request = DecisionsRequest(
            action=action,
            reward=reward,
            propensity=propensity,
            covariates=covariates,
            learn=learn,
            alpha=alpha,
        )
        policies = [
            PolicyModel(getattr(m.card, "model_label", None) or m.card.model, m.pipeline, m.card)
            for m in models or []
        ]
        return self._state.service(ExperimentEngine).evaluate_policy(
            self._id, request, self.conditions, policies
        )

    def ml_targets(self) -> Targets:
        """Every column as a possible target: the problem it makes (regression, binary,
        multiclass) or why it cannot be one."""
        from quantilo.ml.engine import MlEngine

        return self._state.service(MlEngine).targets(self._id, self.conditions)

    def ml(
        self,
        target: str,
        *,
        classes: Classes | dict[str, Any] | None = None,
        validation: Validation | dict[str, Any] | None = None,
    ) -> MachineLearning:
        """Models predicting `target`: `.train()`, `.compare()`, `.tune()`, `.preview()`.
        `validation`: e.g. `{"method": "kfold", "folds": 10, "metric": "f1_macro"}`;
        `classes`: a number cut into classes, e.g. `{"edges": [18, 65], "names": ["young",
        "adult", "senior"]}`."""
        from quantilo.ml.engine import MlEngine
        from quantilo.ml.facade import MachineLearning

        self._record(target)
        engine = self._state.service(MlEngine)
        return MachineLearning(engine, self._id, target, self.conditions, classes, validation)

    def segments(
        self,
        columns: list[str] | None = None,
        *,
        method: MethodChoice = "auto",
        k: int | Literal["auto"] = "auto",
        preparation: Any = None,
        seed: int = 42,
    ) -> Segments:
        """Groups of similar rows: `.plan()` (columns and method chosen, with reasons), `.fit()`.
        `columns=None` lets the app choose; `method`: auto, kmeans, kprototypes, gmm, ward,
        hdbscan; `preparation`: the ML preparation choices, e.g. `{"columns": {"age": {"scaling":
        "standard"}}}`."""
        from quantilo.segments import facade
        from quantilo.segments.engine import SegmentsEngine

        for name in columns or []:
            self._known(name)
        engine = self._state.service(SegmentsEngine)
        chosen = facade.request(columns, method, k, preparation, seed)
        return facade.Segments(self, engine, self._id, self.conditions, chosen)

    def unusual_rows(
        self,
        columns: list[str] | None = None,
        *,
        method: Literal["auto", "isolation_forest", "mahalanobis", "lof", "segment"] = "auto",
        segments: FittedSegments | None = None,
        preparation: Any = None,
        seed: int = 42,
    ) -> FoundUnusualRows:
        """Rows unlike the others across several columns: a score per row, a proposed threshold
        and, for the most unusual rows, the columns that make them so. `method="segment"`
        measures each row from its own segment's centre (pass `segments=ds.segments().fit()`)."""
        from quantilo.segments import facade
        from quantilo.segments.engine import SegmentsEngine

        for name in columns or []:
            self._known(name)
        engine = self._state.service(SegmentsEngine)
        chosen = facade.unusual_request(columns, method, preparation, seed)
        labels = segments.as_labels() if segments is not None else None
        return facade.FoundUnusualRows(engine.unusual(self._id, self.conditions, chosen, labels))

    def synthetic(self, use_gpu: bool | None = None) -> Synthesizer:
        """Synthetic copies of the whole dataset (filters do not apply, as in the app).
        `use_gpu`: whether the deep methods may use the GPU (`None`: when PyTorch sees one)."""
        from quantilo.synthetic.engine import SyntheticEngine
        from quantilo.synthetic.facade import Synthesizer, gpu_compute

        key = f"SyntheticEngine:{use_gpu}"
        services = self._state._services
        if key not in services:
            found: list[tuple[str | None, bool]] = []

            def compute() -> tuple[str | None, bool]:
                if not found:
                    found.append(gpu_compute(use_gpu))
                return found[0]

            services[key] = SyntheticEngine(self._state, compute)
        return Synthesizer(self.unfiltered(), services[key], self._id)

    # --- Cleaning (065) ----------------------------------------------------------------------

    def clean(self, steps: list[Any] | None = None) -> Cleaning:
        """What to clean, as the app's Clean page: `.suggestions` (each with its reason and its
        step), `.steps` (the automatic ones to start with), `.use()` / `.skip()` a suggestion,
        `.add(op=…)` a step, `.preview()`, then `.apply()` for the cleaned dataset (this one is
        never changed). Whole dataset: filters do not apply."""
        from quantilo.cleaning.facade import Cleaning

        return Cleaning(self, steps)

    # --- Versions (031) ----------------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        """The statistical photo the app keeps of every version: schema, types, rows, missing
        values, quantiles, top categories, periods, duplicates. Whole dataset (no filters)."""
        info = self._state.info
        key = f"Snapshot:{info.imported_at}:{info.types_version}"
        services = self._state._services
        if key not in services:
            connection = duckdb.connect()
            try:
                source = f"read_parquet({lit(info.storage_path)})"
                services[key] = take_snapshot(
                    connection, source, self.types, info.imported_at, info.types_version
                )
            finally:
                connection.close()
        return services[key]

    def changes(
        self,
        other: Dataset,
        *,
        used: dict[str, list[str]] | None = None,
        focus: list[str] | None = None,
    ) -> Changes:
        """What changed since `other` (the previous version, or a file to compare with): schema
        with guessed renames, quality, dates, distributions ranked by PSI, and a level (No
        notable change, Check, Breaking). `used`: column → what depends on it (a column used
        that disappears is Breaking); `focus`: columns listed first."""
        return compare_snapshots(
            other.snapshot(),
            self.snapshot(),
            before_label=other.name,
            after_label=self.name,
            used=used,
            focus=focus or [],
        )

    def row_changes(
        self, other: Dataset, key: str | None = None, alignment: RowAlignment | None = None
    ) -> RowChanges:
        """Rows added, removed and modified since `other`, joined on `key` (by default the first
        column that identifies a row in both), or paired by a cleaning's row map (`alignment`,
        e.g. `RowAlignment(None, cleaning.rowmap)` for a cleaned version against its original)."""
        chosen = key or ROW_KEY
        if alignment is None and key is None:
            chosen = next(iter(key_candidates(other.snapshot(), self.snapshot())), None)
        if chosen is None:
            raise RowKeyError("No column identifies a row in both versions: give a key.")
        connection = duckdb.connect()
        try:
            return row_changes(
                connection,
                str(other.path),
                other.types,
                str(self.path),
                self.types,
                chosen,
                alignment,
            )
        finally:
            connection.close()

    def relation_changes(self, other: Dataset) -> list[RelationChange]:
        """The 10 strongest relations of each version, with their value in the other."""
        connection = duckdb.connect()
        try:
            return relation_changes(
                connection, str(other.path), other.types, str(self.path), self.types
            )
        finally:
            connection.close()

    def kpi_changes(
        self, other: Dataset, kpis: dict[str, Definition | dict[str, Any]]
    ) -> list[KpiChange]:
        """KPIs (name → definition) computed on both versions, with the periods they share."""
        found: list[KpiChange] = []
        for name, raw in kpis.items():
            definition = raw if isinstance(raw, Definition) else Definition.model_validate(raw)
            values = []
            problem = None
            for side in (other, self):
                connection = duckdb.connect()
                try:
                    source = f"read_parquet({lit(str(side.path))})"
                    values.append(compute_value(connection, source, side.types, name, definition))
                except ExplorerError as error:
                    values.append(None)
                    problem = error.message
                finally:
                    connection.close()
            found.append(compare_kpi(name, values[0], values[1], problem, values[1] is None))
        return found

    def report(self, path: str | os.PathLike[str] | None = None) -> str:
        """The app's HTML report (one self-contained file); written to `path` when given.
        Returns the HTML."""
        state = self._state
        services = ReportServices(
            state,
            state.service(ProfilingService),
            state.service(VariableService),
            state.service(CorrelationService),
            state.service(CompareService),
        )
        content, _name = generate(services, self._id, self.conditions, __version__)
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return content


def _format_of(path: Path) -> Format:
    suffix = path.suffix.lower()
    formats: dict[str, Format] = {".csv": "csv", ".parquet": "parquet", ".xlsx": "xlsx"}
    if suffix not in formats:
        raise ExplorerError(f"Unknown export format {suffix!r}: use .csv, .parquet or .xlsx.")
    return formats[suffix]


# --- Opening ---------------------------------------------------------------------------------


def open(
    path: str | os.PathLike[str],
    *,
    name: str | None = None,
    delimiter: str | None = None,
    encoding: str | None = None,
    decimal: str | None = None,
    header: bool = True,
    sheets: list[str] | None = None,
    files: list[str] | None = None,
    columns: list[str] | None = None,
    sample_pct: float = 100,
    cache_dir: str | os.PathLike[str] | None = None,
    refresh: bool = False,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Dataset:
    """Open a CSV, Parquet or Excel file, or a folder of them, and detect the column types.

    The CSV dialect is detected unless given (`delimiter`, `encoding`, `decimal`). The source is
    copied once to Parquet in `cache_dir`; opening it again while it is unchanged reuses the copy
    and the detection (`refresh=True` reads it again).
    """
    source = Path(path).expanduser().resolve()
    csv = None
    if delimiter or encoding or decimal or not header:
        csv = CsvOptions.model_validate(
            {
                "delimiter": delimiter or ",",
                "encoding": encoding or "utf-8",
                "decimal": decimal or ".",
                "header": header,
            }
        )
    options = ImportOptions(
        csv=csv, sheets=sheets, files=files, columns=columns, sample_pct=sample_pct
    )
    found = list_source_files(source, options.files)
    key = fingerprint(found, options.model_dump(mode="json"))
    folder = Path(cache_dir) if cache_dir else default_cache_dir()
    storage = folder / f"{key[:24]}.parquet"
    if refresh or not storage.is_file():
        import_to_parquet(found, options, storage)
    dataset_name = (name or "").strip() or (source.stem if source.is_file() else source.name)
    return _load(storage, dataset_name, str(source), thresholds, refresh)


def _load(
    storage: Path, name: str, source_path: str, thresholds: Thresholds, refresh: bool = False
) -> Dataset:
    stat = storage.stat()
    stamp = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
    connection = duckdb.connect()
    try:
        parquet = f"read_parquet({lit(str(storage))})"
        row = connection.execute(f"SELECT count(*) FROM {parquet}").fetchone()
        info = DatasetInfo(
            id=storage.stem,
            name=name,
            source_path=source_path,
            storage_path=str(storage),
            imported_at=stamp,
            row_count=int(row[0]) if row else 0,
            size_bytes=stat.st_size,
        )
        records = None if refresh else _cached_detection(storage, stamp, thresholds)
        if records is None:
            records = []
            for detection in detect_columns(connection, parquet, thresholds):
                record = detection_record(info.id, detection)
                record.summary = summarize_record(str(storage), record, connection)
                records.append(record)
            _save_detection(storage, stamp, thresholds, records)
    finally:
        connection.close()
    return Dataset(_State(info, {record.name: record for record in records}))


def _detection_file(storage: Path) -> Path:
    return storage.with_suffix(".columns.json")


def _cached_detection(
    storage: Path, stamp: str, thresholds: Thresholds
) -> list[ColumnRecord] | None:
    try:
        saved = json.loads(_detection_file(storage).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if saved.get("key") != [DETECTION_VERSION, stamp, asdict(thresholds)]:
        return None
    return [ColumnRecord(**record) for record in saved["columns"]]


def _save_detection(
    storage: Path, stamp: str, thresholds: Thresholds, records: list[ColumnRecord]
) -> None:
    payload = {
        "key": [DETECTION_VERSION, stamp, asdict(thresholds)],
        "columns": [asdict(record) for record in records],
    }
    _detection_file(storage).write_text(json.dumps(payload), encoding="utf-8")


def from_arrow(
    table: Any,
    name: str = "dataset",
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Dataset:
    """A dataset from a pyarrow Table (or anything DuckDB reads: a pandas or polars frame)."""
    folder = Path(cache_dir) if cache_dir else default_cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.register("source_table", table)
        partial = folder / "from-memory.parquet.partial"
        # R-ING-01 (092): times with a time zone stored in UTC, as on import
        sql, notes = normalized_sql(connection, "SELECT * FROM source_table")
        connection.execute(
            f"COPY ({sql}) TO {lit(str(partial))} "
            f"(FORMAT parquet, COMPRESSION zstd{copy_metadata(notes)})"
        )
    finally:
        connection.close()
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()[:24]
    storage = folder / f"{digest}.parquet"
    if storage.is_file():
        partial.unlink()
    else:
        partial.replace(storage)
    return _load(storage, name, "", thresholds)


def from_pandas(
    frame: Any,
    name: str = "dataset",
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Dataset:
    """A dataset from a pandas DataFrame (the index is left out)."""
    return from_arrow(
        frame.reset_index(drop=True), name, cache_dir=cache_dir, thresholds=thresholds
    )
