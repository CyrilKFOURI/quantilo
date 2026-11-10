"""Time series page: one series of a dataset, read through the active filters, then analysed."""

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import duckdb
import numpy as np

from quantilo.catalog import Catalog
from quantilo.compare.analyze import (
    MAX_SERIES,
    CompareColumn,
    Comparer,
    number_sql,
)
from quantilo.errors import ExplorerError
from quantilo.filters.model import Condition, filters_key
from quantilo.filters.scope import scoped_source, where_for
from quantilo.ingestion.sql import lit
from quantilo.timeseries import autocorrelation as ac
from quantilo.timeseries import breaks as br
from quantilo.timeseries import builder, choice, methods
from quantilo.timeseries import drivers as dv
from quantilo.timeseries import forecast as fc
from quantilo.timeseries import groups as gr
from quantilo.timeseries import hierarchy as hi
from quantilo.timeseries import residuals as rs
from quantilo.timeseries import series as ts
from quantilo.timeseries import smoothing as sm
from quantilo.timeseries import stationarity as st
from quantilo.timeseries import volatility as vo
from quantilo.timeseries.decomposition import (
    Decomposition,
    DecompositionError,
    decompose,
    trend_component,
)
from quantilo.timeseries.forecasting import (
    ForecastJob,
    ForecastJobs,
    future_periods,
)
from quantilo.timeseries.methods import MethodResult
from quantilo.timeseries.seasonality import (
    SeasonalProfile,
    YearOverlay,
    period_end,
    profiles,
    starts,
    year_overlay,
)
from quantilo.variable.source import clean

DATES = {"datetime"}
NUMBERS = {"numeric_continuous", "numeric_discrete"}
CATEGORIES = {"categorical_nominal", "categorical_ordinal", "boolean"}
SUM_ROLES = ts.SUM_ROLES
CACHE_SIZE = 16
MAX_CANDIDATES = 50  # columns read when suggesting drivers
MAX_PARENTS = 20  # columns checked as parents of the group (039)


class TimeSeriesError(ExplorerError):
    def __init__(self, message: str, code: str = "time_series") -> None:
        super().__init__(message, code=code, status_code=422)


@dataclass
class Request:
    time: str | None = None
    value: str | None = None
    group: str | None = None
    series: str | None = None  # the group analysed, when grouped
    options: ts.Options = field(default_factory=ts.Options)
    gaps: ts.Gaps = "leave"
    season: int | None = None  # cycle length override, in periods


@dataclass
class SeriesSetup:
    """The configuration actually used: every field chosen or defaulted."""

    time: str
    value: str
    group: str | None
    series: str | None
    dates: list[str]
    numbers: list[str]
    categories: list[str]
    groups: list[str]  # labels of the group column with their own series
    period: ts.Period
    aggregate: ts.Aggregate
    period_auto: bool
    aggregate_auto: bool
    coarsened: bool
    first_partial: bool
    last_partial: bool
    gaps: ts.Gaps
    season: int | None
    periods: int
    rows: int  # rows with a date and a value
    total: int  # rows after the filters
    # 093: why this date, value and aggregate (R-TS-01 to R-TS-06), shown under the chart
    reasons: list[str] = field(default_factory=list)


@dataclass
class SeriesLine:
    label: str | None
    values: list[float | None]
    counts: list[int]
    rolling_mean: list[float | None] = field(default_factory=list)
    rolling_std: list[float | None] = field(default_factory=list)
    # Gaps filled by Kalman (064): 95 % band at the filled periods (None elsewhere), and how
    filled_low: list[float | None] = field(default_factory=list)
    filled_high: list[float | None] = field(default_factory=list)
    filled_by: str | None = None


@dataclass
class SeriesFigures:
    start: str
    end: str
    periods: int
    missing: int  # periods without any row
    change: float | None  # last quarter of the series against the first, in %


@dataclass
class SeriesOverview:
    setup: SeriesSetup
    periods: list[str]
    analyzed: SeriesLine
    lines: list[SeriesLine]  # every group, for the overlay (empty when not grouped)
    figures: SeriesFigures
    window: int | None
    align: ts.Align
    methods: list[MethodResult]
    message: str | None = None


@dataclass
class SeriesSmoothing:
    """Smoothed curves of the analysed series (064), the methods suggested by the data and the
    ones that do not apply to this period."""

    setup: SeriesSetup
    periods: list[str]
    curves: list[sm.Curve]
    suggestions: list[sm.Suggestion]
    unavailable: dict[str, str]
    message: str | None = None


@dataclass
class SeriesSeasonality:
    setup: SeriesSetup
    detrend: str  # how the trend was removed before comparing seasons
    profiles: list[SeasonalProfile]
    overlay: YearOverlay | None
    methods: list[MethodResult]
    filled: int  # empty periods interpolated for the methods
    message: str | None = None


@dataclass
class SeriesDecomposition:
    setup: SeriesSetup
    periods: list[str]
    observed: list[float | None]
    result: Decomposition | None
    filled: int
    message: str | None = None


@dataclass
class SeriesStationarity:
    setup: SeriesSetup
    transform: st.Transform
    regression: st.Regression  # deterministic terms of ADF, KPSS and PP: level (c) or trend (ct)
    periods: list[str]  # periods of the transformed series (differences drop the first ones)
    transformed: list[float]
    synthesis: st.Synthesis | None
    suggestion: st.Suggestion | None
    methods: list[MethodResult]
    filled: int
    message: str | None = None


@dataclass
class SeriesAutocorrelation:
    setup: SeriesSetup
    transform: st.Transform
    correlogram: ac.Correlogram | None
    ar: ac.ArFit | None
    methods: list[MethodResult]
    filled: int
    message: str | None = None


@dataclass
class SeriesVolatility:
    setup: SeriesSetup
    residuals_from: str  # "STL remainder" or "straight-line residuals"
    periods: list[str]
    residuals: list[float]
    rolling_std: list[float | None]
    window: int
    rising: bool | None
    log_helps: bool
    methods: list[MethodResult]
    filled: int
    message: str | None = None


@dataclass
class SeriesResiduals:
    setup: SeriesSetup
    distribution: rs.Distribution | None
    filled: int
    message: str | None = None


@dataclass
class BreakPoint:
    index: int
    period: str
    before: float  # level at the end of the segment that ends (its mean, or its line)
    after: float  # level at the start of the segment that starts
    slope_before: float = 0.0  # per period (trend cost)
    slope_after: float = 0.0


@dataclass
class AnomalyRow:
    index: int
    period: str
    start: str  # ISO bounds of the period, to filter its rows
    end: str
    value: float
    expected: float
    score: float


@dataclass
class SeriesBreaks:
    setup: SeriesSetup
    periods: list[str]
    values: list[float]  # the series searched (without its seasonality when it has one)
    deseasonalized: bool
    breaks: br.Breaks | None
    points: list[BreakPoint]
    methods: list[MethodResult]
    anomalies: list[AnomalyRow]
    anomaly_method: br.Anomaly
    threshold: float
    residuals_from: str
    filled: int
    message: str | None = None


@dataclass
class SeriesDrivers:
    """The columns that could explain the series (028 A4), most correlated first."""

    setup: SeriesSetup
    horizon: int
    candidates: list[dv.DriverCandidate]
    calendar: list[str]  # calendar variables that make sense for the period


@dataclass
class ParentCandidate:
    """A categorical column that could sit above the group in a hierarchy (039)."""

    column: str
    values: int  # distinct labels
    parent: bool  # each label of the group goes with a single label of this column


@dataclass
class GroupsSetup:
    """Every series of the group column, before forecasting them all (039)."""

    setup: SeriesSetup
    group: str
    groups: int  # labels of the group column
    limit: int  # series forecast on their own by default; the others pooled into "Others"
    max_limit: int
    summable: bool  # sums and counts add up: totals can be reconciled
    candidates: list[ParentCandidate]
    levels: list[str]  # the hierarchy recognised above the group, finest first
    browser: bool  # the browser version: few series, one after the other


@dataclass
class DriverForecast:
    """One driver model's forecast with given future driver values (scenarios, 039)."""

    model: str
    info: str | None
    future: list[str]
    forecast: list[float]
    low80: list[float]
    high80: list[float]
    low95: list[float]
    high95: list[float]
    drivers: list[dv.DriverInfo]
    notes: list[str]


@dataclass
class _Prepared:
    setup: SeriesSetup
    built: builder.Aggregated
    analyzed: SeriesLine
    lines: list[SeriesLine]
    window: int | None  # None: too few periods for rolling statistics
    # The analysed values without incomplete edge periods of a sum or count (their total is too
    # low and would distort every method), with their periods.
    core: list[float | None]
    core_periods: list[str]


def _column(view: Any) -> CompareColumn:
    return CompareColumn(view.name, view.type, view.parser, view.order, view.role)


def _choose(requested: str | None, names: list[str], fallback: str | None) -> str | None:
    return requested if requested in names else fallback


class TimeSeriesService:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns
        self._cache: OrderedDict[tuple[Any, ...], _Prepared] = OrderedDict()
        self._stats: OrderedDict[tuple[Any, ...], dict[str, choice.Stats]] = OrderedDict()
        self.forecasts = ForecastJobs()
        self.group_forecasts = gr.GroupJobs()

    def clear(self) -> None:
        self._cache.clear()
        self._stats.clear()
        self.forecasts.clear()
        self.group_forecasts.clear()

    # --- Series --------------------------------------------------------------------------------

    def prepare(self, dataset_id: str, request: Request, conditions: list[Condition]) -> _Prepared:
        dataset = self.columns.ready(dataset_id)
        views = [v for v in self.columns.columns(dataset_id) if not v.excluded]
        by_name = {v.name: v for v in views}
        dates = [v.name for v in views if v.type in DATES]
        numbers = [v.name for v in views if v.type in NUMBERS]
        categories = [v.name for v in views if v.type in CATEGORIES]
        if not dates:
            raise TimeSeriesError("This dataset has no date column.", "needs_date")
        if not numbers:
            raise TimeSeriesError("This dataset has no numeric column.", "needs_number")
        found = self._numbers_stats(dataset_id, dataset, [by_name[n] for n in [*numbers, *dates]])
        chosen_by = choice.choose(
            [by_name[n] for n in dates],
            [by_name[n] for n in numbers],
            request.time,
            request.value,
            found,
        )
        time, value = chosen_by.time, chosen_by.value
        group = _choose(request.group, categories, None)
        options = request.options
        if options.aggregate == "auto":
            options = replace(options, aggregate=chosen_by.aggregate)
        key = (
            dataset_id,
            dataset.imported_at,
            time,
            value,
            group,
            request.series,
            (options.period, options.aggregate, options.window, options.align),
            request.gaps,
            request.season,
            tuple(
                (v.name, v.type, v.role, v.parser.model_dump_json())
                for v in (by_name[time], by_name[value], by_name.get(group or ""))
                if v is not None
            ),
            filters_key(conditions),
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        where = where_for(views, conditions)
        connection = duckdb.connect()
        try:
            source = scoped_source(dataset.storage_path, where, numbered=True)
            comparer = Comparer(
                connection,
                source,
                _column(by_name[time]),
                _column(by_name[value]),
                color=_column(by_name[group]) if group else None,
            )
            aliases = {"t": comparer.x, "v": comparer.y}
            if comparer.color:
                aliases["c"] = comparer.color
            relation = comparer.rows_of(aliases)
            total = comparer.total()
            (rows,) = comparer.one(f"SELECT count(*) FROM {relation}")
            if rows < 3:
                raise TimeSeriesError("Fewer than 3 rows have a date and a value.", "too_few_rows")
            labels = (
                comparer.kept_labels(relation, "c", comparer.color)[:MAX_SERIES]
                if comparer.color
                else None
            )
            if labels is not None and request.series and request.series not in labels:
                # A series picked among every group (039): analysed even beyond the first eight.
                (found,) = comparer.one(
                    f"SELECT count(*) FROM {relation} WHERE c = {lit(request.series)}"
                )
                if found:
                    labels.append(request.series)
            built = builder.aggregate(connection, relation, by_name[value].role, options, labels)
            built.aggregate_auto = request.options.aggregate == "auto"
        finally:
            connection.close()

        season = request.season or ts.season_length(built.period)
        # Kalman fills the analysed series only; the other groups drawn alongside are
        # interpolated (one structural fit per series would cost too much, 064).
        cheap: ts.Gaps = "interpolate" if request.gaps == "kalman" else request.gaps

        def line(item: builder.Values) -> SeriesLine:
            return SeriesLine(item.label, ts.fill_gaps(item.values, cheap), item.counts)

        def kalman_line(item: builder.Values) -> SeriesLine:
            filled = sm.kalman_fill(item.values, season)
            return SeriesLine(
                item.label, filled.values, item.counts, [], [], filled.low, filled.high, filled.info
            )

        lines = [line(item) for item in built.groups]
        chosen = None
        if lines:
            chosen = (
                request.series if request.series in [x.label for x in lines] else lines[0].label
            )
        source = next((x for x in built.groups if x.label == chosen), None) or built.overall
        if request.gaps == "kalman":
            analyzed = kalman_line(source)
            lines = [analyzed if x.label == chosen else x for x in lines]
        else:
            analyzed = next((x for x in lines if x.label == chosen), None) or line(built.overall)
        window = None
        if len(built.periods) >= ts.MIN_PERIODS:
            window = ts.resolve_window(options.window, built.period, len(built.periods))
            for item in {id(x): x for x in [analyzed, *lines]}.values():
                item.rolling_mean, item.rolling_std = ts.rolling(item.values, window, options.align)
        setup = SeriesSetup(
            time,
            value,
            group,
            chosen,
            dates,
            numbers,
            categories,
            labels or [],
            built.period,
            built.aggregate,
            built.period_auto,
            built.aggregate_auto,
            built.coarsened,
            built.first_partial,
            built.last_partial,
            request.gaps,
            season,
            len(built.periods),
            int(rows),
            total,
            [
                *chosen_by.reasons[:1],
                *([chosen_by.aggregate_reason] if request.options.aggregate == "auto" else []),
                *chosen_by.reasons[1:],
            ],
        )
        summed = built.aggregate in ("sum", "count")
        start = 1 if built.first_partial and summed else 0
        stop = len(built.periods) - (1 if built.last_partial and summed else 0)
        prepared = _Prepared(
            setup,
            built,
            analyzed,
            lines,
            window,
            analyzed.values[start:stop],
            built.periods[start:stop],
        )
        self._cache[key] = prepared
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return prepared

    def _numbers_stats(
        self, dataset_id: str, dataset: Any, views: list[Any]
    ) -> dict[str, choice.Stats]:
        """Min, max and whole-number flag of each numeric column, whole-minute flag of each date
        column, on every row (R-TS-04, R-TS-06)."""
        key = (dataset_id, dataset.imported_at, tuple((v.name, v.type) for v in views))
        if key not in self._stats:
            connection = duckdb.connect()
            try:
                source = scoped_source(dataset.storage_path, where_for(views, []))
                self._stats[key] = choice.stats(connection, source, views)
            finally:
                connection.close()
            if len(self._stats) > CACHE_SIZE:
                self._stats.popitem(last=False)
        return self._stats[key]

    # --- Sections ------------------------------------------------------------------------------

    def overview(
        self,
        dataset_id: str,
        request: Request,
        conditions: list[Condition],
        chosen: list[str] | None = None,
    ) -> SeriesOverview:
        prepared = self.prepare(dataset_id, request, conditions)
        setup, built, analyzed = prepared.setup, prepared.built, prepared.analyzed
        known = [v for v in analyzed.values if v is not None]
        quarter = max(1, len(known) // 4)
        first = float(np.mean(known[:quarter])) if known else 0.0
        last = float(np.mean(known[-quarter:])) if known else 0.0
        figures = SeriesFigures(
            built.periods[0],
            built.periods[-1],
            len(built.periods),
            sum(1 for count in analyzed.counts if count == 0),
            clean((last - first) / abs(first) * 100) if first else None,
        )
        message = None
        results: list[MethodResult] = []
        if setup.periods < ts.MIN_PERIODS:
            message = f"Too few periods to analyse ({ts.MIN_PERIODS} needed)."
        else:
            tested = builder.trend_input(built, analyzed.values)
            for key in chosen or methods.DEFAULT_TREND:
                results.append(methods.TREND[key](tested, setup.season))
        return SeriesOverview(
            setup,
            built.periods,
            analyzed,
            prepared.lines,
            figures,
            prepared.window,
            request.options.align,
            results,
            message,
        )

    def smoothing(
        self,
        dataset_id: str,
        request: Request,
        conditions: list[Condition],
        chosen: list[str] | None = None,
        params: sm.SmoothParams | None = None,
    ) -> SeriesSmoothing:
        """The chosen smoothers (none: only the suggestions, which cost no model fit)."""
        prepared = self.prepare(dataset_id, request, conditions)
        setup, built, analyzed = prepared.setup, prepared.built, prepared.analyzed
        blocked = sm.unavailable(built.period)
        view = SeriesSmoothing(
            setup, built.periods, [], [], {str(k): v for k, v in blocked.items()}
        )
        if setup.periods < ts.MIN_PERIODS:
            view.message = f"Too few periods to smooth ({ts.MIN_PERIODS} needed)."
            return view
        params = params or sm.SmoothParams()
        if params.window is None:
            params.window = prepared.window
        params.align = request.options.align
        keys: list[sm.SmoothKey] = [k for k in sm.SMOOTHERS if k in (chosen or [])]
        view.curves = sm.smooth(analyzed.values, keys, built.period, setup.season, params)
        view.suggestions = [
            s
            for s in sm.suggest(analyzed.values, built.period, setup.season)
            if s.key not in keys and s.key not in blocked
        ]
        return view

    def seasonality(
        self,
        dataset_id: str,
        request: Request,
        conditions: list[Condition],
        chosen: list[str] | None = None,
    ) -> SeriesSeasonality:
        prepared = self.prepare(dataset_id, request, conditions)
        setup, built = prepared.setup, prepared.built
        if setup.periods < ts.MIN_PERIODS:
            message = f"Too few periods to analyse ({ts.MIN_PERIODS} needed)."
            return SeriesSeasonality(setup, "", [], None, [], 0, message)
        series, filled = ts.complete(prepared.core)
        trend, detrend = trend_component(series, setup.season)
        detrended = series - trend
        moments = starts(prepared.core_periods, built.period)
        results = []
        for key in chosen or methods.DEFAULT_SEASONALITY:
            method, wants_detrended = methods.SEASONALITY[key]
            results.append(method(detrended if wants_detrended else series, setup.season))
        return SeriesSeasonality(
            setup,
            detrend,
            profiles(built.period, moments, series, detrended),
            year_overlay(built.period, moments, prepared.core),
            results,
            filled,
        )

    def _transformed(
        self, prepared: _Prepared, how: st.Transform
    ) -> tuple[np.ndarray, np.ndarray, int, str | None]:
        """The complete series, its transformation, the interpolated periods, or why not."""
        series, filled = ts.complete(prepared.core)
        if prepared.setup.periods < st.MIN_PERIODS:
            return series, series, filled, f"Needs at least {st.MIN_PERIODS} periods."
        try:
            return series, st.transform(series, how, prepared.setup.season), filled, None
        except st.TransformError as error:
            return series, series, filled, str(error)

    def stationarity(
        self,
        dataset_id: str,
        request: Request,
        conditions: list[Condition],
        how: st.Transform = "none",
        chosen: list[str] | None = None,
    ) -> SeriesStationarity:
        prepared = self.prepare(dataset_id, request, conditions)
        setup, built = prepared.setup, prepared.built
        series, changed, filled, problem = self._transformed(prepared, how)
        view = SeriesStationarity(setup, how, "c", [], [], None, None, [], filled, problem)
        if problem:
            return view
        regression: st.Regression = "c"
        if how in ("none", "log") and setup.season and len(changed) >= 2 * setup.season:
            try:
                parts = decompose(changed, "stl", setup.season, built.period)
                if (parts.strength_trend or 0) > 0.6:
                    regression = "ct"  # a strong trend: test stationarity around it
            except DecompositionError:
                pass
        view.regression = regression
        view.periods = prepared.core_periods[len(prepared.core_periods) - len(changed) :]
        view.transformed = [float(v) for v in changed]
        keys = chosen or list(st.DEFAULT_STATIONARITY)
        results = {key: st.STATIONARITY[key](changed, regression) for key in keys}
        adf = results.get("adf") or st.adf(changed, regression)
        kpss = results.get("kpss") or st.kpss_test(changed, regression)
        view.synthesis = st.synthesis(adf, kpss)
        view.suggestion = st.suggest(series, setup.season)
        view.methods = list(results.values())
        return view

    def autocorrelation(
        self,
        dataset_id: str,
        request: Request,
        conditions: list[Condition],
        how: st.Transform = "none",
        chosen: list[str] | None = None,
        pacf: ac.PacfMethod = "ywm",
        ar_order: int | None = None,
    ) -> SeriesAutocorrelation:
        prepared = self.prepare(dataset_id, request, conditions)
        setup = prepared.setup
        _, changed, filled, problem = self._transformed(prepared, how)
        view = SeriesAutocorrelation(setup, how, None, None, [], filled, problem)
        if problem:
            return view
        view.correlogram = ac.correlogram(changed, setup.season, pacf)
        view.ar = ac.fit_ar(changed, ar_order, setup.season)
        view.methods = [
            ac.AUTOCORRELATION[key](changed, setup.season)
            for key in chosen or ac.DEFAULT_AUTOCORRELATION
        ]
        return view

    def volatility(
        self,
        dataset_id: str,
        request: Request,
        conditions: list[Condition],
        chosen: list[str] | None = None,
        spread_window: int | None = None,
    ) -> SeriesVolatility:
        prepared = self.prepare(dataset_id, request, conditions)
        setup, built = prepared.setup, prepared.built
        series, filled = ts.complete(prepared.core)
        window = ts.resolve_window(
            spread_window or prepared.window, built.period, max(setup.periods, 8)
        )
        view = SeriesVolatility(setup, "", [], [], [], window, None, False, [], filled)
        if setup.periods < vo.MIN_PERIODS:
            view.message = f"Needs at least {vo.MIN_PERIODS} periods."
            return view
        found = rs.remainder(series, setup.season)
        spread = vo.rolling_spread(found.values, window)
        direction = vo.summary(series, spread, found.fitted)
        view.residuals_from = found.method
        view.periods = prepared.core_periods
        view.residuals = [float(v) for v in found.values]
        view.rolling_std = spread
        view.rising, view.log_helps = direction.rising, direction.log_helps
        view.methods = [vo.VOLATILITY[key](found.values) for key in chosen or vo.DEFAULT_VOLATILITY]
        return view

    def residuals(
        self,
        dataset_id: str,
        request: Request,
        conditions: list[Condition],
        source: rs.Source = "residuals",
    ) -> SeriesResiduals:
        prepared = self.prepare(dataset_id, request, conditions)
        series, filled = ts.complete(prepared.core)
        view = SeriesResiduals(prepared.setup, None, filled)
        try:
            view.distribution = rs.distribution(series, source, prepared.setup.season)
        except rs.DistributionError as error:
            view.message = str(error)
        return view

    def breaks(
        self,
        dataset_id: str,
        request: Request,
        conditions: list[Condition],
        search: br.Search = "pelt",
        cost: br.Cost | Literal["auto"] = "auto",
        sensitivity: br.Sensitivity = "medium",
        chosen: list[str] | None = None,
        anomaly: br.Anomaly = "robust_z",
        threshold: float = 3.5,
    ) -> SeriesBreaks:
        prepared = self.prepare(dataset_id, request, conditions)
        setup, built = prepared.setup, prepared.built
        series, filled = ts.complete(prepared.core)
        view = SeriesBreaks(setup, [], [], False, None, [], [], [], anomaly, threshold, "", filled)
        if setup.periods < 10:
            view.message = "Needs at least 10 periods."
            return view
        values, deseasonalized = rs.deseasonalized(series, setup.season)
        chosen_cost: br.Cost
        if cost == "auto":
            # A steady trend cut into mean levels gives a staircase of false breaks.
            trending = methods.mann_kendall_test([float(v) for v in values]).significant
            chosen_cost = "trend" if trending else "mean"
        else:
            chosen_cost = cost
        found = br.find_breaks(values, search, chosen_cost, sensitivity, setup.season)
        found.cost_auto = cost == "auto"
        view.periods, view.values, view.deseasonalized = (
            prepared.core_periods,
            [float(v) for v in values],
            deseasonalized,
        )
        view.breaks = found
        view.points = [
            BreakPoint(
                point,
                prepared.core_periods[point],
                before.last if chosen_cost == "trend" else before.mean,
                after.first if chosen_cost == "trend" else after.mean,
                before.slope,
                after.slope,
            )
            for point, before, after in zip(
                found.points, found.segments, found.segments[1:], strict=False
            )
        ]
        view.methods = [br.BREAK_TESTS[key](values) for key in chosen or br.DEFAULT_BREAK_TESTS]
        remainder = rs.remainder(series, setup.season)
        view.residuals_from = remainder.method
        observed = series
        if anomaly == "kalman":
            # The structural model has its own seasonal part and handles the empty periods.
            observed = np.array([np.nan if v is None else v for v in prepared.core], dtype=float)
            view.residuals_from = br.kalman_source(observed, setup.season)
        moments = starts(prepared.core_periods, built.period)
        view.anomalies = [
            AnomalyRow(
                point.index,
                prepared.core_periods[point.index],
                moments[point.index].isoformat(),
                period_end(moments[point.index], built.period).isoformat(),
                point.value,
                point.expected,
                point.score,
            )
            for point in br.anomalies(remainder.values, observed, anomaly, threshold, setup.season)
        ]
        return view

    def start_forecast(
        self,
        dataset_id: str,
        request: Request,
        conditions: list[Condition],
        horizon: int | None = None,
        chosen: list[str] | None = None,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int] = (0, 0, 0),
        drivers: list[dv.Driver] | None = None,
    ) -> ForecastJob:
        prepared = self.prepare(dataset_id, request, conditions)
        setup, built = prepared.setup, prepared.built
        if setup.periods < 10:
            raise TimeSeriesError("A forecast needs at least 10 periods.", "too_few_periods")
        horizon = horizon or fc.default_horizon(built.period, len(prepared.core))
        keys = chosen or list(fc.DEFAULT_MODELS)
        if not setup.season:
            keys = [key for key in keys if key not in ("seasonal_naive", "holt_winters")] or [
                "naive"
            ]
        models = {key: self._model(key, order, seasonal_order) for key in keys}
        driver_set = (
            self._driver_set(dataset_id, conditions, prepared, drivers, horizon)
            if drivers
            else None
        )
        key = (
            id(prepared),
            tuple(prepared.core_periods[-3:]),
            len(prepared.core),
            horizon,
            tuple(keys),
            order,
            seasonal_order,
            tuple(drivers or ()),
        )
        return self.forecasts.start(
            key,
            prepared.core,
            prepared.core_periods,
            built.period,
            setup.season,
            horizon,
            models,
            driver_set,
        )

    @staticmethod
    def _model(
        key: str, order: tuple[int, int, int], seasonal_order: tuple[int, int, int]
    ) -> fc.Model:
        if key == "sarima":
            return fc.sarima_manual(order, seasonal_order)
        if key == "sarimax":
            return fc.sarimax_model()  # its own cache of ARIMA orders
        return fc.MODELS[key]

    # --- Drivers (028 A4) ----------------------------------------------------------------------

    def _raw_drivers(
        self,
        dataset_id: str,
        conditions: list[Condition],
        prepared: _Prepared,
        columns: list[tuple[str, bool]],
        future: list[str],
    ) -> tuple[dict[str, dv.RawDriver], dict[str, bool]]:
        """Columns aggregated per period like the series (sum for amounts and quantities, mean
        otherwise; the maximum for a flag), over the analysed group's rows, on the series'
        periods then the forecast periods. Also whether each column only holds 0 and 1."""
        if not columns:
            return {}, {}
        dataset = self.columns.ready(dataset_id)
        views = [v for v in self.columns.columns(dataset_id) if not v.excluded]
        by_name = {v.name: v for v in views}
        setup, period = prepared.setup, prepared.built.period
        connection = duckdb.connect()
        try:
            source = scoped_source(
                dataset.storage_path, where_for(views, conditions), numbered=True
            )
            comparer = Comparer(
                connection,
                source,
                _column(by_name[setup.time]),
                _column(by_name[setup.value]),
                color=_column(by_name[setup.group]) if setup.group else None,
            )
            selected = [f"{comparer.value_sql(comparer.x)} AS t"]
            measures, checks = [], []
            for i, (name, flag) in enumerate(columns):
                view = by_name[name]
                value = number_sql(CompareColumn(view.name, "numeric_continuous", view.parser))
                selected.append(f"CASE WHEN isfinite({value}) THEN {value} END AS d{i}")
                aggregate = "max" if flag else ts.default_aggregate(view.role)
                measures.append(ts.AGGREGATE_SQL[aggregate].replace("(v)", f"(d{i})"))
                checks.append(f"bool_and(d{i} IS NULL OR d{i} IN (0, 1))")
            condition = "t IS NOT NULL"
            if comparer.color is not None and setup.series is not None:
                selected.append(f"{comparer.value_sql(comparer.color)} AS c")
                if setup.series == builder.OTHERS:
                    listed = ", ".join(lit(label) for label in setup.groups) or "NULL"
                    condition += f" AND c IS NOT NULL AND c NOT IN ({listed})"
                else:
                    condition += f" AND c = {lit(setup.series)}"
            relation = (
                f"(SELECT * FROM (SELECT {', '.join(selected)} FROM {source}) WHERE {condition})"
            )
            truncated = f"date_trunc('{period}', t)"
            rows = {
                str(row[0]): row[1:]
                for row in connection.execute(
                    f"SELECT strftime({truncated}, '{ts.FORMATS[period]}'), {', '.join(measures)} "
                    f"FROM {relation} GROUP BY {truncated}"
                ).fetchall()
            }
            binary = connection.execute(f"SELECT {', '.join(checks)} FROM {relation}").fetchone()
        finally:
            connection.close()

        def at(label: str, i: int) -> float | None:
            return clean(rows[label][i]) if label in rows else None

        raw = {
            name: dv.RawDriver(
                [at(label, i) for label in prepared.core_periods],
                [at(label, i) for label in future],
            )
            for i, (name, _) in enumerate(columns)
        }
        flags = {name: bool(binary and binary[i]) for i, (name, _) in enumerate(columns)}
        return raw, flags

    def _driver_set(
        self,
        dataset_id: str,
        conditions: list[Condition],
        prepared: _Prepared,
        drivers: list[dv.Driver],
        horizon: int,
    ) -> dv.DriverSet:
        setup, period = prepared.setup, prepared.built.period
        types = {v.name: v.type for v in self.columns.columns(dataset_id) if not v.excluded}
        for driver in drivers:
            if driver.kind == "calendar":
                continue
            if driver.column in (setup.time, setup.value):
                raise dv.DriverError(f"{driver.column!r} is the series itself: it cannot drive it.")
            if types.get(driver.column) not in {*NUMBERS, "boolean"}:
                raise dv.DriverError(
                    f"A driver needs a numeric or yes/no column: {driver.column!r}."
                )
        future = future_periods(prepared.core_periods[-1], period, horizon)
        raw, _ = self._raw_drivers(
            dataset_id,
            conditions,
            prepared,
            [(d.column, d.kind == "flag") for d in drivers if d.kind != "calendar"],
            future,
        )
        return dv.build(
            drivers, raw, starts(prepared.core_periods, period), starts(future, period), period
        )

    def drivers(
        self,
        dataset_id: str,
        request: Request,
        conditions: list[Condition],
        horizon: int | None = None,
    ) -> SeriesDrivers:
        """Numeric and yes/no columns that could be drivers, with the correlation of their
        period-to-period changes with the series, and the calendar variables for the period."""
        prepared = self.prepare(dataset_id, request, conditions)
        setup, period = prepared.setup, prepared.built.period
        horizon = horizon or fc.default_horizon(period, len(prepared.core))
        views = [v for v in self.columns.columns(dataset_id) if not v.excluded]
        names = [
            v.name
            for v in views
            if (v.type in NUMBERS or v.type == "boolean")
            and v.name not in (setup.time, setup.value)
        ][:MAX_CANDIDATES]
        booleans = {v.name for v in views if v.type == "boolean"}
        future = future_periods(prepared.core_periods[-1], period, horizon)
        raw, binary = self._raw_drivers(
            dataset_id, conditions, prepared, [(name, False) for name in names], future
        )
        series, _ = ts.complete(prepared.core)
        found = [
            dv.candidate(name, name in booleans or binary[name], series, raw[name])
            for name in names
        ]
        found.sort(key=lambda c: (not c.suggested, -abs(c.correlation or 0)))
        return SeriesDrivers(setup, horizon, found, dv.calendar_parts(period))

    def forecast_with_drivers(
        self,
        dataset_id: str,
        request: Request,
        conditions: list[Condition],
        drivers: list[dv.Driver],
        future: dict[str, list[float]] | None = None,
        model: str = "tslm",
        horizon: int | None = None,
        changes: list[dv.ScenarioChange] | None = None,
    ) -> DriverForecast:
        """The final forecast of one driver model (`sarimax`, `tslm`, `gbm_lags`) with given
        future values of the known-in-advance drivers: the engine of what-if scenarios (039).
        `changes` (± x %, a fixed value, over a range of periods) apply after `future`."""
        if model not in fc.DRIVER_MODELS:
            raise TimeSeriesError(
                f"{model!r} does not use drivers: {', '.join(fc.DRIVER_MODELS)}.", "bad_driver"
            )
        prepared = self.prepare(dataset_id, request, conditions)
        setup, period = prepared.setup, prepared.built.period
        if setup.periods < 10:
            raise TimeSeriesError("A forecast needs at least 10 periods.", "too_few_periods")
        horizon = horizon or fc.default_horizon(period, len(prepared.core))
        driver_set = self._driver_set(dataset_id, conditions, prepared, drivers, horizon)
        chosen = driver_set.with_future(future) if future else driver_set
        if changes:
            labels = future_periods(prepared.core_periods[-1], period, horizon)
            chosen = chosen.with_future(dv.scenario_future(chosen, labels, changes))
        series, _ = ts.complete(prepared.core)
        try:
            prediction = dv.forecast_with_drivers(
                series, horizon, setup.season, self._model(model, (1, 1, 1), (0, 0, 0)), chosen
            )
        except fc.ModelError as error:
            raise TimeSeriesError(str(error), "model_failed") from error
        return DriverForecast(
            model,
            prediction.info,
            future_periods(prepared.core_periods[-1], period, horizon),
            [float(v) for v in prediction.mean],
            [float(v) for v in prediction.low80],
            [float(v) for v in prediction.high80],
            [float(v) for v in prediction.low95],
            [float(v) for v in prediction.high95],
            chosen.infos,
            chosen.notes,
        )

    def decomposition(
        self,
        dataset_id: str,
        request: Request,
        conditions: list[Condition],
        method: str = "stl",
    ) -> SeriesDecomposition:
        prepared = self.prepare(dataset_id, request, conditions)
        setup, built = prepared.setup, prepared.built
        series, filled = ts.complete(prepared.core)
        observed = [None if v is None else float(v) for v in prepared.core]
        view = SeriesDecomposition(setup, prepared.core_periods, observed, None, filled)
        if not setup.season:
            view.message = "Yearly data has no cycle to separate."
            return view
        try:
            view.result = decompose(series, method, setup.season, built.period)  # type: ignore[arg-type]
        except DecompositionError as error:
            view.message = str(error)
        return view

    # --- Every group (039) ---------------------------------------------------------------------

    def _grouped(
        self, dataset_id: str, conditions: list[Condition], setup: SeriesSetup, parents: list[str]
    ) -> tuple[duckdb.DuckDBPyConnection, str]:
        """A connection and the rows with a date, a value and a group (`t`, `v`, `c`), with the
        parent columns as `p0`, `p1`… (a missing label reads "(missing)")."""
        dataset = self.columns.ready(dataset_id)
        views = [v for v in self.columns.columns(dataset_id) if not v.excluded]
        by_name = {v.name: v for v in views}
        assert setup.group is not None
        connection = duckdb.connect()
        source = scoped_source(dataset.storage_path, where_for(views, conditions), numbered=True)
        comparer = Comparer(
            connection,
            source,
            _column(by_name[setup.time]),
            _column(by_name[setup.value]),
            color=_column(by_name[setup.group]),
        )
        assert comparer.color is not None
        selected = [
            f"{comparer.value_sql(comparer.x)} AS t",
            f"{comparer.value_sql(comparer.y)} AS v",
            f"{comparer.value_sql(comparer.color)} AS c",
        ]
        for i, name in enumerate(parents):
            label = comparer.value_sql(_column(by_name[name]))
            selected.append(f"coalesce({label}, '(missing)') AS p{i}")
        relation = (
            f"(SELECT * FROM (SELECT {', '.join(selected)} FROM {source}) "
            "WHERE t IS NOT NULL AND v IS NOT NULL AND isfinite(v) AND c IS NOT NULL)"
        )
        return connection, relation

    @staticmethod
    def _depends(
        connection: duckdb.DuckDBPyConnection, relation: str, child: str, parent: str
    ) -> bool:
        """Whether each `child` label goes with a single `parent` label."""
        (most,) = connection.execute(
            f"SELECT max(k) FROM (SELECT {child}, count(DISTINCT {parent}) AS k "
            f"FROM {relation} GROUP BY {child})"
        ).fetchone() or (None,)
        return most == 1

    def group_setup(
        self, dataset_id: str, request: Request, conditions: list[Condition]
    ) -> GroupsSetup:
        """How many series the group column holds, and the hierarchy recognised above it: the
        categorical columns where each group label goes with a single label (store → region)."""
        prepared = self.prepare(dataset_id, request, conditions)
        setup = prepared.setup
        if not setup.group:
            raise TimeSeriesError("Choose a group column first.", "needs_group")
        names = [c for c in setup.categories if c != setup.group][:MAX_PARENTS]
        connection, relation = self._grouped(dataset_id, conditions, setup, names)
        try:
            (groups,) = connection.execute(
                f"SELECT count(DISTINCT c) FROM {relation}"
            ).fetchone() or (0,)
            candidates: list[ParentCandidate] = []
            if names:
                distinct = ", ".join(f"count(DISTINCT p{i})" for i in range(len(names)))
                counted = connection.execute(f"SELECT {distinct} FROM {relation}").fetchone() or ()
                for i, name in enumerate(names):
                    values = int(counted[i])
                    parent = 2 <= values < groups and self._depends(
                        connection, relation, "c", f"p{i}"
                    )
                    candidates.append(ParentCandidate(name, values, parent))
            levels: list[str] = []
            current, size = "c", int(groups)
            for candidate in sorted(
                (c for c in candidates if c.parent), key=lambda c: (-c.values, c.column)
            ):
                at = f"p{names.index(candidate.column)}"
                if candidate.values < size and self._depends(connection, relation, current, at):
                    levels.append(candidate.column)
                    current, size = at, candidate.values
        finally:
            connection.close()
        gr.warm_up()  # the processes start while the user looks at the options
        browser = gr.IN_BROWSER
        return GroupsSetup(
            setup,
            setup.group,
            int(groups),
            gr.BROWSER_GROUPS if browser else gr.MAX_GROUPS,
            gr.BROWSER_GROUPS if browser else gr.LIMIT_RANGE[1],
            prepared.built.aggregate in ("sum", "count"),
            candidates,
            levels,
            browser,
        )

    def start_group_forecast(
        self,
        dataset_id: str,
        request: Request,
        conditions: list[Condition],
        horizon: int | None = None,
        chosen: list[str] | None = None,
        limit: int | None = None,
        use_global: bool = False,
        reconcile: gr.Reconcile | None = None,
        levels: list[str] | None = None,
        workers: int | None = None,
    ) -> gr.GroupJob:
        """Forecast every series of the group column (the `limit` largest on their own, the rest
        pooled), each with its best model; optionally a global model, and coherent totals over
        the hierarchy `levels` (parent columns, finest first; None: the one recognised)."""
        prepared = self.prepare(dataset_id, request, conditions)
        setup, built = prepared.setup, prepared.built
        if not setup.group:
            raise TimeSeriesError("Choose a group column first.", "needs_group")
        if setup.periods < 10:
            raise TimeSeriesError("A forecast needs at least 10 periods.", "too_few_periods")
        if reconcile is not None and built.aggregate not in ("sum", "count"):
            raise TimeSeriesError(
                "Totals only add up for sums and counts: choose Sum or Count to reconcile.",
                "not_summable",
            )
        horizon = horizon or fc.default_horizon(built.period, len(prepared.core))
        cap = gr.BROWSER_GROUPS if gr.IN_BROWSER else gr.LIMIT_RANGE[1]
        default = gr.BROWSER_GROUPS if gr.IN_BROWSER else gr.MAX_GROUPS
        limit = max(1, min(limit or default, cap))
        keys = [k for k in chosen or gr.DEFAULT_MODELS if k not in ("sarima", "sarimax")]
        unknown = [k for k in keys if k not in fc.MODELS]
        if unknown:
            raise TimeSeriesError(f"Unknown method(s): {', '.join(unknown)}.", "unknown_method")
        keys = keys or list(gr.DEFAULT_MODELS)
        if reconcile is None:
            parents: list[str] = []
        elif levels is None:
            parents = self.group_setup(dataset_id, request, conditions).levels
        else:
            parents = list(levels)
        for name in parents:
            if name not in setup.categories or name == setup.group:
                raise TimeSeriesError(
                    f"{name!r} is not a category column above the group.", "bad_level"
                )
        nodes, tree, meta = self._group_nodes(
            dataset_id, conditions, prepared, limit, parents, reconcile is not None
        )
        future = future_periods(prepared.core_periods[-1], built.period, horizon)
        season = setup.season
        per_year = round(ts.PER_YEAR[built.period])
        key = (
            id(prepared),
            tuple(prepared.core_periods[-3:]),
            horizon,
            tuple(keys),
            limit,
            use_global,
            reconcile,
            tuple(parents),
        )

        def work(job: gr.GroupJob, cancelled: Callable[[], bool]) -> gr.GroupForecast | None:
            return gr.forecast_groups(
                job,
                nodes,
                tree,
                keys,
                horizon,
                season,
                per_year,
                prepared.core_periods,
                future,
                meta,
                cancelled,
                use_global=use_global,
                reconcile=reconcile,
                workers=workers,
            )

        return self.group_forecasts.start(key, len(nodes), work)

    def _group_nodes(
        self,
        dataset_id: str,
        conditions: list[Condition],
        prepared: _Prepared,
        limit: int,
        parents: list[str],
        with_tree: bool,
    ) -> tuple[list[gr.Node], hi.Tree | None, dict[str, Any]]:
        """One series per kept label (largest first), the others pooled into "Others" (one per
        parent with a hierarchy, so it stays a tree); with a tree, also its parents and total."""
        setup, built = prepared.setup, prepared.built
        assert setup.group is not None
        summed = built.aggregate in ("sum", "count")
        connection, relation = self._grouped(dataset_id, conditions, setup, parents)
        try:
            for i, name in enumerate(parents):
                child = "c" if i == 0 else f"p{i - 1}"
                if not self._depends(connection, relation, child, f"p{i}"):
                    below = setup.group if i == 0 else parents[i - 1]
                    raise TimeSeriesError(
                        f"{name!r} is not above {below!r}: one {below} value goes with several "
                        f"{name} values.",
                        "not_a_hierarchy",
                    )
            measure = "sum(abs(v))" if built.aggregate == "sum" else "count(*)"
            ranked = connection.execute(
                f"SELECT c, {measure} AS m FROM {relation} GROUP BY c ORDER BY m DESC, c"
            ).fetchall()
            kept = [str(label) for label, _ in ranked[:limit]]
            listed = ", ".join(lit(label) for label in kept) or "NULL"
            pooled = f"'{builder.OTHERS} · ' || p0" if parents else f"'{builder.OTHERS}'"
            bucket = f"CASE WHEN c IN ({listed}) THEN c ELSE {pooled} END"
            period = built.period
            truncated = f"date_trunc('{period}', t)"
            above_sql = "".join(f"p{i}, " for i in range(len(parents)))
            found = connection.execute(
                f"SELECT g, {above_sql}strftime({truncated}, '{ts.FORMATS[period]}'), "
                f"{ts.AGGREGATE_SQL[built.aggregate]}, count(*) "
                f"FROM (SELECT {bucket} AS g, * FROM {relation}) "
                f"GROUP BY g, {above_sql}{truncated}"
            ).fetchall()
        finally:
            connection.close()
        depth = len(parents)
        cells: dict[str, dict[str, tuple[float | None, int]]] = {}
        above: dict[str, list[str]] = {}
        for row in found:
            label = str(row[0])
            above.setdefault(label, [str(p) for p in row[1 : 1 + depth]])
            cells.setdefault(label, {})[str(row[1 + depth])] = (
                clean(row[2 + depth]),
                int(row[3 + depth]),
            )
        kept_set = set(kept)
        others = sorted(label for label in cells if label not in kept_set)
        order = [label for label in kept if label in cells] + others
        start = 1 if built.first_partial and summed else 0
        stop = len(built.periods) - (1 if built.last_partial and summed else 0)
        empty = 0.0 if summed else None
        leaves: list[gr.Node] = []
        for label in order:
            values = [cells[label][p][0] if p in cells[label] else empty for p in built.periods]
            counts = [cells[label][p][1] if p in cells[label] else 0 for p in built.periods]
            # Kalman is for the series studied; every group is interpolated (cost, 064).
            gaps: ts.Gaps = "interpolate" if setup.gaps == "kalman" else setup.gaps
            filled, _ = ts.complete(ts.fill_gaps(values, gaps)[start:stop])
            first = next((i for i, k in enumerate(counts[start:stop]) if k), 0)
            leaves.append(
                gr.Node(
                    label, setup.group, [float(v) for v in filled], first, label not in kept_set
                )
            )
        notes = []
        pooled_count = len(ranked) - len(kept)
        if pooled_count:
            where = " (one series per parent)" if parents else ""
            notes.append(
                f"{pooled_count} smaller {setup.group} values are pooled into “Others”{where} "
                f"and forecast as one series: the {len(kept)} largest are forecast on their own."
            )
        if gr.IN_BROWSER:
            notes.append(
                f"Browser version: at most {gr.BROWSER_GROUPS} series, "
                "forecast one after the other."
            )
        meta: dict[str, Any] = {
            "group": setup.group,
            "groups": len(ranked),
            "kept": len(kept),
            "pooled": pooled_count,
            "limit": limit,
            "notes": notes,
        }
        if not with_tree:
            return leaves, None, meta
        tree = hi.build_tree(
            [leaf.label for leaf in leaves],
            setup.group,
            [(name, [above[leaf.label][i] for leaf in leaves]) for i, name in enumerate(parents)],
        )
        summing = tree.summing()
        totals = summing @ np.array([leaf.values for leaf in leaves])
        nodes: list[gr.Node] = []
        for i in range(len(tree.labels)):
            if i in tree.bottom:
                nodes.append(leaves[tree.bottom.index(i)])
                continue
            below = [leaves[j] for j in np.flatnonzero(summing[i])]
            nodes.append(
                gr.Node(
                    tree.labels[i],
                    tree.levels[i],
                    [float(v) for v in totals[i]],
                    min(leaf.start for leaf in below),
                )
            )
        return nodes, tree, meta
