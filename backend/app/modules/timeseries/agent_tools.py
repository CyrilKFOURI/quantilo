"""What the time-series module lets the AI agent do (spec 026b): read a series (trend,
seasonality, stationarity) and forecast it, through this module's own service."""

from typing import Any, ClassVar, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from quantilo.timeseries import drivers as dv
from quantilo.timeseries import forecast as fc
from quantilo.timeseries import series as ts
from quantilo.timeseries.service import Request, TimeSeriesError, TimeSeriesService

from app.core.tools import AppLink, Tool, ToolContext, ToolError, ToolOutput, compact, wait_for


class SeriesInput(BaseModel):
    time: str = Field(description="The date or time column.")
    value: str | None = Field(None, description="The number to follow; empty to count rows.")
    period: Literal["auto", "hour", "day", "week", "month", "quarter", "year"] = Field(
        "auto", description="Aggregation step."
    )
    agg: Literal["auto", "sum", "mean", "median", "min", "max", "count"] = Field(
        "auto", description="How values of one period are combined (auto: sum for amounts)."
    )

    def request(self) -> Request:
        return Request(
            time=self.time,
            value=self.value,
            options=ts.Options(period=self.period, aggregate=self.agg),
        )


def _link(args: SeriesInput, setup: Any = None) -> AppLink:
    """098: the page opens empty without a date and a value, so the link names the ones used."""
    time = getattr(setup, "time", None) or args.time
    value = args.value or getattr(setup, "value", None)
    params = {"time": time} | ({"value": value} if value else {})
    return AppLink("timeseries", params, "Open in Time series")


def _methods(results: list[Any]) -> list[Any]:
    return [compact(m, max_items=8) for m in results]


class AnalyzeTimeSeries(Tool[SeriesInput]):
    name: ClassVar[str] = "analyze_time_series"
    description: ClassVar[str] = (
        "Read one series over time: its setup (period, aggregation), key figures, trend tests "
        "(Mann-Kendall, Sen's slope), seasonality tests and calendar profile, and stationarity "
        "tests (ADF, KPSS, Phillips-Perron) with the suggested differencing. Use it before a "
        "forecast or to answer 'is it growing / seasonal?'."
    )
    Input: ClassVar[type[BaseModel]] = SeriesInput

    def __init__(self, service: TimeSeriesService) -> None:
        self.service = service

    def label(self, args: SeriesInput) -> str:
        return f"Analysing {args.value or 'rows'} over {args.time}"

    def run(self, ctx: ToolContext, args: SeriesInput) -> ToolOutput:
        request = args.request()
        try:
            overview = self.service.overview(ctx.dataset_id, request, ctx.conditions, None)
            seasonality = self.service.seasonality(ctx.dataset_id, request, ctx.conditions, None)
            stationarity = self.service.stationarity(
                ctx.dataset_id, request, ctx.conditions, "none", None
            )
        except TimeSeriesError as error:
            raise ToolError(error.message) from error
        content = {
            "setup": compact(overview.setup),
            "figures": compact(overview.figures, max_items=8),
            "trend": _methods(overview.methods),
            "seasonality": {
                "detrended_by": seasonality.detrend,
                "methods": _methods(seasonality.methods),
            },
            "stationarity": {
                "synthesis": compact(stationarity.synthesis),
                "suggestion": compact(stationarity.suggestion),
                "methods": _methods(stationarity.methods),
            },
            "notes": [
                m for m in (overview.message, seasonality.message, stationarity.message) if m
            ],
        }
        setup = overview.setup
        summary = f"{len(overview.periods)} periods ({getattr(setup, 'period', '')})"
        return ToolOutput(summary, content, _link(args, setup))


class DriverInput(BaseModel):
    column: str = Field(
        description="A numeric or yes/no column; for kind=calendar: weekday, month or month_end; "
        "for kind=holidays: the country's ISO code (FR, US)."
    )
    kind: Literal["column", "flag", "calendar", "holidays"] = Field(
        "column",
        description="column: a column aggregated per period; flag: a 0/1 column (events); "
        "calendar: day of week, month or end of month; holidays: the public holidays of a "
        "country (hourly, daily or weekly series only).",
    )
    known: bool = Field(
        False,
        description="Known in advance (planned price, promotion): its future values are given. "
        "False: past only (visits), lagged by the horizon so nothing leaks from the future.",
    )
    future: Literal["file", "constant", "last"] = Field(
        "file",
        description="When known in advance: future rows of the file, a constant, or the last "
        "value repeated.",
    )
    value: float | None = Field(None, description="The constant, when future=constant.")
    subdiv: str | None = Field(
        None, description="kind=holidays: a subdivision code (CA for California), if any."
    )

    def driver(self) -> dv.Driver:
        return dv.Driver(self.column, self.kind, self.known, self.future, self.value, self.subdiv)


DRIVER_DEFAULTS = ["naive", "seasonal_naive", "ets_auto", "arima_auto", "tslm", "sarimax"]


class ForecastInput(SeriesInput):
    horizon: int | None = Field(None, ge=1, le=500, description="Periods ahead; empty: automatic.")
    group: str | None = Field(
        None, description="A category column: one series per value (store, product, country)."
    )
    series: str | None = Field(
        None, description="With a group and every_group=false: the group value to forecast."
    )
    every_group: bool = Field(
        False,
        description="With a group: forecast every value of it at once (the 200 largest, the "
        "rest pooled into Others), each with its best model; drivers are not used.",
    )
    reconcile: Literal["none", "auto", "bottom_up", "top_down", "mint_shrink"] = Field(
        "none",
        description="With every_group, for sums and counts: make the forecasts add up over the "
        "hierarchy recognised above the group (store -> region -> total). auto: MinT when the "
        "history allows, bottom-up otherwise.",
    )
    models: list[str] | None = Field(
        None,
        description="Model keys; empty: naive, seasonal_naive, ets_auto, arima_auto (with "
        "drivers, also tslm and sarimax). Only sarimax, tslm, gbm_lags, prophet_like and "
        "chronos_2 use drivers. Several seasons (daily, hourly data): prophet_like, stlf. "
        "AI models (optional extra, only if already downloaded: you cannot download one, "
        "the user confirms it in the page or Settings > Models): chronos_bolt_tiny, "
        "chronos_bolt_small, chronos_bolt_base, chronos_2, timesfm_2p5 (pretrained, good on "
        "short series), nhits, nbeats (trained, 500 periods or more). A missing one returns "
        "its reason in its row.",
    )
    drivers: list[DriverInput] = Field(
        default_factory=list,
        max_length=dv.MAX_DRIVERS,
        description="Outside variables that may explain the series (correlated columns, prices, "
        "promotions, calendar). Each driver model is also scored without them.",
    )


class Forecast(Tool[ForecastInput]):
    name: ClassVar[str] = "forecast"
    kind = "compute"
    description: ClassVar[str] = (
        "Forecast a series: baseline (naive, seasonal naive) and statistical models (ETS, auto "
        "ARIMA) scored by rolling-origin validation, the best refitted with prediction "
        "intervals. Drivers (other columns, calendar, 0/1 flags) feed SARIMAX, the linear "
        "regression, gradient boosting and the Prophet-like model; the result says whether "
        "each driver helps. With a "
        "group and every_group=true, every series of the group at once (best model per "
        "series); reconcile makes stores add up to regions and the total. Needs at least 10 "
        "periods. Analyse the series first."
    )
    Input: ClassVar[type[BaseModel]] = ForecastInput

    def __init__(self, service: TimeSeriesService) -> None:
        self.service = service

    def label(self, args: ForecastInput) -> str:
        if args.every_group and args.group:
            return f"Forecasting {args.value or 'rows'} for every {args.group}"
        return f"Forecasting {args.value or 'rows'}"

    def run(self, ctx: ToolContext, args: ForecastInput) -> ToolOutput:
        if args.every_group:
            return self._every_group(ctx, args)
        drivers = [d.driver() for d in args.drivers] or None
        models = args.models or (DRIVER_DEFAULTS if drivers else None)
        unknown = [key for key in models or [] if key not in fc.MODELS]
        if unknown:
            raise ToolError(f"Unknown model(s): {', '.join(unknown)}.")
        try:
            job = self.service.start_forecast(
                ctx.dataset_id,
                self._request(args),
                ctx.conditions,
                args.horizon,
                models,
                drivers=drivers,
            )
        except (TimeSeriesError, dv.DriverError) as error:
            raise ToolError(error.message) from error
        forecasts = self.service.forecasts

        def poll() -> Any:
            found = forecasts.get(job.id)
            if found is None:
                raise ToolError("The forecast is no longer available: run it again.")
            return found

        job = wait_for(
            poll,
            lambda j: j.status != "running",
            ctx,
            describe=lambda j: (
                f"Model {j.done + 1} of {j.total}: {j.current or ''}",
                j.done / max(j.total, 1),
            ),
            cancel=lambda: forecasts.cancel(job.id),
        )
        if job.result is None:
            raise ToolError(f"Forecast {job.status}: {job.error or 'no result'}")
        content = compact(job.result, max_items=12)
        return ToolOutput("Forecast ready", content, _link(args))

    @staticmethod
    def _request(args: ForecastInput) -> Request:
        request = args.request()
        request.group, request.series = args.group, args.series
        return request

    def _every_group(self, ctx: ToolContext, args: ForecastInput) -> ToolOutput:
        if not args.group:
            raise ToolError("every_group needs a group column.")
        unknown = [key for key in args.models or [] if key not in fc.MODELS]
        if unknown:
            raise ToolError(f"Unknown model(s): {', '.join(unknown)}.")
        try:
            job = self.service.start_group_forecast(
                ctx.dataset_id,
                self._request(args),
                ctx.conditions,
                args.horizon,
                args.models,
                reconcile=None if args.reconcile == "none" else args.reconcile,
            )
        except TimeSeriesError as error:
            raise ToolError(error.message) from error
        jobs = self.service.group_forecasts

        def poll() -> Any:
            found = jobs.get(job.id)
            if found is None:
                raise ToolError("The forecast is no longer available: run it again.")
            return found

        job = wait_for(
            poll,
            lambda j: j.status != "running",
            ctx,
            describe=lambda j: (f"Series {j.done} of {j.total}", j.done / max(j.total, 1)),
            cancel=lambda: jobs.cancel(job.id),
        )
        result = job.result
        if result is None:
            raise ToolError(f"Forecast {job.status}: {job.error or 'no result'}")
        rows = [
            {
                "group": r.label,
                "level": r.level,
                "best": r.best,
                "mase": r.mase,
                "wape": r.wape,
                "total": r.total,
                "change_vs_last_year_pct": r.change,
                "short_history": r.short_history or None,
            }
            for r in result.rows[:30]
        ]
        content = {
            "group": result.group,
            "horizon": result.horizon,
            "future": result.future,
            "series": {"groups": result.groups, "kept": result.kept, "pooled": result.pooled},
            "best_models": compact(result.best_models),
            "hierarchy": compact(result.hierarchy),
            "notes": result.notes,
            "rows": compact(rows, max_items=30),
        }
        link = _link(args)
        link.params["group"] = args.group
        summary = f"{result.kept + (1 if result.pooled else 0)} series forecast"
        return ToolOutput(summary, content, link)


def agent_tools(app: FastAPI) -> list[Tool[Any]]:
    service: TimeSeriesService = app.state.timeseries
    return [AnalyzeTimeSeries(service), Forecast(service)]
