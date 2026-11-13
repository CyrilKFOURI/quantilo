from typing import Annotated, Literal

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from quantilo.filters.model import Condition, parse_filters
from quantilo.timeseries import autocorrelation as ac
from quantilo.timeseries import breaks as br
from quantilo.timeseries import drivers as dv
from quantilo.timeseries import forecast as fc
from quantilo.timeseries import groups as gr
from quantilo.timeseries import methods
from quantilo.timeseries import regimes as rg
from quantilo.timeseries import residuals as rs
from quantilo.timeseries import smoothing as sm
from quantilo.timeseries import stationarity as st
from quantilo.timeseries import volatility as vo
from quantilo.timeseries.decomposition import Method
from quantilo.timeseries.forecasting import ForecastJob
from quantilo.timeseries.series import Aggregate, Align, Gaps, Options, Period
from quantilo.timeseries.service import (
    DriverForecast,
    GroupsSetup,
    SeriesAutocorrelation,
    SeriesBreaks,
    SeriesDecomposition,
    SeriesDrivers,
    SeriesOverview,
    SeriesRegimes,
    SeriesResiduals,
    SeriesSeasonality,
    SeriesSmoothing,
    SeriesStationarity,
    SeriesVolatility,
    TimeSeriesError,
    TimeSeriesService,
)
from quantilo.timeseries.service import Request as SeriesRequest

router = APIRouter(tags=["module:timeseries"])

Name = Annotated[str | None, Query(min_length=1)]


def get_service(request: Request) -> TimeSeriesService:
    return request.app.state.timeseries


def active_filters(
    filters: Annotated[str | None, Query(max_length=100_000)] = None,
) -> list[Condition]:
    return parse_filters(filters)


def series_request(
    time: Name = None,
    value: Name = None,
    group: Name = None,
    series: Annotated[str | None, Query()] = None,
    period: Period | Literal["auto"] = "auto",
    agg: Aggregate | Literal["auto"] = "auto",
    window: Annotated[int | None, Query(ge=2, le=365)] = None,
    align: Align = "trailing",
    gaps: Gaps = "leave",
    season: Annotated[int | None, Query(ge=2, le=400)] = None,
) -> SeriesRequest:
    return SeriesRequest(
        time, value, group, series, Options(period, agg, window, align), gaps, season
    )


def chosen_methods(text: str | None, catalogue: dict) -> list[str] | None:
    if not text:
        return None
    keys = [key.strip() for key in text.split(",") if key.strip()]
    unknown = [key for key in keys if key not in catalogue]
    if unknown:
        raise TimeSeriesError(f"Unknown method(s): {', '.join(unknown)}.", "unknown_method")
    return keys


Service = Annotated[TimeSeriesService, Depends(get_service)]
Series = Annotated[SeriesRequest, Depends(series_request)]
Filters = Annotated[list[Condition], Depends(active_filters)]
Methods = Annotated[str | None, Query(alias="methods", description="Comma-separated method keys.")]


@router.get("/datasets/{dataset_id}/overview")
def overview(
    dataset_id: str, service: Service, request: Series, filters: Filters, methods_: Methods = None
) -> SeriesOverview:
    """The series, its rolling statistics, key figures and trend methods."""
    return service.overview(dataset_id, request, filters, chosen_methods(methods_, methods.TREND))


@router.get("/datasets/{dataset_id}/smoothing")
def smoothing(
    dataset_id: str,
    service: Service,
    request: Series,
    filters: Filters,
    methods_: Methods = None,
    ewma_alpha: Annotated[float | None, Query(gt=0, lt=1)] = None,
    lowess_frac: Annotated[float | None, Query(gt=0, le=1)] = None,
    savgol_window: Annotated[int | None, Query(ge=5, le=401)] = None,
    savgol_order: Annotated[int | None, Query(ge=1, le=5)] = None,
    kalman_model: sm.KalmanForm | Literal["auto"] = "auto",
    hp_lambda: Annotated[float | None, Query(gt=0, le=1e9)] = None,
) -> SeriesSmoothing:
    """Smoothed curves of the series (EWMA, Kalman filtered / smoothed, LOWESS, Savitzky-Golay,
    Hodrick-Prescott), each with its automatic parameters and whether it is causal, and the
    methods the data suggests (rules only: no model is fitted without a method)."""
    keys = chosen_methods(methods_, {key: None for key in sm.SMOOTHERS})
    params = sm.SmoothParams(
        ewma_alpha=ewma_alpha,
        lowess_frac=lowess_frac,
        savgol_window=savgol_window,
        savgol_order=savgol_order,
        kalman_model=kalman_model,
        hp_lambda=hp_lambda,
    )
    return service.smoothing(dataset_id, request, filters, keys, params)


@router.get("/datasets/{dataset_id}/regimes")
def regimes(
    dataset_id: str,
    service: Service,
    request: Series,
    filters: Filters,
    k: Annotated[int | None, Query(ge=2, le=3)] = None,
    on: rg.On = "auto",
) -> SeriesRegimes:
    """Regimes of the series: a Markov-switching model (statsmodels MarkovRegression, its own
    mean and variance per regime), 2 or 3 regimes by BIC, on the level or on the period changes
    by rule (R-RG-01/02). Slow (seconds): called only on demand."""
    return service.regimes(dataset_id, request, filters, k, on)


@router.get("/datasets/{dataset_id}/seasonality")
def seasonality(
    dataset_id: str, service: Service, request: Series, filters: Filters, methods_: Methods = None
) -> SeriesSeasonality:
    """Calendar profiles, one line per year, and seasonality methods."""
    keys = chosen_methods(methods_, methods.SEASONALITY)
    return service.seasonality(dataset_id, request, filters, keys)


@router.get("/datasets/{dataset_id}/stationarity")
def stationarity(
    dataset_id: str,
    service: Service,
    request: Series,
    filters: Filters,
    transform: st.Transform = "none",
    methods_: Methods = None,
) -> SeriesStationarity:
    """Unit-root and stationarity tests on the series or a transformation of it."""
    keys = chosen_methods(methods_, st.STATIONARITY)
    return service.stationarity(dataset_id, request, filters, transform, keys)


@router.get("/datasets/{dataset_id}/autocorrelation")
def autocorrelation(
    dataset_id: str,
    service: Service,
    request: Series,
    filters: Filters,
    transform: st.Transform = "none",
    methods_: Methods = None,
    pacf: ac.PacfMethod = "ywm",
    ar: Annotated[int | None, Query(ge=1, le=24, description="AR order (default: by AIC).")] = None,
) -> SeriesAutocorrelation:
    """ACF, PACF, a reading of their shapes, and autocorrelation / randomness tests."""
    keys = chosen_methods(methods_, ac.AUTOCORRELATION)
    return service.autocorrelation(dataset_id, request, filters, transform, keys, pacf, ar)


@router.get("/datasets/{dataset_id}/volatility")
def volatility(
    dataset_id: str,
    service: Service,
    request: Series,
    filters: Filters,
    methods_: Methods = None,
    spread: Annotated[int | None, Query(ge=2, le=365, description="Rolling window.")] = None,
) -> SeriesVolatility:
    """Rolling spread of the residuals and heteroscedasticity / ARCH tests."""
    keys = chosen_methods(methods_, vo.VOLATILITY)
    return service.volatility(dataset_id, request, filters, keys, spread)


@router.get("/datasets/{dataset_id}/residuals")
def residuals(
    dataset_id: str,
    service: Service,
    request: Series,
    filters: Filters,
    source: rs.Source = "residuals",
) -> SeriesResiduals:
    """Distribution of the residuals, the series or its returns: statistics, normality, laws."""
    return service.residuals(dataset_id, request, filters, source)


@router.get("/datasets/{dataset_id}/breaks")
def breaks(
    dataset_id: str,
    service: Service,
    request: Series,
    filters: Filters,
    search: br.Search = "pelt",
    cost: br.Cost | Literal["auto"] = "auto",
    sensitivity: br.Sensitivity = "medium",
    methods_: Methods = None,
    anomaly: br.Anomaly = "robust_z",
    threshold: Annotated[float, Query(ge=2, le=6)] = 3.5,
) -> SeriesBreaks:
    """Change points, single-break tests and anomalies."""
    keys = chosen_methods(methods_, br.BREAK_TESTS)
    return service.breaks(
        dataset_id, request, filters, search, cost, sensitivity, keys, anomaly, threshold
    )


def _orders(text: str, name: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(part) for part in text.split(","))
    except ValueError:
        values = ()
    if len(values) != 3 or not all(0 <= v <= 3 for v in values):
        raise TimeSeriesError(
            f"{name} must be three integers from 0 to 3, like 1,1,1.", "bad_order"
        )
    return values  # type: ignore[return-value]


class ForecastOptions(BaseModel):
    """What does not fit in the query string: the drivers (028 A4, public holidays A5) and the
    tuning options (028 A7: ARIMA bounds and criterion, gradient-boosting grid, Prophet-like
    changepoints, a time limit per model)."""

    drivers: list[dv.Driver] = Field(default_factory=list, max_length=dv.MAX_DRIVERS)
    tuning: fc.Tuning = Field(default_factory=fc.Tuning)


@router.post("/datasets/{dataset_id}/forecast", status_code=202)
def start_forecast(
    dataset_id: str,
    service: Service,
    request: Series,
    filters: Filters,
    horizon: Annotated[int | None, Query(ge=1, le=500)] = None,
    methods_: Methods = None,
    order: str = "1,1,1",
    sorder: str = "0,0,0",
    options: Annotated[ForecastOptions | None, Body()] = None,
) -> ForecastJob:
    """Start (or reuse) a forecast job: models scored by rolling origin, then refitted. The
    body's drivers are used by SARIMAX, the linear regression, gradient boosting and the
    Prophet-like model; its tuning sets the options of the models and the time limit."""
    keys = chosen_methods(methods_, fc.MODELS)
    return service.start_forecast(
        dataset_id,
        request,
        filters,
        horizon,
        keys,
        _orders(order, "order"),
        _orders(sorder, "seasonal order"),
        options.drivers if options else None,
        options.tuning if options else None,
    )


@router.get("/datasets/{dataset_id}/drivers")
def drivers(
    dataset_id: str,
    service: Service,
    request: Series,
    filters: Filters,
    horizon: Annotated[int | None, Query(ge=1, le=500)] = None,
) -> SeriesDrivers:
    """Columns that could drive the series (correlation of their changes, future values in the
    file), the calendar variables that make sense for the period, and the public holiday
    calendar (countries, the one proposed by rule and why)."""
    return service.drivers(dataset_id, request, filters, horizon)


def _job(service: TimeSeriesService, job_id: str) -> ForecastJob:
    job = service.forecasts.get(job_id)
    if job is None:
        raise TimeSeriesError("This forecast is no longer available: run it again.", "unknown_job")
    return job


@router.get("/forecast-jobs/{job_id}")
def forecast_job(job_id: str, service: Service) -> ForecastJob:
    """Progress, then the result, of a forecast job."""
    return _job(service, job_id)


@router.delete("/forecast-jobs/{job_id}")
def cancel_forecast(job_id: str, service: Service) -> ForecastJob:
    """Stop a forecast after the model being fitted."""
    _job(service, job_id)
    return service.forecasts.cancel(job_id)  # type: ignore[return-value]


@router.get("/datasets/{dataset_id}/decomposition")
def decomposition(
    dataset_id: str, service: Service, request: Series, filters: Filters, method: Method = "stl"
) -> SeriesDecomposition:
    """Trend, seasonal and remainder components."""
    return service.decomposition(dataset_id, request, filters, method)


# --- Every group, coherent totals, scenarios (039) ------------------------------------------------


@router.get("/datasets/{dataset_id}/groups")
def groups(dataset_id: str, service: Service, request: Series, filters: Filters) -> GroupsSetup:
    """How many series the group column holds and the hierarchy recognised above it (columns
    where each group value goes with a single value: store → region)."""
    return service.group_setup(dataset_id, request, filters)


def _levels(text: str | None) -> list[str] | None:
    """None: the hierarchy recognised; "": the total only; "region,country": these parents."""
    if text is None:
        return None
    return [name.strip() for name in text.split(",") if name.strip()]


@router.post("/datasets/{dataset_id}/forecast-groups", status_code=202)
def start_group_forecast(
    dataset_id: str,
    service: Service,
    request: Series,
    filters: Filters,
    horizon: Annotated[int | None, Query(ge=1, le=500)] = None,
    methods_: Methods = None,
    limit: Annotated[int | None, Query(ge=gr.LIMIT_RANGE[0], le=gr.LIMIT_RANGE[1])] = None,
    global_model: Annotated[bool, Query(alias="global")] = False,
    reconcile: Literal["none", "auto", "bottom_up", "top_down", "mint_shrink"] = "none",
    levels: Annotated[
        str | None, Query(description="Parent columns, finest first; empty: the total only.")
    ] = None,
) -> gr.GroupJob:
    """Start (or reuse) a forecast of every series of the group: the best model per series,
    optionally a global LightGBM model and coherent totals over the hierarchy."""
    keys = chosen_methods(methods_, fc.MODELS)
    return service.start_group_forecast(
        dataset_id,
        request,
        filters,
        horizon,
        keys,
        limit,
        global_model,
        None if reconcile == "none" else reconcile,
        _levels(levels),
    )


def _group_job(service: TimeSeriesService, job_id: str) -> gr.GroupJob:
    job = service.group_forecasts.get(job_id)
    if job is None:
        raise TimeSeriesError("This forecast is no longer available: run it again.", "unknown_job")
    return job


@router.get("/group-forecast-jobs/{job_id}")
def group_forecast_job(job_id: str, service: Service) -> gr.GroupJob:
    """Progress (series done), then the result, of a forecast of every group."""
    return _group_job(service, job_id)


@router.delete("/group-forecast-jobs/{job_id}")
def cancel_group_forecast(job_id: str, service: Service) -> gr.GroupJob:
    """Stop: the series not started yet are dropped."""
    _group_job(service, job_id)
    return service.group_forecasts.cancel(job_id)  # type: ignore[return-value]


@router.get("/group-forecast-jobs/{job_id}/forecast.csv")
def group_forecast_csv(job_id: str, service: Service) -> Response:
    """Every forecast in long format (group, level, period, forecast, 80 % and 95 % bounds)."""
    job = _group_job(service, job_id)
    if job.result is None:
        raise TimeSeriesError("This forecast has not finished.", "not_finished")
    return Response(
        gr.long_csv(job.result),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="forecast-every-group.csv"'},
    )


class ScenarioOptions(BaseModel):
    """The drivers and what changes in their future values (039 story 6)."""

    drivers: list[dv.Driver] = Field(min_length=1, max_length=dv.MAX_DRIVERS)
    future: dict[str, list[float]] | None = None
    changes: list[dv.ScenarioChange] = Field(default_factory=list, max_length=20)
    tuning: fc.Tuning = Field(default_factory=fc.Tuning)


@router.post("/datasets/{dataset_id}/scenario")
def scenario(
    dataset_id: str,
    service: Service,
    request: Series,
    filters: Filters,
    options: Annotated[ScenarioOptions, Body()],
    model: str = "tslm",
    horizon: Annotated[int | None, Query(ge=1, le=500)] = None,
) -> DriverForecast:
    """One driver model's forecast (SARIMAX, linear regression, gradient boosting) with other
    future values of the known-in-advance drivers: ± x %, a fixed value, over a range of
    periods. Without changes: the base forecast. Same as `TimeSeries.scenario`."""
    return service.forecast_with_drivers(
        dataset_id,
        request,
        filters,
        options.drivers,
        options.future,
        model,
        horizon,
        options.changes or None,
        options.tuning,
    )
