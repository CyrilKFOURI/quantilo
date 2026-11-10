"""One time series of a dataset, for the library: the app's Time series page as methods.

series = ds.timeseries(time="date", value="sales")
series.overview(); series.stationarity(); series.forecast(horizon=12)
"""

import time as clock
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from quantilo.errors import ExplorerError
from quantilo.filters.model import Condition
from quantilo.timeseries import autocorrelation as ac
from quantilo.timeseries import breaks as br
from quantilo.timeseries import drivers as dv
from quantilo.timeseries import groups as gr
from quantilo.timeseries import residuals as rs
from quantilo.timeseries import smoothing as sm
from quantilo.timeseries import stationarity as st
from quantilo.timeseries.forecasting import SeriesForecast
from quantilo.timeseries.service import (
    DriverForecast,
    GroupsSetup,
    Request,
    SeriesAutocorrelation,
    SeriesBreaks,
    SeriesDecomposition,
    SeriesDrivers,
    SeriesOverview,
    SeriesResiduals,
    SeriesSeasonality,
    SeriesSetup,
    SeriesSmoothing,
    SeriesStationarity,
    SeriesVolatility,
    TimeSeriesService,
)

DriverItems = Sequence["dv.Driver | str | Mapping[str, Any]"]


class TimeSeries:
    """A value over time (one series, or one group of a grouped series). Unset choices (time
    column, period, aggregation, season) are made by the app's rules; `setup()` shows them."""

    def __init__(
        self,
        service: TimeSeriesService,
        dataset_id: str,
        request: Request,
        conditions: list[Condition],
        methods: list[str] | None = None,
    ) -> None:
        self._service = service
        self._id = dataset_id
        self.request = request
        self._conditions = conditions
        self._methods = methods  # the tests to run (None: the default ones ★)

    def _args(self) -> tuple[str, Request, list[Condition]]:
        return self._id, self.request, self._conditions

    def setup(self) -> SeriesSetup:
        """Every choice actually used: time column, value, period, aggregation, season."""
        return self.overview().setup

    def overview(self) -> SeriesOverview:
        """The series, its rolling mean and spread, its trend (Mann-Kendall, Sen's slope)."""
        return self._service.overview(*self._args(), self._methods)

    def smoothing(
        self,
        methods: Sequence[sm.SmoothKey] = ("moving_average", "kalman_smoothed"),
        params: sm.SmoothParams | None = None,
    ) -> SeriesSmoothing:
        """Smoothed curves (EWMA, LOWESS, Savitzky-Golay, Kalman, HP), each with its automatic
        parameters and whether it is causal, plus the methods the data suggests (064)."""
        return self._service.smoothing(*self._args(), list(methods), params)

    def seasonality(self) -> SeriesSeasonality:
        return self._service.seasonality(*self._args(), self._methods)

    def decomposition(self, method: str = "stl") -> SeriesDecomposition:
        """Trend, seasonal and remainder: `stl`, `mstl`, `classical_additive` or
        `classical_multiplicative`."""
        return self._service.decomposition(*self._args(), method)

    def stationarity(self, transform: st.Transform = "none") -> SeriesStationarity:
        """ADF, KPSS, Phillips–Perron… and the suggested differencing."""
        return self._service.stationarity(*self._args(), transform, self._methods)

    def autocorrelation(
        self,
        transform: st.Transform = "none",
        pacf: ac.PacfMethod = "ywm",
        ar_order: int | None = None,
    ) -> SeriesAutocorrelation:
        """ACF, PACF, Ljung–Box, Durbin–Watson."""
        return self._service.autocorrelation(
            *self._args(), transform, self._methods, pacf, ar_order
        )

    def volatility(self, spread_window: int | None = None) -> SeriesVolatility:
        """Heteroscedasticity tests (Breusch–Pagan, White, ARCH-LM…) and the rolling spread."""
        return self._service.volatility(*self._args(), self._methods, spread_window)

    def residuals(self, source: rs.Source = "residuals") -> SeriesResiduals:
        return self._service.residuals(*self._args(), source)

    def breaks(
        self,
        search: br.Search = "pelt",
        cost: br.Cost | Literal["auto"] = "auto",
        sensitivity: br.Sensitivity = "medium",
        anomaly: br.Anomaly = "robust_z",
        threshold: float = 3.5,
    ) -> SeriesBreaks:
        """Change points and anomalies."""
        return self._service.breaks(
            *self._args(), search, cost, sensitivity, self._methods, anomaly, threshold
        )

    def forecast(
        self,
        horizon: int | None = None,
        models: list[str] | None = None,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int] = (0, 0, 0),
        drivers: DriverItems | None = None,
    ) -> SeriesForecast:
        """Naive models, ETS, ARIMA… validated on rolling origins and ranked by MASE (waits for
        every model). `drivers`: outside variables used by `sarimax`, `tslm` and `gbm_lags`, as
        `Driver("visits")` (past only), `Driver("price", known=True)` (future values from the
        file), `Driver("weekday", kind="calendar")`, `Driver("holiday", kind="flag")`; a plain
        column name is a past-only driver."""
        job = self._service.start_forecast(
            *self._args(), horizon, models, order, seasonal_order, as_drivers(drivers)
        )
        while job.status == "running":
            clock.sleep(0.05)
        if job.result is None:
            raise ExplorerError(job.error or "The forecast failed.", code="forecast")
        return job.result

    def drivers(self, horizon: int | None = None) -> SeriesDrivers:
        """Columns that could drive the series, most correlated first (changes vs changes)."""
        return self._service.drivers(*self._args(), horizon)

    def scenario(
        self,
        drivers: DriverItems,
        future: dict[str, list[float]] | None = None,
        model: str = "tslm",
        horizon: int | None = None,
        changes: Sequence["dv.ScenarioChange | Mapping[str, Any]"] | None = None,
    ) -> DriverForecast:
        """One driver model's forecast with other future values for the known-in-advance
        drivers: `future={"price": [9.9, 9.9, 12.0, …]}` (one value per forecast period), or
        `changes=[ScenarioChange("price", "percent", -10, start="2025-12", end="2025-12")]`."""
        found = [
            c if isinstance(c, dv.ScenarioChange) else dv.ScenarioChange(**c) for c in changes or []
        ]
        return self._service.forecast_with_drivers(
            *self._args(), as_drivers(drivers) or [], future, model, horizon, found or None
        )

    def groups(self) -> GroupsSetup:
        """How many series the group column holds, and the hierarchy recognised above it."""
        return self._service.group_setup(*self._args())

    def forecast_groups(
        self,
        horizon: int | None = None,
        models: list[str] | None = None,
        limit: int | None = None,
        global_model: bool = False,
        reconcile: gr.Reconcile | None = None,
        levels: list[str] | None = None,
        workers: int | None = None,
    ) -> gr.GroupForecast:
        """Forecast every series of the group column, each with its best model (default:
        seasonal naive, ETS, combination; Croston-type added to intermittent series). The `limit`
        largest (200) are forecast on their own, the others pooled into "Others".
        `global_model`: also one LightGBM model over every series. `reconcile`: coherent totals
        over the hierarchy (`auto`, `mint_shrink`, `bottom_up`, `top_down`), `levels` = parent
        columns, finest first (None: the ones recognised). `workers`: processes (one per core)."""
        job = self._service.start_group_forecast(
            *self._args(), horizon, models, limit, global_model, reconcile, levels, workers
        )
        while job.status == "running":
            clock.sleep(0.05)
        if job.result is None:
            raise ExplorerError(job.error or "The forecast failed.", code="forecast")
        return job.result


def as_drivers(items: DriverItems | None) -> list[dv.Driver] | None:
    """Drivers given as `Driver`, a column name (past only) or a dict of `Driver` fields."""
    if not items:
        return None
    found = []
    for item in items:
        if isinstance(item, dv.Driver):
            found.append(item)
        elif isinstance(item, str):
            found.append(dv.Driver(item))
        else:
            found.append(dv.Driver(**item))
    return found
