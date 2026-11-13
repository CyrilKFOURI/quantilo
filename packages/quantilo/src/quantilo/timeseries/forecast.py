"""Forecast a series and compare models honestly on a rolling-origin validation.

Every model returns point forecasts with 80 % and 95 % prediction intervals. Models are scored
on the last periods they did not see (several origins), against a naive reference (MASE).
"""

import hashlib
import itertools
import math
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from quantilo.timeseries import series as ts
from quantilo.timeseries import stationarity as st
from quantilo.timeseries.autocorrelation import ljung_box
from quantilo.timeseries.decomposition import stl, strength
from quantilo.timeseries.methods import MethodResult
from quantilo.variable.source import clean

Z80, Z95 = float(stats.norm.ppf(0.9)), float(stats.norm.ppf(0.975))
ARIMA_BUDGET = 20.0  # seconds for the automatic ARIMA search
MAX_SEASONAL_ARIMA = 24  # longer cycles make seasonal ARIMA impractical (weekly data, m = 52)
HORIZON = {"hour": 24, "day": 30, "week": 12, "month": 12, "quarter": 8, "year": 3}


def default_horizon(period: str, periods: int) -> int:
    """The usual horizon for the period, at most a quarter of the series (room to validate)."""
    return max(1, min(HORIZON[period], periods // 4))


Criterion = Literal["aicc", "aic", "bic"]
GbmGrid = Literal["fixed", "small", "wide"]


@dataclass(frozen=True)
class Tuning:
    """Options of the models (028 A7, ⚙ Options, never needed): the bounds and criterion of the
    automatic ARIMA search (Hyndman-Khandakar's defaults), the grid of gradient boosting, the
    trend prior of the Prophet-like model, and a time limit per model (None: no limit)."""

    max_p: int = 3
    max_q: int = 3
    max_seasonal_p: int = 1
    max_seasonal_q: int = 1
    max_d: int = 2
    criterion: Criterion = "aicc"
    gbm_grid: GbmGrid = "fixed"
    changepoint_scale: float = 0.05
    changepoints: int = 25
    time_limit: float | None = None  # seconds per model

    def __post_init__(self) -> None:
        checks = [
            (0 <= self.max_p <= 5, "max_p goes from 0 to 5"),
            (0 <= self.max_q <= 5, "max_q goes from 0 to 5"),
            (0 <= self.max_seasonal_p <= 2, "max_seasonal_p goes from 0 to 2"),
            (0 <= self.max_seasonal_q <= 2, "max_seasonal_q goes from 0 to 2"),
            (0 <= self.max_d <= 2, "max_d goes from 0 to 2"),
            (0 < self.changepoint_scale <= 10, "changepoint_scale is above 0, at most 10"),
            (0 <= self.changepoints <= 100, "changepoints goes from 0 to 100"),
            (
                self.time_limit is None or 1 <= self.time_limit <= 3600,
                "time_limit goes from 1 to 3600 seconds",
            ),
        ]
        wrong = [text for ok, text in checks if not ok]
        if wrong:
            raise ValueError("; ".join(wrong))

    @property
    def arima(self) -> tuple[int, int, int, int, int, str]:
        return (
            self.max_p,
            self.max_q,
            self.max_seasonal_p,
            self.max_seasonal_q,
            self.max_d,
            self.criterion,
        )


DEFAULT_TUNING = Tuning()


@dataclass
class Component:
    """One part of a structural fit (Prophet-like, 028 A6 story 4): over the fitted history,
    then the forecast periods, in the series' unit."""

    name: str  # "Trend", "weekly", "Public holidays (FR)", a driver's column
    kind: Literal["trend", "season", "holidays", "driver"]
    values: list[float]


@dataclass
class Prediction:
    mean: np.ndarray
    low80: np.ndarray
    high80: np.ndarray
    low95: np.ndarray
    high95: np.ndarray
    info: str | None = None  # chosen form or orders, e.g. "ETS(A,Ad,A)", "ARIMA(1,1,1)"
    resid: np.ndarray | None = None  # in-sample residuals (for diagnostics)
    # No interval of its own: the final one is replaced by conformal intervals from validation.
    approximate: bool = False
    # With drivers (028 A4), one value per driver column given to the model: coefficient and
    # p-value (linear models), or importance (gradient boosting); `exog` = which columns.
    coef: np.ndarray | None = None
    pvalues: np.ndarray | None = None
    importance: np.ndarray | None = None
    exog: list[int] | None = None
    # Structural models: trend, seasons (then the drivers' effects, added by `drivers.Bound`)
    # over the fitted history and the horizon; `start` = first period of the history they cover.
    components: list[Component] = field(default_factory=list)
    exog_effects: np.ndarray | None = None  # (fitted + h, driver columns): column × coefficient
    start: int = 0


class ModelError(ValueError):
    """The model cannot be fitted on this series (too short, values ≤ 0…)."""


class Unavailable(ModelError):
    """The model cannot run on this machine (028 B1: extra or weights missing): the reason is
    the same at every origin, so it is reported instead of skipping them."""


class TimeLimit(ModelError):
    """A model went over its time limit (028 A7): it stops between two fits and says so."""


def _quiet(compute: Callable):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute()


def _band(mean: np.ndarray, sigma: np.ndarray, info: str | None = None, resid=None) -> Prediction:
    return Prediction(
        mean,
        mean - Z80 * sigma,
        mean + Z80 * sigma,
        mean - Z95 * sigma,
        mean + Z95 * sigma,
        info,
        resid,
    )


# --- Benchmarks (intervals from Hyndman & Athanasopoulos, FPP3 §5.5) ----------------------------


def _rms(resid: np.ndarray) -> float:
    """Root mean square of the residuals: unlike the standard deviation, it keeps a systematic
    error (a seasonal naive forecast of a growing series is always too low)."""
    return float(np.sqrt(np.mean(resid**2))) if len(resid) else 0.0


def naive(y: np.ndarray, h: int, season: int | None) -> Prediction:
    resid = np.diff(y)
    sigma = _rms(resid)
    steps = np.arange(1, h + 1)
    return _band(np.full(h, y[-1]), sigma * np.sqrt(steps), resid=resid)


def seasonal_naive(y: np.ndarray, h: int, season: int | None) -> Prediction:
    if not season or len(y) < season + 2:
        raise ModelError("Needs a seasonal period and more than one full cycle.")
    resid = y[season:] - y[:-season]
    sigma = _rms(resid)
    steps = np.arange(h)
    mean = np.array([y[len(y) - season + (i % season)] for i in steps])
    cycles = steps // season + 1
    return _band(mean, sigma * np.sqrt(cycles), resid=resid)


def drift(y: np.ndarray, h: int, season: int | None) -> Prediction:
    n = len(y)
    slope = (y[-1] - y[0]) / (n - 1)
    resid = np.diff(y) - slope
    sigma = float(np.std(resid, ddof=1))
    steps = np.arange(1, h + 1)
    return _band(y[-1] + slope * steps, sigma * np.sqrt(steps * (1 + steps / (n - 1))), resid=resid)


def mean_model(y: np.ndarray, h: int, season: int | None) -> Prediction:
    sigma = float(np.std(y, ddof=1))
    return _band(
        np.full(h, y.mean()), np.full(h, sigma * math.sqrt(1 + 1 / len(y))), resid=y - y.mean()
    )


# --- Exponential smoothing (ETS state-space models) ----------------------------------------------


def _ets(
    y: np.ndarray,
    h: int,
    error: str,
    trend: str | None,
    damped: bool,
    seasonal: str | None,
    season: int | None,
):
    model = ETSModel(
        pd.Series(y),
        error=error,
        trend=trend,
        damped_trend=damped,
        seasonal=seasonal,
        seasonal_periods=season if seasonal else None,
    )
    return _quiet(lambda: model.fit(disp=False, maxiter=200))


def _ets_prediction(fit, n: int, h: int, label: str) -> Prediction:
    frames = [
        _quiet(lambda a=alpha: fit.get_prediction(start=n, end=n + h - 1).summary_frame(alpha=a))
        for alpha in (0.2, 0.05)
    ]
    mean = np.asarray(frames[0]["mean"], dtype=float)
    return Prediction(
        mean,
        np.asarray(frames[0]["pi_lower"], dtype=float),
        np.asarray(frames[0]["pi_upper"], dtype=float),
        np.asarray(frames[1]["pi_lower"], dtype=float),
        np.asarray(frames[1]["pi_upper"], dtype=float),
        label,
        np.asarray(fit.resid, dtype=float),
    )


def _ets_label(error: str, trend: str | None, damped: bool, seasonal: str | None) -> str:
    code = {"add": "A", "mul": "M", None: "N"}
    return f"ETS({code[error]},{code[trend]}{'d' if damped else ''},{code[seasonal]})"


def ses(y: np.ndarray, h: int, season: int | None) -> Prediction:
    return _ets_prediction(_ets(y, h, "add", None, False, None, season), len(y), h, "ETS(A,N,N)")


def holt(y: np.ndarray, h: int, season: int | None) -> Prediction:
    return _ets_prediction(_ets(y, h, "add", "add", False, None, season), len(y), h, "ETS(A,A,N)")


def holt_winters(y: np.ndarray, h: int, season: int | None) -> Prediction:
    if not season or len(y) < 2 * season + 2:
        raise ModelError("Needs two full cycles.")
    return _ets_prediction(_ets(y, h, "add", "add", False, "add", season), len(y), h, "ETS(A,A,A)")


def ets_auto(y: np.ndarray, h: int, season: int | None) -> Prediction:
    """Every admissible ETS form, the lowest AICc kept (multiplicative only for values > 0)."""
    positive = bool((y > 0).all())
    errors = ["add", "mul"] if positive else ["add"]
    trends: list[tuple[str | None, bool]] = [(None, False), ("add", False), ("add", True)]
    seasonals: list[str | None] = [None]
    if season and 2 <= season <= 52 and len(y) >= 2 * season + 2:
        seasonals += ["add", "mul"] if positive else ["add"]
    best = None
    for error, (trend, damped), seasonal in itertools.product(errors, trends, seasonals):
        if error == "add" and seasonal == "mul":
            continue  # numerically unstable combination (FPP3 §8.6)
        try:
            fit = _ets(y, h, error, trend, damped, seasonal, season)
        except Exception:
            continue
        if np.isfinite(fit.aicc) and (best is None or fit.aicc < best[0].aicc):
            best = (fit, _ets_label(error, trend, damped, seasonal))
    if best is None:
        raise ModelError("No ETS form could be fitted.")
    return _ets_prediction(best[0], len(y), h, best[1])


STL_LONGEST = 400  # periods: longer cycles (yearly on hourly data) are left out of STLF


def stlf_model(
    period: ts.Period | None = None,
) -> Callable[[np.ndarray, int, int | None], Prediction]:
    """STLF (FPP3 §13.2, 028 A6): the seasons removed by STL (one cycle) or MSTL (several: the
    cycles of the Prophet-like rule, rounded), the seasonally adjusted series forecast by the
    automatic non-seasonal ETS, then the last cycle of each seasonal component added back.
    Its own intervals (the ETS ones, the seasonal part taken as known) only serve the validation:
    they covered ~50 % on rolling origins, so the final ones are conformal (`run_model`)."""

    def fit(y: np.ndarray, h: int, season: int | None) -> Prediction:
        from statsmodels.tsa.seasonal import MSTL

        from quantilo.timeseries.prophet_like import cycles

        n = len(y)
        used, _ = cycles(period, season, n)
        lengths = sorted({round(c.length) for c in used if 2 <= round(c.length) <= STL_LONGEST})
        lengths = [m for m in lengths if n >= 2 * m]
        if not lengths:
            raise ModelError("Needs two full cycles of a season.")
        if len(lengths) == 1:
            seasonal = [stl(y, lengths[0]).seasonal]
        else:
            parts = _quiet(lambda: MSTL(y, periods=lengths).fit())
            seasonal = list(np.asarray(parts.seasonal, dtype=float).reshape(n, -1).T)
        adjusted = y - np.sum(seasonal, axis=0)
        base = ets_auto(adjusted, h, None)
        steps = np.arange(h)
        ahead = np.sum(
            [component[n - m + steps % m] for component, m in zip(seasonal, lengths, strict=True)],
            axis=0,
        )
        method = "STL" if len(lengths) == 1 else "MSTL"
        prediction = Prediction(
            base.mean + ahead,
            base.low80 + ahead,
            base.high80 + ahead,
            base.low95 + ahead,
            base.high95 + ahead,
            f"{method} ({', '.join(map(str, lengths))}) + {base.info}",
            base.resid,
        )
        prediction.approximate = True
        return prediction

    return fit


def prophet_model(
    period: ts.Period | None = None, tuning: Tuning = DEFAULT_TUNING
) -> Callable[..., Prediction]:
    """The Prophet-like structural model (028 A6, `prophet_like.py`), drivers included; each
    public holiday gets its own column (`separate_holidays`, 028 A5). `tuning`: number of
    candidate changepoints and the prior scale of their slope changes (028 A7)."""

    def fit(
        y: np.ndarray,
        h: int,
        season: int | None,
        x: np.ndarray | None = None,
        x_future: np.ndarray | None = None,
    ) -> Prediction:
        from quantilo.timeseries import prophet_like

        built = prophet_like.model(period, tuning.changepoints, tuning.changepoint_scale)
        return built(y, h, season, x, x_future)

    fit.separate_holidays = True  # type: ignore[attr-defined]
    return fit


def theta(y: np.ndarray, h: int, season: int | None) -> Prediction:
    seasonal = bool(season and season >= 2 and len(y) >= 2 * season and (y > 0).all())
    fit = _quiet(
        lambda: ThetaModel(y, period=season if seasonal else 1, deseasonalize=seasonal).fit()
    )
    mean = np.asarray(fit.forecast(h), dtype=float)
    i80 = np.asarray(_quiet(lambda: fit.prediction_intervals(h, alpha=0.2)), dtype=float)
    i95 = np.asarray(_quiet(lambda: fit.prediction_intervals(h, alpha=0.05)), dtype=float)
    return Prediction(
        mean,
        i80[:, 0],
        i80[:, 1],
        i95[:, 0],
        i95[:, 1],
        "Theta" + (" (deseasonalized)" if seasonal else ""),
    )


# --- Structural model (Kalman, 064) -------------------------------------------------------------


def structural(y: np.ndarray, h: int, season: int | None) -> Prediction:
    """Unobserved components: local level or local linear trend, with a fixed seasonal pattern
    over two full cycles, the form with the lowest AIC (`smoothing.kalman`); native intervals."""
    from quantilo.timeseries import smoothing as sm

    try:
        fit = sm.kalman(y, season)
    except sm.KalmanError as error:
        raise ModelError(str(error)) from error
    forecast = _quiet(lambda: fit.results.get_forecast(h))
    mean = np.asarray(forecast.predicted_mean, dtype=float) * fit.scale + fit.center
    i80 = np.asarray(forecast.conf_int(alpha=0.2), dtype=float) * fit.scale + fit.center
    i95 = np.asarray(forecast.conf_int(alpha=0.05), dtype=float) * fit.scale + fit.center
    return Prediction(
        mean,
        i80[:, 0],
        i80[:, 1],
        i95[:, 0],
        i95[:, 1],
        f"UCM({fit.label})",
        (np.asarray(y, dtype=float) - fit.predicted)[fit.burn :],
    )


# --- ARIMA ---------------------------------------------------------------------------------------


def _sarimax(
    y: np.ndarray,
    order: tuple[int, int, int],
    seasonal: tuple[int, int, int, int],
    exog: np.ndarray | None = None,
):
    model = SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal,
        trend="c" if order[1] + seasonal[1] == 0 else None,
    )
    return _quiet(lambda: model.fit(disp=False, maxiter=100))


def _arima_prediction(fit, h: int, label: str, exog: np.ndarray | None = None) -> Prediction:
    forecast = fit.get_forecast(h, exog=exog)
    mean = np.asarray(forecast.predicted_mean, dtype=float)
    i80 = np.asarray(forecast.conf_int(alpha=0.2), dtype=float)
    i95 = np.asarray(forecast.conf_int(alpha=0.05), dtype=float)
    burn = max(fit.model.k_diff + fit.model.k_seasonal_diff * fit.model.seasonal_periods, 1)
    return Prediction(
        mean,
        i80[:, 0],
        i80[:, 1],
        i95[:, 0],
        i95[:, 1],
        label,
        np.asarray(fit.resid, dtype=float)[burn:],
    )


def _arima_label(order: tuple[int, int, int], seasonal: tuple[int, int, int, int]) -> str:
    text = f"ARIMA({order[0]},{order[1]},{order[2]})"
    if seasonal[3]:
        text += f"({seasonal[0]},{seasonal[1]},{seasonal[2]})[{seasonal[3]}]"
    return text


def differences(y: np.ndarray, season: int | None, max_d: int = 2) -> tuple[int, int, int]:
    """Seasonal differences D (strong seasonality) then d (ADF and KPSS, at most `max_d`), and
    the usable cycle."""
    m = season if season and 2 <= season <= MAX_SEASONAL_ARIMA and len(y) >= 3 * season else 0
    seasonal_d = 0
    if m:
        parts = stl(y, m)
        if (strength(parts.seasonal, parts.resid) or 0) > 0.64:
            seasonal_d = 1
    base = y[m:] - y[:-m] if seasonal_d else y
    d = 0
    while d < max_d:
        verdict = st.judge(base)
        if verdict is None or verdict.verdict == "stationary":
            break
        base, d = np.diff(base), d + 1
    return d, seasonal_d, m


def arima_auto(
    y: np.ndarray, h: int, season: int | None, tuning: Tuning = DEFAULT_TUNING
) -> Prediction:
    """Hyndman-Khandakar stepwise search, simplified: differences from tests, then neighbours of
    the best (p, q, P, Q) by AICc (or the criterion of `tuning`), within a time budget."""
    known = _known_orders(y, season, tuning)
    if known is not None:
        order, seasonal, label = known
        try:
            return _arima_prediction(_sarimax(y, order, seasonal), h, label)
        except Exception:
            pass  # search again below
    order, seasonal, label, fit = arima_search(y, season, tuning)
    _remember_orders(y, season, (order, seasonal, label), tuning)
    return _arima_prediction(fit, h, label)


def arima_model(
    tuning: Tuning = DEFAULT_TUNING,
) -> Callable[[np.ndarray, int, int | None], Prediction]:
    """The automatic ARIMA with the bounds and criterion of `tuning` (028 A7)."""

    def fit(y: np.ndarray, h: int, season: int | None) -> Prediction:
        return arima_auto(y, h, season, tuning)

    return fit


ORDERS_KEPT = 128  # series whose automatic ARIMA orders are remembered
_orders: OrderedDict[tuple[Any, ...], tuple[Any, Any, str]] = OrderedDict()
_orders_lock = threading.Lock()


def _orders_key(y: np.ndarray, season: int | None, tuning: Tuning) -> tuple[Any, ...]:
    data = np.ascontiguousarray(y, dtype=float).tobytes()
    return hashlib.sha1(data, usedforsecurity=False).digest(), season, tuning.arima


def _known_orders(
    y: np.ndarray, season: int | None, tuning: Tuning = DEFAULT_TUNING
) -> tuple[Any, Any, str] | None:
    """The orders already found for exactly this series and search (the same training window
    searched by ARIMA (automatic) and by SARIMAX, or a forecast run again)."""
    key = _orders_key(y, season, tuning)
    with _orders_lock:
        found = _orders.get(key)
        if found is not None:
            _orders.move_to_end(key)
        return found


def _remember_orders(
    y: np.ndarray,
    season: int | None,
    found: tuple[Any, Any, str],
    tuning: Tuning = DEFAULT_TUNING,
) -> None:
    with _orders_lock:
        _orders[_orders_key(y, season, tuning)] = found
        while len(_orders) > ORDERS_KEPT:
            _orders.popitem(last=False)


def arima_orders(
    y: np.ndarray, season: int | None, tuning: Tuning = DEFAULT_TUNING
) -> tuple[Any, Any, str]:
    """Orders and label of the automatic ARIMA for this series, searched once."""
    known = _known_orders(y, season, tuning)
    if known is None:
        order, seasonal, label, _ = arima_search(y, season, tuning)
        known = (order, seasonal, label)
        _remember_orders(y, season, known, tuning)
    return known


def arima_search(
    y: np.ndarray, season: int | None, tuning: Tuning = DEFAULT_TUNING
) -> tuple[tuple[int, int, int], tuple[int, int, int, int], str, Any]:
    """The orders chosen by the automatic ARIMA, their label and the fitted model. `tuning`
    bounds p, q, P, Q and d and picks the criterion (AICc by default, as Hyndman-Khandakar)."""
    d, seasonal_d, m = differences(y, season, tuning.max_d)
    started = time.monotonic()
    budget = min(ARIMA_BUDGET, tuning.time_limit or ARIMA_BUDGET)
    tried: dict[tuple[int, int, int, int], float] = {}
    fits: dict[tuple[int, int, int, int], object] = {}
    top = (
        tuning.max_p,
        tuning.max_q,
        tuning.max_seasonal_p if m else 0,
        tuning.max_seasonal_q if m else 0,
    )

    def score(p: int, q: int, sp: int, sq: int) -> float:
        key = (p, q, sp, sq)
        if key in tried:
            return tried[key]
        try:
            fit = _sarimax(y, (p, d, q), (sp, seasonal_d, sq, m) if m else (0, 0, 0, 0))
            found = float(getattr(fit, tuning.criterion))
            value = found if np.isfinite(found) else math.inf
            fits[key] = fit
        except Exception:
            value = math.inf
        tried[key] = value
        return value

    starts = (
        [(2, 2, 1, 1), (0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1)]
        if m
        else [(2, 2, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)]
    )
    bounded = [tuple(min(v, b) for v, b in zip(start, top, strict=True)) for start in starts]
    for start in dict.fromkeys(bounded):
        score(*start)
    improved = True
    while improved and time.monotonic() - started < budget:
        improved = False
        best = min(tried, key=tried.__getitem__)
        p, q, sp, sq = best
        for dp, dq, dsp, dsq in itertools.product((-1, 0, 1), repeat=4):
            candidate = (p + dp, q + dq, sp + dsp, sq + dsq)
            if (
                min(candidate) < 0
                or any(v > b for v, b in zip(candidate, top, strict=True))
                or candidate in tried
            ):
                continue
            if score(*candidate) < tried[best]:
                improved = True
            if time.monotonic() - started > budget:
                break
    best = min(tried, key=tried.__getitem__)
    if not math.isfinite(tried[best]):
        raise ModelError("No ARIMA order could be fitted.")
    p, q, sp, sq = best
    order, seasonal = (p, d, q), (sp, seasonal_d, sq, m) if m else (0, 0, 0, 0)
    label = _arima_label(order, seasonal)
    if tuning.criterion != "aicc":
        label += f" by {tuning.criterion.upper()}"
    if season and season > MAX_SEASONAL_ARIMA:
        label += " (cycle too long for seasonal terms)"
    if time.monotonic() - started > budget:
        label += " (search stopped at the time limit)"
    return order, seasonal, label, fits[best]


def sarimax_model(tuning: Tuning = DEFAULT_TUNING) -> Callable[..., Prediction]:
    """SARIMAX (028 A4): the orders of the automatic ARIMA found without drivers, then the model
    refitted with them as regressors (regression with ARIMA errors). The orders are searched once
    per training window (shared with ARIMA (automatic) under the same `tuning`), so the "with /
    without each driver" comparisons only refit."""

    def fit(
        y: np.ndarray,
        h: int,
        season: int | None,
        x: np.ndarray | None = None,
        x_future: np.ndarray | None = None,
    ) -> Prediction:
        order, seasonal, label = arima_orders(y, season, tuning)
        if x is None or x_future is None or not x.shape[1]:
            try:
                result = _sarimax(y, order, seasonal)
            except Exception as error:
                raise ModelError(f"SARIMAX could not be fitted: {error}") from error
            return _arima_prediction(result, h, f"{label} · no drivers")
        try:
            result = _sarimax(y, order, seasonal, x)
            prediction = _arima_prediction(result, h, label, x_future[:h])
        except Exception as error:
            raise ModelError(f"SARIMAX could not be fitted: {error}") from error
        names = list(result.model.param_names)
        params, pvalues = np.asarray(result.params), np.asarray(result.pvalues)
        at = [names.index(name) for name in result.model.exog_names]
        prediction.coef, prediction.pvalues = params[at], pvalues[at]
        return prediction

    return fit


def sarima_manual(
    order: tuple[int, int, int], seasonal: tuple[int, int, int]
) -> Callable[[np.ndarray, int, int | None], Prediction]:
    def fit(y: np.ndarray, h: int, season: int | None) -> Prediction:
        m = season if season and seasonal != (0, 0, 0) else 0
        full = (*seasonal, m) if m else (0, 0, 0, 0)
        try:
            result = _sarimax(y, order, full)
        except Exception as error:
            raise ModelError(f"SARIMA could not be fitted: {error}") from error
        return _arima_prediction(result, h, _arima_label(order, full))

    return fit


# --- Regression on time, and machine learning on lags --------------------------------------------


def _season_dummies(t: np.ndarray, season: int | None) -> np.ndarray:
    if not season or season < 2 or season > 52:
        return np.empty((len(t), 0))
    position = t % season
    return np.column_stack([(position == k).astype(float) for k in range(1, season)])


def linear_trend_season(
    y: np.ndarray,
    h: int,
    season: int | None,
    x: np.ndarray | None = None,
    x_future: np.ndarray | None = None,
) -> Prediction:
    """Linear regression on a time trend and one indicator per position of the cycle
    (Hyndman's TSLM), with its exact prediction intervals. Drivers (028 A4) are extra columns;
    the seasonal indicators are dropped when they repeat a calendar driver (day of week)."""
    import statsmodels.api as sm

    n = len(y)
    use_season = bool(season and 2 <= season <= 52 and n >= 2 * season)
    t = np.arange(n + h, dtype=float)
    k = 0 if x is None or x_future is None else x.shape[1]
    exog = np.vstack([x, x_future[:h]]) if k and x is not None and x_future is not None else None

    def design_of(seasonal: bool) -> np.ndarray:
        parts = [np.ones(n + h), t, _season_dummies(t, season if seasonal else None)]
        return np.column_stack([*parts, exog] if exog is not None else parts)

    design = design_of(use_season)
    if exog is not None and use_season and np.linalg.matrix_rank(design[:n]) < design.shape[1]:
        use_season = False
        design = design_of(False)
    fit = _quiet(lambda: sm.OLS(y, design[:n]).fit())
    frames = [fit.get_prediction(design[n:]).summary_frame(alpha=a) for a in (0.2, 0.05)]
    label = "Trend + seasonal dummies" if use_season else "Linear trend"
    prediction = Prediction(
        np.asarray(frames[0]["mean"], dtype=float),
        np.asarray(frames[0]["obs_ci_lower"], dtype=float),
        np.asarray(frames[0]["obs_ci_upper"], dtype=float),
        np.asarray(frames[1]["obs_ci_lower"], dtype=float),
        np.asarray(frames[1]["obs_ci_upper"], dtype=float),
        label,
        np.asarray(fit.resid, dtype=float),
    )
    if k:
        prediction.coef = np.asarray(fit.params, dtype=float)[-k:]
        prediction.pvalues = np.asarray(fit.pvalues, dtype=float)[-k:]
    return prediction


GBM_LEAVES = 15  # leaves per tree of the fixed gradient boosting


def lag_grid(grid: GbmGrid, lags: int) -> list[tuple[int, int]]:
    """The (lags, leaves per tree) tried by gradient boosting (028 A7): fixed = the default
    alone; small = half or all of the cycle's lags × 7 or 15 leaves (4); wide = half, all or
    twice the lags × 7, 15 or 31 leaves (9)."""
    if grid == "fixed":
        return [(lags, GBM_LEAVES)]
    counts = [max(3, lags // 2), lags] + ([min(2 * lags, 48)] if grid == "wide" else [])
    leaves = [7, 15] + ([31] if grid == "wide" else [])
    return list(dict.fromkeys((c, leaf) for c in counts for leaf in leaves))


def _lag_model(
    kind: str, grid: GbmGrid = "fixed"
) -> Callable[[np.ndarray, int, int | None], Prediction]:
    """Gradient boosting or a small neural network predicting each period from the previous
    ones, on the series with its linear trend removed (trees cannot extrapolate a trend), then
    forecasting one step at a time. Its own intervals (in-sample one-step errors widened with the
    horizon) only serve the validation; the final ones are conformal (`run_model`).

    Drivers (028 A4) enter twice: in the linear part removed first (trend + drivers, so that a
    driver leaving its past range still moves the forecast) and as features of the period being
    predicted (for what is not linear).

    `grid` (gradient boosting, 028 A7): the lags and leaves of `lag_grid`, each scored on the last
    h periods of the training window (an inner hold-out, so the rolling validation stays honest),
    the best refitted on the whole window."""

    def one(
        y: np.ndarray,
        h: int,
        season: int | None,
        x: np.ndarray | None,
        x_future: np.ndarray | None,
        lags: int,
        leaves: int,
        importance: bool = True,
    ) -> Prediction:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        n = len(y)
        if n < lags + 10:
            raise ModelError(f"Needs at least {lags + 10} periods.")
        k = 0 if x is None or x_future is None else x.shape[1]
        drivers = (
            np.vstack([x, x_future[:h]]) if k and x is not None and x_future is not None else None
        )
        t = np.arange(n + h, dtype=float)
        basis = np.column_stack(
            [np.ones(n + h), t] if drivers is None else [np.ones(n + h), t, drivers]
        )
        beta = np.linalg.lstsq(basis[:n], y, rcond=None)[0]
        linear = basis @ beta
        rest = y - linear[:n]

        def features(history: np.ndarray, position: int, given: np.ndarray | None) -> list[float]:
            row = list(history[-lags:][::-1])
            if season and season >= 2:
                angle = 2 * np.pi * (position % season) / season
                row += [float(np.sin(angle)), float(np.cos(angle))]
            if given is not None:
                row += [float(v) for v in given[position]]
            return row

        inputs = np.array([features(rest[:i], i, drivers) for i in range(lags, n)])
        target = rest[lags:]
        if kind == "gbm":
            model = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=leaves, random_state=42
            )
        else:
            model = make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=2_000, random_state=42
                ),
            )
        _quiet(lambda: model.fit(inputs, target))
        resid = target - np.asarray(model.predict(inputs), dtype=float)
        history = list(rest)
        steps = []
        for step in range(h):
            row = features(np.array(history), n + step, drivers)
            value = float(model.predict(np.array([row]))[0])
            history.append(value)
            steps.append(value + linear[n + step])
        sigma = float(np.sqrt(np.mean(resid**2))) * np.sqrt(np.arange(1, h + 1))
        label = (
            "Gradient boosting" if kind == "gbm" else "Neural network (MLP)"
        ) + f" on {lags} lags"
        prediction = _band(np.array(steps), sigma, label, resid)
        prediction.approximate = True
        if drivers is not None and importance:

            def fitted(given: np.ndarray) -> np.ndarray:
                """In-sample one-step prediction of y with these driver values."""
                base = np.column_stack([np.ones(n), t[:n], given]) @ beta
                rows = inputs.copy()
                rows[:, inputs.shape[1] - k :] = given[lags:]
                return base[lags:] + np.asarray(model.predict(rows), dtype=float)

            prediction.importance = _shuffled_error(fitted, drivers[:n], y[lags:])
        return prediction

    def fit(
        y: np.ndarray,
        h: int,
        season: int | None,
        x: np.ndarray | None = None,
        x_future: np.ndarray | None = None,
    ) -> Prediction:
        n = len(y)
        lags = max(3, min(season or 12, 24, n // 4))
        if kind != "gbm" or grid == "fixed":
            return one(y, h, season, x, x_future, lags, GBM_LEAVES)
        tried = lag_grid(grid, lags)
        hold = max(1, min(h, n // 5))
        best: tuple[float, int, int] | None = None
        for count, leaves in tried:
            past = None if x is None else x[:-hold]
            given = None if x is None else x[-hold:]
            try:
                trial = one(y[:-hold], hold, season, past, given, count, leaves, False)
            except ModelError:
                continue
            error = float(np.mean(np.abs(trial.mean - y[-hold:])))
            if best is None or error < best[0]:
                best = (error, count, leaves)
        if best is None:
            raise ModelError(f"Needs at least {lags + 10 + hold} periods for the grid.")
        _, count, leaves = best
        prediction = one(y, h, season, x, x_future, count, leaves)
        prediction.info = (
            f"Gradient boosting on {count} lags, {leaves} leaves (grid {grid}: best of "
            f"{len(tried)} on the last {hold} periods)"
        )
        return prediction

    return fit


def _shuffled_error(
    fitted: Callable[[np.ndarray], np.ndarray], given: np.ndarray, actual: np.ndarray
) -> np.ndarray:
    """Permutation importance of each driver column: how much the in-sample RMSE rises, in %,
    when that column is shuffled (fixed seed, mean of 5 shuffles)."""
    rng = np.random.default_rng(0)
    base = float(np.sqrt(np.mean((actual - fitted(given)) ** 2))) or 1e-12
    rises = []
    for column in range(given.shape[1]):
        errors = []
        for _ in range(5):
            shuffled = given.copy()
            shuffled[:, column] = rng.permutation(shuffled[:, column])
            errors.append(float(np.sqrt(np.mean((actual - fitted(shuffled)) ** 2))))
        rises.append((float(np.mean(errors)) - base) / base * 100)
    return np.array(rises)


# --- Intermittent demand (many zeros) ------------------------------------------------------------

INTERMITTENT_ALPHA = 0.1  # the classic smoothing constant of Croston-type methods
ADI_CUT, CV2_CUT = 1.32, 0.49  # Syntetos, Boylan & Croston (2005)


@dataclass
class DemandPattern:
    """Syntetos-Boylan classification of a series with zeros."""

    zeros_pct: float
    adi: float  # average number of periods per non-zero value
    cv2: float  # squared coefficient of variation of the non-zero values
    kind: str  # smooth, erratic, intermittent or lumpy
    intermittent: bool  # Croston-type models are worth trying


def demand_pattern(y: np.ndarray) -> DemandPattern | None:
    """None when the series has negative values, no zero or fewer than two non-zero values."""
    if (y < 0).any() or (y > 0).all():
        return None
    sizes = y[y > 0]
    if len(sizes) < 2:
        return None
    adi = len(y) / len(sizes)
    cv2 = float((np.std(sizes) / np.mean(sizes)) ** 2)
    kind = {
        (False, False): "smooth",
        (False, True): "erratic",
        (True, False): "intermittent",
        (True, True): "lumpy",
    }[(adi >= ADI_CUT, cv2 >= CV2_CUT)]
    zeros = float(np.mean(y == 0))
    return DemandPattern(
        round(zeros * 100, 1), float(adi), cv2, kind, adi >= ADI_CUT or zeros >= 0.3
    )


def _demands(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Non-zero values and the number of periods since the previous one (the first one counts
    from the start of the series)."""
    if (y < 0).any():
        raise ModelError("Croston-type models need values ≥ 0.")
    at = np.flatnonzero(y > 0)
    if len(at) == 0:
        raise ModelError("Every value is zero: there is nothing to forecast.")
    return y[at], np.diff(np.concatenate(([-1], at))).astype(float)


def _ses_level(values: np.ndarray, alpha: float) -> float:
    level = float(values[0])
    for value in values[1:]:
        level += alpha * (float(value) - level)
    return level


def _flat(value: float, h: int, info: str, y: np.ndarray) -> Prediction:
    prediction = _band(np.full(h, value), np.full(h, _rms(y - value)), info, y - value)
    prediction.approximate = True
    return prediction


def croston(y: np.ndarray, h: int, season: int | None) -> Prediction:
    """Croston (1972): exponential smoothing of the non-zero sizes and of the intervals between
    them, updated only when a demand occurs; forecast = size / interval."""
    sizes, intervals = _demands(y)
    a = INTERMITTENT_ALPHA
    return _flat(_ses_level(sizes, a) / _ses_level(intervals, a), h, "Croston (α = 0.1)", y)


def sba(y: np.ndarray, h: int, season: int | None) -> Prediction:
    """Syntetos-Boylan approximation (2005): Croston times (1 - α/2), which removes its bias."""
    sizes, intervals = _demands(y)
    a = INTERMITTENT_ALPHA
    value = (1 - a / 2) * _ses_level(sizes, a) / _ses_level(intervals, a)
    return _flat(value, h, "SBA (α = 0.1)", y)


def tsb(y: np.ndarray, h: int, season: int | None) -> Prediction:
    """Teunter, Syntetos & Babai (2011): the probability of a demand is updated every period, so
    it decays when demand stops; the size only when a demand occurs."""
    sizes, _ = _demands(y)
    a = b = INTERMITTENT_ALPHA
    first = int(np.flatnonzero(y > 0)[0])
    probability, size = 1.0, float(sizes[0])
    for value in y[first + 1 :]:
        probability += b * (float(value > 0) - probability)
        if value > 0:
            size += a * (float(value) - size)
    return _flat(probability * size, h, "TSB (α = β = 0.1)", y)


def adida(y: np.ndarray, h: int, season: int | None) -> Prediction:
    """Nikolopoulos et al. (2011): sum the series over buckets as long as the average interval
    between demands (most zeros disappear), forecast the sums by simple exponential smoothing
    (α by least squares), then spread that forecast evenly over the periods of a bucket."""
    sizes, _ = _demands(y)
    bucket = max(1, round(len(y) / len(sizes)))
    usable = len(y) - len(y) % bucket
    sums = y[len(y) - usable :].reshape(-1, bucket).sum(axis=1)
    if len(sums) < 3:
        raise ModelError("Too few periods for the aggregation.")

    def sse(alpha: float) -> float:
        level, total = float(sums[0]), 0.0
        for value in sums[1:]:
            total += (float(value) - level) ** 2
            level += alpha * (float(value) - level)
        return total

    alpha = float(min(np.linspace(0.05, 0.95, 19), key=sse))
    value = _ses_level(sums, alpha) / bucket
    return _flat(value, h, f"ADIDA (buckets of {bucket}, α = {alpha:.2f})", y)


Model = Callable[[np.ndarray, int, int | None], Prediction]

# AI models (028 phase B): pretrained (`pretrained.py`, the forecast-ai extra) and trained
# networks (`neural.py`). Their modules import this one: imported at the first call only.
PRETRAINED_MODELS = (
    "chronos_bolt_tiny",
    "chronos_bolt_small",
    "chronos_bolt_base",
    "chronos_2",
    "timesfm_2p5",
)
NEURAL_MODELS = ("nhits", "nbeats")
AI_MODELS = (*PRETRAINED_MODELS, *NEURAL_MODELS)


def ai_model(key: str, tuning: Tuning = DEFAULT_TUNING) -> Callable[..., Prediction]:
    """A pretrained model, or N-HiTS / N-BEATS with the time limit of `tuning` (028 A7)."""

    def fit(
        y: np.ndarray,
        h: int,
        season: int | None,
        x: np.ndarray | None = None,
        x_future: np.ndarray | None = None,
    ) -> Prediction:
        if key in NEURAL_MODELS:
            from quantilo.timeseries import neural

            return neural.model(key, tuning)(y, h, season)
        from quantilo.timeseries import pretrained

        return pretrained.model(key)(y, h, season, x, x_future)

    return fit


MODELS: dict[str, Model] = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "ets_auto": ets_auto,
    "arima_auto": arima_auto,
    "drift": drift,
    "mean": mean_model,
    "ses": ses,
    "holt": holt,
    "holt_winters": holt_winters,
    "theta": theta,
    "sarima": naive,  # replaced by sarima_manual(...) with the user's orders
    "tslm": linear_trend_season,
    "structural": structural,
    "gbm_lags": _lag_model("gbm"),  # lag_model("gbm", grid) with a grid (028 A7)
    "mlp_lags": _lag_model("mlp"),
    "croston": croston,
    "sba": sba,
    "tsb": tsb,
    "adida": adida,
    "combination": naive,  # not a model: averages the best ones, after the others (`combine`)
    "sarimax": sarimax_model(),  # a fresh one (its own cache of orders) per forecast
    # 028 A6; their cycles follow the period, inferred here from the season (the service gives
    # the period itself: `PERIOD_MODELS`).
    "prophet_like": prophet_model(),
    "stlf": stlf_model(),
    **{key: ai_model(key) for key in AI_MODELS},
}
lag_model = _lag_model
INTERMITTENT_MODELS = ("croston", "sba", "tsb", "adida")
# The models that use drivers (028 A4, A6).
DRIVER_MODELS = ("sarimax", "tslm", "gbm_lags", "prophet_like", "chronos_2")
PERIOD_MODELS = {"prophet_like": prophet_model, "stlf": stlf_model}  # built for a period
COMBINED = 3  # models averaged by "combination"
DEFAULT_MODELS = ("naive", "seasonal_naive", "ets_auto", "arima_auto")


# --- Validation and scores -----------------------------------------------------------------------


@dataclass
class Scores:
    mae: float | None
    rmse: float | None
    mape: float | None  # only when every actual value is above 0
    smape: float | None
    wape: float | None  # sum of absolute errors / sum of actual values, in %
    mase: float | None  # < 1: better than the naive reference
    rmsse: float | None
    bias: float | None  # mean error (forecast - actual)
    coverage80: float | None  # share of actual values inside the 80 % interval
    coverage95: float | None
    winkler95: float | None  # interval width plus penalties for misses (lower is better)


@dataclass
class Backtest:
    """One rolling origin: the model saw the first `origin` periods and forecast the next ones."""

    origin: int
    forecast: list[float]
    low95: list[float | None]
    high95: list[float | None]


@dataclass
class DriverEffect:
    """What one driver brings to one model (028 A4)."""

    driver: str
    coefficient: float | None  # change of the forecast per unit (linear models, one column)
    p_value: float | None
    importance: float | None  # gradient boosting: rise of the in-sample error when shuffled, %
    mase_without: float | None  # the same model without this driver, on the same origins
    # helps: MASE at least 5 % lower with it; no_help: not lower at all; unclear: in between
    # (a few % is within the noise of a handful of origins).
    verdict: Literal["helps", "unclear", "no_help"] | None


@dataclass
class ModelResult:
    key: str
    info: str | None
    scores: Scores | None
    vs_reference: float | None  # relative MASE improvement over the reference, in %
    forecast: list[float] = field(default_factory=list)
    low80: list[float] = field(default_factory=list)
    high80: list[float] = field(default_factory=list)
    low95: list[float] = field(default_factory=list)
    high95: list[float] = field(default_factory=list)
    residual_check: MethodResult | None = None  # Ljung-Box on the residuals (ARIMA, ETS)
    seconds: float = 0.0
    error: str | None = None
    backtests: list[Backtest] = field(default_factory=list)
    uses_drivers: bool = False
    mase_without_drivers: float | None = None  # the same model without any driver
    drivers: list[DriverEffect] = field(default_factory=list)
    # Prophet-like (028 A6 story 4): trend, each season, the holidays' and drivers' effects of the
    # final fit, from period `components_start` of the history to the end of the horizon.
    components: list[Component] = field(default_factory=list)
    components_start: int = 0
    timed_out: bool = False  # stopped at the time limit (028 A7)


def origins(n: int, h: int, season: int | None) -> list[int]:
    """Training lengths of the rolling origins: up to 5, each followed by h unseen periods."""
    minimum = max((season or 1) + 2, 10)  # one cycle; models needing more skip that origin
    found = [n - h - i * max(1, h // 2) for i in range(5)]
    return sorted(t for t in found if t >= minimum)[-5:] or ([n - h] if n - h >= 4 else [])


def _scale(train: np.ndarray, season: int | None) -> tuple[float, float]:
    """In-sample MAE and RMSE of the seasonal naive (naive without a season): MASE, RMSSE scales."""
    m = season if season and len(train) > season + 1 else 1
    diff = train[m:] - train[:-m]
    return float(np.mean(np.abs(diff))) or 1.0, float(np.sqrt(np.mean(diff**2))) or 1.0


@dataclass
class OriginRun:
    size: int  # periods seen
    prediction: Prediction


def _over(deadline: float | None, limit: float | None, what: str) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise TimeLimit(f"Stopped at the time limit ({limit:g} s) {what}.")


def origin_runs(
    y: np.ndarray, model: Model, h: int, season: int | None, limit: float | None = None
) -> list[OriginRun]:
    """The model fitted at each rolling origin; with `limit` (seconds), it stops between two
    origins once over it (`TimeLimit`)."""
    runs = []
    deadline = time.monotonic() + limit if limit else None
    sizes = origins(len(y), h, season)
    for done, size in enumerate(sizes):
        test = y[size : size + h]
        if not len(test):
            continue
        _over(deadline, limit, f"after {done} of {len(sizes)} rolling origins")
        try:
            runs.append(OriginRun(size, model(y[:size], len(test), season)))
        except Unavailable:
            raise
        except ModelError:
            continue  # too little history at this origin for this model
    return runs


def evaluate(
    y: np.ndarray, model: Model, h: int, season: int | None, limit: float | None = None
) -> Scores:
    return score(y, origin_runs(y, model, h, season, limit), h, season)


def score(y: np.ndarray, runs: list[OriginRun], h: int, season: int | None) -> Scores:
    errors, actuals, scaled_abs, scaled_sq = [], [], [], []
    in80, in95, winkler = [], [], []
    for run in runs:
        train, test, prediction = y[: run.size], y[run.size : run.size + h], run.prediction
        error = prediction.mean[: len(test)] - test
        mae_scale, rmse_scale = _scale(train, season)
        errors.extend(error)
        actuals.extend(test)
        scaled_abs.extend(np.abs(error) / mae_scale)
        scaled_sq.extend((error / rmse_scale) ** 2)
        lo80, hi80 = prediction.low80[: len(test)], prediction.high80[: len(test)]
        lo, hi = prediction.low95[: len(test)], prediction.high95[: len(test)]
        known = np.isfinite(lo) & np.isfinite(hi)  # a combination has none at its first origin
        in80.extend(((test >= lo80) & (test <= hi80))[known])
        in95.extend(((test >= lo) & (test <= hi))[known])
        penalty = np.where(
            test < lo, 2 / 0.05 * (lo - test), np.where(test > hi, 2 / 0.05 * (test - hi), 0)
        )
        winkler.extend((hi - lo + penalty)[known])
    if not errors:
        raise ModelError("The series is too short to validate this horizon.")
    e, a = np.array(errors), np.array(actuals)
    positive = bool((a > 0).all())
    denominator = np.abs(a) + np.abs(a + e)
    total = float(np.sum(np.abs(a)))
    return Scores(
        mae=clean(np.mean(np.abs(e))),
        rmse=clean(np.sqrt(np.mean(e**2))),
        mape=clean(np.mean(np.abs(e / a)) * 100) if positive else None,
        smape=clean(
            np.mean(
                np.divide(2 * np.abs(e), denominator, out=np.zeros_like(e), where=denominator > 0)
            )
            * 100
        ),
        wape=clean(np.sum(np.abs(e)) / total * 100) if total > 0 else None,
        mase=clean(np.mean(scaled_abs)),
        rmsse=clean(np.sqrt(np.mean(scaled_sq))),
        bias=clean(np.mean(e)),
        coverage80=clean(np.mean(in80)) if in80 else None,
        coverage95=clean(np.mean(in95)) if in95 else None,
        winkler95=clean(np.mean(winkler)) if winkler else None,
    )


def conformal(
    y: np.ndarray, runs: list[OriginRun], mean: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Intervals from validation errors (split conformal): the error at step i is divided by √i
    so that every step pools together, the 80 % and 95 % quantiles are taken, then widened back
    by √i. None with fewer than 8 errors."""
    scaled = [
        abs(run.prediction.mean[i] - y[run.size + i]) / math.sqrt(i + 1)
        for run in runs
        for i in range(min(len(run.prediction.mean), len(y) - run.size))
    ]
    if len(scaled) < 8:
        return None
    root = np.sqrt(np.arange(1, len(mean) + 1))
    q80 = float(np.quantile(scaled, 0.8, method="higher")) * root
    q95 = float(np.quantile(scaled, 0.95, method="higher")) * root
    return mean - q80, mean + q80, mean - q95, mean + q95


def _numbers(values: np.ndarray) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in values]


def _backtests(runs: list[OriginRun]) -> list[Backtest]:
    return [
        Backtest(
            run.size,
            [float(v) for v in run.prediction.mean],
            _numbers(run.prediction.low95),
            _numbers(run.prediction.high95),
        )
        for run in runs
    ]


def run_model(
    key: str,
    model: Model,
    y: np.ndarray,
    h: int,
    season: int | None,
    limit: float | None = None,
) -> ModelResult:
    """Score a model on the rolling origins, then refit it on the whole series. `limit`
    (seconds, 028 A7): over it, the model stops between two fits and its row says so."""
    started = time.monotonic()
    try:
        runs = origin_runs(y, model, h, season, limit)
        scores = score(y, runs, h, season)
        _over(started + limit if limit else None, limit, "before the final fit")
        final = model(y, h, season)
    except (ModelError, ValueError, np.linalg.LinAlgError) as error:
        return ModelResult(
            key,
            None,
            None,
            None,
            error=str(error),
            seconds=time.monotonic() - started,
            timed_out=isinstance(error, TimeLimit),
        )
    if final.approximate and (bands := conformal(y, runs, final.mean)):
        final.low80, final.high80, final.low95, final.high95 = bands
        final.info = f"{final.info} · conformal intervals"
    check = None
    if (
        final.resid is not None
        and len(final.resid) >= 20
        and key not in ("naive", "seasonal_naive", "drift", "mean")
    ):
        check = ljung_box(final.resid, season)
    return ModelResult(
        key,
        final.info,
        scores,
        None,
        [float(v) for v in final.mean],
        [float(v) for v in final.low80],
        [float(v) for v in final.high80],
        [float(v) for v in final.low95],
        [float(v) for v in final.high95],
        check,
        time.monotonic() - started,
        backtests=_backtests(runs),
        components=final.components,
        components_start=final.start,
    )


def combine(results: list[ModelResult], y: np.ndarray, h: int, season: int | None) -> ModelResult:
    """Mean of the COMBINED best models, scored without hindsight: at each rolling origin the
    models are ranked on the earlier origins only (all are averaged at the first one), and the
    validation intervals come from the combination's own earlier errors. The final forecast
    averages the COMBINED best on the whole validation."""
    started = time.monotonic()
    scored = [r for r in results if r.scores and r.scores.mase is not None and r.backtests]
    if len(scored) < 2:
        return ModelResult(
            "combination", None, None, None, error="Needs at least two models that could be scored."
        )
    by_origin = {r.key: {b.origin: b for b in r.backtests} for r in scored}
    sizes = sorted({size for backtests in by_origin.values() for size in backtests})

    def past_error(result: ModelResult, before: list[int]) -> float:
        errors = [
            abs(v - y[size + i])
            for size in before
            if (backtest := by_origin[result.key].get(size))
            for i, v in enumerate(backtest.forecast)
        ]
        return float(np.mean(errors)) if errors else math.inf

    runs: list[OriginRun] = []
    for k, size in enumerate(sizes):
        available = [r for r in scored if size in by_origin[r.key]]
        if k:
            available = sorted(available, key=lambda r: past_error(r, sizes[:k]))[:COMBINED]
        mean = np.mean([by_origin[r.key][size].forecast for r in available], axis=0)
        bands = conformal(y, runs, mean)
        nan = np.full(len(mean), np.nan)
        low80, high80, low95, high95 = bands or (nan, nan, nan, nan)
        runs.append(OriginRun(size, Prediction(mean, low80, high80, low95, high95)))

    def mase(result: ModelResult) -> float:
        return result.scores.mase if result.scores and result.scores.mase else math.inf

    members = sorted(scored, key=mase)[:COMBINED]
    mean = np.mean([r.forecast for r in members], axis=0)
    low80, high80, low95, high95 = (
        [float(v) for v in band] for band in conformal(y, runs, mean) or (mean,) * 4
    )
    names = " + ".join((r.info or r.key).split(" ·")[0] for r in members)
    return ModelResult(
        "combination",
        f"Mean of {names}",
        score(y, runs, h, season),
        None,
        [float(v) for v in mean],
        low80,
        high80,
        low95,
        high95,
        seconds=time.monotonic() - started,
        backtests=_backtests(runs),
    )


def compare_to_reference(results: list[ModelResult], season: int | None) -> str:
    """Fill each model's improvement over the naive reference; returns the reference key."""
    reference_key = (
        "seasonal_naive"
        if season and any(r.key == "seasonal_naive" and r.scores for r in results)
        else "naive"
    )
    reference = next((r for r in results if r.key == reference_key and r.scores), None)
    if reference and reference.scores and reference.scores.mase:
        for result in results:
            if result.scores and result.scores.mase is not None:
                result.vs_reference = clean((1 - result.scores.mase / reference.scores.mase) * 100)
    return reference_key
