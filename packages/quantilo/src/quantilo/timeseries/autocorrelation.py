"""Does the series remember its past? ACF, PACF, portmanteau and randomness tests, and a reading
of the classic shapes."""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf, pacf

from quantilo.timeseries.methods import ALPHA, MethodResult
from quantilo.variable.source import clean

PacfMethod = Literal["ywm", "ols", "burg"]
PACF_METHODS: tuple[PacfMethod, ...] = ("ywm", "ols", "burg")
MIN_PERIODS = 20


@dataclass
class Correlogram:
    lags: list[int]
    acf: list[float]
    pacf: list[float]
    band: float  # ± 1.96 / √n
    readings: list[str] = field(default_factory=list)  # keys of the guided reading
    ar_order: int | None = None  # where the PACF cuts off, when it does


def _quiet(compute: Callable):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute()


def _not_run(key: str, n: int, reason: str) -> MethodResult:
    return MethodResult(key, None, None, n, None, note=reason)


def portmanteau_lags(n: int, season: int | None) -> list[int]:
    """Lag 1, one cycle and two cycles, never beyond n / 5."""
    candidates = [1, 10] if not season or season < 2 else [1, season, 2 * season]
    return sorted({lag for lag in candidates if lag <= max(1, n // 5)})


def correlogram(series: np.ndarray, season: int | None, method: PacfMethod = "ywm") -> Correlogram:
    n = len(series)
    nlags = max(1, min(40, n // 4))
    autocorrelations = _quiet(lambda: acf(series, nlags=nlags, fft=True))
    partials = _quiet(lambda: pacf(series, nlags=min(nlags, n // 2 - 1), method=method))
    band = 1.96 / math.sqrt(n)
    lags = list(range(1, nlags + 1))
    result = Correlogram(
        lags,
        [float(v) for v in autocorrelations[1:]],
        [float(v) for v in partials[1:]],
        band,
    )
    read(result, season)
    return result


def read(result: Correlogram, season: int | None) -> None:
    """Classic shapes, each from a simple, tested rule."""
    acf_values, pacf_values, band = result.acf, result.pacf, result.band
    significant = [abs(v) > band for v in acf_values]
    if len(acf_values) >= 10 and acf_values[9] > 0.5:
        result.readings.append("slow_decay")
    if season and season < len(acf_values) and significant[season - 1]:
        doubled = 2 * season <= len(acf_values) and significant[2 * season - 1]
        if doubled or len(acf_values) < 2 * season:
            result.readings.append("seasonal_spikes")
    if sum(significant[:10]) <= 1 and not result.readings:
        result.readings.append("white_noise")  # one significant lag in ten is chance
        return
    pacf_significant = [abs(v) > band for v in pacf_values]
    last = max((i for i, s in enumerate(pacf_significant[:10]) if s), default=None)
    if (
        last is not None
        and last + 4 <= len(pacf_significant)
        and not any(pacf_significant[last + 1 : last + 4])
        and "slow_decay" not in result.readings
    ):
        result.ar_order = last + 1
        result.readings.append("pacf_cutoff")


def _trend_fit(series: np.ndarray):
    x = np.column_stack([np.ones(len(series)), np.arange(len(series), dtype=float)])
    return OLS(series, x).fit()


def ljung_box(series: np.ndarray, season: int | None, boxpierce: bool = False) -> MethodResult:
    key = "box_pierce" if boxpierce else "ljung_box"
    n = len(series)
    if n < MIN_PERIODS or np.ptp(series) == 0:
        return _not_run(key, n, f"Needs at least {MIN_PERIODS} periods that vary.")
    lags = portmanteau_lags(n, season)
    table = _quiet(lambda: acorr_ljungbox(series, lags=lags, boxpierce=boxpierce))
    column = "bp" if boxpierce else "lb"
    statistics = [float(v) for v in table[f"{column}_stat"]]
    pvalues = [float(v) for v in table[f"{column}_pvalue"]]
    first = next((lag for lag, p in zip(lags, pvalues, strict=True) if p < ALPHA), None)
    shown = ", ".join(f"lag {lag}: p = {p:.3g}" for lag, p in zip(lags, pvalues, strict=True))
    index = lags.index(first) if first is not None else len(lags) - 1
    return MethodResult(
        key,
        clean(statistics[index]),
        clean(pvalues[index]),
        n,
        first is not None,
        value=float(first) if first is not None else None,
        note=f"{shown}.",
    )


def box_pierce(series: np.ndarray, season: int | None) -> MethodResult:
    return ljung_box(series, season, boxpierce=True)


def durbin_watson_test(series: np.ndarray, season: int | None) -> MethodResult:
    """On the residuals of a straight-line trend; about 2 = no first-order autocorrelation."""
    n = len(series)
    if n < MIN_PERIODS or np.ptp(series) == 0:
        return _not_run("durbin_watson", n, f"Needs at least {MIN_PERIODS} periods that vary.")
    statistic = float(durbin_watson(_trend_fit(series).resid))
    linked = statistic < 1.5 or statistic > 2.5
    reading = "positive" if statistic < 1.5 else "negative" if statistic > 2.5 else "none"
    return MethodResult(
        "durbin_watson",
        statistic,
        None,
        n,
        linked,
        value=statistic,
        note=f"Autocorrelation: {reading}.",
    )


def breusch_godfrey(series: np.ndarray, season: int | None) -> MethodResult:
    n = len(series)
    if n < MIN_PERIODS or np.ptp(series) == 0:
        return _not_run("breusch_godfrey", n, f"Needs at least {MIN_PERIODS} periods that vary.")
    lags = min(season or 4, n // 5) if season else min(4, n // 5)
    lm, p, _, _ = _quiet(lambda: acorr_breusch_godfrey(_trend_fit(series), nlags=max(1, lags)))
    return MethodResult(
        "breusch_godfrey",
        clean(lm),
        clean(p),
        n,
        bool(p < ALPHA),
        note=f"Up to lag {max(1, lags)}.",
    )


def runs_test(series: np.ndarray, season: int | None) -> MethodResult:
    """Runs above / below the median: too few runs = values cluster (memory)."""
    n = len(series)
    if n < MIN_PERIODS or np.ptp(series) == 0:
        return _not_run("runs", n, f"Needs at least {MIN_PERIODS} periods that vary.")
    z, p = _quiet(lambda: runstest_1samp(series, cutoff="median", correction=True))
    return MethodResult("runs", clean(z), clean(p), n, clean(p) is not None and bool(p < ALPHA))


def turning_points(series: np.ndarray, season: int | None) -> MethodResult:
    n = len(series)
    if n < MIN_PERIODS:
        return _not_run("turning_points", n, f"Needs at least {MIN_PERIODS} periods.")
    middle, before, after = series[1:-1], series[:-2], series[2:]
    count = int(
        (((middle > before) & (middle > after)) | ((middle < before) & (middle < after))).sum()
    )
    expected = 2 * (n - 2) / 3
    variance = (16 * n - 29) / 90
    z = (count - expected) / math.sqrt(variance)
    p = float(2 * stats.norm.sf(abs(z)))
    return MethodResult(
        "turning_points",
        z,
        p,
        n,
        bool(p < ALPHA),
        value=count,
        note=f"Expected {expected:.0f} for a random series.",
    )


def bartels(series: np.ndarray, season: int | None) -> MethodResult:
    """Bartels' rank von Neumann ratio: about 2 for a random order, lower when values persist."""
    n = len(series)
    if n < MIN_PERIODS or np.ptp(series) == 0:
        return _not_run("bartels", n, f"Needs at least {MIN_PERIODS} periods that vary.")
    ranks = stats.rankdata(series)
    ratio = float(np.sum(np.diff(ranks) ** 2) / np.sum((ranks - ranks.mean()) ** 2))
    variance = 4 * (n - 2) * (5 * n * n - 2 * n - 9) / (5 * n * (n + 1) * (n - 1) ** 2)
    z = (ratio - 2) / math.sqrt(variance)
    p = float(2 * stats.norm.sf(abs(z)))
    return MethodResult("bartels", z, p, n, bool(p < ALPHA), value=ratio)


AutocorrelationMethod = Callable[[np.ndarray, int | None], MethodResult]
AUTOCORRELATION: dict[str, AutocorrelationMethod] = {
    "ljung_box": ljung_box,
    "durbin_watson": durbin_watson_test,
    "box_pierce": box_pierce,
    "breusch_godfrey": breusch_godfrey,
    "runs": runs_test,
    "turning_points": turning_points,
    "bartels": bartels,
}
DEFAULT_AUTOCORRELATION = ("ljung_box", "durbin_watson")


MAX_AR = 12


@dataclass
class ArOrder:
    order: int
    aic: float | None
    bic: float | None


@dataclass
class ArCoefficient:
    term: str  # "const" or "lag k"
    estimate: float
    std_error: float | None
    p_value: float | None


@dataclass
class ArFit:
    order: int
    chosen_by: str  # "you" or "AIC"
    coefficients: list[ArCoefficient]
    sigma: float | None  # standard deviation of the innovations
    aic: float | None
    bic: float | None
    residuals: MethodResult  # Ljung-Box on the residuals
    orders: list[ArOrder]  # AIC / BIC for every order tried


def fit_ar(series: np.ndarray, order: int | None, season: int | None) -> ArFit | None:
    """An autoregressive model AR(p): p as asked, or the order with the lowest AIC."""
    from statsmodels.tsa.ar_model import AutoReg

    n = len(series)
    highest = min(MAX_AR, max(1, n // 5))
    if n < MIN_PERIODS or np.ptp(series) == 0:
        return None
    fits = {}
    orders = []
    for p in range(1, highest + 1):
        fit = _quiet(lambda p=p: AutoReg(series, lags=p, trend="c").fit())
        fits[p] = fit
        orders.append(ArOrder(p, clean(fit.aic), clean(fit.bic)))
    if order is None or order > highest:
        chosen = min(fits, key=lambda p: fits[p].aic if np.isfinite(fits[p].aic) else np.inf)
        chosen_by = "AIC"
    else:
        chosen, chosen_by = order, "you"
    fit = fits[chosen]
    names = ["const", *[f"lag {k}" for k in range(1, chosen + 1)]]
    params, errors, pvalues = (
        np.asarray(x, dtype=float) for x in (fit.params, fit.bse, fit.pvalues)
    )
    coefficients = [
        ArCoefficient(name, float(params[i]), clean(errors[i]), clean(pvalues[i]))
        for i, name in enumerate(names)
    ]
    resid = np.asarray(fit.resid, dtype=float)
    return ArFit(
        chosen,
        chosen_by,
        coefficients,
        clean(float(np.sqrt(fit.sigma2))),
        clean(fit.aic),
        clean(fit.bic),
        ljung_box(resid, season),
        orders,
    )
