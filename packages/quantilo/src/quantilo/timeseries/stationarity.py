"""Is the series stationary, and which transformation makes it so?

ADF and KPSS come from statsmodels; Phillips-Perron, DF-GLS and the variance ratio are written
here on numpy (no `arch` dependency), with MacKinnon's response surfaces for p-values.
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.stattools import adfuller, kpss, range_unit_root_test, zivot_andrews

from quantilo.timeseries.methods import ALPHA, MethodResult
from quantilo.variable.source import clean

Transform = Literal[
    "none", "log", "diff", "seasonal_diff", "log_diff", "diff2", "seasonal_diff_diff"
]
TRANSFORMS: tuple[Transform, ...] = (
    "none",
    "log",
    "diff",
    "seasonal_diff",
    "log_diff",
    "diff2",
    "seasonal_diff_diff",
)
Regression = Literal["c", "ct"]
Verdict = Literal["stationary", "unit_root", "trend_or_difference", "inconclusive"]
MIN_PERIODS = 20


@dataclass
class Synthesis:
    verdict: Verdict
    adf_rejects: bool
    kpss_rejects: bool


@dataclass
class Suggestion:
    transform: Transform
    verdict: Verdict


class TransformError(ValueError):
    pass


def transform(series: np.ndarray, how: Transform, season: int | None) -> np.ndarray:
    if how in ("log", "log_diff") and (series <= 0).any():
        raise TransformError("A log needs every value above 0.")
    needed = (season or 0) + (2 if how == "seasonal_diff_diff" else 1)
    if how in ("seasonal_diff", "seasonal_diff_diff") and (not season or len(series) <= needed):
        raise TransformError("A seasonal difference needs more than one full cycle.")
    if how == "none":
        return series
    if how == "log":
        return np.log(series)
    if how == "diff":
        return np.diff(series)
    if how == "seasonal_diff":
        return series[season:] - series[:-season]  # type: ignore[misc]
    if how == "diff2":
        return np.diff(series, n=2)
    if how == "seasonal_diff_diff":
        return np.diff(series[season:] - series[:-season])  # type: ignore[misc]
    return np.diff(np.log(series))


def _quiet(compute: Callable):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute()


def _not_run(key: str, n: int, reason: str) -> MethodResult:
    return MethodResult(key, None, None, n, None, note=reason)


def _critical(values) -> str:
    return "Critical values: " + ", ".join(f"{k} {v:.2f}" for k, v in values.items()) + "."


# --- Unit-root tests (H0: unit root, rejecting = stationary) ------------------------------------


def adf(series: np.ndarray, regression: Regression = "c") -> MethodResult:
    n = len(series)
    if n < MIN_PERIODS or np.ptp(series) == 0:
        return _not_run("adf", n, f"Needs at least {MIN_PERIODS} periods that vary.")
    statistic, p, lags, _, critical, _ = _quiet(
        lambda: adfuller(series, regression=regression, autolag="AIC")
    )
    note = f"{_critical(critical)} Lags chosen by AIC: {lags}; deterministic terms: {regression}."
    return MethodResult("adf", clean(statistic), clean(p), n, bool(p < ALPHA), note=note)


def kpss_test(series: np.ndarray, regression: Regression = "c") -> MethodResult:
    """H0: stationary. statsmodels caps its p-value to [0.01, 0.10]: said in the note."""
    n = len(series)
    if n < MIN_PERIODS or np.ptp(series) == 0:
        return _not_run("kpss", n, f"Needs at least {MIN_PERIODS} periods that vary.")
    statistic, p, lags, critical = _quiet(lambda: kpss(series, regression=regression, nlags="auto"))
    bounded = (
        " (p is beyond the table: > 0.10)"
        if p >= 0.1
        else " (p is beyond the table: < 0.01)"
        if p <= 0.01
        else ""
    )
    around = "trend" if regression == "ct" else "level"
    note = f"{_critical(critical)} Lags: {lags}; around a {around}{bounded}."
    return MethodResult("kpss", clean(statistic), clean(p), n, bool(p < ALPHA), note=note)


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coefficients, their standard errors and the residuals."""
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    s2 = float(resid @ resid) / (len(y) - x.shape[1])
    cov = s2 * np.linalg.inv(x.T @ x)
    return beta, np.sqrt(np.diag(cov)), resid


def _deterministic(n: int, regression: Regression) -> np.ndarray:
    columns = [np.ones(n)]
    if regression == "ct":
        columns.append(np.arange(1, n + 1, dtype=float))
    return np.column_stack(columns)


def phillips_perron(
    series: np.ndarray, regression: Regression = "c", lags: int | None = None
) -> MethodResult:
    """Z_t statistic: the Dickey-Fuller t corrected for autocorrelation and heteroscedasticity
    with a Newey-West long-run variance (Bartlett kernel)."""
    n = len(series)
    if n < MIN_PERIODS or np.ptp(series) == 0:
        return _not_run("phillips_perron", n, f"Needs at least {MIN_PERIODS} periods that vary.")
    y, lagged = np.diff(series), series[:-1]
    t = len(y)
    x = np.column_stack([_deterministic(t, regression), lagged])
    beta, errors, resid = _ols(y, x)
    t_stat = beta[-1] / errors[-1]
    lags = math.ceil(12 * (t / 100) ** 0.25) if lags is None else lags
    gamma0 = float(resid @ resid) / t
    long_run = gamma0
    for j in range(1, lags + 1):
        long_run += 2 * (1 - j / (lags + 1)) * float(resid[j:] @ resid[:-j]) / t
    s = math.sqrt(float(resid @ resid) / (t - x.shape[1]))
    z_t = math.sqrt(gamma0 / long_run) * t_stat - 0.5 * (long_run - gamma0) / math.sqrt(
        long_run
    ) * (t * errors[-1] / s)
    p = float(mackinnonp(z_t, regression=regression, N=1))
    critical = dict(
        zip(("1%", "5%", "10%"), mackinnoncrit(N=1, regression=regression, nobs=t), strict=True)
    )
    note = f"{_critical(critical)} Newey-West lags: {lags}."
    return MethodResult("phillips_perron", clean(z_t), clean(p), n, bool(p < ALPHA), note=note)


def df_gls(series: np.ndarray, regression: Regression = "c") -> MethodResult:
    """Elliott-Rothenberg-Stock: remove the mean by GLS, then a Dickey-Fuller test without
    deterministic terms (lags by BIC). More powerful than ADF near a unit root."""
    n = len(series)
    if n < MIN_PERIODS or np.ptp(series) == 0:
        return _not_run("df_gls", n, f"Needs at least {MIN_PERIODS} periods that vary.")
    if regression == "ct":
        regression = "c"  # p-values exist for the constant case only
    alpha = 1 - 7 / n
    z = np.ones((n, 1))
    yd = np.concatenate([[series[0]], series[1:] - alpha * series[:-1]])
    zd = np.vstack([z[:1], z[1:] - alpha * z[:-1]])
    beta, *_ = np.linalg.lstsq(zd, yd, rcond=None)
    detrended = series - z @ beta
    max_lags = int(12 * (n / 100) ** 0.25)
    statistic, _, lags, *_ = _quiet(
        lambda: adfuller(detrended, maxlag=max_lags, regression="n", autolag="BIC")
    )
    p = float(mackinnonp(statistic, regression="n", N=1))
    critical = dict(
        zip(("1%", "5%", "10%"), mackinnoncrit(N=1, regression="n", nobs=n), strict=True)
    )
    note = (
        f"{_critical(critical)} Lags chosen by BIC (AIC over-fits here): {lags}; demeaned by GLS."
    )
    return MethodResult("df_gls", clean(statistic), clean(p), n, bool(p < ALPHA), note=note)


def zivot_andrews_test(series: np.ndarray, regression: Regression = "c") -> MethodResult:
    """A unit-root test that allows one structural break at an unknown date."""
    n = len(series)
    if n < 30 or np.ptp(series) == 0:
        return _not_run("zivot_andrews", n, "Needs at least 30 periods that vary.")
    if n > 1_000:
        return _not_run(
            "zivot_andrews", n, "Too slow beyond 1,000 periods: choose a coarser period."
        )
    statistic, p, critical, lags, breakpoint = _quiet(
        lambda: zivot_andrews(series, regression=regression, autolag="AIC")
    )
    note = f"{_critical(critical)} Most likely break at period {int(breakpoint) + 1}; lags: {lags}."
    return MethodResult(
        "zivot_andrews",
        clean(statistic),
        clean(p),
        n,
        bool(p < ALPHA),
        value=float(breakpoint) + 1,
        note=note,
    )


def variance_ratio(series: np.ndarray, regression: Regression = "c", q: int = 4) -> MethodResult:
    """Lo-MacKinlay, heteroscedasticity-robust: does the variance of q-period changes equal q
    times that of one-period changes, as for a random walk?"""
    changes = np.diff(series)
    n = len(changes)
    if n < 4 * q or np.ptp(changes) == 0:
        return _not_run(
            "variance_ratio", len(series), f"Needs at least {4 * q + 1} periods that vary."
        )
    mu = changes.mean()
    centred = changes - mu
    var_1 = float(centred @ centred) / (n - 1)
    sums = series[q:] - series[:-q] - q * mu
    m = q * (n - q + 1) * (1 - q / n)
    var_q = float(sums @ sums) / m
    ratio = var_q / var_1
    total = float(centred @ centred) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = n * float((centred[j:] ** 2) @ (centred[:-j] ** 2)) / total  # Lo-MacKinlay δ_j
        theta += (2 * (q - j) / q) ** 2 * delta
    z = (ratio - 1) * math.sqrt(n) / math.sqrt(theta) if theta > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    note = f"q = {q}. Below 1: changes revert (mean-reverting); above 1: they persist."
    return MethodResult(
        "variance_ratio", clean(z), clean(p), len(series), bool(p < ALPHA), value=ratio, note=note
    )


def rur(series: np.ndarray, regression: Regression = "c") -> MethodResult:
    """Range unit-root test (statsmodels: H0 stationary)."""
    n = len(series)
    if n < 25 or np.ptp(series) == 0:
        return _not_run("rur", n, "Needs at least 25 periods that vary.")
    statistic, p, critical = _quiet(lambda: range_unit_root_test(series, result_object=False))
    return MethodResult(
        "rur", clean(statistic), clean(p), n, bool(p < ALPHA), note=_critical(critical)
    )


StationarityMethod = Callable[[np.ndarray, Regression], MethodResult]
STATIONARITY: dict[str, StationarityMethod] = {
    "adf": adf,
    "kpss": kpss_test,
    "phillips_perron": phillips_perron,
    "df_gls": df_gls,
    "zivot_andrews": zivot_andrews_test,
    "variance_ratio": variance_ratio,
    "rur": rur,
}
DEFAULT_STATIONARITY = ("adf", "kpss", "phillips_perron")


def synthesis(adf_result: MethodResult, kpss_result: MethodResult) -> Synthesis | None:
    """ADF and KPSS decision table: they have opposite null hypotheses, so together they tell
    stationary, unit root, trend/difference-stationary or inconclusive apart."""
    if adf_result.significant is None or kpss_result.significant is None:
        return None
    adf_rejects, kpss_rejects = adf_result.significant, kpss_result.significant
    if adf_rejects and not kpss_rejects:
        verdict: Verdict = "stationary"
    elif not adf_rejects and kpss_rejects:
        verdict = "unit_root"
    elif adf_rejects and kpss_rejects:
        verdict = "trend_or_difference"
    else:
        verdict = "inconclusive"
    return Synthesis(verdict, adf_rejects, kpss_rejects)


def judge(series: np.ndarray, regression: Regression = "c") -> Synthesis | None:
    return synthesis(adf(series, regression), kpss_test(series, regression))


def suggest(series: np.ndarray, season: int | None) -> Suggestion | None:
    """The simplest transformation that ADF and KPSS agree is stationary (2 differences at most)."""
    candidates: list[Transform] = [
        "none",
        "diff",
        "seasonal_diff",
        "log_diff",
        "seasonal_diff_diff",
        "diff2",
    ]
    for how in candidates:
        try:
            changed = transform(series, how, season)
        except TransformError:
            continue
        found = judge(changed)
        if found and found.verdict == "stationary":
            return Suggestion(how, found.verdict)
    return None
