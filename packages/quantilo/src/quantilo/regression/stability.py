"""Stability over time (073): are the coefficients the same at the start and at the end of a
dated series? Recursive residuals with CUSUM and CUSUM of squares (statsmodels RecursiveLS), a
Chow test at the strongest break of the residuals (found by `timeseries/breaks.py`), rolling
coefficients over a window chosen by rule, and the reading of a factor model (alpha a period and
a year, betas, R²) when the target looks like a return. Rules R-REG-15, R-REG-STAB-01…05.

Everything is numpy / scipy / statsmodels on one thread: Pyodide-compatible."""

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.diagnostic import breaks_cusumolsresid

from quantilo.regression import analyze as an
from quantilo.regression.analyze import RegressionCheck
from quantilo.regression.design import INTERCEPT

LEAST_SQUARES = ("ols", "wls", "log_ols", "boxcox")
MIN_ROWS = 30  # R-REG-STAB-01
ALPHA = 0.05
STRONG = 0.01  # R-REG-15
SEARCHED = 0.001  # R-REG-STAB-03: a Chow date found in the data must pass a stricter bar
MAX_POINTS = 400  # drawn per curve
MAX_CURVES = 6  # small multiples, 3 a row
TRIM = 0.15  # R-REG-STAB-03: a break this close to an end is not tested
RETURN_MEDIAN = 0.05  # R-REG-STAB-04: a return is a small number around 0…
RETURN_SPREAD = 0.5  # …that rarely moves by more than half
# R-REG-STAB-04: days between two rows → periods a year
FREQUENCIES = (
    (1.6, 252.0, "a trading day"),  # weekdays only: a median gap of one day, weekends missing
    (1.6, 365.0, "a day"),
    (8.0, 52.0, "a week"),
    (35.0, 12.0, "a month"),
    (95.0, 4.0, "a quarter"),
    (370.0, 1.0, "a year"),
)


@dataclass
class CusumPath:
    dates: list[str]
    values: list[float]
    lower: list[float]
    upper: list[float]
    crossed: bool
    first: str | None  # the first date out of the band


@dataclass
class ChowTest:
    at: str  # the first date of the second part
    index: int
    f: float | None
    p_value: float | None
    before: list[float]
    after: list[float]
    searched: bool  # the date was found in the data (R-REG-STAB-03)
    sentence: str


@dataclass
class RollingCurve:
    term: str
    dates: list[str]
    estimates: list[float | None]
    low: list[float | None]
    high: list[float | None]
    full: float  # the whole sample
    smallest: float | None
    largest: float | None
    sentence: str = ""


@dataclass
class FactorReading:
    periods: float  # a year
    period: str  # "a month"
    alpha: float
    alpha_low: float | None
    alpha_high: float | None
    alpha_year: float
    alpha_year_low: float | None
    alpha_year_high: float | None
    alpha_p: float | None
    betas: list[tuple[str, float]]
    r2: float
    sentence: str


@dataclass
class StabilityTable:
    available: bool
    reason: str
    date: str | None = None
    rows: int = 0
    cusum: CusumPath | None = None
    cusum_squares: CusumPath | None = None
    cusum_p: float | None = None  # Ploberger–Krämer CUSUM of the OLS residuals
    chow: ChowTest | None = None
    window: int = 0
    window_reason: str = ""
    rolling: list[RollingCurve] = field(default_factory=list)
    rolled: int = 0  # curves before the cut to MAX_CURVES
    factor: FactorReading | None = None
    stable: bool = True
    sentence: str = ""
    notes: list[str] = field(default_factory=list)


def _unavailable(fit: an.Fitted) -> str | None:
    """R-REG-STAB-01: a dated series fitted by least squares, with enough rows."""
    if fit.inputs.shape.kind != "series" or fit.inputs.dates is None:
        return (
            "Needs one series in time: one row per date, so that the start and the end of the "
            "rows can be compared (R-REG-STAB-01). Panel data are compared over periods in Panel."
        )
    if fit.ls is None or fit.model not in LEAST_SQUARES:
        label = fit.family.label if fit.family is not None else fit.model
        return f"Stability over time is read on least squares; the {label} model is not (yet)."
    n, k = fit.design.x.shape
    if n < max(MIN_ROWS, 3 * k + 6):
        return f"{n} rows for {k} terms: too few to compare the start and the end of the series."
    return None


def _arrays(fit: an.Fitted) -> tuple[np.ndarray, np.ndarray]:
    """The (weighted) design and target, rows in time order."""
    assert fit.ls is not None
    a = fit.ls.arrays
    return a.x, a.x @ fit.ls.beta + a.e


def _thin(n: int) -> np.ndarray:
    step = max(1, math.ceil(n / MAX_POINTS))
    picked = np.arange(0, n, step)
    return picked if picked[-1] == n - 1 else np.append(picked, n - 1)


def recursive(x: np.ndarray, y: np.ndarray) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RecursiveLS(y, x).fit()


def _path(dates: np.ndarray, values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> CusumPath:
    out = (values < lower) | (values > upper)
    first = str(dates[int(np.argmax(out))]) if out.any() else None
    keep = _thin(len(values))
    return CusumPath(
        [str(d) for d in dates[keep]],
        [float(v) for v in values[keep]],
        [float(v) for v in lower[keep]],
        [float(v) for v in upper[keep]],
        bool(out.any()),
        first,
    )


def chow(x: np.ndarray, y: np.ndarray, at: int) -> tuple[float | None, float | None]:
    """F = ((SSR − SSR₁ − SSR₂)/k) / ((SSR₁ + SSR₂)/(n − 2k)), classic errors (Chow 1960)."""
    n, k = x.shape

    def ssr(xs: np.ndarray, ys: np.ndarray) -> float:
        beta = np.linalg.lstsq(xs, ys, rcond=None)[0]
        e = ys - xs @ beta
        return float(e @ e)

    whole = ssr(x, y)
    parts = ssr(x[:at], y[:at]) + ssr(x[at:], y[at:])
    if parts <= 0 or n - 2 * k <= 0:
        return None, None
    f = ((whole - parts) / k) / (parts / (n - 2 * k))
    return float(f), float(stats.f.sf(f, k, n - 2 * k))


def _break(x: np.ndarray, y: np.ndarray) -> int:
    """R-REG-STAB-03: the date where the Chow F is largest (Quandt), kept 15 % away from both
    ends, among at most 400 candidates; running sums of X'X, X'y and y'y make a candidate cost
    k³, not n k²."""
    n, k = x.shape
    edge = max(k + 2, math.ceil(TRIM * n))
    if n - 2 * edge < 0:
        return n // 2
    candidates = np.unique(
        np.linspace(edge, n - edge, min(MAX_POINTS, n - 2 * edge + 1)).astype(int)
    )
    total_xx, total_xy, total_yy = x.T @ x, x.T @ y, float(y @ y)

    def ssr(a: np.ndarray, b: np.ndarray, c: float) -> float:
        beta = np.linalg.lstsq(a, b, rcond=None)[0]
        return max(c - float(beta @ b), 0.0)

    whole = ssr(total_xx, total_xy, total_yy)
    xx, xy, yy, start = np.zeros((k, k)), np.zeros(k), 0.0, 0
    best, at = -1.0, n // 2
    for t in candidates:
        block = x[start:t]
        xx, xy, yy = (
            xx + block.T @ block,
            xy + block.T @ y[start:t],
            yy + float(y[start:t] @ y[start:t]),
        )
        start = t
        parts = ssr(xx, xy, yy) + ssr(total_xx - xx, total_xy - xy, total_yy - yy)
        if parts > 0 and (whole - parts) / parts > best:
            best, at = (whole - parts) / parts, int(t)
    return at


def window_of(n: int, k: int) -> tuple[int, str]:
    """R-REG-STAB-02: a fifth of the rows, at least 10 rows a term, at most half the rows."""
    window = min(max(n // 5, 10 * k, 24), n // 2)
    return window, (
        f"R-REG-STAB-02: windows of {window:,} rows (a fifth of the {n:,} rows, at least 10 a "
        "term, at most half): long enough to estimate, short enough to move."
    )


def _rolling(
    fit: an.Fitted, x: np.ndarray, y: np.ndarray, dates: np.ndarray, window: int
) -> tuple[list[RollingCurve], int]:
    assert fit.ls is not None
    k = x.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found: Any = RollingOLS(y, x, window=window).fit(cov_type="nonrobust")
    params = np.asarray(found.params, dtype=float)
    bse = np.asarray(found.bse, dtype=float)
    se = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
    groups = fit.design.groups
    own = [j for j in range(1, k) if not groups or groups[j] == fit.design.columns[j]]
    own.sort(key=lambda j: -abs(fit.ls.beta[j] / se[j]) if se[j] > 0 else 0)  # type: ignore[union-attr]
    keep = _thin(len(y))
    keep = keep[keep >= window - 1]
    curves: list[RollingCurve] = []
    for j in own[:MAX_CURVES]:
        values = params[:, j]
        finite = values[np.isfinite(values)]
        curve = RollingCurve(
            fit.design.terms[j],
            [str(d) for d in dates[keep]],
            [_clean(float(values[i])) for i in keep],
            [_clean(float(values[i] - 1.96 * bse[i, j])) for i in keep],
            [_clean(float(values[i] + 1.96 * bse[i, j])) for i in keep],
            float(fit.ls.beta[j]),
            float(finite.min()) if len(finite) else None,
            float(finite.max()) if len(finite) else None,
        )
        curve.sentence = _rolling_sentence(curve, window)
        curves.append(curve)
    return curves, len(own)


def _clean(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _rolling_sentence(curve: RollingCurve, window: int) -> str:
    low, high = curve.smallest, curve.largest
    if low is None or high is None:
        return f"`{curve.term}`: too few rows a window."
    flips = low < 0 < high
    text = (
        f"Over windows of {window:,} rows, the effect of `{curve.term}` goes from {an.fmt(low)} "
        f"to {an.fmt(high)} ({an.fmt(curve.full)} on the whole series)"
    )
    return text + (": it changes sign." if flips else ".")


def frequency(dates: np.ndarray) -> tuple[float, str] | None:
    """R-REG-STAB-04: periods a year from the median gap between two dates."""
    parsed = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    if len(parsed) < 3:
        return None
    gaps = parsed.diff().dropna().dt.total_seconds().to_numpy() / 86_400
    gaps = gaps[gaps > 0]
    if not len(gaps):
        return None
    median = float(np.median(gaps))
    if median <= FREQUENCIES[0][0]:
        weekend = float(np.mean(parsed.dt.day_of_week >= 5))
        return (252.0, "a trading day") if weekend <= 0.02 else (365.0, "a day")
    for limit, periods, label in FREQUENCIES[2:]:
        if median <= limit:
            return periods, label
    return None


def _looks_like_returns(y: np.ndarray) -> bool:
    finite = y[np.isfinite(y)]
    return bool(
        len(finite)
        and abs(float(np.median(finite))) < RETURN_MEDIAN
        and float(np.std(finite)) < RETURN_SPREAD
    )


def factor_reading(fit: an.Fitted, dates: np.ndarray) -> FactorReading | None:
    """R-REG-STAB-04: a target that looks like a return (a small number around 0), numbers on
    the right, an intercept and a known frequency → alpha a period and a year, betas, R²."""
    assert fit.ls is not None
    if fit.model != "ols" or fit.design.columns[0] != INTERCEPT or fit.transform is not None:
        return None
    kinds = fit.design.kinds
    if any(kind != "num" for kind, _ in kinds.values()):
        return None
    if not _looks_like_returns(fit.design.y):
        return None
    found = frequency(dates)
    if found is None:
        return None
    periods, label = found
    alpha = float(fit.ls.beta[0])
    se = math.sqrt(max(float(fit.covariance[0, 0]), 0))
    crit = float(stats.t.ppf(0.975, fit.df))
    low, high = (alpha - crit * se, alpha + crit * se) if se > 0 else (None, None)
    p = float(2 * stats.t.sf(abs(alpha / se), fit.df)) if se > 0 else None
    betas = [(fit.design.terms[j], float(fit.ls.beta[j])) for j in range(1, len(fit.design.terms))]
    year = alpha * periods
    sentence = (
        f"Read as a factor model: alpha {alpha * 100:.3g} % {label}, {year * 100:.3g} % a year "
        f"[{_pct(low, periods)} ; {_pct(high, periods)}], {an.fmt_p(p)}; R² {fit.ls.r2:.2f} of "
        "the variance follows the factors. Alpha is the mean return the factors leave unexplained "
        "(R-REG-STAB-04)."
    )
    return FactorReading(
        periods,
        label,
        alpha,
        low,
        high,
        year,
        low * periods if low is not None else None,
        high * periods if high is not None else None,
        p,
        betas,
        float(fit.ls.r2),
        sentence,
    )


def _pct(value: float | None, periods: float) -> str:
    return "—" if value is None else f"{value * periods * 100:.3g} %"


def section(fit: an.Fitted) -> StabilityTable:
    """Advanced › Stability over time (073)."""
    missing = _unavailable(fit)
    if missing is not None:
        return StabilityTable(False, missing)
    assert fit.ls is not None and fit.inputs.dates is not None
    dates = np.asarray(fit.inputs.dates)[fit.design.used]
    x, y = _arrays(fit)
    n, k = x.shape
    found = recursive(x, y)
    cusum = np.asarray(found.cusum, dtype=float)
    squares = np.asarray(found.cusum_squares, dtype=float)
    tail = dates[n - len(cusum) :]
    points = np.arange(n - len(cusum) + 1, n + 1)  # Brown, Durbin & Evans: t = k+1…n
    low, high = (
        np.asarray(b, dtype=float) for b in found._cusum_significance_bounds(ALPHA, points=points)
    )
    low2, high2 = (
        np.asarray(b, dtype=float)
        for b in found._cusum_squares_significance_bounds(ALPHA, points=points)
    )
    path = _path(tail, cusum, low, high)
    path2 = _path(dates[n - len(squares) :], squares, low2, high2)
    _, cusum_p, _ = breaks_cusumolsresid(fit.ls.resid, ddof=k)
    at, searched = _break(x, y), True
    f, p = chow(x, y, at)
    before = np.linalg.lstsq(x[:at], y[:at], rcond=None)[0]
    after = np.linalg.lstsq(x[at:], y[at:], rcond=None)[0]
    window, why = window_of(n, k)
    rolling, rolled = _rolling(fit, x, y, dates, window) if window >= k + 5 else ([], 0)
    test = ChowTest(
        str(dates[at]),
        at,
        f,
        p,
        [float(b) for b in before],
        [float(b) for b in after],
        searched,
        _chow_sentence(str(dates[at]), p, searched),
    )
    unstable = path.crossed or (p is not None and p < SEARCHED)  # CUSUM² is the spread
    table = StabilityTable(
        True,
        "",
        fit.inputs.shape.date,
        n,
        path,
        path2,
        float(cusum_p),
        test,
        window,
        why,
        rolling,
        rolled,
        factor_reading(fit, dates),
        not unstable,
        _sentence(path, path2, test),
    )
    if fit.ls.weights is not None:
        table.notes.append(
            f"The rows are weighted by `{fit.inputs.weights}`, as in the first plane."
        )
    if fit.inputs.sampled:
        table.notes.append(
            "The rows are a reproducible sample of the series: gaps between dates are wider."
        )
    return table


def _chow_sentence(at: str, p: float | None, searched: bool) -> str:
    if p is None:
        return "The Chow test could not be computed here."
    verdict = "differ" if p < SEARCHED else "do not clearly differ"
    text = f"Before and after {at}, the coefficients {verdict} (Chow F test, {an.fmt_p(p)})."
    if searched:
        text += (
            " The date is the one where the coefficients differ most, searched in the data: the "
            "p-value is optimistic (R-REG-STAB-03)."
        )
    return text


def _sentence(path: CusumPath, squares: CusumPath, test: ChowTest) -> str:
    """The coefficients first (CUSUM, Chow), then the spread of the errors (CUSUM of squares)."""
    if path.crossed:
        text = (
            f"The CUSUM of the recursive residuals leaves its 5 % band at {path.first}: the "
            "coefficients drift over time. Read the rolling coefficients below (R-REG-15)."
        )
    elif test.p_value is not None and test.p_value < SEARCHED:
        text = (
            f"The coefficients before and after {test.at} differ (Chow {an.fmt_p(test.p_value)}); "
            "read the rolling coefficients below (R-REG-15)."
        )
    else:
        text = "Nothing says the coefficients change over the series (CUSUM within its 5 % band)."
    if squares.crossed:
        text += (
            f" The CUSUM of squares leaves its band at {squares.first}: the spread of the errors "
            "changes over time, which HAC standard errors allow for."
        )
    return text


def proposal(fit: an.Fitted) -> RegressionCheck | None:
    """R-REG-15 in the first plane, at the cost of one more pass over the rows: the CUSUM of the
    OLS residuals (Ploberger–Krämer), and the Chow test at the date where it is largest. That
    date is searched, so it must reach p < 0.001 rather than 0.01 (R-REG-STAB-03)."""
    if _unavailable(fit) is not None or fit.ls is None or fit.inputs.dates is None:
        return None
    k = fit.design.x.shape[1]
    _, p, _ = breaks_cusumolsresid(fit.ls.resid, ddof=k)
    p = float(p)
    x, y = _arrays(fit)
    at = _break(x, y)
    _, chow_p = chow(x, y, at)
    date = fit.inputs.shape.date
    if p < STRONG or (chow_p is not None and chow_p < SEARCHED):
        when = str(np.asarray(fit.inputs.dates)[fit.design.used][at])
        why = (
            f"the residuals drift along `{date}` (CUSUM {an.fmt_p(p)})"
            if p < STRONG
            else f"the coefficients before and after {when} differ (Chow {an.fmt_p(chow_p)})"
        )
        return RegressionCheck(
            "stability",
            "warn",
            "Stability over time",
            f"{why[0].upper()}{why[1:]}: one set of coefficients may not hold over the whole "
            "series. Open Stability over time for the rolling coefficients.",
            "R-REG-15",
            "stability",
        )
    return RegressionCheck(
        "stability",
        "ok",
        "Stability over time",
        f"The coefficients look the same along `{date}` (CUSUM {an.fmt_p(p)}, Chow at the "
        f"strongest date {an.fmt_p(chow_p)}).",
        "R-REG-15",
        "stability",
    )
