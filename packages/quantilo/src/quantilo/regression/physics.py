"""Physics and metrology (075), all in Advanced › More: the fit weighted by the measurement
uncertainties (χ², reduced χ², absolute or scaled sigma, pulls), errors on x (Deming, York, total
least squares) and non-linear curves (exponential, power, logistic, Michaelis–Menten, Gaussian peak)
by `scipy.optimize.least_squares`. Rules R-REG-PHY-01…05.

`scipy.odr` is deprecated since SciPy 1.17 and removed in 1.19: Deming, York and TLS are closed
forms and an SVD in numpy. One thread, no subprocess: Pyodide-compatible."""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

from quantilo.regression import analyze as an
from quantilo.regression.design import INTERCEPT

LEAST_SQUARES = ("ols", "wls")
BAND = 0.99  # R-REG-PHY-02
PULL_HIGH = 3.0  # R-REG-PHY-03
MAX_CURVE_ROWS = 20_000  # R-REG-PHY-05
CURVE_POINTS = 200
DATA_POINTS = 2_000
SEED = 42
YORK_STEPS = 200


class PhysicsError(ValueError):
    pass


# --- Uncertainties -------------------------------------------------------------------------------


@dataclass
class WeightedTerm:
    term: str
    estimate: float
    absolute_se: float | None  # (X'WX)⁻¹: the uncertainties as they are
    scaled_se: float | None  # × χ²/df
    ci_low: float | None  # with the sigma chosen by R-REG-PHY-02
    ci_high: float | None
    p_value: float | None


@dataclass
class Pull:
    row: int  # file row number
    pull: float


@dataclass
class UncertaintyFit:
    sigma: str
    sigma_reason: str
    rows: int
    dropped: int  # σ ≤ 0 or missing
    chi2: float
    df: int
    reduced: float
    p_value: float
    band_low: float
    band_high: float
    absolute: bool  # R-REG-PHY-02
    absolute_reason: str
    terms: list[WeightedTerm]
    pull_sd: float
    pull_high: int  # |pull| > 3
    largest: list[Pull]
    fitted: list[float]  # drawn: pulls against the fitted values
    pulls: list[float]
    sentence: str
    pull_sentence: str


def _band(df: int) -> tuple[float, float]:
    tail = (1 - BAND) / 2
    return float(stats.chi2.ppf(tail, df) / df), float(stats.chi2.ppf(1 - tail, df) / df)


def weighted_fit(
    x: np.ndarray, y: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Least squares weighted by 1/σ²: coefficients, (X'WX)⁻¹ (absolute sigma), χ², df."""
    w = 1 / sigma**2
    xw = x * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    beta = np.linalg.lstsq(xw, yw, rcond=None)[0]
    absolute = np.linalg.pinv(xw.T @ xw)
    r = (y - x @ beta) / sigma
    return beta, absolute, float(r @ r), x.shape[0] - x.shape[1]


def uncertainty_fit(
    fit: an.Fitted, sigma: np.ndarray, name: str, reason: str
) -> UncertaintyFit | str:
    """Advanced › More › Uncertainties (R-REG-PHY-01…03); `sigma` over the used rows."""
    if fit.model not in LEAST_SQUARES or fit.ls is None:
        return "The weighted fit by uncertainties is read on least squares of the target as it is."
    usable = np.isfinite(sigma) & (sigma > 0)
    x, y = fit.design.x[usable], fit.design.y[usable]
    s = sigma[usable]
    n, k = x.shape
    if n < k + 2:
        return f"{n} rows with an uncertainty above 0: too few for {k} terms."
    beta, absolute, chi2, df = weighted_fit(x, y, s)
    reduced = chi2 / df
    low, high = _band(df)
    p = float(stats.chi2.sf(chi2, df))
    keep_absolute = low <= reduced <= high
    if keep_absolute:
        why = (
            f"R-REG-PHY-02: χ²/df = {reduced:.3g} lies in its 99 % band [{low:.3g} ; {high:.3g}]: "
            "the uncertainties explain the scatter; intervals use them as they are (absolute "
            "sigma)."
        )
    elif reduced > high:
        why = (
            f"R-REG-PHY-02: χ²/df = {reduced:.3g} is above its 99 % band (up to {high:.3g}): the "
            f"scatter is {math.sqrt(reduced):.2g} times the stated uncertainties; they are too "
            "small, or the model misses something. Intervals are scaled by √(χ²/df)."
        )
    else:
        why = (
            f"R-REG-PHY-02: χ²/df = {reduced:.3g} is below its 99 % band (from {low:.3g}): the "
            "points scatter less than their uncertainties say; they are likely overestimated. "
            "Intervals are scaled by √(χ²/df)."
        )
    se_abs = np.sqrt(np.clip(np.diag(absolute), 0, None))
    se_scaled = se_abs * math.sqrt(reduced)
    chosen = se_abs if keep_absolute else se_scaled
    terms = []
    for j, term in enumerate(fit.design.terms):
        se = float(chosen[j])
        if keep_absolute:  # known sigma: normal quantiles
            crit, p_j = (
                1.959963984540054,
                float(2 * stats.norm.sf(abs(beta[j] / se))) if se else None,
            )
        else:
            crit = float(stats.t.ppf(0.975, df))
            p_j = float(2 * stats.t.sf(abs(beta[j] / se), df)) if se else None
        terms.append(
            WeightedTerm(
                term,
                float(beta[j]),
                float(se_abs[j]),
                float(se_scaled[j]),
                float(beta[j] - crit * se) if se else None,
                float(beta[j] + crit * se) if se else None,
                p_j,
            )
        )
    fitted = x @ beta
    pulls = (y - fitted) / s
    rows = fit.inputs.rows[fit.design.used][usable]
    order = np.argsort(-np.abs(pulls))[:5]
    keep = _sample(n, DATA_POINTS)
    sd = float(np.std(pulls, ddof=k))
    high_count = int((np.abs(pulls) > PULL_HIGH).sum())
    sentence = (
        f"Weighted by 1/σ² of `{name}`: χ² = {chi2:,.1f} for {df:,} degrees of freedom "
        f"(χ²/df = {reduced:.3g}, {an.fmt_p(p)})."
    )
    pull_sentence = (
        f"The pulls (y − fit)/σ have a standard deviation of {sd:.2f}"
        + (" (1 when the uncertainties are right)" if abs(sd - 1) > 0.1 else ", as expected")
        + f"; {high_count:,} beyond ±3 (R-REG-PHY-03)."
    )
    return UncertaintyFit(
        name,
        reason,
        n,
        int((~usable).sum()),
        chi2,
        df,
        reduced,
        p,
        low,
        high,
        keep_absolute,
        why,
        terms,
        sd,
        high_count,
        [Pull(int(rows[i]), float(pulls[i])) for i in order],
        [float(v) for v in fitted[keep]],
        [float(v) for v in pulls[keep]],
        sentence,
        pull_sentence,
    )


def _sample(n: int, size: int) -> np.ndarray:
    if n <= size:
        return np.arange(n)
    return np.sort(np.random.default_rng(SEED).choice(n, size, replace=False))


# --- Errors on x ---------------------------------------------------------------------------------


@dataclass
class LineFit:
    method: str  # "ols", "deming", "york", "tls"
    label: str
    slope: float
    intercept: float
    slope_se: float | None
    intercept_se: float | None
    reason: str


@dataclass
class TlsTerm:
    term: str
    estimate: float
    least_squares: float


@dataclass
class ErrorsInX:
    column: str | None  # one column: Deming, York
    ratio: float | None  # λ = σy²/σx² of Deming
    ratio_reason: str
    lines: list[LineFit]
    tls: list[TlsTerm]  # several columns
    attenuation: float | None  # 1 − OLS / Deming
    sentence: str
    notes: list[str] = field(default_factory=list)


def deming(x: np.ndarray, y: np.ndarray, ratio: float) -> tuple[float, float]:
    """Deming's slope for λ = σ²(error on y) / σ²(error on x)."""
    sxx = float(np.var(x, ddof=1))
    syy = float(np.var(y, ddof=1))
    sxy = float(np.cov(x, y, ddof=1)[0, 1])
    if sxy == 0:
        return 0.0, float(np.mean(y))
    d = syy - ratio * sxx
    slope = (d + math.sqrt(d * d + 4 * ratio * sxy * sxy)) / (2 * sxy)
    return slope, float(np.mean(y)) - slope * float(np.mean(x))


def deming_jackknife(x: np.ndarray, y: np.ndarray, ratio: float) -> tuple[float, float]:
    """Jackknife standard errors of Deming's slope and intercept, leave-one-out by sums (O(n))."""
    n = len(x)
    sx, sy = x.sum(), y.sum()
    sxx, syy, sxy = (x * x).sum(), (y * y).sum(), (x * y).sum()
    m = n - 1
    lx, ly = (sx - x) / m, (sy - y) / m
    vxx = ((sxx - x * x) - m * lx * lx) / (m - 1)
    vyy = ((syy - y * y) - m * ly * ly) / (m - 1)
    vxy = ((sxy - x * y) - m * lx * ly) / (m - 1)
    d = vyy - ratio * vxx
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = (d + np.sqrt(d * d + 4 * ratio * vxy * vxy)) / (2 * vxy)
    intercepts = ly - slopes * lx
    factor = (n - 1) / n
    return (
        math.sqrt(factor * float(((slopes - slopes.mean()) ** 2).sum())),
        math.sqrt(factor * float(((intercepts - intercepts.mean()) ** 2).sum())),
    )


def york(
    x: np.ndarray, y: np.ndarray, sx: np.ndarray, sy: np.ndarray
) -> tuple[float, float, float, float]:
    """York's straight line with an error on each x and y (York et al. 2004, uncorrelated):
    slope, intercept and their standard errors."""
    wx, wy = 1 / sx**2, 1 / sy**2
    b = float(np.polyfit(x, y, 1)[0])
    w = xbar = ybar = beta = None
    for _ in range(YORK_STEPS):
        w = wx * wy / (wx + b * b * wy)
        xbar = float((w * x).sum() / w.sum())
        ybar = float((w * y).sum() / w.sum())
        u, v = x - xbar, y - ybar
        beta = w * (u / wy + b * v / wx)
        new = float((w * beta * v).sum() / (w * beta * u).sum())
        if abs(new - b) <= 1e-12 * max(1.0, abs(b)):
            b = new
            break
        b = new
    assert w is not None and xbar is not None and ybar is not None and beta is not None
    a = ybar - b * xbar
    fitted_x = xbar + beta
    mean_x = float((w * fitted_x).sum() / w.sum())
    var_b = 1 / float((w * (fitted_x - mean_x) ** 2).sum())
    var_a = 1 / float(w.sum()) + mean_x**2 * var_b
    return b, a, math.sqrt(var_b), math.sqrt(var_a)


def total_least_squares(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Orthogonal regression by SVD of the centred [X, y]: the coefficients of X (intercept
    excluded), in the units of the columns as they are."""
    data = np.column_stack([x, y])
    centred = data - data.mean(axis=0)
    v = np.linalg.svd(centred, full_matrices=False)[2][-1]
    return -v[:-1] / v[-1]


def _numeric_only(fit: an.Fitted) -> list[int] | None:
    """Every explanatory term a numeric column as it is (R-REG-PHY-04), or None."""
    columns = []
    for j, term in enumerate(fit.design.terms):
        if term == INTERCEPT:
            continue
        column = fit.design.columns[j]
        group = fit.design.groups[j] if fit.design.groups else column
        if group != column or fit.design.kinds.get(column, ("cat", []))[0] != "num":
            return None
        columns.append(j)
    return columns


def errors_in_x(
    fit: an.Fitted,
    sigma_y: np.ndarray | None,
    sigma_x: np.ndarray | None,
) -> ErrorsInX | str:
    """Advanced › More › Errors on x (R-REG-PHY-04); the σ over the used rows."""
    if fit.model not in LEAST_SQUARES or fit.ls is None:
        return "Errors on x are read on least squares of the target as it is."
    columns = _numeric_only(fit)
    if not columns or fit.design.columns[0] != INTERCEPT:
        return (
            "Needs numeric explanatory columns only, with an intercept, no curve or category "
            "(R-REG-PHY-04)."
        )
    y = fit.design.y
    if len(columns) > 1:
        x = fit.design.x[:, columns]
        found = total_least_squares(x, y)
        tls = [
            TlsTerm(fit.design.terms[j], float(found[i]), float(fit.ls.beta[j]))
            for i, j in enumerate(columns)
        ]
        return ErrorsInX(
            None,
            None,
            "",
            [],
            tls,
            None,
            "Total least squares put an error on every column and on the target; with several "
            "columns, Deming's ratio has no single value (R-REG-PHY-04).",
            [
                "TLS depends on the units: a column in mm and one in km do not weigh the same. "
                "Rescale the columns to comparable errors before reading it."
            ],
        )
    j = columns[0]
    x = fit.design.x[:, j]
    column = fit.design.terms[j]
    se_ls = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
    lines = [
        LineFit(
            "ols",
            "Least squares",
            float(fit.ls.beta[j]),
            float(fit.ls.beta[0]),
            float(se_ls[j]),
            float(se_ls[0]),
            "Every error on the target, none on x (the first plane).",
        )
    ]
    both = sigma_y is not None and sigma_x is not None
    if both:
        assert sigma_y is not None and sigma_x is not None
        ok = np.isfinite(sigma_y) & np.isfinite(sigma_x) & (sigma_y > 0) & (sigma_x > 0)
        ratio = float(np.mean(sigma_y[ok] ** 2) / np.mean(sigma_x[ok] ** 2)) if ok.any() else 1.0
        ratio_reason = (
            f"R-REG-PHY-04: λ = mean σy² / mean σx² = {ratio:.3g}, from the uncertainty columns."
        )
    else:
        ratio = 1.0
        ratio_reason = (
            "R-REG-PHY-04: λ = 1 (orthogonal regression): the same error on both, in the units "
            "of the columns. Change the units, the line changes; name the uncertainty columns to "
            "set λ."
        )
    slope, intercept = deming(x, y, ratio)
    se_slope, se_intercept = deming_jackknife(x, y, ratio)
    lines.append(
        LineFit(
            "deming",
            "Deming",
            slope,
            intercept,
            se_slope,
            se_intercept,
            f"Errors on both, in the ratio λ = {ratio:.3g}; standard errors by jackknife.",
        )
    )
    notes: list[str] = []
    if both:
        assert sigma_y is not None and sigma_x is not None
        ok = np.isfinite(sigma_y) & np.isfinite(sigma_x) & (sigma_y > 0) & (sigma_x > 0)
        if ok.sum() >= 3:
            b, a, sb, sa = york(x[ok], y[ok], sigma_x[ok], sigma_y[ok])
            lines.append(
                LineFit(
                    "york",
                    "York",
                    b,
                    a,
                    sb,
                    sa,
                    "Each point's own σx and σy (York et al. 2004, errors uncorrelated).",
                )
            )
            if int((~ok).sum()):
                notes.append(f"York leaves out {int((~ok).sum()):,} rows without both σ above 0.")
    attenuation = 1 - lines[0].slope / slope if slope else None
    if attenuation is not None and abs(attenuation) >= 0.01:
        direction = "understate" if attenuation > 0 else "overstate"
        sentence = (
            f"If `{column}` carries error, least squares {direction} the slope by "
            f"{abs(attenuation):.0%}: {an.fmt(lines[0].slope)} against {an.fmt(slope)} for Deming "
            "(attenuation, R-REG-PHY-04)."
        )
        if not both:
            sentence += (
                " With λ = 1, read it only if the target and the column share their units and "
                "their errors."
            )
    else:
        sentence = (
            f"Least squares and Deming agree on the slope of `{column}` ({an.fmt(slope)}): an "
            "error on x changes little here."
        )
    return ErrorsInX(column, ratio, ratio_reason, lines, [], attenuation, sentence, notes)


# --- Curves --------------------------------------------------------------------------------------

Shape = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class CurveParameter:
    name: str
    estimate: float
    std_error: float | None
    ci_low: float | None
    ci_high: float | None


@dataclass
class CurveFit:
    kind: str
    label: str
    formula: str
    converged: bool
    parameters: list[CurveParameter]
    r2: float | None
    aic: float | None
    delta_aic: float | None
    xs: list[float]
    ys: list[float]
    error: str | None = None


@dataclass
class CurvesTable:
    available: bool
    reason: str
    column: str | None = None
    rows: int = 0
    sampled: bool = False
    weighted: str | None = None
    fits: list[CurveFit] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # shapes the data do not admit, and why
    data_x: list[float] = field(default_factory=list)
    data_y: list[float] = field(default_factory=list)
    best: str | None = None
    sentence: str = ""


def _line(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p[0] + p[1] * x


def _exponential(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p[0] * np.exp(np.clip(p[1] * x, -700, 700)) + p[2]


def _power(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p[0] * np.power(x, p[1])


def _logistic(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p[0] + p[1] / (1 + np.exp(np.clip(-p[2] * (x - p[3]), -700, 700)))


def _michaelis(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p[0] * x / (p[1] + x)


def _gaussian(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p[0] * np.exp(-((x - p[1]) ** 2) / (2 * p[2] ** 2)) + p[3]


SHAPES: dict[str, tuple[str, str, list[str], Shape]] = {
    "line": ("Straight line", "a + b·x", ["a", "b"], _line),
    "exponential": ("Exponential", "a·exp(b·x) + c", ["a", "b", "c"], _exponential),
    "power": ("Power law", "a·x^b", ["a", "b"], _power),
    "logistic": ("Logistic", "c + L / (1 + exp(−k(x − x0)))", ["c", "L", "k", "x0"], _logistic),
    "michaelis": ("Michaelis–Menten", "Vmax·x / (Km + x)", ["Vmax", "Km"], _michaelis),
    "gaussian": ("Gaussian peak", "A·exp(−(x − μ)²/(2s²)) + c", ["A", "μ", "s", "c"], _gaussian),
}


def _starts(kind: str, x: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """Written starting points (R-REG-PHY-05): fits on logs, quantiles, the peak."""
    span = float(np.ptp(x)) or 1.0
    slope = float(np.polyfit(x, y, 1)[0])
    if kind == "line":
        return [np.polyfit(x, y, 1)[::-1]]
    if kind == "exponential":
        found = []
        for offset in (0.0, float(y.min()) - 0.1 * (float(np.ptp(y)) or 1.0)):
            shifted = y - offset
            if (shifted > 0).all():
                b, log_a = np.polyfit(x, np.log(shifted), 1)
                found.append(np.array([math.exp(log_a), b, offset]))
        sign = 1.0 if slope >= 0 else -1.0
        found.append(np.array([sign * (float(np.ptp(y)) or 1.0), sign / span, float(np.mean(y))]))
        # a decay or a saturation towards the level of the largest x (b < 0)
        order = np.argsort(x)
        tenth = max(1, len(x) // 10)
        first, last = float(np.mean(y[order[:tenth]])), float(np.mean(y[order[-tenth:]]))
        rate = -3 / span
        found.append(np.array([(first - last) * math.exp(-rate * float(x.min())), rate, last]))
        return found
    if kind == "power":
        b, log_a = np.polyfit(np.log(x), np.log(y), 1)
        return [np.array([math.exp(log_a), b])]
    if kind == "logistic":
        sign = 1.0 if slope >= 0 else -1.0
        low, high = float(np.quantile(y, 0.02)), float(np.quantile(y, 0.98))
        return [np.array([low, high - low, sign * 4 / span, float(np.median(x))])]
    if kind == "michaelis":
        return [np.array([float(y.max()), float(np.median(x[x > 0])) if (x > 0).any() else 1.0])]
    base = float(np.median(y))
    peak = int(np.argmax(np.abs(y - base)))
    return [np.array([float(y[peak]) - base, float(x[peak]), span / 6, base])]


def _admitted(kind: str, x: np.ndarray, y: np.ndarray) -> str | None:
    if kind == "power" and not ((x > 0).all() and (y > 0).all()):
        return "Power law: needs x > 0 and y > 0."
    if kind == "michaelis" and (x < 0).any():
        return "Michaelis–Menten: needs x ≥ 0."
    return None


def fit_curve(
    kind: str, x: np.ndarray, y: np.ndarray, sigma: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Parameters, their covariance (scaled by the residual variance, as `curve_fit`), the
    (weighted) sum of squares, and whether the fit converged."""
    function = SHAPES[kind][3]
    weight = 1 / sigma if sigma is not None else np.ones_like(y)
    best: tuple[np.ndarray, np.ndarray, float, bool] | None = None
    for start in _starts(kind, x, y):

        def residuals(p: np.ndarray, fn: Shape = function) -> np.ndarray:
            return (fn(x, p) - y) * weight

        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                found = optimize.least_squares(residuals, start, method="lm", max_nfev=2_000)
            except (ValueError, np.linalg.LinAlgError):
                continue
        if not np.isfinite(found.fun).all():
            continue
        ssr = float(found.fun @ found.fun)
        if best is not None and ssr >= best[2]:
            continue
        jac = found.jac
        n, p = jac.shape
        _, s, vt = np.linalg.svd(jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(jac.shape) * (s[0] if len(s) else 0)
        s = s[s > threshold]
        vt = vt[: len(s)]
        covariance = (vt.T / s**2) @ vt
        covariance = covariance * (ssr / (n - p)) if n > p else covariance * np.inf
        best = (found.x, covariance, ssr, bool(found.success))
    if best is None:
        raise PhysicsError("No starting point leads to a finite fit.")
    return best


def curves(fit: an.Fitted, sigma: np.ndarray | None, sigma_name: str | None) -> CurvesTable:
    """Advanced › More › Curve (R-REG-PHY-05): one numeric column, the shapes ranked by AIC."""
    if fit.model not in LEAST_SQUARES or fit.ls is None:
        return CurvesTable(False, "Curves are fitted to the target as it is.")
    columns = _numeric_only(fit)
    if columns is None or len(columns) != 1:
        return CurvesTable(
            False, "A curve takes one numeric explanatory column, no category or term."
        )
    j = columns[0]
    x, y = fit.design.x[:, j], fit.design.y
    s = sigma
    if s is not None:
        ok = np.isfinite(s) & (s > 0)
        x, y, s = x[ok], y[ok], s[ok]
    n = len(y)
    picked = _sample(n, MAX_CURVE_ROWS)
    x, y = x[picked], y[picked]
    if s is not None:
        s = s[picked]
    table = CurvesTable(True, "", fit.design.terms[j], len(y), n > MAX_CURVE_ROWS, sigma_name)
    grid = np.linspace(float(x.min()), float(x.max()), CURVE_POINTS)
    tss = float(((y - np.average(y, weights=None if s is None else 1 / s**2)) ** 2).sum())
    for kind, (label, formula, names, function) in SHAPES.items():
        refused = _admitted(kind, x, y)
        if refused is not None:
            table.skipped.append(refused)
            continue
        try:
            params, covariance, ssr, converged = fit_curve(kind, x, y, s)
        except PhysicsError as error:
            table.fits.append(CurveFit(kind, label, formula, False, [], None, None, None, [], [],
                                       str(error)))  # fmt: skip
            continue
        k = len(params)
        df = max(len(y) - k, 1)
        crit = float(stats.t.ppf(0.975, df))
        parameters = []
        for i, name in enumerate(names):
            se = math.sqrt(covariance[i, i]) if math.isfinite(covariance[i, i]) else None
            parameters.append(
                CurveParameter(
                    name,
                    float(params[i]),
                    se,
                    float(params[i] - crit * se) if se is not None else None,
                    float(params[i] + crit * se) if se is not None else None,
                )
            )
        plain = float(((function(x, params) - y) ** 2).sum())
        r2 = 1 - plain / tss if tss > 0 else None
        aic = len(y) * math.log(max(ssr, 1e-300) / len(y)) + 2 * k
        with np.errstate(all="ignore"):
            curve = function(grid, params)
        table.fits.append(
            CurveFit(
                kind,
                label,
                formula,
                converged,
                parameters,
                r2,
                aic,
                None,
                [float(v) for v in grid],
                [float(v) for v in curve],
            )
        )
    fitted = [f for f in table.fits if f.aic is not None and f.converged]
    if fitted:
        best = min(fitted, key=lambda f: f.aic or 0)
        for f in table.fits:
            if f.aic is not None and best.aic is not None:
                f.delta_aic = f.aic - best.aic
        table.fits.sort(key=lambda f: (f.aic is None, f.aic or 0))
        table.best = best.kind
        table.sentence = _curve_sentence(table, best)
    keep = _sample(len(y), DATA_POINTS)
    table.data_x, table.data_y = [float(v) for v in x[keep]], [float(v) for v in y[keep]]
    return table


def _curve_sentence(table: CurvesTable, best: CurveFit) -> str:
    line = next((f for f in table.fits if f.kind == "line"), None)
    parts = ", ".join(
        f"{p.name} = {an.fmt(p.estimate)} [{an.fmt(p.ci_low)} ; {an.fmt(p.ci_high)}]"
        for p in best.parameters
    )
    if best.kind == "line":
        return (
            f"The straight line fits `{table.column}` best (lowest AIC): no curve does better "
            "(R-REG-PHY-05)."
        )
    gain = (
        f" ({line.aic - (best.aic or 0):.0f} AIC points below the line)"
        if line and line.aic
        else ""
    )
    return f"{best.label} fits best{gain}: {best.formula} with {parts} (R-REG-PHY-05)."


# --- The fold of More ----------------------------------------------------------------------------


@dataclass
class MeasurementTable:
    available: bool
    reason: str
    uncertainty: UncertaintyFit | None = None
    uncertainty_reason: str = ""  # why there is none
    errors: ErrorsInX | None = None
    errors_reason: str = ""
    sigma_x: str | None = None
    sigma_x_reason: str = ""


def measurement(
    fit: an.Fitted,
    sigma: np.ndarray | None,
    sigma_name: str | None,
    sigma_reason: str,
    sigma_x: np.ndarray | None,
    sigma_x_name: str | None,
    sigma_x_reason: str,
) -> MeasurementTable:
    """Advanced › More › Uncertainties and errors on x (075): the σ over the used rows."""
    if fit.model not in LEAST_SQUARES or fit.ls is None:
        return MeasurementTable(False, "Read on least squares of the target as it is.")
    table = MeasurementTable(True, "", sigma_x=sigma_x_name, sigma_x_reason=sigma_x_reason)
    if sigma is not None and sigma_name is not None:
        found = uncertainty_fit(fit, sigma, sigma_name, sigma_reason)
        if isinstance(found, str):
            table.uncertainty_reason = found
        else:
            table.uncertainty = found
    else:
        table.uncertainty_reason = (
            "No column of role uncertainty names the target: pick the column of its σ to fit by "
            "1/σ² (R-REG-PHY-01)."
        )
    errors = errors_in_x(fit, sigma, sigma_x)
    if isinstance(errors, str):
        table.errors_reason = errors
    else:
        table.errors = errors
    return table
