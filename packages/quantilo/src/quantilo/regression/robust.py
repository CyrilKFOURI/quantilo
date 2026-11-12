"""Robust and quantile regression (072): Huber and Tukey (statsmodels RLM) side by side with
least squares, quantile regression (statsmodels QuantReg) with a kernel sandwich that stays right
when the spread changes, the coefficients along the distribution (τ = 0.1…0.9) with a test of
"same effect at the bottom and at the top", Student-t errors, and the law and tail of the
residuals. Rules R-REG-11, R-REG-12, R-REG-ROB-01…04, R-REG-QR-01…03.

Everything is numpy / scipy / statsmodels on one thread: Pyodide-compatible."""

import contextlib
import io
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import statsmodels.api as statsmodels_api
from scipy import stats
from statsmodels.miscmodels.tmodel import TLinearModel
from statsmodels.regression.quantile_regression import QuantReg

from quantilo.compare.multiple import MultipleTesting, correct
from quantilo.regression import analyze as an
from quantilo.regression import diagnostics as dg
from quantilo.regression.fit import least_squares

sm: Any = statsmodels_api
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)  # R-REG-QR-01
PROCESS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # R-REG-QR-02
QR_ROWS = 50_000
PROCESS_ROWS = 20_000
LAW_ROWS = 20_000
MAX_CURVES = 9  # small multiples, 3 a row
HUBER_K = 1.345
TUKEY_C = 4.685
MAX_ITER = 1_000
SEED = 42
LEAST_SQUARES = ("ols", "wls", "log_ols", "boxcox")

Estimator = Literal["ols", "huber", "tukey", "quantile", "t"]


class RobustError(ValueError):
    pass


@dataclass
class RobustColumn:
    kind: Estimator
    label: str
    estimates: list[float | None]
    low: list[float | None]
    high: list[float | None]
    p_values: list[float | None]
    note: str  # its standard errors, and the rows it weighs less


@dataclass
class ResidualLaw:
    name: str  # "Normal", "Student t", "Laplace"
    aic: float
    df: float | None = None  # Student t: degrees of freedom
    best: bool = False


@dataclass
class RobustTable:
    available: bool
    reason: str
    terms: list[str] = field(default_factory=list)
    columns: list[RobustColumn] = field(default_factory=list)
    quantile: float = 0.5
    quantiles: list[float] = field(default_factory=lambda: list(QUANTILES))
    moved: list[str] = field(default_factory=list)  # R-REG-ROB-02
    downweighted: int = 0  # Huber weight < 1
    rejected: int = 0  # Tukey weight 0
    rows: int = 0
    quantile_rows: int = 0  # a sample beyond QR_ROWS
    laws: list[ResidualLaw] = field(default_factory=list)
    law_sentence: str = ""
    tail_xi: float | None = None
    tail_sentence: str = ""
    sentence: str = ""
    notes: list[str] = field(default_factory=list)  # weights or groups left aside (R-REG-ROB-03)


@dataclass
class QuantileCurve:
    term: str
    taus: list[float]
    estimates: list[float]
    low: list[float | None]
    high: list[float | None]
    ols: float
    ols_low: float | None
    ols_high: float | None
    difference: float | None  # β(0.9) − β(0.1)
    p_value: float | None  # 0.1 against 0.9, the cross-quantile Wald test
    q_value: float | None = None
    sentence: str = ""


@dataclass
class TErrors:
    available: bool
    reason: str
    converged: bool = False
    df: float | None = None
    scale: float | None = None
    terms: list[str] = field(default_factory=list)
    estimates: list[float] = field(default_factory=list)
    low: list[float | None] = field(default_factory=list)
    high: list[float | None] = field(default_factory=list)
    aic: float | None = None
    ols_aic: float | None = None
    sentence: str = ""


@dataclass
class QuantileProcess:
    available: bool
    reason: str
    taus: list[float] = field(default_factory=lambda: list(PROCESS))
    curves: list[QuantileCurve] = field(default_factory=list)
    tested: int = 0
    changing: int = 0
    rows: int = 0
    sampled: bool = False
    multiple: MultipleTesting | None = None
    sentence: str = ""
    t_errors: TErrors | None = None


# --- Arrays ------------------------------------------------------------------------------------


def _unavailable(fit: an.Fitted) -> str | None:
    if fit.ls is None or fit.model not in LEAST_SQUARES:
        label = fit.family.label if fit.family is not None else fit.model
        return (
            f"Robust and quantile regression are for a number explained by least squares; the "
            f"{label} model has its own robust standard errors (R-REG-ROB-03)."
        )
    if fit.design.x.shape[1] >= len(fit.design.y) - 2:
        return "Too few rows for the number of terms."
    return None


def _notes(fit: an.Fitted) -> list[str]:
    """R-REG-ROB-03: what these estimators leave aside, said."""
    found: list[str] = []
    if fit.ls is not None and fit.ls.weights is not None:
        found.append(
            f"The weights (`{fit.inputs.weights}`) are not used here: each row counts once."
        )
    if fit.se.kind in ("cluster", "cluster2", "dk", "hac"):
        found.append(
            f"Their standard errors assume independent rows, unlike the {an.in_text(fit.se.label)} "
            "ones of the first plane."
        )
    if fit.transform is not None:
        found.append(f"On {_scale(fit)}: a quantile of the log is the log of the quantile.")
    return found


def _scale(fit: an.Fitted) -> str:
    target = fit.inputs.target
    if fit.transform is not None and fit.transform.kind == "log_ols":
        return f"log(`{target}`)"
    if fit.transform is not None and fit.transform.kind == "boxcox":
        return f"the Box-Cox transform of `{target}`"
    return f"`{target}`"


def _sample(n: int, limit: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(SEED).choice(n, limit, replace=False))


def _intervals(
    beta: np.ndarray, se: np.ndarray, df: int | None
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """t(df) intervals and p-values, normal ones when df is None."""
    crit = float(stats.t.ppf(0.975, df)) if df else float(stats.norm.ppf(0.975))
    low: list[float | None] = []
    high: list[float | None] = []
    p: list[float | None] = []
    for b, s in zip(beta, se, strict=True):
        if not math.isfinite(s) or s <= 0:
            low.append(None)
            high.append(None)
            p.append(None)
            continue
        low.append(float(b - crit * s))
        high.append(float(b + crit * s))
        z = abs(b / s)
        p.append(float(2 * (stats.t.sf(z, df) if df else stats.norm.sf(z))))
    return low, high, p


# --- Quantile regression and its kernel sandwich -----------------------------------------------


def hall_sheather(n: int, tau: float, alpha: float = 0.05) -> float:
    z = stats.norm.ppf(1 - alpha / 2)
    x0 = stats.norm.ppf(tau)
    f0 = stats.norm.pdf(x0)
    return float(n ** (-1 / 3) * z ** (2 / 3) * ((1.5 * f0**2) / (2 * x0**2 + 1)) ** (1 / 3))


def density_matrix(x: np.ndarray, e: np.ndarray, tau: float) -> np.ndarray:
    """H = Σ f̂ᵢ xᵢxᵢ′, f̂ᵢ the density of the errors at 0 near row i (Powell's kernel estimate as
    in R quantreg, summary.rq(se = "ker")): Gaussian kernel, Hall–Sheather bandwidth."""
    n = len(e)
    h = hall_sheather(n, tau)
    h = min(h, 0.999 * tau, 0.999 * (1 - tau))
    spread = min(float(np.std(e, ddof=1)), float(np.subtract(*np.percentile(e, [75, 25]))) / 1.34)
    width = (stats.norm.ppf(tau + h) - stats.norm.ppf(tau - h)) * max(spread, 1e-12)
    f = stats.norm.pdf(e / width) / width
    return (x * f[:, None]).T @ x


@dataclass
class QuantileFit:
    tau: float
    beta: np.ndarray
    resid: np.ndarray
    h: np.ndarray  # Σ f̂ᵢ xᵢxᵢ′
    converged: bool


def quantile_fit(x: np.ndarray, y: np.ndarray, tau: float) -> QuantileFit:
    """QuantReg (IRLS) on columns and target scaled to unit spread: its stopping rule is an
    absolute change of the coefficients, which a price in the millions never meets."""
    spread = x.std(axis=0)
    columns = np.where(spread > 0, spread, 1.0)
    scale = float(y.std()) or 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found: Any = QuantReg(y / scale, x / columns).fit(q=tau, max_iter=MAX_ITER)
    beta = np.asarray(found.params, dtype=float) * scale / columns
    resid = y - x @ beta
    converged = int(getattr(found, "iterations", 0)) < MAX_ITER
    return QuantileFit(tau, beta, resid, density_matrix(x, resid, tau), converged)


def cross_covariance(first: QuantileFit, second: QuantileFit, xtx: np.ndarray) -> np.ndarray:
    """Cov(β̂(τ₁), β̂(τ₂)) = (min(τ₁, τ₂) − τ₁τ₂) H₁⁻¹ X′X H₂⁻¹ (Koenker 2005, § 3.5)."""
    t1, t2 = first.tau, second.tau
    weight = min(t1, t2) - t1 * t2
    return weight * np.linalg.pinv(first.h) @ xtx @ np.linalg.pinv(second.h)


def quantile_covariance(fit: QuantileFit, xtx: np.ndarray) -> np.ndarray:
    v = cross_covariance(fit, fit, xtx)
    return (v + v.T) / 2


# --- Advanced › Robust & quantile ---------------------------------------------------------------


def table(fit: an.Fitted, quantile: float = 0.5) -> RobustTable:
    """OLS, Huber, Tukey and one quantile on the same rows and terms (R-REG-ROB-01…03,
    R-REG-QR-01), and the law and tail of the residuals (R-REG-QR-03)."""
    if quantile not in QUANTILES:
        raise RobustError(
            f"Quantile {quantile} is not offered: pick one of "
            f"{', '.join(str(q) for q in QUANTILES)}."
        )
    missing = _unavailable(fit)
    if missing is not None:
        return RobustTable(False, missing, quantile=quantile)
    assert fit.ls is not None
    x, y = fit.design.x, fit.design.y
    n, k = x.shape
    terms = list(fit.design.terms)
    ols_beta = fit.ls.beta
    ols_se = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
    low, high, p = _intervals(ols_beta, ols_se, fit.df)
    columns = [
        RobustColumn(
            "ols",
            "Least squares",
            [float(b) for b in ols_beta],
            low,
            high,
            p,
            f"{fit.se.label} standard errors, as in the first plane",
        )
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        huber = sm.RLM(y, x, M=sm.robust.norms.HuberT(t=HUBER_K)).fit()
        tukey = sm.RLM(y, x, M=sm.robust.norms.TukeyBiweight(c=TUKEY_C)).fit(
            start_params=huber.params
        )
    downweighted = int((np.asarray(huber.weights) < 1 - 1e-9).sum())
    rejected = int((np.asarray(tukey.weights) <= 1e-9).sum())
    fitted: tuple[tuple[Estimator, str, Any, str], ...] = (
        (
            "huber",
            "Huber",
            huber,
            f"k = {HUBER_K}, scale by MAD; {downweighted:,} rows weigh less than 1",
        ),
        (
            "tukey",
            "Tukey biweight",
            tukey,
            f"c = {TUKEY_C}, started from Huber; {rejected:,} rows set aside (weight 0)",
        ),
    )
    for kind, label, found, note in fitted:
        beta = np.asarray(found.params, dtype=float)
        se = np.asarray(found.bse, dtype=float)
        low, high, p = _intervals(beta, se, None)
        columns.append(
            RobustColumn(
                kind,
                label,
                [float(b) for b in beta],
                low,
                high,
                p,
                note + " (Huber's H1 standard errors)",
            )
        )
    rows = _sample(n, QR_ROWS)
    xq, yq = x[rows], y[rows]
    qfit = quantile_fit(xq, yq, quantile)
    qse = np.sqrt(np.clip(np.diag(quantile_covariance(qfit, xq.T @ xq)), 0, None))
    low, high, p = _intervals(qfit.beta, qse, max(len(rows) - k, 1))
    what = "median" if quantile == 0.5 else f"{quantile:g} quantile"
    columns.append(
        RobustColumn(
            "quantile",
            f"Quantile τ = {quantile:g}",
            [float(b) for b in qfit.beta],
            low,
            high,
            p,
            f"the {what} of {_scale(fit)}; kernel sandwich standard errors, right when the spread "
            "changes" + (f"; on a sample of {len(rows):,} rows" if len(rows) < n else ""),
        )
    )
    moved = [
        terms[j]
        for j in range(1, k)
        if ols_se[j] > 0 and abs(float(huber.params[j]) - float(ols_beta[j])) > ols_se[j]
    ]
    laws, law_sentence = residual_law(fit.ls.resid)
    xi = dg.tail_index(fit.ls.resid)
    found = RobustTable(
        True,
        "",
        terms,
        columns,
        quantile,
        list(QUANTILES),
        moved,
        downweighted,
        rejected,
        n,
        len(rows),
        laws,
        law_sentence,
        xi,
        _tail_sentence(xi),
        _sentence(moved, downweighted, n),
        _notes(fit),
    )
    if not qfit.converged:
        found.notes.append("The quantile fit stopped before converging: read it with care.")
    return found


def _sentence(moved: list[str], downweighted: int, n: int) -> str:
    share = downweighted / max(n, 1)
    start = (
        f"Huber weighs {downweighted:,} rows ({share:.0%}) less than least squares does"
        if downweighted
        else "Huber weighs no row less than least squares does"
    )
    if not moved:
        return (
            f"{start}; no coefficient moves by more than one standard error: the rows that "
            "weigh heavily do not change the conclusion (R-REG-ROB-02)."
        )
    names = ", ".join(f"`{m}`" for m in moved[:3]) + (" …" if len(moved) > 3 else "")
    verb = "moves" if len(moved) == 1 else "move"
    return (
        f"{start}; {names} {verb} by more than one standard error: a few rows drive "
        f"{'it' if len(moved) == 1 else 'them'} (R-REG-ROB-02)."
    )


def _tail_sentence(xi: float | None) -> str:
    if xi is None:
        return "Too few rows to read the tail of the residuals."
    if xi > dg.HEAVY_XI:
        moments = 1 / xi if xi > 0 else math.inf
        return (
            f"The residuals have a heavy tail (Hill ξ = {xi:.2f}, about {moments:.0f} moments): a "
            "few large errors can pull least squares; Huber or Student-t errors resist them."
        )
    return f"The tail of the residuals is not heavy (Hill ξ = {xi:.2f})."


def residual_law(resid: np.ndarray) -> tuple[list[ResidualLaw], str]:
    """R-REG-QR-03: normal, Student t and Laplace fitted to the standardized residuals."""
    e = resid[np.isfinite(resid)]
    if len(e) < 20:
        return [], "Too few rows to compare laws."
    e = e[_sample(len(e), LAW_ROWS)]
    s = float(np.std(e)) or 1.0
    z = (e - float(np.mean(e))) / s
    laws: list[ResidualLaw] = []
    normal = float(stats.norm.logpdf(z).sum())
    laws.append(ResidualLaw("Normal", 2 * 2 - 2 * normal))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, loc, scale = stats.t.fit(z)
    student = float(stats.t.logpdf(z, df, loc, scale).sum())
    laws.append(ResidualLaw("Student t", 2 * 3 - 2 * student, float(df)))
    median = float(np.median(z))
    spread = float(np.mean(np.abs(z - median))) or 1.0
    laplace = float(stats.laplace.logpdf(z, median, spread).sum())
    laws.append(ResidualLaw("Laplace", 2 * 2 - 2 * laplace))
    best = min(laws, key=lambda law: law.aic)
    best.best = True
    if best.name == "Normal":
        text = "The residuals follow a normal law best (lowest AIC)."
    elif best.name == "Student t":
        text = (
            f"The residuals follow a Student t law best (ν = {best.df:.1f}, lowest AIC): their "
            "tails are heavier than normal."
        )
    else:
        text = "The residuals follow a Laplace law best (lowest AIC): a sharp peak, heavy tails."
    return laws, text


# --- Advanced › More › Along the distribution ---------------------------------------------------


def process(fit: an.Fitted) -> QuantileProcess:
    """R-REG-QR-02: the coefficients at τ = 0.1…0.9, the test of 0.1 against 0.9 corrected for
    multiple tests, and Student-t errors (R-REG-ROB-04)."""
    missing = _unavailable(fit)
    if missing is not None:
        return QuantileProcess(False, missing)
    assert fit.ls is not None
    x_all, y_all = fit.design.x, fit.design.y
    rows = _sample(len(y_all), PROCESS_ROWS)
    x, y = x_all[rows], y_all[rows]
    n, k = x.shape
    xtx = x.T @ x
    fits = [quantile_fit(x, y, tau) for tau in PROCESS]
    covs = [quantile_covariance(f, xtx) for f in fits]
    bottom, top = fits[0], fits[-1]
    cross = cross_covariance(bottom, top, xtx)
    crit = float(stats.t.ppf(0.975, max(n - k, 1)))
    ols_se = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
    groups = fit.design.groups
    # 070: a column's own effect; the parts of a curve or of an interaction do not read alone
    own = [j for j in range(1, k) if not groups or groups[j] == fit.design.columns[j]]
    order = sorted(
        own,
        key=lambda j: -abs(fit.ls.beta[j] / ols_se[j]) if ols_se[j] > 0 else 0,  # type: ignore[union-attr]
    )
    curves: list[QuantileCurve] = []
    for j in own:
        estimates = [float(f.beta[j]) for f in fits]
        ses = [math.sqrt(max(float(c[j, j]), 0)) for c in covs]
        low = [e - crit * s if s > 0 else None for e, s in zip(estimates, ses, strict=True)]
        high = [e + crit * s if s > 0 else None for e, s in zip(estimates, ses, strict=True)]
        difference = estimates[-1] - estimates[0]
        var = float(covs[0][j, j] + covs[-1][j, j] - 2 * cross[j, j])
        p = float(2 * stats.norm.sf(abs(difference) / math.sqrt(var))) if var > 0 else None
        ols = float(fit.ls.beta[j])
        ols_low, ols_high = ols - crit * ols_se[j], ols + crit * ols_se[j]
        curves.append(
            QuantileCurve(
                fit.design.terms[j],
                list(PROCESS),
                estimates,
                low,
                high,
                ols,
                ols_low,
                ols_high,
                difference,
                p,
            )
        )
    adjusted, summary = correct([c.p_value for c in curves], noun="effects")
    changing = 0
    for curve, q in zip(curves, adjusted, strict=True):
        curve.q_value = q
        curve.sentence = _curve_sentence(curve)
        if q is not None and q < 0.05:
            changing += 1
    shown = {fit.design.terms[j] for j in order[:MAX_CURVES]}
    target = _scale(fit)
    sentence = (
        f"{changing} of {len(curves)} effects change along the distribution of {target} "
        f"(τ = 0.1 against 0.9, {summary.label if summary.method != 'none' else 'no correction'})."
        if curves
        else "No column to follow along the distribution."
    )
    if len(own) < k - 1:
        sentence += " The parts of a curve or of an interaction are fitted, not followed alone."
    if len(rows) < len(y_all):
        sentence += f" Computed on a reproducible sample of {len(rows):,} rows."
    return QuantileProcess(
        True,
        "",
        list(PROCESS),
        [c for c in curves if c.term in shown],
        len(curves),
        changing,
        len(rows),
        len(rows) < len(y_all),
        summary,
        sentence,
        t_errors(fit, rows),
    )


def _curve_sentence(curve: QuantileCurve) -> str:
    q = curve.q_value
    first, last = curve.estimates[0], curve.estimates[-1]
    stat = f"adj. p = {q:.2g}" if q is not None and q >= 0.001 else "adj. p < 0.001"
    if q is None:
        stat = "not testable"
    if q is not None and q < 0.05:
        way = "grows" if last > first else "shrinks"
        return (
            f"The effect of `{curve.term}` {way} along the distribution: {an.fmt(first)} at "
            f"τ = 0.1, {an.fmt(last)} at τ = 0.9 ({stat})."
        )
    return (
        f"The effect of `{curve.term}` stays about the same along the distribution "
        f"({an.fmt(first)} at τ = 0.1, {an.fmt(last)} at τ = 0.9, {stat})."
    )


def t_errors(fit: an.Fitted, rows: np.ndarray) -> TErrors:
    """R-REG-ROB-04: the linear model with Student-t errors, by maximum likelihood."""
    assert fit.ls is not None
    x, y = fit.design.x[rows], fit.design.y[rows]
    k = x.shape[1]
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            found: Any = TLinearModel(y, x).fit(method="bfgs", maxiter=500, disp=0)
    except (ValueError, np.linalg.LinAlgError) as error:
        return TErrors(False, f"Student-t errors could not be fitted: {error}")
    params = np.asarray(found.params, dtype=float)
    se = np.asarray(found.bse, dtype=float)
    converged = bool(found.mle_retvals.get("converged", False))
    beta, df, scale = params[:k], float(params[k]), float(params[k + 1])
    low, high, _ = _intervals(beta, se[:k], None)
    ols = least_squares(x, y)
    ols_aic = float(ols.aic)
    aic = float(found.aic)
    better = aic < ols_aic
    sentence = (
        f"With Student-t errors, ν = {df:.1f}"
        + (": heavy tails" if df < 10 else ": close to normal")
        + f"; AIC {an.fmt(aic)} against {an.fmt(ols_aic)} for normal errors"
        + (" (better)." if better else ".")
    )
    if not converged:
        sentence += " The fit stopped before converging: read it with care."
    return TErrors(
        True,
        "",
        converged,
        df,
        scale,
        list(fit.design.terms),
        [float(b) for b in beta],
        low,
        high,
        aic,
        ols_aic,
        sentence,
    )
