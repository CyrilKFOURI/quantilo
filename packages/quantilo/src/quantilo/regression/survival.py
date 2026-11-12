"""Health (076), in Advanced › More: the Cox model of a time to event (statsmodels PHReg, Breslow
ties) with the proportional-hazards test of the Schoenfeld residuals, and the Tobit model of a
target held at a bound (maximum likelihood with scipy). Rules R-REG-06, R-REG-SURV-01, 02,
R-REG-TOB-01.

The duration and event are read by the rules of 062 (`compare/durations.py`, R-SRV-01 and 03).
One thread, no subprocess: Pyodide-compatible."""

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy import optimize, special, stats
from statsmodels.duration.hazard_regression import PHReg

from quantilo.regression import analyze as an
from quantilo.regression.analyze import RegressionCheck
from quantilo.regression.design import INTERCEPT
from quantilo.regression.fit import least_squares

MIN_EVENTS = 10  # R-REG-SURV-01
MAX_COX_ROWS = 50_000
MAX_TOBIT_ROWS = 20_000  # R-REG-TOB-01
PH_P = 0.01  # R-REG-SURV-02
CENSORED_SHARE = 0.10  # R-REG-06
SEED = 42
LEAST_SQUARES = ("ols", "wls")


def _sample(n: int, size: int) -> np.ndarray:
    if n <= size:
        return np.arange(n)
    return np.sort(np.random.default_rng(SEED).choice(n, size, replace=False))


# --- R-REG-06 in the first plane ------------------------------------------------------------------


def time_to_event(target: str, event: str, reason: str) -> RegressionCheck:
    return RegressionCheck(
        "time_to_event",
        "warn",
        "Time to event",
        f"`{target}` reads as a duration and `{event}` as the event that ends it: least squares on "
        "a duration still running are biased. Open More for the Cox model.",
        "R-REG-06",
        "more",
    )


def bounds(y: np.ndarray) -> tuple[float | None, float | None]:
    """R-REG-06, R-REG-TOB-01: the minimum and the maximum when each holds ≥ 10 % of the rows."""
    finite = y[np.isfinite(y)]
    if len(finite) < 20 or np.ptp(finite) == 0:
        return None, None
    low, high = float(finite.min()), float(finite.max())
    left = low if float(np.mean(finite == low)) >= CENSORED_SHARE else None
    right = high if float(np.mean(finite == high)) >= CENSORED_SHARE else None
    return left, right


def censored_check(fit: an.Fitted) -> RegressionCheck | None:
    if fit.model not in LEAST_SQUARES or fit.ls is None:
        return None
    left, right = bounds(fit.design.y)
    if left is None and right is None:
        return None
    y = fit.design.y
    parts = []
    if left is not None:
        parts.append(
            f"{float(np.mean(y == left)):.0%} of the rows sit at the minimum {an.fmt(left)}"
        )
    if right is not None:
        parts.append(
            f"{float(np.mean(y == right)):.0%} at the maximum {an.fmt(right)}"
            if parts
            else f"{float(np.mean(y == right)):.0%} of the rows sit at the maximum {an.fmt(right)}"
        )
    return RegressionCheck(
        "censored",
        "info",
        "Censored values",
        f"{' and '.join(parts)}: a value held at a bound is not like the others, least squares "
        "flatten the slopes there. Open More for the Tobit model.",
        "R-REG-06",
        "more",
    )


# --- Cox ------------------------------------------------------------------------------------------


@dataclass
class HazardTerm:
    term: str
    coefficient: float
    std_error: float | None
    hazard_ratio: float
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    ph_p: float | None  # Schoenfeld (R-REG-SURV-02)
    proportional: bool | None
    sentence: str = ""


@dataclass
class CoxTable:
    available: bool
    reason: str
    duration: str | None = None
    event: str | None = None
    event_reason: str = ""
    rows: int = 0
    events: int = 0
    censored: int = 0
    sampled: bool = False
    terms: list[HazardTerm] = field(default_factory=list)
    lr: float | None = None
    lr_df: int = 0
    lr_p: float | None = None
    sentence: str = ""
    ph_sentence: str = ""
    notes: list[str] = field(default_factory=list)


def schoenfeld_test(
    residuals: np.ndarray, times: np.ndarray, events: np.ndarray
) -> list[float | None]:
    """R-REG-SURV-02: per column, the correlation of the Schoenfeld residuals with the rank of the
    event time, t-tested (Grambsch & Therneau 1994, unscaled)."""
    found: list[float | None] = []
    happened = events > 0
    ranks = stats.rankdata(times[happened])
    d = int(happened.sum())
    for j in range(residuals.shape[1]):
        r_j = residuals[happened, j]
        ok = np.isfinite(r_j)
        if ok.sum() < 4 or np.ptp(r_j[ok]) == 0:
            found.append(None)
            continue
        rho = float(np.corrcoef(r_j[ok], ranks[ok])[0, 1])
        if not math.isfinite(rho) or abs(rho) >= 1:
            found.append(None)
            continue
        dof = int(ok.sum()) - 2
        t = rho * math.sqrt(dof) / math.sqrt(1 - rho * rho)
        found.append(float(2 * stats.t.sf(abs(t), dof)) if d > 2 else None)
    return found


def cox(fit: an.Fitted, event: np.ndarray, name: str, reason: str) -> CoxTable:
    """Advanced › More › Survival (076, R-REG-SURV-01, 02): `event` is 1/0 over the used rows."""
    durations = fit.original if fit.original is not None else fit.design.y
    columns = [j for j, term in enumerate(fit.design.terms) if term != INTERCEPT]
    if not columns:
        return CoxTable(False, "Add an explanatory column: the Cox model compares hazards.")
    ok = np.isfinite(event) & np.isfinite(durations) & (durations >= 0)
    if (durations < 0).any():
        reason_neg = f" {int((durations < 0).sum()):,} rows with a negative duration are left out."
    else:
        reason_neg = ""
    picked = np.flatnonzero(ok)
    sampled = len(picked) > MAX_COX_ROWS
    picked = picked[_sample(len(picked), MAX_COX_ROWS)]
    t, e, x = durations[picked], event[picked], fit.design.x[np.ix_(picked, columns)]
    events = int(e.sum())
    if events < MIN_EVENTS:
        return CoxTable(
            False,
            f"{events} events among {len(t):,} rows: the Cox model needs at least {MIN_EVENTS} "
            "(R-REG-SURV-01).",
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PHReg(t, x, status=e, ties="breslow")
        found: Any = model.fit()
        null: float = cast(Any, model).loglike(np.zeros(x.shape[1]))
        residuals = np.asarray(found.schoenfeld_residuals, dtype=float)
    params = np.asarray(found.params, dtype=float)
    bse = np.asarray(found.bse, dtype=float)
    lr = 2 * (float(found.llf) - null)
    ph = schoenfeld_test(residuals, t, e)
    terms: list[HazardTerm] = []
    for i, j in enumerate(columns):
        b, se, shape = float(params[i]), float(bse[i]), ph[i]
        good = math.isfinite(se) and se > 0
        p = float(2 * stats.norm.sf(abs(b / se))) if good else None
        term = HazardTerm(
            fit.design.terms[j],
            b,
            se if good else None,
            math.exp(b),
            math.exp(b - 1.959963984540054 * se) if good else None,
            math.exp(b + 1.959963984540054 * se) if good else None,
            p,
            shape,
            None if shape is None else shape >= PH_P,
        )
        term.sentence = _hazard_sentence(term, name)
        terms.append(term)
    table = CoxTable(
        True,
        "",
        fit.inputs.target,
        name,
        reason,
        len(t),
        events,
        len(t) - events,
        sampled,
        terms,
        lr,
        len(columns),
        float(stats.chi2.sf(lr, len(columns))),
    )
    clear = [term for term in terms if term.p_value is not None and term.p_value < 0.05]
    head = (
        f"Cox model of `{fit.inputs.target}` until `{name}`: {events:,} events, "
        f"{len(t) - events:,} still followed (censored), likelihood-ratio {an.fmt_p(table.lr_p)}."
    )
    table.sentence = head + (
        f" {min(clear, key=lambda term: term.p_value or 1).sentence}" if clear else ""
    )
    changing = [term for term in terms if term.proportional is False]
    names = ", ".join(f"`{term.term}`" for term in changing[:3])
    if len(changing) > 1:
        table.ph_sentence = (
            f"The effects of {names} change with time (Schoenfeld p < 0.01): proportional "
            "hazards do not hold for them; split the follow-up or add an interaction with time "
            "(R-REG-SURV-02)."
        )
    elif changing:
        table.ph_sentence = (
            f"The effect of {names} changes with time (Schoenfeld p < 0.01): proportional hazards "
            "do not hold for it; split the follow-up or add an interaction with time "
            "(R-REG-SURV-02)."
        )
    else:
        table.ph_sentence = (
            "No effect changes clearly with time (Schoenfeld residuals, p ≥ 0.01): proportional "
            "hazards hold (R-REG-SURV-02)."
        )
    if sampled:
        table.notes.append(
            f"A reproducible sample of {MAX_COX_ROWS:,} rows is fitted (R-REG-SURV-01)."
        )
    if reason_neg:
        table.notes.append(reason_neg.strip())
    table.notes.append("Ties by Breslow's method; survival curves by group: Compare › Durations.")
    return table


def _hazard_sentence(term: HazardTerm, event: str) -> str:
    ratio = an.fmt(term.hazard_ratio)
    return (
        f"Each extra unit of `{term.term}` multiplies the hazard of `{event}` by {ratio} "
        f"[{an.fmt(term.ci_low)} ; {an.fmt(term.ci_high)}], {an.fmt_p(term.p_value)}."
    )


# --- Tobit ----------------------------------------------------------------------------------------


@dataclass
class TobitTerm:
    term: str
    estimate: float  # on the latent value
    std_error: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    marginal: float  # McDonald–Moffitt: on the observed value
    least_squares: float


@dataclass
class TobitTable:
    available: bool
    reason: str
    lower: float | None = None
    upper: float | None = None
    rows: int = 0
    at_lower: int = 0
    at_upper: int = 0
    sampled: bool = False
    terms: list[TobitTerm] = field(default_factory=list)
    sigma: float | None = None
    sigma_se: float | None = None
    llf: float | None = None
    converged: bool = True
    sentence: str = ""
    notes: list[str] = field(default_factory=list)


def _tobit_parts(
    theta: np.ndarray, x: np.ndarray, y: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[float, np.ndarray]:
    """Minus the log-likelihood and its gradient in (β, log σ)."""
    k = x.shape[1]
    beta, log_sigma = theta[:k], theta[k]
    sigma = math.exp(log_sigma)
    xb = x @ beta
    mid = ~(low | high)
    llf = 0.0
    grad_beta = np.zeros(k)
    grad_log = 0.0
    if mid.any():
        z = (y[mid] - xb[mid]) / sigma
        llf += float((-0.5 * z * z - 0.5 * math.log(2 * math.pi) - log_sigma).sum())
        grad_beta += x[mid].T @ (z / sigma)
        grad_log += float((z * z - 1).sum())
    if low.any():
        z = (y[low] - xb[low]) / sigma  # P(y* ≤ L) = Φ((L − xb)/σ)
        logcdf = special.log_ndtr(z)
        ratio = np.exp(stats.norm.logpdf(z) - logcdf)
        llf += float(logcdf.sum())
        grad_beta += x[low].T @ (-ratio / sigma)
        grad_log += float((-ratio * z).sum())
    if high.any():
        z = (xb[high] - y[high]) / sigma  # P(y* ≥ U) = Φ((xb − U)/σ)
        logcdf = special.log_ndtr(z)
        ratio = np.exp(stats.norm.logpdf(z) - logcdf)
        llf += float(logcdf.sum())
        grad_beta += x[high].T @ (ratio / sigma)
        grad_log += float((-ratio * z).sum())
    return -llf, -np.append(grad_beta, grad_log)


def tobit_fit(
    x: np.ndarray, y: np.ndarray, lower: float | None, upper: float | None
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Tobit by maximum likelihood (BFGS, exact gradient, OLS start): (β, log σ), their
    covariance from the numerical Hessian of the gradient, the log-likelihood, convergence."""
    low = y <= lower if lower is not None else np.zeros(len(y), dtype=bool)
    high = y >= upper if upper is not None else np.zeros(len(y), dtype=bool)
    start_beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ start_beta
    start = np.append(start_beta, math.log(max(float(np.std(resid)), 1e-8)))
    found = optimize.minimize(
        _tobit_parts, start, args=(x, y, low, high), jac=True, method="BFGS",
        options={"gtol": 1e-8, "maxiter": 2_000},
    )  # fmt: skip
    theta = found.x
    m = len(theta)
    hessian = np.zeros((m, m))
    for i in range(m):
        step = 1e-5 * max(1.0, abs(theta[i]))
        up, down = theta.copy(), theta.copy()
        up[i] += step
        down[i] -= step
        hessian[:, i] = (
            _tobit_parts(up, x, y, low, high)[1] - _tobit_parts(down, x, y, low, high)[1]
        ) / (2 * step)
    hessian = (hessian + hessian.T) / 2
    covariance = np.linalg.pinv(hessian)
    return theta, covariance, -float(found.fun), bool(found.success)


def tobit(fit: an.Fitted) -> TobitTable:
    """Advanced › More › Censored (076, R-REG-TOB-01)."""
    if fit.model not in LEAST_SQUARES or fit.ls is None:
        return TobitTable(False, "The Tobit model reads the target as it is, with least squares.")
    lower, upper = bounds(fit.design.y)
    if lower is None and upper is None:
        return TobitTable(
            False,
            "No bound holds 10 % of the rows: the target is not held at a minimum or a maximum "
            "(R-REG-TOB-01).",
        )
    n = len(fit.design.y)
    picked = _sample(n, MAX_TOBIT_ROWS)
    x, y = fit.design.x[picked], fit.design.y[picked]
    theta, covariance, llf, converged = tobit_fit(x, y, lower, upper)
    k = x.shape[1]
    beta, sigma = theta[:k], math.exp(theta[k])
    xb = x @ beta
    share = np.ones(len(y))
    if upper is not None:
        share = stats.norm.cdf((upper - xb) / sigma)
    if lower is not None:
        share = share - stats.norm.cdf((lower - xb) / sigma)
    mean_share = float(np.mean(share))
    ls = least_squares(x, y)
    terms = []
    for j, term in enumerate(fit.design.terms):
        se = math.sqrt(covariance[j, j]) if covariance[j, j] > 0 else None
        p = float(2 * stats.norm.sf(abs(beta[j] / se))) if se else None
        terms.append(
            TobitTerm(
                term,
                float(beta[j]),
                se,
                float(beta[j] - 1.959963984540054 * se) if se else None,
                float(beta[j] + 1.959963984540054 * se) if se else None,
                p,
                float(beta[j]) * mean_share,
                float(ls.beta[j]),
            )
        )
    sigma_se = sigma * math.sqrt(covariance[k, k]) if covariance[k, k] > 0 else None
    table = TobitTable(
        True,
        "",
        lower,
        upper,
        len(y),
        int((y <= lower).sum()) if lower is not None else 0,
        int((y >= upper).sum()) if upper is not None else 0,
        n > MAX_TOBIT_ROWS,
        terms,
        sigma,
        sigma_se,
        llf,
        converged,
    )
    table.sentence = _tobit_sentence(table, fit.inputs.target, mean_share)
    if not converged:
        table.notes.append("The likelihood did not fully converge: read the estimates with care.")
    if fit.ls.weights is not None:
        table.notes.append("The Tobit model does not use the row weights.")
    if table.sampled:
        table.notes.append(f"A reproducible sample of {MAX_TOBIT_ROWS:,} rows is fitted.")
    return table


def _tobit_sentence(table: TobitTable, target: str, share: float) -> str:
    where = []
    if table.lower is not None:
        where.append(f"{table.at_lower:,} rows at {an.fmt(table.lower)}")
    if table.upper is not None:
        where.append(f"{table.at_upper:,} at {an.fmt(table.upper)}")
    effects = [t for t in table.terms[1:] if t.p_value is not None]
    text = (
        f"Tobit model of `{target}`, censored with {' and '.join(where)} (R-REG-TOB-01). On the "
        f"observed value, each effect is the latent one × {share:.2f} (the share of rows between "
        "the bounds, McDonald–Moffitt)."
    )
    if effects:
        best = min(effects, key=lambda t: t.p_value or 1)
        text += (
            f" `{best.term}`: {an.fmt(best.estimate)} on the latent value, {an.fmt(best.marginal)} "
            f"on the observed one, against {an.fmt(best.least_squares)} for least squares."
        )
    return text
