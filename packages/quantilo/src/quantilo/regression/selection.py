"""Selection (077), in Advanced › More: which terms does a lasso keep, what least squares say on
those (post-lasso), the effect of one column once the controls are chosen by two lassos (double
selection, Belloni, Chernozhukov & Hansen 2014), and how often each term is kept on half samples
(stability selection, Meinshausen & Bühlmann 2010). The penalty is the plug-in one of Belloni et
al., not cross-validation: deterministic and fast. Rules R-REG-16, R-REG-SEL-01…05.

Stepwise selection is left out on purpose: its p-values are wrong after the search.
Everything is numpy / scikit-learn on one thread: Pyodide-compatible."""

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from quantilo.regression import analyze as an
from quantilo.regression import covariance as cov
from quantilo.regression.analyze import RegressionCheck
from quantilo.regression.fit import least_squares

LEAST_SQUARES = ("ols", "wls", "log_ols", "boxcox")
SEL_ROWS = 20_000
HALVES = 100  # R-REG-SEL-04
KEEP_SHARE = 0.6  # R-REG-SEL-04: kept on 60 % of the half samples
MANY_TERMS = 30  # R-REG-16
ROWS_PER_TERM = 10  # R-REG-16
C = 1.1  # Belloni et al.
SEED = 42


@dataclass
class SelectedTerm:
    term: str
    lasso: float  # in the term's own unit, 0 when dropped
    kept: bool
    post: float | None  # post-lasso least squares
    post_low: float | None
    post_high: float | None
    share: float | None  # stability selection: share of half samples that keep it
    stable: bool = False


@dataclass
class DoubleSelection:
    term: str
    estimate: float
    low: float | None
    high: float | None
    p_value: float | None
    controls: list[str]  # kept by either lasso
    ols: float  # the first plane, every control in
    ols_low: float | None
    ols_high: float | None
    sentence: str


@dataclass
class SelectionTable:
    available: bool
    reason: str
    rows: int = 0
    sampled: bool = False
    penalty: float = 0.0
    penalty_reason: str = ""
    terms: list[SelectedTerm] = field(default_factory=list)
    kept: int = 0
    stable: int = 0
    double: DoubleSelection | None = None
    double_reason: str = ""
    sentence: str = ""
    notes: list[str] = field(default_factory=list)


def _unavailable(fit: an.Fitted) -> str | None:
    if fit.ls is None or fit.model not in LEAST_SQUARES:
        label = fit.family.label if fit.family is not None else fit.model
        return f"Selection is offered for a number explained by least squares, not the {label}."
    if fit.design.x.shape[1] < 3:
        return "Needs at least two explanatory terms to choose among."
    return None


def penalty(n: int, p: int) -> float:
    """R-REG-SEL-01: λ = 2c√n Φ⁻¹(1 − γ/(2p)), c = 1.1, γ = 0.1 / log n (Belloni et al.)."""
    gamma = 0.1 / math.log(max(n, 3))
    return 2 * C * math.sqrt(n) * float(stats.norm.ppf(1 - gamma / (2 * max(p, 1))))


def lasso(x: np.ndarray, y: np.ndarray, iterations: int = 3) -> tuple[np.ndarray, float]:
    """Plug-in lasso on standardized columns (each counts the same), σ re-estimated from the
    post-lasso residuals a few times; coefficients back in each column's unit."""
    n, p = x.shape
    centre, spread = x.mean(axis=0), x.std(axis=0)
    spread = np.where(spread > 0, spread, 1.0)
    z = (x - centre) / spread
    yc = y - y.mean()
    lam = penalty(n, p)
    sigma = float(np.std(yc)) or 1.0
    coef = np.zeros(p)
    for _ in range(iterations):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = Lasso(alpha=lam * sigma / (2 * n), max_iter=5_000)
            model.fit(z, yc)
        coef = np.asarray(model.coef_, dtype=float)
        kept = np.flatnonzero(coef)
        if len(kept):
            e = yc - z[:, kept] @ np.linalg.lstsq(z[:, kept], yc, rcond=None)[0]
            sigma = math.sqrt(float(e @ e) / max(n - len(kept), 1)) or sigma
        else:
            sigma = float(np.std(yc)) or sigma
    return coef / spread, lam


def _sample(n: int) -> np.ndarray:
    if n <= SEL_ROWS:
        return np.arange(n)
    return np.sort(np.random.default_rng(SEED).choice(n, SEL_ROWS, replace=False))


def double_selection(x: np.ndarray, y: np.ndarray, j: int) -> tuple[float, float, float, list[int]]:
    """Lasso of y on the controls, lasso of d = x_j on the controls, least squares of y on d and
    the union, HC3 errors (R-REG-SEL-03). x has the intercept first."""
    controls = [c for c in range(1, x.shape[1]) if c != j]
    w = x[:, controls]
    first, _ = lasso(w, y)
    second, _ = lasso(w, x[:, j])
    union = [controls[i] for i in range(len(controls)) if first[i] != 0 or second[i] != 0]
    design = x[:, [0, j, *union]]
    ls = least_squares(design, y)
    se = math.sqrt(max(float(cov.hc(ls.arrays, "hc3")[1, 1]), 0))
    return float(ls.beta[1]), se, float(ls.df_resid), union


def section(fit: an.Fitted) -> SelectionTable:
    """Advanced › More › Selection (077)."""
    missing = _unavailable(fit)
    if missing is not None:
        return SelectionTable(False, missing)
    assert fit.ls is not None
    a = fit.ls.arrays
    x_all, y_all = a.x, a.x @ fit.ls.beta + a.e
    rows = _sample(len(y_all))
    x, y = x_all[rows], y_all[rows]
    n, k = x.shape
    names = fit.design.terms
    coef, lam = lasso(x[:, 1:], y)
    kept = [j + 1 for j in np.flatnonzero(coef)]
    post = np.full(k, np.nan)
    post_se = np.full(k, np.nan)
    df = n - len(kept) - 1
    if kept:
        ls = least_squares(x[:, [0, *kept]], y)
        v = cov.hc(ls.arrays, "hc3")
        post[kept] = ls.beta[1:]
        post_se[kept] = np.sqrt(np.clip(np.diag(v)[1:], 0, None))
    crit = float(stats.t.ppf(0.975, max(df, 1)))
    shares = _stability(x, y)
    terms: list[SelectedTerm] = []
    for j in range(1, k):
        has = math.isfinite(post_se[j]) and post_se[j] > 0
        share = float(shares[j - 1])
        terms.append(
            SelectedTerm(
                names[j],
                float(coef[j - 1]),
                j in kept,
                float(post[j]) if math.isfinite(post[j]) else None,
                float(post[j] - crit * post_se[j]) if has else None,
                float(post[j] + crit * post_se[j]) if has else None,
                share,
                share >= KEEP_SHARE,
            )
        )
    table = SelectionTable(
        True,
        "",
        n,
        n < len(y_all),
        lam,
        (
            f"R-REG-SEL-01: plug-in penalty λ = {lam:.1f} (Belloni et al.): large enough that a "
            "column of pure noise is almost never kept; no cross-validation, the same answer "
            "every time."
        ),
        terms,
        len(kept),
        sum(1 for t in terms if t.stable),
    )
    focus = _focus(fit)
    if focus is None:
        table.double_reason = (
            "R-REG-SEL-03: pick a focus column (the conclusion's column) among the numbers to "
            "estimate its effect by double selection."
        )
    else:
        estimate, se, dof, union = double_selection(x, y, focus)
        crit_d = float(stats.t.ppf(0.975, max(dof, 1)))
        ols_se = math.sqrt(max(float(fit.covariance[focus, focus]), 0))
        crit_o = float(stats.t.ppf(0.975, fit.df))
        ols = float(fit.ls.beta[focus])
        p = float(2 * stats.t.sf(abs(estimate / se), dof)) if se > 0 else None
        table.double = DoubleSelection(
            names[focus],
            estimate,
            estimate - crit_d * se if se > 0 else None,
            estimate + crit_d * se if se > 0 else None,
            p,
            [names[c] for c in union],
            ols,
            ols - crit_o * ols_se if ols_se > 0 else None,
            ols + crit_o * ols_se if ols_se > 0 else None,
            _double_sentence(names[focus], estimate, se, crit_d, p, len(union), k - 2),
        )
    table.sentence = (
        f"The lasso keeps {len(kept)} of {k - 1} terms; {table.stable} are kept on at least "
        f"{KEEP_SHARE:.0%} of {HALVES} half samples (stability selection, R-REG-SEL-04). The "
        "post-lasso intervals ignore the search: read them as a description, not a test "
        "(R-REG-SEL-02)."
    )
    if table.sampled:
        table.notes.append(f"Computed on a reproducible sample of {n:,} rows.")
    return table


def _focus(fit: an.Fitted) -> int | None:
    """R-REG-SEL-03: the focus column when it is a single numeric term."""
    focus = fit.inputs.focus
    if focus is None:
        return None
    for j, column in enumerate(fit.design.columns):
        if j and column == focus and fit.design.terms[j] == focus:
            return j
    return None


def _double_sentence(
    term: str, estimate: float, se: float, crit: float, p: float | None, union: int, of: int
) -> str:
    interval = f"[{an.fmt(estimate - crit * se)} ; {an.fmt(estimate + crit * se)}]" if se else ""
    return (
        f"With the controls chosen by two lassos ({union} of {of} kept), the effect of `{term}` "
        f"is {an.fmt(estimate)} {interval}, {an.fmt_p(p)}: this interval stays valid after the "
        "selection (double selection, R-REG-SEL-03)."
    )


def _stability(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """R-REG-SEL-04: the share of 100 half samples whose lasso keeps each term."""
    n, k = x.shape
    rng = np.random.default_rng(SEED)
    counts = np.zeros(k - 1)
    half = n // 2
    for _ in range(HALVES):
        rows = rng.choice(n, half, replace=False)
        coef, _ = lasso(x[rows, 1:], y[rows], iterations=1)
        counts += coef != 0
    return counts / HALVES


def proposal(fit: an.Fitted) -> RegressionCheck | None:
    """R-REG-16: many terms for the rows → double selection proposed (counts only, no fit)."""
    if _unavailable(fit) is not None:
        return None
    n, k = fit.design.x.shape
    terms = k - 1
    if terms <= MANY_TERMS and terms * ROWS_PER_TERM <= n:
        return None
    why = (
        f"{terms} terms"
        if terms > MANY_TERMS
        else f"{terms} terms for {n:,} rows (fewer than 10 a term)"
    )
    return RegressionCheck(
        "selection",
        "info",
        "Many terms",
        f"{why}: some effects are estimated loosely. Advanced › More › Selection shows which "
        "terms a lasso keeps, and double selection estimates the focus column's effect with "
        "chosen controls (R-REG-16).",
        "R-REG-16",
        "selection",
    )
