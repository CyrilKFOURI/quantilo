"""Mixed model (076), in Advanced › More: a random intercept by group (a patient, a school, a
store), a random slope when the user picks one, fitted by statsmodels MixedLM (REML); the share of
variance between groups (ICC) and the likelihood-ratio test against least squares. A yes/no or
count target takes a random intercept on the log-odds or log scale (`glmm.py`, adaptive
Gauss–Hermite). Rules R-REG-MIX-01, 02. One thread: Pyodide-compatible."""

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from statsmodels.genmod.families import Binomial, Poisson
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.mixed_linear_model import MixedLM

from quantilo.regression import analyze as an
from quantilo.regression import glmm

MIN_GROUPS = 5  # R-REG-MIX-01
MAX_ROWS = 20_000
SEED = 42
LEAST_SQUARES = ("ols", "wls")
GENERALISED = ("logit", "poisson")  # R-REG-MIX-02
MAX_GLMM_ROWS = 20_000
LOGISTIC_VARIANCE = math.pi**2 / 3  # of the latent logistic error


@dataclass
class MixedTerm:
    term: str
    estimate: float
    std_error: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    least_squares: float
    ls_se: float | None


@dataclass
class MixedTable:
    available: bool
    reason: str
    group: str | None = None
    group_reason: str = ""
    slope: str | None = None
    groups: int = 0
    rows: int = 0
    sampled: bool = False
    terms: list[MixedTerm] = field(default_factory=list)
    group_variance: float | None = None
    slope_variance: float | None = None
    residual_variance: float | None = None
    icc: float | None = None
    lr: float | None = None
    lr_p: float | None = None
    converged: bool = True
    sentence: str = ""
    notes: list[str] = field(default_factory=list)
    family: str = "gaussian"  # R-REG-MIX-02: "logit" or "poisson" for a generalised model


def _sample_groups(codes: np.ndarray, size: int) -> np.ndarray:
    """R-REG-MIX-01: whole groups drawn at random (seed 42) until `size` rows."""
    if len(codes) <= size:
        return np.arange(len(codes))
    groups = np.unique(codes)
    order = np.random.default_rng(SEED).permutation(groups)
    counts = {int(g): int((codes == g).sum()) for g in groups}
    chosen, total = [], 0
    for g in order:
        if total + counts[int(g)] > size and chosen:
            break
        chosen.append(int(g))
        total += counts[int(g)]
    return np.flatnonzero(np.isin(codes, chosen))


def mixed(
    fit: an.Fitted, codes: np.ndarray | None, group: str | None, reason: str, slope: str | None
) -> MixedTable:
    """Advanced › More › Mixed model (076, R-REG-MIX-01, 02): `codes` over the used rows."""
    generalised = fit.model in GENERALISED
    if not generalised and (fit.model not in LEAST_SQUARES or fit.ls is None):
        return MixedTable(
            False,
            "The mixed model reads the target as it is with least squares, or a yes/no or count "
            "target with its family (R-REG-MIX-02).",
        )
    if codes is None or group is None:
        return MixedTable(
            False,
            "Set Grouped by (a patient, a school, a store) in How: the mixed model needs the "
            "groups whose rows look alike (R-REG-MIX-01).",
        )
    ok = codes >= 0
    picked = np.flatnonzero(ok)
    if generalised and len(picked) > MAX_GLMM_ROWS:
        # R-REG-MIX-02: rows drawn within every group (whole groups would leave few of them)
        picked = np.sort(np.random.default_rng(SEED).choice(picked, MAX_GLMM_ROWS, replace=False))
    else:
        picked = picked[_sample_groups(codes[picked], MAX_ROWS)]
    x, y, g = fit.design.x[picked], fit.design.y[picked], codes[picked]
    distinct, sizes = np.unique(g, return_counts=True)
    if len(distinct) < MIN_GROUPS or sizes.max() < 2:
        return MixedTable(
            False,
            f"{len(distinct)} groups of `{group}`, the largest of {int(sizes.max())} rows: the "
            f"mixed model needs {MIN_GROUPS} groups and one with 2 rows at least (R-REG-MIX-01).",
        )
    if generalised:
        offset = None
        if fit.glm is not None and fit.glm.offset is not None:
            offset = np.asarray(fit.glm.offset, dtype=float)[picked]
        return _generalised(fit, x, y, g, offset, group, reason, n_rows=int(ok.sum()), slope=slope)
    assert fit.ls is not None
    exog_re = np.ones((len(y), 1))
    if slope is not None:
        if slope not in fit.design.terms:
            raise ValueError(f"`{slope}` is not a term of the model.")
        exog_re = np.column_stack([exog_re, x[:, fit.design.terms.index(slope)]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MixedLM(y, x, groups=g, exog_re=exog_re)
        found: Any = model.fit(reml=True)
        ml: Any = model.fit(reml=False)
    params = np.asarray(found.fe_params, dtype=float)
    bse = np.asarray(found.bse_fe, dtype=float)
    cov_re = np.asarray(found.cov_re, dtype=float)
    scale = float(found.scale)
    n = len(y)
    ols = np.linalg.lstsq(x, y, rcond=None)[0]
    e = y - x @ ols
    ols_llf = -n / 2 * (math.log(2 * math.pi) + math.log(float(e @ e) / n) + 1)
    lr = max(2 * (float(ml.llf) - ols_llf), 0.0)
    # the variance sits at its bound under the null: ½ χ²(q−1) + ½ χ²(q), q random effects
    q = exog_re.shape[1]
    lr_p = 0.5 * float(stats.chi2.sf(lr, q)) + (
        0.5 * float(stats.chi2.sf(lr, q - 1)) if q > 1 else 0.0
    )
    se_ls = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
    full_ls = fit.ls.beta
    terms = []
    for j, term in enumerate(fit.design.terms):
        se = float(bse[j]) if j < len(bse) and math.isfinite(bse[j]) and bse[j] > 0 else None
        p = float(2 * stats.norm.sf(abs(params[j] / se))) if se else None
        terms.append(
            MixedTerm(
                term,
                float(params[j]),
                se,
                float(params[j] - 1.959963984540054 * se) if se else None,
                float(params[j] + 1.959963984540054 * se) if se else None,
                p,
                float(full_ls[j]),
                float(se_ls[j]) if se_ls[j] > 0 else None,
            )
        )
    tau = float(cov_re[0, 0])
    icc = tau / (tau + scale) if tau + scale > 0 else None
    table = MixedTable(
        True,
        "",
        group,
        reason,
        slope,
        len(distinct),
        n,
        n < int(ok.sum()),
        terms,
        tau,
        float(cov_re[1, 1]) if q > 1 else None,
        scale,
        icc,
        lr,
        lr_p,
        bool(getattr(found, "converged", True)),
    )
    verdict = "improves" if lr_p < 0.05 else "does not clearly improve"
    table.sentence = (
        f"{icc:.0%} of the variance lies between the {len(distinct):,} groups of `{group}` (ICC); "
        f"the random intercept {verdict} the fit (likelihood ratio {an.fmt_p(lr_p)}, "
        "R-REG-MIX-01)."
        if icc is not None
        else f"The random intercept {verdict} the fit ({an.fmt_p(lr_p)})."
    )
    if slope is not None:
        table.notes.append(
            f"The slope of `{slope}` varies by group (variance {an.fmt(table.slope_variance)})."
        )
    if not table.converged:
        table.notes.append("The fit did not fully converge: read the variances with care.")
    if table.sampled:
        table.notes.append(f"Whole groups are drawn at random up to {MAX_ROWS:,} rows.")
    if fit.ls.weights is not None:
        table.notes.append("The mixed model does not use the row weights.")
    return table


def _generalised(
    fit: an.Fitted,
    x: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    offset: np.ndarray | None,
    group: str,
    reason: str,
    n_rows: int,
    slope: str | None,
) -> MixedTable:
    """R-REG-MIX-02: a random intercept on the scale of the link (log-odds, log of the rate),
    against the same model without it on the same rows; the variance at its bound under the
    null: p = ½ P(χ²₁ > LR)."""
    link: glmm.Link = "logit" if fit.model == "logit" else "poisson"
    family = Binomial() if link == "logit" else Poisson()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain: Any = GLM(y, x, family=family, offset=offset).fit()
    start = np.asarray(plain.params, dtype=float)
    found = glmm.fit(link, x, y, g, start, offset)
    se_all = np.sqrt(np.clip(np.diag(found.cov), 0, None))
    plain_se = np.asarray(plain.bse, dtype=float)
    terms = []
    for j, term in enumerate(fit.design.terms):
        se = float(se_all[j]) if math.isfinite(se_all[j]) and se_all[j] > 0 else None
        b = float(found.beta[j])
        terms.append(
            MixedTerm(
                term,
                b,
                se,
                b - 1.959963984540054 * se if se else None,
                b + 1.959963984540054 * se if se else None,
                float(2 * stats.norm.sf(abs(b / se))) if se else None,
                float(start[j]),
                float(plain_se[j]) if math.isfinite(plain_se[j]) and plain_se[j] > 0 else None,
            )
        )
    tau2 = found.tau**2
    lr = max(2 * (found.llf - float(plain.llf)), 0.0)
    lr_p = 0.5 * float(stats.chi2.sf(lr, 1)) if lr > 0 else 1.0
    icc = tau2 / (tau2 + LOGISTIC_VARIANCE) if link == "logit" else None
    distinct = len(np.unique(g))
    table = MixedTable(
        True,
        "",
        group,
        reason,
        None,
        distinct,
        len(y),
        len(y) < n_rows,
        terms,
        tau2,
        None,
        LOGISTIC_VARIANCE if link == "logit" else None,
        icc,
        lr,
        lr_p,
        found.converged,
        family=link,
    )
    verdict = "improves" if lr_p < 0.05 else "does not clearly improve"
    if link == "logit":
        table.sentence = (
            f"{icc:.0%} of the variance of the latent log-odds lies between the {distinct:,} "
            f"groups of `{group}` (ICC, logistic error π²/3); the random intercept {verdict} the "
            f"fit (likelihood ratio {an.fmt_p(lr_p)}, R-REG-MIX-02)."
        )
    else:
        spread = math.exp(found.tau)
        table.sentence = (
            f"The rate differs between the {distinct:,} groups of `{group}`: a typical group's "
            f"rate is ×{spread:.2f} or ÷{spread:.2f} of the average one (σ {found.tau:.3g} on the "
            f"log scale); the random intercept {verdict} the fit (likelihood ratio "
            f"{an.fmt_p(lr_p)}, R-REG-MIX-02)."
        )
    table.notes.append(
        f"R-REG-MIX-02: maximum likelihood, {glmm.NODES}-point adaptive Gauss–Hermite "
        "quadrature around each group's mode; the coefficients are "
        + (
            "log-odds for a given group (conditional), larger than the averaged ones of the "
            "model without groups."
            if link == "logit"
            else "logs of the rate for a given group; the slopes keep their meaning, only the "
            "level moves by group."
        )
    )
    if slope is not None:
        table.notes.append("A random slope is fitted for least squares only (R-REG-MIX-02).")
    if not found.converged:
        table.notes.append("The fit did not fully converge: read the variances with care.")
    if table.sampled:
        table.notes.append(
            f"R-REG-MIX-02: a reproducible sample of {MAX_GLMM_ROWS:,} rows, drawn within every "
            "group, is fitted."
        )
    return table
