"""Mixed model (076), in Advanced › More: a random intercept by group (a patient, a school, a
store), a random slope when the user picks one, fitted by statsmodels MixedLM (REML); the share of
variance between groups (ICC) and the likelihood-ratio test against least squares. Rule
R-REG-MIX-01. One thread: Pyodide-compatible."""

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

from quantilo.regression import analyze as an

MIN_GROUPS = 5  # R-REG-MIX-01
MAX_ROWS = 20_000
SEED = 42
LEAST_SQUARES = ("ols", "wls")


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
    """Advanced › More › Mixed model (076, R-REG-MIX-01): `codes` over the used rows."""
    if fit.model not in LEAST_SQUARES or fit.ls is None:
        return MixedTable(False, "The mixed model reads the target as it is, with least squares.")
    if codes is None or group is None:
        return MixedTable(
            False,
            "Set Grouped by (a patient, a school, a store) in How: the mixed model needs the "
            "groups whose rows look alike (R-REG-MIX-01).",
        )
    ok = codes >= 0
    picked = np.flatnonzero(ok)
    picked = picked[_sample_groups(codes[picked], MAX_ROWS)]
    x, y, g = fit.design.x[picked], fit.design.y[picked], codes[picked]
    distinct, sizes = np.unique(g, return_counts=True)
    if len(distinct) < MIN_GROUPS or sizes.max() < 2:
        return MixedTable(
            False,
            f"{len(distinct)} groups of `{group}`, the largest of {int(sizes.max())} rows: the "
            f"mixed model needs {MIN_GROUPS} groups and one with 2 rows at least (R-REG-MIX-01).",
        )
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
