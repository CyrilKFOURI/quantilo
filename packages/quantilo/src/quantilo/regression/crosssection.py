"""The cross-section of a quant (074), all in Advanced › More: Fama–MacBeth (one regression a
period, the mean of the coefficients with Newey–West errors), the information coefficient of each
column (rank correlation with the target a period, ICIR), several outcomes on the same columns
(coefficient matrix, Wilks and Pillai, the GRS test of the alphas) and the regression on principal
components. Rules R-REG-XS-01…06.

Everything is numpy / scipy on one thread: Pyodide-compatible."""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from quantilo.correlations import structure as pca
from quantilo.regression import analyze as an
from quantilo.regression import covariance as cov
from quantilo.regression import stability as st
from quantilo.regression.design import INTERCEPT
from quantilo.regression.fit import least_squares

LEAST_SQUARES = st.LEAST_SQUARES
MIN_PERIODS = 10  # R-REG-XS-01
MIN_IC_ROWS = 5  # R-REG-XS-03
IC_T = 2.0
MAX_OUTPUTS = 10  # R-REG-XS-04
MAX_IC_COLUMNS = 30
ALPHA = 0.05


class CrossSectionError(ValueError):
    pass


# --- Fama–MacBeth and IC ------------------------------------------------------------------------


@dataclass
class MacBethTerm:
    term: str
    estimate: float
    std_error: float | None
    ci_low: float | None
    ci_high: float | None
    t: float | None
    p_value: float | None
    pooled: float  # the first plane's coefficient
    pooled_se: float | None
    positive: float  # share of the periods with a positive coefficient


@dataclass
class ColumnIc:
    column: str
    mean: float
    sd: float | None
    icir: float | None
    t: float | None  # Newey–West
    p_value: float | None
    positive: float  # share of the periods with IC > 0
    periods: int
    ranks: bool  # |t| ≥ 2 (R-REG-XS-03)
    sentence: str = ""


@dataclass
class CoefficientSeries:
    term: str
    periods: list[str]
    values: list[float]


@dataclass
class CrossSectionTable:
    available: bool
    reason: str
    entity: str | None = None
    period: str | None = None
    periods: int = 0  # used
    skipped: int = 0  # periods with too few rows
    rows: float = 0.0  # mean rows a period
    lag: int = 0
    lag_reason: str = ""
    terms: list[MacBethTerm] = field(default_factory=list)
    r2: float | None = None  # mean R² of the cross-sections
    ics: list[ColumnIc] = field(default_factory=list)
    series: list[CoefficientSeries] = field(default_factory=list)
    sentence: str = ""
    ic_sentence: str = ""
    notes: list[str] = field(default_factory=list)


def newey_west_mean(values: np.ndarray, lag: int) -> float:
    """Variance of the mean of a series with Bartlett weights (Newey & West 1987), no
    small-sample correction: the HAC of a regression on a constant."""
    t = len(values)
    e = values - values.mean()
    s = float(e @ e) / t
    for lead in range(1, min(lag, t - 1) + 1):
        s += 2 * (1 - lead / (lag + 1)) * float(e[lead:] @ e[:-lead]) / t
    return max(s, 0.0) / t


def _mean_test(values: np.ndarray, lag: int) -> tuple[float | None, float | None, float | None]:
    """Standard error, t and p of the mean of a series (R-REG-XS-02), t with T − 1 df."""
    t = len(values)
    if t < 2:
        return None, None, None
    se = math.sqrt(newey_west_mean(values, lag))
    if se <= 0:
        return None, None, None
    stat = float(values.mean()) / se
    return se, stat, float(2 * stats.t.sf(abs(stat), t - 1))


def _unavailable(fit: an.Fitted) -> str | None:
    """R-REG-XS-01: a panel fitted by least squares."""
    shape = fit.inputs.shape
    if shape.kind != "panel" or fit.grouping.period_codes is None:
        return (
            "Needs a panel: entities seen over periods (a stock a month, a store a week), so that "
            "there is one cross-section a period (R-REG-XS-01)."
        )
    if fit.ls is None or fit.model not in LEAST_SQUARES:
        label = fit.family.label if fit.family is not None else fit.model
        return f"Fama–MacBeth is read on least squares; the {label} model is not (yet)."
    return None


def _periods(fit: an.Fitted) -> tuple[np.ndarray, list[int], list[str]]:
    """Codes of the periods over the used rows, in time order, and their labels."""
    assert fit.grouping.period_codes is not None
    codes = fit.grouping.period_codes
    order = fit.inputs.period_order
    ranks = order[fit.design.used] if order is not None else codes
    present = sorted(
        {int(c) for c in codes if c >= 0}, key=lambda c: float(np.min(ranks[codes == c]))
    )
    dates = fit.inputs.dates
    labels: list[str] = []
    for c in present:
        rows = np.flatnonzero(codes == c)
        labels.append(str(dates[fit.design.used][rows[0]]) if dates is not None else str(c))
    return codes, present, labels


def _numeric_terms(fit: an.Fitted) -> list[int]:
    """Columns of the design that are a numeric column as it is (not an indicator, a curve)."""
    kinds = fit.design.kinds
    found = []
    for j, term in enumerate(fit.design.terms):
        column = fit.design.columns[j]
        group = fit.design.groups[j] if fit.design.groups else column
        if term == INTERCEPT or group != column or kinds.get(column, ("cat", []))[0] != "num":
            continue
        found.append(j)
    return found


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < MIN_IC_ROWS:
        return None
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def cross_section(fit: an.Fitted) -> CrossSectionTable:
    """Advanced › More › Fama–MacBeth and IC (074, R-REG-XS-01…03)."""
    missing = _unavailable(fit)
    if missing is not None:
        return CrossSectionTable(False, missing)
    assert fit.ls is not None
    x, y = fit.ls.arrays.x, fit.ls.arrays.x @ fit.ls.beta + fit.ls.arrays.e  # weighted if WLS
    codes, present, labels = _periods(fit)
    k = x.shape[1]
    betas: list[np.ndarray] = []
    r2s: list[float] = []
    kept: list[str] = []
    sizes: list[int] = []
    raw_y, raw_x = fit.design.y, fit.design.x
    numeric = _numeric_terms(fit)
    ic: dict[int, list[float]] = {j: [] for j in numeric[:MAX_IC_COLUMNS]}
    skipped = 0
    for code, label in zip(present, labels, strict=True):
        rows = codes == code
        xs, ys = x[rows], y[rows]
        for j in ic:
            value = _spearman(raw_x[rows, j], raw_y[rows])
            if value is not None:
                ic[j].append(value)
        if len(ys) < k + 2 or np.linalg.matrix_rank(xs) < k:
            skipped += 1
            continue
        beta = np.linalg.lstsq(xs, ys, rcond=None)[0]
        e = ys - xs @ beta
        tss = float(((ys - ys.mean()) ** 2).sum())
        betas.append(beta)
        r2s.append(1 - float(e @ e) / tss if tss > 0 else float("nan"))
        kept.append(label)
        sizes.append(int(rows.sum()))
    if len(betas) < MIN_PERIODS:
        return CrossSectionTable(
            False,
            f"{len(betas)} periods have at least {k + 2} rows for {k} terms: Fama–MacBeth needs "
            f"{MIN_PERIODS} (R-REG-XS-01).",
        )
    series = np.vstack(betas)
    t = len(series)
    lag = cov.newey_west_lag(t)
    se_pooled = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
    terms: list[MacBethTerm] = []
    crit = float(stats.t.ppf(0.975, t - 1))
    for j, term in enumerate(fit.design.terms):
        column = series[:, j]
        se, stat, p = _mean_test(column, lag)
        mean = float(column.mean())
        terms.append(
            MacBethTerm(
                term,
                mean,
                se,
                mean - crit * se if se is not None else None,
                mean + crit * se if se is not None else None,
                stat,
                p,
                float(fit.ls.beta[j]),
                float(se_pooled[j]) if se_pooled[j] > 0 else None,
                float(np.mean(column > 0)),
            )
        )
    ics = [_ic(fit.design.terms[j], np.asarray(values), lag) for j, values in ic.items() if values]
    ics.sort(key=lambda found: -abs(found.t or 0))
    keep = st._thin(t)
    drawn = [
        CoefficientSeries(
            fit.design.terms[j], [kept[i] for i in keep], [float(series[i, j]) for i in keep]
        )
        for j in range(1, k)
    ][: st.MAX_CURVES]
    finite_r2 = [r for r in r2s if math.isfinite(r)]
    table = CrossSectionTable(
        True,
        "",
        fit.inputs.shape.entity,
        fit.inputs.shape.period,
        t,
        skipped,
        float(np.mean(sizes)),
        lag,
        f"R-REG-XS-02: Newey–West on the {t} coefficients a term, lag {lag} = ⌊4(T/100)^(2/9)⌋: "
        "coefficients of neighbouring periods can look alike.",
        terms,
        float(np.mean(finite_r2)) if finite_r2 else None,
        ics,
        drawn,
    )
    table.sentence = _macbeth_sentence(table, fit)
    table.ic_sentence = _ic_sentence(ics, fit.inputs.target)
    if skipped:
        table.notes.append(
            f"{skipped} periods with fewer than {k + 2} rows (or columns without variation) are "
            "left out (R-REG-XS-01)."
        )
    if fit.ls.weights is not None:
        table.notes.append(f"Each cross-section is weighted by `{fit.inputs.weights}`.")
    return table


def _ic(term: str, values: np.ndarray, lag: int) -> ColumnIc:
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
    _, stat, p = _mean_test(values, lag)
    mean = float(values.mean())
    icir = mean / sd if sd else None
    found = ColumnIc(
        term,
        mean,
        sd,
        icir,
        stat,
        p,
        float(np.mean(values > 0)),
        len(values),
        stat is not None and abs(stat) >= IC_T,
    )
    found.sentence = (
        f"`{term}` ranks the target with a mean IC of {an.fmt(mean)} (ICIR {an.fmt(icir)}, "
        f"t {an.fmt(stat)}, positive in {found.positive:.0%} of {len(values)} periods)."
    )
    return found


def _macbeth_sentence(table: CrossSectionTable, fit: an.Fitted) -> str:
    clear = [term for term in table.terms[1:] if term.p_value is not None and term.p_value < ALPHA]
    head = (
        f"One cross-section a period: {table.periods} periods of {table.rows:.0f} rows on "
        f"average (R-REG-XS-01)."
    )
    if not clear:
        return head + " No mean coefficient differs clearly from 0 (Newey–West)."
    best = min(clear, key=lambda term: term.p_value or 1)
    return (
        f"{head} The clearest is `{best.term}`: {an.fmt(best.estimate)} on average "
        f"[{an.fmt(best.ci_low)} ; {an.fmt(best.ci_high)}], {an.fmt_p(best.p_value)} "
        f"({an.fmt(best.pooled)} pooled, in the first plane)."
    )


def _ic_sentence(ics: list[ColumnIc], target: str) -> str:
    if not ics:
        return "No numeric column to rank the target with."
    ranking = [ic for ic in ics if ic.ranks]
    if not ranking:
        return (
            f"No column ranks `{target}` consistently from one period to the next (|t| < 2, "
            "R-REG-XS-03)."
        )
    return ranking[0].sentence


# --- Several outcomes ---------------------------------------------------------------------------


@dataclass
class OutputCoefficient:
    estimate: float
    std_error: float | None
    p_value: float | None


@dataclass
class MultivariateTest:
    term: str
    df: int
    wilks: float
    wilks_f: float | None
    wilks_df1: float
    wilks_df2: float
    wilks_p: float | None
    pillai: float
    pillai_f: float | None
    pillai_df1: float
    pillai_df2: float
    pillai_p: float | None


@dataclass
class GrsTest:
    f: float
    df1: int
    df2: int
    p_value: float
    alphas: list[float]
    sentence: str


@dataclass
class OutputsTable:
    available: bool
    reason: str
    outputs: list[str] = field(default_factory=list)  # the first plane's target first
    terms: list[str] = field(default_factory=list)
    coefficients: list[list[OutputCoefficient]] = field(default_factory=list)  # term × output
    r2: list[float | None] = field(default_factory=list)
    rows: int = 0
    tests: list[MultivariateTest] = field(default_factory=list)
    grs: GrsTest | None = None
    grs_reason: str = ""
    sentence: str = ""
    notes: list[str] = field(default_factory=list)


def _f_wilks(lam: float, p: int, q: int, df_e: int) -> tuple[float | None, float, float]:
    """Rao's F for Wilks' lambda (p outcomes, q hypothesis df, df_e error df)."""
    t = math.sqrt((p * p * q * q - 4) / (p * p + q * q - 5)) if p * p + q * q - 5 > 0 else 1.0
    df1 = p * q
    df2 = (df_e - (p - q + 1) / 2) * t - (p * q - 2) / 2
    root = lam ** (1 / t)
    if root <= 0 or df2 <= 0:
        return None, df1, df2
    return (1 - root) / root * df2 / df1, df1, df2


def _f_pillai(v: float, p: int, q: int, df_e: int) -> tuple[float | None, float, float]:
    s = min(p, q)
    m = (abs(p - q) - 1) / 2
    n = (df_e - p - 1) / 2
    df1 = s * (2 * m + s + 1)
    df2 = s * (2 * n + s + 1)
    if s - v <= 0 or df2 <= 0:
        return None, df1, df2
    return (2 * n + s + 1) / (2 * m + s + 1) * v / (s - v), df1, df2


def multivariate_test(
    x: np.ndarray, ys: np.ndarray, columns: list[int], term: str
) -> MultivariateTest:
    """Wilks' lambda and Pillai's trace of the columns `columns` of x being 0 for every outcome."""
    n, k = x.shape
    p = ys.shape[1]
    inverse = np.linalg.pinv(x.T @ x)
    b = inverse @ x.T @ ys
    e = ys - x @ b
    error = e.T @ e
    picked = b[columns]
    middle = np.linalg.pinv(inverse[np.ix_(columns, columns)])
    hyp = picked.T @ middle @ picked
    q, df_e = len(columns), n - k
    lam = float(np.linalg.det(error) / np.linalg.det(error + hyp))
    v = float(np.trace(hyp @ np.linalg.pinv(hyp + error)))
    wf, w1, w2 = _f_wilks(lam, p, q, df_e)
    pf, p1, p2 = _f_pillai(v, p, q, df_e)
    return MultivariateTest(
        term,
        q,
        lam,
        wf,
        w1,
        w2,
        float(stats.f.sf(wf, w1, w2)) if wf is not None else None,
        v,
        pf,
        p1,
        p2,
        float(stats.f.sf(pf, p1, p2)) if pf is not None else None,
    )


def grs(x: np.ndarray, ys: np.ndarray) -> tuple[float, int, int, float, np.ndarray]:
    """Gibbons, Ross & Shanken (1989), in Cochrane's form with maximum-likelihood estimates:
    F = (T − N − L)/N · α'Σ⁻¹α / (1 + μ'Ω⁻¹μ) ~ F(N, T − N − L)."""
    t, n = ys.shape
    factors = x[:, 1:]
    l_count = factors.shape[1]
    b = np.linalg.lstsq(x, ys, rcond=None)[0]
    e = ys - x @ b
    sigma = e.T @ e / t
    mu = factors.mean(axis=0)
    centred = factors - mu
    omega = centred.T @ centred / t
    alpha = b[0]
    quad = float(alpha @ np.linalg.solve(sigma, alpha))
    scale = 1 + float(mu @ np.linalg.solve(omega, mu)) if l_count else 1.0
    df2 = t - n - l_count
    f = df2 / n * quad / scale
    return f, n, df2, float(stats.f.sf(f, n, df2)), alpha


def outputs(fit: an.Fitted, extra: dict[str, np.ndarray]) -> OutputsTable:
    """Advanced › More › Several outcomes (074, R-REG-XS-04, 05): the same columns explain the
    target and the outcomes in `extra` (values over the used rows)."""
    if fit.ls is None or fit.model not in LEAST_SQUARES or fit.transform is not None:
        return OutputsTable(False, "Several outcomes are read on least squares of the target.")
    if not extra:
        return OutputsTable(
            False, "Add an outcome: another numeric column the same columns may explain."
        )
    if len(extra) > MAX_OUTPUTS:
        raise CrossSectionError(f"At most {MAX_OUTPUTS} more outcomes (R-REG-XS-04).")
    names = [fit.inputs.target, *extra]
    ys = np.column_stack([fit.design.y, *extra.values()])
    complete = np.isfinite(ys).all(axis=1)
    x = fit.design.x[complete]
    ys = ys[complete]
    n, k = x.shape
    if n < k + len(names) + 2:
        return OutputsTable(False, f"{n} complete rows for {k} terms and {len(names)} outcomes.")
    table = OutputsTable(True, "", names, list(fit.design.terms), rows=n)
    columns: list[list[OutputCoefficient]] = []
    for j in range(len(names)):
        ls = least_squares(x, ys[:, j])
        se = np.sqrt(np.clip(np.diag(cov.hc(ls.arrays, "hc3")), 0, None))
        column = []
        for i in range(k):
            ok = se[i] > 0
            stat = ls.beta[i] / se[i] if ok else None
            p = float(2 * stats.t.sf(abs(stat), n - k)) if stat is not None else None
            column.append(OutputCoefficient(float(ls.beta[i]), float(se[i]) if ok else None, p))
        columns.append(column)
        table.r2.append(float(ls.r2) if math.isfinite(ls.r2) else None)
    table.coefficients = [[columns[j][i] for j in range(len(names))] for i in range(k)]
    groups = fit.design.groups or fit.design.columns
    for group in dict.fromkeys(groups[1:]):
        picked = [j for j in range(1, k) if groups[j] == group]
        table.tests.append(multivariate_test(x, ys, picked, group))
    table.grs, table.grs_reason = _grs(fit, x, ys, names, complete)
    table.sentence = _outputs_sentence(table)
    if not complete.all():
        table.notes.append(
            f"{int((~complete).sum()):,} rows without a value in one of the outcomes are left out."
        )
    table.notes.append(
        "Each outcome's standard errors are HC3 (R-REG-SE-05); the joint tests assume the "
        "outcomes' errors share one covariance across rows."
    )
    return table


def _grs(
    fit: an.Fitted, x: np.ndarray, ys: np.ndarray, names: list[str], complete: np.ndarray
) -> tuple[GrsTest | None, str]:
    """R-REG-XS-05: dated returns on numeric factors with an intercept, T > N + L."""
    t, n = ys.shape
    factors = x.shape[1] - 1
    if fit.inputs.shape.kind != "series" or fit.inputs.dates is None:
        return None, "GRS needs one row per date (R-REG-XS-05)."
    if fit.design.columns[0] != INTERCEPT or any(
        kind != "num" for kind, _ in fit.design.kinds.values()
    ):
        return None, "GRS needs an intercept and numeric factors (R-REG-XS-05)."
    if not all(st._looks_like_returns(ys[:, j]) for j in range(n)):
        return None, "GRS reads returns: every outcome a small number around 0 (R-REG-STAB-04)."
    if t <= n + factors:
        return None, f"{t} dates for {n} outcomes and {factors} factors: GRS needs T > N + L."
    f, df1, df2, p, alpha = grs(x, ys)
    verdict = "are not jointly 0" if p < ALPHA else "can be jointly 0"
    sentence = (
        f"The alphas of {', '.join(f'`{name}`' for name in names)} {verdict}: GRS F = {f:.3g}, "
        f"{an.fmt_p(p)} (R-REG-XS-05). An alpha that looks small on its own may not be, with the "
        "others."
    )
    return GrsTest(f, df1, df2, p, [float(a) for a in alpha], sentence), ""


def _outputs_sentence(table: OutputsTable) -> str:
    clear = [test for test in table.tests if test.pillai_p is not None and test.pillai_p < 0.05]
    if not clear:
        return (
            f"No column moves the {len(table.outputs)} outcomes jointly (Pillai's trace, p ≥ 0.05)."
        )
    best = min(clear, key=lambda test: test.pillai_p or 1)
    return (
        f"`{best.term}` moves the {len(table.outputs)} outcomes jointly (Pillai's trace "
        f"{best.pillai:.3f}, {an.fmt_p(best.pillai_p)}); {len(clear)} of {len(table.tests)} "
        "columns do."
    )


# --- Principal components -------------------------------------------------------------------------


@dataclass
class ComponentEffect:
    term: str
    estimate: float  # per unit of the column, through the components kept
    ci_low: float | None
    ci_high: float | None
    least_squares: float  # the first plane
    ls_low: float | None
    ls_high: float | None


@dataclass
class ComponentCoefficient:
    component: str  # "PC1"
    label: str  # its heaviest columns
    share: float  # of the variance of the columns
    estimate: float
    std_error: float | None
    p_value: float | None


@dataclass
class ComponentsTable:
    available: bool
    reason: str
    columns: list[str] = field(default_factory=list)
    kept: int = 0
    kept_reason: str = ""
    cumulative: float = 0.0
    components: list[ComponentCoefficient] = field(default_factory=list)
    effects: list[ComponentEffect] = field(default_factory=list)
    r2: float | None = None
    ls_r2: float | None = None
    proposed: bool = False
    sentence: str = ""
    notes: list[str] = field(default_factory=list)


def retained(values: np.ndarray) -> int:
    """R-REG-XS-06: the components whose variance is at least 1/30² of the first's; those below
    make the condition number exceed 30 (an exact or near-exact combination)."""
    if not len(values) or values[0] <= 0:
        return 1
    return max(1, int((values >= values[0] / an.CONDITION_HIGH**2).sum()))


def collinear(fit: an.Fitted) -> bool:
    """R-REG-XS-06: VIF > 10 or a condition number > 30 among the explanatory columns."""
    vifs, condition = an._collinearity(fit.design)
    return any((v.vif or 0) > an.VIF_HIGH for v in vifs) or (condition or 0) > an.CONDITION_HIGH


def components(fit: an.Fitted) -> ComponentsTable:
    """Advanced › More › Principal components (074, R-REG-XS-06)."""
    if fit.ls is None or fit.model not in LEAST_SQUARES or fit.transform is not None:
        return ComponentsTable(False, "The regression on components is read on least squares.")
    numeric = _numeric_terms(fit)
    if len(numeric) < 2:
        return ComponentsTable(False, "Needs at least 2 numeric explanatory columns.")
    x = fit.design.x
    others = [j for j in range(x.shape[1]) if j not in numeric]
    raw = x[:, numeric]
    sd = raw.std(axis=0)
    centre = raw.mean(axis=0)
    z = (raw - centre) / sd
    values, vectors = pca.eigen(pca.correlation(z))
    kept = retained(values)
    thresholds, _ = pca.parallel_thresholds(z)
    real = pca.retained_count(values, thresholds)
    names = [fit.design.terms[j] for j in numeric]
    scores = z @ vectors[:, :kept]
    design = np.column_stack([x[:, others], scores])
    w = fit.ls.weights
    ls = least_squares(design, fit.design.y, w)
    covariance = cov.hc(ls.arrays, "hc3")
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    df = max(ls.n - ls.k, 1)
    crit = float(stats.t.ppf(0.975, df))
    total = float(values.sum())
    table = ComponentsTable(
        True,
        "",
        names,
        kept,
        f"R-REG-XS-06: {len(numeric) - kept} of {len(numeric)} components are near-exact "
        f"combinations of the others (variance below 1/{an.CONDITION_HIGH}² of the first's, "
        f"κ > {an.CONDITION_HIGH}) and are dropped; parallel analysis (078) sees {real} above "
        "chance.",
        float(values[:kept].sum() / total) if total > 0 else 0.0,
        proposed=len(numeric) >= 3 and collinear(fit),
    )
    start = len(others)
    for c in range(kept):
        estimate, s = float(ls.beta[start + c]), float(se[start + c])
        p = float(2 * stats.t.sf(abs(estimate / s), df)) if s > 0 else None
        table.components.append(
            ComponentCoefficient(
                f"PC{c + 1}",
                pca.component_label(vectors[:, c], names),
                float(values[c] / total) if total > 0 else 0.0,
                estimate,
                s if s > 0 else None,
                p,
            )
        )
    # back to the columns' units: β = diag(1/sd) V γ
    a = vectors[:, :kept] / sd[:, None]
    gamma = ls.beta[start:]
    back = a @ gamma
    back_cov = a @ covariance[start:, start:] @ a.T
    se_ls = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
    for i, j in enumerate(numeric):
        s = math.sqrt(max(float(back_cov[i, i]), 0))
        ls_beta = float(fit.ls.beta[j])
        table.effects.append(
            ComponentEffect(
                fit.design.terms[j],
                float(back[i]),
                float(back[i] - crit * s) if s > 0 else None,
                float(back[i] + crit * s) if s > 0 else None,
                ls_beta,
                ls_beta - crit * float(se_ls[j]) if se_ls[j] > 0 else None,
                ls_beta + crit * float(se_ls[j]) if se_ls[j] > 0 else None,
            )
        )
    table.r2 = float(ls.r2) if math.isfinite(ls.r2) else None
    table.ls_r2 = float(fit.ls.r2) if math.isfinite(fit.ls.r2) else None
    table.sentence = _components_sentence(table)
    if others[1:]:
        table.notes.append("The other terms (categories, curves) stay in the model as they are.")
    if kept == len(numeric):
        table.notes.append("Every component is kept: the coefficients are those of least squares.")
    return table


def _components_sentence(table: ComponentsTable) -> str:
    flips = [
        e.term for e in table.effects if e.estimate * e.least_squares < 0 and e.least_squares != 0
    ]
    text = (
        f"{table.kept} of {len(table.columns)} components keep {table.cumulative:.0%} of the "
        f"variance of the columns; on them, R² {an.fmt(table.r2)} against {an.fmt(table.ls_r2)} "
        "with every column."
    )
    if flips:
        text += (
            f" The sign of {', '.join(f'`{f}`' for f in flips[:3])} differs from the first plane: "
            "collinear columns share their effect there."
        )
    return text
