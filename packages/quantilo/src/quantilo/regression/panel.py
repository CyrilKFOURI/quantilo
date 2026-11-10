"""Panel estimators of a regression (071): when the rows are entities seen over periods
(`ml/panel.py`), fixed effects by entity, by period or both (within), first differences, between,
random effects (Swamy–Arora GLS), the Mundlak test that chooses between fixed and random effects
(R-REG-PAN-01), the classic Hausman test next to it, and the R-REG-14 check of the first plane.
Every estimator with several rows per entity has standard errors clustered by entity
(R-REG-PAN-04). All numpy and scipy (sparse group means): Pyodide-compatible."""

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import sparse, stats

from quantilo.regression import covariance as cov
from quantilo.regression.analyze import Fitted, RegressionCheck, fmt, fmt_p
from quantilo.regression.design import INTERCEPT, LeftOutTerm, _estimable
from quantilo.regression.fit import LeastSquares, least_squares
from quantilo.regression.rules import FEW_CLUSTERS

EstimatorKey = Literal["pooled", "entity", "time", "twoway", "fd", "between", "re"]
LABELS: dict[EstimatorKey, str] = {
    "pooled": "Pooled OLS",
    "entity": "Fixed effects (entity)",
    "time": "Fixed effects (period)",
    "twoway": "Fixed effects (two-way)",
    "fd": "First differences",
    "between": "Between (entity means)",
    "re": "Random effects (GLS)",
}
ORDER: tuple[EstimatorKey, ...] = ("pooled", "entity", "time", "twoway", "fd", "between", "re")
CONSTANT_SHARE = 1e-10  # R-REG-14 / PAN-05: within variance below this share of the total
TOLERANCE = 1e-10  # alternating projections (two-way, unbalanced)
MAX_SWEEPS = 1_000
LEVEL = 0.05  # Mundlak, time effects
SERIAL = 0.8  # R-REG-PAN-03
MIN_PAIRS = 10


# --- Results -------------------------------------------------------------------------------------


@dataclass
class PanelCoefficient:
    estimate: float | None = None
    std_error: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None


@dataclass
class PanelEstimator:
    key: EstimatorKey
    label: str
    available: bool
    reason: str  # what it compares, or why it cannot be computed
    coefficients: list[PanelCoefficient] = field(default_factory=list)  # aligned with terms
    dropped: list[LeftOutTerm] = field(default_factory=list)
    rows: int = 0
    groups: int = 0
    r2: float | None = None
    r2_kind: str = ""
    se_label: str = ""
    theta: float | None = None  # random effects: the mean of θᵢ


@dataclass
class PanelTest:
    key: str
    title: str
    statistic: float | None
    df: int | None
    df2: int | None
    p_value: float | None
    sentence: str
    rule: str | None = None


@dataclass
class PanelTable:
    available: bool
    reason: str
    entity: str | None = None
    period: str | None = None
    entities: int = 0
    periods: int = 0
    balanced: bool = False
    terms: list[str] = field(default_factory=list)
    varying: list[str] = field(default_factory=list)  # columns that change within an entity
    fixed: list[str] = field(default_factory=list)  # … that do not
    estimators: list[PanelEstimator] = field(default_factory=list)
    recommended: EstimatorKey | None = None
    recommended_rule: str | None = None
    recommended_reason: str = ""
    tests: list[PanelTest] = field(default_factory=list)
    sentence: str = ""
    notes: list[str] = field(default_factory=list)


# --- Group means ---------------------------------------------------------------------------------


class Groups:
    """Rows → groups: means and demeaning through one sparse indicator matrix."""

    def __init__(self, codes: np.ndarray) -> None:
        self.codes, self.count = cov.codes_of(codes)
        n = len(self.codes)
        self.sizes = np.bincount(self.codes, minlength=self.count).astype(float)
        self.matrix = sparse.csr_matrix(
            (np.ones(n), (self.codes, np.arange(n))), shape=(self.count, n)
        )

    def means(self, a: np.ndarray) -> np.ndarray:
        flat = a.ndim == 1
        found = (self.matrix @ (a[:, None] if flat else a)) / self.sizes[:, None]
        return found[:, 0] if flat else found

    def demean(self, a: np.ndarray) -> np.ndarray:
        return a - self.means(a)[self.codes]


def twoway(a: np.ndarray, entity: Groups, period: Groups, balanced: bool) -> np.ndarray:
    """Both effects out: a closed form for a balanced panel (one row per cell), alternating
    projections otherwise (they converge to the projection on both sets of indicators)."""
    if balanced:
        return a - entity.means(a)[entity.codes] - period.means(a)[period.codes] + a.mean(axis=0)
    out = entity.demean(a)
    scale = max(float(np.abs(a).max()) if a.size else 1.0, 1.0)
    for _ in range(MAX_SWEEPS):
        new = entity.demean(period.demean(out))
        change = float(np.abs(new - out).max()) if new.size else 0.0
        out = new
        if change < TOLERANCE * scale:
            break
    return period.demean(out)


# --- A fit ---------------------------------------------------------------------------------------


@dataclass
class _Estimate:
    index: list[int]  # design terms estimated
    ls: LeastSquares
    robust: np.ndarray
    classic: np.ndarray
    df: int
    dropped: list[LeftOutTerm]
    se_label: str


def _estimate(
    x: np.ndarray,
    y: np.ndarray,
    index: list[int],
    terms: list[str],
    columns: list[str],
    *,
    cluster: np.ndarray | None,
    extra: int = 0,
    absorbed: int = 0,
    sigma2: float | None = None,
) -> _Estimate | None:
    """OLS of y on the columns of x (design terms `index`; term 0 = the intercept when present).
    `extra`: effects absorbed but not nested in the clusters (they count in k, R-REG-PAN-04);
    `absorbed`: every effect absorbed (they count in the classic variance)."""
    has_intercept = bool(index) and index[0] == 0
    full = x if has_intercept else np.column_stack([np.ones(len(y)), x])
    names = [terms[i] for i in index] if has_intercept else [INTERCEPT, *(terms[i] for i in index)]
    sources = (
        [columns[i] for i in index] if has_intercept else [INTERCEPT, *(columns[i] for i in index)]
    )
    keep, dropped = _estimable(full, names, sources)
    keep = [j - 1 for j in keep[1:]] if not has_intercept else list(keep)
    kept = [index[j] for j in keep]
    xk = x[:, keep]
    n, k = xk.shape
    if k == 0 or n - k - max(extra, absorbed) <= 0:
        return None
    ls = least_squares(xk, y)
    sse = float(ls.arrays.e @ ls.arrays.e)
    s2 = sigma2 if sigma2 is not None else sse / max(n - k - absorbed, 1)
    classic = s2 * ls.arrays.bread
    if cluster is not None:
        groups = len(np.unique(cluster))
        robust = cov.cluster(ls.arrays, cluster) * (n - k) / max(n - k - extra, 1)
        df, label = max(groups - 1, 1), "Clustered by entity"
    else:
        robust, df, label = cov.hc(ls.arrays, "hc3"), max(n - k, 1), "HC3"
    return _Estimate(kept, ls, robust, classic, df, dropped, label)


def _coefficients(found: _Estimate, size: int) -> list[PanelCoefficient]:
    out = [PanelCoefficient() for _ in range(size)]
    ses = np.sqrt(np.clip(np.diag(found.robust), 0, None))
    half = float(stats.t.ppf(0.975, found.df))
    for j, i in enumerate(found.index):
        b, se = float(found.ls.beta[j]), float(ses[j])
        if not math.isfinite(se) or se <= 0:
            out[i] = PanelCoefficient(b)
            continue
        p = float(2 * stats.t.sf(abs(b / se), found.df))
        out[i] = PanelCoefficient(b, se, b - half * se, b + half * se, p)
    return out


def _wald(beta: np.ndarray, v: np.ndarray) -> float | None:
    try:
        return float(beta @ np.linalg.solve(v, beta))
    except np.linalg.LinAlgError:
        return None


# --- The section ---------------------------------------------------------------------------------


def _unavailable(reason: str) -> PanelTable:
    return PanelTable(False, reason)


def _absorbed(
    x: np.ndarray, within: np.ndarray, names: list[str], columns: list[str], reason: str
) -> tuple[list[int], list[LeftOutTerm]]:
    """R-REG-PAN-05: terms whose within variance is (almost) nothing are absorbed by the effects."""
    total = x.var(axis=0)
    left = within.var(axis=0)
    keep, dropped = [], []
    for j in range(x.shape[1]):
        if total[j] > 0 and left[j] > CONSTANT_SHARE * total[j]:
            keep.append(j)
        else:
            dropped.append(LeftOutTerm(names[j], columns[j], reason))
    return keep, dropped


def varying_columns(fit: Fitted, entity: Groups) -> tuple[list[str], list[str]]:
    """Explanatory columns that change within an entity, and those that do not."""
    design = fit.design
    x = design.x[:, 1:]
    within = entity.demean(x)
    total, left = x.var(axis=0), within.var(axis=0)
    varying: dict[str, bool] = {}
    for j, column in enumerate(design.columns[1:]):
        moves = bool(total[j] > 0 and left[j] > CONSTANT_SHARE * total[j])
        varying[column] = varying.get(column, False) or moves
    return [c for c, v in varying.items() if v], [c for c, v in varying.items() if not v]


def _names(columns: list[str], limit: int = 3) -> str:
    shown = [f"`{c}`" for c in columns[:limit]]
    more = len(columns) - limit
    if more > 0:
        return ", ".join(shown) + f" and {more} more"
    if len(shown) > 1:
        return ", ".join(shown[:-1]) + f" and {shown[-1]}"
    return shown[0] if shown else ""


def proposal(fit: Fitted) -> RegressionCheck | None:
    """R-REG-14: a panel whose columns change within an entity → the Panel section."""
    shape = fit.inputs.shape
    if shape.kind != "panel" or fit.ls is None or fit.grouping.entity_codes is None:
        return None
    if fit.design.x.shape[1] < 2:
        return None
    entity = Groups(fit.grouping.entity_codes)
    varying, _ = varying_columns(fit, entity)
    who = f"`{shape.entity}`"
    if varying:
        text = (
            f"{_names(varying)} change within a {who}: fixed effects compare each {who} with "
            f"itself, setting aside what makes one {who} differ from another. Open Panel to "
            "compare fixed and random effects (Mundlak test)."
        )
    else:
        text = (
            f"No explanatory column changes within a {who}: fixed effects would have nothing to "
            "estimate; between and random effects remain in Panel."
        )
    return RegressionCheck("panel", "info", "Panel data", text, "R-REG-14", "panel")


def section(fit: Fitted) -> PanelTable:
    shape = fit.inputs.shape
    if shape.kind != "panel" or shape.entity is None or shape.period is None:
        return _unavailable(
            "Needs panel data: an entity (store, customer, ticker) seen over several periods. "
            "None was recognised in these rows."
        )
    if fit.ls is None:
        return _unavailable(
            "Panel estimators are linear here: not for a yes/no target (R-REG-PAN-08)."
        )
    if fit.grouping.entity_codes is None or fit.grouping.period_codes is None:
        return _unavailable("The entity and period of the rows are missing.")
    return _section(fit)


def _section(fit: Fitted) -> PanelTable:
    inputs, design = fit.inputs, fit.design
    shape = inputs.shape
    assert fit.ls is not None and fit.grouping.entity_codes is not None
    assert fit.grouping.period_codes is not None and shape.entity and shape.period
    who, when = f"`{shape.entity}`", f"`{shape.period}`"
    entity = Groups(fit.grouping.entity_codes)
    period = Groups(fit.grouping.period_codes)
    y, x_full = design.y, design.x
    n, k_full = x_full.shape
    terms, columns = design.terms, design.columns
    size = len(terms)
    x = x_full[:, 1:]
    slopes = list(range(1, k_full))
    cells = len(np.unique(entity.codes * period.count + period.codes))
    single = cells == n
    balanced = single and n == entity.count * period.count
    varying, fixed = varying_columns(fit, entity)
    notes: list[str] = []
    if fit.ls.weights is not None:
        notes.append(
            "The panel estimators are unweighted: demeaning and weights do not mix "
            "(R-REG-PAN-08); the pooled column keeps the weights."
        )
    if entity.count < FEW_CLUSTERS:
        notes.append(
            f"Only {entity.count} {who}: clustered intervals are too narrow below "
            f"{FEW_CLUSTERS} groups (R-REG-SE-03)."
        )
    cluster = entity.codes
    estimators: dict[EstimatorKey, PanelEstimator] = {}
    fits: dict[EstimatorKey, _Estimate] = {}

    # Pooled: the first-plane fit, its own standard errors.
    pooled_se = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
    half = float(stats.t.ppf(0.975, fit.df))
    pooled = []
    for j in range(size):
        b, se = float(fit.ls.beta[j]), float(pooled_se[j])
        p = float(2 * stats.t.sf(abs(b / se), fit.df)) if se > 0 else None
        pooled.append(PanelCoefficient(b, se, b - half * se, b + half * se, p))
    estimators["pooled"] = PanelEstimator(
        "pooled",
        LABELS["pooled"],
        True,
        f"The first-plane fit: every row counts alike, differences between and within {who} mixed.",
        pooled,
        [],
        n,
        entity.count,
        fit.ls.r2,
        "R²",
        fit.se.label,
    )

    absorbed_by_entity = f"does not change within a {who}: absorbed by the {who} effects"
    absorbed_by_period = f"does not change within a {when}: absorbed by the {when} effects"

    # Fixed effects by entity.
    xe, ye = entity.demean(x), entity.demean(y)
    found, dropped = _within(
        x, xe, ye, terms, columns, absorbed_by_entity, cluster, extra=1, absorbed=entity.count
    )
    _store(
        estimators,
        fits,
        "entity",
        found,
        dropped,
        size,
        n,
        entity.count,
        f"Compares each {who} with itself over time: what makes {who} differ is set aside.",
        "within R²",
    )
    # Fixed effects by period.
    xt, yt = period.demean(x), period.demean(y)
    found, dropped = _within(
        x, xt, yt, terms, columns, absorbed_by_period, cluster, period.count, period.count
    )
    _store(
        estimators,
        fits,
        "time",
        found,
        dropped,
        size,
        n,
        entity.count,
        f"Compares the {who} within each {when}: shocks common to a {when} are set aside.",
        "within R²",
    )
    # Two-way.
    x2, y2 = twoway(x, entity, period, balanced), twoway(y, entity, period, balanced)
    reasons: list[str] = []
    for j in range(x.shape[1]):
        column = x[:, j]
        if entity.demean(column).var() <= CONSTANT_SHARE * max(column.var(), 1e-300):
            reasons.append(absorbed_by_entity)
        elif period.demean(column).var() <= CONSTANT_SHARE * max(column.var(), 1e-300):
            reasons.append(absorbed_by_period)
        else:
            reasons.append(f"moves only with the {who} and {when} effects: absorbed")
    keep, dropped = [], []
    total, left = x.var(axis=0), x2.var(axis=0)
    for j in range(x.shape[1]):
        if total[j] > 0 and left[j] > CONSTANT_SHARE * total[j]:
            keep.append(j)
        else:
            dropped.append(LeftOutTerm(terms[j + 1], columns[j + 1], reasons[j]))
    found = _estimate(
        x2[:, keep],
        y2,
        [slopes[j] for j in keep],
        terms,
        columns,
        cluster=cluster,
        extra=period.count,
        absorbed=entity.count + period.count - 1,
    )
    _store(
        estimators,
        fits,
        "twoway",
        found,
        dropped + (found.dropped if found else []),
        size,
        n,
        entity.count,
        f"Both at once: each {who} against itself, net of what every {who} shared that {when}.",
        "within R²",
    )
    # First differences.
    ranks = inputs.period_order[design.used] if inputs.period_order is not None else None
    if not single:
        estimators["fd"] = PanelEstimator(
            "fd",
            LABELS["fd"],
            False,
            f"Several rows share a {who} and a {when}: there is no single previous row to "
            "subtract (R-REG-PAN-06).",
        )
    elif ranks is None:
        estimators["fd"] = PanelEstimator(
            "fd", LABELS["fd"], False, f"The order of the {when} values is unknown."
        )
    else:
        order = np.lexsort((ranks, entity.codes))
        e_sorted, r_sorted = entity.codes[order], ranks[order]
        follows = (e_sorted[1:] == e_sorted[:-1]) & (r_sorted[1:] - r_sorted[:-1] == 1)
        after, before = order[1:][follows], order[:-1][follows]
        dx, dy = x_full[after] - x_full[before], y[after] - y[before]
        dx[:, 0] = 1.0  # a constant: the common trend
        lost = n - len(after)
        keep, dropped = _absorbed(x, dx[:, 1:], terms[1:], columns[1:], absorbed_by_entity)
        found = (
            _estimate(
                dx[:, [0, *(j + 1 for j in keep)]],
                dy,
                [0, *(slopes[j] for j in keep)],
                terms,
                columns,
                cluster=entity.codes[after],
            )
            if len(after) > 2
            else None
        )
        _store(
            estimators,
            fits,
            "fd",
            found,
            dropped + (found.dropped if found else []),
            size,
            len(after),
            entity.count,
            f"Change from one {when} to the next within a {who}; {lost:,} rows without the "
            "previous period are lost (R-REG-PAN-06). The constant is the common trend.",
            "R² of the changes",
        )
    # Between.
    mx, my = entity.means(x_full), entity.means(y)
    found = (
        _estimate(mx, my, list(range(k_full)), terms, columns, cluster=None)
        if entity.count > k_full + 1
        else None
    )
    _store(
        estimators,
        fits,
        "between",
        found,
        found.dropped if found else [],
        size,
        entity.count,
        entity.count,
        f"One row per {who} (its means): how {who} differ, not how one changes. HC3, the "
        f"{who} being independent rows.",
        "R² of the means",
    )
    # Random effects (Swamy–Arora).
    re_found, theta, re_note = _random_effects(
        y, x_full, terms, columns, entity, fits.get("entity"), fits.get("between")
    )
    if re_note:
        notes.append(re_note)
    if re_found is not None:
        r2 = float(np.corrcoef(y, x_full[:, re_found.index] @ re_found.ls.beta)[0, 1] ** 2)
        estimators["re"] = PanelEstimator(
            "re",
            LABELS["re"],
            True,
            f"A weighted mix of within and between (θ = {theta:.2f}, 1 = fixed effects): "
            f"keeps the columns that do not change within a {who}, but is biased if the {who} "
            "effect is tied to the columns.",
            _coefficients(re_found, size),
            re_found.dropped,
            n,
            entity.count,
            r2,
            "overall R²",
            re_found.se_label,
            theta,
        )
        fits["re"] = re_found
    else:
        estimators["re"] = PanelEstimator(
            "re", LABELS["re"], False, "Needs the within and between fits."
        )

    tests: list[PanelTest] = []
    mundlak = _mundlak(y, x_full, terms, columns, entity, who)
    if mundlak is not None:
        tests.append(mundlak)
    hausman = _hausman(fits.get("entity"), fits.get("re"))
    if hausman is not None:
        tests.append(hausman)
    time_test = _time_effects(fits.get("entity"), fits.get("twoway"), n, entity, period, when)
    if time_test is not None:
        tests.append(time_test)
    serial = _serial(fits.get("entity"), entity, ranks, xe, ye, who)
    if serial is not None:
        tests.append(serial)

    recommended, rule, reason = _recommend(
        estimators, mundlak, time_test, varying, fixed, who, when
    )
    if serial is not None and serial.statistic is not None and serial.statistic > SERIAL:
        reason += (
            f" The within errors are close to a random walk (ρ = {serial.statistic:.2f}): "
            "first differences are worth comparing (R-REG-PAN-03)."
        )
    sentence = _conclusion(fit, estimators, recommended, reason, terms, columns)
    return PanelTable(
        True,
        "",
        shape.entity,
        shape.period,
        entity.count,
        period.count,
        balanced,
        list(terms),
        varying,
        fixed,
        [estimators[key] for key in ORDER if key in estimators],
        recommended,
        rule,
        reason,
        tests,
        sentence,
        notes,
    )


def _within(
    x: np.ndarray,
    transformed: np.ndarray,
    y: np.ndarray,
    terms: list[str],
    columns: list[str],
    reason: str,
    cluster: np.ndarray,
    extra: int,
    absorbed: int,
) -> tuple[_Estimate | None, list[LeftOutTerm]]:
    keep, dropped = _absorbed(x, transformed, terms[1:], columns[1:], reason)
    found = _estimate(
        transformed[:, keep],
        y,
        [j + 1 for j in keep],
        terms,
        columns,
        cluster=cluster,
        extra=extra,
        absorbed=absorbed,
    )
    if found is None:
        return None, dropped
    return found, dropped + found.dropped


def _store(
    estimators: dict[EstimatorKey, PanelEstimator],
    fits: dict[EstimatorKey, _Estimate],
    key: EstimatorKey,
    found: _Estimate | None,
    dropped: list[LeftOutTerm],
    size: int,
    rows: int,
    groups: int,
    reason: str,
    r2_kind: str,
) -> None:
    if found is None:
        why = "Nothing left to estimate"
        if dropped:
            why += ": " + "; ".join(f"`{d.term}` {d.reason}" for d in dropped[:3])
        estimators[key] = PanelEstimator(key, LABELS[key], False, why + ".", dropped=dropped)
        return
    fits[key] = found
    estimators[key] = PanelEstimator(
        key,
        LABELS[key],
        True,
        reason,
        _coefficients(found, size),
        dropped,
        rows,
        groups,
        found.ls.r2,
        r2_kind,
        found.se_label,
    )


def _random_effects(
    y: np.ndarray,
    x_full: np.ndarray,
    terms: list[str],
    columns: list[str],
    entity: Groups,
    within: _Estimate | None,
    between: _Estimate | None,
) -> tuple[_Estimate | None, float, str | None]:
    """R-REG-PAN-07: Swamy–Arora variances, θᵢ = 1 − √(σ²ₑ / (Tᵢ σ²ᵤ + σ²ₑ))."""
    if between is None:
        return None, 0.0, None
    n, g = len(y), entity.count
    if within is not None:
        sse_w = float(within.ls.arrays.e @ within.ls.arrays.e)
        sigma_e = sse_w / max(n - g - within.ls.k, 1)
    else:  # nothing changes within an entity: the within residuals are y's deviations
        demeaned = entity.demean(y)
        sigma_e = float(demeaned @ demeaned) / max(n - g, 1)
    sse_b = float(between.ls.arrays.e @ between.ls.arrays.e)
    sigma_b = sse_b / max(g - between.ls.k, 1)
    harmonic = g / float((1 / entity.sizes).sum())
    sigma_u = sigma_b - sigma_e / harmonic
    note = None
    if sigma_u <= 0:
        sigma_u = 0.0
        note = (
            "The entity effects' variance comes out below 0: random effects equal pooled OLS "
            "here (R-REG-PAN-07)."
        )
    theta = 1 - np.sqrt(sigma_e / (entity.sizes * sigma_u + sigma_e))
    shrink = theta[entity.codes]
    xs = x_full - shrink[:, None] * entity.means(x_full)[entity.codes]
    ys = y - shrink * entity.means(y)[entity.codes]
    found = _estimate(
        xs,
        ys,
        list(range(x_full.shape[1])),
        terms,
        columns,
        cluster=entity.codes,
        sigma2=sigma_e,
    )
    return found, float(theta.mean()), note


def _mundlak(
    y: np.ndarray,
    x_full: np.ndarray,
    terms: list[str],
    columns: list[str],
    entity: Groups,
    who: str,
) -> PanelTest | None:
    """R-REG-PAN-01: pooled OLS on X and the entity means of the columns that change within an
    entity; a clustered Wald test that the means add nothing."""
    x = x_full[:, 1:]
    means = entity.means(x)[entity.codes]
    within = x - means
    moves = [
        j for j in range(x.shape[1]) if within[:, j].var() > CONSTANT_SHARE * max(x[:, j].var(), 0)
    ]
    if not moves:
        return None
    augmented = np.column_stack([x_full, means[:, moves]])
    names = [*terms, *(f"mean of {terms[j + 1]}" for j in moves)]
    sources = [*columns, *(columns[j + 1] for j in moves)]
    keep, _ = _estimable(augmented, names, sources)
    tested = [i for i, j in enumerate(keep) if j >= x_full.shape[1]]
    if not tested:
        return None
    ls = least_squares(augmented[:, keep], y)
    v = cov.cluster(ls.arrays, entity.codes)
    q = len(tested)
    wald = _wald(ls.beta[tested], v[np.ix_(tested, tested)])
    if wald is None:
        return None
    f = wald / q
    df2 = max(entity.count - 1, 1)
    p = float(stats.f.sf(f, q, df2))
    if p < LEVEL:
        text = (
            f"The {who} effect is tied to the columns (Mundlak {fmt_p(p)}): random effects "
            "would be biased; fixed effects stay unbiased."
        )
    else:
        text = (
            f"No sign that the {who} effect is tied to the columns (Mundlak {fmt_p(p)}): random "
            "effects are unbiased and more precise."
        )
    return PanelTest("mundlak", "Mundlak (robust Hausman)", f, q, df2, p, text, "R-REG-PAN-01")


def _hausman(fe: _Estimate | None, re: _Estimate | None) -> PanelTest | None:
    """Classic Hausman on the slopes both estimate; informs, does not decide (R-REG-PAN-01)."""
    if fe is None or re is None:
        return None
    common = [i for i in fe.index if i in re.index and i != 0]
    if not common:
        return None
    a = [fe.index.index(i) for i in common]
    b = [re.index.index(i) for i in common]
    d = fe.ls.beta[a] - re.ls.beta[b]
    v = fe.classic[np.ix_(a, a)] - re.classic[np.ix_(b, b)]
    v = (v + v.T) / 2
    values, vectors = np.linalg.eigh(v)
    top = float(np.abs(values).max()) if values.size else 0.0
    positive = values > 1e-10 * max(top, 1e-300)
    definite = bool(values.min() > -1e-8 * max(top, 1e-300))
    if not positive.any():
        return None
    projected = vectors[:, positive].T @ d
    h = float((projected**2 / values[positive]).sum())
    df = int(positive.sum())
    p = float(stats.chi2.sf(h, df))
    verdict = "differ" if p < LEVEL else "agree"
    text = (
        f"Fixed and random effects {verdict} (classic Hausman χ² = {fmt(h)}, {fmt_p(p)}). It "
        "assumes independent, equal-variance errors, so it informs; Mundlak decides."
    )
    if not definite:
        text += " Its variance difference is not positive definite here: read Mundlak."
    return PanelTest("hausman", "Hausman (classic)", h, df, None, p, text)


def _time_effects(
    fe: _Estimate | None,
    two: _Estimate | None,
    n: int,
    entity: Groups,
    period: Groups,
    when: str,
) -> PanelTest | None:
    """R-REG-PAN-02: classic F of the period indicators, two-way against entity effects."""
    if fe is None or two is None or period.count < 2:
        return None
    sse_1 = float(fe.ls.arrays.e @ fe.ls.arrays.e)
    sse_2 = float(two.ls.arrays.e @ two.ls.arrays.e)
    df1 = period.count - 1 - (fe.ls.k - two.ls.k)
    df2 = n - entity.count - (period.count - 1) - two.ls.k
    if df1 <= 0 or df2 <= 0 or sse_2 <= 0:
        return None
    f = max((sse_1 - sse_2) / df1, 0.0) / (sse_2 / df2)
    p = float(stats.f.sf(f, df1, df2))
    if p < LEVEL:
        text = (
            f"Shocks common to a {when} matter (F = {fmt(f)}, {fmt_p(p)}): two-way fixed effects "
            "keep them out of the coefficients."
        )
    else:
        text = f"No clear shock common to a {when} (F = {fmt(f)}, {fmt_p(p)})."
    return PanelTest("time", "Period effects", f, df1, df2, p, text, "R-REG-PAN-02")


def _serial(
    fe: _Estimate | None,
    entity: Groups,
    ranks: np.ndarray | None,
    xe: np.ndarray,
    ye: np.ndarray,
    who: str,
) -> PanelTest | None:
    """R-REG-PAN-03: AR(1) of the within residuals over consecutive periods of an entity."""
    if fe is None or ranks is None:
        return None
    residuals = fe.ls.arrays.e
    order = np.lexsort((ranks, entity.codes))
    e_sorted, r_sorted = entity.codes[order], ranks[order]
    follows = (e_sorted[1:] == e_sorted[:-1]) & (r_sorted[1:] - r_sorted[:-1] == 1)
    if int(follows.sum()) < MIN_PAIRS:
        return None
    now, before = residuals[order[1:][follows]], residuals[order[:-1][follows]]
    denominator = float(before @ before)
    if denominator <= 0:
        return None
    rho = float(now @ before) / denominator
    if rho > SERIAL:
        text = (
            f"Within a {who}, one period's error carries into the next (ρ = {rho:.2f}): close to "
            "a random walk, where first differences are more efficient (R-REG-PAN-03). "
            "Clustered standard errors stay valid either way."
        )
    else:
        text = (
            f"Within a {who}, errors carry over moderately from one period to the next "
            f"(ρ = {rho:.2f}); clustered standard errors allow for it."
        )
    return PanelTest("serial", "Serial correlation", rho, None, None, None, text, "R-REG-PAN-03")


def _recommend(
    estimators: dict[EstimatorKey, PanelEstimator],
    mundlak: PanelTest | None,
    time_test: PanelTest | None,
    varying: list[str],
    fixed: list[str],
    who: str,
    when: str,
) -> tuple[EstimatorKey | None, str | None, str]:
    def ok(key: EstimatorKey) -> bool:
        return key in estimators and estimators[key].available

    if not varying:
        if ok("re"):
            return (
                "re",
                "R-REG-14",
                f"No column changes within a {who}: fixed effects have nothing to estimate; "
                "random effects use both kinds of differences.",
            )
        return "pooled", "R-REG-14", f"No column changes within a {who}."
    time_matters = (
        time_test is not None and time_test.p_value is not None and time_test.p_value < LEVEL
    )
    if mundlak is None or mundlak.p_value is None or mundlak.p_value < LEVEL:
        why = (
            f"Random effects would be biased: the {who} effect is tied to the columns "
            f"(Mundlak {fmt_p(mundlak.p_value)})."
            if mundlak is not None and mundlak.p_value is not None
            else "The Mundlak test cannot be computed: fixed effects are the safe choice."
        )
        if time_matters and ok("twoway"):
            assert time_test is not None
            return (
                "twoway",
                "R-REG-PAN-01, R-REG-PAN-02",
                f"{why} Shocks common to a {when} matter too ({fmt_p(time_test.p_value)}): "
                "two-way fixed effects.",
            )
        if ok("entity"):
            return "entity", "R-REG-PAN-01", f"{why} Fixed effects by {who}."
        return "pooled", "R-REG-PAN-01", why
    if ok("re"):
        why = (
            f"No sign that the {who} effect is tied to the columns (Mundlak "
            f"{fmt_p(mundlak.p_value)}): random effects are unbiased and more precise than "
            "fixed effects"
        )
        why += f", and keep {_names(fixed)}, which fixed effects absorb." if fixed else "."
        if time_matters:
            why += f" Shocks common to a {when} matter: compare with two-way fixed effects."
        return "re", "R-REG-PAN-01", why
    return "entity", "R-REG-PAN-01", f"Fixed effects by {who}."


def _conclusion(
    fit: Fitted,
    estimators: dict[EstimatorKey, PanelEstimator],
    recommended: EstimatorKey | None,
    reason: str,
    terms: list[str],
    columns: list[str],
) -> str:
    if recommended is None:
        return reason
    chosen = estimators[recommended]
    pooled = estimators["pooled"]
    candidates = [
        i
        for i in range(1, len(terms))
        if chosen.coefficients[i].estimate is not None
        and chosen.coefficients[i].std_error is not None
    ]
    focus = fit.inputs.focus
    pick = next((i for i in candidates if columns[i] == focus), None)
    if pick is None and candidates:
        pick = max(
            candidates,
            key=lambda i: abs(
                (chosen.coefficients[i].estimate or 0) / (chosen.coefficients[i].std_error or 1)
            ),
        )
    lead = f"{chosen.label} recommended. {reason}"
    if pick is None:
        return lead
    c, base = chosen.coefficients[pick], pooled.coefficients[pick]
    return (
        f"{lead} With it, `{terms[pick]}` moves `{fit.inputs.target}` by {fmt(c.estimate)} "
        f"[{fmt(c.ci_low)} ; {fmt(c.ci_high)}] per unit (pooled OLS: {fmt(base.estimate)})."
    )
