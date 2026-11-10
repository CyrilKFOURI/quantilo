"""A regression from loaded arrays to its result (067): the fit, the standard errors chosen by
rule, coefficients with intervals, adjusted p-values and evidence, quality, checks in sentences
and the residual plots; then the Advanced sections, computed from the same fit when opened."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Literal, cast

import numpy as np
from scipy import stats

from quantilo.compare.multiple import MultipleTesting, correct
from quantilo.correlations.evidence import Evidence, effect_size, grade
from quantilo.regression import covariance as cov
from quantilo.regression import diagnostics as dg
from quantilo.regression.covariance import LABELS, Kind
from quantilo.regression.design import (
    INTERCEPT,
    Design,
    LeftOutTerm,
    MissingRows,
    ReferenceLevel,
    Variable,
    build,
)
from quantilo.regression.fit import (
    LeastSquares,
    Logistic,
    least_squares,
    logistic,
    logistic_predict,
    lstsq_predict,
)
from quantilo.regression.influence import Influence, dfbetas, influence, top_rows
from quantilo.regression.rules import (
    SE_KINDS,
    Grouping,
    RowStructure,
    SeChoice,
    SeRequest,
    available,
    choose,
    two_way_hint,
)

Model = Literal["ols", "wls", "logit"]
Status = Literal["ok", "warn", "bad", "info"]
MODEL_LABELS: dict[Model, str] = {
    "ols": "Least squares (OLS)",
    "wls": "Weighted least squares (WLS)",
    "logit": "Logistic",
}
FOLDS = 5
SEED = 42
PLOT_POINTS = 5_000  # R-REGUI-04
QQ_POINTS = 400
VIF_HIGH = 10
CONDITION_HIGH = 30
OVERFIT = 0.1  # R-REG-25
INFLUENTIAL_SHARE = 0.01  # R-REG-11
SPEC_P = 0.01
LARGE_N = 500  # normality matters little for the coefficients beyond


# --- Inputs and results --------------------------------------------------------------------------


@dataclass
class Inputs:
    target: str
    y: np.ndarray  # floats with NaN; a yes/no target as 0/1
    binary: tuple[str, str] | None  # (reference, positive) labels of a yes/no target
    variables: list[Variable]
    refused: list[LeftOutTerm]  # columns that cannot enter the design, with the reason
    rows: np.ndarray  # the file row number of every row read
    loaded: int
    total: int
    sampled: bool
    shape: RowStructure = field(default_factory=RowStructure)
    weights: str | None = None
    weight_values: np.ndarray | None = None
    cluster: str | None = None
    cluster_codes: np.ndarray | None = None  # −1 = missing
    entity_codes: np.ndarray | None = None
    period_codes: np.ndarray | None = None
    requested: SeRequest = "auto"
    focus: str | None = None
    weights_reason: str | None = None
    period_order: np.ndarray | None = None  # 071: each row's period rank (first differences)


@dataclass
class RegressionCoefficient:
    term: str
    column: str
    level: str | None
    estimate: float  # per unit (log-odds for the logistic model)
    std_error: float | None
    ci_low: float | None
    ci_high: float | None
    statistic: float | None
    p_value: float | None
    q_value: float | None = None  # adjusted over the coefficients (051), intercept left out
    evidence: Evidence = "none"
    standardized: float | None = None  # per standard deviation of the column (and of the target)
    std_low: float | None = None
    std_high: float | None = None
    odds_ratio: float | None = None  # logistic
    marginal: float | None = None  # logistic: average marginal effect, in probability
    marginal_low: float | None = None
    marginal_high: float | None = None
    sentence: str = ""


@dataclass
class RegressionQuality:
    rows: int  # rows used
    loaded: int  # rows read (a sample beyond 200 000)
    total: int  # rows after the global filters
    sampled: bool
    missing: list[MissingRows]
    r2: float | None
    adj_r2: float | None
    oos_r2: float | None
    oos_how: str
    aic: float | None
    bic: float | None
    f: float | None
    f_p: float | None
    df_model: int
    df_resid: int
    pseudo_r2: float | None = None  # logistic: McFadden


@dataclass
class RegressionCheck:
    key: str
    status: Status
    title: str
    sentence: str
    rule: str | None = None
    action: str | None = None  # an Advanced section, "compare" or "correlations"
    column: str | None = None  # the column the action is about


@dataclass
class ResidualPlots:
    fitted: list[float]
    residuals: list[float]
    rows: list[int]
    leverage: list[float]
    cook: list[float]
    cook_threshold: float
    qq_theoretical: list[float]
    qq_sample: list[float]
    points: int  # drawn
    of: int  # rows used


@dataclass
class RegressionReport:
    target: str
    positive: str | None  # the yes level of a yes/no target
    model: Model
    model_label: str
    model_reason: str
    columns: list[str]  # explanatory columns asked for
    focus: str | None
    terms: list[RegressionCoefficient]
    se: SeChoice
    weights: str | None
    quality: RegressionQuality
    multiple: MultipleTesting
    sentence: str
    checks: list[RegressionCheck]
    references: list[ReferenceLevel]
    left_out: list[LeftOutTerm]
    structure: RowStructure
    hints: list[str] = field(default_factory=list)
    plots: ResidualPlots | None = None


# --- Sections (Advanced) -------------------------------------------------------------------------


@dataclass
class SeColumn:
    kind: Kind
    label: str
    available: bool
    reason: str  # why not, or what it allows for
    chosen: bool
    std_errors: list[float | None]
    p_values: list[float | None]


@dataclass
class StandardErrorTable:
    terms: list[str]
    estimates: list[float]
    kinds: list[SeColumn]
    sentence: str


@dataclass
class InfluentialRow:
    row: int  # file row number (Data page)
    leverage: float
    cook: float
    studentized: float
    bonferroni: float
    dfbetas: list[float]


@dataclass
class InfluenceTable:
    terms: list[str]
    rows: list[InfluentialRow]
    influential: int
    outliers: int
    cook_threshold: float
    sentence: str


@dataclass
class TermsTable:
    references: list[ReferenceLevel]
    left_out: list[LeftOutTerm]
    vifs: list[dg.Vif]
    condition: float | None
    sentence: str


@dataclass
class CoefficientShift:
    term: str
    estimate: float
    without: float | None
    change: float | None  # relative change


@dataclass
class MoreChecks:
    white: dg.DiagnosticTest | None
    white_sentence: str
    removed: int
    shifts: list[CoefficientShift]
    shift_sentence: str


# --- Formatting ----------------------------------------------------------------------------------


def finite[T](found: T) -> T:
    """NaN and infinities become None, everywhere in a result (JSON has no NaN)."""
    if is_dataclass(found) and not isinstance(found, type):
        for item in fields(found):
            setattr(found, item.name, finite(getattr(found, item.name)))
        return found
    if isinstance(found, list):
        return cast(T, [finite(item) for item in found])
    if isinstance(found, float) and not math.isfinite(found):
        return cast(T, None)
    return found


def fmt(value: float | None) -> str:
    """Three significant digits, thousands separated, no exponent between 1e-4 and 1e15."""
    if value is None or not math.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-4 or magnitude >= 1e15:
        return f"{value:.2e}"
    decimals = max(0, 2 - math.floor(math.log10(magnitude)))
    return f"{value:,.{decimals}f}".replace("-", "−")


def fmt_p(p: float | None) -> str:
    if p is None:
        return "p = —"
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}"


def in_text(label: str) -> str:
    """A label inside a sentence: "clustered"; HC3, HAC and names keep their capitals."""
    return label.lower() if label in ("Classic", "Clustered", "Two-way clustered") else label


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}{'' if count == 1 else 's'}"


# --- The fit -------------------------------------------------------------------------------------


@dataclass
class Fitted:
    """Everything the sections need, kept by the service between requests."""

    inputs: Inputs
    design: Design
    model: Model
    ls: LeastSquares | None
    logit: Logistic | None
    grouping: Grouping
    se: SeChoice
    df: int  # degrees of freedom of the t intervals
    covariance: np.ndarray
    found: Influence | None
    result: RegressionReport = field(init=False)


def _used(values: np.ndarray | None, used: np.ndarray) -> np.ndarray | None:
    return values[used] if values is not None else None


def _df(choice: SeChoice, n: int, k: int) -> int:
    if choice.kind in ("cluster", "cluster2", "dk") and choice.clusters:
        return max(choice.clusters - 1, 1)
    return max(n - k, 1)


def covariance_of(
    kind: Kind, ls: LeastSquares, grouping: Grouping, shape: RowStructure
) -> np.ndarray:
    a = ls.arrays
    if kind == "classic":
        return cov.classic(a)
    if kind in ("hc0", "hc1", "hc2", "hc3"):
        return cov.hc(a, kind)
    if kind == "hac":
        return cov.hac(a, cov.newey_west_lag(a.n))
    if kind == "cluster":
        codes = (
            grouping.cluster_codes if grouping.cluster_codes is not None else grouping.entity_codes
        )
        assert codes is not None
        return cov.cluster(a, codes)
    assert grouping.entity_codes is not None and grouping.period_codes is not None
    if kind == "cluster2":
        return cov.cluster2(a, grouping.entity_codes, grouping.period_codes)
    periods = len(np.unique(grouping.period_codes))
    return cov.driscoll_kraay(a, grouping.period_codes, cov.newey_west_lag(periods))


def _logit_cov(kind: Kind, grouping: Grouping) -> tuple[str, Mapping[str, object]]:
    if kind == "classic":
        return "nonrobust", {}
    if kind == "hac":
        return "HAC", {}
    if kind == "cluster":
        codes = (
            grouping.cluster_codes if grouping.cluster_codes is not None else grouping.entity_codes
        )
        return "cluster", {"groups": codes}
    return "HC0", {}


def run(inputs: Inputs) -> Fitted:
    extra: dict[str, np.ndarray] = {}
    weights = inputs.weight_values
    if weights is not None and inputs.weights:
        weights = np.where(weights > 0, weights, np.nan)
        extra[inputs.weights] = weights
    if inputs.cluster_codes is not None and inputs.cluster:
        extra[inputs.cluster] = inputs.cluster_codes
    if inputs.shape.kind == "panel":
        if inputs.entity_codes is not None and inputs.shape.entity:
            extra[inputs.shape.entity] = inputs.entity_codes
        if inputs.period_codes is not None and inputs.shape.period:
            extra[inputs.shape.period] = inputs.period_codes
    design = build(inputs.y, inputs.variables, extra)
    for item in design.missing:
        if item.column == "target":
            item.column = inputs.target
    used = design.used
    grouping = Grouping(
        inputs.cluster,
        _used(inputs.cluster_codes, used),
        _used(inputs.entity_codes, used) if inputs.shape.kind == "panel" else None,
        _used(inputs.period_codes, used) if inputs.shape.kind == "panel" else None,
    )
    w = _used(weights, used) if inputs.weights else None
    n, k = design.x.shape
    choice = choose(inputs.requested, inputs.shape, grouping, n)
    if inputs.binary is not None:
        if choice.kind not in ("classic", "hac", "cluster"):
            # HC1–3 need the hat matrix of least squares: the logistic model has its own sandwich
            choice = replace(
                choice,
                kind="hc0",
                label=LABELS["hc0"],
                reason=choice.reason.replace("HC3", "a robust sandwich")
                + " The logistic model uses HC0.",
            )
        cov_type, kwds = _logit_cov(choice.kind, grouping)
        if choice.kind == "hac":
            kwds = {"maxlags": choice.lag or cov.newey_west_lag(n)}
        found = logistic(design.x, design.y, cov_type, kwds)
        fitted = Fitted(
            inputs,
            design,
            "logit",
            None,
            found,
            grouping,
            choice,
            _df(choice, n, k),
            found.cov,
            None,
        )
    else:
        ls = least_squares(design.x, design.y, w)
        matrix = covariance_of(choice.kind, ls, grouping, inputs.shape)
        model: Model = "wls" if w is not None else "ols"
        fitted = Fitted(
            inputs,
            design,
            model,
            ls,
            None,
            grouping,
            choice,
            _df(choice, n, k),
            matrix,
            influence(ls.arrays),
        )
    fitted.result = finite(_result(fitted))
    return fitted


# --- Coefficients --------------------------------------------------------------------------------


def _interval(
    estimate: float, se: float | None, df: int
) -> tuple[float | None, float | None, float | None, float | None]:
    if se is None or not math.isfinite(se) or se <= 0:
        return None, None, None, None
    t = estimate / se
    p = float(2 * stats.t.sf(abs(t), df))
    half = float(stats.t.ppf(0.975, df)) * se
    return estimate - half, estimate + half, t, p


def _terms(fit: Fitted) -> list[RegressionCoefficient]:
    design, df = fit.design, fit.df
    beta = fit.logit.beta if fit.logit is not None else fit.ls.beta  # type: ignore[union-attr]
    ses = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
    sd_x = design.x.std(axis=0)
    if fit.logit is not None:
        mean = float(design.y.mean())
        sd_y = math.sqrt(mean * (1 - mean)) or 1.0
    else:
        sd_y = float(design.y.std()) or 1.0
    terms: list[RegressionCoefficient] = []
    for j, name in enumerate(design.terms):
        b = float(beta[j])
        se = float(ses[j]) if math.isfinite(ses[j]) else None
        low, high, t, p = _interval(b, se, df)
        term = RegressionCoefficient(
            name, design.columns[j], design.levels[j], b, se, low, high, t, p
        )
        if j > 0:
            if fit.logit is not None:
                term.odds_ratio = math.exp(b) if b < 700 else None
                ame = float(fit.logit.ame[j - 1])
                ame_se = math.sqrt(max(float(fit.logit.ame_cov[j - 1, j - 1]), 0))
                ame_low, ame_high, _, _ = _interval(ame, ame_se, df)
                term.marginal, term.marginal_low, term.marginal_high = ame, ame_low, ame_high
                scale = sd_x[j] / sd_y
                term.standardized = ame * scale if math.isfinite(ame) else None
                if ame_low is not None and ame_high is not None:
                    term.std_low, term.std_high = ame_low * scale, ame_high * scale
            else:
                scale = sd_x[j] / sd_y
                term.standardized = b * scale
                if low is not None and high is not None:
                    term.std_low, term.std_high = low * scale, high * scale
        terms.append(term)
    return terms


def _adjust(terms: list[RegressionCoefficient]) -> MultipleTesting:
    body = terms[1:]
    adjusted, summary = correct([t.p_value for t in body], noun="effects")
    for term, q in zip(body, adjusted, strict=True):
        term.q_value = q
        size = (
            effect_size("pearson", term.standardized)
            if term.standardized is not None
            else "negligible"
        )
        term.evidence = grade(size, q)
    return summary


def _sentence(
    term: RegressionCoefficient, fit: Fitted, references: dict[str, str], tests: int
) -> str:
    target = fit.inputs.target
    ci = f"[{fmt(term.ci_low)} ; {fmt(term.ci_high)}]"
    if term.term == INTERCEPT:
        return (
            f"With every column at 0 (and categories at their reference), `{target}` is "
            f"{fmt(term.estimate)} {ci}."
        )
    level = term.level
    if fit.logit is not None and fit.inputs.binary is not None:
        positive = fit.inputs.binary[1]
        value = (term.marginal or 0.0) * 100
        ci = f"[{fmt((term.marginal_low or 0) * 100)} ; {fmt((term.marginal_high or 0) * 100)}]"
        odds = f"odds × {fmt(term.odds_ratio)}" if term.odds_ratio is not None else "odds unbounded"
        way = "raises" if value >= 0 else "lowers"
        if level is None:
            text = (
                f"Each extra unit of `{term.column}` {way} the probability of `{target}` = "
                f"{positive} "
                f"by {fmt(abs(value))} points {ci} on average ({odds}), other columns held fixed."
            )
        else:
            side = "higher" if value >= 0 else "lower"
            text = (
                f"With `{term.column}` = {level}, the probability of `{target}` = {positive} is "
                f"{fmt(abs(value))} points {side} {ci} than with "
                f"{references.get(term.column, 'the reference')} "
                f"({odds}), other columns held fixed."
            )
    elif level is None:
        if term.estimate >= 0:
            text = (
                f"Each extra unit of `{term.column}` adds {fmt(term.estimate)} to `{target}` "
                f"{ci}, other columns held fixed."
            )
        else:
            text = (
                f"Each extra unit of `{term.column}` takes {fmt(-term.estimate)} off "
                f"`{target}` {ci}, other columns held fixed."
            )
    else:
        side = "higher" if term.estimate >= 0 else "lower"
        text = (
            f"With `{term.column}` = {level}, `{target}` is {fmt(abs(term.estimate))} {side} {ci} "
            f"than with {references.get(term.column, 'the reference')}, other columns held fixed."
        )
    if term.q_value is not None and term.q_value >= 0.05:
        text += f" Not clear after correcting for {tests} tests (adj. p = {term.q_value:.2g})."
    return text


# --- Quality -------------------------------------------------------------------------------------


def folds(fit_inputs: Inputs, grouping: Grouping, n: int) -> tuple[list[np.ndarray], str]:
    """R-REG-25: contiguous blocks for a series, whole groups for clusters, else shuffled."""
    if fit_inputs.shape.kind == "series":
        return np.array_split(np.arange(n), FOLDS), f"{FOLDS} folds, contiguous blocks in time"
    codes = grouping.cluster_codes if grouping.cluster_codes is not None else grouping.entity_codes
    name = fit_inputs.cluster or fit_inputs.shape.entity
    rng = np.random.default_rng(SEED)
    if codes is not None and len(np.unique(codes)) >= FOLDS:
        groups = np.unique(codes)
        fold_of = dict(
            zip(
                rng.permutation(groups).tolist(),
                [i % FOLDS for i in range(len(groups))],
                strict=True,
            )
        )
        assign = np.array([fold_of[c] for c in codes.tolist()])
        return [np.flatnonzero(assign == f) for f in range(FOLDS)], f"{FOLDS} folds by `{name}`"
    return np.array_split(rng.permutation(n), FOLDS), f"{FOLDS} random folds"


def out_of_sample(fit: Fitted) -> tuple[float | None, str]:
    x, y = fit.design.x, fit.design.y
    n = len(y)
    if n < 2 * FOLDS:
        return None, "too few rows"
    parts, how = folds(fit.inputs, fit.grouping, n)
    w = fit.ls.weights if fit.ls is not None else None
    loss = base = 0.0
    for test in parts:
        train = np.setdiff1d(np.arange(n), test, assume_unique=True)
        if len(train) <= x.shape[1] or not len(test):
            continue
        if fit.logit is not None:
            if len(np.unique(y[train])) < 2:
                continue
            try:
                p = np.clip(logistic_predict(x[train], y[train], x[test]), 1e-12, 1 - 1e-12)
            except (ValueError, np.linalg.LinAlgError):
                return None, how
            rate = np.clip(y[train].mean(), 1e-12, 1 - 1e-12)
            loss += float(-(y[test] * np.log(p) + (1 - y[test]) * np.log(1 - p)).sum())
            base += float(-(y[test] * np.log(rate) + (1 - y[test]) * np.log(1 - rate)).sum())
            continue
        wt = w[train] if w is not None else None
        predicted = lstsq_predict(x[train], y[train], x[test], wt)
        centre = float(np.average(y[train], weights=wt))
        weight = w[test] if w is not None else np.ones(len(test))
        loss += float(weight @ (y[test] - predicted) ** 2)
        base += float(weight @ (y[test] - centre) ** 2)
    return (1 - loss / base if base > 0 else None), how


def _wald(fit: Fitted) -> tuple[float | None, float | None]:
    k = fit.design.x.shape[1]
    if k < 2:
        return None, None
    beta = fit.logit.beta if fit.logit is not None else fit.ls.beta  # type: ignore[union-attr]
    b = beta[1:]
    v = fit.covariance[1:, 1:]
    try:
        f = float(b @ np.linalg.pinv(v) @ b) / (k - 1)
    except np.linalg.LinAlgError:
        return None, None
    return f, float(stats.f.sf(f, k - 1, fit.df))


def _quality(fit: Fitted) -> RegressionQuality:
    inputs, design = fit.inputs, fit.design
    n, k = design.x.shape
    oos, how = out_of_sample(fit)
    f, f_p = _wald(fit)
    if fit.logit is not None:
        lg = fit.logit
        return RegressionQuality(
            n,
            inputs.loaded,
            inputs.total,
            inputs.sampled,
            design.missing,
            None,
            None,
            oos,
            how,
            lg.aic,
            lg.bic,
            f,
            f_p,
            k - 1,
            n - k,
            lg.pseudo_r2,
        )
    ls = fit.ls
    assert ls is not None
    return RegressionQuality(
        n,
        inputs.loaded,
        inputs.total,
        inputs.sampled,
        design.missing,
        ls.r2,
        ls.adj_r2,
        oos,
        how,
        ls.aic,
        ls.bic,
        f,
        f_p,
        k - 1,
        n - k,
    )


# --- Checks --------------------------------------------------------------------------------------

ROBUST = ("hc0", "hc1", "hc2", "hc3", "hac", "cluster", "cluster2", "dk")


def _checks(fit: Fitted, quality: RegressionQuality) -> list[RegressionCheck]:
    checks: list[RegressionCheck] = []
    design, inputs, se = fit.design, fit.inputs, fit.se
    n = len(design.y)
    if fit.logit is not None and fit.logit.separated:
        checks.append(
            RegressionCheck(
                "separation",
                "bad",
                "Separation",
                "Some rows are predicted with certainty (perfect separation): a column splits "
                f"`{inputs.target}` exactly, its coefficient runs to infinity. Leave it out "
                "or merge its levels.",
                "R-REG-01",
                "terms",
            )
        )
    if fit.ls is not None:
        a = fit.ls.arrays
        bp = dg.breusch_pagan(a.e, a.x)
        if bp.p_value is not None and bp.p_value < SPEC_P:
            robust = se.kind in ROBUST
            checks.append(
                RegressionCheck(
                    "spread",
                    "ok" if robust else "bad",
                    "Spread of the residuals",
                    f"The spread of the residuals changes with the fitted values (Breusch–Pagan "
                    f"{fmt_p(bp.p_value)}). "
                    + (
                        f"The {in_text(se.label)} standard errors already allow for it."
                        if robust
                        else "The classic standard errors assume it does not: try robust ones."
                    ),
                    "R-REG-12",
                    None if robust else "standard_errors",
                )
            )
        else:
            checks.append(
                RegressionCheck(
                    "spread",
                    "ok",
                    "Spread of the residuals",
                    f"The spread of the residuals looks even (Breusch–Pagan {fmt_p(bp.p_value)}); "
                    "robust standard errors stay right either way.",
                    "R-REG-12",
                )
            )
        if inputs.shape.kind == "series" and n > 10:
            lag = cov.newey_west_lag(n)
            bg = dg.breusch_godfrey(a.e, a.x, lag)
            dw = dg.durbin_watson(a.e)
            correlated = bg.p_value is not None and bg.p_value < SPEC_P
            handled = se.kind == "hac"
            status: Status = "ok" if not correlated or handled else "bad"
            follow = "follow each other" if correlated else "do not follow each other"
            text = (
                f"Residuals {follow} in time "
                f"(Durbin–Watson {dw:.2f}, Breusch–Godfrey {fmt_p(bg.p_value)} at {lag} lags)."
            )
            if correlated:
                text += (
                    " HAC standard errors allow for it." if handled else " Try HAC standard errors."
                )
            checks.append(
                RegressionCheck(
                    "autocorrelation",
                    status,
                    "Errors in time",
                    text,
                    "R-REG-SE-02",
                    None if status == "ok" else "standard_errors",
                )
            )
        rs = dg.reset(
            design.y * (np.sqrt(fit.ls.weights) if fit.ls.weights is not None else 1),
            a.x,
            a.x @ fit.ls.beta,
        )
        if rs.p_value is not None and rs.p_value < SPEC_P:
            numbers = {
                t: design.x[:, j]
                for j, t in enumerate(design.terms)
                if j > 0 and design.levels[j] is None
            }
            bend = dg.bending(fit.ls.resid, numbers)
            text = f"The effects may not be straight lines (RESET {fmt_p(rs.p_value)})"
            text += f": `{bend}` bends the most. See `{inputs.target}` against it." if bend else "."
            checks.append(
                RegressionCheck(
                    "specification",
                    "warn",
                    "Straight lines",
                    text,
                    "R-REG-10",
                    "compare" if bend else None,
                    bend,
                )
            )
        else:
            checks.append(
                RegressionCheck(
                    "specification",
                    "ok",
                    "Straight lines",
                    f"No sign of a missing curve (RESET {fmt_p(rs.p_value)}).",
                    "R-REG-10",
                )
            )
        found = dg.normality(fit.ls.resid)
        test = found.shapiro or found.jarque_bera
        normal = test.p_value is None or test.p_value >= SPEC_P
        if normal:
            text = f"The residuals look normal ({test.name} {fmt_p(test.p_value)})."
            status = "ok"
        elif n >= LARGE_N:
            text = (
                f"The residuals are not normal ({test.name} {fmt_p(test.p_value)}, skewness "
                f"{found.skewness:.1f}), but at n = {n:,} normality matters little for the "
                "coefficients' intervals."
            )
            status = "info"
        else:
            text = (
                f"The residuals are not normal ({test.name} {fmt_p(test.p_value)}) and there "
                "are only "
                f"{n:,} rows: read the intervals with care."
            )
            status = "warn"
        checks.append(RegressionCheck("normality", status, "Normality of the residuals", text))
    vifs, condition = dg.collinearity(design.x, design.terms)
    worst = max((v for v in vifs if v.vif is not None), key=lambda v: v.vif or 0, default=None)
    if worst is not None and worst.vif is not None and worst.vif > VIF_HIGH:
        checks.append(
            RegressionCheck(
                "collinearity",
                "bad",
                "Columns that repeat each other",
                f"`{worst.term}` and `{worst.partner}` say almost the same thing (VIF "
                f"{worst.vif:.0f}, "
                f"r = {worst.correlation:.2f}): their effects share out at random and the "
                "intervals widen.",
                "R-REG-13",
                "correlations",
                worst.term,
            )
        )
    elif condition > CONDITION_HIGH:
        checks.append(
            RegressionCheck(
                "collinearity",
                "warn",
                "Columns that repeat each other",
                f"The columns are close to dependent (condition number {condition:.1f} > "
                f"{CONDITION_HIGH}); "
                "intervals may be wide.",
                "R-REG-13",
                "terms",
            )
        )
    else:
        largest = (
            f"largest VIF {worst.vif:.1f}"
            if worst is not None and worst.vif is not None
            else "one column"
        )
        checks.append(
            RegressionCheck(
                "collinearity",
                "ok",
                "Columns that repeat each other",
                f"No column repeats another ({largest}).",
                "R-REG-13",
            )
        )
    if design.left_out:
        first = design.left_out[0]
        more = (
            f" (and {_plural(len(design.left_out) - 1, 'other')})"
            if len(design.left_out) > 1
            else ""
        )
        checks.append(
            RegressionCheck(
                "left_out",
                "warn",
                "Left out",
                f"`{first.term}` was left out: {first.reason}{more}.",
                "R-REG-13",
                "terms",
            )
        )
    if fit.found is not None:
        share = fit.found.influential / max(n, 1)
        outliers = fit.found.outliers
        if share > INFLUENTIAL_SHARE or outliers:
            text = (
                f"{_plural(fit.found.influential, 'row')} ({share:.1%}) weigh heavily on the "
                "coefficients "
                f"(Cook's distance > 4/n)"
            )
            text += (
                f", {_plural(outliers, 'residual')} beyond the Bonferroni threshold."
                if outliers
                else "."
            )
            checks.append(
                RegressionCheck(
                    "influence", "warn", "Influential rows", text, "R-REG-11", "influence"
                )
            )
        else:
            checks.append(
                RegressionCheck(
                    "influence",
                    "ok",
                    "Influential rows",
                    "No row weighs much more than the others on the coefficients.",
                    "R-REG-11",
                )
            )
    fit_r2 = quality.r2 if quality.r2 is not None else quality.pseudo_r2
    if quality.oos_r2 is not None and fit_r2 is not None:
        name = "R²" if quality.r2 is not None else "pseudo-R²"
        gap = fit_r2 - quality.oos_r2
        if gap > OVERFIT:
            checks.append(
                RegressionCheck(
                    "overfitting",
                    "warn",
                    "Out of sample",
                    f"Out of sample the model explains much less ({name} {quality.oos_r2:.2f} "
                    "against "
                    f"{fit_r2:.2f}): it fits noise. Fewer columns or merged levels would hold "
                    "better.",
                    "R-REG-25",
                    "terms",
                )
            )
        else:
            checks.append(
                RegressionCheck(
                    "overfitting",
                    "ok",
                    "Out of sample",
                    f"Out of sample {name} {quality.oos_r2:.2f}, close to {fit_r2:.2f} on the "
                    "rows used: "
                    "no sign of overfitting.",
                    "R-REG-25",
                )
            )
    if se.warning:
        checks.append(
            RegressionCheck(
                "clusters", "warn", "Few groups", se.warning, "R-REG-SE-03", "standard_errors"
            )
        )
    if inputs.sampled:
        checks.append(
            RegressionCheck(
                "sample",
                "info",
                "Sample",
                f"Computed on a reproducible sample of {inputs.loaded:,} of the "
                f"{inputs.total:,} rows.",
            )
        )
    return checks


# --- Plots ---------------------------------------------------------------------------------------


def _plots(fit: Fitted) -> ResidualPlots | None:
    if fit.ls is None or fit.found is None:
        return None
    n = fit.ls.n
    positions = np.arange(n)
    if n > PLOT_POINTS:
        rng = np.random.default_rng(SEED)
        picked = rng.choice(n, PLOT_POINTS - 20, replace=False)
        positions = np.unique(np.concatenate([picked, top_rows(fit.found)]))
    s = math.sqrt(float(fit.ls.arrays.e @ fit.ls.arrays.e) / max(fit.ls.df_resid, 1))
    standardized = fit.ls.arrays.e / (s * np.sqrt(np.clip(1 - fit.found.leverage, 1e-12, None)))
    m = min(n, QQ_POINTS)
    probabilities = (np.arange(m) + 0.5) / m
    rows = fit.inputs.rows[fit.design.used]
    return ResidualPlots(
        [float(v) for v in fit.ls.fitted[positions]],
        [float(v) for v in fit.ls.arrays.e[positions]],
        [int(v) for v in rows[positions]],
        [float(v) for v in fit.found.leverage[positions]],
        [float(v) for v in fit.found.cook[positions]],
        fit.found.cook_threshold,
        [float(v) for v in stats.norm.ppf(probabilities)],
        [float(v) for v in np.quantile(standardized, probabilities)],
        len(positions),
        n,
    )


# --- The result ----------------------------------------------------------------------------------


def _model_reason(fit: Fitted) -> str:
    inputs = fit.inputs
    if fit.model == "logit" and inputs.binary is not None:
        return (
            f"R-REG-01: `{inputs.target}` is yes/no ({inputs.binary[0]} / "
            f"{inputs.binary[1]}); a straight "
            "line would leave [0, 1]. Effects read as average changes in probability."
        )
    if fit.model == "wls":
        return inputs.weights_reason or f"R-REG-SE-04: rows weighted by `{inputs.weights}`."
    return "A number to explain: least squares, in the units of the data."


def _result(fit: Fitted) -> RegressionReport:
    inputs, design = fit.inputs, fit.design
    terms = _terms(fit)
    multiple = _adjust(terms)
    references = {r.column: f"`{r.column}` = {r.level} (the reference)" for r in design.references}
    for term in terms:
        term.sentence = _sentence(term, fit, references, multiple.tests)
    quality = _quality(fit)
    body = terms[1:]
    chosen = None
    if inputs.focus:
        chosen = next((t for t in body if t.column == inputs.focus), None)
    if chosen is None:
        clear = [
            t
            for t in body
            if t.q_value is not None and t.q_value < 0.05 and t.standardized is not None
        ]
        chosen = max(clear, key=lambda t: abs(t.standardized or 0), default=None)
    if chosen is not None:
        sentence = f"{chosen.sentence} {multiple.sentence}"
    elif body:
        sentence = f"No column has a clear effect on `{inputs.target}`. {multiple.sentence}"
    else:
        sentence = f"No column left to explain `{inputs.target}`."
    hints = [h for h in [two_way_hint(inputs.shape)] if h]
    return RegressionReport(
        inputs.target,
        inputs.binary[1] if inputs.binary else None,
        fit.model,
        MODEL_LABELS[fit.model],
        _model_reason(fit),
        [v.name for v in inputs.variables],
        inputs.focus,
        terms,
        fit.se,
        inputs.weights if fit.model == "wls" else None,
        quality,
        multiple,
        sentence,
        _checks(fit, quality),
        design.references,
        [*inputs.refused, *design.left_out],
        inputs.shape,
        hints,
        _plots(fit),
    )


# --- Sections ------------------------------------------------------------------------------------

WHAT: dict[Kind, str] = {
    "classic": "assumes equal variances and independent rows",
    "hc0": "unequal variances (White)",
    "hc1": "unequal variances, × n/(n − k)",
    "hc2": "unequal variances, corrected by the leverage",
    "hc3": "unequal variances, jackknife-like: the safest of the HC family",
    "hac": "unequal variances and errors close in time",
    "cluster": "rows of one group share their shocks",
    "cluster2": "shocks shared within entities and within periods",
    "dk": "shocks shared across entities, correlated in time",
}


def standard_errors(fit: Fitted) -> StandardErrorTable:
    terms = fit.result.terms
    kinds: list[SeColumn] = []
    n, k = fit.design.x.shape
    for kind in SE_KINDS:
        missing = available(kind, fit.inputs.shape, fit.grouping)
        chosen = kind == fit.se.kind
        if missing is not None:
            kinds.append(SeColumn(kind, LABELS[kind], False, missing, False, [], []))
            continue
        if fit.logit is not None and kind not in ("classic", "hc0", "hac", "cluster"):
            kinds.append(SeColumn(kind, LABELS[kind], False, "least squares only", False, [], []))
            continue
        if kind == fit.se.kind:
            matrix = fit.covariance
            choice = fit.se
        elif fit.logit is not None:
            choice = choose(kind, fit.inputs.shape, fit.grouping, n)
            cov_type, kwds = _logit_cov(kind, fit.grouping)
            if kind == "hac":
                kwds = {"maxlags": cov.newey_west_lag(n)}
            matrix = logistic(fit.design.x, fit.design.y, cov_type, kwds).cov
        else:
            assert fit.ls is not None
            choice = choose(kind, fit.inputs.shape, fit.grouping, n)
            matrix = covariance_of(kind, fit.ls, fit.grouping, fit.inputs.shape)
        df = _df(choice, n, k)
        ses = np.sqrt(np.clip(np.diag(matrix), 0, None))
        p_values: list[float | None] = []
        for term, se in zip(terms, ses, strict=True):
            p_values.append(float(2 * stats.t.sf(abs(term.estimate / se), df)) if se > 0 else None)
        kinds.append(
            SeColumn(
                kind, LABELS[kind], True, WHAT[kind], chosen, [float(s) for s in ses], p_values
            )
        )
    classic = next((c for c in kinds if c.kind == "classic" and c.available), None)
    current = next((c for c in kinds if c.chosen), None)
    sentence = f"Standard errors: {fit.se.label} ({fit.se.reason})"
    if classic and current and current.kind != "classic" and len(terms) > 1:
        ratios = [
            c / s
            for c, s in zip(current.std_errors[1:], classic.std_errors[1:], strict=True)
            if s and c
        ]
        if ratios:
            median = float(np.median(ratios))
            sentence = (
                f"{current.label} standard errors are {median:.2f} × the classic ones (median "
                "over the "
                f"coefficients). {fit.se.reason}"
            )
    return StandardErrorTable([t.term for t in terms], [t.estimate for t in terms], kinds, sentence)


def influence_table(fit: Fitted) -> InfluenceTable:
    terms = [t.term for t in fit.result.terms]
    if fit.ls is None or fit.found is None:
        return InfluenceTable(terms, [], 0, 0, 0.0, "Influence is computed for least squares only.")
    rows = top_rows(fit.found)
    betas = dfbetas(fit.ls.arrays, fit.found, rows)
    numbers = fit.inputs.rows[fit.design.used]
    listed = [
        InfluentialRow(
            int(numbers[i]),
            float(fit.found.leverage[i]),
            float(fit.found.cook[i]),
            float(fit.found.studentized[i]),
            float(fit.found.bonferroni[i]),
            [float(v) for v in betas[r]],
        )
        for r, i in enumerate(rows)
    ]
    n = fit.ls.n
    sentence = (
        f"{_plural(fit.found.influential, 'row')} of {n:,} have a Cook's distance above 4/n = "
        f"{fit.found.cook_threshold:.2g}; |DFBETAS| above 2/√n = {2 / math.sqrt(n):.2g} moves a "
        "coefficient by more than its noise."
    )
    return InfluenceTable(
        terms, listed, fit.found.influential, fit.found.outliers, fit.found.cook_threshold, sentence
    )


def terms_table(fit: Fitted) -> TermsTable:
    vifs, condition = dg.collinearity(fit.design.x, fit.design.terms)
    count = len(fit.result.left_out)
    sentence = f"{_plural(len(fit.design.terms) - 1, 'term')} estimated"
    sentence += f", {_plural(count, 'column')} left out with the reason." if count else "."
    return TermsTable(fit.design.references, fit.result.left_out, vifs, condition, sentence)


def more(fit: Fitted) -> MoreChecks:
    if fit.ls is None or fit.found is None:
        return MoreChecks(None, "White's test is for least squares.", 0, [], "")
    a = fit.ls.arrays
    test = dg.white(a.e, a.x)
    if test is None:
        white_sentence = (
            f"White's test is skipped beyond {dg.WHITE_MAX_TERMS} terms (too many products)."
        )
    else:
        verdict = "changes" if test.p_value is not None and test.p_value < SPEC_P else "looks even"
        white_sentence = (
            f"With every square and cross-product, the spread {verdict} (White "
            f"{fmt_p(test.p_value)}). "
            "It informs, it does not choose the standard errors (R-REG-SE-05)."
        )
    keep = fit.found.cook <= fit.found.cook_threshold
    removed = int((~keep).sum())
    shifts: list[CoefficientShift] = []
    if removed > 0 and int(keep.sum()) > fit.design.x.shape[1] + 1:
        w = fit.ls.weights[keep] if fit.ls.weights is not None else None
        again = least_squares(fit.design.x[keep], fit.design.y[keep], w)
        for j, term in enumerate(fit.result.terms):
            new = float(again.beta[j])
            change = (new - term.estimate) / abs(term.estimate) if term.estimate else None
            shifts.append(CoefficientShift(term.term, term.estimate, new, change))
    moved = [s for s in shifts[1:] if s.change is not None and abs(s.change) > 0.2]
    shift_sentence = (
        f"Without the {_plural(removed, 'influential row')}, "
        + (
            f"{_plural(len(moved), 'coefficient')} move by more than 20 %."
            if moved
            else "no coefficient moves by more than 20 %."
        )
        if removed
        else "No influential row to leave out."
    )
    return MoreChecks(test, white_sentence, removed, shifts, shift_sentence)
