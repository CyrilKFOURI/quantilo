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
from quantilo.regression import families as fm
from quantilo.regression import terms as tm
from quantilo.regression.covariance import LABELS, Kind
from quantilo.regression.design import (
    INTERCEPT,
    Design,
    DesignError,
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

Model = Literal[
    "ols", "wls", "logit", "probit", "poisson", "negbin", "gamma", "log_ols", "boxcox",
    "fractional", "mnlogit",
]  # fmt: skip
Status = Literal["ok", "warn", "bad", "info"]
MODEL_LABELS: dict[Model, str] = {
    "ols": "Least squares (OLS)",
    "wls": "Weighted least squares (WLS)",
    "logit": "Logistic",
    **{kind: fm.LABELS[kind] for kind in fm.FAMILIES if kind not in ("ols", "logit")},
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
    # 069: the family of the model
    target_role: str | None = None
    outcomes: list[str] = field(default_factory=list)  # classes of a category target (codes in y)
    family: fm.FamilyRequest = "auto"
    exposure: str | None = None  # R-REG-02: its log is the offset of a count model
    exposure_values: np.ndarray | None = None
    exposure_reason: str | None = None
    out_of_sample: bool = True  # off for the alternatives of Advanced › More
    period_order: np.ndarray | None = None  # 071: each row's period rank (first differences)
    plan: tm.Plan = field(default_factory=tm.Plan)  # 070: curves, logs, interactions, contrasts
    dates: np.ndarray | None = None  # 073: a series' date of each row read, as text


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
    ratio: float | None = None  # 069: exp(b), rate ratio or multiplier (log link, log(y))
    ratio_low: float | None = None
    ratio_high: float | None = None
    outcome: str | None = None  # multinomial logit: the class this coefficient is about
    group: str | None = None  # 070: the term it comes from (a column, `spline:age`, …)


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
    pseudo_how: str | None = None  # 069: McFadden or deviance explained


@dataclass
class RegressionCheck:
    key: str
    status: Status
    title: str
    sentence: str
    rule: str | None = None
    action: str | None = None  # an Advanced section, "compare" or "correlations"
    column: str | None = None  # the column the action is about
    family: str | None = None  # action "family": the family to switch to (069)
    term: str | None = None  # action "term": the term to add (070, `spline:age`)


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
class FamilyInfo:
    """The family of the model (069): chosen by rule from the target, or in the settings."""

    kind: str
    label: str
    rule: str | None
    reason: str
    reads: fm.Reads  # how the effects read: units, points of probability, ratios (%)
    requested: str
    target_kind: str
    options: list[fm.FamilyOption]
    exposure: str | None = None
    exposure_reason: str | None = None
    alpha: float | None = None  # negative binomial
    dispersion: float | None = None  # counts: Pearson χ² / (n − k)
    smearing: float | None = None  # log(y): Duan's factor
    boxcox_lambda: float | None = None
    boxcox_low: float | None = None
    boxcox_high: float | None = None
    outcomes: list[str] = field(default_factory=list)  # multinomial: the reference first
    note: str | None = None


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
    family: FamilyInfo | None = None


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
    # 070: the terms in use and how they read
    specs: list[str] = field(default_factory=list)
    center: bool = False
    coding: str = "reference"
    notes: list[tm.TermNote] = field(default_factory=list)
    tests: list[tm.TermTest] = field(default_factory=list)
    curves: list[tm.EffectCurve] = field(default_factory=list)
    slopes: list[tm.SimpleSlope] = field(default_factory=list)
    contrasts: list[tm.LevelEffect] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)  # columns a partial plot can show


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
    # 069: the family, its fit when it is not least squares, its target-specific numbers
    family: fm.FamilyChoice | None = None
    profile: fm.TargetProfile | None = None
    glm: fm.GlmFit | None = None
    transform: fm.Transform | None = None
    original: np.ndarray | None = None  # the target before log or Box-Cox
    smear: float | None = None
    dispersion: fm.Overdispersion | None = None
    negbin_aic: float | None = None
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


def _mle_choice(choice: SeChoice, label: str) -> SeChoice:
    """HC1–3 need the hat matrix of least squares: a likelihood model has its own sandwich."""
    if choice.kind in ("classic", "hac", "cluster"):
        return choice
    model = "logistic" if label == MODEL_LABELS["logit"] else label
    return replace(
        choice,
        kind="hc0",
        label=LABELS["hc0"],
        reason=choice.reason.replace("HC3", "a robust sandwich") + f" The {model} model uses HC0.",
    )


def _offset(inputs: Inputs, family: fm.Family) -> np.ndarray | None:
    """R-REG-02: the log of the exposure, for counts (NaN where it is missing or ≤ 0)."""
    if family not in fm.OFFSET or not inputs.exposure or inputs.exposure_values is None:
        return None
    values = inputs.exposure_values
    return np.log(np.where(values > 0, values, np.nan))


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
    design = build(inputs.y, inputs.variables, extra, inputs.plan)
    profile = fm.profile(design.y, inputs.binary, inputs.outcomes, inputs.target_role)
    family = fm.choose(profile, inputs.family, inputs.target, inputs.binary)
    offset_all = _offset(inputs, family.kind)
    if offset_all is not None and inputs.exposure:
        # R-REG-02: the exposure is the offset, not a column of the design
        variables = [v for v in inputs.variables if v.name != inputs.exposure]
        design = build(inputs.y, variables, {**extra, inputs.exposure: offset_all}, inputs.plan)
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
    if family.kind == "logit":
        choice = _mle_choice(choice, MODEL_LABELS["logit"])
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
    elif family.kind in fm.MLE:
        choice = _mle_choice(choice, family.label)
        cov_type, kwds = _logit_cov(choice.kind, grouping)
        if choice.kind == "hac":
            kwds = {"maxlags": choice.lag or cov.newey_west_lag(n)}
        offset = _used(offset_all, used)
        glm_weights = w if family.kind in fm.GLM_WEIGHTS else None
        try:
            glm = fm.fit(
                family.kind, design.x, design.y, cov_type, kwds, offset, glm_weights,
                inputs.outcomes or None,
            )  # fmt: skip
        except (np.linalg.LinAlgError, ValueError) as error:
            raise DesignError(f"The {family.label} model could not be fitted: {error}") from error
        fitted = Fitted(
            inputs,
            design,
            cast(Model, family.kind),
            None,
            None,
            grouping,
            choice,
            _df(choice, n, k),
            glm.cov,
            None,
        )
        fitted.glm = glm
        if family.kind == "poisson":
            fitted.dispersion = fm.overdispersion(design.y, glm.mu, k)
            if fm.overdispersed(fitted.dispersion):
                # R-REG-03: the negative binomial is proposed, compared in AIC
                try:
                    fitted.negbin_aic = fm.fit(
                        "negbin", design.x, design.y, "nonrobust", None, offset
                    ).aic
                except (np.linalg.LinAlgError, ValueError):
                    fitted.negbin_aic = None
    else:
        changed = fm.transform(family.kind, design.x, design.y, w)
        original = design.y
        if family.kind != "ols":
            design = replace(design, y=changed.y)
        ls = least_squares(design.x, design.y, w)
        model: Model = "wls" if w is not None and family.kind == "ols" else cast(Model, family.kind)
        fitted = Fitted(
            inputs,
            design,
            model,
            ls,
            None,
            grouping,
            choice,
            _df(choice, n, k),
            covariance_of(choice.kind, ls, grouping, inputs.shape),
            influence(ls.arrays),
        )
        if family.kind != "ols":
            fitted.transform, fitted.original = changed, original
        if family.kind == "log_ols":
            fitted.smear = fm.smearing(ls.resid, w)
    fitted.family, fitted.profile = family, profile
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
    if fit.glm is not None:
        return _glm_terms(fit, fit.glm)
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
            name,
            design.columns[j],
            design.levels[j],
            b,
            se,
            low,
            high,
            t,
            p,
            group=design.groups[j] if design.groups else None,
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
        if fit.transform is not None and fit.transform.kind == "log_ols":
            term.ratio = _exp(b)
            term.ratio_low, term.ratio_high = _exp(low), _exp(high)
        terms.append(term)
    return terms


def _exp(value: float | None) -> float | None:
    return math.exp(value) if value is not None and value < 700 else None


def _glm_terms(fit: Fitted, glm: fm.GlmFit) -> list[RegressionCoefficient]:
    """069: coefficients of a likelihood family; the multinomial logit gives one block per class
    against the reference, named `class · term`."""
    design, df = fit.design, fit.df
    kind = glm.kind
    ses = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
    sd_x = design.x.std(axis=0)
    k = design.x.shape[1]
    equations = len(glm.beta) // k
    reads = fm.READS[kind]
    terms: list[RegressionCoefficient] = []
    for e in range(equations):
        outcome = glm.outcomes[e + 1] if kind == "mnlogit" else None
        if kind == "mnlogit":
            codes = glm.codes if glm.codes is not None else design.y.astype(int)
            share = float((codes == e + 1).mean())
            sd_y = math.sqrt(share * (1 - share)) or 1.0
        elif reads == "points":
            mean = float(design.y.mean())
            sd_y = math.sqrt(max(mean * (1 - mean), 0)) or 1.0
        else:
            sd_y = float(design.y.std()) or 1.0
        for j in range(k):
            i = e * k + j
            b = float(glm.beta[i])
            se = float(ses[i]) if math.isfinite(ses[i]) else None
            low, high, t, p = _interval(b, se, df)
            name = design.terms[j] if outcome is None else f"{outcome} · {design.terms[j]}"
            term = RegressionCoefficient(
                name,
                design.columns[j],
                design.levels[j],
                b,
                se,
                low,
                high,
                t,
                p,
                outcome=outcome,
                group=design.groups[j] if design.groups else None,
            )
            if j > 0:
                ame = float(glm.marginal[i])
                ame_se = float(glm.marginal_se[i])
                if math.isfinite(ame):
                    ame_low, ame_high, _, _ = _interval(ame, ame_se, df)
                    term.marginal, term.marginal_low, term.marginal_high = ame, ame_low, ame_high
                    scale = sd_x[j] / sd_y
                    term.standardized = ame * scale
                    if ame_low is not None and ame_high is not None:
                        term.std_low, term.std_high = ame_low * scale, ame_high * scale
                if reads == "ratio":
                    term.ratio, term.ratio_low, term.ratio_high = _exp(b), _exp(low), _exp(high)
                elif kind in ("fractional", "mnlogit"):
                    term.odds_ratio = _exp(b)
            terms.append(term)
    return terms


def _adjust(terms: list[RegressionCoefficient]) -> MultipleTesting:
    body = [t for t in terms if t.column != INTERCEPT]
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
    term: RegressionCoefficient,
    fit: Fitted,
    references: dict[str, str],
    tests: int,
    index: int = -1,
) -> str:
    target = fit.inputs.target
    ci = f"[{fmt(term.ci_low)} ; {fmt(term.ci_high)}]"
    special = _term_sentence(term, index, fit, references)
    if special is not None:
        if term.q_value is not None and term.q_value >= 0.05:
            special += (
                f" Not clear after correcting for {tests} tests (adj. p = {term.q_value:.2g})."
            )
        return special
    if fit.family is not None and fit.family.kind not in ("ols", "logit"):
        text = _family_sentence(term, fit, references) + _main_suffix(term, fit, references)
        if term.column != INTERCEPT and term.q_value is not None and term.q_value >= 0.05:
            text += f" Not clear after correcting for {tests} tests (adj. p = {term.q_value:.2g})."
        return text
    if term.term == INTERCEPT:
        return (
            f"With {_origin(fit)} (and categories at their reference), `{target}` is "
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
    text += _main_suffix(term, fit, references)
    if term.level is None and term.column.endswith(" › Trend"):  # R-REG-TERM-08
        source = fit.design.sources.get(term.column, term.column)
        text = text.replace(f"Each extra unit of `{term.column}`", f"Each year later in `{source}`")
    if term.q_value is not None and term.q_value >= 0.05:
        text += f" Not clear after correcting for {tests} tests (adj. p = {term.q_value:.2g})."
    return text


def _recipe(fit: Fitted, index: int) -> tm.Recipe | None:
    recipes = fit.design.recipes
    if index < 0 or not recipes:
        return None
    return recipes[index % len(recipes)]


def _term_sentence(
    term: RegressionCoefficient, index: int, fit: Fitted, references: dict[str, str]
) -> str | None:
    """070: the sentence of a term that is not a column as it is: part of a curve, an
    interaction, the log of a column (R-REG-TERM-02…07)."""
    group = term.group or ""
    target = fit.inputs.target
    reads = fit.family.reads if fit.family is not None else "units"
    ci = f"[{fmt(term.ci_low)} ; {fmt(term.ci_high)}]"
    design = fit.design
    if group.startswith(("spline:", "square:", "cube:")):
        return (
            f"`{term.term}` is part of the curve of `{term.column}` ({fmt(term.estimate)} {ci}): "
            "read the curve and its joint test in Advanced › Terms."
        )
    recipe = _recipe(fit, index)
    if group.startswith("interaction:") and recipe is not None and recipe.kind == "product":
        first, second = recipe.parts
        if reads != "units":
            return (
                f"`{term.term}`: the effect of one column changes with the other "
                f"({fmt(term.estimate)} {ci} on the model's scale); read the slopes in "
                "Advanced › Terms."
            )
        if first.kind != "ind" and second.kind != "ind":
            a, b = first.variables[0], second.variables[0]
            way = "grows" if term.estimate >= 0 else "shrinks"
            return (
                f"The effect of each extra unit of `{a}` on `{target}` {way} by "
                f"{fmt(abs(term.estimate))} {ci} for each extra unit of `{b}` (and the other way "
                "round), other columns held fixed."
            )
        if first.kind != "ind":
            a = first.variables[0]
            b = second.variables[0]
            level = term.term.rsplit(" = ", 1)[-1]
            side = "more" if term.estimate >= 0 else "less"
            return (
                f"With `{b}` = {level}, each extra unit of `{a}` adds {fmt(abs(term.estimate))} "
                f"{side} {ci} to `{target}` than with {references.get(b, 'the reference')}, other "
                "columns held fixed."
            )
        return (
            f"The combination `{term.term}` is {fmt(term.estimate)} {ci} away from what its two "
            "levels give separately, other columns held fixed."
        )
    source = design.sources.get(term.column, term.column)
    if term.level is None and source != term.column and term.column.startswith("log("):
        plus = term.column.startswith("log(1 +")
        what = f"1 + `{source}`" if plus else f"`{source}`"
        if reads == "ratio":
            way = "raises" if term.estimate >= 0 else "lowers"
            return (
                f"A 1 % rise in {what} {way} `{target}` by about {fmt(abs(term.estimate))} % "
                f"{ci} (an elasticity), other columns held fixed."
            ) + _main_suffix(term, fit, references)
        if reads == "units":
            low = None if term.ci_low is None else term.ci_low / 100
            high = None if term.ci_high is None else term.ci_high / 100
            way = "adds" if term.estimate >= 0 else "takes"
            off = "to" if term.estimate >= 0 else "off"
            return (
                f"A 1 % rise in {what} {way} {fmt(abs(term.estimate) / 100)} {off} `{target}` "
                f"[{fmt(low)} ; {fmt(high)}], other columns held fixed."
            ) + _main_suffix(term, fit, references)
    return None


def _origin(fit: Fitted) -> str:
    """Where the intercept is read (R-REG-TERM-05, 08): at 0, at the means once centred, and a
    date's trend at its round first year, the origin it is counted from (fix3: `yr_built` read
    "at 0" was read at 1900)."""
    design = fit.design
    if design.centres:
        return "every number at its mean"
    starts = [
        f"`{r.variables[0]}` at {r.centre:.0f}"
        for r in design.recipes
        if r.kind == "num" and r.centre and r.variables
    ]
    return "every column at 0" + (f" but {', '.join(starts)}" if starts else "")


def _main_suffix(term: RegressionCoefficient, fit: Fitted, references: dict[str, str]) -> str:
    """070: where a main effect reads when the column also has a square or an interaction."""
    design, plan = fit.design, fit.inputs.plan
    if not plan.specs or term.column == INTERCEPT or term.group != term.column:
        return ""
    name = term.column
    source = design.sources.get(name, name)
    parts: list[str] = []
    if term.level is None and any(
        s.kind in ("square", "cube") and s.columns[0] == source for s in plan.specs
    ):
        mean = next(
            (r.centre for r in design.recipes if r.kind == "power" and r.variables == (name,)), None
        )
        parts.append(f"This is the slope at the mean of `{name}` ({fmt(mean)}).")
    for spec in plan.specs:
        if spec.kind != "interaction" or source not in spec.columns:
            continue
        other = spec.columns[1] if spec.columns[0] == source else spec.columns[0]
        shown = next((n for n in design.kinds if design.sources.get(n, n) == other), other)
        kind = design.kinds.get(shown, ("num", []))[0]
        if kind == "cat":
            if any(r.coding == "deviation" for r in design.references if r.column == shown):
                at = f"at the average of the `{shown}` levels"
            else:
                at = f"with {references.get(shown, 'the reference')}"
        elif shown in design.centres:
            at = f"at the mean of `{shown}` ({fmt(design.centres[shown])})"
        else:
            at = f"at `{shown}` = 0"
        parts.append(f"Read {at}: its interaction with `{shown}` is a separate term.")
    return (" " + " ".join(parts)) if parts else ""


def _pct(value: float | None) -> str:
    """A ratio as a signed change: 1.12 → "+12 %"."""
    if value is None:
        return "—"
    change = (value - 1) * 100
    return f"{'+' if change >= 0 else '−'}{fmt(abs(change))} %"


def _family_sentence(term: RegressionCoefficient, fit: Fitted, references: dict[str, str]) -> str:
    """069: an effect read in the family's own terms (points of probability, %, Box-Cox units)."""
    assert fit.family is not None
    kind, target = fit.family.kind, fit.inputs.target
    glm = fit.glm
    exposed = glm is not None and glm.offset is not None
    rate = " per unit of exposure" if exposed else ""
    b, low, high = term.estimate, term.ci_low, term.ci_high
    lam = fit.transform.lam if fit.transform is not None else None
    if term.column == INTERCEPT:
        start = f"With {_origin(fit)} (and categories at their reference), "
        if kind in ("poisson", "negbin", "gamma"):
            return (
                f"{start}the expected `{target}`{rate} is {fmt(_exp(b))} "
                f"[{fmt(_exp(low))} ; {fmt(_exp(high))}]."
            )
        if kind == "log_ols":
            smear = fit.smear or 1.0
            centre = _exp(b)
            return (
                f"{start}`{target}` is about {fmt(centre * smear if centre else None)} "
                f"(geometric mean {fmt(centre)} × Duan's smearing factor {smear:.3f})."
            )
        if kind == "boxcox" and lam is not None:
            return (
                f"{start}the Box-Cox transform of `{target}` (λ = {lam:.2f}) is {fmt(b)}, about "
                f"{fmt(fm.boxcox_inverse(b, lam))} in units."
            )
        if kind == "mnlogit" and glm is not None:
            return (
                f"{start}the log-odds of `{target}` = {term.outcome} against "
                f"{glm.outcomes[0]} are {fmt(b)}."
            )
        level = float(stats.norm.cdf(b)) if kind == "probit" else 1 / (1 + math.exp(-b))
        what = "share" if kind == "fractional" else "probability"
        return f"{start}the {what} of `{target}` is {fmt(level * 100)} %."
    reference = references.get(term.column, "the reference")
    level = term.level
    if fit.family.reads == "points":
        if kind == "mnlogit":
            what = f"the probability of `{target}` = {term.outcome}"
        elif kind == "fractional":
            what = f"the share `{target}`"
        else:
            positive = fit.inputs.binary[1] if fit.inputs.binary else "yes"
            what = f"the probability of `{target}` = {positive}"
        value = (term.marginal or 0.0) * 100
        ci = f"[{fmt((term.marginal_low or 0) * 100)} ; {fmt((term.marginal_high or 0) * 100)}]"
        odds = f" (odds × {fmt(term.odds_ratio)})" if term.odds_ratio is not None else ""
        if level is None:
            way = "raises" if value >= 0 else "lowers"
            return (
                f"Each extra unit of `{term.column}` {way} {what} by {fmt(abs(value))} points "
                f"{ci} on average{odds}, other columns held fixed."
            )
        side = "higher" if value >= 0 else "lower"
        return (
            f"With `{term.column}` = {level}, {what} is {fmt(abs(value))} points {side} {ci} "
            f"than with {reference}{odds}, other columns held fixed."
        )
    if fit.family.reads == "ratio":
        what = f"`{target}`" if kind == "log_ols" else f"the expected `{target}`{rate}"
        ci = f"[{_pct(term.ratio_low)} ; {_pct(term.ratio_high)}]"
        if level is None:
            return (
                f"Each extra unit of `{term.column}` multiplies {what} by {fmt(term.ratio)} "
                f"({_pct(term.ratio)}) {ci}, other columns held fixed."
            )
        return (
            f"With `{term.column}` = {level}, {what} is {_pct(term.ratio)} {ci} against "
            f"{reference}, other columns held fixed."
        )
    # Box-Cox: the transform's units, and what it means at the median of the target
    ci = f"[{fmt(low)} ; {fmt(high)}]"
    median = float(np.median(fit.original)) if fit.original is not None else math.nan
    slope = b * median ** (1 - lam) if lam is not None and median > 0 else None
    near = (
        f": about {'+' if (slope or 0) >= 0 else '−'}{fmt(abs(slope))} on `{target}` at its "
        f"median ({fmt(median)})"
        if slope is not None
        else ""
    )
    lam_text = f"λ = {lam:.2f}" if lam is not None else "Box-Cox"
    if level is None:
        return (
            f"Each extra unit of `{term.column}` adds {fmt(b)} to the Box-Cox transform of "
            f"`{target}` ({lam_text}) {ci}{near}, other columns held fixed."
        )
    return (
        f"With `{term.column}` = {level}, the Box-Cox transform of `{target}` ({lam_text}) is "
        f"{fmt(b)} {ci} away from {reference}{near}, other columns held fixed."
    )


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
    if not fit.inputs.out_of_sample:
        return None, "not computed for an alternative"
    if n < 2 * FOLDS:
        return None, "too few rows"
    parts, how = folds(fit.inputs, fit.grouping, n)
    if fit.glm is not None:
        g = fit.glm
        values = g.codes.astype(float) if g.codes is not None else y
        return fm.out_of_sample(g.kind, x, values, parts, g.offset, g.weights), how
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


def _beta(fit: Fitted) -> np.ndarray:
    if fit.glm is not None:
        return fit.glm.beta
    return fit.logit.beta if fit.logit is not None else fit.ls.beta  # type: ignore[union-attr]


def _wald(fit: Fitted) -> tuple[float | None, float | None]:
    beta = _beta(fit)
    intercepts = fit.glm.intercepts if fit.glm is not None else [0]
    keep = [i for i in range(len(beta)) if i not in intercepts]
    if not keep:
        return None, None
    b = beta[keep]
    v = fit.covariance[np.ix_(keep, keep)]
    q = len(keep)
    try:
        f = float(b @ np.linalg.pinv(v) @ b) / q
    except np.linalg.LinAlgError:
        return None, None
    return f, float(stats.f.sf(f, q, fit.df))


def _quality(fit: Fitted) -> RegressionQuality:
    inputs, design = fit.inputs, fit.design
    n, k = design.x.shape
    oos, how = out_of_sample(fit)
    f, f_p = _wald(fit)
    if fit.glm is not None:
        g = fit.glm
        params = len(g.beta)
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
            g.aic,
            g.bic,
            f,
            f_p,
            params - len(g.intercepts),
            n - params,
            g.pseudo_r2,
            g.pseudo_how,
        )
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
    checks.extend(_family_checks(fit))
    separated = (fit.logit is not None and fit.logit.separated) or (
        fit.glm is not None and fit.glm.separated
    )
    if separated:
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
            driver = dg.spread_driver(a.e, a.x)
            if driver is not None and fit.model in ("ols", "wls", "log_ols", "boxcox"):
                column = design.sources.get(design.columns[driver], design.columns[driver])
                checks.append(
                    RegressionCheck(
                        "quantile",
                        "info",
                        "Low and high values",
                        f"The spread follows `{design.terms[driver]}` most: the effects may "
                        f"differ for low and high values of `{inputs.target}`. Quantile "
                        "regression shows it.",
                        "R-REG-12",
                        "robust",
                        column,
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
                design.sources.get(t, t): design.x[:, j]
                for j, t in enumerate(design.terms)
                if j > 0
                and design.levels[j] is None
                and (not design.recipes or design.recipes[j].kind == "num")
            }
            bend = dg.bending(fit.ls.resid, numbers)
            curved = {s.columns[0] for s in inputs.plan.specs if s.kind == "spline"}
            text = f"The effects may not be straight lines (RESET {fmt_p(rs.p_value)})"
            if bend and bend not in curved:
                text += (
                    f": `{bend}` bends the most. Try a curve on it, or see `{inputs.target}` "
                    "against it."
                )
            elif bend:
                text += (
                    f": `{bend}` bends the most, even with its curve. See `{inputs.target}` "
                    "against it."
                )
            else:
                text += "."
            try_curve = bool(bend) and bend not in curved
            checks.append(
                RegressionCheck(
                    "specification",
                    "warn",
                    "Straight lines",
                    text,
                    "R-REG-10",
                    ("term" if try_curve else "compare") if bend else None,
                    bend,
                    term=f"spline:{bend}" if try_curve else None,
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
    vifs, condition = _collinearity(design)
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
        xi = dg.tail_index(fit.ls.resid) if fit.ls is not None else None
        heavy = xi is not None and xi > dg.HEAVY_XI
        if share > INFLUENTIAL_SHARE or outliers or heavy:
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
            else:
                text = "No row stands out by Cook's distance."
            if heavy and xi is not None:
                text += f" The residuals have a heavy tail (Hill ξ = {xi:.2f})."
            text += " Huber's robust fit shows whether they change the coefficients."
            checks.append(
                RegressionCheck("influence", "warn", "Influential rows", text, "R-REG-11", "robust")
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
    checks.extend(_term_checks(fit))
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


def _collinearity(design: Design) -> tuple[list[dg.Vif], float]:
    """VIF per column (067); with curves or interactions (070), a curve read with its column
    (generalized VIF) and the interaction columns left out of the alarm."""
    termed = ("spline:", "square:", "cube:", "interaction:")
    if not design.recipes or not any(g.startswith(termed) for g in design.groups):
        return dg.collinearity(design.x, design.terms)
    owners: list[str | None] = []
    for j, recipe in enumerate(design.recipes):
        if j == 0 or recipe.kind == "product":
            owners.append(None)
        elif recipe.kind in ("spline", "power", "num"):
            owners.append(recipe.variables[0])
        else:
            owners.append(design.terms[j])
    return dg.grouped_collinearity(design.x, design.terms, owners)


def _term_checks(fit: Fitted) -> list[RegressionCheck]:
    """070: a term that could not be made (said, never dropped in silence), and centring when
    two numbers interact (R-REG-TERM-05)."""
    found: list[RegressionCheck] = []
    design, plan = fit.design, fit.inputs.plan
    refused = [n for n in design.notes if n.status == "refused"]
    if refused:
        first = refused[0]
        more = f" (and {_plural(len(refused) - 1, 'other')})" if len(refused) > 1 else ""
        found.append(
            RegressionCheck(
                "terms",
                "warn",
                "Terms",
                f"{first.label}: {first.reason}{more}.",
                first.rule,
                "terms",
            )
        )
    if not plan.center:
        for spec in plan.specs:
            if spec.kind != "interaction":
                continue
            shown = [
                next((n for n in design.kinds if design.sources.get(n, n) == c), c)
                for c in spec.columns
            ]
            if all(design.kinds.get(n, ("cat", []))[0] == "num" for n in shown):
                a, b = shown
                found.append(
                    RegressionCheck(
                        "centering",
                        "info",
                        "Centering",
                        f"`{a}` and `{b}` interact: the effect of each is read where the other "
                        "is 0. Center the numbers to read it at the mean of the other.",
                        "R-REG-TERM-05",
                        "center",
                    )
                )
                break
    return found


def _family_checks(fit: Fitted) -> list[RegressionCheck]:
    """069: what the target says about the family (R-REG-03, 04, 05), and a fit that did not
    converge."""
    if fit.family is None or fit.profile is None:
        return []
    found = [
        RegressionCheck(h.key, h.status, h.title, h.sentence, h.rule, h.action, None, h.family)
        for h in fm.hints(
            fit.profile,
            fit.family.kind,
            fit.inputs.target,
            fit.dispersion,
            fit.glm.aic if fit.glm is not None else None,
            fit.negbin_aic,
        )
    ]
    if fit.glm is not None and not fit.glm.converged:
        found.append(
            RegressionCheck(
                "convergence",
                "warn",
                "Convergence",
                f"The {fit.family.label} fit stopped before converging: read the coefficients "
                "with care, or leave out a column with rare levels.",
            )
        )
    return found


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
    if fit.family is not None:
        reason = fit.family.reason
        if inputs.weights and not _weighted(fit):
            reason += f" The {fit.family.label} model does not use the weights."
        return reason
    return "A number to explain: least squares, in the units of the data."


def _weighted(fit: Fitted) -> bool:
    if fit.glm is not None:
        return fit.glm.weights is not None
    return fit.ls is not None and fit.ls.weights is not None


def _family_info(fit: Fitted) -> FamilyInfo | None:
    """069: the family as the page shows it, with its options and its own numbers."""
    family, p = fit.family, fit.profile
    if family is None or p is None:
        return None
    inputs = fit.inputs
    info = FamilyInfo(
        family.kind,
        family.label,
        family.rule,
        _model_reason(fit),
        family.reads,
        family.requested,
        p.kind,
        fm.options(p),
    )
    if fit.glm is not None:
        info.alpha = fit.glm.alpha
        info.outcomes = fit.glm.outcomes
        if fit.glm.offset is not None:
            info.exposure = inputs.exposure
            info.exposure_reason = inputs.exposure_reason
    if fit.dispersion is not None:
        info.dispersion = fit.dispersion.ratio
    target = inputs.target
    if fit.transform is not None and fit.transform.kind == "log_ols":
        info.smearing = fit.smear
        info.note = (
            f"R², residuals and checks are on log(`{target}`); Duan's smearing factor "
            f"{(fit.smear or 1):.3f} brings predictions back to units."
        )
    elif fit.transform is not None and fit.transform.kind == "boxcox":
        t = fit.transform
        info.boxcox_lambda, info.boxcox_low, info.boxcox_high = t.lam, t.lam_low, t.lam_high
        interval = (
            f" [{t.lam_low:.2f} ; {t.lam_high:.2f}]"
            if t.lam_low is not None and t.lam_high is not None
            else ""
        )
        info.note = (
            f"Box-Cox λ = {(t.lam or 0):.2f}{interval} (profile likelihood): R², residuals and "
            f"checks are on the transform of `{target}`."
        )
    return info


def _result(fit: Fitted) -> RegressionReport:
    inputs, design = fit.inputs, fit.design
    terms = _terms(fit)
    multiple = _adjust(terms)
    references = {
        r.column: (
            f"the average of the `{r.column}` levels"
            if r.coding == "deviation"
            else f"`{r.column}` = {r.level} (the reference)"
        )
        for r in design.references
    }
    for index, term in enumerate(terms):
        term.sentence = _sentence(term, fit, references, multiple.tests, index)
    quality = _quality(fit)
    body = terms[1:]
    chosen = None
    if inputs.focus:
        chosen = next((t for t in body if t.column == inputs.focus), None)
    if chosen is None:
        clear = [
            t
            for t in body
            if t.q_value is not None
            and t.q_value < 0.05
            and t.standardized is not None
            and t.group in (None, t.column)  # 070: a column's own effect, not a curve's part
        ]
        chosen = max(clear, key=lambda t: abs(t.standardized or 0), default=None)
    if chosen is not None:
        sentence = f"{chosen.sentence} {multiple.sentence}"
    elif body:
        sentence = f"No column has a clear effect on `{inputs.target}`. {multiple.sentence}"
    else:
        sentence = f"No column left to explain `{inputs.target}`."
    if fit.family is not None and fit.family.kind != "ols":
        # 069: the family leads the first-plane sentence, with its rule
        why = fit.family.rule or "chosen in the settings"
        sentence = f"{fit.family.label} model ({why}). {sentence}"
    hints = [h for h in [two_way_hint(inputs.shape)] if h]
    left_out = [*inputs.refused, *design.left_out]
    exposure = inputs.exposure if fit.glm is not None and fit.glm.offset is not None else None
    if exposure and any(v.name == exposure for v in inputs.variables):
        left_out.append(
            LeftOutTerm(
                exposure,
                exposure,
                "the exposure: its log is the offset of the count model (R-REG-02)",
            )
        )
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
        inputs.weights if _weighted(fit) else None,
        quality,
        multiple,
        sentence,
        _checks(fit, quality),
        design.references,
        left_out,
        inputs.shape,
        hints,
        _plots(fit),
        _family_info(fit),
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
        likelihood = fit.logit is not None or fit.glm is not None
        if likelihood and kind not in ("classic", "hc0", "hac", "cluster"):
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
        elif fit.glm is not None:
            g = fit.glm
            choice = choose(kind, fit.inputs.shape, fit.grouping, n)
            cov_type, kwds = _logit_cov(kind, fit.grouping)
            if kind == "hac":
                kwds = {"maxlags": cov.newey_west_lag(n)}
            matrix = fm.fit(
                g.kind, fit.design.x, fit.design.y, cov_type, kwds, g.offset, g.weights,
                fit.inputs.outcomes or None,
            ).cov  # fmt: skip
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


def _scale(fit: Fitted) -> str:
    target = fit.inputs.target
    if fit.transform is not None and fit.transform.kind == "log_ols":
        return f"log(`{target}`)"
    if fit.transform is not None and fit.transform.kind == "boxcox":
        return f"Box-Cox transform of `{target}`"
    return f"`{target}`"


def terms_table(fit: Fitted) -> TermsTable:
    design, plan = fit.design, fit.inputs.plan
    vifs, condition = _collinearity(design)
    count = len(fit.result.left_out)
    sentence = f"{_plural(len(design.terms) - 1, 'term')} estimated"
    sentence += f", {_plural(count, 'column')} left out with the reason." if count else "."
    table = TermsTable(design.references, fit.result.left_out, vifs, condition, sentence)
    table.specs = [s.key for s in plan.specs]
    table.center, table.coding, table.notes = plan.center, plan.coding, design.notes
    table.numbers = [
        design.sources.get(n, n) for n, (kind, _) in design.kinds.items() if kind == "num"
    ]
    beta = _beta(fit)
    if design.recipes and len(beta) == design.x.shape[1]:
        reading = tm.read(
            design.groups,
            design.terms,
            design.recipes,
            design.sources,
            design.kinds,
            design.values,
            design.x,
            plan,
            tm.Estimates(beta, fit.covariance, fit.df, fit.ls is not None, _scale(fit)),
        )
        table.tests, table.curves = reading.tests, reading.curves
        table.slopes, table.contrasts = reading.slopes, reading.contrasts
    return table


def partial_plot(fit: Fitted, column: str) -> tm.PartialPlot:
    """070 (R-REG-TERM-11): added-variable and component + residual plots of one column."""
    if fit.ls is None:
        return tm.PartialPlot(column, False, "Partial plots are for least squares.")
    design = fit.design
    return tm.partial(
        column,
        design.recipes,
        design.sources,
        design.kinds,
        design.values,
        design.x,
        design.y,
        fit.ls.beta,
        fit.ls.resid,
        fit.ls.weights,
        fit.inputs.rows[design.used],
        _scale(fit),
    )


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


# --- Other families (Advanced › More, 069) -------------------------------------------------------

MAX_EFFECTS = 12


@dataclass
class AlternativeEffect:
    term: str
    value: float | None  # in the reading of the family: points, % change or units
    low: float | None
    high: float | None


@dataclass
class FamilyAlternative:
    kind: str
    label: str
    available: bool
    reason: str  # what it is for, or why it cannot be fitted
    reads: fm.Reads
    sentence: str = ""
    aic: float | None = None
    pseudo_r2: float | None = None
    r2: float | None = None
    note: str | None = None
    effects: list[AlternativeEffect] = field(default_factory=list)
    error: str | None = None


@dataclass
class FamilyAlternatives:
    current: str
    alternatives: list[FamilyAlternative]
    sentence: str


def _reading(term: RegressionCoefficient, reads: fm.Reads) -> AlternativeEffect:
    if reads == "points":
        scale = [None if v is None else v * 100 for v in (term.marginal, term.marginal_low,
                                                         term.marginal_high)]  # fmt: skip
        return AlternativeEffect(term.term, *scale)
    if reads == "ratio":
        change = [None if v is None else (v - 1) * 100 for v in (term.ratio, term.ratio_low,
                                                                term.ratio_high)]  # fmt: skip
        return AlternativeEffect(term.term, *change)
    return AlternativeEffect(term.term, term.estimate, term.ci_low, term.ci_high)


def family_alternatives(fit: Fitted) -> FamilyAlternatives:
    """The niche families (probit, Box-Cox, fractional logit, multinomial logit) fitted on the
    same rows and columns, each with its strongest effect: computed only when More is opened."""
    current = fit.family.kind if fit.family is not None else fit.model
    p = fit.profile
    found: list[FamilyAlternative] = []
    for option in fm.options(p) if p is not None else []:
        if not option.exotic or option.kind == current:
            continue
        reads = fm.READS[option.kind]
        if not option.available:
            found.append(FamilyAlternative(option.kind, option.label, False, option.reason, reads))
            continue
        alternative = FamilyAlternative(option.kind, option.label, True, option.reason, reads)
        try:
            other = run(replace(fit.inputs, family=option.kind, out_of_sample=False)).result
        except (DesignError, fm.FamilyError, ValueError, np.linalg.LinAlgError) as error:
            alternative.error = str(error)
            found.append(alternative)
            continue
        alternative.sentence = other.sentence
        alternative.aic = other.quality.aic
        alternative.pseudo_r2 = other.quality.pseudo_r2
        alternative.r2 = other.quality.r2
        alternative.note = other.family.note if other.family is not None else None
        alternative.effects = [_reading(t, reads) for t in other.terms if t.column != INTERCEPT][
            :MAX_EFFECTS
        ]
        found.append(alternative)
    fitted = sum(1 for a in found if a.available and a.error is None)
    sentence = (
        f"{_plural(fitted, 'other family')} fitted on the same rows and columns, for comparison."
        if fitted
        else "No other family fits this target."
    )
    return FamilyAlternatives(current, found, sentence)
