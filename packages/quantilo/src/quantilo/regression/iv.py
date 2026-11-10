"""Instrumental variables (071): two-stage least squares written in numpy, only when the user
names the endogenous column and its instruments (R-REG-IV-01, never guessed). Standard errors by
the same rule as the first plane, on the predicted columns (R-REG-IV-02); the strength of the
instruments (first-stage F, Montiel Olea–Pflueger effective F, R-REG-IV-03); overidentification
(Hansen J, Sargan, R-REG-IV-04); endogeneity (Wu–Hausman by control function, R-REG-IV-05);
the Anderson–Rubin test, valid with weak instruments (R-REG-IV-06)."""

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import stats

from quantilo.regression import covariance as cov
from quantilo.regression.analyze import Inputs, Status, fmt, fmt_p
from quantilo.regression.covariance import LABELS, Arrays, Kind
from quantilo.regression.design import INTERCEPT, Design, LeftOutTerm, MissingRows, build
from quantilo.regression.rules import Grouping, SeChoice, choose

EFFECTIVE_CRITICAL = 23.1  # Montiel Olea–Pflueger simplified, τ = 10 %, 5 %, one instrument
RULE_OF_THUMB = 10.0  # Staiger–Stock (1997), Stock–Yogo (2005)
LEVEL = 0.05


class IvError(ValueError):
    pass


@dataclass
class IvCoefficient:
    term: str
    column: str
    endogenous: bool
    estimate: float
    std_error: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    ols: float | None = None  # least squares on the same rows, for comparison
    ols_low: float | None = None
    ols_high: float | None = None


@dataclass
class FirstStage:
    term: str  # the endogenous term
    f: float | None  # Wald of the excluded instruments / q, with the chosen standard errors
    classic_f: float | None
    effective_f: float | None  # Montiel Olea–Pflueger, one endogenous term only
    partial_r2: float | None
    df1: int
    df2: int
    p_value: float | None
    status: Status
    sentence: str


@dataclass
class IvTest:
    key: str
    title: str
    statistic: float | None
    df: int | None
    df2: int | None
    p_value: float | None
    status: Status
    sentence: str
    rule: str | None = None


@dataclass
class ReducedForm:
    term: str  # an instrument term
    estimate: float
    std_error: float | None
    p_value: float | None


@dataclass
class IvTable:
    target: str
    endogenous: list[str]
    instruments: list[str]
    instrument_terms: list[str]
    rows: int
    se: SeChoice
    coefficients: list[IvCoefficient]
    first_stages: list[FirstStage]
    tests: list[IvTest]
    reduced_form: list[ReducedForm]
    left_out: list[LeftOutTerm]
    missing: list[MissingRows]
    sentence: str
    notes: list[str] = field(default_factory=list)


# --- Linear algebra ------------------------------------------------------------------------------


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, Arrays]:
    q, r = np.linalg.qr(x)
    beta = np.linalg.solve(r, q.T @ y)
    inverse = np.linalg.solve(r, np.eye(r.shape[0]))
    e = y - x @ beta
    h = np.einsum("ij,ij->i", q, q)
    return beta, e, Arrays(x, e, inverse @ inverse.T, h)


def _covariance(kind: Kind, a: Arrays, grouping: Grouping) -> np.ndarray:
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


def _df(choice: SeChoice, n: int, k: int) -> int:
    if choice.kind in ("cluster", "cluster2", "dk") and choice.clusters:
        return max(choice.clusters - 1, 1)
    return max(n - k, 1)


def _wald(beta: np.ndarray, v: np.ndarray) -> float | None:
    try:
        found = float(beta @ np.linalg.solve(v, beta))
    except np.linalg.LinAlgError:
        return None
    return found if math.isfinite(found) else None


def _meat(z: np.ndarray, e: np.ndarray, kind: Kind, grouping: Grouping) -> np.ndarray:
    """Σ of the moment contributions zᵢeᵢ, robust as the standard errors (Hansen J)."""
    u = z * e[:, None]
    if kind in ("cluster", "cluster2", "dk"):
        codes = (
            grouping.cluster_codes if grouping.cluster_codes is not None else grouping.entity_codes
        )
        if kind == "dk" or codes is None:
            codes = grouping.period_codes
        if codes is not None:
            groups, count = cov.codes_of(codes)
            sums = np.empty((count, u.shape[1]))
            for j in range(u.shape[1]):
                sums[:, j] = np.bincount(groups, weights=u[:, j], minlength=count)
            return sums.T @ sums
    if kind == "hac":
        lag = cov.newey_west_lag(len(e))
        s = u.T @ u
        for lagged in range(1, min(lag, len(u) - 1) + 1):
            gamma = u[lagged:].T @ u[:-lagged]
            s += (1 - lagged / (lag + 1)) * (gamma + gamma.T)
        return s
    return u.T @ u


# --- The estimate --------------------------------------------------------------------------------


def _design(inputs: Inputs) -> tuple[Design, Grouping, np.ndarray | None]:
    """The design of the augmented columns (explanatory + instruments), rows complete for all,
    as the first plane builds it."""
    extra: dict[str, np.ndarray] = {}
    weights = inputs.weight_values
    if weights is not None and inputs.weights:
        weights = np.where(weights > 0, weights, np.nan)
        extra[inputs.weights] = weights
    if inputs.cluster_codes is not None and inputs.cluster:
        extra[inputs.cluster] = inputs.cluster_codes
    panel = inputs.shape.kind == "panel"
    if panel and inputs.entity_codes is not None and inputs.shape.entity:
        extra[inputs.shape.entity] = inputs.entity_codes
    if panel and inputs.period_codes is not None and inputs.shape.period:
        extra[inputs.shape.period] = inputs.period_codes
    design = build(inputs.y, inputs.variables, extra)
    for item in design.missing:
        if item.column == "target":
            item.column = inputs.target
    used = design.used

    def pick(values: np.ndarray | None) -> np.ndarray | None:
        return values[used] if values is not None else None

    grouping = Grouping(
        inputs.cluster,
        pick(inputs.cluster_codes),
        pick(inputs.entity_codes) if panel else None,
        pick(inputs.period_codes) if panel else None,
    )
    w = pick(weights) if inputs.weights else None
    return design, grouping, w


def _interval(
    b: float, se: float | None, df: int
) -> tuple[float | None, float | None, float | None]:
    if se is None or not math.isfinite(se) or se <= 0:
        return None, None, None
    half = float(stats.t.ppf(0.975, df)) * se
    return b - half, b + half, float(2 * stats.t.sf(abs(b / se), df))


def _names(items: list[str]) -> str:
    quoted = [f"`{item}`" for item in items]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + f" and {quoted[-1]}"


def estimate(inputs: Inputs, endogenous: list[str], instruments: list[str]) -> IvTable:
    if inputs.outcomes:
        raise IvError(
            "IV here is linear: a category with several classes has no mean to explain; pick a "
            "number (069, R-REG-07)."
        )
    if inputs.binary is not None:
        raise IvError(
            "IV here is linear: for a yes/no target it would be a linear probability model; "
            "pick a number to explain."
        )
    design, grouping, w = _design(inputs)
    missing_columns = [name for name in instruments if name not in design.columns]
    refused = {item.column: item.reason for item in inputs.refused}
    for name in missing_columns:
        if name in refused:
            raise IvError(f"`{name}` cannot be an instrument: {refused[name]}.")
    terms, columns = design.terms, design.columns
    exogenous = [j for j, c in enumerate(columns) if c not in instruments and c not in endogenous]
    endo = [j for j, c in enumerate(columns) if c in endogenous]
    excluded = [j for j, c in enumerate(columns) if c in instruments]
    left_out = [*inputs.refused, *design.left_out]
    if not endo:
        lost = [d for d in design.left_out if d.column in endogenous]
        why = f": {lost[0].reason}" if lost else ""
        raise IvError(f"{_names(endogenous)} left the design{why}.")
    if not excluded:
        lost = [d for d in design.left_out if d.column in instruments]
        why = f" ({lost[0].reason})" if lost else ""
        raise IvError(
            f"No instrument left: {_names(instruments)} adds nothing beyond the explanatory "
            f"columns{why}."
        )
    if len(excluded) < len(endo):
        raise IvError(
            f"{len(endo)} endogenous terms but {len(excluded)} instrument terms: IV needs at "
            "least as many instruments as endogenous terms (order condition)."
        )
    y = design.y
    x_all = design.x
    if w is not None:
        root = np.sqrt(w)
        y, x_all = y * root, x_all * root[:, None]
    regressors = sorted(exogenous + endo)
    x = x_all[:, regressors]
    z = x_all[:, exogenous + excluded]
    x_ex = x_all[:, exogenous]
    n, k = x.shape
    q = len(excluded)
    if n <= z.shape[1] + 1:
        raise IvError("Fewer rows than instruments and columns: nothing to estimate.")

    # Standard errors by the first-plane rule, HC2/HC3 → HC1 (R-REG-IV-02).
    choice = choose(inputs.requested, inputs.shape, grouping, n)
    notes: list[str] = []
    if choice.kind in ("hc2", "hc3"):
        choice = replace(
            choice,
            kind="hc1",
            label=LABELS["hc1"],
            reason=choice.reason + " IV uses HC1: the leverage of HC2/HC3 belongs to least "
            "squares, not to two-stage least squares (R-REG-IV-02).",
        )
    kind = choice.kind
    df = _df(choice, n, k)

    # First stage and 2SLS.
    pi, _, _ = _ols(z, x)
    x_hat = z @ pi
    beta, _, a_hat = _ols(x_hat, y)
    e = y - x @ beta
    a = Arrays(x_hat, e, a_hat.bread, a_hat.h)
    v_iv = _covariance(kind, a, grouping)
    ses_iv = np.sqrt(np.clip(np.diag(v_iv), 0, None))
    beta_ols, _, a_ols = _ols(x, y)
    v_ols = _covariance(kind, a_ols, grouping)
    ses_ols = np.sqrt(np.clip(np.diag(v_ols), 0, None))
    coefficients: list[IvCoefficient] = []
    for j, index in enumerate(regressors):
        b, se = float(beta[j]), float(ses_iv[j])
        low, high, p = _interval(b, se, df)
        o_low, o_high, _ = _interval(float(beta_ols[j]), float(ses_ols[j]), df)
        coefficients.append(
            IvCoefficient(
                terms[index],
                columns[index],
                index in endo,
                b,
                se if se > 0 else None,
                low,
                high,
                p,
                float(beta_ols[j]),
                o_low,
                o_high,
            )
        )

    # Strength of the instruments (R-REG-IV-03).
    first_stages: list[FirstStage] = []
    position = list(range(len(exogenous), len(exogenous) + q))
    residuals = np.empty((n, len(endo)))
    z_tilde = z[:, position]
    if x_ex.shape[1]:
        z_tilde = z_tilde - x_ex @ np.linalg.lstsq(x_ex, z_tilde, rcond=None)[0]
    df_first = _df(choice, n, z.shape[1])
    for m, index in enumerate(endo):
        target = x_all[:, index]
        coef, v, arrays = _ols(z, target)
        residuals[:, m] = v
        robust = _covariance(kind, arrays, grouping)
        wald = _wald(coef[position], robust[np.ix_(position, position)])
        f = wald / q if wald is not None else None
        p = float(stats.f.sf(f, q, df_first)) if f is not None else None
        sse_full = float(v @ v)
        if x_ex.shape[1]:
            _, r, _ = _ols(x_ex, target)
            sse_restricted = float(r @ r)
        else:
            sse_restricted = float(target @ target)
        partial = 1 - sse_full / sse_restricted if sse_restricted > 0 else None
        classic_f = (
            ((sse_restricted - sse_full) / q) / (sse_full / max(n - z.shape[1], 1))
            if sse_full > 0
            else None
        )
        effective = None
        if len(endo) == 1:
            effective = _effective_f(z_tilde, target, x_ex, kind, grouping)
        first_stages.append(
            _first_stage(terms[index], f, classic_f, effective, partial, q, df_first, p, len(endo))
        )

    tests: list[IvTest] = []
    tests.append(_endogeneity(x, y, residuals, kind, grouping, choice, [terms[i] for i in endo]))
    reduced, anderson = _anderson_rubin(
        z, y, position, kind, grouping, df_first, q, excluded, terms
    )
    tests.append(anderson)
    if q > len(endo):
        tests.append(_overidentification(x, z, y, e, kind, grouping, q - len(endo)))
    else:
        tests.append(
            IvTest(
                "overid",
                "Overidentification",
                None,
                None,
                None,
                None,
                "info",
                "As many instruments as endogenous terms: their exclusion cannot be tested; it "
                "rests on what you know of the data (R-REG-IV-04).",
                "R-REG-IV-04",
            )
        )
    if w is not None:
        notes.append(f"Rows weighted by `{inputs.weights}` in every stage.")
    sentence = _conclusion(inputs.target, coefficients, first_stages, tests, instruments)
    return IvTable(
        inputs.target,
        list(dict.fromkeys(columns[i] for i in endo)),
        list(instruments),
        [terms[i] for i in excluded],
        n,
        choice,
        coefficients,
        first_stages,
        tests,
        reduced,
        left_out,
        design.missing,
        sentence,
        notes,
    )


def _effective_f(
    z_tilde: np.ndarray, x: np.ndarray, x_ex: np.ndarray, kind: Kind, grouping: Grouping
) -> float | None:
    """Montiel Olea & Pflueger (2013): F_eff = π̂′Qπ̂ / tr(Σ̂_π Q), Q = Z̃′Z̃, on the instruments
    and the endogenous column cleaned of the exogenous ones (Frisch–Waugh)."""
    x_tilde = x - x_ex @ np.linalg.lstsq(x_ex, x, rcond=None)[0] if x_ex.shape[1] else x
    coef, _, arrays = _ols(z_tilde, x_tilde)
    sigma = _covariance(kind, arrays, grouping)
    gram = z_tilde.T @ z_tilde
    denominator = float(np.trace(sigma @ gram))
    if denominator <= 0:
        return None
    return float(coef @ gram @ coef) / denominator


def _first_stage(
    term: str,
    f: float | None,
    classic_f: float | None,
    effective: float | None,
    partial: float | None,
    q: int,
    df2: int,
    p: float | None,
    endogenous: int,
) -> FirstStage:
    strength = effective if effective is not None else f
    status: Status
    if strength is None:
        status, verdict = "info", "cannot be measured"
    elif effective is not None and effective >= EFFECTIVE_CRITICAL:
        status, verdict = "ok", "strong"
    elif strength >= RULE_OF_THUMB:
        status = "warn" if effective is not None else "ok"
        verdict = "usable but not strong" if effective is not None else "above the rule of thumb"
    else:
        status, verdict = "bad", "weak"
    if effective is not None:
        text = (
            f"Instruments for `{term}`: effective F = {fmt(effective)} ({verdict}; Montiel Olea–"
            f"Pflueger, robust; {EFFECTIVE_CRITICAL} keeps the bias under 10 % of OLS's), "
            f"first-stage F = {fmt(f)} (rule of thumb > {RULE_OF_THUMB:g}), partial R² "
            f"{fmt(partial)}."
        )
    else:
        text = (
            f"Instruments for `{term}`: first-stage F = {fmt(f)} ({verdict}; rule of thumb > "
            f"{RULE_OF_THUMB:g}), partial R² {fmt(partial)}. With {endogenous} endogenous terms, "
            "the effective F is not defined: read each first stage."
        )
    if status == "bad":
        text += (
            " Weak instruments pull IV towards OLS and make its intervals too narrow: read the "
            "Anderson–Rubin test."
        )
    return FirstStage(term, f, classic_f, effective, partial, q, df2, p, status, text)


def _endogeneity(
    x: np.ndarray,
    y: np.ndarray,
    residuals: np.ndarray,
    kind: Kind,
    grouping: Grouping,
    choice: SeChoice,
    names: list[str],
) -> IvTest:
    """R-REG-IV-05: control function; the first-stage residuals added to least squares."""
    augmented = np.column_stack([x, residuals])
    beta, _, arrays = _ols(augmented, y)
    v = _covariance(kind, arrays, grouping)
    tested = list(range(x.shape[1], augmented.shape[1]))
    q = len(tested)
    wald = _wald(beta[tested], v[np.ix_(tested, tested)])
    df2 = _df(choice, len(y), augmented.shape[1])
    if wald is None:
        return IvTest(
            "endogeneity",
            "Endogeneity (Wu–Hausman)",
            None,
            q,
            df2,
            None,
            "info",
            "The endogeneity test cannot be computed.",
            "R-REG-IV-05",
        )
    f = wald / q
    p = float(stats.f.sf(f, q, df2))
    who = _names(names)
    if p < LEVEL:
        status: Status = "warn"
        text = (
            f"{who} is endogenous (Wu–Hausman F = {fmt(f)}, {fmt_p(p)}): least squares is "
            "biased; keep IV."
        )
    else:
        status = "ok"
        text = (
            f"No evidence that {who} is endogenous (Wu–Hausman F = {fmt(f)}, {fmt_p(p)}): least "
            "squares is more precise and stays the reference."
        )
    return IvTest(
        "endogeneity", "Endogeneity (Wu–Hausman)", f, q, df2, p, status, text, "R-REG-IV-05"
    )


def _anderson_rubin(
    z: np.ndarray,
    y: np.ndarray,
    position: list[int],
    kind: Kind,
    grouping: Grouping,
    df2: int,
    q: int,
    excluded: list[int],
    terms: list[str],
) -> tuple[list[ReducedForm], IvTest]:
    """R-REG-IV-06: the reduced form y on the instruments; its Wald test is Anderson–Rubin's."""
    beta, _, arrays = _ols(z, y)
    v = _covariance(kind, arrays, grouping)
    ses = np.sqrt(np.clip(np.diag(v), 0, None))
    reduced = []
    for j, index in zip(position, excluded, strict=True):
        se = float(ses[j])
        _, _, p = _interval(float(beta[j]), se, df2)
        reduced.append(ReducedForm(terms[index], float(beta[j]), se if se > 0 else None, p))
    wald = _wald(beta[position], v[np.ix_(position, position)])
    if wald is None:
        return reduced, IvTest(
            "anderson_rubin",
            "Anderson–Rubin",
            None,
            q,
            df2,
            None,
            "info",
            "The Anderson–Rubin test cannot be computed.",
            "R-REG-IV-06",
        )
    f = wald / q
    p = float(stats.f.sf(f, q, df2))
    if p < LEVEL:
        text = (
            f"The instruments move the target directly in the reduced form (Anderson–Rubin "
            f"F = {fmt(f)}, {fmt_p(p)}): the endogenous effect is not zero, a conclusion that "
            "holds even with weak instruments."
        )
    else:
        text = (
            f"The reduced form shows no effect (Anderson–Rubin F = {fmt(f)}, {fmt_p(p)}): no "
            "evidence that the endogenous effect differs from zero, whatever the instruments' "
            "strength."
        )
    return reduced, IvTest(
        "anderson_rubin", "Anderson–Rubin", f, q, df2, p, "info", text, "R-REG-IV-06"
    )


def _overidentification(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    kind: Kind,
    grouping: Grouping,
    df: int,
) -> IvTest:
    """R-REG-IV-04: Hansen J from two-step GMM (robust weight from the 2SLS residuals), Sargan
    n R² of the 2SLS residuals on the instruments."""
    n = len(y)
    _, rest, _ = _ols(z, e)
    centred = e - e.mean()
    r2 = 1 - float(rest @ rest) / float(centred @ centred) if centred @ centred > 0 else 0.0
    sargan = n * r2
    sargan_p = float(stats.chi2.sf(sargan, df))
    j_stat, j_p = None, None
    try:
        weight = np.linalg.inv(_meat(z, e, kind, grouping) / n)
        zx, zy = z.T @ x / n, z.T @ y / n
        beta = np.linalg.solve(zx.T @ weight @ zx, zx.T @ weight @ zy)
        moments = z.T @ (y - x @ beta) / n
        j_stat = float(n * moments @ weight @ moments)
        j_p = float(stats.chi2.sf(j_stat, df))
    except np.linalg.LinAlgError:
        pass
    robust = kind != "classic" and j_stat is not None
    statistic, p = (j_stat, j_p) if robust else (sargan, sargan_p)
    name = "Hansen J" if robust else "Sargan"
    other = f"; Sargan {fmt_p(sargan_p)}" if robust else ""
    if p is not None and p < LEVEL:
        status: Status = "bad"
        text = (
            f"At least one instrument looks invalid ({name} = {fmt(statistic)}, {fmt_p(p)}"
            f"{other}): the instruments disagree on the effect, so one of them reaches the "
            "target by another path."
        )
    else:
        status = "ok"
        text = (
            f"The instruments agree with each other ({name} = {fmt(statistic)}, {fmt_p(p)}"
            f"{other}). The test assumes at least one of them is valid."
        )
    return IvTest(
        "overid",
        f"Overidentification ({name})",
        statistic,
        df,
        None,
        p,
        status,
        text,
        "R-REG-IV-04",
    )


def _conclusion(
    target: str,
    coefficients: list[IvCoefficient],
    first_stages: list[FirstStage],
    tests: list[IvTest],
    instruments: list[str],
) -> str:
    endogenous = [c for c in coefficients if c.endogenous and c.term != INTERCEPT]
    if not endogenous:
        return "No endogenous term left."
    lead = endogenous[0]
    text = (
        f"Instrumenting `{lead.term}` with {_names(instruments)}: each extra unit of "
        f"`{lead.term}` moves `{target}` by {fmt(lead.estimate)} [{fmt(lead.ci_low)} ; "
        f"{fmt(lead.ci_high)}] (least squares: {fmt(lead.ols)})."
    )
    weak = [s for s in first_stages if s.status == "bad"]
    if weak:
        text += " The instruments are weak: trust the Anderson–Rubin test more than the interval."
    endogeneity = next((t for t in tests if t.key == "endogeneity"), None)
    if endogeneity is not None and endogeneity.p_value is not None:
        text += (
            " Least squares is biased here."
            if endogeneity.p_value < LEVEL
            else " No evidence of endogeneity: least squares stays the more precise reading."
        )
    return text
