"""Coefficients that move (073, second part), in Advanced › More: each coefficient let free to
drift as a random walk, estimated by the Kalman filter and smoother (the state-space model of
064, a statsmodels `MLEModel`), and regimes of the coefficients by a Markov-switching regression
(`MarkovRegression`, as the regimes of a series in 062). Rules R-REG-STAB-05…08.

Both are proposed when the first plane's check R-REG-15 says the coefficients change; both are
computed only when their fold is opened (onion rule). numpy / scipy / statsmodels on one thread:
Pyodide-compatible."""

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import special, stats
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.statespace.mlemodel import MLEModel

from quantilo.regression import analyze as an
from quantilo.regression import stability as st
from quantilo.regression.design import INTERCEPT

MAX_KALMAN_ROWS = 1_000  # R-REG-STAB-06: the variances are estimated on the last rows
MAX_REGIME_ROWS = 1_000  # R-REG-STAB-07, as R-RG-04 in 062
MIN_REGIME_ROWS = 40
MAX_REGIME_TERMS = 4  # R-REG-STAB-07: 2 × (k + 1) parameters a regime stay estimable
MIN_SHARE = 0.02  # R-REG-STAB-07: a smaller regime fits a few points
NARROW = 50.0  # R-REG-STAB-07: σ ratio beyond which a regime is degenerate
MAX_CURVES = st.MAX_CURVES
Z95 = 1.959963984540054


# --- Kalman: coefficients as random walks -----------------------------------------------------


class Drifting(MLEModel):
    """y_t = x_t′β_t + ε_t, β_t = β_{t−1} + η_t, ε ~ N(0, σ²), η ~ N(0, diag(q)); exact diffuse
    start, so that q = 0 is recursive least squares (= OLS at the last date)."""

    def __init__(self, y: np.ndarray, x: np.ndarray) -> None:
        n, k = x.shape
        super().__init__(y, k_states=k, k_posdef=k, initialization="diffuse")
        self.k = k
        self["design"] = x.T[np.newaxis, :, :]
        self["transition"] = np.eye(k)
        self["selection"] = np.eye(k)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        resid = y - x @ beta
        self.s2 = float(resid @ resid) / max(n - k, 1)
        spread = np.var(x, axis=0)
        spread[spread == 0] = 1.0  # the intercept
        self._start = np.r_[self.s2, self.s2 * 0.01 / spread]

    @property
    def param_names(self) -> list[str]:
        return ["sigma2.error", *(f"sigma2.{j}" for j in range(self.k))]

    @property
    def start_params(self) -> np.ndarray:
        return self._start

    def transform_params(self, unconstrained: np.ndarray) -> np.ndarray:
        return np.asarray(unconstrained, dtype=float) ** 2

    def untransform_params(self, constrained: np.ndarray) -> np.ndarray:
        return np.sqrt(np.clip(np.asarray(constrained, dtype=float), 0, None))

    def update(
        self,
        params: Any,
        transformed: bool = True,
        includes_fixed: bool = False,
        complex_step: bool = False,
    ) -> Any:
        params = super().update(params, transformed, includes_fixed, complex_step)
        self["obs_cov", 0, 0] = params[0]
        self["state_cov"] = np.diag(params[1:])
        return params


def _quiet(compute: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            return compute()


def drift_test(lr: float, k: int) -> float:
    """R-REG-STAB-06: q = 0 sits on the edge of the parameter space, so the likelihood ratio
    follows a mixture of χ²: weights C(k, j)/2ᵏ on χ²_j (Self & Liang 1987), the variances taken
    as independent."""
    if lr <= 0:
        return 1.0
    total = 0.0
    for j in range(1, k + 1):
        total += float(special.comb(k, j)) / 2**k * float(stats.chi2.sf(lr, j))
    return min(1.0, total)


def kalman_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The variances (σ², q₁…q_k) by maximum likelihood, the log-likelihood, and that of fixed
    coefficients (q = 0, σ² = SSR/(n − k), its maximum: the recursive residuals are the n − k
    innovations, each of variance σ² f_t)."""
    model = Drifting(y, x)
    found: Any = _quiet(lambda: model.fit(disp=False, maxiter=200, method="lbfgs"))
    params = np.asarray(found.params, dtype=float)
    fixed = np.r_[model.s2, np.zeros(x.shape[1])]
    restricted = float(_quiet(lambda: model.loglike(fixed)))
    free = float(found.llf)
    if not math.isfinite(free) or free < restricted:  # the optimum is at the edge
        return fixed, restricted, restricted
    return params, free, restricted


@dataclass
class DriftCurve:
    term: str
    dates: list[str]
    estimates: list[float | None]
    low: list[float | None]
    high: list[float | None]
    fixed: float  # the first plane's coefficient
    smallest: float | None
    largest: float | None
    step: float  # σ of the random walk: how much the coefficient moves a period
    sentence: str = ""


@dataclass
class VaryingTable:
    available: bool
    reason: str
    date: str | None = None
    rows: int = 0
    fitted_rows: int = 0  # the rows the variances are estimated on (R-REG-STAB-06)
    curves: list[DriftCurve] = field(default_factory=list)
    shown: int = 0  # terms before the cut to MAX_CURVES
    lr: float | None = None
    p_value: float | None = None
    drifts: bool = False
    sigma: float | None = None  # σ of the errors with drifting coefficients
    fixed_sigma: float | None = None
    sentence: str = ""
    notes: list[str] = field(default_factory=list)


def _own(fit: an.Fitted) -> list[int]:
    """The terms shown: each own column (not an indicator of a category beyond its group),
    strongest first, the intercept last (it moves with the level, rarely the question)."""
    k = fit.design.x.shape[1]
    groups = fit.design.groups
    se = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
    assert fit.ls is not None
    beta = fit.ls.beta
    own = [
        j
        for j in range(k)
        if fit.design.columns[j] != INTERCEPT and (not groups or groups[j] == fit.design.columns[j])
    ]
    own.sort(key=lambda j: -abs(beta[j] / se[j]) if se[j] > 0 else 0)
    return own


def varying(fit: an.Fitted) -> VaryingTable:
    """Advanced › More › Coefficients that drift (Kalman), R-REG-STAB-05, 06."""
    missing = st._unavailable(fit)
    if missing is not None:
        return VaryingTable(False, missing)
    assert fit.ls is not None and fit.inputs.dates is not None
    dates = np.asarray(fit.inputs.dates)[fit.design.used]
    x, y = st._arrays(fit)
    n, k = x.shape
    tail = slice(max(0, n - MAX_KALMAN_ROWS), n)
    params, free, restricted = kalman_fit(x[tail], y[tail])
    model = Drifting(y, x)
    smoothed: Any = _quiet(lambda: model.smooth(params))
    states = np.asarray(smoothed.smoothed_state, dtype=float)  # k × n
    spread = np.asarray(smoothed.smoothed_state_cov, dtype=float)  # k × k × n
    lr = max(0.0, 2 * (free - restricted))
    p = drift_test(lr, k)
    table = VaryingTable(
        True,
        "",
        fit.inputs.shape.date,
        n,
        tail.stop - tail.start,
        lr=lr,
        p_value=p,
        drifts=p < st.ALPHA,
        sigma=math.sqrt(max(float(params[0]), 0)),
        fixed_sigma=math.sqrt(max(Drifting(y[tail], x[tail]).s2, 0)),
    )
    keep = st._thin(n)
    keep = keep[keep >= k]  # the diffuse start: the first k dates hold no estimate yet
    own = _own(fit)
    table.shown = len(own)
    for j in own[:MAX_CURVES]:
        values = states[j]
        se = np.sqrt(np.clip(spread[j, j], 0, None))
        path = values[k:]
        curve = DriftCurve(
            fit.design.terms[j],
            [str(d) for d in dates[keep]],
            [st._clean(float(values[i])) for i in keep],
            [st._clean(float(values[i] - Z95 * se[i])) for i in keep],
            [st._clean(float(values[i] + Z95 * se[i])) for i in keep],
            float(fit.ls.beta[j]),
            float(path.min()) if len(path) else None,
            float(path.max()) if len(path) else None,
            math.sqrt(max(float(params[1 + j]), 0)),
        )
        curve.sentence = _curve_sentence(curve)
        table.curves.append(curve)
    table.sentence = _varying_sentence(table, k)
    if table.fitted_rows < n:
        table.notes.append(
            f"R-REG-STAB-06: the variances are estimated on the last {table.fitted_rows:,} "
            f"rows, then the smoother runs over all {n:,} with them (cost)."
        )
    table.notes.append(
        "Each coefficient may drift as a random walk; the curve is its estimate with all the "
        "rows (smoothed: it uses future values), the band its 95 % interval."
    )
    if fit.ls.weights is not None:
        table.notes.append(
            f"The rows are weighted by `{fit.inputs.weights}`, as in the first plane."
        )
    return table


def _curve_sentence(curve: DriftCurve) -> str:
    low, high = curve.smallest, curve.largest
    if low is None or high is None:
        return f"`{curve.term}`: too few rows."
    if curve.step <= 1e-12 * max(abs(curve.fixed), 1.0):
        return f"The effect of `{curve.term}` does not drift: {an.fmt(curve.fixed)} throughout."
    text = (
        f"Let free to drift, the effect of `{curve.term}` goes from {an.fmt(low)} to "
        f"{an.fmt(high)} ({an.fmt(curve.fixed)} fixed), moving by about {an.fmt(curve.step)} a row"
    )
    return text + (": it changes sign." if low < 0 < high else ".")


def _varying_sentence(table: VaryingTable, k: int) -> str:
    test = f"likelihood ratio {an.fmt(table.lr)}, {an.fmt_p(table.p_value)}"
    if table.drifts:
        return (
            f"Coefficients that drift fit the series better than fixed ones ({test}, "
            f"{k} variances tested at 0, R-REG-STAB-06): read the curves below rather than one "
            "number."
        )
    return (
        f"Letting the coefficients drift does not fit clearly better ({test}, R-REG-STAB-06): "
        "the fixed coefficients of the first plane hold."
    )


# --- Markov switching: regimes of the coefficients --------------------------------------------


@dataclass
class RegimeTerm:
    term: str
    estimate: float
    std_error: float | None
    ci_low: float | None
    ci_high: float | None


@dataclass
class MarkovRegime:
    name: str
    share: float  # of the dates, by smoothed probability
    duration: float | None  # periods: 1 / (1 − p_ii)
    sigma: float
    terms: list[RegimeTerm]
    last: float  # probability at the last date


@dataclass
class RegimesTable:
    available: bool
    reason: str
    date: str | None = None
    rows: int = 0
    fitted_rows: int = 0
    regimes: list[MarkovRegime] = field(default_factory=list)
    k_regimes: int = 2
    dates: list[str] = field(default_factory=list)
    probabilities: list[list[float]] = field(default_factory=list)  # regime × date
    bic: float | None = None
    bic_one: float | None = None
    worth: bool = False
    switching: bool = True  # each regime has its own spread of the errors (R-REG-STAB-07)
    focus: str | None = None  # the strongest effect, quoted in the sentence
    sentence: str = ""
    notes: list[str] = field(default_factory=list)


def _markov(
    y: np.ndarray, exog: np.ndarray | None, k_regimes: int, constant: bool, switching: bool
) -> tuple[Any, Any, bool] | None:
    model = MarkovRegression(
        y,
        k_regimes=k_regimes,
        trend="c" if constant else "n",
        exog=exog,
        switching_variance=switching,
    )
    try:
        found: Any = _quiet(lambda: model.fit(disp=False))
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not math.isfinite(float(found.bic)):
        return None
    return model, found, switching


def _degenerate(fitted: tuple[Any, Any, bool], k_regimes: int) -> bool:
    """R-REG-STAB-07: a regime that holds less than 2 % of the rows, or whose errors are 50
    times narrower than another's, is the likelihood fitting a few points exactly: dropped."""
    model, found, switching = fitted
    probs = np.asarray(found.smoothed_marginal_probabilities, dtype=float)
    if probs.mean(axis=0).min() < MIN_SHARE:
        return True
    if not switching:
        return False
    names = list(model.param_names)
    params = np.asarray(found.params, dtype=float)
    spreads = [float(params[names.index(f"sigma2[{r}]")]) for r in range(k_regimes)]
    return min(spreads) <= 0 or math.sqrt(max(spreads) / min(spreads)) > NARROW


def _regimes_unavailable(fit: an.Fitted) -> str | None:
    missing = st._unavailable(fit)
    if missing is not None:
        return missing
    own = [j for j, c in enumerate(fit.design.columns) if c != INTERCEPT]
    if len(own) > MAX_REGIME_TERMS:
        return (
            f"{len(own)} terms: regimes are estimated for {MAX_REGIME_TERMS} terms at most, as "
            "each regime has its own coefficients (R-REG-STAB-07). Keep the main columns."
        )
    return None


def regimes(fit: an.Fitted, k_regimes: int = 2) -> RegimesTable:
    """Advanced › More › Regimes (Markov switching), R-REG-STAB-07, 08."""
    missing = _regimes_unavailable(fit)
    if missing is not None:
        return RegimesTable(False, missing)
    assert fit.ls is not None and fit.inputs.dates is not None
    dates = np.asarray(fit.inputs.dates)[fit.design.used]
    x, y = st._arrays(fit)
    n = len(y)
    start = max(0, n - MAX_REGIME_ROWS)
    x, y, dates = x[start:], y[start:], dates[start:]
    m = len(y)
    if m < MIN_REGIME_ROWS:
        return RegimesTable(False, f"{m} rows: regimes need {MIN_REGIME_ROWS} at least.")
    k_regimes = 3 if k_regimes == 3 else 2
    constant = fit.design.columns[0] == INTERCEPT
    exog = x[:, 1:] if constant else x
    scale_y = float(np.std(y)) or 1.0
    scale_x = np.std(exog, axis=0) if exog.shape[1] else np.zeros(0)
    scale_x[scale_x == 0] = 1.0
    fits = [
        _markov(y / scale_y, exog / scale_x if exog.shape[1] else None, k_regimes, constant, both)
        for both in (True, False)
    ]
    usable = [f for f in fits if f is not None and not _degenerate(f, k_regimes)]
    if not usable:
        return RegimesTable(False, "The Markov-switching fit did not converge on these rows.")
    model, found, switching = min(usable, key=lambda f: float(f[1].bic))
    names = list(model.param_names)
    params = np.asarray(found.params, dtype=float)
    bse = np.asarray(found.bse, dtype=float)
    probs = np.asarray(found.smoothed_marginal_probabilities, dtype=float)  # m × regimes
    durations = np.asarray(found.expected_durations, dtype=float).ravel()
    order = list(fit.design.terms)  # the intercept first, as `const` then x1… in statsmodels
    own = _own(fit)
    focus = fit.design.terms[own[0]] if own else None
    raw: list[MarkovRegime] = []
    for r in range(k_regimes):
        rows: list[RegimeTerm] = []
        for i, term in enumerate(order):
            label = f"const[{r}]" if constant and i == 0 else f"x{i if constant else i + 1}[{r}]"
            at = names.index(label)
            unit = scale_y if constant and i == 0 else scale_y / scale_x[i - 1 if constant else i]
            b, se = params[at] * unit, bse[at] * unit
            good = math.isfinite(se) and se > 0
            rows.append(
                RegimeTerm(
                    term,
                    float(b),
                    float(se) if good else None,
                    float(b - Z95 * se) if good else None,
                    float(b + Z95 * se) if good else None,
                )
            )
        label = f"sigma2[{r}]" if switching else "sigma2"
        sigma = math.sqrt(max(float(params[names.index(label)]), 0)) * scale_y
        duration = (
            float(durations[r]) if r < len(durations) and math.isfinite(durations[r]) else None
        )
        raw.append(
            MarkovRegime("", float(probs[:, r].mean()), duration, sigma, rows, float(probs[-1, r]))
        )
    if switching:
        ranked = sorted(range(k_regimes), key=lambda r: raw[r].sigma)
        labels = ["calm", "turbulent"] if k_regimes == 2 else ["calm", "moderate", "turbulent"]
    else:  # one spread: the regimes differ by their coefficients, ranked by the focus effect
        at = order.index(focus) if focus in order else len(order) - 1
        ranked = sorted(range(k_regimes), key=lambda r: raw[r].terms[at].estimate)
        labels = ["low", "high"] if k_regimes == 2 else ["low", "middle", "high"]
    for rank, r in enumerate(ranked):
        raw[r].name = labels[rank]
    keep = st._thin(m)
    table = RegimesTable(
        True,
        "",
        fit.inputs.shape.date,
        n,
        m,
        [raw[r] for r in ranked],
        k_regimes,
        [str(d) for d in dates[keep]],
        [[float(v) for v in probs[keep, r]] for r in ranked],
    )
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    s2 = float(resid @ resid) / m
    ll_one = -0.5 * m * (math.log(2 * math.pi * s2) + 1)
    table.bic_one = float(-2 * ll_one + (x.shape[1] + 1) * math.log(m))
    # the BIC on the scale of y: the fit ran on y / sd(y), its density differs by that factor
    table.bic = float(found.bic) + 2 * m * math.log(scale_y)
    table.worth = table.bic < table.bic_one
    table.switching = switching
    table.focus = focus
    table.sentence = _regimes_sentence(table)
    if start > 0:
        table.notes.append(
            f"R-REG-STAB-07: the last {m:,} rows of {n:,} are fitted (cost, as R-RG-04)."
        )
    table.notes.append(
        (
            "R-REG-STAB-07: each regime has its own spread of the errors (lower BIC than one "
            "spread); regimes named by that spread."
            if switching
            else "R-REG-STAB-07: one spread of the errors for all regimes (lower BIC than one "
            f"each); regimes named by the effect of `{focus or INTERCEPT}`."
        )
        + " A regime's duration is 1 / (1 − its probability of staying), in rows; the "
        "probabilities use all the rows (smoothed)."
    )
    return table


def _regimes_sentence(table: RegimesTable) -> str:
    focus = table.focus
    parts = []
    for regime in table.regimes:
        effect = next((t for t in regime.terms if t.term == focus), None)
        rows = round(regime.duration) if regime.duration else 0
        lasts = f", about {rows} row{'s' if rows != 1 else ''} at a time" if rows else ""
        coefficient = f", `{focus}` {an.fmt(effect.estimate)}" if focus and effect else ""
        spread = f", σ {an.fmt(regime.sigma)}" if table.switching else ""
        parts.append(f"{regime.name} ({regime.share:.0%} of the rows{lasts}{coefficient}{spread})")
    verdict = (
        f"BIC {table.bic:,.0f} against {table.bic_one:,.0f} for one set of coefficients: the "
        "regimes are worth their parameters"
        if table.worth and table.bic is not None and table.bic_one is not None
        else f"BIC {table.bic or 0:,.0f} against {table.bic_one or 0:,.0f} for one set of "
        "coefficients: the regimes do not pay for their parameters"
    )
    last = max(table.regimes, key=lambda r: r.last)
    return (
        f"{len(table.regimes)} regimes: " + "; ".join(parts) + f". {verdict} (R-REG-STAB-08). "
        f"The last row is {last.name} (probability {last.last:.0%})."
    )
