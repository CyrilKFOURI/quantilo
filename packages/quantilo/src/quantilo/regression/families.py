"""The family of a regression chosen from its target (069), each choice with its rule and reason:
yes/no → logistic (R-REG-01), counts → Poisson with robust errors and an exposure offset
(R-REG-02), overdispersion → negative binomial proposed (R-REG-03), positive and skewed → log(Y)
with Duan's smearing or a Gamma model proposed (R-REG-04), shares → fractional logit proposed
(R-REG-05), a few classes → multinomial logit (R-REG-07).

Pure numerics: the profile of the target, the choice, the fits (statsmodels GLM and discrete
models), average marginal effects by the delta method, Cameron–Trivedi's overdispersion test,
Box-Cox's profile likelihood, the out-of-sample deviance. `analyze` turns them into the report."""

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
from scipy import optimize, special, stats

Family = Literal[
    "ols", "logit", "probit", "poisson", "negbin", "gamma", "log_ols", "boxcox", "fractional",
    "mnlogit",
]  # fmt: skip
FamilyRequest = Literal[
    "auto", "ols", "logit", "probit", "poisson", "negbin", "gamma", "log_ols", "boxcox",
    "fractional", "mnlogit",
]  # fmt: skip
Reads = Literal["units", "points", "ratio"]
TargetKind = Literal["binary", "category", "count", "share", "positive", "number"]

FAMILIES: tuple[Family, ...] = (
    "ols", "logit", "probit", "poisson", "negbin", "gamma", "log_ols", "boxcox", "fractional",
    "mnlogit",
)  # fmt: skip
EXOTIC: tuple[Family, ...] = ("probit", "boxcox", "fractional", "mnlogit")  # Advanced › More
LEAST_SQUARES: tuple[Family, ...] = ("ols", "log_ols", "boxcox")
MLE: tuple[Family, ...] = ("probit", "poisson", "negbin", "gamma", "fractional", "mnlogit")
OFFSET: tuple[Family, ...] = ("poisson", "negbin")  # the exposure enters as log offset
GLM_WEIGHTS: tuple[Family, ...] = ("poisson", "gamma", "fractional")  # var_weights of a GLM
LABELS: dict[Family, str] = {
    "ols": "Least squares (OLS)",
    "logit": "Logistic",
    "probit": "Probit",
    "poisson": "Poisson",
    "negbin": "Negative binomial",
    "gamma": "Gamma (log link)",
    "log_ols": "Log-linear",
    "boxcox": "Box-Cox least squares",
    "fractional": "Fractional logit",
    "mnlogit": "Multinomial logit",
}
READS: dict[Family, Reads] = {
    "ols": "units",
    "logit": "points",
    "probit": "points",
    "fractional": "points",
    "mnlogit": "points",
    "poisson": "ratio",
    "negbin": "ratio",
    "gamma": "ratio",
    "log_ols": "ratio",
    "boxcox": "units",
}

COUNT_DISTINCT = 50  # R-REG-02: fewer distinct values than this …
COUNT_MEAN = 20  # … for a mean below this: counts of events
SKEWED = 1.0  # R-REG-04: |skewness| above this
OVERDISPERSED_P = 0.01  # R-REG-03
OVERDISPERSED_RATIO = 2.0
MIN_CLASSES, MAX_CLASSES = 3, 10  # R-REG-07
EXPOSURE_ROLE = "exposure"
# Whole numbers ≥ 0 that are not counts of events: a rating, a calendar part, an age, a code.
NOT_COUNTS = (
    "rating", "year", "month", "weekday", "hour", "age", "percentage", "angle", "postal_code",
    "latitude", "longitude", "identifier", "propensity",
)  # fmt: skip
LAMBDA_RANGE = (-2.0, 2.0)  # Box-Cox
CERTAIN = 1e-8  # a fitted probability this close to 0 or 1: separation


class FamilyError(ValueError):
    pass


# --- The target ----------------------------------------------------------------------------------


@dataclass
class TargetProfile:
    kind: TargetKind
    rows: int
    mean: float
    variance: float
    skewness: float
    distinct: int
    minimum: float
    maximum: float
    integers: bool
    at_bounds: int  # shares: rows exactly at 0 or 1
    outcomes: list[str] = field(default_factory=list)  # classes, most frequent first
    role: str | None = None


def profile(
    y: np.ndarray,
    binary: tuple[str, str] | None,
    outcomes: Sequence[str] | None = None,
    role: str | None = None,
) -> TargetProfile:
    """What the target looks like over the rows used (y finite)."""
    values = y[np.isfinite(y)]
    n = len(values)
    distinct = len(np.unique(values))
    mean = float(values.mean()) if n else math.nan
    variance = float(values.var(ddof=1)) if n > 1 else math.nan
    skewness = float(stats.skew(values)) if n > 2 and variance > 0 else 0.0
    low = float(values.min()) if n else math.nan
    high = float(values.max()) if n else math.nan
    integers = bool(n and np.all(values == np.round(values)))
    bounds = int(((values == 0) | (values == 1)).sum())
    found = TargetProfile(
        "number", n, mean, variance, skewness, distinct, low, high, integers, 0, [], role
    )
    if binary is not None:
        found.kind = "binary"
        return found
    if outcomes:
        found.kind = "category"
        found.outcomes = list(outcomes)
        return found
    counts = (
        integers
        and low >= 0
        and role not in NOT_COUNTS
        and distinct >= 3
        and (role == "quantity" or (distinct < COUNT_DISTINCT and mean < COUNT_MEAN))
    )
    if counts:
        found.kind = "count"
    elif low >= 0 and high <= 1 and not integers:
        found.kind = "share"
        found.at_bounds = bounds
    elif low > 0:
        found.kind = "positive"
    return found


# --- Which families fit the target ---------------------------------------------------------------


@dataclass
class FamilyOption:
    kind: Family
    label: str
    available: bool
    reason: str  # what it is for, or why it cannot be fitted here
    exotic: bool  # Advanced › More


WHAT: dict[Family, str] = {
    "ols": "effects in the units of the data",
    "logit": "yes/no: effects in points of probability, odds ratios",
    "probit": "yes/no with a normal latent scale: effects in points of probability",
    "poisson": "counts: effects as rate ratios, robust errors",
    "negbin": "counts with extra variance: rate ratios, variance μ + αμ²",
    "gamma": "positive and skewed, variance growing with the mean: effects as %",
    "log_ols": "positive and skewed: effects as %, Duan's smearing back to units",
    "boxcox": "positive: the power transform that makes the residuals most normal",
    "fractional": "shares between 0 and 1, bounds included (Papke–Wooldridge)",
    "mnlogit": "3 to 10 classes: the probability of each class",
}


def _missing(kind: Family, p: TargetProfile) -> str | None:
    """Why `kind` cannot be fitted to this target, None when it can."""
    if p.kind == "category":
        return None if kind == "mnlogit" else "needs a number or a yes/no target"
    if kind in ("logit", "probit"):
        return None if p.kind == "binary" else "needs a yes/no target"
    if p.kind == "binary":
        return None if kind == "ols" else "the target is yes/no"
    if kind in ("poisson", "negbin"):
        return None if p.integers and p.minimum >= 0 else "needs whole numbers ≥ 0"
    if kind in ("gamma", "log_ols", "boxcox"):
        return None if p.minimum > 0 else "needs values above 0"
    if kind == "fractional":
        return None if p.minimum >= 0 and p.maximum <= 1 else "needs values between 0 and 1"
    if kind == "mnlogit":
        if MIN_CLASSES <= p.distinct <= MAX_CLASSES:
            return None
        return f"needs {MIN_CLASSES} to {MAX_CLASSES} distinct values"
    return None


def options(p: TargetProfile) -> list[FamilyOption]:
    found = []
    for kind in FAMILIES:
        missing = _missing(kind, p)
        found.append(
            FamilyOption(kind, LABELS[kind], missing is None, missing or WHAT[kind], kind in EXOTIC)
        )
    return found


@dataclass
class FamilyChoice:
    kind: Family
    label: str
    rule: str | None  # the rule that chose it, None when the user did
    reason: str
    reads: Reads
    requested: FamilyRequest = "auto"


def _rule(p: TargetProfile, target: str, binary: tuple[str, str] | None) -> FamilyChoice:
    if p.kind == "binary" and binary is not None:
        return FamilyChoice(
            "logit",
            LABELS["logit"],
            "R-REG-01",
            f"R-REG-01: `{target}` is yes/no ({binary[0]} / {binary[1]}); a straight line would "
            "leave [0, 1]. Effects read as average changes in probability.",
            "points",
        )
    if p.kind == "category":
        return FamilyChoice(
            "mnlogit",
            LABELS["mnlogit"],
            "R-REG-07",
            f"R-REG-07: `{target}` has {len(p.outcomes)} classes: a multinomial logit gives the "
            f"probability of each against {p.outcomes[0]} (the most frequent). Effects read as "
            "average changes in probability.",
            "points",
        )
    if p.kind == "count":
        why = (
            f"the role {p.role}"
            if p.role == "quantity"
            else f"{p.distinct} distinct values, mean {_short(p.mean)}"
        )
        return FamilyChoice(
            "poisson",
            LABELS["poisson"],
            "R-REG-02",
            f"R-REG-02: `{target}` counts events (whole numbers ≥ 0, {why}). Poisson stays right "
            "even when the variance differs from the mean, as long as the standard errors are "
            "robust. Effects read as rate ratios.",
            "ratio",
        )
    return FamilyChoice(
        "ols",
        LABELS["ols"],
        None,
        "A number to explain: least squares, in the units of the data.",
        "units",
    )


def choose(
    p: TargetProfile, requested: FamilyRequest, target: str, binary: tuple[str, str] | None
) -> FamilyChoice:
    """The rule's family, or the one asked for when the target allows it (else the rule, saying
    why)."""
    rule = _rule(p, target, binary)
    if requested == "auto" or requested == rule.kind:
        if requested != "auto":
            rule.requested = requested
        return rule
    kind = cast(Family, requested)
    missing = _missing(kind, p)
    if missing is not None:
        rule.reason = f"{LABELS[kind]} {missing}: {rule.reason}"
        return rule
    return FamilyChoice(
        kind,
        LABELS[kind],
        None,
        f"{LABELS[kind]} chosen in the settings: {WHAT[kind]}.",
        READS[kind],
        requested,
    )


def _short(value: float) -> str:
    return f"{value:.3g}"


# --- Fits ----------------------------------------------------------------------------------------


@dataclass
class GlmFit:
    """A maximum-likelihood fit. `beta` and `cov` are flat (for the multinomial logit: one
    equation after the other, intercepts at `intercepts`); `marginal` and `marginal_se` are
    aligned with `beta` (NaN at the intercepts)."""

    kind: Family
    beta: np.ndarray
    cov: np.ndarray
    mu: np.ndarray  # fitted mean (probabilities n × J for the multinomial logit)
    llf: float
    llnull: float
    aic: float
    bic: float
    pseudo_r2: float | None
    pseudo_how: str
    marginal: np.ndarray
    marginal_se: np.ndarray
    n: int
    k: int  # columns of the design
    intercepts: list[int]
    outcomes: list[str] = field(default_factory=list)  # multinomial: every class, reference first
    alpha: float | None = None  # negative binomial: variance = μ + αμ²
    dispersion: float | None = None  # Pearson χ² / (n − k)
    separated: bool = False
    converged: bool = True
    offset: np.ndarray | None = None
    weights: np.ndarray | None = None
    codes: np.ndarray | None = None  # multinomial: the class of every row


def _link(kind: Family) -> Literal["logit", "probit", "log"]:
    if kind in ("logit", "fractional", "mnlogit"):
        return "logit"
    return "probit" if kind == "probit" else "log"


def marginal_effects(
    kind: Family, x: np.ndarray, beta: np.ndarray, cov: np.ndarray, offset: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Average marginal effects of the non-intercept columns, dμ/dx averaged over the rows, and
    their covariance by the delta method (as statsmodels' `get_margeff(at="overall")`)."""
    eta = x @ beta + (offset if offset is not None else 0.0)
    link = _link(kind)
    if link == "logit":
        m = special.expit(eta)
        d1 = m * (1 - m)
        d2 = d1 * (1 - 2 * m)
    elif link == "probit":
        d1 = stats.norm.pdf(eta)
        d2 = -eta * d1
    else:
        d1 = np.exp(np.clip(eta, -700, 700))
        d2 = d1
    mean = float(d1.mean())
    k = x.shape[1]
    ame = beta[1:] * mean
    gradient = np.zeros((k - 1, k))
    cross = (d2[:, None] * x).mean(axis=0)
    for j in range(1, k):
        gradient[j - 1] = beta[j] * cross
        gradient[j - 1, j] += mean
    return ame, gradient @ cov @ gradient.T


def _fit_result(result: Any) -> tuple[bool, bool]:
    """(converged, separation warned) of a statsmodels result."""
    retvals = getattr(result, "mle_retvals", None)
    if isinstance(retvals, Mapping):
        return bool(retvals.get("converged", True)), False
    return bool(getattr(result, "converged", True)), False


def mn_codes(y: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Classes of a target as 0..J−1, the most frequent first (the reference)."""
    values, counts = np.unique(y, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    rank = {float(values[i]): r for r, i in enumerate(order)}
    codes = np.array([rank[float(v)] for v in y], dtype=int)
    return codes, [_label(float(values[i])) for i in order]


def _label(value: float) -> str:
    return str(int(value)) if value == round(value) else f"{value:g}"


def fit(
    kind: Family,
    x: np.ndarray,
    y: np.ndarray,
    cov_type: str = "HC0",
    cov_kwds: Mapping[str, object] | None = None,
    offset: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    outcomes: Sequence[str] | None = None,
) -> GlmFit:
    """statsmodels GLM (Poisson, Gamma log link, fractional logit with a binomial QMLE) or
    discrete model (probit, negative binomial NB2, multinomial logit). Robust sandwich by
    default; cluster and HAC by the same rules as least squares."""
    import statsmodels.api as sm

    kwds = dict(cov_kwds or {})
    n, k = x.shape
    with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
        warnings.simplefilter("always")
        if kind in ("poisson", "gamma", "fractional"):
            families = {
                "poisson": sm.families.Poisson(),
                "gamma": sm.families.Gamma(sm.families.links.Log()),
                "fractional": sm.families.Binomial(),
            }
            model = sm.GLM(
                y,
                x,
                family=families[kind],
                offset=offset if kind == "poisson" else None,
                var_weights=weights if kind in GLM_WEIGHTS else None,
            )
            result = model.fit(maxiter=200, cov_type=cov_type, cov_kwds=kwds)
            beta = np.asarray(result.params)
            cov = np.asarray(result.cov_params())
            mu = np.asarray(result.fittedvalues)
            deviance, null = float(result.deviance), float(result.null_deviance)
            pseudo = 1 - deviance / null if null > 0 else None
            how = "deviance explained: 1 − deviance of the model / with the intercept alone"
            llf, llnull = float(result.llf), float(result.llnull)
            aic, bic = float(result.aic), -2 * llf + k * math.log(n)
            converged = bool(getattr(result, "converged", True))
            pearson = float(result.pearson_chi2) / max(n - k, 1)
            found = GlmFit(
                kind, beta, cov, mu, llf, llnull, aic, bic, pseudo, how,
                np.empty(0), np.empty(0), n, k, [0],
                dispersion=pearson, converged=converged,
            )  # fmt: skip
        elif kind == "negbin":
            model = sm.NegativeBinomial(y, x, offset=offset, loglike_method="nb2")
            result = model.fit(disp=0, maxiter=300, cov_type=cov_type, cov_kwds=kwds)
            params = np.asarray(result.params)
            beta, alpha = params[:-1], float(params[-1])
            cov = np.asarray(result.cov_params())[:-1, :-1]
            mu = np.exp(x @ beta + (offset if offset is not None else 0.0))
            llf, llnull = float(result.llf), float(result.llnull)
            found = GlmFit(
                kind, beta, cov, mu, llf, llnull, float(result.aic), float(result.bic),
                float(result.prsquared), "McFadden: 1 − log-likelihood of the model / "
                "log-likelihood with the intercept alone",
                np.empty(0), np.empty(0), n, k, [0], alpha=alpha,
                converged=_fit_result(result)[0],
            )  # fmt: skip
        elif kind == "probit":
            result = sm.Probit(y, x).fit(disp=0, maxiter=200, cov_type=cov_type, cov_kwds=kwds)
            beta = np.asarray(result.params)
            mu = np.asarray(result.predict(x))
            found = GlmFit(
                kind, beta, np.asarray(result.cov_params()), mu, float(result.llf),
                float(result.llnull), float(result.aic), float(result.bic),
                float(result.prsquared), "McFadden: 1 − log-likelihood of the model / "
                "log-likelihood with the intercept alone",
                np.empty(0), np.empty(0), n, k, [0], converged=_fit_result(result)[0],
            )  # fmt: skip
            found.separated = bool(np.any((mu < CERTAIN) | (mu > 1 - CERTAIN)))
        elif kind == "mnlogit":
            if outcomes:
                # classes absent from the rows used drop out; the most frequent stays first
                present = np.unique(y.astype(int))
                remap = {int(c): r for r, c in enumerate(present)}
                codes = np.array([remap[int(v)] for v in y], dtype=int)
                labels = [outcomes[int(c)] for c in present]
            else:
                codes, labels = mn_codes(y)
            if len(labels) < 2:
                raise FamilyError("A single class: nothing to explain.")
            result = sm.MNLogit(codes, x).fit(
                disp=0, maxiter=300, method="newton", cov_type=cov_type, cov_kwds=kwds
            )
            params = np.asarray(result.params)  # k × (J − 1)
            beta = params.ravel(order="F")
            mu = np.asarray(result.predict(x))
            equations = params.shape[1]
            marginal = np.full(len(beta), np.nan)
            marginal_se = np.full(len(beta), np.nan)
            try:
                margins = result.get_margeff(at="overall", method="dydx")
                effects = np.asarray(margins.margeff)  # (k − 1) × J, the reference included
                ses = np.asarray(margins.margeff_se)
                for j in range(equations):
                    marginal[j * k + 1 : (j + 1) * k] = effects[:, j + 1]
                    marginal_se[j * k + 1 : (j + 1) * k] = ses[:, j + 1]
            except (ValueError, np.linalg.LinAlgError):
                pass
            found = GlmFit(
                kind, beta, np.asarray(result.cov_params()), mu, float(result.llf),
                float(result.llnull), float(result.aic), float(result.bic),
                float(result.prsquared), "McFadden: 1 − log-likelihood of the model / "
                "log-likelihood with the intercept alone",
                marginal, marginal_se, n, k, [j * k for j in range(equations)], labels,
                converged=_fit_result(result)[0], codes=codes,
            )  # fmt: skip
            found.separated = bool(np.any(mu.max(axis=1) > 1 - CERTAIN))
        else:
            raise FamilyError(f"{kind} is not a maximum-likelihood family.")
    for warning in caught:
        message = str(warning.message).lower()
        if "separation" in message:
            found.separated = True
        if "converge" in message:
            found.converged = False
    if kind != "mnlogit":
        ame, ame_cov = marginal_effects(kind, x, found.beta, found.cov, offset)
        found.marginal = np.concatenate([[np.nan], ame])
        found.marginal_se = np.concatenate([[np.nan], np.sqrt(np.clip(np.diag(ame_cov), 0, None))])
        if kind == "fractional":
            found.separated = False  # shares at the bounds are expected, not a separation
    found.offset, found.weights = offset, weights
    return found


def predict(
    kind: Family,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    offset_train: np.ndarray | None,
    offset_test: np.ndarray | None,
    weights: np.ndarray | None,
    classes: int = 0,
) -> np.ndarray:
    """The fitted mean of held-out rows (probabilities n × J for the multinomial logit)."""
    found = fit(kind, x_train, y_train, "nonrobust", None, offset_train, weights)
    eta = x_test @ found.beta[: x_test.shape[1]] if kind != "mnlogit" else None
    if kind == "mnlogit":
        params = found.beta.reshape(x_train.shape[1], -1, order="F")
        scores = np.column_stack([np.zeros(len(x_test)), x_test @ params])
        probabilities = special.softmax(scores, axis=1)
        if probabilities.shape[1] < classes:  # a class absent from the training rows
            pad = np.zeros((len(x_test), classes - probabilities.shape[1]))
            probabilities = np.column_stack([probabilities, pad])
        return probabilities
    assert eta is not None
    if offset_test is not None:
        eta = eta + offset_test
    link = _link(kind)
    if link == "logit":
        return special.expit(eta)
    if link == "probit":
        return stats.norm.cdf(eta)
    return np.exp(np.clip(eta, -700, 700))


def deviance(kind: Family, y: np.ndarray, mu: np.ndarray) -> float:
    """Total unit deviance (log loss for the multinomial logit, codes in y)."""
    if kind == "mnlogit":
        rows = np.arange(len(y))
        return float(-2 * np.log(np.clip(mu[rows, y.astype(int)], 1e-12, None)).sum())
    mu = np.clip(mu, 1e-12, None)
    if kind in ("poisson", "negbin"):
        term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
        return float(2 * (term - (y - mu)).sum())
    if kind == "gamma":
        return float(2 * (-np.log(y / mu) + (y - mu) / mu).sum())
    mu = np.clip(mu, 1e-12, 1 - 1e-12)
    yes = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    no = np.where(y < 1, (1 - y) * np.log(np.where(y < 1, 1 - y, 1) / (1 - mu)), 0.0)
    return float(2 * (yes + no).sum())


def out_of_sample(
    kind: Family,
    x: np.ndarray,
    y: np.ndarray,
    parts: list[np.ndarray],
    offset: np.ndarray | None,
    weights: np.ndarray | None,
) -> float | None:
    """1 − deviance on held-out rows / deviance of the training mean (rate × exposure for
    counts with an offset): the R² of a GLM on rows it did not see."""
    n = len(y)
    loss = base = 0.0
    classes = int(y.max()) + 1 if kind == "mnlogit" else 0
    for test in parts:
        train = np.setdiff1d(np.arange(n), test, assume_unique=True)
        if len(train) <= x.shape[1] or not len(test) or len(np.unique(y[train])) < 2:
            continue
        o_train = offset[train] if offset is not None else None
        o_test = offset[test] if offset is not None else None
        w = weights[train] if weights is not None else None
        try:
            mu = predict(kind, x[train], y[train], x[test], o_train, o_test, w, classes)
        except (ValueError, np.linalg.LinAlgError, FamilyError):
            return None
        if kind == "mnlogit":
            shares = np.bincount(y[train].astype(int), minlength=classes) / len(train)
            centre = np.tile(np.clip(shares, 1e-12, None), (len(test), 1))
        elif offset is not None and o_test is not None and o_train is not None:
            rate = y[train].sum() / np.exp(o_train).sum()
            centre = rate * np.exp(o_test)
        else:
            centre = np.full(len(test), float(np.average(y[train], weights=w)))
        loss += deviance(kind, y[test], mu)
        base += deviance(kind, y[test], centre)
    return 1 - loss / base if base > 0 else None


# --- Counts: overdispersion (R-REG-03) -----------------------------------------------------------


@dataclass
class Overdispersion:
    statistic: float  # t of α in the auxiliary regression
    p_value: float  # one-sided, α > 0
    alpha: float  # the auxiliary estimate of α (variance = μ + αμ²)
    ratio: float  # Pearson χ² / (n − k): variance over mean after the model


def overdispersion(y: np.ndarray, mu: np.ndarray, k: int) -> Overdispersion:
    """Cameron & Trivedi (1990): regress ((y − μ)² − y) / μ on μ without a constant; α > 0 means
    the variance grows faster than the mean (NB2)."""
    mu = np.clip(mu, 1e-12, None)
    z = ((y - mu) ** 2 - y) / mu
    sxx = float(mu @ mu)
    alpha = float(z @ mu) / sxx
    n = len(y)
    resid = z - alpha * mu
    s2 = float(resid @ resid) / max(n - 1, 1)
    se = math.sqrt(s2 / sxx) if s2 > 0 else math.inf
    t = alpha / se if se > 0 else 0.0
    p = float(stats.t.sf(t, max(n - 1, 1)))
    ratio = float(((y - mu) ** 2 / mu).sum()) / max(n - k, 1)
    return Overdispersion(t, p, alpha, ratio)


def overdispersed(found: Overdispersion) -> bool:
    return found.p_value < OVERDISPERSED_P and found.ratio > OVERDISPERSED_RATIO


# --- Transformed least squares (R-REG-04, Box-Cox) -----------------------------------------------


@dataclass
class Transform:
    kind: Family
    y: np.ndarray  # the transformed target
    lam: float | None = None  # Box-Cox λ
    lam_low: float | None = None
    lam_high: float | None = None


def boxcox_values(y: np.ndarray, lam: float) -> np.ndarray:
    return np.log(y) if abs(lam) < 1e-6 else (y**lam - 1) / lam


def boxcox_lambda(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, float | None, float | None]:
    """λ maximising the profile likelihood of the regression of (y^λ − 1)/λ on x, with its 95 %
    likelihood-ratio interval: −n/2 log(SSE(λ)/n) + (λ − 1) Σ log y."""
    root = np.sqrt(weights) if weights is not None else np.ones(len(y))
    q, _ = np.linalg.qr(x * root[:, None])
    logs = float(np.log(y).sum())
    n = len(y)
    # The geometric mean keeps y^λ in range for large values
    scale = math.exp(logs / n)
    z = y / scale

    def profile_llf(lam: float) -> float:
        v = boxcox_values(z, lam) * root
        e = v - q @ (q.T @ v)
        sse = max(float(e @ e), 1e-300)
        return -n / 2 * math.log(sse / n) + (lam - 1) * float(np.log(z).sum())

    grid = np.linspace(*LAMBDA_RANGE, 81)
    values = np.array([profile_llf(float(g)) for g in grid])
    best = int(np.argmax(values))
    low_edge = grid[max(best - 1, 0)]
    high_edge = grid[min(best + 1, len(grid) - 1)]
    found = optimize.minimize_scalar(
        lambda lam: -profile_llf(lam), bounds=(float(low_edge), float(high_edge)), method="bounded"
    )
    lam = float(cast(Any, found).x)
    top = profile_llf(lam)
    cut = top - stats.chi2.ppf(0.95, 1) / 2

    def edge(a: float, b: float) -> float | None:
        fa, fb = profile_llf(a) - cut, profile_llf(b) - cut
        if fa * fb > 0:
            return None
        root = optimize.brentq(lambda v: profile_llf(v) - cut, a, b)
        return float(cast(Any, root))

    return lam, edge(LAMBDA_RANGE[0], lam), edge(lam, LAMBDA_RANGE[1])


def transform(
    kind: Family, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> Transform:
    if kind == "log_ols":
        return Transform(kind, np.log(y))
    if kind == "boxcox":
        lam, low, high = boxcox_lambda(x, y, weights)
        return Transform(kind, boxcox_values(y, lam), lam, low, high)
    return Transform(kind, y)


def smearing(resid: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Duan (1983): E[y | x] = exp(x b) × mean(exp(residuals)) on log(y)."""
    return float(np.average(np.exp(resid), weights=weights))


def boxcox_inverse(value: float, lam: float) -> float:
    if abs(lam) < 1e-6:
        return math.exp(value)
    base = lam * value + 1
    return base ** (1 / lam) if base > 0 else math.nan


# --- Hints on the target (R-REG-03, 04, 05) ------------------------------------------------------


@dataclass
class FamilyHint:
    key: str
    status: Literal["ok", "warn", "bad", "info"]
    title: str
    sentence: str
    rule: str | None
    action: str | None  # "family" (switch) or "more" (Advanced › More)
    family: str | None


def hints(
    p: TargetProfile,
    current: Family,
    target: str,
    dispersion: Overdispersion | None = None,
    poisson_aic: float | None = None,
    negbin_aic: float | None = None,
) -> list[FamilyHint]:
    found: list[FamilyHint] = []
    if current == "poisson" and dispersion is not None:
        if overdispersed(dispersion):
            compare = ""
            if negbin_aic is not None and poisson_aic is not None:
                better = "better" if negbin_aic < poisson_aic else "not better"
                compare = (
                    f" A negative binomial fits {better} (AIC {_aic(negbin_aic)} against "
                    f"{_aic(poisson_aic)})."
                )
            found.append(
                FamilyHint(
                    "overdispersion",
                    "warn",
                    "Extra variance",
                    f"`{target}` varies {dispersion.ratio:.1f} × more than its mean once the "
                    f"columns are in (Cameron–Trivedi p {_p(dispersion.p_value)}). The robust "
                    "errors keep the Poisson intervals right." + compare,
                    "R-REG-03",
                    "family",
                    "negbin",
                )
            )
        else:
            if dispersion.p_value < OVERDISPERSED_P:
                text = (
                    f"The test finds a little extra variance (Cameron–Trivedi p "
                    f"{_p(dispersion.p_value)}), but `{target}` varies only "
                    f"{dispersion.ratio:.1f} × its mean once the columns are in: the robust "
                    "errors are enough."
                )
            else:
                text = (
                    f"The variance of `{target}` stays close to its mean once the columns are in "
                    f"({dispersion.ratio:.1f} ×, Cameron–Trivedi p {_p(dispersion.p_value)})."
                )
            found.append(
                FamilyHint(
                    "overdispersion",
                    "ok",
                    "Extra variance",
                    text,
                    "R-REG-03",
                    None,
                    None,
                )
            )
    if current == "ols" and p.kind == "positive" and abs(p.skewness) > SKEWED:
        found.append(
            FamilyHint(
                "skewed_target",
                "info",
                "Skewed target",
                f"`{target}` is positive and skewed (skewness {p.skewness:.1f}): on "
                f"log(`{target}`) the effects read as %, and Duan's smearing brings predictions "
                "back to units. A Gamma model with a log link is the alternative.",
                "R-REG-04",
                "family",
                "log_ols",
            )
        )
    if current == "ols" and p.kind == "share":
        bounds = (
            f", {p.at_bounds:,} rows exactly at 0 or 1" if p.at_bounds else ", none at the bounds"
        )
        found.append(
            FamilyHint(
                "share_target",
                "info",
                "Share between 0 and 1",
                f"`{target}` is a share between 0 and 1{bounds}: a straight line can leave "
                "[0, 1]. A fractional logit keeps the predictions inside and the bounds "
                "(Papke–Wooldridge), see Advanced › More.",
                "R-REG-05",
                "more",
                "fractional",
            )
        )
    return found


def _aic(value: float) -> str:
    return f"{value:,.0f}"


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.2g}"
