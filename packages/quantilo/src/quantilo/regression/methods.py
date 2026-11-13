"""Which methods Advanced › More offers for these data (074, 075, 076), each with the rule that
proposes it and its reason. Cheap: only the first plane's fit and the columns' types and roles
are read; every method is computed when its fold is opened (onion rule)."""

import re
from dataclasses import dataclass, field
from typing import Literal, cast

from quantilo.compare import durations as du
from quantilo.regression import analyze as an
from quantilo.regression import crosssection as xs
from quantilo.regression import mixed as mx
from quantilo.regression import physics as ph
from quantilo.regression import stability as st
from quantilo.regression import survival as sv
from quantilo.regression import varying as vr
from quantilo.typing.types import AnalysisType, Parser

MethodKind = Literal[
    "cross_section", "outputs", "components", "measurement", "curves", "survival", "tobit",
    "mixed", "varying", "regimes",
]  # fmt: skip
UNCERTAINTY_ROLE = "uncertainty"
TITLES: dict[MethodKind, str] = {
    "cross_section": "Fama–MacBeth and IC",
    "outputs": "Several outcomes",
    "components": "Principal components",
    "measurement": "Uncertainties and errors on x",
    "curves": "Curve",
    "survival": "Survival (Cox)",
    "tobit": "Censored (Tobit)",
    "mixed": "Mixed model",
    "varying": "Coefficients that drift (Kalman)",
    "regimes": "Regimes (Markov switching)",
}


@dataclass
class MethodColumn:
    """What the rules read of a column: its name, type and role, as in Columns."""

    name: str
    type: str
    role: str | None
    distinct: int | None = None
    minimum: float | None = None


@dataclass
class MoreMethod:
    kind: MethodKind
    title: str
    proposed: bool  # a rule on the data proposes it
    available: bool
    reason: str  # why it is proposed, what it gives, or why it does not apply
    rule: str


@dataclass
class Pick:
    column: str
    reason: str


@dataclass
class MoreMethods:
    methods: list[MoreMethod]
    numbers: list[str] = field(default_factory=list)  # outcomes and uncertainties to pick from
    events: list[str] = field(default_factory=list)  # yes/no columns
    slopes: list[str] = field(default_factory=list)  # numeric terms a random slope can take
    sigma: Pick | None = None  # R-REG-PHY-01
    sigma_x: Pick | None = None
    event: Pick | None = None  # R-SRV-01
    group: Pick | None = None  # R-REG-MIX-01
    categories: list[str] = field(default_factory=list)  # R-REG-SURV-03: strata to pick from
    start: Pick | None = None  # R-REG-SURV-04


def _words(name: str) -> set[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return set(re.findall(r"[a-z0-9]+", spaced.lower()))


def uncertainty_for(column: str, uncertainties: list[str], fallback: bool) -> Pick | None:
    """R-REG-PHY-01: the uncertainty column whose name takes up the column's, else the only one
    (said so) when `fallback`."""
    own = _words(column)
    for name in uncertainties:
        if own and own <= _words(name):
            return Pick(
                name, f"R-REG-PHY-01: `{name}` has the role uncertainty and names `{column}`."
            )
    if fallback and len(uncertainties) == 1:
        name = uncertainties[0]
        return Pick(
            name,
            f"R-REG-PHY-01: `{name}` is the only column of role uncertainty: read as the "
            f"uncertainty of `{column}`. Check that your data was read correctly.",
        )
    return None


START_WORDS = {"start", "entry", "begin", "tstart", "enter", "entered"}


def start_column(target: MethodColumn, numbers: list[str]) -> Pick | None:
    """R-REG-SURV-04: a numeric column named like the start of an interval (start, entry,
    begin) next to a duration: the rows are intervals (start, duration]."""
    for name in numbers:
        if _words(name) & START_WORDS:
            return Pick(
                name,
                f"R-REG-SURV-04: `{name}` is named like the start of an interval: each row is "
                f"read from `{name}` to `{target.name}`. Check that your data was read correctly.",
            )
    return None


def _duration_column(column: MethodColumn) -> du.DurationColumn:
    kind = cast(AnalysisType, column.type)
    return du.DurationColumn(
        column.name, kind, Parser(), column.role, column.distinct, None, None, column.minimum
    )


def event_column(target: MethodColumn, columns: list[MethodColumn]) -> Pick | None:
    """R-REG-06 / R-SRV-01: the target reads as a duration and a yes/no column as its event."""
    if not du._is_duration(_duration_column(target)):
        return None
    for column in columns:
        if column.name != target.name and du._is_event(_duration_column(column)):
            return Pick(
                column.name,
                f"R-SRV-01: `{target.name}` reads as a duration and `{column.name}` as the event "
                "that ends it (their names). Check that your data was read correctly.",
            )
    return None


def listing(
    fit: an.Fitted, target: MethodColumn, columns: list[MethodColumn], used: set[str]
) -> MoreMethods:
    """The methods of More for this fit; `columns` are the dataset's, `used` the model's."""
    least = fit.ls is not None and fit.model in ("ols", "wls") and fit.transform is None
    numbers = [
        c.name
        for c in columns
        if c.type in ("numeric_continuous", "numeric_discrete")
        and c.name != target.name
        and c.name not in used
    ]
    uncertainties = [c.name for c in columns if c.role == UNCERTAINTY_ROLE and c.name in numbers]
    events = [
        c.name
        for c in columns
        if c.name != target.name and (c.type == "boolean" or (c.distinct == 2 and c.type != "text"))
    ]
    numeric_terms = xs._numeric_terms(fit)
    one_column = ph._numeric_only(fit)
    one = one_column is not None and len(one_column) == 1
    found = MoreMethods(
        [],
        numbers,
        events,
        [fit.design.terms[j] for j in numeric_terms],
    )
    xcol = fit.design.terms[one_column[0]] if one and one_column else None
    found.sigma = uncertainty_for(fit.inputs.target, uncertainties, True)
    if xcol is not None:
        rest = [u for u in uncertainties if not found.sigma or u != found.sigma.column]
        found.sigma_x = uncertainty_for(xcol, rest, False)
    found.event = event_column(target, columns)
    if found.event is not None:
        found.start = start_column(target, numbers)
    found.categories = [
        c.name
        for c in columns
        if c.name != target.name
        and c.type in ("categorical_nominal", "categorical_ordinal", "boolean")
        and (c.distinct is None or c.distinct <= sv.MAX_STRATA)
    ]
    if fit.inputs.cluster:
        found.group = Pick(
            fit.inputs.cluster, f"R-REG-MIX-01: `{fit.inputs.cluster}` is chosen in Grouped by."
        )
    elif fit.inputs.shape.kind == "panel" and fit.inputs.shape.entity:
        entity = fit.inputs.shape.entity
        found.group = Pick(entity, f"R-REG-MIX-01: `{entity}` is the entity of the panel.")
    add = found.methods.append

    # 074
    panel = fit.inputs.shape.kind == "panel"
    add(
        MoreMethod(
            "cross_section",
            TITLES["cross_section"],
            panel and least,
            panel and least,
            (
                f"R-REG-XS-01: a panel of {fit.inputs.shape.entities:,} entities over "
                f"{fit.inputs.shape.periods:,} periods: one cross-section a period, and the IC of "
                "each column."
                if panel
                else "Needs a panel: entities seen over periods (R-REG-XS-01)."
            ),
            "R-REG-XS-01",
        )
    )
    returns = fit.inputs.shape.kind == "series" and least and st._looks_like_returns(fit.design.y)
    add(
        MoreMethod(
            "outputs",
            TITLES["outputs"],
            bool(returns and numbers),
            bool(least and numbers),
            (
                "R-REG-XS-05: the target looks like a return on dated rows: add the other assets' "
                "returns for the GRS test of all the alphas."
                if returns and numbers
                else "R-REG-XS-04: the same columns explain other numeric columns too: coefficient "
                "matrix and joint tests."
                if least and numbers
                else "Needs least squares and another numeric column outside the model."
            ),
            "R-REG-XS-04",
        )
    )
    collinear = least and len(numeric_terms) >= 3 and xs.collinear(fit)
    add(
        MoreMethod(
            "components",
            TITLES["components"],
            collinear,
            least and len(numeric_terms) >= 2,
            (
                "R-REG-XS-06: collinear columns (VIF > 10 or κ > 30) share one direction; on "
                "their principal components the coefficients are stable."
                if collinear
                else "R-REG-XS-06: the numeric columns regressed through their principal "
                "components (parallel analysis, 078)."
                if least and len(numeric_terms) >= 2
                else "Needs least squares and 2 numeric explanatory columns."
            ),
            "R-REG-XS-06",
        )
    )
    # 075
    measured = found.sigma is not None
    add(
        MoreMethod(
            "measurement",
            TITLES["measurement"],
            least and measured,
            least and (bool(numbers) or one_column is not None),
            (
                f"R-REG-PHY-01: `{found.sigma.column}` holds the uncertainty of the target: the "
                "fit weighted by 1/σ², χ² and pulls."
                if least and found.sigma is not None
                else "R-REG-PHY-04: the slope when x carries error too (Deming, York, TLS), and "
                "the fit weighted by an uncertainty column you pick."
                if least
                else "Read on least squares of the target as it is."
            ),
            "R-REG-PHY-01",
        )
    )
    bends = any(c.key == "specification" and c.status == "warn" for c in fit.result.checks)
    add(
        MoreMethod(
            "curves",
            TITLES["curves"],
            least and one and (bends or measured),
            least and one,
            (
                "R-REG-PHY-05: one numeric column"
                + (" and a straight line that misses a bend" if bends else "")
                + ": exponential, power, logistic, Michaelis–Menten and Gaussian peak ranked by "
                "AIC."
                if least and one
                else "A curve takes one numeric explanatory column, with least squares."
            ),
            "R-REG-PHY-05",
        )
    )
    # 076
    timed = found.event is not None
    add(
        MoreMethod(
            "survival",
            TITLES["survival"],
            timed,
            bool(events) and bool(numeric_terms or len(fit.design.terms) > 1),
            (
                "R-REG-06: the target reads as a duration with its event: the Cox model reads the "
                "rows still followed as censored."
                if timed
                else "Pick the yes/no column of the event to read the target as a time to it."
                if events
                else "Needs a yes/no column for the event."
            ),
            "R-REG-06",
        )
    )
    lower, upper = sv.bounds(fit.design.y) if least else (None, None)
    held = lower is not None or upper is not None
    add(
        MoreMethod(
            "tobit",
            TITLES["tobit"],
            held,
            held,
            (
                "R-REG-06: 10 % of the rows or more sit at a bound of the target: Tobit reads them "
                "as censored."
                if held
                else "No bound of the target holds 10 % of the rows (R-REG-TOB-01)."
            ),
            "R-REG-TOB-01",
        )
    )
    mixable = least or fit.model in mx.GENERALISED
    add(
        MoreMethod(
            "mixed",
            TITLES["mixed"],
            mixable and found.group is not None,
            mixable and found.group is not None,
            (
                f"R-REG-MIX-01: rows of one `{found.group.column}` look alike: a random intercept "
                "by group, its share of the variance (ICC)."
                + (
                    " The target keeps its family (R-REG-MIX-02)."
                    if fit.model in mx.GENERALISED
                    else ""
                )
                if mixable and found.group is not None
                else "Set Grouped by (a patient, a school, a store) in How (R-REG-MIX-01)."
                if mixable
                else "Needs least squares, a logistic or a Poisson model (R-REG-MIX-02)."
            ),
            "R-REG-MIX-01",
        )
    )
    # 073: coefficients that move, proposed when R-REG-15 says they change
    dated = st._unavailable(fit) is None
    moving = any(c.key == "stability" and c.status == "warn" for c in fit.result.checks)
    add(
        MoreMethod(
            "varying",
            TITLES["varying"],
            dated and moving,
            dated,
            (
                "R-REG-STAB-05: the coefficients change over the series (R-REG-15): each one let "
                "free to drift as a random walk, estimated by the Kalman smoother."
                if dated and moving
                else "R-REG-STAB-05: each coefficient let free to drift along the dates (Kalman), "
                "tested against fixed coefficients."
                if dated
                else "Needs one series in time fitted by least squares (R-REG-STAB-01)."
            ),
            "R-REG-STAB-05",
        )
    )
    few = vr._regimes_unavailable(fit) is None
    add(
        MoreMethod(
            "regimes",
            TITLES["regimes"],
            few and moving,
            few,
            (
                "R-REG-STAB-07: the coefficients change over the series (R-REG-15): two regimes, "
                "each with its coefficients and the spread of its errors (Markov switching)."
                if few and moving
                else "R-REG-STAB-07: two or three regimes of the coefficients and of the spread "
                "of the errors (Markov switching), a few seconds."
                if few
                else (vr._regimes_unavailable(fit) or "")
            ),
            "R-REG-STAB-07",
        )
    )
    found.methods.sort(key=lambda m: (not m.proposed, not m.available))
    return found
