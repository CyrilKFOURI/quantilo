"""Who should get which variant (spec 055 § B, R-PL1 to R-PL8): on a randomized experiment, a
depth 1-3 policy tree grown on doubly robust (AIPW) scores, its value measured by
cross-validation against giving everyone the same variant, and the Qini curve.

The input is one row per unit: the variant label (`VARIANT`), the metric (`VALUE`) and the
covariates by name (service.py reads and aggregates it). The rule is a hypothesis to confirm
with a new test, never a result.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from quantilo.experiments import policy_core as core
from quantilo.experiments.analyze import VARIANT, _gap, _value
from quantilo.experiments.policy_offer import (  # noqa: F401 (the module's public names)
    MIN_COVARIATES,
    MIN_UNITS,
    VALUE,
    PolicyOffer,
    offer,
)
from quantilo.ml.data import Field as DataField
from quantilo.typing.types import Parser

CAVEAT = (
    "Hypothesis to confirm with a new test: the rule was learnt on this test's units; covariates "
    "must be measured before the test, not changed by it."
)
METHOD = (
    "Doubly robust (AIPW) scores from outcome models cross-fitted on 5 folds (gradient boosting, "
    "one per variant) and the observed share of each variant as its propensity; a greedy tree of "
    "depth 0 (one variant for all) to 3 maximizing the summed scores, the depth chosen by "
    "cross-validation "
    "(one-standard-error rule); each unit judged by the rule learnt without it."
)

PolicyKind = Literal["rate", "mean", "count"]
Verdict = Literal["worth_testing", "no_better"]


@dataclass
class PolicyLeafView:
    conditions: str  # "" = everyone
    variant: str
    units: int
    share: float


@dataclass
class PolicyValue:
    key: str  # "rule" or the variant
    label: str  # "The rule", "Everyone on control"
    value: float | None
    low: float | None
    high: float | None


@dataclass
class PolicyGain:
    versus: str  # the variant everyone would get
    diff: float | None
    low: float | None
    high: float | None


@dataclass
class PolicyQini:
    variant: str
    control: str
    shares: list[float]
    gains: list[float]
    random: list[float]
    auuc: float
    sentence: str


@dataclass
class PolicyLearning:
    metric: str
    kind: PolicyKind
    variants: list[str]
    control: str
    units: int
    sampled: bool  # the units are a reproducible sample (R-PC6)
    covariates: list[core.PolicyCovariate]
    depth: int
    depth_values: list[float]  # out-of-fold value at depth 0 (one variant for all), 1, 2, 3
    rule: str
    leaves: list[PolicyLeafView]
    values: list[PolicyValue]  # the rule, then everyone on each variant
    gains: list[PolicyGain]  # the rule against everyone on each variant
    best_constant: str
    verdict: Verdict
    sentence: str
    qini: list[PolicyQini]  # empty until asked ("Advanced › More")
    caveat: str
    method: str
    draws: int


@dataclass
class PolicyUnavailable:
    reason: str


def _per_unit(kind: str, value: float | None) -> str:
    return _value(kind, value)  # type: ignore[arg-type]


def learn(
    data: Any,
    fields: list[DataField],
    metric: str,
    kind: PolicyKind,
    order: list[str],
    sampled: bool = False,
    alpha: float = 0.05,
    qini: bool = False,
) -> PolicyLearning | PolicyUnavailable:
    """`order`: the variants, the control first. `data`: one row per unit. The Qini curve is
    "Advanced › More" (055, onion rule): computed only when asked."""
    found = scores(data, fields, kind, order)
    if isinstance(found, PolicyUnavailable):
        return found
    learnt = learn_scores(found, metric, sampled, alpha)
    if qini:
        learnt.qini = qini_curves(found)
    return learnt


@dataclass
class PolicyScores:
    """What the rule and the Qini curve are computed from (kept by the service, so that opening
    the Qini curve later costs no new cross-fitting)."""

    kind: PolicyKind
    order: list[str]
    covariates: core.Covariates
    mu: np.ndarray
    gamma: np.ndarray
    fold: np.ndarray


def scores(
    data: Any, fields: list[DataField], kind: PolicyKind, order: list[str]
) -> PolicyScores | PolicyUnavailable:
    """R-PL1 to R-PL3, R-PL9 and R-PC1/R-PC2: the units, their covariates and their doubly
    robust scores."""
    data = data.dropna(subset=[VALUE])
    data = data[data[VARIANT].isin(order)].reset_index(drop=True)
    counts = data[VARIANT].value_counts()
    units = {label: int(counts.get(label, 0)) for label in order}
    checked = offer(kind, [f.name for f in fields], units)
    if not checked.available:
        return PolicyUnavailable(checked.reason or "")
    y = data[VALUE].to_numpy(float)
    binary = kind == "rate"
    target = DataField(VALUE, "boolean" if binary else "numeric_continuous", Parser())
    frame = data.assign(**{VALUE: y})
    changed = core.imbalanced(frame, fields, VARIANT)
    kept = [f for f in fields if f.name not in changed]
    covs = core.covariates(frame, kept, target, binary)
    covs.columns = [
        *(c for c in covs.columns),
        *(core.PolicyCovariate(name, False, reason) for name, reason in changed.items()),
    ]
    position_of = {f.name: i for i, f in enumerate(fields)}
    covs.columns.sort(key=lambda c: position_of.get(c.column, len(position_of)))
    if len(covs.used) < MIN_COVARIATES:
        left = "; ".join(f"{c.column}: {c.reason}" for c in covs.columns if not c.used)
        return PolicyUnavailable(
            f"Fewer than {MIN_COVARIATES} usable covariates once the preparation rules are "
            f"applied ({left})."
        )
    position = {label: i for i, label in enumerate(order)}
    actions = data[VARIANT].map(position).to_numpy(int)
    k = len(order)
    fold = core.folds(len(y))
    mu = core.outcomes(covs.x, actions, y, k, fold, binary)
    shares = np.bincount(actions, minlength=k) / len(actions)
    gamma = core.aipw(mu, shares[actions], actions, y)
    return PolicyScores(kind, order, covs, mu, gamma, fold)


def learn_scores(
    found: PolicyScores, metric: str, sampled: bool = False, alpha: float = 0.05
) -> PolicyLearning:
    """R-PL4 to R-PL6: the rule, its cross-validated value against everyone on each variant."""
    kind, order, covs, gamma, fold = (
        found.kind,
        found.order,
        found.covariates,
        found.gamma,
        found.fold,
    )
    k = len(order)
    total = len(gamma)
    chosen = core.choose_depth(covs.x, gamma, covs.features, fold)
    tree = core.grow(covs.x, gamma, covs.features, chosen.depth)
    found_leaves = core.leaves(tree, covs.features)
    rule = core.rule_sentence(found_leaves, order)
    leaf_views = [
        PolicyLeafView(leaf.conditions, order[leaf.action], leaf.units, leaf.units / total)
        for leaf in found_leaves
    ]

    rows = np.arange(total)
    rule_values = gamma[rows, chosen.actions]
    columns = np.column_stack([rule_values, gamma, rule_values[:, None] - gamma])
    samples = core.bootstrap(columns)
    low, high = core.interval(samples, alpha)
    means = columns.mean(axis=0)
    values = [PolicyValue("rule", "The rule", *_triple(means, low, high, 0))]
    gains = []
    for i, label in enumerate(order):
        values.append(PolicyValue(label, f"Everyone on {label}", *_triple(means, low, high, 1 + i)))
        gains.append(PolicyGain(label, *_triple(means, low, high, 1 + k + i)))
    best = int(np.argmax(means[1 : 1 + k]))
    against = gains[best]
    verdict: Verdict = "worth_testing" if (against.low or 0) > 0 else "no_better"
    points = " points" if kind == "rate" else ""
    level = f"{(1 - alpha) * 100:.0f} %"
    head = (
        f"The rule reaches {_per_unit(kind, means[0])} vs {_per_unit(kind, means[1 + best])} for "
        f"everyone on {order[best]}: {_gap(kind, against.diff)}{points}, {level} CI "
        f"[{_gap(kind, against.low)} ; {_gap(kind, against.high)}] (cross-validated)."
    )
    single = chosen.depth == 0 or len({leaf.action for leaf in found_leaves}) == 1
    if single:
        verdict = "no_better"
        head = (
            f"No rule beats giving everyone {order[best]} "
            f"({_per_unit(kind, means[1 + best])}): cross-validation keeps one variant for all."
        )
        tail = ""
    elif verdict == "worth_testing":
        tail = " Worth a new test."
    elif (against.high or 0) < 0:
        tail = f" Worse than giving everyone {order[best]}: no rule found here beats it."
    else:
        tail = " No better than giving everyone the same variant: the interval includes 0."
    return PolicyLearning(
        metric,
        kind,
        order,
        order[0],
        total,
        sampled,
        covs.columns,
        chosen.depth,
        chosen.values,
        rule,
        leaf_views,
        values,
        gains,
        order[best],
        verdict,
        head + tail,
        [],
        CAVEAT,
        METHOD,
        core.DRAWS,
    )


def qini_curves(found: PolicyScores) -> list[PolicyQini]:
    """R-PL7: each variant against the control, units ranked by their out-of-fold predicted
    gain."""
    kind, order, mu, gamma = found.kind, found.order, found.mu, found.gamma
    points = " points" if kind == "rate" else ""
    curves = []
    for i in range(1, len(order)):
        shares_q, gain_q, random_q, auuc = core.qini(gamma, mu[:, i] - mu[:, 0], i, 0)
        sentence = (
            f"Treating first the units with the largest predicted gain from {order[i]}: the area "
            f"between the curve and random targeting is {_gap(kind, auuc)}{points} per unit "
            "(0 = the ranking is no better than chance)."
        )
        curves.append(PolicyQini(order[i], order[0], shares_q, gain_q, random_q, auuc, sentence))
    return curves


def _triple(
    means: np.ndarray, low: np.ndarray, high: np.ndarray, i: int
) -> tuple[float | None, float | None, float | None]:
    return core.finite(means[i]), core.finite(low[i]), core.finite(high[i])
