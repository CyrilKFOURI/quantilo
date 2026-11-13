"""Evaluate a policy from logged decisions (spec 055 § C, R-PO1 to R-PO7): off-policy evaluation
of candidate policies (always one action, the learnt rule, a saved ML model) on a log of
decisions, with the DM, IPS, SNIPS and DR estimators (Dudík, Langford & Li 2011), the
diagnostics first.

The input is one row per decision: the action taken (`ACTION`), its reward (`REWARD`), the
logged propensity (`PROPENSITY`, optional) and the covariates by name (service.py reads it).
Never called "reinforcement learning": decisions, policies.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantilo.experiments import policy_core as core
from quantilo.experiments.analyze import _gap, _value
from quantilo.experiments.decisions import (  # noqa: F401 (the module's public names)
    ACTION,
    ACTION_HINT,
    ACTION_ROLE,
    LOW_ESS,
    LOW_PROPENSITY,
    MAX_ACTIONS,
    MIN_ACTIONS,
    PROPENSITY,
    PROPENSITY_ROLE,
    REWARD,
    REWARD_HINT,
    DecisionsRequest,
    LoggedDecisions,
    detect,
)
from quantilo.interpret.matching import MatchingOffer
from quantilo.ml.data import Field as DataField
from quantilo.typing.types import Parser

OBSERVATIONAL = (
    "Observational: valid only if no hidden factor drove the decisions. No propensity column, so "
    "the probability of each decision is estimated from the covariates."
)
METHOD = (
    "Reward models cross-fitted on 5 folds (gradient boosting, one per action); weights = the "
    "policy's probability of the logged action ÷ its logged propensity (floored at 0.01); "
    "intervals by a seeded Poisson bootstrap."
)

Estimator = Literal["dm", "ips", "snips", "dr"]
PolicySource = Literal["logged", "constant", "learned", "model"]


# --- What the evaluation returns ----------------------------------------------------------------


@dataclass
class EstimatorValue:
    estimator: Estimator
    value: float | None
    low: float | None
    high: float | None


@dataclass
class PolicyEstimate:
    key: str
    label: str
    source: PolicySource
    description: str
    estimates: list[EstimatorValue]  # DM, IPS, SNIPS, DR (the logged policy: its mean)
    value: float | None  # DR (the logged policy: the observed mean)
    low: float | None
    high: float | None
    gain: float | None  # against the logged decisions (DR)
    gain_low: float | None
    gain_high: float | None
    ess: float | None  # effective sample size of the weights
    ess_share: float | None
    reliable: bool
    clipped: int  # weights whose propensity was floored
    choices: dict[str, float]  # share of the units it gives each action
    sentence: str


@dataclass
class ActionOverlap:
    action: str
    logged_share: float
    rare_share: float  # units where its (estimated) propensity is below 0.05


@dataclass
class DecisionsDiagnostics:
    propensity: Literal["column", "estimated"]
    min_propensity: float | None
    below: float  # share of the units whose logged propensity is below 0.05
    clipped: int  # rows whose propensity was floored at 0.01
    dropped: int  # rows left out: propensity outside ]0, 1]
    overlap: list[ActionOverlap]
    sentences: list[str]


@dataclass
class PolicyEvaluation:
    action: str
    reward: str
    kind: Literal["rate", "mean"]
    actions: list[str]
    units: int
    sampled: bool
    covariates: list[core.PolicyCovariate]
    diagnostics: DecisionsDiagnostics
    policies: list[PolicyEstimate]
    rule: str | None  # the learnt rule, in words
    warnings: list[str]
    skipped: list[str]  # models that cannot act as a policy, with why
    method: str
    draws: int
    # 123: whether matched groups (FLAME, "Advanced") can compare the two actions (R-MAT-01)
    matching: MatchingOffer | None = None


@dataclass
class CandidatePolicy:
    """A policy given from outside (a saved model): each unit's probability of each action."""

    key: str
    label: str
    description: str
    probabilities: np.ndarray


@dataclass
class EvaluationUnavailable:
    reason: str


# --- Policies from a saved model (R-PO5) ---------------------------------------------------------


def model_policy(
    predict: Any,
    task: str,
    target: str,
    labels: list[str],
    positive: str | None,
    inputs: list[str],
    data: pd.DataFrame,
    action: str,
    actions: list[str],
    reward: str,
) -> np.ndarray:
    """A saved model as a policy: its target is the action (the action it predicts), or the
    reward with the action among its inputs (the action that maximizes its prediction).
    `predict(frame)`: regression → numbers, classifier → probabilities (n × labels).
    Raises ValueError with the reason when the model is neither."""
    n, k = len(data), len(actions)
    probabilities = np.zeros((n, k))
    if target == action:
        if task == "regression":
            raise ValueError("it predicts the action as a number, not as a choice.")
        chances = np.asarray(predict(data), dtype=float)
        position = {label: j for j, label in enumerate(labels)}
        known = [a for a in actions if a in position]
        if not known:
            raise ValueError("none of the actions it predicts is in the log.")
        scores = np.column_stack(
            [chances[:, position[a]] if a in position else np.full(n, -1.0) for a in actions]
        )
        probabilities[np.arange(n), scores.argmax(axis=1)] = 1.0
        return probabilities
    if target != reward:
        raise ValueError(f"its target is {target}, neither the action nor the reward.")
    if action not in inputs:
        raise ValueError(f"it predicts {reward} without reading {action}: it cannot choose one.")
    scores = np.zeros((n, k))
    for j, label in enumerate(actions):
        frame = data.copy()
        frame[action] = label
        out = np.asarray(predict(frame), dtype=float)
        if task != "regression":
            column = labels.index(positive) if positive in labels else len(labels) - 1
            out = out[:, column]
        scores[:, j] = out
    probabilities[np.arange(n), scores.argmax(axis=1)] = 1.0
    return probabilities


# --- Evaluation ---------------------------------------------------------------------------------


def _ess(weights: np.ndarray) -> float | None:
    total = float((weights**2).sum())
    return float(weights.sum()) ** 2 / total if total > 0 else None


def evaluate(
    data: Any,
    fields: list[DataField],
    action: str,
    reward: str,
    has_propensity: bool,
    sampled: bool = False,
    extra: list[CandidatePolicy] | None = None,
    skipped: list[str] | None = None,
    dropped: int = 0,
    alpha: float = 0.05,
    learn: bool = False,
) -> PolicyEvaluation | EvaluationUnavailable:
    """`learn`: also grow and evaluate a rule (R-PO5), the slow part ("Advanced › More")."""
    data = data.dropna(subset=[ACTION, REWARD]).reset_index(drop=True)
    data[ACTION] = data[ACTION].astype(str)
    counts = data[ACTION].value_counts()
    actions = sorted(str(a) for a in counts.index)
    if not MIN_ACTIONS <= len(actions) <= MAX_ACTIONS:
        return EvaluationUnavailable(
            f"{action} has {len(actions)} values: a policy chooses among 2 to 10 actions."
        )
    if len(data) < 100:
        return EvaluationUnavailable(
            "Fewer than 100 logged decisions: too few to evaluate a policy."
        )
    y = data[REWARD].to_numpy(float)
    binary = bool(np.isin(y, [0.0, 1.0]).all())
    kind: Literal["rate", "mean"] = "rate" if binary else "mean"
    target = DataField(REWARD, "boolean" if binary else "numeric_continuous", Parser())
    covs = core.covariates(data, fields, target, binary)
    if not covs.used:
        left = "; ".join(f"{c.column}: {c.reason}" for c in covs.columns)
        return EvaluationUnavailable(
            f"No usable covariate once the preparation rules are applied ({left})."
        )
    position = {label: j for j, label in enumerate(actions)}
    a = data[ACTION].map(position).to_numpy(int)
    k, n = len(actions), len(y)
    rows = np.arange(n)
    fold = core.folds(n)
    mu = core.outcomes(covs.x, a, y, k, fold, binary)
    warnings: list[str] = []
    overlap: list[ActionOverlap] = []
    model_note: str | None = None
    if has_propensity:
        p = data[PROPENSITY].to_numpy(float)
        source: Literal["column", "estimated"] = "column"
    else:
        estimated = core.propensities(covs.x, a, k, fold)
        e = estimated.values
        p = e[rows, a]
        model_note = (
            f"Propensity estimated by {estimated.model} (cross-fitted, the classifier with the "
            f"lower cross-validated log loss, {estimated.log_loss:.3f})."
        )
        source = "estimated"
        warnings.append(OBSERVATIONAL)
        for j, label in enumerate(actions):
            overlap.append(
                ActionOverlap(
                    label, float((a == j).mean()), float((e[:, j] < LOW_PROPENSITY).mean())
                )
            )
    clipped_rows = p < core.FLOOR
    pf = np.maximum(p, core.FLOOR)
    gamma = core.aipw(mu, pf, a, y)

    # candidate policies: probabilities n × k
    candidates: list[tuple[str, str, PolicySource, str, np.ndarray]] = []
    for j, label in enumerate(actions):
        matrix = np.zeros((n, k))
        matrix[:, j] = 1.0
        candidates.append(
            (f"always:{label}", f"Always {label}", "constant", f"{label} for every unit.", matrix)
        )
    rule: str | None = None
    if learn:
        chosen = core.choose_depth(covs.x, gamma, covs.features, fold)
        tree = core.grow(covs.x, gamma, covs.features, chosen.depth)
        rule = core.rule_sentence(core.leaves(tree, covs.features), actions, "Choose", "for")
        learned = np.zeros((n, k))
        learned[rows, chosen.actions] = 1.0
        candidates.append(
            (
                "rule",
                "Learnt rule",
                "learned",
                f"{rule} (a depth-{chosen.depth} tree on doubly robust scores; each unit judged by "
                "the rule learnt without it).",
                learned,
            )
        )
    for policy in extra or []:
        candidates.append(
            (policy.key, policy.label, "model", policy.description, policy.probabilities)
        )

    # per-unit terms: logged reward, then DM, IPS (w·r), w, DR per policy
    columns = [y]
    weights = []
    for _, _, _, _, matrix in candidates:
        w = matrix[rows, a] / pf
        dm = (matrix * mu).sum(axis=1)
        dr = (matrix * gamma).sum(axis=1)
        weights.append(w)
        columns += [dm, w * y, w, dr, dr - y]
    stacked = np.column_stack(columns)
    samples = core.bootstrap(stacked)
    means = stacked.mean(axis=0)
    level = f"{(1 - alpha) * 100:.0f} %"
    logged_mean = float(means[0])
    logged_low, logged_high = core.interval(samples[:, :1], alpha)
    policies = [
        PolicyEstimate(
            "logged",
            "Logged decisions",
            "logged",
            "What was actually done: the observed mean reward.",
            [],
            logged_mean,
            core.finite(logged_low[0]),
            core.finite(logged_high[0]),
            None,
            None,
            None,
            float(n),
            1.0,
            True,
            0,
            {label: float((a == j).mean()) for j, label in enumerate(actions)},
            f"Logged decisions: {_value(kind, logged_mean)} per unit, {level} CI "
            f"[{_value(kind, core.finite(logged_low[0]))} ; "
            f"{_value(kind, core.finite(logged_high[0]))}].",
        )
    ]
    points = " points" if kind == "rate" else ""
    for i, (key, label, source_kind, description, matrix) in enumerate(candidates):
        base = 1 + 5 * i
        dm_s, ips_s, w_s, dr_s, gain_s = (samples[:, base + j] for j in range(5))
        with np.errstate(divide="ignore", invalid="ignore"):
            snips_s = ips_s / w_s
        snips = float(means[base + 1] / means[base + 2]) if means[base + 2] > 0 else float("nan")
        # DM's interval takes the spread of DR's: resampling the units with the reward model
        # fixed would ignore the model's own error, the larger part of DM's (R-PO3)
        dm_s = means[base] + (dr_s - means[base + 3])
        draws = np.column_stack([dm_s, ips_s, snips_s, dr_s, gain_s])
        low, high = core.interval(draws, alpha)
        values = [means[base], means[base + 1], snips, means[base + 3], means[base + 4]]
        estimates = [
            EstimatorValue(name, core.finite(values[j]), core.finite(low[j]), core.finite(high[j]))
            for j, name in enumerate(("dm", "ips", "snips", "dr"))
        ]
        w = weights[i]
        ess = _ess(w)
        share = ess / n if ess is not None else None
        reliable = share is not None and share >= LOW_ESS
        clipped = int((clipped_rows & (matrix[rows, a] > 0)).sum())
        dr, dr_low, dr_high = estimates[3].value, estimates[3].low, estimates[3].high
        gain, gain_low, gain_high = (
            core.finite(values[4]),
            core.finite(low[4]),
            core.finite(high[4]),
        )
        sentence = (
            f"{label}: {_value(kind, dr)} per unit (DR, {level} CI [{_value(kind, dr_low)} ; "
            f"{_value(kind, dr_high)}]) vs {_value(kind, logged_mean)} for the logged decisions: "
            f"{_gap(kind, gain)}{points}."
        )
        if not reliable:
            sentence += (
                f" Unreliable: the logged decisions rarely did what this policy does (effective "
                f"sample {ess or 0:,.0f} units, {(share or 0) * 100:.1f} % of the log)."
            )
        policies.append(
            PolicyEstimate(
                key,
                label,
                source_kind,
                description,
                estimates,
                dr,
                dr_low,
                dr_high,
                gain,
                gain_low,
                gain_high,
                ess,
                share,
                reliable,
                clipped,
                {lab: float(matrix[:, j].mean()) for j, lab in enumerate(actions)},
                sentence,
            )
        )

    minimum = float(p.min()) if n else None
    below = float((p < LOW_PROPENSITY).mean()) if n else 0.0
    sentences = [model_note] if model_note else []
    if minimum is not None:
        sentences.append(
            f"Smallest propensity {minimum:.3g}; {below * 100:.1f} % of the decisions had a "
            f"probability below {LOW_PROPENSITY:g}."
        )
    if clipped_rows.any():
        sentences.append(
            f"{int(clipped_rows.sum()):,} propensities below {core.FLOOR:g} were raised to it "
            f"(weights at most {1 / core.FLOOR:,.0f}): a small bias, much less variance."
        )
    if dropped:
        sentences.append(f"{dropped:,} rows left out: their propensity is not in ]0, 1].")
    for item in overlap:
        if item.rare_share >= 0.05:
            sentences.append(
                f"{item.action} was almost never chosen for {item.rare_share * 100:.0f} % of the "
                "units (estimated probability below 0.05): policies that choose it there lean on "
                "the reward model."
            )
    unreliable = [policy.label for policy in policies if not policy.reliable]
    if unreliable:
        sentences.append(
            f"Effective sample below {LOW_ESS * 100:.0f} % of the log for {', '.join(unreliable)}: "
            "their estimates are unreliable."
        )
    diagnostics = DecisionsDiagnostics(
        source, minimum, below, int(clipped_rows.sum()), dropped, overlap, sentences
    )
    return PolicyEvaluation(
        action,
        reward,
        kind,
        actions,
        n,
        sampled,
        covs.columns,
        diagnostics,
        policies,
        rule,
        warnings,
        list(skipped or []),
        METHOD,
        core.DRAWS,
    )
