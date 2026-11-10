"""Logged decisions (spec 055 § C, R-PO1): does a dataset look like a log of decisions, and
what the Policies tab asks to evaluate. No scikit-learn here: `import quantilo` stays light; the
evaluation itself is in offpolicy.py."""

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

ACTION, REWARD, PROPENSITY = "__action", "__reward", "__propensity"
ACTION_ROLE, PROPENSITY_ROLE = "action", "propensity"
MIN_ACTIONS, MAX_ACTIONS = 2, 10
LOW_PROPENSITY = 0.05
LOW_ESS = 0.10  # share of the units below which a policy's estimate is unreliable
REWARD_HINT = re.compile(
    r"reward|outcome|convert|conversion|revenue|click|purchas|response|success|gain|profit|"
    r"payoff|return|sales|order|signup|subscri|retain|churn",
    re.I,
)
ACTION_HINT = re.compile(r"action|treatment|offer|arm|campaign|decision|chosen|policy", re.I)

# --- Detection (R-PO1) --------------------------------------------------------------------------


@dataclass
class LoggedDecisions:
    detected: bool
    sentence: str  # what was found, or why not
    action: str | None
    actions: list[str]  # the action column's levels
    reward: str | None
    propensity: str | None
    covariates: list[str]
    action_columns: list[str] = field(default_factory=list)  # every possible choice
    reward_columns: list[str] = field(default_factory=list)
    propensity_columns: list[str] = field(default_factory=list)


class DecisionsRequest(BaseModel):
    action: str = Field(description="The column naming the decision taken (2 to 10 actions).")
    reward: str = Field(description="The reward observed after the decision (number or yes/no).")
    propensity: str | None = Field(
        default=None,
        description="Probability of the logged action, in ]0, 1]; estimated when unset.",
    )
    covariates: list[str] | None = Field(
        default=None, description="Columns describing each unit; every usable one when unset."
    )
    models: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Saved ML models (run ids in the app) to evaluate as policies.",
    )
    learn: bool = Field(
        default=False,
        description=(
            "Also learn a rule from the log (a cross-validated tree on doubly robust scores) and "
            'evaluate it: slower, the page\'s "Advanced > More".'
        ),
    )
    alpha: float = Field(default=0.05, gt=0, lt=0.5)


def detect(views: list[Any], levels: dict[str, int], valid_propensity: set[str]) -> LoggedDecisions:
    """`views`: the dataset's columns (name, type, role); `levels`: distinct values per column;
    `valid_propensity`: numeric columns whose every value lies in ]0, 1]."""
    from quantilo.typing.types import KEY_ROLES, NUMERIC_TYPES

    def is_action(v: Any) -> bool:
        named = v.role == ACTION_ROLE or bool(ACTION_HINT.search(v.name))
        return (
            named
            and v.type
            in ("categorical_nominal", "categorical_ordinal", "boolean", "numeric_discrete", "text")
            and MIN_ACTIONS <= levels.get(v.name, 0) <= MAX_ACTIONS
        )

    actions = [v.name for v in views if is_action(v)]
    actions.sort(
        key=lambda name: 0 if next(v for v in views if v.name == name).role == ACTION_ROLE else 1
    )
    propensities = [
        v.name
        for v in views
        if v.type in NUMERIC_TYPES
        and v.name in valid_propensity
        and (
            v.role == PROPENSITY_ROLE
            or re.search(r"propens|pscore|prob|p_action|logging", v.name, re.I)
        )
    ]
    rewards = [
        v.name
        for v in views
        if (v.type == "boolean" or v.type in NUMERIC_TYPES)
        and v.role not in KEY_ROLES
        and v.name not in propensities
        and v.name not in actions
        and REWARD_HINT.search(v.name)
    ]
    action = actions[0] if actions else None
    reward = rewards[0] if rewards else None
    propensity = propensities[0] if propensities else None
    taken = {action, reward, propensity}
    covariates = [
        v.name
        for v in views
        if v.name not in taken
        and v.type
        in (
            "numeric_continuous",
            "numeric_discrete",
            "categorical_nominal",
            "categorical_ordinal",
            "boolean",
        )
        and v.role not in KEY_ROLES
        and levels.get(v.name, 0) >= 2
    ]
    if action is None:
        why = (
            "No column names a decision (action, treatment, offer, arm, campaign) with 2 to 10 "
            "values."
        )
    elif reward is None:
        why = (
            f"{action} looks like a decision, but no column names its reward (reward, outcome, "
            "converted, revenue…)."
        )
    elif not covariates:
        why = (
            f"{action} and {reward} look like logged decisions, but no column describes the units."
        )
    else:
        why = ""
    detected = not why
    action_levels = levels.get(action, 0) if action else 0
    sentence = (
        f"This looks like logged decisions: {action} ({action_levels} actions) and its reward "
        f"{reward}, with {len(covariates)} column{'s' if len(covariates) > 1 else ''} "
        "describing the units"
        + (f" and the propensity {propensity}." if propensity else "; no propensity column.")
        if detected
        else why
    )
    return LoggedDecisions(
        detected,
        sentence,
        action if detected else None,
        [],
        reward if detected else None,
        propensity if detected else None,
        covariates if detected else [],
        actions,
        rewards,
        propensities,
    )
