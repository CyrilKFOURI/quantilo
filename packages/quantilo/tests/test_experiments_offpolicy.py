"""055 Experiments, evaluate a policy from logged decisions: detection, diagnostics and the DM,
IPS, SNIPS and DR estimators on a synthetic log whose true policy values are known."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantilo.experiments import offpolicy as op
from quantilo.ml.data import Field
from quantilo.typing.types import Parser

ACTIONS = ["email", "none", "sms"]


def true_reward(x1: np.ndarray, x2: np.ndarray, action: int) -> np.ndarray:
    """P(reward = 1 | x, action): sms helps when x1 > 0, email when x2 > 0."""
    base = 0.25 + 0.05 * np.tanh(x2)
    if action == 0:  # email
        return base + np.where(x2 > 0, 0.15, -0.05)
    if action == 2:  # sms
        return base + np.where(x1 > 0, 0.20, -0.10)
    return base


def log(n: int = 8_000, seed: int = 11, rare: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """A log of decisions: the logging policy depends on x1 (known propensities)."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    region = rng.choice(["north", "south"], n)
    logits = np.column_stack([0.3 * x1, np.zeros(n), -0.5 * x1 + (-3.5 if rare else 0.0)])
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    chosen = np.array([rng.choice(3, p=row) for row in probs])
    truth = np.column_stack([true_reward(x1, x2, j) for j in range(3)])
    reward = (rng.random(n) < truth[np.arange(n), chosen]).astype(float)
    frame = pd.DataFrame(
        {
            op.ACTION: np.array(ACTIONS)[chosen],
            op.REWARD: reward,
            op.PROPENSITY: probs[np.arange(n), chosen],
            "x1": x1,
            "x2": x2,
            "region": region,
        }
    )
    return frame, truth


FIELDS = [
    Field("x1", "numeric_continuous", Parser()),
    Field("x2", "numeric_continuous", Parser()),
    Field("region", "categorical_nominal", Parser()),
]


def targeted(frame: pd.DataFrame) -> op.CandidatePolicy:
    """sms when x1 > 0, else email: a policy that depends on the covariates."""
    choice = np.where(frame["x1"].to_numpy() > 0, 2, 0)
    matrix = np.zeros((len(frame), 3))
    matrix[np.arange(len(frame)), choice] = 1.0
    return op.CandidatePolicy("targeted", "Targeted", "sms if x1 > 0, else email", matrix)


def by_key(result: op.PolicyEvaluation) -> dict[str, op.PolicyEstimate]:
    return {p.key: p for p in result.policies}


@pytest.fixture(scope="module")
def logged() -> tuple[pd.DataFrame, np.ndarray, op.PolicyEvaluation]:
    frame, truth = log()
    result = op.evaluate(
        frame.copy(), FIELDS, "offer", "converted", True, extra=[targeted(frame)], learn=True
    )
    assert isinstance(result, op.PolicyEvaluation)
    return frame, truth, result


def test_estimators_recover_the_true_values(logged) -> None:
    frame, truth, result = logged
    policies = by_key(result)
    n = len(frame)
    targets = {f"always:{a}": truth[:, j].mean() for j, a in enumerate(ACTIONS)}
    choice = np.where(frame["x1"].to_numpy() > 0, 2, 0)
    targets["targeted"] = truth[np.arange(n), choice].mean()
    for key, value in targets.items():
        policy = policies[key]
        estimates = {e.estimator: e for e in policy.estimates}
        assert set(estimates) == {"dm", "ips", "snips", "dr"}
        for estimator in estimates.values():
            assert estimator.low is not None and estimator.high is not None
            assert estimator.low <= value <= estimator.high, (key, estimator, value)
        assert estimates["dr"].value == pytest.approx(value, abs=0.02)
        assert estimates["ips"].value == pytest.approx(value, abs=0.02)
        assert policy.reliable


def test_logged_policy_is_the_observed_mean(logged) -> None:
    frame, _, result = logged
    logged_policy = by_key(result)["logged"]
    assert logged_policy.value == pytest.approx(frame[op.REWARD].mean())
    assert logged_policy.low is not None and logged_policy.value is not None
    assert logged_policy.low < logged_policy.value
    assert result.kind == "rate" and result.actions == ACTIONS


def test_learnt_rule_beats_the_logged_decisions(logged) -> None:
    _, _, result = logged
    rule = by_key(result)["rule"]
    assert rule.source == "learned" and result.rule is not None
    assert result.rule.startswith("Choose")
    assert rule.gain is not None and rule.gain_low is not None and rule.gain_low > 0
    assert "vs" in rule.sentence and "for the logged decisions" in rule.sentence


def test_diagnostics_come_with_the_log(logged) -> None:
    _, _, result = logged
    d = result.diagnostics
    assert d.propensity == "column" and d.min_propensity is not None and d.min_propensity > 0
    assert d.sentences[0].startswith("Smallest propensity")
    assert not result.warnings  # the propensity is known: not observational


def test_estimated_propensity_warns_and_still_recovers() -> None:
    frame, truth = log(seed=12)
    result = op.evaluate(frame.drop(columns=op.PROPENSITY), FIELDS, "offer", "converted", False)
    assert isinstance(result, op.PolicyEvaluation)
    assert result.diagnostics.propensity == "estimated"
    assert result.warnings and result.warnings[0].startswith(
        "Observational: valid only if no hidden factor"
    )
    assert len(result.diagnostics.overlap) == 3
    policies = by_key(result)
    assert result.diagnostics.sentences[0].startswith("Propensity estimated by")
    for j, a in enumerate(ACTIONS):
        for estimator in policies[f"always:{a}"].estimates:
            assert estimator.low is not None and estimator.high is not None
            assert estimator.low <= truth[:, j].mean() <= estimator.high, (a, estimator)


def test_rare_action_is_clipped_and_unreliable() -> None:
    frame, _ = log(seed=13, rare=True)
    result = op.evaluate(frame, FIELDS, "offer", "converted", True)
    assert isinstance(result, op.PolicyEvaluation)
    sms = by_key(result)["always:sms"]
    assert sms.clipped > 0 and not sms.reliable
    assert "Unreliable" in sms.sentence
    assert result.diagnostics.clipped > 0
    assert any("raised to it" in s for s in result.diagnostics.sentences)


def test_the_rule_is_learnt_only_when_asked() -> None:
    frame, _ = log(n=2_000, seed=15)
    result = op.evaluate(frame, FIELDS, "offer", "converted", True)
    assert isinstance(result, op.PolicyEvaluation)
    assert result.rule is None and "rule" not in by_key(result)


def test_evaluation_is_deterministic() -> None:
    frame, _ = log(n=2_000, seed=14)
    a = op.evaluate(frame.copy(), FIELDS, "offer", "converted", True)
    b = op.evaluate(frame.copy(), FIELDS, "offer", "converted", True)
    assert a == b


def test_model_policy_reads_a_classifier_of_the_action() -> None:
    data = pd.DataFrame({"x1": [-1.0, 1.0]})

    def predict(frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack([frame["x1"] < 0, frame["x1"] >= 0]).astype(float)

    matrix = op.model_policy(
        predict,
        "binary",
        "offer",
        ["email", "sms"],
        None,
        ["x1"],
        data,
        "offer",
        ACTIONS,
        "converted",
    )
    assert matrix.tolist() == [[1, 0, 0], [0, 0, 1]]


def test_model_policy_maximizes_a_reward_model() -> None:
    data = pd.DataFrame({"x1": [-1.0, 1.0], "offer": ["none", "none"]})

    def predict(frame: pd.DataFrame) -> np.ndarray:
        gain = np.where(frame["offer"] == "sms", frame["x1"], 0.0)
        return np.column_stack([1 - (0.5 + gain / 4), 0.5 + gain / 4])

    matrix = op.model_policy(
        predict,
        "binary",
        "converted",
        ["False", "True"],
        "True",
        ["x1", "offer"],
        data,
        "offer",
        ACTIONS,
        "converted",
    )
    assert matrix[1, 2] == 1.0 and matrix[0, 2] == 0.0
    with pytest.raises(ValueError, match="without reading offer"):
        op.model_policy(
            predict,
            "binary",
            "converted",
            ["False", "True"],
            "True",
            ["x1"],
            data,
            "offer",
            ACTIONS,
            "converted",
        )
    with pytest.raises(ValueError, match="neither the action nor the reward"):
        op.model_policy(
            predict, "binary", "other", [], None, ["x1"], data, "offer", ACTIONS, "converted"
        )


def view(name: str, kind: str, role: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(name=name, type=kind, role=role)


def test_detection_needs_an_action_a_reward_and_covariates() -> None:
    views = [
        view("offer", "categorical_nominal", "action"),
        view("converted", "boolean"),
        view("p_action", "numeric_continuous", "propensity"),
        view("age", "numeric_continuous", "age"),
        view("user_id", "identifier", "identifier"),
    ]
    levels = {"offer": 3, "converted": 2, "p_action": 50, "age": 60, "user_id": 1000}
    found = op.detect(views, levels, {"p_action"})
    assert found.detected and found.action == "offer" and found.reward == "converted"
    assert found.propensity == "p_action" and found.covariates == ["age"]
    assert found.sentence.startswith("This looks like logged decisions")
    ab = [
        view("group", "categorical_nominal"),
        view("converted", "boolean"),
        view("landing_page", "categorical_nominal"),
    ]
    missed = op.detect(ab, {"group": 2, "converted": 2, "landing_page": 2}, set())
    assert not missed.detected and "No column names a decision" in missed.sentence
