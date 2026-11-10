"""055 through the library: `exp.policy()`, `ds.logged_decisions()` and `ds.evaluate_policy()` on
datasets read like the app reads them (types, roles, filters, units)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo.experiments.analyze import ExperimentError


@pytest.fixture(scope="module")
def ab(tmp_path_factory: pytest.TempPathFactory) -> dx.Dataset:
    """A randomized test with covariates: new_checkout helps mobile users and hurts desktop ones."""
    rng = np.random.default_rng(21)
    n = 3_000
    device = rng.choice(["mobile", "desktop"], n)
    age = rng.integers(18, 70, n)
    group = rng.choice(["control", "new_checkout"], n)
    p = 0.2 + np.where((group == "new_checkout") & (device == "mobile"), 0.15, 0.0)
    p -= np.where((group == "new_checkout") & (device == "desktop"), 0.15, 0.0)
    frame = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "group": group,
            "converted": rng.random(n) < p,
            "device": device,
            "age": age,
        }
    )
    return dx.from_pandas(frame, "ab", cache_dir=tmp_path_factory.mktemp("ab"))


def test_analysis_offers_the_policy_and_learns_the_rule(ab: dx.Dataset) -> None:
    exp = ab.experiment(variant="group", metrics=["converted"])
    offer = exp.analyze().policy_offer
    assert offer is not None and offer.available
    assert set(offer.covariates) == {"device", "age"}
    learnt = exp.policy()
    assert "device = mobile" in learnt.rule and "new_checkout" in learnt.rule
    assert learnt.verdict == "worth_testing"
    assert learnt.values[0].key == "rule"
    assert exp.policy() is learnt  # cached
    assert learnt.qini == []  # "Advanced > More": only when asked
    with_qini = exp.policy(qini=True)
    assert with_qini.rule == learnt.rule and with_qini.qini[0].variant == "new_checkout"


def test_policy_refuses_too_few_covariates(ab: dx.Dataset) -> None:
    exp = ab.experiment(variant="group", metrics=["converted"])
    with pytest.raises(ExperimentError, match="cannot describe the units"):
        exp.policy(covariates=["user_id", "age"])


def test_policy_needs_two_covariates(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "group": rng.choice(["A", "B"], 1_000),
            "converted": rng.random(1_000) < 0.3,
            "age": rng.integers(18, 70, 1_000),
        }
    )
    ds = dx.from_pandas(frame, "one", cache_dir=tmp_path)
    exp = ds.experiment(variant="group", metrics=["converted"])
    offer = exp.analyze().policy_offer
    assert offer is not None and not offer.available and "at least 2" in (offer.reason or "")
    with pytest.raises(ExperimentError):
        exp.policy()


@pytest.fixture(scope="module")
def decisions(tmp_path_factory: pytest.TempPathFactory) -> dx.Dataset:
    rng = np.random.default_rng(5)
    n = 4_000
    age = rng.integers(18, 70, n).astype(float)
    region = rng.choice(["north", "south"], n)
    young = age < 35
    p_sms = np.where(young, 0.6, 0.2)
    offer = np.where(rng.random(n) < p_sms, "sms", "email")
    lift = np.where(young, 0.2, -0.1)  # sms helps the young, puts off the others
    reward = rng.random(n) < (0.3 + np.where(offer == "sms", lift, 0.0))
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "offer": offer,
            "propensity": np.where(offer == "sms", p_sms, 1 - p_sms),
            "converted": reward,
            "age": age,
            "region": region,
        }
    )
    return dx.from_pandas(frame, "log", cache_dir=tmp_path_factory.mktemp("log"))


def test_logged_decisions_are_detected(decisions: dx.Dataset, ab: dx.Dataset) -> None:
    found = decisions.logged_decisions()
    assert found.detected
    assert (found.action, found.reward, found.propensity) == ("offer", "converted", "propensity")
    assert found.actions == ["email", "sms"]
    assert set(found.covariates) == {"age", "region"}
    assert not ab.logged_decisions().detected


def test_evaluate_policy_with_a_saved_model(decisions: dx.Dataset) -> None:
    model = decisions.ml("offer").train()
    result = decisions.evaluate_policy(
        "offer", "converted", propensity="propensity", models=[model], learn=True
    )
    keys = [p.key for p in result.policies]
    assert keys[:3] == ["logged", "always:email", "always:sms"]
    assert "rule" in keys and "model:0" in keys
    assert result.diagnostics.propensity == "column" and not result.warnings
    rule = next(p for p in result.policies if p.key == "rule")
    assert rule.gain is not None and rule.gain > 0
    assert result.rule is not None and "age" in result.rule
    estimated = decisions.evaluate_policy("offer", "converted")
    assert estimated.diagnostics.propensity == "estimated"
    assert estimated.warnings[0].startswith("Observational")
