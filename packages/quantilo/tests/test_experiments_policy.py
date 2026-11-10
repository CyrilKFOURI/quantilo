"""055 Experiments, policies: the shared core (AIPW, bootstrap, policy tree, Qini) and "who should
get which variant" on a randomized test."""

import numpy as np
import pandas as pd
import pytest

from quantilo.experiments import policy as pl
from quantilo.experiments import policy_core as core
from quantilo.experiments.analyze import VARIANT
from quantilo.ml.data import Field
from quantilo.typing.types import Parser


def fields(*specs: tuple[str, str]) -> list[Field]:
    return [Field(name, kind, Parser()) for name, kind in specs]  # type: ignore[arg-type]


def heterogeneous(n: int = 4_000, seed: int = 3, effect: bool = True) -> pd.DataFrame:
    """A randomized test: B converts better than control only when x1 ≥ 0."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    device = rng.choice(["mobile", "desktop", "tablet"], n)
    group = rng.choice(["control", "B"], n)
    lift = np.where(x1 >= 0, 0.20, -0.10) if effect else np.zeros(n)
    p = 0.30 + np.where(group == "B", lift, 0.0)
    return pd.DataFrame(
        {
            VARIANT: group,
            pl.VALUE: (rng.random(n) < p).astype(float),
            "x1": x1,
            "x2": x2,
            "device": device,
        }
    )


COVARIATES = fields(
    ("x1", "numeric_continuous"), ("x2", "numeric_continuous"), ("device", "categorical_nominal")
)


# --- Core ---------------------------------------------------------------------------------------


def test_bootstrap_is_seeded_and_covers_the_mean() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(10, 2, 2_000)
    a = core.bootstrap(values)
    b = core.bootstrap(values)
    assert np.array_equal(a, b)
    low, high = core.interval(a)
    se = values.std(ddof=1) / np.sqrt(len(values))
    assert low[0] < values.mean() < high[0]
    assert high[0] - low[0] == pytest.approx(2 * 1.96 * se, rel=0.15)


def test_aipw_scores_are_unbiased_with_known_propensity() -> None:
    rng = np.random.default_rng(2)
    n = 20_000
    actions = rng.integers(0, 2, n)
    y = np.where(actions == 1, 2.0, 1.0) + rng.normal(0, 1, n)
    mu = np.zeros((n, 2))  # a wrong outcome model: the propensity fixes it
    gamma = core.aipw(mu, np.full(n, 0.5), actions, y)
    assert gamma[:, 0].mean() == pytest.approx(1.0, abs=0.06)
    assert gamma[:, 1].mean() == pytest.approx(2.0, abs=0.06)


def test_tree_splits_where_the_best_action_changes() -> None:
    rng = np.random.default_rng(4)
    n = 2_000
    x = rng.normal(0, 1, (n, 2))
    gamma = np.column_stack([np.zeros(n), np.where(x[:, 0] > 0.5, 1.0, -1.0)])
    features = [core.Feature("a", "a", "number"), core.Feature("b", "b", "number")]
    tree = core.grow(x, gamma, features, 2)
    assert tree.feature == 0 and tree.threshold == pytest.approx(0.5, abs=0.15)
    found = core.leaves(tree, features)
    assert {leaf.action for leaf in found} == {0, 1}
    sentence = core.rule_sentence(found, ["control", "B"])
    assert sentence.startswith("Give B to a ≥") and sentence.endswith("control elsewhere.")


def test_rule_for_everyone_when_nothing_differs() -> None:
    n = 500
    gamma = np.column_stack([np.zeros(n), np.ones(n)])
    x = np.random.default_rng(0).normal(0, 1, (n, 1))
    tree = core.grow(x, gamma, [core.Feature("a", "a", "number")], 3)
    assert tree.leaf and tree.action == 1
    assert core.rule_sentence(core.leaves(tree, []), ["A", "B"]).startswith("Give B to everyone")


def test_paths_are_simplified() -> None:
    features = [
        core.Feature("age", "age", "number"),
        core.Feature("device = mobile", "device", "level", "mobile"),
    ]
    path = [
        core.Condition(0, False, 30.0),
        core.Condition(0, True, 50.0),
        core.Condition(1, False, 0.5),
    ]
    assert core._path_text(path, features) == "30 ≤ age < 50 and device = mobile"


def test_qini_is_above_random_with_a_good_ranking() -> None:
    n = 1_000
    uplift = np.linspace(-1, 1, n)
    gamma = np.column_stack([np.zeros(n), uplift])
    shares, gains, random, auuc = core.qini(gamma, uplift, 1, 0)
    assert len(shares) == core.QINI_POINTS and shares[0] == 0 and shares[-1] == 1
    assert gains[-1] == pytest.approx(random[-1])
    assert auuc > 0.1


def test_covariates_follow_the_preparation_rules() -> None:
    data = heterogeneous(600)
    data["user"] = [f"u{i}" for i in range(len(data))]
    data["when"] = pd.Timestamp("2026-01-01")
    data["constant"] = 1.0
    specs = [
        *COVARIATES,
        *fields(("user", "identifier"), ("when", "datetime"), ("constant", "numeric_continuous")),
    ]
    target = Field(pl.VALUE, "boolean", Parser())
    covs = core.covariates(data, specs, target, True)
    assert covs.used == ["x1", "x2", "device"]
    reasons = {c.column: c.reason for c in covs.columns if not c.used}
    assert set(reasons) == {"user", "when", "constant"}
    assert "date" in (reasons["when"] or "")
    names = [f.name for f in covs.features]
    assert "x1" in names and "device = mobile" in names
    assert covs.x.shape == (600, len(names)) and np.isfinite(covs.x).all()


# --- B: who should get which variant ------------------------------------------------------------


def test_policy_finds_the_subgroup_and_beats_both_constants() -> None:
    result = pl.learn(heterogeneous(), COVARIATES, "converted", "rate", ["control", "B"], qini=True)
    assert isinstance(result, pl.PolicyLearning)
    assert "x1 ≥" in result.rule and result.rule.endswith("control elsewhere.")
    assert result.rule.startswith("Give B to")
    rule, everyone_control, everyone_b = result.values
    assert rule.key == "rule" and everyone_control.label == "Everyone on control"
    assert rule.value is not None and everyone_b.value is not None
    assert rule.value > everyone_b.value and rule.value > (everyone_control.value or 0)
    assert result.verdict == "worth_testing"
    assert "Worth a new test" in result.sentence
    assert result.caveat.startswith("Hypothesis to confirm with a new test")
    assert result.qini[0].variant == "B" and result.qini[0].auuc > 0
    assert 1 <= result.depth <= 3 and len(result.depth_values) == 4
    assert sum(leaf.units for leaf in result.leaves) == result.units


def test_policy_without_heterogeneity_is_no_better() -> None:
    result = pl.learn(
        heterogeneous(effect=False, seed=9), COVARIATES, "converted", "rate", ["control", "B"]
    )
    assert isinstance(result, pl.PolicyLearning)
    assert result.verdict == "no_better"
    gain = result.gains[0] if result.best_constant == "control" else result.gains[1]
    assert gain.low is not None and gain.low <= 0


def test_qini_only_when_asked() -> None:
    result = pl.learn(heterogeneous(1_500), COVARIATES, "converted", "rate", ["control", "B"])
    assert isinstance(result, pl.PolicyLearning) and result.qini == []


def test_policy_is_deterministic() -> None:
    data = heterogeneous(1_500)
    a = pl.learn(data, COVARIATES, "converted", "rate", ["control", "B"])
    b = pl.learn(data, COVARIATES, "converted", "rate", ["control", "B"])
    assert a == b


def test_policy_needs_units_and_covariates() -> None:
    small = heterogeneous(300)
    result = pl.learn(small, COVARIATES, "converted", "rate", ["control", "B"])
    assert isinstance(result, pl.PolicyUnavailable) and "200 units" in result.reason
    one = pl.offer("rate", ["x1"], {"control": 500, "B": 500})
    assert not one.available and "at least 2" in (one.reason or "")
    ratio = pl.offer("ratio", ["x1", "x2"], {"control": 500, "B": 500})
    assert not ratio.available and "ratio" in (ratio.reason or "")
