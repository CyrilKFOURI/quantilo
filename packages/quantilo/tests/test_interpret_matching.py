"""123: FLAME (almost-exact matching) in DuckDB, against a confounded case of known effect."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo.experiments.analyze import ExperimentError
from quantilo.interpret import matching as mt


def confounded(n: int = 6_000, seed: int = 7) -> tuple[pd.DataFrame, float]:
    """The treatment is given more often in the north and to pro plans, which also raise the
    outcome; its effect is 1 on basic plans and 3 on pro plans. noise1 and noise2 change
    nothing."""
    rng = np.random.default_rng(seed)
    region = rng.choice(["north", "south", "east"], n)
    plan = rng.choice(["basic", "pro"], n)
    age = rng.integers(18, 80, n).astype(float)
    noise1 = rng.choice(list("abcde"), n)
    noise2 = rng.normal(size=n)
    p = 0.15 + 0.5 * (region == "north") + 0.2 * (plan == "pro")
    t = rng.random(n) < p
    effect = np.where(plan == "pro", 3.0, 1.0)
    y = 5 * (region == "north") + 2 * (plan == "pro") + 0.02 * age + effect * t
    y = y + rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {
            mt.TREATED: t.astype(int),
            mt.OUTCOME: y,
            mt.ROW: np.arange(n),
            "region": region,
            "plan": plan,
            "age": age,
            "noise1": noise1,
            "noise2": noise2,
        }
    )
    return frame, float(effect.mean())


COVARIATES = ["region", "plan", "age", "noise1", "noise2"]


@pytest.fixture(scope="module")
def found() -> tuple[mt.MatchedGroups, float]:
    frame, truth = confounded()
    result = mt.flame(frame, COVARIATES, "offer", "yes", "no", "spend")
    assert isinstance(result, mt.MatchedGroups)
    return result, truth


def test_the_matched_effect_finds_the_known_effect_the_naive_gap_misses(
    found: tuple[mt.MatchedGroups, float],
) -> None:
    result, truth = found
    effects = {e.key: e for e in result.effects}
    naive, ate = effects["naive"], effects["ate"]
    assert naive.value is not None and naive.value > truth + 1.0  # confounded by the region
    assert ate.value is not None and ate.low is not None and ate.high is not None
    assert abs(ate.value - truth) < 0.25
    assert ate.low <= truth <= ate.high
    assert result.matched_treated / result.treated_units > 0.9
    assert "matched exactly" in result.sentence and result.kind == "mean"


def test_each_group_has_its_own_effect(found: tuple[mt.MatchedGroups, float]) -> None:
    result, _ = found
    pro = [g for g in result.groups if "plan = pro" in g.conditions]
    basic = [g for g in result.groups if "plan = basic" in g.conditions]
    assert pro and basic

    def mean(groups: list[mt.MatchedGroup]) -> float:
        sizes = np.array([g.treated + g.control for g in groups], dtype=float)
        return float(np.dot(sizes, [g.effect for g in groups]) / sizes.sum())

    assert abs(mean(pro) - 3.0) < 0.5 and abs(mean(basic) - 1.0) < 0.5
    group = result.groups[0]
    assert group.treated >= 1 and group.control >= 1 and group.rows
    assert group.low is not None and group.high is not None and group.low < group.effect


def test_the_least_useful_covariates_are_dropped_first(
    found: tuple[mt.MatchedGroups, float],
) -> None:
    result, _ = found
    assert result.levels[0].level == 1 and result.levels[0].covariates == COVARIATES
    dropped = [level.dropped for level in result.levels[1:]]
    assert dropped and dropped[0] in ("noise1", "noise2")
    if "region" in dropped:  # the strongest confounder goes last, if ever
        assert dropped.index("region") > dropped.index(dropped[0])
    errors = [level.prediction_error for level in result.levels]
    assert all(0 <= e < 1 for e in errors)
    assert result.stopped
    assert result.held_out > 0 and result.units + result.held_out == 6_000


def test_matching_balances_the_confounders(found: tuple[mt.MatchedGroups, float]) -> None:
    result, _ = found
    region = next(row for row in result.balance if row.column == "region")
    assert region.before is not None and region.before > 0.3
    assert region.after is not None and region.after <= mt.BALANCED and region.balanced
    assert {row.column for row in result.balance} == set(COVARIATES)


def test_the_same_call_gives_the_same_groups() -> None:
    frame, _ = confounded(1_500, seed=3)
    first = mt.flame(frame, COVARIATES, "offer", "yes", "no", "spend")
    second = mt.flame(frame, COVARIATES, "offer", "yes", "no", "spend")
    assert isinstance(first, mt.MatchedGroups) and isinstance(second, mt.MatchedGroups)
    assert first.sentence == second.sentence
    assert [g.rows for g in first.groups] == [g.rows for g in second.groups]


def test_a_rate_outcome_reads_in_points() -> None:
    frame, _ = confounded(2_000, seed=5)
    frame[mt.OUTCOME] = (frame[mt.OUTCOME] > frame[mt.OUTCOME].median()).astype(float)
    result = mt.flame(frame, ["region", "plan"], "offer", "yes", "no", "converted")
    assert isinstance(result, mt.MatchedGroups) and result.kind == "rate"
    assert "points" in result.effects[1].sentence


def test_too_few_units_or_no_covariate_say_why() -> None:
    frame, _ = confounded(400, seed=1)
    few = frame.copy()
    few[mt.TREATED] = 0
    few.loc[:9, mt.TREATED] = 1
    refused = mt.flame(few, COVARIATES, "offer", "yes", "no", "spend")
    assert isinstance(refused, mt.MatchingUnavailable) and "Fewer than" in refused.reason
    flat = frame.assign(region="north")
    nothing = mt.flame(flat, ["region"], "offer", "yes", "no", "spend")
    assert isinstance(nothing, mt.MatchingUnavailable) and "No covariate" in nothing.reason


def test_numbers_are_cut_into_bins_and_categories_keep_their_main_values() -> None:
    codes, texts, how = mt.discrete("age", pd.Series(np.arange(100, dtype=float)))
    assert len(texts) == 4 and texts[0] == "age ≤ 24.75" and texts[-1] == "age > 74.25"
    assert "quantiles" in how and codes.min() == 0 and codes.max() == 3
    _, kept, _ = mt.discrete("kids", pd.Series([0.0, 1.0, 2.0, None]))
    assert kept == ["kids = 0", "kids = 1", "kids = 2", "kids (empty)"]
    many = pd.Series([f"c{i % 15}" for i in range(300)])
    _, labels, why = mt.discrete("city", many)
    assert len(labels) == mt.TOP_CATEGORIES + 1 and "city = Other" in labels and "Other" in why


def test_the_offer_says_why() -> None:
    assert mt.offer(["email", "sms"], ["age"], {"email": 300, "sms": 300}).available
    three = mt.offer(["a", "b", "c"], ["age"], {"a": 300, "b": 300, "c": 300})
    assert three.available and "R-MAT-10" in three.reason  # 123 bis: one pair at a time
    many = mt.offer([str(i) for i in range(11)], ["age"], {str(i): 300 for i in range(11)})
    assert not many.available and "two to 10" in many.reason
    assert not mt.offer(["a", "b"], [], {"a": 300, "b": 300}).available
    small = mt.offer(["a", "b"], ["age"], {"a": 300, "b": 10})
    assert not small.available and "b" in small.reason


def test_the_library_matches_a_dataset_and_the_policies_offer_it(tmp_path: Path) -> None:
    frame, truth = confounded(3_000, seed=11)
    data = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(len(frame))],
            "offer": np.where(frame[mt.TREATED] == 1, "coupon", "control"),
            "spend": frame[mt.OUTCOME].round(3),
            "region": frame["region"],
            "plan": frame["plan"],
            "age": frame["age"],
        }
    )
    ds = dx.from_pandas(data, "offers", cache_dir=tmp_path)
    found = ds.matched_groups("offer", "spend")
    assert found.treated == "coupon" and found.control == "control"
    ate = next(e for e in found.effects if e.key == "ate")
    assert ate.value is not None and abs(ate.value - truth) < 0.3
    assert "customer_id" not in [c.column for c in found.covariates]
    assert found.groups[0].rows and max(found.groups[0].rows) < len(data)
    flipped = ds.matched_groups("offer", "spend", treated="control", covariates=["region"])
    assert flipped.treated == "control" and [c.column for c in flipped.covariates] == ["region"]
    with pytest.raises(ExperimentError, match="not a value"):
        ds.matched_groups("offer", "spend", treated="nothing")
    # 123 bis (R-MAT-10): three regions, one pair at a time
    pair = ds.matched_groups("region", "spend", treated="north", control="south")
    assert pair.treated == "north" and pair.control == "south"
    assert pair.treated_units + pair.control_units <= int(
        np.count_nonzero(np.asarray(data["region"]) != "east")
    )
    offered = ds.evaluate_policy("offer", "spend").matching
    assert offered is not None and offered.available and "R-MAT-01" in offered.reason


# --- 123 bis: DAME (R-MAT-09) -------------------------------------------------------------------


def test_dame_finds_the_effect_and_may_bring_a_covariate_back() -> None:
    frame, truth = confounded()
    found = mt.flame(frame, COVARIATES, "offer", "yes", "no", "spend", method="dame")
    assert isinstance(found, mt.MatchedGroups)
    ate = next(e for e in found.effects if e.key == "ate")
    assert ate.value is not None and abs(ate.value - truth) < 0.3
    assert "DAME" in found.method and "R-MAT-09" in found.method
    first = found.levels[1].dropped if len(found.levels) > 1 else None
    assert first is not None and ("noise" in first or "age" in first)
    # a level matches on a set of covariates, never on one already used
    sets = [tuple(sorted(level.covariates)) for level in found.levels]
    assert len(sets) == len(set(sets))
    flame = mt.flame(frame, COVARIATES, "offer", "yes", "no", "spend")
    assert isinstance(flame, mt.MatchedGroups)
    assert found.matched_treated >= flame.matched_treated * 0.95
