"""034 Experiments: every test and interval checked against statsmodels / SciPy, and the spec's
acceptance scenarios."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_effectsize,
    proportions_ztest,
)
from statsmodels.stats.rates import test_poisson_2indep as poisson_test
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

import quantilo as dx
from quantilo.experiments import stats as st
from quantilo.experiments.analyze import ExperimentError, ExperimentResult


def ones(n: int, hits: int) -> np.ndarray:
    return np.r_[np.ones(hits), np.zeros(n - hits)]


def load(frame: pd.DataFrame, tmp: Path, name: str = "exp") -> dx.Dataset:
    return dx.from_pandas(frame, name, cache_dir=tmp)


def only(result: ExperimentResult, metric: str | None = None, variant: str | None = None):
    found = [
        c
        for m in result.metrics
        for c in m.comparisons
        if (metric is None or m.column == metric) and (variant is None or c.variant == variant)
    ]
    assert len(found) == 1
    return found[0]


@pytest.fixture(scope="module")
def conversion(tmp_path_factory: pytest.TempPathFactory) -> dx.Dataset:
    """10,000 rows per variant, 11.1 % vs 12.4 % (the spec's first scenario)."""
    frame = pd.DataFrame(
        {
            "group": ["A"] * 10_000 + ["B"] * 10_000,
            "converted": np.r_[ones(10_000, 1_110), ones(10_000, 1_240)].astype(bool),
        }
    )
    return load(frame, tmp_path_factory.mktemp("conversion"))


# --- Rates ---------------------------------------------------------------------------------------


def test_two_proportions_match_statsmodels(conversion: dx.Dataset) -> None:
    result = conversion.experiment(variant="group", metrics=["converted"]).analyze()

    c = only(result)
    z, p = proportions_ztest([1240, 1110], [10_000, 10_000])
    low, high = confint_proportions_2indep(1240, 10_000, 1110, 10_000, method="newcomb")
    assert result.design.control == "A" and c.variant == "B" and c.test == "z_proportions"
    assert c.p_value == pytest.approx(p) and c.statistic == pytest.approx(z)
    assert c.p_value == pytest.approx(0.004, abs=0.001)
    assert c.diff_low == pytest.approx(low) and c.diff_high == pytest.approx(high)
    assert c.diff_low == pytest.approx(0.004, abs=0.001)
    assert c.diff_high == pytest.approx(0.022, abs=0.001)
    assert c.lift == pytest.approx(0.124 / 0.111 - 1)
    assert c.significant and c.verdict == "better"
    assert c.sentence == (
        "B reaches 12.4 % vs 11.1 % for A: +1.3 points (+11.7 %), 95 % CI [+0.4 ; +2.2]. "
        "Unlikely to be chance (p = 0.004)."
    )
    assert result.correction == "One comparison: no adjustment needed."
    assert not result.srm.alarm


def test_sample_ratio_mismatch_raises_the_banner(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {"group": ["A"] * 5_300 + ["B"] * 4_700, "converted": rng.random(10_000) < 0.1}
    )

    result = load(frame, tmp_path).experiment("group", "converted").analyze()

    assert result.srm.p_value == pytest.approx(stats.chisquare([5300, 4700]).pvalue)
    assert result.srm.p_value is not None and result.srm.p_value < 0.001
    assert result.srm.alarm
    assert result.srm.message.startswith("The groups are not the size you planned")
    planned = load(frame, tmp_path / "planned").experiment(
        "group", "converted", expected={"A": 53, "B": 47}
    )
    assert not planned.analyze().srm.alarm


def test_holm_over_three_variants_and_two_metrics(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 3_000
    group = np.repeat(["control", "B", "C"], n)
    lift = np.select([group == "B", group == "C"], [0.02, 0.01], 0.0)
    frame = pd.DataFrame(
        {
            "variant": group,
            "converted": rng.random(3 * n) < 0.10 + lift,
            "revenue": rng.gamma(2.0, 20, 3 * n) * (1 + lift * 5),
        }
    )

    result = load(frame, tmp_path).experiment("variant", ["converted", "revenue"]).analyze()

    every = [c for m in result.metrics for c in m.comparisons]
    assert result.design.control == "control"
    assert result.comparisons == 4 and len(every) == 4
    expected = multipletests([c.p_value for c in every], method="holm")[1]
    assert [c.p_adjusted for c in every] == pytest.approx(list(expected))
    assert result.correction == "p adjusted for 4 comparisons (Holm)"
    assert all("adjusted for 4 comparisons" in c.sentence for c in every)
    assert [m.primary for m in result.metrics] == [True, False]
    assert [m.kind for m in result.metrics] == ["rate", "mean"]


# --- Means, counts, ratios -----------------------------------------------------------------------


def test_welch_matches_statsmodels_and_skew_adds_checks(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    a, b = rng.lognormal(3, 1.2, 800), rng.lognormal(3.1, 1.2, 700)
    frame = pd.DataFrame({"arm": ["a"] * 800 + ["b"] * 700, "basket": np.r_[a, b]})

    result = load(frame, tmp_path).experiment("arm", "basket").analyze()

    c = only(result)
    means = CompareMeans(DescrStatsW(b), DescrStatsW(a))
    low, high = means.tconfint_diff(usevar="unequal")
    t, p, _ = means.ttest_ind(usevar="unequal")
    assert c.test == "welch_t"
    assert (c.diff_low, c.diff_high) == pytest.approx((low, high))
    assert c.p_value == pytest.approx(p) and c.statistic == pytest.approx(t)
    checks = {check.key: check for check in c.checks}
    assert checks["mann_whitney"].p_value == pytest.approx(
        stats.mannwhitneyu(b, a, alternative="two-sided").pvalue  # type: ignore[attr-defined]
    )
    bootstrap = checks["bootstrap"]
    assert bootstrap.low is not None and bootstrap.high is not None
    assert c.diff is not None and bootstrap.low < c.diff < bootstrap.high
    again = st.bootstrap_diff(a, b, 0.05)
    assert again == pytest.approx((bootstrap.low, bootstrap.high))  # seeded: same interval


def test_poisson_counts_when_variance_is_close_to_the_mean(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    a, b = rng.poisson(3.0, 2_000), rng.poisson(3.2, 2_000)
    frame = pd.DataFrame({"group": ["A"] * 2_000 + ["B"] * 2_000, "sessions": np.r_[a, b]})

    c = only(load(frame, tmp_path).experiment("group", "sessions").analyze())

    reference = poisson_test(b.sum(), 2_000, a.sum(), 2_000, compare="diff", method="score")
    assert c.test == "poisson"
    assert c.p_value == pytest.approx(reference.pvalue)
    assert c.diff == pytest.approx(b.mean() - a.mean())
    assert c.checks[0].key == "welch_t"


def test_overdispersed_counts_use_welch(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    a, b = rng.negative_binomial(1, 0.2, 2_000), rng.negative_binomial(1, 0.19, 2_000)
    frame = pd.DataFrame({"group": ["A"] * 2_000 + ["B"] * 2_000, "visits": np.r_[a, b]})

    c = only(load(frame, tmp_path).experiment("group", "visits").analyze())

    assert c.test == "welch_t"
    assert c.p_value == pytest.approx(stats.ttest_ind(b, a, equal_var=False).pvalue)  # type: ignore[attr-defined]


def test_ratio_of_sums_by_delta_method_with_bootstrap_check(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    n = 1_500
    orders = rng.poisson(2, 2 * n) + 1
    revenue = orders * rng.gamma(4, 10, 2 * n) * np.r_[np.ones(n), np.full(n, 1.05)]
    frame = pd.DataFrame({"group": ["A"] * n + ["B"] * n, "revenue": revenue, "orders": orders})

    result = (
        load(frame, tmp_path)
        .experiment("group", [{"column": "revenue", "denominator": "orders"}])
        .analyze()
    )

    metric = result.metrics[0]
    c = metric.comparisons[0]
    assert metric.kind == "ratio" and metric.label == "revenue ÷ orders"
    assert c.control_value == pytest.approx(revenue[:n].sum() / orders[:n].sum())
    assert c.test == "delta_ratio"
    bootstrap = c.checks[0]
    assert bootstrap.key == "bootstrap" and bootstrap.low is not None
    width, check = c.diff_high - c.diff_low, bootstrap.high - bootstrap.low  # type: ignore[operator]
    assert check == pytest.approx(width, rel=0.15)  # the delta method agrees with the bootstrap


def test_delta_method_standard_error_against_a_direct_formula() -> None:
    rng = np.random.default_rng(6)
    y, x = rng.gamma(3, 2, 500), rng.poisson(3, 500) + 1.0
    ratio, se = st.ratio_se(y, x)

    linearised = (y - ratio * x) / x.mean()  # influence of each unit on the ratio
    assert ratio == pytest.approx(y.sum() / x.sum())
    assert se == pytest.approx(linearised.std(ddof=1) / np.sqrt(500))


# --- Units, CUPED, power -------------------------------------------------------------------------


def test_repeated_user_ids_suggest_a_unit_then_aggregate(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    users = np.repeat(np.arange(1_000), 3)
    frame = pd.DataFrame(
        {
            "user_id": users,
            "group": np.where(users % 2 == 0, "A", "B"),
            "clicked": rng.random(3_000) < 0.2,
            "spend": rng.gamma(2, 5, 3_000),
        }
    )
    ds = load(frame, tmp_path)

    loose = ds.experiment("group", ["clicked"]).analyze()
    assert loose.suggested_unit == "user_id"
    assert any("user_id repeats (3.0 rows per value)" in w for w in loose.warnings)

    by_user = ds.experiment("group", ["clicked", "spend"], unit="user_id").analyze()
    assert by_user.suggested_unit is None and by_user.units == 1_000 and by_user.rows == 3_000
    assert [v.units for v in by_user.variants] == [500, 500]
    clicked, spend = by_user.metrics
    per_user = frame.groupby("user_id").agg(clicked=("clicked", "max"), spend=("spend", "sum"))
    group = np.where(per_user.index % 2 == 0, "A", "B")
    assert clicked.values[0] == pytest.approx(per_user.clicked[group == "A"].mean())
    assert spend.values[1] == pytest.approx(per_user.spend[group == "B"].mean())
    assert clicked.aggregation == "1 if any row of the user_id is 1"
    assert spend.aggregation == "summed per user_id"


def test_cuped_removes_rho_squared_of_the_variance(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    n = 20_000
    pre = rng.normal(0, 1, n)
    rho = 0.6
    metric = 50 + 10 * (rho * pre + np.sqrt(1 - rho**2) * rng.normal(0, 1, n))
    group = rng.choice(["A", "B"], n)
    metric = metric + np.where(group == "B", 0.3, 0.0)
    frame = pd.DataFrame({"group": group, "spend": metric, "spend_before": pre})
    ds = load(frame, tmp_path)

    plain = only(ds.experiment("group", "spend").analyze())
    result = ds.experiment("group", [{"column": "spend", "pre": "spend_before"}]).analyze()

    cuped = result.metrics[0].cuped
    adjusted = result.metrics[0].comparisons[0]
    assert cuped is not None
    assert cuped.reduction == pytest.approx(rho**2, abs=0.02)  # the variance left is 1 − ρ²
    assert cuped.correlation == pytest.approx(rho, abs=0.02)
    assert re.match(r"CUPED reduced the variance by 3[4-8] % \(correlation", cuped.sentence)
    assert adjusted.diff_high - adjusted.diff_low < plain.diff_high - plain.diff_low  # type: ignore[operator]
    assert adjusted.diff_low < 0.3 < adjusted.diff_high  # type: ignore[operator]  (the true effect)


def test_underpowered_result_never_says_no_effect(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({"group": ["A"] * 300 + ["B"] * 300, "converted": rng.random(600) < 0.1})

    c = only(load(frame, tmp_path).experiment("group", "converted").analyze())

    assert not c.significant
    assert c.power is not None and c.power < 0.5
    assert c.verdict == "underpowered"
    assert "Not enough data to conclude: with this sample you could only detect a lift of ±" in (
        c.sentence
    )
    assert "no effect" not in c.sentence.lower()


def test_power_and_mde_follow_normal_ind_power() -> None:
    estimate = st.two_proportions(1110, 10_000, 1240, 10_000, 0.05)
    power, mde = st.power_and_mde(estimate, 10_000, 10_000, 0.05)

    sigma = estimate.se / np.sqrt(2 / 10_000)
    assert power == pytest.approx(
        NormalIndPower().power(estimate.diff / sigma, nobs1=10_000, alpha=0.05, ratio=1)
    )
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    assert mde == pytest.approx(z * estimate.se, rel=1e-3)


# --- Subgroups and time --------------------------------------------------------------------------


def test_segments_warn_and_cumulative_ends_on_the_overall_difference(tmp_path: Path) -> None:
    rng = np.random.default_rng(10)
    n = 6_000
    days = pd.Timestamp("2026-03-01") + pd.to_timedelta(rng.integers(0, 14, n), unit="D")
    group = rng.choice(["A", "B"], n)
    frame = pd.DataFrame(
        {
            "group": group,
            "device": rng.choice(["mobile", "desktop", "tablet"], n, p=[0.5, 0.4, 0.1]),
            "day": days,
            "converted": rng.random(n) < np.where(group == "B", 0.13, 0.11),
        }
    )

    result = (
        load(frame, tmp_path)
        .experiment("group", "converted", segment="device", time="day")
        .analyze()
    )

    assert result.segments is not None and result.cumulative is not None
    assert [s.level for s in result.segments.levels] == ["mobile", "desktop", "tablet"]
    assert result.segments.warning.startswith("Subgroups are hypotheses to confirm")
    line = result.cumulative.lines[0]
    assert result.cumulative.grain == "day" and len(line.periods) == 14
    assert line.diff[-1] == pytest.approx(only(result).diff)
    assert "inflates false positives" in result.cumulative.warning


def test_wrong_variant_columns_are_refused(tmp_path: Path) -> None:
    frame = pd.DataFrame({"g": [str(i % 7) for i in range(70)], "y": np.arange(70.0)})
    with pytest.raises(ExperimentError, match="2 to 5 variants"):
        load(frame, tmp_path).experiment("g", "y").analyze()


# --- Plan ----------------------------------------------------------------------------------------


def test_sample_size_matches_normal_ind_power() -> None:
    plan = dx.sample_size(baseline=0.10, effect=0.05, alpha=0.05, power=0.8)

    effect = proportion_effectsize(0.105, 0.10)
    reference = NormalIndPower().solve_power(effect_size=effect, alpha=0.05, power=0.8, ratio=1)
    assert plan.control == pytest.approx(reference, rel=0.01)
    assert plan.total == 2 * plan.control
    assert plan.target == pytest.approx(0.105)
    assert plan.curve and plan.curve[len(plan.curve) // 2].lift == pytest.approx(0.05, rel=0.02)


def test_sample_size_for_a_mean_with_duration_and_three_variants() -> None:
    plan = dx.sample_size(
        kind="mean", baseline=50, sd=20, effect=2, relative=False, variants=3, daily=1_000
    )

    reference = NormalIndPower().solve_power(effect_size=0.1, alpha=0.025, power=0.8, ratio=1)
    assert plan.alpha_per_comparison == pytest.approx(0.025)
    assert plan.control == int(np.ceil(reference))
    assert plan.total == 3 * plan.control and plan.days == int(np.ceil(plan.total / 1_000))
    assert "α is shared by 2 comparisons" in plan.sentence


def test_results_as_tables(conversion: dx.Dataset) -> None:
    result = conversion.experiment("group", "converted").analyze()

    table = dx.to_pandas(result)
    assert list(table[["metric", "variant", "kind"]].iloc[0]) == ["converted", "B", "rate"]
    assert dx.to_dict(result)["srm"]["alarm"] is False
    curve = dx.to_pandas(dx.sample_size(baseline=0.2, effect=0.1))
    assert {"per_group", "lift", "absolute"} <= set(curve.columns)


def test_candidates_guess_the_variant_control_metric_and_unit(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    users = np.repeat(np.arange(400), 2)
    frame = pd.DataFrame(
        {
            "user_id": users,
            "country": rng.choice(["FR", "DE", "ES"], 800),
            "test_group": np.where(users % 2 == 0, "control", "treatment"),
            "converted": rng.random(800) < 0.3,
            "basket": rng.gamma(2, 20, 800),
        }
    )

    found = load(frame, tmp_path).experiment_candidates()

    assert found.variants[0].column == "test_group" and found.variants[0].control == "control"
    kinds = {m.column: m.kind for m in found.metrics}
    assert kinds["converted"] == "rate" and kinds["basket"] == "mean"
    assert found.units[0].column == "user_id" and found.units[0].rows_per_value == 2
    assert found.suggested is not None
    assert found.suggested.variant == "test_group" and found.suggested.unit == "user_id"
    assert found.suggested.metrics[0].column == "converted"
