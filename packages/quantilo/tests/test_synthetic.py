"""Synthetic data (023): plan rules, methods, and every evaluation formula on hand-made cases."""

import numpy as np
import pandas as pd
import pytest

from quantilo.synthetic import (
    compare,
    detection,
    diagnostic,
    encode,
    identifiers,
    privacy,
)
from quantilo.synthetic import quality as q
from quantilo.synthetic.generate import Spec, group_like, synthesize
from quantilo.synthetic.methods import Cancelled, Options, draw_in_groups
from quantilo.synthetic.plan import PlanColumn, make_plan
from quantilo.synthetic.report import evaluate
from quantilo.synthetic.utility import UtilityError, utility

METHODS = ["cart", "copula", "bayes", "independent"]


def mixed(n: int = 3_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(50, 10, n)
    y = 0.9 * (x - 50) + rng.normal(0, 4.4, n) + 50  # correlation ≈ 0.9
    y[rng.random(n) < 0.2] = np.nan
    group = np.where(x > 55, "hi", np.where(x < 45, "lo", "mid")).astype(object)
    group[rng.random(n) < 0.1] = None
    return pd.DataFrame(
        {
            "id": [f"C{i:05d}" for i in range(n)],
            "x": np.round(x, 2),
            "y": y,
            "k": rng.integers(0, 5, n).astype(float),
            "group": group,
            "day": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
        }
    )


SPECS = [
    Spec("id", "identifier", "unique"),
    Spec("x", "numeric_continuous", "model"),
    Spec("y", "numeric_continuous", "model"),
    Spec("k", "numeric_discrete", "model"),
    Spec("group", "categorical_nominal", "model"),
    Spec("day", "datetime", "model"),
]


# --- Plan ------------------------------------------------------------------------------------


def column(
    name: str,
    kind: str,
    role: str | None = None,
    present: int = 100,
    distinct: int = 10,
    excluded: bool = False,
) -> PlanColumn:
    return PlanColumn(name, kind, role, 100, present, distinct, excluded)  # type: ignore[arg-type]


def test_plan_rules_give_each_column_a_treatment_and_its_reason() -> None:
    found = make_plan(
        [
            column("customer_id", "identifier", "identifier", distinct=100),
            column("revenue", "numeric_continuous"),
            column("children", "numeric_discrete"),
            column("country", "categorical_nominal"),
            column("signup", "datetime"),
            column("email", "text", "email"),
            column("comment", "text", "free_text"),
            column("zip", "categorical_nominal", "postal_code", distinct=300),
            column("empty", "numeric_continuous", present=2),
            column("legacy", "ignored", excluded=True),
        ],
        100,
    )
    rules = {c.name: (c.treatment, c.rule) for c in found.columns}
    assert rules == {
        "customer_id": ("unique", "R-SYN-05"),
        "revenue": ("model", "R-SYN-01"),
        "children": ("model", "R-SYN-02"),
        "country": ("model", "R-SYN-03"),
        "signup": ("model", "R-SYN-04"),
        "email": ("leave", "R-SYN-06"),
        "comment": ("leave", "R-SYN-06"),
        "zip": ("model", "R-SYN-08"),
        "empty": ("leave", "R-SYN-07"),
        "legacy": ("leave", "R-SYN-07"),
    }
    by_name = {c.name: c for c in found.columns}
    assert "fake" in by_name["email"].options and by_name["comment"].options == ["leave"]
    assert by_name["zip"].grouped and "model" not in by_name["customer_id"].options
    assert found.method == "cart" and found.error is None and found.smooth


def test_plan_says_why_nothing_can_be_generated() -> None:
    assert make_plan([column("a", "numeric_continuous")], 20).error
    assert make_plan([column("t", "text", "free_text")], 100).error


# --- Encoding and identifiers ----------------------------------------------------------------


def test_columns_are_learned_and_decoded_inside_their_bounds() -> None:
    values = pd.Series([1.25, 3.5, np.nan, 10.0])
    learned = encode.learn_column("v", values, "numeric_continuous")
    assert (learned.minimum, learned.maximum, learned.decimals) == (1.25, 10.0, 2)
    decoded = encode.decode_column(np.array([0.0, 5.123, 99.0, np.nan]), learned)
    assert decoded.iloc[:3].tolist() == [1.25, 5.12, 10.0] and np.isnan(decoded.iloc[3])
    custom = encode.decode_column(np.array([0.0, 99.0]), learned, (2.0, 8.0))
    assert custom.tolist() == [2.0, 8.0]
    days = encode.learn_column(
        "d", pd.Series(pd.to_datetime(["2020-01-01", "2020-03-01"])), "datetime"
    )
    assert days.day_only and days.minimum is not None
    back = encode.decode_column(np.array([days.minimum + 40_000]), days)
    assert back.iloc[0] == pd.Timestamp("2020-01-01")  # rounded to the day
    ordinal = encode.learn_column(
        "s", pd.Series(["high", "low", "low"]), "categorical_ordinal", ["low", "medium", "high"]
    )
    assert ordinal.levels == ["low", "high"]


def test_rare_levels_are_grouped_into_other() -> None:
    values = pd.Series(["a"] * 6 + ["b"] * 2 + [None])
    grouped = encode.group_rare(values)
    assert grouped.tolist()[-3:] == ["Other", "Other", None]


def test_new_identifiers_keep_the_format_and_never_reuse_a_real_value() -> None:
    real = pd.Series([f"C{i:05d}" for i in range(1, 601)])
    made = identifiers.new_identifiers(real, 1_000, seed=1)
    assert made.is_unique and not made.isin(real).any()
    assert made.str.fullmatch(r"C\d{5}").all()
    numbers = identifiers.new_identifiers(pd.Series([3, 8, 12]), 50, seed=1)
    assert numbers.is_unique and not numbers.isin([3, 8, 12]).any()
    shaped = identifiers.new_identifiers(pd.Series(["ab-12", "cd-34"]), 20, seed=1)
    assert shaped.str.fullmatch(r"[a-z]{2}-\d{2}").all() and shaped.is_unique


def test_fake_values_keep_only_the_shape() -> None:
    real = pd.Series(["anna@mail.com", "bob@site.org", None, None])
    fake = identifiers.fake_values(real, 400, seed=3)
    present = fake.dropna()
    assert present.str.fullmatch(r"[a-z]+@[a-z]+\.[a-z]{3}").all()
    assert not present.isin(real.dropna()).any()
    assert 0.4 < fake.isna().mean() < 0.6


def test_drawing_in_groups_takes_values_of_the_same_group() -> None:
    rng = np.random.default_rng(0)
    drawn = draw_in_groups(
        np.array([0, 0, 1, 1]), np.array([1.0, 2.0, 10.0, 20.0]), np.array([1] * 50), rng
    )
    assert set(drawn) <= {10.0, 20.0}


# --- Methods ---------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_every_method_respects_bounds_levels_integers_and_empty_rates(method: str) -> None:
    train = mixed()
    result = synthesize(train, SPECS, method, Options(rows=2_000, seed=7))  # type: ignore[arg-type]
    frame = result.frame
    assert list(frame.columns) == [s.name for s in SPECS] and len(frame) == 2_000
    assert bool(frame["x"].between(train["x"].min(), train["x"].max()).all())
    assert set(frame["group"].dropna()) <= {"hi", "lo", "mid"}
    assert bool((frame["k"] == frame["k"].round()).all())
    for name in ("y", "group"):
        share = frame[name].isna().to_numpy().mean()
        assert abs(share - train[name].isna().to_numpy().mean()) <= 0.03
    assert bool(frame["day"].between(train["day"].min(), train["day"].max()).all())
    assert frame["id"].is_unique and not encode.column_of(frame, "id").isin(train["id"]).any()
    checks = diagnostic.diagnose(
        frame, result.learned, ["id"], result.bounds, [s.name for s in SPECS]
    )
    assert checks.score == 1.0 and not checks.failures


def test_the_same_seed_gives_the_same_data() -> None:
    train = mixed(800)
    first = synthesize(train, SPECS, "cart", Options(rows=500, seed=3)).frame
    second = synthesize(train, SPECS, "cart", Options(rows=500, seed=3)).frame
    pd.testing.assert_frame_equal(first, second)


def test_cart_and_copula_keep_a_strong_correlation_and_independent_loses_it() -> None:
    train = mixed(4_000).drop(columns=["id"])
    specs = SPECS[1:]
    real_r = encode.column_of(train, "x").corr(encode.column_of(train, "y"))
    found = {}
    for method in METHODS:
        result = synthesize(train, specs, method, Options(rows=4_000, seed=1))  # type: ignore[arg-type]
        found[method] = result
        r = encode.column_of(result.frame, "x").corr(encode.column_of(result.frame, "y"))
        if method in ("cart", "copula"):
            assert abs(r - real_r) <= 0.1, method
        if method == "independent":
            assert abs(r) < 0.1
    trends = {m: q.quality(train, r.frame, r.learned).trends or 0 for m, r in found.items()}
    assert trends["cart"] > trends["independent"] and trends["copula"] > trends["independent"]


def test_bayesian_network_shows_its_tree_and_cart_its_order() -> None:
    train = mixed(1_000).drop(columns=["id"])
    bayes = synthesize(train, SPECS[1:], "bayes", Options(rows=300)).fitted
    assert len(bayes.edges) == len(SPECS[1:]) - 1
    cart = synthesize(train, SPECS[1:], "cart", Options(rows=300)).fitted
    assert sorted(cart.order) == sorted(s.name for s in SPECS[1:])


def test_a_stopped_run_raises_cancelled() -> None:
    with pytest.raises(Cancelled):
        synthesize(mixed(300), SPECS, "cart", Options(rows=100), should_stop=lambda: True)


# --- Diagnostic ------------------------------------------------------------------------------


def test_diagnostic_names_the_columns_of_a_tampered_set() -> None:
    train = mixed(500)
    result = synthesize(train, SPECS, "independent", Options(rows=200))
    tampered = result.frame.copy()
    tampered.loc[0, "x"] = 10_000.0
    tampered.loc[1, "group"] = "invented"
    tampered.loc[2, "id"] = tampered.loc[3, "id"]
    tampered.loc[4, "k"] = 1.5
    found = diagnostic.diagnose(
        tampered, result.learned, ["id"], result.bounds, [s.name for s in SPECS]
    )
    failed = {(c.check, c.column) for c in found.checks if c.score < 1}
    assert failed == {
        ("Boundary adherence", "x"),
        ("Category adherence", "group"),
        ("Key uniqueness", "id"),
        ("Integer adherence", "k"),
    }
    assert found.score < 1 and len(found.failures) == 4


# --- Quality formulas (hand-computed) --------------------------------------------------------


def test_ks_and_tv_complements() -> None:
    assert q.ks_complement(np.array([1.0, 2, 3, 4]), np.array([1.0, 2, 3, 4])) == 1.0
    assert q.ks_complement(np.array([1.0, 2, 3, 4]), np.array([3.0, 4, 5, 6])) == pytest.approx(0.5)
    real = pd.Series(["a", "a", "b", None])  # a .5, b .25, empty .25
    synthetic = pd.Series(["a", "b", "b", "b"])  # a .25, b .75
    assert q.tv_complement(real, synthetic) == pytest.approx(1 - 0.5 * (0.25 + 0.5 + 0.25))


def test_correlation_and_contingency_similarity() -> None:
    x = np.arange(10.0)
    score, real_r, synthetic_r = q.correlation_similarity(x, x, x, -x)
    assert (real_r, synthetic_r, score) == (
        pytest.approx(1.0),
        pytest.approx(-1.0),
        pytest.approx(0.0),
    )
    same = np.array([0, 0, 1, 1])
    assert q.contingency_similarity(same, same, same, same) == 1.0
    # real cells (0,0) and (1,1) at .5 each; synthetic all in (0,1)
    score = q.contingency_similarity(same, same, np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]))
    assert score == pytest.approx(1 - 0.5 * 2)


def test_coverage_and_missing_values() -> None:
    real = np.array([0.0, 10.0, 5.0])
    assert q.range_coverage(real, np.array([2.0, 8.0])) == pytest.approx(0.6)
    assert q.category_coverage(pd.Series(["a", "b", "c", "d"]), pd.Series(["a", "b"])) == 0.5
    assert q.missing_similarity(pd.Series([1, None]), pd.Series([1, 2, 3, None])) == pytest.approx(
        0.75
    )
    values = np.r_[np.zeros(96), [100.0, 100, 100, 100]]
    assert q.outlier_coverage(values, np.r_[np.zeros(98), [100.0, 100]]) == pytest.approx(0.5)


def test_weak_pairs_are_listed_but_do_not_count() -> None:
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    learned = [
        encode.learn_column(n, encode.column_of(real, n), "numeric_continuous") for n in ("a", "b")
    ]
    found = q.quality(real, real.sample(frac=1, random_state=1), learned)
    assert len(found.pair_scores) == 1 and not found.pair_scores[0].meets
    assert found.trends is None and found.score == found.shapes


# --- Detection, utility, privacy -------------------------------------------------------------


def test_a_copy_is_undetectable_and_flagged_as_copied() -> None:
    train = mixed(1_500)
    holdout = mixed(500, seed=9)
    result = synthesize(train, SPECS, "cart", Options(rows=1_500))
    copied = train.copy()
    report = evaluate(
        train, holdout, copied, result.learned, result.fitted, ["id"], result.bounds, []
    )
    assert report.scores.detection > 0.9
    assert report.privacy.copy_share == 1.0 and report.privacy.score == 0.0
    assert report.privacy.warnings


def test_a_healthy_synthetic_set_protects_the_real_rows() -> None:
    train, holdout = mixed(2_000), mixed(600, seed=5)
    learned = synthesize(train, SPECS, "cart", Options(rows=2_000, seed=2))
    found = privacy.privacy(train, holdout, learned.frame, learned.learned)
    assert found.unique_copies == 0 and found.overfitting is not None and found.overfitting >= 0.9
    assert found.baseline is not None and found.histogram is not None


def test_barely_noised_training_rows_trigger_the_memorisation_warning() -> None:
    train, holdout = mixed(2_000), mixed(600, seed=5)
    learned = synthesize(train, SPECS, "independent", Options(rows=10)).learned
    noisy = train.copy()
    noisy["y"] = noisy["y"] + 0.001
    found = privacy.privacy(train, holdout, noisy, learned)
    assert found.overfitting is not None and found.overfitting < privacy.MEMORISATION
    assert any("Memorisation" in w for w in found.warnings)


def test_detection_names_what_gives_the_synthetic_data_away() -> None:
    train = mixed(1_000).drop(columns=["id"])
    learned = synthesize(train, SPECS[1:], "independent", Options(rows=10)).learned
    fake = train.copy()
    fake["x"] = fake["x"] + 30
    found = detection.detect(train, fake, learned)
    assert found.score < 0.2 and found.giveaways[0].column == "x"


def test_utility_tests_both_models_on_the_same_held_out_rows() -> None:
    train, holdout = mixed(1_500), mixed(500, seed=4)
    result = synthesize(train, SPECS, "cart", Options(rows=1_500))
    found = utility(train, result.frame, holdout, result.learned, "x")
    assert found.task == "regression" and found.test_rows == 500
    assert {m.name for m in found.metrics} == {"R²", "MAE"} and found.ratio is not None
    classes = utility(train, result.frame, holdout, result.learned, "group")
    assert classes.task == "classification" and "keeps" in classes.sentence
    with pytest.raises(UtilityError):
        utility(train, result.frame, holdout, result.learned, "day")


def test_disclosure_is_compared_with_a_random_guess() -> None:
    train = mixed(1_000)
    result = synthesize(train, SPECS, "cart", Options(rows=1_000))
    found = privacy.disclosure(train, result.frame, result.learned, ["x", "k"], "group")
    assert found.baseline == pytest.approx(1 / 4)  # hi, lo, mid and empty
    assert [v.name for v in found.variants] == ["CAP", "Zero CAP", "Generalized CAP"]
    assert all(0 <= v.protection <= 1 for v in found.variants)
    with pytest.raises(privacy.PrivacyError):
        privacy.disclosure(train, result.frame, result.learned, ["group"], "group")


def test_held_out_rows_are_grouped_like_the_training_rows() -> None:
    train = pd.DataFrame({"c": ["a"] * 10 + ["b"]})
    result = synthesize(
        train,
        [Spec("c", "categorical_nominal", "model", grouped=True)],
        "independent",
        Options(rows=20),
    )
    grouped = group_like(pd.DataFrame({"c": ["a", "z"]}), result.learned)
    assert grouped["c"].tolist() == ["a", "Other"]


# --- Side by side ----------------------------------------------------------------------------


def test_counts_are_shares_of_each_set_and_histograms_share_intervals() -> None:
    real = mixed(1_000)
    result = synthesize(real, SPECS, "cart", Options(rows=400))
    learned = {i.name: i for i in result.learned}
    detail = compare.column_detail(real, result.frame, learned["y"], 0.9, 10, learned["group"])
    values = detail.counts[0]
    assert values.real_share == pytest.approx(values.real / 1_000)
    assert values.synthetic_share == pytest.approx(values.synthetic / 400)
    assert detail.histogram is not None and len(detail.histogram.edges) == 11
    assert sum(detail.histogram.real) == pytest.approx(1) and sum(
        detail.histogram.synthetic
    ) == pytest.approx(1)
    assert [s.label for s in detail.stats][:3] == ["Min", "5 %", "Q1"]
    levels = compare.column_detail(real, result.frame, learned["group"], 0.9)
    assert {row.value for row in levels.levels} == {"hi", "lo", "mid"}


def test_summaries_put_the_most_damaged_column_first_and_links_list_worst_pairs() -> None:
    real = mixed(1_000).drop(columns=["id"])
    result = synthesize(real, SPECS[1:], "independent", Options(rows=1_000))
    found = q.quality(real, result.frame, result.learned)
    lines = compare.summaries(real, result.frame, result.learned, found)
    scores = [line.score for line in lines if line.score is not None]
    assert len(scores) == len(lines)
    assert scores == sorted(scores)
    links = compare.links(real, result.frame, result.learned)
    assert links.worst[0].a in {"x", "y", "group"} and len(links.columns) == 5
    view = compare.overview(real, result.frame, result.learned)
    assert view.pca is not None and len(view.mean_spread) == 4
