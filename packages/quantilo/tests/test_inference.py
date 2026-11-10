"""051: multiple-testing corrections and bootstrap intervals."""

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from quantilo.compare import bootstrap as bs
from quantilo.compare.multiple import MIN_TESTS, adjust, choose, correct

P_VALUES = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216, 0.5, 0.9]


@pytest.mark.parametrize(
    ("method", "reference"), [("bh", "fdr_bh"), ("holm", "holm"), ("bonferroni", "bonferroni")]
)
def test_adjustments_match_statsmodels(method: str, reference: str) -> None:
    expected = multipletests(P_VALUES, method=reference)[1]
    found = adjust(P_VALUES, method)  # type: ignore[arg-type]
    assert np.allclose([v for v in found if v is not None], expected)


def test_missing_p_values_stay_missing_and_are_not_counted() -> None:
    found = adjust([0.01, None, 0.04], "bonferroni")
    assert found[1] is None
    assert found[0] == pytest.approx(0.02) and found[2] == pytest.approx(0.08)


def test_no_correction_returns_the_p_values() -> None:
    assert adjust([0.01, None], "none") == [0.01, None]


def test_the_rule_picks_bh_from_ten_tests_holm_below_nothing_for_one() -> None:
    assert choose(MIN_TESTS)[0] == "bh"
    assert choose(MIN_TESTS - 1)[0] == "holm"
    assert choose(2)[0] == "holm"
    assert choose(1)[0] == "none"
    assert choose(0)[0] == "none"
    assert choose(40, "bonferroni")[0] == "bonferroni"


def test_the_reason_says_why() -> None:
    _, reason = choose(190)
    assert "190 tests" in reason and "10 would pass" in reason
    _, chosen = choose(190, "holm")
    assert "chosen" in chosen.lower()


def test_correct_counts_what_the_correction_removes() -> None:
    adjusted, summary = correct(P_VALUES, noun="links")
    assert summary.method == "bh" and summary.tests == 12
    assert summary.significant == 5  # p < 0.05 before
    assert summary.kept == sum(1 for q in adjusted if q is not None and q < 0.05)
    assert summary.kept < summary.significant
    assert summary.sentence == (
        f"{summary.kept} of the 5 significant links stay significant after correcting for "
        "12 tests (Benjamini–Hochberg)."
    )


def test_correct_with_one_test_needs_no_correction() -> None:
    adjusted, summary = correct([0.03])
    assert adjusted == [0.03]
    assert summary.method == "none" and summary.sentence == "One test: no correction needed."


def test_correct_with_nothing_significant() -> None:
    _, summary = correct([0.2, 0.5, 0.7])
    assert summary.sentence.startswith("No result reaches p < 0.05")


# --- Bootstrap -----------------------------------------------------------------------------------


def test_the_rule_for_a_mean_interval() -> None:
    rng = np.random.default_rng(1)
    small = rng.normal(10, 2, 12)
    assert bs.mean_rule(len(small), small)[0] == "bootstrap"
    skewed = rng.lognormal(0, 1.2, 500)
    method, reason = bs.mean_rule(500, skewed)
    assert method == "bootstrap" and "skewness" in reason
    normal = rng.normal(0, 1, 500)
    assert bs.mean_rule(500, normal)[0] == "t"
    # many values: the mean is near normal anyway
    assert bs.mean_rule(50_000, skewed)[0] == "t"
    # too few to resample
    assert bs.mean_rule(4, small[:4])[0] == "t"


def test_replicates_are_seeded_and_independent_per_stream() -> None:
    values = np.random.default_rng(2).normal(0, 1, 200)
    first = bs.replicates(values, len(values), stream=0)
    again = bs.replicates(values, len(values), stream=0)
    other = bs.replicates(values, len(values), stream=1)
    assert np.array_equal(first.means, again.means)
    assert not np.array_equal(first.means, other.means)
    assert len(first.means) == bs.DRAWS


def test_bootstrap_interval_of_a_mean_is_close_to_student_on_normal_data() -> None:
    values = np.random.default_rng(3).normal(50, 5, 400)
    rep = bs.replicates(values, len(values), stream=0)
    low, high = bs.interval(rep.means, len(values))
    margin = stats.t.ppf(0.975, 399) * values.std(ddof=1) / np.sqrt(400)
    assert low == pytest.approx(values.mean() - margin, rel=0.01)
    assert high == pytest.approx(values.mean() + margin, rel=0.01)


def test_small_samples_get_a_wider_interval_than_the_plain_percentile() -> None:
    values = np.random.default_rng(4).normal(0, 1, 10)
    rep = bs.replicates(values, len(values), stream=0)
    plain = np.quantile(rep.means, [0.025, 0.975])
    low, high = bs.interval(rep.means, len(values))
    assert low < plain[0] and high > plain[1]


def test_a_large_group_is_rescaled_from_its_sample() -> None:
    rng = np.random.default_rng(5)
    population = rng.exponential(1, 40_000)
    sample = population[:1_000]
    rep = bs.replicates(sample, len(population), stream=0)
    low, high = bs.interval(rep.medians, len(population))
    # the median's standard error on 40,000 values, about 1.2533 σ / √n for an exponential
    width = high - low
    expected = 2 * 1.96 / (2 * np.exp(-np.log(2)) * np.sqrt(40_000))
    assert width == pytest.approx(expected, rel=0.25)


def test_the_budget_caps_the_values_resampled() -> None:
    assert bs.held_size(groups=2, available=2_500) == 2_000
    assert bs.held_size(groups=14, available=1_000) == bs.BUDGET // (bs.DRAWS * 14)
    assert bs.held_size(groups=1, available=40) == 40
