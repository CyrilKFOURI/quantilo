"""062: regimes of a series, a statsmodels MarkovRegression read with rules."""

import numpy as np
import pytest
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from quantilo.timeseries import regimes as rg


@pytest.fixture(scope="module")
def walk() -> np.ndarray:
    rng = np.random.default_rng(0)
    steps = np.concatenate([rng.normal(0, 1, 200), rng.normal(0, 4, 150), rng.normal(0, 1, 200)])
    return 100 + np.cumsum(steps)


def test_a_random_walk_is_read_through_its_changes(walk) -> None:
    found = rg.find_regimes(walk, k=2)
    assert found.on == "changes" and "R-RG-01" in found.on_reason
    calm, turbulent = found.regimes
    assert (calm.name, turbulent.name) == ("calm", "turbulent")
    assert calm.std == pytest.approx(1, rel=0.15)
    assert turbulent.std == pytest.approx(4, rel=0.15)
    assert calm.share == pytest.approx(400 / 549, abs=0.05)
    # the turbulent block is periods 200-349 of the changes
    assert np.mean(np.array(found.most_likely[210:340]) == 1) > 0.9
    assert all(sum(row) == pytest.approx(1) for row in found.transition)
    assert found.sentence and "calm" in found.sentence


def test_same_numbers_as_statsmodels(walk) -> None:
    found = rg.find_regimes(walk, k=2, on="changes")
    direct = MarkovRegression(np.diff(walk), k_regimes=2, trend="c", switching_variance=True).fit(
        disp=False
    )
    durations = sorted(float(d) for d in direct.expected_durations)
    ours = sorted(r.expected_duration or 0 for r in found.regimes)
    assert ours == pytest.approx(durations, rel=0.02)
    sigmas = sorted(np.sqrt(direct.params[-2:]))
    assert [r.std for r in found.regimes] == pytest.approx(sigmas, rel=0.02)


def test_number_of_regimes_by_bic(walk) -> None:
    found = rg.find_regimes(walk)
    assert found.k == 2 and "R-RG-02" in found.k_reason
    assert {f.k for f in found.fits} == {2, 3}


def test_stationary_level_regimes() -> None:
    rng = np.random.default_rng(2)
    level = np.concatenate([rng.normal(0, 1, 150), rng.normal(5, 1, 150), rng.normal(0, 1, 150)])
    found = rg.find_regimes(level, k=2, on="level")
    assert found.on == "level"
    low, high = found.regimes
    assert (low.name, high.name) == ("low", "high")
    assert high.mean == pytest.approx(5, abs=0.3)
    assert found.switches == 2


def test_too_short_or_flat() -> None:
    assert rg.find_regimes(np.arange(10.0)).message
    assert rg.find_regimes(np.ones(100), on="level").message == "The series does not vary."
