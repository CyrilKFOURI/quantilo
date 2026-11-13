"""Coefficients that move (073, second part): Kalman random-walk coefficients and Markov-switching
regimes, R-REG-STAB-05…08."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import quantilo as dx
from quantilo.regression import varying as vr

N = 400


def _series(beta_after: float, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=N, freq="ME")
    mkt, smb = rng.normal(0.01, 0.04, N), rng.normal(0, 0.03, N)
    beta = np.where(np.arange(N) < N // 2, 0.8, beta_after)
    ret = 0.002 + beta * mkt + 0.3 * smb + rng.normal(0, 0.01, N)
    return pd.DataFrame({"date": dates, "ret": ret, "mkt": mkt, "smb": smb})


@pytest.fixture(scope="module")
def broken(tmp_path_factory) -> dx.Dataset:
    return dx.from_pandas(_series(1.4), "broken", cache_dir=tmp_path_factory.mktemp("b"))


@pytest.fixture(scope="module")
def steady(tmp_path_factory) -> dx.Dataset:
    return dx.from_pandas(_series(0.8, 2), "steady", cache_dir=tmp_path_factory.mktemp("s"))


def test_the_drift_test_is_a_mixture_of_chi_squares():
    assert vr.drift_test(0.0, 2) == 1.0
    assert vr.drift_test(3.84, 1) == pytest.approx(0.5 * stats.chi2.sf(3.84, 1))
    two = 0.5 * stats.chi2.sf(5.0, 1) + 0.25 * stats.chi2.sf(5.0, 2)
    assert vr.drift_test(5.0, 2) == pytest.approx(two)


def test_fixed_coefficients_are_least_squares_at_every_date():
    rng = np.random.default_rng(3)
    x = np.column_stack([np.ones(200), rng.normal(size=200)])
    y = x @ np.array([1.0, 2.0]) + rng.normal(0, 0.5, 200)
    model = vr.Drifting(y, x)
    smoothed = model.smooth(np.r_[model.s2, 0.0, 0.0])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    states = np.asarray(smoothed.smoothed_state)
    np.testing.assert_allclose(states[:, -1], beta, rtol=1e-6)
    np.testing.assert_allclose(states[:, 50], beta, rtol=1e-6)


def test_a_beta_that_moves_is_followed_by_the_kalman_smoother(broken):
    listing = broken.regression_methods("ret", ["mkt", "smb"])
    proposed = {m.kind for m in listing.methods if m.proposed}
    assert {"varying", "regimes"} <= proposed
    table = broken.regression_method("varying", "ret", ["mkt", "smb"])
    assert isinstance(table, vr.VaryingTable) and table.available
    assert table.drifts and table.p_value is not None and table.p_value < 0.001
    mkt = next(c for c in table.curves if c.term == "mkt")
    assert mkt.estimates[1] == pytest.approx(0.8, abs=0.2)
    assert mkt.estimates[-1] == pytest.approx(1.4, abs=0.2)
    assert "R-REG-STAB-06" in table.sentence


def test_steady_coefficients_do_not_drift(steady):
    table = steady.regression_method("varying", "ret", ["mkt", "smb"])
    assert isinstance(table, vr.VaryingTable) and table.available
    assert not table.drifts
    listing = steady.regression_methods("ret", ["mkt", "smb"])
    varying = next(m for m in listing.methods if m.kind == "varying")
    assert varying.available and not varying.proposed


def test_two_regimes_of_the_beta_are_found(broken):
    table = broken.regression_method("regimes", "ret", ["mkt", "smb"])
    assert isinstance(table, vr.RegimesTable) and table.available
    assert table.worth and table.focus == "mkt"
    betas = sorted(next(t.estimate for t in r.terms if t.term == "mkt") for r in table.regimes)
    assert betas[0] == pytest.approx(0.8, abs=0.1)
    assert betas[1] == pytest.approx(1.4, abs=0.1)
    assert len(table.probabilities) == 2 and len(table.probabilities[0]) == len(table.dates)
    assert "R-REG-STAB-08" in table.sentence


def test_regimes_need_a_dated_series(tmp_path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"y": rng.normal(size=80), "x": rng.normal(size=80)})
    ds = dx.from_pandas(frame, "plain", cache_dir=tmp_path)
    table = ds.regression_method("regimes", "y", ["x"])
    assert not table.available and "R-REG-STAB-01" in table.reason
