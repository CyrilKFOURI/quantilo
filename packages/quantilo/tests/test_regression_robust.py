"""Robust and quantile regression (072) against statsmodels and a bootstrap."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as statsmodels_api
from statsmodels.regression.quantile_regression import QuantReg

import quantilo as dx
from quantilo.regression import robust as rb
from quantilo.regression.robust import QuantileProcess
from quantilo.regression.service import RegressionError

sm: Any = statsmodels_api
N = 1_000


@pytest.fixture(scope="module")
def spread(tmp_path_factory) -> dx.Dataset:
    """The spread of y grows with x: the slope is 1 at the median, steeper at the top."""
    rng = np.random.default_rng(21)
    x = rng.uniform(0, 10, N)
    z = rng.normal(size=N)
    y = 5 + 1.0 * x + 0.5 * z + rng.normal(size=N) * (0.5 + 0.4 * x)
    frame = pd.DataFrame({"y": y, "x": x, "z": z, "flag": (rng.random(N) < 0.4)})
    return dx.from_pandas(frame, "spread", cache_dir=tmp_path_factory.mktemp("spread"))


@pytest.fixture(scope="module")
def outliers(tmp_path_factory) -> dx.Dataset:
    rng = np.random.default_rng(8)
    x = rng.normal(size=N)
    y = 1 + 2 * x + rng.normal(size=N)
    bad = rng.choice(N, 50, replace=False)
    x[bad] = rng.uniform(2, 3, 50)  # 5 % of rows far out, pulling the slope down
    y[bad] = rng.uniform(-12, -8, 50)
    frame = pd.DataFrame({"y": y, "x": x, "w": rng.uniform(0.5, 2, N)})
    return dx.from_pandas(frame, "outliers", cache_dir=tmp_path_factory.mktemp("outliers"))


def _xy(ds: dx.Dataset, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = ds.to_pandas()
    return sm.add_constant(frame[columns].to_numpy(dtype=float)), frame["y"].to_numpy()


def test_huber_and_tukey_match_statsmodels_and_resist_outliers(outliers):
    found = outliers.regression_robust("y", ["x"])
    assert found.available
    by = {c.kind: c for c in found.columns}
    x, y = _xy(outliers, ["x"])
    huber = sm.RLM(y, x, M=sm.robust.norms.HuberT()).fit()
    tukey = sm.RLM(y, x, M=sm.robust.norms.TukeyBiweight()).fit(start_params=huber.params)
    np.testing.assert_allclose(by["huber"].estimates, huber.params, rtol=1e-8)
    np.testing.assert_allclose(by["tukey"].estimates, tukey.params, rtol=1e-8)
    ols = by["ols"].estimates[1]
    assert ols is not None and ols < 1.5  # least squares is pulled down
    assert by["tukey"].estimates[1] == pytest.approx(2, abs=0.15)
    assert abs((by["huber"].estimates[1] or 0) - 2) < abs(ols - 2)
    assert found.moved == ["x"] and "`x` moves by more than one standard error" in found.sentence
    assert found.rejected >= 40
    first = outliers.regression("y", ["x"])
    check = next(c for c in first.checks if c.key == "influence")
    assert check.status == "warn" and check.action == "robust" and check.rule == "R-REG-11"


def test_weights_are_said_to_be_left_aside(outliers):
    found = outliers.regression_robust("y", ["x"], weights="w")
    assert any("are not used here" in note for note in found.notes)


def test_quantile_matches_quantreg_and_its_kernel_errors(spread):
    found = spread.regression_robust("y", ["x", "z"], quantile=0.9)
    column = next(c for c in found.columns if c.kind == "quantile")
    x, y = _xy(spread, ["x", "z"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = QuantReg(y, x).fit(q=0.9, max_iter=2_000)
    np.testing.assert_allclose(column.estimates, expected.params, rtol=1e-3)

    def loss(beta: np.ndarray) -> float:  # the check loss: ours is at least as low
        e = y - x @ beta
        return float(np.sum(np.where(e > 0, 0.9 * e, -0.1 * e)))

    assert loss(np.array(column.estimates)) <= loss(expected.params) * (1 + 1e-7)
    assert column.label == "Quantile τ = 0.9"
    # the kernel sandwich against a pairs bootstrap (heteroscedastic errors)
    fit = rb.quantile_fit(x, y, 0.5)
    se = np.sqrt(np.diag(rb.quantile_covariance(fit, x.T @ x)))
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(200):
        pick = rng.integers(0, N, N)
        draws.append(rb.quantile_fit(x[pick], y[pick], 0.5).beta)
    boot = np.std(np.array(draws), axis=0, ddof=1)
    ratio = se / boot  # Powell's kernel errs on the wide side when the spread changes a lot
    assert np.all(ratio > 0.85) and np.all(ratio < 1.5), ratio
    with pytest.raises(RegressionError):
        spread.regression_robust("y", ["x"], quantile=0.3)


def test_kernel_errors_are_close_to_the_iid_ones_when_errors_are_even():
    rng = np.random.default_rng(2)
    x = sm.add_constant(rng.normal(size=(2_000, 2)))
    y = x @ np.array([1.0, 2.0, -1.0]) + rng.normal(size=2_000)
    fit = rb.quantile_fit(x, y, 0.5)
    mine = np.sqrt(np.diag(rb.quantile_covariance(fit, x.T @ x)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iid = QuantReg(y, x).fit(q=0.5, vcov="iid").bse
    np.testing.assert_allclose(mine, iid, rtol=0.12)
    cross = rb.cross_covariance(fit, fit, x.T @ x)
    np.testing.assert_allclose((cross + cross.T) / 2, rb.quantile_covariance(fit, x.T @ x))


def test_effects_change_along_the_distribution(spread):
    found = spread.regression_section("quantiles", "y", ["x", "z"])
    assert isinstance(found, QuantileProcess) and found.available
    curves = {c.term: c for c in found.curves}
    x = curves["x"]
    assert x.taus == list(rb.PROCESS)
    assert x.estimates[-1] > x.estimates[0] + 0.5
    assert x.q_value is not None and x.q_value < 0.001
    assert x.sentence.startswith("The effect of `x` grows along the distribution")
    z = curves["z"]
    assert z.q_value is not None and z.q_value > 0.05  # z's effect does not depend on the level
    assert found.changing == 1 and found.tested == 2
    assert found.t_errors is not None and found.t_errors.available
    checks = {c.key: c for c in spread.regression("y", ["x", "z"]).checks}
    assert checks["quantile"].rule == "R-REG-12" and checks["quantile"].column == "x"
    assert checks["quantile"].action == "robust"


def test_student_errors_and_residual_law(tmp_path):
    rng = np.random.default_rng(5)
    x = rng.normal(size=3_000)
    frame = pd.DataFrame({"y": 1 + x + rng.standard_t(3, size=3_000), "x": x})
    ds = dx.from_pandas(frame, "heavy", cache_dir=tmp_path)
    found = ds.regression_robust("y", ["x"])
    best = next(law for law in found.laws if law.best)
    assert best.name == "Student t" and best.df is not None and 2 < best.df < 5
    assert found.tail_xi is not None and found.tail_xi > 0.25
    assert "heavy tail" in found.tail_sentence
    process = ds.regression_section("quantiles", "y", ["x"])
    assert isinstance(process, QuantileProcess)
    t = process.t_errors
    assert t is not None and t.df is not None and 2 < t.df < 5
    assert t.estimates[1] == pytest.approx(1, abs=0.1)
    assert t.aic is not None and t.ols_aic is not None and t.aic < t.ols_aic
    check = next(c for c in ds.regression("y", ["x"]).checks if c.key == "influence")
    assert "heavy tail" in check.sentence and check.action == "robust"


def test_a_yes_no_target_has_no_robust_section(spread):
    found = spread.regression_robust("flag", ["x"])
    assert not found.available and "logistic" in found.reason.lower()
    process = spread.regression_section("quantiles", "flag", ["x"])
    assert isinstance(process, QuantileProcess) and not process.available
