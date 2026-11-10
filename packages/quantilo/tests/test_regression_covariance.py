"""Every sandwich of regression/covariance.py against statsmodels."""

from typing import Any

import numpy as np
import pytest
import statsmodels.api as statsmodels_api

from quantilo.regression import covariance as cov

sm: Any = statsmodels_api

N = 600


@pytest.fixture(scope="module")
def data() -> dict[str, Any]:
    rng = np.random.default_rng(3)
    entity = np.repeat(np.arange(30), 20)
    period = np.tile(np.arange(20), 30)
    x = rng.normal(size=(N, 2)) + entity[:, None] * 0.05
    shock = rng.normal(size=30)[entity] + rng.normal(size=20)[period]
    y = 1 + x @ np.array([0.5, -0.3]) + shock + rng.normal(size=N) * (1 + np.abs(x[:, 0]))
    return {"x": sm.add_constant(x), "y": y, "entity": entity, "period": period}


def arrays(data: dict[str, Any]) -> tuple[cov.Arrays, Any]:
    fit = sm.OLS(data["y"], data["x"]).fit()
    bread = np.linalg.inv(data["x"].T @ data["x"])
    h = np.einsum("ij,jk,ik->i", data["x"], bread, data["x"])
    return cov.Arrays(data["x"], fit.resid, bread, h), fit


@pytest.mark.parametrize("kind", ["hc0", "hc1", "hc2", "hc3"])
def test_hc_matches_statsmodels(data, kind):
    a, fit = arrays(data)
    expected = fit.get_robustcov_results(kind.upper()).cov_params()
    np.testing.assert_allclose(cov.hc(a, kind), expected, rtol=1e-8)


def test_classic_matches_statsmodels(data):
    a, fit = arrays(data)
    np.testing.assert_allclose(cov.classic(a), fit.cov_params(), rtol=1e-10)


def test_hac_matches_statsmodels(data):
    a, fit = arrays(data)
    expected = fit.get_robustcov_results("HAC", maxlags=5, use_correction=True).cov_params()
    np.testing.assert_allclose(cov.hac(a, 5), expected, rtol=1e-8)


def test_cluster_matches_statsmodels(data):
    a, fit = arrays(data)
    expected = fit.get_robustcov_results("cluster", groups=data["entity"]).cov_params()
    np.testing.assert_allclose(cov.cluster(a, data["entity"]), expected, rtol=1e-8)


def test_two_way_cluster_matches_statsmodels(data):
    a, fit = arrays(data)
    groups = np.column_stack([data["entity"], data["period"]])
    expected = fit.get_robustcov_results("cluster", groups=groups).cov_params()
    np.testing.assert_allclose(
        cov.cluster2(a, data["entity"], data["period"]), expected, rtol=1e-6, atol=1e-12
    )


def test_driscoll_kraay_matches_statsmodels(data):
    a, fit = arrays(data)
    expected = fit.get_robustcov_results(
        "hac-groupsum", time=data["period"], maxlags=3, use_correction="hac"
    ).cov_params()
    np.testing.assert_allclose(cov.driscoll_kraay(a, data["period"], 3), expected, rtol=1e-8)


def test_newey_west_lag():
    assert cov.newey_west_lag(100) == 4
    assert cov.newey_west_lag(1259) == 7  # five years of trading days
    assert cov.newey_west_lag(5) == 2
    assert cov.newey_west_lag(1) == 1


def test_negative_eigenvalues_are_clipped():
    v = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and −1
    fixed = cov._psd(v)
    assert np.linalg.eigvalsh(fixed).min() >= -1e-12
