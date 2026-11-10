"""Design, least squares, diagnostics and influence of regression/ against statsmodels."""

from typing import Any

import numpy as np
import pytest
import statsmodels.api as statsmodels_api
from statsmodels.stats import diagnostic as statsmodels_diagnostic
from statsmodels.stats.outliers_influence import OLSInfluence as _OLSInfluence
from statsmodels.stats.stattools import durbin_watson

from quantilo.regression import diagnostics as dg
from quantilo.regression.design import INTERCEPT, OTHERS, DesignError, Variable, build, codes
from quantilo.regression.fit import least_squares, logistic
from quantilo.regression.influence import dfbetas, influence, top_rows

sm: Any = statsmodels_api
smd: Any = statsmodels_diagnostic
OLSInfluence: Any = _OLSInfluence

N = 800


@pytest.fixture(scope="module")
def sample() -> dict[str, Any]:
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    e = np.zeros(N)
    for t in range(1, N):  # AR(1) errors with heteroscedasticity
        e[t] = 0.5 * e[t - 1] + rng.normal() * (1 + 0.5 * abs(x1[t]))
    y = 2 + 1.5 * x1 - 0.7 * x2 + 0.3 * x1**2 + e
    return {"x": sm.add_constant(np.column_stack([x1, x2])), "y": y, "w": rng.uniform(0.5, 2, N)}


def test_ols_matches_statsmodels(sample):
    fit = least_squares(sample["x"], sample["y"])
    expected = sm.OLS(sample["y"], sample["x"]).fit()
    np.testing.assert_allclose(fit.beta, expected.params, rtol=1e-10)
    assert fit.r2 == pytest.approx(expected.rsquared)
    assert fit.adj_r2 == pytest.approx(expected.rsquared_adj)
    assert fit.aic == pytest.approx(expected.aic)
    assert fit.bic == pytest.approx(expected.bic)
    np.testing.assert_allclose(fit.arrays.h, expected.get_influence().hat_matrix_diag, rtol=1e-8)


def test_wls_matches_statsmodels(sample):
    fit = least_squares(sample["x"], sample["y"], sample["w"])
    expected = sm.WLS(sample["y"], sample["x"], weights=sample["w"]).fit()
    np.testing.assert_allclose(fit.beta, expected.params, rtol=1e-10)
    assert fit.r2 == pytest.approx(expected.rsquared)
    assert fit.llf == pytest.approx(expected.llf)


def test_diagnostics_match_statsmodels(sample):
    x, y = sample["x"], sample["y"]
    fit = least_squares(x, y)
    res = sm.OLS(y, x).fit()
    bp = dg.breusch_pagan(fit.resid, x)
    lm, p, _, _ = smd.het_breuschpagan(res.resid, x)
    assert bp.statistic == pytest.approx(lm) and bp.p_value == pytest.approx(p)
    w = dg.white(fit.resid, x)
    assert w is not None
    lm, p, _, _ = smd.het_white(res.resid, x)
    assert w.statistic == pytest.approx(lm) and w.p_value == pytest.approx(p)
    assert dg.durbin_watson(fit.resid) == pytest.approx(durbin_watson(res.resid))
    bg = dg.breusch_godfrey(fit.resid, x, 4)
    lm, p, _, _ = smd.acorr_breusch_godfrey(res, nlags=4, result_object=False)
    assert bg.statistic == pytest.approx(lm, rel=1e-6) and bg.p_value == pytest.approx(p, rel=1e-6)
    lb = dg.ljung_box(fit.resid, 6)
    expected = smd.acorr_ljungbox(res.resid, lags=[6])
    assert lb.statistic == pytest.approx(float(expected["lb_stat"].iloc[0]))
    rs = dg.reset(y, x, fit.fitted)
    found = smd.linear_reset(res, power=3, test_type="fitted", use_f=True)
    assert rs.statistic == pytest.approx(float(found.fvalue), rel=1e-6)
    assert rs.p_value is not None and rs.p_value < 0.01  # the x1² term is missing
    assert dg.bending(fit.resid, {"x1": x[:, 1], "x2": x[:, 2]}) == "x1"


def test_collinearity():
    rng = np.random.default_rng(1)
    a = rng.normal(size=500)
    b = a + rng.normal(size=500) * 0.01
    c = rng.normal(size=500)
    x = sm.add_constant(np.column_stack([a, b, c]))
    vifs, condition = dg.collinearity(x, [INTERCEPT, "a", "b", "c"])
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    assert vifs[0].vif == pytest.approx(variance_inflation_factor(x, 1), rel=1e-6)
    assert vifs[0].partner == "b"
    assert condition > 30


def test_influence_matches_statsmodels(sample):
    fit = least_squares(sample["x"], sample["y"])
    found = influence(fit.arrays)
    expected = OLSInfluence(sm.OLS(sample["y"], sample["x"]).fit())
    np.testing.assert_allclose(found.cook, expected.cooks_distance[0], rtol=1e-8)
    np.testing.assert_allclose(found.studentized, expected.resid_studentized_external, rtol=1e-8)
    rows = top_rows(found)
    np.testing.assert_allclose(dfbetas(fit.arrays, found, rows), expected.dfbetas[rows], rtol=1e-6)
    assert len(rows) == 20


def test_design_indicators_and_missing():
    city, labels = codes(["b", "a", "b", "c", None, "b", "a", "c"] * 10)
    assert labels[0] == "b"  # most frequent first
    target = np.arange(80, dtype=float)
    target[3] = np.nan
    area = np.linspace(0, 1, 80) ** 2
    area[5] = np.nan
    design = build(target, [Variable("area", "num", area), Variable("city", "cat", city, labels)])
    assert design.terms == [INTERCEPT, "area", "city = a", "city = c"]
    assert design.references[0].level == "b"
    assert {m.column: m.rows for m in design.missing} == {"target": 1, "area": 1, "city": 10}
    assert len(design.y) == 80 - 12  # rows 3 and 5 are not among the missing cities


def test_design_top_levels_merge_into_others():
    labels = [f"l{i:02d}" for i in range(30)]
    values = np.repeat(np.arange(30), np.arange(60, 30, -1))
    target = np.random.default_rng(0).normal(size=len(values))
    design = build(target, [Variable("shop", "cat", values, labels)])
    assert design.terms[-1] == f"shop = {OTHERS}"
    assert design.references[0].others == 9  # 30 levels: reference + 20 kept + 9 merged


def test_design_leaves_out_constant_and_collinear_columns():
    rng = np.random.default_rng(2)
    above = rng.uniform(500, 3000, 300)
    basement = np.where(rng.random(300) < 0.4, rng.uniform(100, 900, 300), 0.0)
    living = above + basement
    target = living * 200 + rng.normal(size=300) * 1000
    design = build(
        target,
        [
            Variable("living", "num", living),
            Variable("above", "num", above),
            Variable("basement", "num", basement),
            Variable("flat", "num", np.ones(300)),
        ],
    )
    assert design.terms == [INTERCEPT, "living", "above"]
    reasons = {item.term: item.reason for item in design.left_out}
    assert "perfectly collinear" in reasons["basement"]
    assert "`living`" in reasons["basement"] and "`above`" in reasons["basement"]
    assert reasons["flat"] == "constant on the rows used"


def test_design_needs_rows():
    with pytest.raises(DesignError):
        build(np.array([1.0, np.nan]), [Variable("a", "num", np.array([1.0, 2.0]))])


def test_logistic_average_marginal_effects():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2000, 2))
    p = 1 / (1 + np.exp(-(0.3 + 1.2 * x[:, 0] - 0.5 * x[:, 1])))
    y = (rng.random(2000) < p).astype(float)
    design = sm.add_constant(x)
    fit = logistic(design, y)
    expected = sm.Logit(y, design).fit(disp=0, cov_type="HC0")
    np.testing.assert_allclose(fit.beta, expected.params, rtol=1e-6)
    np.testing.assert_allclose(fit.ame, expected.get_margeff().margeff, rtol=1e-6)
    assert not fit.separated
    assert 0 < fit.pseudo_r2 < 1


def test_logistic_reports_separation():
    x = np.linspace(-1, 1, 100)
    y = (x > 0).astype(float)
    fit = logistic(sm.add_constant(x), y)
    assert fit.separated
