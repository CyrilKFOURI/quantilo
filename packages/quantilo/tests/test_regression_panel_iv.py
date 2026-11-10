"""Panel estimators and instrumental variables of the regression (071)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as statsmodels_api

import quantilo as dx
from quantilo.regression.iv import IvTable
from quantilo.regression.panel import Groups, PanelTable, twoway
from quantilo.regression.service import RegressionError

sm: Any = statsmodels_api

STORES, WEEKS = 60, 30


def _panel(seed: int, tied: bool, drop: float = 0.0) -> pd.DataFrame:
    """y = 2·x + 1.5·size + αᵢ + δₜ + ε; x tied to αᵢ when `tied` (random effects biased)."""
    rng = np.random.default_rng(seed)
    store = np.repeat(np.arange(1, STORES + 1), WEEKS)
    week = np.tile(np.arange(WEEKS), STORES)
    date = pd.date_range("2021-01-04", periods=WEEKS, freq="W-MON")[week]
    alpha = rng.normal(0, 2, STORES)[store - 1]
    delta = rng.normal(0, 1.5, WEEKS)[week]
    size = rng.normal(10, 3, STORES)[store - 1]
    x = (0.8 * alpha if tied else 0.0) + rng.normal(0, 1, STORES * WEEKS)
    y = 2 * x + 1.5 * size + alpha + delta + rng.normal(0, 1, STORES * WEEKS)
    frame = pd.DataFrame({"Store": store, "Date": date, "Sales": y, "Price": x, "Size": size})
    if drop:
        frame = frame.sample(frac=1 - drop, random_state=seed).sort_index()
    return frame


@pytest.fixture(scope="module")
def tied(tmp_path_factory) -> dx.Dataset:
    return dx.from_pandas(_panel(1, True), "tied", cache_dir=tmp_path_factory.mktemp("tied"))


def _section(ds: dx.Dataset, columns: list[str]) -> PanelTable:
    found = ds.regression_section("panel", "Sales", columns)
    assert isinstance(found, PanelTable)
    return found


def _estimate(table: PanelTable, key: str, term: str) -> float:
    estimator = next(e for e in table.estimators if e.key == key)
    value = estimator.coefficients[table.terms.index(term)].estimate
    assert value is not None
    return value


def test_the_first_plane_proposes_the_panel_section(tied):
    found = tied.regression("Sales", ["Price", "Size"])
    check = next(c for c in found.checks if c.key == "panel")
    assert check.rule == "R-REG-14" and check.action == "panel"
    assert "`Price`" in check.sentence and "`Size`" not in check.sentence.split(":")[0]


def test_within_matches_least_squares_with_one_indicator_per_entity(tied):
    table = _section(tied, ["Price", "Size"])
    assert table.available and table.entity == "Store" and table.period == "Date"
    assert table.balanced and table.entities == STORES and table.periods == WEEKS
    assert table.varying == ["Price"] and table.fixed == ["Size"]
    frame = tied.to_pandas()
    dummies = pd.get_dummies(frame["Store"], prefix="s", drop_first=True, dtype=float)
    x = sm.add_constant(pd.concat([frame[["Price"]], dummies], axis=1))
    expected = sm.OLS(frame["Sales"], x).fit(
        cov_type="cluster", cov_kwds={"groups": frame["Store"]}
    )
    entity = next(e for e in table.estimators if e.key == "entity")
    price = entity.coefficients[table.terms.index("Price")]
    assert price.estimate == pytest.approx(expected.params["Price"], rel=1e-9)
    # R-REG-PAN-04: the store indicators are nested in the clusters, they do not count in k.
    n, k_dummies = len(frame), x.shape[1]
    scale = np.sqrt((n - k_dummies) / (n - 2))
    assert price.std_error == pytest.approx(expected.bse["Price"] * scale, rel=1e-6)
    # R-REG-PAN-05: the store size does not change within a store.
    dropped = {d.term: d.reason for d in entity.dropped}
    assert "does not change within a `Store`" in dropped["Size"]
    assert entity.coefficients[table.terms.index("Size")].estimate is None
    between = next(e for e in table.estimators if e.key == "between")
    assert between.coefficients[table.terms.index("Size")].estimate is not None


def test_two_way_matches_indicators_for_both(tied):
    table = _section(tied, ["Price"])
    frame = tied.to_pandas()
    stores = pd.get_dummies(frame["Store"], prefix="s", drop_first=True, dtype=float)
    weeks = pd.get_dummies(frame["Date"], prefix="w", drop_first=True, dtype=float)
    x = sm.add_constant(pd.concat([frame[["Price"]], stores, weeks], axis=1))
    expected = sm.OLS(frame["Sales"], x).fit()
    assert _estimate(table, "twoway", "Price") == pytest.approx(expected.params["Price"], rel=1e-8)


def test_correlated_effects_choose_fixed_effects(tied):
    table = _section(tied, ["Price", "Size"])
    tests = {t.key: t for t in table.tests}
    assert tests["mundlak"].p_value is not None and tests["mundlak"].p_value < 0.001
    assert "random effects would be biased" in tests["mundlak"].sentence
    assert tests["time"].p_value is not None and tests["time"].p_value < 0.001
    assert table.recommended == "twoway"
    assert table.recommended_rule == "R-REG-PAN-01, R-REG-PAN-02"
    assert table.recommended_reason.startswith("Random effects would be biased")
    assert abs(_estimate(table, "twoway", "Price") - 2) < 0.1
    assert abs(_estimate(table, "pooled", "Price") - 2) > 0.3  # pooled OLS is biased
    assert "hausman" in tests and tests["hausman"].p_value is not None
    assert tests["hausman"].p_value < 0.05
    assert "Fixed effects (two-way) recommended" in table.sentence


def test_independent_effects_choose_random_effects(tmp_path):
    ds = dx.from_pandas(_panel(2, False), cache_dir=tmp_path)
    table = _section(ds, ["Price", "Size"])
    tests = {t.key: t for t in table.tests}
    assert tests["mundlak"].p_value is not None and tests["mundlak"].p_value > 0.05
    assert table.recommended == "re" and table.recommended_rule == "R-REG-PAN-01"
    re = next(e for e in table.estimators if e.key == "re")
    assert re.theta is not None and 0 < re.theta < 1
    assert abs(_estimate(table, "re", "Price") - 2) < 0.1
    assert abs(_estimate(table, "re", "Size") - 1.5) < 0.3  # kept, unlike fixed effects
    assert re.se_label == "Clustered by entity"


def test_first_differences_and_mundlak_equal_within(tied):
    table = _section(tied, ["Price"])
    frame = tied.to_pandas().sort_values(["Store", "Date"])
    changes = frame.groupby("Store")[["Sales", "Price"]].diff().dropna()
    expected = sm.OLS(changes["Sales"], sm.add_constant(changes[["Price"]])).fit()
    fd = next(e for e in table.estimators if e.key == "fd")
    assert fd.rows == STORES * (WEEKS - 1)
    assert _estimate(table, "fd", "Price") == pytest.approx(expected.params["Price"], rel=1e-8)


def test_unbalanced_two_way_converges_to_the_indicators(tmp_path):
    frame = _panel(3, True, drop=0.15)
    ds = dx.from_pandas(frame, cache_dir=tmp_path)
    table = _section(ds, ["Price"])
    assert not table.balanced
    stores = pd.get_dummies(frame["Store"], prefix="s", drop_first=True, dtype=float)
    weeks = pd.get_dummies(frame["Date"], prefix="w", drop_first=True, dtype=float)
    x = sm.add_constant(pd.concat([frame[["Price"]], stores, weeks], axis=1))
    expected = sm.OLS(frame["Sales"], x).fit()
    assert _estimate(table, "twoway", "Price") == pytest.approx(expected.params["Price"], rel=1e-7)
    fd = next(e for e in table.estimators if e.key == "fd")
    assert fd.available and "rows without the previous period are lost" in fd.reason


def test_twoway_closed_form_equals_alternating_projections():
    rng = np.random.default_rng(4)
    entity = np.repeat(np.arange(8), 5)
    period = np.tile(np.arange(5), 8)
    a = rng.normal(size=(40, 2))
    e, t = Groups(entity), Groups(period)
    np.testing.assert_allclose(twoway(a, e, t, True), twoway(a, e, t, False), atol=1e-9)


def test_no_panel_no_section(tmp_path):
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"y": rng.normal(size=200), "x": rng.normal(size=200)})
    ds = dx.from_pandas(frame, cache_dir=tmp_path)
    table = _section_any(ds)
    assert not table.available and "panel data" in table.reason
    assert all(c.key != "panel" for c in ds.regression("y", ["x"]).checks)


def _section_any(ds: dx.Dataset) -> PanelTable:
    found = ds.regression_section("panel", "y", ["x"])
    assert isinstance(found, PanelTable)
    return found


# --- Instrumental variables ----------------------------------------------------------------------

N = 3_000


def _iv_frame(seed: int) -> pd.DataFrame:
    """y = 1 + 2·x + 0.5·c + 3·u + ε, x = 0.7·z1 + 0.5·z2 + 0.3·c + u + v: x is endogenous."""
    rng = np.random.default_rng(seed)
    z1, z2, c, u = (rng.normal(size=N) for _ in range(4))
    x = 0.7 * z1 + 0.5 * z2 + 0.3 * c + u + rng.normal(size=N)
    y = 1 + 2 * x + 0.5 * c + 3 * u + rng.normal(size=N)
    bad = rng.normal(size=N)
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "c": c,
            "z1": z1,
            "z2": z2,
            "noise": rng.normal(size=N),
            "leak": bad,
            "y_leak": y + 2 * bad,
            "x_leak": x + 0.6 * bad,
        }
    )


@pytest.fixture(scope="module")
def instrumented(tmp_path_factory) -> dx.Dataset:
    return dx.from_pandas(_iv_frame(7), "iv", cache_dir=tmp_path_factory.mktemp("iv"))


def _coefficient(table: IvTable, term: str):
    return next(c for c in table.coefficients if c.term == term)


def test_iv_recovers_the_known_coefficient(instrumented):
    table = instrumented.regression_iv("y", ["x", "c"], endogenous=["x"], instruments=["z1", "z2"])
    x = _coefficient(table, "x")
    assert x.endogenous and x.ci_low is not None and x.ci_high is not None
    assert x.ci_low < 2 < x.ci_high
    assert x.ols is not None and x.ols_low is not None and x.ols_low > 2.3  # OLS is biased
    assert table.se.kind == "hc1" and "R-REG-IV-02" in table.se.reason
    frame = instrumented.to_pandas()
    z = sm.add_constant(frame[["c", "z1", "z2"]]).to_numpy()
    xs = sm.add_constant(frame[["x", "c"]]).to_numpy()
    x_hat = z @ np.linalg.lstsq(z, xs, rcond=None)[0]
    beta = np.linalg.lstsq(x_hat, frame["y"].to_numpy(), rcond=None)[0]
    np.testing.assert_allclose([c.estimate for c in table.coefficients], beta, rtol=1e-9)
    tests = {t.key: t for t in table.tests}
    assert tests["endogeneity"].p_value is not None and tests["endogeneity"].p_value < 0.001
    assert "is endogenous" in tests["endogeneity"].sentence
    assert tests["overid"].title == "Overidentification (Hansen J)"
    assert tests["overid"].p_value is not None and tests["overid"].p_value > 0.05
    assert tests["anderson_rubin"].p_value is not None and tests["anderson_rubin"].p_value < 0.001
    stage = table.first_stages[0]
    assert stage.effective_f is not None and stage.effective_f > 100 and stage.status == "ok"
    assert table.sentence.startswith("Instrumenting `x` with `z1` and `z2`")


def test_classic_errors_match_statsmodels_2sls(instrumented):
    from statsmodels.sandbox.regression.gmm import IV2SLS

    table = instrumented.regression_iv(
        "y", ["x", "c"], endogenous=["x"], instruments=["z1", "z2"], se="classic"
    )
    frame = instrumented.to_pandas()
    expected = IV2SLS(
        frame["y"], sm.add_constant(frame[["x", "c"]]), sm.add_constant(frame[["c", "z1", "z2"]])
    ).fit()
    np.testing.assert_allclose([c.estimate for c in table.coefficients], expected.params, rtol=1e-9)
    np.testing.assert_allclose([c.std_error for c in table.coefficients], expected.bse, rtol=1e-9)
    overid = next(t for t in table.tests if t.key == "overid")
    assert overid.title == "Overidentification (Sargan)"


def test_one_instrument_effective_f_equals_the_robust_first_stage_f(instrumented):
    table = instrumented.regression_iv("y", ["x", "c"], endogenous=["x"], instruments=["z1"])
    stage = table.first_stages[0]
    assert stage.f is not None and stage.effective_f is not None
    assert stage.effective_f == pytest.approx(stage.f, rel=0.01)
    overid = next(t for t in table.tests if t.key == "overid")
    assert overid.statistic is None and "cannot be tested" in overid.sentence


def test_a_weak_instrument_turns_red(instrumented):
    table = instrumented.regression_iv("y", ["x", "c"], endogenous=["x"], instruments=["noise"])
    stage = table.first_stages[0]
    assert stage.effective_f is not None and stage.effective_f < 10
    assert stage.status == "bad" and "Anderson–Rubin" in stage.sentence
    assert "weak" in table.sentence


def test_an_invalid_instrument_fails_the_overidentification_test(instrumented):
    table = instrumented.regression_iv(
        "y_leak", ["x_leak", "c"], endogenous=["x_leak"], instruments=["z1", "leak"]
    )
    overid = next(t for t in table.tests if t.key == "overid")
    assert overid.p_value is not None and overid.p_value < 0.001 and overid.status == "bad"


def test_iv_is_never_guessed(instrumented):
    with pytest.raises(RegressionError, match="never guessed"):
        instrumented.regression_iv("y", ["x", "c"], endogenous=["x"], instruments=[])
    with pytest.raises(RegressionError, match="already in the model"):
        instrumented.regression_iv("y", ["x", "c"], endogenous=["x"], instruments=["c"])
    with pytest.raises(RegressionError, match="not among the explanatory columns"):
        instrumented.regression_iv("y", ["c"], endogenous=["x"], instruments=["z1"])
    with pytest.raises(RegressionError, match="order condition"):
        instrumented.regression_iv("y", ["x", "c"], endogenous=["x", "c"], instruments=["z1"])
