"""The cross-section of a quant (074): Fama–MacBeth, IC, several outcomes and GRS, principal
components, against statsmodels."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as statsmodels_api
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS

import quantilo as dx
from quantilo.regression import crosssection as xs
from quantilo.regression.methods import MoreMethods

sm: Any = statsmodels_api
STOCKS, MONTHS = 40, 60


def _stocks(seed: int = 3) -> pd.DataFrame:
    """ret = 0.02·mom − 0.01·size + month effect + ε; mom persistent across months."""
    rng = np.random.default_rng(seed)
    store = np.repeat(np.arange(1, STOCKS + 1), MONTHS)
    month = np.tile(np.arange(MONTHS), STOCKS)
    date = pd.date_range("2015-01-31", periods=MONTHS, freq="ME")[month]
    mom = rng.normal(0, 1, STOCKS * MONTHS)
    size = rng.normal(0, 1, STOCKS)[store - 1] + rng.normal(0, 0.2, STOCKS * MONTHS)
    shock = rng.normal(0, 0.03, MONTHS)[month]
    ret = 0.02 * mom - 0.01 * size + shock + rng.normal(0, 0.05, STOCKS * MONTHS)
    return pd.DataFrame(
        {
            "Ticker": [f"T{s:02d}" for s in store],
            "Date": date,
            "Return": ret,
            "Mom": mom,
            "Size": size,
        }
    )


@pytest.fixture(scope="module")
def stocks(tmp_path_factory) -> dx.Dataset:
    return dx.from_pandas(_stocks(), "stocks", cache_dir=tmp_path_factory.mktemp("stocks"))


def test_fama_macbeth_is_the_mean_of_cross_sections_with_newey_west(stocks):
    table = stocks.regression_method("cross_section", "Return", ["Mom", "Size"])
    assert isinstance(table, xs.CrossSectionTable)
    assert table.available, table.reason
    assert table.periods == MONTHS and table.entity == "Ticker" and table.period == "Date"
    frame = stocks.to_pandas()
    betas = []
    for _, rows in frame.groupby("Date"):
        x = sm.add_constant(rows[["Mom", "Size"]].to_numpy())
        betas.append(sm.OLS(rows["Return"].to_numpy(), x).fit().params)
    series = np.vstack(betas)
    lag = int(np.floor(4 * (MONTHS / 100) ** (2 / 9)))
    assert table.lag == lag
    for j, term in enumerate(table.terms):
        assert term.estimate == pytest.approx(series[:, j].mean(), rel=1e-9)
        hac = sm.OLS(series[:, j], np.ones(MONTHS)).fit(
            cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}
        )
        assert term.std_error == pytest.approx(float(hac.bse[0]), rel=1e-6)
    mom = next(t for t in table.terms if t.term == "Mom")
    assert mom.p_value is not None and mom.p_value < 0.001
    assert "`Mom`" in table.sentence


def test_ic_ranks_mom_and_not_noise(stocks):
    table = stocks.regression_method("cross_section", "Return", ["Mom", "Size"])
    assert isinstance(table, xs.CrossSectionTable)
    first = table.ics[0]
    assert first.column == "Mom" and first.ranks and first.periods == MONTHS
    frame = stocks.to_pandas()
    expected = (
        frame.groupby("Date")[["Mom", "Return"]]
        .corr(method="spearman")
        .xs("Mom", level=1)["Return"]
    )
    assert first.mean == pytest.approx(expected.mean(), rel=1e-9)
    assert first.icir == pytest.approx(expected.mean() / expected.std(ddof=1), rel=1e-9)


def test_cross_section_needs_a_panel(tmp_path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"y": rng.normal(size=200), "x": rng.normal(size=200)})
    ds = dx.from_pandas(frame, "plain", cache_dir=tmp_path)
    table = ds.regression_method("cross_section", "y", ["x"])
    assert isinstance(table, xs.CrossSectionTable)
    assert not table.available and "R-REG-XS-01" in table.reason


def _assets(seed: int = 5, alpha: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 240
    mkt, smb = rng.normal(0.008, 0.045, n), rng.normal(0.002, 0.03, n)
    frame = pd.DataFrame(
        {"Date": pd.date_range("2000-01-31", periods=n, freq="ME"), "Mkt": mkt, "Smb": smb}
    )
    for i, (b, s) in enumerate(((1.0, 0.2), (0.8, 0.6), (1.2, -0.3))):
        frame[f"R{i + 1}"] = alpha + b * mkt + s * smb + rng.normal(0, 0.02, n)
    return frame


def test_several_outcomes_match_statsmodels_mv_test(tmp_path):
    frame = _assets()
    ds = dx.from_pandas(frame, "assets", cache_dir=tmp_path)
    table = ds.regression_method("outputs", "R1", ["Mkt", "Smb"], outputs=["R2", "R3"])
    assert isinstance(table, xs.OutputsTable) and table.available, table.reason
    assert table.outputs == ["R1", "R2", "R3"] and table.rows == len(frame)
    x = sm.add_constant(frame[["Mkt", "Smb"]].to_numpy())
    ys = frame[["R1", "R2", "R3"]].to_numpy()
    model = _MultivariateOLS(ys, x)
    tested = model.fit().mv_test(hypotheses=[("Mkt", np.array([[0.0, 1.0, 0.0]]))])
    stats = tested.results["Mkt"]["stat"]
    mkt = next(t for t in table.tests if t.term == "Mkt")
    assert mkt.wilks == pytest.approx(stats.loc["Wilks' lambda", "Value"], rel=1e-8)
    assert mkt.wilks_f == pytest.approx(stats.loc["Wilks' lambda", "F Value"], rel=1e-6)
    assert mkt.pillai == pytest.approx(stats.loc["Pillai's trace", "Value"], rel=1e-8)
    assert mkt.pillai_f == pytest.approx(stats.loc["Pillai's trace", "F Value"], rel=1e-6)
    assert mkt.pillai_df2 == pytest.approx(stats.loc["Pillai's trace", "Den DF"], rel=1e-9)
    for j, name in enumerate(table.outputs):
        own = sm.OLS(frame[name], x).fit(cov_type="HC3")
        for i in range(3):
            cell = table.coefficients[i][j]
            assert cell.estimate == pytest.approx(own.params.iloc[i], rel=1e-8)
            assert cell.std_error == pytest.approx(own.bse.iloc[i], rel=1e-6)


def test_grs_of_one_asset_is_the_square_of_alpha_t_and_flags_alphas(tmp_path):
    frame = _assets()
    x = sm.add_constant(frame[["Mkt", "Smb"]].to_numpy())
    f, df1, df2, _, _ = xs.grs(x, frame[["R1"]].to_numpy())
    t = sm.OLS(frame["R1"], x).fit().tvalues.iloc[0]
    assert f == pytest.approx(t * t, rel=1e-9) and df1 == 1 and df2 == len(frame) - 3
    ds = dx.from_pandas(_assets(alpha=0.006), "alphas", cache_dir=tmp_path)
    table = ds.regression_method("outputs", "R1", ["Mkt", "Smb"], outputs=["R2", "R3"])
    assert isinstance(table, xs.OutputsTable)
    assert table.grs is not None, table.grs_reason
    assert table.grs.p_value < 0.001 and "are not jointly 0" in table.grs.sentence
    offered = ds.regression_methods("R1", ["Mkt", "Smb"])
    assert isinstance(offered, MoreMethods)
    outputs = next(m for m in offered.methods if m.kind == "outputs")
    assert outputs.proposed and "R-REG-XS-05" in outputs.reason


def _collinear(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 500
    base = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "a": base + rng.normal(0, 0.02, n),
            "b": base + rng.normal(0, 0.02, n),
            "c": base + rng.normal(0, 0.02, n),
            "d": rng.normal(size=n),
        }
    )
    frame["y"] = frame["a"] + frame["b"] + frame["c"] + 0.5 * frame["d"] + rng.normal(0, 1, n)
    return frame


def test_principal_components_are_proposed_for_collinear_columns(tmp_path):
    ds = dx.from_pandas(_collinear(), "collinear", cache_dir=tmp_path)
    offered = ds.regression_methods("y", ["a", "b", "c", "d"])
    components = next(m for m in offered.methods if m.kind == "components")
    assert components.proposed and components.available
    table = ds.regression_method("components", "y", ["a", "b", "c", "d"])
    assert isinstance(table, xs.ComponentsTable) and table.available
    assert table.kept == 2
    # the three collinear columns share their effect: about 1 each, against wild OLS values
    shared = [e.estimate for e in table.effects if e.term in ("a", "b", "c")]
    assert all(0.7 < v < 1.3 for v in shared)


def test_every_component_kept_gives_least_squares(tmp_path, monkeypatch):
    monkeypatch.setattr(xs, "retained", lambda values: len(values))
    frame = _collinear()
    ds = dx.from_pandas(frame, "all", cache_dir=tmp_path)
    table = ds.regression_method("components", "y", ["a", "b", "c", "d"])
    assert isinstance(table, xs.ComponentsTable) and table.kept == 4
    expected = sm.OLS(frame["y"], sm.add_constant(frame[["a", "b", "c", "d"]])).fit().params
    for effect in table.effects:
        assert effect.estimate == pytest.approx(expected[effect.term], rel=1e-7)
        assert effect.least_squares == pytest.approx(expected[effect.term], rel=1e-7)
