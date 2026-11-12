"""Stability over time (073), resampling and selection (077), and ML's coefficients through the
common engine (079), against statsmodels."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as statsmodels_api
from statsmodels.stats.diagnostic import recursive_olsresiduals

import quantilo as dx
from quantilo.ml.data import Field
from quantilo.ml.train import train
from quantilo.regression import inference as inf
from quantilo.regression import resampling as rs
from quantilo.regression import selection as sl
from quantilo.regression import stability as st
from quantilo.typing.types import Parser

sm: Any = statsmodels_api
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


# --- 073 -----------------------------------------------------------------------------------------


def _xy(frame: pd.DataFrame, columns: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    return sm.add_constant(frame[columns].to_numpy(dtype=float)), frame[target].to_numpy()


def test_cusum_matches_recursive_residuals_of_statsmodels():
    frame = _series(1.4)
    x, y = _xy(frame, ["mkt", "smb"], "ret")
    found = st.recursive(x, y)
    k = x.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaled = np.asarray(recursive_olsresiduals(sm.OLS(y, x).fit(), skip=k)[4])[k:]
    np.testing.assert_allclose(np.asarray(found.resid_recursive)[k:], scaled, rtol=1e-6)
    expected = np.cumsum(scaled) / np.std(scaled, ddof=1)
    np.testing.assert_allclose(np.asarray(found.cusum), expected, rtol=1e-6)


def test_chow_matches_three_least_squares():
    frame = _series(1.4)
    x, y = _xy(frame, ["mkt", "smb"], "ret")
    at = 150
    f, p = st.chow(x, y, at)
    ssr = [sm.OLS(yy, xx).fit().ssr for xx, yy in ((x, y), (x[:at], y[:at]), (x[at:], y[at:]))]
    k = x.shape[1]
    expected = ((ssr[0] - ssr[1] - ssr[2]) / k) / ((ssr[1] + ssr[2]) / (N - 2 * k))
    assert f == pytest.approx(expected, rel=1e-9)
    assert p is not None and p < 1e-6


def test_a_break_in_a_beta_is_found_and_said(broken):
    report = broken.regression("ret", ["mkt", "smb"])
    check = next(c for c in report.checks if c.key == "stability")
    assert check.status == "warn" and check.action == "stability" and check.rule == "R-REG-15"
    found = broken.regression_section("stability", "ret", ["mkt", "smb"])
    assert isinstance(found, st.StabilityTable) and found.available
    assert found.chow is not None and abs(found.chow.index - N // 2) <= 10
    assert found.chow.searched and "optimistic" in found.chow.sentence
    assert not found.stable and "R-REG-15" in found.sentence
    mkt = next(c for c in found.rolling if c.term == "mkt")
    assert mkt.smallest is not None and mkt.largest is not None
    assert mkt.smallest < 0.95 and mkt.largest > 1.25
    assert found.window == 80 and "R-REG-STAB-02" in found.window_reason
    assert found.chow.before[1] == pytest.approx(0.8, abs=0.1)
    assert found.chow.after[1] == pytest.approx(1.4, abs=0.1)


def test_a_steady_series_stays_quiet_and_reads_as_a_factor_model(steady):
    report = steady.regression("ret", ["mkt", "smb"])
    check = next(c for c in report.checks if c.key == "stability")
    assert check.status == "ok"
    found = steady.regression_section("stability", "ret", ["mkt", "smb"])
    assert isinstance(found, st.StabilityTable)
    assert found.cusum is not None and not found.cusum.crossed
    factor = found.factor
    assert factor is not None and factor.periods == 12 and factor.period == "a month"
    assert factor.alpha_year == pytest.approx(12 * factor.alpha)
    assert factor.alpha == pytest.approx(report.terms[0].estimate)
    assert "R-REG-STAB-04" in factor.sentence


def test_stability_needs_a_series(tmp_path):
    frame = _series(0.8).drop(columns="date")
    ds = dx.from_pandas(frame, "flat", cache_dir=tmp_path)
    found = ds.regression_section("stability", "ret", ["mkt", "smb"])
    assert isinstance(found, st.StabilityTable)
    assert not found.available and "R-REG-STAB-01" in found.reason
    assert all(c.key != "stability" for c in ds.regression("ret", ["mkt", "smb"]).checks)


@pytest.mark.parametrize(
    ("freq", "periods"), [("D", 365.0), ("B", 252.0), ("W", 52.0), ("QE", 4.0), ("YE", 1.0)]
)
def test_frequency_by_rule(freq, periods):
    dates = pd.date_range("2001-01-01", periods=60, freq=freq).strftime("%Y-%m-%d").to_numpy()
    found = st.frequency(dates)
    assert found is not None and found[0] == periods


# --- 077: resampling ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clustered(tmp_path_factory) -> dx.Dataset:
    rng = np.random.default_rng(5)
    n, g = 1_500, 15
    group = rng.integers(0, g, n)
    x, z = rng.normal(size=n) + rng.normal(size=g)[group], rng.normal(size=n)
    y = 1 + 0.0 * x + 0.5 * z + rng.normal(size=g)[group] + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "x": x, "z": z, "g": [f"g{i}" for i in group]})
    return dx.from_pandas(frame, "clustered", cache_dir=tmp_path_factory.mktemp("c"))


def test_pairs_bootstrap_agrees_with_hc0(tmp_path):
    rng = np.random.default_rng(3)
    n = 2_000
    x = rng.normal(size=n)
    y = 1 + 2 * x + rng.normal(size=n) * (0.5 + np.abs(x))
    xs = sm.add_constant(x)
    reps, m = rs.bootstrap(xs, y, "pairs")
    assert m == n
    expected = sm.OLS(y, xs).fit(cov_type="HC0").bse
    np.testing.assert_allclose(reps.std(axis=0, ddof=1), expected, rtol=0.12)


def test_cluster_bootstrap_and_wild_cluster(clustered):
    found = clustered.regression_section("resampling", "y", ["x", "z"], cluster="g")
    assert isinstance(found, rs.ResamplingTable) and found.available
    assert found.scheme == "cluster" and "R-REG-RS-01" in found.scheme_reason
    assert found.wild and found.clusters == 15 and found.weights == "Rademacher"
    by = {t.term: t for t in found.terms}
    assert by["z"].wild_p is not None and by["z"].wild_p < 0.01
    assert by["x"].wild_p is not None and by["x"].wild_p > 0.05
    frame = clustered.to_pandas()
    xs = sm.add_constant(frame[["x", "z"]].to_numpy())
    cl = sm.OLS(frame["y"].to_numpy(), xs).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(frame["g"])[0]}
    )
    assert by["x"].boot_se == pytest.approx(cl.bse[1], rel=0.35)
    assert found.permutation is None and found.permutation_reason


def test_wild_cluster_shortcut_equals_refitting_each_draw():
    """The fast algebra against a refit of every draw with statsmodels' CR1 errors."""
    rng = np.random.default_rng(9)
    n, g, draws = 300, 9, 40
    codes = rng.integers(0, g, n)
    x = sm.add_constant(rng.normal(size=(n, 2)))
    y = x @ np.array([1.0, 0.3, 0.0]) + rng.normal(size=g)[codes] + rng.normal(size=n)
    found, label = rs.wild_cluster(x, y, codes, [1, 2], draws=draws, seed=7)
    assert label == "Webb"
    v = rs.WEBB[np.random.default_rng(7).integers(0, 6, (g, draws))]
    for j, p in zip([1, 2], found, strict=True):
        t = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": codes}).tvalues[j]
        rest = np.delete(x, j, axis=1)
        r = sm.OLS(y, rest).fit()
        beyond = 0
        for b in range(draws):
            ys = r.fittedvalues + r.resid * v[codes, b]
            star = sm.OLS(ys, x).fit(cov_type="cluster", cov_kwds={"groups": codes}).tvalues[j]
            beyond += abs(star) >= abs(t) - 1e-12
        assert p == pytest.approx((beyond + 1) / (draws + 1))


def test_freedman_lane_permutation():
    rng = np.random.default_rng(4)
    n = 300
    x = sm.add_constant(rng.normal(size=(n, 2)))
    y = x @ np.array([0.0, 1.0, 0.0]) + rng.standard_t(3, n)
    t, p_strong = rs.freedman_lane(x, y, 1)
    assert p_strong == pytest.approx(1 / 1000)
    assert t == pytest.approx(sm.OLS(y, x).fit().tvalues[1], rel=1e-9)
    _, p_null = rs.freedman_lane(x, y, 2)
    assert p_null > 0.05


def test_resampling_pairs_section_and_series_blocks(tmp_path, steady):
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"x": rng.normal(size=500)})
    frame["y"] = 1 + frame["x"] + rng.normal(size=500)
    found = dx.from_pandas(frame, "p", cache_dir=tmp_path).regression_section(
        "resampling", "y", ["x"]
    )
    assert isinstance(found, rs.ResamplingTable)
    assert found.scheme == "pairs" and found.permutation is not None
    assert found.permutation.term == "x" and "R-REG-RS-05" in found.sentence
    blocks = steady.regression_section("resampling", "ret", ["mkt", "smb"])
    assert isinstance(blocks, rs.ResamplingTable) and blocks.scheme == "block"


# --- 077: selection -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wide(tmp_path_factory) -> dx.Dataset:
    rng = np.random.default_rng(11)
    n, p = 600, 40
    w = rng.normal(size=(n, p))
    d = w[:, 0] + 0.5 * w[:, 1] + rng.normal(size=n)  # confounded by c0 and c1
    y = 1.5 * d + 2 * w[:, 0] - w[:, 1] + rng.normal(size=n)
    frame = pd.DataFrame(w, columns=[f"c{i}" for i in range(p)])
    frame["d"], frame["y"] = d, y
    return dx.from_pandas(frame, "wide", cache_dir=tmp_path_factory.mktemp("w"))


def test_selection_keeps_signals_and_double_selection_matches_ols_hc3(wide):
    columns = ["d", *[f"c{i}" for i in range(40)]]
    report = wide.regression("y", columns, focus="d")
    check = next(c for c in report.checks if c.key == "selection")
    assert check.rule == "R-REG-16" and check.action == "selection"
    found = wide.regression_section("selection", "y", columns, focus="d")
    assert isinstance(found, sl.SelectionTable) and found.available
    by = {t.term: t for t in found.terms}
    assert by["d"].kept and by["c0"].kept and by["c1"].kept
    assert sum(t.kept for t in found.terms) <= 6
    assert by["c0"].stable and by["c0"].share == 1.0
    double = found.double
    assert double is not None and double.term == "d"
    assert {"c0", "c1"} <= set(double.controls)
    frame = wide.to_pandas()
    design = sm.add_constant(frame[["d", *double.controls]].to_numpy())
    expected = sm.OLS(frame["y"].to_numpy(), design).fit(cov_type="HC3")
    assert double.estimate == pytest.approx(expected.params[1], rel=1e-8)
    assert double.low is not None and double.high is not None
    assert double.low < 1.5 < double.high


def test_plugin_penalty_keeps_noise_out():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(800, 30))
    y = rng.normal(size=800)
    coef, lam = sl.lasso(x, y)
    assert lam == pytest.approx(sl.penalty(800, 30))
    assert (coef != 0).sum() <= 1


# --- 079: ML through the common engine ----------------------------------------------------------


def test_inference_is_hc3_with_adjusted_p():
    rng = np.random.default_rng(6)
    n = 500
    x = sm.add_constant(rng.normal(size=(n, 3)))
    y = x @ np.array([1.0, 2.0, 0.0, 0.1]) + rng.normal(size=n) * (1 + np.abs(x[:, 1]))
    found = inf.coefficients(x, y)
    expected = sm.OLS(y, x).fit(cov_type="HC3")
    np.testing.assert_allclose(found.std_errors, expected.bse, rtol=1e-8)
    assert found.q_values[0] is None and found.multiple.method == "holm"
    assert all(
        q is not None and p is not None and q >= p
        for p, q in zip(found.p_values[1:], found.q_values[1:], strict=True)
    )


def test_ml_linear_coefficients_use_the_engine():
    rng = np.random.default_rng(7)
    n = 800
    frame = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    frame["y"] = 2 * frame["a"] - frame["b"] + rng.normal(0, 1, n) * (1 + frame["a"].abs())
    frame["__row"] = np.arange(n)
    raw = Parser(kind="raw")
    fields = [Field(c, "numeric_continuous", raw) for c in ("a", "b", "y")]
    trained = train(
        frame,
        fields,
        fields[-1],
        "regression",
        "linear_regression",
        {"regularization": "none"},
        n,
        False,
    )
    assert trained.regression is not None and trained.regression.inference is not None
    assert "HC3" in trained.regression.inference and "R-REG-SE-05" in trained.regression.inference
    assert all(c.q_value is not None for c in trained.coefficients)
