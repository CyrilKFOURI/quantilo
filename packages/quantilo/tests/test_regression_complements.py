"""Complements of 074–077 (b3): stratified Cox, intervals and adjusted survival curves; the
generalised mixed model; resampling of non-linear families and the two-way wild bootstrap; York
with correlated errors and curves typed by the user; the two-pass Fama–MacBeth with Shanken's
correction and the weighted IC."""

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, optimize, special
from statsmodels.duration.hazard_regression import PHReg

import quantilo as dx
from quantilo.regression import crosssection as xs
from quantilo.regression import formula as fx
from quantilo.regression import glmm
from quantilo.regression import mixed as mx
from quantilo.regression import physics as ph
from quantilo.regression import resampling as rs
from quantilo.regression import survival as sv

# --- 076: Cox ----------------------------------------------------------------------------------


def _clinic(seed: int = 4, n: int = 900) -> pd.DataFrame:
    """Two sites with different baseline hazards, one dose effect shared by both."""
    rng = np.random.default_rng(seed)
    site = rng.choice(["north", "south"], n)
    dose = rng.normal(10, 3, n)
    base = np.where(site == "north", 0.02, 0.08)
    rate = base * np.exp(0.1 * (dose - 10))
    time = rng.exponential(1 / rate)
    follow = rng.uniform(5, 60, n)
    return pd.DataFrame(
        {
            "duration": np.minimum(time, follow).round(3),
            "died": np.where(time <= follow, "yes", "no"),
            "dose": dose,
            "site": site,
        }
    )


@pytest.fixture(scope="module")
def clinic(tmp_path_factory) -> dx.Dataset:
    return dx.from_pandas(_clinic(), "clinic", cache_dir=tmp_path_factory.mktemp("c"))


def test_stratified_cox_matches_phreg_with_strata(clinic):
    table = clinic.regression_method("survival", "duration", ["dose", "site"], strata="site")
    assert isinstance(table, sv.CoxTable) and table.available, table.reason
    frame = _clinic()
    event = (frame["died"] == "yes").to_numpy(dtype=float)
    codes = (frame["site"] == "north").to_numpy(dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model: Any = PHReg(
            frame["duration"].to_numpy(),
            frame[["dose"]].to_numpy(),
            status=event,
            strata=codes,
            ties="breslow",
        ).fit()
    assert [t.term for t in table.terms] == ["dose"]  # the strata column has no ratio
    assert table.terms[0].coefficient == pytest.approx(float(model.params[0]), rel=1e-6)
    assert table.strata == "site" and table.strata_count == 2
    assert {c.label for c in table.curves} == {"site = north", "site = south"}
    south = next(c for c in table.curves if c.label == "site = south")
    north = next(c for c in table.curves if c.label == "site = south".replace("south", "north"))
    assert south.median is not None and (north.median is None or north.median > south.median)
    assert "R-REG-SURV-03" in " ".join(table.notes)


def test_adjusted_curves_follow_the_baseline_and_the_profile(clinic):
    table = clinic.regression_method("survival", "duration", ["dose", "site"])
    assert isinstance(table, sv.CoxTable) and table.available
    labels = [c.label for c in table.curves]
    assert len(labels) == 2 and all(label.startswith("site = ") for label in labels)
    for curve in table.curves:
        assert curve.survival[0] == 1.0 and all(
            a >= b - 1e-12 for a, b in zip(curve.survival, curve.survival[1:], strict=False)
        )
    assert "R-REG-SURV-05" in table.curves_sentence


def test_a_start_column_reads_rows_as_intervals(tmp_path):
    frame = _clinic(5, 600)
    rng = np.random.default_rng(9)
    frame["start"] = (frame["duration"] * rng.uniform(0, 0.5, len(frame))).round(3)
    ds = dx.from_pandas(frame, "intervals", cache_dir=tmp_path)
    offered = ds.regression_methods("duration", ["dose"])
    assert offered.start is not None and offered.start.column == "start"
    table = ds.regression_method("survival", "duration", ["dose"])
    assert isinstance(table, sv.CoxTable) and table.available and table.start == "start"
    event = (frame["died"] == "yes").to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model: Any = PHReg(
            frame["duration"].to_numpy(),
            frame[["dose"]].to_numpy(),
            status=event,
            entry=frame["start"].to_numpy(),
            ties="breslow",
        ).fit()
    assert table.terms[0].coefficient == pytest.approx(float(model.params[0]), rel=1e-6)
    off = ds.regression_method("survival", "duration", ["dose"], start="")
    assert isinstance(off, sv.CoxTable) and off.start is None


def test_a_category_breaking_proportional_hazards_is_proposed_as_strata(tmp_path):
    rng = np.random.default_rng(2)
    n = 1500
    group = rng.choice(["a", "b"], n)
    early = rng.exponential(1 / np.where(group == "a", 0.2, 0.02))
    late = 10 + rng.exponential(1 / np.where(group == "a", 0.01, 0.2))
    time = np.where(early < 10, early, late)
    frame = pd.DataFrame(
        {"duration": np.minimum(time, 40), "event": time <= 40, "x": rng.normal(size=n), "g": group}
    )
    ds = dx.from_pandas(frame, "crossing", cache_dir=tmp_path)
    table = ds.regression_method("survival", "duration", ["x", "g"], event="event")
    assert isinstance(table, sv.CoxTable) and table.available
    assert "R-REG-SURV-03" in table.strata_hint


# --- 076: generalised mixed model ---------------------------------------------------------------


def test_adaptive_quadrature_matches_numerical_integration():
    rng = np.random.default_rng(0)
    x = np.column_stack([np.ones(30), rng.normal(size=30)])
    y = (rng.uniform(size=30) < 0.4).astype(float)
    beta, tau = np.array([-0.3, 0.8]), 1.3
    groups = np.zeros(30, dtype=int)
    eta = x @ beta

    def integrand(u: float) -> float:
        rows = float(np.sum(y * (eta + u) - np.logaddexp(0.0, eta + u)))
        return math.exp(rows - 0.5 * u * u / tau**2) / math.sqrt(2 * math.pi * tau**2)

    exact = math.log(integrate.quad(integrand, -15, 15, limit=200, epsabs=0, epsrel=1e-12)[0])
    assert glmm.loglike("logit", x, y, groups, beta, tau) == pytest.approx(exact, abs=1e-6)
    counts = rng.poisson(2.0, 30).astype(float)

    def poisson(u: float) -> float:
        rows = float(np.sum(counts * (eta + u) - np.exp(eta + u) - special.gammaln(counts + 1)))
        return math.exp(rows - 0.5 * u * u / tau**2) / math.sqrt(2 * math.pi * tau**2)

    exact = math.log(integrate.quad(poisson, -15, 15, limit=200, epsabs=0, epsrel=1e-12)[0])
    assert glmm.loglike("poisson", x, counts, groups, beta, tau) == pytest.approx(exact, abs=1e-6)


def _schools(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(40), 50)
    u = rng.normal(0, 1.0, 40)[g]
    hours = rng.normal(5, 2, len(g))
    eta = -1.0 + 0.3 * hours + u
    passed = rng.uniform(size=len(g)) < special.expit(eta)
    return pd.DataFrame({"passed": passed, "hours": hours, "school": [f"s{i}" for i in g]})


def test_a_random_intercept_logit_recovers_its_variance(tmp_path):
    ds = dx.from_pandas(_schools(), "schools", cache_dir=tmp_path)
    table = ds.regression_method("mixed", "passed", ["hours"], cluster="school")
    assert isinstance(table, mx.MixedTable) and table.available, table.reason
    assert table.family == "logit" and table.groups == 40
    hours = next(t for t in table.terms if t.term == "hours")
    assert hours.estimate == pytest.approx(0.3, abs=0.06)
    assert table.group_variance is not None and math.sqrt(table.group_variance) == pytest.approx(
        1.0, abs=0.3
    )
    assert table.lr_p is not None and table.lr_p < 0.001
    assert abs(hours.least_squares) < abs(hours.estimate)  # the averaged slope is flatter
    assert "R-REG-MIX-02" in table.sentence


# --- 077: non-linear families and two-way wild bootstrap -----------------------------------------


def test_a_logistic_model_is_resampled_by_refitting(tmp_path):
    rng = np.random.default_rng(6)
    n = 800
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    churn = rng.uniform(size=n) < special.expit(-0.5 + 0.9 * x1 - 0.4 * x2)
    ds = dx.from_pandas(pd.DataFrame({"churn": churn, "x1": x1, "x2": x2}), cache_dir=tmp_path)
    table = ds.regression_section("resampling", "churn", ["x1", "x2"])
    assert isinstance(table, rs.ResamplingTable)
    assert table.available, table.reason
    assert table.family == "Logistic" and table.draws == rs.NONLINEAR_DRAWS
    x1_term = next(t for t in table.terms if t.term == "x1")
    assert x1_term.high is not None and x1_term.low is not None
    assert x1_term.boot_low is not None and x1_term.boot_high is not None
    first_plane_se = (x1_term.high - x1_term.low) / (2 * 1.959963984540054)
    assert x1_term.boot_se == pytest.approx(first_plane_se, rel=0.25)
    assert x1_term.boot_low < 0.9 < x1_term.boot_high
    assert "R-REG-RS-06" in table.notes[0] and not table.wild


def _two_way_brute(x, y, first, second, j, v_first):
    """Refit every draw and compute the CGM variance of β_j the slow way."""
    n, k = x.shape

    def var_j(e):
        c = np.linalg.pinv(x.T @ x) @ x.T
        total = 0.0
        for codes, sign in ((first, 1), (second, 1), (first * 1000 + second, -1)):
            _, inv = np.unique(codes, return_inverse=True)
            g = inv.max() + 1
            sums = np.bincount(inv, c[j] * e, g)
            total += sign * g / (g - 1) * (n - 1) / (n - k) * float(sums @ sums)
        return total

    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    t = beta[j] / math.sqrt(var_j(y - x @ beta))
    rest = np.delete(x, j, axis=1)
    fitted = rest @ np.linalg.lstsq(rest, y, rcond=None)[0]
    resid = y - fitted
    _, inv = np.unique(first, return_inverse=True)
    beyond = 0
    for b in range(v_first.shape[1]):
        ystar = fitted + resid * v_first[inv, b]
        bs = np.linalg.lstsq(x, ystar, rcond=None)[0]
        vs = var_j(ystar - x @ bs)
        ts = bs[j] / math.sqrt(vs) if vs > 0 else 0.0
        beyond += abs(ts) >= abs(t) - 1e-12
    return (beyond + 1) / (v_first.shape[1] + 1)


def test_the_two_way_wild_bootstrap_matches_refitting_each_draw():
    rng = np.random.default_rng(7)
    entities, periods = 8, 30
    first = np.repeat(np.arange(entities), periods)
    second = np.tile(np.arange(periods), entities)
    x = np.column_stack([np.ones(len(first)), rng.normal(size=len(first))])
    y = 0.1 * x[:, 1] + rng.normal(size=8)[first] + rng.normal(size=30)[second]
    y = y + rng.normal(size=len(first))
    draws = 99
    found, label, on_first = rs.wild_two_way(x, y, first, second, [1], draws=draws, seed=3)
    assert label == "Webb" and on_first  # 8 entities < 30 periods, fewer than 12
    v = rs.WEBB[np.random.default_rng(3).integers(0, 6, (entities, draws))]
    assert found[0] == pytest.approx(_two_way_brute(x, y, first, second, 1, v), abs=1e-12)


# --- 075: York with correlated errors, formulas typed by the user --------------------------------


def test_york_with_correlated_errors_minimises_its_weighted_squares():
    rng = np.random.default_rng(11)
    n = 25
    x_true = np.linspace(0, 10, n)
    sx, sy = rng.uniform(0.2, 0.6, n), rng.uniform(0.3, 0.8, n)
    r = rng.uniform(-0.8, 0.8, n)
    x = x_true + rng.normal(0, sx)
    y = 1.0 + 0.7 * x_true + rng.normal(0, sy)
    b, a, _, _ = ph.york(x, y, sx, sy, r)
    wx, wy = 1 / sx**2, 1 / sy**2

    def s(params: np.ndarray) -> float:
        intercept, slope = params
        w = wx * wy / (wx + slope**2 * wy - 2 * slope * r * np.sqrt(wx * wy))
        return float((w * (y - intercept - slope * x) ** 2).sum())

    best = optimize.minimize(s, np.array([a, b]), method="Nelder-Mead",
                             options={"xatol": 1e-10, "fatol": 1e-12})  # fmt: skip
    assert b == pytest.approx(best.x[1], abs=1e-6) and a == pytest.approx(best.x[0], abs=1e-6)
    plain = ph.york(x, y, sx, sy)
    assert ph.york(x, y, sx, sy, np.zeros(n))[0] == pytest.approx(plain[0], rel=1e-12)


@pytest.mark.parametrize(
    "text", ["__import__('os')", "x.real", "a * foo(x)", "a + b", "x + 1", "a if x else b"]
)
def test_a_formula_that_is_not_a_curve_of_x_is_refused(text):
    with pytest.raises(fx.FormulaError):
        fx.parse(text)


def test_a_typed_formula_is_fitted_like_the_built_in_shapes(tmp_path):
    rng = np.random.default_rng(12)
    x = np.linspace(0, 5, 120)
    y = 3.0 * np.exp(-0.8 * x) + 0.5 + rng.normal(0, 0.05, len(x))
    ds = dx.from_pandas(pd.DataFrame({"dose": x, "level": y}), cache_dir=tmp_path)
    table = ds.regression_method("curves", "level", ["dose"], formula="a * exp(-k * x) + c")
    assert isinstance(table, ph.CurvesTable) and table.available
    assert table.formula == "a * exp(-k * x) + c"
    mine = next(f for f in table.fits if f.kind == "formula")
    values = {p.name: p.estimate for p in mine.parameters}
    assert values["a"] == pytest.approx(3.0, abs=0.1)
    assert values["k"] == pytest.approx(0.8, abs=0.05)
    built = next(f for f in table.fits if f.kind == "exponential")
    assert mine.aic == pytest.approx(built.aic, abs=1e-6)  # the same curve, the same optimum
    wrong = ds.regression_method("curves", "level", ["dose"], formula="a * exp(")
    assert isinstance(wrong, ph.CurvesTable) and wrong.formula_error and wrong.available


# --- 074: two passes with Shanken's correction, weighted IC ---------------------------------------


def _assets(seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assets, months = 40, 120
    dates = pd.date_range("2010-01-31", periods=months, freq="ME")
    market = rng.normal(0.008, 0.045, months)
    beta = rng.uniform(0.4, 1.6, assets)
    size = rng.normal(0, 1, (assets, months))
    rows = []
    for i in range(assets):
        ret = beta[i] * market + 0.002 * size[i] + rng.normal(0, 0.03, months)
        for t in range(months):
            rows.append((f"A{i:02d}", dates[t], ret[t], market[t], size[i, t], 1 + i))
    return pd.DataFrame(rows, columns=["Store", "date", "ret", "mkt", "size", "cap"])


def test_a_factor_is_read_by_two_passes_with_shanken(tmp_path):
    frame = _assets()
    ds = dx.from_pandas(frame, "assets", cache_dir=tmp_path)
    table = ds.regression_method("cross_section", "ret", ["mkt", "size"])
    assert isinstance(table, xs.CrossSectionTable) and table.available, table.reason
    assert table.two_pass is not None and table.two_pass.factors == ["mkt"]
    assert [t.term for t in table.terms] == ["(Intercept)", "size"]  # mkt is not priced each month
    premium = next(t for t in table.two_pass.terms if t.term == "mkt")
    # by hand: betas by asset, then a cross-section a month on [1, beta]
    betas = {}
    for asset, rows in frame.groupby("Store"):
        design = np.column_stack([np.ones(len(rows)), rows["mkt"]])
        betas[asset] = np.linalg.lstsq(design, rows["ret"].to_numpy(), rcond=None)[0][1]
    lambdas, realised = [], []
    for _, rows in frame.groupby("date"):
        design = np.column_stack(
            [np.ones(len(rows)), rows["Store"].map(lambda store: betas[store])]
        )
        lambdas.append(np.linalg.lstsq(design, rows["ret"].to_numpy(), rcond=None)[0])
        realised.append(rows["mkt"].iloc[0])
    lam = np.vstack(lambdas)
    t = len(lam)
    var_f = float(np.var(realised, ddof=1))
    c = lam[:, 1].mean() ** 2 / var_f
    expected = math.sqrt(np.var(lam[:, 1], ddof=1) / t * (1 + c) + var_f / t)
    assert premium.estimate == pytest.approx(lam[:, 1].mean(), rel=1e-9)
    assert premium.shanken_se == pytest.approx(expected, rel=1e-9)
    assert premium.shanken_se is not None and premium.fm_se is not None
    assert premium.shanken_se > premium.fm_se
    assert premium.factor_mean == pytest.approx(float(np.mean(realised)), rel=1e-9)
    assert "R-REG-XS-08" in table.two_pass.sentence


def test_the_ic_is_weighted_by_the_row_weights(tmp_path):
    frame = _assets(22)
    ds = dx.from_pandas(frame, "weighted", cache_dir=tmp_path)
    table = ds.regression_method("cross_section", "ret", ["size"], weights="cap")
    assert isinstance(table, xs.CrossSectionTable) and table.available
    assert table.weights == "cap"
    ic = next(found for found in table.ics if found.column == "size")
    by_hand = []
    for _, rows in frame.groupby("date"):
        rx = rows["size"].rank().to_numpy()
        ry = rows["ret"].rank().to_numpy()
        w = rows["cap"].to_numpy(dtype=float)
        mx_, my_ = np.average(rx, weights=w), np.average(ry, weights=w)
        cov_ = np.average((rx - mx_) * (ry - my_), weights=w)
        sx_ = np.average((rx - mx_) ** 2, weights=w)
        sy_ = np.average((ry - my_) ** 2, weights=w)
        by_hand.append(cov_ / math.sqrt(sx_ * sy_))
    assert ic.weighted == pytest.approx(float(np.mean(by_hand)), rel=1e-9)
