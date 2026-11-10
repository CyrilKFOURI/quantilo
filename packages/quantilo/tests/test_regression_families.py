"""Families chosen from the target (069): rules R-REG-01…05 and 07, fits checked against
statsmodels, marginal effects, overdispersion, Box-Cox, Duan's smearing, the More section."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as statsmodels_api

import quantilo as dx
from quantilo.regression import families as fm
from quantilo.regression.analyze import FamilyAlternatives, StandardErrorTable
from quantilo.regression.service import RegressionError

sm: Any = statsmodels_api
N = 1_500


# --- Pure functions ------------------------------------------------------------------------------


def test_profile_reads_the_target():
    rng = np.random.default_rng(1)
    assert fm.profile(np.array([0.0, 1, 1, 0]), ("no", "yes")).kind == "binary"
    counts = rng.poisson(2, 500).astype(float)
    assert fm.profile(counts, None).kind == "count"
    assert fm.profile(counts, None, role="rating").kind != "count"  # a rating is not a count
    big = rng.poisson(200, 500).astype(float)
    assert fm.profile(big, None).kind == "positive"  # mean ≥ 20, no role quantity
    assert fm.profile(big, None, role="quantity").kind == "count"
    assert fm.profile(rng.uniform(0, 1, 500), None).kind == "share"
    assert fm.profile(rng.lognormal(0, 1, 500), None).kind == "positive"
    assert fm.profile(rng.normal(size=500), None).kind == "number"
    assert fm.profile(np.array([0.0, 1, 2]), None, ["a", "b", "c"]).kind == "category"


def test_choose_follows_the_rules_and_falls_back_saying_why():
    counts = fm.profile(np.random.default_rng(2).poisson(2, 300).astype(float), None)
    choice = fm.choose(counts, "auto", "injured", None)
    assert choice.kind == "poisson" and choice.rule == "R-REG-02" and choice.reads == "ratio"
    forced = fm.choose(counts, "negbin", "injured", None)
    assert (
        forced.kind == "negbin"
        and forced.rule is None
        and "chosen in the settings" in forced.reason
    )
    numbers = fm.profile(np.random.default_rng(3).normal(size=300), None)
    fallback = fm.choose(numbers, "poisson", "y", None)
    assert fallback.kind == "ols" and fallback.reason.startswith("Poisson needs whole numbers ≥ 0")
    binary = fm.profile(np.array([0.0, 1] * 50), ("No", "Yes"))
    assert fm.choose(binary, "auto", "Churn", ("No", "Yes")).rule == "R-REG-01"
    options = {o.kind: o for o in fm.options(binary)}
    assert options["probit"].available and options["probit"].exotic
    assert not options["poisson"].available and options["ols"].available


def test_marginal_effects_match_statsmodels():
    rng = np.random.default_rng(4)
    x = sm.add_constant(np.column_stack([rng.normal(size=N), rng.integers(0, 2, N)]))
    yb = (rng.random(N) < 1 / (1 + np.exp(-(0.3 + x[:, 1] - 0.5 * x[:, 2])))).astype(float)
    probit = sm.Probit(yb, x).fit(disp=0)
    found = fm.fit("probit", x, yb, "nonrobust")
    np.testing.assert_allclose(found.marginal[1:], probit.get_margeff().margeff, rtol=1e-6)
    np.testing.assert_allclose(found.marginal_se[1:], probit.get_margeff().margeff_se, rtol=1e-4)
    yc = rng.poisson(np.exp(0.2 + 0.4 * x[:, 1])).astype(float)
    poisson = sm.Poisson(yc, x).fit(disp=0)
    found = fm.fit("poisson", x, yc, "nonrobust")
    np.testing.assert_allclose(found.beta, poisson.params, rtol=1e-6)
    np.testing.assert_allclose(found.marginal[1:], poisson.get_margeff().margeff, rtol=1e-6)
    np.testing.assert_allclose(found.marginal_se[1:], poisson.get_margeff().margeff_se, rtol=1e-4)


def test_overdispersion_tells_poisson_from_negative_binomial():
    rng = np.random.default_rng(5)
    x = sm.add_constant(rng.normal(size=N))
    mu = np.exp(1 + 0.5 * x[:, 1])
    plain = rng.poisson(mu).astype(float)
    fitted = fm.fit("poisson", x, plain).mu
    assert not fm.overdispersed(fm.overdispersion(plain, fitted, 2))
    extra = rng.negative_binomial(1.0, 1 / (1 + mu)).astype(float)  # α = 1
    found = fm.overdispersion(extra, fm.fit("poisson", x, extra).mu, 2)
    assert fm.overdispersed(found) and 0.5 < found.alpha < 1.5 and found.ratio > 2


def test_boxcox_lambda_finds_the_power():
    rng = np.random.default_rng(6)
    x = sm.add_constant(rng.uniform(0, 2, N))
    logged = np.exp(1 + x[:, 1] + rng.normal(0, 0.2, N))
    lam, low, high = fm.boxcox_lambda(x, logged)
    assert abs(lam) < 0.15 and low is not None and high is not None and low < lam < high
    squared = (3 + x[:, 1] + rng.normal(0, 0.1, N)) ** 2
    assert abs(fm.boxcox_lambda(x, squared)[0] - 0.5) < 0.15


def test_smearing_is_the_mean_of_the_exponentiated_residuals():
    resid = np.array([-0.5, 0.0, 0.5])
    assert fm.smearing(resid) == pytest.approx(np.exp(resid).mean())


# --- Through the dataset -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def crashes(tmp_path_factory) -> dx.Dataset:
    rng = np.random.default_rng(7)
    speed = rng.uniform(20, 90, N)
    night = rng.choice(["dark", "light"], N, p=[0.3, 0.7])
    borough = rng.choice(["Queens", "Bronx", "Brooklyn"], N, p=[0.5, 0.3, 0.2])
    hours = rng.uniform(1, 10, N)  # exposure
    mu = hours * np.exp(-2.5 + 0.02 * speed + 0.4 * (night == "dark"))
    injured = rng.poisson(mu)
    crowded = rng.negative_binomial(1.0, 1 / (1 + mu))
    price = np.exp(11 + 0.01 * speed + 0.3 * (borough == "Bronx") + rng.normal(0, 0.6, N))
    share = np.clip(1 / (1 + np.exp(-(-1 + 0.02 * speed))) + rng.normal(0, 0.1, N), 0, 1)
    churn = rng.random(N) < 1 / (1 + np.exp(-(-2 + 0.03 * speed)))
    premium = rng.random(N) < 1 / (1 + np.exp(-(-3 + 0.04 * speed)))
    plan = np.where(premium, "premium", np.where(rng.random(N) < 0.6, "basic", "standard"))
    frame = pd.DataFrame(
        {
            "injured": injured,
            "crowded": crowded,
            "speed": speed,
            "night": night,
            "borough": borough,
            "person_years": hours,
            "price": price,
            "share": share,
            "Churn": np.where(churn, "Yes", "No"),
            "plan": plan,
        }
    )
    return dx.from_pandas(frame, "crashes", cache_dir=tmp_path_factory.mktemp("crashes"))


def _design(frame: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(
        pd.DataFrame(
            {"speed": frame["speed"], "night = dark": (frame["night"] == "dark").astype(float)}
        )
    )


def test_counts_get_poisson_with_robust_errors_and_the_exposure_offset(crashes):
    assert crashes.column("person_years").role == "exposure"
    found = crashes.regression("injured", ["speed", "night", "person_years"])
    assert found.model == "poisson" and found.family is not None
    assert found.family.rule == "R-REG-02" and found.family.reads == "ratio"
    assert found.family.exposure == "person_years"
    assert found.se.kind == "hc0" and "The Poisson model uses HC0." in found.se.reason
    assert [t.term for t in found.terms] == ["(Intercept)", "speed", "night = dark"]
    assert any(item.term == "person_years" for item in found.left_out)
    frame = crashes.to_pandas()
    expected = sm.GLM(
        frame["injured"],
        _design(frame),
        family=sm.families.Poisson(),
        offset=np.log(frame["person_years"]),
    ).fit(cov_type="HC0")
    np.testing.assert_allclose([t.estimate for t in found.terms], expected.params, rtol=1e-6)
    np.testing.assert_allclose([t.std_error for t in found.terms], expected.bse, rtol=1e-6)
    night = found.terms[2]
    assert night.ratio == pytest.approx(np.exp(expected.params.iloc[2]))
    assert 0.3 < np.log(night.ratio or 0) < 0.5
    assert "per unit of exposure" in night.sentence and "%" in night.sentence
    assert found.sentence.startswith("Poisson model (R-REG-02).")
    assert found.quality.pseudo_r2 is not None and found.quality.oos_r2 is not None
    checks = {c.key: c for c in found.checks}
    assert checks["overdispersion"].status == "ok"
    off = crashes.regression("injured", ["speed", "night"], exposure="")
    assert off.family is not None and off.family.exposure is None


def test_overdispersed_counts_propose_the_negative_binomial(crashes):
    found = crashes.regression("crowded", ["speed", "night"], exposure="")
    check = next(c for c in found.checks if c.key == "overdispersion")
    assert check.status == "warn" and check.rule == "R-REG-03"
    assert check.action == "family" and check.family == "negbin" and "AIC" in check.sentence
    nb = crashes.regression("crowded", ["speed", "night"], exposure="", family="negbin")
    assert nb.model == "negbin" and nb.family is not None and nb.family.rule is None
    frame = crashes.to_pandas()
    expected = sm.NegativeBinomial(frame["crowded"], _design(frame)).fit(disp=0, cov_type="HC0")
    np.testing.assert_allclose([t.estimate for t in nb.terms], expected.params[:-1], rtol=1e-4)
    assert nb.family.alpha == pytest.approx(expected.params.iloc[-1], rel=1e-3)
    assert nb.quality.aic is not None and nb.quality.aic < (found.quality.aic or 0)


def test_skewed_positive_target_proposes_log_and_gamma(crashes):
    found = crashes.regression("price", ["speed", "borough"])
    assert found.model == "ols"
    check = next(c for c in found.checks if c.key == "skewed_target")
    assert check.rule == "R-REG-04" and check.family == "log_ols"
    logged = crashes.regression("price", ["speed", "borough"], family="log_ols")
    frame = crashes.to_pandas()
    x = sm.add_constant(
        pd.DataFrame(
            {
                "speed": frame["speed"],
                "Bronx": (frame["borough"] == "Bronx").astype(float),
                "Brooklyn": (frame["borough"] == "Brooklyn").astype(float),
            }
        )
    )
    expected = sm.OLS(np.log(frame["price"]), x).fit(cov_type="HC3")
    np.testing.assert_allclose([t.estimate for t in logged.terms], expected.params, rtol=1e-8)
    np.testing.assert_allclose([t.std_error for t in logged.terms], expected.bse, rtol=1e-8)
    assert logged.family is not None and logged.family.smearing == pytest.approx(
        np.exp(expected.resid).mean()
    )
    bronx = next(t for t in logged.terms if t.level == "Bronx")
    assert bronx.ratio == pytest.approx(np.exp(expected.params["Bronx"]))
    assert "%" in bronx.sentence and "Duan" in logged.terms[0].sentence
    assert logged.family.note is not None and "log(`price`)" in logged.family.note
    gamma = crashes.regression("price", ["speed", "borough"], family="gamma")
    expected = sm.GLM(frame["price"], x, family=sm.families.Gamma(sm.families.links.Log())).fit(
        cov_type="HC0"
    )
    np.testing.assert_allclose([t.estimate for t in gamma.terms], expected.params, rtol=1e-5)
    np.testing.assert_allclose([t.std_error for t in gamma.terms], expected.bse, rtol=1e-5)


def test_share_target_proposes_the_fractional_logit(crashes):
    found = crashes.regression("share", ["speed"])
    check = next(c for c in found.checks if c.key == "share_target")
    assert check.rule == "R-REG-05" and check.action == "more"
    fractional = crashes.regression("share", ["speed"], family="fractional")
    frame = crashes.to_pandas()
    expected = sm.GLM(
        frame["share"], sm.add_constant(frame["speed"]), family=sm.families.Binomial()
    ).fit(cov_type="HC0")
    np.testing.assert_allclose([t.estimate for t in fractional.terms], expected.params, rtol=1e-6)
    assert fractional.terms[1].marginal is not None and "share" in fractional.terms[1].sentence
    assert not any(c.key == "separation" for c in fractional.checks)


def test_yes_no_target_stays_logistic_and_takes_probit_or_least_squares(crashes):
    found = crashes.regression("Churn", ["speed"])
    assert found.model == "logit" and found.family is not None and found.family.rule == "R-REG-01"
    probit = crashes.regression("Churn", ["speed"], family="probit")
    frame = crashes.to_pandas()
    y = frame["Churn"].astype(str).str.lower().isin(["yes", "true"]).astype(float)
    expected = sm.Probit(y, sm.add_constant(frame["speed"])).fit(disp=0, cov_type="HC0")
    np.testing.assert_allclose([t.estimate for t in probit.terms], expected.params, rtol=1e-5)
    assert probit.terms[1].marginal == pytest.approx(expected.get_margeff().margeff[0], rel=1e-5)
    assert "probability of `Churn` =" in probit.terms[1].sentence
    lpm = crashes.regression("Churn", ["speed"], family="ols")
    assert lpm.model == "ols" and lpm.se.kind == "hc3"


def test_a_few_classes_get_the_multinomial_logit(crashes):
    found = crashes.regression("plan", ["speed"])
    assert found.model == "mnlogit" and found.family is not None
    assert found.family.rule == "R-REG-07"
    reference = found.family.outcomes[0]
    frame = crashes.to_pandas()
    order = list(frame["plan"].value_counts().index)
    assert reference == order[0]
    codes = frame["plan"].map({level: i for i, level in enumerate(order)})
    expected = sm.MNLogit(codes, sm.add_constant(frame["speed"])).fit(disp=0, cov_type="HC0")
    np.testing.assert_allclose(
        [t.estimate for t in found.terms], np.asarray(expected.params).ravel(order="F"), rtol=1e-5
    )
    assert [t.outcome for t in found.terms] == [order[1], order[1], order[2], order[2]]
    assert found.multiple.tests == 2  # intercepts left out
    assert (
        found.terms[1].marginal is not None and "probability of `plan` =" in found.terms[1].sentence
    )
    table = crashes.regression_section("standard_errors", "plan", ["speed"])
    assert isinstance(table, StandardErrorTable)
    assert len(table.terms) == 4 and next(k for k in table.kinds if k.kind == "hc0").available


def test_more_than_ten_classes_are_refused(tmp_path):
    frame = pd.DataFrame({"kind": [f"k{i % 12}" for i in range(240)], "x": np.arange(240.0)})
    ds = dx.from_pandas(frame, cache_dir=tmp_path)
    with pytest.raises(RegressionError, match="12 levels"):
        ds.regression("kind", ["x"])


def test_a_family_the_target_cannot_take_falls_back_with_the_reason(crashes):
    found = crashes.regression("price", ["speed"], family="poisson")
    assert found.model == "ols" and found.family is not None
    assert found.family.reason.startswith("Poisson needs whole numbers ≥ 0")


def test_poisson_errors_side_by_side_and_clustered(crashes):
    table = crashes.regression_section("standard_errors", "injured", ["speed", "night"])
    assert isinstance(table, StandardErrorTable)
    kinds = {k.kind: k for k in table.kinds}
    assert kinds["hc0"].available and kinds["hc0"].chosen and kinds["classic"].available
    assert not kinds["hc3"].available
    clustered = crashes.regression("injured", ["speed", "night"], cluster="borough")
    assert clustered.se.kind == "cluster" and clustered.se.clusters == 3


def test_the_other_families_wait_for_more(crashes):
    binary = crashes.regression_section("families", "Churn", ["speed"])
    assert isinstance(binary, FamilyAlternatives) and binary.current == "logit"
    probit = next(a for a in binary.alternatives if a.kind == "probit")
    assert probit.available and probit.effects and probit.sentence.startswith("Probit model")
    assert not next(a for a in binary.alternatives if a.kind == "boxcox").available
    skewed = crashes.regression_section("families", "price", ["speed"])
    assert isinstance(skewed, FamilyAlternatives)
    boxcox = next(a for a in skewed.alternatives if a.kind == "boxcox")
    assert boxcox.available and boxcox.note is not None and "λ" in boxcox.note
    assert boxcox.effects[0].term == "speed"
