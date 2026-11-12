"""Health (076): Cox against statsmodels PHReg with the Schoenfeld test, Tobit by maximum
likelihood, the mixed model against MixedLM, and R-REG-06 in the first plane."""

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.regression.mixed_linear_model import MixedLM

import quantilo as dx
from quantilo.regression import mixed as mx
from quantilo.regression import survival as sv


def _churn(seed: int = 1, varying: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 600
    charges = rng.normal(70, 20, n)
    age = rng.normal(50, 10, n)
    rate = np.exp(-4 + 0.02 * charges + 0.06 * (age - 50))
    time = rng.exponential(1 / rate)
    if varying:  # age raises the hazard early only: proportional hazards fail
        late = rng.exponential(1 / np.exp(-4 + 0.02 * charges - 0.06 * (age - 50)))
        time = np.where(time > 15, 15 + late, time)
    follow = rng.uniform(5, 80, n)
    frame = pd.DataFrame(
        {
            "tenure": np.minimum(time, follow).round(2),
            "Churn": np.where(time <= follow, "Yes", "No"),
            "MonthlyCharges": charges,
            "Age": age,
        }
    )
    return frame


@pytest.fixture(scope="module")
def churn(tmp_path_factory) -> dx.Dataset:
    return dx.from_pandas(_churn(), "churn", cache_dir=tmp_path_factory.mktemp("c"))


def test_the_first_plane_says_time_to_event(churn):
    report = churn.regression("tenure", ["MonthlyCharges", "Age"])
    check = next(c for c in report.checks if c.key == "time_to_event")
    assert check.rule == "R-REG-06" and check.action == "more" and "`Churn`" in check.sentence
    offered = churn.regression_methods("tenure", ["MonthlyCharges", "Age"])
    assert offered.event is not None and offered.event.column == "Churn"
    survival = next(m for m in offered.methods if m.kind == "survival")
    assert survival.proposed and offered.methods[0].proposed


def test_cox_matches_statsmodels_phreg(churn):
    table = churn.regression_method("survival", "tenure", ["MonthlyCharges", "Age"])
    assert isinstance(table, sv.CoxTable) and table.available, table.reason
    frame = _churn()
    event = (frame["Churn"] == "Yes").to_numpy(dtype=float)
    x = frame[["MonthlyCharges", "Age"]].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found: Any = PHReg(frame["tenure"].to_numpy(), x, status=event, ties="breslow").fit()
    assert [t.coefficient for t in table.terms] == pytest.approx(list(found.params), rel=1e-6)
    assert [t.std_error for t in table.terms] == pytest.approx(list(found.bse), rel=1e-5)
    assert table.events == int(event.sum()) and table.event == "Churn = True"
    assert table.terms[0].hazard_ratio == pytest.approx(math.exp(found.params[0]), rel=1e-6)
    assert table.lr_p is not None and table.lr_p < 0.001
    assert all(t.proportional for t in table.terms)


def test_schoenfeld_flags_an_effect_that_changes_with_time(tmp_path):
    ds = dx.from_pandas(_churn(3, varying=True), "varying", cache_dir=tmp_path)
    table = ds.regression_method("survival", "tenure", ["MonthlyCharges", "Age"])
    assert isinstance(table, sv.CoxTable) and table.available
    age = next(t for t in table.terms if t.term == "Age")
    assert age.ph_p is not None and age.ph_p < 0.01 and age.proportional is False
    assert "`Age`" in table.ph_sentence


def test_tobit_without_censoring_is_least_squares():
    rng = np.random.default_rng(0)
    x = np.column_stack([np.ones(300), rng.normal(size=300)])
    y = x @ np.array([1.0, 2.0]) + rng.normal(0, 1, 300)
    theta, _, _, converged = sv.tobit_fit(x, y, None, None)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    assert converged
    np.testing.assert_allclose(theta[:2], beta, rtol=1e-6)
    assert math.exp(theta[2]) == pytest.approx(math.sqrt(resid @ resid / 300), rel=1e-6)


def test_tobit_recovers_the_latent_slope_at_zero(tmp_path):
    rng = np.random.default_rng(5)
    n = 2_000
    income = rng.normal(0, 1, n)
    spend = np.maximum(0.0, 0.5 + 2.0 * income + rng.normal(0, 1.5, n))
    ds = dx.from_pandas(pd.DataFrame({"income": income, "spend": spend}), "t", cache_dir=tmp_path)
    report = ds.regression("spend", ["income"])
    check = next(c for c in report.checks if c.key == "censored")
    assert check.rule == "R-REG-06" and check.action == "more"
    table = ds.regression_method("tobit", "spend", ["income"])
    assert isinstance(table, sv.TobitTable) and table.available and table.lower == 0
    slope = next(t for t in table.terms if t.term == "income")
    assert slope.estimate == pytest.approx(2.0, abs=0.12)
    assert slope.least_squares < 1.8 and 0 < slope.marginal < slope.estimate
    assert table.sigma == pytest.approx(1.5, abs=0.1)


def _schools(seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    schools, pupils = 30, 25
    school = np.repeat(np.arange(schools), pupils)
    effect = rng.normal(0, 3, schools)[school]
    hours = rng.normal(5, 2, schools * pupils)
    score = 50 + 2 * hours + effect + rng.normal(0, 4, schools * pupils)
    return pd.DataFrame({"school": [f"S{s}" for s in school], "hours": hours, "score": score})


def test_mixed_model_matches_mixedlm(tmp_path):
    frame = _schools()
    ds = dx.from_pandas(frame, "schools", cache_dir=tmp_path)
    offered = ds.regression_methods("score", ["hours"], cluster="school")
    mixed = next(m for m in offered.methods if m.kind == "mixed")
    assert mixed.proposed and offered.group is not None and offered.group.column == "school"
    table = ds.regression_method("mixed", "score", ["hours"], cluster="school")
    assert isinstance(table, mx.MixedTable) and table.available, table.reason
    codes = pd.factorize(frame["school"])[0]
    x = np.column_stack([np.ones(len(frame)), frame["hours"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found: Any = MixedLM(frame["score"].to_numpy(), x, groups=codes).fit(reml=True)
    assert [t.estimate for t in table.terms] == pytest.approx(list(found.fe_params), rel=1e-6)
    assert table.group_variance == pytest.approx(float(np.asarray(found.cov_re)[0, 0]), rel=1e-4)
    assert table.icc is not None and 0.2 < table.icc < 0.5
    assert table.lr_p is not None and table.lr_p < 0.001 and table.groups == 30


def test_mixed_model_needs_groups(tmp_path):
    ds = dx.from_pandas(_schools(), "nogroup", cache_dir=tmp_path)
    table = ds.regression_method("mixed", "score", ["hours"])
    assert isinstance(table, mx.MixedTable) and not table.available
    assert "Grouped by" in table.reason
