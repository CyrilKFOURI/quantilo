"""The regression service and `ds.regression` (067): rules, results and sections."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as statsmodels_api

import quantilo as dx
from quantilo.errors import ColumnNotFound
from quantilo.regression import service as service_module
from quantilo.regression.analyze import InfluenceTable, MoreChecks, StandardErrorTable, TermsTable
from quantilo.regression.service import RegressionError

sm: Any = statsmodels_api

N = 1_200


@pytest.fixture(scope="module")
def houses(tmp_path_factory) -> dx.Dataset:
    rng = np.random.default_rng(5)
    area = rng.uniform(30, 200, N)
    rooms = np.clip(np.round(area / 25 + rng.normal(0, 1, N)), 1, 9)
    city = rng.choice(["Paris", "Lyon", "Lille"], N, p=[0.5, 0.3, 0.2])
    noise = rng.normal(0, 1, N) * area * 40  # spread grows with the area
    price = 50_000 + 3_000 * area + 4_000 * rooms + np.where(city == "Lyon", -60_000, 0) + noise
    frame = pd.DataFrame(
        {"price": price, "area": area, "rooms": rooms, "city": city, "noise": rng.normal(size=N)}
    )
    return dx.from_pandas(frame, "houses", cache_dir=tmp_path_factory.mktemp("houses"))


def test_plain_rows_get_hc3_and_match_statsmodels(houses):
    found = houses.regression("price", ["area", "rooms", "city"])
    assert found.model == "ols"
    assert found.se.kind == "hc3" and found.se.rule == "R-REG-SE-05"
    names = [t.term for t in found.terms]
    assert names == ["(Intercept)", "area", "rooms", "city = Lyon", "city = Lille"]
    frame = houses.to_pandas()
    x = sm.add_constant(
        pd.DataFrame(
            {
                "area": frame["area"],
                "rooms": frame["rooms"],
                "Lyon": (frame["city"] == "Lyon").astype(float),
                "Lille": (frame["city"] == "Lille").astype(float),
            }
        )
    )
    expected = sm.OLS(frame["price"], x).fit(cov_type="HC3")
    np.testing.assert_allclose([t.estimate for t in found.terms], expected.params, rtol=1e-8)
    np.testing.assert_allclose([t.std_error for t in found.terms], expected.bse, rtol=1e-8)
    area = found.terms[1]
    assert area.ci_low is not None and area.ci_high is not None
    assert area.ci_low < 3_000 < area.ci_high
    assert area.evidence in ("strong", "moderate")
    assert area.sentence.startswith("Each extra unit of `area` adds")
    assert "`city` = Paris (the reference)" in found.terms[3].sentence
    assert found.references[0].level == "Paris"
    assert found.multiple.method == "holm" and found.multiple.tests == 4
    assert found.quality.rows == N and found.quality.oos_r2 is not None
    assert 0 < found.quality.oos_r2 <= found.quality.r2 + 0.05
    assert found.plots is not None and found.plots.points == N
    checks = {c.key: c for c in found.checks}
    assert checks["spread"].status == "ok"  # Breusch–Pagan rejects but HC3 already allows for it
    assert "Breusch–Pagan p < 0.001" in checks["spread"].sentence
    assert "autocorrelation" not in checks  # rows have no order


def test_classic_errors_turn_the_spread_check_red(houses):
    found = houses.regression("price", ["area", "rooms", "city"], se="classic")
    assert found.se.kind == "classic" and found.se.rule is None
    spread = next(c for c in found.checks if c.key == "spread")
    assert spread.status == "bad" and spread.action == "standard_errors"


def test_unknown_columns_are_named(houses):
    with pytest.raises(ColumnNotFound):
        houses.regression("price", ["surface"])


def test_suggestion_picks_the_amount_and_its_strongest_links(houses):
    found = houses.regression_suggestion()
    assert found.target is not None and found.target.column == "price"
    assert "amount" in found.target.reason
    picked = [p.column for p in found.columns]
    assert {"area", "rooms"} <= set(picked) and "noise" not in picked
    assert all(p.reason.startswith("R-REG-21") for p in found.columns)
    auto = houses.regression()
    assert auto.target == "price" and auto.columns == picked


def test_sections(houses):
    errors = houses.regression_section("standard_errors", "price", ["area", "rooms", "city"])
    assert isinstance(errors, StandardErrorTable)
    kinds = {k.kind: k for k in errors.kinds}
    assert kinds["hc3"].chosen and kinds["hc3"].available
    assert not kinds["hac"].available and "time order" in kinds["hac"].reason
    assert kinds["classic"].std_errors[1] != pytest.approx(kinds["hc3"].std_errors[1])
    table = houses.regression_section("influence", "price", ["area", "rooms", "city"])
    assert isinstance(table, InfluenceTable)
    assert len(table.rows) == 20 and len(table.rows[0].dfbetas) == 5
    assert table.rows[0].cook >= table.rows[-1].cook
    terms = houses.regression_section("terms", "price", ["area", "rooms", "city"])
    assert isinstance(terms, TermsTable)
    assert {v.term for v in terms.vifs} >= {"area", "rooms"}
    more = houses.regression_section("more", "price", ["area", "rooms", "city"])
    assert isinstance(more, MoreChecks)
    assert more.white is not None and more.white.p_value is not None
    assert more.removed > 0 and len(more.shifts) == 5


def test_collinear_columns_are_left_out(tmp_path):
    rng = np.random.default_rng(8)
    above = rng.uniform(500, 3000, 400)
    basement = np.where(rng.random(400) < 0.4, rng.uniform(100, 900, 400), 0.0)
    frame = pd.DataFrame(
        {
            "price": 200 * (above + basement) + rng.normal(0, 5_000, 400),
            "sqft_living": above + basement,
            "sqft_above": above,
            "sqft_basement": basement,
        }
    )
    ds = dx.from_pandas(frame, cache_dir=tmp_path)
    found = ds.regression("price", ["sqft_living", "sqft_above", "sqft_basement"])
    assert [t.term for t in found.terms] == ["(Intercept)", "sqft_living", "sqft_above"]
    assert "perfectly collinear" in found.left_out[0].reason
    assert any(c.key == "left_out" for c in found.checks)


def test_panel_rows_are_clustered_by_entity(tmp_path):
    rng = np.random.default_rng(9)
    stores, weeks = 45, 30
    store = np.repeat(np.arange(1, stores + 1), weeks)
    week = np.tile(pd.date_range("2020-01-06", periods=weeks, freq="W-MON"), stores)
    effect = rng.normal(0, 5, stores)[store - 1]
    temperature = rng.normal(15, 8, stores * weeks)
    sales = 100 + 0.5 * temperature + effect + rng.normal(0, 1, stores * weeks)
    frame = pd.DataFrame(
        {"Store": store, "Date": week, "Weekly_Sales": sales, "Temperature": temperature}
    )
    ds = dx.from_pandas(frame, cache_dir=tmp_path)
    found = ds.regression("Weekly_Sales", ["Temperature"])
    assert found.structure.kind == "panel"
    assert found.structure.entity == "Store" and found.structure.period == "Date"
    assert found.se.kind == "cluster" and found.se.rule == "R-REG-SE-01"
    assert found.se.clusters == stores and found.se.warning is None
    assert any("two-way" in hint for hint in found.hints)
    assert "by `Store`" in found.quality.oos_how
    errors = ds.regression_section("standard_errors", "Weekly_Sales", ["Temperature"])
    assert isinstance(errors, StandardErrorTable)
    kinds = {k.kind: k for k in errors.kinds}
    assert kinds["cluster2"].available and kinds["dk"].available


def test_a_dated_series_gets_hac(tmp_path):
    rng = np.random.default_rng(10)
    days = 600
    market = rng.normal(0, 0.01, days)
    error = np.zeros(days)
    for t in range(1, days):
        error[t] = 0.6 * error[t - 1] + rng.normal(0, 0.005)
    stock = 0.0002 + 1.2 * market + error
    dates = pd.bdate_range("2015-01-01", periods=days)
    order = rng.permutation(days)  # rows out of order: the engine sorts them
    frame = pd.DataFrame({"date": dates[order], "stock": stock[order], "market": market[order]})
    ds = dx.from_pandas(frame, cache_dir=tmp_path)
    found = ds.regression("stock", ["market"])
    assert found.structure.kind == "series" and found.structure.date == "date"
    assert found.se.kind == "hac" and found.se.rule == "R-REG-SE-02"
    assert found.se.lag == 5  # ⌊4 (600/100)^(2/9)⌋
    autocorrelation = next(c for c in found.checks if c.key == "autocorrelation")
    assert autocorrelation.status == "ok" and "HAC" in autocorrelation.sentence
    assert "contiguous blocks" in found.quality.oos_how
    beta = found.terms[1]
    assert beta.ci_low is not None and beta.ci_high is not None
    assert beta.ci_low < 1.2 < beta.ci_high


def test_weight_role_gives_wls(tmp_path):
    rng = np.random.default_rng(11)
    x = rng.normal(size=500)
    weight = rng.uniform(1, 5, 500)
    y = 1 + 2 * x + rng.normal(size=500) / np.sqrt(weight)
    ds = dx.from_pandas(pd.DataFrame({"y": y, "x": x, "survey_weight": weight}), cache_dir=tmp_path)
    assert ds.column("survey_weight").role == "weight"
    found = ds.regression("y", ["x"])
    assert found.model == "wls" and found.weights == "survey_weight"
    expected = sm.WLS(y, sm.add_constant(x), weights=weight).fit()
    np.testing.assert_allclose([t.estimate for t in found.terms], expected.params, rtol=1e-8)
    off = ds.regression("y", ["x"], weights="")
    assert off.model == "ols"


def test_yes_no_target_is_logistic(tmp_path):
    rng = np.random.default_rng(12)
    age = rng.uniform(1, 80, 1_500)
    sex = rng.choice(["male", "female"], 1_500)
    p = 1 / (1 + np.exp(-(1.5 - 0.03 * age - 2.0 * (sex == "male"))))
    survived = rng.random(1_500) < p
    ds = dx.from_pandas(
        pd.DataFrame({"Survived": survived, "Age": age, "Sex": sex}), cache_dir=tmp_path
    )
    found = ds.regression("Survived", ["Age", "Sex"])
    assert found.model == "logit" and found.positive == "True"
    assert found.se.kind == "hc0"
    sex_term = next(t for t in found.terms if t.column == "Sex")
    assert sex_term.marginal is not None and abs(sex_term.marginal) > 0.2
    assert "points" in sex_term.sentence and "odds ×" in sex_term.sentence
    assert found.quality.pseudo_r2 is not None and found.plots is None


def test_categories_with_many_levels_are_refused_as_target(tmp_path):
    frame = pd.DataFrame({"kind": list("abc") * 50, "x": np.arange(150.0)})
    ds = dx.from_pandas(frame, cache_dir=tmp_path)
    with pytest.raises(RegressionError, match="3 levels"):
        ds.regression("kind", ["x"])


def test_cluster_column_with_few_groups_warns(houses):
    found = houses.regression("price", ["area"], cluster="city")
    assert found.se.kind == "cluster" and found.se.rule == "R-REG-SE-03"
    assert found.se.clusters == 3 and found.se.warning is not None
    assert any(c.key == "clusters" for c in found.checks)


def test_large_data_is_sampled(tmp_path, monkeypatch):
    monkeypatch.setattr(service_module, "MAX_ROWS", 300)
    rng = np.random.default_rng(13)
    x = rng.normal(size=1_000)
    ds = dx.from_pandas(
        pd.DataFrame({"y": 2 * x + rng.normal(size=1_000), "x": x}), cache_dir=tmp_path
    )
    found = ds.regression("y", ["x"])
    assert found.quality.sampled and found.quality.loaded == 300 and found.quality.total == 1_000
    assert any(c.key == "sample" for c in found.checks)


def test_filters_apply(houses):
    lyon = houses.where("city", "eq", "Lyon").regression("price", ["area"])
    assert lyon.quality.rows < N
