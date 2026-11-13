"""Terms of a regression (070): curves, squares, logs, interactions, contrasts, centring, dates,
the Terms section and the partial plots, against statsmodels on the same columns."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as statsmodels_api
from scipy.interpolate import CubicSpline

import quantilo as dx
from quantilo.regression import terms as tm
from quantilo.regression.analyze import TermsTable
from quantilo.regression.service import RegressionError

sm: Any = statsmodels_api

N = 1_500


@pytest.fixture(scope="module")
def homes(tmp_path_factory) -> dx.Dataset:
    rng = np.random.default_rng(11)
    area = rng.uniform(30, 200, N)
    age = rng.uniform(0, 80, N)
    lot = np.where(rng.random(N) < 0.1, 0.0, rng.lognormal(6, 1, N))
    city = rng.choice(["Paris", "Lyon", "Lille"], N, p=[0.5, 0.3, 0.2])
    sold = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.uniform(0, 4 * 365.25, N), unit="D")
    years = (sold - pd.Timestamp("2020-01-01")).days / 365.25
    slope = np.where(city == "Lyon", 1_500, 3_000)  # the effect of the area depends on the city
    price = (
        40_000
        + slope * area
        + 25 * (age - 40) ** 2  # a curve in the age
        + 8_000 * np.log1p(lot)
        + 5_000 * years
        + rng.normal(0, 15_000, N)
    )
    frame = pd.DataFrame(
        {
            "price": price,
            "area": area,
            "age": age,
            "lot": lot,
            "city": city,
            "sold": sold,
            "rooms": np.round(area / 25 + rng.normal(0, 1, N)),
        }
    )
    return dx.from_pandas(frame, "homes", cache_dir=tmp_path_factory.mktemp("homes"))


def _frame(ds: dx.Dataset) -> pd.DataFrame:
    return ds.to_pandas()


def test_restricted_spline_is_a_natural_cubic_spline():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 10, 400))
    knots = tm.knots_of(x)
    assert len(knots) == 5
    inside = (x >= knots[0]) & (x <= knots[-1])
    truth = CubicSpline(knots, [1.0, 3.0, -2.0, 0.5, 4.0], bc_type="natural")(x[inside])
    basis = np.column_stack([np.ones(inside.sum()), x[inside], tm.rcs(x[inside], knots)])
    coef, *_ = np.linalg.lstsq(basis, truth, rcond=None)
    np.testing.assert_allclose(basis @ coef, truth, atol=1e-8)
    beyond = np.linspace(knots[-1], knots[-1] + 5, 20)  # linear beyond the outer knot
    second = np.diff(tm.rcs(beyond, knots), n=2, axis=0)
    np.testing.assert_allclose(second, 0, atol=1e-9)


def test_parse_terms():
    assert tm.parse("spline:age") == tm.TermSpec("spline", ("age",))
    assert tm.parse("interaction:a|b").columns == ("a", "b")
    assert tm.parse("reference:city|Lyon").level == "Lyon"
    with pytest.raises(tm.TermError):
        tm.parse("curve:age")
    with pytest.raises(tm.TermError):
        tm.parse("interaction:a|a")
    plan = tm.plan_of(["square:x", "cube:x", "reference:c|a", "reference:c|b"])
    assert [s.key for s in plan.specs] == ["cube:x", "reference:c|b"]


def test_spline_matches_statsmodels_and_reads_as_a_curve(homes):
    found = homes.regression("price", ["area", "age"], terms=["spline:age"], se="hc3")
    names = [t.term for t in found.terms]
    assert names[:3] == ["(Intercept)", "area", "age"]
    assert names[3:] == ["age · curve 1", "age · curve 2", "age · curve 3"]
    assert found.terms[3].group == "spline:age"
    assert "part of the curve of `age`" in found.terms[3].sentence
    assert "curve" not in found.sentence  # the conclusion speaks of a column, not a curve's part
    collinearity = next(c for c in found.checks if c.key == "collinearity")
    assert collinearity.status == "ok"  # a curve's columns are read with their column (GVIF)
    frame = _frame(homes)
    age = frame["age"].to_numpy()
    basis = tm.rcs(age, tm.knots_of(age))
    x = np.column_stack([np.ones(N), frame["area"], age, basis])
    expected = sm.OLS(frame["price"].to_numpy(), x).fit(cov_type="HC3")
    np.testing.assert_allclose([t.estimate for t in found.terms], expected.params, rtol=1e-7)
    np.testing.assert_allclose([t.std_error for t in found.terms], expected.bse, rtol=1e-7)
    table = homes.regression_section("terms", "price", ["area", "age"], terms=["spline:age"])
    assert isinstance(table, TermsTable)
    tests = {t.question: t for t in table.tests}
    curve = tests["Is the effect of `age` a curve?"]
    assert curve.p_value is not None and curve.p_value < 0.001 and curve.df_num == 3
    assert curve.sentence.startswith("`age` is a curve")
    drawn = next(c for c in table.curves if c.column == "age")
    lowest = int(np.argmin(drawn.lines[0].y))
    assert 30 < drawn.x[lowest] < 50  # the true curve is lowest at 40
    assert all(lo <= y <= hi for lo, y, hi in zip(drawn.lines[0].low, drawn.lines[0].y,
                                                  drawn.lines[0].high, strict=True))  # fmt: skip
    note = next(n for n in table.notes if n.key == "spline:age")
    assert note.status == "used" and "5 knots" in note.reason


def test_square_is_taken_around_the_mean(homes):
    found = homes.regression("price", ["area"], terms=["square:age"])
    names = [t.term for t in found.terms]
    assert names == ["(Intercept)", "area", "age", "age²"]  # age added (R-REG-TERM-01)
    frame = _frame(homes)
    age = frame["age"].to_numpy()
    x = np.column_stack([np.ones(N), frame["area"], age, (age - age.mean()) ** 2])
    expected = sm.OLS(frame["price"].to_numpy(), x).fit()
    np.testing.assert_allclose([t.estimate for t in found.terms], expected.params, rtol=1e-7)
    assert found.terms[3].estimate == pytest.approx(25, rel=0.15)
    assert "slope at the mean of `age`" in found.terms[2].sentence
    table = homes.regression_section("terms", "price", ["area"], terms=["square:age"])
    assert isinstance(table, TermsTable)
    assert any(n.status == "added" and n.label == "`age`" for n in table.notes)


def test_interaction_with_a_category_gives_one_slope_per_level(homes):
    found = homes.regression("price", ["area", "city"], terms=["interaction:area|city"])
    names = [t.term for t in found.terms]
    assert "area × city = Lyon" in names and "area × city = Lille" in names
    lyon = next(t for t in found.terms if t.term == "area × city = Lyon")
    assert lyon.estimate == pytest.approx(-1_500, abs=300)
    assert "With `city` = Lyon, each extra unit of `area` adds" in lyon.sentence
    area = next(t for t in found.terms if t.term == "area")
    assert "`city` = Paris (the reference)" in area.sentence
    table = homes.regression_section(
        "terms", "price", ["area", "city"], terms=["interaction:area|city"]
    )
    assert isinstance(table, TermsTable)
    slopes = {s.at: s.slope for s in table.slopes}
    assert slopes["`city` = Lyon"] == pytest.approx(1_500, abs=300)
    assert slopes["`city` = Paris"] == pytest.approx(3_000, abs=300)
    assert slopes["`city` = Lyon"] == pytest.approx(area.estimate + lyon.estimate)
    test = next(t for t in table.tests if t.key == "interaction:area|city")
    assert test.p_value is not None and test.p_value < 0.001
    lines = next(c for c in table.curves if c.column == "area").lines
    assert [line.label for line in lines] == ["`city` = Paris", "`city` = Lyon", "`city` = Lille"]


def test_two_numbers_interacting_propose_centring(homes):
    found = homes.regression("price", ["area", "age"], terms=["interaction:area|age"])
    checks = {c.key: c for c in found.checks}
    assert checks["centering"].rule == "R-REG-TERM-05" and checks["centering"].action == "center"
    area = next(t for t in found.terms if t.term == "area")
    assert "Read at `age` = 0" in area.sentence
    centred = homes.regression(
        "price", ["area", "age"], terms=["interaction:area|age"], center=True
    )
    assert "centering" not in {c.key for c in centred.checks}
    product = [t.estimate for t in found.terms if t.term == "area × age"]
    again = [t.estimate for t in centred.terms if t.term == "area × age"]
    np.testing.assert_allclose(product, again, rtol=1e-8)  # the product's slope does not change
    area_c = next(t for t in centred.terms if t.term == "area")
    assert "at the mean of `age`" in area_c.sentence
    assert centred.terms[0].sentence.startswith("With every number at its mean")
    frame = _frame(homes)
    mean = float(frame["price"].to_numpy(dtype=float).mean())
    assert centred.terms[0].estimate == pytest.approx(mean, rel=0.05)


def test_log_of_a_column_with_zeros_and_negatives(homes):
    found = homes.regression("price", ["area", "lot"], terms=["log:lot"])
    names = [t.term for t in found.terms]
    assert names == ["(Intercept)", "area", "log(1 + lot)"]
    term = found.terms[2]
    assert term.estimate == pytest.approx(8_000, rel=0.1)
    assert term.sentence.startswith("A 1 % rise in 1 + `lot` adds")
    frame = _frame(homes)
    x = sm.add_constant(np.column_stack([frame["area"], np.log1p(frame["lot"])]))
    expected = sm.OLS(frame["price"].to_numpy(), x).fit()
    np.testing.assert_allclose([t.estimate for t in found.terms], expected.params, rtol=1e-7)
    elastic = homes.regression("price", ["area", "lot"], terms=["log:area"], family="log_ols")
    log_area = next(t for t in elastic.terms if t.term == "log(area)")
    assert "(an elasticity)" in log_area.sentence
    shifted = homes.regression("price", ["area", "age"], terms=["log:age"])  # age has no zero
    assert "log(age)" in [t.term for t in shifted.terms]


def test_negative_values_refuse_the_log(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    frame = pd.DataFrame({"y": 2 * x + rng.normal(size=300), "x": x})
    ds = dx.from_pandas(frame, "negative", cache_dir=tmp_path)
    found = ds.regression("y", ["x"], terms=["log:x"])
    assert [t.term for t in found.terms] == ["(Intercept)", "x"]
    check = next(c for c in found.checks if c.key == "terms")
    assert check.status == "warn" and "below 0" in check.sentence


def test_reference_and_deviation_coding(homes):
    chosen = homes.regression("price", ["city"], terms=["reference:city|Lyon"])
    assert chosen.references[0].level == "Lyon"
    assert "chosen in the settings" in chosen.references[0].reason
    deviation = homes.regression("price", ["city"], coding="deviation")
    frame = _frame(homes)
    city, price = frame["city"].to_numpy().astype(str), frame["price"].to_numpy(dtype=float)
    means = {level: float(price[city == level].mean()) for level in np.unique(city)}
    grand = float(np.mean(list(means.values())))  # unweighted average of the levels
    for term in deviation.terms[1:]:
        level = term.level
        assert level is not None
        assert term.estimate == pytest.approx(means[level] - grand, rel=1e-8)
        assert "the average of the `city` levels" in term.sentence
    table = homes.regression_section("terms", "price", ["city"], coding="deviation")
    assert isinstance(table, TermsTable)
    derived = next(c for c in table.contrasts if c.derived)
    assert derived.level == "Paris"
    assert derived.estimate == pytest.approx(means["Paris"] - grand, rel=1e-8)
    assert derived.low is not None and derived.low < derived.estimate


def test_dates_as_terms(homes):
    found = homes.regression(
        "price",
        ["area", "age", "lot", "city", "sold"],
        terms=["trend:sold", "square:age", "log:lot", "interaction:area|city"],
    )
    assert "sold" not in {item.term for item in found.left_out}  # it entered as its trend
    plain = homes.regression("price", ["area", "sold"])
    left = {item.term: item.reason for item in plain.left_out}
    assert "Advanced › Terms" in left["sold"]
    trend = next(t for t in found.terms if t.term == "sold › Trend")
    assert trend.estimate == pytest.approx(5_000, rel=0.25)
    months = homes.regression("price", ["area"], terms=["month:sold"])
    assert any(t.term.startswith("sold › Month = ") for t in months.terms)
    assert "Each year later in `sold` adds" in trend.sentence
    curved = homes.regression("price", ["area", "sold"], terms=["spline:sold"])
    names = [t.term for t in curved.terms]
    assert "sold › Trend" in names and "sold › Trend · curve 1" in names  # a date's curve
    assert "sold" not in {item.term for item in curved.left_out}
    with pytest.raises(RegressionError):
        homes.regression("price", ["area"], terms=["trend:city"])


def test_reset_proposes_a_curve(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 10, 600)
    z = rng.normal(size=600)
    frame = pd.DataFrame({"y": x + 0.5 * x**2 + z + rng.normal(size=600), "x": x, "z": z})
    ds = dx.from_pandas(frame, "bent", cache_dir=tmp_path)
    found = ds.regression("y", ["x", "z"])
    check = next(c for c in found.checks if c.key == "specification")
    assert check.status == "warn" and check.action == "term"
    assert check.term == "spline:x" and check.column == "x"
    assert "Try a curve" in check.sentence
    curved = ds.regression("y", ["x", "z"], terms=["spline:x"])
    check = next(c for c in curved.checks if c.key == "specification")
    assert check.status == "ok"


def test_partial_plot_slope_is_the_coefficient(homes):
    found = homes.regression("price", ["area", "age", "rooms"])
    plot = homes.regression_partial("area", "price", ["area", "age", "rooms"])
    assert plot.available
    area = next(t for t in found.terms if t.term == "area")
    assert plot.slope == pytest.approx(area.estimate, rel=1e-8)  # Frisch–Waugh–Lovell
    assert plot.points == N and len(plot.x_resid) == N
    curved = homes.regression_partial("age", "price", ["area", "age"], terms=["spline:age"])
    assert curved.available and len(curved.line_x) == tm.CURVE_POINTS
    assert "curve included" in curved.sentence and "straight part" in curved.sentence
    city = homes.regression_partial("city", "price", ["area", "city"])
    assert not city.available


@pytest.mark.parametrize("term", ["spline:area", "interaction:area|age", "interaction:area|city"])
def test_partial_plot_slope_stays_the_coefficient_with_curves_and_interactions(homes, term):
    # fix3 (kc_house): with `sqft_living × waterfront` the plot said 213 for a coefficient of 199,
    # its curve and its interactions were not taken out with the other columns
    found = homes.regression("price", ["area", "age", "city"], terms=[term])
    plot = homes.regression_partial("area", "price", ["area", "age", "city"], terms=[term])
    area = next(t for t in found.terms if t.term == "area")
    assert plot.slope == pytest.approx(area.estimate, rel=1e-8)


def test_the_intercept_says_where_a_trend_starts(homes):
    # fix3 (kc_house, `yr_built` read as a date): "with every column at 0" was read at 1900
    found = homes.regression("price", ["area"], terms=["trend:sold"])
    intercept = found.terms[0]
    assert "but `sold › Trend` at 2020" in intercept.sentence
    frame = _frame(homes)
    sold = pd.to_datetime(frame["sold"])
    years = sold.dt.year + (sold.dt.day_of_year - 1 + sold.dt.hour / 24) / np.where(
        sold.dt.is_leap_year, 366, 365
    )
    x = sm.add_constant(np.column_stack([frame["area"], years.to_numpy(dtype=float) - 2020]))
    expected = sm.OLS(frame["price"].to_numpy(), x).fit()
    assert intercept.estimate == pytest.approx(expected.params[0], rel=1e-6)


def test_a_falling_elasticity_says_lowers(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.lognormal(3, 0.5, 400)
    frame = pd.DataFrame({"y": np.exp(5 - 0.4 * np.log(x) + rng.normal(0, 0.1, 400)), "x": x})
    ds = dx.from_pandas(frame, "falling", cache_dir=tmp_path)
    found = ds.regression("y", ["x"], terms=["log:x"], family="log_ols")
    term = next(t for t in found.terms if t.term == "log(x)")
    assert term.sentence.startswith("A 1 % rise in `x` lowers `y` by about 0.3")
