import duckdb
import numpy as np
import pyarrow as pa
import pytest
from scipy import stats

from quantilo.typing.types import AnalysisType, Parser
from quantilo.variable import profiles
from quantilo.variable.analyze import analyze
from quantilo.variable.fitting import fit
from quantilo.variable.numeric import describe, ecdf, normality, qq
from quantilo.variable.outliers import outliers
from quantilo.variable.source import VariableInput, VariableSource

RAW = Parser(kind="raw")


def source_of(
    values, analysis_type: AnalysisType = "numeric_continuous", name: str = "v"
) -> VariableSource:
    connection = duckdb.connect()
    table = pa.table({"file_row_number": pa.array(range(len(values))), name: pa.array(values)})
    connection.register("t", table)
    return VariableSource(connection, "t", VariableInput(name, analysis_type, RAW))


def draws(kind: str, size: int = 5_000) -> list[float]:
    rng = np.random.default_rng(7)
    generators = {
        "normal": lambda: rng.normal(10, 2, size),
        "lognormal": lambda: rng.lognormal(0, 0.8, size),
        "gamma": lambda: rng.gamma(2.0, 3.0, size),
        "poisson": lambda: rng.poisson(4.0, size).astype(float),
    }
    return [float(v) for v in generators[kind]()]


# --- Descriptive statistics ---------------------------------------------------------------------


def test_statistics_match_numpy_and_scipy() -> None:
    values = draws("gamma", 2_000)
    x = np.array(values)

    summary = describe(source_of([*values, None, float("inf")]))

    assert (summary.rows, summary.count, summary.missing, summary.infinite) == (2_002, 2_000, 1, 1)
    assert summary.mean == pytest.approx(x.mean(), rel=1e-9)
    assert summary.std == pytest.approx(x.std(ddof=1), rel=1e-9)
    assert summary.median == pytest.approx(np.median(x), rel=1e-9)
    assert summary.p5 == pytest.approx(np.quantile(x, 0.05), rel=1e-9)
    assert summary.p99 == pytest.approx(np.quantile(x, 0.99), rel=1e-9)
    assert summary.iqr == pytest.approx(np.quantile(x, 0.75) - np.quantile(x, 0.25), rel=1e-9)
    assert summary.mad == pytest.approx(np.median(np.abs(x - np.median(x))), rel=1e-9)
    assert summary.skewness == pytest.approx(stats.skew(x, bias=False), rel=1e-6)
    assert summary.kurtosis == pytest.approx(stats.kurtosis(x, bias=False), rel=1e-6)
    assert summary.sum == pytest.approx(x.sum(), rel=1e-9)
    assert summary.cv == pytest.approx(x.std(ddof=1) / x.mean(), rel=1e-9)


def test_zeros_and_negatives_are_counted() -> None:
    summary = describe(source_of([-2.0, -1.0, 0.0, 0.0, 3.0]))

    assert (summary.zeros, summary.negatives, summary.distinct) == (2, 2, 4)


# --- Normality ----------------------------------------------------------------------------------


def test_normal_draws_are_not_rejected() -> None:
    source = source_of(draws("normal"))

    result = normality(source, describe(source))

    assert [test.key for test in result.tests] == [
        "shapiro",
        "dagostino",
        "anderson",
        "jarque_bera",
        "lilliefors",
    ]
    assert all(test.p_value is not None and test.p_value > 0.01 for test in result.tests)


def test_lognormal_draws_are_rejected_by_every_test() -> None:
    source = source_of(draws("lognormal"))

    result = normality(source, describe(source))

    assert all(test.p_value is not None and test.p_value < 0.05 for test in result.tests)


def test_tests_run_on_a_sample_of_at_most_5000_values() -> None:
    source = source_of(draws("normal", 12_000))

    result = normality(source, describe(source))

    assert (result.n, result.sample) == (12_000, 5_000)
    assert all(test.n == 5_000 for test in result.tests)


def test_normality_needs_enough_varying_values() -> None:
    few = source_of([1.0, 2.0, 3.0])
    constant = source_of([5.0] * 50)

    assert normality(few, describe(few)).reason == "Needs at least 8 values."
    assert normality(constant, describe(constant)).reason == "The variable is constant."


def test_sample_is_reproducible() -> None:
    source = source_of(draws("normal", 9_000))

    assert np.array_equal(source.sample(1_000), source.sample(1_000))


# --- Curves -------------------------------------------------------------------------------------


def test_ecdf_and_qq() -> None:
    source = source_of(draws("normal"))
    summary = describe(source)

    curve = ecdf(source, summary)
    plot = qq(source, summary)

    assert curve is not None and curve.y[0] == 0 and curve.y[-1] == 1
    assert curve.x == sorted(curve.x)
    assert plot is not None and len(plot.theoretical) == len(plot.sample) == 200
    assert plot.slope == pytest.approx(2, rel=0.1)  # the std of the draws
    assert plot.intercept == pytest.approx(10, rel=0.05)  # their mean


# --- Fitted laws and transformations ------------------------------------------------------------


def test_gamma_draws_rank_gamma_in_the_top_two() -> None:
    result = fit(source_of(draws("gamma")), discrete=False)

    assert [law.key for law in result.laws[:2]].count("gamma") == 1
    gamma = next(law for law in result.laws if law.key == "gamma")
    assert gamma.params["a"] == pytest.approx(2.0, rel=0.15)
    assert len(gamma.density) == len(result.grid) == 200
    assert result.laws == sorted(result.laws, key=lambda law: law.aic)


def test_laws_need_a_matching_support() -> None:
    values = [v - 15 for v in draws("normal")]  # mostly negative

    keys = {law.key for law in fit(source_of(values), discrete=False).laws}

    assert {"lognormal", "gamma", "weibull", "exponential", "pareto", "beta"}.isdisjoint(keys)
    assert {"normal", "student_t", "logistic", "uniform"} <= keys


def test_counts_try_discrete_laws() -> None:
    result = fit(source_of(draws("poisson")), discrete=True)

    assert result.laws[0].key in ("poisson", "negative_binomial")
    assert result.laws[0].discrete


def test_transformations_reduce_skewness() -> None:
    result = fit(source_of(draws("lognormal")), discrete=False)

    by_key = {t.key: t for t in result.transforms}
    assert set(by_key) == {"log", "sqrt", "boxcox", "yeojohnson"}
    log = by_key["log"]
    assert log.skewness_before is not None and log.skewness_after is not None
    assert abs(log.skewness_after) < abs(log.skewness_before) / 5
    assert by_key["boxcox"].lmbda == pytest.approx(0, abs=0.15)


# --- Outliers -----------------------------------------------------------------------------------


def test_outlier_rules_and_rows() -> None:
    values = [float(v) for v in range(100)] + [1_000.0, -500.0]
    source = source_of(values)

    result = outliers(source, describe(source), "iqr")

    rules = {rule.method: rule for rule in result.rules}
    assert rules["iqr"].count == 2 and rules["iqr3"].count == 2
    assert rules["iqr"].pct == pytest.approx(200 / 102)
    assert [(row.row, row.value) for row in result.rows] == [(101, 1_000.0), (102, -500.0)]


# --- Profiles -----------------------------------------------------------------------------------


def test_categorical_profile() -> None:
    values = ["a"] * 90 + ["b"] * 9 + ["c"] + [None]
    profile = profiles.categorical(source_of(values, "categorical_nominal"))

    assert (profile.rows, profile.missing, profile.distinct) == (101, 1, 3)
    shares = np.array([0.9, 0.09, 0.01])
    assert profile.entropy == pytest.approx(-(shares * np.log2(shares)).sum())
    assert profile.herfindahl == pytest.approx((shares**2).sum())
    assert profile.imbalanced and profile.top == {"label": "a", "count": 90}
    assert profile.rare_count == 0  # "c" is exactly 1%
    assert profile.chi2_p is not None and profile.chi2_p < 0.001


def test_date_profile_finds_gaps() -> None:
    from datetime import datetime, timedelta

    start = datetime(2024, 1, 1)
    days = [start + timedelta(days=d) for d in range(60) if not 10 <= d < 13]

    profile = profiles.dates(source_of(days, "datetime"), year_only=False)

    assert profile.step == "day" and profile.granularity == "daily"
    assert (profile.periods, profile.gap_count) == (60, 3)
    assert profile.gaps == ["2024-01-11", "2024-01-12", "2024-01-13"]
    assert len(profile.calendar) == 57


def test_text_profile_counts_words() -> None:
    values = ["The red car", "a red bike", "Red, red wine!", None]

    profile = profiles.text(source_of(values, "text"))

    assert profile.words[0] == {"label": "red", "count": 4}
    assert (profile.min_length, profile.max_length) == (10, 14)


def test_identifier_profile() -> None:
    values = ["AB-001", "AB-002", "AB-002", "XY-1", None]

    profile = profiles.identifier(source_of(values, "identifier"))

    assert profile.duplicated_values == 1
    assert profile.duplicates == [{"label": "AB-002", "count": 2}]
    assert profile.patterns[0] == {"label": "AA-999", "count": 3}
    assert profile.unique_pct == pytest.approx(75)


def test_analyze_fills_the_section_of_the_type() -> None:
    numeric = analyze(source_of(draws("normal", 500)))
    discrete = analyze(source_of([1.0, 2.0, 2.0, 3.0] * 10, "numeric_discrete"))

    assert numeric.numeric is not None and numeric.categorical is None
    assert len(numeric.numeric.sample) == 500
    assert discrete.numeric is not None and discrete.categorical is not None
