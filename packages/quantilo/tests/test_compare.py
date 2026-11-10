from datetime import datetime, timedelta
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pytest
from scipy import stats

from quantilo.compare.analyze import (
    CompareColumn,
    NotComparableError,
    compare,
    keep_slots,
    plan_layout,
)
from quantilo.compare.stats import GroupStats, anovas
from quantilo.typing.types import Parser

RAW = Parser(kind="raw")
N = 3_000


@pytest.fixture(scope="module")
def connection() -> duckdb.DuckDBPyConnection:
    rng = np.random.default_rng(3)
    groups = rng.choice(["a", "b", "c"], size=N, p=[0.5, 0.3, 0.2])
    shift = {"a": 0.0, "b": 1.0, "c": 3.0}
    spread = {"a": 1.0, "b": 1.0, "c": 4.0}
    x = rng.normal(0, 1, N)
    table = pa.table(
        {
            "file_row_number": pa.array(range(N)),
            "x": x,
            "y": 2 * x + rng.normal(0, 1, N),
            "z": rng.normal(0, 1, N),
            "g": groups,
            "two": np.where(groups == "a", "a", "other"),
            "v": [rng.normal(shift[g], spread[g]) for g in groups],
            "h": rng.choice(["u", "w"], size=N),
            "flag": rng.random(N) < 0.3,
            "when": [datetime(2024, 1, 1) + timedelta(days=int(i % 90)) for i in range(N)],
            "level": rng.choice([3, 1, 2], size=N).astype(float),  # a whole number, 3 values
            "seven": rng.integers(0, 7, size=N).astype(str),  # a category of 7 values
        }
    )
    con = duckdb.connect()
    con.register("t", table)
    con.execute("CREATE TABLE data AS SELECT * FROM t")
    return con


def col(name: str, kind: str) -> CompareColumn:
    return CompareColumn(name, kind, RAW)  # type: ignore[arg-type]


NUM, CAT, BOOL, DATE = "numeric_continuous", "categorical_nominal", "boolean", "datetime"


def frame(con: duckdb.DuckDBPyConnection, sql: str) -> dict[str, np.ndarray]:
    rows = con.execute(sql).fetchnumpy()
    return {key: np.asarray(value) for key, value in rows.items()}


def test_numeric_pair_matches_scipy(connection: duckdb.DuckDBPyConnection) -> None:
    result = compare(connection, "data", col("x", NUM), col("y", NUM))
    data = frame(connection, "SELECT x, y FROM data")

    pearson = next(t for t in result.tests if t.key == "pearson")
    expected: Any = stats.pearsonr(data["x"], data["y"])
    assert result.kind == "scatter" and result.rows == N
    assert pearson.statistic == pytest.approx(expected.statistic, rel=1e-9)
    assert pearson.p_value == pytest.approx(expected.pvalue, abs=1e-12)
    assert pearson.effect is not None and pearson.effect.magnitude == "large"
    slope: Any = stats.linregress(data["x"], data["y"])
    assert result.regression is not None
    assert result.regression.slope == pytest.approx(slope.slope, rel=1e-9)
    assert result.points is not None and len(result.points.x) == N
    assert result.density is not None and sum(map(sum, result.density.counts)) == N
    spearman = next(t for t in result.tests if t.key == "spearman")
    rho: Any = stats.spearmanr(data["x"], data["y"])
    assert spearman.statistic == pytest.approx(rho.statistic)


def test_two_groups_welch_and_cohen_d(connection: duckdb.DuckDBPyConnection) -> None:
    result = compare(connection, "data", col("two", CAT), col("v", NUM))
    data = frame(connection, "SELECT two, v FROM data")
    a, b = data["v"][data["two"] == "a"], data["v"][data["two"] == "other"]

    welch = next(t for t in result.tests if t.key == "welch_t")
    expected: Any = stats.ttest_ind(a, b, equal_var=False)
    assert result.kind == "groups" and result.group_axis == "x"
    assert welch.statistic == pytest.approx(expected.statistic, rel=1e-9)
    assert welch.p_value == pytest.approx(expected.pvalue, rel=1e-6, abs=1e-300)
    assert welch.recommended and welch.effect is not None and welch.effect.name == "d"
    assert [g.label for g in result.groups] == ["a", "other"]


def test_several_groups_anova_and_levene(connection: duckdb.DuckDBPyConnection) -> None:
    result = compare(connection, "data", col("v", NUM), col("g", CAT))
    data = frame(connection, "SELECT g, v FROM data")
    samples = [data["v"][data["g"] == label] for label in ("a", "b", "c")]

    by_key = {t.key: t for t in result.tests}
    f_test: Any = stats.f_oneway(*samples)
    kruskal: Any = stats.kruskal(*samples)
    assert by_key["anova"].statistic == pytest.approx(f_test.statistic, rel=1e-9)
    assert by_key["levene"].p_value is not None and by_key["levene"].p_value < 0.05
    assert by_key["welch_anova"].recommended and not by_key["anova"].recommended
    assert by_key["kruskal"].statistic == pytest.approx(kruskal.statistic, rel=1e-9)
    assert result.group_axis == "y"
    c = next(g for g in result.groups if g.label == "c")
    assert c.ci_low is not None and c.ci_low < (c.mean or 0) < (c.ci_high or 0)


def test_welch_anova_matches_reference() -> None:
    # Classic ANOVA must equal scipy exactly; Welch's ANOVA has no scipy twin, so it is checked
    # against Alexander-Govern, another heteroscedastic test of the same null hypothesis.
    groups = [GroupStats(10, 5.0, 1.0), GroupStats(12, 6.0, 4.0), GroupStats(8, 8.0, 9.0)]
    rng = np.random.default_rng(0)
    samples = [rng.normal(g.mean, np.sqrt(g.var), g.n) for g in groups]
    exact = [GroupStats(len(s), float(s.mean()), float(s.var(ddof=1))) for s in samples]

    classic, welch = anovas(exact)

    f_test: Any = stats.f_oneway(*samples)
    welch_reference: Any = stats.alexandergovern(*samples)
    assert classic.statistic == pytest.approx(
        f_test.statistic, rel=1e-9
    )  # same null; F from Welch's formula
    assert welch.p_value is not None and abs(welch.p_value - welch_reference.pvalue) < 0.05


def test_categories_chi2_and_cramers_v(connection: duckdb.DuckDBPyConnection) -> None:
    result = compare(connection, "data", col("g", CAT), col("flag", BOOL))
    data = frame(connection, "SELECT g, flag FROM data")

    observed = np.array(
        [
            [np.sum((data["g"] == g) & (data["flag"] == f)) for g in ("a", "b", "c")]
            for f in (False, True)
        ]
    )
    expected: Any = stats.chi2_contingency(observed, correction=False)
    chi2 = next(t for t in result.tests if t.key == "chi2")
    assert result.kind == "contingency" and result.contingency is not None
    assert sum(map(sum, result.contingency.counts)) == N
    assert chi2.statistic == pytest.approx(expected.statistic, rel=1e-9)
    assert chi2.effect is not None and chi2.effect.name == "V"


def test_two_by_two_offers_fisher(connection: duckdb.DuckDBPyConnection) -> None:
    result = compare(connection, "data", col("h", CAT), col("flag", BOOL))

    assert {t.key for t in result.tests} == {"chi2", "fisher"}


def test_date_and_number_timeline(connection: duckdb.DuckDBPyConnection) -> None:
    result = compare(connection, "data", col("when", DATE), col("v", NUM), color=col("two", CAT))

    assert result.kind == "timeline" and result.step == "day"
    assert {s.label for s in result.series} == {"a", "other"}
    assert result.tests[0].key == "mann_kendall"
    assert all(s.rolling_mean for s in result.series)


def test_date_and_category_counts(connection: duckdb.DuckDBPyConnection) -> None:
    result = compare(connection, "data", col("g", CAT), col("when", DATE))

    assert result.kind == "date_counts"
    assert sum(sum(v or 0 for v in s.values) for s in result.series) == N


def test_three_numbers_give_3d_points(connection: duckdb.DuckDBPyConnection) -> None:
    result = compare(connection, "data", col("x", NUM), col("y", NUM), col("z", NUM), col("g", CAT))

    assert result.points is not None and result.points.z is not None and result.points.color
    assert set(result.points.color) == {"a", "b", "c"}


def test_points_are_sampled_beyond_the_limit(connection: duckdb.DuckDBPyConnection) -> None:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE big AS SELECT range AS file_row_number, random() AS x, random() AS y "
        "FROM range(30000)"
    )

    result = compare(con, "big", col("x", NUM), col("y", NUM))

    assert result.sample == 20_000 and result.rows == 30_000


@pytest.mark.parametrize(
    ("x", "y", "z", "color", "message"),
    [
        (col("x", NUM), col("x", NUM), None, None, "two different"),
        (col("x", NUM), col("y", NUM), col("g", CAT), None, "three numeric"),
        (col("x", NUM), col("y", NUM), None, col("z", NUM), "Color needs a category"),
        (col("x", NUM), col("y", "text"), None, None, "can't be compared"),
        (col("when", DATE), col("when2", DATE), None, None, "Two dates"),
    ],
)
def test_invalid_choices(connection, x, y, z, color, message) -> None:
    with pytest.raises(NotComparableError, match=message):
        compare(connection, "data", x, y, z, color)


DISCRETE = "numeric_discrete"


def levels(name: str) -> CompareColumn:
    return CompareColumn(name, DISCRETE, RAW, as_category=True)  # type: ignore[arg-type]


def test_a_whole_number_read_as_categories_gives_ordered_groups(connection) -> None:
    result = compare(connection, "data", levels("level"), col("v", NUM))
    assert result.kind == "groups" and [g.label for g in result.groups] == ["1", "2", "3"]
    assert result.roles["level"] == "cat" and result.tests
    as_numbers = compare(connection, "data", col("level", DISCRETE), col("v", NUM))
    assert as_numbers.kind == "scatter"
    crossed = compare(connection, "data", levels("level"), col("h", CAT))
    assert crossed.kind == "contingency" and crossed.contingency is not None
    assert crossed.contingency.columns == ["1", "2", "3"]


def test_a_colour_splits_each_group_into_boxes(connection) -> None:
    result = compare(connection, "data", col("g", CAT), col("v", NUM), color=col("h", CAT))
    assert result.kind == "groups" and set(result.color_levels) == {"u", "w"}
    pairs = [(g.label, g.color) for g in result.groups]
    assert set(pairs) == {(a, b) for a in ("a", "b", "c") for b in ("u", "w")}
    total = sum(g.n for g in result.groups)
    assert total == result.rows == N
    plain = compare(connection, "data", col("g", CAT), col("v", NUM))
    # the tests still compare the groups, all colours together
    assert [t.p_value for t in result.tests] == pytest.approx(
        [t.p_value for t in plain.tests], rel=1e-6, abs=1e-12
    )
    assert result.message and "all h values together" in result.message
    assert result.layout is not None and not result.layout.swapped
    numeric_colour = compare(
        connection, "data", col("g", CAT), col("v", NUM), color=levels("level")
    )
    assert numeric_colour.color_levels == ["1", "2", "3"]


def test_the_layout_puts_the_category_with_more_values_on_the_axis_within_15_boxes() -> None:
    assert plan_layout(2, 3) == (False, 2, 3)  # Survived x Embarked: 6 boxes, no swap
    assert plan_layout(2, 7) == (True, 7, 2)  # a colour of 7 goes on the axis: 14 boxes
    assert plan_layout(7, 10) == (True, 3, 5)  # 10 on the axis, 5 colours: 3 x 5 = 15 boxes
    assert plan_layout(20, 0) == (False, 15, 1)  # no colour: 15 groups
    assert keep_slots(["a", "b", "c"], 3) == ["a", "b", "c"]
    assert keep_slots(["a", "b", "c", "d"], 3) == ["a", "b"]  # + "Others"


def test_a_colour_with_many_values_goes_on_the_axis_whatever_the_places(connection) -> None:
    result = compare(connection, "data", col("h", CAT), col("v", NUM), color=col("seven", CAT))
    assert result.layout is not None and result.layout.swapped
    assert (result.layout.axis, result.layout.color) == ("seven", "h")
    assert len(result.groups) == 7 * 2 and set(result.color_levels) == {"u", "w"}
    assert {g.label for g in result.groups} == {str(k) for k in range(7)}
    swapped_places = compare(
        connection, "data", col("v", NUM), col("h", CAT), color=col("seven", CAT)
    )
    assert swapped_places.layout is not None and swapped_places.layout.axis == "seven"
    assert swapped_places.group_axis == "y"
    scatter = compare(connection, "data", col("x", NUM), col("y", NUM), color=col("seven", CAT))
    assert scatter.points is not None and scatter.points.color is not None


def test_two_large_categories_are_cut_to_15_boxes(connection) -> None:
    result = compare(connection, "data", col("g", CAT), col("v", NUM), color=col("seven", CAT))
    # seven (7 values) goes on the axis, g (3) in colour: 7 x 3 = 21 > 15, so 5 x 3 with Others
    assert result.layout is not None and result.layout.axis == "seven"
    assert len(result.groups) <= 15 and result.layout.axis_kept == 5
    assert "Others" in {g.label for g in result.groups}
    assert result.message and "at most 15" in result.message


# --- 051: bootstrap intervals and pairwise differences -------------------------------------------


def small_groups() -> duckdb.DuckDBPyConnection:
    rng = np.random.default_rng(11)
    labels = ["big"] * 600 + ["small"] * 12 + ["skewed"] * 400
    values = np.concatenate(
        [rng.normal(10, 2, 600), rng.normal(12, 2, 12), rng.lognormal(2, 1.2, 400)]
    )
    table = pa.table({"file_row_number": pa.array(range(len(labels))), "g": labels, "v": values})
    con = duckdb.connect()
    con.register("t", table)
    con.execute("CREATE TABLE data AS SELECT * FROM t")
    return con


def test_small_or_skewed_groups_get_a_bootstrap_interval_with_its_reason() -> None:
    con = small_groups()
    result = compare(con, "data", col("g", CAT), col("v", NUM))
    by_label = {g.label: g for g in result.groups}
    small, skewed, big = by_label["small"], by_label["skewed"], by_label["big"]
    assert small.ci_method == "bootstrap" and small.ci_reason and "12 values" in small.ci_reason
    assert skewed.ci_method == "bootstrap" and "skewness" in (skewed.ci_reason or "")
    assert big.ci_method == "t"
    for group in result.groups:
        assert group.ci_low is not None and group.ci_high is not None
        assert group.ci_low < (group.mean or 0) < group.ci_high
        assert group.median_low is not None and group.median_high is not None
        assert group.median_low <= (group.median or 0) <= group.median_high
        assert group.median_reason
    assert result.intervals and "bootstrap for 2 of 3 groups" in result.intervals
    again = compare(con, "data", col("g", CAT), col("v", NUM))
    assert [(g.ci_low, g.median_high) for g in again.groups] == [
        (g.ci_low, g.median_high) for g in result.groups
    ]


def test_pairwise_differences_are_welch_tests_adjusted_by_holm(connection) -> None:
    result = compare(connection, "data", col("v", NUM), col("g", CAT))
    data = frame(connection, "SELECT g, v FROM data")
    assert len(result.differences) == 3
    assert result.multiple is not None and result.multiple.method == "holm"
    for d in result.differences:
        a, b = data["v"][data["g"] == d.a], data["v"][data["g"] == d.b]
        expected: Any = stats.ttest_ind(a, b, equal_var=False)
        assert d.p_value == pytest.approx(expected.pvalue, rel=1e-6, abs=1e-300)
        assert d.diff == pytest.approx(b.mean() - a.mean())
        assert d.p_adjusted is not None and d.p_adjusted >= (d.p_value or 0)
        assert d.interval == "welch" and d.low is not None and d.low < d.diff < (d.high or 0)


def test_a_difference_with_a_small_group_is_bootstrapped() -> None:
    result = compare(small_groups(), "data", col("g", CAT), col("v", NUM))
    pair = next(d for d in result.differences if {d.a, d.b} == {"big", "small"})
    assert pair.interval == "bootstrap" and pair.reason and "small" in pair.reason
    assert pair.low is not None and pair.high is not None and pair.low < pair.diff < pair.high


def test_many_groups_use_benjamini_hochberg() -> None:
    rng = np.random.default_rng(12)
    labels = rng.choice([f"g{k}" for k in range(6)], size=3_000)
    values = rng.normal(0, 1, 3_000) + np.array([int(label[1]) * 0.1 for label in labels])
    table = pa.table({"file_row_number": pa.array(range(3_000)), "g": labels, "v": values})
    con = duckdb.connect()
    con.register("t", table)
    con.execute("CREATE TABLE data AS SELECT * FROM t")
    result = compare(con, "data", col("g", CAT), col("v", NUM))
    assert len(result.differences) == 15
    assert result.multiple is not None and result.multiple.method == "bh"
    assert "15 tests" in result.multiple.sentence


def test_colored_groups_compare_the_axis_groups_pairwise(connection) -> None:
    result = compare(connection, "data", col("g", CAT), col("v", NUM), color=col("h", CAT))
    assert {(d.a, d.b) for d in result.differences} == {("a", "b"), ("a", "c"), ("b", "c")}
    assert all(g.ci_method in ("t", "bootstrap") for g in result.groups)


# --- LOWESS curve on the scatter (064) ----------------------------------------------------------


def _pairs(n: int, bend: bool) -> duckdb.DuckDBPyConnection:
    rng = np.random.default_rng(9)
    x = rng.uniform(-3, 3, n)
    y = (x**2 if bend else 2 * x) + rng.normal(0, 1, n)
    con = duckdb.connect()
    con.register("t", pa.table({"file_row_number": pa.array(range(n)), "x": x, "y": y}))
    con.execute("CREATE TABLE data AS SELECT * FROM t")
    return con


def test_lowess_is_on_for_a_bent_relation_and_off_for_a_line() -> None:
    bent = compare(_pairs(400, True), "data", col("x", NUM), col("y", NUM)).lowess
    straight = compare(_pairs(400, False), "data", col("x", NUM), col("y", NUM)).lowess

    assert bent is not None and straight is not None
    assert bent.suggested and not straight.suggested
    assert bent.r2 - bent.line_r2 >= 0.05
    assert len(bent.x) == len(bent.y) == 100 and bent.x == sorted(bent.x)
    assert "bends" in bent.reason and "line" in straight.reason
    assert min(bent.y) < 1.5  # the curve goes down to the bottom of the parabola


def test_no_lowess_under_30_points() -> None:
    result = compare(_pairs(20, True), "data", col("x", NUM), col("y", NUM))
    assert result.lowess is None
