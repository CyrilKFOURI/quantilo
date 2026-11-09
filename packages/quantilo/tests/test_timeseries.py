from datetime import datetime, timedelta

import duckdb
import numpy as np
import pyarrow as pa
import pytest
from scipy import stats

from quantilo.compare.analyze import CompareColumn, compare
from quantilo.timeseries import series as ts
from quantilo.timeseries.trend import mann_kendall, trend
from quantilo.typing.types import Parser

RAW = Parser(kind="raw")


def table(rows: list[tuple[datetime, float, str]]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register(
        "t",
        pa.table(
            {
                "file_row_number": pa.array(range(len(rows))),
                "when": [r[0] for r in rows],
                "v": [r[1] for r in rows],
                "g": [r[2] for r in rows],
            }
        ),
    )
    con.execute("CREATE TABLE data AS SELECT * FROM t")
    return con


def when() -> CompareColumn:
    return CompareColumn("when", "datetime", RAW)


def value(role: str | None = None) -> CompareColumn:
    return CompareColumn("v", "numeric_continuous", RAW, semantic_role=role)


def daily(days: int, per_day: int = 1, start: datetime = datetime(2024, 1, 1)) -> list:
    rng = np.random.default_rng(5)
    return [
        (start + timedelta(days=d), float(10 + d * 0.5 + rng.normal(0, 1)), "ab"[k % 2])
        for d in range(days)
        for k in range(per_day)
    ]


# --- aggregation ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("aggregate", "expected"),
    [("sum", 60.0), ("mean", 20.0), ("median", 20.0), ("min", 10.0), ("max", 30.0), ("count", 3.0)],
)
def test_rows_of_a_period_are_aggregated(aggregate: str, expected: float) -> None:
    rows = [(datetime(2024, 1, 1), v, "a") for v in (10.0, 20.0, 30.0)] + daily(20)[1:]
    con = table(rows)

    result = compare(con, "data", when(), value(), options=ts.Options("day", aggregate))  # type: ignore[arg-type]

    assert result.series[0].periods[0] == "2024-01-01"
    assert result.series[0].values[0] == pytest.approx(expected)
    assert result.series[0].counts[0] == 3


def test_default_aggregate_follows_the_role() -> None:
    con = table(daily(30, per_day=2))

    summed = compare(con, "data", when(), value("amount"))
    averaged = compare(con, "data", when(), value("age"))

    assert summed.time is not None and summed.time.aggregate == "sum" and summed.time.aggregate_auto
    assert averaged.time is not None and averaged.time.aggregate == "mean"


def test_empty_periods_are_gaps_except_for_sums_and_counts() -> None:
    rows = [r for r in daily(30) if r[0].day != 10]  # 2024-01-10 missing
    con = table(rows)

    mean = compare(con, "data", when(), value(), options=ts.Options("day", "mean"))
    total = compare(con, "data", when(), value(), options=ts.Options("day", "sum"))

    index = mean.series[0].periods.index("2024-01-10")
    assert mean.series[0].values[index] is None
    assert total.series[0].values[index] == 0
    assert mean.series[0].counts[index] == 0


# --- period ---------------------------------------------------------------------------------------


def test_auto_period_is_the_typical_step_coarsened_beyond_400_periods() -> None:
    con = table(daily(3 * 365))

    result = compare(con, "data", when(), value())

    assert result.time is not None
    assert (result.time.period, result.time.coarsened, result.time.period_auto) == (
        "week",
        True,
        True,
    )
    assert 150 <= result.time.periods <= 160


def test_typical_period() -> None:
    assert ts.typical_period(86_400, False) == "day"
    assert ts.typical_period(7 * 86_400, False) == "week"
    assert ts.typical_period(31 * 86_400, False) == "month"
    assert ts.typical_period(3_600, False) == "day"  # no time of day: never hourly
    assert ts.typical_period(3_600, True) == "hour"


def test_chosen_period_is_kept_up_to_2000_periods() -> None:
    assert ts.choose_period("day", "day", 1_000 * 86_400) == ("day", False)
    assert ts.choose_period("day", "day", 3_000 * 86_400) == ("week", True)


# --- rolling --------------------------------------------------------------------------------------


def test_rolling_trailing_skips_gaps_and_needs_half_a_window() -> None:
    means, stds = ts.rolling([1.0, 2.0, None, 4.0, 5.0], 4, "trailing")

    # pandas: rolling(4, min_periods=2).mean() / .std()
    assert means == [None, 1.5, 1.5, pytest.approx(7 / 3), pytest.approx(11 / 3)]
    assert stds[0] is None and stds[1] == pytest.approx(np.std([1, 2], ddof=1))
    assert stds[4] == pytest.approx(np.std([2, 4, 5], ddof=1))


def test_rolling_centered() -> None:
    means, _ = ts.rolling([1.0, 2.0, 3.0, 4.0, 5.0], 3, "centered")

    assert means == [1.5, 2.0, 3.0, 4.0, 4.5]


def test_default_window_never_exceeds_a_quarter_of_the_series() -> None:
    assert ts.default_window("day", 400) == 7
    assert ts.default_window("day", 12) == 3
    assert ts.default_window("month", 40) == 3


# --- trend ----------------------------------------------------------------------------------------


def test_mann_kendall_reference_values() -> None:
    s, z, p, tau = mann_kendall(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    assert s == 10 and tau == 1
    assert z == pytest.approx(9 / np.sqrt(5 * 4 * 15 / 18))
    assert p == pytest.approx(0.02749, abs=1e-4)


def test_mann_kendall_matches_kendall_without_ties() -> None:
    rng = np.random.default_rng(1)
    values = np.cumsum(rng.normal(0.1, 1, 200))

    s, _, _, tau = mann_kendall(values)

    expected = float(np.asarray(stats.kendalltau(np.arange(200), values), dtype=float)[0])
    assert tau == pytest.approx(expected) and s == round(expected * 200 * 199 / 2)


def test_mann_kendall_corrects_for_ties() -> None:
    values = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0])

    s, z, _, _ = mann_kendall(values)

    ties = np.array([2, 2, 3])
    variance = (8 * 7 * 21 - (ties * (ties - 1) * (2 * ties + 5)).sum()) / 18
    assert z == pytest.approx((s - 1) / np.sqrt(variance))


def test_sen_slope_and_gaps_keep_their_position() -> None:
    values: list[float | None] = [2.0 * i for i in range(20)]
    values[5] = None

    found = trend(values)

    assert found is not None
    assert found.slope == pytest.approx(2.0) and found.n == 19
    assert found.slope_low <= 2.0 <= found.slope_high


def test_flat_series_has_no_trend() -> None:
    assert trend([3.0] * 10) is None


# --- through Compare ------------------------------------------------------------------------------


def test_timeline_reports_trend_and_rolling_statistics() -> None:
    con = table(daily(120))

    result = compare(con, "data", when(), value())

    assert result.time is not None and result.time.window == 7
    assert len(result.series[0].rolling_mean) == result.time.periods
    test = result.tests[0]
    assert test.key == "mann_kendall" and test.p_value is not None and test.p_value < 0.001
    assert result.trend is not None and result.trend.slope == pytest.approx(0.5, abs=0.05)
    assert result.trend.per_year is not None and result.trend.per_year > 0


def test_groups_get_their_own_series_and_one_combined_test() -> None:
    con = table(daily(60, per_day=2))

    result = compare(
        con, "data", when(), value(), color=CompareColumn("g", "categorical_nominal", RAW)
    )

    assert [s.label for s in result.series] == ["a", "b"]
    assert all(s.rolling_mean for s in result.series)
    assert result.tests[0].note and "all groups combined" in result.tests[0].note


def test_too_few_periods_draw_the_curve_without_test() -> None:
    con = table(daily(5))

    result = compare(con, "data", when(), value(), options=ts.Options("day"))

    assert result.series[0].values and result.tests == []
    assert result.time is not None and result.time.window is None
    assert result.message and "8 needed" in result.message


def test_incomplete_edge_periods_are_left_out_of_the_trend_test() -> None:
    # 2024-01-03 is a Wednesday and 2024-03-28 a Thursday: both weeks are partial.
    rows = [(datetime(2024, 1, 3) + timedelta(days=d), 10.0 + d, "a") for d in range(86)]
    con = table(rows)

    weekly = compare(con, "data", when(), value(), options=ts.Options("week", "sum"))
    daily_view = compare(con, "data", when(), value(), options=ts.Options("day", "sum"))

    assert weekly.time is not None
    assert (weekly.time.first_partial, weekly.time.last_partial) == (True, True)
    assert weekly.tests[0].n == weekly.time.periods - 2
    assert daily_view.time is not None
    assert not daily_view.time.first_partial and not daily_view.time.last_partial


def test_window_and_alignment_options() -> None:
    con = table(daily(60))

    result = compare(
        con, "data", when(), value(), options=ts.Options("day", "mean", 10, "centered")
    )

    assert result.time is not None
    assert (result.time.window, result.time.align) == (10, "centered")
    assert result.series[0].rolling_mean[0] is not None  # a centred window sees ahead
