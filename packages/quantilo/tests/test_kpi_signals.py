"""Signals of a KPI (054): unusual periods and level shifts (P12), quality (P13), mix effect and
Simpson's paradox (P14), the started period at an equal date (P9)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

from quantilo.ingestion.sql import lit
from quantilo.kpi import mix as kpi_mix
from quantilo.kpi.compute import compute
from quantilo.kpi.definition import Definition
from quantilo.typing.types import Parser

DASH = chr(0x2013)


@dataclass
class Col:
    name: str
    type: Any
    role: str | None = None
    order: list[str] | None = None
    parser: Parser = field(default_factory=Parser)


COLUMNS = [
    Col("day", "datetime"),
    Col("value", "numeric_continuous", "amount"),
    Col("score", "numeric_continuous"),
    Col("group", "categorical_nominal"),
    Col("order_id", "categorical_nominal", "order_key"),
    Col("item", "categorical_nominal"),
]


def source(tmp_path: Path, frame: pd.DataFrame, name: str = "rows") -> str:
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path)
    return f"read_parquet({lit(str(path))})"


def run(src: str, columns: list[Col] | None = None, **definition: Any):
    connection = duckdb.connect()
    try:
        return compute(connection, src, columns or COLUMNS, Definition.model_validate(definition))
    finally:
        connection.close()


def kinds(result) -> list[str]:
    return [s.kind for s in result.signals]


def of(result, kind: str):
    return [s for s in result.signals if s.kind == kind]


MONTHLY_SUM = {
    "measures": [{"id": "m1", "agg": "sum", "column": "value"}],
    "time": {"column": "day", "grain": "month"},
}


def daily(start: str, end: str, level, seed: int = 3, noise: float = 0.03) -> pd.DataFrame:
    """One row a day; `level(day)` gives its value, plus a little noise."""
    days = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    values = np.array([level(d) for d in days]) * (1 + rng.normal(0, noise, len(days)))
    return pd.DataFrame(
        {
            "day": days,
            "value": values,
            "score": rng.normal(50, 5, len(days)),
            "group": rng.choice(["A", "B"], len(days)),
            "order_id": [f"o{i}" for i in range(len(days))],
            "item": "x",
        }
    )


def seasonal(day: pd.Timestamp) -> float:
    return 100 + 20 * np.sin(2 * np.pi * day.month / 12)


# --- P12: unusual periods and level shifts ---------------------------------------------------


def test_an_unusual_month_is_flagged_with_its_robust_z(tmp_path: Path) -> None:
    frame = daily("2023-01-01", "2025-12-31", seasonal, noise=0.25)
    frame.loc[(frame.day.dt.year == 2025) & (frame.day.dt.month == 3), "value"] *= 2.5
    result = run(source(tmp_path, frame), **MONTHLY_SUM)
    anomalies = of(result, "anomaly")
    assert [a.cells[0].period for a in anomalies] == ["2025-03-01"]
    signal = anomalies[0]
    assert signal.message.startswith("Mar 2025 unusually high (robust z ")
    assert signal.rule == "R-SIG-01" and signal.scope == "series" and signal.measure == "m1"
    assert signal.score is not None and signal.score >= 3.5


def test_a_partial_period_is_never_flagged(tmp_path: Path) -> None:
    frame = daily("2023-01-01", "2026-01-02", seasonal)  # Jan 2026: two days, a tiny sum
    result = run(source(tmp_path, frame), **MONTHLY_SUM)
    assert result.periods[-1].partial
    assert all(c.period != "2026-01-01" for s in result.signals for c in s.cells)


def test_no_series_signal_under_12_complete_periods(tmp_path: Path) -> None:
    frame = daily("2025-01-01", "2025-10-31", seasonal)
    frame.loc[frame.day.dt.month == 5, "value"] *= 4
    result = run(source(tmp_path, frame), **MONTHLY_SUM)
    assert "anomaly" not in kinds(result) and "level_shift" not in kinds(result)


def test_a_level_shift_says_since_when_and_by_how_much(tmp_path: Path) -> None:
    frame = daily(
        "2021-01-01", "2024-12-31", lambda d: 100 if d < pd.Timestamp("2023-07-01") else 160
    )
    result = run(source(tmp_path, frame), **MONTHLY_SUM)
    shifts = of(result, "level_shift")
    assert [s.cells[0].period for s in shifts] == ["2023-07-01"]
    assert "level shift since Jul 2023 (+" in shifts[0].message
    assert shifts[0].score == pytest.approx(0.6, abs=0.06)


def test_a_steady_series_has_no_series_signal(tmp_path: Path) -> None:
    result = run(source(tmp_path, daily("2021-01-01", "2024-12-31", seasonal)), **MONTHLY_SUM)
    assert "anomaly" not in kinds(result) and "level_shift" not in kinds(result)


# --- P13: quality ----------------------------------------------------------------------------


def test_cells_of_a_mean_on_few_values_are_low_n_but_never_a_sum(tmp_path: Path) -> None:
    frame = daily("2024-01-01", "2024-12-31", lambda d: 100)
    frame["group"] = "A"
    frame.loc[frame.index[::40], "group"] = "B"  # B: about one row a month
    src = source(tmp_path, frame)
    split = {"time": {"column": "day", "grain": "month"}, "groups": [{"column": "group"}]}
    result = run(src, measures=[{"id": "m1", "agg": "mean", "column": "score"}], **split)
    (low,) = of(result, "low_n")
    assert low.rule == "R-SIG-03" and low.scope == "cells" and low.measure == "m1"
    cells = {(c.period, c.g1): c.value for c in low.cells}
    assert 0 < (cells.get(("2024-01-01", "B")) or 0) < 30
    assert ("2024-01-01", None) not in cells  # the month's total rests on 31 days
    assert (None, "B") in cells  # B over the whole year: 10 rows
    summed = run(src, measures=[{"id": "m1", "agg": "sum", "column": "value"}], **split)
    assert "low_n" not in kinds(summed)


def test_a_ratio_rests_on_the_smaller_n_of_its_measures(tmp_path: Path) -> None:
    frame = daily("2024-01-01", "2024-03-31", lambda d: 100)
    frame["order_id"] = [f"o{i // 10}" for i in range(len(frame))]  # 10 lines an order
    result = run(
        source(tmp_path, frame),
        measures=[
            {"id": "m1", "agg": "sum", "column": "value"},
            {"id": "m2", "agg": "distinct", "column": "order_id"},
        ],
        ratio={"numerator": "m1", "denominator": "m2"},
        time={"column": "day", "grain": "month"},
    )
    (low,) = of(result, "low_n")
    assert low.measure == "ratio"
    assert {c.period: c.value for c in low.cells}["2024-01-01"] == pytest.approx(4, abs=1)


def test_a_thin_period_is_flagged(tmp_path: Path) -> None:
    frame = daily("2024-01-01", "2024-12-31", lambda d: 100)
    frame = pd.DataFrame(frame[(frame.day.dt.month != 6) | (frame.day.dt.day <= 3)])  # June: 3 rows
    result = run(source(tmp_path, frame), **MONTHLY_SUM)
    (thin,) = of(result, "thin_period")
    assert [c.period for c in thin.cells] == ["2024-06-01"]
    assert thin.message.startswith("Jun 2024 has under 20 % of the median period's rows")


@pytest.mark.parametrize(("share", "flagged"), [(0.3, True), (0.1, False)])
def test_a_measure_empty_on_more_than_a_fifth_of_the_rows(
    tmp_path: Path, share: float, flagged: bool
) -> None:
    frame = daily("2024-01-01", "2024-12-31", lambda d: 100)
    frame.loc[frame.sample(frac=share, random_state=1).index, "score"] = np.nan
    result = run(
        source(tmp_path, frame),
        measures=[{"id": "m1", "agg": "mean", "column": "score"}],
        time={"column": "day", "grain": "month"},
    )
    empty = of(result, "empty_measure")
    assert bool(empty) is flagged
    if flagged:
        assert empty[0].message.startswith("Average score has no value on 30 % of the rows")
        assert empty[0].scope == "measure"


@pytest.mark.parametrize(("copies", "flagged"), [(20, True), (3, False)])
def test_rows_repeated_exactly_inside_an_order_inflate_a_sum(
    tmp_path: Path, copies: int, flagged: bool
) -> None:
    frame = daily("2022-01-01", "2024-12-31", lambda d: 100).head(1000)
    frame["order_id"] = [f"o{i // 4}" for i in range(len(frame))]  # four lines an order: normal
    frame["item"] = [f"i{i % 4}" for i in range(len(frame))]
    frame = pd.concat([frame, frame.head(copies)], ignore_index=True)
    src = source(tmp_path, frame)
    result = run(src, **MONTHLY_SUM)
    found = of(result, "duplicates")
    assert bool(found) is flagged
    if flagged:
        assert found[0].message.startswith(f"{copies} rows (2.0 %) are exact copies")
        assert "of the same order_id" in found[0].message
    averaged = run(
        src, measures=[{"id": "m1", "agg": "mean", "column": "score"}], time=MONTHLY_SUM["time"]
    )
    assert "duplicates" not in kinds(averaged)


# --- P14: mix and Simpson --------------------------------------------------------------------


def two_months(rows: dict[tuple[str, str], tuple[int, float]]) -> pd.DataFrame:
    """(month, group) -> (rows, score of every row); every day of both months has rows."""
    parts = []
    for (month, group), (count, score) in rows.items():
        days = pd.date_range(f"2024-{month}-01", periods=28, freq="D")
        parts.append(
            pd.DataFrame(
                {
                    "day": [days[i % 28] for i in range(count)],
                    "value": 1.0,
                    "score": score,
                    "group": group,
                    "order_id": "o",
                    "item": "x",
                }
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    frame.loc[len(frame)] = [pd.Timestamp("2024-02-29"), 1.0, 0.15, "B", "o", "x"]
    return frame


MEAN_BY_GROUP = {
    "measures": [{"id": "m1", "agg": "mean", "column": "score"}],
    "time": {"column": "day", "grain": "month"},
    "groups": [{"column": "group"}],
}


def test_the_change_of_a_mean_splits_into_rate_and_mix(tmp_path: Path) -> None:
    frame = two_months({("01", "A"): (500, 0.5), ("01", "B"): (500, 0.1),
                        ("02", "A"): (400, 0.55), ("02", "B"): (600, 0.05)})  # fmt: skip
    result = run(source(tmp_path, frame), **MEAN_BY_GROUP)
    (mix,) = of(result, "mix")
    assert mix.rule == "R-SIG-07" and mix.scope == "change"
    assert "vs Jan 2024: rate effect" in mix.message and "mix effect" in mix.message
    total = next(s for s in result.series if s.g1 is None)
    before, after = total.values[0], total.values[1]
    assert before is not None and after is not None
    assert mix.cells[0].value == pytest.approx(after - before)


def test_simpson_when_the_total_falls_while_the_groups_rise(tmp_path: Path) -> None:
    frame = two_months({("01", "A"): (800, 0.5), ("01", "B"): (200, 0.1),
                        ("02", "A"): (200, 0.55), ("02", "B"): (799, 0.15)})  # fmt: skip
    result = run(source(tmp_path, frame), **MEAN_BY_GROUP)
    assert "mix" not in kinds(result)
    (simpson,) = of(result, "simpson")
    assert simpson.severity == "warning" and simpson.rule == "R-SIG-08"
    assert simpson.message.startswith("Simpson's paradox: Average score falls while groups")
    assert "100 % of group rise" in simpson.message


def test_the_decomposition_is_exact_and_handles_a_new_group() -> None:
    parts = [
        kpi_mix.Part("A", 0.5, 0.6, 50, 30),
        kpi_mix.Part("B", 0.2, 0.25, 50, 50),
        kpi_mix.Part("C", None, 0.9, 0, 20),
    ]
    r0 = (0.5 * 50 + 0.2 * 50) / 100
    r1 = (0.6 * 30 + 0.25 * 50 + 0.9 * 20) / 100
    found = kpi_mix.decompose(r0, r1, parts)
    assert found is not None
    assert found.rate + found.mix == pytest.approx(r1 - r0)
    assert kpi_mix.decompose(r0, r1 + 0.2, parts) is None  # weights that do not add up


def test_no_mix_without_a_split_or_for_a_sum(tmp_path: Path) -> None:
    frame = two_months({("01", "A"): (500, 0.5), ("02", "A"): (400, 0.55)})
    src = source(tmp_path, frame)
    assert "mix" not in kinds(
        run(src, measures=MEAN_BY_GROUP["measures"], time=MONTHLY_SUM["time"])
    )
    summed = run(
        src, **{**MEAN_BY_GROUP, "measures": [{"id": "m1", "agg": "sum", "column": "value"}]}
    )
    assert "mix" not in kinds(summed) and "simpson" not in kinds(summed)


# --- P9: the started period at an equal date -------------------------------------------------


def test_the_started_month_is_compared_at_an_equal_date(tmp_path: Path) -> None:
    frame = daily("2025-01-01", "2026-11-10", lambda d: 100)
    frame["value"] = 10.0
    result = run(
        source(tmp_path, frame),
        measures=[
            {"id": "m1", "agg": "sum", "column": "value"},
            {"id": "m2", "agg": "mean", "column": "value"},
        ],
        time={"column": "day", "grain": "month"},
    )
    tile = result.tiles[0]
    assert tile.period == "Oct 2026"  # R-TILE-01 unchanged: the last complete month
    to_date = tile.to_date
    assert to_date is not None
    assert (to_date.period, to_date.label, to_date.previous_label, to_date.last_year_label) == (
        "Nov 2026",
        f"Nov 1{DASH}10",
        f"Oct 1{DASH}10",
        f"Nov 1{DASH}10 last year",
    )
    assert to_date.value == pytest.approx(100) and to_date.previous == pytest.approx(100)
    assert to_date.last_year == pytest.approx(100) and to_date.change == pytest.approx(0)
    assert to_date.elapsed == pytest.approx(10 / 30, abs=1e-3)
    assert to_date.projected == pytest.approx(300)
    mean = result.tiles[1].to_date
    assert mean is not None and mean.value == pytest.approx(10) and mean.projected is None


def test_no_comparison_at_date_when_the_last_period_is_complete_or_before_the_data(
    tmp_path: Path,
) -> None:
    complete = run(source(tmp_path, daily("2025-01-01", "2025-10-31", lambda d: 1)), **MONTHLY_SUM)
    assert complete.tiles[0].to_date is None
    young = run(
        source(tmp_path, daily("2025-10-15", "2025-11-10", lambda d: 1), "y"), **MONTHLY_SUM
    )
    to_date = young.tiles[0].to_date
    assert to_date is not None and to_date.previous is None and to_date.last_year is None


def test_a_kpi_without_date_still_has_a_signals_list(tmp_path: Path) -> None:
    result = run(
        source(tmp_path, daily("2024-01-01", "2024-03-31", lambda d: 1)),
        measures=[{"id": "m1", "agg": "count"}],
    )
    assert result.signals == [] and result.tiles[0].to_date is None


def test_an_unusual_sum_with_more_rows_says_so(tmp_path: Path) -> None:
    frame = daily("2023-01-01", "2025-12-31", seasonal, noise=0.25)
    march = frame[(frame.day.dt.year == 2025) & (frame.day.dt.month == 3)]
    frame = pd.DataFrame(pd.concat([frame, march, march], ignore_index=True))  # 3 x the rows
    frame["order_id"] = [f"o{i}" for i in range(len(frame))]
    (signal,) = of(run(source(tmp_path, frame), **MONTHLY_SUM), "anomaly")
    assert signal.message.startswith("Mar 2025 unusually high, with 200 % more rows than usual")


def test_an_empty_period_is_not_thin(tmp_path: Path) -> None:
    frame = daily("2024-01-01", "2024-12-31", lambda d: 100)
    frame = pd.DataFrame(frame[frame.day.dt.month != 6])  # June: no row at all
    assert "thin_period" not in kinds(run(source(tmp_path, frame), **MONTHLY_SUM))


def test_at_the_year_grain_the_previous_period_is_last_year(tmp_path: Path) -> None:
    frame = daily("2023-01-01", "2025-11-10", lambda d: 1)
    frame["value"] = 1.0
    result = run(
        source(tmp_path, frame),
        measures=[{"id": "m1", "agg": "sum", "column": "value"}],
        time={"column": "day", "grain": "year"},
    )
    to_date = result.tiles[0].to_date
    assert to_date is not None and to_date.label == f"Jan 1 {DASH} Nov 10"
    assert (
        to_date.previous == to_date.last_year == pytest.approx(315)
    )  # Jan 1 - Nov 10 2024, a leap year
