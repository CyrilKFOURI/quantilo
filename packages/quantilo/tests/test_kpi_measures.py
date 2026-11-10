"""Business measures of a KPI (053): margin and difference, share of values, weighted mean,
delay between two dates, rolling mean and trend, and the suggestions that use them."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quantilo.ingestion.sql import lit
from quantilo.kpi import trend as kpi_trend
from quantilo.kpi.compute import KpiError, compute
from quantilo.kpi.definition import Definition, Measure
from quantilo.kpi.suggest import suggest
from quantilo.typing.types import Parser


@dataclass
class Col:
    name: str
    type: Any
    role: str | None = None
    order: list[str] | None = None
    parser: Parser = field(default_factory=Parser)
    summary: dict[str, Any] | None = None


COLUMNS = [
    Col("ordered", "datetime"),
    Col("shipped", "datetime"),
    Col("revenue", "numeric_continuous", "amount"),
    Col("cost", "numeric_continuous", "cost"),
    Col("price", "numeric_continuous", "unit_price"),
    Col("units", "numeric_discrete", "quantity"),
    Col("order_id", "categorical_nominal", "order_key"),
    Col("status", "categorical_nominal", "status"),
    Col("country", "categorical_nominal", "country"),
    Col("churned", "boolean"),
]


def orders(rows: int = 3_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ordered = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, rows), "D")
    ordered = ordered.sort_values()
    lines = rng.integers(1, 4, rows)
    order_ids = np.repeat(np.arange(rows), lines)[:rows].astype(str)
    revenue = rng.gamma(2.0, 60.0, rows).round(2)
    price = rng.uniform(5, 50, rows).round(2)
    units = rng.integers(1, 9, rows).astype(float)
    units[::17] = -1.0  # returns: never a weight
    return pd.DataFrame(
        {
            "ordered": ordered,
            "shipped": ordered + pd.to_timedelta(rng.integers(0, 8, rows) * 24 + 6, "h"),
            "revenue": revenue,
            "cost": (revenue * rng.uniform(0.4, 0.9, rows)).round(2),
            "price": price,
            "units": units,
            "order_id": order_ids,
            "status": pd.Series(
                rng.choice(["paid", "cancelled", "pending", ""], rows, p=[0.6, 0.2, 0.15, 0.05])
            ).replace("", None),
            "country": rng.choice(["FR", "DE", "IT"], rows),
            "churned": rng.random(rows) < 0.3,
        }
    )


@pytest.fixture
def data(tmp_path: Path) -> tuple[str, pd.DataFrame]:
    frame = orders()
    path = tmp_path / "orders.parquet"
    frame.to_parquet(path)
    return f"read_parquet({lit(str(path))})", frame


def run(source: str, columns: list[Col] | None = None, **definition: Any):
    connection = duckdb.connect()
    try:
        return compute(
            connection, source, columns or COLUMNS, Definition.model_validate(definition)
        )
    finally:
        connection.close()


def total(result, measure: str = "m1"):
    return next(s for s in result.series if s.measure == measure and s.g1 is None and s.g2 is None)


def by_country(result, measure: str = "m1") -> dict[str, float | None]:
    return {
        s.g1: s.overall
        for s in result.series
        if s.measure == measure and s.g1 is not None and s.g2 is None
    }


# --- Backward compatibility ----------------------------------------------------------------------


def test_a_definition_saved_before_053_still_loads_with_its_defaults() -> None:
    old = {
        "measures": [{"id": "m1", "agg": "sum", "column": "revenue"}, {"id": "m2", "agg": "count"}],
        "ratio": {"numerator": "m1", "denominator": "m2", "percent": False},
        "view": {"show_as": "running"},
    }
    definition = Definition.model_validate(old)
    assert definition.ratio is not None and definition.ratio.kind == "ratio"
    assert definition.measures[0].values == []
    assert definition.measures[0].weight is None
    assert definition.measures[0].key is None
    assert definition.measures[0].until is None


def test_fields_of_another_aggregate_are_dropped_not_refused() -> None:
    measure = Measure(id="m1", agg="sum", column="revenue", values=["x"], weight="units", key="a")
    assert measure.values == [] and measure.weight is None and measure.key is None
    assert Measure(id="m1", agg="share", column="status", values=["paid"]).columns() == ["status"]
    with pytest.raises(ValidationError, match="values"):
        Measure(id="m1", agg="share", column="status")
    with pytest.raises(ValidationError, match="weight"):
        Measure(id="m1", agg="weighted_mean", column="price")
    with pytest.raises(ValidationError, match="delay"):
        Measure(id="m1", agg="sum", column="ordered", until="shipped")
    delay = Measure(id="m1", agg="median", column="ordered", until="shipped")
    assert delay.columns() == ["ordered", "shipped"]


# --- P4: margin and difference (R-MES-03, R-MES-04) ----------------------------------------------


def test_margin_is_computed_per_cell_from_the_sums(data) -> None:
    source, frame = data
    result = run(
        source,
        measures=[
            {"id": "m1", "agg": "sum", "column": "revenue"},
            {"id": "m2", "agg": "sum", "column": "cost"},
        ],
        ratio={"numerator": "m1", "denominator": "m2", "kind": "margin"},
        groups=[{"column": "country"}],
    )
    expected = frame.groupby("country").apply(
        lambda g: (g.revenue.sum() - g.cost.sum()) / g.revenue.sum(), include_groups=False
    )
    got = by_country(result, "ratio")
    for country, value in expected.items():
        assert got[country] == pytest.approx(value)
    ratio = next(m for m in result.measures if m.id == "ratio")
    assert ratio.format.percent and not ratio.additive
    assert ratio.label == "Margin (revenue - cost)"
    whole = (frame.revenue.sum() - frame.cost.sum()) / frame.revenue.sum()
    assert result.value == pytest.approx(whole)


def test_a_difference_of_sums_adds_up_a_difference_with_a_mean_does_not(data) -> None:
    source, frame = data
    result = run(
        source,
        measures=[
            {"id": "m1", "agg": "sum", "column": "revenue"},
            {"id": "m2", "agg": "sum", "column": "cost"},
        ],
        ratio={"numerator": "m1", "denominator": "m2", "kind": "difference"},
        groups=[{"column": "country"}],
    )
    ratio = next(m for m in result.measures if m.id == "ratio")
    assert ratio.additive and not ratio.format.percent
    assert ratio.label == "revenue - cost"
    parts = by_country(result, "ratio")
    assert sum(v or 0 for v in parts.values()) == pytest.approx(
        frame.revenue.sum() - frame.cost.sum()
    )
    mixed = Definition.model_validate(
        {
            "measures": [
                {"id": "m1", "agg": "sum", "column": "revenue"},
                {"id": "m2", "agg": "mean", "column": "cost"},
            ],
            "ratio": {"numerator": "m1", "denominator": "m2", "kind": "difference"},
        }
    )
    assert not mixed.additive("ratio")


# --- P5: share of values (R-MES-05) --------------------------------------------------------------


def test_share_of_rows_whose_status_is_in_the_values(data) -> None:
    source, frame = data
    result = run(
        source,
        measures=[{"id": "m1", "agg": "share", "column": "status", "values": ["paid", "pending"]}],
        groups=[{"column": "country"}],
    )
    filled = frame[frame.status.notna()]
    expected = filled.status.isin(["paid", "pending"]).mean()
    assert result.value == pytest.approx(expected)
    got = by_country(result)
    for country, group in filled.groupby("country"):
        assert got[country] == pytest.approx(group.status.isin(["paid", "pending"]).mean())
    measure = result.measures[0]
    assert measure.format.percent and not measure.additive
    assert measure.label == "% status: paid, pending"


def test_share_of_distinct_keys_counts_each_order_once(data) -> None:
    source, frame = data
    result = run(
        source,
        measures=[
            {
                "id": "m1",
                "agg": "share",
                "column": "status",
                "values": ["cancelled"],
                "key": "order_id",
            }
        ],
    )
    filled = frame[frame.status.notna()]
    hit = filled[filled.status == "cancelled"].order_id.nunique()
    assert result.value == pytest.approx(hit / filled.order_id.nunique())
    assert result.measures[0].label == "% of order_id: cancelled"


def test_share_of_a_boolean_reads_its_labels(data) -> None:
    source, frame = data
    result = run(
        source, measures=[{"id": "m1", "agg": "share", "column": "churned", "values": ["True"]}]
    )
    assert result.value == pytest.approx(frame.churned.mean())


# --- P18: weighted mean (R-MES-06) ---------------------------------------------------------------


def test_weighted_mean_ignores_rows_without_a_positive_weight(data) -> None:
    source, frame = data
    result = run(
        source,
        measures=[{"id": "m1", "agg": "weighted_mean", "column": "price", "weight": "units"}],
        groups=[{"column": "country"}],
    )
    kept = frame[frame.units > 0]
    assert result.value == pytest.approx((kept.price * kept.units).sum() / kept.units.sum())
    got = by_country(result)
    for country, group in kept.groupby("country"):
        assert got[country] == pytest.approx((group.price * group.units).sum() / group.units.sum())
    assert result.measures[0].label == "Weighted average price"
    assert not result.measures[0].additive


# --- P16b: delay (R-MES-07) ----------------------------------------------------------------------


def test_delay_between_two_dates_in_days(data) -> None:
    source, frame = data
    days = (frame.shipped - frame.ordered).dt.total_seconds() / 86_400
    mean = run(
        source, measures=[{"id": "m1", "agg": "mean", "column": "ordered", "until": "shipped"}]
    )
    assert mean.value == pytest.approx(days.mean())
    assert mean.measures[0].label == "Average days ordered → shipped"
    assert mean.measures[0].format.suffix == " d"
    median = run(
        source, measures=[{"id": "m1", "agg": "median", "column": "ordered", "until": "shipped"}]
    )
    assert median.value == pytest.approx(days.median())


def test_wrong_columns_name_the_field(data) -> None:
    source, _ = data
    cases = [
        ({"id": "m1", "agg": "share", "column": "revenue", "values": ["1"]}, "measures.m1.column"),
        (
            {"id": "m1", "agg": "weighted_mean", "column": "price", "weight": "status"},
            "measures.m1.weight",
        ),
        (
            {"id": "m1", "agg": "mean", "column": "revenue", "until": "shipped"},
            "measures.m1.column",
        ),
        ({"id": "m1", "agg": "mean", "column": "ordered", "until": "units"}, "measures.m1.until"),
        (
            {"id": "m1", "agg": "share", "column": "status", "values": ["x"], "key": "gone"},
            "measures.m1.key",
        ),
    ]
    for measure, where in cases:
        with pytest.raises(KpiError) as error:
            run(source, measures=[measure])
        assert error.value.field == where


# --- Show as: trend (R-SHOW-02) ------------------------------------------------------------------


def test_trend_is_stl_with_two_cycles_lowess_otherwise() -> None:
    months = 36
    rng = np.random.default_rng(3)
    season = np.tile(np.sin(np.arange(12) / 12 * 2 * np.pi) * 10, 3)
    values = list(np.arange(months) * 2.0 + 100 + season + rng.normal(0, 1, months))
    partial = [False] * months
    fitted, decision = kpi_trend.trend(values, partial, "month")
    assert decision.value == "stl" and decision.rule == "R-SHOW-02"
    assert "may change" in decision.reason
    last = fitted[-1]
    assert fitted[0] is not None and last is not None and abs(last - (100 + 2 * 35)) < 6
    short, decision = kpi_trend.trend(values[:18], partial[:18], "month")
    assert decision.value == "lowess" and short[0] is not None
    edge = [*partial[:17], True]
    cut, _ = kpi_trend.trend(values[:18], edge, "month")
    assert cut[-1] is None and cut[-2] is not None
    none, decision = kpi_trend.trend(values[:3], partial[:3], "month")
    assert all(v is None for v in none) and decision.value == "none"


def test_trend_is_only_computed_when_shown(data) -> None:
    source, _ = data
    plain = run(
        source,
        measures=[{"id": "m1", "agg": "sum", "column": "revenue"}],
        time={"column": "ordered"},
    )
    assert total(plain).trend is None and plain.resolved.trend is None
    shown = run(
        source,
        measures=[{"id": "m1", "agg": "sum", "column": "revenue"}],
        time={"column": "ordered", "grain": "month"},
        groups=[{"column": "country"}],
        view={"show_as": "trend"},
    )
    line = total(shown)
    assert line.trend is not None and len(line.trend) == len(line.values)
    assert shown.resolved.trend is not None and shown.resolved.trend.value == "stl"
    assert any(shown.resolved.trend.reason in why for why in shown.resolved.why)
    assert all(s.trend is not None for s in shown.series)


# --- Suggestions (R-SUG-20 to 24, R-SUG-06 revised) ----------------------------------------------


def summary(distinct: int, levels: list[str] | None = None, rows: int = 1_000) -> dict[str, Any]:
    return {
        "rows": rows,
        "null_count": 0,
        "distinct_count": distinct,
        "histogram": [{"label": level, "count": 10} for level in levels or []],
    }


def suggestion_columns() -> list[Col]:
    return [
        Col("order_date", "datetime", summary=summary(900)),
        Col("ship_date", "datetime", summary=summary(900)),
        Col("Revenue", "numeric_continuous", "amount", summary=summary(990)),
        Col("Cost", "numeric_continuous", "cost", summary=summary(990)),
        Col("Order ID", "categorical_nominal", "order_key", summary=summary(400)),
        Col(
            "Status",
            "categorical_nominal",
            "status",
            summary=summary(3, ["Paid", "Cancelled", "Pending"]),
        ),
        Col("Partner", "boolean", summary=summary(2)),
        Col("Churn", "boolean", summary=summary(2)),
    ]


def test_suggestions_offer_margin_share_of_status_and_delay() -> None:
    found = {s.rule: s for s in suggest(suggestion_columns())}
    margin = found["R-SUG-20"]
    assert margin.definition.ratio is not None and margin.definition.ratio.kind == "margin"
    assert margin.name == "Margin"
    assert margin.reasons[0].startswith("R-SUG-20 · ")
    share = found["R-SUG-22"].definition.measures[0]
    assert share.agg == "share" and share.values == ["Paid"] and share.key == "Order ID"
    assert "R-SUG-22" in found["R-SUG-22"].reasons[0]
    delay = found["R-SUG-24"].definition.measures[0]
    assert (delay.agg, delay.column, delay.until) == ("mean", "order_date", "ship_date")
    churn = found["R-SUG-06"].definition.measures[0]
    assert churn.column == "Churn"
    assert "outcome" in found["R-SUG-06"].reasons[0]


def test_a_profit_column_gives_the_profit_margin() -> None:
    columns = [
        Col("Order Date", "datetime", summary=summary(900)),
        Col("Sales", "numeric_continuous", "amount", summary=summary(990)),
        Col("Profit", "numeric_continuous", "amount", summary=summary(990)),
    ]
    found = {s.rule: s for s in suggest(columns)}
    profit = found["R-SUG-21"]
    assert profit.name == "Profit margin"
    ratio = profit.definition.ratio
    assert ratio is not None and ratio.kind == "ratio" and ratio.percent
    measures = {m.id: m.column for m in profit.definition.measures}
    assert measures[ratio.numerator] == "Profit" and measures[ratio.denominator] == "Sales"


def test_a_unit_price_with_a_quantity_is_a_weighted_mean() -> None:
    columns = [
        Col("InvoiceDate", "datetime", summary=summary(900)),
        Col("Price", "numeric_continuous", "amount", summary=summary(500)),
        Col("Quantity", "numeric_discrete", "quantity", summary=summary(50)),
    ]
    found = {s.rule: s for s in suggest(columns)}
    price = found["R-SUG-09"]
    measure = price.definition.measures[0]
    assert (measure.agg, measure.column, measure.weight) == ("weighted_mean", "Price", "Quantity")
    assert any(r.startswith("R-SUG-23 · ") for r in price.reasons)
    alone = {s.rule: s for s in suggest(columns[:2])}
    assert alone["R-SUG-09"].definition.measures[0].agg == "mean"


def test_a_weighted_percentage_reads_in_percent() -> None:
    from quantilo.kpi.rules import resolve_format

    shown = resolve_format("weighted_mean", [0.14, 0.2], "percentage")
    assert shown.percent and shown.scale == 100.0
