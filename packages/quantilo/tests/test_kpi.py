"""KPI builder (022): automatic choices rule by rule, and every number against pandas."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quantilo.ingestion.sql import lit
from quantilo.kpi import rules
from quantilo.kpi.compute import KpiError, compute
from quantilo.kpi.definition import Definition
from quantilo.kpi.probe import measure
from quantilo.kpi.suggest import Facts, choose_date, hierarchies, split_candidates, suggest
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
    Col("date", "datetime"),
    Col("revenue", "numeric_continuous", "amount"),
    Col("units", "numeric_discrete", "quantity"),
    Col("customer_id", "identifier", "identifier"),
    Col("country", "categorical_nominal", "country"),
    Col("segment", "categorical_nominal"),
    Col("churned", "boolean"),
    Col("rating", "numeric_continuous", "rating"),
    Col("profit", "numeric_continuous", "amount"),
    Col("size", "categorical_ordinal", None, ["S", "M", "L"]),
]


def sales(rows: int = 4_000, end: str = "2025-12-20", seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, int((pd.Timestamp(end) - pd.Timestamp("2023-01-01")).days) + 1, rows), "D"
    )
    dates = dates.sort_values()
    frame = pd.DataFrame(
        {
            "date": dates,
            "revenue": rng.gamma(2.0, 50.0, rows).round(2),
            "units": rng.integers(1, 6, rows),
            "customer_id": rng.integers(1, 400, rows).astype(str),
            "country": rng.choice(
                ["FR", "DE", "IT", "ES", "BE", "NL"], rows, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]
            ),
            "segment": rng.choice(["Retail", "Pro"], rows),
            "churned": rng.random(rows) < 0.2,
            "rating": rng.integers(1, 6, rows).astype(float),
            "profit": rng.normal(0, 40, rows).round(2),
            "size": rng.choice(["L", "S", "M"], rows),
        }
    )
    return frame


@pytest.fixture
def data(tmp_path: Path) -> tuple[str, pd.DataFrame]:
    frame = sales()
    path = tmp_path / "sales.parquet"
    frame.to_parquet(path)
    return f"read_parquet({lit(str(path))})", frame


def run(source: str, **definition: Any):
    connection = duckdb.connect()
    try:
        return compute(connection, source, COLUMNS, Definition.model_validate(definition))
    finally:
        connection.close()


def total(result, measure: str = "m1"):
    return next(s for s in result.series if s.measure == measure and s.g1 is None and s.g2 is None)


def cell(result, g1, measure: str = "m1", g2=None):
    return next(s for s in result.series if s.measure == measure and s.g1 == g1 and s.g2 == g2)


SUM_BY_MONTH = {
    "measures": [{"id": "m1", "agg": "sum", "column": "revenue"}],
    "time": {"column": "date", "grain": "month"},
}


# --- Numbers against pandas --------------------------------------------------------------------


def test_sum_by_month_by_country_top_3_matches_pandas(data) -> None:
    source, frame = data
    result = run(source, **SUM_BY_MONTH, groups=[{"column": "country"}], top=3)

    split = result.splits[0]
    assert split.levels == ["FR", "DE", "IT", "Other"] and split.others == 3
    months = frame.assign(m=frame["date"].dt.to_period("M"))
    expected = months.groupby("m")["revenue"].sum()
    assert len(result.periods) == len(expected) == 36
    assert total(result).values == pytest.approx(list(expected.values))
    others = months[~months["country"].isin(["FR", "DE", "IT"])].groupby("m")["revenue"].sum()
    assert cell(result, "Other").values == pytest.approx(list(others.values))
    parts = np.sum([[v or 0.0 for v in cell(result, g).values] for g in split.levels], axis=0)
    assert list(parts) == pytest.approx(total(result).values)  # the groups add up to the total
    assert total(result).overall == pytest.approx(frame["revenue"].sum())


def test_non_additive_totals_are_recomputed_on_the_rows(data) -> None:
    source, frame = data
    result = run(
        source,
        measures=[
            {"id": "m1", "agg": "median", "column": "revenue"},
            {"id": "m2", "agg": "distinct", "column": "customer_id"},
        ],
        groups=[{"column": "country"}],
    )

    medians = frame.groupby("country")["revenue"].median()
    assert total(result).overall == pytest.approx(frame["revenue"].median())  # R-TAB-07
    assert total(result).overall != pytest.approx(medians.sum())
    assert total(result, "m2").overall == frame["customer_id"].nunique()  # each customer once
    assert (
        sum(cell(result, g, "m2").overall or 0 for g in medians.index)
        > frame["customer_id"].nunique()
    )


def test_ratio_is_a_ratio_of_sums_per_cell(data) -> None:
    source, frame = data
    result = run(
        source,
        measures=[
            {"id": "m1", "agg": "sum", "column": "revenue"},
            {"id": "m2", "agg": "distinct", "column": "customer_id"},
        ],
        ratio={"numerator": "m1", "denominator": "m2"},
        time={"column": "date", "grain": "quarter"},
    )

    quarters = frame.groupby(frame["date"].dt.to_period("Q"))
    expected = quarters["revenue"].sum() / quarters["customer_id"].nunique()
    assert total(result, "ratio").values == pytest.approx(list(expected.values))
    assert result.measures[-1].label == "revenue per customer_id"
    assert result.name == "Revenue per customer_id by quarter"


def test_incomplete_last_month_is_flagged_and_left_out_of_the_tile(data) -> None:
    source, frame = data
    result = run(source, **SUM_BY_MONTH)

    assert result.periods[-1].partial and result.periods[-1].label == "Dec 2025"
    tile = result.tiles[0]
    months = frame.groupby(frame["date"].dt.to_period("M"))["revenue"].sum()
    assert tile.period == "Nov 2025" and tile.value == pytest.approx(months["2025-11"])
    assert tile.previous == pytest.approx(months["2025-10"])
    assert tile.last_year == pytest.approx(months["2024-11"])
    assert tile.change == pytest.approx((months["2025-11"] - months["2025-10"]) / months["2025-10"])


def test_last_12_months_still_compares_with_last_year(data) -> None:
    source, frame = data
    result = run(
        source,
        measures=[{"id": "m1", "agg": "count"}],
        time={"column": "date", "grain": "month", "period": {"kind": "last", "n": 12}},
    )

    counts = frame.groupby(frame["date"].dt.to_period("M")).size()
    assert next(p.label for p in result.periods) == "Jan 2025" and len(result.periods) == 12
    assert total(result).last_year == pytest.approx(list(counts["2024-01":"2024-12"].values))
    assert total(result).previous[0] == counts["2024-12"]
    assert result.rows == int(counts["2025-01":].sum())


def test_true_share_and_page_period_range(data) -> None:
    source, frame = data
    result = run(
        source,
        measures=[{"id": "m1", "agg": "true_share", "column": "churned"}],
        time={
            "column": "date",
            "period": {"kind": "range", "from": "2024-01-01", "to": "2024-12-31"},
        },
    )

    year = frame[frame["date"].dt.year == 2024]
    assert total(result).overall == pytest.approx(year["churned"].mean())
    assert result.measures[0].format.percent and result.measures[0].format.scale == 100


def test_empty_periods_are_zero_for_sums_and_empty_for_means(tmp_path: Path) -> None:
    frame = sales(400)
    frame = frame[frame["date"].dt.month != 6]  # no June at all
    path = tmp_path / "gaps.parquet"
    frame.to_parquet(path)
    source = f"read_parquet({lit(str(path))})"

    summed = run(source, **SUM_BY_MONTH)
    june = [p.label for p in summed.periods].index("Jun 2023")
    assert total(summed).values[june] == 0
    mean = run(
        source,
        measures=[{"id": "m1", "agg": "mean", "column": "rating"}],
        time={"column": "date", "grain": "month"},
    )
    assert total(mean).values[june] is None


def test_errors_name_the_field(data) -> None:
    source, _ = data
    with pytest.raises(KpiError) as error:
        run(source, measures=[{"id": "m1", "agg": "sum", "column": "country"}])
    assert error.value.field == "measures.m1.column"
    with pytest.raises(KpiError) as error:
        run(source, measures=[{"id": "m1", "agg": "count"}], groups=[{"column": "gone"}])
    assert error.value.code == "unknown_column"
    with pytest.raises(KpiError) as error:
        run(
            source,
            measures=[{"id": "m1", "agg": "count"}],
            time={"column": "date", "period": {"kind": "range", "from": "2030-01-01"}},
        )
    assert error.value.code == "no_rows"


def test_ordinal_split_keeps_its_natural_order(data) -> None:
    source, _ = data
    result = run(source, measures=[{"id": "m1", "agg": "count"}], groups=[{"column": "size"}])
    assert result.splits[0].levels == ["S", "M", "L"] and result.splits[0].ordinal


# --- Automatic choices ---------------------------------------------------------------------------


def test_grain_auto(data) -> None:
    source, _ = data
    assert (
        run(
            source, measures=[{"id": "m1", "agg": "count"}], time={"column": "date"}
        ).resolved.grain.value  # type: ignore[union-attr]
        == "month"
    )  # 3 years
    six_months = run(
        source,
        measures=[{"id": "m1", "agg": "count"}],
        time={
            "column": "date",
            "period": {"kind": "range", "from": "2025-01-01", "to": "2025-06-30"},
        },
    )
    assert six_months.resolved.grain is not None
    assert six_months.resolved.grain.value == "week"


def test_grain_rules() -> None:
    year = rules.GRAIN_SECONDS["year"]
    decision, warning = rules.resolve_grain("auto", "day", 3 * year)
    assert (decision.value, decision.rule, warning) == ("month", "R-TIME-01", None)
    decision, warning = rules.resolve_grain("day", "month", 3 * year)  # monthly data by day
    assert decision.value == "month" and decision.rule == "R-TIME-02" and warning
    assert not next(o for o in decision.options if o.value == "week").allowed
    decision, warning = rules.resolve_grain("day", "day", 10 * year)  # 3,653 days
    assert decision.value == "week" and decision.rule == "R-TIME-03"


def facts(**changes: Any) -> rules.Facts:
    base = dict(
        time=True,
        periods=12,
        groups=1,
        levels=[4],
        additive=True,
        negative=False,
        dominant=0.4,
        empty=0.0,
        measure="revenue",
        split="country",
    )
    return rules.Facts(**{**base, **changes})


@pytest.mark.parametrize(
    ("changes", "chart", "rule"),
    [
        ({"groups": 0, "levels": []}, "bars", "R-VIS-01"),
        ({"groups": 0, "levels": [], "periods": 40}, "area", "R-VIS-02"),
        ({"groups": 0, "levels": [], "additive": False}, "line", "R-VIS-03"),
        ({}, "stacked_bars", "R-VIS-04"),
        ({"periods": 40}, "stacked_area", "R-VIS-04"),
        ({"additive": False}, "line", "R-VIS-05"),
        ({"groups": 2, "levels": [4, 3]}, "stacked_bars", "R-VIS-06 + R-VIS-04"),
        ({"time": False, "groups": 0, "levels": []}, "none", "R-VIS-07"),
        ({"time": False}, "hbar", "R-VIS-08"),
        ({"time": False, "groups": 2, "levels": [4, 3]}, "hbar_stacked", "R-VIS-09"),
        ({"time": False, "groups": 2, "levels": [4, 3], "additive": False}, "heatmap", "R-VIS-10"),
        ({"periods": 3}, "grouped_bars", "R-VIS-11"),
        ({"dominant": 0.8}, "line", "R-VIS-12"),
        ({"empty": 0.5}, "stacked_bars", "R-VIS-13"),
        ({"groups": 0, "levels": [], "empty": 0.5, "periods": 40}, "bars", "R-VIS-13"),
        ({"time": False, "groups": 2, "levels": [10, 12]}, "heatmap", "R-VIS-14"),
        ({"negative": True}, "grouped_bars", "R-VIS-21"),
        ({"negative": True, "periods": 40}, "line", "R-VIS-21"),
    ],
)
def test_chart_rules(changes: dict, chart: str, rule: str) -> None:
    decision = rules.resolve_chart(facts(**changes))
    assert (decision.value, decision.rule) == (chart, rule)


def test_stacking_is_refused_for_levels_and_negative_values() -> None:
    mean = {o.value: o for o in rules.chart_options(facts(additive=False))}
    assert not mean["stacked_bars"].allowed and "R-VIS-20" in (mean["stacked_bars"].reason or "")
    assert mean["line"].allowed
    negative = {o.value: o for o in rules.chart_options(facts(negative=True))}
    assert not negative["stacked_area"].allowed and "R-VIS-21" in (
        negative["stacked_area"].reason or ""
    )
    assert not rules.share_allowed(facts(additive=False)).allowed
    assert rules.share_allowed(facts()).allowed


@pytest.mark.parametrize(
    ("time", "groups", "periods", "layout", "rule"),
    [
        (True, 0, 36, "periods_rows", "R-TAB-01"),
        (True, 1, 12, "periods_columns", "R-TAB-02"),
        (True, 1, 36, "periods_rows", "R-TAB-03"),
        (True, 2, 36, "periods_columns", "R-TAB-04"),
        (False, 1, 0, "groups_rows", "R-TAB-05"),
        (False, 2, 0, "crosstab", "R-TAB-06"),
    ],
)
def test_table_layout_rules(time: bool, groups: int, periods: int, layout: str, rule: str) -> None:
    decision = rules.resolve_layout(time, groups, periods)
    assert (decision.value, decision.rule) == (layout, rule)


def test_shading_and_formats() -> None:
    assert rules.resolve_shading("periods_columns", 1).value == "on"
    assert rules.resolve_shading("periods_rows", 0).value == "off"
    assert rules.resolve_format("sum", [12.0, 30.0], "amount").decimals == 0
    assert rules.resolve_format("mean", [3.21, 4.5], None).decimals == 2
    assert rules.resolve_format("mean", [321.0], None).decimals == 1
    assert rules.resolve_format("mean", [0.12, 0.4], "percentage").scale == 100
    assert rules.resolve_format(None, [0.5], None, percent_ratio=True).percent


def test_charts_follow_the_measure(data) -> None:
    source, _ = data
    by_country = {"time": {"column": "date", "grain": "month"}, "groups": [{"column": "country"}]}
    mean = run(source, measures=[{"id": "m1", "agg": "mean", "column": "rating"}], **by_country)
    assert mean.measures[0].chart.value == "line"  # a mean never stacks
    profit = run(
        source,
        measures=[{"id": "m1", "agg": "sum", "column": "profit"}],
        time={"column": "date", "grain": "quarter"},
        groups=[{"column": "segment"}],
    )
    assert profit.measures[0].chart.rule == "R-VIS-21"  # negative quarters: side by side
    single = run(
        source,
        measures=[{"id": "m1", "agg": "count"}],
        time={
            "column": "date",
            "grain": "year",
            "period": {"kind": "range", "from": "2024-01-01", "to": "2024-12-31"},
        },
        groups=[{"column": "country"}],
    )
    assert not single.resolved.time and single.measures[0].chart.value == "hbar"  # R-TIME-04


# --- Suggestions ---------------------------------------------------------------------------------


def summarised() -> list[Col]:
    summaries = {
        "customer_id": {"rows": 4000, "null_count": 0, "distinct_count": 399},
        "country": {"rows": 4000, "null_count": 0, "distinct_count": 6},
        "segment": {"rows": 4000, "null_count": 0, "distinct_count": 2},
        "size": {"rows": 4000, "null_count": 0, "distinct_count": 3},
        "churned": {"rows": 4000, "null_count": 0, "distinct_count": 2},
    }
    return [Col(c.name, c.type, c.role, c.order, c.parser, summaries.get(c.name)) for c in COLUMNS]


def test_suggestions_follow_the_columns() -> None:
    found = suggest(summarised())
    # 048: ranked by score (R-SUG-16), no longer in rule order; 053: profit ÷ revenue (R-SUG-21)
    assert [s.name for s in found] == [
        "Total revenue",
        "Total profit",
        "Profit margin",
        "Total units",
        "Average unit price",
        "Rows",
        "Distinct customer_id",
        "Revenue per customer_id",
        "% churned",
        "Average rating",
    ]
    assert [s.score for s in found] == sorted((s.score for s in found), reverse=True)
    assert all(s.definition.time and s.definition.time.column == "date" for s in found)
    assert split_candidates(summarised()) == ["country", "size", "segment"]


def test_every_suggestion_says_why_with_its_rule() -> None:
    for suggestion in suggest(summarised()):
        assert suggestion.reasons[0].startswith(f"{suggestion.rule} · ")
        score = f"R-SUG-16 · Score {suggestion.score}/100"
        assert any(r.startswith(score) for r in suggestion.reasons)
        assert 0 < suggestion.score <= 100


FULL = {"rows": 9_994, "null_count": 0}


def orders(*extra: Col) -> list[Col]:
    def col(name: str, kind: str, role: str | None = None, distinct: int | None = None) -> Col:
        summary = {**FULL, "distinct_count": distinct or 2_000}
        return Col(name, kind, role, summary=summary)

    return [
        col("Order Date", "datetime", None, 1_237),
        col("Ship Date", "datetime", None, 1_334),
        col("Order ID", "categorical_nominal", "order_key", 5_009),
        col("Customer ID", "categorical_nominal", "customer_key", 793),
        col("Region", "categorical_nominal", "region", 4),
        col("State", "categorical_nominal", "region", 49),
        col("Segment", "categorical_nominal", None, 3),
        col("Sales", "numeric_continuous", "amount"),
        col("Quantity", "numeric_discrete", "quantity"),
        col("Price", "numeric_continuous", "amount"),
        *extra,
    ]


def columns_of(definition: Definition) -> list[tuple[str, str | None]]:
    return [(m.agg, m.column) for m in definition.measures]


def test_business_ratios_from_order_and_customer_keys() -> None:
    found = {s.name: s for s in suggest(orders())}

    basket = found["Average basket"].definition
    assert basket.ratio is not None
    assert columns_of(basket) == [("sum", "Sales"), ("distinct", "Order ID")]
    units = found["Units per order"].definition
    assert columns_of(units) == [("sum", "Quantity"), ("distinct", "Order ID")]
    price = found["Average unit price"].definition
    assert columns_of(price) == [("sum", "Sales"), ("sum", "Quantity")]
    per_customer = found["Sales per customer"].definition
    assert columns_of(per_customer) == [("sum", "Sales"), ("distinct", "Customer ID")]
    assert "Distinct Customer ID" in found and "Distinct Order ID" in found
    assert "5,009 orders" in found["Average basket"].reasons[0]


def test_a_price_is_averaged_never_summed() -> None:
    found = {s.name: s for s in suggest(orders())}

    assert "Total Price" not in found
    # 053: weighted by Quantity (R-SUG-23)
    assert found["Weighted average Price"].rule == "R-SUG-09"
    assert found["Weighted average Price"].definition.measures[0].agg == "weighted_mean"
    unit = Col("unit_price", "numeric_continuous", "unit_price")
    assert any(s.name == "Weighted average unit_price" for s in suggest([*orders()[:-1], unit]))


def test_the_date_is_the_order_one_never_a_birth_date() -> None:
    columns = orders()
    time = suggest(columns)[0].definition.time
    assert time is not None and time.column == "Order Date"
    shipped = [c for c in columns if c.name != "Order Date"]
    birth = Col("birth_date", "datetime")
    updated = Col("last_updated", "datetime")
    chosen, reason = choose_date([birth, updated, *shipped])
    assert chosen is not None and chosen.name == "Ship Date"
    assert reason and "birth_date is never a time axis" in reason
    alone, why = choose_date([birth])
    assert alone is None and why and why.startswith("R-SUG-17")
    # equal names: the best-filled date
    sparse = Col("date_a", "datetime", summary={"rows": 10, "null_count": 5, "distinct_count": 4})
    full = Col("date_b", "datetime", summary={"rows": 10, "null_count": 0, "distinct_count": 9})
    assert choose_date([sparse, full])[0] is full


def test_amounts_in_several_currencies_are_split_by_currency() -> None:
    currency = Col(
        "currency", "categorical_nominal", "currency", summary={**FULL, "distinct_count": 3}
    )
    found = {s.name: s for s in suggest(orders(currency))}

    assert [g.column for g in found["Total Sales"].definition.groups] == ["currency"]
    assert any(r.startswith("R-SUG-19") for r in found["Total Sales"].reasons)
    assert found["Rows"].definition.groups == []


def test_hierarchies_and_twins_from_the_rows() -> None:
    code = Col("Region code", "categorical_nominal", summary={**FULL, "distinct_count": 4})
    columns = orders(code)
    facts = Facts(
        dependency={
            ("State", "Region"): 1.0,
            ("Region", "State"): 0.2,
            ("Region", "Region code"): 1.0,
            ("Region code", "Region"): 0.995,
            ("State", "Region code"): 1.0,
            ("Region code", "State"): 0.2,
        }
    )

    found = hierarchies(columns, facts)
    assert (found[0].parent, found[0].child) == ("Region", "State")
    assert found[0].reason.startswith("R-SUG-18 · Each State belongs to one Region (100% of rows)")
    splits = split_candidates(columns, facts)
    assert "Region" in splits and "Region code" not in splits  # twins never together


def test_a_measure_that_moves_with_a_better_one_ranks_lower() -> None:
    columns = orders(Col("Sales incl VAT", "numeric_continuous", "amount"))
    plain = {s.name: s.score for s in suggest(columns)}
    facts = Facts(correlation={frozenset(("Sales", "Sales incl VAT")): 0.999})
    found = {s.name: s for s in suggest(columns, facts)}

    assert found["Total Sales incl VAT"].score == plain["Total Sales incl VAT"] // 2
    assert any("Moves with Sales (r = 1.00)" in r for r in found["Total Sales incl VAT"].reasons)


def test_one_order_per_customer_drops_the_customer_kpis() -> None:
    facts = Facts(dependency={("Order ID", "Customer ID"): 1.0, ("Customer ID", "Order ID"): 1.0})
    names = [s.name for s in suggest(orders(), facts)]
    assert "Average basket" in names and "Sales per customer" not in names


def test_more_suggestions_than_the_page_shows_stop_at_twelve() -> None:
    found = suggest(orders())
    assert 6 < len(found) <= 12


def test_the_probe_measures_dependencies_and_correlations(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    states = [f"S{i}" for i in range(20)]
    region = {s: f"R{i % 4}" for i, s in enumerate(states)}
    picked = rng.choice(states, 3_000)
    sales = rng.gamma(2.0, 50.0, 3_000)
    frame = pd.DataFrame(
        {
            "state": picked,
            "region": [region[s] for s in picked],
            "segment": rng.choice(["A", "B", "C"], 3_000),
            "sales": sales,
            "sales_vat": sales * 1.2,
            "units": rng.integers(1, 6, 3_000),
        }
    )
    path = tmp_path / "p.parquet"
    frame.to_parquet(path)
    source = f"read_parquet({lit(str(path))}, file_row_number = true)"
    labels = [Col(name, "categorical_nominal") for name in ("state", "region", "segment")]
    numbers = [Col(name, "numeric_continuous") for name in ("sales", "sales_vat", "units")]

    facts = measure(duckdb.connect(), source, labels, numbers)

    assert facts.fixes("state", "region") == pytest.approx(1.0)
    assert (facts.fixes("region", "state") or 1) < 0.5
    assert (facts.fixes("segment", "state") or 1) < 0.5
    assert facts.corr("sales", "sales_vat") == pytest.approx(1.0)
    assert abs(facts.corr("sales", "units") or 1) < 0.2


def test_without_a_date_suggestions_split_at_once() -> None:
    columns = [c for c in summarised() if c.type != "datetime"]
    found = suggest(columns)
    assert found[0].definition.time is None
    assert found[0].definition.groups[0].column == "country"


def test_a_column_without_readable_values_is_never_suggested() -> None:
    columns = [
        *summarised(),
        Col(
            "store",
            "numeric_discrete",
            "duration",
            summary={"rows": 9, "null_count": 9, "distinct_count": 0},
        ),
    ]
    columns = [c for c in columns if c.name != "rating"]
    assert all("store" not in s.name for s in suggest(columns))


def test_the_view_keeps_a_line_over_bars_and_the_bar_spacing() -> None:
    old = Definition.model_validate({"measures": [{"id": "m1", "agg": "count"}]})
    assert (old.view.line, old.view.spacing) == (False, "normal")  # saved before 045

    chosen = Definition.model_validate(
        {"measures": [{"id": "m1", "agg": "count"}], "view": {"line": True, "spacing": "wide"}}
    )
    assert (chosen.view.line, chosen.view.spacing) == (True, "wide")


def test_the_view_keeps_a_palette_of_its_own() -> None:
    assert (
        Definition.model_validate({"measures": [{"id": "m1", "agg": "count"}]}).view.palette is None
    )
    chosen = Definition.model_validate(
        {"measures": [{"id": "m1", "agg": "count"}], "view": {"palette": "tableau20"}}
    )
    assert chosen.view.palette == "tableau20"
    with pytest.raises(ValidationError):
        Definition.model_validate(
            {"measures": [{"id": "m1", "agg": "count"}], "view": {"palette": "<x>"}}
        )
    with pytest.raises(ValueError):
        Definition.model_validate(
            {"measures": [{"id": "m1", "agg": "count"}], "view": {"spacing": "huge"}}
        )
