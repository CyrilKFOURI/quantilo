"""Cohorts and funnels (060), targets and the kind of dataset (061)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quantilo.ingestion.sql import lit
from quantilo.kpi import cohort as kpi_cohort
from quantilo.kpi.archetypes import classify, templates
from quantilo.kpi.cohort import CohortError, propose, stages_of
from quantilo.kpi.compute import KpiError, compute
from quantilo.kpi.definition import Definition
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


def summary(distinct: int, levels: list[str] | None = None, rows: int = 1_000) -> dict[str, Any]:
    return {
        "rows": rows,
        "null_count": 0,
        "distinct_count": distinct,
        "histogram": [{"label": level, "count": 1} for level in levels or []],
    }


def parquet(tmp_path: Path, frame: pd.DataFrame, numbered: bool = False) -> str:
    path = tmp_path / "rows.parquet"
    frame.to_parquet(path)
    options = ", file_row_number = true" if numbered else ""
    return f"read_parquet({lit(str(path))}{options})"


CUSTOMER = Col("customer", "categorical_nominal", "customer_key")
DAY = Col("day", "datetime")


def visits() -> pd.DataFrame:
    """a, b, c arrive in January; a comes back in February and March, b in March; d arrives in
    February and comes back in March; e arrives in March."""
    rows = [
        ("a", "2024-01-03"),
        ("a", "2024-01-20"),
        ("b", "2024-01-10"),
        ("c", "2024-01-15"),
        ("a", "2024-02-02"),
        ("d", "2024-02-11"),
        ("a", "2024-03-05"),
        ("b", "2024-03-07"),
        ("d", "2024-03-09"),
        ("e", "2024-03-30"),
    ]
    return pd.DataFrame(
        {"customer": [r[0] for r in rows], "day": pd.to_datetime([r[1] for r in rows])}
    )


def cohorts(source: str, grain: Any = "month"):
    connection = duckdb.connect()
    try:
        return kpi_cohort.retention(connection, source, CUSTOMER, DAY, grain)
    finally:
        connection.close()


# --- 060 · Retention by cohort -------------------------------------------------------------------


def test_retention_counts_each_customer_once_per_period(tmp_path: Path) -> None:
    result = cohorts(parquet(tmp_path, visits()))
    assert [c.label for c in result.cohorts] == ["Jan 2024", "Feb 2024", "Mar 2024"]
    assert [c.size for c in result.cohorts] == [3, 1, 1]
    january, february, march = result.cohorts
    assert january.retained == pytest.approx([1.0, 1 / 3, 2 / 3])
    assert february.retained[:2] == pytest.approx([1.0, 1.0])
    assert february.retained[2] is None  # not observed yet
    assert march.retained[1:] == [None, None]
    # new and returning customers per period, from the same pass
    assert [(p.new, p.returning) for p in result.periods] == [(3, 0), (1, 1), (1, 3)]
    assert result.customers == 5
    assert result.returning_share == pytest.approx(3 / 5)
    assert "R-COH-05" in result.reason
    assert "next month" in result.summary


def test_the_grain_follows_the_span_of_the_data(tmp_path: Path) -> None:
    source = parquet(tmp_path, visits())
    weekly = cohorts(source, "auto")
    assert weekly.grain == "week"  # 3 months: fewer than 4
    assert "R-COH-03" in weekly.grain_reason


def test_too_few_periods_or_returning_customers_are_said(tmp_path: Path) -> None:
    short = visits().loc[lambda f: f["day"] < "2024-02-15"]
    with pytest.raises(CohortError, match="R-COH-02"):
        cohorts(parquet(tmp_path, short))
    once = pd.DataFrame(
        {
            "customer": [f"c{i}" for i in range(30)],
            "day": pd.date_range("2024-01-01", periods=30, freq="5D"),
        }
    )
    with pytest.raises(CohortError, match="come back"):
        cohorts(parquet(tmp_path, once))


def test_an_incomplete_last_period_is_said(tmp_path: Path) -> None:
    frame = visits()
    frame.loc[frame["customer"] == "e", "day"] = pd.Timestamp("2024-03-12")
    frame = frame.loc[frame["day"] <= "2024-03-12"]
    result = cohorts(parquet(tmp_path, frame))
    assert result.periods[-1].partial
    assert any("incomplete" in w for w in result.warnings)


# --- 060 · Funnel --------------------------------------------------------------------------------

STAGE = Col("stage", "categorical_nominal", "status")
DEAL = Col("deal", "categorical_nominal", "order_key")
STAGES = ["Lead", "Qualified", "Proposal", "Won"]


def run_funnel(source: str, key: Col | None):
    connection = duckdb.connect()
    try:
        return kpi_cohort.funnel(connection, source, STAGE, STAGES, key)
    finally:
        connection.close()


def test_funnel_counts_each_key_at_its_furthest_stage(tmp_path: Path) -> None:
    rows = [
        ("d1", "Lead"),
        ("d1", "Qualified"),
        ("d1", "Proposal"),
        ("d1", "Won"),
        ("d2", "Lead"),
        ("d2", "Qualified"),
        ("d3", "Lead"),
        ("d4", "Lead"),
        ("d4", "Proposal"),  # skipped Qualified: passed it all the same
        ("d5", "Lost"),
    ]
    frame = pd.DataFrame({"deal": [r[0] for r in rows], "stage": [r[1] for r in rows]})
    result = run_funnel(parquet(tmp_path, frame, numbered=True), DEAL)
    assert [s.reached for s in result.stages] == [4, 3, 2, 1]
    assert result.stages[1].conversion == pytest.approx(0.75)
    assert result.stages[-1].of_first == pytest.approx(0.25)
    assert result.outside == 1 and result.outside_labels == ["Lost"]
    assert "R-FUN-03" in result.summary and "R-FUN-02" in result.reason


def test_funnel_without_a_key_counts_rows(tmp_path: Path) -> None:
    frame = pd.DataFrame({"stage": ["Lead"] * 5 + ["Qualified"] * 3 + ["Won"] * 2})
    result = run_funnel(parquet(tmp_path, frame, numbered=True), None)
    assert [s.reached for s in result.stages] == [10, 5, 2, 2]
    assert result.key is None


def test_known_funnels_are_found_in_the_values() -> None:
    found = stages_of(["Won", "Lost", "Lead", "qualified", "Proposal_Sent"])
    assert found is not None
    assert found[0] == ["Lead", "qualified", "Proposal_Sent", "Won"]
    assert stages_of(["Low", "Medium", "High"]) is None  # an order, not a funnel
    assert stages_of(["Lead", "Won"]) is None  # 3 stages at least


def test_proposals_say_their_rule_and_reason() -> None:
    columns = [
        Col("customer", "categorical_nominal", "customer_key", summary=summary(200)),
        Col("ordered", "datetime"),
        Col("deal", "categorical_nominal", "order_key", summary=summary(600)),
        Col("stage", "categorical_nominal", "status", summary=summary(5, [*STAGES, "Lost"])),
    ]
    found = propose(columns, columns[1])
    assert [a.rule for a in found] == ["R-COH-01", "R-FUN-01"]
    assert found[0].customer == "customer" and found[0].date == "ordered"
    assert "repeats (200 values over 1,000 rows)" in found[0].reason
    assert found[1].stages == STAGES and found[1].key == "deal"
    # a customer key that never repeats (one row each) gives no cohort
    unique = Col("customer", "categorical_nominal", "customer_key", summary=summary(1_000))
    assert propose([unique, columns[1]], columns[1]) == []


# --- 061 · Targets -------------------------------------------------------------------------------

SALES = [
    Col("month", "datetime"),
    Col("sales", "numeric_continuous", "amount"),
    Col("budget", "numeric_continuous", "target"),
    Col("region", "categorical_nominal", "region"),
]


def monthly(tmp_path: Path) -> str:
    months = pd.date_range("2024-01-01", periods=6, freq="MS")
    frame = pd.DataFrame(
        {
            "month": list(months) * 2,
            "sales": [80.0, 90, 100, 110, 120, 130] * 2,
            "budget": [100.0] * 12,
            "region": ["N"] * 6 + ["S"] * 6,
        }
    )
    return parquet(tmp_path, frame)


def kpi(source: str, **definition: Any):
    connection = duckdb.connect()
    try:
        return compute(connection, source, SALES, Definition.model_validate(definition))
    finally:
        connection.close()


def test_a_target_column_is_summed_over_the_tiles_period(tmp_path: Path) -> None:
    result = kpi(
        monthly(tmp_path),
        measures=[{"id": "m1", "agg": "sum", "column": "sales"}],
        time={"column": "month", "grain": "month"},
        goal={"column": "budget"},
    )
    goal = result.tiles[0].goal
    assert goal is not None and goal.source == "column"
    assert result.tiles[0].value == pytest.approx(260)  # June, both regions
    assert goal.target == pytest.approx(200)
    assert goal.attained == pytest.approx(1.3)
    assert goal.gap == pytest.approx(60)
    assert "R-GOAL-01" in goal.reason and "Jun 2024" in goal.reason


def test_a_typed_target_and_a_mean_against_its_target(tmp_path: Path) -> None:
    typed = kpi(
        monthly(tmp_path),
        measures=[{"id": "m1", "agg": "sum", "column": "sales"}],
        goal={"value": 1_000},
    )
    goal = typed.tiles[0].goal
    assert goal is not None and goal.attained == pytest.approx(1_260 / 1_000)
    assert "R-GOAL-02" in goal.reason
    mean = kpi(
        monthly(tmp_path),
        measures=[{"id": "m1", "agg": "mean", "column": "sales"}],
        goal={"column": "budget"},
    )
    goal = mean.tiles[0].goal
    assert goal is not None and goal.target == pytest.approx(100)  # averaged, never summed
    assert "does not add up" in goal.reason


def test_a_target_is_a_column_or_a_value_and_a_number(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Definition.model_validate(
            {"measures": [{"id": "m1", "agg": "count"}], "goal": {"value": 3, "column": "budget"}}
        )
    with pytest.raises(ValidationError):
        Definition.model_validate(
            {"measures": [{"id": "m1", "agg": "count"}], "goal": {"measure": "m2", "value": 3}}
        )
    with pytest.raises(KpiError) as error:
        kpi(monthly(tmp_path), measures=[{"id": "m1", "agg": "count"}], goal={"column": "region"})
    assert error.value.field == "goal.column"


def test_an_amount_beside_a_target_column_is_suggested_against_it() -> None:
    columns = [
        Col("month", "datetime"),
        Col("sales", "numeric_continuous", "amount", summary=summary(900)),
        Col("budget", "numeric_continuous", "target", summary=summary(300)),
    ]
    found = {s.rule: s for s in suggest(columns)}
    assert "R-SUG-31" in found
    goal = found["R-SUG-31"].definition.goal
    assert goal is not None and goal.column == "budget"
    assert found["R-SUG-31"].name == "Sales vs budget"
    # a 0/1 machine-learning target is no business target
    binary = Col("target", "numeric_discrete", "target", summary=summary(2))
    assert "R-SUG-31" not in {s.rule for s in suggest([*columns[:2], binary])}


# --- 061 · The kind of dataset -------------------------------------------------------------------


def test_every_kind_has_its_rules() -> None:
    kinds = {t.id for t in templates()}
    assert kinds == {"sales", "subscription", "support", "hr", "web"}
    assert all(t.rules and t.needs for t in templates())


def test_a_sales_dataset_gets_a_pack_of_its_kpis() -> None:
    columns = [
        Col("Order Date", "datetime"),
        Col("Order ID", "categorical_nominal", "order_key", summary=summary(500)),
        Col("Customer ID", "categorical_nominal", "customer_key", summary=summary(300)),
        Col("Sales", "numeric_continuous", "amount", summary=summary(900)),
        Col("Profit", "numeric_continuous", "amount", summary=summary(900)),
        Col("Quantity", "numeric_discrete", "quantity", summary=summary(14)),
        Col("Region", "categorical_nominal", "region", summary=summary(4)),
    ]
    found = suggest(columns)
    kind = classify(columns, found)
    assert kind is not None and kind.id == "sales"
    assert kind.message == f"Looks like sales data · {len(kind.kpis)} KPIs"
    assert 3 <= len(kind.kpis) <= 5
    names = {s.name for s in found}
    assert set(kind.kpis) <= names
    assert kind.reasons[0].startswith("R-ARC-01 · Looks like sales data: Sales (amount)")
    assert "nothing is saved until you confirm" in kind.reasons[1]


def test_a_subscription_dataset_and_no_kind_at_all() -> None:
    columns = [
        Col("customerID", "identifier", "customer_key", summary=summary(1_000)),
        Col("tenure", "numeric_discrete", "duration", summary=summary(70)),
        Col("Contract", "categorical_nominal", summary=summary(3)),
        Col("MonthlyCharges", "numeric_continuous", "amount", summary=summary(900)),
        Col("Churn", "boolean", summary=summary(2)),
    ]
    kind = classify(columns, suggest(columns))
    assert kind is not None and kind.id == "subscription"
    plain = [Col("x", "numeric_continuous", summary=summary(900)), Col("y", "categorical_nominal")]
    assert classify(plain, suggest(plain)) is None


def test_on_real_shapes_numbers_stay_put(tmp_path: Path) -> None:
    """A larger random log: every cohort starts at 100 % and retention never exceeds 1."""
    rng = np.random.default_rng(60)
    days = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 365, 5_000), "D")
    frame = pd.DataFrame({"customer": rng.integers(0, 600, 5_000).astype(str), "day": days})
    result = cohorts(parquet(tmp_path, frame))
    assert all(c.retained[0] == 1.0 for c in result.cohorts)
    assert all(v is None or 0 <= v <= 1 for c in result.cohorts for v in c.retained)
    assert sum(p.new for p in result.periods) == result.customers
