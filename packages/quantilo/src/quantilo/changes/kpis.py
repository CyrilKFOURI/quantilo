"""Saved KPIs before and after: the headline value, and the periods both versions hold.

A past period whose value changes (September in both files, not the same total) points at a
correction made back in the source: it is counted apart (`revised`).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import duckdb

from quantilo.catalog import ColumnView
from quantilo.changes.snapshot import KpiPoint, KpiValue
from quantilo.kpi.compute import KpiResult, compute
from quantilo.kpi.definition import RATIO_ID, Definition

REVISED = 1e-6  # relative change below which a period is the same


@dataclass
class KpiPeriodChange:
    label: str
    before: float | None
    after: float | None
    change: float | None  # relative (0.05 = +5 %)


@dataclass
class KpiChange:
    name: str
    measure: str
    status: Literal["ok", "broken", "unavailable"]
    before: float | None = None
    after: float | None = None
    change: float | None = None
    problem: str | None = None  # why a value is missing ("column amount is gone")
    common: list[KpiPeriodChange] = field(default_factory=list)  # periods in both versions
    revised: int = 0  # common periods whose value changed


def relative(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return (after - before) / abs(before)


def kpi_value(name: str, result: KpiResult) -> KpiValue:
    """The headline measure (the ratio when there is one), overall and per period."""
    measure = next(
        (m for m in result.measures if m.id == RATIO_ID),
        result.measures[0] if result.measures else None,
    )
    if measure is None:
        return KpiValue(name, "", None)
    line = next(
        (s for s in result.series if s.measure == measure.id and s.g1 is None and s.g2 is None),
        None,
    )
    if line is None:
        return KpiValue(name, measure.label, None)
    labels = [p.label for p in result.periods]
    points = (
        [KpiPoint(label, value) for label, value in zip(labels, line.values, strict=True)]
        if len(labels) == len(line.values)
        else []
    )
    return KpiValue(name, measure.label, line.overall, points)


def compute_value(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    views: Sequence[ColumnView],
    name: str,
    definition: Definition,
) -> KpiValue:
    used = [v for v in views if not v.excluded]
    return kpi_value(name, compute(connection, source, used, definition))


def compare_kpi(
    name: str,
    before: KpiValue | None,
    after: KpiValue | None,
    problem: str | None = None,
    broken: bool = False,
) -> KpiChange:
    if after is None or before is None:
        known = after or before
        return KpiChange(
            name,
            known.measure if known else "",
            "broken" if broken else "unavailable",
            before.value if before else None,
            after.value if after else None,
            problem=problem,
        )
    old = {p.label: p.value for p in before.periods}
    common = [
        KpiPeriodChange(p.label, old[p.label], p.value, relative(old[p.label], p.value))
        for p in after.periods
        if p.label in old
    ]
    revised = sum(
        1
        for c in common
        if c.before is not None
        and c.after is not None
        and not math.isclose(c.before, c.after, rel_tol=REVISED, abs_tol=1e-12)
    )
    return KpiChange(
        name,
        after.measure,
        "ok",
        before.value,
        after.value,
        relative(before.value, after.value),
        common=common,
        revised=revised,
    )
