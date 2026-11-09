"""Ready-made KPIs and the best columns to split by, read from column types and roles.

Spec 022, section 1 (rules R-SUG-01 to R-SUG-08).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from quantilo.kpi.definition import Definition, Group, Measure, Ratio, Time
from quantilo.kpi.rules import measure_label, ratio_label
from quantilo.typing.types import AnalysisType

MAX_SUGGESTIONS = 6
MAX_SPLITS = 3
REPEATED = 0.9  # an identifier with fewer distinct values than this share of rows repeats
SPLIT_ROLES = ("country", "region", "city", "status", "gender")
LEVEL_ROLES = ("rating", "percentage", "duration", "age")
NUMBERS: frozenset[str] = frozenset({"numeric_continuous", "numeric_discrete"})
SPLITS: frozenset[str] = frozenset({"categorical_nominal", "categorical_ordinal", "boolean"})
MIN_LEVELS, MAX_LEVELS = 2, 50


class Column(Protocol):
    name: str
    type: AnalysisType
    role: str | None
    summary: dict[str, Any] | None


@dataclass
class Suggestion:
    rule: str
    name: str
    definition: Definition


def _distinct(column: Column) -> int:
    return int((column.summary or {}).get("distinct_count") or 0)


def _valid(column: Column) -> int:
    summary = column.summary or {}
    return int(summary.get("rows") or 0) - int(summary.get("null_count") or 0)


def _missing(column: Column) -> float:
    summary = column.summary or {}
    rows = int(summary.get("rows") or 0)
    return int(summary.get("null_count") or 0) / rows if rows else 1.0


def repeated(column: Column) -> bool:
    """An identifier whose values repeat (customer_id), unlike a row id (R-SUG-04)."""
    valid = _valid(column)
    return column.type == "identifier" and valid > 0 and _distinct(column) < REPEATED * valid


def split_candidates(columns: Sequence[Column]) -> list[str]:
    """Columns worth splitting by, best first (R-SUG-08)."""

    def distance(levels: int) -> int:
        return 0 if 3 <= levels <= 12 else (3 - levels if levels < 3 else levels - 12)

    fitting = [c for c in columns if c.type in SPLITS and MIN_LEVELS <= _distinct(c) <= MAX_LEVELS]
    fitting.sort(key=lambda c: (c.role not in SPLIT_ROLES, distance(_distinct(c)), _missing(c)))
    return [c.name for c in fitting[:MAX_SPLITS]]


def readable(column: Column) -> bool:
    """A column with at least one value readable in its type (a summary of 0 distinct: none)."""
    return column.summary is None or _distinct(column) > 0


def suggest(columns: Sequence[Column]) -> list[Suggestion]:
    columns = [c for c in columns if readable(c)]
    dates = [c.name for c in columns if c.type == "datetime"]
    time = Time(column=dates[0]) if dates else None
    splits = split_candidates(columns)
    groups = [] if time or not splits else [Group(column=splits[0])]
    used: set[str] = set()
    found: list[Suggestion] = []

    def add(rule: str, measures: list[Measure], ratio: Ratio | None = None) -> None:
        definition = Definition(measures=measures, ratio=ratio, time=time, groups=groups)
        name = ratio_label(ratio, measures) if ratio else measure_label(measures[0])
        found.append(Suggestion(rule, name[:1].upper() + name[1:], definition))
        used.update(m.column for m in measures if m.column)

    def free(role: str | tuple[str, ...], limit: int) -> list[Column]:
        roles = (role,) if isinstance(role, str) else role
        return [c for c in columns if c.type in NUMBERS and c.role in roles and c.name not in used][
            :limit
        ]

    add("R-SUG-01", [Measure(id="m1", agg="count")])
    amounts = free("amount", 2)
    for column in amounts:
        add("R-SUG-02", [Measure(id="m1", agg="sum", column=column.name)])
    for column in free("quantity", 1):
        add("R-SUG-03", [Measure(id="m1", agg="sum", column=column.name)])
    ids = [c for c in columns if repeated(c)]
    if ids:
        add("R-SUG-04", [Measure(id="m1", agg="distinct", column=ids[0].name)])
        if amounts:
            add(
                "R-SUG-05",
                [
                    Measure(id="m1", agg="sum", column=amounts[0].name),
                    Measure(id="m2", agg="distinct", column=ids[0].name),
                ],
                Ratio(numerator="m1", denominator="m2"),
            )
    booleans = [c for c in columns if c.type == "boolean" and c.name not in used]
    if booleans:
        add("R-SUG-06", [Measure(id="m1", agg="true_share", column=booleans[0].name)])
    for column in free(LEVEL_ROLES, 1):
        add("R-SUG-07", [Measure(id="m1", agg="mean", column=column.name)])
    return found[:MAX_SUGGESTIONS]
