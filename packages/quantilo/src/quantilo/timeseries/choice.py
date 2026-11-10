"""The time series the page opens on, when the user has not chosen (093, rules R-TS-01 to R-TS-06).

The date of the event, then a measure that adds up (summed), a level (averaged), a measurement
taken once per date (averaged) or, when no column measures anything, the number of rows per
period. Identifiers, codes and coordinates are never a default value. Every choice carries its
reason, shown under the chart; the user's own choices are never changed.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import duckdb

from quantilo import semantics
from quantilo.timeseries import series as ts
from quantilo.typing.parsers import typed_sql

LEVEL_ROLES = frozenset(["unit_price", "rating", "percentage", "duration", "age"])
LEVEL_WORDS = frozenset(
    [
        *("price", "prices", "prix", "tarif", "salary", "salaries", "salaire", "wage", "wages"),
        *("avg", "average", "mean", "moyenne", "moyen", "rate", "balance", "pct", "percent"),
    ]
)
MEASUREMENT_ROWS = 2.0  # R-TS-04: at most this many rows per distinct date: a reading per date
MIN_LEVELS = 30  # R-TS-04: whole numbers with fewer distinct values are levels or codes
NOT_VALUES = frozenset(["year", "month", "weekday", "hour", "latitude", "longitude", "angle"])


@dataclass
class Stats:
    """What the values of a numeric or date column say (one query for all of them)."""

    low: float | None
    high: float | None
    integer: bool  # numbers: whole numbers only; dates: whole minutes only (a sampling grid)
    filled: int


@dataclass
class Choice:
    time: str
    value: str
    aggregate: ts.Aggregate
    reasons: list[str] = field(default_factory=list)  # date and value (R-TS-01, 05, 06)
    aggregate_reason: str = ""  # R-TS-02 to 05, shown when the aggregate is automatic


def stats(
    connection: duckdb.DuckDBPyConnection, source: str, views: Sequence[Any]
) -> dict[str, Stats]:
    if not views:
        return {}
    parts = []
    for view in views:
        if view.type == "datetime":
            moment = f"CAST({typed_sql(view.name, view.parser)} AS TIMESTAMP)"
            parts += [
                "NULL",
                "NULL",
                f"count(*) FILTER (WHERE date_part('second', {moment}) <> 0)",
                f"count({moment})",
            ]
            continue
        value = f"CAST({typed_sql(view.name, view.parser)} AS DOUBLE)"
        parts += [
            f"min({value})",
            f"max({value})",
            f"count(*) FILTER (WHERE {value} <> trunc({value}))",
            f"count({value})",
        ]
    row: tuple[Any, ...] = connection.execute(
        f"SELECT {', '.join(parts)} FROM {source}"
    ).fetchone() or tuple([None] * len(parts))
    found = {}
    for index, view in enumerate(views):
        low, high, fractional, filled = row[4 * index : 4 * index + 4]
        found[view.name] = Stats(
            None if low is None else float(low),
            None if high is None else float(high),
            int(fractional or 0) == 0,
            int(filled or 0),
        )
    return found


def _distinct(view: Any) -> int:
    return int((view.summary or {}).get("distinct_count") or 0)


def _rows(view: Any) -> int:
    summary = view.summary or {}
    return int(summary.get("rows") or 0) - int(summary.get("null_count") or 0)


def choose_time(dates: Sequence[Any]) -> tuple[Any, str]:
    """R-TS-01: the date of the event first; a birth or construction date last."""
    ranked = sorted(enumerate(dates), key=lambda item: (semantics.date_tier(item[1].name), item[0]))
    chosen = ranked[0][1]
    tier = semantics.date_tier(chosen.name)
    others = [view.name for _, view in ranked[1:3]]
    if not others:
        why = "the only date column"
    elif tier == 0:
        why = f"the date of the event itself, preferred to {' and '.join(others)}"
    elif tier == 4:
        why = "only dates of a characteristic (birth, construction): read with care"
    else:
        later = [v.name for _, v in ranked[1:] if semantics.date_tier(v.name) == 4]
        why = (
            f"an event date, preferred to {' and '.join(later)} (a birth or construction date)"
            if later
            else f"the first date column, before {' and '.join(others)}"
        )
    return chosen, f"R-TS-01 · Over {chosen.name}: {why}."


def _skipped(view: Any, stat: Stats | None) -> str | None:
    """R-TS-06: why a number is never the default value."""
    code = semantics.code_by_name(view.name, view.role, view.type)
    if code is not None:
        return code.reason
    if semantics.is_coordinate(view.name, view.role):
        return "a coordinate"
    if view.role in NOT_VALUES or semantics.is_date_part(view.name):
        return f"a {view.role}"
    if stat is not None:
        by_values = semantics.code_by_values(stat.low, stat.high, stat.integer)
        by_values = by_values or semantics.sequence(
            stat.low, stat.high, stat.integer, _distinct(view)
        )
        if by_values is not None:
            return by_values.reason
    return None


def _probability(stat: Stats | None) -> bool:
    return bool(
        stat
        and not stat.integer
        and stat.low is not None
        and stat.high is not None
        and stat.low >= 0
        and stat.high <= 1
    )


def _level(view: Any, stat: Stats | None) -> str | None:
    """R-TS-03: what makes a number a level (averaged), or None."""
    words = set(semantics.tokens(view.name))
    if view.role in LEVEL_ROLES:
        return f"{view.role.replace('_', ' ')}"
    if semantics.is_ratio(view.name):
        return "a ratio (its name says 'per')"
    if view.role in ts.SUM_ROLES and words & LEVEL_WORDS:
        return f"a price or level (its name says {sorted(words & LEVEL_WORDS)[0]})"
    if view.role in ts.SUM_ROLES and _probability(stat):
        return "a probability or share (every value between 0 and 1)"
    return None


def aggregate_for(view: Any, stat: Stats | None) -> tuple[ts.Aggregate, str]:
    """R-TS-02 / R-TS-03: sum what adds up, average a level (a sum of prices means nothing)."""
    level = _level(view, stat)
    if level is not None:
        return (
            "mean",
            f"R-TS-03 · {view.name} is {_an(level)}: averaged per period, as a sum of "
            "prices or rates means nothing.",
        )
    if view.role in ts.SUM_ROLES:
        role = view.role.replace("_", " ")
        return "sum", f"R-TS-02 · {view.name} is {_an(role)}: summed per period."
    return "mean", f"R-TS-04 · {view.name} is a measurement: averaged per period."


def _an(text: str) -> str:
    if text.startswith(("a ", "an ")):
        return text
    return f"{'an' if text[:1] in 'aeiou' else 'a'} {text}"


def choose_value(
    numbers: Sequence[Any], time: Any, found: dict[str, Stats]
) -> tuple[str, ts.Aggregate, str, list[str]]:
    """R-TS-02 to R-TS-06: the value, its aggregate and why, and the columns left out."""
    skipped: dict[str, str] = {}
    usable = []
    for view in numbers:
        why = _skipped(view, found.get(view.name))
        if why is None:
            usable.append(view)
        else:
            skipped[view.name] = why
    notes = []
    if skipped:
        shown = ", ".join(list(skipped)[:4]) + (" …" if len(skipped) > 4 else "")
        first = next(iter(skipped.values()))
        notes.append(
            f"R-TS-06 · {shown}: identifiers, codes or coordinates ({first}), never a value by "
            "default."
        )
    flows = [
        v
        for v in usable
        if v.role in ts.SUM_ROLES
        and _level(v, found.get(v.name)) is None
        and (v.role != "amount" or semantics.is_money(v.name))
    ]
    if flows:
        aggregate, why = aggregate_for(flows[0], found.get(flows[0].name))
        return flows[0].name, aggregate, why, notes
    levels = [v for v in usable if _level(v, found.get(v.name)) is not None]
    if levels:
        aggregate, why = aggregate_for(levels[0], found.get(levels[0].name))
        return levels[0].name, aggregate, why, notes
    measures = [v for v in usable if (stat := found.get(v.name)) is not None and not stat.integer]
    if measures:
        why = (
            f"R-TS-04 · {measures[0].name} has decimal values and no role: a measurement, "
            "averaged per period (whole numbers without a role are often codes or levels)."
        )
        return measures[0].name, "mean", why, notes
    rows, distinct = _rows(time), _distinct(time)
    grid = found.get(time.name)  # readings fall on whole minutes; events carry their seconds
    once = distinct > 0 and rows <= MEASUREMENT_ROWS * distinct and bool(grid and grid.integer)
    readings = [v for v in usable if _distinct(v) >= MIN_LEVELS]
    if once and readings:
        why = (
            f"R-TS-04 · About one row per date ({rows:,} rows, {distinct:,} dates): "
            f"{readings[0].name} is a reading taken at each date, averaged per period."
        )
        return readings[0].name, "mean", why, notes
    counted = max(numbers, key=lambda v: found[v.name].filled if v.name in found else 0)
    why = (
        "R-TS-05 · No column is a measure (amount, quantity, price, level): the series counts "
        f"the rows per period (rows with a {counted.name})."
    )
    return counted.name, "count", why, notes


def choose(
    dates: Sequence[Any],
    numbers: Sequence[Any],
    requested_time: str | None,
    requested_value: str | None,
    found: dict[str, Stats],
) -> Choice:
    """The defaults of the series; the user's choices stay as they are."""
    by_name = {v.name: v for v in [*dates, *numbers]}
    reasons: list[str] = []
    if requested_time in {v.name for v in dates}:
        time = by_name[requested_time]
    else:
        time, why = choose_time(dates)
        reasons.append(why)
    if requested_value in {v.name for v in numbers}:
        view = by_name[requested_value]
        aggregate, why = aggregate_for(view, found.get(view.name))
        return Choice(time.name, view.name, aggregate, reasons, why)
    value, aggregate, why, notes = choose_value(numbers, time, found)
    return Choice(time.name, value, aggregate, [*reasons, *notes], why)
