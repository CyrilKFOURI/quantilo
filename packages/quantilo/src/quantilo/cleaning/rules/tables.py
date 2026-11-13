"""Rules between the tables of a project (audit 2026-11-12, N32), computed with Clean's Advanced
layer when the dataset is related in a project (088, 089). The relations are the ones the user
accepted on the Model page; each one is checked exactly on every row (`model.check`, R-JOIN-06).

- T-01 (information): rows of this table whose key has no match in the table it points to
  ("37 rows of orders point to a customerID missing from customers"). Never removed: a join keeps
  them with empty columns (R-JOIN-01); the suggestion names their values, so the page shows the
  rows on the Data page, where they can be pinned (099).
- T-01:near (decision): an orphan one letter away from exactly one key ('ALFKJ' for 'ALFKI',
  R-REL-15), the readable fix R-REL-15 announced: a `map_values`, never applied on its own.
- T-02 (information): a key that repeats in this table while another table points to it: its
  rows would match twice, so the build refuses the relation (R-BLD-03).

Every reading is an exact SQL count; nothing is joined by guess (C-49).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb

from quantilo.cleaning import catalogue
from quantilo.cleaning import model as m
from quantilo.cleaning.suggestion import CleanSuggestion, Example
from quantilo.ingestion.sql import ident
from quantilo.model.build import LinkCheck, ModelError, check
from quantilo.model.keys import Match, TableInput, key_expression, link_kinds, source

ORDER = {"T-01": 400, "T-02": 410}
SHOWN = 5  # values named in a reason
MAX_WHERE = 1_000  # values of a Data filter (filters/model.py MAX_VALUES)
MAX_MAP = 500  # orphan values fixed by one step at most


@dataclass
class TableLink:
    """An accepted relation of a project, with its two tables as Clean reads them."""

    child: TableInput
    child_columns: list[str]
    parent: TableInput
    parent_columns: list[str]
    match: Match = "exact"
    relation_id: str | None = None
    project: str | None = None  # the project's name, for the reason


def _n(count: int) -> str:
    return f"{count:,}"


def _plural(count: int, noun: str) -> str:
    return f"{_n(count)} {noun}" + ("" if count == 1 else "s")


def _shown(value: str) -> str:
    return value.replace(chr(31), " · ")  # a key of several columns (keys.key_expression)


def _key(columns: list[str]) -> str:
    return " + ".join(columns)


def _raw_values(
    connection: duckdb.DuckDBPyConnection,
    table: TableInput,
    columns: list[str],
    other: TableInput,
    other_columns: list[str],
    match: Match,
    orphans: bool,
) -> list[Any] | None:
    """The stored values of `table`'s one key column whose key is missing from `other`
    (`orphans`) or repeats in `table` itself; None for a key of several columns or too many."""
    if len(columns) != 1:
        return None
    kinds = link_kinds(table, columns, other, other_columns)
    if not isinstance(kinds, tuple):
        return None
    mine = key_expression(table, columns, kinds[0], match)
    if orphans:
        theirs = key_expression(other, other_columns, kinds[1], match)
        where = (
            f"{mine} IS NOT NULL AND {mine} NOT IN "
            f"(SELECT {theirs} FROM {source(other)} WHERE {theirs} IS NOT NULL)"
        )
    else:
        where = (
            f"{mine} IN (SELECT {mine} FROM {source(table)} WHERE {mine} IS NOT NULL "
            f"GROUP BY 1 HAVING count(*) > 1)"
        )
    column = ident(columns[0])
    found = connection.execute(
        f"SELECT DISTINCT {column} FROM {source(table)} WHERE {where} ORDER BY 1 "
        f"LIMIT {MAX_WHERE + 1}"
    ).fetchall()
    values = [row[0] for row in found if row[0] is not None]
    if not values or len(values) > MAX_WHERE:
        return None
    return [str(v) for v in values]  # RowCondition compares as text, numbers as numbers


def _where(column: str, values: list[Any] | None) -> list[m.RowCondition]:
    if not values:
        return []
    return [m.RowCondition(column=column, op="in", values=values)]


def _suggestion(
    rule: str,
    mode: str,
    table: TableInput,
    columns: list[str],
    title: str,
    reason: str,
    examples: list[tuple[str, str]],
    key: str,
    step: Any = None,
    cells: int = 0,
    where: list[m.RowCondition] | None = None,
) -> CleanSuggestion:
    view = {v.name: v for v in table.columns}
    position = min((view[c].position for c in columns if c in view), default=0)
    if step is not None:
        step.origin = "suggested"
        step.rule = rule
    found = CleanSuggestion(
        id=f"{rule}:{'+'.join(columns)}:{key}",
        rule=rule,
        layer="advanced",
        mode=mode,  # type: ignore[arg-type]
        impact="values" if step is not None else "rows",
        risk="review",
        order=ORDER[rule] * 10_000 + position,
        title=title,
        reason=reason,
        columns=columns,
        cells=cells,
        examples=[Example(before=b, after=a) for b, a in examples[:SHOWN]],
        step=step,
        where=where or [],
    )
    if step is not None:
        step.suggestion = found.id
    return found


def _orphans(
    connection: duckdb.DuckDBPyConnection, link: TableLink, found: LinkCheck
) -> list[CleanSuggestion]:
    child, parent, columns = link.child, link.parent, link.child_columns
    key = _key(columns)
    values = ", ".join(f"'{_shown(e.value)}' ({_plural(e.rows, 'row')})" for e in found.orphans)
    raw = _raw_values(
        connection, child, columns, parent, link.parent_columns, link.match, orphans=True
    )
    count = f" ({_plural(len(raw), 'value')} in all)" if raw else ""
    misses = len(found.near_misses)
    verb = "is" if misses == 1 else "are"
    near = (
        f" {_plural(misses, 'value')} {verb} one letter away from a key: see the fix below."
        if misses
        else ""
    )
    where = _where(columns[0], raw) if len(columns) == 1 else []
    see = (
        " Open them on the Data page to look at them or pin them."
        if where
        else " The examples name the first of them."
    )
    reason = (
        f"{_plural(found.orphan_rows, 'row')} of {child.name} hold a {key} that {parent.name} does "
        f"not have: {values}{count}. They are kept: a join keeps them with the {parent.name} "
        f"columns empty (R-JOIN-01), and nothing removes them unless you add a step.{near}{see} "
        "Check your data."
    )
    title = (
        f"{_plural(found.orphan_rows, 'row')} of {child.name} point to a {key} missing from "
        f"{parent.name}"
    )
    suggestions = [
        _suggestion(
            "T-01",
            "I",
            child,
            columns,
            title,
            reason,
            [(f"{key} = '{_shown(e.value)}'", f"not in {parent.name}") for e in found.orphans],
            f"{parent.id}",
            where=where,
        )
    ]
    if found.near_misses and len(columns) == 1:
        fix = _near_fix(connection, link, found)
        if fix is not None:
            suggestions.append(fix)
    return suggestions


def _near_fix(
    connection: duckdb.DuckDBPyConnection, link: TableLink, found: LinkCheck
) -> CleanSuggestion | None:
    """T-01:near: each orphan one letter away from exactly one key → that key (R-REL-15)."""
    child, parent, column = link.child, link.parent, link.child_columns[0]
    kinds = link_kinds(child, link.child_columns, parent, link.parent_columns)
    if not isinstance(kinds, tuple):
        return None
    pairs = dict(found.near_misses[:MAX_MAP])
    mine = key_expression(child, link.child_columns, kinds[0], link.match)
    rows = connection.execute(
        f"SELECT {ident(column)}, {mine} AS k, count(*) FROM {source(child)} "
        f"WHERE {mine} IN (SELECT unnest(?)) GROUP BY 1, 2 ORDER BY 3 DESC, 1",
        [list(pairs)],
    ).fetchall()
    mapping: dict[Any, Any] = {}
    cells = 0
    shown = []
    for raw, canonical, count in rows:
        if raw is None or canonical not in pairs:
            continue
        mapping[str(raw)] = pairs[canonical]
        cells += int(count)
        shown.append((str(raw), pairs[canonical], int(count)))
    if not mapping:
        return None
    listed = ", ".join(f"'{a}' ({_plural(c, 'row')}) → '{b}'" for a, b, c in shown[:SHOWN])
    reason = (
        f"{listed}: missing from {parent.name}, and one letter away from exactly one of its keys "
        f"(R-REL-15). Replaced if you tick it; a key is never matched by guess. Check your data."
    )
    title = (
        f"Fix {_plural(len(mapping), 'key')} of {column} one letter away from a {parent.name} key"
    )
    return _suggestion(
        "T-01",
        "S",
        child,
        [column],
        title,
        reason,
        [(a, b) for a, b, _ in shown],
        f"{parent.id}:near",
        step=m.MapValues(id="s0", column=column, map=mapping),
        cells=cells,
    )


def _duplicates(
    connection: duckdb.DuckDBPyConnection, link: TableLink, found: LinkCheck
) -> CleanSuggestion:
    child, parent, columns = link.child, link.parent, link.parent_columns
    key = _key(columns)
    values = ", ".join(f"'{_shown(e.value)}' ({_plural(e.rows, 'row')})" for e in found.duplicates)
    raw = _raw_values(
        connection, parent, columns, child, link.child_columns, link.match, orphans=False
    )
    where = _where(columns[0], raw) if len(columns) == 1 else []
    repeat = "repeats" if found.duplicate_keys == 1 else "repeat"
    reason = (
        f"{_plural(found.duplicate_keys, 'value')} of {key} {repeat} in {parent.name} ({values}), "
        f"while {child.name} points to {parent.name} by it: each of its rows would match several "
        f"rows, so the project's build refuses this relation (R-BLD-03). Keep one row per {key} "
        "(Remove duplicate rows) once you know which one is right. Check your data."
    )
    title = f"{_plural(found.duplicate_keys, 'key')} of {key} {repeat} in {parent.name}"
    return _suggestion(
        "T-02",
        "I",
        parent,
        columns,
        title,
        reason,
        [(f"{key} = '{_shown(e.value)}'", f"{_plural(e.rows, 'row')}") for e in found.duplicates],
        f"{child.id}",
        where=where,
    )


def suggest_tables(
    connection: duckdb.DuckDBPyConnection, dataset_id: str, links: list[TableLink]
) -> list[CleanSuggestion]:
    """T-01 for the relations where `dataset_id` points to another table, T-02 for those where
    another table points to it. Step ids are left to the caller (numbered after the others)."""
    found: list[CleanSuggestion] = []
    for link in links:
        if dataset_id not in (link.child.id, link.parent.id):
            continue
        try:
            counts = check(
                connection,
                link.child,
                link.child_columns,
                link.parent,
                link.parent_columns,
                link.match,
            )
        except (ModelError, duckdb.Error):  # refused or broken: the Model page says why
            continue
        if link.child.id == dataset_id and counts.orphan_rows:
            found += _orphans(connection, link, counts)
        if link.parent.id == dataset_id and counts.duplicate_keys:
            found.append(_duplicates(connection, link, counts))
    return catalogue.annotate(sorted(found, key=lambda s: s.order))
