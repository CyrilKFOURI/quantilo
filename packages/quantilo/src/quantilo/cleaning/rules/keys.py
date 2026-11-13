"""A Basic rule on rows (audit 2026-11-12, N22): B-23, a key that repeats with differences.

A column that names one record each (an identifier by its type or its name: `*id`, `*_link`,
`*_key`, `code`) and is unique on 98 % of the distinct rows, whose values still repeat on rows that
differ elsewhere, but on half their columns at most: the same record entered twice, once updated
(IMDB: 79 films whose votes changed). Rows that share a key and little else are two records (a
product code drawn twice by chance) and are left alone, and so are rows whose date differs (a
house sold twice: two sales of one house, not one sale entered twice).
B-14 removes the exact copies; B-23 says which columns differ and proposes one row per key, the
latest by an update date when there is one, else the most complete. A reference shared by rows (an
order, a customer, a sale: `Invoice`, `id_mutation`) is never unique on 98 % of the rows, so it is
left alone. A second suggestion only marks the rows, for those who want to look first.
"""

from __future__ import annotations

from typing import Any

from quantilo import semantics
from quantilo.cleaning import model as m
from quantilo.cleaning.compile import INTEGERS, as_text, is_temporal, is_text, literal_list
from quantilo.cleaning.rules.basic import Column, Context, _blank, n, q
from quantilo.ingestion.sql import ident

KEY_UNIQUE = 0.98  # B-23: the key is unique on this share of the distinct rows
KEY_ROWS = 20  # … of this many rows at least
LINK_WORDS = frozenset({"link", "links", "url", "uri", "href", "permalink"})
SHARED_ROLES = frozenset({"customer_key", "order_key", "postal_code", "phone", "email"})
SAME_RECORD = 0.5  # … the rows of a repeated key differ on at most this share of the columns
SAME_GROUPS = 0.8  # … in this share of the repeated keys (dirty_traps: SKUs drawn twice by chance)
EVENT_GROUPS = 0.5  # … and no date but an update time differs in this share of them
MAX_KEYS = 3  # key columns proposed at most
DIFFER_SHOWN = 4
EXAMPLES = 3
REPEATED = "repeated key"


def key_columns(context: Context) -> list[Column]:
    """Columns that may name one record each: text or whole numbers, an identifier by type or by
    name, nearly as many distinct values as rows."""
    found = []
    for column in context.columns:
        if not (is_text(column.dtype) or column.dtype in INTEGERS) or column.nonnull < KEY_ROWS:
            continue
        view = column.view
        role = view.role if view is not None else None
        if role in SHARED_ROLES:
            continue
        typed = view is not None and view.type == "identifier"
        code = semantics.code_by_name(column.name, role)
        words = set(semantics.tokens(column.name))
        named = (code is not None and code.kind == "identifier") or "code" in words
        linked = bool(words & LINK_WORDS)
        if not (typed or named or linked):
            continue
        if column.distinct < 0.9 * column.nonnull:  # approximate: a first cheap filter
            continue
        found.append(column)
    return found


def _repeating(context: Context, columns: list[Column]) -> dict[str, int]:
    """Those whose values repeat at all (exact count on the rows as they are): the rows a step
    keeping one row per key would remove, exact copies included."""
    if not columns:
        return {}
    parts = ", ".join(f"count(DISTINCT {ident(c.name)}), count({ident(c.name)})" for c in columns)
    row = context.one(f"SELECT {parts} FROM {context.source}")
    removed = {c.name: int(row[2 * i + 1]) - int(row[2 * i]) for i, c in enumerate(columns)}
    return {name: count for name, count in removed.items() if count > 0}


def _distinct_rows(context: Context) -> str:
    empty = " AND ".join(_blank(c) for c in context.columns)
    return f"(SELECT DISTINCT * FROM {context.source} WHERE NOT ({empty}))"


def _update_column(context: Context, key: Column) -> Column | None:
    """S-06: a date saying when the row was last changed ('updated_at', 'last_modified')."""
    for column in context.columns:
        if column is key or not is_temporal(column.dtype) or not column.nonnull:
            continue
        if semantics.date_tier(column.name) == 3:
            return column
    return None


def _event_date(
    context: Context, differing: list[tuple[str, int]], groups: int, update: Column | None
) -> str | None:
    """A date other than an update time that differs within most repeated keys: each row is an
    event of the same thing (a sale, a visit), not the same row entered twice."""
    for name, count in differing:
        column = next(c for c in context.columns if c.name == name)
        dated = is_temporal(column.dtype) or (
            column.view is not None and column.view.type == "datetime"
        )
        if dated and column is not update and count >= EVENT_GROUPS * groups:
            return name
    return None


def _differences(
    context: Context, key: Column, distinct: str
) -> tuple[int, list[tuple[str, int]], list[str]]:
    """Keys repeated among the distinct rows: how many, the columns that differ within them (with
    the number of keys where they do), and the first keys."""
    others = [c for c in context.columns if c is not key]
    name = ident(key.name)
    differ = ", ".join(
        f"count(DISTINCT coalesce({as_text(c.name, c.dtype)}, chr(0))) > 1" for c in others
    )
    found = context.connection.execute(
        f"SELECT {as_text(key.name, key.dtype)}, {differ or 'true'} FROM {distinct} "
        f"WHERE {name} IS NOT NULL GROUP BY {name} HAVING count(*) > 1 ORDER BY min({name})"
    ).fetchall()
    counts = [0] * len(others)
    alike = 0
    for row in found:
        flags = [bool(flag) for flag in row[1 : 1 + len(others)]]
        alike += sum(flags) <= SAME_RECORD * len(others)
        for index, flag in enumerate(flags):
            counts[index] += flag
    if found and alike < SAME_GROUPS * len(found):
        return 0, [], []  # rows that share a key but little else: two records, not one twice
    columns = sorted(
        ((c.name, k) for c, k in zip(others, counts, strict=True) if k), key=lambda i: -i[1]
    )
    return len(found), columns, [str(row[0]) for row in found]


def _examples(
    context: Context, key: Column, distinct: str, keys: list[str], differing: list[str]
) -> list[tuple[str | None, str | None]]:
    """'tt0499549: num_voted_users 886204 / 886120' → 'one row kept'."""
    if not keys or not differing:
        return []
    shown = [c for c in context.columns if c.name in differing[:DIFFER_SHOWN]]
    texts = ", ".join(as_text(c.name, c.dtype) for c in shown)
    key_text = as_text(key.name, key.dtype)
    rows = context.connection.execute(
        f"SELECT {key_text}, {texts} FROM {distinct} WHERE {key_text} IN "
        f"{literal_list(keys[:EXAMPLES])} ORDER BY 1"
    ).fetchall()
    by_key: dict[str, list[tuple[Any, ...]]] = {}
    for row in rows:
        by_key.setdefault(str(row[0]), []).append(row[1:])
    examples: list[tuple[str | None, str | None]] = []
    for value, group in by_key.items():
        parts = []
        for index, column in enumerate(shown):
            seen = list(dict.fromkeys(row[index] for row in group))
            if len(seen) > 1:
                parts.append(
                    f"{column.name} "
                    + " / ".join(q(None if v is None else str(v), 24) for v in seen[:3])
                )
        examples.append((f"{q(value, 40)}: " + "; ".join(parts[:2]), "one row kept"))
    return examples


def _key_values(context: Context, key: Column, keys: list[str]) -> list[str]:
    """The repeated keys as stored and after the Basic text fixes (a mark found either way)."""
    stages = context.stages.get(key.name)
    found = set(keys)
    if stages is not None:
        from quantilo.cleaning.rules.advanced import _final_of

        finals = {raw: final for final, raws in _final_of(stages).items() for raw in raws}
        found |= {finals.get(k, k) for k in keys}
    return sorted(found)


def b23_repeated_keys(context: Context) -> None:
    if context.rows < KEY_ROWS or len(context.columns) < 2:
        return
    alone = _repeating(context, key_columns(context))
    candidates = [c for c in context.columns if c.name in alone]
    if not candidates:
        return
    distinct = _distinct_rows(context)
    parts = ", ".join(
        f"count({ident(c.name)}), count(DISTINCT {ident(c.name)})" for c in candidates
    )
    row = context.one(f"SELECT count(*), {parts} FROM {distinct}")
    proposed = 0
    for index, key in enumerate(candidates):
        present, unique = int(row[1 + 2 * index]), int(row[2 + 2 * index])
        extra = present - unique
        if extra <= 0 or unique < KEY_UNIQUE * present or proposed >= MAX_KEYS:
            continue
        groups, differing, keys = _differences(context, key, distinct)
        if not groups or not differing:
            continue
        update = _update_column(context, key)
        event = _event_date(context, differing, groups, update)
        if event is not None:
            continue  # kc_house: a house sold twice, two sales (its `date` differs), no duplicate
        proposed += 1
        listed = ", ".join(f"{name} in {n(k)}" for name, k in differing[:DIFFER_SHOWN])
        more = len(differing) - DIFFER_SHOWN
        listed += f" and {more} more columns" if more > 0 else ""
        kept = (
            f"the latest by {update.name} kept"
            if update is not None
            else "the most complete kept (fewest empty cells, then the first)"
        )
        reason = (
            f"{n(groups)} {key.name} values repeat on {n(groups + extra)} rows that differ "
            f"elsewhere ({listed}), while {key.name} is unique on {unique / present:.1%} of the "
            f"rows: the same record entered twice, one of them updated? One row per {key.name}, "
            f"{kept}; {n(extra)} rows removed."
        )
        copies = alone[key.name] - extra
        if copies > 0:
            reason += (
                f" The {rows_word(copies)} that are exact copies are B-14's: without it, this "
                f"step removes {n(alone[key.name])} rows."
            )
        reason += " Two real records can share a key by mistake too. Check your data."
        examples = _examples(context, key, distinct, keys, [name for name, _ in differing])
        context.add(
            "B-23",
            "S",
            key,
            f"Keep one row per {key.name} ({rows_word(extra)} with differences)",
            reason,
            m.DropDuplicates(
                id="s0",
                keys=[key.name],
                keep="latest" if update is not None else "most_complete",
                order_by=update.name if update is not None else None,
            ),
            rows=extra,
            examples=examples,
        )
        values = _key_values(context, key, keys)
        if len(values) > m.MAX_VALUES:
            continue
        flag = f"{key.name}_repeated"
        taken = {c.name for c in context.columns}
        while flag in taken:
            flag += "_"
        context.add(
            "B-23",
            "S",
            key,
            f"Mark the {n(groups + extra)} rows whose {key.name} repeats",
            f"The same {n(groups)} repeated {key.name} values, only marked: a column {flag} says "
            f"{q(REPEATED)} on their {n(groups + extra)} rows, to filter, pin or compare them "
            "before deciding. Nothing is removed. Check your data.",
            m.ConditionalColumn(
                id="s0",
                to=flag,
                cases=[
                    m.Case(
                        conditions=[m.RowCondition(column=key.name, op="in", values=values)],
                        value=REPEATED,
                    )
                ],
                otherwise=None,
            ),
            cells=groups + extra,
            examples=[(v, REPEATED) for v in keys[:EXAMPLES]],
            key="mark",
        )


def rows_word(count: int) -> str:
    return f"{n(count)} row" + ("" if count == 1 else "s")
