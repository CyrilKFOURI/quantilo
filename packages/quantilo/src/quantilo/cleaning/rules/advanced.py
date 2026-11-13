"""The Advanced layer (097, study § 2.2): computed only when asked (C-16), every suggestion in
mode S or I with risk "review" (C-17). Bounded on 5 M rows: distinct values (≤ 50,000) for the
spellings, ≤ 2,000 of them for the neighbours, a reproducible sample of 200,000 rows for the
dependencies, the units and the totals, one SQL pass for the dates, the ages and the
near-duplicates.

- A-01, A-02, A-03: spellings of one value (fingerprint, 2-letter fingerprint, a rare one close to
  a frequent one by edit distance or sound), merged into the most frequent (`map_values`).
- A-05: a minority of a number column 100 or 1,000 times off (cents, grams, 0–1 percentages).
- A-10: a column that determines another (city → country) except on a few rows.
- A-11, A-12, A-13: dates in the wrong order, totals that are not their parts, ages that contradict
  the birth date. Each one proposes the conditional step (C-45, `set_where`, `swap_values`).
- A-15: rows equal once normalized, apart from an identifier or a free text.
- B1 (rules/shapes.py, rules/distributions.py, rules/linkage.py): A-04 shapes of codes, phones and
  e-mails; A-14 dates written another way (day and month, two-digit years, time zones); A-06
  outliers, A-07 floating-point noise, A-08 Benford, A-09 last digits; A-16 near-duplicates by
  Fellegi–Sunter; A-17 rows mostly empty.
- F-02 (rules/formats.py, audit 2026-11-12 N21): identifiers whose check digit fails (SIREN, SIRET,
  French VAT, IBAN, EAN / UPC, ISBN), flagged, never corrected; F-01 (N20) phones in the
  international form; F-04 (N24) countries written one way; F-05 (N23) quantities with
  a unit read as numbers.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import duckdb
import numpy as np

from quantilo import semantics
from quantilo.catalog import ColumnView
from quantilo.cleaning import formula as fx
from quantilo.cleaning import model as m
from quantilo.cleaning import phonetic as ph
from quantilo.cleaning import text as tx
from quantilo.cleaning.compile import (
    INTEGERS,
    _date_expr,
    as_text,
    is_number,
    is_temporal,
    is_text,
)
from quantilo.cleaning.rules import basic
from quantilo.cleaning.rules.basic import Column, Context, TextStages, n, q
from quantilo.cleaning.suggestion import CleanSuggestion, Example, MergeGroup, MergeValue, Mode
from quantilo.ingestion.sql import ident, lit
from quantilo.typing.parsers import clean_text, number_sql

SAMPLE_ROWS = 200_000
SEED = 97
LISTED = 10  # rows shown in an information's examples
NEIGHBOURS = 2_000  # A-03: distinct spellings compared pairwise at most
RATIO = 10  # C-21: the rare spelling has at most a tenth of the rows of the frequent one
MIN_LENGTH = 4
MAX_LENGTH = 60  # A-03: longer values are sentences or lists, compared by their keys only
LONG_WORD = 7  # C-25: one letter more or less is a typo from this length
FD_SHARE = 0.99  # A-10: A determines B on this share of rows (C-27)
FD_GAIN = 0.8  # … and knowing A removes this share of the errors of guessing B alone
FD_KEYS = 10  # … with at least this many values of A
FD_GROUP = 3  # … of this many rows each on average (pairs of rows agree by chance)
FD_COLUMNS = 14
FD_MAX = 10
FD_SECOND_ROWS = 5  # C-27: a second answer this frequent for one value of A is real
FD_SECOND_SHARE = 0.1  # (Dominique: M and F), not an error
FD_SECOND_NAMES = 0.02  # N15: from a first name, a second answer from 2 % of its rows (Claude: F)
FD_MAJORITY = 3  # N15: a value of A is corrected only if its majority has this many rows
FD_EXPLAINED = 3  # N15: rows of A = a and C = c that all say b′: a real exception, not a typo
FD_COPY = 0.95  # … a column equal to B on this share of rows is a copy of B, no evidence
FD_CONTEXT = 24  # … C among this many text columns (the finest first)
FD_CONTEXT_ROWS = 50_000  # … read on this many rows at most
UNIT_FACTORS = (100, 1000)
UNIT_TOLERANCE = 0.15  # log10: the two groups' quantiles are this close once rescaled
UNIT_SHARE = (0.01, 0.30)
UNIT_MIN_VALUES = 10
UNIT_SPLITS = 30  # the widest gaps between distinct values tried as the split
# N17: in a percentage, a minority of small values a hundred times smaller is a real low tail
# (scores of 0.04 %) unless a gap of this many powers of ten parts it from the others
UNIT_PERCENT_GAP = 0.5
PERCENT_WORDS = {"pct", "percent", "percentage", "pourcentage", "pourcent", "100g"}
ORDER_SHARE = 0.95  # A-11
TOTAL_SHARE = 0.95  # A-12
TOTAL_COLUMNS = 10
TOTAL_RELATIVE = 0.01
TOTAL_ABSOLUTE = 0.011
TOTAL_PROBE = 1_000  # rows a relation is first tried on
AGE_SHARE = 0.9  # A-13
NEAR_SHARE = 0.05  # A-15: normalized duplicates are at most this share of the rows
NEAR_KEYS = 3
NEAR_ANCHOR = 20  # A-15: a key column naming the record (a date, a customer…) has this many values
SEPARATORS = r"[\s.,;:'\"`_/\\()\[\]-]+"  # A-15: what normalizing removes
LIST_SHARE = 0.3  # N13: rows holding a separator for a column of lists
LIST_VALUES = 5  # … among this many distinct values at least
LIST_DISTINCT = 0.1  # … or this share of the distinct values, when their items stand alone
LIST_REPEAT = 0.5  # … and this share of item occurrences seen in two values or more
FREE_WORDS = {"note", "notes", "comment", "comments", "remark", "remarks", "message", "feedback"}
# N44 (audit 2026-11-12): bounds on a wide file. A-01 to A-03 on this many category text columns
# at most, those with the fewest distinct values per row; the checks not started after this
# many seconds are left out. Either way a note (A-00) names what was not checked.
SPELLING_COLUMNS = 60
TIME_BUDGET = 15.0
ORDER = {
    "A-00": 100,  # N44: what a wide file left unchecked, first
    "A-14": 105,  # C-46: dates rewritten before B-11 reads them (110)
    "A-01": 200,
    "A-02": 210,
    "A-03": 220,
    "F-01": 224,  # N20: phones in the international form, before A-04's layout (rules/formats.py)
    "A-21": 222,  # N30: business synonyms (rules/synonyms.py), after the spellings
    "A-04": 225,
    "F-02": 226,  # N21: identifiers with a check digit (rules/formats.py)
    "F-04": 227,  # N24: countries written several ways (rules/formats.py)
    "F-05": 228,  # N23: quantities with a unit read as numbers (rules/formats.py)
    "A-05": 230,
    "A-07": 232,
    "A-06": 235,
    "A-15": 240,
    "A-16": 245,
    "A-20": 247,  # N36: one entity written several ways, numbered (rules/entities.py)
    "A-17": 250,
    "A-18": 255,  # N33: a value of a group repeated on its rows
    "A-19": 256,  # N34: two columns that say the same thing
    "A-10": 300,
    "A-11": 310,
    "A-12": 320,
    "A-13": 330,
    "A-08": 340,
    "A-09": 350,
}
# 096 (shared with the Basic layer): how each rule changes the dataset; by step when a rule
# proposes two steps (A-06: a flag column, or the values capped).
IMPACT: dict[str, m.Impact] = {
    "A-00": "values",  # N44: a note, no step
    "A-01": "values",
    "A-02": "values",
    "A-03": "values",
    "A-04": "values",
    "A-05": "values",
    "A-06": "values",
    "A-07": "format",  # 0.30000000000000004 → 0.3: the number as written, nothing lost
    "A-08": "values",
    "A-09": "values",
    "A-10": "values",
    "A-11": "values",
    "A-12": "values",
    "A-13": "values",
    "A-14": "values",
    "A-15": "rows",
    "A-16": "rows",
    "A-20": "columns",  # N36: an entity number, the names unchanged
    "A-17": "rows",
    "A-18": "values",  # N33: an information, no step
    "A-19": "columns",  # N34: a copy removed, or an information
    "F-02": "columns",  # N21: a flag column (the IBAN's electronic form says "format")
    "F-04": "values",  # N24: a country written one way (variants merged)
    "F-01": "format",  # N20: the same number, written in the international form
    "F-05": "columns",  # N23: new columns of numbers (grams, litres, metres) and units
    "A-21": "values",  # N30: business synonyms merged
}
STEP_IMPACT: dict[tuple[str, str], m.Impact] = {
    ("A-06", "conditional_column"): "columns",
}
FIX_CASES = 200  # A-10: values of A corrected by one step at most
NULL_TOKENS = basic.SURE_TOKENS | basic.WEAK_TOKENS
START_WORDS = {"start", "begin", "beginning", "from", "open", "opened", "checkin", "pickup"}
START_WORDS |= {"arrival", "debut", "hire", "hired", "entry", "admission", "admitted"}
# N16: the words of a record's first date (nyc_311 created_date, a filing, an order)
START_WORDS |= {"created", "received", "filed", "submitted", "issued", "ordered"}
END_WORDS = {"end", "ending", "finish", "finished", "until", "close", "closed", "checkout"}
END_WORDS |= {"dropoff", "departure", "fin", "termination", "terminated", "exit", "discharge"}
END_WORDS |= {"discharged", "expiry", "expiration", "expires", "due"}
END_WORDS |= {"resolved", "shipped", "delivered"}  # N16
SWAP_DAYS = 60  # N16: two dates swapped are this close; further apart, one of them is wrong
BIRTH_WORDS = {"birth", "birthday", "dob", "born", "naissance", "birthdate"}
SINGLE_TOKEN = re.compile(r"^(?:[a-z]|\d+)$")


def _number_text(value: float) -> str:
    rounded = round(value, 9)
    if rounded == int(rounded) and abs(rounded) < 1e15:
        return str(int(rounded))
    return format(rounded, ".12g")


@dataclass
class Unit:
    """The spellings of one value up to case and accents (B-03's group)."""

    key: str  # folded
    display: str  # the most frequent spelling (the accented one kept, CLN-5)
    rows: int
    spellings: set[str] = field(default_factory=set)  # final and raw values

    @property
    def length(self) -> int:
        return len(self.key)


@dataclass
class Advanced:
    context: Context
    rows_source: str  # read_parquet(…, file_row_number = true)
    basic_found: list[CleanSuggestion]
    found: list[CleanSuggestion] = field(default_factory=list)
    merges: dict[str, dict[str, str]] = field(default_factory=dict)  # column → final → canonical
    # column → (raw → after the Basic text fixes, folded key → its display): `variants`
    spellings: dict[str, tuple[dict[str, str], dict[str, str]]] = field(default_factory=dict)
    sampled: bool = False
    sample_rows: int = 0
    # N44: when the checks started, and what the bounds left out (column names, check names)
    started: float = field(default_factory=time.monotonic)
    unchecked: list[str] = field(default_factory=list)  # beyond SPELLING_COLUMNS
    late_columns: list[str] = field(default_factory=list)  # A-01 to A-03, out of time
    late_checks: list[str] = field(default_factory=list)

    def out_of_time(self) -> bool:
        return time.monotonic() - self.started > TIME_BUDGET

    def add(
        self,
        rule: str,
        mode: Mode,
        columns: list[Column],
        title: str,
        reason: str,
        step: Any = None,
        cells: int = 0,
        rows: int = 0,
        examples: Sequence[tuple[str | None, str | None]] | None = None,
        key: str | None = None,
        impact: m.Impact | None = None,
    ) -> None:
        if step is not None:
            step.origin = "suggested"
            step.rule = rule
        if impact is None:
            impact = STEP_IMPACT.get((rule, step.op if step is not None else "")) or IMPACT[rule]
        target = "+".join(c.name for c in columns) if columns else "rows"
        suggestion = CleanSuggestion(
            id=f"{rule}:{target}" + (f":{key}" if key else ""),
            rule=rule,
            layer="advanced",
            mode=mode,
            order=ORDER[rule] * 10_000 + (columns[0].position if columns else 0),
            title=title,
            reason=reason,
            columns=[c.name for c in columns],
            cells=cells,
            rows=rows,
            examples=[Example(before=b, after=a) for b, a in (examples or [])[:LISTED]],
            step=step,
            impact=impact,
            risk="review",
        )
        if step is not None:
            step.suggestion = suggestion.id
        self.found.append(suggestion)

    def column(self, name: str) -> Column | None:
        return next((c for c in self.context.columns if c.name == name), None)


def _id() -> str:
    return "s0"  # numbered with the others (suggest.py)


def row_label(row: int) -> str:
    return f"row {row + 1:,}"  # decision 3: from 1, the first data row


def rows_list(rows: list[int], total: int) -> str:
    shown = ", ".join(f"{r + 1:,}" for r in rows[:5])
    more = f" and {n(total - min(len(rows), 5))} more" if total > 5 else ""
    return f"rows {shown}{more}"


# --- Spellings (A-01, A-02, A-03) ----------------------------------------------------------------


def _final_of(stages: TextStages) -> dict[str, list[str]]:
    """Each value after the Basic text rules → the raw values that give it."""
    raws: dict[str, list[str]] = defaultdict(list)
    for raw, _count in stages.raw:
        value = raw
        for mapping in (stages.invisible, stages.spaces, stages.encoding, stages.html):
            value = mapping.get(value, value)
        raws[value].append(raw)
    return raws


def units_of(stages: TextStages) -> list[Unit]:
    """B-03's groups: spellings equal up to case and accents, null words left out."""
    raws = _final_of(stages)
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for value, count in stages.final:
        if value.strip() and value.strip().lower() not in NULL_TOKENS:
            groups[ph.fold(value)].append((value, count))
    units = []
    for key, spellings in groups.items():
        ordered = basic._accent_first(sorted(spellings, key=lambda item: (-item[1], item[0])))
        names = {v for v, _ in spellings} | {r for v, _ in spellings for r in raws.get(v, [])}
        units.append(Unit(key, ordered[0][0], sum(c for _, c in spellings), names))
    return sorted(units, key=lambda u: (-u.rows, u.key))


class Groups:
    """Union-find over units; a group's canonical value is its most frequent unit (C-20)."""

    def __init__(self, units: list[Unit]) -> None:
        self.units = units
        self.parent = list(range(len(units)))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # units are sorted by rows: the smaller index (more rows) stays the root
            self.parent[max(root_a, root_b)] = min(root_a, root_b)

    def members(self) -> dict[int, list[int]]:
        found: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.units)):
            found[self.find(index)].append(index)
        return found


def _same_numbers(a: str, b: str) -> bool:
    return ph.digits(a) == ph.digits(b)


def _token_swap(a: str, b: str) -> bool:
    """C-22: the values differ in one short word ('model s' / 'model x', 'size 2' / 'size 3')."""
    left, right = a.split(), b.split()
    if len(left) != len(right):
        return False
    different = [(x, y) for x, y in zip(left, right, strict=True) if x != y]
    return len(different) == 1 and any(SINGLE_TOKEN.match(w) for w in different[0])


def _spelling_columns(advanced: Advanced) -> list[tuple[Column, TextStages]]:
    found = []
    for column, stages in advanced.context.texts():
        if not basic._is_category(column) or basic._converted(column) or basic._is_code(stages):
            continue
        found.append((column, stages))
    if len(found) <= SPELLING_COLUMNS:
        return found
    # N44: the most categorical (fewest distinct values per row) first, checked in their order
    ranked = sorted(found, key=lambda i: (i[0].distinct / max(i[0].nonnull, 1), i[0].position))
    kept = {column.name for column, _ in ranked[:SPELLING_COLUMNS]}
    advanced.unchecked += [column.name for column, _ in found if column.name not in kept]
    return [(column, stages) for column, stages in found if column.name in kept]


def _merge_suggestion(
    advanced: Advanced,
    rule: str,
    column: Column,
    groups: Groups,
    roots: dict[int, list[int]],
    how: str,
    person: bool,
) -> None:
    """One `map_values` for the groups found by a method: every spelling of a non-canonical
    member (raw and normalized, C-23) → the canonical display."""
    mapping: dict[str, str] = {}
    weights: dict[str, int] = {}
    pairs: list[tuple[Unit, list[Unit]]] = []
    for root, members in roots.items():
        canonical = groups.units[root]
        others = [groups.units[i] for i in members if i != root]
        if not others:
            continue
        pairs.append((canonical, others))
        for unit in others:
            for spelling in unit.spellings:
                if spelling != canonical.display:
                    mapping[spelling] = canonical.display
                    weights[spelling] = unit.rows
            advanced.merges.setdefault(column.name, {})[unit.key] = canonical.key
    if not mapping:
        return
    cells = sum(unit.rows for _, others in pairs for unit in others)
    pairs.sort(key=lambda pair: -sum(u.rows for u in pair[1]))
    canonical, others = pairs[0]
    shown = ", ".join(f"{q(u.display)} ({n(u.rows)} rows)" for u in others[:3])
    title = (
        f"Merge {len(others) + 1} spellings of {q(canonical.display)}"
        if len(pairs) == 1
        else f"Merge near spellings in {column.name} ({len(pairs)} groups)"
    )
    reason = (
        f"{shown} → {q(canonical.display)} ({n(canonical.rows)} rows, the most frequent): {how}. "
        f"{n(cells)} cells in {len(pairs)} groups. Check each group before merging."
    )
    if person:
        reason += " Names of people: two spellings may be two people (CLN-5)."
    examples = [
        (unit.display, canon.display)
        for canon, members in pairs
        for unit in sorted(members, key=lambda u: -u.rows)
    ]
    step_map = basic._capped(mapping, weights)
    advanced.add(
        rule,
        "S",
        [column],
        title,
        reason,
        m.MapValues(id=_id(), column=column.name, map=step_map),
        cells=cells,
        examples=examples,
    )
    advanced.found[-1].clusters = _clusters(pairs, step_map)


def _clusters(pairs: list[tuple[Unit, list[Unit]]], step_map: dict[str, Any]) -> list[MergeGroup]:
    """N38: each group of a merge with its values and their rows, the most frequent first; only
    the spellings the (capped) map holds, so the page's review writes back exactly that map."""
    clusters = []
    for canonical, others in pairs:
        values = [
            MergeValue(
                value=unit.display,
                rows=unit.rows,
                spellings=sorted(s for s in unit.spellings if s in step_map),
            )
            for unit in sorted(others, key=lambda u: (-u.rows, u.display))
        ]
        values = [value for value in values if value.spellings]
        if values:
            clusters.append(
                MergeGroup(target=canonical.display, rows=canonical.rows, values=values)
            )
    return clusters


def spellings(advanced: Advanced) -> None:
    for column, stages in _spelling_columns(advanced):
        if advanced.out_of_time():  # N44
            advanced.late_columns.append(column.name)
            continue
        units = units_of(stages)
        if len(units) < 2:
            continue
        person = basic._person_names(column)
        listed = not person and _is_list(units)
        groups = Groups(units)
        # A-01: fingerprint (word order, punctuation); a list: the set of its items' (N13)
        key: Callable[[str], str] = (
            (lambda v: ph.fingerprint(v, ordered=True)) if person else ph.fingerprint
        )
        _join(groups, _buckets(units, _list_key(key) if listed else key))
        first = {i: groups.find(i) for i in range(len(units))}
        how = HOW_LIST if listed else HOW
        _emit(advanced, "A-01", column, groups, {i: i for i in range(len(units))}, person, how)
        if person:
            # N14: no A-02 nor A-03 on names of people: 'LE ROUX' and 'LEROUX' are two families
            continue
        # A-02: 2-letter fingerprint in order (spaces and separators only, N14)
        _join(groups, _buckets(units, _list_key(_ordered_pairs) if listed else _ordered_pairs))
        second = {i: groups.find(i) for i in range(len(units))}
        _emit(advanced, "A-02", column, groups, first, person, how)
        # A-03: a rare spelling close to a frequent one
        _neighbours(advanced, column, groups)
        _emit(advanced, "A-03", column, groups, second, person, how)


def _ordered_pairs(value: str) -> str:
    return ph.ngram_fingerprint(value, ordered=True)


def _list_key(key: Callable[[str], str]) -> Callable[[str], str]:
    return lambda value: ph.items_key(value, key)


def _join(groups: Groups, buckets: dict[str, list[int]]) -> None:
    units = groups.units
    for indexes in buckets.values():
        for index in indexes[1:]:
            if _same_numbers(units[indexes[0]].key, units[index].key):
                groups.union(indexes[0], index)


def _is_list(units: list[Unit]) -> bool:
    """N13: a column of lists ('en:vegan,en:no-gluten'), seen one of two ways:
    - most rows are lists: at least 30 % of the rows hold a separator (`,` `;` `|`) and the items
      come back from one value to another (half of the item occurrences are items seen in two
      distinct values or more): Open Food Facts' categories, labels, states;
    - its items stand alone too: at least 10 % of the distinct values are lists, and half of their
      items are also whole values of the column (brands: 'Kroger, The Kroger Co.' and 'The Kroger
      Co.'). A company name with a legal suffix ('Pepperidge Farm, Inc.': 'Inc.' never stands
      alone) or a city with its state ('BROOKLYN, NY', 4 % of the values) is not a list.
    A free text with commas has items that never come back."""
    lists = [unit for unit in units if ph.ITEM_SEPARATOR.search(unit.key)]
    if len(lists) < LIST_VALUES:
        return False
    items = [{ph.fingerprint(item) for item in ph.items(unit.key)} for unit in lists]
    occurrences = sum(len(found) for found in items)
    if not occurrences:
        return False
    total = sum(unit.rows for unit in units)
    if sum(unit.rows for unit in lists) >= LIST_SHARE * total:
        seen = Counter(item for found in items for item in found)
        if sum(c for c in seen.values() if c >= 2) >= LIST_REPEAT * occurrences:
            return True
    if len(lists) < LIST_DISTINCT * len(units):
        return False
    alone = {ph.fingerprint(unit.key) for unit in units if not ph.ITEM_SEPARATOR.search(unit.key)}
    standing = sum(1 for found in items for item in found if item in alone)
    return standing >= LIST_REPEAT * occurrences


def _buckets(units: list[Unit], key: Any) -> dict[str, list[int]]:
    found: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        if len(unit.key) >= 2:
            found[key(unit.key)].append(index)
    return {k: v for k, v in found.items() if len(v) > 1 and k}


HOW = {
    "A-01": "same words once case, accents, punctuation and word order are ignored (fingerprint)",
    "A-02": "same letters in the same order once spaces and separators are removed (2-letter "
    "fingerprint)",
    "A-03": "a rare spelling one or two letters away from a frequent one (same sound, letters "
    "swapped, or one letter missing in a long word)",
}


# N13: a column of lists is compared item by item, never word by word across its items
HOW_LIST = {
    **HOW,
    "A-01": "same items in another order or written another way (case, accents, punctuation); "
    "a column of lists, compared item by item, never word by word across items",
    "A-02": "same items once spaces and separators inside each item are removed; a column of "
    "lists, compared item by item",
}


def _emit(
    advanced: Advanced,
    rule: str,
    column: Column,
    groups: Groups,
    before: dict[int, int],
    person: bool,
    how: dict[str, str] = HOW,
) -> None:
    """The groups this method joined (`before`: each unit's group before it), each one's
    canonical = its most frequent unit; the units of the canonical's earlier group are left to the
    earlier method."""
    earlier: dict[int, list[int]] = defaultdict(list)
    for index, root in before.items():
        earlier[root].append(index)
    joined: dict[int, list[int]] = defaultdict(list)
    for old_root in earlier:
        joined[groups.find(old_root)].append(old_root)
    roots = {
        root: [root, *(i for old in olds if old != before[root] for i in earlier[old])]
        for root, olds in joined.items()
        if len(olds) > 1
    }
    if roots:
        _merge_suggestion(advanced, rule, column, groups, roots, how[rule], person)


def _neighbours(advanced: Advanced, column: Column, groups: Groups) -> None:
    """A-03: each rare group → its closest frequent one, frequencies 10 times apart (C-21):
    Damerau–Levenshtein ≤ 1 for 4 to 8 letters, ≤ 2 from 9, and (C-25) the same sound, two
    letters swapped, or one letter more or less in a word of 7 letters or more ('Brookyn'):
    'Clifton' and 'Clinton', 'Conroe' and 'Monroe', 'Hollis' and 'Hollins' are two towns.
    Candidates are blocked by sound, first two and last two letters (a typo keeps one of them)."""
    members = groups.members()
    if len(members) > NEIGHBOURS or len(members) < 2:
        return
    rows = {r: sum(groups.units[i].rows for i in members[r]) for r in members}
    keys = {
        r: groups.units[r].key
        for r in members
        if MIN_LENGTH <= groups.units[r].length <= MAX_LENGTH
    }
    sounds = {r: ph.phonetic(k) for r, k in keys.items()}
    blocks: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r, k in keys.items():
        for block in (("p", sounds[r]), ("s", k[:2]), ("e", k[-2:])):
            if block[1]:
                blocks[block].append(r)
    chosen: dict[int, int] = {}
    for rare, key in keys.items():
        best: tuple[int, int, int] | None = None  # distance, -rows, root
        seen: set[int] = set()
        for block in (("p", sounds[rare]), ("s", key[:2]), ("e", key[-2:])):
            for often in blocks.get(block, []):
                if often in seen or often == rare or rows[often] < RATIO * rows[rare]:
                    continue
                seen.add(often)
                other = keys[often]
                if abs(len(other) - len(key)) > 2:
                    continue
                limit = 2 if len(key) >= 9 else 1
                distance = ph.damerau(key, other, limit)
                if distance > limit or distance == 0:
                    continue
                same_sound = sounds[rare] != "" and sounds[rare] == sounds[often]
                if not _typo(key, other, same_sound, distance):
                    continue
                if not _same_numbers(key, other) or _token_swap(key, other):
                    continue
                candidate = (distance, -rows[often], often)
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            chosen[rare] = best[2]
    for rare, often in chosen.items():
        if groups.find(often) != groups.find(rare):
            groups.union(often, rare)


def _typo(a: str, b: str, same_sound: bool, distance: int) -> bool:
    """C-25: the edit looks like a slip of the keyboard, not another word."""
    if same_sound:
        return True
    if distance != 1:
        return False
    if len(a) == len(b):  # a swap of two neighbours ('Contractro'), not a substitution
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 2 and diff[1] == diff[0] + 1
    return min(len(a), len(b)) >= LONG_WORD  # one letter more or less in a long word


# --- Sample --------------------------------------------------------------------------------------

SAMPLE = "quantilo_advanced_sample"
ROW = "__row"


def make_sample(advanced: Advanced) -> None:
    """A reproducible sample (CLAUDE.md: `ORDER BY hash(row number + seed)`), or every row."""
    context = advanced.context
    advanced.sampled = context.rows > SAMPLE_ROWS
    order = f"hash(file_row_number + {SEED})" if advanced.sampled else "file_row_number"
    context.connection.execute(
        f"CREATE OR REPLACE TEMP TABLE {SAMPLE} AS SELECT file_row_number AS {ROW}, "
        f"* EXCLUDE (file_row_number) FROM {advanced.rows_source} ORDER BY {order} "
        f"LIMIT {SAMPLE_ROWS}"
    )
    advanced.sample_rows = min(context.rows, SAMPLE_ROWS)


def scaled(advanced: Advanced, count: int) -> str:
    if not advanced.sampled:
        return n(count)
    estimate = round(count * advanced.context.rows / max(advanced.sample_rows, 1))
    return f"about {n(estimate)} (from a sample of {n(advanced.sample_rows)} rows)"


def number_expr(column: Column) -> str | None:
    """The column as a DOUBLE: a number, or text the typing reads as numbers (B-07)."""
    if is_number(column.dtype):
        return f"CAST({ident(column.name)} AS DOUBLE)"
    view = column.view
    if is_text(column.dtype) and view is not None and view.parser.kind == "number":
        return number_sql(clean_text(column.name), view.parser.decimal)
    return None


def _measures(advanced: Advanced) -> list[Column]:
    """Number columns that measure something: no identifier, code, date part, coordinate, flag."""
    found = []
    for column in advanced.context.columns:
        if number_expr(column) is None or not column.nonnull:
            continue
        view = column.view
        role = column.role
        if view is not None and view.type in ("identifier", "boolean", "ignored"):
            continue
        if view is not None and view.role in semantics.KEY_ROLES:
            continue
        if semantics.code_by_name(column.name) is not None or semantics.is_date_part(column.name):
            continue
        if semantics.is_coordinate(column.name, view.role if view else None):
            continue
        if role is not None and role.id in ("year", "month", "weekday", "hour", "identifier"):
            continue
        found.append(column)
    return found


# --- Units (A-05) --------------------------------------------------------------------------------


def _quantiles(values: np.ndarray) -> np.ndarray:
    return np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])


def _unit_split(logs: np.ndarray) -> tuple[float, int, float] | None:
    """The split of log10 values into a majority and a minority that is the majority × or ÷ a
    factor (100, 1000): (threshold, signed exponent of the minority, mismatch)."""
    ordered = np.sort(logs)
    unique = np.unique(ordered)
    if len(unique) < 4:
        return None
    gaps = np.diff(unique)
    candidates = np.argsort(gaps)[::-1][:UNIT_SPLITS]
    best: tuple[float, int, float] | None = None
    for index in candidates:
        if gaps[index] < 0.1:
            continue
        threshold = float((unique[index] + unique[index + 1]) / 2)
        low, high = ordered[ordered < threshold], ordered[ordered >= threshold]
        share_high = len(high) / len(ordered)
        minority, majority, sign = (high, low, 1) if share_high <= 0.5 else (low, high, -1)
        share = len(minority) / len(ordered)
        if not UNIT_SHARE[0] <= share <= UNIT_SHARE[1] or len(minority) < UNIT_MIN_VALUES:
            continue
        for factor in UNIT_FACTORS:
            shift = sign * math.log10(factor)
            mismatch = float(np.mean(np.abs(_quantiles(minority) - shift - _quantiles(majority))))
            spread_major = float(np.subtract(*np.quantile(majority, [0.75, 0.25])))
            spread_minor = float(np.subtract(*np.quantile(minority, [0.75, 0.25])))
            if spread_major > 0 and not 0.5 <= (spread_minor + 0.05) / (spread_major + 0.05) <= 2:
                continue
            if mismatch <= UNIT_TOLERANCE and (best is None or mismatch < best[2]):
                best = (threshold, sign * round(math.log10(factor)), mismatch)
    return best


def percent(column: Column) -> bool:
    """N17: a percentage (or a share per 100 g), by its name or its role: '% Votants',
    'discount_pct', 'fruits-vegetables-legumes_100g'."""
    view_role = column.view.role if column.view else None
    role = column.role.id if column.role is not None else None
    words = set(semantics.tokens(column.name))
    return "%" in column.name or "percentage" in (role, view_role) or bool(words & PERCENT_WORDS)


def units(advanced: Advanced) -> None:
    connection = advanced.context.connection
    for column in _measures(advanced):
        expr = number_expr(column)
        assert expr is not None
        rows = connection.execute(
            f"SELECT v FROM (SELECT {expr} AS v FROM {SAMPLE}) WHERE v > 0 AND isfinite(v)"
        ).fetchall()
        values = np.array([r[0] for r in rows], dtype=float)
        if len(values) < 50 or len(np.unique(values)) < 5:
            continue
        split = _unit_split(np.log10(values))
        if split is None:
            continue
        threshold, exponent, _mismatch = split
        if exponent < 0 and percent(column):
            logs = np.log10(values)
            gap = float(logs[logs >= threshold].min() - logs[logs < threshold].max())
            if gap < UNIT_PERCENT_GAP:
                continue  # N17: the low tail of real small percentages, not fractions
        factor = 10 ** abs(exponent)
        bound = 10**threshold
        test = f"v >= {bound!r}" if exponent > 0 else f"v > 0 AND v < {bound!r}"
        _unit_suggestion(advanced, column, expr, test, exponent, factor)


def _unit_target(advanced: Advanced, column: Column) -> str:
    if is_number(column.dtype):
        return column.dtype
    for found in advanced.basic_found:
        if found.rule == "B-07" and found.columns == [column.name] and found.step is not None:
            return "BIGINT" if getattr(found.step, "to", "") == "integer" else "DOUBLE"
    return "DOUBLE"


def _unit_suggestion(
    advanced: Advanced, column: Column, expr: str, test: str, exponent: int, factor: int
) -> None:
    context = advanced.context
    raw = as_text(column.name, column.dtype)
    shown = (
        "CASE WHEN v = trunc(v) AND abs(v) < 1e15 THEN CAST(CAST(v AS BIGINT) AS VARCHAR) "
        "ELSE CAST(v AS VARCHAR) END"
    )
    found = context.connection.execute(
        f"SELECT raw, {shown}, v, count(*) FROM (SELECT {raw} AS raw, {expr} AS v FROM "
        f"{context.source}) WHERE {test} GROUP BY 1, 2, 3 ORDER BY 4 DESC, 3"
    ).fetchall()
    count = sum(int(r[3]) for r in found)
    total, median = context.one(
        f"SELECT count(*), median(v) FILTER (WHERE v > 0) FROM (SELECT {expr} AS v FROM "
        f"{context.source}) WHERE v IS NOT NULL AND NOT ({test})"
    )
    if not count or not total:
        return
    divide = exponent > 0
    mapping: dict[str, str | None] = {}
    integral = True
    for text, number_text, value, _rows in found:
        new = float(value) / factor if divide else float(value) * factor
        integral = integral and new == int(new)
        for key in {str(text).strip(), str(text), str(number_text)}:
            mapping[key] = _number_text(new)
    share = count / (count + int(total))
    word = {100: "a hundred", 1000: "a thousand"}[factor]
    how = "larger" if divide else "smaller"
    hint = {
        (100, True): "cents among euros, or percentages among fractions?",
        (100, False): "fractions (0–1) among percentages (0–100), or euros among cents?",
        (1000, True): "grams among kilograms, or units among thousands?",
        (1000, False): "kilograms among grams, or thousands among units?",
    }[(factor, divide)]
    reason = (
        f"{n(count)} values ({share:.1%}) are about {word} times {how} than the others "
        f"(median {float(median):,.4g}), with the same spread once rescaled: {hint} "
        f"{'Divided' if divide else 'Multiplied'} by {n(factor)}, they would match."
    )
    target = _unit_target(advanced, column)
    can_map = len(mapping) <= m.MAX_MAP and (integral or target not in INTEGERS)
    step = None
    if can_map:
        step = m.MapValues(id=_id(), column=column.name, map=mapping)
    else:
        reason += " No step proposed: too many distinct values, or the column stores whole numbers."
    advanced.add(
        "A-05",
        "S" if step is not None else "I",
        [column],
        f"Rescale {n(count)} values of {column.name} ({'÷' if divide else '×'}{n(factor)})",
        reason,
        step,
        cells=count,
        examples=[
            (str(r[0]), _number_text(float(r[2]) / factor if divide else float(r[2]) * factor))
            for r in found[:3]
        ],
    )


# --- Dependencies (A-10) -------------------------------------------------------------------------


def _dependency_columns(advanced: Advanced) -> list[Column]:
    rows = max(advanced.context.rows, 1)
    found = []
    for column in advanced.context.columns:
        view = column.view
        if not column.nonnull or column.distinct < 2 or column.distinct > 0.5 * rows:
            continue
        if view is not None and view.type in ("text", "ignored", "boolean"):
            continue
        postal = column.role is not None and column.role.id == "postal_code"
        if is_text(column.dtype):
            if column.name not in advanced.context.stages or basic._converted(column):
                continue
        elif not (postal or (view is not None and view.role in ("postal_code", "identifier"))):
            continue
        found.append(column)

    def priority(column: Column) -> tuple[int, int]:
        geo = column.role is not None and column.role.id in (
            "city",
            "country",
            "region",
            "postal_code",
        )
        return (0 if geo else 1, column.distinct)

    return sorted(found, key=priority)[:FD_COLUMNS]


def _codes(advanced: Advanced, column: Column, values: list[Any]) -> tuple[np.ndarray, list[str]]:
    """Each sampled value → an integer (−1: empty), equal for spellings Basic and A-01 to A-03
    would merge."""
    stages = advanced.context.stages.get(column.name)
    finals: dict[str, str] = {}
    if stages is not None:
        for raw, final in ((r, f) for f, rs in _final_of(stages).items() for r in rs):
            finals[raw] = final
    merges = advanced.merges.get(column.name, {})
    index: dict[str, int] = {}
    labels: list[str] = []
    cache: dict[Any, int] = {}
    out = np.full(len(values), -1, dtype=np.int64)
    for position, value in enumerate(values):
        if value is None:
            continue
        code = cache.get(value)
        if code is None:
            text = finals.get(str(value), str(value))
            key = ph.fold(text)
            if not key or key.strip().lower() in NULL_TOKENS:
                cache[value] = -1
                continue
            key = merges.get(key, key)
            code = index.get(key)
            if code is None:
                code = index[key] = len(labels)
                labels.append(text)
            cache[value] = code
        out[position] = code
    return out, labels


@dataclass
class Dependency:
    a: Column
    b: Column
    share: float  # rows whose B is the majority B of their A
    gain: float  # errors removed compared with guessing B alone
    keys: int  # values of A
    groups: float  # rows per value of A
    paired: int  # values of A with two rows or more
    majority: dict[int, int]  # code of A → code of its majority B
    support: dict[int, tuple[int, int]]  # code of A → (rows of its majority B, rows with a B)
    accepted: set[tuple[int, int]]  # C-27: second answers frequent enough to be real
    errors: int  # rows that are neither the majority nor an accepted second answer

    @property
    def holds(self) -> bool:
        return self.share >= FD_SHARE and self.gain >= FD_GAIN

    @property
    def reportable(self) -> bool:
        return (
            self.holds
            and self.errors > 0
            and self.keys >= FD_KEYS
            and self.groups >= FD_GROUP
            and self.paired >= 5
        )


def _dependency(
    a: Column, b: Column, codes_a: np.ndarray, codes_b: np.ndarray, nb: int
) -> Dependency | None:
    mask = (codes_a >= 0) & (codes_b >= 0)
    total = int(mask.sum())
    if total < 50:
        return None
    joint = codes_a[mask] * nb + codes_b[mask]
    unique, counts = np.unique(joint, return_counts=True)
    owners = unique // nb
    starts = np.flatnonzero(np.r_[True, owners[1:] != owners[:-1]])
    tops = np.maximum.reduceat(counts, starts)
    sizes = np.add.reduceat(counts, starts)
    baseline = total - int(np.bincount(codes_b[mask]).max())
    if baseline <= 0:
        return None
    errors = total - int(tops.sum())
    lengths = np.diff(np.r_[starts, len(unique)])
    group_size = np.repeat(sizes, lengths)
    group_top = np.repeat(tops, lengths)
    majority: dict[int, int] = {}
    support: dict[int, tuple[int, int]] = {}
    accepted: set[tuple[int, int]] = set()
    alternative = 0
    second = FD_SECOND_NAMES if basic._person_names(a) else FD_SECOND_SHARE
    for value, count, size, top in zip(
        unique.tolist(), counts.tolist(), group_size.tolist(), group_top.tolist(), strict=True
    ):
        owner, answer = divmod(value, nb)
        if count == top and owner not in majority:
            majority[owner] = answer
            support[owner] = (top, size)
        elif count >= FD_SECOND_ROWS and count / size >= second:
            accepted.add((owner, answer))
            alternative += count
    return Dependency(
        a,
        b,
        share=1 - errors / total,
        gain=1 - errors / baseline,
        keys=len(starts),
        groups=total / len(starts),
        paired=int((sizes >= 2).sum()),
        majority=majority,
        support=support,
        accepted=accepted,
        errors=errors - alternative,
    )


def dependencies(advanced: Advanced) -> None:
    columns = _dependency_columns(advanced)
    if len(columns) < 2:
        return
    connection = advanced.context.connection
    texts = ", ".join(as_text(c.name, c.dtype) for c in columns)
    rows = connection.execute(f"SELECT {ROW}, {texts} FROM {SAMPLE} ORDER BY {ROW}").fetchall()
    if not rows:
        return
    numbers = np.array([r[0] for r in rows], dtype=np.int64)
    coded: dict[str, tuple[np.ndarray, list[str]]] = {}
    raw: dict[str, list[Any]] = {}
    for position, column in enumerate(columns, start=1):
        raw[column.name] = [r[position] for r in rows]
        coded[column.name] = _codes(advanced, column, raw[column.name])
    found: dict[tuple[str, str], Dependency] = {}
    for a in columns:
        for b in columns:
            labels = coded[b.name][1]
            if a is b or len(labels) < 2:
                continue
            dependency = _dependency(a, b, coded[a.name][0], coded[b.name][0], len(labels))
            if dependency is not None and dependency.holds:
                found[(a.name, b.name)] = dependency
    reported = [d for d in found.values() if d.reportable and not _implied(d, found, columns)]
    reported.sort(key=lambda d: (-d.errors, d.a.position))
    for dependency in reported[:FD_MAX]:
        _dependency_suggestion(advanced, dependency, coded, raw, numbers, found)


def _implied(
    dependency: Dependency, found: dict[tuple[str, str], Dependency], columns: list[Column]
) -> bool:
    """A → C said by A → B and B → C (zip → city → country); of two equivalent columns (B ↔ C,
    a copy) the first one speaks; of A → B and B → A, the finer A (more values)."""
    a, c = dependency.a, dependency.b
    for b in columns:
        if b is a or b is c or (a.name, b.name) not in found or (b.name, c.name) not in found:
            continue
        if (c.name, b.name) in found and c.position < b.position:
            continue  # B and C are copies: A → C is the one kept
        return True
    reverse = found.get((c.name, a.name))
    if reverse is not None and reverse.reportable:
        return (reverse.keys, -reverse.a.position) > (dependency.keys, -a.position)
    return False


@dataclass
class Kept:
    """N15: why a disagreeing row is left alone (a real exception, not a typo)."""

    kind: str  # "unclear", "copied", "agrees", "explained"
    text: str  # said in the reason


def _exceptions(
    advanced: Advanced,
    dependency: Dependency,
    coded: dict[str, tuple[np.ndarray, list[str]]],
    found: dict[tuple[str, str], Dependency],
    numbers: np.ndarray,
    positions: np.ndarray,
) -> dict[int, Kept]:
    """N15: the world has repeated exceptions (Marble Hill: ZIP 10463 and the Bronx's district 8,
    in Manhattan; ZIP 11237 on both sides of the Brooklyn–Queens border; two towns named Wayne).
    A disagreeing row is left alone, never changed, when
    - its value of A has no clear majority (fewer than 3 rows, or not twice the others: 'MONTVALE'
      NY on one row, NJ on the other);
    - its B is written in another column of the row (311: city 'MANHATTAN' where most rows of
      ZIP 10023 write 'NEW YORK', and borough 'MANHATTAN'): another naming, not a slip;
    - another column that determines B agrees with its B (borough MANHATTAN with city NEW YORK):
      the row is consistent, A may be the odd one, or the place is on a border;
    - a third column explains it: every row with this A and this C (3 or more) has this B (ZIP
      3168 and suburb Notting Hill: Southern Metropolitan, where Clayton is South-Eastern; ZIP
      10463 and tax block 02215: Manhattan). The third column is any text column of the table,
      read on the rows of the values of A that disagree (50,000 rows at most).
    A typo stays alone: the planted 'Paris, Belgium' rows have Paris ZIP codes (75…) and nothing
    else in common, so their step stays."""
    a, b = dependency.a, dependency.b
    codes_a, labels_a = coded[a.name]
    codes_b, labels_b = coded[b.name]
    others = [name for name in coded if name not in (a.name, b.name)]
    keys = sorted({int(codes_a[p]) for p in positions.tolist()})
    order = np.argsort(codes_a, kind="stable")
    bounds = zip(
        keys,
        np.searchsorted(codes_a[order], keys, side="left").tolist(),
        np.searchsorted(codes_a[order], keys, side="right").tolist(),
        strict=True,
    )
    rows_of = {key: order[low:high] for key, low, high in bounds}
    context = {name: coded[name] for name in others}
    context |= _context(advanced, set(coded), numbers, rows_of)
    # a copy of B (311's park_borough, equal to borough) proves nothing about a row's B
    folded_b = {ph.fold(label): code for code, label in enumerate(labels_b)}
    context = {k: v for k, v in context.items() if not _copy_of_b(v, codes_b, folded_b)}
    others = [name for name in others if name in context]
    out: dict[int, Kept] = {}
    for position in positions.tolist():
        key, answer = int(codes_a[position]), int(codes_b[position])
        top, size = dependency.support.get(key, (0, 0))
        said = f"{q(labels_a[key])} has {b.name} {q(labels_b[answer])}"
        if top < FD_MAJORITY or top < 2 * (size - top):
            out[position] = Kept(
                "unclear",
                f"{q(labels_a[key])} has no clear {b.name} ({top} of its {size} rows agree)",
            )
            continue
        kept = _copied(context, codes_b, labels_b, position, said)
        if kept is None:
            kept = _agrees(dependency, coded, found, others, position, said)
        if kept is None:
            kept = _explained(context, rows_of[key], codes_b, position, said)
        if kept is not None:
            out[position] = kept
    return out


def _copy_of_b(
    pair: tuple[np.ndarray, list[str]], codes_b: np.ndarray, folded_b: dict[str, int]
) -> bool:
    """A column that writes the very value of B on 95 % of the rows where both are known (among
    the rows read) is a copy of B: it repeats B's slips too."""
    codes, labels = pair
    as_b = np.array([folded_b.get(ph.fold(label), -2) for label in labels] + [-3], dtype=np.int64)
    known = (codes >= 0) & (codes_b >= 0)
    if not known.any():
        return False
    return float((as_b[codes[known]] == codes_b[known]).mean()) >= FD_COPY


def _copied(
    context: dict[str, tuple[np.ndarray, list[str]]],
    codes_b: np.ndarray,
    labels_b: list[str],
    position: int,
    said: str,
) -> Kept | None:
    """Another column of the row (A aside: a shifted A is an error; copies of B aside) holds this
    very B: borough 'MANHATTAN' where the city of most rows of ZIP 10023 is 'NEW YORK'."""
    wanted = ph.fold(labels_b[int(codes_b[position])])
    for name, (codes, labels) in context.items():
        code = int(codes[position])
        if code >= 0 and ph.fold(labels[code]) == wanted:
            return Kept("copied", f"{said}, the same as its {name}")
    return None


def _agrees(
    dependency: Dependency,
    coded: dict[str, tuple[np.ndarray, list[str]]],
    found: dict[tuple[str, str], Dependency],
    others: list[str],
    position: int,
    said: str,
) -> Kept | None:
    """Another column D → B whose clear majority, on this row's D, is this row's B."""
    b = dependency.b
    answer = int(coded[b.name][0][position])
    for name in others:
        other = found.get((name, b.name))
        code = int(coded[name][0][position])
        if other is None or code < 0 or other.majority.get(code) != answer:
            continue
        if other.support.get(code, (0, 0))[0] < FD_MAJORITY:
            continue
        return Kept("agrees", f"{said}, as {name} {q(coded[name][1][code])} says")
    return None


def _explained(
    context: dict[str, tuple[np.ndarray, list[str]]],
    same_a: np.ndarray,
    codes_b: np.ndarray,
    position: int,
    said: str,
) -> Kept | None:
    """A third column C: all the rows of this A and this row's C (3 or more) have this B."""
    answer = int(codes_b[position])
    for name, (codes_c, labels_c) in context.items():
        code = int(codes_c[position])
        if code < 0:
            continue
        group = same_a[(codes_c[same_a] == code) & (codes_b[same_a] >= 0)]
        if len(group) >= FD_EXPLAINED and bool((codes_b[group] == answer).all()):
            return Kept(
                "explained", f"{said} on all its {len(group)} rows with {name} {q(labels_c[code])}"
            )
    return None


def _context_columns(advanced: Advanced, skip: set[str]) -> list[Column]:
    """Text columns that may explain an exception (an address, a block, a neighbourhood)."""
    found = []
    for column in advanced.context.columns:
        view = column.view
        if column.name in skip or not is_text(column.dtype) or column.distinct < 2:
            continue
        if view is not None and view.type in ("ignored", "boolean"):
            continue
        if basic._converted(column):
            continue
        found.append(column)
    return sorted(found, key=lambda c: -c.distinct)[:FD_CONTEXT]


def _context(
    advanced: Advanced,
    skip: set[str],
    numbers: np.ndarray,
    rows_of: dict[int, np.ndarray],
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """N15: the other text columns, read on the rows of the values of A that disagree only (the
    smallest values first, FD_CONTEXT_ROWS at most); −1 elsewhere."""
    columns = _context_columns(advanced, skip)
    if not columns:
        return {}
    chosen: list[np.ndarray] = []
    total = 0
    for rows in sorted(rows_of.values(), key=len):
        if total + len(rows) > FD_CONTEXT_ROWS:
            break
        chosen.append(rows)
        total += len(rows)
    if not chosen:
        return {}
    positions = np.unique(np.concatenate(chosen))
    ids = numbers[positions].tolist()
    texts = ", ".join(as_text(c.name, c.dtype) for c in columns)
    fetched = advanced.context.connection.execute(
        f"SELECT {ROW}, {texts} FROM {SAMPLE} WHERE {ROW} IN (SELECT unnest($ids)) ORDER BY {ROW}",
        {"ids": ids},
    ).fetchall()
    where = np.searchsorted(numbers, np.array([r[0] for r in fetched], dtype=np.int64))
    out: dict[str, tuple[np.ndarray, list[str]]] = {}
    for index, column in enumerate(columns, start=1):
        codes = np.full(len(numbers), -1, dtype=np.int64)
        labels: list[str] = []
        seen: dict[str, int] = {}
        for at, row in zip(where.tolist(), fetched, strict=True):
            value = row[index]
            if value is None or not str(value).strip():
                continue
            key = ph.fold(str(value))
            code = seen.get(key)
            if code is None:
                code = seen[key] = len(labels)
                labels.append(str(value))
            codes[at] = code
        out[column.name] = (codes, labels)
    return out


def _dependency_suggestion(
    advanced: Advanced,
    dependency: Dependency,
    coded: dict[str, tuple[np.ndarray, list[str]]],
    raw: dict[str, list[Any]],
    numbers: np.ndarray,
    found: dict[tuple[str, str], Dependency] | None = None,
) -> None:
    a, b = dependency.a, dependency.b
    codes_a, labels_a = coded[a.name]
    codes_b, labels_b = coded[b.name]
    expected = np.array(
        [dependency.majority.get(int(code), -1) for code in codes_a.tolist()], dtype=np.int64
    )
    wrong = (codes_a >= 0) & (codes_b >= 0) & (codes_b != expected)
    for position in np.flatnonzero(wrong).tolist():
        if (int(codes_a[position]), int(codes_b[position])) in dependency.accepted:
            wrong[position] = False
    positions = np.flatnonzero(wrong)
    if not len(positions):
        return
    kept = _exceptions(advanced, dependency, coded, found or {}, numbers, positions)
    typos = np.array([p for p in positions.tolist() if p not in kept], dtype=np.int64)
    per_key: dict[int, int] = defaultdict(int)
    for position in positions.tolist():
        per_key[int(codes_a[position])] += 1
    worst = max(per_key, key=lambda k: per_key[k])
    worst_rows = int(((codes_a == worst) & (codes_b >= 0)).sum())
    worst_share = 1 - per_key[worst] / max(worst_rows, 1)
    shown = [*typos.tolist(), *(p for p in positions.tolist() if p in kept)][:LISTED]
    examples = [
        (
            f"{row_label(int(numbers[p]))}: {a.name} {q(str(raw[a.name][p]))}, "
            f"{b.name} {q(str(raw[b.name][p]))}",
            f"{labels_b[int(codes_b[p])]} (kept)" if p in kept else labels_b[int(expected[p])],
        )
        for p in shown
    ]
    count = len(positions)
    # N15: a person's sex (or any B) is never corrected from their name (CLN-5)
    names = basic._person_names(a)
    step = (
        _dependency_step(advanced, dependency, codes_a, codes_b, raw, typos)
        if len(typos) and not names
        else None
    )
    reason = (
        f"{a.name} determines {b.name} on {dependency.share:.1%} of rows: "
        f"{q(labels_a[worst])} has {b.name} {q(labels_b[dependency.majority[worst]])} on "
        f"{worst_share:.1%} of its rows. {scaled(advanced, count)} rows disagree "
        f"({rows_list(sorted(int(numbers[p]) for p in shown), count)})"
    )
    if kept:
        texts = list(dict.fromkeys(k.text for k in kept.values()))
        more = f" (and {len(texts) - 2} more)" if len(texts) > 2 else ""
        alone = "all of them" if len(kept) == count else f"{scaled(advanced, len(kept))} of them"
        reason += (
            f". {alone[0].upper()}{alone[1:]} look like real exceptions, repeated or consistent "
            f"with the rest of the row: {'; '.join(texts[:2])}{more}. Not changed."
        )
        if len(typos):
            reason += (
                f" The other {scaled(advanced, len(typos))} look like a typo or a wrong value in "
                "one of the two columns."
            )
    else:
        reason += ": a typo or a wrong value in one of the two columns."
    if step is not None:
        rows = "these rows" if not kept else "those rows"
        reason += (
            f" The step sets {b.name} to the majority value of its {a.name} on {rows} only (the "
            f"wrong values it saw are listed, a new one is never changed); if {a.name} is the "
            "wrong column, leave it unticked."
        )
        if advanced.sampled:
            reason += " Found on a sample: other rows may disagree too."
    elif names and len(typos):
        reason += (
            f" {a.name} holds names of people: no step changes a person's {b.name} from their "
            "name (CLN-5). Information only: check these rows."
        )
    else:
        reason += " Information only: check these rows."
    advanced.add(
        "A-10",
        "S" if step is not None else "I",
        [a, b],
        f"{b.name} disagrees with {a.name} on {scaled(advanced, count)} rows",
        reason,
        step,
        rows=0,
        cells=len(typos) if step is not None else count,
        examples=examples,
    )


def variants(advanced: Advanced, column: Column, raw: str) -> set[str]:
    """The spellings a value may have when a step reads it (C-23): as stored, after the Basic
    text fixes, and the form B-03 or A-01 to A-03 would merge it into."""
    found = {raw}
    stages = advanced.context.stages.get(column.name)
    if stages is None:
        return found
    cache = advanced.spellings.get(column.name)
    if cache is None:
        finals = {r: f for f, rs in _final_of(stages).items() for r in rs}
        displays = {unit.key: unit.display for unit in units_of(stages)}
        cache = advanced.spellings[column.name] = (finals, displays)
    finals, displays = cache
    final = finals.get(raw, raw)
    found.add(final)
    key = ph.fold(final)
    merged = advanced.merges.get(column.name, {}).get(key, key)
    for one in (key, merged):
        if one in displays:
            found.add(displays[one])
    return found


def _dependency_step(
    advanced: Advanced,
    dependency: Dependency,
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    raw: dict[str, list[Any]],
    positions: np.ndarray,
) -> m.SetWhere | None:
    """C-45: one case per value of A that has wrong rows: where A is that value (any of its
    spellings) and B one of the wrong values seen, B takes the majority value of A, written as
    most of its rows write it."""
    a, b = dependency.a, dependency.b
    wrong_b: dict[int, set[str]] = defaultdict(set)
    wrong_a: dict[int, set[str]] = defaultdict(set)
    for position in positions.tolist():
        code = int(codes_a[position])
        wrong_a[code].add(str(raw[a.name][position]))
        wrong_b[code].add(str(raw[b.name][position]))
    written: dict[int, Counter[str]] = defaultdict(Counter)
    for position in np.flatnonzero(np.isin(codes_a, list(wrong_a))).tolist():
        code = int(codes_a[position])
        if int(codes_b[position]) == dependency.majority.get(code) and raw[b.name][position]:
            written[code][str(raw[b.name][position])] += 1
    cases = []
    for code in sorted(wrong_a, key=lambda c: -len(wrong_b[c]))[:FIX_CASES]:
        if not written[code]:
            continue
        target = written[code].most_common(1)[0][0]
        spelled_a = sorted({v for raw_a in wrong_a[code] for v in variants(advanced, a, raw_a)})
        spelled_b = sorted({v for raw_b in wrong_b[code] for v in variants(advanced, b, raw_b)})
        spelled_b = [v for v in spelled_b if v != target]
        if not spelled_b or len(spelled_a) > m.MAX_VALUES or len(spelled_b) > m.MAX_VALUES:
            continue
        conditions = [
            m.RowCondition(column=a.name, op="in", values=spelled_a),
            m.RowCondition(column=b.name, op="in", values=spelled_b),
        ]
        cases.append(m.Case(conditions=conditions, value=target))
    if not cases:
        return None
    return m.SetWhere(id=_id(), column=b.name, cases=cases)


# --- Dates in order (A-11), ages (A-13) ---------------------------------------------------------


def date_expr(advanced: Advanced, column: Column) -> str | None:
    """The column as a DATE: a date, or text that B-11 reads with its formats (C-24)."""
    if is_temporal(column.dtype):
        return f"CAST({ident(column.name)} AS DATE)"
    if not is_text(column.dtype):
        return None
    for found in advanced.basic_found:
        step = found.step
        if found.rule in ("B-11", "B-17") and found.columns == [column.name] and step is not None:
            value = f"trim(regexp_replace({ident(column.name)}, {lit(tx.SQL_SPACE_RUN)}, ' ', 'g'))"
            parsed = _date_expr(value, list(step.formats), bool(step.excel_serial))  # type: ignore[union-attr]
            return f"CAST({parsed} AS DATE)"
    return None


def display_sql(column: Column) -> str:
    """The value as text, a date without its midnight time."""
    text = as_text(column.name, column.dtype)
    if is_temporal(column.dtype):
        return f"regexp_replace({text}, ' 00:00:00$', '')"
    return text


def _dates(advanced: Advanced) -> list[tuple[Column, str]]:
    found = []
    for column in advanced.context.columns:
        expr = date_expr(advanced, column)
        if expr is not None and column.nonnull:
            found.append((column, expr))
    return found


def _ordered(first: Column, second: Column) -> bool:
    """By their names, `first` comes before `second` (C-24, S-06 of 093)."""
    a, b = set(semantics.tokens(first.name)), set(semantics.tokens(second.name))
    if a & BIRTH_WORDS and not b & BIRTH_WORDS:
        return True
    if a & START_WORDS and b & END_WORDS:
        return (a - START_WORDS) == (b - END_WORDS) or not (a - START_WORDS - {"date", "time"})
    tier_a, tier_b = semantics.date_tier(first.name), semantics.date_tier(second.name)
    return tier_a == 0 and tier_b in (2, 3)


def _apart(days: float) -> str:
    """'3 days', '2 months', '6 years'."""
    if days >= 730:
        return f"{days / 365.25:,.0f} years"
    if days >= 60:
        return f"{days / 30.44:,.0f} months"
    return f"{days:,.0f} days" if days >= 2 else "a day or less"


def date_order(advanced: Advanced) -> None:
    """A-11: an end before its start. N16: an end written more coarsely than its start (at :00
    seconds, at midnight: nyc_311 `closed_date`) is compared at its own precision, so the same
    minute or day written two ways is no inversion (said once when it is all there is, never
    swapped); two dates less than 60 days apart are swapped (S, `swap_values` on those rows
    only); further apart one of them is wrong: listed, not changed (I)."""
    dates = _dates(advanced)
    context = advanced.context
    for first, first_expr in dates:
        for second, second_expr in dates:
            if first is second or not _ordered(first, second):
                continue
            stamps = is_temporal(first.dtype) and is_temporal(second.dtype)
            stamps = stamps and "DATE" not in (first.dtype, second.dtype)
            a, b = first_expr, second_expr
            if stamps:
                a = f"CAST({ident(first.name)} AS TIMESTAMP)"
                b = f"CAST({ident(second.name)} AS TIMESTAMP)"
            both = f"({a}) IS NOT NULL AND ({b}) IS NOT NULL"
            after = f"{both} AND ({a}) > ({b})"
            cut = (  # the start at the end's precision: its day, or its minute
                f"CASE WHEN date_trunc('day', {b}) = {b} THEN date_trunc('day', {a}) "
                f"WHEN date_trunc('minute', {b}) = {b} THEN date_trunc('minute', {a}) ELSE {a} END"
                if stamps
                else a
            )
            real = f"({cut}) > ({b})"
            days = f"date_diff('day', CAST({b} AS DATE), CAST({a} AS DATE))"
            total, wrong, tiny, far, same, median = context.one(
                f"SELECT count(*) FILTER (WHERE {both}), count(*) FILTER (WHERE {after}), "
                f"count(*) FILTER (WHERE {after} AND NOT {real}), "
                f"count(*) FILTER (WHERE {after} AND {real} AND {days} >= {SWAP_DAYS}), "
                f"count(*) FILTER (WHERE {after} AND {real} AND {days} = 0), "
                f"median({days}) FILTER (WHERE {after} AND {real} AND {days} >= {SWAP_DAYS}) "
                f"FROM {advanced.rows_source}"
            )
            total, wrong, tiny, far, same = int(total), int(wrong), int(tiny), int(far), int(same)
            inverted = wrong - tiny
            if not total or not wrong:
                continue
            spans = OrderSpans(a, b, after, real, days, tiny, far, inverted - far - same, same)
            if not inverted:
                if 1 - tiny / total >= ORDER_SHARE:
                    _same_precision(advanced, first, second, spans)
            elif 1 - inverted / total >= ORDER_SHARE:
                _out_of_order(advanced, first, second, spans, total, float(median or 0))


@dataclass
class OrderSpans:
    """N16: the rows of A-11 by how far apart their two dates are."""

    a: str
    b: str
    after: str  # SQL: both read and the first after the second
    real: str  # … and still after it at the second's precision
    days: str
    tiny: int  # the same minute or day, the end written without its seconds or time
    far: int  # 60 days or more apart: one date is wrong
    close: int  # in between: swapped dates
    same: int = 0  # timestamps hours apart on one day: a time typed or rounded, not swapped

    def listed(self, advanced: Advanced, first: Column, second: Column, where: str) -> list[Any]:
        return advanced.context.connection.execute(
            f"SELECT file_row_number, {display_sql(first)}, {display_sql(second)} "
            f"FROM {advanced.rows_source} WHERE {self.after} AND {where} "
            f"ORDER BY file_row_number LIMIT {LISTED}"
        ).fetchall()


def _examples(first: Column, second: Column, listed: list[Any]) -> list[tuple[str, str]]:
    return [(f"{row_label(r[0])}: {first.name} {r[1]}", f"{second.name} {r[2]}") for r in listed]


def _same_precision(advanced: Advanced, first: Column, second: Column, spans: OrderSpans) -> None:
    listed = spans.listed(advanced, first, second, "true")
    advanced.add(
        "A-11",
        "I",
        [first, second],
        f"{second.name} a little before {first.name} on {n(spans.tiny)} rows (the same minute)",
        f"{second.name} is before {first.name} on {n(spans.tiny)} rows "
        f"({rows_list([r[0] for r in listed], spans.tiny)}), always in the same minute (or day): "
        f"{second.name} is written there without its seconds (:00) or its time. Not an error "
        "and not swapped; a duration computed from them is negative by a few seconds. Not "
        "changed.",
        cells=spans.tiny,
        examples=_examples(first, second, listed),
        key="minute",
    )


def _out_of_order(
    advanced: Advanced,
    first: Column,
    second: Column,
    spans: OrderSpans,
    total: int,
    median: float,
) -> None:
    wrong = spans.close + spans.far + spans.same
    listed = spans.listed(advanced, first, second, spans.real)
    order = f"{first.name} comes before {second.name} on {1 - wrong / total:.1%} of rows"
    rows = rows_list([r[0] for r in listed], wrong)
    left = []  # the rows the step leaves, and why
    if spans.far:
        left.append(
            f"{n(spans.far)} {'is' if spans.far == 1 else 'are'} {SWAP_DAYS} days or more apart "
            f"({_apart(median)}, median): one date is wrong there, not swapped"
        )
    if spans.same:
        left.append(
            f"{n(spans.same)} {'is' if spans.same == 1 else 'are'} hours apart on the same day: a "
            "time typed or rounded, not swapped"
        )
    if spans.tiny:
        left.append(
            f"{n(spans.tiny)} more {'is' if spans.tiny == 1 else 'are'} in the same minute or day, "
            f"{second.name} written without its seconds or time: no inversion"
        )
    said = " ".join(f"{part[0].upper()}{part[1:]}." for part in left)
    if not spans.close:
        advanced.add(
            "A-11",
            "I",
            [first, second],
            f"{second.name} before {first.name} on {n(wrong)} rows",
            f"{order}, after it on {n(wrong)} ({rows}). {said} Not changed: check these rows.",
            cells=wrong,
            examples=_examples(first, second, listed),
        )
        return
    a, b = fx.quoted(first.name), fx.quoted(second.name)
    when = f"{a} > {b}"
    if spans.tiny or spans.same:  # not the rows of one day
        when += f" and date_diff('day', {b}, {a}) >= 1"
    if spans.far:
        when += f" and date_diff('day', {b}, {a}) < {SWAP_DAYS}"
    reason = (
        f"{order}, after it on {n(wrong)} ({rows}): swapped or mistyped dates. The step swaps "
        f"the two dates on the {n(spans.close)} rows where {first.name} is after {second.name} "
        f"by less than {SWAP_DAYS} days; it needs both read as dates (Types)."
        + (f" {said} They are left as they are." if said else "")
        + " If one date is simply wrong, leave it unticked and check these rows."
    )
    advanced.add(
        "A-11",
        "S",
        [first, second],
        f"{second.name} before {first.name} on {n(wrong)} rows",
        reason,
        m.SwapValues(id=_id(), columns=[first.name, second.name], when=when),
        cells=spans.close,
        examples=_examples(first, second, listed),
    )


def ages(advanced: Advanced) -> None:
    context = advanced.context
    dates = _dates(advanced)
    births = [(c, e) for c, e in dates if set(semantics.tokens(c.name)) & BIRTH_WORDS]
    if not births:
        return
    for age in context.columns:
        role = age.role
        if role is None or role.id != "age" or number_expr(age) is None:
            continue
        value = f"({number_expr(age)})"
        for birth, birth_expr in births:
            references = [(c, e) for c, e in dates if c is not birth and c is not age]
            best: tuple[float, str, str, int, int] | None = None
            source: Column | float | None = None  # the reference date, or the year of the ages
            plausible = f"{value} BETWEEN 0 AND 130 AND ({birth_expr}) IS NOT NULL"
            for reference, reference_expr in references:
                computed = f"floor(date_diff('day', {birth_expr}, {reference_expr}) / 365.25)"
                where = f"{plausible} AND ({reference_expr}) IS NOT NULL"
                total, good = context.one(
                    f"SELECT count(*), count(*) FILTER (WHERE abs({value} - {computed}) <= 1) "
                    f"FROM {advanced.rows_source} WHERE {where}"
                )
                if total and (best is None or good / total > best[0]):
                    best = (good / total, computed, where, int(total), int(good))
                    source = reference
            year = f"({value} + year({birth_expr}))"
            mode = context.one(
                f"SELECT mode({year}) FROM {advanced.rows_source} WHERE {plausible}"
            )[0]
            if mode is not None:
                total, good = context.one(
                    f"SELECT count(*), count(*) FILTER (WHERE abs({year} - {mode}) <= 1) FROM "
                    f"{advanced.rows_source} WHERE {plausible}"
                )
                if total and (best is None or good / total > best[0]):
                    computed = f"({float(mode)!r} - year({birth_expr}))"
                    best = (good / total, computed, plausible, int(total), int(good))
                    source = float(mode)
            if best is None:
                continue
            share, computed, where, total, good = best
            wrong = total - good
            if share < AGE_SHARE or not wrong:
                continue
            listed = context.connection.execute(
                f"SELECT file_row_number, {display_sql(age)}, {display_sql(birth)}, "
                f"CAST({computed} AS BIGINT) "
                f"FROM {advanced.rows_source} WHERE {where} AND abs({value} - {computed}) > 1 "
                f"ORDER BY file_row_number LIMIT {LISTED}"
            ).fetchall()
            step = _age_step(age, birth, source)
            on = f"on {source.name}" if isinstance(source, Column) else f"in {int(source or 0)}"
            reason = (
                f"{age.name} matches {birth.name} (within a year) on {share:.1%} of rows; "
                f"{n(wrong)} rows disagree ({rows_list([r[0] for r in listed], wrong)}). "
                f"The step writes the age {birth.name} gives {on} on these rows only; it needs "
                "the dates read as dates (Types). If the birth date is the wrong one, leave it "
                "unticked."
            )
            advanced.add(
                "A-13",
                "S",
                [age, birth],
                f"{age.name} contradicts {birth.name} on {n(wrong)} rows",
                reason,
                step,
                cells=wrong,
                examples=[
                    (f"{row_label(r[0])}: {age.name} {r[1]}, {birth.name} {r[2]}", f"{r[3]}")
                    for r in listed
                ],
            )


def _age_step(age: Column, birth: Column, source: Column | float | None) -> m.SetWhere:
    """C-45: the age in whole years at the reference date (birthday passed or not), or the year
    of the ages minus the birth year, where the age is more than a year off."""
    born = fx.quoted(birth.name)
    if isinstance(source, Column):
        at = fx.quoted(source.name)
        passed = (
            f"CASE WHEN month({at}) * 100 + day({at}) < month({born}) * 100 + day({born}) "
            "THEN 1 ELSE 0 END"
        )
        computed = f"year({at}) - year({born}) - {passed}"
    else:
        computed = f"{int(source or 0)} - year({born})"
    test = f"abs({fx.quoted(age.name)} - ({computed})) > 1"
    return m.SetWhere(id=_id(), column=age.name, cases=[m.Case(when=test, formula=computed)])


# --- Totals (A-12) -------------------------------------------------------------------------------


def _total_priority(column: Column) -> int:
    words = set(semantics.tokens(column.name))
    if words & {"total", "sum", "subtotal", "amount", "revenue", "sales", "montant", "net"}:
        return 0
    if semantics.is_money(column.name):
        return 1
    return 2


def _relations(names: list[str]) -> list[tuple[str, tuple[int, ...]]]:
    found: list[tuple[str, tuple[int, ...]]] = []
    count = len(names)
    for i in range(count):
        for j in range(i + 1, count):
            found += [("product", (i, j)), ("sum", (i, j))]
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(count):
                if k in (i, j):
                    continue
                found.append(("product_plus", (i, j, k)))
                if k > j:
                    found.append(("sum", (i, j, k)))
    return found


def _formula(kind: str, parts: list[str]) -> str:
    if kind == "product":
        return f"{parts[0]} × {parts[1]}"
    if kind == "product_plus":
        return f"{parts[0]} × {parts[1]} + {parts[2]}"
    return " + ".join(parts)


def _compute(kind: str, arrays: list[np.ndarray]) -> np.ndarray:
    if kind == "product":
        return arrays[0] * arrays[1]
    if kind == "product_plus":
        return arrays[0] * arrays[1] + arrays[2]
    return np.sum(arrays, axis=0)


def totals(advanced: Advanced) -> None:
    columns = sorted(_measures(advanced), key=_total_priority)[:TOTAL_COLUMNS]
    if len(columns) < 3:
        return
    exprs = [number_expr(c) for c in columns]
    rows = advanced.context.connection.execute(
        f"SELECT {ROW}, {', '.join(str(e) for e in exprs)} FROM {SAMPLE} ORDER BY {ROW}"
    ).fetchall()
    if len(rows) < 50:
        return
    numbers = np.array([r[0] for r in rows], dtype=np.int64)
    data = np.array([[np.nan if v is None else v for v in r[1:]] for r in rows], dtype=float).T
    distinct = [len(np.unique(col[np.isfinite(col)])) for col in data]
    flags = [percent(c) for c in columns]
    probe = data[:, np.random.default_rng(SEED).permutation(data.shape[1])[:TOTAL_PROBE]]
    reported: list[int] = []
    for t, target in enumerate(columns):
        if distinct[t] < 10 or t in reported:
            continue
        y = data[t]
        others = [i for i in range(len(columns)) if i != t and distinct[i] >= 2]
        if any(_share(y, data[i]) >= TOTAL_SHARE for i in others):
            continue  # a copy of another column, not a total of parts
        best: tuple[int, float, str, tuple[int, ...], np.ndarray, np.ndarray] | None = None
        for kind, local in _relations([columns[i].name for i in others]):
            parts = [others[i] for i in local]
            if any(p in reported for p in parts) or not _plausible(probe, t, kind, parts):
                continue
            if len({flags[i] for i in (t, *parts)}) > 1:
                continue  # N17: a percentage is never added to counts (Votants + % Votants)
            arrays = [data[p] for p in parts]
            mask = np.isfinite(y) & np.all([np.isfinite(a) for a in arrays], axis=0)
            if mask.sum() < 50:
                continue
            if any(np.mean(a[mask] != 0) < 0.2 for a in arrays):
                continue
            expected = _compute(kind, arrays)
            close = np.abs(y - expected) <= np.maximum(TOTAL_RELATIVE * np.abs(y), TOTAL_ABSOLUTE)
            share = float(np.mean(close[mask]))
            if share < TOTAL_SHARE or share == 1.0:
                continue
            rank = (len(parts), -share)
            if best is None or rank < (best[0], -best[1]):
                best = (len(parts), share, kind, tuple(parts), mask & ~close, expected)
        if best is None:
            continue
        _, share, kind, parts, wrong, expected = best
        reported.append(t)
        named = [columns[p] for p in parts]
        factors = 3 if kind == "sum" else 2  # the terms read in the file's order, the addend last
        named = sorted(named[:factors], key=lambda c: c.position) + named[factors:]
        _total_suggestion(advanced, target, named, kind, share, wrong, expected, y, numbers)


def _plausible(probe: np.ndarray, t: int, kind: str, parts: list[int]) -> bool:
    """A relation worth checking on the whole sample: it holds on most of 1,000 random rows."""
    y = probe[t]
    expected = _compute(kind, [probe[p] for p in parts])
    mask = np.isfinite(y) & np.isfinite(expected)
    if mask.sum() < 20:
        return True  # too few rows to judge: the full check decides
    close = np.abs(y - expected) <= np.maximum(TOTAL_RELATIVE * np.abs(y), TOTAL_ABSOLUTE)
    return float(np.mean(close[mask])) >= TOTAL_SHARE - 0.1


def _share(y: np.ndarray, x: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(x)
    if not mask.any():
        return 0.0
    close = np.abs(y[mask] - x[mask]) <= np.maximum(TOTAL_RELATIVE * np.abs(y[mask]), 1e-9)
    return float(np.mean(close))


def decimals_of(values: np.ndarray) -> int:
    """The fewest decimals (0 to 6) that write at least 99 % of the values exactly."""
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 2
    for places in range(7):
        if np.mean(np.abs(finite - np.round(finite, places)) < 1e-9) >= 0.99:
            return places
    return 6


def _total_step(target: Column, parts: list[Column], kind: str, y: np.ndarray) -> m.SetWhere:
    """C-45: the total recalculated (rounded as the totals are written) where it differs from
    its parts by more than 1 % (and 0.011)."""
    names = [fx.quoted(p.name) for p in parts]
    if kind == "product":
        computed = f"{names[0]} * {names[1]}"
    elif kind == "product_plus":
        computed = f"{names[0]} * {names[1]} + {names[2]}"
    else:
        computed = " + ".join(names)
    total = fx.quoted(target.name)
    test = (
        f"abs({total} - ({computed})) > greatest({TOTAL_RELATIVE!r} * abs({total}), "
        f"{TOTAL_ABSOLUTE!r})"
    )
    value = f"round({computed}, {decimals_of(y)})"
    return m.SetWhere(id=_id(), column=target.name, cases=[m.Case(when=test, formula=value)])


def _total_suggestion(
    advanced: Advanced,
    target: Column,
    parts: list[Column],
    kind: str,
    share: float,
    wrong: np.ndarray,
    expected: np.ndarray,
    y: np.ndarray,
    numbers: np.ndarray,
) -> None:
    positions = np.flatnonzero(wrong)
    count = len(positions)
    formula = _formula(kind, [p.name for p in parts])
    listed = [int(numbers[p]) for p in positions[:LISTED].tolist()]
    units = [
        f.columns[0]
        for f in advanced.found
        if f.rule == "A-05" and f.columns and f.columns[0] in {p.name for p in parts}
    ]
    warning = (
        f" {', '.join(units)} also has values in another unit (A-05): tick that fix too, it "
        "runs before this one, or these rows get a wrong total."
        if units
        else ""
    )
    reason = (
        f"{target.name} = {formula} (within 1 %) on {share:.1%} of rows; "
        f"{scaled(advanced, count)} rows differ ({rows_list(listed, count)}): a wrong total, "
        f"or a part in another unit. The step recalculates {target.name} on the rows where it "
        f"differs by more than 1 %; it needs the parts read as numbers (Types).{warning}"
    )
    step = _total_step(target, parts, kind, y)
    advanced.add(
        "A-12",
        "S",
        [target, *parts],
        f"{target.name} is not {formula} on {scaled(advanced, count)} rows",
        reason,
        step,
        cells=count,
        examples=[
            (
                f"{row_label(int(numbers[p]))}: {target.name} {_number_text(float(y[p]))}",
                f"{formula} = {_number_text(float(expected[p]))}",
            )
            for p in positions[:LISTED].tolist()
        ],
    )


# --- A value of the group on each of its rows (A-18) ---------------------------------------------

HEADER_SHARE = 0.95  # N33: the measure is constant in this share of the key's groups of 2 rows +
HEADER_GROUPS = 20  # … with at least this many such groups
HEADER_KEYS = 3  # keys tried at most
HEADER_ROLES = ("identifier", "order_key")  # a record's key (not a customer's: a dimension)
NOT_SUMMED = ("age", "rating", "year", "month", "weekday", "hour", "duration")


def _header_keys(advanced: Advanced) -> list[Column]:
    """N33: a key shared by rows (dvf's id_mutation): an identifier by its role or its name,
    repeated, filled on nearly every row."""
    rows = max(advanced.context.rows, 1)
    found = []
    for column in advanced.context.columns:
        view = column.view
        role = view.role if view is not None else None
        code = semantics.code_by_name(column.name)
        named = code is not None and code.kind == "identifier"
        if role == "customer_key" or not (role in HEADER_ROLES or named):
            continue
        if column.nonnull < 0.95 * rows or not HEADER_GROUPS <= column.distinct <= 0.9 * rows:
            continue
        found.append(column)
    return sorted(found, key=lambda c: c.distinct)[:HEADER_KEYS]


def _summed(column: Column) -> bool:
    """A measure one adds up over rows: not a percentage, a ratio, an age, a rating."""
    view_role = column.view.role if column.view else None
    role = column.role.id if column.role is not None else None
    if percent(column) or semantics.is_ratio(column.name):
        return False
    return role not in NOT_SUMMED and view_role not in NOT_SUMMED


def repeated_values(advanced: Advanced) -> None:
    """N33 (A-18): a measure equal on every row of a key's group (dvf: valeur_fonciere on each
    line of its id_mutation, 226 lines for one sale) is a value of the group: a sum over rows counts
    it once per row. Information only (no step can keep it on one row of each group)."""
    measures = [c for c in _measures(advanced) if _summed(c) and c.distinct >= 10]
    if not measures:
        return
    connection = advanced.context.connection
    for key in _header_keys(advanced):
        name = ident(key.name)
        tests = [c for c in measures if c is not key]
        if not tests:
            continue
        flags = ", ".join(
            f"count(DISTINCT {ident(c.name)}) = 1 AND count({ident(c.name)}) = count(*) AS c{i}"
            for i, c in enumerate(tests)
        )
        counts = ", ".join(
            f"count(*) FILTER (WHERE c{i}), sum(n) FILTER (WHERE c{i})" for i in range(len(tests))
        )
        row = connection.execute(
            f"SELECT count(*), {counts} FROM (SELECT count(*) AS n, {flags} FROM {SAMPLE} "
            f"WHERE {name} IS NOT NULL GROUP BY {name} HAVING count(*) > 1)"
        ).fetchone()
        assert row is not None
        groups = int(row[0])
        if groups < HEADER_GROUPS:
            continue
        for index, column in enumerate(tests):
            constant, cells = int(row[1 + 2 * index]), int(row[2 + 2 * index] or 0)
            if constant < HEADER_SHARE * groups:
                continue
            _repeated_suggestion(advanced, key, column, groups, constant, cells)


def _repeated_suggestion(
    advanced: Advanced, key: Column, column: Column, groups: int, constant: int, cells: int
) -> None:
    measure, name = ident(column.name), ident(key.name)
    top = advanced.context.connection.execute(
        f"SELECT CAST({name} AS VARCHAR), count(*), {as_text(column.name, column.dtype)} "
        f"FROM {SAMPLE} WHERE {name} IS NOT NULL GROUP BY ALL HAVING count(*) > 1 AND "
        f"count(DISTINCT {measure}) = 1 ORDER BY 2 DESC, 1 LIMIT 3"
    ).fetchall()
    if not top:
        return
    largest, size, value = top[0]
    advanced.add(
        "A-18",
        "I",
        [column, key],
        f"{column.name} repeats on every row of its {key.name}",
        f"{column.name} is the same on every row of its {key.name} ({n(int(size))} rows for "
        f"{largest}, {value} each; {constant / groups:.0%} of the {scaled(advanced, groups)} "
        f"{key.name} with several rows): a value of the {key.name}, not of the row. A sum over "
        f"rows counts it once per row: sum it once per {key.name} (group by {key.name} first, "
        "then add up). Not changed.",
        cells=cells,
        examples=[
            (f"{key.name} {r[0]}: {n(int(r[1]))} rows", f"{column.name} {r[2]}") for r in top
        ],
        key="header",
    )


# --- Two columns that say the same thing (A-19) ---------------------------------------------------

REDUNDANT_ROWS = 50  # N34: filled rows at least
REDUNDANT_BATCH = 40  # columns (or pairs) per query: a wide file stays a few short queries
REDUNDANT_PAIRS = 200  # … and past this many pairs to compare, their partitions first
REDUNDANT_UNIQUE = 0.95  # … a column with more distinct values names its rows (an identifier)
EPOCH = (1e8, 4e9)  # seconds since 1970 between 1973 and 2096


def _redundant_candidates(advanced: Advanced) -> list[Column]:
    found = []
    for column in advanced.context.columns:
        view = column.view
        role = view.role if view is not None else None
        if (view is not None and view.type == "identifier") or role in semantics.KEY_ROLES:
            continue
        if column.nonnull < REDUNDANT_ROWS or column.distinct < 2:
            continue
        found.append(column)
    return found


def _batches(items: list[Any]) -> list[list[Any]]:
    return [items[i : i + REDUNDANT_BATCH] for i in range(0, len(items), REDUNDANT_BATCH)]


def _one_row(advanced: Advanced, parts: str, source: str = SAMPLE) -> tuple[Any, ...]:
    row = advanced.context.connection.execute(f"SELECT {parts} FROM {source}").fetchone()
    assert row is not None
    return row


def redundant_columns(advanced: Advanced) -> None:
    """N34 (A-19): columns one to one on every row of the sample, with the same blanks (adult's
    education and education-num, chicago's timestamp and its label): the same information twice.
    A copy, or a text or a number that reads back to the other column (a date written as text,
    seconds since 1970), is removed if ticked (S, `drop_columns`); otherwise an information:
    which one to keep (codes or labels) is a choice."""
    columns = _redundant_candidates(advanced)
    # the profile first (no query): one to one columns have the same blanks on the whole file and
    # about as many distinct values (an estimate: within a quarter)
    by_filled: dict[int, list[Column]] = defaultdict(list)
    for column in columns:
        by_filled[column.nonnull].append(column)
    columns = [
        c
        for members in by_filled.values()
        for c in members
        if any(o is not c and 0.8 <= o.distinct / c.distinct <= 1.25 for o in members)
    ]
    if len(columns) < 2:
        return
    by_size: dict[tuple[int, int], list[Column]] = defaultdict(list)
    for batch in _batches(columns):
        row = _one_row(
            advanced,
            ", ".join(f"count(DISTINCT {ident(c.name)}), count({ident(c.name)})" for c in batch),
        )
        for index, column in enumerate(batch):
            distinct, filled = int(row[2 * index]), int(row[2 * index + 1])
            if distinct >= 2 and filled >= REDUNDANT_ROWS and distinct < REDUNDANT_UNIQUE * filled:
                by_size[(distinct, filled)].append(column)
    pairs = [
        (a, b, size[0])
        for size, members in by_size.items()
        for i, a in enumerate(members)
        for b in members[i + 1 :]
    ]
    if len(pairs) > REDUNDANT_PAIRS:
        # many alike columns (a wide file of categories): first the rows each value groups, one
        # number per column (the first row of each row's value, hashed with the row and summed,
        # never XOR-ed); one to one columns part the rows alike, only those are compared
        signature: dict[str, int] = {}
        alike = [c for members in by_size.values() if len(members) > 1 for c in members]
        for batch in _batches(alike):
            firsts = ", ".join(
                f"min({ROW}) OVER (PARTITION BY {ident(c.name)}) AS f{i}"
                for i, c in enumerate(batch)
            )
            sums = ", ".join(f"sum(hash({ROW}, f{i}))" for i in range(len(batch)))
            row = _one_row(advanced, sums, f"(SELECT {ROW}, {firsts} FROM {SAMPLE})")
            signature.update({c.name: int(row[i]) for i, c in enumerate(batch)})
        pairs = [(a, b, d) for a, b, d in pairs if signature[a.name] == signature[b.name]]
    if not pairs:
        return
    checked: list[Any] = []
    for batch in _batches(pairs):
        checked += _one_row(
            advanced,
            ", ".join(
                f"count(DISTINCT ({ident(a.name)}, {ident(b.name)})), "
                f"count(*) FILTER (WHERE ({ident(a.name)} IS NULL) <> ({ident(b.name)} IS NULL))"
                for a, b, _ in batch
            ),
        )
    parent: dict[str, str] = {}

    def find(name: str) -> str:
        while parent.get(name, name) != name:
            name = parent[name]
        return name

    values: dict[str, int] = {}
    for index, (a, b, distinct) in enumerate(pairs):
        if int(checked[2 * index]) == distinct and int(checked[2 * index + 1]) == 0:
            root_a, root_b = find(a.name), find(b.name)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
            values[a.name] = values[b.name] = distinct
    groups: dict[str, list[Column]] = defaultdict(list)
    for column in columns:
        if column.name in values:
            groups[find(column.name)].append(column)
    for members in groups.values():
        members.sort(key=lambda c: c.position)
        _redundant_suggestion(advanced, members, values[members[0].name])


def _reads_back(advanced: Advanced, keep: Column, drop: Column) -> str | None:
    """How `drop` is `keep` written another way, checked on every sampled row: a copy, a date
    written as text, seconds since 1970; None when it is not (codes and labels)."""
    a, b = ident(keep.name), ident(drop.name)
    tests: list[tuple[str, str]] = [
        (
            "a copy of {keep}",
            f"{as_text(keep.name, keep.dtype)} IS DISTINCT FROM {as_text(drop.name, drop.dtype)}",
        ),
    ]
    if is_temporal(keep.dtype):
        stamp = f"CAST({a} AS TIMESTAMP)"
        text_date = date_expr(advanced, drop) if is_text(drop.dtype) else None
        if text_date is not None:
            tests.append(
                (
                    "{keep} written as text",
                    f"CAST({stamp} AS DATE) IS DISTINCT FROM ({text_date})",
                )
            )
        if is_number(drop.dtype):
            epoch = f"make_timestamp(CAST({b} AS BIGINT) * 1000000)"
            tests.append(
                (
                    "{keep} in seconds since 1970",
                    f"{b} NOT BETWEEN {EPOCH[0]} AND {EPOCH[1]} "
                    f"OR {epoch} IS DISTINCT FROM {stamp}",
                )
            )
    for how, differs in tests:
        (count,) = advanced.context.one(
            f"SELECT count(*) FILTER (WHERE {a} IS NOT NULL AND ({differs})) FROM {SAMPLE}"
        )
        if not int(count):
            return how
    return None


def _redundant_suggestion(advanced: Advanced, members: list[Column], distinct: int) -> None:
    first, others = members[0], members[1:]
    for keep in sorted(members, key=lambda c: (not is_temporal(c.dtype), c.position)):
        for drop in (c for c in members if c is not keep):
            how = _reads_back(advanced, keep, drop) if len(members) == 2 else None
            if how is None:
                continue
            how = how.format(keep=keep.name)
            advanced.add(
                "A-19",
                "S",
                [drop, keep],
                f"Remove {drop.name}: {how}",
                f"{drop.name} is {how} (checked on every row of the sample, "
                f"{n(distinct)} values one to one, the same blanks): the same information twice. "
                f"Removed if you tick it; {keep.name} stays. A filter or a KPI may name it.",
                m.DropColumns(id=_id(), columns=[drop.name]),
                examples=[(drop.name, None)],
                key="copy",
            )
            return
    names = ", ".join(c.name for c in others)
    advanced.add(
        "A-19",
        "I",
        members,
        f"{names} {'holds' if len(others) == 1 else 'hold'} the same information as {first.name}",
        f"{', '.join(c.name for c in members)}: {n(distinct)} values one to one on every row of "
        f"the sample, with the same blanks ({first.name} → {others[0].name}). One can go if you "
        "keep the correspondence elsewhere; which one to keep (codes or labels) is a choice. "
        "Not changed.",
        examples=[(c.name, None) for c in members],
        key="same",
    )


# --- Near-duplicates (A-15) ----------------------------------------------------------------------


def _trivial(column: Column, rows: int) -> str | None:
    """Why a column does not count when comparing rows: an identifier, free text, a timestamp."""
    view = column.view
    if view is not None and view.type == "identifier":
        return "identifier"
    code = semantics.code_by_name(column.name)
    if code is not None and code.kind == "identifier" and column.distinct >= 0.9 * max(rows, 1):
        return "identifier"
    words = set(semantics.tokens(column.name))
    if (view is not None and view.role == "free_text") or words & FREE_WORDS:
        return "free text"
    if semantics.date_tier(column.name) == 3:
        return "update time"
    return None


def _anchor(advanced: Advanced, column: Column) -> bool:
    """C-26: a column that says which record a row is: an event date, or a key shared by rows
    (an order, a customer), with enough values to make a chance match unlikely."""
    if column.distinct < NEAR_ANCHOR:
        return False
    if date_expr(advanced, column) is not None:
        return semantics.date_tier(column.name) != 4  # a birth date is a trait, not an event
    view = column.view
    if view is not None and view.role in semantics.KEY_ROLES:
        return True
    code = semantics.code_by_name(column.name)
    return code is not None and code.kind == "identifier"


def _normalized(column: Column) -> str:
    name = ident(column.name)
    if is_text(column.dtype):  # separators dropped; symbols kept ('<=50K' is not '>50K')
        spaced = f"regexp_replace({name}, {lit(tx.SQL_SPACE_RUN)}, ' ', 'g')"
        return f"regexp_replace(lower(strip_accents({spaced})), {lit(SEPARATORS)}, '', 'g')"
    return f"CAST({name} AS VARCHAR)"  # numbers exactly: 40.71143 is not 40.71122


def near_duplicates(advanced: Advanced) -> None:
    context = advanced.context
    if context.rows < 3:
        return
    trivial = {c.name: why for c in context.columns if (why := _trivial(c, context.rows))}
    keys = [c for c in context.columns if c.name not in trivial and c.nonnull]
    if len(keys) < NEAR_KEYS or not any(_anchor(advanced, c) for c in keys):
        return  # C-26: without a date or a key, equal rows may be two customers alike
    empty = " AND ".join(basic._empty(c) for c in context.columns)
    normalized = ", ".join(_normalized(c) for c in keys)
    every = ", ".join(ident(c.name) for c in context.columns)
    rows, near, near_groups = _groups(advanced, normalized, empty)
    if (rows - near) > NEAR_SHARE * rows or near == rows:
        return  # too many to be typos, or none (then no exact copy either)
    exact, exact_groups = _groups(advanced, every, empty)[1:]
    extra = exact - near
    if extra <= 0:
        return
    pairs = _near_pairs(near_groups, exact_groups)
    examples = _near_examples(advanced, pairs)
    ignored = sorted(trivial)
    apart = (
        f" apart from {', '.join(ignored[:4])} ({', '.join(sorted(set(trivial.values())))})"
        if ignored
        else ""
    )
    reason = (
        f"{n(extra)} rows repeat another row once case, accents, spaces and punctuation are "
        f"ignored{apart}: the same record entered twice? The most complete row is kept. Counted "
        "on normalized values: tick the Basic spacing and case fixes first, or fewer rows will "
        "match."
    )
    advanced.add(
        "A-15",
        "S",
        [],
        f"Remove {n(extra)} near-duplicate rows",
        reason,
        m.DropDuplicates(id=_id(), keys=[c.name for c in keys], keep="most_complete"),
        rows=extra,
        examples=examples,
        key="near",
        impact="rows",
    )


def _groups(advanced: Advanced, key: str, empty: str) -> tuple[int, int, list[list[int]]]:
    """N44: the rows with a value (not empty), how many distinct `row(key)` they hold, and the
    rows of each value held twice or more: one hash aggregation, where two `count(DISTINCT row(…))`
    and two windows sorting every row on all its columns took several times longer on wide
    files."""
    rows, distinct, groups = advanced.context.one(
        "SELECT coalesce(sum(k), 0), count(*), list(r) FILTER (WHERE k > 1) FROM (SELECT "
        f"count(*) AS k, list(file_row_number) AS r FROM {advanced.rows_source} WHERE NOT "
        f"({empty}) GROUP BY row({key}))"
    )
    return int(rows), int(distinct), [[int(r) for r in group] for group in groups or []]


def _near_pairs(near: list[list[int]], exact: list[list[int]]) -> list[tuple[int, int]]:
    """The first three rows (in the file's order) that repeat an earlier row once normalized
    without being an exact copy of an earlier row, each with the first row of its group."""
    first_row = {r: min(rows) for rows in near for r in rows}
    first_copy = {r: min(rows) for rows in exact for r in rows}
    found = [r for r, first in first_row.items() if r != first and first_copy.get(r, r) == r]
    return [(r, first_row[r]) for r in sorted(found)[:3]]


def _near_examples(advanced: Advanced, pairs: list[tuple[int, int]]) -> list[tuple[str, str]]:
    """'row 53: brands 'X'' → 'same as row 52 (brands 'x')', the rows read in one query."""
    if not pairs:
        return []
    columns = advanced.context.columns
    texts = ", ".join(as_text(c.name, c.dtype) for c in columns)
    wanted = sorted({r for pair in pairs for r in pair})
    found = {
        int(row[0]): row[1:]
        for row in advanced.context.connection.execute(
            f"SELECT file_row_number, {texts} FROM {advanced.rows_source} WHERE file_row_number "
            f"IN ({', '.join(str(r) for r in wanted)}) ORDER BY file_row_number"
        ).fetchall()
    }
    examples = []
    for row, first in pairs:
        if row not in found or first not in found:
            examples.append((row_label(row), row_label(first)))
            continue
        original, copy = found[min(row, first)], found[max(row, first)]
        differ = [
            (c.name, a, b) for c, a, b in zip(columns, original, copy, strict=True) if a != b
        ][:3]
        shown = ", ".join(f"{name} {q(str(b))}" for name, _, b in differ)
        before = ", ".join(f"{name} {q(str(a))}" for name, a, _ in differ)
        examples.append((f"{row_label(row)}: {shown}", f"same as {row_label(first)} ({before})"))
    return examples


# --- Entry ---------------------------------------------------------------------------------------


def profile(
    connection: duckdb.DuckDBPyConnection, parquet_path: str, views: list[ColumnView]
) -> tuple[Context, str]:
    context = basic.profile(connection, f"read_parquet({lit(parquet_path)})", views)
    return context, f"read_parquet({lit(parquet_path)}, file_row_number = true)"


def run(
    context: Context, rows_source: str, basic_found: list[CleanSuggestion]
) -> list[CleanSuggestion]:
    # they import this module
    from quantilo.cleaning.rules import distributions, entities, formats, linkage, shapes

    context.close()  # the Basic rules' table of values: not read here
    advanced = Advanced(context, rows_source, basic_found)
    if not context.rows or not context.columns:
        return []
    checks: list[tuple[str, Callable[[Advanced], None]]] = [
        ("A-14", shapes.dates),
        ("A-01 to A-03", spellings),
        ("A-21", _synonyms),
        ("A-04", shapes.codes),
        ("F-02", formats.check_digits),
        ("F-04", formats.countries),
        ("F-01", formats.phones),
        ("F-05", formats.quantities),
    ]
    sampled: list[tuple[str, Callable[[Advanced], None]]] = [
        ("A-05", units),
        ("A-07", distributions.noise),
        ("A-06", distributions.outliers),
        ("A-15", near_duplicates),
        ("A-16", linkage.fellegi_sunter),
        ("A-20", entities.entities),
        ("A-17", linkage.mostly_empty),
        ("A-18", repeated_values),
        ("A-19", redundant_columns),
        ("A-10", dependencies),
        ("A-11", date_order),
        ("A-12", totals),
        ("A-13", ages),
        ("A-08", distributions.benford),
        ("A-09", distributions.last_digits),
    ]
    _checks(advanced, checks)
    make_sample(advanced)
    try:
        _checks(advanced, sampled)
    finally:
        context.connection.execute(f"DROP TABLE IF EXISTS {SAMPLE}")
    _budget_note(advanced)
    return sorted(advanced.found, key=lambda s: s.order)


def _synonyms(advanced: Advanced) -> None:
    """N30 (A-21); imported here because rules/synonyms.py imports this module."""
    from quantilo.cleaning.rules import synonyms

    synonyms.business(advanced)


def _checks(advanced: Advanced, checks: list[tuple[str, Callable[[Advanced], None]]]) -> None:
    for rule, check in checks:
        if advanced.out_of_time():  # N44: the time budget spent, the rest is named in a note
            advanced.late_checks.append(rule)
        else:
            check(advanced)


def _budget_note(advanced: Advanced) -> None:
    """N44: an information naming what the bounds left out on a wide or slow file."""
    columns = [*advanced.unchecked, *advanced.late_columns]
    if not columns and not advanced.late_checks:
        return
    seconds = f"{TIME_BUDGET:g} s"
    parts = []
    if advanced.unchecked:
        parts.append(
            f"{n(len(advanced.unchecked))} category text columns not checked for near spellings "
            f"(A-01 to A-03): there are more than {SPELLING_COLUMNS}, and the {SPELLING_COLUMNS} "
            "with the fewest distinct values per row were checked."
        )
    if advanced.late_columns:
        parts.append(
            f"{n(len(advanced.late_columns))} more not checked for near spellings: the time limit "
            f"({seconds}) was reached."
        )
    if advanced.late_checks:
        parts.append(
            f"Checks not run, the time limit ({seconds}) reached: "
            + ", ".join(advanced.late_checks)
            + "."
        )
    shown = ", ".join(columns[:LISTED]) + (
        f" and {n(len(columns) - LISTED)} more" if len(columns) > LISTED else ""
    )
    if columns:
        parts.append(f"Open them one by one to check them: {shown}.")
    why = "time limit" if advanced.late_columns or advanced.late_checks else "too many columns"
    title = (
        f"{n(len(columns))} columns not checked ({why}): open them one by one"
        if columns
        else f"{len(advanced.late_checks)} checks not run (time limit)"
    )
    advanced.add(
        "A-00",
        "I",
        [],
        title,
        " ".join(parts),
        examples=[(name, None) for name in columns],
        key="budget",
    )
