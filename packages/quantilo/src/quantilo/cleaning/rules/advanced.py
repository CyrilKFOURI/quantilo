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
  the birth date.
- A-15: rows equal once normalized, apart from an identifier or a free text.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import duckdb
import numpy as np

from quantilo import semantics
from quantilo.catalog import ColumnView
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
from quantilo.cleaning.suggestion import CleanSuggestion, Example, Mode
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
UNIT_FACTORS = (100, 1000)
UNIT_TOLERANCE = 0.15  # log10: the two groups' quantiles are this close once rescaled
UNIT_SHARE = (0.01, 0.30)
UNIT_MIN_VALUES = 10
UNIT_SPLITS = 30  # the widest gaps between distinct values tried as the split
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
FREE_WORDS = {"note", "notes", "comment", "comments", "remark", "remarks", "message", "feedback"}
ORDER = {
    "A-01": 200,
    "A-02": 210,
    "A-03": 220,
    "A-05": 230,
    "A-15": 240,
    "A-10": 300,
    "A-11": 310,
    "A-12": 320,
    "A-13": 330,
}
NULL_TOKENS = basic.SURE_TOKENS | basic.WEAK_TOKENS
START_WORDS = {"start", "begin", "beginning", "from", "open", "opened", "checkin", "pickup"}
START_WORDS |= {"arrival", "debut", "hire", "hired", "entry", "admission", "admitted"}
END_WORDS = {"end", "ending", "finish", "finished", "until", "close", "closed", "checkout"}
END_WORDS |= {"dropoff", "departure", "fin", "termination", "terminated", "exit", "discharge"}
END_WORDS |= {"discharged", "expiry", "expiration", "expires", "due"}
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
    sampled: bool = False
    sample_rows: int = 0

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
        impact: m.Impact = "values",
    ) -> None:
        if step is not None:
            step.origin = "suggested"
            step.rule = rule
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
    return found


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
    advanced.add(
        rule,
        "S",
        [column],
        title,
        reason,
        m.MapValues(id=_id(), column=column.name, map=basic._capped(mapping, weights)),
        cells=cells,
        examples=examples,
    )


def spellings(advanced: Advanced) -> None:
    for column, stages in _spelling_columns(advanced):
        units = units_of(stages)
        if len(units) < 2:
            continue
        person = basic._person_names(column)
        groups = Groups(units)
        # A-01: fingerprint (word order, punctuation)
        key = (lambda v: ph.fingerprint(v, ordered=True)) if person else ph.fingerprint
        for _, indexes in _buckets(units, key).items():
            for index in indexes[1:]:
                if _same_numbers(units[indexes[0]].key, units[index].key):
                    groups.union(indexes[0], index)
        first = {i: groups.find(i) for i in range(len(units))}
        _emit(advanced, "A-01", column, groups, {i: i for i in range(len(units))}, person)
        # A-02: 2-letter fingerprint (spaces and separators)
        for _, indexes in _buckets(units, ph.ngram_fingerprint).items():
            for index in indexes[1:]:
                if _same_numbers(units[indexes[0]].key, units[index].key):
                    groups.union(indexes[0], index)
        second = {i: groups.find(i) for i in range(len(units))}
        _emit(advanced, "A-02", column, groups, first, person)
        # A-03: a rare spelling close to a frequent one (not on names of people)
        if not person:
            _neighbours(advanced, column, groups)
            _emit(advanced, "A-03", column, groups, second, person)


def _buckets(units: list[Unit], key: Any) -> dict[str, list[int]]:
    found: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        if len(unit.key) >= 2:
            found[key(unit.key)].append(index)
    return {k: v for k, v in found.items() if len(v) > 1 and k}


HOW = {
    "A-01": "same words once case, accents, punctuation and word order are ignored (fingerprint)",
    "A-02": "same letters once spaces and separators are removed (2-letter fingerprint)",
    "A-03": "a rare spelling one or two letters away from a frequent one (same sound, letters "
    "swapped, or one letter missing in a long word)",
}


def _emit(
    advanced: Advanced,
    rule: str,
    column: Column,
    groups: Groups,
    before: dict[int, int],
    person: bool,
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
        _merge_suggestion(advanced, rule, column, groups, roots, HOW[rule], person)


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
    accepted: set[tuple[int, int]] = set()
    alternative = 0
    for value, count, size, top in zip(
        unique.tolist(), counts.tolist(), group_size.tolist(), group_top.tolist(), strict=True
    ):
        owner, answer = divmod(value, nb)
        if count == top and owner not in majority:
            majority[owner] = answer
        elif count >= FD_SECOND_ROWS and count / size >= FD_SECOND_SHARE:
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
        _dependency_suggestion(advanced, dependency, coded, raw, numbers)


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


def _dependency_suggestion(
    advanced: Advanced,
    dependency: Dependency,
    coded: dict[str, tuple[np.ndarray, list[str]]],
    raw: dict[str, list[Any]],
    numbers: np.ndarray,
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
    per_key: dict[int, int] = defaultdict(int)
    for position in positions.tolist():
        per_key[int(codes_a[position])] += 1
    worst = max(per_key, key=lambda k: per_key[k])
    worst_rows = int(((codes_a == worst) & (codes_b >= 0)).sum())
    worst_share = 1 - per_key[worst] / max(worst_rows, 1)
    listed = [int(numbers[p]) for p in positions[:LISTED].tolist()]
    examples = [
        (
            f"{row_label(int(numbers[p]))}: {a.name} {q(str(raw[a.name][p]))}, "
            f"{b.name} {q(str(raw[b.name][p]))}",
            labels_b[int(expected[p])],
        )
        for p in positions[:LISTED].tolist()
    ]
    count = len(positions)
    reason = (
        f"{a.name} determines {b.name} on {dependency.share:.1%} of rows: "
        f"{q(labels_a[worst])} has {b.name} {q(labels_b[dependency.majority[worst]])} on "
        f"{worst_share:.1%} of its rows. {scaled(advanced, count)} rows disagree "
        f"({rows_list(listed, count)}): a typo or a wrong value in one of the two columns. "
        "Information only: check these rows (no step changes a value according to another "
        "column yet)."
    )
    advanced.add(
        "A-10",
        "I",
        [a, b],
        f"{b.name} disagrees with {a.name} on {scaled(advanced, count)} rows",
        reason,
        rows=0,
        cells=count,
        examples=examples,
    )


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


def date_order(advanced: Advanced) -> None:
    dates = _dates(advanced)
    context = advanced.context
    for first, first_expr in dates:
        for second, second_expr in dates:
            if first is second or not _ordered(first, second):
                continue
            both = f"({first_expr}) IS NOT NULL AND ({second_expr}) IS NOT NULL"
            total, wrong = context.one(
                f"SELECT count(*) FILTER (WHERE {both}), count(*) FILTER (WHERE {both} AND "
                f"({first_expr}) > ({second_expr})) FROM {advanced.rows_source}"
            )
            total, wrong = int(total), int(wrong)
            if not wrong or not total or 1 - wrong / total < ORDER_SHARE:
                continue
            listed = context.connection.execute(
                f"SELECT file_row_number, {display_sql(first)}, {display_sql(second)} "
                f"FROM {advanced.rows_source} WHERE {both} "
                f"AND ({first_expr}) > ({second_expr}) ORDER BY file_row_number LIMIT {LISTED}"
            ).fetchall()
            reason = (
                f"{first.name} comes before {second.name} on {1 - wrong / total:.1%} of rows, "
                f"after it on {n(wrong)} ({rows_list([r[0] for r in listed], wrong)}): swapped "
                "or mistyped dates. Information only: check these rows."
            )
            advanced.add(
                "A-11",
                "I",
                [first, second],
                f"{second.name} before {first.name} on {n(wrong)} rows",
                reason,
                cells=wrong,
                examples=[
                    (f"{row_label(r[0])}: {first.name} {r[1]}", f"{second.name} {r[2]}")
                    for r in listed
                ],
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
            plausible = f"{value} BETWEEN 0 AND 130 AND ({birth_expr}) IS NOT NULL"
            for _reference, reference_expr in references:
                computed = f"floor(date_diff('day', {birth_expr}, {reference_expr}) / 365.25)"
                where = f"{plausible} AND ({reference_expr}) IS NOT NULL"
                total, good = context.one(
                    f"SELECT count(*), count(*) FILTER (WHERE abs({value} - {computed}) <= 1) "
                    f"FROM {advanced.rows_source} WHERE {where}"
                )
                if total and (best is None or good / total > best[0]):
                    best = (good / total, computed, where, int(total), int(good))
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
            reason = (
                f"{age.name} matches {birth.name} (within a year) on {share:.1%} of rows; "
                f"{n(wrong)} rows disagree ({rows_list([r[0] for r in listed], wrong)}). "
                "Information only: check these rows."
            )
            advanced.add(
                "A-13",
                "I",
                [age, birth],
                f"{age.name} contradicts {birth.name} on {n(wrong)} rows",
                reason,
                cells=wrong,
                examples=[
                    (f"{row_label(r[0])}: {age.name} {r[1]}, {birth.name} {r[2]}", f"{r[3]}")
                    for r in listed
                ],
            )


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
    reason = (
        f"{target.name} = {formula} (within 1 %) on {share:.1%} of rows; "
        f"{scaled(advanced, count)} rows differ ({rows_list(listed, count)}): a wrong total, "
        "or a part in another unit. Information only: check these rows."
    )
    advanced.add(
        "A-12",
        "I",
        [target, *parts],
        f"{target.name} is not {formula} on {scaled(advanced, count)} rows",
        reason,
        cells=count,
        examples=[
            (
                f"{row_label(int(numbers[p]))}: {target.name} {_number_text(float(y[p]))}",
                f"{formula} = {_number_text(float(expected[p]))}",
            )
            for p in positions[:LISTED].tolist()
        ],
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
    empty = " AND ".join(basic._blank(c) for c in context.columns)
    normalized = ", ".join(_normalized(c) for c in keys)
    every = ", ".join(ident(c.name) for c in context.columns)
    rows, near, exact = (
        int(v)
        for v in context.one(
            f"SELECT count(*), count(DISTINCT row({normalized})), count(DISTINCT row({every})) "
            f"FROM {context.source} WHERE NOT ({empty})"
        )
    )
    extra = exact - near
    if extra <= 0 or (rows - near) > NEAR_SHARE * rows:
        return
    pairs = context.connection.execute(
        f"SELECT file_row_number, first_row FROM (SELECT file_row_number, "
        f"min(file_row_number) OVER (PARTITION BY row({normalized})) AS first_row, "
        f"min(file_row_number) OVER (PARTITION BY row({every})) AS first_copy "
        f"FROM {advanced.rows_source} WHERE NOT ({empty})) "
        f"WHERE file_row_number <> first_row AND first_copy = file_row_number "
        f"ORDER BY file_row_number LIMIT 3"
    ).fetchall()
    examples = [_near_example(advanced, int(row), int(first)) for row, first in pairs]
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


def _near_example(advanced: Advanced, row: int, first: int) -> tuple[str, str]:
    columns = advanced.context.columns
    texts = ", ".join(as_text(c.name, c.dtype) for c in columns)
    found = advanced.context.connection.execute(
        f"SELECT file_row_number, {texts} FROM {advanced.rows_source} WHERE file_row_number IN "
        f"({row}, {first}) ORDER BY file_row_number"
    ).fetchall()
    if len(found) < 2:
        return (row_label(row), row_label(first))
    original, copy = found[0][1:], found[1][1:]
    differ = [(c.name, a, b) for c, a, b in zip(columns, original, copy, strict=True) if a != b][:3]
    shown = ", ".join(f"{name} {q(str(b))}" for name, _, b in differ)
    before = ", ".join(f"{name} {q(str(a))}" for name, a, _ in differ)
    return (f"{row_label(row)}: {shown}", f"same as {row_label(first)} ({before})")


# --- Entry ---------------------------------------------------------------------------------------


def profile(
    connection: duckdb.DuckDBPyConnection, parquet_path: str, views: list[ColumnView]
) -> tuple[Context, str]:
    context = basic.profile(connection, f"read_parquet({lit(parquet_path)})", views)
    return context, f"read_parquet({lit(parquet_path)}, file_row_number = true)"


def run(
    context: Context, rows_source: str, basic_found: list[CleanSuggestion]
) -> list[CleanSuggestion]:
    advanced = Advanced(context, rows_source, basic_found)
    if not context.rows or not context.columns:
        return []
    spellings(advanced)
    make_sample(advanced)
    try:
        units(advanced)
        near_duplicates(advanced)
        dependencies(advanced)
        date_order(advanced)
        totals(advanced)
        ages(advanced)
    finally:
        context.connection.execute(f"DROP TABLE IF EXISTS {SAMPLE}")
    return sorted(advanced.found, key=lambda s: s.order)
