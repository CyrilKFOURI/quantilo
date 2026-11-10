"""The Basic layer (study § 2.1, B-01 to B-19): computed when Clean opens, a few seconds at most.

Text rules work on each column's distinct values (50,000 at most, C-14), each one on the value
the previous rules left (C-06): invisible characters, then spaces, encoding, HTML entities, null
tokens, then case. Number, date and row rules are one SQL query each.
"""

import contextlib
import math
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import duckdb

from quantilo import semantics
from quantilo.catalog import ColumnView
from quantilo.cleaning import model as m
from quantilo.cleaning import text as tx
from quantilo.cleaning.compile import (
    _date_expr,
    as_text,
    is_integer,
    is_number,
    is_temporal,
    is_text,
)
from quantilo.cleaning.suggestion import CleanSuggestion, Example, Mode
from quantilo.ingestion.sql import ident, lit
from quantilo.typing.knowledge_base import Role, load
from quantilo.typing.parsers import DATE_FORMATS, clean_text, number_sql

MAX_DISTINCT = 50_000  # text columns with more distinct values get the SQL-only checks
MAX_EXAMPLES = 3
DIGIT = re.compile(r"\d")
SENTINEL_SHARE = 0.005  # B-09, B-16: a value standing for "missing" is at least 0.5 % of rows
NEGATIVE_SHARE = 0.05  # B-10: fewer negatives than this in an amount or a quantity
LENGTH_SHARE = 0.9  # B-13: the dominant length of a code
DATE_COVERAGE = 0.95  # B-11: share of values the chosen formats must read

# B-04 (decision 8): sure tokens are automatic, those that can be a real category are suggested.
SURE_TOKENS = {"", "n/a", "#n/a", "#na", "na", "nan", "null", "nil", "#value!", "#ref!"}
SURE_TOKENS |= {"#div/0!", "#null!", "#num!", "#name?", "(blank)", "not available", "missing"}
WEAK_TOKENS = {"none", "unknown", "-", "--", "?", "tbd", "n.a.", "undefined", "not applicable"}
# 093 (CLN-8): words for "not given" that can also be a real answer.
WEAK_TOKENS |= {"unspecified", "not specified", "not stated", "inconnu", "non renseigne"}
WEAK_TOKENS |= {"non renseigné"}
REAL_NA = {"na", "n/a"}  # 093 (CLN-4): the most frequent of few levels may mean "none"
REAL_NA_LEVELS = 12
NUMBER_SHARE = 0.8  # 093 (CLN-8): a column at least this numeric, with a few short words
SUPPRESSED_LENGTH = 3
NUMBER_TEXT = re.compile(r"^\s*-?\d+(?:[.,]\d+)?\s*$")
PEAK_CANDIDATES = 5  # 093 (CLN-1): most frequent values checked for a round isolated peak
LEADING_ZERO_SHARE = 0.01  # 093 (CLN-6): values like '00928' keep a code as text
# 093 (CLN-5): columns of people's names, whose spellings may be two people.
PERSON_WORDS = {"nom", "prenom", "surname", "firstname", "lastname", "forename", "givenname"}
PERSON_PAIRS = {"first", "last", "given", "family", "middle", "maiden"}
# 093 (CLN-9): "Less than 1 year", "More than 50 years": read as their bound.
BOUND_BELOW = re.compile(
    r"^\s*(?:less than|under|below|fewer than|<)\s*(\d+(?:[.,]\d+)?)\s*[a-z ]*$", re.I
)
BOUND_ABOVE = re.compile(
    r"^\s*(?:more than|over|above|greater than|>)\s*(\d+(?:[.,]\d+)?)\s*[a-z ]*$", re.I
)
BOUND_AFTER = re.compile(
    r"^\s*(\d+(?:[.,]\d+)?)\s*[a-z ]*?(?:or more|or older|and over|and above|or less|"
    r"or fewer|and under|\+)\s*$",
    re.I,
)
SENTINELS = (-1, -9, -99, -999, -9999, -99999, 99, 999, 9999, 99999, 999999, 9999999)
SENTINEL_DATES = {"1900-01-01": "A", "0001-01-01": "A", "9999-12-31": "A", "1970-01-01": "S"}
CATEGORY_TYPES = ("categorical_nominal", "categorical_ordinal", "boolean")
ORDER = {
    "B-15": 10,
    "B-02": 20,
    "B-01": 30,
    "B-05": 40,
    "B-06": 50,
    "B-04": 60,
    "B-03": 70,
    "B-12": 80,
    "B-13": 90,
    "B-07": 100,
    "B-11": 110,
    "B-17": 110,
    "B-18": 120,
    "B-09": 130,
    "B-16": 140,
    "B-10": 150,
    "B-14": 160,
    "B-19": 170,
}


def n(count: int | float) -> str:
    return f"{count:,}"


def q(value: str | None, width: int = 40) -> str:
    if value is None:
        return "empty"
    shown = value if len(value) <= width else value[: width - 1] + "…"
    return f"'{shown}'"


def visible(value: str) -> str:
    """Invisible characters spelled out, for an example."""
    return "".join(f"[U+{ord(c):04X}]" if tx.remove_invisible(c) == "" else c for c in value)


@dataclass
class Column:
    name: str
    position: int
    dtype: str
    view: ColumnView | None
    nonnull: int
    distinct: int
    low: str | None
    high: str | None
    values: list[tuple[str, int]] | None = None  # text: distinct values and their rows

    @cached_property
    def role(self) -> Role | None:
        """From the name: a role a value outside its range kept the detection from applying."""
        return load().match_role(self.name)

    @property
    def type(self) -> str | None:
        return self.view.type if self.view else None


@dataclass
class TextStages:
    """A text column's distinct values through the text rules, in order (C-06)."""

    raw: list[tuple[str, int]]
    invisible: dict[str, str] = field(default_factory=dict)  # raw → without invisible
    spaces: dict[str, str] = field(default_factory=dict)  # previous → normalized spaces
    encoding: dict[str, str] = field(default_factory=dict)
    unfixable: int = 0  # rows whose value looks mis-encoded but does not round-trip
    html: dict[str, str] = field(default_factory=dict)
    final: list[tuple[str, int]] = field(default_factory=list)  # value after them, rows


def text_stages(values: list[tuple[str, int]]) -> TextStages:
    stages = TextStages(values)
    final: dict[str, int] = defaultdict(int)
    for raw, count in values:
        value = raw
        cleaned = tx.remove_invisible(value)
        if cleaned != value:
            stages.invisible[value] = cleaned
            value = cleaned
        spaced = tx.normalize_spaces(value)
        if spaced != value:
            stages.spaces[value] = spaced
            value = spaced
        repaired = tx.fix_encoding(value)
        if repaired is not None:
            stages.encoding[value] = repaired
            value = repaired
        elif tx.MOJIBAKE.search(value):
            stages.unfixable += count
        decoded = tx.unescape_html(value)
        if decoded is not None:
            stages.html[value] = decoded
            value = decoded
        final[value] += count
    stages.final = sorted(final.items(), key=lambda item: (-item[1], item[0]))
    return stages


@dataclass
class Context:
    connection: duckdb.DuckDBPyConnection
    source: str  # read_parquet(…)
    rows: int
    columns: list[Column]
    found: list[CleanSuggestion] = field(default_factory=list)
    stages: dict[str, TextStages] = field(default_factory=dict)

    def one(self, sql: str) -> tuple[Any, ...]:
        row = self.connection.execute(sql).fetchone()
        assert row is not None
        return row

    def add(
        self,
        rule: str,
        mode: Mode,
        column: Column | None,
        title: str,
        reason: str,
        step: Any = None,
        cells: int = 0,
        rows: int = 0,
        examples: list[tuple[str | None, str | None]] | None = None,
        key: str | None = None,  # tells apart two suggestions of a rule on one column
    ) -> None:
        if step is not None:
            step.origin = "auto" if mode == "A" else "suggested"
            step.rule = rule
        target = column.name if column else "rows"
        suggestion = CleanSuggestion(
            id=f"{rule}:{target}" + (f":{key}" if key else ""),
            rule=rule,
            mode=mode,
            order=ORDER[rule] * 10_000 + (column.position if column else 0),
            title=title,
            reason=reason,
            columns=[column.name] if column else [],
            cells=cells,
            rows=rows,
            examples=[Example(before=b, after=a) for b, a in (examples or [])[:MAX_EXAMPLES]],
            step=step,
        )
        if step is not None:
            step.suggestion = suggestion.id
        self.found.append(suggestion)

    def texts(self) -> list[tuple[Column, TextStages]]:
        return [(c, self.stages[c.name]) for c in self.columns if c.name in self.stages]


def _id() -> str:
    return "s0"  # numbered once every suggestion is known (suggest.py)


# --- Profile -----------------------------------------------------------------------------------


def profile(connection: duckdb.DuckDBPyConnection, source: str, views: list[ColumnView]) -> Context:
    described = connection.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
    by_name = {view.name: view for view in views}
    parts = ["count(*)"]
    for name, dtype, *_ in described:
        text = as_text(name, dtype)
        parts += [
            f"count({ident(name)})",
            f"approx_count_distinct({ident(name)})",
            f"min({text})",
            f"max({text})",
        ]
    row = connection.execute(f"SELECT {', '.join(parts)} FROM {source}").fetchone()
    assert row is not None
    columns = []
    for index, (name, dtype, *_) in enumerate(described):
        base = 1 + index * 4
        columns.append(
            Column(
                name=name,
                position=index,
                dtype=dtype,
                view=by_name.get(name),
                nonnull=int(row[base]),
                distinct=int(row[base + 1]),
                low=row[base + 2],
                high=row[base + 3],
            )
        )
    context = Context(connection, source, int(row[0]), columns)
    for column in columns:
        if is_text(column.dtype) and 0 < column.distinct <= MAX_DISTINCT:
            column.values = [
                (str(value), int(count))
                for value, count in connection.execute(
                    f"SELECT {ident(column.name)}, count(*) FROM {source} "
                    f"WHERE {ident(column.name)} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC, 1"
                ).fetchall()
            ]
            context.stages[column.name] = text_stages(column.values)
    return context


# --- Text --------------------------------------------------------------------------------------


def _changed(stages: TextStages, mapping: dict[str, str]) -> int:
    """Rows whose value at this stage is a key of `mapping` (counted on the raw values)."""
    total = 0
    for raw, count in stages.raw:
        value = raw
        for previous in (stages.invisible, stages.spaces, stages.encoding, stages.html):
            if previous is mapping:
                break
            value = previous.get(value, value)
        if value in mapping:
            total += count
    return total


def _examples(mapping: dict[str, str], show: Callable[[str], str] = lambda v: v) -> list[Any]:
    return [(show(before), after) for before, after in list(mapping.items())[:MAX_EXAMPLES]]


def _large_text(context: Context, column: Column, rule: str, pattern: str, replace: str) -> int:
    """SQL-only count of the rows a text normalization changes (too many distinct values)."""
    name = ident(column.name)
    if rule == "B-01":
        expr = f"trim(regexp_replace({name}, {lit(pattern)}, ' ', 'g'), ' ')"
    else:
        expr = f"regexp_replace({name}, {lit(pattern)}, {lit(replace)}, 'g')"
    return int(context.one(f"SELECT count(*) FROM {context.source} WHERE {expr} <> {name}")[0])


def b02_invisible(context: Context) -> None:
    for column in context.columns:
        if not is_text(column.dtype) or not column.nonnull:
            continue
        stages = context.stages.get(column.name)
        if stages is None:
            cells = _large_text(context, column, "B-02", tx.SQL_INVISIBLE, "")
            examples: list[Any] = []
        else:
            cells = _changed(stages, stages.invisible)
            examples = _examples(stages.invisible, visible)
        if cells:
            context.add(
                "B-02",
                "A",
                column,
                f"Remove invisible characters in {column.name}",
                f"Invisible characters (zero-width spaces, byte-order marks, control characters) "
                f"found in {n(cells)} values.",
                m.RemoveInvisible(id=_id(), columns=[column.name]),
                cells=cells,
                examples=examples,
            )


def b01_spaces(context: Context) -> None:
    for column in context.columns:
        if not is_text(column.dtype) or not column.nonnull:
            continue
        stages = context.stages.get(column.name)
        if stages is None:
            cells = _large_text(context, column, "B-01", tx.SQL_SPACE_RUN, " ")
            examples: list[Any] = []
        else:
            cells = _changed(stages, stages.spaces)
            examples = _examples(stages.spaces)
        if cells:
            context.add(
                "B-01",
                "A",
                column,
                f"Trim extra spaces in {column.name}",
                f"{n(cells)} values had extra spaces (at the ends, doubled, tabs or no-break "
                "spaces).",
                m.NormalizeSpaces(id=_id(), columns=[column.name]),
                cells=cells,
                examples=examples,
            )


def b05_encoding(context: Context) -> None:
    for column, stages in context.texts():
        if not stages.encoding:
            continue
        cells = _changed(stages, stages.encoding)
        mode: Mode = "S" if stages.unfixable else "A"
        unfixable = (
            f" {n(stages.unfixable)} more values look damaged but cannot be repaired safely."
            if stages.unfixable
            else ""
        )
        first = next(iter(stages.encoding.items()))
        context.add(
            "B-05",
            mode,
            column,
            f"Repair the encoding of {column.name}",
            f"{n(cells)} values were saved in one encoding and read in another: {q(first[0])} → "
            f"{q(first[1])} (wrong encoding).{unfixable}",
            m.MapValues(id=_id(), column=column.name, map=_capped(stages.encoding)),
            cells=cells,
            examples=_examples(stages.encoding),
        )


def b06_html(context: Context) -> None:
    for column, stages in context.texts():
        if not stages.html:
            continue
        cells = _changed(stages, stages.html)
        context.add(
            "B-06",
            "A",
            column,
            f"Decode HTML codes in {column.name}",
            f"HTML codes such as '&amp;' decoded in {n(cells)} values.",
            m.MapValues(id=_id(), column=column.name, map=_capped(stages.html)),
            cells=cells,
            examples=_examples(stages.html),
        )


def _capped(mapping: dict[str, str], weights: dict[str, int] | None = None) -> dict[str, Any]:
    items = list(mapping.items())
    if weights is not None:
        items.sort(key=lambda item: -weights.get(item[0], 0))
    return dict(items[: m.MAX_MAP])


def _converted(column: Column) -> bool:
    """A text column the typing reads as numbers, dates or booleans: a conversion rule handles
    its null tokens."""
    view = column.view
    return bool(view and view.parser.kind in ("number", "date", "boolean"))


def b04_tokens(context: Context) -> None:
    for column, stages in context.texts():
        if _converted(column):
            continue
        sure: dict[str, int] = {}
        weak: dict[str, int] = {}
        for value, count in stages.final:
            token = value.strip().lower()
            if token in SURE_TOKENS:
                sure[value] = count
            elif token in WEAK_TOKENS:
                weak[value] = count
        notes: list[str] = []
        real = _real_na(stages, sure)
        if real is not None:
            weak[real] = sure.pop(real)
            notes.append(
                f" {q(real)} is the most frequent value of only {len(stages.final) - 1} others: "
                "it may mean 'none' (nothing of the kind, as 'no pool'), a real category (CLN-4)."
            )
        suppressed = _suppressed(stages, {**sure, **weak})
        if suppressed:
            weak.update(suppressed)
            notes.append(
                " They sit in a column of numbers: likely suppressed or hidden values, which "
                "become empty so the numbers can be read."
            )
        levels: list[tuple[Mode, dict[str, int]]] = [("A", sure), ("S", weak)]
        for mode, found in levels:
            if not found:
                continue
            cells = sum(found.values())
            shown = ", ".join(q(v) for v in list(found)[:4])
            blank = all(v.strip() == "" for v in found)
            reason = (
                f"{n(cells)} blank strings read as missing."
                if blank
                else f"{n(cells)} values are words for 'missing' ({shown}): read as empty."
            )
            if mode == "S":
                reason += "".join(notes) + " Check they are not a real category."
            context.add(
                "B-04",
                mode,
                column,
                f"Read {shown} as missing in {column.name}",
                reason,
                m.NullValues(
                    id=_id(),
                    column=column.name,
                    values=sorted({v.strip().lower() for v in found}),
                    match="token",
                ),
                cells=cells,
                examples=[(v, None) for v in list(found)[:3]],
                key="maybe" if mode == "S" else None,
            )
            context.stages[column.name].final = [(v, c) for v, c in stages.final if v not in found]


def _real_na(stages: TextStages, sure: dict[str, int]) -> str | None:
    """CLN-4: 'NA' as the most frequent value of a column with few levels often means 'none'
    (no alley, no pool): suggested, not automatic."""
    values = [(v, c) for v, c in stages.final if v.strip()]
    if not values or len(values) - 1 > REAL_NA_LEVELS:
        return None
    top = values[0][0]
    return top if top in sure and top.strip().lower() in REAL_NA else None


def _suppressed(stages: TextStages, tokens: dict[str, int]) -> dict[str, int]:
    """CLN-8: in a column of numbers, one to three short words ('s', 'x', '*') stand for a value
    that is hidden or suppressed."""
    values = [(v, c) for v, c in stages.final if v.strip() and v not in tokens]
    total = sum(c for _, c in values)
    words = {v: c for v, c in values if not NUMBER_TEXT.match(v)}
    if not total or not 1 <= len(words) <= 3:
        return {}
    if (total - sum(words.values())) < NUMBER_SHARE * total:
        return {}
    if any(len(v.strip()) > SUPPRESSED_LENGTH or DIGIT.search(v) for v in words):
        return {}
    return words


def _is_code(stages: TextStages) -> bool:
    """Values that are mostly codes ('85123A'): their case can carry meaning (A-04, M4)."""
    values = [v for v, _ in stages.final[:2_000]]
    return bool(values) and sum(1 for v in values if DIGIT.search(v)) >= 0.5 * len(values)


def _is_category(column: Column) -> bool:
    if column.view is None:
        return False
    if column.view.type in CATEGORY_TYPES:
        return True
    return column.view.type == "text" and column.distinct <= 0.2 * max(column.nonnull, 1)


def _accent_first(spellings: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """CLN-5: the accented spelling is kept unless the plain one has twice its rows: folding
    accents loses information ('MÜLLER' → 'MULLER'), restoring them cannot."""
    accented = [item for item in spellings if not item[0].isascii()]
    if not accented or not spellings[0][0].isascii():
        return spellings
    best = accented[0]
    if spellings[0][1] >= 2 * best[1]:
        return spellings
    return [best, *[item for item in spellings if item is not best]]


def _person_names(column: Column) -> bool:
    """CLN-5: 'nom', 'prénom', 'last_name', 'owner first name': names of people."""
    words = set(semantics.tokens(column.name))
    return bool(words & PERSON_WORDS) or ("name" in words and bool(words & PERSON_PAIRS))


def b03_case(context: Context) -> None:
    for column, stages in context.texts():
        if not _is_category(column) or _converted(column) or _is_code(stages):
            continue
        groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for value, count in stages.final:
            if value.strip():
                groups[tx.fold(value)].append((value, count))
        mapping: dict[str, str] = {}
        weights: dict[str, int] = {}
        merged = []
        for spellings in groups.values():
            if len(spellings) < 2:
                continue
            spellings = _accent_first(spellings)
            target = spellings[0][0]  # most frequent (stages.final is sorted by rows)
            merged.append(spellings)
            for value, count in spellings[1:]:
                mapping[value] = target
                weights[value] = count
        if not mapping:
            continue
        cells = sum(weights.values())
        biggest = max(merged, key=lambda group: sum(c for _, c in group))
        others = ", ".join(q(v) for v, _ in biggest[1:3])
        title = (
            f"Merge {len(biggest)} spellings of {q(biggest[0][0])}"
            if len(merged) == 1
            else f"Merge spellings in {column.name} ({len(merged)} groups)"
        )
        person = _person_names(column)
        accented = not biggest[0][0].isascii() and any(v.isascii() for v, _ in biggest)
        kept = "the accented spelling, kept (CLN-5)" if accented else "most frequent"
        context.add(
            "B-03",
            "S" if person else "A",
            column,
            title,
            f"{others} → {q(biggest[0][0])} ({kept}): same value up to case or accents. "
            f"{n(cells)} cells in {len(merged)} groups."
            + (
                " Names of people: two spellings may be two people, so the merge is suggested, "
                "not applied (CLN-5)."
                if person
                else ""
            ),
            m.MapValues(id=_id(), column=column.name, map=_capped(mapping, weights)),
            cells=cells,
            examples=[(b, a) for b, a in sorted(mapping.items(), key=lambda i: -weights[i[0]])],
        )


def b12_decimal_zero(context: Context) -> None:
    pattern = re.compile(r"^-?\d+\.0+$")
    for column in context.columns:
        if not column.nonnull:
            continue
        stages = context.stages.get(column.name)
        if stages is not None:
            values = [v for v, _ in stages.final]
            if values and all(pattern.match(v) for v in values):
                context.add(
                    "B-12",
                    "A",
                    column,
                    f"Remove the '.0' of {column.name}",
                    f"All {n(column.nonnull)} values end in '.0': {q(values[0])} → "
                    f"{q(values[0].split('.')[0])}.",
                    m.ReplaceValues(
                        id=_id(),
                        column=column.name,
                        find=r"^(-?[0-9]+)\.0+$",
                        replace=r"\1",
                        match="regex",
                    ),
                    cells=column.nonnull,
                    examples=[(v, v.split(".")[0]) for v in values[:3]],
                )
            continue
        if column.dtype not in ("DOUBLE", "FLOAT", "REAL") or not _identifier_like(column):
            continue
        name = ident(column.name)
        whole = context.one(
            f"SELECT count(*) FILTER (WHERE {name} <> trunc({name})), any_value({name}) "
            f"FROM {context.source} WHERE {name} IS NOT NULL"
        )
        if whole[0] == 0 and whole[1] is not None:
            sample = f"{float(whole[1]):.1f}"
            context.add(
                "B-12",
                "A",
                column,
                f"Store {column.name} as text without '.0'",
                f"An identifier stored as decimal numbers: all {n(column.nonnull)} values end in "
                f"'.0' ({q(sample)} → {q(sample[:-2])}).",
                m.ChangeType(id=_id(), column=column.name, to="text"),
                cells=column.nonnull,
                examples=[(sample, sample[:-2])],
            )


def _identifier_like(column: Column) -> bool:
    if column.type == "identifier" or (column.view and column.view.role == "identifier"):
        return True
    role = column.role
    if role is not None and role.id in ("identifier", "postal_code"):
        return True
    words = set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", column.name).lower()))
    return bool(words & {"id", "code", "number", "num", "no", "key", "ref"})


def b13_leading_zeros(context: Context) -> None:
    for column in context.columns:
        if not column.nonnull:
            continue
        postal = column.role is not None and column.role.id == "postal_code"
        if not (postal or column.type == "identifier"):
            continue
        if is_text(column.dtype):
            stages = context.stages.get(column.name)
            if stages is None or not all(v.isdigit() for v, _ in stages.final):
                continue
            lengths: dict[int, int] = defaultdict(int)
            for value, count in stages.final:
                lengths[len(value)] += count
        elif is_integer(column.dtype):
            name = ident(column.name)
            lengths = defaultdict(int)
            negative = context.one(f"SELECT count(*) FROM {context.source} WHERE {name} < 0")[0]
            if negative:
                continue
            for length, count in context.connection.execute(
                f"SELECT length(CAST({name} AS VARCHAR)), count(*) FROM {context.source} "
                f"WHERE {name} IS NOT NULL GROUP BY 1"
            ).fetchall():
                lengths[int(length)] += int(count)
        else:
            continue
        if not lengths:
            continue
        dominant = max(lengths, key=lambda length: lengths[length])
        share = lengths[dominant] / sum(lengths.values())
        shorter = sum(count for length, count in lengths.items() if length < dominant)
        if share < LENGTH_SHARE or not shorter or dominant < 3:
            continue
        if min(lengths) < dominant - 2:  # every length down to one digit: a counter, not codes
            continue
        what = "postal codes" if postal else "codes"
        context.add(
            "B-13",
            "A" if postal else "S",
            column,
            f"Restore the leading zeros of {column.name}",
            f"{n(shorter)} {what} are shorter than the usual {dominant} digits "
            f"({share:.0%} of values): they lost their leading zero.",
            m.PadZeros(id=_id(), column=column.name, length=dominant),
            cells=shorter,
            examples=[("5207", "05207")] if dominant == 5 else [],
        )


# --- Types -------------------------------------------------------------------------------------


def b07_numbers(context: Context) -> None:
    for column in context.columns:
        view = column.view
        if not (is_text(column.dtype) and view and view.parser.kind == "number"):
            continue
        if view.type not in ("numeric_continuous", "numeric_discrete"):
            continue
        code = semantics.code_by_name(column.name)
        if _leading_zeros(context, column) or (code is not None and code.kind == "identifier"):
            continue  # CLN-6: a code ('00928', 'PID') is not a number (B-13 says so)
        cleaned = clean_text(column.name)
        number = number_sql(cleaned, view.parser.decimal)
        shown = (
            f"CASE WHEN {number} = trunc({number}) AND abs({number}) < 1e15 THEN "
            f"CAST(CAST({number} AS BIGINT) AS VARCHAR) ELSE CAST({number} AS VARCHAR) END"
        )
        row = context.one(
            f"SELECT count(*) FILTER (WHERE {number} IS NOT NULL), "
            f"count(*) FILTER (WHERE {cleaned} IS NOT NULL AND {number} IS NULL), "
            f"count(*) FILTER (WHERE {number} IS NOT NULL AND {number} <> trunc({number})), "
            f"count(*) FILTER (WHERE {ident(column.name)} IS NOT NULL AND {cleaned} IS NULL), "
            f"count(*) FILTER (WHERE {number} IS NOT NULL AND {ident(column.name)} <> "
            f"{shown}) FROM {context.source}"
        )
        read, unreadable, fractional, tokens, rewritten = (int(v) for v in row)
        if not read:
            continue
        bad = [
            str(v[0])
            for v in context.connection.execute(
                f"SELECT {cleaned} FROM {context.source} WHERE {cleaned} IS NOT NULL AND "
                f"{number} IS NULL GROUP BY 1 ORDER BY count(*) DESC LIMIT 3"
            ).fetchall()
        ]
        good = context.connection.execute(
            f"SELECT {ident(column.name)}, {number} FROM {context.source} WHERE {number} IS NOT "
            f"NULL AND {ident(column.name)} <> CAST({number} AS VARCHAR) LIMIT 3"
        ).fetchall()
        integer = fractional == 0 and view.type == "numeric_discrete"
        how = "decimal comma" if view.parser.decimal == "," else "decimal point"
        reason = f"{n(read)} values are numbers stored as text ({how})."
        bounds = _bounds(context, column, cleaned, number) if unreadable else {}
        if bounds:
            shown = ", ".join(f"{q(k)} → {v}" for k, v in list(bounds.items())[:3])
            context.add(
                "B-07",
                "S",
                column,
                f"Read the bounds in {column.name} as numbers",
                f"Values written in words are read as their bound: {shown}. 'Less than 1' "
                "becomes 0.5; the bound keeps the order and an approximate value (CLN-9).",
                m.MapValues(id=_id(), column=column.name, map=dict(bounds)),
                cells=int(
                    context.one(
                        f"SELECT count(*) FROM {context.source} WHERE {ident(column.name)} IN "
                        f"({', '.join(lit(v) for v in bounds)})"
                    )[0]
                ),
                examples=list(bounds.items()),
                key="bounds",
            )
            reason += " Apply the bounds suggestion first to keep the values written in words."
        if tokens:
            reason += f" {n(tokens)} blank or 'missing' values become empty."
        if unreadable:
            reason += (
                f" {n(unreadable)} values cannot be read and become empty: "
                + ", ".join(q(v) for v in bad)
                + "."
            )
        context.add(
            "B-07",
            "S" if unreadable else "A",
            column,
            f"Store {column.name} as numbers",
            reason,
            m.ChangeType(
                id=_id(),
                column=column.name,
                to="integer" if integer else "number",
                decimal=view.parser.decimal,
            ),
            cells=rewritten + tokens + unreadable,
            examples=[*((str(b), f"{a:g}") for b, a in good), *((v, None) for v in bad)],
        )


def _bound(text: str) -> str | None:
    """CLN-9: 'Less than 1 year' → '0.5', 'More than 50 years' → '50', '65 or more' → '65'."""
    for pattern in (BOUND_BELOW, BOUND_ABOVE, BOUND_AFTER):
        match = pattern.match(text)
        if match:
            value = float(match.group(1).replace(",", "."))
            if pattern is BOUND_BELOW and value == 1:
                value = 0.5
            return f"{value:g}"
    return None


def _bounds(context: Context, column: Column, cleaned: str, number: str) -> dict[str, str]:
    """The unreadable values, when every one of them is a bound written in words."""
    raw = [
        str(v)
        for (v,) in context.connection.execute(
            f"SELECT DISTINCT {ident(column.name)} FROM {context.source} WHERE {cleaned} IS NOT "
            f"NULL AND {number} IS NULL LIMIT 21"
        ).fetchall()
    ]
    if not raw or len(raw) > 20:
        return {}
    found = {value: _bound(value) for value in raw}
    if any(v is None for v in found.values()):
        return {}
    return {k: v for k, v in sorted(found.items()) if v is not None}


def _leading_zeros(context: Context, column: Column) -> int:
    """CLN-6: rows of a text column of digits that start with a zero ('00928')."""
    stages = context.stages.get(column.name)
    if stages is None or not is_text(column.dtype) or semantics.is_date_part(column.name):
        return 0  # '01' for January is a month number, not a code
    values = [(v.strip(), c) for v, c in stages.final if v.strip()]
    if not values or not all(v.isdigit() for v, _ in values):
        return 0
    zeros = sum(c for v, c in values if len(v) > 1 and v.startswith("0"))
    return zeros if zeros >= LEADING_ZERO_SHARE * sum(c for _, c in values) else 0


def b13_keep_text(context: Context) -> None:
    """CLN-6: digits with leading zeros are a code: kept as text, never read as numbers."""
    done = {s.columns[0] for s in context.found if s.rule == "B-13" and s.columns}
    for column in context.columns:
        if column.name in done or not column.view or column.view.parser.kind != "number":
            continue
        zeros = _leading_zeros(context, column)
        if not zeros:
            continue
        sample = next(v for v, _ in context.stages[column.name].final if v.startswith("0"))
        context.add(
            "B-13",
            "S",
            column,
            f"Keep {column.name} as text",
            f"{n(zeros)} values start with a leading zero ({q(sample)}): a code, which stored as "
            f"a number would lose it ({q(sample)} → {q(sample.lstrip('0') or '0')}). Kept as "
            "text (CLN-6).",
            m.ChangeType(id=_id(), column=column.name, to="text"),
            cells=zeros,
            examples=[(sample, sample)],
        )


def _swapped(date_format: str) -> str:
    return date_format.replace("%d", "\0").replace("%m", "%d").replace("\0", "%m")


SERIAL = "regexp_full_match(v, '[0-9]{5}') AND TRY_CAST(v AS INTEGER) BETWEEN 20000 AND 60000"


def _swap_index(index: int) -> int:
    """The same format with day and month swapped (itself when it has no such twin)."""
    swapped = _swapped(DATE_FORMATS[index])
    return DATE_FORMATS.index(swapped) if swapped in DATE_FORMATS else index


def _greedy(values: list[tuple[str, int, list[bool]]]) -> list[int]:
    """Up to three formats (their indexes), each reading most of the values still unread."""
    chosen: list[int] = []
    unread = list(values)
    for _ in range(3):
        gains = [0] * len(DATE_FORMATS)
        for _value, count, reads in unread:
            for index, ok in enumerate(reads):
                if ok:
                    gains[index] += count
        for index in chosen:
            gains[index] = 0
            gains[_swap_index(index)] = 0  # one day/month order
        best = max(range(len(gains)), key=lambda i: gains[i])
        if gains[best] == 0:
            break
        chosen.append(best)
        unread = [item for item in unread if not item[2][best]]
        if not unread:
            break
    return chosen


def b11_dates(context: Context) -> None:
    """Text dates, read value by value (B-11, B-17): up to three formats chosen greedily on the
    distinct values; day/month order decided on the whole column (values with a day above 12)."""
    for column, stages in context.texts():
        view = column.view
        if view and view.type not in ("datetime", "categorical_nominal", "text", "identifier"):
            continue
        total = sum(c for v, c in stages.final if v.strip())
        digits = sum(c for v, c in stages.final if DIGIT.search(v))
        if not total or digits < 0.5 * total:
            continue
        value = f"trim(regexp_replace({ident(column.name)}, {lit(tx.SQL_SPACE_RUN)}, ' ', 'g'))"
        tests = ", ".join(f"TRY_STRPTIME(v, {lit(f)}) IS NOT NULL" for f in DATE_FORMATS)
        rows = context.connection.execute(
            f"SELECT v, count(*), [{tests}], {SERIAL} FROM (SELECT {value} AS v FROM "
            f"{context.source}) WHERE v <> '' GROUP BY v"
        ).fetchall()
        values = [(str(v), int(c), [bool(r) for r in reads]) for v, c, reads, _ in rows]
        by_format = [sum(c for _, c, reads in values if reads[i]) for i in range(len(DATE_FORMATS))]
        if max(by_format, default=0) < 0.3 * total:
            continue
        indexes = _greedy(values)
        for position, index in enumerate(indexes):  # the order reading more values wins
            swapped = _swap_index(index)
            if by_format[swapped] > by_format[index]:
                indexes[position] = swapped
        chosen = [DATE_FORMATS[i] for i in indexes]
        serials = sum(int(c) for _, c, reads, serial in rows if serial and not any(reads))
        read = sum(c for _, c, reads in values if any(reads[i] for i in indexes))
        excel = serials > 0
        covered = read + serials
        if covered < DATE_COVERAGE * total:
            continue
        ambiguous = [
            DATE_FORMATS[i]
            for i in indexes
            if _swap_index(i) != i and by_format[_swap_index(i)] >= by_format[i]
        ]
        with_time = any("%H" in f for f in chosen)
        unread = total - covered
        mode: Mode = "S" if ambiguous or unread else "A"
        reason = (
            f"{n(covered)} dates stored as text"
            + (f" in {len(chosen)} formats ({', '.join(chosen)})" if len(chosen) > 1 else "")
            + (f", {n(serials)} as Excel day numbers" if excel else "")
            + "."
        )
        if ambiguous:
            reason += " Day and month could be swapped: check the order."
        if unread:
            reason += f" {n(unread)} values cannot be read and become empty."
        parsed = _date_expr("v", chosen, excel)
        if not with_time:
            parsed = f"CAST({parsed} AS DATE)"
        reads_of = {v: reads for v, _, reads in values}
        changed = 0
        examples: list[tuple[str | None, str | None]] = []
        seen: set[int] = set()
        for text, count, after in context.connection.execute(
            f"SELECT v, c, CAST({parsed} AS VARCHAR) FROM (SELECT {value} AS v, count(*) AS c "
            f"FROM {context.source} WHERE {ident(column.name)} IS NOT NULL GROUP BY 1) "
            "ORDER BY c DESC, v"
        ).fetchall():
            if after == text:
                continue
            changed += int(count)
            reads = reads_of.get(str(text), [])
            kind = next((i for i in indexes if reads and reads[i]), -1)
            if kind not in seen and len(examples) < MAX_EXAMPLES:
                seen.add(kind)
                examples.append((str(text), None if after is None else str(after)))
        if not changed:
            continue
        formats = len(chosen) + int(excel)
        context.add(
            "B-17" if excel and not chosen else "B-11",
            mode,
            column,
            (
                f"Unify {formats} date formats in {column.name}"
                if formats > 1
                else f"Store {column.name} as dates"
            ),
            reason,
            m.ChangeType(
                id=_id(),
                column=column.name,
                to="datetime" if with_time else "date",
                formats=chosen,
                excel_serial=excel,
            ),
            cells=changed,
            examples=examples,
        )


def b18_booleans(context: Context) -> None:
    kb = load()
    trues = {w for pair in kb.boolean_pairs for w in pair[0]}
    falses = {w for pair in kb.boolean_pairs for w in pair[1]} - trues
    for column, stages in context.texts():
        values = [(v, c) for v, c in stages.final if v.strip()]
        if len(values) < 2:
            continue
        words = {v.strip().lower() for v, _ in values}
        found_true, found_false = words & trues, words & falses
        if words - trues - falses or not found_true or not found_false:
            continue
        mixed = len(values) > 2
        cells = sum(c for _, c in values)
        spellings = ", ".join(q(v) for v, _ in values[:6])
        context.add(
            "B-18",
            "A" if mixed else "S",
            column,
            f"Store {column.name} as true / false",
            (
                f"{len(values)} spellings of yes and no mixed ({spellings}): unified."
                if mixed
                else f"Two values ({spellings}) mean yes and no: stored as true / false."
            ),
            m.ChangeType(
                id=_id(),
                column=column.name,
                to="boolean",
                true_values=sorted(found_true),
                false_values=sorted(found_false),
            ),
            cells=cells,
            examples=[(v, "true" if v.strip().lower() in trues else "false") for v, _ in values],
        )


# --- Numbers and dates -------------------------------------------------------------------------


def _numbers(context: Context) -> list[Column]:
    return [
        c
        for c in context.columns
        if is_number(c.dtype) and c.nonnull and c.type not in ("identifier", "boolean")
    ]


def _round(value: float) -> bool:
    """CLN-1: -200, -100000, 99999: a negative number with one significant digit, or nines only
    (at least 2 digits). A positive round maximum (100 %, 100 points) is a cap, not a code."""
    if value != int(value) or value == 0:
        return False
    digits = str(abs(int(value)))
    if len(digits) < 2:
        return False
    return set(digits) == {"9"} or (value < 0 and len(digits.rstrip("0")) == 1)


def _number(value: float) -> str:
    return str(int(value)) if value == int(value) else repr(float(value))


def _peaks(context: Context, columns: list[Column]) -> dict[str, list[float]]:
    """CLN-1: the round values among each column's most frequent ones (one query) and its two
    extremes."""
    if not columns:
        return {}
    parts = ", ".join(f"approx_top_k({ident(c.name)}, {PEAK_CANDIDATES})" for c in columns)
    row = context.one(f"SELECT {parts} FROM {context.source}")
    found: dict[str, list[float]] = {}
    for column, tops in zip(columns, row, strict=True):
        values = [float(v) for v in tops or [] if v is not None and math.isfinite(float(v))]
        for end in (column.low, column.high):  # a code for "missing" is the extreme itself
            if end is not None:
                with contextlib.suppress(ValueError):
                    values.append(float(end))
        kept = {v for v in values if math.isfinite(v) and _round(v) and v not in SENTINELS}
        found[column.name] = sorted(kept)
    return found


def b09_sentinels(context: Context) -> dict[str, list[float]]:
    """-999, 9999…: frequent, and either far from the other values (beyond 3 IQR) or a peak
    (repeated ten times more than a typical value) beyond the whiskers (FAHES). 093 (CLN-1, CLN-2):
    any round value among the five most frequent ones (-200, -100000) is a candidate too when it is
    the column's extreme and far from the rest; (0, 0) positions are read as missing (CLN-3)."""
    found: dict[str, list[float]] = {}
    columns = _numbers(context)
    peaks = _peaks(context, columns)
    for column in columns:
        name = ident(column.name)
        extra = peaks.get(column.name, [])
        values = [*(float(v) for v in SENTINELS), *extra]
        candidates = ", ".join(_number(v) for v in values)
        others = f"{name} NOT IN ({candidates})"
        row = context.one(
            f"SELECT approx_quantile({name}, 0.25) FILTER (WHERE {others}), "
            f"approx_quantile({name}, 0.75) FILTER (WHERE {others}), "
            f"min({name}) FILTER (WHERE {others}), max({name}) FILTER (WHERE {others}), "
            + ", ".join(f"count(*) FILTER (WHERE {name} = {_number(v)})" for v in values)
            + f" FROM {context.source}"
        )
        q1, q3, low, high = row[0], row[1], row[2], row[3]
        if q1 is None or q3 is None:
            continue
        iqr = float(q3) - float(q1)
        repeat = column.nonnull / max(column.distinct, 1)  # rows per distinct value
        role = column.role
        for index, (value, count) in enumerate(zip(values, row[4:], strict=True)):
            share = int(count) / column.nonnull
            if share < SENTINEL_SHARE:
                continue
            far = not (float(q1) - 3 * iqr <= value <= float(q3) + 3 * iqr)
            peak = int(count) >= max(5, 10 * repeat) and not (
                float(q1) - 1.5 * iqr <= value <= float(q3) + 1.5 * iqr
            )
            listed = index < len(SENTINELS)
            if listed and not (far or peak):
                continue
            alone = value < float(low) or value > float(high)
            # the only negatives of a column of positives (refunds are negatives of an amount)
            flows = role is not None and role.id in ("amount", "quantity", "cost")
            below = value < 0 <= float(low) and not flows
            if not listed and not (alone and (far or below)):
                continue  # CLN-1: a round value counts only as the isolated extreme
            outside_role = role is not None and not role.accepts_range(value, value)
            text = _number(value)
            found.setdefault(column.name, []).append(float(value))
            why = (
                "far from the other values"
                if far
                else f"{int(count) / repeat:,.0f} times as frequent as a typical value"
            )
            if not listed:
                why = (
                    "a round number beyond every other value"
                    + (", the only negatives of the column" if below else " and far from them")
                    + ": the shape of a code for 'no measure' (CLN-1)"
                )
            context.add(
                "B-09",
                "A" if outside_role else "S",
                column,
                f"Read {text} as missing in {column.name}",
                f"{text} appears {n(int(count))} times ({share:.1%}), {why} (middle half "
                f"{float(q1):,.4g} to {float(q3):,.4g}): likely a code for 'missing'.",
                m.NullValues(id=_id(), column=column.name, values=[text], match="exact"),
                cells=int(count),
                examples=[(text, None)],
                key=text,
            )
    _zero_positions(context, columns, found)
    return found


def _zero_positions(context: Context, columns: list[Column], found: dict[str, list[float]]) -> None:
    """CLN-3: latitude = longitude = 0 is a geocoder's default ("null island"), not a place,
    when 0 is far from the other positions."""
    latitude = next((c for c in columns if c.role is not None and c.role.id == "latitude"), None)
    longitude = next((c for c in columns if c.role is not None and c.role.id == "longitude"), None)
    if latitude is None or longitude is None:
        return
    lat, lon = ident(latitude.name), ident(longitude.name)
    zero = f"{lat} = 0 AND {lon} = 0"
    count, q1, q3 = context.one(
        f"SELECT count(*) FILTER (WHERE {zero}), "
        f"approx_quantile({lat}, 0.25) FILTER (WHERE NOT ({zero})), "
        f"approx_quantile({lat}, 0.75) FILTER (WHERE NOT ({zero})) FROM {context.source}"
    )
    if not count or q1 is None or q3 is None:
        return
    iqr = float(q3) - float(q1)
    if float(q1) - 3 * iqr <= 0 <= float(q3) + 3 * iqr:
        return  # positions around the equator and Greenwich: 0 can be real
    share = int(count) / max(latitude.nonnull, 1)
    for column in (latitude, longitude):
        found.setdefault(column.name, []).append(0.0)
        context.add(
            "B-09",
            "S",
            column,
            f"Read 0 as missing in {column.name}",
            f"{n(int(count))} rows ({share:.1%}) sit at (0, 0), the default point of geocoders "
            f"in the ocean, far from the other positions (middle half of {latitude.name} "
            f"{float(q1):,.4g} to {float(q3):,.4g}): latitude and longitude 0 read as missing "
            "(CLN-3).",
            m.NullValues(id=_id(), column=column.name, values=["0"], match="exact"),
            cells=int(count),
            examples=[("0", None)],
            key="0,0",
        )


def b10_ranges(context: Context, sentinels: dict[str, list[float]]) -> None:
    for column in _numbers(context):
        name = ident(column.name)
        skip = sentinels.get(column.name, [])
        kept = f" AND {name} NOT IN ({', '.join(repr(v) for v in skip)})" if skip else ""
        role = column.role
        if role is not None and (role.min is not None or role.max is not None):
            tests = []
            if role.min is not None:
                tests.append(f"{name} < {role.min!r}")
            if role.max is not None:
                tests.append(f"{name} > {role.max!r}")
            outside, sample = context.one(
                f"SELECT count(*), list({as_text(column.name, column.dtype)})[1:3] "
                f"FROM {context.source} WHERE ({' OR '.join(tests)}){kept}"
            )
            if outside and outside / column.nonnull < NEGATIVE_SHARE:
                bounds = (
                    f"{role.min:g} to {role.max:g}"
                    if role.min is not None and role.max is not None
                    else ""
                )
                context.add(
                    "B-10",
                    "S",
                    column,
                    f"Empty the impossible values of {column.name}",
                    f"{n(int(outside))} values are outside the plausible range of "
                    f"{role.label.lower()} ({bounds}): "
                    + ", ".join(q(str(v)) for v in sample)
                    + ".",
                    m.NullOutside(id=_id(), column=column.name, min=role.min, max=role.max),
                    cells=int(outside),
                    examples=[(str(v), None) for v in sample],
                )
                continue
        if role is None or role.id not in ("amount", "quantity"):
            continue
        if semantics.is_ratio(column.name):
            continue  # CLN-7: an intensity (per unit) can be negative by nature
        negatives = int(
            context.one(f"SELECT count(*) FROM {context.source} WHERE {name} < 0{kept}")[0]
        )
        if negatives and negatives / column.nonnull < NEGATIVE_SHARE:
            share = f"{negatives / column.nonnull:.1%}"
            sales = role.id == "quantity" or semantics.is_money(column.name)
            what = "quantities" if role.id == "quantity" else "amounts"
            reason = (
                f"Refunds or cancellations? {n(negatives)} negative {what} ({share}). Remove them "
                "only if they are errors."
                if sales
                else f"{n(negatives)} negative values ({share}): errors, or real values (net "
                "emissions, losses, balances)? Remove them only if they are errors (CLN-7)."
            )
            context.add(
                "B-10",
                "S",
                column,
                f"Remove the rows with a negative {column.name}",
                reason,
                m.FilterRows(
                    id=_id(),
                    conditions=[m.RowCondition(column=column.name, op="lt", value="0")],
                    action="remove",
                ),
                rows=negatives,
            )


def b16_sentinel_dates(context: Context) -> None:
    for column in context.columns:
        if not is_temporal(column.dtype) or not column.nonnull:
            continue
        name = ident(column.name)
        for day, mode in SENTINEL_DATES.items():
            count, texts = context.one(
                f"SELECT count(*), list(DISTINCT {as_text(column.name, column.dtype)})[1:20] "
                f"FROM {context.source} WHERE CAST({name} AS DATE) = DATE {lit(day)}"
            )
            share = int(count) / column.nonnull
            if share < SENTINEL_SHARE:
                continue
            meaning = "'no end date'" if day.startswith("9999") else "'unknown date'"
            context.add(
                "B-16",
                mode,  # type: ignore[arg-type]
                column,
                f"Read {day} as missing in {column.name}",
                f"{day} is used {n(int(count))} times ({share:.1%}): a placeholder meaning "
                f"{meaning}, not a real date.",
                m.NullValues(id=_id(), column=column.name, values=sorted(texts), match="exact"),
                cells=int(count),
                examples=[(texts[0], None)] if texts else [],
                key=day,
            )


# --- Rows and columns --------------------------------------------------------------------------


def _blank(column: Column) -> str:
    return f"coalesce(trim({as_text(column.name, column.dtype)}), '') = ''"


def b15_empty_rows(context: Context) -> None:
    if len(context.columns) < 2:
        return
    empty = " AND ".join(_blank(c) for c in context.columns)
    header = " AND ".join(
        f"coalesce(lower(trim({as_text(c.name, c.dtype)})) = {lit(c.name.strip().lower())}, false)"
        for c in context.columns
    )
    blanks, headers = (
        int(v)
        for v in context.one(
            f"SELECT count(*) FILTER (WHERE {empty}), count(*) FILTER (WHERE {header}) "
            f"FROM {context.source}"
        )
    )
    if blanks:
        context.add(
            "B-15",
            "A",
            None,
            f"Remove {n(blanks)} empty rows",
            f"{n(blanks)} rows have no value at all.",
            m.DropEmptyRows(id=_id()),
            rows=blanks,
        )
        context.found[-1].id = "B-15:empty"
        context.found[-1].step.suggestion = "B-15:empty"  # type: ignore[union-attr]
    if headers:
        context.add(
            "B-15",
            "A",
            None,
            f"Remove {n(headers)} repeated header rows",
            f"{n(headers)} rows repeat the column names: files pasted one after the other.",
            m.DropHeaderRows(id=_id()),
            rows=headers,
        )
        context.found[-1].id = "B-15:header"
        context.found[-1].step.suggestion = "B-15:header"  # type: ignore[union-attr]


def b14_duplicates(context: Context) -> None:
    if not context.columns or context.rows < 2:
        return
    empty = " AND ".join(_blank(c) for c in context.columns)
    rows, distinct = (
        int(v)
        for v in context.one(
            f"SELECT count(*), (SELECT count(*) FROM (SELECT DISTINCT * FROM {context.source} "
            f"WHERE NOT ({empty}))) FROM {context.source} WHERE NOT ({empty})"
        )
    )
    copies = rows - distinct
    if not copies:
        return
    identified = any(c.type == "identifier" for c in context.columns)
    reason = f"{n(copies)} rows are exact copies of another row ({copies / max(rows, 1):.1%})."
    reason += (
        " Even their identifier repeats: a row entered twice."
        if identified
        else " Without an identifier they may be real repeats (the same purchase twice)."
    )
    context.add(
        "B-14",
        "A" if identified else "S",
        None,
        f"Remove {n(copies)} identical rows",
        reason,
        m.DropDuplicates(id=_id(), keys=None, keep="first"),
        rows=copies,
    )


def b19_columns(context: Context) -> None:
    for column in context.columns:
        if context.rows < 2:
            return
        if column.nonnull == 0:
            title, reason = f"Remove the empty column {column.name}", "It has no value at all."
        elif column.low is not None and column.low == column.high:
            title = f"Remove the constant column {column.name}"
            reason = f"Always {q(column.low)}: no information."
        else:
            continue
        context.add(
            "B-19",
            "S",
            column,
            title,
            reason + " Kept unless you remove it: a filter or a KPI may name it.",
            m.DropColumns(id=_id(), columns=[column.name]),
        )


def run(context: Context) -> list[CleanSuggestion]:
    b15_empty_rows(context)
    b02_invisible(context)
    b01_spaces(context)
    b05_encoding(context)
    b06_html(context)
    b04_tokens(context)
    b03_case(context)
    b12_decimal_zero(context)
    b13_leading_zeros(context)
    b13_keep_text(context)
    b07_numbers(context)
    b11_dates(context)
    b18_booleans(context)
    sentinels = b09_sentinels(context)
    b16_sentinel_dates(context)
    b10_ranges(context, sentinels)
    b14_duplicates(context)
    b19_columns(context)
    return sorted(context.found, key=lambda s: s.order)
