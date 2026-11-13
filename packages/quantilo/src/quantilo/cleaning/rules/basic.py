"""The Basic layer (study § 2.1, B-01 to B-19): computed when Clean opens, a few seconds at most.

Text rules work on each column's distinct values (50,000 at most, C-14), each one on the value
the previous rules left (C-06): invisible characters, then spaces, encoding, HTML entities, null
tokens, then case. Number, date and row rules are one SQL query each.
"""

import contextlib
import datetime
import itertools
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
    Stage,
    _date_expr,
    as_text,
    assembled_date_sql,
    is_integer,
    is_number,
    is_temporal,
    is_text,
)
from quantilo.cleaning.suggestion import CleanSuggestion, Example, Mode
from quantilo.ingestion.sql import ident, lit
from quantilo.profiling import aggregates
from quantilo.reasons import say, say_count
from quantilo.typing import date_parts
from quantilo.typing.knowledge_base import Role, load
from quantilo.typing.parsers import (
    ALL_DATE_FORMATS,
    AMOUNT_FORMS,
    CURRENCY_PATTERN,
    clean_text,
    currency_sql,
    has_time,
    month_words_sql,
    needs_letters,
    null_tokens_sql,
    number_sql,
    strptime_sql,
    typed_sql,
)

MAX_DISTINCT = 50_000  # text columns with more distinct values get the SQL-only checks
MAX_EXAMPLES = 3
DIGIT = re.compile(r"\d")
LETTER = re.compile("[A-Za-z]")  # N19: B-11 tries the formats needing a letter only then
PLAIN = re.compile(r"[^0-9.,+%-]")  # N18: B-07 reads the forms of the other values only
SENTINEL_SHARE = 0.005  # B-09, B-16: a value standing for "missing" is at least 0.5 % of rows
REPEATED_COLUMNS = 3  # N09: a sentinel found in this many number columns is the file's code
NEGATIVE_SHARE = 0.05  # B-10: fewer negatives than this in an amount or a quantity
LENGTH_SHARE = 0.9  # B-13: the dominant length of a code
DATE_COVERAGE = 0.95  # B-11: share of values the chosen formats must read
VALUES_TABLE = "quantilo_clean_values"  # N43: the text columns' distinct values (`profile`)
_TABLES = itertools.count(1)  # one table per profile: two on one connection do not meet

# B-04 (decision 8): sure tokens are automatic, those that can be a real category are suggested.
SURE_TOKENS = {"", "n/a", "#n/a", "#na", "nan", "null", "nil", "#value!", "#ref!"}
SURE_TOKENS |= {"#div/0!", "#null!", "#num!", "#name?", "(blank)", "not available", "missing"}
WEAK_TOKENS = {"none", "unknown", "-", "--", "?", "tbd", "n.a.", "undefined", "not applicable"}
# 093 (CLN-8): words for "not given" that can also be a real answer.
WEAK_TOKENS |= {"unspecified", "not specified", "not stated", "inconnu"}
# N07 (audit 2026-11-12): French words and Excel's French error codes (kiwi 'n/d', a French probe).
FRENCH_SURE = {"n/d", "n.d.", "#n/d", "#valeur!", "#nom?", "#nombre!", "non communiqué"}
FRENCH_SURE |= {"non communique", "non renseigné", "non renseigne"}
FRENCH_WEAK = {"nc", "n.c.", "nd", "néant", "neant", "s.o.", "sans objet"}
SURE_TOKENS |= FRENCH_SURE
# 'NC', 'ND' are also codes (North Carolina, North Dakota): not read in a column of 2-letter codes.
SHORT_CODES = {"nc", "nd"}
CODE_COLUMN_SHARE = 0.5
DASHES = {chr(0x2014), chr(0x2013)}  # em and en dashes, alone in a cell
WEAK_TOKENS |= FRENCH_WEAK | DASHES | {"unknown/invalid", "not disclosed", "(none)"}
# N06 (audit 2026-11-12): a bare 'NA' is often "none" (Ames: no garage, no basement), not missing.
WEAK_TOKENS |= {"na"}
REAL_NA = {"na", "n/a"}  # 093 (CLN-4): the most frequent of few levels may mean "none"
REAL_NA_LEVELS = 12
# N06 (B-04:none): 'NA' on the rows where a number column of the same subject is 0 or empty.
NONE_TOKEN = "na"
NONE_COVER = 0.95  # share of the 'NA' rows where that number is 0 or empty
NONE_SHARE = 0.5  # share of the rows where it is 0 or empty that hold 'NA'
NONE_VALUE = "None"
NONE_SUBJECT_SKIP = {"total", "num", "nb", "no", "number", "count", "is", "has", "the", "of"}
# N08 (B-04:file): a weak token that is a minority value in two columns or more is the file's own
# code for "missing" (adult, diabetes: '?'), read as empty in every column that holds it.
FILE_TOKENS = {"?", "-", "--", "unknown"}
FILE_SHARE = 0.3  # at most this share of the rows in each column that makes it the file's code
FILE_COLUMNS = 2
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
    "B-24": 5,  # N47: the real names first; the other fixes are found again on them
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
    "B-22": 135,  # N10: a 0 that means 'never' in a year column, before B-10 reads the years
    "B-16": 140,
    "B-10": 150,
    "B-14": 160,
    "B-23": 161,  # N22 (rules/keys.py): a key that repeats with differences, after the copies
    "B-19": 170,
    "B-20": 165,  # fix1: after the value fixes, before B-19 can drop a constant year column
    "B-21": 166,  # fix1: rows that sum up other rows (World, Total), after the value fixes
}
# 096 (spec catalogue): how each rule changes the dataset; by step for the rules with two steps.
IMPACT: dict[str, m.Impact] = {
    "B-01": "format",
    "B-02": "format",
    "B-05": "format",  # a map_values, but only the encoding changes
    "B-06": "format",
    "B-03": "values",
    "B-04": "values",
    "B-09": "values",
    "B-22": "values",  # N10: 0 read as empty (or, without a step, a 0 worth a look)
    "B-16": "values",
    "B-11": "types",
    "B-17": "types",
    "B-18": "types",
    "B-14": "rows",
    "B-15": "rows",
    "B-19": "columns",
    "B-20": "columns",  # fix1: a new date column assembled from its parts
    "B-21": "rows",  # fix1: rows that sum up other rows, left out
    "B-24": "columns",  # N47: a later row read as the column names
    "B-23": "rows",  # N22: one row per key
}
STEP_IMPACT: dict[tuple[str, str], m.Impact] = {
    ("B-07", "map_values"): "values",  # bounds in words
    ("B-07", "change_type"): "types",
    ("B-07", "extract_text"): "columns",  # N18: the currency into its own column
    ("B-10", "null_outside"): "values",
    ("B-10", "null_values"): "values",  # N11: impossible years emptied
    ("B-10", "filter_rows"): "rows",
    ("B-12", "replace_values"): "format",  # a trailing '.0'
    ("B-18", "fill_empty"): "values",  # N27: the blanks of a checkbox written false
    ("B-12", "change_type"): "types",
    ("B-13", "pad_zeros"): "values",
    ("B-13", "change_type"): "types",
    ("B-23", "conditional_column"): "columns",  # N22: the repeated keys only marked
}


def n(count: int | float) -> str:
    return f"{count:,}"


def rows_of(count: int, noun: str) -> str:
    """'1 empty row', '7 empty rows'."""
    return f"{n(count)} {noun}" + ("" if count == 1 else "s")


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


def _plain(value: str) -> bool:
    """Printable ASCII with single inner spaces and no '&': no invisible character, no space to
    fix, no UTF-8 read as Windows-1252 (non-ASCII), no HTML code (`text_stages` skips it)."""
    return (
        value.isascii()
        and value.isprintable()
        and "&" not in value
        and "  " not in value
        and not value.startswith(" ")
        and not value.endswith(" ")
    )


def text_stages(values: list[tuple[str, int]], keep_newlines: bool = False) -> TextStages:
    stages = TextStages(values)
    final: dict[str, int] = defaultdict(int)
    for raw, count in values:
        value = raw
        if _plain(value):  # N43: no text rule changes it
            final[value] += count
            continue
        cleaned = tx.remove_invisible(value)
        if cleaned != value:
            stages.invisible[value] = cleaned
            value = cleaned
        spaced = tx.normalize_spaces(value, keep_newlines)
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
    # N43: a temporary table (k = column position, v = distinct value, c = its rows) of the
    # text columns with `stages`, made by `profile`; dropped by `close`
    values_table: str | None = None

    def one(self, sql: str) -> tuple[Any, ...]:
        row = self.connection.execute(sql).fetchone()
        assert row is not None
        return row

    def close(self) -> None:
        if self.values_table is not None:
            with contextlib.suppress(duckdb.Error):
                self.connection.execute(f"DROP TABLE IF EXISTS {self.values_table}")
            self.values_table = None

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
        op = step.op if step is not None else ""
        impact: m.Impact | None = STEP_IMPACT.get((rule, op)) or IMPACT.get(rule)
        assert impact is not None, f"{rule}: no impact in the catalogue"
        suggestion = CleanSuggestion(
            id=f"{rule}:{target}" + (f":{key}" if key else ""),
            rule=rule,
            mode=mode,
            impact=impact,
            risk="safe" if mode == "A" else "review",  # CL-02
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
    texts = [c for c in columns if is_text(c.dtype) and 0 < c.distinct <= MAX_DISTINCT]
    if texts:
        # N43: every text column's distinct values in one scan (UNPIVOT), kept in a temporary
        # table that B-07 and B-11 read instead of the rows (`Context.values_table`)
        table = f"{VALUES_TABLE}_{next(_TABLES)}"
        picked = ", ".join(f'{ident(c.name)} AS "{c.position}"' for c in texts)
        connection.execute(
            f"CREATE OR REPLACE TEMP TABLE {table} AS SELECT CAST(k AS INTEGER) AS k, v, "
            f"count(*) AS c FROM (UNPIVOT (SELECT {picked} FROM {source}) ON COLUMNS(*) "
            "INTO NAME k VALUE v) GROUP BY ALL"
        )
        context.values_table = table
        found: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for position, value, count in connection.execute(
            f"SELECT k, v, c FROM {table} ORDER BY k, c DESC, v"
        ).fetchall():
            found[int(position)].append((str(value), int(count)))
        for column in texts:
            column.values = found.get(column.position, [])
            context.stages[column.name] = text_stages(column.values, free_text(column))
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


def free_text(column: Column) -> bool:
    """N29: a column of free text (role `free_text`, by the typing or the name: comments, notes,
    descriptions), whose line breaks B-01 keeps."""
    view_role = column.view.role if column.view else None
    return view_role == "free_text" or (column.role is not None and column.role.id == "free_text")


def _large_text(context: Context, column: Column, rule: str, pattern: str, replace: str) -> int:
    """SQL-only count of the rows a text normalization changes (too many distinct values)."""
    name = ident(column.name)
    if rule == "B-01":
        expr = tx.normalize_spaces_sql(name, free_text(column))
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
                say(
                    "B-02",
                    "Invisible characters (zero-width spaces, byte-order marks, control "
                    "characters) found in {count} values.",
                    count=n(cells),
                ),
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
            keep = free_text(column)
            context.add(
                "B-01",
                "A",
                column,
                f"Trim extra spaces in {column.name}",
                say(
                    "B-01.free_text" if keep else "B-01",
                    "{count} values had extra spaces (at the ends, doubled, tabs or no-break "
                    "spaces). Line breaks are kept: a free text keeps its paragraphs (N29)."
                    if keep
                    else "{count} values had extra spaces (at the ends, doubled, tabs or no-break "
                    "spaces).",
                    count=n(cells),
                ),
                m.NormalizeSpaces(id=_id(), columns=[column.name], keep_newlines=keep),
                cells=cells,
                examples=examples,
            )


def b05_encoding(context: Context) -> None:
    for column, stages in context.texts():
        if not stages.encoding:
            continue
        cells = _changed(stages, stages.encoding)
        mode: Mode = "S" if stages.unfixable else "A"
        first = next(iter(stages.encoding.items()))
        reason = say(
            "B-05",
            "{count} values were saved in one encoding and read in another: {before} → {after} "
            "(wrong encoding).",
            count=n(cells), before=q(first[0]), after=q(first[1]),
        )  # fmt: skip
        if stages.unfixable:
            reason += say(
                "B-05.unfixable",
                " {damaged} more values look damaged but cannot be repaired safely.",
                damaged=n(stages.unfixable),
            )
        context.add(
            "B-05",
            mode,
            column,
            f"Repair the encoding of {column.name}",
            reason,
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
            say("B-06", "HTML codes such as '&amp;' decoded in {count} values.", count=n(cells)),
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


def _tokens(stages: TextStages) -> tuple[dict[str, int], dict[str, int]]:
    """B-04: the sure and the weak words for "missing" among a column's values, with their rows."""
    sure: dict[str, int] = {}
    weak: dict[str, int] = {}
    total = sum(c for _, c in stages.final)
    letters = sum(c for v, c in stages.final if len(v.strip()) == 2 and v.strip().isalpha())
    codes = bool(total) and letters >= CODE_COLUMN_SHARE * total  # states, countries: 'NC', 'ND'
    for value, count in stages.final:
        token = value.strip().lower()
        if token in SURE_TOKENS:
            sure[value] = count
        elif token in WEAK_TOKENS and not (codes and token in SHORT_CODES):
            weak[value] = count
    return sure, weak


def _file_tokens(context: Context, found: list[tuple[Column, dict[str, int]]]) -> dict[str, int]:
    """N08: the weak tokens that are a minority value (30 % of the rows at most) in two columns or
    more: the file's own code for "missing", and in how many columns it is found."""
    minority: dict[str, int] = defaultdict(int)
    present: dict[str, int] = defaultdict(int)
    for _, weak in found:
        rows: dict[str, int] = defaultdict(int)
        for value, count in weak.items():
            token = value.strip().lower()
            if token in FILE_TOKENS:
                rows[token] += count
        for token, count in rows.items():
            present[token] += 1
            if count <= FILE_SHARE * context.rows:
                minority[token] += 1
    return {t: present[t] for t, k in minority.items() if k >= FILE_COLUMNS}


def b04_tokens(context: Context) -> None:
    columns = [(c, s, *_tokens(s)) for c, s in context.texts() if not _converted(c)]
    shared = _file_tokens(context, [(c, weak) for c, _, _, weak in columns])
    for column, stages, sure, weak in columns:
        notes: list[str] = []
        if _none_reading(context, column, weak):
            weak = {v: c for v, c in weak.items() if v.strip().lower() != NONE_TOKEN}
        real = _real_na(stages, {**sure, **weak})
        if real is not None:
            weak[real] = sure.pop(real, None) or weak[real]
            notes.append(
                say(
                    "B-04.real",
                    " {real} is the most frequent value of only {others} others: it may mean "
                    "'none' (nothing of the kind, as 'no pool'), a real category (CLN-4).",
                    real=q(real),
                    others=len(stages.final) - 1,
                )
            )
        suppressed = _suppressed(stages, {**sure, **weak})
        if suppressed:
            weak.update(suppressed)
            notes.append(
                say(
                    "B-04.suppressed",
                    " They sit in a column of numbers: likely suppressed or hidden values, which "
                    "become empty so the numbers can be read.",
                )
            )
        coded = {
            v: c for v, c in weak.items() if v.strip().lower() in shared and v not in suppressed
        }
        weak = {v: c for v, c in weak.items() if v not in coded}
        levels: list[tuple[Mode, dict[str, int], str | None]] = [
            ("A", sure, None),
            ("A", coded, "file"),
            ("S", weak, "maybe"),
        ]
        for mode, found, key in levels:
            if not found:
                continue
            cells = sum(found.values())
            shown = ", ".join(q(v) for v in list(found)[:4])
            blank = all(v.strip() == "" for v in found)
            reason = (
                say("B-04.blank", "{count} blank strings read as missing.", count=n(cells))
                if blank
                else say(
                    "B-04",
                    "{count} values are words for 'missing' ({values}): read as empty.",
                    count=n(cells), values=shown,
                )
            )  # fmt: skip
            if key == "file":
                token = next(iter(found)).strip().lower()
                reason = say(
                    "B-04.file",
                    "{value} stands for 'missing' in {columns} columns of this file ({rows} in "
                    "{column}): read as empty everywhere (N08). Check your data.",
                    value=q(token), columns=shared[token], rows=rows_of(cells, "row"),
                    column=column.name,
                )  # fmt: skip
            if any(v.strip().lower() in FRENCH_SURE | FRENCH_WEAK for v in found):
                reason += say(
                    "B-04.french",
                    " Words for 'not given' in French and in Excel's French error codes.",
                )
            if mode == "S":
                for note in notes:
                    reason += note
                reason += say("B-04.category", " Check they are not a real category.")
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
                key=key,
            )
            context.stages[column.name].final = [(v, c) for v, c in stages.final if v not in found]


def _real_na(stages: TextStages, tokens: dict[str, int]) -> str | None:
    """CLN-4: 'NA' as the most frequent value of a column with few levels often means 'none'
    (no alley, no pool): suggested, not automatic."""
    values = [(v, c) for v, c in stages.final if v.strip()]
    if not values or len(values) - 1 > REAL_NA_LEVELS:
        return None
    top = values[0][0]
    return top if top in tokens and top.strip().lower() in REAL_NA else None


def _number_sql(column: Column) -> str | None:
    """A column's value as a number: stored as one, or text the typing reads as numbers."""
    if column.type in ("identifier", "boolean"):
        return None
    if is_number(column.dtype):
        return ident(column.name)
    view = column.view
    if is_text(column.dtype) and view is not None and view.parser.kind == "number":
        return typed_sql(view.name, view.parser)
    return None


def _same_subject(subject: str, name: str) -> bool:
    """'garage' in 'Garage Area', 'bsmt' in 'Total Bsmt SF', 'fireplace' in 'Fireplaces'."""
    return any(w in (subject, subject + "s") or subject == w + "s" for w in semantics.tokens(name))


def _none_reading(context: Context, column: Column, weak: dict[str, int]) -> bool:
    """N06 (B-04:none): a bare 'NA' on the rows where a number column of the same subject is 0 or
    empty ('Garage Type' NA where 'Garage Area' is 0) means "none of it" (no garage), not a missing
    value: written 'None' (mode S) so it stays a category. True when proposed."""
    found = {v: c for v, c in weak.items() if v.strip().lower() == NONE_TOKEN}
    words = semantics.tokens(column.name)
    if not found or not _is_category(column) or not words:
        return False
    subject = words[0]
    if len(subject) < 3 or subject.isdigit() or subject in NONE_SUBJECT_SKIP:
        return False
    partners = [
        (c, expr)
        for c in context.columns
        if c is not column
        and _same_subject(subject, c.name)
        and (expr := _number_sql(c)) is not None
    ]
    if not partners:
        return False
    na = f"lower(trim({ident(column.name)})) = {lit(NONE_TOKEN)}"
    parts = [f"count(*) FILTER (WHERE {na})"]
    for _, expr in partners:
        zero = f"coalesce({expr}, 0) = 0"
        parts += [
            f"count(*) FILTER (WHERE {na} AND {zero})",
            f"count(*) FILTER (WHERE {zero})",
            f"count(*) FILTER (WHERE {na} AND {expr} IS NULL)",
        ]
    row = context.one(f"SELECT {', '.join(parts)} FROM {context.source}")
    rows = int(row[0])
    best: tuple[float, Column, int, int] | None = None
    for index, (partner, _) in enumerate(partners):
        covered, zeros, empty = (int(v) for v in row[1 + 3 * index : 4 + 3 * index])
        if not rows or covered < NONE_COVER * rows or covered < NONE_SHARE * zeros:
            continue
        share = covered / zeros
        if best is None or share > best[0]:
            best = (share, partner, covered, empty)
    if best is None:
        return False
    _, partner, covered, empty = best
    how = "empty" if empty == covered else "0 or empty" if empty else "0"
    shown = q(next(iter(found)))
    where = (
        say(
            "B-04.none",
            "{value} sits on the {rows} rows where {partner} is {how}: it means 'no {subject}', "
            "not a missing value. Written '{written}' so it stays a category (N06). Check your "
            "data.",
            value=shown, rows=n(rows), partner=partner.name, how=how, subject=subject,
            written=NONE_VALUE,
        )
        if covered == rows
        else say(
            "B-04.none_some",
            "{value} sits on {rows} rows, {covered} of them where {partner} is {how}: it means "
            "'no {subject}', not a missing value. Written '{written}' so it stays a category "
            "(N06). Check your data.",
            value=shown, rows=n(rows), covered=n(covered), partner=partner.name, how=how,
            subject=subject, written=NONE_VALUE,
        )
    )  # fmt: skip
    context.add(
        "B-04",
        "S",
        column,
        f"Write {shown} as '{NONE_VALUE}' (no {subject}) in {column.name}",
        where,
        m.MapValues(id=_id(), column=column.name, map=dict.fromkeys(found, NONE_VALUE)),
        cells=sum(found.values()),
        examples=[(v, NONE_VALUE) for v in found],
        key="none",
    )
    stages = context.stages[column.name]
    stages.final = [(v, c) for v, c in stages.final if v not in found]
    return True


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
        if accented:
            reason = say(
                "B-03.accented",
                "{others} → {value} (the accented spelling, kept (CLN-5)): same value up to "
                "case or accents. {cells} cells in {groups} groups.",
                others=others, value=q(biggest[0][0]), cells=n(cells), groups=len(merged),
            )  # fmt: skip
        else:
            reason = say(
                "B-03",
                "{others} → {value} (most frequent): same value up to case or accents. {cells} "
                "cells in {groups} groups.",
                others=others, value=q(biggest[0][0]), cells=n(cells), groups=len(merged),
            )  # fmt: skip
        if person:
            reason += say(
                "B-03.person",
                " Names of people: two spellings may be two people, so the merge is suggested, "
                "not applied (CLN-5).",
            )
        context.add(
            "B-03",
            "S" if person else "A",
            column,
            title,
            reason,
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
                say(
                    "B-12",
                    "An identifier stored as decimal numbers: all {count} values end in '.0' "
                    "({before} → {after}).",
                    count=n(column.nonnull),
                    before=q(sample),
                    after=q(sample[:-2]),
                ),
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
        mode: Mode = "A" if postal else "S"
        reason = say(
            "B-13" if postal else "B-13.codes",
            "{count} postal codes are shorter than the usual {digits} digits ({share} of "
            "values): they lost their leading zero."
            if postal
            else "{count} codes are shorter than the usual {digits} digits ({share} of values): "
            "they lost their leading zero.",
            count=n(shorter),
            digits=dominant,
            share=f"{share:.0%}",
        )
        step = m.PadZeros(id=_id(), column=column.name, length=dominant)
        if postal:
            guard = _postal_countries(context, column, dominant)
            if guard is None:
                continue  # every shorter code is of a country whose codes are that short
            if guard.mode is not None:
                mode, reason, cells = guard.mode, guard.reason, guard.padded
                step.conditions = guard.conditions
                shorter = cells
        context.add(
            "B-13",
            mode,
            column,
            f"Restore the leading zeros of {column.name}",
            reason,
            step,
            cells=shorter,
            examples=[("5207", "05207")] if dominant == 5 else [],
        )


@dataclass
class PostalGuard:
    """N12: what the country column says of the shorter postal codes (no mode: no country)."""

    mode: Mode | None = None
    reason: str = ""
    padded: int = 0
    conditions: list[m.RowCondition] = field(default_factory=list)


def _countries(found: list[tuple[str, int]]) -> str:
    names = [name for name, _ in found[:3]]
    more = len(found) - len(names)
    return ", ".join(names) + (f" and {more} more" if more else "")


def _postal_countries(context: Context, column: Column, dominant: int) -> PostalGuard | None:
    """N12 (B-13 guard): with a country column, the shorter postal codes lost their zero only in
    the rows of a country whose codes have the usual length (US, France, Germany…); Belgian or Swiss
    codes have 4 digits by nature. Automatic only when every shorter code is of such a country;
    else suggested, padding those rows only. None when no shorter code needs a zero."""
    country = next(
        (
            c
            for c in context.columns
            if c is not column
            and c.nonnull
            and (
                (c.role is not None and c.role.id == "country")
                or (c.view and c.view.role == "country")
            )
        ),
        None,
    )
    if country is None:
        return PostalGuard()
    text = f"trim({as_text(column.name, column.dtype)})"
    rows = context.connection.execute(
        f"SELECT {as_text(country.name, country.dtype)}, count(*) FROM {context.source} "
        f"WHERE {ident(column.name)} IS NOT NULL AND length({text}) < {dominant} "
        "GROUP BY 1 ORDER BY 2 DESC, 1"
    ).fetchall()
    lengths = dict(load().postal_lengths)
    padded: list[tuple[str, int]] = []
    natural: dict[int, list[tuple[str, int]]] = defaultdict(list)
    unknown: list[tuple[str, int]] = []
    for value, count in rows:
        name = tx.normalize_spaces(str(value)) if value is not None else None
        length = lengths.get(name.lower()) if name else None
        if length == dominant and name:
            padded.append((name, int(count)))
        elif length is not None and name:
            natural[length].append((name, int(count)))
        else:
            unknown.append((name or "no country", int(count)))
    if not padded and not unknown:
        return None
    pad = sum(c for _, c in padded)
    reason: str = (
        say(
            "B-13.countries",
            "{count} postal codes of {countries} rows are shorter than the usual {digits} "
            "digits: they lost their leading zero.",
            count=n(pad),
            countries=_countries(padded),
            digits=dominant,
        )
        if padded
        else ""
    )
    if not natural and not unknown:
        return PostalGuard("A", reason + say("CHECK", " Check your data."), pad)
    for length, found in sorted(natural.items()):
        reason += say(
            "B-13.natural",
            " {rows} of {countries} have {length}-digit codes by nature: not padded.",
            rows=rows_of(sum(c for _, c in found), "row"),
            countries=_countries(found),
            length=length,
        )
    named = [name for name, _ in unknown if name != "no country"]
    if unknown:
        reason += say(
            "B-13.unknown",
            " {rows} of {countries} are shorter too, with no known length for their country: "
            "padded if you tick it.",
            rows=rows_of(sum(c for _, c in unknown), "row"),
            countries=_countries(unknown),
        )
    targets = [name for name, _ in padded] + named
    conditions = (
        [m.RowCondition(column=country.name, op="in", values=targets)]
        if natural and targets
        else []
    )
    if natural and not targets:
        return None
    cells = pad + sum(c for name, c in unknown if not natural or name != "no country")
    said = reason + say("CHECK", " Check your data.")
    return PostalGuard("S", said if said == said.strip() else said.strip(), cells, conditions)


# --- Types -------------------------------------------------------------------------------------


def _number_tests(raw: str, cleaned: str, number: str) -> list[str]:
    """B-07's five tests on a value: read as a number, unreadable, with decimals, a blank or a
    word for 'missing', written another way once read."""
    shown = (
        f"CASE WHEN {number} = trunc({number}) AND abs({number}) < 1e15 THEN "
        f"CAST(CAST({number} AS BIGINT) AS VARCHAR) ELSE CAST({number} AS VARCHAR) END"
    )
    return [
        f"{number} IS NOT NULL",
        f"{cleaned} IS NOT NULL AND {number} IS NULL",
        f"{number} IS NOT NULL AND {number} <> trunc({number})",
        f"{raw} IS NOT NULL AND {cleaned} IS NULL",
        f"{number} IS NOT NULL AND {raw} <> {shown}",
    ]


@dataclass
class NumberCounts:
    """B-07 on one column: its five counts, and where its first examples are read."""

    counts: tuple[int, ...]
    good: str  # SQL: the first three values rewritten once read, with their number
    numbers: dict[str, float] = field(default_factory=dict)  # those values → their number


def _number_counts(context: Context, columns: list[Column]) -> dict[str, NumberCounts]:
    """N43: B-07's counts for every column whose distinct values are known, in one query per
    decimal mark on those values (each weighted by its rows), not one per column on the rows:
    each test only reads the value, so the counts are the same. The values rewritten once read
    go to a temporary table, where the examples' query finds them."""
    table = context.values_table
    known = [c for c in columns if c.name in context.stages]
    if table is None or not known:
        return {}
    found: dict[str, NumberCounts] = {}
    good = f"{table}_numbers"
    groups: dict[str, list[Column]] = defaultdict(list)
    for column in known:
        assert column.view is not None
        groups[column.view.parser.decimal].append(column)
    for index, (decimal, group) in enumerate(groups.items()):
        keys = ", ".join(str(c.position) for c in group)
        read = (
            f"(SELECT k, v, c, t, {number_sql('t', decimal)} AS x FROM (SELECT k, v, c, "
            f"{clean_text('v')} AS t FROM {table} WHERE k IN ({keys})))"
        )
        tests = _number_tests("v", "t", "x")
        rows = context.connection.execute(
            "SELECT k, "
            + ", ".join(f"coalesce(sum(c) FILTER (WHERE {test}), 0)" for test in tests)
            + f" FROM {read} GROUP BY k"
        ).fetchall()
        context.connection.execute(
            f"{'CREATE OR REPLACE' if index == 0 else 'INSERT INTO'} "
            + (f"TEMP TABLE {good} AS " if index == 0 else f"{good} ")
            + f"SELECT k, v, x FROM {read} WHERE x IS NOT NULL AND v <> CAST(x AS VARCHAR)"
        )
        counts = {int(k): tuple(int(v) for v in values) for k, *values in rows}
        for column in group:
            name = ident(column.name)
            found[column.name] = NumberCounts(
                counts.get(column.position, (0, 0, 0, 0, 0)),
                f"SELECT {name} FROM {context.source} WHERE {name} IN "
                f"(SELECT v FROM {good} WHERE k = {column.position}) LIMIT 3",
            )
    for k, v, x in context.connection.execute(f"SELECT k, v, x FROM {good}").fetchall():
        column = next(c for c in known if c.position == int(k))
        found[column.name].numbers[str(v)] = x
    return found


def b07_numbers(context: Context) -> None:
    candidates = []
    for column in context.columns:
        view = column.view
        if not (is_text(column.dtype) and view and view.parser.kind == "number"):
            continue
        if view.type not in ("numeric_continuous", "numeric_discrete"):
            continue
        code = semantics.code_by_name(column.name)
        if _leading_zeros(context, column) or (code is not None and code.kind == "identifier"):
            continue  # CLN-6: a code ('00928', 'PID') is not a number (B-13 says so)
        candidates.append(column)
    try:
        _b07_numbers(context, candidates, _number_counts(context, candidates))
    finally:
        if context.values_table is not None:
            context.connection.execute(f"DROP TABLE IF EXISTS {context.values_table}_numbers")


def _b07_numbers(
    context: Context, candidates: list[Column], batched: dict[str, NumberCounts]
) -> None:
    for column in candidates:
        view = column.view
        assert view is not None
        cleaned = clean_text(column.name)
        number = number_sql(cleaned, view.parser.decimal)
        known = batched.get(column.name)
        if known is None:  # more distinct values than the profile reads: tested on the rows
            tests = _number_tests(ident(column.name), cleaned, number)
            row = context.one(
                "SELECT "
                + ", ".join(f"count(*) FILTER (WHERE {test})" for test in tests)
                + f" FROM {context.source}"
            )
            counts = tuple(int(v) for v in row)
        else:
            counts = known.counts
        read, unreadable, fractional, tokens, rewritten = counts
        if not read:
            continue
        bad = (
            [
                str(v[0])
                for v in context.connection.execute(
                    f"SELECT {cleaned} FROM {context.source} WHERE {cleaned} IS NOT NULL AND "
                    f"{number} IS NULL GROUP BY 1 ORDER BY count(*) DESC LIMIT 3"
                ).fetchall()
            ]
            if unreadable
            else []
        )
        if known is None:
            good = context.connection.execute(
                f"SELECT {ident(column.name)}, {number} FROM {context.source} WHERE {number} IS "
                f"NOT NULL AND {ident(column.name)} <> CAST({number} AS VARCHAR) LIMIT 3"
            ).fetchall()
        else:
            good = [
                (v, known.numbers[str(v)])
                for (v,) in context.connection.execute(known.good).fetchall()
            ]
        integer = fractional == 0 and view.type == "numeric_discrete"
        how = "decimal comma" if view.parser.decimal == "," else "decimal point"
        # N18: the ways of writing an amount the reading met, its currencies and a negative
        # example, in one query on the distinct numbers read
        brackets = lit(AMOUNT_FORMS[4][1] + "|" + AMOUNT_FORMS[5][1])
        # values made of digits, separators and % only ('69,55%') have none of these forms
        stages = context.stages.get(column.name)
        plain = stages is not None and not any(PLAIN.search(v) for v, _ in stages.final)
        row: tuple[Any, ...] = (0,) * len(AMOUNT_FORMS) + ([], None)
        distinct = (  # N43: the distinct values are known, the rows need not be read again
            f"SELECT DISTINCT {cleaned} AS v FROM {context.source}"
            if known is None
            else f"SELECT DISTINCT {clean_text('v')} AS v FROM {context.values_table} "
            f"WHERE k = {column.position}"
        )
        if not plain:
            row = context.one(
                "SELECT "
                + "".join(
                    f"count(*) FILTER (WHERE regexp_matches(v, {lit(regex)})), "
                    for _, regex in AMOUNT_FORMS
                )
                + f"list(DISTINCT {currency_sql('v', code=True)}), "
                + f"min(v) FILTER (WHERE x < 0 AND regexp_matches(v, {brackets})) "
                + f"FROM (SELECT v, {number_sql('v', view.parser.decimal)} AS x FROM "
                + f"({distinct})) WHERE x IS NOT NULL"
            )
        found_forms, currencies = row[: len(AMOUNT_FORMS)], sorted(c for c in row[-2] if c)
        forms = [label for (label, _), k in zip(AMOUNT_FORMS, found_forms, strict=True) if k]
        if len(currencies) == 1:
            forms = [f"in {currencies[0]}" if f == "a currency" else f for f in forms]
        reason = say(
            "B-07",
            "{count} values are numbers stored as text ({forms}).",
            count=n(read), forms=", ".join([how, *forms]),
        )  # fmt: skip
        negative = row[-1]
        if negative:
            (value,) = context.one(f"SELECT {number_sql(lit(negative), view.parser.decimal)}")
            reason += say(
                "B-07.negative",
                " {value} becomes {number}.",
                value=q(str(negative)), number=f"{value:.15g}",
            )  # fmt: skip
        if len(currencies) > 1:
            shown_currencies = ", ".join(currencies[:4])
            context.add(
                "B-07",
                "S",
                column,
                f"Split the currency of {column.name} into its own column",
                f"{column.name} mixes {len(currencies)} currencies ({shown_currencies}): stored "
                "as numbers they would add up as one. The currency goes first to a new column "
                f"'{column.name} currency'; nothing is converted (no exchange rate is known). "
                "Check your data.",
                m.ExtractText(
                    id=_id(),
                    column=column.name,
                    how="regex",
                    pattern=CURRENCY_PATTERN,
                    to=f"{column.name} currency",
                ),
                cells=read,
                key="currency",
            )
            reason += say(
                "B-07.currencies",
                " It mixes {currencies} currencies: split them first.",
                currencies=len(currencies),
            )
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
            reason += say(
                "B-07.bounds",
                " Apply the bounds suggestion first to keep the values written in words.",
            )
        if tokens:
            reason += say(
                "B-07.missing",
                " {tokens} blank or 'missing' values become empty.",
                tokens=n(tokens),
            )
        if unreadable:
            reason += say(
                "B-07.unreadable",
                " {unreadable} values cannot be read and become empty: {unread}.",
                unreadable=n(unreadable), unread=", ".join(q(v) for v in bad),
            )  # fmt: skip
        automatic = not unreadable and len(currencies) < 2
        if automatic:
            reason += say("CHECK", " Check your data.")
        context.add(
            "B-07",
            "A" if automatic else "S",
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
    swapped = _swapped(ALL_DATE_FORMATS[index])
    return ALL_DATE_FORMATS.index(swapped) if swapped in ALL_DATE_FORMATS else index


def _greedy(values: list[tuple[str, int, list[bool]]]) -> list[int]:
    """Up to three formats (their indexes), each reading most of the values still unread."""
    chosen: list[int] = []
    unread = list(values)
    for _ in range(3):
        gains = [0] * len(ALL_DATE_FORMATS)
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
    distinct values; day/month order decided on the whole column (values with a day above 12).
    N43: the formats are tried on every candidate column's distinct values in one query."""
    candidates: list[tuple[Column, int]] = []
    for column, stages in context.texts():
        view = column.view
        if view and view.type not in ("datetime", "categorical_nominal", "text", "identifier"):
            continue
        tokens = set(load().null_tokens)  # N19: 'NULL' becomes empty (B-04): no unread date
        total = sum(c for v, c in stages.final if v.strip() and v.strip().lower() not in tokens)
        digits = sum(c for v, c in stages.final if DIGIT.search(v))
        if not total or digits < 0.5 * total:
            continue
        candidates.append((column, total))
    if not candidates or context.values_table is None:
        return
    value = f"trim(regexp_replace(v, {lit(tx.SQL_SPACE_RUN)}, ' ', 'g'))"
    # a column without a letter skips the formats needing one (month names, AM/PM, ISO's T)
    lettered = {
        c.position
        for c, _ in candidates
        if any(LETTER.search(v) for v, _ in context.stages[c.name].final)
    }
    keys = ", ".join(str(c.position) for c, _ in candidates)
    letters = f"k IN ({', '.join(str(k) for k in sorted(lettered))})" if lettered else "false"
    tests = ", ".join(
        f"{strptime_sql('v', f, 'm')} IS NOT NULL"
        if not needs_letters(f)
        else (f"({letters} AND {strptime_sql('v', f, 'm')} IS NOT NULL)" if lettered else "false")
        for f in ALL_DATE_FORMATS
    )
    months = f"CASE WHEN {letters} THEN {month_words_sql('v')} ELSE v END" if lettered else "v"
    tried: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
    for k, *row in context.connection.execute(
        f"SELECT k, v, c, [{tests}], {SERIAL} FROM (SELECT k, v, c, {months} AS m FROM "
        f"(SELECT k, {value} AS v, sum(c) AS c FROM {context.values_table} WHERE k IN ({keys}) "
        f"GROUP BY 1, 2) WHERE v <> '' AND lower(v) NOT IN {null_tokens_sql()})"
    ).fetchall():
        tried[int(k)].append(tuple(row))
    for column, total in candidates:
        _b11_dates(context, column, total, tried[column.position], value)


def _b11_dates(
    context: Context, column: Column, total: int, rows: list[tuple[Any, ...]], value: str
) -> None:
    stages = context.stages[column.name]
    tokens = set(load().null_tokens)
    values = [(str(v), int(c), [bool(r) for r in reads]) for v, c, reads, _ in rows]
    by_format = [sum(c for _, c, reads in values if reads[i]) for i in range(len(ALL_DATE_FORMATS))]
    if max(by_format, default=0) < 0.3 * total:
        return
    indexes = _greedy(values)
    for position, index in enumerate(indexes):  # the order reading more values wins
        swapped = _swap_index(index)
        if by_format[swapped] > by_format[index]:
            indexes[position] = swapped
    chosen = [ALL_DATE_FORMATS[i] for i in indexes]
    serials = sum(int(c) for _, c, reads, serial in rows if serial and not any(reads))
    read = sum(c for _, c, reads in values if any(reads[i] for i in indexes))
    excel = serials > 0
    covered = read + serials
    if covered < DATE_COVERAGE * total:
        return
    ambiguous = [
        ALL_DATE_FORMATS[i]
        for i in indexes
        if _swap_index(i) != i and by_format[_swap_index(i)] >= by_format[i]
    ]
    with_time = any(has_time(f) for f in chosen)
    unread = total - covered
    mode: Mode = "S" if ambiguous or unread else "A"
    reason = say("B-11", "{count} dates stored as text", count=n(covered))
    if any("%f" in f for f in chosen):
        reason += say("B-11.milliseconds", " with milliseconds")
    if len(chosen) > 1:
        reason += say(
            "B-11.formats",
            " in {formats} formats ({listed})",
            formats=len(chosen), listed=", ".join(chosen),
        )  # fmt: skip
    if excel:
        reason += say("B-11.excel", ", {serials} as Excel day numbers", serials=n(serials))
    reason += say("B-11.end", ".")
    if ambiguous:
        reason += say("B-11.swapped", " Day and month could be swapped: check the order.")
    if unread:
        reason += say(
            "B-11.unread",
            " {unread} values cannot be read and become empty.",
            unread=n(unread),
        )
    missing = [(v.strip(), c) for v, c in stages.final if v.strip().lower() in tokens]
    if missing:
        words = ", ".join(q(v) for v, _ in missing[:3] if v) or "blank"
        count = n(sum(c for _, c in missing))
        reason += say(
            "B-11.missing",
            " {missing} values for 'missing' ({words}) become empty.",
            missing=count, words=words,
        )  # fmt: skip
    if any("%b" in f or "%B" in f for f in chosen):
        reason += say(
            "B-11.months",
            " Month names are read in English and in French (janv., févr., août…).",
        )
    parsed = _date_expr("v", chosen, excel)
    # N19: '05/01/24' read with %Y is the year 24: never stored on its own (A-14 writes the
    # century first)
    (short,) = context.one(
        f"SELECT coalesce(sum(c) FILTER (WHERE year({parsed}) < 100), 0) FROM (SELECT "
        f"{value} AS v, sum(c) AS c FROM {context.values_table} WHERE k = {column.position} "
        "GROUP BY 1)"
    )
    if short:
        mode = "S"
        reason += say(
            "B-11.century",
            " {short} dates have a two-digit year and would read as the years 0 to 99: write "
            "their century first (A-14).",
            short=n(short),
        )
    if mode == "A":
        reason += say("CHECK", " Check your data.")
    if not with_time:
        parsed = f"CAST({parsed} AS DATE)"
    reads_of = {v: reads for v, _, reads in values}
    changed = 0
    examples: list[tuple[str | None, str | None]] = []
    seen: set[int] = set()
    for text, count, after in context.connection.execute(
        f"SELECT v, c, CAST({parsed} AS VARCHAR) FROM (SELECT {value} AS v, sum(c) AS c "
        f"FROM {context.values_table} WHERE k = {column.position} GROUP BY 1) "
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
        return
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


# N27: the marks of a ticked box, alone with blanks in a column (a checkbox); and 'V' / 'F' (vrai,
# faux), read here only, so the typing of an import does not change.
MARKS = {"x", "✓", "✔", "☑", "☒", "1"}
CLEAN_PAIRS = ((("v",), ("f",)),)


def _blanks(context: Context, stages: TextStages) -> int:
    """Rows of a text column with no value: empty, or blank text."""
    return context.rows - sum(c for v, c in stages.final if v.strip())


def b18_booleans(context: Context) -> None:
    kb = load()
    pairs = (*kb.boolean_pairs, *CLEAN_PAIRS)
    trues = {w for pair in pairs for w in pair[0]}
    falses = {w for pair in pairs for w in pair[1]} - trues
    for column, stages in context.texts():
        values = [(v, c) for v, c in stages.final if v.strip()]
        words = {v.strip().lower() for v, _ in values}
        if values and words <= MARKS and len(words) == 1:
            _checkbox(context, column, stages, values)
            continue
        if len(values) < 2:
            continue
        marks = (words & MARKS) - trues
        found_true, found_false = (words & trues) | marks, words & falses
        if words - trues - falses - marks or not found_true or not found_false:
            continue
        mixed = len(values) > 2
        cells = sum(c for _, c in values)
        spellings = ", ".join(q(v) for v, _ in values[:6])
        reason = (
            say(
                "B-18.mixed",
                "{count} spellings of yes and no mixed ({values}): unified.",
                count=len(values), values=spellings,
            )
            if mixed
            else say(
                "B-18",
                "Two values ({values}) mean yes and no: stored as true / false.",
                values=spellings,
            )
        )  # fmt: skip
        if marks:  # N27: 'X' among 'Oui' / 'Non' is a ticked box, a guess said as one
            shown = ", ".join(q(v) for v, _ in values if v.strip().lower() in marks)
            reason += say(
                "B-18.marks",
                " {marks} read as yes (a ticked box); blanks stay empty. Check your data.",
                marks=shown,
            )
        context.add(
            "B-18",
            "A" if mixed and not marks else "S",
            column,
            f"Store {column.name} as true / false",
            reason,
            m.ChangeType(
                id=_id(),
                column=column.name,
                to="boolean",
                true_values=sorted(found_true),
                false_values=sorted(found_false),
            ),
            cells=cells,
            examples=[
                (v, "true" if v.strip().lower() in found_true else "false") for v, _ in values
            ],
            key="mark" if marks else None,
        )
    for column in context.columns:  # N27: a boolean read at import with only 'true' and blanks
        blanks = context.rows - column.nonnull
        if column.dtype == "BOOLEAN" and column.low == column.high == "true" and blanks:
            context.add(
                "B-18",
                "S",
                column,
                f"Read the blanks of {column.name} as false",
                f"Only true ({rows_of(column.nonnull, 'row')}) and blanks "
                f"({rows_of(blanks, 'row')}): a checkbox, where a blank means no. Blanks "
                "written false if you tick it. Check your data.",
                m.FillEmpty(id=_id(), column=column.name, value="false"),
                cells=blanks,
                examples=[(None, "false")],
                key="checkbox",
            )


def _checkbox(
    context: Context, column: Column, stages: TextStages, values: list[tuple[str, int]]
) -> None:
    """N27: one mark ('X', a tick) and blanks: a checkbox, true where ticked, false where blank."""
    blanks = _blanks(context, stages)
    if not blanks:
        return  # a constant column (B-19), not a box left blank on some rows
    marked = sum(c for _, c in values)
    shown = ", ".join(q(v) for v, _ in values[:3])
    context.add(
        "B-18",
        "S",
        column,
        f"Store {column.name} as true / false (a checkbox)",
        f"One mark ({shown}, {rows_of(marked, 'row')}) and blanks ({rows_of(blanks, 'row')}): a "
        "checkbox. Stored as true / false; a blank becomes false. Check your data.",
        m.ChangeType(
            id=_id(),
            column=column.name,
            to="boolean",
            true_values=sorted({v.strip().lower() for v, _ in values}),
            false_values=[""],  # a blank cell is false (compile.py)
        ),
        cells=marked + blanks,
        examples=[(values[0][0], "true"), (None, "false")],
        key="checkbox",
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


@dataclass
class Sentinel:
    """A value B-09 reads as a code for "missing" in one column, before it is said."""

    column: Column
    value: float
    text: str
    count: int
    mode: Mode
    reason: str
    isolated: bool  # beyond every other value of the column: far from them, or the only negative
    repeated: bool = False


def _repeated_sentinels(pending: list[Sentinel]) -> None:
    """N09 (B-09:repeated): one value found in 3 number columns or more, beyond every other value
    of each one (far from them, or their only negatives): -200 in the 13 sensors of air quality,
    -100000 in 3 of Chicago's. The file's code for "no measure": automatic in each column."""
    columns: dict[float, list[Sentinel]] = defaultdict(list)
    for item in pending:
        if item.isolated:
            columns[item.value].append(item)
    for items in columns.values():
        if len({item.column.name for item in items}) < REPEATED_COLUMNS:
            continue
        for item in items:
            item.mode = "A"
            item.repeated = True
            item.reason = (
                f"{item.text} appears in {len(items)} measure columns ({rows_of(item.count, 'row')}"
                f" in {item.column.name}), always beyond every other value: the file's code for "
                "'no measure'. Read as empty (N09). Check your data."
            )


def b09_sentinels(context: Context) -> dict[str, list[float]]:
    """-999, 9999…: frequent, and either far from the other values (beyond 3 IQR) or a peak
    (repeated ten times more than a typical value) beyond the whiskers (FAHES). 093 (CLN-1, CLN-2):
    any round value among the five most frequent ones (-200, -100000) is a candidate too when it is
    the column's extreme and far from the rest; (0, 0) positions are read as missing (CLN-3)."""
    found: dict[str, list[float]] = {}
    pending: list[Sentinel] = []
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
            head = say(
                "B-09",
                "{value} appears {count} times ({share}), ",
                value=text,
                count=n(int(count)),
                share=f"{share:.1%}",
            )
            if not listed:
                why = (
                    say(
                        "B-09.negatives",
                        "a round number beyond every other value, the only negatives of the "
                        "column: the shape of a code for 'no measure' (CLN-1)",
                    )
                    if below
                    else say(
                        "B-09.round",
                        "a round number beyond every other value and far from them: the shape of "
                        "a code for 'no measure' (CLN-1)",
                    )
                )
            elif far:
                why = say("B-09.far", "far from the other values")
            else:
                why = say(
                    "B-09.frequent",
                    "{times} times as frequent as a typical value",
                    times=f"{int(count) / repeat:,.0f}",
                )
            tail = say(
                "B-09.end",
                " (middle half {q1} to {q3}): likely a code for 'missing'.",
                q1=f"{float(q1):,.4g}",
                q3=f"{float(q3):,.4g}",
            )
            pending.append(
                Sentinel(
                    column,
                    float(value),
                    text,
                    int(count),
                    "A" if outside_role else "S",
                    head + why + tail,
                    alone and (far or below),
                )
            )
    _repeated_sentinels(pending)
    for item in pending:
        context.add(
            "B-09",
            item.mode,
            item.column,
            f"Read {item.text} as missing in {item.column.name}",
            item.reason,
            m.NullValues(id=_id(), column=item.column.name, values=[item.text], match="exact"),
            cells=item.count,
            examples=[(item.text, None)],
            key=f"{item.text}:repeated" if item.repeated else item.text,
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


# --- A 0 that means 'never' (N10, B-22) -------------------------------------------------------

YEARS = (1000, 2200)  # the year role's range (roles.toml): a year column's real values
RARE_ZERO = 0.01  # B-22: a count or a built area at 0 on at most this share of the rows
ROOM_WORDS = {"room", "rooms", "bedroom", "bedrooms", "bathroom", "bathrooms", "bath", "baths"}
ROOM_WORDS |= {"bed", "beds", "piece", "pieces", "chambre", "chambres"}
COUNT_WORDS = {"number", "of", "nb", "nbr", "num", "no", "nombre", "de", "total", "count"}
BUILT_WORDS = {"living", "liv", "building", "built", "bati", "habitable", "habitation", "floor"}
# N10: plausible ranges of physical measures named in a column, for B-10. Kept here, not in the
# knowledge base, so the typing does not read new roles; only measures without a unit question
# (an air temperature may be in °F: not listed).
PHYSICAL: tuple[tuple[frozenset[str], frozenset[str], str, float, float, str], ...] = (
    (frozenset({"water"}), frozenset({"temperature", "temp"}), "water temperature", -2, 40, " °C"),
    (
        frozenset(),
        frozenset({"humidity", "humidite", "hygrometrie"}),
        "relative humidity",
        0,
        100,
        " %",
    ),
    (
        frozenset({"battery", "batterie"}),
        frozenset({"level", "percent", "pct", "charge"}),
        "battery level",
        0,
        100,
        " %",
    ),
)
NOT_RELATIVE = {"absolute", "specific", "absolue"}


def _year_column(column: Column) -> bool:
    view_role = column.view.role if column.view else None
    return view_role == "year" or (column.role is not None and column.role.id == "year")


def _count_words(words: tuple[str, ...]) -> bool:
    """'bedrooms', 'Bedroom2', 'number of rooms': a count of rooms and nothing else."""
    stems = [word.rstrip("0123456789") for word in words]
    return any(s in ROOM_WORDS for s in stems) and all(
        s in ROOM_WORDS or s in COUNT_WORDS for s in stems
    )


def _built_area(column: Column) -> bool:
    """'BuildingArea', 'sqft_living': the floor area of a building (a land area can be 0)."""
    role = column.role
    words = set(semantics.tokens(column.name))
    return role is not None and role.id == "surface" and bool(words & BUILT_WORDS)


def b22_zero_never(context: Context, sentinels: dict[str, list[float]]) -> None:
    """N10: 0 among years (kc_house `yr_renovated`, 20,699 rows: never renovated) is read as
    empty if ticked; a count of rooms or a built area at 0 on a few rows among positives is
    shown, not changed. The zeros read as 'never' are left out of B-10 (`sentinels`)."""
    years = [c for c in _numbers(context) if _year_column(c)]
    counts = [
        c
        for c in _numbers(context)
        if c not in years and (_count_words(semantics.tokens(c.name)) or _built_area(c))
    ]
    columns = years + counts
    if not columns:
        return
    parts = []
    for column in columns:
        name = ident(column.name)
        parts += [
            f"count(*) FILTER (WHERE {name} = 0)",
            f"min({name}) FILTER (WHERE {name} <> 0)",
            f"max({name}) FILTER (WHERE {name} <> 0)",
        ]
    row = context.one(f"SELECT {', '.join(parts)} FROM {context.source}")
    for index, column in enumerate(columns):
        zeros, low, high = row[3 * index : 3 * index + 3]
        zeros = int(zeros)
        if not zeros or low is None or high is None:
            continue
        share = zeros / column.nonnull
        if column in years:
            if not (YEARS[0] <= float(low) and float(high) <= YEARS[1]):
                continue
            sentinels.setdefault(column.name, []).append(0.0)
            context.add(
                "B-22",
                "S",
                column,
                f"Read 0 as 'never' in {column.name}",
                f"0 is on {rows_of(zeros, 'row')} ({share:.1%}) of {column.name}, whose other "
                f"values are all years ({_number(float(low))} to {_number(float(high))}): it "
                "stands for 'never' or 'unknown' (a house never renovated), not the year 0. Read "
                "as empty, the averages and ranges of years make sense; the rows keep their "
                "other values. Check your data.",
                m.NullValues(id=_id(), column=column.name, values=["0"], match="exact"),
                cells=zeros,
                examples=[("0", None)],
            )
        elif share <= RARE_ZERO and float(low) > 0:
            context.add(
                "B-22",
                "I",
                column,
                f"{rows_of(zeros, 'row')} with 0 {column.name}",
                say(
                    "B-22",
                    "{column} is 0 on {rows} ({share}), while every other row has {low} or more: "
                    "a real 0, or a value not given and written 0? Not changed: look at these "
                    "rows. Check your data.",
                    column=column.name,
                    rows=rows_of(zeros, "row"),
                    share=f"{share:.2%}",
                    low=_number(float(low)),
                ),
                cells=zeros,
                examples=[("0", None)],
                key="rare",
            )


def _physical(column: Column) -> tuple[str, float, float, str] | None:
    """N10: the plausible range a column's name gives (`PHYSICAL`), when its role has none."""
    words = set(semantics.tokens(column.name))
    if column.name.strip().lower() == "rh":
        words.add("humidity")  # relative humidity, as sensors write it
    if words & NOT_RELATIVE:
        return None
    for needed, any_of, label, low, high, unit in PHYSICAL:
        if needed <= words and words & any_of:
            return label, float(low), float(high), unit
    return None


def b10_ranges(context: Context, sentinels: dict[str, list[float]]) -> None:
    for column in _numbers(context):
        name = ident(column.name)
        skip = sentinels.get(column.name, [])
        kept = f" AND {name} NOT IN ({', '.join(repr(v) for v in skip)})" if skip else ""
        role = column.role
        bounds: tuple[str, float | None, float | None, str] | None = None
        if role is not None and (role.min is not None or role.max is not None):
            bounds = (role.label.lower(), role.min, role.max, "")
        elif (physical := _physical(column)) is not None:
            bounds = physical
        if bounds is not None:
            label, low, high, unit = bounds
            tests = []
            if low is not None:
                tests.append(f"{name} < {low!r}")
            if high is not None:
                tests.append(f"{name} > {high!r}")
            outside, sample = context.one(
                f"SELECT count(*), list({as_text(column.name, column.dtype)})[1:3] "
                f"FROM {context.source} WHERE ({' OR '.join(tests)}){kept}"
            )
            if outside and outside / column.nonnull < NEGATIVE_SHARE:
                span = f"{low:g} to {high:g}{unit}" if low is not None and high is not None else ""
                context.add(
                    "B-10",
                    "S",
                    column,
                    f"Empty the impossible values of {column.name}",
                    say(
                        "B-10",
                        "{count} values are outside the plausible range of {label} ({span}): "
                        "{values}.",
                        count=n(int(outside)),
                        label=label,
                        span=span,
                        values=", ".join(q(str(v)) for v in sample),
                    ),
                    m.NullOutside(id=_id(), column=column.name, min=low, max=high),
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
            if not sales:
                reason = say(
                    "B-10.negative",
                    "{count} negative values ({share}): errors, or real values (net emissions, "
                    "losses, balances)? Remove them only if they are errors (CLN-7).",
                    count=n(negatives),
                    share=share,
                )
            elif role.id == "quantity":
                reason = say(
                    "B-10.refunds",
                    "Refunds or cancellations? {count} negative quantities ({share}). Remove them "
                    "only if they are errors.",
                    count=n(negatives),
                    share=share,
                )
            else:
                reason = say(
                    "B-10.refunds_amounts",
                    "Refunds or cancellations? {count} negative amounts ({share}). Remove them "
                    "only if they are errors.",
                    count=n(negatives),
                    share=share,
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


# --- Impossible years (N11) --------------------------------------------------------------------

YEAR_FLOOR = 1800  # N11: a year column "starts after 1800" when its other values do
YEAR_GAP = 200  # … and a year this far before all of them is a typing error, not an old building
YEAR_ISOLATED = 0.001  # … on at most this share of the rows (1196 in melbourne's YearBuilt)
YEAR_VALUES = 20  # … and this many distinct values at most


def _year_hints(value: int, low: float, high: float) -> list[int]:
    """The years a mistyped one may have been: two digits swapped (1196 → 1916) or the doubled
    digit typed on its neighbour (1196 → 1996), kept if they fall among the column's other years."""
    digits = str(value)
    found: set[int] = set()
    for i in range(len(digits) - 1):
        found.add(int(digits[:i] + digits[i + 1] + digits[i] + digits[i + 2 :]))
        if digits[i] != digits[i + 1]:
            continue
        if i + 2 < len(digits):  # aab → abb
            found.add(int(digits[:i] + digits[i] + digits[i + 2] * 2 + digits[i + 3 :]))
        if i >= 1:  # baa → bba
            found.add(int(digits[: i - 1] + digits[i - 1] * 2 + digits[i] + digits[i + 2 :]))
    return sorted(y for y in found if y != value and low <= y <= high)


def b10_years(context: Context, sentinels: dict[str, list[float]]) -> None:
    """N11: a year column whose values start after 1800 (melbourne YearBuilt, 1830 to 2018) with a
    few years far before them (1196) or in the future: typing errors, emptied if ticked (S), the
    likely right year said in the reason, never written for you. B-10 already empties what is
    outside 1000 to 2200 (the year role's range)."""
    future = datetime.date.today().year + 1
    columns = [c for c in _numbers(context) if _year_column(c)]
    if not columns:
        return
    parts = []
    for column in columns:
        name = ident(column.name)
        skip = sentinels.get(column.name, [])
        known = f" AND {name} NOT IN ({', '.join(repr(v) for v in skip)})" if skip else ""
        early = f"{name} BETWEEN {YEARS[0]} AND {YEAR_FLOOR - 1}"
        odd = f"({early} OR {name} BETWEEN {future + 1} AND {YEARS[1]}){known}"
        usual = f"{name} BETWEEN {YEAR_FLOOR} AND {future}"
        parts += [
            f"count(*) FILTER (WHERE {odd})",
            f"min({name}) FILTER (WHERE {usual})",
            f"max({name}) FILTER (WHERE {usual})",
            f"list(DISTINCT {name}) FILTER (WHERE {odd})",
        ]
    row = context.one(f"SELECT {', '.join(parts)} FROM {context.source}")
    for index, column in enumerate(columns):
        count, low, high, odd = row[4 * index : 4 * index + 4]
        count = int(count)
        if not count or low is None or high is None or count > YEAR_ISOLATED * column.nonnull:
            continue
        low, high = float(low), float(high)
        values = sorted(float(v) for v in odd or [] if v == int(v))
        early = [v for v in values if v < YEAR_FLOOR]
        if len(values) != len(odd or []) or len(values) > YEAR_VALUES:
            continue  # not whole years, or too many to be a few typing errors
        if any(v > low - YEAR_GAP for v in early):
            continue  # an old year close to the others may be real
        shown = [_number(v) for v in values]
        hints = [
            f"{text} → {' or '.join(map(str, _year_hints(int(v), low, high)))}"
            for v, text in zip(values, shown, strict=True)
            if _year_hints(int(v), low, high)
        ]
        span = f"every other {column.name} ({_number(low)} to {_number(high)})"
        late = len(values) - len(early)
        if len(values) == 1 and early:
            found = f"{shown[0]} is {_number(low - early[0])} years before {span}"
        elif len(values) == 1:
            found = f"{shown[0]} is after {future}, in the future, unlike {span}"
        else:
            parts_said = []
            if early:
                gap = _number(low - max(early))
                parts_said.append(f"{len(early)} at least {gap} years before")
            if late:
                parts_said.append(f"{late} after {future}")
            found = (
                f"{rows_of(count, 'value')} of {column.name} ({', '.join(shown)}) are far from "
                f"{span}: {', '.join(parts_said)}"
            )
        reason = (
            f"{found}: a typing error? Emptied if you tick it"
            + (
                f"; two digits may be swapped or a doubled digit misplaced ({'; '.join(hints)}), "
                "never written for you."
                if hints
                else "."
            )
            + " Check your data."
        )
        context.add(
            "B-10",
            "S",
            column,
            f"Empty {rows_of(count, 'impossible year')} in {column.name}",
            reason,
            m.NullValues(id=_id(), column=column.name, values=shown, match="exact"),
            cells=count,
            examples=[(text, None) for text in shown],
            key="year",
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
            if day.startswith("9999"):
                said = say(
                    "B-16.end",
                    "{day} is used {count} times ({share}): a placeholder meaning 'no end date', "
                    "not a real date.",
                    day=day,
                    count=n(int(count)),
                    share=f"{share:.1%}",
                )
            else:
                said = say(
                    "B-16",
                    "{day} is used {count} times ({share}): a placeholder meaning 'unknown date', "
                    "not a real date.",
                    day=day,
                    count=n(int(count)),
                    share=f"{share:.1%}",
                )
            context.add(
                "B-16",
                mode,  # type: ignore[arg-type]
                column,
                f"Read {day} as missing in {column.name}",
                said,
                m.NullValues(id=_id(), column=column.name, values=sorted(texts), match="exact"),
                cells=int(count),
                examples=[(texts[0], None)] if texts else [],
                key=day,
            )


# --- Rows and columns --------------------------------------------------------------------------


def _blank(column: Column) -> str:
    return f"coalesce(trim({as_text(column.name, column.dtype)}), '') = ''"


def _typed(column: Column) -> bool:
    return is_number(column.dtype) or is_temporal(column.dtype) or column.dtype == "BOOLEAN"


def _empty(column: Column) -> str:
    """`_blank`, cheaper for the counts (N43): a number, a date or a boolean is never blank
    once written as text, only empty."""
    return f"{ident(column.name)} IS NULL" if _typed(column) else _blank(column)


NAMED_WORDS = {"nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
NAMED_WORDS |= {"true", "false"}


def _never_named(column: Column) -> bool:
    """N43: a number, date or boolean column whose name no value of it is written as: a number's,
    a date's or a boolean's text holds a digit, or is one of `NAMED_WORDS`."""
    if not _typed(column):
        return False
    name = column.name.strip().lower()
    return not DIGIT.search(name) and name not in NAMED_WORDS


def b15_empty_rows(context: Context) -> None:
    if len(context.columns) < 2:
        return
    empty = " AND ".join(_empty(c) for c in context.columns)
    header = " AND ".join(
        f"coalesce(lower(trim({as_text(c.name, c.dtype)})) = {lit(c.name.strip().lower())}, false)"
        for c in context.columns
    )
    if any(_never_named(c) for c in context.columns):  # N43: no row can repeat the names
        blanks, headers = (
            int(context.one(f"SELECT count(*) FROM {context.source} WHERE {empty}")[0]),
            0,
        )
    else:
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
            f"Remove {rows_of(blanks, 'empty row')}",
            say_count(
                "B-15",
                blanks,
                "{count} row has no value at all.",
                "{count} rows have no value at all.",
            ),
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
            f"Remove {rows_of(headers, 'repeated header row')}",
            say(
                "B-15.headers",
                "{count} rows repeat the column names: files pasted one after the other.",
                count=n(headers),
            ),
            m.DropHeaderRows(id=_id()),
            rows=headers,
        )
        context.found[-1].id = "B-15:header"
        context.found[-1].step.suggestion = "B-15:header"  # type: ignore[union-attr]


def b14_duplicates(context: Context) -> None:
    if not context.columns or context.rows < 2:
        return
    empty = " AND ".join(_empty(c) for c in context.columns)
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
    reason = say(
        "B-14",
        "{count} rows are exact copies of another row ({share}).",
        count=n(copies), share=f"{copies / max(rows, 1):.1%}",
    )  # fmt: skip
    reason += (
        say("B-14.identified", " Even their identifier repeats: a row entered twice.")
        if identified
        else say(
            "B-14.repeats",
            " Without an identifier they may be real repeats (the same purchase twice).",
        )
    )
    context.add(
        "B-14",
        "A" if identified else "S",
        None,
        f"Remove {rows_of(copies, 'identical row')}",
        reason,
        m.DropDuplicates(id=_id(), keys=None, keep="first"),
        rows=copies,
    )


def b19_columns(context: Context) -> None:
    boxes = {  # N27: a checkbox ('X' and blanks) is not a constant column
        s.columns[0] for s in context.found if s.rule == "B-18" and s.id.endswith(":checkbox")
    }
    for column in context.columns:
        if context.rows < 2:
            return
        if column.name in boxes:
            continue
        if column.nonnull == 0:
            title = f"Remove the empty column {column.name}"
            reason = say("B-19.empty", "It has no value at all.")
        elif column.low is not None and column.low == column.high:
            title = f"Remove the constant column {column.name}"
            reason = say("B-19", "Always {value}: no information.", value=q(column.low))
        else:
            continue
        context.add(
            "B-19",
            "S",
            column,
            title,
            reason + say("B-19.kept", " Kept unless you remove it: a filter or a KPI may name it."),
            m.DropColumns(id=_id(), columns=[column.name]),
        )


def _already_dated(context: Context, year: str, month: str) -> str | None:
    """A date column whose year and month are these parts on 95 % of the rows (a `Date` beside
    the `Year` and `Month` derived from it): nothing to assemble."""
    for column in context.columns:
        view = column.view
        if view is None or view.type != "datetime" or view.role == "year" or column.nonnull == 0:
            continue
        moment = f"CAST({typed_sql(view.name, view.parser)} AS TIMESTAMP)"
        (same,) = context.one(
            f"SELECT count(*) FILTER (WHERE year({moment}) = TRY_CAST({ident(year)} AS DOUBLE) "
            f"AND month({moment}) = TRY_CAST({ident(month)} AS DOUBLE)) FROM {context.source}"
        )
        if same >= DATE_COVERAGE * column.nonnull:
            return column.name
    return None


def b20_assemble_dates(context: Context) -> None:
    """fix1 (TYP-6): a year, a month and a day (an hour, a minute) in separate columns whose names
    say the same thing ("an, mois, jour", "Yr Sold, Mo Sold") make one date: proposed as a new
    column (mode S) when 95 % of the rows with every part make a valid date."""
    by_name = {c.name: c for c in context.columns}
    names = set(by_name)
    stage = Stage("source", [(c.name, c.dtype) for c in context.columns])
    for group in date_parts.groups(c.name for c in context.columns):
        if _already_dated(context, group.parts["year"], group.parts["month"]):
            continue
        target = group.name
        while target in names:
            target = f"{target} (assembled)"
        parts = group.parts
        step = m.AssembleDate(
            id=_id(),
            year=parts["year"],
            month=parts["month"],
            day=parts.get("day"),
            hour=parts.get("hour"),
            minute=parts.get("minute"),
            time=parts.get("time"),
            to=target,
        )
        given = " AND ".join(f"{ident(c)} IS NOT NULL" for c in group.columns)
        tries = [step]
        hour = parts.get("hour")
        if hour and not parts.get("minute") and by_name[hour].dtype in ("VARCHAR", "TIME"):
            # C1: an "heure" column written 07:40 is a time of day, not a whole hour
            tries.append(step.model_copy(update={"hour": None, "time": hour}))
        for candidate in tries:
            made_sql = assembled_date_sql(candidate, stage)
            filled, made = context.one(
                f"SELECT count(*) FILTER (WHERE {given}), count({made_sql}) FROM {context.source}"
            )
            if filled and made >= DATE_COVERAGE * filled:
                step = candidate
                break
        else:
            continue
        shown = ", ".join(ident(c) for c in group.columns)
        examples: list[tuple[str | None, str | None]] = [
            (" · ".join("" if v is None else str(v) for v in row[:-1]), str(row[-1]))
            for row in context.connection.execute(
                f"SELECT {shown}, CAST({made_sql} AS VARCHAR) FROM {context.source} "
                f"WHERE {made_sql} IS NOT NULL LIMIT {MAX_EXAMPLES}"
            ).fetchall()
        ]
        first = "" if "day" in group.parts else " (the 1st of the month: there is no day)"
        context.add(
            "B-20",
            "S",
            by_name[group.parts["year"]],
            f"Assemble a date from {', '.join(group.columns)}",
            f"{group.listed} hold the parts of one date: {n(made)} of {n(filled)} rows with "
            f"every part ({made / filled:.1%}) make a valid date. A new column '{target}' holds "
            f"it{first}; the part columns stay as they are, and a date that cannot exist "
            "(31 February, a missing part) stays empty.",
            step,
            cells=made,
            examples=examples,
        )
        context.found[-1].columns = group.columns
        names.add(target)


def b21_aggregate_rows(context: Context) -> None:
    """fix1 (KPI-2): rows whose label sums up other rows (OWID's World, Europe, High-income
    countries; a Total row), found by the written signals of `profiling/aggregates.py`: proposed
    to leave out (mode S), never removed on their own."""
    views = [c.view for c in context.columns if c.view is not None]
    labels = {c.name: c.values for c in context.columns if c.values is not None}
    by_name = {c.name: c for c in context.columns}
    for found in aggregates.find(context.connection, context.source, views, labels):
        if len(found.found) > m.MAX_VALUES:  # one condition holds this many values at most
            continue
        column = by_name[found.column]
        first = found.found[0]
        context.add(
            "B-21",
            "S",
            column,
            f"Leave out {n(found.rows)} row{'s' if found.rows != 1 else ''} that sum up other "
            f"{column.name} rows",
            f"{len(found.found)} of the {n(found.labels)} labels of {column.name} sum up other "
            f"rows rather than being one ({found.shown()}; '{first.label}': {first.why}). In a "
            f"total over {column.name} they count the same rows twice. Removed only if you tick "
            "it; the original dataset keeps them, and a filter can leave them out instead.",
            m.FilterRows(
                id=_id(),
                conditions=[m.RowCondition(column=column.name, op="in", values=found.names)],
                combine="all",
                action="remove",
            ),
            rows=found.rows,
            examples=[(item.label, item.why) for item in found.found],
        )


# N47 (B-24): names given by the reader when a file has none ('column_2', 'Unnamed: 3').
GENERIC_NAME = re.compile(r"^(?:column[ _]?\d+|unnamed:?[ _]?\d+(?:_level_\d+)?)$", re.I)
HEADER_ROWS = 10  # rows read to find the real names
HEADER_SHARE = 0.8  # share of the columns whose cell on that row looks like a name
NAME_LENGTH = 60


def _name_like(value: Any) -> str | None:
    """A cell that can be a column name: short text, not a number, not mostly digits."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > NAME_LENGTH or NUMBER_TEXT.match(text):
        return None
    digits = sum(ch.isdigit() for ch in text)
    return text if digits * 2 < len(text) else None


def _first_rows(context: Context, count: int) -> list[tuple[Any, ...]]:
    """The first rows in the file's order, every column as text."""
    source = context.source
    if source.startswith("read_parquet(") and source.endswith(")"):
        source = source[:-1] + ", file_row_number = true)"
        order = "ORDER BY file_row_number "
    else:
        order = ""
    texts = ", ".join(as_text(c.name, c.dtype) for c in context.columns)
    return context.connection.execute(
        f"SELECT {texts} FROM {source} {order}LIMIT {int(count)}"
    ).fetchall()


def b24_shifted_header(context: Context) -> None:
    """N47 (B-24): columns named 'column_N' or 'Unnamed: N' by the reader, and among the first ten
    rows one whose cells are, for 80 % of the columns, short distinct texts that are not numbers:
    the real column names, under a note (kiwi). Suggested: that row becomes the header and the
    rows above it leave the table."""
    columns = context.columns
    generic = [c.name for c in columns if GENERIC_NAME.match(c.name.strip())]
    if len(columns) < 2 or len(generic) < max(2, len(columns) / 2) or context.rows < 2:
        return
    for index, row in enumerate(_first_rows(context, HEADER_ROWS)):
        names = [_name_like(v) for v in row]
        kept = [v for v in names if v is not None]
        distinct = {v.lower() for v in kept}
        if len(distinct) < len(kept) or len(kept) < HEADER_SHARE * len(columns):
            continue
        if all(
            (v or "").lower() == c.name.strip().lower() for v, c in zip(names, columns, strict=True)
        ):
            return  # the names repeated: B-15's repeated header
        shown = ", ".join(kept[:3]) + ("…" if len(kept) > 3 else "")
        above = (
            f"the {'row' if index == 1 else f'{index} rows'} above "
            f"{'is a note' if index == 1 else 'are notes'}, removed with it"
            if index
            else f"the current names are the reader's ({', '.join(generic[:2])}…)"
        )
        context.add(
            "B-24",
            "S",
            None,
            f"Use row {index + 1} as the column names",
            f"Row {index + 1} looks like the real column names ({shown}): {above}. Tick it alone "
            "and save the version: the other fixes are found again on the real names (N47). "
            "Check your data.",
            m.PromoteHeader(id=_id(), row=index + 1),
            rows=index + 1,
            examples=[(c.name, v) for c, v in zip(columns, names, strict=True) if v is not None],
        )
        context.found[-1].id = "B-24:header"
        context.found[-1].step.suggestion = "B-24:header"  # type: ignore[union-attr]
        context.found[-1].columns = [c.name for c in columns]
        return


def run(context: Context) -> list[CleanSuggestion]:
    try:
        _run(context)
    finally:
        context.close()
    return sorted(context.found, key=lambda s: s.order)


def _run(context: Context) -> None:
    b24_shifted_header(context)
    b15_empty_rows(context)
    b02_invisible(context)
    b01_spaces(context)
    b05_encoding(context)
    from quantilo.cleaning.rules import encoding  # N28: B-05's readings and the lost characters

    encoding.b05_more(context)
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
    b22_zero_never(context, sentinels)
    b16_sentinel_dates(context)
    b10_ranges(context, sentinels)
    b10_years(context, sentinels)
    b14_duplicates(context)
    from quantilo.cleaning.rules import keys  # it imports this module

    keys.b23_repeated_keys(context)
    b19_columns(context)
    b20_assemble_dates(context)
    from quantilo.cleaning.rules import date_time  # N19 (B-20): it reads this module

    date_time.b20_date_and_time(context)
    b21_aggregate_rows(context)
