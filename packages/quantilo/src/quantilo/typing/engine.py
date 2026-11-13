"""Detect the analysis type of every column of a dataset.

Profiles a fixed random sample in DuckDB, then applies ordered rules (each one explained)
combined with the knowledge base. Candidate parsers are evaluated with the very SQL used
later to read the data, so detection and conversion always agree.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import duckdb

from quantilo.ingestion.normalize import stored_notes
from quantilo.ingestion.sql import ident, lit
from quantilo.reasons import Reason, say
from quantilo.typing import date_parts
from quantilo.typing.knowledge_base import (
    MEASURE_ROLES,
    KnowledgeBase,
    Role,
    load,
    name_note,
    name_tokens,
)
from quantilo.typing.knowledge_base.synonyms import load_synonyms
from quantilo.typing.parsers import (
    ALL_DATE_FORMATS,
    CURRENCY_PATTERN,
    NUMBER_PATTERNS,
    TIME_FORMATS,
    VALUE_PATTERNS,
    clean_text,
    currency_text_sql,
    month_words_sql,
    needs_letters,
    number_sql,
    strptime_sql,
)
from quantilo.typing.text import TextMeasures, language_reason, measure
from quantilo.typing.types import (
    NUMERIC_TYPES,
    AnalysisType,
    ColumnDetection,
    Confidence,
    Parser,
)

SAMPLE_ROWS = 10_000
SAMPLE_TABLE = "detection_sample"
VALUES_TABLE = "detection_values"  # the distinct texts of the column being detected, with counts
TEXT_MEASURE_LIMIT = 2_000  # distinct texts measured for the text verdicts (R-ING-06)


@dataclass(frozen=True)
class Thresholds:
    """Detection thresholds, adjustable in Settings (defaults below)."""

    parse_rate: float = 0.95  # share of values read as numbers / dates to convert the column
    empty_rate: float = 0.95  # share of empty values to ignore the column
    discrete_max: int = 20  # at most this many distinct whole numbers: discrete
    identifier_unique: float = 0.95  # share of distinct values to call text an identifier


DEFAULT_THRESHOLDS = Thresholds()
TOP_VALUES = 30

# Text verdicts (050: docs/features/050-texte-detection-fiche/spec.md, R-TXT-1 to R-TXT-11).
CATEGORY_MAX_DISTINCT = 50  # at most this many distinct texts: categories
CATEGORY_TOP_COVERAGE = 0.8  # the 20 most frequent texts cover this share: categories
CATEGORY_MAX_UNIQUE = 0.2  # fewer unique texts than this share: categories
SHORT_WORDS = (2, 8)  # median words of a short text (a label)
STOPWORDS_SENTENCE = 0.15  # stop words of a language among the words: sentences
PUNCTUATION_SENTENCE = 0.3  # texts with sentence punctuation: sentences
FREE_MEDIAN_WORDS = 5
FREE_MEAN_LENGTH = 50
ADDRESS_SHARE = 0.5  # texts with a number and a street word: addresses
TEXT_NAME_ROLES = ("free_text", "label", "address")

TEXT_PARSER = Parser(kind="text")
RAW_PARSER = Parser(kind="raw")

# 092 (docs/features/092-corrections-multidomaine-import-types/spec.md)
LOST_LIMIT = 0.005  # R-TYP-01: more unreadable values than this share keep the column as text
MARKER_MAX_LENGTH = 3  # R-TYP-02: "s", "x", "*", ".." read as missing in a numeric column
MARKER_MIN_NUMBERS = 0.5
LEADING_ZERO_SHARE = 0.01  # R-TYP-03
UNIQUE_CODE_SHARE = 0.99  # R-TYP-05: all distinct whole numbers...
UNIQUE_CODE_DIGITS = 6  # ...of at least this many digits...
CODE_MIN_VALUES = 100
FIXED_LENGTH_SHARE = 0.95  # ...of one length; or repeated numbers of one length of at least
FIXED_CODE_DIGITS = 7  # this many digits
LIST_SHARE = 0.3  # R-TYP-09: values holding ";"...
LIST_MAX_CHOICES = 100  # ...made of at most this many choices
RANGE_MAX_DISTINCT = 30  # R-TYP-10
RANGE_NUMBERED_SHARE = 0.8
PLACE_ROLES = ("region", "city", "country")  # R-TYP-08 (b)
PARQUET_SOURCE = re.compile(r"read_parquet\('((?:[^']|'')*)'\)")
DASHES = "|".join(("-", chr(0x2013), chr(0x2014), "to", "à"))  # en and em dashes
_NUMBER = r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?"
RANGE_VALUE = re.compile(
    rf"^\W{{0,2}}[$€£]?\s*({_NUMBER})\s*(?:{DASHES})\s*[$€£]?\s*({_NUMBER})\W{{0,2}}"
    r"(?:\s+\D.*)?$",
    re.IGNORECASE,
)
FIRST_NUMBER = re.compile(_NUMBER)
LOWER_BOUND = re.compile(r"\b(under|less than|below|fewer than|up to|moins de)\b|<|≤", re.I)
UPPER_BOUND = re.compile(r"\b(over|more than|older|above|or more|plus de|et plus)\b|>|≥|\+", re.I)


UNIX_SECONDS = (946_684_800, 2_147_483_647)  # 2000-01-01 to 2038-01-19


def unix_time(low: float | None, high: float | None, digits: int | None) -> bool:
    """Whole numbers that read as Unix times (seconds or milliseconds since 1970, 2000-2038):
    a date, never a code by their shape (R-TYP-05)."""
    if low is None or high is None or digits not in (10, 13):
        return False
    scale = 1 if digits == 10 else 1000
    return UNIX_SECONDS[0] * scale <= low and high <= UNIX_SECONDS[1] * scale


def rule(code: str, template: str, **values: Any) -> Reason:
    """A reason given by a 092 rule: the sentence and its code, which the interface lists
    (`R-TYP-05.length` is said "(R-TYP-05)"); `say` keeps the code and vars (M02)."""
    return say(code, f"{template} ({code.split('.')[0]})", **values)


def name_reason(role: Role) -> Reason:
    """The reason a role read from the column's name gives."""
    return say("TYP.name", "name suggests {role}", role=role.label.lower())


@dataclass
class Profile:
    rows: int
    present: int
    distinct: int
    top: list[tuple[str, int]]
    minimum: float | None = None
    maximum: float | None = None
    integer: bool = False
    avg_length: float = 0
    spaced: int = 0
    number_matches: dict[str, int] = field(default_factory=dict)
    percent: int = 0
    currency: int = 0
    date_matches: dict[str, int] = field(default_factory=dict)
    pattern_matches: dict[str, int] = field(default_factory=dict)
    # 092: digit count shared by most whole numbers and its share (R-TYP-05), values starting
    # with 0 (R-TYP-03), holding ";" (R-TYP-09), read as times of day (R-TYP-07)
    length_mode: int | None = None
    length_share: float = 0
    # N18: what decided the decimal separator when some values read both ways ('1.234'), and
    # whether that was an inference (confidence medium)
    decimal_reason: Reason | None = None
    decimal_doubt: bool = False
    grouped: int = 0  # values with a dot or a comma before three digits ('1.234'): R-TYP-15
    leading_zero: int = 0
    lists: int = 0
    time_matches: dict[str, int] = field(default_factory=dict)

    @property
    def unique_ratio(self) -> float:
        return self.distinct / self.present if self.present else 0

    def rate(self, count: int) -> float:
        return count / self.present if self.present else 0


def storage_kind(dtype: str) -> str:
    upper = dtype.upper()
    if upper == "BOOLEAN":
        return "boolean"
    if upper.startswith(("DATE", "TIMESTAMP")):
        return "temporal"
    if upper.startswith("TIME"):
        return "time"
    if upper.startswith(("VARCHAR", "CHAR", "TEXT", "STRING")):
        return "text"
    numeric = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
               "UINTEGER", "UBIGINT", "DOUBLE", "FLOAT", "REAL", "DECIMAL")  # fmt: skip
    if upper.startswith(numeric):
        return "numeric"
    return "other"


def percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def examples(profile: Profile, count: int = 3) -> str:
    return ", ".join(value for value, _ in profile.top[:count])


def unique_values(ratio: float) -> Reason:
    return say("TYP.unique", "{share} unique values", share=percent(ratio))


ROW_INDEX = re.compile(r"column0+|unnamed[ _:]*0|index|level[ _]0")


def is_row_index(name: str, profile: Profile) -> bool:
    """An unnamed first column of distinct whole numbers: `df.to_csv()` without `index=False`."""
    return bool(
        ROW_INDEX.fullmatch(name.strip().lower()) and profile.integer and profile.unique_ratio == 1
    )


class ColumnDetector:
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        kb: KnowledgeBase,
        rows: int,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
    ) -> None:
        self.connection = connection
        self.kb = kb
        self.rows = rows
        self.thresholds = thresholds

    def one(self, sql: str) -> tuple:
        return self.connection.execute(sql).fetchone() or ()

    def top_values(self, expression: str) -> list[tuple[str, int]]:
        rows = self.connection.execute(
            f"SELECT CAST(v AS VARCHAR), count(*) AS c FROM (SELECT {expression} AS v FROM "
            f"{SAMPLE_TABLE}) WHERE v IS NOT NULL GROUP BY v ORDER BY c DESC, v LIMIT {TOP_VALUES}"
        ).fetchall()
        return [(str(value), int(count)) for value, count in rows]

    # --- Profiles ------------------------------------------------------------------------------

    def numeric_profile(self, column: str) -> Profile:
        col = ident(column)
        present, distinct, low, high, integer = self.one(
            f"SELECT count({col}), count(DISTINCT {col}), min({col}), max({col}), "
            f"coalesce(bool_and({col} = floor({col})), false) FROM {SAMPLE_TABLE}"
        )
        profile = Profile(
            rows=self.rows,
            present=present,
            distinct=distinct,
            top=self.top_values(col),
            minimum=float(low) if low is not None else None,
            maximum=float(high) if high is not None else None,
            integer=bool(integer),
        )
        if profile.integer and present and (low or 0) >= 0:
            digits = f"length(CAST(TRY_CAST({col} AS HUGEINT) AS VARCHAR))"
            row = self.one(
                f"SELECT {digits} AS l, count(*) AS c FROM {SAMPLE_TABLE} WHERE {col} IS NOT NULL "
                "GROUP BY l ORDER BY c DESC, l LIMIT 1"
            )
            if row and row[0] is not None:
                profile.length_mode, profile.length_share = int(row[0]), row[1] / present
        return profile

    def text_profile(self, column: str) -> Profile:
        """R-ING-06: measured once on the distinct texts, each weighted by its count."""
        self.connection.execute(
            f"CREATE OR REPLACE TEMP TABLE {VALUES_TABLE} AS SELECT v, count(*) AS c FROM "
            f"(SELECT {clean_text(column)} AS v FROM {SAMPLE_TABLE}) WHERE v IS NOT NULL GROUP BY v"
        )
        value = "v"

        def weighted(condition: str) -> str:
            return f"coalesce(sum(c) FILTER (WHERE {condition}), 0)"

        aggregates = [
            "coalesce(sum(c), 0)",
            "count(*)",
            f"coalesce(sum(length({value}) * c) / sum(c), 0)",
            weighted(f"contains({value}, ' ')"),
            weighted(f"contains({value}, '%')"),
            weighted(f"regexp_matches({value}, {lit(CURRENCY_PATTERN)})"),
            weighted(f"regexp_full_match({value}, '0[0-9]+')"),
            weighted(f"contains({value}, ';')"),
            weighted(f"regexp_matches({value}, '[.,][0-9]{{3}}')"),
            weighted(f"regexp_matches({value}, '[A-Za-z]')"),
        ]
        aggregates += [
            weighted(f"regexp_full_match({currency_text_sql(value)}, {lit(pattern)})")
            for pattern in NUMBER_PATTERNS.values()
        ]
        aggregates += [
            weighted(f"regexp_full_match({value}, {lit(pattern)})")
            for pattern in VALUE_PATTERNS.values()
        ]
        row = self.one(f"SELECT {', '.join(aggregates)} FROM {VALUES_TABLE}")
        present, distinct, avg_length, spaced, percent_count, currency, zeros, lists = row[:8]
        grouped, lettered = int(row[8]), int(row[9])
        offset = 10
        number_matches = dict(zip(NUMBER_PATTERNS, row[offset : offset + 2], strict=True))
        offset += 2
        pattern_matches = dict(zip(VALUE_PATTERNS, row[offset:], strict=True))
        # Only short texts with a digit can be dates or times: the others skip the 39 formats
        # (a filtered table, since DuckDB evaluates every FILTER condition on every row). Month
        # names are rewritten once (`m`, N19) for the formats that read them; a column without a
        # letter skips the formats that need one (month names, AM/PM, ISO's T).
        formats = (*ALL_DATE_FORMATS, *TIME_FORMATS)
        parsed = [
            weighted(f"{strptime_sql('v', fmt, 'm')} IS NOT NULL")
            if lettered or not needs_letters(fmt)
            else "0"
            for fmt in formats
        ]
        months = month_words_sql("v") if lettered else "v"
        dated = self.one(
            f"SELECT {', '.join(parsed)} FROM (SELECT v, c, {months} AS m FROM "
            f"{VALUES_TABLE} WHERE length(v) <= 40 AND regexp_matches(v, '[0-9]'))"
        )
        date_matches = {
            fmt: int(count) for fmt, count in zip(ALL_DATE_FORMATS, dated, strict=False)
        }
        time_matches = {
            fmt: int(count)
            for fmt, count in zip(TIME_FORMATS, dated[len(ALL_DATE_FORMATS) :], strict=True)
        }
        top = self.connection.execute(
            f"SELECT v, c FROM {VALUES_TABLE} ORDER BY c DESC, v LIMIT {TOP_VALUES}"
        ).fetchall()
        return Profile(
            rows=self.rows,
            present=int(present),
            distinct=int(distinct),
            top=[(str(text), int(count)) for text, count in top],
            avg_length=float(avg_length),
            spaced=int(spaced),
            number_matches=number_matches,
            percent=int(percent_count),
            currency=int(currency),
            date_matches=date_matches,
            pattern_matches=pattern_matches,
            leading_zero=int(zeros),
            lists=int(lists),
            grouped=grouped,
            time_matches=time_matches,
        )

    def parsed_number_profile(self, column: str, decimal: str) -> tuple[int, float, float, bool]:
        """Distinct numbers, range and whole-ness of the current text column read as numbers."""
        parsed = number_sql("v", decimal)
        distinct, low, high, integer = self.one(
            f"SELECT count(DISTINCT p), min(p), max(p), coalesce(bool_and(p = floor(p)), false) "
            f"FROM (SELECT {parsed} AS p FROM {VALUES_TABLE}) WHERE p IS NOT NULL"
        )
        return int(distinct), float(low), float(high), bool(integer)

    def decimal_choice(self, profile: Profile) -> str:
        """R-TYP-15 (N18): the decimal separator, decided on the whole column. The reading that
        reads more values wins ('12,5' reads only with a decimal comma), as before; the reason says
        so when some values read both ways ('1.234': 1.234 or 1234). When every value reads both
        ways, the separator is a thousands one if the numbers written without it are all below
        1,000 (250 beside 1.234), a decimal one if some are not (1500 beside 1.234: a writer
        grouping thousands would write 1.500); with no such number, the dot stays a decimal point
        and the comma a thousands separator, as before, and the reason says it is a guess."""
        dot, comma = profile.number_matches["."], profile.number_matches[","]
        best = "," if comma > dot else "."
        if not dot or not comma or not profile.grouped:
            return best  # only '1.234' or '1,234' can read two ways
        point, decimal_comma = number_sql("v", "."), number_sql("v", ",")
        both, dotted, commas, plain, largest, either, only_point, only_comma = self.one(
            "SELECT coalesce(sum(c) FILTER (WHERE d <> k), 0), "
            "coalesce(sum(c) FILTER (WHERE d <> k AND contains(v, '.')), 0), "
            "coalesce(sum(c) FILTER (WHERE d <> k AND contains(v, ',')), 0), "
            "coalesce(sum(c) FILTER (WHERE d IS NOT NULL AND NOT regexp_matches(v, '[.,]')), 0), "
            "max(abs(d)) FILTER (WHERE d IS NOT NULL AND NOT regexp_matches(v, '[.,]')), "
            "arg_max(v, c) FILTER (WHERE d <> k), "
            "arg_max(v, c) FILTER (WHERE d IS NOT NULL AND k IS NULL), "
            "arg_max(v, c) FILTER (WHERE k IS NOT NULL AND d IS NULL) "
            f"FROM (SELECT v, c, {point} AS d, {decimal_comma} AS k FROM {VALUES_TABLE})"
        )
        if not both:
            return best  # every value reads the same both ways: nothing to decide
        if dot != comma:
            only, shown = (dot - comma, only_point) if best == "." else (comma - dot, only_comma)
            if best == ".":
                profile.decimal_reason = rule(
                    "R-TYP-15.dot",
                    "dot read as the decimal point: {only} values such as '{value}' read only "
                    "this way ({both} such as '{either}' read both ways)",
                    only=only, value=shown, both=both, either=either,
                )  # fmt: skip
            else:
                profile.decimal_reason = rule(
                    "R-TYP-15.comma",
                    "comma read as the decimal separator: {only} values such as '{value}' read "
                    "only this way ({both} such as '{either}' read both ways)",
                    only=only, value=shown, both=both, either=either,
                )  # fmt: skip
            return best
        profile.decimal_doubt = True
        if not dotted or not commas:
            grouping = "," if dotted else "."  # the reading where this separator groups thousands
            if plain and largest is not None and largest < 1000:
                if dotted:
                    profile.decimal_reason = rule(
                        "R-TYP-15.thousands_dot",
                        "the dot is read as a thousands separator: the {both} values with one "
                        "have groups of three digits (such as '{either}') and the {plain} "
                        "numbers written without one are all below 1,000",
                        both=both, either=either, plain=plain,
                    )  # fmt: skip
                else:
                    profile.decimal_reason = rule(
                        "R-TYP-15.thousands_comma",
                        "the comma is read as a thousands separator: the {both} values with one "
                        "have groups of three digits (such as '{either}') and the {plain} "
                        "numbers written without one are all below 1,000",
                        both=both, either=either, plain=plain,
                    )  # fmt: skip
                return grouping
            if plain:
                if dotted:
                    profile.decimal_reason = rule(
                        "R-TYP-15.decimal_dot",
                        "the dot is read as the decimal separator: '{either}' could be "
                        "thousands, but numbers up to {largest} are written without one",
                        either=either, largest=f"{largest:g}",
                    )  # fmt: skip
                else:
                    profile.decimal_reason = rule(
                        "R-TYP-15.decimal_comma",
                        "the comma is read as the decimal separator: '{either}' could be "
                        "thousands, but numbers up to {largest} are written without one",
                        either=either, largest=f"{largest:g}",
                    )  # fmt: skip
                return "." if dotted else ","
        profile.decimal_reason = rule(
            "R-TYP-15.ambiguous",
            "'{either}' reads both as a decimal and as thousands, and no value tells: the dot is "
            "read as the decimal point, the comma as a thousands separator (ambiguous)",
            either=either,
        )
        return "."

    def digit_length(self) -> tuple[int | None, float]:
        """Most common length of the current column's digit-only texts, and its share."""
        row = self.one(
            f"SELECT length(v) AS l, sum(c) AS n, (SELECT sum(c) FROM {VALUES_TABLE} "
            f"WHERE regexp_full_match(v, '[0-9]+')) FROM {VALUES_TABLE} "
            "WHERE regexp_full_match(v, '[0-9]+') GROUP BY l ORDER BY n DESC, l LIMIT 1"
        )
        if not row or row[0] is None:
            return None, 0
        return int(row[0]), float(row[1]) / float(row[2])

    # --- Shared rules --------------------------------------------------------------------------

    def empty_or_constant(self, profile: Profile) -> tuple[str, str] | None:
        if profile.rows == 0 or profile.present / profile.rows < 1 - self.thresholds.empty_rate:
            empty = percent(1 - profile.present / max(profile.rows, 1))
            return "ignored", say("TYP.empty", "{share} of values are empty", share=empty)
        if profile.distinct == 1:
            value = repr(profile.top[0][0])
            return "ignored", say("TYP.constant", "a single value everywhere: {value}", value=value)
        return None

    def numeric_type(
        self, role: Role | None, integer: bool, distinct: int
    ) -> tuple[AnalysisType, Confidence, list[str]]:
        if role and role.type in NUMERIC_TYPES:
            return role.type, "high", [name_reason(role)]
        if integer and distinct <= self.thresholds.discrete_max:
            confidence: Confidence = "high" if distinct <= 12 else "medium"
            discrete = say(
                "TYP.discrete",
                "only {count} distinct whole numbers: shown as categories",
                count=distinct,
            )
            return "numeric_discrete", confidence, [discrete]
        confidence = "high" if distinct > 30 else "medium"
        continuous = say("TYP.continuous", "{count} distinct numeric values", count=distinct)
        return "numeric_continuous", confidence, [continuous]

    # --- Detection by storage ------------------------------------------------------------------

    def detect(self, name: str, dtype: str, position: int) -> ColumnDetection:
        found = self._detect(name, dtype, position)
        # N30 (R-SYN-01): a role read through a business synonym says which one ('Qté')
        if found.role is not None and found.role == getattr(self.kb.match_role(name), "id", None):
            why = load_synonyms().reason(name)
            if why and why not in found.reasons:
                found.reasons.append(why)
        return found

    def _detect(self, name: str, dtype: str, position: int) -> ColumnDetection:
        kind = storage_kind(dtype)
        role = self.kb.match_role(name)

        def make(
            analysis_type: AnalysisType,
            confidence: Confidence,
            reasons: list[str],
            parser: Parser = RAW_PARSER,
            role_id: str | None = None,
            order: list[str] | None = None,
            candidates: dict[str, Parser] | None = None,
        ) -> ColumnDetection:
            return ColumnDetection(
                name=name,
                position=position,
                storage_dtype=dtype,
                type=analysis_type,
                confidence=confidence,
                reasons=reasons,
                parser=parser,
                role=role_id,
                order=order,
                candidates=candidates or {},
            )

        if kind == "boolean":
            return make("boolean", "high", [say("TYP.stored_boolean", "stored as true/false")])
        if kind == "temporal":
            stored = say("TYP.stored_date", "stored as a date")
            return make("datetime", "high", [stored], candidates={"date": RAW_PARSER})
        if kind == "time":
            # R-TYP-07: a time of day alone is no date; as hours it keeps its order and distances
            time_parser = Parser(kind="time")
            stored_time = rule(
                "R-TYP-07.stored",
                "stored as a time of day: read as hours since midnight (07:40 → 7.67)",
            )
            reasons: list[str] = [stored_time]
            return make("numeric_continuous", "high", reasons, time_parser, "hour",
                        candidates={"number": time_parser, "text": TEXT_PARSER})  # fmt: skip
        if kind == "other":
            unsupported = say("TYP.unsupported", "unsupported storage type {dtype}", dtype=dtype)
            return make("ignored", "high", [unsupported])
        if kind == "numeric":
            return self.detect_numeric(name, role, make)
        return self.detect_text(name, role, make)

    def detect_numeric(self, name: str, role: Role | None, make) -> ColumnDetection:
        profile = self.numeric_profile(name)
        candidates = {
            "number": RAW_PARSER,
            "text": TEXT_PARSER,
            "boolean": Parser(kind="boolean"),
        }
        if profile.integer:
            candidates["date"] = Parser(kind="year")
        if ignored := self.empty_or_constant(profile):
            return make("ignored", "high", [ignored[1]], candidates=candidates)
        if is_row_index(name, profile):
            index = say("TYP.row_index", "row numbers left by an export (pandas index)")
            return make("ignored", "high", [index], candidates=candidates)
        if role and not role.accepts_range(profile.minimum, profile.maximum):
            role = None
        # Decimal values are measures: a categorical name ("..._country_score") does not win.
        if role and role.type == "categorical_nominal" and not profile.integer:
            role = None
        # A customer or order key is a whole number or a code: "basket" in decimals is a value.
        if role and role.suffixes and not profile.integer:
            role = None
        # R-TYP-08 (b): mostly distinct whole numbers are no region ("x_coordinate_state_plane")
        if role and role.id in PLACE_ROLES and profile.unique_ratio > 0.5:
            role = None
        role_reason = [name_reason(role)] if role else []
        if role and role.id == "amount" and profile.integer and self.kb.only_total(name):
            # R-TYP-14: "total ads" in whole numbers counts ads; "TotalCharges" in cents is money
            role = self.kb.role("quantity")
            role_reason = [
                rule("R-TYP-14", "the name says 'total' with no money word and the values are "
                     "whole numbers: read as a count, not an amount")
            ]  # fmt: skip
        note = name_note(name)
        if role and note and self.kb.match_role(name, notes=False) != role:
            role_reason = [
                *role_reason,
                rule(
                    "R-TYP-13",
                    "'{note}' closing the name is a note, not the word it is about",
                    note=note,
                ),
            ]

        if role and role.id == "year" and profile.integer:
            between = say("TYP.year_range", "whole numbers between {low} and {high}",
                          low=f"{profile.minimum:.0f}", high=f"{profile.maximum:.0f}")  # fmt: skip
            return make("datetime", "high", [*role_reason, between], Parser(kind="year"), "year",
                        candidates=candidates)  # fmt: skip
        if role and role.type == "categorical_nominal":
            return make(role.type, "high", role_reason, TEXT_PARSER, role.id, candidates=candidates)
        if role and role.key:
            if profile.unique_ratio >= 0.9:
                unique = unique_values(profile.unique_ratio)
                return make("identifier", "high", [*role_reason, unique], TEXT_PARSER, role.id,
                            candidates=candidates)  # fmt: skip
            # Repeated ids (e.g. store_id) reference another table: they are categories.
            shared = say(
                "TYP.shared_key",
                "only {share} unique values: a reference shared by rows, used as categories",
                share=percent(profile.unique_ratio),
            )
            reasons = [*role_reason, shared]
            return make("categorical_nominal", "medium", reasons, TEXT_PARSER, role.id,
                        candidates=candidates)  # fmt: skip
        values = {value for value, _ in profile.top}
        zero_one = profile.distinct == 2 and values <= {"0", "1", "0.0", "1.0"}
        count_reason: list[str] = []
        if zero_one and role and role.id == "quantity":
            # R-TYP-11: "number_of_cyclist_killed" is 0 or 1 in a sample, but it is a count
            count_reason = [rule("R-TYP-11", "the name says a count: 0 and 1 kept as numbers")]
        elif zero_one:
            zero_one_reason = say("TYP.zero_one", "only the values 0 and 1")
            return make("boolean", "high", [zero_one_reason],
                        Parser(kind="boolean"), candidates=candidates)  # fmt: skip
        if not count_reason and (
            code := self.code_reading(
                name,
                role,
                integer=profile.integer,
                minimum=profile.minimum,
                maximum=profile.maximum,
                unique_ratio=profile.unique_ratio,
                present=profile.present,
                length=(profile.length_mode, profile.length_share),
            )
        ):
            analysis_type, confidence, reasons = code
            return make(analysis_type, confidence, reasons, TEXT_PARSER,
                        role.id if role and role.key else None, candidates=candidates)  # fmt: skip
        analysis_type, confidence, reasons = self.numeric_type(
            role, profile.integer, profile.distinct
        )
        reasons = list(dict.fromkeys([*role_reason, *count_reason, *reasons]))
        return make(analysis_type, confidence, reasons, RAW_PARSER,
                    role.id if role else None, candidates=candidates)  # fmt: skip

    # --- Codes (092: R-TYP-03, R-TYP-04, R-TYP-05) ----------------------------------------------

    def code_name(self, name: str) -> tuple[str, str] | None:
        """R-TYP-04, shared with `quantilo.semantics` (S-01): see KnowledgeBase.name_code."""
        return self.kb.name_code(name)

    def code_reading(
        self,
        name: str,
        role: Role | None,
        *,
        integer: bool,
        minimum: float | None,
        maximum: float | None = None,
        unique_ratio: float,
        present: int,
        length: tuple[int | None, float],
        leading_zero: float = 0,
        zero_examples: str = "",
    ) -> tuple[AnalysisType, Confidence, list[str]] | None:
        """Whole numbers that name things rather than measure them: a code (categories) or, when
        nearly all distinct, an identifier. None when the rules do not apply."""
        if not integer or (minimum is not None and minimum < 0):
            return None
        measured = role is not None and role.id in MEASURE_ROLES
        reason: str | None = None
        confidence: Confidence = "high"
        if leading_zero >= LEADING_ZERO_SHARE and not measured:
            reason = rule(
                "R-TYP-03",
                "{share} of values start with 0 (e.g. {examples}): a code, kept as text so the "
                "zeros stay",
                share=percent(leading_zero), examples=zero_examples,
            )  # fmt: skip
        elif named := self.code_name(name):
            # a measure named by another word wins ("YearsCode" is a number of years); not one
            # named by the head word itself ("item_nbr" is an item number, not a quantity)
            kind, word = named
            if kind == "code" and word == "#":
                reason = rule("R-TYP-04.sign", "the name says a code or a number (a number "
                              "sign): whole numbers read as codes, not as a measure")  # fmt: skip
            elif kind == "code":
                reason = rule(
                    "R-TYP-04.code",
                    "the name says a code or a number ('{word}'): whole numbers read as codes, "
                    "not as a measure",
                    word=word,
                )
            else:
                reason = rule(
                    "R-TYP-04.category",
                    "the name says a category ('{word}'): whole numbers read as category codes, "
                    "not as a measure",
                    word=word,
                )
        elif (
            not measured
            and role is None
            and not (set(name_tokens(name)) & self.kb.date_words)
            and not unix_time(minimum, maximum, length[0])
        ):
            digits, share = length
            if digits and share >= FIXED_LENGTH_SHARE:
                if (
                    unique_ratio >= UNIQUE_CODE_SHARE
                    and present >= CODE_MIN_VALUES
                    and digits >= UNIQUE_CODE_DIGITS
                ):
                    reason = rule(
                        "R-TYP-05.unique",
                        "all values are different whole numbers of {digits} digits: an "
                        "identifier, not a measure",
                        digits=digits,
                    )
                elif digits >= FIXED_CODE_DIGITS:
                    reason = rule(
                        "R-TYP-05.length",
                        "{share} of values have exactly {digits} digits: a code (a measure "
                        "would vary in length)",
                        share=percent(share), digits=digits,
                    )  # fmt: skip
                confidence = "medium"
        if reason is None:
            return None
        unique = unique_values(unique_ratio)
        if unique_ratio >= 0.9:
            return "identifier", confidence, [reason, unique]
        return "categorical_nominal", confidence, [reason, unique]

    def quick_ignored(self, name: str) -> str | None:
        """R-ING-06: an empty or constant text column, told by one light query (wide files hold
        dozens of them)."""
        present, distinct, first = self.one(
            f"SELECT count(v), count(DISTINCT v), min(v) FROM "
            f"(SELECT {clean_text(name)} AS v FROM {SAMPLE_TABLE})"
        )
        profile = Profile(rows=self.rows, present=present, distinct=distinct, top=[(first, 1)])
        ignored = self.empty_or_constant(profile)
        return ignored[1] if ignored else None

    def detect_text(self, name: str, role: Role | None, verdict) -> ColumnDetection:
        if (ignored := self.quick_ignored(name)) is not None:
            return verdict("ignored", "high", [ignored], candidates={"text": TEXT_PARSER})
        profile = self.text_profile(name)
        candidates: dict[str, Parser] = {"text": TEXT_PARSER}
        best_decimal = self.decimal_choice(profile)
        if profile.number_matches[best_decimal]:
            candidates["number"] = Parser(kind="number", decimal=best_decimal)  # type: ignore[arg-type]
        best_format = max(ALL_DATE_FORMATS, key=lambda fmt: profile.date_matches[fmt])
        if profile.date_matches[best_format]:
            candidates["date"] = Parser(kind="date", format=best_format)
        lowered = {value.lower() for value, _ in profile.top}
        pair = self.kb.boolean_pair_for(lowered) if profile.distinct <= 2 else None
        if pair:
            candidates["boolean"] = Parser(
                kind="boolean", true_values=pair[0], false_values=pair[1]
            )
        # Reasons of a 092 rule that changed how the values are read, whatever the verdict.
        notes: list[str] = []

        def make(
            analysis_type: AnalysisType, confidence: Confidence, reasons: list[str], *args: Any,
            **kwargs: Any,
        ) -> ColumnDetection:  # fmt: skip
            return verdict(analysis_type, confidence, [*notes, *reasons], *args, **kwargs)

        if ignored := self.empty_or_constant(profile):
            return make("ignored", "high", [ignored[1]], candidates=candidates)
        # R-TYP-08 (a): a role whose values have a shape needs it ("ON-SITE" is no URL)
        if (
            role
            and role.pattern in VALUE_PATTERNS
            and profile.rate(profile.pattern_matches[role.pattern]) < 0.5
        ):
            role = None
        role_reason = [name_reason(role)] if role else []

        if role and role.type == "text":
            return self.named_text(name, role, profile, make, candidates)
        if pair:
            reason = say("TYP.two_values", "two values: {values}",
                         values=" / ".join(value for value, _ in profile.top))  # fmt: skip
            return make("boolean", "high", [reason], candidates["boolean"], candidates=candidates)
        if role and role.id in ("postal_code", "phone"):
            kept = say("TYP.kept_text", "kept as text to preserve leading zeros and formatting")
            reasons: list[str] = [*role_reason, kept]
            return make("categorical_nominal", "high", reasons, TEXT_PARSER, role.id,
                        candidates=candidates)  # fmt: skip

        numbers = profile.number_matches[best_decimal]
        number_rate = profile.rate(numbers)
        # Mostly numbers, the rest the same numbers behind letters: codes. Read as numbers,
        # the lettered ones would become empty (Online Retail's cancelled invoices, 040).
        codes = profile.pattern_matches["code"]
        coded = codes > 0 and profile.rate(numbers + codes) >= 0.999
        # A quoted "0007200179" named id is a key: as a number it loses its zeros (040).
        identifier = role is not None and role.key
        # A customer or order key in decimals is a value ("basket"), not a key (048).
        if (
            identifier
            and role
            and role.suffixes
            and number_rate >= self.thresholds.parse_rate
            and not self.parsed_number_profile(name, best_decimal)[3]
        ):
            role, identifier = None, False
        marker = self.missing_marker(profile, numbers, best_decimal)
        if (number_rate >= self.thresholds.parse_rate or marker) and not identifier:
            reading = self.numbers_or_codes(
                name, role, profile, best_decimal, marker, coded, make, candidates, notes
            )
            if reading is not None:
                return reading
            role_reason = [name_reason(role)] if role else []

        date_rate = profile.rate(profile.date_matches[best_format])
        if date_rate >= self.thresholds.parse_rate:
            reasons: list[str] = [say("TYP.dates", "{share} of values read as dates ({format})",
                           share=percent(date_rate), format=best_format)]  # fmt: skip
            confidence = "high" if date_rate >= 0.99 else "medium"
            swapped = swap_day_month(best_format)
            if swapped and profile.date_matches.get(swapped) == profile.date_matches[best_format]:
                confidence = "medium"
                reasons.append(
                    say("TYP.day_first", "day/month order is ambiguous: day first assumed")
                )
            return make("datetime", confidence, reasons, candidates["date"], candidates=candidates)
        if second := self.second_date_format(profile, best_format):
            other, rate = second
            parser = Parser(kind="date", format=best_format, formats=[best_format, other])
            candidates["date"] = parser
            reason = rule(
                "R-TYP-06",
                "{share} of values read as dates in two formats ({first} and {second})",
                share=percent(rate), first=best_format, second=other,
            )  # fmt: skip
            return make("datetime", "medium", [reason], parser, candidates=candidates)

        best_time = max(TIME_FORMATS, key=lambda fmt: profile.time_matches[fmt])
        time_rate = profile.rate(profile.time_matches[best_time])
        if time_rate >= self.thresholds.parse_rate:
            parser = Parser(kind="time", format=best_time)
            candidates["number"] = parser
            reason = rule(
                "R-TYP-07",
                "{share} of values are times of day ({format}, e.g. {examples}): read as hours "
                "since midnight",
                share=percent(time_rate), format=best_time, examples=examples(profile, 2),
            )  # fmt: skip
            return make("numeric_continuous", "high", [reason], parser, "hour",
                        candidates=candidates)  # fmt: skip

        order = self.kb.order_for(lowered) if profile.distinct <= 12 else None
        if order:
            by_lower = {value.lower(): value for value, _ in profile.top}
            ordered = [by_lower[item] for item in order if item in by_lower]
            reason = say("TYP.order", "values follow a known order: {order}",
                         order=" < ".join(ordered))  # fmt: skip
            return make("categorical_ordinal", "high", [reason], TEXT_PARSER,
                        role.id if role else None, ordered, candidates)  # fmt: skip
        if ranges := self.range_order(profile):
            shown = " < ".join(ranges[:4]) + (" < …" if len(ranges) > 4 else "")
            reason = rule("R-TYP-10", "values are ranges ordered by their lower bound: {order}",
                          order=shown)  # fmt: skip
            return make("categorical_ordinal", "high", [reason], TEXT_PARSER,
                        role.id if role else None, ranges, candidates)  # fmt: skip

        for pattern in ("email", "url", "uuid"):
            if profile.rate(profile.pattern_matches[pattern]) >= self.thresholds.parse_rate:
                pattern_role = "identifier" if pattern == "uuid" else pattern
                analysis_type: AnalysisType = (
                    "identifier" if profile.unique_ratio >= 0.9 else "categorical_nominal"
                )
                looks = {
                    "email": say("TYP.emails", "values look like emails"),
                    "url": say("TYP.urls", "values look like URLs"),
                    "uuid": say("TYP.uuids", "values look like uuids"),
                }[pattern]
                share = percent(profile.unique_ratio)
                reasons: list[str] = [looks, say("TYP.unique_share", "{share} unique", share=share)]
                return make(analysis_type, "high", reasons, TEXT_PARSER, pattern_role,
                            candidates=candidates)  # fmt: skip

        if role and role.key and profile.unique_ratio >= 0.9:
            reasons: list[str] = [*role_reason, unique_values(profile.unique_ratio)]
            return make("identifier", "high", reasons, TEXT_PARSER, role.id, candidates=candidates)

        if choices := self.choice_list(profile):
            return make("text", "high", [choices], TEXT_PARSER, "choices", candidates=candidates)

        spaced_rate = profile.rate(profile.spaced)
        # R-TXT-1: unique single words are keys, however long (hashes, tokens).
        if (
            profile.unique_ratio >= self.thresholds.identifier_unique
            and profile.present >= 20
            and spaced_rate < 0.1
        ):
            share = percent(profile.unique_ratio)
            reasons: list[str] = [
                say(
                    "TYP.unique_examples",
                    "{share} unique values, e.g. {examples}",
                    share=share,
                    examples=examples(profile),
                )
            ]
            return make("identifier", "medium", reasons, TEXT_PARSER,
                        role.id if role else None, candidates=candidates)  # fmt: skip
        return self.text_verdict(name, role, profile, make, candidates)

    # --- Numbers written as text (092: R-TYP-01 to R-TYP-05) ------------------------------------

    def unreadable_split(self, decimal: str) -> tuple[int, list[str], int, list[str], int]:
        """Unreadable texts of the current column (as numbers): those holding a digit (codes:
        "2A", "V57"), their examples, then the words ("see note", "s", or a bound such as
        "Less than 1 year"), their examples and how many distinct words."""
        number = number_sql("v", decimal)
        rows = self.connection.execute(
            f"SELECT v, c, regexp_matches(v, '[0-9]') AS d FROM {VALUES_TABLE} "
            f"WHERE {number} IS NULL ORDER BY c DESC, v"
        ).fetchall()
        # a bound ("Less than 1 year", "65 or older") is no code: a word around a number
        bound = [
            bool(d and (LOWER_BOUND.search(str(v)) or UPPER_BOUND.search(str(v))))
            for v, _, d in rows
        ]
        coded = [(str(v), int(c)) for (v, c, d), b in zip(rows, bound, strict=True) if d and not b]
        words = [(str(v), int(c)) for (v, c, d), b in zip(rows, bound, strict=True) if not d or b]
        return (
            sum(c for _, c in coded),
            [v for v, _ in coded[:3]],
            sum(c for _, c in words),
            [v for v, _ in words[:3]],
            len(words),
        )

    def missing_marker(self, profile: Profile, numbers: int, decimal: str) -> list[str] | None:
        """R-TYP-02: words in place of numbers ("see note", "heavy" in a column of numbers, or
        "s" = suppressed in 12 % of the scores) are no information to keep: read as missing.
        Not when codes holding a digit would be lost (R-TYP-01)."""
        lost = profile.present - numbers
        if not lost or profile.rate(numbers) < MARKER_MIN_NUMBERS:
            return None
        coded, _codes, worded, words, distinct = self.unreadable_split(decimal)
        if not worded or profile.rate(coded) > LOST_LIMIT:
            return None
        short = distinct <= 2 and all(len(word) <= MARKER_MAX_LENGTH for word in words)
        if profile.rate(numbers) >= self.thresholds.parse_rate or short:
            return words
        return None

    def zero_examples(self, profile: Profile) -> str:
        shown = [v for v, _ in profile.top if len(v) > 1 and v.startswith("0") and v.isdigit()]
        if not shown:
            shown = [
                str(row[0])
                for row in self.connection.execute(
                    f"SELECT v FROM {VALUES_TABLE} WHERE regexp_full_match(v, '0[0-9]+') "
                    "ORDER BY c DESC, v LIMIT 2"
                ).fetchall()
            ]
        return ", ".join(shown[:2])

    def numbers_or_codes(
        self,
        name: str,
        role: Role | None,
        profile: Profile,
        decimal: str,
        marker: list[str] | None,
        coded: bool,
        make,
        candidates: dict[str, Parser],
        notes: list[str],
    ) -> ColumnDetection | None:
        """Text that reads as numbers: a code (R-TYP-03 to 05), numbers (with a marker read as
        missing, R-TYP-02), or None to go on with the text rules (R-TYP-01: values would be
        lost, or letter-prefixed codes)."""
        numbers = profile.number_matches[decimal]
        coded_lost, codes, *_words = self.unreadable_split(decimal)
        if profile.rate(coded_lost) > LOST_LIMIT and not marker:
            notes.append(
                rule(
                    "R-TYP-01",
                    "{share} of values, such as {examples}, cannot be read as numbers: kept as "
                    "text so that no value is lost",
                    share=percent(profile.rate(coded_lost)),
                    examples=", ".join(codes),
                )
            )
        _distinct, low, high, integer = self.parsed_number_profile(name, decimal)
        if role and role.id in PLACE_ROLES and profile.unique_ratio > 0.5:
            role = None  # R-TYP-08 (b)
        if integer:  # a code keeps its words as text too (no marker reading)
            code = self.code_reading(
                name,
                None if role and (role.key or role.type == "categorical_nominal") else role,
                integer=True,
                minimum=low,
                maximum=high,
                unique_ratio=profile.unique_ratio,
                present=profile.present,
                length=self.digit_length(),
                leading_zero=profile.rate(profile.leading_zero),
                zero_examples=self.zero_examples(profile),
            )
            if code:
                analysis_type, confidence, reasons = code
                kept = role.id if role and role.type == "categorical_nominal" else None
                return make(analysis_type, confidence, reasons, TEXT_PARSER, kept,
                            candidates=candidates)  # fmt: skip
            if role and role.type == "categorical_nominal" and not notes:
                # a categorical name on whole numbers written as text ("store" = "12")
                reasons = [name_reason(role), say("TYP.codes_kept", "whole numbers kept as codes")]
                return make("categorical_nominal", "high", reasons, TEXT_PARSER, role.id,
                            candidates=candidates)  # fmt: skip
        if notes or coded:
            return None
        return self.text_number(
            name, role, profile, decimal, profile.rate(numbers), make, candidates, marker
        )

    # --- Dates, lists, ranges (092: R-TYP-06, R-TYP-09, R-TYP-10) -------------------------------

    def second_date_format(self, profile: Profile, best: str) -> tuple[str, float] | None:
        """R-TYP-06: a second format that, with the best one, reads every value as a date (all but
        0.5 %: no value lost, as R-TYP-01; otherwise the Clean page unifies them, B-11)."""
        if not profile.date_matches[best]:
            return None
        needed = (1 - LOST_LIMIT) * profile.present
        swapped = swap_day_month(best)
        others = sorted(
            (fmt for fmt in ALL_DATE_FORMATS if fmt not in (best, swapped)),
            key=lambda fmt: -profile.date_matches[fmt],
        )
        for other in others[:3]:
            count = profile.date_matches[other]
            if not count or profile.date_matches[best] + count < needed:
                continue
            (read,) = self.one(
                f"SELECT coalesce(sum(c) FILTER (WHERE {strptime_sql('v', best)} IS NOT NULL "
                f"OR {strptime_sql('v', other)} IS NOT NULL), 0) FROM {VALUES_TABLE}"
            )
            if read >= needed:
                return other, profile.rate(int(read))
        return None

    def choice_list(self, profile: Profile) -> str | None:
        """R-TYP-09: answers of a multiple-choice question, "Python;SQL;Rust"."""
        if profile.distinct <= CATEGORY_MAX_DISTINCT or profile.rate(profile.lists) < LIST_SHARE:
            return None
        items = self.connection.execute(
            f"SELECT i, sum(c) AS n FROM (SELECT trim(unnest(string_split(v, ';'))) AS i, c "
            f"FROM {VALUES_TABLE}) WHERE i <> '' GROUP BY i ORDER BY n DESC, i "
            f"LIMIT {LIST_MAX_CHOICES + 1}"
        ).fetchall()
        if len(items) > LIST_MAX_CHOICES or len(items) > profile.distinct / 2:
            return None
        shown = ", ".join(str(item) for item, _ in items[:3])
        return rule(
            "R-TYP-09",
            "{share} of values are lists separated by ';' of {count} choices (e.g. {examples}): "
            "a multiple-choice answer, not categories",
            share=percent(profile.rate(profile.lists)), count=len(items), examples=shown,
        )  # fmt: skip

    @staticmethod
    def range_order(profile: Profile) -> list[str] | None:
        """R-TYP-10: ranges ("[70-80)", "25-34 years old") ordered on their lower bound."""
        if not 3 <= profile.distinct <= RANGE_MAX_DISTINCT:
            return None
        keyed: list[tuple[float, str]] = []
        plain: list[str] = []
        ranges = 0
        for value, _ in profile.top:
            if match := RANGE_VALUE.match(value):
                ranges += 1
                keyed.append((float(match.group(1).replace(",", "")), value))
            elif found := FIRST_NUMBER.search(value):
                key = float(found.group().replace(",", ""))
                if LOWER_BOUND.search(value):
                    key -= 0.5
                elif UPPER_BOUND.search(value):
                    key += 0.5
                keyed.append((key, value))
            else:
                plain.append(value)
        if ranges < 2 or len(keyed) < RANGE_NUMBERED_SHARE * profile.distinct:
            return None
        return [value for _, value in sorted(keyed)] + plain

    # --- Text verdicts (050) -------------------------------------------------------------------

    def text_values(self, column: str) -> list[tuple[str, int]]:
        """The current text column's distinct texts with their counts (at most
        TEXT_MEASURE_LIMIT of them, picked by a hash of the text: R-ING-06)."""
        rows = self.connection.execute(
            f"SELECT v, c FROM {VALUES_TABLE} ORDER BY hash(v), v LIMIT {TEXT_MEASURE_LIMIT}"
        ).fetchall()
        return [(str(value), int(count)) for value, count in rows]

    def categories_by_count(self, profile: Profile, named_text: bool) -> str | None:
        """R-TXT-2: why the texts are categories ("" when obvious), None when they are not."""
        if profile.distinct <= CATEGORY_MAX_DISTINCT:
            return ""
        coverage = profile.rate(sum(count for _, count in profile.top[:20]))
        if coverage >= CATEGORY_TOP_COVERAGE:
            return say("TYP.top20", "the 20 most frequent values cover {share} of rows",
                       share=percent(coverage))  # fmt: skip
        if profile.unique_ratio < CATEGORY_MAX_UNIQUE and not named_text:
            return say("TYP.few_unique", "only {share} unique values",
                       share=percent(profile.unique_ratio))  # fmt: skip
        return None

    @staticmethod
    def words_reason(measures: TextMeasures) -> str:
        median, p90 = f"{measures.median_words:.0f}", f"{measures.p90_words:.0f}"
        return say("TYP.words", "median {median} words (P90 {p90})", median=median, p90=p90)

    @staticmethod
    def is_short_label(measures: TextMeasures) -> bool:
        low, high = SHORT_WORDS
        return (
            low <= measures.median_words <= high
            and measures.stopword_share < STOPWORDS_SENTENCE
            and measures.sentence_share < PUNCTUATION_SENTENCE
        )

    def label_reasons(self, measures: TextMeasures) -> list[str]:
        return [
            self.words_reason(measures),
            say(
                "TYP.labels",
                "few stop words ({stopwords}) and little sentence punctuation ({punctuation}): "
                "short labels",
                stopwords=percent(measures.stopword_share),
                punctuation=percent(measures.sentence_share),
            ),
        ]

    def sentence_reasons(self, measures: TextMeasures) -> list[str]:
        reasons = [
            self.words_reason(measures),
            say("TYP.stopwords", "{share} stop words", share=percent(measures.stopword_share)),
            say(
                "TYP.punctuation",
                "{share} with sentence punctuation",
                share=percent(measures.sentence_share),
            ),
        ]
        language = language_reason(measures, self.kb)
        return [*reasons, language] if language else reasons

    def named_text(
        self, name: str, role: Role, profile: Profile, make, candidates: dict[str, Parser]
    ) -> ColumnDetection:
        """R-TXT-9: a free-text name keeps the text type; the values choose the role."""
        role_reason = name_reason(role)
        measures = measure(self.text_values(name), self.kb)
        categories = self.categories_by_count(profile, named_text=True)
        if categories is None and self.is_short_label(measures):
            labels = say("TYP.name_labels", "name suggests {role}, but values are short labels",
                         role=role.label.lower())  # fmt: skip
            reasons = [labels, *self.label_reasons(measures)]
            return make("text", "medium", reasons, TEXT_PARSER, "label", candidates=candidates)
        confidence: Confidence = "high" if profile.avg_length > 20 else "medium"
        reasons = [role_reason, self.words_reason(measures)]
        if language := language_reason(measures, self.kb):
            reasons.append(language)
        return make("text", confidence, reasons, TEXT_PARSER, role.id, candidates=candidates)

    def text_verdict(
        self, name: str, role: Role | None, profile: Profile, make, candidates: dict[str, Parser]
    ) -> ColumnDetection:
        """R-TXT-2 to R-TXT-8: categories, address, free text, short text, in this order."""
        role_reason = [name_reason(role)] if role else []
        role_id = role.id if role else None

        def categories(reason: str = "") -> ColumnDetection:
            if role and role.type == "categorical_nominal":
                return make("categorical_nominal", "high", role_reason, TEXT_PARSER, role.id,
                            candidates=candidates)  # fmt: skip
            confidence: Confidence = (
                "medium" if profile.distinct <= CATEGORY_MAX_DISTINCT else "low"
            )
            reasons: list[str] = [say("TYP.distinct", "{count} distinct values, e.g. {examples}",
                           count=profile.distinct, examples=examples(profile))]  # fmt: skip
            kept = role_id if role_id != "label" else None  # R-TXT-11
            if reason:
                reasons.append(reason)
            return make("categorical_nominal", confidence, reasons, TEXT_PARSER, kept,
                        candidates=candidates)  # fmt: skip

        reason = self.categories_by_count(profile, named_text=role_id in TEXT_NAME_ROLES)
        if reason is not None:
            return categories(reason)
        # A repeated id ("CA-2017-100111") is a reference, never a sentence nor a label.
        if role_id == "identifier":
            return categories()

        measures = measure(self.text_values(name), self.kb)
        words = self.words_reason(measures)
        # R-TXT-3, before short text: an address is also 3 to 8 words without stop words.
        by_name = role_id == "address" and measures.median_words >= 2
        by_values = (
            measures.address_share >= ADDRESS_SHARE
            and measures.sentence_share < PUNCTUATION_SENTENCE
        )
        if by_name or by_values:
            reasons = [*role_reason] if by_name else []
            if measures.address_share:
                reasons.append(
                    say(
                        "TYP.addresses",
                        "{share} look like street addresses (a number and a street word)",
                        share=percent(measures.address_share),
                    )
                )
            if measures.postal_share >= 0.1:
                reasons.append(say("TYP.postal", "{share} with a postal code",
                                   share=percent(measures.postal_share)))  # fmt: skip
            sure = (by_name and by_values) or measures.address_share >= 0.8
            return make("text", "high" if sure else "medium", [*reasons, words], TEXT_PARSER,
                        "address", candidates=candidates)  # fmt: skip

        # R-TXT-4: sentences hold grammatical words or end with punctuation.
        long_enough = (
            measures.median_words >= FREE_MEDIAN_WORDS
            or profile.avg_length > FREE_MEAN_LENGTH
            or (measures.median_words >= 2 and profile.unique_ratio >= 0.5)
        )
        sentences = (
            measures.stopword_share >= STOPWORDS_SENTENCE
            or measures.sentence_share >= PUNCTUATION_SENTENCE
        )
        if long_enough and sentences:
            clear = measures.stopword_share >= 0.25 or measures.sentence_share >= 0.5
            return make("text", "high" if clear else "medium",
                        [*role_reason, *self.sentence_reasons(measures)], TEXT_PARSER,
                        "free_text", candidates=candidates)  # fmt: skip

        # R-TXT-5: a categorical name (city, status...) wins over a short-text reading.
        if role and role.type == "categorical_nominal":
            return categories()

        # R-TXT-6: short texts that name things (product names, titles).
        if self.is_short_label(measures):
            confidence = "high" if role_id in TEXT_NAME_ROLES else "medium"
            return make("text", confidence, [*role_reason, *self.label_reasons(measures)],
                        TEXT_PARSER, "label", candidates=candidates)  # fmt: skip

        # R-TXT-7: long values, the former rule (keyword lists, text without known stop words).
        spaced_rate = profile.rate(profile.spaced)
        if profile.avg_length > FREE_MEAN_LENGTH or (
            profile.unique_ratio > 0.5 and spaced_rate > 0.5 and profile.avg_length > 20
        ):
            long = say("TYP.long", "long values (average {length} characters)",
                       length=f"{profile.avg_length:.0f}")  # fmt: skip
            reasons = [long, words]
            return make("text", "medium", reasons, TEXT_PARSER, "free_text",
                        candidates=candidates)  # fmt: skip
        return categories()

    def text_number(
        self,
        name: str,
        role: Role | None,
        profile: Profile,
        decimal: str,
        rate: float,
        make,
        candidates: dict[str, Parser],
        marker: list[str] | None = None,
    ) -> ColumnDetection:
        distinct, low, high, integer = self.parsed_number_profile(name, decimal)
        parser = candidates["number"]
        if role and not role.accepts_range(low, high):
            role = None
        if role is None and profile.rate(profile.percent) >= 0.5:
            role = self.kb.role("percentage")
        elif role is None and profile.rate(profile.currency) >= 0.5:
            role = self.kb.role("amount")
        shown = [value for value, _ in profile.top if value not in (marker or [])][:2]
        reasons = [say("TYP.numbers", "{share} of values read as numbers (e.g. {examples})",
                       share=percent(rate), examples=", ".join(shown))]  # fmt: skip
        if decimal == ",":
            reasons.append(say("TYP.comma", "comma used as decimal separator"))
        if profile.decimal_reason:
            reasons.append(profile.decimal_reason)
        if marker:
            missing = profile.present - profile.number_matches[decimal]
            tokens = ", ".join(repr(token) for token in marker)
            reasons.append(
                rule(
                    "R-TYP-02",
                    "words in place of numbers ({words}: {share} of values) read as missing",
                    words=tokens,
                    share=percent(profile.rate(missing)),
                )
            )
        if role and role.id == "year" and integer:
            year = say("TYP.name", "name suggests {role}", role="year")
            return make("datetime", "high", [*reasons, year], Parser(kind="year"),
                        "year", candidates=candidates)  # fmt: skip
        analysis_type, confidence, type_reasons = self.numeric_type(role, integer, distinct)
        if rate < 0.99 or profile.decimal_doubt:
            confidence = "medium"
        role_reason = [name_reason(role)] if role else []
        all_reasons = list(dict.fromkeys([*reasons, *role_reason, *type_reasons]))
        return make(analysis_type, confidence, all_reasons, parser,
                    role.id if role else None, candidates=candidates)  # fmt: skip


def swap_day_month(date_format: str) -> str | None:
    if "%d" in date_format and "%m" in date_format:
        return date_format.replace("%d", "\0").replace("%m", "%d").replace("\0", "%m")
    return None


# R-TYP-12: how a date part may be read (a single year everywhere is "ignored", still a part)
PART_TYPES = frozenset(
    [
        *("datetime", "numeric_discrete", "numeric_continuous"),
        *("categorical_ordinal", "categorical_nominal", "ignored"),
    ]
)


def date_part_notes(detections: list[ColumnDetection]) -> None:
    """R-TYP-12 (fix1): a year, a month and a day read as separate columns are pointed at on the
    year's reasons; Clean checks their values and proposes the assembled date (B-20). Nothing is
    assembled here: the reading of each column stays as it is."""
    by_name = {d.name: d for d in detections}
    for group in date_parts.groups(by_name):
        if any(by_name[c].type not in PART_TYPES for c in group.columns):
            continue
        year = by_name[group.parts["year"]]
        year.reasons.append(
            rule(
                "R-TYP-12",
                "{columns} look like the parts of one date: Clean proposes to assemble them "
                "into a '{name}' column (B-20)",
                columns=group.listed,
                name=group.name,
            )
        )


def detect_columns(
    connection: duckdb.DuckDBPyConnection,
    source_sql: str,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> list[ColumnDetection]:
    """Detect every column of a query (typically `read_parquet(...)`)."""
    connection.execute("SET threads = 1")  # seeded sampling is only reproducible on one thread
    connection.execute(
        f"CREATE OR REPLACE TEMP TABLE {SAMPLE_TABLE} AS SELECT * FROM {source_sql} "
        f"USING SAMPLE {SAMPLE_ROWS} ROWS (reservoir, 42)"
    )
    connection.execute("RESET threads")
    rows = int((connection.execute(f"SELECT count(*) FROM {SAMPLE_TABLE}").fetchone() or (0,))[0])
    columns = connection.execute(f"DESCRIBE {SAMPLE_TABLE}").fetchall()
    detector = ColumnDetector(connection, load(), rows, thresholds)
    detections = [
        detector.detect(name, dtype, position)
        for position, (name, dtype, *_rest) in enumerate(columns)
    ]
    connection.execute(f"DROP TABLE IF EXISTS {VALUES_TABLE}")
    date_part_notes(detections)
    # What the import changed in a column (R-ING-01) is said first in its reasons (092).
    if (stored := PARQUET_SOURCE.fullmatch(source_sql.strip())) is not None:
        notes = stored_notes(connection, stored.group(1).replace("''", "'"))
        for detection in detections:
            detection.reasons = [*notes.get(detection.name, []), *detection.reasons]
    return detections
