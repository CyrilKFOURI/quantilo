"""Detect the analysis type of every column of a dataset.

Profiles a fixed random sample in DuckDB, then applies ordered rules (each one explained)
combined with the knowledge base. Candidate parsers are evaluated with the very SQL used
later to read the data, so detection and conversion always agree.
"""

import re
from dataclasses import dataclass, field

import duckdb

from quantilo.ingestion.sql import ident, lit
from quantilo.typing.knowledge_base import KnowledgeBase, Role, load
from quantilo.typing.parsers import (
    DATE_FORMATS,
    NUMBER_PATTERNS,
    VALUE_PATTERNS,
    clean_text,
    number_sql,
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
        return Profile(
            rows=self.rows,
            present=present,
            distinct=distinct,
            top=self.top_values(col),
            minimum=float(low) if low is not None else None,
            maximum=float(high) if high is not None else None,
            integer=bool(integer),
        )

    def text_profile(self, column: str) -> Profile:
        value = "v"
        aggregates = [
            f"count({value})",
            f"count(DISTINCT {value})",
            f"coalesce(avg(length({value})), 0)",
            f"count(*) FILTER (WHERE contains({value}, ' '))",
            f"count(*) FILTER (WHERE contains({value}, '%'))",
            f"count(*) FILTER (WHERE regexp_matches({value}, '[€$£]'))",
        ]
        aggregates += [
            f"count(*) FILTER (WHERE regexp_full_match({value}, {lit(pattern)}))"
            for pattern in NUMBER_PATTERNS.values()
        ]
        aggregates += [f"count(TRY_STRPTIME({value}, {lit(fmt)}))" for fmt in DATE_FORMATS]
        aggregates += [
            f"count(*) FILTER (WHERE regexp_full_match({value}, {lit(pattern)}))"
            for pattern in VALUE_PATTERNS.values()
        ]
        row = self.one(
            f"SELECT {', '.join(aggregates)} FROM "
            f"(SELECT {clean_text(column)} AS {value} FROM {SAMPLE_TABLE})"
        )
        present, distinct, avg_length, spaced, percent_count, currency = row[:6]
        offset = 6
        number_matches = dict(zip(NUMBER_PATTERNS, row[offset : offset + 2], strict=True))
        offset += 2
        date_matches = {
            fmt: int(count)
            for fmt, count in zip(
                DATE_FORMATS, row[offset : offset + len(DATE_FORMATS)], strict=True
            )
        }
        offset += len(DATE_FORMATS)
        pattern_matches = dict(zip(VALUE_PATTERNS, row[offset:], strict=True))
        return Profile(
            rows=self.rows,
            present=present,
            distinct=distinct,
            top=self.top_values(clean_text(column)),
            avg_length=float(avg_length),
            spaced=spaced,
            number_matches=number_matches,
            percent=percent_count,
            currency=currency,
            date_matches=date_matches,
            pattern_matches=pattern_matches,
        )

    def parsed_number_profile(self, column: str, decimal: str) -> tuple[int, float, float, bool]:
        parsed = number_sql(clean_text(column), decimal)
        distinct, low, high, integer = self.one(
            f"SELECT count(DISTINCT p), min(p), max(p), coalesce(bool_and(p = floor(p)), false) "
            f"FROM (SELECT {parsed} AS p FROM {SAMPLE_TABLE}) WHERE p IS NOT NULL"
        )
        return int(distinct), float(low), float(high), bool(integer)

    # --- Shared rules --------------------------------------------------------------------------

    def empty_or_constant(self, profile: Profile) -> tuple[str, str] | None:
        if profile.rows == 0 or profile.present / profile.rows < 1 - self.thresholds.empty_rate:
            return (
                "ignored",
                f"{percent(1 - profile.present / max(profile.rows, 1))} of values are empty",
            )
        if profile.distinct == 1:
            return "ignored", f"a single value everywhere: {profile.top[0][0]!r}"
        return None

    def numeric_type(
        self, role: Role | None, integer: bool, distinct: int
    ) -> tuple[AnalysisType, Confidence, list[str]]:
        if role and role.type in NUMERIC_TYPES:
            return role.type, "high", [f"name suggests {role.label.lower()}"]
        if integer and distinct <= self.thresholds.discrete_max:
            confidence: Confidence = "high" if distinct <= 12 else "medium"
            return (
                "numeric_discrete",
                confidence,
                [f"only {distinct} distinct whole numbers: shown as categories"],
            )
        confidence = "high" if distinct > 30 else "medium"
        return "numeric_continuous", confidence, [f"{distinct} distinct numeric values"]

    # --- Detection by storage ------------------------------------------------------------------

    def detect(self, name: str, dtype: str, position: int) -> ColumnDetection:
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
            return make("boolean", "high", ["stored as true/false"])
        if kind == "temporal":
            return make("datetime", "high", ["stored as a date"], candidates={"date": RAW_PARSER})
        if kind == "other":
            return make("ignored", "high", [f"unsupported storage type {dtype}"])
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
            return make("ignored", "high", ["row numbers left by an export (pandas index)"],
                        candidates=candidates)  # fmt: skip
        if role and not role.accepts_range(profile.minimum, profile.maximum):
            role = None
        # Decimal values are measures: a categorical name ("..._country_score") does not win.
        if role and role.type == "categorical_nominal" and not profile.integer:
            role = None
        # A customer or order key is a whole number or a code: "basket" in decimals is a value.
        if role and role.suffixes and not profile.integer:
            role = None
        role_reason = [f"name suggests {role.label.lower()}"] if role else []

        if role and role.id == "year" and profile.integer:
            return make(
                "datetime", "high", [*role_reason, "whole numbers between "
                f"{profile.minimum:.0f} and {profile.maximum:.0f}"],
                Parser(kind="year"), "year", candidates=candidates,
            )  # fmt: skip
        if role and role.type == "categorical_nominal":
            return make(role.type, "high", role_reason, TEXT_PARSER, role.id, candidates=candidates)
        if role and role.key:
            unique = f"{percent(profile.unique_ratio)} unique values"
            if profile.unique_ratio >= 0.9:
                return make("identifier", "high", [*role_reason, unique], TEXT_PARSER, role.id,
                            candidates=candidates)  # fmt: skip
            # Repeated ids (e.g. store_id) reference another table: they are categories.
            reasons = [
                *role_reason,
                f"only {unique}: a reference shared by rows, used as categories",
            ]
            return make("categorical_nominal", "medium", reasons, TEXT_PARSER, role.id,
                        candidates=candidates)  # fmt: skip
        values = {value for value, _ in profile.top}
        if profile.distinct == 2 and values <= {"0", "1", "0.0", "1.0"}:
            return make("boolean", "high", ["only the values 0 and 1"],
                        Parser(kind="boolean"), candidates=candidates)  # fmt: skip
        analysis_type, confidence, reasons = self.numeric_type(
            role, profile.integer, profile.distinct
        )
        reasons = list(dict.fromkeys([*role_reason, *reasons]))
        return make(analysis_type, confidence, reasons, RAW_PARSER,
                    role.id if role else None, candidates=candidates)  # fmt: skip

    def detect_text(self, name: str, role: Role | None, make) -> ColumnDetection:
        profile = self.text_profile(name)
        candidates: dict[str, Parser] = {"text": TEXT_PARSER}
        best_decimal = max(NUMBER_PATTERNS, key=lambda d: (profile.number_matches[d], d == "."))
        if profile.number_matches[best_decimal]:
            candidates["number"] = Parser(kind="number", decimal=best_decimal)  # type: ignore[arg-type]
        best_format = max(DATE_FORMATS, key=lambda fmt: profile.date_matches[fmt])
        if profile.date_matches[best_format]:
            candidates["date"] = Parser(kind="date", format=best_format)
        lowered = {value.lower() for value, _ in profile.top}
        pair = self.kb.boolean_pair_for(lowered) if profile.distinct <= 2 else None
        if pair:
            candidates["boolean"] = Parser(
                kind="boolean", true_values=pair[0], false_values=pair[1]
            )

        if ignored := self.empty_or_constant(profile):
            return make("ignored", "high", [ignored[1]], candidates=candidates)
        role_reason = [f"name suggests {role.label.lower()}"] if role else []

        if role and role.type == "text":
            return self.named_text(name, role, profile, make, candidates)
        if pair:
            reason = f"two values: {' / '.join(value for value, _ in profile.top)}"
            return make("boolean", "high", [reason], candidates["boolean"], candidates=candidates)
        if role and role.id in ("postal_code", "phone"):
            reasons = [*role_reason, "kept as text to preserve leading zeros and formatting"]
            return make("categorical_nominal", "high", reasons, TEXT_PARSER, role.id,
                        candidates=candidates)  # fmt: skip

        number_rate = profile.rate(profile.number_matches[best_decimal])
        # Mostly numbers, the rest the same numbers behind letters: codes. Read as numbers,
        # the lettered ones would become empty (Online Retail's cancelled invoices, 040).
        codes = profile.pattern_matches["code"]
        coded = codes > 0 and profile.rate(profile.number_matches[best_decimal] + codes) >= 0.999
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
        if number_rate >= self.thresholds.parse_rate and not coded and not identifier:
            return self.text_number(
                name, role, profile, best_decimal, number_rate, make, candidates
            )

        date_rate = profile.rate(profile.date_matches[best_format])
        if date_rate >= self.thresholds.parse_rate:
            reasons = [f"{percent(date_rate)} of values read as dates ({best_format})"]
            confidence = "high" if date_rate >= 0.99 else "medium"
            swapped = swap_day_month(best_format)
            if swapped and profile.date_matches.get(swapped) == profile.date_matches[best_format]:
                confidence = "medium"
                reasons.append("day/month order is ambiguous: day first assumed")
            return make("datetime", confidence, reasons, candidates["date"], candidates=candidates)

        order = self.kb.order_for(lowered) if profile.distinct <= 12 else None
        if order:
            by_lower = {value.lower(): value for value, _ in profile.top}
            ordered = [by_lower[item] for item in order if item in by_lower]
            reason = "values follow a known order: " + " < ".join(ordered)
            return make("categorical_ordinal", "high", [reason], TEXT_PARSER,
                        role.id if role else None, ordered, candidates)  # fmt: skip

        for pattern in ("email", "url", "uuid"):
            if profile.rate(profile.pattern_matches[pattern]) >= self.thresholds.parse_rate:
                pattern_role = "identifier" if pattern == "uuid" else pattern
                analysis_type: AnalysisType = (
                    "identifier" if profile.unique_ratio >= 0.9 else "categorical_nominal"
                )
                reasons = [f"values look like {'URLs' if pattern == 'url' else pattern + 's'}",
                           f"{percent(profile.unique_ratio)} unique"]  # fmt: skip
                return make(analysis_type, "high", reasons, TEXT_PARSER, pattern_role,
                            candidates=candidates)  # fmt: skip

        if role and role.key and profile.unique_ratio >= 0.9:
            reasons = [*role_reason, f"{percent(profile.unique_ratio)} unique values"]
            return make("identifier", "high", reasons, TEXT_PARSER, role.id, candidates=candidates)

        spaced_rate = profile.rate(profile.spaced)
        # R-TXT-1: unique single words are keys, however long (hashes, tokens).
        if (
            profile.unique_ratio >= self.thresholds.identifier_unique
            and profile.present >= 20
            and spaced_rate < 0.1
        ):
            reasons = [f"{percent(profile.unique_ratio)} unique values, e.g. {examples(profile)}"]
            return make("identifier", "medium", reasons, TEXT_PARSER,
                        role.id if role else None, candidates=candidates)  # fmt: skip
        return self.text_verdict(name, role, profile, make, candidates)

    # --- Text verdicts (050) -------------------------------------------------------------------

    def text_values(self, column: str) -> list[tuple[str, int]]:
        rows = self.connection.execute(
            f"SELECT v, count(*) FROM (SELECT {clean_text(column)} AS v FROM {SAMPLE_TABLE}) "
            "WHERE v IS NOT NULL GROUP BY v"
        ).fetchall()
        return [(str(value), int(count)) for value, count in rows]

    def categories_by_count(self, profile: Profile, named_text: bool) -> str | None:
        """R-TXT-2: why the texts are categories ("" when obvious), None when they are not."""
        if profile.distinct <= CATEGORY_MAX_DISTINCT:
            return ""
        coverage = profile.rate(sum(count for _, count in profile.top[:20]))
        if coverage >= CATEGORY_TOP_COVERAGE:
            return f"the 20 most frequent values cover {percent(coverage)} of rows"
        if profile.unique_ratio < CATEGORY_MAX_UNIQUE and not named_text:
            return f"only {percent(profile.unique_ratio)} unique values"
        return None

    @staticmethod
    def words_reason(measures: TextMeasures) -> str:
        return f"median {measures.median_words:.0f} words (P90 {measures.p90_words:.0f})"

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
            f"few stop words ({percent(measures.stopword_share)}) and little sentence "
            f"punctuation ({percent(measures.sentence_share)}): short labels",
        ]

    def sentence_reasons(self, measures: TextMeasures) -> list[str]:
        reasons = [
            self.words_reason(measures),
            f"{percent(measures.stopword_share)} stop words",
            f"{percent(measures.sentence_share)} with sentence punctuation",
        ]
        language = language_reason(measures, self.kb)
        return [*reasons, language] if language else reasons

    def named_text(
        self, name: str, role: Role, profile: Profile, make, candidates: dict[str, Parser]
    ) -> ColumnDetection:
        """R-TXT-9: a free-text name keeps the text type; the values choose the role."""
        role_reason = f"name suggests {role.label.lower()}"
        measures = measure(self.text_values(name), self.kb)
        categories = self.categories_by_count(profile, named_text=True)
        if categories is None and self.is_short_label(measures):
            reasons = [f"{role_reason}, but values are short labels", *self.label_reasons(measures)]
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
        role_reason = [f"name suggests {role.label.lower()}"] if role else []
        role_id = role.id if role else None

        def categories(reason: str = "") -> ColumnDetection:
            if role and role.type == "categorical_nominal":
                return make("categorical_nominal", "high", role_reason, TEXT_PARSER, role.id,
                            candidates=candidates)  # fmt: skip
            confidence: Confidence = (
                "medium" if profile.distinct <= CATEGORY_MAX_DISTINCT else "low"
            )
            reasons = [f"{profile.distinct} distinct values, e.g. {examples(profile)}"]
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
                    f"{percent(measures.address_share)} look like street addresses "
                    "(a number and a street word)"
                )
            if measures.postal_share >= 0.1:
                reasons.append(f"{percent(measures.postal_share)} with a postal code")
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
            reasons = [f"long values (average {profile.avg_length:.0f} characters)", words]
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
    ) -> ColumnDetection:
        distinct, low, high, integer = self.parsed_number_profile(name, decimal)
        parser = candidates["number"]
        if role and not role.accepts_range(low, high):
            role = None
        if role is None and profile.rate(profile.percent) >= 0.5:
            role = self.kb.role("percentage")
        elif role is None and profile.rate(profile.currency) >= 0.5:
            role = self.kb.role("amount")
        reasons = [f"{percent(rate)} of values read as numbers (e.g. {examples(profile, 2)})"]
        if decimal == ",":
            reasons.append("comma used as decimal separator")
        if role and role.id == "year" and integer:
            return make("datetime", "high", [*reasons, "name suggests year"], Parser(kind="year"),
                        "year", candidates=candidates)  # fmt: skip
        analysis_type, confidence, type_reasons = self.numeric_type(role, integer, distinct)
        if rate < 0.99:
            confidence = "medium"
        role_reason = [f"name suggests {role.label.lower()}"] if role else []
        all_reasons = list(dict.fromkeys([*reasons, *role_reason, *type_reasons]))
        return make(analysis_type, confidence, all_reasons, parser,
                    role.id if role else None, candidates=candidates)  # fmt: skip


def swap_day_month(date_format: str) -> str | None:
    if "%d" in date_format and "%m" in date_format:
        return date_format.replace("%d", "\0").replace("%m", "%d").replace("\0", "%m")
    return None


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
    return [
        detector.detect(name, dtype, position)
        for position, (name, dtype, *_rest) in enumerate(columns)
    ]
