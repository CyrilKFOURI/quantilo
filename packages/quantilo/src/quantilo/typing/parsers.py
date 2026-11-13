"""SQL expressions reading a stored column in its analysis type (see `Parser`)."""

import re

from quantilo.ingestion.sql import ident, lit
from quantilo.typing.knowledge_base import load
from quantilo.typing.types import Parser

# --- Numbers (N18: French / EU / US amounts) ---------------------------------------------------

# Spaces inside an amount: space, no-break space, thin no-break space (U+202F, French typography).
_SPACE = "[ \u00a0\u202f]"
# A currency sign or ISO 4217 code, before or after the number ("1 234,56 €", "EUR 12,50"). Codes in
# capitals (and 'eur', 'usd', as before): a lowercase 'try' or 'mad' after a number is a word.
CURRENCY_SIGNS = "€$£¥"
CURRENCY_CODES = (
    *("EUR", "USD", "GBP", "CHF", "CAD", "AUD", "NZD", "JPY", "CNY", "HKD", "SGD", "INR"),
    *("SEK", "NOK", "DKK", "ISK", "PLN", "CZK", "HUF", "RON", "BGN", "TRY", "RUB", "UAH"),
    *("MAD", "TND", "DZD", "XOF", "XAF", "ZAR", "AED", "SAR", "ILS", "KRW", "BRL", "MXN"),
    *("eur", "usd"),
)
_CODE = "(?:" + "|".join(CURRENCY_CODES) + ")"
# The patterns see each code as the sign U+00A4 (`currency_text_sql`): the codes are then written
# once per expression, since DuckDB compiles a pattern for each use.
_CODE_SIGN = "¤"
_CURRENCY = f"[{CURRENCY_SIGNS}{_CODE_SIGN}]"
_PREFIX = f"(?:{_CURRENCY}{_SPACE}*)?"
_SUFFIX = f"{_SPACE}*(?:%|{_CURRENCY})?"
_SIGN = "[+\\-\u2212]"  # U+2212: the minus sign of typeset reports
_MINUS = "[\\-\u2212]"
# A first group of thousands never starts with 0: '0,500' and '0.125' are decimals in any locale.
_LEAD = "[1-9]\\d{0,2}"
# 1,234,567.89 · 1'234.50 (Swiss) · 1 234.5 · 1234.5 · 1.2e3 · .5
_DOT_GROUPS = f"{_LEAD}(?:,\\d{{3}})+|{_LEAD}(?:['\u2019]\\d{{3}})+|{_LEAD}(?:{_SPACE}\\d{{3}})+"
_DOT_CORE = f"(?:(?:{_DOT_GROUPS}|\\d+)(?:\\.\\d+)?(?:[eE][+-]?\\d+)?|\\.\\d+)"
# 1 234 567,89 · 1.234,5 · 1 234,5 (thin space) · 1234,5 (one separator kind per number)
_COMMA_GROUPS = f"{_LEAD}(?:\\.\\d{{3}})+|{_LEAD}(?:{_SPACE}\\d{{3}})+"
_COMMA_CORE = f"(?:{_COMMA_GROUPS}|\\d+)(?:,\\d+)?"


def _number_pattern(core: str) -> str:
    """A signed number ('-12', '€-12', '-€12'), an accounting negative in brackets ('(1 234,56)',
    '(12 €)', '($12)') or a trailing minus ('1 234,56-'), with its currency or percent; on the
    text where every currency code is the sign U+00A4 (`currency_text_sql`)."""
    start = f"(?:{_SIGN}{_CURRENCY}{_SPACE}*|{_CURRENCY}{_SPACE}*)?"
    bracketed = f"\\({_SPACE}*{_PREFIX}{core}{_SUFFIX}{_SPACE}*\\)"
    body = f"(?:{_SIGN}?{core}|{bracketed}|{core}{_SPACE}*{_MINUS})"
    return f"^{start}{body}{_SUFFIX}$"


NUMBER_PATTERNS = {".": _number_pattern(_DOT_CORE), ",": _number_pattern(_COMMA_CORE)}
# What number_sql takes out of a matched value to keep its digits, separator, sign, exponent and
# the opening bracket of an accounting negative.
_NOT_DIGITS = {".": "[^0-9.eE+\\-\u2212(]", ",": "[^0-9,+\\-\u2212(]"}
# Names of the currencies a value is written with ('€', 'EUR'), for the profile and `currency_sql`.
CURRENCY_PATTERN = f"[{CURRENCY_SIGNS}]|{_CODE}"
# N18: the ways of writing an amount a reason names (B-07), with the regex finding each.
AMOUNT_FORMS = (
    ("thin spaces as thousands", "[0-9]\u202f[0-9]{3}"),
    ("no-break spaces as thousands", "[0-9]\u00a0[0-9]{3}"),
    ("spaces as thousands", "[0-9] [0-9]{3}"),
    ("apostrophes as thousands", "[0-9]['\u2019][0-9]{3}"),
    ("accounting negatives in brackets", r"^\("),
    ("minus signs after the number", f"[0-9]{_SPACE}*{_MINUS}"),
    ("a currency", CURRENCY_PATTERN),
)

# Formats tried on text columns, most common first.
DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y%m%dT%H%M%S",  # ISO 8601 basic format
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%d %B %Y",
    "%B %d, %Y",
    "%d %b %Y",
    # N19: milliseconds (Northwind's 1998-04-10 00:00:00.000, SQL Server exports), ISO with a final
    # Z (read as UTC, the clock it names), 12-Jan-2024 and Jan 12, 2024. Month names are read in
    # English and French (`strptime_sql`).
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%d-%b-%Y",
    "%b %d, %Y",
)
# N19: a 12-hour clock (Socrata's 01/05/2024 12:00:00 AM), each with its day/month twin. Apart from
# DATE_FORMATS because a reader that tells a date from a date and time by '%H' would drop the time:
# read them through `ALL_DATE_FORMATS` and `has_time`.
TWELVE_HOUR_FORMATS = (
    "%m/%d/%Y %I:%M:%S %p",
    "%d/%m/%Y %I:%M:%S %p",
    "%m/%d/%Y %I:%M %p",
    "%d/%m/%Y %I:%M %p",
)
ALL_DATE_FORMATS = (*DATE_FORMATS, *TWELVE_HOUR_FORMATS)

# N19: month names DuckDB cannot read (it reads English only, and not 'Sept' nor 'Jan.'), each
# rewritten as its English abbreviation before a format with %b or %B reads it: French in full or
# abbreviated (janv., févr., déc.), with or without accents, any case, and English 'Sept', 'Jan.'.
# Longest words first: 'janvier' before 'janv'.
MONTH_WORDS = (
    ("Jan", ("janvier", "janv", "january", "jan")),
    ("Feb", ("février", "fevrier", "february", "févr", "fevr", "fév", "fev", "feb")),
    ("Mar", ("mars", "march", "mar")),
    ("Apr", ("avril", "april", "avr", "apr")),
    ("May", ("mai", "may")),
    ("Jun", ("juin", "june", "jun")),
    ("Jul", ("juillet", "july", "juil", "jul")),
    ("Aug", ("août", "aout", "august", "aug")),
    ("Sep", ("septembre", "september", "sept", "sep")),
    ("Oct", ("octobre", "october", "oct")),
    ("Nov", ("novembre", "november", "nov")),
    ("Dec", ("décembre", "decembre", "december", "déc", "dec")),
)
_FIRST_OF_MONTH = r"\b([0-9]{1,2})er\b"  # '1er mars 2024' (French ordinal of the first day)


def _month_regex(words: tuple[str, ...]) -> str:
    return r"\b(?:" + "|".join(words) + r")\b\.?"


def needs_letters(date_format: str) -> bool:
    """Whether a format reads only texts holding a letter (a month name, AM/PM, ISO's T or Z): a
    column without any letter skips it, as DuckDB plans every format it is given."""
    return bool(re.search("[A-Za-z]", re.sub("%.", "", date_format))) or any(
        directive in date_format for directive in ("%b", "%B", "%p")
    )


def has_time(date_format: str) -> bool:
    """Whether a format reads a time of day (24-hour or 12-hour clock)."""
    return "%H" in date_format or "%I" in date_format


def english_months(text: str) -> str:
    """Python twin of `month_words_sql`: '12 janv. 2024' → '12 Jan 2024', '1er mars' → '1 Mar'."""
    text = re.sub(_FIRST_OF_MONTH, r"\1", text, flags=re.IGNORECASE)
    for english, words in MONTH_WORDS:
        text = re.sub(_month_regex(words), english, text, flags=re.IGNORECASE)
    return text


def month_words_sql(value: str) -> str:
    """The text with its month names written as English abbreviations (see MONTH_WORDS); values
    without a letter are left as they are."""
    first, day = lit("(?i)" + _FIRST_OF_MONTH), lit(r"\1")
    rewritten = f"regexp_replace({value}, {first}, {day})"
    for english, words in MONTH_WORDS:
        regex = lit("(?i)" + _month_regex(words))
        rewritten = f"regexp_replace({rewritten}, {regex}, {lit(english)})"
    return f"CASE WHEN regexp_matches({value}, '[A-Za-z]') THEN {rewritten} ELSE {value} END"


def strptime_sql(value: str, date_format: str, months: str | None = None) -> str:
    """TRY_STRPTIME, month names read in English and French: a format with %B reads abbreviations
    too, since every month name is first rewritten as one (N19). `months`: the value already
    rewritten (`month_words_sql`), when a query tries several formats on it."""
    if "%b" in date_format or "%B" in date_format:
        rewritten = months or month_words_sql(value)
        return f"TRY_STRPTIME({rewritten}, {lit(date_format.replace('%B', '%b'))})"
    return f"TRY_STRPTIME({value}, {lit(date_format)})"


# Times of day tried on text columns (092, R-TYP-07): "07:40", "18.00.00".
TIME_FORMATS = ("%H:%M:%S", "%H:%M", "%H.%M.%S", "%Hh%M")

VALUE_PATTERNS = {
    "email": r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$",
    "url": r"^(?:https?://|www\.)\S+$",
    "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
    "phone": r"^\+?[\d\s().-]{8,20}$",
    # a number behind a few letters: "C489449" (a cancelled invoice), "INV-2024001"
    "code": r"^[A-Za-z]{1,4}[-_]?\d{3,}$",
}


def null_tokens_sql() -> str:
    return "(" + ", ".join(lit(token) for token in load().null_tokens) + ")"


def clean_text(column: str) -> str:
    """Trimmed text value, NULL for empty strings and null tokens ("N/A", "-", ...)."""
    text = f"trim(CAST({ident(column)} AS VARCHAR))"
    return f"CASE WHEN lower({text}) IN {null_tokens_sql()} THEN NULL ELSE {text} END"


def number_sql(value: str, decimal: str) -> str:
    """Parse cleaned text as a number with the given decimal separator; NULL if it doesn't fit.
    N18: an accounting negative in brackets or a trailing minus is negative, a currency code is
    a sign, out of the digits (the E of 'SEK' is no exponent). `value` is written twice only, as
    before N18: DuckDB plans the whole expression (often `clean_text`) at each use."""
    coded = currency_text_sql(value)
    unwanted = lit(_NOT_DIGITS[decimal])
    # the codes only matter to the digits with a decimal point (an E kept for the exponent)
    digits = f"regexp_replace({coded if decimal == '.' else value}, {unwanted}, '', 'g')"
    digits = f"replace({digits}, '\u2212', '-')"
    digits = f"regexp_replace({digits}, '^\\((.*)$', '-\\1')"  # (1 234,56) → -1234,56
    digits = f"regexp_replace({digits}, '^([^-].*)-$', '-\\1')"  # 1 234,56- → -1234,56
    if decimal == ",":
        digits = f"replace({digits}, ',', '.')"
    pattern = lit(NUMBER_PATTERNS[decimal])
    return f"CASE WHEN regexp_full_match({coded}, {pattern}) THEN TRY_CAST({digits} AS DOUBLE) END"


def currency_text_sql(value: str) -> str:
    """The text with each currency code written as the sign U+00A4, as NUMBER_PATTERNS read it."""
    return f"regexp_replace({value}, {lit(_CODE)}, {lit(_CODE_SIGN)}, 'g')"


def currency_sql(value: str, code: bool = False) -> str:
    """The currency sign or code written with an amount ('€', 'EUR'); NULL when there is none (N18:
    a column mixing currencies is said, never converted). `code`: the ISO code, so that '€' and
    'EUR' are one currency ('$' stays '$': dollars of several countries)."""
    written = f"nullif(regexp_extract({value}, {lit(CURRENCY_PATTERN)}, 0), '')"
    if not code:
        return written
    return f"CASE {written} WHEN '€' THEN 'EUR' WHEN '£' THEN 'GBP' ELSE upper({written}) END"


def date_sql(value: str, date_format: str, formats: list[str] | None = None) -> str:
    """Text read as a date (DATE) or a date and time (TIMESTAMP); with several `formats`, the
    first one that reads the value (092, R-TYP-06)."""
    tried = formats or [date_format]
    parsed = [strptime_sql(value, fmt) for fmt in tried]
    combined = parsed[0] if len(parsed) == 1 else f"coalesce({', '.join(parsed)})"
    return combined if any(has_time(fmt) for fmt in tried) else f"CAST({combined} AS DATE)"


def time_sql(raw: str, text: str, time_format: str | None) -> str:
    """Hours since midnight (7:40 -> 7.67) of a TIME value or of text read with `time_format`."""
    moment = (
        f"CAST({raw} AS TIME)"
        if time_format is None
        else f"CAST(TRY_STRPTIME({text}, {lit(time_format)}) AS TIME)"
    )
    return f"(epoch({moment}) / 3600.0)"


def typed_sql(column: str, parser: Parser) -> str:
    """Expression giving the column's value in its analysis type."""
    if parser.kind == "derived" and parser.derived is not None:
        from quantilo.typing.derived import derived_sql  # derived reads this module

        return derived_sql(parser.derived)
    raw = ident(column)
    if parser.kind == "raw":
        return raw
    if parser.kind == "year":
        return f"make_date(TRY_CAST({raw} AS INTEGER), 1, 1)"
    text = clean_text(column)
    if parser.kind == "text":
        return text
    if parser.kind == "number":
        return number_sql(text, parser.decimal)
    if parser.kind == "date":
        return date_sql(text, parser.format or "%Y-%m-%d", parser.formats)
    if parser.kind == "time":
        return time_sql(raw, text, parser.format)
    if not parser.true_values:
        return f"TRY_CAST({raw} AS BOOLEAN)"
    trues = ", ".join(lit(value) for value in parser.true_values)
    falses = ", ".join(lit(value) for value in parser.false_values)
    lowered = f"lower({text})"
    return f"CASE WHEN {lowered} IN ({trues}) THEN true WHEN {lowered} IN ({falses}) THEN false END"


def present_sql(column: str, parser: Parser) -> str:
    """Expression that is non-NULL when the stored value is present (before parsing)."""
    if parser.kind == "derived":
        return typed_sql(column, parser)
    raw_kinds = ("raw", "year") + (("time",) if parser.format is None else ())
    return ident(column) if parser.kind in raw_kinds else clean_text(column)
