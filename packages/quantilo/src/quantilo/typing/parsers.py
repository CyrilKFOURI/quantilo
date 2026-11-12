"""SQL expressions reading a stored column in its analysis type (see `Parser`)."""

from quantilo.ingestion.sql import ident, lit
from quantilo.typing.knowledge_base import load
from quantilo.typing.types import Parser

# Optional currency before, percent / currency after, around a number.
_PREFIX = r"(?:[€$£]\s*)?"
_SUFFIX = r"\s*(?:%|€|\$|£|eur|EUR|usd|USD)?"
# 1,234,567.89 · 1234.5 · -3 · 1.2e3 · .5
_DOT_CORE = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+)"
# 1 234 567,89 · 1.234,5 · 1234,5 · -3 (space, no-break space or dot as thousands separator)
_COMMA_CORE = "[+-]?(?:\\d{1,3}(?:[ \u00a0.]\\d{3})+|\\d+)(?:,\\d+)?"

NUMBER_PATTERNS = {
    ".": f"^{_PREFIX}{_DOT_CORE}{_SUFFIX}$",
    ",": f"^{_PREFIX}{_COMMA_CORE}{_SUFFIX}$",
}

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
)

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
    """Parse cleaned text as a number with the given decimal separator; NULL if it doesn't fit."""
    no_words = f"regexp_replace({value}, '(?i)eur|usd', '', 'g')"
    if decimal == ",":
        digits = f"replace(regexp_replace({no_words}, '[^0-9,+-]', '', 'g'), ',', '.')"
    else:
        digits = f"regexp_replace({no_words}, '[^0-9.eE+-]', '', 'g')"
    pattern = lit(NUMBER_PATTERNS[decimal])
    return f"CASE WHEN regexp_full_match({value}, {pattern}) THEN TRY_CAST({digits} AS DOUBLE) END"


def date_sql(value: str, date_format: str, formats: list[str] | None = None) -> str:
    """Text read as a date (DATE) or a date and time (TIMESTAMP); with several `formats`, the
    first one that reads the value (092, R-TYP-06)."""
    tried = formats or [date_format]
    parsed = [f"TRY_STRPTIME({value}, {lit(fmt)})" for fmt in tried]
    combined = parsed[0] if len(parsed) == 1 else f"coalesce({', '.join(parsed)})"
    return combined if any("%H" in fmt for fmt in tried) else f"CAST({combined} AS DATE)"


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
