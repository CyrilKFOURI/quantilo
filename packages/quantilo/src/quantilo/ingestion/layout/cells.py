"""What a cell looks like: number, date, number written as text, null token, period label, text.

Pure functions on single values, shared by every step of the detection (Pytheas-style cell
signatures: the rules read these kinds, never the raw cells).
"""

import datetime as dt
import functools
import re
from typing import Literal

Kind = Literal["e", "n", "d", "b", "t", "z", "s"]
"""e empty, n number, d date/time, b boolean, t number written as text, z null token, s text."""

NBSP = "\xa0"
_SPACES = re.compile(r"[\s\xa0  ]+")

# T2: strong tokens are null in any column; weak ones only in a numeric column (a cash book keeps
# its "-" and "X" reconciliation marks).
STRONG_NULLS = frozenset(
    {
        "n/d",
        "n/a",
        "nd",
        "na",
        "#n/a",
        "…",
        "...",
        "---",
        "—",
        "(blank)",
        "(na)",
        "(x)",
        "(d)",
        "(z)",
        "(s)",
        "null",
    }
)
WEAK_NULLS = frozenset({"-", "–", "x", "*", ":", "..", "(b)", "///", "ns", "nc"})

TOTAL_START = re.compile(
    r"^(grand\s+total|totals?\b|sum\b|sub-?totals?\b|sous-total|total\s|ensemble\b|all\b|"
    r"do not delete this totals row)",
    re.I,
)
TOTAL_WORD = re.compile(r"\b(total|totals|subtotal|sous-total)\b", re.I)
NOTE_START = re.compile(
    r"^(notes?\b|sources?\b|footnotes?\b|lecture\s*:|champ\s*:|n\.\s?b\.|nb\s*:|\(\d+\)|\\\d+\\|"
    r"\d+\.\s|[a-z]\s?:\s|n/d\s*=|suggested citation|release date|symbols?\b|---|\*\s|reading\b|"
    r"return to|back to|this worksheet|some cells|if shorthand|to enable|information on|"
    r"footnotes are)",
    re.I,
)
UNIT_CELL = re.compile(
    r"^\s*[\[(][^\])]{1,60}[\])]\s*$|\$/|\bin (thousands|millions|billions)\b|"
    r"\b(usd|eur|mad|gbp)\b|^\s*%\s*$|\(\$|\bbtu\b",
    re.I,
)
FOOTNOTE_REF = re.compile(r"\\\d+\\")
TRAILING_REF = re.compile(r"\s*\((\d{1,3})\)\s*$")
FLAGGED_NUMBER = re.compile(r"^\s*(-?\d[\d\s.,]*?)\s*([a-z]{1,2}|\(\d{1,3}\)|\*{1,3})\s*$", re.I)
LEADING_ZERO = re.compile(r"^0\d")
SHORT_CODE = re.compile(r"^(?=.*[A-Za-z])[A-Za-z0-9]{1,4}$")

_MONTHS = (
    r"jan(uary|vier)?|feb(ruary)?|f[ée]v(rier)?|mar(ch|s)?|apr(il)?|avr(il)?|may|mai|june?|juin|"
    r"july?|juil(let)?|aug(ust)?|ao[uû]t|sep(t(ember)?)?|oct(ober|obre)?|nov(ember|embre)?|"
    r"dec(ember)?|d[ée]c(embre)?"
)
PERIOD = re.compile(
    rf"^((19|20)\d\d(\s*\([^)]{{1,12}}\))?|(18|19|20)\d\d\s*[-/–]\s*\d{{2,4}}|"
    rf"(fall|spring|summer|winter|autumn)\s+(19|20)\d\d|"
    rf"(19|20)\d\d-\d\d-\d\d(t[\d:.]+)?|\d\d?/\d\d?/(19|20)?\d\d|"
    rf"q[1-4]\s*[-/ ]?\s*((19|20)?\d\d)?|((19|20)\d\d)\s*[-/ ]?\s*q[1-4]|"
    rf"h[12]\s*((19|20)\d\d)?|((19|20)\d\d)\s*h[12]|"
    rf"({_MONTHS})\.?(\s*[-/ ]?\s*((19|20)?\d\d))?|((19|20)\d\d)\s*[-/]\s*(0?[1-9]|1[0-2])|"
    rf"(0?[1-9]|[12]\d|3[01])\s*[-/ ]\s*({_MONTHS})\.?(\s*[-/ ]\s*((19|20)?\d\d))?|"
    rf"({_MONTHS})\.?\s+\d{{1,2}},?\s+(19|20)\d\d|week\s*\d{{1,2}}|w\d{{1,2}})$",
    re.I,
)


def clean_text(value: object) -> str:
    """One line, single spaces, no NBSP: what a name or a label reads as (a date at midnight
    reads as its day)."""
    if isinstance(value, dt.datetime) and value.time() == dt.time():
        return value.date().isoformat()
    return _SPACES.sub(" ", str(value)).strip()


def is_blank(value: object) -> bool:
    """G2: None, or a string made only of spaces and NBSP."""
    return value is None or (isinstance(value, str) and not clean_text(value))


@functools.lru_cache(maxsize=65536)
def parse_number(text: str, decimal: str | None = None) -> float | None:
    """A number written as text (T1): "1 234", "1,234.5", "69,55%", "(123)", "$5", "-3.2e4".

    `decimal` forces the decimal separator ("," or "."); None decides from the text alone (a single
    comma followed by exactly 3 digits is a thousands separator)."""
    s = clean_text(text)
    if not s or len(s) > 40:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1].strip()
    s = s.removesuffix("%").strip()
    s = s.lstrip("$€£¥").strip()
    s = s.removeprefix("+")
    if s.startswith("-"):
        negative, s = not negative, s[1:].strip()
    s = s.lstrip("$€£¥").strip()
    s = s.replace(" ", "")
    if not s or not re.fullmatch(r"[\d.,]+([eE][-+]?\d+)?", s) or not s[0].isdigit():
        return None
    comma, dot = "," in s, "." in s
    if comma and dot:
        decimal_mark = "," if s.rfind(",") > s.rfind(".") else "."
    elif comma:
        grouped = re.fullmatch(r"\d{1,3}(,\d{3})+", s)
        decimal_mark = decimal or ("." if grouped or s.count(",") > 1 else ",")
    elif dot:
        grouped = re.fullmatch(r"\d{1,3}(\.\d{3})+", s)
        decimal_mark = "," if (decimal == "," and grouped) or s.count(".") > 1 else "."
    else:
        decimal_mark = "."
    s = s.replace("." if decimal_mark == "," else ",", "").replace(",", ".")
    if s.count(".") > 1:
        return None
    try:
        number = float(s)
    except ValueError:
        return None
    return -number if negative else number


def null_token(text: str) -> Literal["strong", "weak"] | None:
    s = clean_text(text).lower()
    if s in STRONG_NULLS:
        return "strong"
    if s in WEAK_NULLS:
        return "weak"
    return None


def kind(value: object) -> Kind:
    if is_blank(value):
        return "e"
    if isinstance(value, bool):
        return "b"
    if isinstance(value, int | float):
        return "n"
    if isinstance(value, dt.date | dt.datetime | dt.time):
        return "d"
    return _text_kind(str(value))


@functools.lru_cache(maxsize=65536)
def _text_kind(text: str) -> Kind:
    if null_token(text):
        return "z"
    if parse_number(text) is not None and not LEADING_ZERO.match(clean_text(text)):
        return "t"
    return "s"


def numeric_value(value: object) -> float | None:
    """A cell as a number for the arithmetic checks (C9, K1, K2), numbers in text included."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return None if LEADING_ZERO.match(clean_text(value)) else parse_number(value)
    return None


def is_period(value: object) -> bool:
    """A period label (K3): a year, a month, a quarter, "Fall 1990", "1869-70", a date."""
    if isinstance(value, dt.date | dt.datetime):
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return float(value).is_integer() and 1800 <= value <= 2100
    if not isinstance(value, str):
        return False
    return bool(PERIOD.match(clean_text(value)))


def is_year_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return float(value).is_integer() and 1800 <= value <= 2100
    if isinstance(value, str):
        return bool(re.fullmatch(r"(18|19|20)\d\d(\s*\([^)]{1,12}\))?", clean_text(value)))
    return False


def indent_of(value: object) -> int:
    """Leading spaces or NBSP of a label, in steps of 2 (NCES indents with 2 or 4 NBSP)."""
    if not isinstance(value, str):
        return 0
    stripped = value.lstrip(" \xa0  ")
    return (len(value) - len(stripped)) // 2 if stripped else 0


def is_unit_cell(value: object) -> bool:
    return isinstance(value, str) and bool(UNIT_CELL.search(clean_text(value)))


def is_total_label(text: str) -> bool:
    s = clean_text(text)
    return bool(TOTAL_START.match(s) or TOTAL_WORD.search(s[:60]))


def is_note_text(text: str) -> bool:
    s = clean_text(text)
    return bool(NOTE_START.match(s))


def strip_refs(text: str) -> tuple[str, list[str]]:
    """C5-ref: footnote references taken out of a header label ("Instruction\\2\\", "2024 (42)")."""
    refs = [ref.strip("\\") for ref in FOOTNOTE_REF.findall(text)]
    text = FOOTNOTE_REF.sub("", text)
    match = TRAILING_REF.search(text)
    if match and is_year_number(text[: match.start()]):
        refs.append(match.group(1))
        text = text[: match.start()]
    return clean_text(text), refs
