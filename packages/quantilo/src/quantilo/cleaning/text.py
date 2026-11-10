"""Text normalizations written twice, identically: in SQL (compile.py, RE2 escapes) and in
Python (the rules compute their correspondences on distinct values already normalized)."""

import html
import re
import unicodedata

# Spaces: ASCII whitespace, no-break and the Unicode spaces.
_SPACES = [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0xA0, 0x1680, (0x2000, 0x200A), 0x2028, 0x2029]
_SPACES += [0x202F, 0x205F, 0x3000]
# Invisible: zero-width, word joiner, byte-order mark, soft hyphen, control characters but the
# whitespace ones (handled by normalize_spaces).
_INVISIBLE = [(0x200B, 0x200D), 0x2060, 0xFEFF, 0xAD, (0x00, 0x08), (0x0E, 0x1F), 0x7F]


def _klass(points: list[int | tuple[int, int]], sql: bool) -> str:
    def one(point: int) -> str:
        return f"\\x{{{point:x}}}" if sql else f"\\U{point:08x}"

    parts = [f"{one(p[0])}-{one(p[1])}" if isinstance(p, tuple) else one(p) for p in points]
    return "[" + "".join(parts) + "]"


SQL_SPACE_RUN = _klass(_SPACES, sql=True) + "+"
SQL_INVISIBLE = _klass(_INVISIBLE, sql=True)
_space_run = re.compile(_klass(_SPACES, sql=False) + "+")
_invisible = re.compile(_klass(_INVISIBLE, sql=False))

# What UTF-8 bytes look like once read as Windows-1252: 'Ã' or 'Â' then a continuation byte.
_CONTINUATION = "".join(sorted(set(bytes(range(0x80, 0xC0)).decode("cp1252", errors="ignore"))))
MOJIBAKE = re.compile(f"[ÃÂ][{re.escape(_CONTINUATION)}]|â€")
HTML_ENTITY = re.compile(r"&(?:[A-Za-z]{2,8}|#[0-9]{1,6}|#[xX][0-9A-Fa-f]{1,6});")


def normalize_spaces(value: str) -> str:
    return _space_run.sub(" ", value).strip(" ")


def remove_invisible(value: str) -> str:
    return _invisible.sub("", value)


def fold(value: str) -> str:
    """The key two spellings share when only case and accents differ (B-03)."""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def fix_encoding(value: str) -> str | None:
    """UTF-8 read as Windows-1252 ('CafÃ©' → 'Café'), or None when it is not that (ftfy's
    prudence: the repair must round-trip and leave no marker)."""
    if not MOJIBAKE.search(value):
        return None
    try:
        repaired = value.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return None
    return None if MOJIBAKE.search(repaired) or repaired == value else repaired


def unescape_html(value: str) -> str | None:
    if not HTML_ENTITY.search(value):
        return None
    decoded = html.unescape(value)
    return decoded if decoded != value else None
