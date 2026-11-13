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

# N29: a free text keeps its line breaks; only the spaces along a line, around a break and at the
# ends change (the same in SQL and in Python).
_BREAKS: list[int | tuple[int, int]] = [0x0A, 0x0D, 0x2028, 0x2029]
_ALONG = [p for p in _SPACES if p not in _BREAKS]
SQL_ALONG_RUN = _klass(_ALONG, sql=True) + "+"
SQL_BREAK_EDGE = f" ?({_klass(_BREAKS, sql=True)}) ?"
_along_run = re.compile(_klass(_ALONG, sql=False) + "+")
_break_edge = re.compile(f" ?({_klass(_BREAKS, sql=False)}) ?")
# the ends are stripped, never matched by an anchored regex (`X+$` is quadratic on a long
# run of spaces inside a value)
_SPACE_CHARS = "".join(
    chr(c) for p in _SPACES for c in (range(p[0], p[1] + 1) if isinstance(p, tuple) else (p,))
)

# What UTF-8 bytes look like once read as Windows-1252: 'Ã' or 'Â' then a continuation byte. N28:
# read as Latin-1, the continuation bytes 0x80-0x9F are the C1 controls (U+0080-U+009F).
_C1 = "".join(chr(point) for point in range(0x80, 0xA0))
_CONTINUATION = "".join(sorted(set(bytes(range(0x80, 0xC0)).decode("cp1252", errors="ignore"))))
MOJIBAKE = re.compile(f"[ÃÂ][{re.escape(_CONTINUATION + _C1)}]|â[€\x80]")
# N28: UTF-8 read as Mac Roman (an old Mac or a Mac export): 'é' (C3 A9) shows as '√©', the
# apostrophe U+2019 (E2 80 99) as U+201A + 'Äô', 'œ' (C5 93) as '≈ì'. The lead bytes C2, C3,
# C5 and E2 80, then a continuation byte as Mac Roman writes it.
_MAC_CONTINUATION = bytes(range(0x80, 0xC0)).decode("mac_roman")
MAC_MOJIBAKE = re.compile(f"[¬√≈][{re.escape(_MAC_CONTINUATION)}]|" + chr(0x201A) + "Ä")
LOST = chr(0xFFFD)  # N28: the replacement character, a character lost before the file was saved
HTML_ENTITY = re.compile(r"&(?:[A-Za-z]{2,8}|#[0-9]{1,6}|#[xX][0-9A-Fa-f]{1,6});")


def normalize_spaces(value: str, keep_newlines: bool = False) -> str:
    if not keep_newlines:
        return _space_run.sub(" ", value).strip(" ")
    value = _break_edge.sub(r"\1", _along_run.sub(" ", value))
    return value.strip(_SPACE_CHARS)


def normalize_spaces_sql(expr: str, keep_newlines: bool = False) -> str:
    """`normalize_spaces` in SQL (DuckDB, RE2), on an expression."""
    from quantilo.ingestion.sql import lit

    if not keep_newlines:
        return f"trim(regexp_replace({expr}, {lit(SQL_SPACE_RUN)}, ' ', 'g'), ' ')"
    along = f"regexp_replace({expr}, {lit(SQL_ALONG_RUN)}, ' ', 'g')"
    edges = f"regexp_replace({along}, {lit(SQL_BREAK_EDGE)}, '\\1', 'g')"
    return f"trim({edges}, {lit(_SPACE_CHARS)})"


def remove_invisible(value: str) -> str:
    return _invisible.sub("", value)


def fold(value: str) -> str:
    """The key two spellings share when only case and accents differ (B-03)."""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


MAX_ROUNDS = 3  # N28: a value read wrong three times is still repaired, never more


def _single_bytes(value: str) -> bytes:
    """The bytes a Windows-1252 (or Latin-1, for the C1 controls) reading turned into `value`."""
    out = bytearray()
    for char in value:
        try:
            out += char.encode("cp1252")
        except UnicodeError:
            if ord(char) > 0xFF:
                raise
            out.append(ord(char))
    return bytes(out)


def repair_encoding(value: str) -> tuple[str, str] | None:
    """The value as it was written and how it was misread, or None when it is not mojibake
    (ftfy's prudence: the repair must round-trip and leave no marker). The ways (N28):
    - `windows-1252`: UTF-8 read once as Windows-1252 ('CafÃ©' → 'Café');
    - `latin-1`: the same read as Latin-1, its C1 controls kept ('CafÃ\\x83Â©');
    - `double`: read wrong two or three times ('CafÃƒÂ©' → 'Café');
    - `mac-roman`: UTF-8 read as Mac Roman ('Caf√©' → 'Café')."""
    if MOJIBAKE.search(value):
        current, rounds, controls = value, 0, False
        while MOJIBAKE.search(current) and rounds < MAX_ROUNDS:
            controls = controls or any(char in _C1 for char in current)
            try:
                current = _single_bytes(current).decode("utf-8")
            except UnicodeError:
                return None
            rounds += 1
        if current == value or MOJIBAKE.search(current) or MAC_MOJIBAKE.search(current):
            return None
        return current, "double" if rounds > 1 else ("latin-1" if controls else "windows-1252")
    if MAC_MOJIBAKE.search(value):
        try:
            repaired = value.encode("mac_roman").decode("utf-8")
        except UnicodeError:
            return None
        if repaired == value or MAC_MOJIBAKE.search(repaired) or MOJIBAKE.search(repaired):
            return None
        return repaired, "mac-roman"
    return None


def fix_encoding(value: str) -> str | None:
    """`repair_encoding`'s value: what B-05 maps the mojibake to."""
    repaired = repair_encoding(value)
    return None if repaired is None else repaired[0]


def lost_characters(value: str) -> bool:
    """N28: '�' (U+FFFD) in a value: a character replaced before the file was saved, which no
    reading can bring back."""
    return LOST in value


def unescape_html(value: str) -> str | None:
    if not HTML_ENTITY.search(value):
        return None
    decoded = html.unescape(value)
    return decoded if decoded != value else None
