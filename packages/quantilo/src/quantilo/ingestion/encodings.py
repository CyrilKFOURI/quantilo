"""The encoding of a text file, from coded evidence on its bytes (N28, R-ING-08).

DuckDB reads UTF-8, Latin-1 and little-endian UTF-16 itself; the other encodings are read
through a UTF-8 copy written with Python's codecs (`csv_sniff.utf8_copy`), so nothing is
downloaded and the same code runs in the browser (Pyodide).

The rules, in order:
1. A byte-order mark says it: UTF-32, UTF-16 big- or little-endian (a UTF-8 one is skipped by
   DuckDB itself).
2. No mark, but one byte in two is 0 on ASCII text: UTF-16 without a mark (R-ING-08).
3. Valid UTF-8 is UTF-8.
4. Otherwise one byte per character. Each candidate (Windows-1252, Latin-1, ISO-8859-15,
   Mac Roman, DOS CP850) reads the bytes above 0x7F that touch an ASCII letter; a candidate scores
   when it reads them as letters of the same case as their neighbour ('caf' + 0x8E: 'é' as Mac
   Roman, 'Ž' as Windows-1252). Windows-1252 (C1 bytes present) or Latin-1 stays the default
   unless another one reads clearly more letters; ISO-8859-15 when 0xA4 sits by an amount (the
   euro sign there, '¤' in Latin-1). The reason names the evidence.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass

# What the options call each encoding, and Python's codec for it.
CODECS = {
    "utf-8": "utf-8",
    "latin-1": "latin-1",
    "windows-1252": "cp1252",
    "utf-16": "utf-16",
    "utf-16-be": "utf-16-be",
    "utf-32": "utf-32",
    "iso-8859-15": "iso8859_15",
    "mac-roman": "mac_roman",
    "cp850": "cp850",
}
# DuckDB reads these itself (no extension); the others through a UTF-8 copy.
NATIVE = frozenset({"utf-8", "latin-1", "utf-16"})
NAMES = {
    "windows-1252": "Windows-1252",
    "latin-1": "Latin-1",
    "iso-8859-15": "ISO-8859-15",
    "mac-roman": "Mac Roman",
    "cp850": "DOS (CP850)",
}
SINGLE_BYTE = ("windows-1252", "latin-1", "iso-8859-15", "mac-roman", "cp850")
C1_BYTES = re.compile(rb"[\x80-\x9f]")
NON_ASCII = re.compile(rb"[\x80-\xff]")
NUL_SHARE = 0.3  # UTF-16 without a mark: this share of the even (or odd) bytes are 0 ...
NUL_OTHER = 0.02  # ... and almost none of the others
MIN_LETTERS = 3  # a candidate must read at least this many more letters than the default
CLEAR_SHARE = 0.9  # ... and read right this share of the bytes it was judged on
EURO = re.compile(rb"(?:\d ?\xa4|\xa4 ?\d)")
# Non-letters a byte above 0x7F may plausibly be beside a letter: inside a word (a no-break space
# joining two words, an apostrophe, a dash), or at its edge (quotes, an ellipsis, a degree, €).
INSIDE_WORD = frozenset(chr(c) for c in (0xA0, 0x2019, 0x2018, 0x2013, 0x2014, 0x2011, 0xB7))
WORD_EDGE = INSIDE_WORD | frozenset(
    chr(c)
    for c in (0x201C, 0x201D, 0xAB, 0xBB, 0x2026, 0xB0, 0xAE, 0x2122, 0xA9, 0xB2, 0xB3, 0x20AC)
)


@dataclass(frozen=True)
class Evidence:
    encoding: str
    reason: str | None = None  # None: the plain case (UTF-8, or a mark everybody reads)


def codec(encoding: str) -> str:
    """Python's codec for an encoding of the options ('windows-1252' → 'cp1252')."""
    return CODECS.get(encoding, encoding)


def _is_letter(byte: int) -> bool:
    return 0x41 <= byte <= 0x5A or 0x61 <= byte <= 0x7A


def _score(raw: bytes, positions: list[int], encoding: str) -> tuple[int, int, int, bytes | None]:
    """(bytes read plausibly, of which letters, bytes judged, a word read right) for one
    candidate. Plausible: a letter (small after a small one), or the punctuation text holds there
    (a no-break space 0xA0 or an apostrophe inside a word, as in Windows-1252)."""
    table = bytes(range(256)).decode(codec(encoding), errors="replace")
    plausible = letters = judged = 0
    example: bytes | None = None
    for index in positions:
        byte = raw[index]
        before = raw[index - 1] if index else 0
        after = raw[index + 1] if index + 1 < len(raw) else 0
        inside = _is_letter(before) and _is_letter(after)
        if not (inside or _is_letter(before) or _is_letter(after)):
            continue
        judged += 1
        char = table[byte]
        if not char.isalpha():
            plausible += char in (INSIDE_WORD if inside else WORD_EDGE)
            continue
        # after a small letter, a small letter ('OrlÉans' is wrong); a capital may open a word
        if _is_letter(before) and chr(before).islower() and not char.islower():
            continue
        plausible += 1
        letters += 1
        if example is None:
            start = index
            while start > 0 and _is_letter(raw[start - 1]):
                start -= 1
            end = index + 1
            while end < len(raw) and (_is_letter(raw[end]) or raw[end] >= 0x80):
                end += 1
            example = raw[start:end]
    return plausible, letters, judged, example


def _byte_order(raw: bytes) -> Evidence | None:
    if raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return Evidence(
            "utf-32", "A UTF-32 byte-order mark opens the file: read as UTF-32 (R-ING-08)."
        )
    if raw.startswith(codecs.BOM_UTF16_BE):
        return Evidence(
            "utf-16-be",
            "A big-endian UTF-16 byte-order mark opens the file: read as UTF-16 BE (R-ING-08).",
        )
    if raw.startswith(codecs.BOM_UTF16_LE):
        return Evidence("utf-16")
    return None


def _no_mark_utf16(raw: bytes) -> Evidence | None:
    sample = raw[: len(raw) - len(raw) % 2]
    if len(sample) < 8:
        return None
    even = sample[0::2].count(0) / (len(sample) / 2)
    odd = sample[1::2].count(0) / (len(sample) / 2)
    if odd >= NUL_SHARE and even <= NUL_OTHER:
        found = "utf-16"
        side = "little-endian"
    elif even >= NUL_SHARE and odd <= NUL_OTHER:
        found = "utf-16-be"
        side = "big-endian"
    else:
        return None
    zero = max(even, odd)
    return Evidence(
        found,
        f"No byte-order mark, but {zero:.0%} of every other byte is 0, as in text of plain "
        f"letters written in UTF-16: read as UTF-16 {side} (R-ING-08).",
    )


def _single_byte(raw: bytes) -> Evidence:
    default = "windows-1252" if C1_BYTES.search(raw) else "latin-1"
    positions = [found.start() for found in NON_ASCII.finditer(raw)]
    scores = {name: _score(raw, positions, name) for name in SINGLE_BYTE}
    base_plausible, base_letters = scores[default][0], scores[default][1]
    best = max(
        (name for name in SINGLE_BYTE if name != default),
        key=lambda name: (scores[name][0], scores[name][1], -SINGLE_BYTE.index(name)),
    )
    plausible, letters, judged, word = scores[best]
    clear = (
        plausible - base_plausible >= MIN_LETTERS
        and letters - base_letters >= MIN_LETTERS
        and judged > 0
        and plausible / judged >= CLEAR_SHARE
    )
    example = "" if word is None else word.decode(codec(best), errors="replace")
    if clear and best not in ("windows-1252", "latin-1", "iso-8859-15"):
        wrong = "" if word is None else word.decode(codec(default), errors="replace")
        return Evidence(
            best,
            f"Not UTF-8: read as {NAMES[best]}, the only encoding that reads the {letters:,} "
            f"accented letters inside words as letters ('{example}', '{wrong}' as "
            f"{NAMES[default]}) (R-ING-08). Check your data.",
        )
    if default == "latin-1" and EURO.search(raw):
        return Evidence(
            "iso-8859-15",
            "Not UTF-8: byte 0xA4 sits next to amounts, the euro sign in ISO-8859-15 ('¤' in "
            "Latin-1): read as ISO-8859-15 (R-ING-08). Check your data.",
        )
    if clear and best == "iso-8859-15" and default == "latin-1":
        return Evidence(
            best,
            f"Not UTF-8: read as ISO-8859-15, which reads {letters:,} letters inside words "
            f"('{example}') where Latin-1 has symbols (R-ING-08). Check your data.",
        )
    return Evidence(default)


def detect(raw: bytes) -> Evidence:
    """The encoding of a sample of a file's first bytes, and why when it is not the plain case."""
    marked = _byte_order(raw)
    if marked is not None:
        return marked
    nul = _no_mark_utf16(raw)
    if nul is not None:
        return nul
    try:
        # Incremental decoding tolerates a multi-byte character cut by the sample boundary.
        codecs.getincrementaldecoder("utf-8")().decode(raw, final=False)
        return Evidence("utf-8")
    except UnicodeDecodeError:
        return _single_byte(raw)
