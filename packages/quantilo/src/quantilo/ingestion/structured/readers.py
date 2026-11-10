"""Reading JSON and JSON Lines with the standard library, and telling the formats apart.

The decoder keeps what the user should hear about: keys given twice (R-SST-23, the last one
wins, as Python and RFC 8259 do), NaN and Infinity (R-LIM-04), integers too long for Python
(R-LIM-03, kept as text).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantilo.ingestion.structured.errors import StructuredError
from quantilo.ingestion.structured.model import Format

MB = 1_000_000
NATIVE_LIMIT = 200 * MB  # R-LIM-01
BROWSER_LIMIT = 50 * MB
MAX_NESTING = 64  # R-LIM-02
MAX_INT_DIGITS = 4_300  # R-LIM-03: Python's int_max_str_digits
JSONL_SUFFIXES = {".jsonl", ".ndjson", ".jsonlines"}
SNIFF_LINES = 50  # R-DET-02
SNIFF_SHARE = 0.95
HEAD_BYTES = 65_536
EXAMPLES = 5
SKIPPED_LISTED = 100


@dataclass
class ParseNotes:
    """What the parsers saw that the layout reports (rules R-SST-23, R-LIM-03 to 05)."""

    duplicate_keys: int = 0
    duplicate_examples: list[str] = field(default_factory=list)
    constants: int = 0  # NaN, Infinity, -Infinity
    big_ints: int = 0
    lines: int = 0  # JSON Lines: non-blank lines read
    skipped: int = 0  # JSON Lines: lines that are not JSON
    skipped_lines: list[int] = field(default_factory=list)  # the first ones, 1-based
    last_line: int = 0  # JSON Lines: the last line read, good or bad


def size_limit(max_bytes: int | None) -> int:
    if max_bytes is not None:
        return max_bytes
    return BROWSER_LIMIT if sys.platform == "emscripten" else NATIVE_LIMIT


def check_size(path: Path, max_bytes: int | None, what: str, hint: str) -> None:
    size = path.stat().st_size
    limit = size_limit(max_bytes)
    if size > limit:
        raise StructuredError(
            f"This {what} is {size / MB:,.0f} MB, over the {limit / MB:,.0f} MB limit for one "
            f"document: {hint}.",
            rule="R-LIM-01",
        )


def too_deep() -> StructuredError:
    return StructuredError(
        f"Nesting too deep: this document has more than {MAX_NESTING} levels.", rule="R-LIM-02"
    )


class Decoder:
    """`json.JSONDecoder` with hooks that count instead of failing."""

    def __init__(self, notes: ParseNotes) -> None:
        self.notes = notes
        self.decoder = json.JSONDecoder(
            object_pairs_hook=self.pairs, parse_constant=self.constant, parse_int=self.integer
        )

    def pairs(self, pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = dict(pairs)
        if len(result) != len(pairs):
            self.notes.duplicate_keys += len(pairs) - len(result)
            seen: set[str] = set()
            for key, _ in pairs:
                examples = self.notes.duplicate_examples
                if key in seen and key not in examples and len(examples) < EXAMPLES:
                    examples.append(key)
                seen.add(key)
        return result

    def constant(self, name: str) -> float:
        self.notes.constants += 1
        return float(name)  # "NaN", "Infinity", "-Infinity"

    def integer(self, text: str) -> int | str:
        digits = len(text) - (text[0] == "-")
        if digits > MAX_INT_DIGITS:
            self.notes.big_ints += 1
            return text
        return int(text)

    def decode(self, text: str) -> Any:
        try:
            return self.decoder.decode(text)
        except RecursionError as error:
            raise too_deep() from error


def decode_bytes(raw: bytes) -> str:
    try:
        return raw.decode(json.detect_encoding(raw))
    except UnicodeDecodeError as error:
        raise StructuredError(
            "This file is not UTF-8 (or UTF-16/32) encoded, as JSON must be."
        ) from error


def head(path: Path, size: int = HEAD_BYTES) -> bytes:
    with path.open("rb") as stream:
        return stream.read(size)


def detect_format(path: Path) -> Format:
    """R-DET-01: the content decides (`<` XML, `{` or `[` JSON); a JSON Lines suffix says so."""
    start = head(path, 4_096)
    for bom in (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff"):
        start = start.removeprefix(bom)
    first = start.lstrip(b" \t\r\n\x00")[:1]
    if first == b"<":
        return "xml"
    if first in (b"{", b"["):
        return "jsonl" if path.suffix.lower() in JSONL_SUFFIXES else "json"
    if path.suffix.lower() in JSONL_SUFFIXES:
        return "jsonl"
    raise StructuredError(
        f"{path.name} is not a JSON, JSON Lines or XML document (it starts with neither "
        "'{', '[' nor '<').",
        rule="R-DET-01",
    )


def looks_jsonl(path: Path) -> bool:
    """R-DET-02: at least 95 % of the first 50 non-blank lines are JSON values (two at least)."""
    lines = []
    with path.open("rb") as stream:
        for raw in stream:
            if raw.strip():
                lines.append(raw)
            if len(lines) >= SNIFF_LINES:
                break
    if len(lines) < 2:
        return False
    good = 0
    for index, raw in enumerate(lines):
        try:
            json.loads(raw.removeprefix(b"\xef\xbb\xbf") if index == 0 else raw)
            good += 1
        except (ValueError, RecursionError):
            pass
    return good / len(lines) >= SNIFF_SHARE


def load_json(path: Path, max_bytes: int | None, notes: ParseNotes) -> Any:
    """One JSON value, whole (R-LIM-01). `json.JSONDecodeError` is left to the caller (R-DET-02)."""
    check_size(path, max_bytes, "JSON file", "convert it to JSON Lines (one record per line)")
    text = decode_bytes(path.read_bytes())
    return Decoder(notes).decode(text)


def iter_jsonl(path: Path, notes: ParseNotes, after: int = 0) -> Iterator[tuple[int, Any, int]]:
    """(line number, value, bytes) for each JSON line after line `after`; bad lines counted
    (R-LIM-05)."""
    decoder = Decoder(notes)
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, start=1):
            if number <= after:
                continue
            notes.last_line = number
            if not raw.strip():
                continue
            notes.lines += 1
            try:
                line = raw.decode("utf-8-sig" if number == 1 else "utf-8")
                yield number, decoder.decode(line), len(raw)
            except (ValueError, UnicodeDecodeError):  # JSONDecodeError is a ValueError
                notes.skipped += 1
                if len(notes.skipped_lines) < SKIPPED_LISTED:
                    notes.skipped_lines.append(number)


def describe_json_error(error: json.JSONDecodeError) -> str:
    return f"This is not valid JSON: {error.msg} (line {error.lineno}, column {error.colno})."
