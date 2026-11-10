"""Verbose CSV files (V rules): a preamble, a block of column metadata, a combined first column.

The file is read as raw fields with the standard `csv` module (no DuckDB sniffing here: a preamble
is exactly what trips a sniffer). V1 counts raw fields, not filled ones: a clean file whose last
columns are empty keeps its header on row 1 (StatCan control file).
"""

import codecs
import csv
import io
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from quantilo.ingestion.csv_sniff import detect_encoding
from quantilo.ingestion.layout.cells import clean_text
from quantilo.ingestion.layout.grid import LIMITS, Grid, Limits

SAMPLE_BYTES = 4 * 1024 * 1024
DELIMITERS = (",", ";", "\t", "|")
_META_KEY = re.compile(r"^[^:]{1,40}:\s*$")


@dataclass
class CsvShape:
    delimiter: str
    encoding: str
    widths: list[int]  # raw fields per row
    modal: int
    preamble: list[int]  # V1: 0-based rows narrower than the modal width, at the top
    metadata: list[int]  # V2: 0-based rows whose first cell is a "Key:" label
    combined: list[str] | None  # V3: names of the combined first column


def _decode(path: Path, limit: int) -> tuple[str, str]:
    encoding = detect_encoding(path)
    raw = path.read_bytes()[:limit]
    codec = "cp1252" if encoding == "windows-1252" else encoding
    decoder = codecs.getincrementaldecoder("utf-8-sig" if codec == "utf-8" else codec)("replace")
    return decoder.decode(raw, final=True), encoding


def _delimiter(text: str, path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    lines = [line for line in text.splitlines()[:200] if line.strip()]
    best, best_score = ",", -1.0
    for delimiter in DELIMITERS:
        counts = Counter(line.count(delimiter) for line in lines)
        count, rows = counts.most_common(1)[0] if counts else (0, 0)
        score = rows + count / 1000 if count else 0
        if score > best_score:
            best, best_score = delimiter, score
    return best


def read_csv_rows(
    path: Path, delimiter: str | None = None, limit: int | None = None
) -> tuple[list[list[str]], str, str]:
    """Raw fields: the first `limit` rows of a sample, or every row when `limit` is None."""
    text, encoding = _decode(path, SAMPLE_BYTES if limit is not None else 1 << 62)
    delimiter = delimiter or _delimiter(text, path)
    rows = []
    for index, row in enumerate(csv.reader(io.StringIO(text), delimiter=delimiter)):
        if limit is not None and index >= limit:
            break
        rows.append(row)
    return rows, delimiter, encoding


def csv_shape(rows: list[list[str]], delimiter: str, encoding: str) -> CsvShape:
    widths = [len(row) if any(cell.strip() for cell in row) else 0 for row in rows]
    counted = Counter(w for w in widths if w > 1)
    modal = counted.most_common(1)[0][0] if counted else max(widths, default=0)
    preamble = []
    for r, width in enumerate(widths):
        if width >= modal:
            break
        preamble.append(r)
    metadata = [
        r
        for r, row in enumerate(rows[:15])
        if r not in preamble and row and _META_KEY.match(clean_text(row[0]))
    ]
    if len(metadata) < 2:
        metadata = []
    combined = None
    start = len(preamble)
    if start < len(rows) and rows[start]:
        first = clean_text(rows[start][0])
        if "\\" in first and "," in first.split("\\")[0]:
            combined = [part.strip() for part in first.split("\\")[0].split(",")]
    return CsvShape(delimiter, encoding, widths, modal, preamble, metadata, combined)


def load_csv_grid(path: Path, limits: Limits = LIMITS) -> tuple[Grid, CsvShape]:
    rows, delimiter, encoding = read_csv_rows(path, limit=limits.max_rows + 1)
    truncated = len(rows) > limits.max_rows
    rows = rows[: limits.max_rows]
    shape = csv_shape(rows, delimiter, encoding)
    grid = Grid.from_rows(path.stem, [list[object](row[: limits.max_cols]) for row in rows])
    grid.truncated = truncated
    return grid, shape


def stream_csv_end(path: Path, delimiter: str, encoding: str, start_row: int) -> int:
    """B1 for a CSV: from `start_row` (1-based) on, the body runs to the first empty row."""
    codec = {"windows-1252": "cp1252", "utf-8": "utf-8-sig"}.get(encoding, encoding)
    last = start_row - 1
    with path.open(encoding=codec, errors="replace", newline="") as handle:
        for number, row in enumerate(csv.reader(handle, delimiter=delimiter), start=1):
            if number < start_row:
                continue
            if not any(cell.strip() for cell in row):
                break
            last = number
    return last
