"""Guess how a CSV file is written: encoding, delimiter, header, decimal separator."""

import codecs
import hashlib
import re
import tempfile
from pathlib import Path

import duckdb

from quantilo.ingestion.options import CsvOptions, Encoding
from quantilo.ingestion.sql import lit

SAMPLE_BYTES = 256 * 1024
DECIMAL_COMMA = re.compile(r"(?<![\d.,])\d+,\d+(?![\d.,])")
DECIMAL_DOT = re.compile(r"(?<![\d.,])\d+\.\d+(?![\d.,])")
C1_BYTES = re.compile(rb"[\x80-\x9f]")


def detect_encoding(path: Path) -> Encoding:
    raw = path.read_bytes()[:SAMPLE_BYTES]
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    try:
        # Incremental decoding tolerates a multi-byte character cut by the sample boundary.
        codecs.getincrementaldecoder("utf-8")().decode(raw, final=False)
        return "utf-8"
    except UnicodeDecodeError:
        # Bytes 0x80-0x9F are control codes in latin-1 but curly quotes, dashes and € in the
        # Windows code page Excel exports with: DuckDB's latin-1 reader refuses them.
        return "windows-1252" if C1_BYTES.search(raw) else "latin-1"


_copies_dir: Path | None = None


def private_dir() -> Path:
    """This process's own temp folder (mkdtemp: created new, readable by this user only), so
    no other account can plant or read a copy."""
    global _copies_dir
    if _copies_dir is None or not _copies_dir.is_dir():
        _copies_dir = Path(tempfile.mkdtemp(prefix="quantilo-"))
    return _copies_dir


def utf8_copy(path: Path) -> Path:
    """A UTF-8 copy of a Windows-1252 file, which DuckDB only reads through an extension it
    downloads (the app stays offline). Kept while the source is unchanged, so a preview and
    the import that follows convert it once; the import deletes it."""
    stat = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
    folder = private_dir()
    target = folder / f"{key[:16]}.csv"
    if target.exists():
        return target
    decoder = codecs.getincrementaldecoder("cp1252")(errors="replace")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=folder, suffix=".partial", delete=False
    ) as copy:
        try:
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    copy.write(decoder.decode(chunk))
                copy.write(decoder.decode(b"", final=True))
        except BaseException:
            copy.close()
            Path(copy.name).unlink(missing_ok=True)
            raise
    Path(copy.name).replace(target)
    return target


def readable_path(path: Path, encoding: Encoding) -> tuple[Path, Encoding]:
    """The file and encoding DuckDB reads: the UTF-8 copy for Windows-1252."""
    return (utf8_copy(path), "utf-8") if encoding == "windows-1252" else (path, encoding)


def detect_decimal(path: Path, encoding: Encoding, delimiter: str) -> str:
    if delimiter == ",":
        return "."
    text = path.read_bytes()[:SAMPLE_BYTES].decode(encoding, errors="ignore")
    commas = len(DECIMAL_COMMA.findall(text))
    dots = len(DECIMAL_DOT.findall(text))
    return "," if commas > dots else "."


SPACE_LINES = 50  # R-ING-02: lines checked for a constant number of space-separated fields


def space_fields(path: Path, encoding: Encoding) -> int | None:
    """R-ING-02 (092): the number of fields when the first lines all split into the same number
    (at least 2) of non-empty fields on single spaces, as in UCI's `german.data`."""
    raw = path.read_bytes()[:SAMPLE_BYTES]
    text = raw.decode("cp1252" if encoding == "windows-1252" else encoding, errors="replace")
    lines = text.splitlines()
    if len(raw) == SAMPLE_BYTES and len(lines) > 1:
        lines = lines[:-1]  # the last line may be cut by the sample
    lines = [line for line in lines if line.strip()][:SPACE_LINES]
    counts = {len(line.split(" ")) for line in lines}
    if len(lines) < 2 or len(counts) != 1 or any("" in line.split(" ") for line in lines):
        return None
    (count,) = counts
    return count if count >= 2 else None


def sniff_csv(path: Path) -> CsvOptions:
    return sniff_dialect(path)[0]


def sniff_dialect(path: Path) -> tuple[CsvOptions, list[str]]:
    """The dialect, and a sentence per rule that chose it (092: R-ING-02, R-ING-03)."""
    encoding = detect_encoding(path)
    readable, read_as = readable_path(path, encoding)
    notes: list[str] = []
    connection = duckdb.connect()
    try:
        source = f"sniff_csv({lit(str(readable))}, encoding={lit(read_as)})"
        query = f"SELECT Delimiter, HasHeader, len(Columns) FROM {source}"
        delimiter, has_header, width = connection.execute(query).fetchone() or (",", True, 0)
        if width == 1 and (fields := space_fields(path, encoding)):
            spaced = f"sniff_csv({lit(str(readable))}, encoding={lit(read_as)}, delim=' ')"
            row = connection.execute(f"SELECT HasHeader FROM {spaced}").fetchone()
            delimiter, has_header = " ", bool(row[0]) if row else False
            notes.append(f"Columns separated by spaces ({fields} fields on every line) (R-ING-02).")
    finally:
        connection.close()
    delimiter = delimiter if len(delimiter) == 1 else ","
    if path.suffix.lower() == ".tsv":
        delimiter = "\t"
    if not has_header:
        notes.append(
            "No header row found: the first line holds values like the others, so the columns "
            "are named column0, column1… (R-ING-03)."
        )
    options = CsvOptions(
        delimiter=delimiter,
        encoding=encoding,
        header=bool(has_header),
        decimal=detect_decimal(path, encoding, delimiter),  # type: ignore[arg-type]
    )
    return options, notes
