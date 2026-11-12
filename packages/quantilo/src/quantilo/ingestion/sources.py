"""Resolve a path into the list of readable files it designates, and fingerprint them."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.options import Format
from quantilo.ingestion.sniff import Sniff, sniff_path

EXTENSIONS: dict[str, Format] = {
    ".csv": "csv",
    ".tsv": "csv",
    # R-ING-04 (092): delimited text under another name; the dialect detection decides
    ".txt": "csv",
    ".dat": "csv",
    ".data": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".npy": "npy",
}
# 082/085: a single document file is read as linked tables (083); never in a folder union.
DOCUMENT_EXTENSIONS: dict[str, Format] = {
    ".json": "json",
    ".geojson": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".xml": "xml",
}
DOCUMENT_FORMATS = ("json", "jsonl", "xml")


@dataclass(frozen=True)
class SourceFile:
    path: Path
    format: Format
    size: int
    mtime: float
    note: str | None = None  # R-DET-01: the name and the content disagree, the content won


TEXT_EXTENSIONS = (".txt", ".dat", ".data")
OLE2_SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")


def detect_format(path: Path) -> Format | None:
    """The format the name says (a table, or a document); the content is checked by `sniff`."""
    suffix = path.suffix.lower()
    return EXTENSIONS.get(suffix) or DOCUMENT_EXTENSIONS.get(suffix)


def legacy_excel_message(path: Path) -> str:
    """R-ING-05 (092): what a .xls file really is, told by its first bytes (no xlrd)."""
    with path.open("rb") as handle:
        head = handle.read(8)
    if head.startswith(OLE2_SIGNATURE):
        return (
            "This is a legacy Excel 97-2003 file (.xls), which Quantilo does not read: open it in "
            "Excel or LibreOffice and save it as .xlsx (File > Save As), then import the .xlsx."
        )
    if head.startswith(b"PK"):
        return (
            "This .xls file is in fact a newer Excel workbook (.xlsx) that was renamed: rename it "
            "to .xlsx and import it again."
        )
    return (
        "This .xls file is plain text (an export from a website or an application), not an "
        "Excel workbook: rename it to .csv (or save it as .xlsx) and import it again."
    )


def refusal(path: Path, found: Sniff) -> str:
    """R-DET-03: what a file that is not read as a table is, by name."""
    if path.suffix.lower() == ".xls" and found.format in ("xls", "zip", "csv"):
        return legacy_excel_message(path)  # R-ING-05 (092): what the .xls really is
    if found.kind == "image":
        return (
            f"{found.label} detected in {path.name}: images are not tables. They open on "
            "their own page (coming later)."
        )
    if found.kind == "document":
        return f"{path.name} is a {found.label} document: import it on its own, not in a folder."
    return f"{path.name}: {found.reason}"


def source_file(path: Path, documents: bool = True) -> SourceFile:
    """A file to read, its format told by its first bytes (082, R-DET-01): the content wins over
    the name, and the difference is kept in `note`. `documents=False` (a folder) refuses JSON."""
    stat = path.stat()
    if stat.st_size == 0:
        raise IngestionError(f"The file {path.name} is empty.")
    found = sniff_path(path)
    if found.format == "csv" and found.name_format is None:
        # plain text under an unknown name (.md, .log): not taken for a table without a sign
        raise IngestionError(f"Unsupported file type: {path.name}")
    readable = found.kind == "table" or (found.kind == "document" and documents)
    if not readable:
        raise IngestionError(refusal(path, found))
    file_format: Format = found.format  # type: ignore[assignment]  # a table or document format
    note = found.note(path.name)
    if found.format == "csv" and found.name_format in ("excel", "parquet", "npy"):
        # Text is no signature: it never overrides a binary format's name. The reader of that
        # format says what is wrong (a damaged workbook), and a text .xls is explained (R-ING-05).
        if path.suffix.lower() == ".xls":
            raise IngestionError(legacy_excel_message(path))
        file_format, note = found.name_format, None  # type: ignore[assignment]
    return SourceFile(
        path=path,
        format=file_format,
        size=stat.st_size,
        mtime=stat.st_mtime,
        note=note,
    )


def list_source_files(path: Path, only: list[str] | None = None) -> list[SourceFile]:
    """A file gives itself; a folder gives its supported files (not recursive), sorted by name."""
    if not path.exists():
        raise IngestionError(f"Path not found: {path}")
    if path.is_file():
        return [source_file(path)]
    children = [
        child
        for child in sorted(path.iterdir(), key=lambda child: child.name.lower())
        if child.is_file() and child.suffix.lower() in EXTENSIONS and child.stat().st_size > 0
    ]
    # R-ING-04: next to CSV files, a .txt is a readme, not a table
    if any(child.suffix.lower() in (".csv", ".tsv") for child in children):
        children = [child for child in children if child.suffix.lower() not in TEXT_EXTENSIONS]
    files = [source_file(child, documents=False) for child in children]
    if only is not None:
        files = [file for file in files if file.path.name in only]
    if not files:
        raise IngestionError("No CSV, Parquet, Excel or .npy file found in this folder.")
    return files


def fingerprint(files: list[SourceFile], options: dict[str, object]) -> str:
    """Identifies a source in a given state, read with given options."""
    payload = {
        "files": [[str(file.path), file.size, file.mtime] for file in files],
        "options": options,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
