"""Resolve a path into the list of readable files it designates, and fingerprint them."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.options import Format

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


@dataclass(frozen=True)
class SourceFile:
    path: Path
    format: Format
    size: int
    mtime: float


TEXT_EXTENSIONS = (".txt", ".dat", ".data")
OLE2_SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")


def detect_format(path: Path) -> Format | None:
    return EXTENSIONS.get(path.suffix.lower())


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


def source_file(path: Path) -> SourceFile:
    file_format = detect_format(path)
    if file_format is None:
        if path.suffix.lower() == ".xls" and path.is_file():
            raise IngestionError(legacy_excel_message(path))
        raise IngestionError(f"Unsupported file type: {path.name}")
    stat = path.stat()
    if stat.st_size == 0:
        raise IngestionError(f"The file {path.name} is empty.")
    return SourceFile(path=path, format=file_format, size=stat.st_size, mtime=stat.st_mtime)


def list_source_files(path: Path, only: list[str] | None = None) -> list[SourceFile]:
    """A file gives itself; a folder gives its supported files (not recursive), sorted by name."""
    if not path.exists():
        raise IngestionError(f"Path not found: {path}")
    if path.is_file():
        return [source_file(path)]
    children = [
        child
        for child in sorted(path.iterdir(), key=lambda child: child.name.lower())
        if child.is_file() and detect_format(child) and child.stat().st_size > 0
    ]
    # R-ING-04: next to CSV files, a .txt is a readme, not a table
    if any(child.suffix.lower() in (".csv", ".tsv") for child in children):
        children = [child for child in children if child.suffix.lower() not in TEXT_EXTENSIONS]
    files = [source_file(child) for child in children]
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
