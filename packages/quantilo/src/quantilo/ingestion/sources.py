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


def detect_format(path: Path) -> Format | None:
    return EXTENSIONS.get(path.suffix.lower())


def source_file(path: Path) -> SourceFile:
    file_format = detect_format(path)
    if file_format is None:
        if path.suffix.lower() == ".xls":
            raise IngestionError("Legacy .xls files are not supported: save it as .xlsx first.")
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
    files = [
        source_file(child)
        for child in sorted(path.iterdir(), key=lambda child: child.name.lower())
        if child.is_file() and detect_format(child) and child.stat().st_size > 0
    ]
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
