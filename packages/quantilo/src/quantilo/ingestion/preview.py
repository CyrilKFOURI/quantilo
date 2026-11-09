"""Quick look at a source before importing it."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import duckdb

from quantilo.ingestion.errors import readable_errors
from quantilo.ingestion.options import CsvOptions, Format, SourceOptions
from quantilo.ingestion.reader import SourceReader, excel_sheets, load_npy
from quantilo.ingestion.sources import SourceFile, list_source_files
from quantilo.ingestion.sql import lit

PREVIEW_ROWS = 20
EXACT_COUNT_MAX_BYTES = 50 * 1024 * 1024


@dataclass
class ColumnInfo:
    name: str
    dtype: str


@dataclass
class FileInfo:
    name: str
    format: Format
    size: int


@dataclass
class Preview:
    kind: Literal["file", "folder"]
    format: Format | Literal["mixed"]
    files: list[FileInfo]
    columns: list[ColumnInfo]
    rows: list[list[str | None]]
    row_estimate: int | None
    csv: CsvOptions | None = None
    sheets: list[str] | None = None
    schema_mismatch: bool = False
    suggested_name: str = ""
    warnings: list[str] = field(default_factory=list)


def estimate_rows(reader: SourceReader, file: SourceFile) -> int:
    connection = reader.connection
    if file.format == "parquet":
        row = connection.execute(
            f"SELECT SUM(num_rows) FROM parquet_file_metadata({lit(str(file.path))})"
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    if file.format == "excel":
        return reader.excel_rows.get(file.path, 0)
    if file.format == "npy":
        return int(load_npy(file.path).shape[0])
    if file.size <= EXACT_COUNT_MAX_BYTES:
        row = connection.execute(f"SELECT count(*) FROM ({reader.csv_sql(file)})").fetchone()
        return int(row[0]) if row else 0
    with file.path.open("rb") as handle:
        sample = handle.read(1024 * 1024)
    lines = max(sample.count(b"\n"), 1)
    return int(file.size / len(sample) * lines)


def preview_source(path: Path, options: SourceOptions) -> Preview:
    with readable_errors():
        all_files = list_source_files(path)
        files = list_source_files(path, options.files) if options.files else all_files
        connection = duckdb.connect()
        try:
            reader = SourceReader(connection, options)
            reader.npy_limit = PREVIEW_ROWS
            sql = reader.combined_sql(files)
            columns = [ColumnInfo(name, dtype) for name, dtype in reader.columns(sql)]
            rows = connection.execute(
                f"SELECT COLUMNS(*)::VARCHAR FROM ({sql}) LIMIT {PREVIEW_ROWS}"
            ).fetchall()
            first = files[0]
            source_format: Format | Literal["mixed"] = (
                first.format if len({file.format for file in files}) == 1 else "mixed"
            )
            return Preview(
                kind="folder" if path.is_dir() else "file",
                format=source_format,
                files=[FileInfo(f.path.name, f.format, f.size) for f in all_files],
                columns=columns,
                rows=[list(row) for row in rows],
                row_estimate=sum(estimate_rows(reader, file) for file in files),
                csv=reader.csv_options_for(first) if first.format == "csv" else None,
                sheets=excel_sheets(first.path) if first.format == "excel" else None,
                schema_mismatch=reader.schema_mismatch(files),
                suggested_name=path.stem if path.is_file() else path.name,
            )
        finally:
            connection.close()
