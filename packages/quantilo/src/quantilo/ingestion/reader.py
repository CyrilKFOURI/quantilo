"""Turn one or more source files into a single DuckDB query."""

import threading
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from quantilo.ingestion.csv_sniff import readable_path, sniff_dialect
from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.options import CsvOptions, SourceOptions
from quantilo.ingestion.sources import SourceFile
from quantilo.ingestion.sql import ident, lit
from quantilo.ingestion.workbook import load_workbook

FULL_SNIFF_MAX_BYTES = 100 * 1024 * 1024
NPY_BLOCK_ROWS = 100_000
NPY_SAMPLE_SEED = 42


def _column(values: list[object]) -> pa.Array:
    """Typed when the cells agree (numbers, dates, booleans), text when they are mixed."""
    try:
        return pa.array(values)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return pa.array([None if value is None else str(value) for value in values], pa.string())


def read_sheet(path: Path, sheet: str, header_row: int | None = None) -> pa.Table:
    """A sheet as a table: the first non-empty row names the columns, or row `header_row` (1-based)
    with the rows above it skipped (title, notes: feature 080); fully empty rows are skipped."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = [
            row
            for row in workbook[sheet].iter_rows(min_row=header_row or 1, values_only=True)
            if any(c is not None for c in row)
        ]
    finally:
        workbook.close()
    if not rows:
        return pa.table({})
    width = max(len(row) for row in rows)
    names: list[str] = []
    for index in range(width):
        header = rows[0][index] if index < len(rows[0]) else None
        name = (
            str(header).strip()
            if header is not None and str(header).strip()
            else f"column_{index + 1}"
        )
        while name in names:  # duplicated header: keep both columns
            name = f"{name}_{index + 1}"
        names.append(name)
    body = [list(row) + [None] * (width - len(row)) for row in rows[1:]]
    return pa.table({name: _column([row[i] for row in body]) for i, name in enumerate(names)})


def load_npy(path: Path) -> np.ndarray:
    """Memory-mapped, so nothing is read until rows are taken; never unpickled."""
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError as error:
        raise IngestionError(
            f"{path.name} holds Python objects: save it with plain dtypes."
        ) from error
    if array.dtype.names is None and array.ndim not in (1, 2):
        raise IngestionError(
            f"{path.name} has {array.ndim} dimensions: only 1-D or 2-D arrays are tables."
        )
    return array


def npy_names(array: np.ndarray) -> list[str]:
    """A structured array keeps its field names, a 2-D array gives column_1..column_n,
    a 1-D array one "value" column."""
    if array.dtype.names:
        return list(array.dtype.names)
    if array.ndim == 1:
        return ["value"]
    return [f"column_{i + 1}" for i in range(array.shape[1])]


def read_npy(
    path: Path,
    sample_pct: float = 100,
    limit: int | None = None,
    columns: list[str] | None = None,
    stop: threading.Event | None = None,
) -> pa.Table:
    """Only the rows and columns asked for are read, block by block, so a sample of a file
    bigger than memory stays cheap."""
    array = load_npy(path)
    names = npy_names(array)
    keep = [(i, name) for i, name in enumerate(names) if columns is None or name in columns]
    total = array.shape[0]
    rows: np.ndarray | None = None
    if limit is not None:
        rows = np.arange(min(limit, total))
    elif sample_pct < 100:
        size = max(1, round(total * sample_pct / 100))
        rng = np.random.default_rng(NPY_SAMPLE_SEED)
        rows = np.sort(rng.choice(total, size=size, replace=False))
    count = total if rows is None else len(rows)

    def block_columns(block: np.ndarray) -> dict[str, np.ndarray]:
        if block.dtype.names:
            return {name: np.ascontiguousarray(block[name]) for _, name in keep}
        if block.ndim == 1:
            return {"value": np.ascontiguousarray(block)}
        return {name: np.ascontiguousarray(block[:, i]) for i, name in keep}

    batches = []
    for start in range(0, count, NPY_BLOCK_ROWS):
        if stop is not None and stop.is_set():
            raise IngestionError("Import cancelled.")
        block = (
            array[start : start + NPY_BLOCK_ROWS]
            if rows is None
            else array[rows[start : start + NPY_BLOCK_ROWS]]
        )
        batches.append(pa.record_batch(block_columns(block)))
    if not batches:
        return pa.table(block_columns(array[:0]))
    return pa.Table.from_batches(batches)


def excel_sheets(path: Path) -> list[str]:
    workbook = load_workbook(path, read_only=True)
    try:
        return list(workbook.sheetnames)
    finally:
        workbook.close()


class SourceReader:
    """Registers what it needs in a DuckDB connection and returns SQL selecting the data."""

    def __init__(self, connection: duckdb.DuckDBPyConnection, options: SourceOptions) -> None:
        self.connection = connection
        self.options = options
        self.excel_rows: dict[Path, int] = {}
        self.csv_options: dict[Path, CsvOptions] = {}
        self.csv_notes: dict[Path, list[str]] = {}  # rules that chose a detected dialect (092)
        # A preview reads only the first rows of a .npy: the whole array may not fit in memory.
        self.npy_limit: int | None = None
        self.stop: threading.Event | None = None
        self.log_notes: dict[Path, list[str]] = {}  # how each log file was read (090)
        self.copies: list[Path] = []  # UTF-8 copies of Windows-1252 files read
        self._views = 0

    def csv_options_for(self, file: SourceFile) -> CsvOptions:
        if file.path not in self.csv_options:
            if self.options.csv is not None:
                self.csv_options[file.path] = self.options.csv
            else:
                options, notes = sniff_dialect(file.path)
                self.csv_options[file.path], self.csv_notes[file.path] = options, notes
        return self.csv_options[file.path]

    def csv_sql(self, file: SourceFile) -> str:
        options = self.csv_options_for(file)
        sample_size = -1 if file.size <= FULL_SNIFF_MAX_BYTES else 200_000
        path, encoding = readable_path(file.path, options.encoding)
        if path != file.path:
            self.copies.append(path)
        return (
            f"SELECT * FROM read_csv({lit(str(path))}, delim={lit(options.delimiter)}, "
            f"header={str(options.header).lower()}, decimal_separator={lit(options.decimal)}, "
            f"encoding={lit(encoding)}, sample_size={sample_size})"
        )

    def register(self, prefix: str, table: pa.Table) -> str:
        self._views += 1
        view = f"{prefix}_{self._views}"
        self.connection.register(view, table)
        return f"SELECT * FROM {view}"

    def stack_sql(self, file: SourceFile, table_ids: list[str]) -> str:
        """S1 (C1): tables of a workbook stacked, columns matched by name, each row with the sheet
        it comes from (`source_sheet`, as for several sheets read whole)."""
        parts = []
        rows = 0
        for table_id in table_ids:
            sql = self.table_sql(file, table_id)
            rows += self.excel_rows[file.path]
            sheet = table_id.rpartition("!")[0] or table_id
            parts.append(f"SELECT *, {lit(sheet)} AS source_sheet FROM ({sql})")
        self.excel_rows[file.path] = rows
        return " UNION ALL BY NAME ".join(parts)

    def table_sql(self, file: SourceFile, table_id: str) -> str:
        """080b: one detected table of a sheet, read with its plan (or a range typed by hand)."""
        from quantilo.ingestion.layout import find_table
        from quantilo.ingestion.layout.cells import user_null_tokens
        from quantilo.ingestion.layout.read import read_table

        try:
            table = find_table(file.path, table_id)
        except (KeyError, ValueError) as error:
            raise IngestionError(str(error).strip("'\"")) from error
        with user_null_tokens(self.options.null_tokens, self.options.text_tokens):  # T2u
            data = read_table(file.path, table, reshape=bool(self.options.reshape))
        self.excel_rows[file.path] = data.num_rows
        return self.register("excel_table", data)

    def document_sql(self, file: SourceFile) -> str:
        """085: one linked table of a JSON, JSON Lines or XML document (083)."""
        from quantilo.ingestion.documents import main_table, read_document

        layout = read_document(file.path, self.options.records_path, bool(self.options.one_table))
        name = self.options.table or main_table(layout)
        try:
            data = layout.arrow(name)
        except KeyError as error:
            raise IngestionError(f"No table {name!r} in {file.path.name}.") from error
        self.excel_rows[file.path] = data.num_rows
        return self.register("document_table", data)

    def excel_sql(self, file: SourceFile, sheets: list[str] | None) -> str:
        if self.options.stack:
            return self.stack_sql(file, self.options.stack)
        if self.options.table:
            return self.table_sql(file, self.options.table)
        names = sheets or excel_sheets(file.path)[:1]
        parts = []
        self.excel_rows[file.path] = 0
        for sheet in names:
            table = read_sheet(file.path, sheet, (self.options.header_row or {}).get(sheet))
            self._views += 1
            view = f"excel_sheet_{self._views}"
            self.connection.register(view, table)
            self.excel_rows[file.path] += table.num_rows
            sheet_column = f", {lit(sheet)} AS source_sheet" if len(names) > 1 else ""
            parts.append(f"SELECT *{sheet_column} FROM {view}")
        return " UNION ALL BY NAME ".join(parts)

    def file_sql(self, file: SourceFile, sheets: list[str] | None = None) -> str:
        if file.format == "csv":
            return self.csv_sql(file)
        if file.format == "parquet":
            return f"SELECT * FROM read_parquet({lit(str(file.path))})"
        if file.format == "log":
            from quantilo.ingestion.logs import read_log

            self._views += 1
            view = f"log_file_{self._views}"
            table, reading = read_log(
                file.path, limit=self.npy_limit, pattern=self.options.log_pattern
            )
            self.log_notes[file.path] = reading.notes
            self.connection.register(view, table)
            return f"SELECT * FROM {view}"
        if file.format in ("json", "jsonl", "xml"):
            return self.document_sql(file)
        if file.format == "npy":
            self._views += 1
            view = f"npy_array_{self._views}"
            table = read_npy(
                file.path,
                sample_pct=getattr(self.options, "sample_pct", 100),
                limit=self.npy_limit,
                columns=getattr(self.options, "columns", None),
                stop=self.stop,
            )
            self.connection.register(view, table)
            return f"SELECT * FROM {view}"
        return self.excel_sql(file, sheets)

    def columns(self, sql: str) -> list[tuple[str, str]]:
        """(name, DuckDB type) of each column of a query."""
        rows = self.connection.execute(f"DESCRIBE ({sql})").fetchall()
        return [(row[0], row[1]) for row in rows]

    def combined_sql(self, files: list[SourceFile]) -> str:
        if len(files) == 1:
            return self.file_sql(files[0], self.options.sheets)
        parts = [
            f"SELECT *, {lit(file.path.name)} AS source_file FROM ({self.file_sql(file)})"
            for file in files
        ]
        if self.options.folder_mode == "union":
            return " UNION ALL BY NAME ".join(parts)
        part_columns = [[name for name, _ in self.columns(part)] for part in parts]
        common = [name for name in part_columns[0] if all(name in cols for cols in part_columns)]
        selected = ", ".join(ident(name) for name in common)
        return " UNION ALL ".join(f"SELECT {selected} FROM ({part})" for part in parts)

    def schema_mismatch(self, files: list[SourceFile]) -> bool:
        if len(files) < 2:
            return False
        names = [{name for name, _ in self.columns(self.file_sql(file))} for file in files]
        return any(other != names[0] for other in names[1:])
