"""Read a source and write it as one normalized Parquet file."""

import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import duckdb

from quantilo.ingestion.errors import IngestionError, readable_errors
from quantilo.ingestion.normalize import copy_metadata, normalized_sql
from quantilo.ingestion.options import ImportOptions
from quantilo.ingestion.reader import SourceReader
from quantilo.ingestion.sources import SourceFile
from quantilo.ingestion.sql import ident, lit

SAMPLE_SEED = 42
# In the browser (Pyodide) there are no threads: an import cannot be interrupted midway.
CAN_WATCH = sys.platform != "emscripten"

Progress = Callable[[float, str], None]


class ImportCancelled(IngestionError):
    def __init__(self) -> None:
        super().__init__("Import cancelled.")


@dataclass(frozen=True)
class ImportResult:
    row_count: int
    col_count: int
    size_bytes: int


def build_select(reader: SourceReader, files: list[SourceFile], options: ImportOptions) -> str:
    sql = reader.combined_sql(files)
    if options.columns is not None:
        available = {name for name, _ in reader.columns(sql)}
        missing = [name for name in options.columns if name not in available]
        if missing:
            raise IngestionError(f"Unknown columns: {', '.join(missing)}")
        if not options.columns:
            raise IngestionError("Select at least one column to import.")
        selected = ", ".join(ident(name) for name in options.columns)
    else:
        selected = "*"
    sample = ""
    # A .npy is already sampled while it is read (reader.read_npy).
    if options.sample_pct < 100 and any(file.format != "npy" for file in files):
        sample = f" USING SAMPLE {options.sample_pct} PERCENT (bernoulli, {SAMPLE_SEED})"
    return f"SELECT {selected} FROM ({sql}){sample}"


def import_to_parquet(
    files: list[SourceFile],
    options: ImportOptions,
    target: Path,
    progress: Progress = lambda _fraction, _message: None,
    stop: threading.Event | None = None,
) -> ImportResult:
    """Setting `stop` interrupts the import from another thread: ImportCancelled is raised and
    nothing is written."""
    stop = stop or threading.Event()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".parquet.partial")
    finished = threading.Event()
    with readable_errors():
        connection = duckdb.connect()
        watcher = threading.Thread(
            target=_interrupt_on_stop, args=(connection, stop, finished), daemon=True
        )
        if CAN_WATCH:
            watcher.start()
        try:
            if options.sample_pct < 100:
                # Seeded sampling is only reproducible on a single thread.
                connection.execute("SET threads = 1")
            progress(0.1, "Reading source")
            reader = SourceReader(connection, options)
            reader.stop = stop
            try:
                sql, notes = normalized_sql(connection, build_select(reader, files, options))
                progress(0.3, "Writing Parquet")
                connection.execute(
                    f"COPY ({sql}) TO {lit(str(partial))} "
                    f"(FORMAT parquet, COMPRESSION zstd{copy_metadata(notes)})"
                )
            except (duckdb.Error, IngestionError):
                if stop.is_set():
                    raise ImportCancelled from None
                raise
            if stop.is_set():
                raise ImportCancelled
            progress(0.9, "Finalizing")
            partial.replace(target)
            for copy in reader.copies:
                copy.unlink(missing_ok=True)
            stored = f"read_parquet({lit(str(target))})"
            row = connection.execute(f"SELECT count(*) FROM {stored}").fetchone()
            col_count = len(connection.execute(f"DESCRIBE SELECT * FROM {stored}").fetchall())
        finally:
            finished.set()
            if CAN_WATCH:
                watcher.join()
            connection.close()
            partial.unlink(missing_ok=True)
    return ImportResult(
        row_count=int(row[0]) if row else 0,
        col_count=col_count,
        size_bytes=target.stat().st_size,
    )


def _interrupt_on_stop(
    connection: duckdb.DuckDBPyConnection, stop: threading.Event, finished: threading.Event
) -> None:
    while not finished.is_set():
        if stop.wait(0.2):
            connection.interrupt()
            return
