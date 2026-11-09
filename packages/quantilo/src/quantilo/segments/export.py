"""A dataset's rows with new columns (a segment, an unusual score): what "Save as dataset"
writes. The rows are the stored ones after the filters, untouched; the new columns join on the
row number."""

from pathlib import Path

import duckdb
import pandas as pd

from quantilo.filters.scope import Where, scoped_source
from quantilo.ingestion.sql import lit


def with_columns(storage_path: str, where: Where, extra: pd.DataFrame, target: Path) -> int:
    """Write the filtered rows plus `extra`'s columns (joined on its `row` column) to a Parquet
    file, in file order. Returns the rows written."""
    source = scoped_source(storage_path, where, numbered=True)
    connection = duckdb.connect()
    try:
        connection.register("extra_columns", extra)
        connection.execute(
            f"COPY (SELECT s.* EXCLUDE (file_row_number), e.* EXCLUDE (row) FROM {source} s "
            "JOIN extra_columns e ON e.row = s.file_row_number ORDER BY s.file_row_number) "
            f"TO {lit(target.as_posix())} (FORMAT parquet)"
        )
        (rows,) = connection.execute(
            f"SELECT count(*) FROM read_parquet({lit(target.as_posix())})"
        ).fetchone() or (0,)
    finally:
        connection.close()
    return int(rows)
