"""What the import changes in the values before storing them, and the notes that say so (092).

R-ING-01: a timestamp with a time zone is stored in UTC. DuckDB hands a `TIMESTAMP WITH TIME
ZONE` to Python through `pytz`, which is not installed (and no dependency is added): converted
before anything leaves DuckDB, the column never reaches that path. The notes travel with the
Parquet copy (key-value metadata) so the Columns page can repeat them in the column's reasons.
"""

import json

import duckdb

from quantilo.ingestion.sql import ident, lit

NOTES_KEY = "quantilo_notes"
ZONED_TYPES = ("TIMESTAMP WITH TIME ZONE", "TIMESTAMPTZ")
UTC_NOTE = "times with a time zone (such as +02:00 or Z) read in UTC (R-ING-01)"

Notes = dict[str, list[str]]


def zoned_columns(columns: list[tuple[str, str]]) -> list[str]:
    return [name for name, dtype in columns if dtype.upper().startswith(ZONED_TYPES)]


def normalized_sql(connection: duckdb.DuckDBPyConnection, sql: str) -> tuple[str, Notes]:
    """The query with its zoned timestamps in UTC, and a note per changed column."""
    columns = [(row[0], row[1]) for row in connection.execute(f"DESCRIBE ({sql})").fetchall()]
    zoned = zoned_columns(columns)
    if not zoned:
        return sql, {}
    replaced = ", ".join(f"timezone('UTC', {ident(name)}) AS {ident(name)}" for name in zoned)
    return f"SELECT * REPLACE ({replaced}) FROM ({sql})", {name: [UTC_NOTE] for name in zoned}


def note_lines(notes: Notes) -> list[str]:
    """One sentence per changed column, for the import preview."""
    return [f"{name}: {note}." for name, column_notes in notes.items() for note in column_notes]


def copy_metadata(notes: Notes) -> str:
    """The COPY option writing the notes into the Parquet file ("" without notes)."""
    if not notes:
        return ""
    return f", KV_METADATA {{{NOTES_KEY}: {lit(json.dumps(notes))}}}"


def stored_notes(connection: duckdb.DuckDBPyConnection, parquet_path: str) -> Notes:
    """The notes an import wrote into a Parquet file; {} when there are none."""
    try:
        rows = connection.execute(
            f"SELECT key, value FROM parquet_kv_metadata({lit(parquet_path)})"
        ).fetchall()
    except duckdb.Error:
        return {}
    for key, value in rows:
        if bytes(key).decode("utf-8", "replace") == NOTES_KEY:
            try:
                notes = json.loads(bytes(value).decode("utf-8"))
            except ValueError:
                return {}
            return {str(name): [str(note) for note in items] for name, items in notes.items()}
    return {}
