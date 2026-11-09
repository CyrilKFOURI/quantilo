"""Row by row: the two versions joined on an identifier column (DuckDB), rows added, removed and
modified, how many per column, examples, and every difference as CSV.

Needs both versions' rows: the previous Parquet copy is kept unless the user said "Never".
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from quantilo.catalog import ColumnView
from quantilo.changes.snapshot import Snapshot
from quantilo.errors import ExplorerError
from quantilo.ingestion.sql import ident, lit
from quantilo.typing.parsers import typed_sql
from quantilo.typing.types import NUMERIC_TYPES

EXAMPLES = 100
LISTED_KEYS = 20


class RowKeyError(ExplorerError):
    def __init__(self, message: str, code: str = "row_key") -> None:
        super().__init__(message, code=code, status_code=422)


@dataclass
class ColumnEdits:
    column: str
    modified: int  # rows whose value changed


@dataclass
class CellEdit:
    key: str
    column: str
    before: str | None
    after: str | None


@dataclass
class RowChanges:
    key: str
    rows_before: int
    rows_after: int
    added: int
    removed: int
    modified: int  # rows with at least one changed value
    unchanged: int
    columns: list[ColumnEdits]  # most modified first, only columns with changes
    examples: list[CellEdit]
    added_keys: list[str] = field(default_factory=list)
    removed_keys: list[str] = field(default_factory=list)
    compared: list[str] = field(default_factory=list)  # columns present in both versions


def key_candidates(before: Snapshot, after: Snapshot) -> list[str]:
    """Columns that identify a row in both versions (no missing value, all distinct):
    identifiers first, then names ending in id, then the others."""
    found: list[tuple[int, int, str]] = []
    for new in after.columns:
        old = before.column(new.name)
        if old is None or new.type in ("ignored", "text") or new.type == "numeric_continuous":
            continue
        unique = all(
            c.missing == 0 and c.distinct == rows and rows > 0
            for c, rows in ((old, before.rows), (new, after.rows))
        )
        if not unique:
            continue
        named = new.name.lower().endswith("id") or new.name.lower().endswith("_key")
        rank = 0 if new.type == "identifier" else 1 if named else 2
        found.append((rank, new.position, new.name))
    return [name for _rank, _position, name in sorted(found)]


def _value(view: ColumnView, numeric: bool) -> str:
    typed = typed_sql(view.name, view.parser)
    return f"CAST({typed} AS DOUBLE)" if numeric else f"CAST({typed} AS VARCHAR)"


def _relations(
    before_path: str,
    before_views: Sequence[ColumnView],
    after_path: str,
    after_views: Sequence[ColumnView],
    key: str,
) -> tuple[str, str, list[str]]:
    old = {v.name: v for v in before_views}
    new = {v.name: v for v in after_views}
    if key not in old or key not in new:
        raise RowKeyError(f"{key!r} is not in both versions: pick a column they share.")
    compared = [name for name in new if name in old and name != key]
    selects: dict[str, list[str]] = {"b": [], "a": []}
    for side, views in (("b", old), ("a", new)):
        selects[side].append(f"CAST({typed_sql(key, views[key].parser)} AS VARCHAR) AS k")
        for i, name in enumerate(compared):
            numeric = old[name].type in NUMERIC_TYPES and new[name].type in NUMERIC_TYPES
            selects[side].append(f"{_value(views[name], numeric)} AS c{i}")
    before_sql = f"(SELECT {', '.join(selects['b'])} FROM read_parquet({lit(before_path)}))"
    after_sql = f"(SELECT {', '.join(selects['a'])} FROM read_parquet({lit(after_path)}))"
    return before_sql, after_sql, compared


def _check_unique(connection: duckdb.DuckDBPyConnection, relation: str, key: str, which: str):
    row = connection.execute(
        f"SELECT count(*), count(DISTINCT k), count(k) FROM {relation}"
    ).fetchone() or (0, 0, 0)
    total, distinct, present = (int(v) for v in row)
    if present < total:
        raise RowKeyError(
            f"{key!r} has missing values in the {which} version: it can't identify rows."
        )
    if distinct < total:
        raise RowKeyError(
            f"{key!r} repeats in the {which} version ({total - distinct:,} duplicates): "
            "pick a column with one value per row."
        )
    return total


def row_changes(
    connection: duckdb.DuckDBPyConnection,
    before_path: str,
    before_views: Sequence[ColumnView],
    after_path: str,
    after_views: Sequence[ColumnView],
    key: str,
) -> RowChanges:
    b, a, compared = _relations(before_path, before_views, after_path, after_views, key)
    connection.execute(f"CREATE OR REPLACE TEMP VIEW change_before AS SELECT * FROM {b}")
    connection.execute(f"CREATE OR REPLACE TEMP VIEW change_after AS SELECT * FROM {a}")
    rows_before = _check_unique(connection, "change_before", key, "previous")
    rows_after = _check_unique(connection, "change_after", key, "new")

    differs = [f"a.c{i} IS DISTINCT FROM b.c{i}" for i in range(len(compared))]
    per_column = [f"count(*) FILTER (WHERE {d})" for d in differs]
    any_change = " OR ".join(differs) or "false"
    joined = "change_after a JOIN change_before b USING (k)"
    counts = (
        connection.execute(
            f"SELECT count(*), count(*) FILTER (WHERE {any_change}) "
            f"{''.join(', ' + p for p in per_column)} FROM {joined}"
        ).fetchone()
        or ()
    )
    common, modified, *by_column = (int(v or 0) for v in counts)
    edits = sorted(
        (ColumnEdits(name, n) for name, n in zip(compared, by_column, strict=True) if n),
        key=lambda e: (-e.modified, e.column),
    )

    examples: list[CellEdit] = []
    changed = [i for i, name in enumerate(compared) if name in {e.column for e in edits}]
    if changed:
        union = " UNION ALL ".join(
            f"(SELECT k, {i} AS c, CAST(b.c{i} AS VARCHAR) AS old, CAST(a.c{i} AS VARCHAR) AS new "
            f"FROM {joined} WHERE {differs[i]} ORDER BY k LIMIT {EXAMPLES})"
            for i in changed
        )
        rows = connection.execute(
            f"SELECT * FROM ({union}) ORDER BY k, c LIMIT {EXAMPLES}"
        ).fetchall()
        examples = [CellEdit(str(k), compared[c], old, new) for k, c, old, new in rows]

    def keys(sql: str) -> tuple[int, list[str]]:
        found = connection.execute(f"SELECT count(*) FROM ({sql})").fetchone()
        listed = connection.execute(f"SELECT k FROM ({sql}) ORDER BY k LIMIT {LISTED_KEYS}")
        return int(found[0]) if found else 0, [str(k) for (k,) in listed.fetchall()]

    added, added_keys = keys("SELECT k FROM change_after EXCEPT SELECT k FROM change_before")
    removed, removed_keys = keys("SELECT k FROM change_before EXCEPT SELECT k FROM change_after")
    return RowChanges(
        key=key,
        rows_before=rows_before,
        rows_after=rows_after,
        added=added,
        removed=removed,
        modified=modified,
        unchanged=common - modified,
        columns=edits,
        examples=examples,
        added_keys=added_keys,
        removed_keys=removed_keys,
        compared=compared,
    )


FORMULA_START = ("=", "+", "@", "\t", "\r")


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def cell(text: str) -> str:
    """A text a spreadsheet would run as a formula ("=HYPERLINK(...)", "-2+3") gets a leading
    quote; a negative number (a return, -12) stays a number."""
    risky = text.startswith(FORMULA_START) or (text.startswith("-") and not _is_number(text))
    return "'" + text if risky else text


def cell_sql(value: str) -> str:
    """`cell` in SQL, for the values DuckDB writes straight into the CSV."""
    risky = (
        f"(regexp_matches({value}, '^[=+@\\t\\r]') "
        f"OR (starts_with({value}, '-') AND TRY_CAST({value} AS DOUBLE) IS NULL))"
    )
    return f"CASE WHEN {risky} THEN '''' || {value} ELSE {value} END"


def write_differences(
    connection: duckdb.DuckDBPyConnection,
    before_path: str,
    before_views: Sequence[ColumnView],
    after_path: str,
    after_views: Sequence[ColumnView],
    key: str,
    target: Path,
) -> Path:
    """Every difference as CSV: change (modified / added / removed), the key, the column, the old
    and the new value. Added and removed rows take one line each, without a column."""
    b, a, compared = _relations(before_path, before_views, after_path, after_views, key)
    connection.execute(f"CREATE OR REPLACE TEMP VIEW change_before AS SELECT * FROM {b}")
    connection.execute(f"CREATE OR REPLACE TEMP VIEW change_after AS SELECT * FROM {a}")
    _check_unique(connection, "change_before", key, "previous")
    _check_unique(connection, "change_after", key, "new")
    joined = "change_after a JOIN change_before b USING (k)"
    k = cell_sql("CAST(k AS VARCHAR)")
    parts = [
        f"SELECT 'modified', {k}, {lit(cell(name))}, {cell_sql(f'CAST(b.c{i} AS VARCHAR)')}, "
        f"{cell_sql(f'CAST(a.c{i} AS VARCHAR)')} "
        f"FROM {joined} WHERE a.c{i} IS DISTINCT FROM b.c{i}"
        for i, name in enumerate(compared)
    ]
    parts.append(
        f"SELECT 'added', {k}, NULL, NULL, NULL FROM change_after "
        "WHERE k NOT IN (SELECT k FROM change_before)"
    )
    parts.append(
        f"SELECT 'removed', {k}, NULL, NULL, NULL FROM change_before "
        "WHERE k NOT IN (SELECT k FROM change_after)"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"COPY (SELECT * FROM ({' UNION ALL '.join(parts)}) "
        f'AS d("change", {ident(cell(key))}, "column", "before", "after") ORDER BY 1, 2, 3) '
        f"TO {lit(str(target))} (HEADER, DELIMITER ',')"
    )
    return target
