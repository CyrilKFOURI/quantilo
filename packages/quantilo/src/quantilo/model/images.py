"""Tables that name the images of the same project (C1, R-PRJ-11, role `image_ref` of 091).

A clinical table with one patient per row and a column `scan` = `mammo/P0012_left.png`, beside an
image set imported in the same project: the column refers to those images. Found by the values
only, compared three ways (the whole relative path, the file name, the file name without its
extension), each case ignored; a value with `..` or an absolute path (`/`, `C:`, `\\\\server`) is
never resolved and is counted apart. Nothing is changed: the page proposes the role.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import duckdb

from quantilo.catalog import ColumnView
from quantilo.ingestion.sql import ident, lit
from quantilo.model.keys import TableInput

MIN_SHARE = 0.9  # R-PRJ-11: share of the column's distinct values found among the images
MIN_FOUND = 5
MAX_VALUES = 50_000
EXAMPLES = 3
TEXT_TYPES = frozenset({"text", "categorical_nominal", "identifier", "categorical_ordinal"})
By = Literal["path", "file_name", "stem"]
_BY_SQL: dict[By, str] = {
    "path": "lower(rel_path)",
    "file_name": "lower(file_name)",
    "stem": "lower(regexp_replace(file_name, '\\.[^.]*$', ''))",
}
_VALUE_SQL: dict[By, str] = {
    "path": "v",
    "file_name": "regexp_extract(v, '[^/]*$')",
    "stem": "regexp_replace(regexp_extract(v, '[^/]*$'), '\\.[^.]*$', '')",
}


WORDS: dict[By, str] = {
    "path": "path",
    "file_name": "file name",
    "stem": "file name without extension",
}


@dataclass
class ImageSetInput:
    id: str
    name: str
    storage_path: str  # its table: one row per image (`rel_path`, `file_name`)


@dataclass
class ImageRef:
    table_id: str
    column: str
    image_set_id: str
    by: By
    found: int  # distinct values found among the images
    distinct: int  # distinct values compared (refused ones apart)
    refused: int  # values never resolved: `..` or an absolute path
    examples: list[str] = field(default_factory=list)
    rule: str = "R-PRJ-11"
    reason: str = ""


def _candidates(table: TableInput) -> list[ColumnView]:
    return [
        view
        for view in table.columns
        if not view.excluded
        and view.storage_dtype.upper().startswith("VARCHAR")
        and view.type in TEXT_TYPES
        and view.role not in ("free_text", "email", "url", "phone")
    ]


def image_references(
    connection: duckdb.DuckDBPyConnection, table: TableInput, sets: list[ImageSetInput]
) -> list[ImageRef]:
    """Every (column, image set) whose values name the set's images (R-PRJ-11), best first."""
    found: list[ImageRef] = []
    if not sets:
        return found
    for view in _candidates(table):
        column = ident(view.name)
        values = (
            f"SELECT DISTINCT lower(replace(trim({column}), '\\', '/')) AS v FROM read_parquet("
            f"{lit(table.storage_path)}) WHERE {column} IS NOT NULL AND trim({column}) <> ''"
        )
        connection.execute(
            f"CREATE OR REPLACE TEMP TABLE refs AS SELECT v FROM ({values}) "
            f"ORDER BY hash(v) LIMIT {MAX_VALUES}"
        )
        refused_sql = (
            "v LIKE '/%' OR regexp_matches(v, '^[a-z]:') OR regexp_matches(v, '(^|/)\\.\\.(/|$)')"
        )
        counted = connection.execute(
            f"SELECT count(*) FILTER (WHERE NOT ({refused_sql})), "
            f"count(*) FILTER (WHERE {refused_sql}) FROM refs"
        ).fetchone()
        distinct, refused = (int(counted[0]), int(counted[1])) if counted else (0, 0)
        if distinct < MIN_FOUND:
            continue
        for image_set in sets:
            best = _best(connection, image_set, refused_sql, distinct)
            if best is None:
                continue
            by, hits, examples = best
            share = hits / distinct
            found.append(
                ImageRef(
                    table_id=table.id,
                    column=view.name,
                    image_set_id=image_set.id,
                    by=by,
                    found=hits,
                    distinct=distinct,
                    refused=refused,
                    examples=examples,
                    reason=(
                        f"{hits:,} of the {distinct:,} values of '{view.name}' ({share:.0%}) name "
                        f"images of '{image_set.name}' by their {WORDS[by]} (case ignored)"
                        + (
                            f"; {refused:,} values with '..' or an absolute path are never followed"
                            if refused
                            else ""
                        )
                        + ": a reference to images (R-PRJ-11)."
                    ),
                )
            )
    connection.execute("DROP TABLE IF EXISTS refs")
    return sorted(found, key=lambda ref: (-ref.found / max(ref.distinct, 1), ref.column))


def _best(
    connection: duckdb.DuckDBPyConnection, image_set: ImageSetInput, refused_sql: str, distinct: int
) -> tuple[By, int, list[str]] | None:
    """The first way of comparing that finds 90 % of the values, on names unique in the set."""
    for by in ("path", "file_name", "stem"):
        key = _BY_SQL[by]
        images = (
            f"SELECT {key} AS k, count(*) AS n FROM read_parquet({lit(image_set.storage_path)}) "
            "GROUP BY 1"
        )
        row = connection.execute(
            f"WITH images AS ({images}) SELECT count(*), "
            f"(SELECT count(*) FROM images WHERE n > 1) FROM refs JOIN images "
            f"ON {_VALUE_SQL[by]} = images.k WHERE NOT ({refused_sql})"
        ).fetchone()
        hits, ambiguous = (int(row[0]), int(row[1])) if row else (0, 0)
        if ambiguous or hits < MIN_FOUND or hits < MIN_SHARE * distinct:
            continue
        examples = [
            str(value)
            for (value,) in connection.execute(
                f"WITH images AS ({images}) SELECT v FROM refs JOIN images "
                f"ON {_VALUE_SQL[by]} = images.k ORDER BY v LIMIT {EXAMPLES}"
            ).fetchall()
        ]
        return by, hits, examples
    return None
