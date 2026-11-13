"""N37: each column's quality as Power Query and Dataprep show it in one bar: the cells empty, the
cells its type cannot read, and the rest. Measured on the Basic profile (`basic.profile`): the
counts and distinct values it already holds, plus one query on its table of distinct values for the
text columns read as numbers, dates or booleans. The file itself is never read again."""

import duckdb
from pydantic import BaseModel, Field

from quantilo.cleaning.compile import is_text
from quantilo.cleaning.rules.basic import Column, Context
from quantilo.ingestion.sql import ident
from quantilo.typing.knowledge_base import load
from quantilo.typing.parsers import present_sql, typed_sql

UNREADABLE_SHOWN = 10  # the values not read listed under the bar (the first, most rows first)
READ_KINDS = ("number", "date", "boolean", "year", "time")  # the readings that can fail


class QualityValue(BaseModel):
    value: str
    rows: int


class ColumnQuality(BaseModel):
    """`missing`: the cells with no value (empty, blank, or a word read as empty: 'N/A', '-').
    `unreadable`: those with a value its type cannot read ('12,5 kg' in a number column); None
    when it was not counted (a text column with too many distinct values for the profile, whose
    blank cells are not counted in `missing` either). `reading`: what its type reads the stored
    text as (None: stored in its type, or a text column)."""

    column: str
    rows: int
    missing: int
    unreadable: int | None = 0
    unreadable_values: list[QualityValue] = Field(default_factory=list)
    reading: str | None = None


def _reads(column: Column) -> bool:
    """A text column its type reads as something else (a number, a date…), which can fail."""
    view = column.view
    if view is None or view.excluded or not is_text(column.dtype):
        return False
    parser = view.parser
    if parser.kind == "time":
        return parser.format is not None  # without a format, the stored value is cast as it is
    return parser.kind in READ_KINDS


def _blank(value: str, tokens: set[str]) -> bool:
    """As `clean_text` reads it: nothing once trimmed, or a null token."""
    trimmed = value.strip(" ")
    return trimmed == "" or trimmed.lower() in tokens


def measure(context: Context) -> list[ColumnQuality]:
    """Every column's quality, in the file's order. Call it between `basic.profile` and
    `basic.run` (which drops the table of distinct values)."""
    tokens = {token.lower() for token in load().null_tokens}
    found: list[ColumnQuality] = []
    for column in context.columns:
        missing = context.rows - column.nonnull
        quality = ColumnQuality(column=column.name, rows=context.rows, missing=missing)
        if is_text(column.dtype):
            if column.values is None and column.nonnull > 0:
                quality.unreadable = None  # too many distinct values: not profiled
            else:
                quality.missing += sum(c for v, c in column.values or [] if _blank(v, tokens))
        if _reads(column) and column.view is not None:
            quality.reading = column.view.parser.kind
        found.append(quality)
    _unreadable(context, found)
    return found


def _unreadable(context: Context, found: list[ColumnQuality]) -> None:
    """The values each text column read as another type cannot be read as, on the distinct values
    (`Context.values_table`), in one query for them all."""
    table = context.values_table
    columns = [c for c in context.columns if _reads(c) and c.values]
    if table is None or not columns:
        return
    parts = []
    for column in columns:
        view = column.view
        assert view is not None
        name = ident(column.name)
        present = present_sql(column.name, view.parser)
        typed = typed_sql(column.name, view.parser)
        parts.append(
            f"SELECT {column.position} AS k, {name} AS v, c FROM (SELECT v AS {name}, c "
            f"FROM {table} WHERE k = {column.position}) WHERE ({present}) IS NOT NULL "
            f"AND ({typed}) IS NULL"
        )
    query = (
        "SELECT k, sum(c), list(v ORDER BY c DESC, v), list(c ORDER BY c DESC, v) FROM ("
        + " UNION ALL ".join(parts)
        + ") GROUP BY k"
    )
    try:
        rows = context.connection.execute(query).fetchall()
    except duckdb.Error:
        for quality in found:
            if quality.reading is not None:
                quality.unreadable = None  # could not be counted: said, never guessed
        return
    by_position = {c.position: q for c, q in zip(context.columns, found, strict=True)}
    for position, total, values, counts in rows:
        quality = by_position[int(position)]
        quality.unreadable = int(total)
        quality.unreadable_values = [
            QualityValue(value=str(v), rows=int(n))
            for v, n in zip(values[:UNREADABLE_SHOWN], counts[:UNREADABLE_SHOWN], strict=True)
        ]
