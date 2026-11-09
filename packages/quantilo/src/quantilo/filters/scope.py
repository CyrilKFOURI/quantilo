"""The rows an analysis reads: the dataset's Parquet copy, narrowed by the active filters."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from quantilo.filters.model import Condition
from quantilo.filters.sql import (
    ROW,
    FilterColumn,
    RowStep,
    filtered_source,
    row_steps,
    where_sql,
)
from quantilo.ingestion.sql import lit
from quantilo.typing.types import AnalysisType, Parser


class TypedColumn(Protocol):
    name: str
    type: AnalysisType
    parser: Parser


@dataclass(frozen=True)
class Where:
    """Value conditions (a WHERE clause), then row conditions (Unique / Duplicates) in order."""

    clause: str = ""
    steps: tuple[RowStep, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return bool(self.clause or self.steps)


def where_for(columns: Sequence[TypedColumn], conditions: list[Condition]) -> Where:
    """The filters, read through each column's effective type and parser (validated here)."""
    if not conditions:
        return Where()
    known = {c.name: FilterColumn(c.name, c.type, c.parser) for c in columns}
    return Where(where_sql(conditions, known), tuple(row_steps(conditions, known)))


def scoped_source(storage_path: str, where: Where, numbered: bool = False) -> str:
    """`numbered` keeps `file_row_number` (row numbers of the file, 0-based)."""
    with_rows = numbered or bool(where.steps)  # row conditions order rows by their number
    options = ", file_row_number = true" if with_rows else ""
    relation = filtered_source(f"read_parquet({lit(storage_path)}{options})", where.clause)
    for step in where.steps:
        relation = step.apply(relation)
    if with_rows and not numbered:
        relation = f"(SELECT * EXCLUDE ({ROW}) FROM {relation})"
    return relation
