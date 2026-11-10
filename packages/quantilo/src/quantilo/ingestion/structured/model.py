"""What the structured reader finds: tables, columns, relations, the rules that fired and why.

Plain dataclasses, JSON-ready through `quantilo.to_dict` (the Arrow data stays outside the
fields: `layout.arrow(name)`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pyarrow as pa

    from quantilo.dataset import Dataset

Format = Literal["json", "jsonl", "xml"]
Dtype = Literal["int", "float", "bool", "string", "json"]
TableKind = Literal["records", "objects", "scalars", "map", "object", "meta"]
Override = Literal["flatten", "separate", "json", "skip"]
Level = Literal["high", "review", "low"]

HIGH = 0.7  # same levels as the sheet layout (080)
REVIEW = 0.4


@dataclass(frozen=True)
class StructuredOptions:
    """Choices of the reader; the defaults are the study's thresholds (083 spec)."""

    format: Format | None = None  # detected when None (R-DET-01, R-DET-02)
    records_path: str | None = None  # dotted; "" = the root as one record; None = found
    sample: int = 5_000  # records profiled
    sample_bytes: int = 20_000_000  # JSON Lines: or this much of lines
    max_depth: int = 4  # R-SST-11 / R-SST-18
    max_columns: int = 500  # R-SST-18
    max_tables: int = 20  # R-SST-21
    joined_distinct: int = 30  # R-SST-14
    one_table: bool = False  # R-SST-15b
    overrides: dict[str, Override] = field(default_factory=dict)
    max_bytes: int | None = None  # R-LIM-01; None = 200 MB native, 50 MB in the browser
    limit: int | None = None  # records materialized (a preview); None = all


@dataclass
class RuleHit:
    """One rule that fired: its code, its effect on the confidence, a sentence for the user."""

    rule: str
    weight: float
    message: str


@dataclass
class StructuredColumn:
    name: str
    path: str  # dotted, relative to the table's row ("customer.city", "items.count")
    dtype: Dtype
    rule: str
    reason: str
    present: int = 0  # rows where the key was there (explicit nulls included)
    nulls: int = 0  # explicit JSON nulls
    missing: int = 0  # rows without the key (R-SST-19)
    unreadable: int = 0  # values left empty: not of the column's type (R-SST-16)
    examples: list[str] = field(default_factory=list)  # a few unreadable values
    notes: list[str] = field(default_factory=list)


@dataclass
class StructuredTable:
    name: str
    kind: TableKind
    locator: str  # "$.results[*].patient.drug[*]"
    rule: str
    reason: str
    parent: str | None = None
    rows: int = 0
    columns: list[StructuredColumn] = field(default_factory=list)
    selected: bool = True
    natural_key: str | None = None  # R-KEY-02
    warnings: list[str] = field(default_factory=list)

    def column(self, name: str) -> StructuredColumn:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)


@dataclass
class Relation:
    """`parent.parent_key` = `child.child_key` (R-KEY-03)."""

    parent: str
    child: str
    parent_key: str
    child_key: str
    rule: str
    path: str  # dotted field of the parent row ("items")


@dataclass
class RecordsCandidate:
    path: str  # dotted from the root
    count: int


@dataclass
class StructuredLayout:
    source: str
    format: Format
    records_path: str  # dotted from the root, "" for the root
    records_rule: str
    records: int  # records read (all of them unless `limit`)
    sampled: int  # records profiled
    truncated: bool = False  # `limit` stopped the reading
    candidates: list[RecordsCandidate] = field(default_factory=list)
    tables: list[StructuredTable] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    reasons: list[RuleHit] = field(default_factory=list)  # document-level rules
    warnings: list[str] = field(default_factory=list)
    skipped_lines: list[int] = field(default_factory=list)  # JSON Lines (R-LIM-05), 1-based
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self._data: dict[str, pa.Table] = {}

    @property
    def level(self) -> Level:
        if self.confidence >= HIGH:
            return "high"
        return "review" if self.confidence >= REVIEW else "low"

    def table(self, name: str) -> StructuredTable:
        for table in self.tables:
            if table.name == name:
                return table
        known = ", ".join(table.name for table in self.tables)
        raise KeyError(f"No table {name!r} (tables: {known})")

    def arrow(self, name: str) -> pa.Table:
        self.table(name)
        return self._data[name]

    def rows(self, name: str, n: int = 20) -> list[dict[str, Any]]:
        """The first rows of a table, as dicts (the preview of the Structure screen)."""
        return self.arrow(name).slice(0, n).to_pylist()

    def summary(self) -> str:
        """The sentence above the result: "3 tables from orders.json: orders (4 rows), …"."""
        shown = [table for table in self.tables if table.kind != "meta"]
        parts = []
        for index, table in enumerate(shown):
            rows = f"{table.rows:,}".replace(",", " ")
            parts.append(f"{table.name} ({rows} rows)" if index == 0 else f"{table.name} ({rows})")
        count = len(shown)
        name = Path(self.source).name
        return f"{count} table{'s' if count != 1 else ''} from {name}: " + ", ".join(parts) + "."

    def to_dict(self) -> dict[str, Any]:
        from quantilo.results import to_dict

        data = to_dict(self)
        data["level"] = self.level
        data["summary"] = self.summary()
        return data

    def write_parquet(
        self, folder: str | os.PathLike[str], *, selected_only: bool = True
    ) -> dict[str, Path]:
        """One Parquet file per table (the selected ones by default); returns name -> path."""
        from quantilo.ingestion.structured.parquet import write_tables

        names = [t.name for t in self.tables if t.selected or not selected_only]
        return write_tables({name: self._data[name] for name in names}, Path(folder))

    def open(
        self,
        name: str,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> Dataset:
        """A table as a `Dataset` (copied to Parquet, types detected)."""
        from quantilo.dataset import from_arrow

        return from_arrow(self.arrow(name), name, cache_dir=cache_dir)
