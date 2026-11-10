"""What the detection finds: tables, their columns, what was set aside and why (JSON-ready)."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from openpyxl.utils import get_column_letter

Role = Literal["key", "value", "derived", "level", "flag", "section"]
Dtype = Literal["number", "date", "bool", "text"]
TableKind = Literal["table", "key_value", "text"]

HIGH = 0.7
REVIEW = 0.4
PREVIEW_HEADER = 0.8  # W1: the import preview only follows a detection this sure


@dataclass
class RuleHit:
    """One rule that fired: its code, its effect on the confidence, and a sentence for the user."""

    rule: str
    weight: float
    message: str


@dataclass
class ColumnPlan:
    name: str
    source_col: int  # 1-based sheet column
    role: Role = "value"
    dtype: Dtype = "text"
    unit: str | None = None
    decimal: str = "."
    nulls: int = 0  # T2 tokens read as empty
    unreadable: int = 0  # values of another type in a typed column, left empty
    flags: bool = False  # T3: a "<name> flag" column is split off


@dataclass
class Transform:
    """A restructuring: proposed (K rules) or applied by default (lossless clean-ups)."""

    kind: Literal[
        "melt_sparse", "melt_periods", "melt_numbered", "transpose", "fill_down", "split_combined"
    ]
    rule: str
    columns: list[str]
    into: list[str]
    message: str
    default: bool = False
    groups: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class DetectedTable:
    id: str
    sheet: str
    top: int  # 1-based, inclusive (first header row)
    left: int
    bottom: int
    right: int
    header_rows: list[int]
    kind: TableKind = "table"
    title: str | None = None
    notes: list[str] = field(default_factory=list)
    unit: str | None = None  # table-wide unit read in a title ("[Amounts in thousands]")
    units: dict[str, str] = field(default_factory=dict)
    unit_rows: list[int] = field(default_factory=list)
    columns: list[ColumnPlan] = field(default_factory=list)
    dropped_rows: dict[str, list[int]] = field(default_factory=dict)
    sections: dict[int, str] = field(default_factory=dict)  # row -> section label above it
    levels: dict[int, int] = field(default_factory=dict)  # row -> hierarchy level (C10)
    hierarchy_col: int | None = None  # the labelled column the levels describe
    level_col: int | None = None  # an explicit "Indent Level" column (C10-col)
    marked_totals: list[int] = field(default_factory=list)  # C8 keyword only, kept
    transforms: list[Transform] = field(default_factory=list)
    confidence: float = 0.0
    reasons: list[RuleHit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    uncached: int = 0
    first_data_row: int = 0
    truncated: bool = False

    @property
    def range(self) -> str:
        return (
            f"{get_column_letter(self.left)}{self.top}:{get_column_letter(self.right)}{self.bottom}"
        )

    @property
    def level(self) -> Literal["high", "review", "low"]:
        if self.confidence >= HIGH:
            return "high"
        return "review" if self.confidence >= REVIEW else "low"

    @property
    def name_rows(self) -> list[int]:
        """Header rows that name the columns (not units, not column numbers)."""
        skipped = set(self.unit_rows) | set(self.dropped_rows.get("column_numbers", []))
        return [row for row in self.header_rows if row not in skipped]

    def summary(self) -> str:
        """The sentence above the result (study § 6)."""
        header = ", ".join(str(row) for row in self.header_rows) or "none"
        body = self.bottom - self.first_data_row + 1 if self.first_data_row else 0
        removed = {key: len(rows) for key, rows in self.dropped_rows.items() if rows}
        parts = [
            f"Table {self.range}, header row{'s' if len(self.header_rows) > 1 else ''} "
            f"{header}, {body} rows."
        ]
        if removed:
            parts.append(
                "Removed: "
                + ", ".join(f"{n} {key.replace('_', ' ')}" for key, n in removed.items())
                + "."
            )
        nulls = sum(column.nulls for column in self.columns)
        if nulls:
            parts.append(f"{nulls} placeholder{'s' if nulls > 1 else ''} read as empty.")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["range"] = self.range
        data["level"] = self.level
        data["summary"] = self.summary()
        return data


@dataclass
class SheetLayout:
    name: str
    tables: list[DetectedTable] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fallback: bool = False  # S-fallback: nothing sure enough, the import keeps read_sheet
    truncated: bool = False
    first_row: int = 0  # 1-based first non-empty row: where `read_sheet` takes its names

    def preview_header(self) -> int | None:
        """W1: the single header row the import preview should read instead of the first non-empty
        row, when the sheet's first table is sure enough (confidence >= 0.8)."""
        table = next((t for t in self.tables if t.confidence >= REVIEW), None)
        if (
            table is None
            or table.confidence < PREVIEW_HEADER
            or len(table.header_rows) != 1
            or table.header_rows[0] == self.first_row
        ):
            return None
        return table.header_rows[0]


@dataclass
class Layout:
    file: str
    format: Literal["excel", "csv"]
    sheets: list[SheetLayout] = field(default_factory=list)
    section_name: str = ""
    warnings: list[str] = field(default_factory=list)
    delimiter: str | None = None  # CSV only: how `read` splits the rows again
    encoding: str | None = None

    @property
    def tables(self) -> list[DetectedTable]:
        return [table for sheet in self.sheets for table in sheet.tables]

    def table(self, table_id: str) -> DetectedTable:
        for table in self.tables:
            if table.id == table_id:
                return table
        raise KeyError(f"No table {table_id!r}: {', '.join(t.id for t in self.tables)}")

    def sheet(self, name: str) -> SheetLayout:
        for sheet in self.sheets:
            if sheet.name == name:
                return sheet
        raise KeyError(f"No sheet {name!r}")

    def read(self, table: "str | DetectedTable", reshape: bool = False) -> Any:
        """The table as a pyarrow Table: clean-ups applied, reshapes (K rules) when asked."""
        from pathlib import Path

        from quantilo.ingestion.layout.read import read_table

        detected = self.table(table) if isinstance(table, str) else table
        return read_table(Path(self.file), detected, reshape=reshape, delimiter=self.delimiter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "format": self.format,
            "section_name": self.section_name,
            "warnings": list(self.warnings),
            "sheets": [
                {
                    "name": sheet.name,
                    "fallback": sheet.fallback,
                    "truncated": sheet.truncated,
                    "warnings": list(sheet.warnings),
                    "tables": [table.to_dict() for table in sheet.tables],
                }
                for sheet in self.sheets
            ],
        }
