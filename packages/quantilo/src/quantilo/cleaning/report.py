"""What a cleaned version did: rows and cells changed, per column and per step, and how long."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StepReport:
    id: str
    op: str
    rule: str | None
    origin: str
    rows_removed: int = 0


@dataclass
class TypeChange:
    column: str
    before: str
    after: str


@dataclass
class CleanReport:
    rows_before: int
    rows_after: int
    rows_removed: int
    cells_changed: int
    columns_before: int
    columns_after: int
    cells_by_column: dict[str, int] = field(default_factory=dict)
    steps: list[StepReport] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    type_changes: list[TypeChange] = field(default_factory=list)  # filled after the typing
    seconds: float = 0.0
    # C-59: the step that reshaped the table (split into rows, unpivot, pivot, group by, append);
    # the version then has no row map: its rows do not come one by one from the parent's
    reshaped_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CleanReport":
        data = dict(raw)
        data["steps"] = [StepReport(**step) for step in data.get("steps", [])]
        data["type_changes"] = [TypeChange(**change) for change in data.get("type_changes", [])]
        return cls(**data)

    def summary(self) -> str:
        """'1,204 cells changed · 18 rows removed'."""
        parts = [f"{self.cells_changed:,} cells changed"]
        if self.rows_removed:
            parts.append(f"{self.rows_removed:,} rows removed")
        if self.dropped:
            parts.append(f"{len(self.dropped)} columns removed")
        return " · ".join(parts)
