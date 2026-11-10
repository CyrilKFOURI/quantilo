"""What a cleaning is: an ordered list of typed steps (`op` + its fields), never code.

Versioned JSON (`schema: 1`). A step is validated strictly (an unknown field or op is refused);
its values reach SQL as literals, its column names as quoted identifiers (compile.py).
"""

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from quantilo.errors import ExplorerError

SCHEMA_VERSION = 1
MAX_STEPS = 300
MAX_MAP = 5_000  # values a merging step keeps (C-05)
MAX_VALUES = 1_000
Name = Annotated[str, Field(min_length=1, max_length=500)]
Value = Annotated[str, Field(max_length=2_000)]
Origin = Literal["auto", "suggested", "manual"]


class CleanError(ExplorerError):
    def __init__(
        self, message: str, code: str = "invalid_cleaning", status_code: int = 422
    ) -> None:
        super().__init__(message, code=code, status_code=status_code)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StepBase(Strict):
    id: str = Field(pattern=r"^s[0-9]{1,4}$")
    enabled: bool = True
    origin: Origin = "manual"
    rule: str | None = Field(default=None, max_length=20)  # B-01…, when a rule proposed it
    suggestion: str | None = Field(default=None, max_length=600)  # the suggestion's id


# --- Columns ---------------------------------------------------------------------------------


class RenameColumn(StepBase):
    op: Literal["rename_column"] = "rename_column"
    column: Name
    to: Name


class DropColumns(StepBase):
    op: Literal["drop_columns"] = "drop_columns"
    columns: list[Name] = Field(min_length=1, max_length=5_000)


class DuplicateColumn(StepBase):
    op: Literal["duplicate_column"] = "duplicate_column"
    column: Name
    to: Name


TargetType = Literal["text", "number", "integer", "date", "datetime", "boolean"]


class ChangeType(StepBase):
    """Write the column in another type. Text is read with the given decimal separator, date
    formats (tried in order) or true/false words; what cannot be read becomes empty."""

    op: Literal["change_type"] = "change_type"
    column: Name
    to: TargetType
    decimal: Literal[".", ","] = "."
    formats: list[Annotated[str, Field(max_length=40)]] = Field(default_factory=list, max_length=8)
    excel_serial: bool = False  # 20,000 to 60,000 read as Excel day numbers (origin 1899-12-30)
    true_values: list[Value] = Field(default_factory=list, max_length=50)
    false_values: list[Value] = Field(default_factory=list, max_length=50)


# --- Text ------------------------------------------------------------------------------------


class NormalizeSpaces(StepBase):
    """Trim, and one space for any run of spaces, tabs, line breaks or no-break spaces."""

    op: Literal["normalize_spaces"] = "normalize_spaces"
    columns: list[Name] = Field(min_length=1, max_length=5_000)


class RemoveInvisible(StepBase):
    """Remove zero-width characters, byte-order marks, soft hyphens and control characters."""

    op: Literal["remove_invisible"] = "remove_invisible"
    columns: list[Name] = Field(min_length=1, max_length=5_000)


class ChangeCase(StepBase):
    op: Literal["change_case"] = "change_case"
    columns: list[Name] = Field(min_length=1, max_length=5_000)
    case: Literal["lower", "upper", "title"]


class ReplaceValues(StepBase):
    op: Literal["replace_values"] = "replace_values"
    column: Name
    find: Annotated[str, Field(min_length=1, max_length=2_000)]
    replace: Value = ""
    match: Literal["exact", "contains", "regex"] = "exact"
    case_sensitive: bool = True


class MapValues(StepBase):
    """Old value → new value (None: empty). Only the values listed change (C-05)."""

    op: Literal["map_values"] = "map_values"
    column: Name
    map: dict[Value, Value | None] = Field(min_length=1, max_length=MAX_MAP)


# --- Values ----------------------------------------------------------------------------------


class Fill(StepBase):
    """Fill empty cells with the value above (down) or below (up), in the file's row order."""

    op: Literal["fill"] = "fill"
    column: Name
    direction: Literal["down", "up"] = "down"


class FillEmpty(StepBase):
    op: Literal["fill_empty"] = "fill_empty"
    column: Name
    value: Value


class NullValues(StepBase):
    """These values become empty. `exact`: the stored value as text; `token`: trimmed, any case."""

    op: Literal["null_values"] = "null_values"
    column: Name
    values: list[Value] = Field(min_length=1, max_length=MAX_VALUES)
    match: Literal["exact", "token"] = "exact"


class NullOutside(StepBase):
    """Numbers below `min` or above `max` become empty."""

    op: Literal["null_outside"] = "null_outside"
    column: Name
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _bound(self) -> "NullOutside":
        if self.min is None and self.max is None:
            raise ValueError("Give a minimum, a maximum or both.")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("The minimum is above the maximum.")
        return self


class PadZeros(StepBase):
    """Digits shorter than `length` get zeros on the left (a postal code that lost its zero);
    the column is written as text."""

    op: Literal["pad_zeros"] = "pad_zeros"
    column: Name
    length: int = Field(ge=2, le=40)


# --- Rows ------------------------------------------------------------------------------------

RowOperator = Literal["eq", "ne", "lt", "le", "gt", "ge", "contains", "empty", "not_empty", "in"]


class RowCondition(Strict):
    """On the stored value (decision 5 of the spec): numbers compare as numbers, the rest as
    text."""

    column: Name
    op: RowOperator
    value: Value | None = None
    values: list[Value] | None = Field(default=None, max_length=MAX_VALUES)

    @model_validator(mode="after")
    def _operand(self) -> "RowCondition":
        if self.op == "in" and not self.values:
            raise ValueError("`in` needs a list of values.")
        if self.op not in ("in", "empty", "not_empty") and self.value is None:
            raise ValueError(f"`{self.op}` needs a value.")
        return self


class FilterRows(StepBase):
    op: Literal["filter_rows"] = "filter_rows"
    conditions: list[RowCondition] = Field(min_length=1, max_length=20)
    combine: Literal["all", "any"] = "all"
    action: Literal["keep", "remove"] = "remove"


class DropDuplicates(StepBase):
    """Rows equal on `keys` (every column when None): one is kept."""

    op: Literal["drop_duplicates"] = "drop_duplicates"
    keys: list[Name] | None = Field(default=None, max_length=5_000)
    keep: Literal["first", "last", "most_complete"] = "first"


class DropEmptyRows(StepBase):
    op: Literal["drop_empty_rows"] = "drop_empty_rows"


class DropHeaderRows(StepBase):
    """Rows whose every value is its column's name: a header repeated inside the file."""

    op: Literal["drop_header_rows"] = "drop_header_rows"


Step = Annotated[
    RenameColumn
    | DropColumns
    | DuplicateColumn
    | ChangeType
    | NormalizeSpaces
    | RemoveInvisible
    | ChangeCase
    | ReplaceValues
    | MapValues
    | Fill
    | FillEmpty
    | NullValues
    | NullOutside
    | PadZeros
    | FilterRows
    | DropDuplicates
    | DropEmptyRows
    | DropHeaderRows,
    Field(discriminator="op"),
]
OPS = (
    "rename_column",
    "drop_columns",
    "duplicate_column",
    "change_type",
    "normalize_spaces",
    "remove_invisible",
    "change_case",
    "replace_values",
    "map_values",
    "fill",
    "fill_empty",
    "null_values",
    "null_outside",
    "pad_zeros",
    "filter_rows",
    "drop_duplicates",
    "drop_empty_rows",
    "drop_header_rows",
)
ROW_OPS = frozenset({"filter_rows", "drop_duplicates", "drop_empty_rows", "drop_header_rows"})


class Steps(Strict):
    """A cleaning: `{"schema": 1, "steps": [...]}`."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[1] = Field(default=SCHEMA_VERSION, alias="schema")
    steps: list[Step] = Field(default_factory=list, max_length=MAX_STEPS)

    @model_validator(mode="after")
    def _unique(self) -> "Steps":
        ids = [step.id for step in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("Two steps share the same id.")
        return self

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True)


def parse_steps(raw: str | bytes | dict[str, Any] | list[Any]) -> Steps:
    """Steps read from their JSON (or a bare list of steps), refused with the field named."""
    if isinstance(raw, str | bytes):
        try:
            raw = json.loads(raw)
        except ValueError as error:
            raise CleanError(f"Invalid steps: not JSON ({error}).") from None
    if isinstance(raw, list):
        raw = {"schema": SCHEMA_VERSION, "steps": raw}
    if not isinstance(raw, dict):
        raise CleanError("Invalid steps: expected a JSON object.")
    if raw.get("schema", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise CleanError(f"Unknown cleaning schema version {raw.get('schema')!r}.")
    try:
        return Steps.model_validate(raw)
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        if first["type"] == "extra_forbidden":
            raise CleanError(f"Unknown field `{where}` in the steps.") from None
        raise CleanError(f"Invalid steps, field `{where}`: {first['msg']}.") from None


def next_id(steps: list[Any]) -> str:
    """The next free step id (s1, s2…)."""
    used = [int(step.id[1:]) for step in steps]
    return f"s{max(used, default=0) + 1}"
