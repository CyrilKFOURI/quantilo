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
MAX_CASES = 500  # cases of a conditional step (A-10 writes one per value it corrects)
MAX_PARTS = 20  # columns a split makes
MAX_GROUP_ROWS = 20_000  # N35: rows a step lists in its groups, or as not duplicates
Name = Annotated[str, Field(min_length=1, max_length=500)]
Value = Annotated[str, Field(max_length=2_000)]
Formula = Annotated[str, Field(min_length=1, max_length=2_000)]  # formula.py
Origin = Literal["auto", "suggested", "manual"]
# 096/097: what a suggestion's correction touches, and whether it can be applied without a look.
# format: the writing only, nothing lost; values: values change; types: the storage type changes;
# rows: rows go; columns: columns go. safe: automatic (mode A); review: the user decides.
Impact = Literal["format", "values", "types", "rows", "columns"]
Risk = Literal["safe", "review"]
IMPACTS: tuple[Impact, ...] = ("format", "values", "types", "rows", "columns")
RISKS: tuple[Risk, ...] = ("safe", "review")


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


class AssembleDate(StepBase):
    """A new date column from its parts held in separate columns (fix1, B-20): the year, the
    month (a number or a month name), the day (the 1st when there is none), and the hour and minute
    when given, apart or in one time-of-day column (`time`). The part columns stay as they are; a
    row whose parts make no date (a 31 February, a missing part) gets an empty date."""

    op: Literal["assemble_date"] = "assemble_date"
    year: Name
    month: Name
    day: Name | None = None
    hour: Name | None = None
    minute: Name | None = None
    # C1 (fix1): the hour and minute from one time-of-day column ('07:40', '7h40', '0740', 740)
    time: Name | None = None
    to: Name

    @model_validator(mode="after")
    def _one_time(self) -> "AssembleDate":
        if self.time is not None and (self.hour is not None or self.minute is not None):
            raise ValueError("Give an hour (and a minute) or a time column, not both.")
        return self


class LabelRows(StepBase):
    """099 (R-PIN-15): a new text column with the label of each pinned row, the rows given by their
    number in the file (from 0); the other rows are empty, or `other` when given. The numbers are
    those of the current import: after a re-import that reorders the rows, make the step again."""

    op: Literal["label_rows"] = "label_rows"
    to: Name
    labels: dict[int, Annotated[str, Field(min_length=1, max_length=60)]] = Field(
        min_length=1, max_length=MAX_MAP
    )
    other: Value | None = None


class SetCells(StepBase):
    """101 (C-33): given cells of one column get a value (None: empty), the rows given by their
    number in the file (from 0), like `label_rows`. What the Expert checks propose: a label
    corrected, an empty cell filled, a cell error fixed. The numbers are those of the current
    import: on another file the step does not apply (the replay says so)."""

    op: Literal["set_cells"] = "set_cells"
    column: Name
    cells: dict[int, Value | None] = Field(min_length=1, max_length=MAX_MAP)


TargetType = Literal["text", "number", "integer", "date", "datetime", "boolean"]


class ChangeType(StepBase):
    """Write the column in another type. Text is read with the given decimal separator, date
    formats (tried in order) or true/false words; what cannot be read becomes empty. An empty
    word among `false_values` makes a blank cell false (a checkbox)."""

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
    """Trim, and one space for any run of spaces, tabs, line breaks or no-break spaces.
    `keep_newlines` (a free text, N29): the line breaks stay, only the spaces along a line, around
    a break and at the ends change."""

    op: Literal["normalize_spaces"] = "normalize_spaces"
    columns: list[Name] = Field(min_length=1, max_length=5_000)
    keep_newlines: bool = False


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
    the column is written as text. With `conditions`, only the rows meeting all of them are padded
    (N12: the rows of the countries whose codes have that length)."""

    op: Literal["pad_zeros"] = "pad_zeros"
    column: Name
    length: int = Field(ge=2, le=40)
    conditions: "list[RowCondition]" = Field(default_factory=list, max_length=20)


class TransformNumber(StepBase):
    """M5 (study § 3, numbers): round to `decimals`, absolute value, multiply or divide by
    `factor`, clip to [`min`, `max`], or bin by `width` (each value → the start of its bin)."""

    op: Literal["transform_number"] = "transform_number"
    column: Name
    how: Literal["round", "abs", "multiply", "divide", "clip", "bin"] = "round"
    decimals: int = Field(default=0, ge=-10, le=15)
    factor: float = 1.0
    min: float | None = None
    max: float | None = None
    width: float | None = None

    @model_validator(mode="after")
    def _settings(self) -> "TransformNumber":
        if self.how in ("multiply", "divide") and self.factor == 0:
            raise ValueError("The factor cannot be 0.")
        if self.how == "clip":
            if self.min is None and self.max is None:
                raise ValueError("Give a minimum, a maximum or both.")
            if self.min is not None and self.max is not None and self.min > self.max:
                raise ValueError("The minimum is above the maximum.")
        if self.how == "bin" and (self.width is None or self.width <= 0):
            raise ValueError("Give a bin width above 0.")
        return self


# --- Splitting, extracting, calculating (M5, study § 3) -------------------------------------


class SplitColumn(StepBase):
    """New columns `to` from the parts of `column`, placed after it (the column stays): cut at a
    `delimiter`, at character `positions` (from 1), or where digits meet other characters
    ('12kg' → '12', 'kg'). The last new column keeps the rest, so nothing is lost."""

    op: Literal["split_column"] = "split_column"
    column: Name
    by: Literal["delimiter", "positions", "digits"] = "delimiter"
    delimiter: Annotated[str, Field(min_length=1, max_length=20)] = ","
    positions: list[Annotated[int, Field(ge=1, le=100_000)]] = Field(
        default_factory=list, max_length=MAX_PARTS - 1
    )
    to: list[Name] = Field(min_length=2, max_length=MAX_PARTS)

    @model_validator(mode="after")
    def _parts(self) -> "SplitColumn":
        if len(set(self.to)) != len(self.to):
            raise ValueError("Two new columns share a name.")
        if self.by == "positions":
            if sorted(set(self.positions)) != self.positions:
                raise ValueError("Positions go up, each once.")
            if len(self.positions) != len(self.to) - 1:
                raise ValueError("Give one position fewer than new columns.")
        return self


class MergeColumns(StepBase):
    """A new column `to`: the values of `columns` joined by `separator`, empty ones left out; the
    columns stay."""

    op: Literal["merge_columns"] = "merge_columns"
    columns: list[Name] = Field(min_length=2, max_length=50)
    separator: Annotated[str, Field(max_length=20)] = " "
    to: Name


class ExtractText(StepBase):
    """A part of the text: its first or last `count` characters, what lies between `start` and
    `end` (an empty one: the beginning, the end), or what a regular expression `pattern` finds
    (its first group when it has one). In the column itself, or in a new column `to`."""

    op: Literal["extract_text"] = "extract_text"
    column: Name
    how: Literal["first", "last", "between", "regex"] = "first"
    count: int = Field(default=1, ge=1, le=100_000)
    start: Annotated[str, Field(max_length=200)] = ""
    end: Annotated[str, Field(max_length=200)] = ""
    pattern: Annotated[str, Field(max_length=500)] | None = None
    to: Name | None = None

    @model_validator(mode="after")
    def _pattern(self) -> "ExtractText":
        if self.how == "regex" and not self.pattern:
            raise ValueError("Give a pattern.")
        if self.how == "between" and not (self.start or self.end):
            raise ValueError("Give a start, an end or both.")
        return self


class FormatText(StepBase):
    """Text written around the value: `pad_with` (a zero) on the left up to `pad_to`
    characters, then `prefix` and `suffix`. An empty cell stays empty; the column becomes text."""

    op: Literal["format_text"] = "format_text"
    column: Name
    prefix: Annotated[str, Field(max_length=200)] = ""
    suffix: Annotated[str, Field(max_length=200)] = ""
    pad_to: int | None = Field(default=None, ge=1, le=200)
    pad_with: Annotated[str, Field(min_length=1, max_length=1)] = "0"

    @model_validator(mode="after")
    def _something(self) -> "FormatText":
        if not (self.prefix or self.suffix or self.pad_to):
            raise ValueError("Give a prefix, a suffix or a length to pad to.")
        return self


class ParseQuantity(StepBase):
    """N23: a quantity written with its unit ('12 kg', '500 g', '1,5 L', '12 x 33 cl') read as a
    number in the base unit of its kind, in a new column per kind: grams (`to_grams`: mg, g, kg,
    lb, oz), litres (`to_litres`: ml, cl, dl, l, fl oz), metres (`to_metres`: mm, cm, m, km); the
    unit as written in `unit_to`. A multiplier before the quantity is read ('12 x 33 cl' = 3.96 L),
    a comma is the decimal mark; a value with an unknown unit stays empty in the new columns, as a
    value of another kind does. The column itself stays."""

    op: Literal["parse_quantity"] = "parse_quantity"
    column: Name
    to_grams: Name | None = None
    to_litres: Name | None = None
    to_metres: Name | None = None
    unit_to: Name | None = None

    @model_validator(mode="after")
    def _targets(self) -> "ParseQuantity":
        names = [n for n in (self.to_grams, self.to_litres, self.to_metres) if n is not None]
        if not names:
            raise ValueError("Give a new column for grams, litres or metres.")
        if self.unit_to is not None:
            names.append(self.unit_to)
        if len(set(names)) != len(names):
            raise ValueError("Two new columns share a name.")
        return self


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


PadZeros.model_rebuild()


class FilterRows(StepBase):
    op: Literal["filter_rows"] = "filter_rows"
    conditions: list[RowCondition] = Field(min_length=1, max_length=20)
    combine: Literal["all", "any"] = "all"
    action: Literal["keep", "remove"] = "remove"


class Case(Strict):
    """One case of a conditional step (C-45): the rows that meet its `conditions` (all or any)
    and its formula test `when` get `value` (None: empty), or the result of `formula`."""

    conditions: list[RowCondition] = Field(default_factory=list, max_length=20)
    combine: Literal["all", "any"] = "all"
    when: Formula | None = None
    value: Value | None = None
    formula: Formula | None = None

    @model_validator(mode="after")
    def _test(self) -> "Case":
        if not self.conditions and self.when is None:
            raise ValueError("A case needs a condition or a formula test.")
        return self


class SetWhere(StepBase):
    """The conditional step (C-45, "set B = v where A = a"): the first case a row meets gives
    `column` its new value; the other rows keep theirs. What A-10 to A-13 propose."""

    op: Literal["set_where"] = "set_where"
    column: Name
    cases: list[Case] = Field(min_length=1, max_length=MAX_CASES)


class ConditionalColumn(StepBase):
    """A new column `to`: the value of the first case a row meets, else `otherwise`."""

    op: Literal["conditional_column"] = "conditional_column"
    to: Name
    cases: list[Case] = Field(min_length=1, max_length=MAX_CASES)
    otherwise: Value | None = None


class ComputedColumn(StepBase):
    """A new column `to` computed by a formula of the other columns (formula.py)."""

    op: Literal["computed_column"] = "computed_column"
    to: Name
    formula: Formula


class SwapValues(StepBase):
    """The values of two columns swapped on the rows where the formula `when` holds (A-11: an
    end before its start). Both columns must have the same type at this step."""

    op: Literal["swap_values"] = "swap_values"
    columns: list[Name] = Field(min_length=2, max_length=2)
    when: Formula

    @model_validator(mode="after")
    def _two(self) -> "SwapValues":
        if self.columns[0] == self.columns[1]:
            raise ValueError("Swap two different columns.")
        return self


class DropDuplicates(StepBase):
    """Rows equal on `keys` (every column when None): one is kept, the first, the last, the most
    complete (fewest empty cells, then the first) or, N22, the latest by the date `order_by`
    (an update time; rows without one come last, then the first).

    N35, the review by group (Splink / dedupe's review of pairs, Informatica's survivorship):
    `groups` lists the groups by their rows' numbers instead of `keys` (A-16, whose groups come
    from pairs), each in the rule's order (its first row is kept); `survivors` are rows the user
    chose in their group, kept instead of the rule's; `not_duplicates` the groups the user said
    are not duplicates (their rows, every one kept); `merge` fills the kept row's empty cells with
    the first value of the other rows of its group (the rule's order), only when asked. Those four
    name rows of the file the step was made on (`names_rows`)."""

    op: Literal["drop_duplicates"] = "drop_duplicates"
    keys: list[Name] | None = Field(default=None, max_length=5_000)
    keep: Literal["first", "last", "most_complete", "latest"] = "first"
    order_by: Name | None = None
    groups: list[list[int]] | None = Field(default=None, max_length=MAX_MAP)
    survivors: list[int] | None = Field(default=None, max_length=MAX_MAP)
    not_duplicates: list[list[int]] | None = Field(default=None, max_length=MAX_MAP)
    merge: bool = False

    @model_validator(mode="after")
    def _latest(self) -> "DropDuplicates":
        if self.keep == "latest" and self.order_by is None:
            raise ValueError("Keeping the latest row needs a date column (`order_by`).")
        if self.groups is not None:
            rows = [row for group in self.groups for row in group]
            if any(len(group) < 2 for group in self.groups):
                raise ValueError("A group of duplicates has two rows at least.")
            if len(set(rows)) != len(rows) or len(rows) > MAX_GROUP_ROWS:
                raise ValueError(
                    f"Groups of duplicates: distinct rows, {MAX_GROUP_ROWS:,} at most."
                )
        kept = [row for group in self.not_duplicates or [] for row in group]
        if len(kept) > MAX_GROUP_ROWS:
            raise ValueError(f"Rows said not duplicates: {MAX_GROUP_ROWS:,} at most.")
        if len(set(self.survivors or [])) != len(self.survivors or []):
            raise ValueError("A row is chosen twice to be kept.")
        return self


AnchorKind = Literal["postal_code", "phone", "email", "key"]
MAX_ENTITY_RECORDS = 5_000  # N36: records (a spelling and its anchor) a step lists in its clusters


class EntityRecord(Strict):
    """N36: one record of an entity cluster: the name and the anchor as `entity_column` reads them
    (`entity_keys`: folded name; anchor normalized by its kind)."""

    anchor: Value
    name: Value


class EntityColumn(StepBase):
    """N36 (A-20): a new column `to` numbering the entity of each row (1, 2, 3… in the order the
    entities first appear), the names left as they are. A row's record is its name (`column`,
    case, accents and spaces folded) and its anchor (`anchor`, normalized by `anchor_kind`: a
    postcode without spaces, a phone's last 9 digits, an e-mail in small letters, a key as
    written); rows of one record are one entity, and the records of a cluster of `groups` are one
    entity too. The review (N35's, by cluster): `not_entities` lists the clusters (by their index)
    whose records stay apart, `split` the records (by their index in their cluster) taken out of
    a cluster, together another entity. A row without a name has no entity. Value-based: it
    replays on another file."""

    op: Literal["entity_column"] = "entity_column"
    column: Name
    anchor: Name | None = None
    anchor_kind: AnchorKind = "key"
    to: Name
    groups: list[list[EntityRecord]] = Field(default_factory=list, max_length=MAX_MAP)
    not_entities: list[int] | None = Field(default=None, max_length=MAX_MAP)
    split: dict[int, list[int]] | None = Field(default=None, max_length=MAX_MAP)

    @model_validator(mode="after")
    def _clusters(self) -> "EntityColumn":
        records = [record for group in self.groups for record in group]
        if any(len(group) < 2 for group in self.groups):
            raise ValueError("A cluster of an entity has two records at least.")
        if len(records) > MAX_ENTITY_RECORDS:
            raise ValueError(f"Clusters of entities: {MAX_ENTITY_RECORDS:,} records at most.")
        keys = [(record.anchor, record.name) for record in records]
        if len(set(keys)) != len(keys):
            raise ValueError("A record is in two clusters.")
        if any(not 0 <= index < len(self.groups) for index in self.not_entities or []):
            raise ValueError("A cluster said apart is not one of the step's clusters.")
        for index, taken in (self.split or {}).items():
            if not 0 <= index < len(self.groups):
                raise ValueError("A cluster split is not one of the step's clusters.")
            if any(not 0 <= record < len(self.groups[index]) for record in taken):
                raise ValueError("A record taken out of a cluster is not one of its records.")
        return self


def names_rows(step: Any) -> bool:
    """Whether a step names rows by their number in the file (tied to one import, C-41): the
    steps of `ROW_NUMBER_OPS`, and a removal of duplicates with explicit groups (A-16, N35)."""
    if step.op in ROW_NUMBER_OPS:
        return True
    return step.op == "drop_duplicates" and step.groups is not None


def reviewed(step: Any) -> bool:
    """N35: a removal of duplicates whose groups were reviewed (rows chosen or kept by name)."""
    return step.op == "drop_duplicates" and bool(step.survivors or step.not_duplicates)


class DropRows(StepBase):
    """101 (C-33): rows removed by their number in the file (from 0), like `label_rows`."""

    op: Literal["drop_rows"] = "drop_rows"
    rows: list[int] = Field(min_length=1, max_length=MAX_MAP)


class DropEmptyRows(StepBase):
    """Rows with no value at all; with `share` (A-17), rows with more than this share of their
    cells empty."""

    op: Literal["drop_empty_rows"] = "drop_empty_rows"
    share: float | None = Field(default=None, gt=0, lt=1)


class DropHeaderRows(StepBase):
    """Rows whose every value is its column's name: a header repeated inside the file."""

    op: Literal["drop_header_rows"] = "drop_header_rows"


# --- C1: dates, rows and header, structure (rest of M5, study § 3) ------------------------------

DateUnit = Literal["years", "months", "days", "hours", "minutes", "seconds"]
Zone = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_/+\-]+$")]
DatasetRef = Annotated[str, Field(min_length=1, max_length=100)]
MAX_HEADERS = 200  # columns a pivot makes
MAX_KEYS = 10


class ShiftDate(StepBase):
    """C-52: a date or date-time moved by `amount` `unit`s (a negative amount goes back). A date
    (no time) moves by days or more only."""

    op: Literal["shift_date"] = "shift_date"
    column: Name
    amount: int = Field(ge=-1_000_000, le=1_000_000)
    unit: DateUnit = "hours"


class ConvertTimezone(StepBase):
    """C-52: a date-time read in `from_zone` and written in `to_zone` (IANA names such as
    Europe/Paris, or UTC); summer time follows each zone's own rules."""

    op: Literal["convert_timezone"] = "convert_timezone"
    column: Name
    from_zone: Zone = "UTC"
    to_zone: Zone

    @model_validator(mode="after")
    def _two(self) -> "ConvertTimezone":
        if self.from_zone == self.to_zone:
            raise ValueError("Give two different time zones.")
        return self


class KeepRows(StepBase):
    """C-53: the first or last `count` rows kept, or removed, in the file's row order."""

    op: Literal["keep_rows"] = "keep_rows"
    how: Literal["first", "last", "remove_first", "remove_last"] = "first"
    count: int = Field(ge=1, le=100_000_000)


class PromoteHeader(StepBase):
    """C-54: the first row becomes the column names (an empty or repeated one keeps a numbered
    name), and leaves the table. Read on the file it runs on, like Power Query's. N47: `row` reads
    the names on a later row (counted from 1); the rows above it, a note, leave the table too."""

    op: Literal["promote_header"] = "promote_header"
    row: int = Field(default=1, ge=1, le=100)


class SplitRows(StepBase):
    """C-55: one row per part of `column` cut at `delimiter`; the other columns repeat. An empty
    cell keeps its row."""

    op: Literal["split_rows"] = "split_rows"
    column: Name
    delimiter: Annotated[str, Field(min_length=1, max_length=20)] = ","
    trim: bool = True


class Unpivot(StepBase):
    """C-56: `columns` turned into rows, two columns: their name (`names_to`) and their value
    (`values_to`); the other columns repeat. Empty values are kept unless `drop_empty`."""

    op: Literal["unpivot"] = "unpivot"
    columns: list[Name] = Field(min_length=1, max_length=5_000)
    names_to: Name = "attribute"
    values_to: Name = "value"
    drop_empty: bool = False

    @model_validator(mode="after")
    def _names(self) -> "Unpivot":
        if self.names_to == self.values_to:
            raise ValueError("The name and value columns need two names.")
        return self


Aggregate = Literal[
    "count", "count_distinct", "sum", "mean", "median", "min", "max", "first", "last"
]


class Pivot(StepBase):
    """C-56: the values of `column` become columns, filled with `aggregate` of `value_column`
    (rows counted when none) for each group of the other columns. `headers`: the values kept, in
    order; empty, the values found (200 at most, in the order they first appear)."""

    op: Literal["pivot"] = "pivot"
    column: Name
    value_column: Name | None = None
    aggregate: Aggregate = "sum"
    headers: list[Value] = Field(default_factory=list, max_length=MAX_HEADERS)

    @model_validator(mode="after")
    def _counted(self) -> "Pivot":
        if self.value_column is None and self.aggregate != "count":
            raise ValueError("Without a value column, the rows are counted.")
        if self.value_column == self.column:
            raise ValueError("The value column cannot be the column that becomes headers.")
        return self


class Aggregation(Strict):
    """One column of a group: `fn` of `column` (rows counted when none), named `to`."""

    column: Name | None = None
    fn: Aggregate = "sum"
    to: Name

    @model_validator(mode="after")
    def _column(self) -> "Aggregation":
        if self.column is None and self.fn != "count":
            raise ValueError(f"`{self.fn}` needs a column.")
        return self


class GroupBy(StepBase):
    """C-56: one row per combination of `keys`, with the `aggregations`."""

    op: Literal["group_by"] = "group_by"
    keys: list[Name] = Field(min_length=1, max_length=50)
    aggregations: list[Aggregation] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def _names(self) -> "GroupBy":
        names = [*self.keys, *(a.to for a in self.aggregations)]
        if len(set(names)) != len(names):
            raise ValueError("Two columns of the result share a name.")
        return self


class AppendRows(StepBase):
    """C-57: the rows of another dataset added below, columns matched by name (one missing on a
    side is empty there); `source_column` says which dataset each row comes from."""

    op: Literal["append_rows"] = "append_rows"
    dataset: DatasetRef
    label: Value | None = None  # its name when the step was made
    source_column: Name | None = None
    this_label: Value | None = None  # the value `source_column` gives this table's rows


class MergeDataset(StepBase):
    """C-58: columns of another dataset looked up by key (`keys` here = `other_keys` there).
    `left` keeps every row, `inner` keeps the rows found. Refused when the other dataset repeats
    a key: rows would multiply (R-JOIN-04)."""

    op: Literal["merge_dataset"] = "merge_dataset"
    dataset: DatasetRef
    label: Value | None = None
    keys: list[Name] = Field(min_length=1, max_length=MAX_KEYS)
    other_keys: list[Name] = Field(min_length=1, max_length=MAX_KEYS)
    add: list[Name] | None = Field(default=None, max_length=5_000)  # None: all but its keys
    how: Literal["left", "inner"] = "left"
    prefix: Annotated[str, Field(max_length=60)] = ""

    @model_validator(mode="after")
    def _pairs(self) -> "MergeDataset":
        if len(self.keys) != len(self.other_keys):
            raise ValueError("Give as many keys on each side.")
        return self


Step = Annotated[
    RenameColumn
    | DropColumns
    | DuplicateColumn
    | AssembleDate
    | LabelRows
    | ChangeType
    | NormalizeSpaces
    | RemoveInvisible
    | ChangeCase
    | ReplaceValues
    | MapValues
    | SetCells
    | Fill
    | FillEmpty
    | NullValues
    | NullOutside
    | PadZeros
    | TransformNumber
    | SplitColumn
    | MergeColumns
    | ExtractText
    | FormatText
    | ParseQuantity
    | SetWhere
    | ConditionalColumn
    | ComputedColumn
    | SwapValues
    | FilterRows
    | DropDuplicates
    | EntityColumn
    | DropRows
    | DropEmptyRows
    | DropHeaderRows
    | ShiftDate
    | ConvertTimezone
    | KeepRows
    | PromoteHeader
    | SplitRows
    | Unpivot
    | Pivot
    | GroupBy
    | AppendRows
    | MergeDataset,
    Field(discriminator="op"),
]
OPS = (
    "rename_column",
    "drop_columns",
    "duplicate_column",
    "assemble_date",
    "label_rows",
    "change_type",
    "normalize_spaces",
    "remove_invisible",
    "change_case",
    "replace_values",
    "map_values",
    "set_cells",
    "fill",
    "fill_empty",
    "null_values",
    "null_outside",
    "pad_zeros",
    "transform_number",
    "split_column",
    "merge_columns",
    "extract_text",
    "format_text",
    "parse_quantity",
    "set_where",
    "conditional_column",
    "computed_column",
    "swap_values",
    "filter_rows",
    "drop_duplicates",
    "entity_column",
    "drop_rows",
    "drop_empty_rows",
    "drop_header_rows",
    "shift_date",
    "convert_timezone",
    "keep_rows",
    "promote_header",
    "split_rows",
    "unpivot",
    "pivot",
    "group_by",
    "append_rows",
    "merge_dataset",
)
ROW_OPS = frozenset(
    {
        "filter_rows",
        "drop_duplicates",
        "drop_rows",
        "drop_empty_rows",
        "drop_header_rows",
        "keep_rows",
        "promote_header",
    }
)
# C-59: steps that reshape the table. A child row no longer comes from one parent row (several
# rows from one, one from several, rows from another dataset): the rows are numbered again, the
# preview shows the result instead of a cell-by-cell comparison and the version has no row map.
RESHAPE_OPS = frozenset({"split_rows", "unpivot", "pivot", "group_by", "append_rows"})
# C-57, C-58: steps that read another dataset (its Parquet resolved by the caller).
OTHER_DATASET_OPS = frozenset({"append_rows", "merge_dataset"})
# Steps that name rows by their number in the file (099, 101): tied to one import, they do not
# apply to another file (C-41).
ROW_NUMBER_OPS = frozenset({"label_rows", "set_cells", "drop_rows"})

# --- What a suggestion does to the dataset (096, shared with the Advanced layer of 097) ----------

# `Impact` and `Risk` are defined at the top: `format`: no information changes (spaces, case,
# encoding, HTML, a trailing '.0'); `values`: values change (variants merged, sentinels emptied,
# bounds read, units fixed); `types`: a column's type changes; `rows`: rows removed or merged;
# `columns`: columns dropped, split or renamed; `review`: never ticked by default.

# The effect of a step by its op, when a rule does not say (a rule says it when it knows better:
# B-05 repairs the encoding with a `map_values`, which is `format`, not `values`).
OP_IMPACT: dict[str, Impact] = {
    "rename_column": "columns",
    "drop_columns": "columns",
    "duplicate_column": "columns",
    "assemble_date": "columns",
    "label_rows": "columns",
    "change_type": "types",
    "normalize_spaces": "format",
    "remove_invisible": "format",
    "change_case": "format",
    "replace_values": "values",
    "map_values": "values",
    "set_cells": "values",
    "fill": "values",
    "fill_empty": "values",
    "null_values": "values",
    "null_outside": "values",
    "pad_zeros": "values",
    "transform_number": "values",
    "split_column": "columns",
    "merge_columns": "columns",
    "extract_text": "values",
    "format_text": "values",
    "parse_quantity": "columns",
    "set_where": "values",
    "conditional_column": "columns",
    "computed_column": "columns",
    "swap_values": "values",
    "filter_rows": "rows",
    "drop_duplicates": "rows",
    "entity_column": "columns",  # N36: an entity number, nothing merged
    "drop_rows": "rows",
    "drop_empty_rows": "rows",
    "drop_header_rows": "rows",
    # C1: dates move (values), rows go (rows), the table changes shape (columns: its columns change)
    "shift_date": "values",
    "convert_timezone": "values",
    "keep_rows": "rows",
    "promote_header": "columns",
    "split_rows": "rows",
    "unpivot": "columns",
    "pivot": "columns",
    "group_by": "rows",
    "append_rows": "rows",
    "merge_dataset": "columns",
}


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
