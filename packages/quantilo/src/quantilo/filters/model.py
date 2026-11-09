"""A filter is a list of conditions combined with AND; values of one condition combine with OR."""

import json
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator

from quantilo.errors import ExplorerError
from quantilo.typing.dates import DatePart

Operator = Literal[
    "eq",  # equal (numbers, text, dates)
    "gt",
    "gte",
    "lt",
    "lte",
    "between",  # low <= value <= high (numbers, dates)
    "in",  # one of `values` (categories, booleans, discrete numbers)
    "contains",
    "starts_with",
    "is_null",
    "part_in",  # a date part (`part`) is one of `values`, e.g. day_of_week in [1]
    "unique",  # one row per value of `column` (+ `columns`): the first or last (`keep`)
    "duplicated",  # rows whose value (+ `columns`) appears at least twice; negated: once
]
ROW_OPERATORS = ("unique", "duplicated")  # act on rows as a whole, after the value conditions

MAX_CONDITIONS = 50
MAX_VALUES = 1_000
MAX_KEY_COLUMNS = 10


class Condition(BaseModel):
    column: str = Field(min_length=1)
    op: Operator
    value: str | float | None = None
    low: str | float | None = None
    high: str | float | None = None
    values: list[str] = Field(default_factory=list, max_length=MAX_VALUES)
    part: DatePart | None = None
    negate: bool = False
    # Unique values / Duplicates only: more columns making up the value, which row to keep.
    columns: list[str] = Field(default_factory=list, max_length=MAX_KEY_COLUMNS)
    keep: Literal["first", "last"] = "first"
    order_by: str | None = None  # None: the file order

    @field_validator("values", mode="before")
    @classmethod
    def _as_text(cls, values: object) -> object:
        """Values are text: a JSON true or 2 becomes "True" or "2" (an API client or the
        assistant may send them unquoted)."""
        if isinstance(values, list):
            return [v if isinstance(v, str) else str(v) for v in values]
        return values


Conditions = TypeAdapter(list[Condition])


class InvalidFilterError(ExplorerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="invalid_filter", status_code=422)


def parse_filters(raw: str | None) -> list[Condition]:
    """The `filters` query parameter: a JSON list of conditions (empty or absent = no filter)."""
    if not raw:
        return []
    try:
        conditions = Conditions.validate_python(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as error:
        raise InvalidFilterError(f"Invalid filters: {error}") from error
    if len(conditions) > MAX_CONDITIONS:
        raise InvalidFilterError(f"At most {MAX_CONDITIONS} filters.")
    return conditions


def filters_key(conditions: list[Condition]) -> str:
    """Stable text form, for cache keys."""
    return json.dumps([c.model_dump() for c in conditions], sort_keys=True)
