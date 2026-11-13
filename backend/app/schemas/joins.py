"""Join the open dataset with another one (089): proposed keys, preview, new dataset."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.schemas.model import CheckOut, DatasetName, JoinType, ReasonOut

Mode = Literal["lookup", "expand", "summarise"]
Columns = Annotated[list[str], Field(min_length=1, max_length=4)]  # C1: keys of 3 and 4


class JoinSuggestIn(BaseModel):
    other_id: str


class JoinKeyOut(BaseModel):
    left_columns: list[str]
    right_columns: list[str]
    mode: Mode  # the way of joining the counts recommend (R-JN-05)
    cardinality: str  # seen from the open dataset: n:1, 1:1, 1:n, n:m
    confidence: str
    matched_share: float
    reasons: list[ReasonOut]


class JoinColumnOut(BaseModel):
    name: str
    type: str
    default_pick: bool  # R-BLD-02: added by default


class JoinSuggestOut(BaseModel):
    keys: list[JoinKeyOut]
    other_columns: list[JoinColumnOut]  # the columns the lookup can add
    same_project: bool


class MeasureIn(BaseModel):
    column: str
    function: Literal["sum", "mean", "min", "max"]


class JoinIn(BaseModel):
    other_id: str
    left_columns: Columns
    right_columns: Columns
    mode: Mode | None = None  # None: the recommended one
    join_type: JoinType | None = None  # None: the recommended one (R-JOIN-01 to 03)
    columns: list[str] | None = None  # the other dataset's columns to add; None = R-BLD-02
    measures: list[MeasureIn] | None = None  # summarise; None = sum and mean (R-BLD-06)
    name: DatasetName | None = None


class JoinSourceOut(BaseModel):
    id: str
    name: str
    exists: bool


class JoinLineageOut(BaseModel):
    """C1 (R-JN-06): a joined dataset's sources, and why it is out of date."""

    dataset_id: str
    left_id: str
    other_id: str
    left_columns: list[str]
    right_columns: list[str]
    mode: Mode | None
    sources: list[JoinSourceOut]
    out_of_date: list[str]  # empty: current
    can_rebuild: bool
    updated_at: str


class ModeOut(BaseModel):
    mode: Mode
    reason: str | None  # why it cannot be chosen


class JoinPreviewOut(BaseModel):
    mode: Mode
    join_type: JoinType
    rows_before: int
    rows_after: int
    columns: list[str]
    added: list[str]
    rows: list[list[str | None]]
    check: CheckOut
    modes: list[ModeOut]
    reasons: list[ReasonOut]
    warnings: list[ReasonOut]
    refused: list[ReasonOut]
    suggested_name: str
