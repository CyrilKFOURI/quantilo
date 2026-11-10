"""The model of a project (088): tables, relations, layout, analysis datasets."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

JoinType = Literal["left", "inner", "right", "full"]
Recommended = Literal["left", "inner", "right", "full", "none"]
Cardinality = Literal["1:1", "n:1", "n:m"]
Columns = Annotated[list[str], Field(min_length=1, max_length=2)]
DatasetName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class ReasonOut(BaseModel):
    """A rule code and its sentence: why the app reads something this way."""

    code: str
    text: str


class ModelColumnOut(BaseModel):
    name: str
    type: str
    role: str | None
    # what the card counts it as (R-ROLE-03)
    category: Literal["key", "measure", "attribute", "date", "text", "other"]
    linkable: bool
    excluded_by: ReasonOut | None
    # R-BLD-02: picked by default when this table is a dimension of the analysis dataset
    default_pick: bool


class KeyOut(BaseModel):
    columns: list[str]
    primary: bool
    reasons: list[ReasonOut]


class VersionOut(BaseModel):
    id: str
    seq: int  # 0 = the original


class ModelTableOut(BaseModel):
    id: str  # the dataset the model uses (the original, or a cleaned version)
    root_id: str  # its family's original: relations and positions are filed by it
    name: str
    rows: int
    columns: list[ModelColumnOut]
    keys: list[KeyOut]
    measures: int
    attributes: int
    dates: int
    role: Literal["fact", "dimension", "unrelated"]
    depth: int | None
    primary: bool
    role_reasons: list[ReasonOut]
    versions: list[VersionOut]


class ExampleOut(BaseModel):
    value: str
    rows: int


class CheckOut(BaseModel):
    """Exact counts of one relation (R-JOIN-06)."""

    child_rows: int
    child_empty: int
    matched_rows: int
    orphan_rows: int
    orphans: list[ExampleOut]
    child_distinct: int
    parent_rows: int
    parent_distinct: int
    duplicate_keys: int
    duplicates: list[ExampleOut]
    unused_keys: int
    cardinality: Cardinality
    recommended: Recommended
    reasons: list[ReasonOut]
    # what the chosen (or recommended) join does that may surprise (R-JOIN-01, 05, 06)
    warnings: list[ReasonOut]


class RelationOut(BaseModel):
    # an accepted relation: its id; a proposal: "s-" + a stable key of its columns
    id: str
    status: Literal["accepted", "suggested"]
    origin: Literal["detected", "user"]
    child_id: str
    child_columns: list[str]
    parent_id: str
    parent_columns: list[str]
    cardinality: Cardinality
    join_type: Recommended  # accepted: the user's; suggested: the recommended one
    recommended: Recommended
    confidence: Literal["high", "medium", "low"] | None  # proposals only (R-REL-08)
    score: float | None
    inclusion: float | None
    reasons: list[ReasonOut]
    check: CheckOut | None
    # set when the relation no longer fits the data (a column missing in the version used)
    problem: str | None = None


class PositionIn(BaseModel):
    x: float = Field(ge=-10_000, le=10_000)
    y: float = Field(ge=-10_000, le=10_000)


class BuildOut(BaseModel):
    dataset_id: str
    name: str
    kind: Literal["flat", "aggregate"]
    status: Literal["importing", "ready", "error"]
    rows: int | None
    created_at: str
    updated_at: str
    out_of_date: list[str]  # why it is out of date (R-BLD-05); empty = current
    can_rebuild: bool


class ModelOut(BaseModel):
    project_id: str
    tables: list[ModelTableOut]
    relations: list[RelationOut]
    facts: list[str]  # primary first (R-ROLE-02)
    shape: Literal["star", "snowflake", "none"]
    positions: dict[str, PositionIn]  # by root id
    builds: list[BuildOut]
    # what the detection set aside, with its rule (R-REL-12)
    notes: list[ReasonOut] = Field(default_factory=list)
    elapsed_ms: int


class RelationIn(BaseModel):
    child_id: str
    child_columns: Columns
    parent_id: str
    parent_columns: Columns
    join_type: JoinType | None = None  # None: the recommended one
    status: Literal["accepted", "rejected"] = "accepted"
    origin: Literal["detected", "user"] = "detected"


class RelationPatch(BaseModel):
    join_type: JoinType | None = None
    child_columns: Columns | None = None
    parent_columns: Columns | None = None


class CheckIn(BaseModel):
    child_id: str
    child_columns: Columns
    parent_id: str
    parent_columns: Columns
    join_type: JoinType | None = None


class LayoutIn(BaseModel):
    positions: dict[str, PositionIn] = Field(default_factory=dict, max_length=500)
    versions: dict[str, str] | None = None  # root id → dataset id; None: unchanged


class BuildIn(BaseModel):
    fact_id: str | None = None  # None: the primary fact (R-ROLE-02)
    # dimension (dataset id) → its columns to add; a table left out takes R-BLD-02's defaults
    picks: dict[str, list[str]] = Field(default_factory=dict)
    name: DatasetName | None = None


class OutputColumnOut(BaseModel):
    name: str
    table_id: str
    column: str


class StepOut(BaseModel):
    relation_id: str
    child_id: str
    parent_id: str
    prefix: str
    join_type: JoinType
    columns: int
    check: CheckOut


class PreviewOut(BaseModel):
    fact_id: str
    rows_before: int
    rows_after: int
    columns: list[OutputColumnOut]
    steps: list[StepOut]
    refused: list[ReasonOut]
    skipped: list[ReasonOut]
    name: str  # the default name of the analysis dataset


class MeasureIn(BaseModel):
    column: str
    function: Literal["sum", "mean", "min", "max"]


class AggregateIn(BaseModel):
    relation_id: str
    measures: list[MeasureIn] | None = None  # None: sum and mean of the child's measures
    name: DatasetName | None = None


class CreatedOut(BaseModel):
    dataset_id: str
