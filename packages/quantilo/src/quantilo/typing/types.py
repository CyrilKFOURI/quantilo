from dataclasses import dataclass, field
from typing import Literal, get_args

from pydantic import BaseModel

AnalysisType = Literal[
    "numeric_continuous",
    "numeric_discrete",
    "categorical_nominal",
    "categorical_ordinal",
    "boolean",
    "datetime",
    "text",
    "identifier",
    "ignored",
]
ANALYSIS_TYPES: tuple[AnalysisType, ...] = get_args(AnalysisType)
NUMERIC_TYPES: tuple[AnalysisType, ...] = ("numeric_continuous", "numeric_discrete")
# Roles of columns that name a thing rather than measure it: detected like identifiers, never a
# metric (048, R-ROLE-02).
KEY_ROLES: frozenset[str] = frozenset({"identifier", "customer_key", "order_key"})
TEXT_LIKE_TYPES: tuple[AnalysisType, ...] = (
    "categorical_nominal",
    "categorical_ordinal",
    "text",
    "identifier",
)

Confidence = Literal["high", "medium", "low"]


class TopicRule(BaseModel):
    """A topic as the Topic rule reads it (059, R-TOP-05): its words and their weights."""

    label: str
    words: dict[str, float]


class DerivedSpec(BaseModel):
    """What a derived column is computed from (058): a source column and a part of it."""

    source: str
    part: str  # "length", "words", ..., "has" (with `option`), "count", "topic"
    option: str | None = None  # a list of choices: the option of "has <option>"
    topics: list[TopicRule] = []  # "topic": the topics, in order


class Parser(BaseModel):
    """How to read a stored value in its analysis type, applied at query time.

    - raw: the stored value as is
    - number: text with a given decimal separator (thousands separators, %, currency stripped)
    - date: text with a strptime format
    - boolean: text mapped with true/false vocabularies (empty lists: cast the value)
    - year: integer year → date (January 1st)
    - text: the value as text, null tokens removed
    - time: a time of day (TIME, or text with a strptime `format`) → hours since midnight (092)
    - derived: a value computed from another column (`derived`, 058): the length of a text...

    A date read with two formats (092, R-TYP-06) lists them in `formats`, the first one in `format`.
    """

    kind: Literal["raw", "number", "date", "boolean", "year", "text", "time", "derived"] = "raw"
    decimal: Literal[".", ","] = "."
    format: str | None = None
    formats: list[str] = []
    true_values: list[str] = []
    false_values: list[str] = []
    derived: DerivedSpec | None = None


@dataclass
class ColumnDetection:
    name: str
    position: int
    storage_dtype: str
    type: AnalysisType
    confidence: Confidence
    reasons: list[str]
    parser: Parser
    role: str | None = None
    order: list[str] | None = None
    # Best parser found for each kind, to switch type later without re-scanning.
    candidates: dict[str, Parser] = field(default_factory=dict)
