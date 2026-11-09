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
TEXT_LIKE_TYPES: tuple[AnalysisType, ...] = (
    "categorical_nominal",
    "categorical_ordinal",
    "text",
    "identifier",
)

Confidence = Literal["high", "medium", "low"]


class Parser(BaseModel):
    """How to read a stored value in its analysis type, applied at query time.

    - raw: the stored value as is
    - number: text with a given decimal separator (thousands separators, %, currency stripped)
    - date: text with a strptime format
    - boolean: text mapped with true/false vocabularies (empty lists: cast the value)
    - year: integer year → date (January 1st)
    - text: the value as text, null tokens removed
    """

    kind: Literal["raw", "number", "date", "boolean", "year", "text"] = "raw"
    decimal: Literal[".", ","] = "."
    format: str | None = None
    true_values: list[str] = []
    false_values: list[str] = []


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
