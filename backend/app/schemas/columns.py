from typing import Literal

from pydantic import BaseModel
from quantilo.typing.knowledge_base import load
from quantilo.typing.types import AnalysisType, Parser

from app.services.columns.service import ColumnChange, ColumnView


class HistogramBar(BaseModel):
    label: str
    count: int


class DateProfile(BaseModel):
    has_time: bool
    min: str | None
    max: str | None
    step: str
    parts: list[str]


class ColumnSummary(BaseModel):
    rows: int
    null_count: int
    invalid_count: int
    distinct_count: int
    examples: list[str]
    histogram: list[HistogramBar]
    date: DateProfile | None = None


class ColumnOut(BaseModel):
    name: str
    position: int
    storage_dtype: str
    type: AnalysisType
    role: str | None
    role_label: str | None
    parser: Parser
    order: list[str] | None
    detected_type: AnalysisType
    detected_role: str | None
    confidence: Literal["high", "medium", "low"]
    reasons: list[str]
    overridden: bool
    excluded: bool
    summary: ColumnSummary | None

    @classmethod
    def of(cls, view: ColumnView) -> "ColumnOut":
        role = load().role(view.role)
        return cls(
            **{name: getattr(view, name) for name in cls.model_fields if hasattr(view, name)},
            role_label=role.label if role else None,
        )


class ColumnChangesRequest(BaseModel):
    changes: list[ColumnChange]


class ConversionOut(BaseModel):
    type: AnalysisType
    parser: Parser
    summary: ColumnSummary


class RoleOut(BaseModel):
    id: str
    label: str
