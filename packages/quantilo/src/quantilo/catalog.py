"""Columns of a dataset (detected types, user corrections) and the catalog the analyses read.

The analyses never know where datasets live: they ask a `Catalog` for a dataset's storage and
its typed columns. The app's catalog keeps them in SQLite; the library's `Dataset` in memory.
Both apply corrections with the same rules (`apply_change`).
"""

import json
from dataclasses import dataclass
from typing import Any, Protocol, cast

import duckdb
from pydantic import BaseModel

from quantilo.errors import ExplorerError
from quantilo.ingestion.sql import lit
from quantilo.typing.conversion import parser_for
from quantilo.typing.summary import summarize
from quantilo.typing.types import AnalysisType, ColumnDetection, Confidence, Parser


class DatasetRef(Protocol):
    """What the analyses need to know about a dataset."""

    @property
    def id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def source_path(self) -> str: ...
    @property
    def storage_path(self) -> str: ...  # the normalized Parquet copy
    @property
    def imported_at(self) -> str | None: ...  # changes when the data changes (cache keys)
    @property
    def row_count(self) -> int | None: ...
    @property
    def size_bytes(self) -> int | None: ...
    @property
    def types_version(self) -> int: ...  # changes when a column's type changes


@dataclass
class ColumnRecord:
    """A column as detected, with the user's corrections (JSON text, as stored by the app)."""

    dataset_id: str
    name: str
    position: int
    storage_dtype: str
    detected_type: str
    detected_parser: str
    candidates: str
    confidence: str
    reasons: str
    detected_role: str | None = None
    detected_order: str | None = None
    user_type: str | None = None
    user_role: str | None = None
    user_parser: str | None = None
    user_order: str | None = None
    excluded: int = 0
    summary: str | None = None


USER_FIELDS = ("user_type", "user_role", "user_parser", "user_order", "excluded")


class ColumnChange(BaseModel):
    """A correction of one column. `None` fields are left unchanged."""

    name: str
    type: AnalysisType | None = None
    role: str | None = None  # "" = no role
    excluded: bool | None = None
    order: list[str] | None = None
    reset: bool = False


@dataclass
class ColumnView:
    name: str
    position: int
    storage_dtype: str
    type: AnalysisType
    role: str | None
    parser: Parser
    order: list[str] | None
    detected_type: AnalysisType
    detected_role: str | None
    confidence: Confidence
    reasons: list[str]
    overridden: bool
    excluded: bool
    summary: dict[str, Any] | None


class Catalog(Protocol):
    """Where the analyses find a dataset and its typed columns."""

    def ready(self, dataset_id: str) -> DatasetRef: ...
    def columns(self, dataset_id: str) -> list[ColumnView]: ...


def _loads(value: str | None) -> Any:
    return json.loads(value) if value else None


def _parser(value: str | None) -> Parser | None:
    return Parser.model_validate_json(value) if value else None


def effective(record: ColumnRecord) -> tuple[AnalysisType, str | None, Parser, list[str] | None]:
    analysis_type = cast(AnalysisType, record.user_type or record.detected_type)
    role = record.detected_role if record.user_role is None else (record.user_role or None)
    parser = _parser(record.user_parser) or Parser.model_validate_json(record.detected_parser)
    order = _loads(record.user_order) or _loads(record.detected_order)
    return analysis_type, role, parser, order


def to_view(record: ColumnRecord) -> ColumnView:
    analysis_type, role, parser, order = effective(record)
    return ColumnView(
        name=record.name,
        position=record.position,
        storage_dtype=record.storage_dtype,
        type=analysis_type,
        role=role,
        parser=parser,
        order=order,
        detected_type=cast(AnalysisType, record.detected_type),
        detected_role=record.detected_role,
        confidence=cast(Confidence, record.confidence),
        reasons=json.loads(record.reasons),
        overridden=any(
            value is not None for value in (record.user_type, record.user_role, record.user_order)
        ),
        excluded=bool(record.excluded),
        summary=_loads(record.summary),
    )


def detection_record(dataset_id: str, detection: ColumnDetection) -> ColumnRecord:
    return ColumnRecord(
        dataset_id=dataset_id,
        name=detection.name,
        position=detection.position,
        storage_dtype=detection.storage_dtype,
        detected_type=detection.type,
        detected_role=detection.role,
        detected_parser=detection.parser.model_dump_json(),
        detected_order=json.dumps(detection.order) if detection.order else None,
        candidates=json.dumps(
            {kind: parser.model_dump() for kind, parser in detection.candidates.items()}
        ),
        confidence=detection.confidence,
        reasons=json.dumps(detection.reasons),
    )


def summarize_record(
    storage_path: str, record: ColumnRecord, connection: duckdb.DuckDBPyConnection | None = None
) -> str:
    """The column's summary (JSON) in its effective type."""
    analysis_type, role, parser, order = effective(record)
    own = connection is None
    duck = connection or duckdb.connect()
    try:
        source = f"read_parquet({lit(storage_path)})"
        summary = summarize(duck, source, record.name, analysis_type, parser, order, role)
    finally:
        if own:
            duck.close()
    return json.dumps(summary)


def apply_change(record: ColumnRecord, change: ColumnChange, valid_roles: set[str]) -> None:
    """Apply a user's correction to a record (in place)."""
    if change.reset:
        record.user_type = record.user_role = record.user_parser = record.user_order = None
    if change.type is not None:
        detected_type = cast(AnalysisType, record.detected_type)
        if change.type == detected_type:
            record.user_type = record.user_parser = None
        else:
            candidates = {
                kind: Parser.model_validate(parser)
                for kind, parser in json.loads(record.candidates).items()
            }
            parser = parser_for(
                change.type,
                candidates,
                detected_type,
                Parser.model_validate_json(record.detected_parser),
            )
            record.user_type = change.type
            record.user_parser = parser.model_dump_json()
    if change.role is not None:
        if change.role and change.role not in valid_roles:
            raise ExplorerError(
                f"Unknown role {change.role!r}", code="invalid_role", status_code=422
            )
        record.user_role = None if change.role == (record.detected_role or "") else change.role
    if change.excluded is not None:
        record.excluded = int(change.excluded)
    if change.order is not None:
        record.user_order = json.dumps(change.order) if change.order else None
