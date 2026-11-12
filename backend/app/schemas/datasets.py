from typing import Any, Literal

from pydantic import BaseModel, Field
from quantilo.ingestion.options import ImportOptions, SourceOptions

from app.services.datasets.repository import DatasetRecord, JobRecord


class JobOut(BaseModel):
    id: str
    kind: str
    dataset_id: str | None
    status: Literal["pending", "running", "done", "failed"]
    progress: float
    message: str | None
    created_at: str
    finished_at: str | None

    @classmethod
    def of(cls, job: JobRecord) -> "JobOut":
        return cls.model_validate(job, from_attributes=True)


class DatasetOut(BaseModel):
    id: str
    name: str
    source_path: str
    source_kind: Literal["file", "folder"]
    format: str
    status: Literal["importing", "ready", "error"]
    error: str | None
    row_count: int | None
    col_count: int | None
    size_bytes: int | None
    sample_pct: float
    created_at: str
    imported_at: str | None
    last_opened_at: str | None
    types_version: int
    types_reviewed_at: str | None
    source_state: Literal["ok", "changed", "missing"]
    job: JobOut | None = None

    @classmethod
    def of(
        cls,
        record: DatasetRecord,
        source_state: Literal["ok", "changed", "missing"],
        job: JobRecord | None = None,
    ) -> "DatasetOut":
        return cls(
            **{name: getattr(record, name) for name in cls.model_fields if hasattr(record, name)},
            source_state=source_state,
            job=JobOut.of(job) if job else None,
        )


class PreviewRequest(BaseModel):
    path: str
    options: SourceOptions = Field(default_factory=lambda: SourceOptions())


class ImportRequest(BaseModel):
    path: str
    name: str | None = None
    options: ImportOptions = Field(default_factory=lambda: ImportOptions())
    # 087, R-PRJ-05: put the new dataset in this project (an import from an open project only)
    project_id: str | None = None


class ImportResponse(BaseModel):
    dataset: DatasetOut
    job: JobOut | None
    reused: bool


class RenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class UploadOut(BaseModel):
    path: str
    size: int


class SampleOut(BaseModel):
    name: str
    path: str
    format: str
    size: int


class HeadOut(BaseModel):
    columns: list[str]
    rows: list[list[str | None]]


# --- Sources (085): one file imported as several tables ------------------------------------


class TableRequest(BaseModel):
    id: str = Field(min_length=1)  # a table id of the preview (`Preview.tables[].id`), or a range
    name: str | None = Field(default=None, max_length=200)


class SourceImportRequest(BaseModel):
    path: str
    tables: list[TableRequest] = Field(min_length=1, max_length=100)
    name: str | None = None  # the source's name; each table gets "<name> - <table>"
    options: ImportOptions = Field(default_factory=lambda: ImportOptions())
    project_id: str | None = None  # 087, R-PRJ-05: every table goes in this project


class SourceOut(BaseModel):
    id: str
    kind: Literal["workbook", "csv", "json", "jsonl", "xml", "image"]
    name: str
    source_path: str
    size_bytes: int
    created_at: str
    dataset_ids: list[str]


class SourceImportResponse(BaseModel):
    source: SourceOut
    imports: list[ImportResponse]


class LinkOut(BaseModel):
    parent_id: str
    child_id: str
    parent_key: str
    child_key: str
    rule: str


class OriginOut(BaseModel):
    """Where a dataset comes from: standalone (`source` null) or one table of a source."""

    source: SourceOut | None
    locator: str | None
    layout: dict[str, Any] | None  # the table's detection: rules, confidence, summary
    links: list[LinkOut]


class OriginalRequest(BaseModel):
    path: str
    sheet: str | None = None
