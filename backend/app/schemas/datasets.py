from typing import Literal

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
