from fastapi import APIRouter, Query, Response
from quantilo.ingestion.preview import ColumnInfo

from app.api.deps import Datasets, Projects
from app.schemas.datasets import (
    DatasetOut,
    HeadOut,
    ImportRequest,
    ImportResponse,
    JobOut,
    RenameRequest,
)
from app.services.datasets.repository import DatasetRecord
from app.services.datasets.service import DatasetService

router = APIRouter(prefix="/datasets", tags=["datasets"])


def to_out(service: DatasetService, record: DatasetRecord) -> DatasetOut:
    job = service.latest_job(record.id) if record.status == "importing" else None
    return DatasetOut.of(record, service.source_state(record), job)


@router.get("")
def list_datasets(service: Datasets) -> list[DatasetOut]:
    return [to_out(service, record) for record in service.list_all()]


@router.post("", status_code=201)
def import_dataset(body: ImportRequest, service: Datasets, projects: Projects) -> ImportResponse:
    if body.project_id is not None:
        projects.get(body.project_id)  # 404 before anything is imported
    record, job = service.start_import(body.path, body.name, body.options)
    if body.project_id is not None:
        projects.move([record.id], body.project_id)
    fresh = service.get(record.id)
    return ImportResponse(
        dataset=to_out(service, fresh),
        job=JobOut.of(service.job(job.id)) if job else None,
        reused=job is None,
    )


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, service: Datasets) -> DatasetOut:
    return to_out(service, service.get(dataset_id))


@router.patch("/{dataset_id}")
def rename_dataset(dataset_id: str, body: RenameRequest, service: Datasets) -> DatasetOut:
    return to_out(service, service.rename(dataset_id, body.name))


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, service: Datasets) -> Response:
    service.delete(dataset_id)
    return Response(status_code=204)


@router.post("/{dataset_id}/reimport", status_code=202)
def reimport_dataset(dataset_id: str, service: Datasets) -> JobOut:
    return JobOut.of(service.job(service.reimport(dataset_id).id))


@router.post("/{dataset_id}/open")
def open_dataset(dataset_id: str, service: Datasets) -> DatasetOut:
    return to_out(service, service.mark_opened(dataset_id))


@router.get("/{dataset_id}/schema")
def dataset_schema(dataset_id: str, service: Datasets) -> list[ColumnInfo]:
    return service.schema(dataset_id)


@router.get("/{dataset_id}/head")
def dataset_head(
    dataset_id: str, service: Datasets, limit: int = Query(20, ge=1, le=500)
) -> HeadOut:
    columns = [column.name for column in service.schema(dataset_id)]
    return HeadOut(columns=columns, rows=service.head(dataset_id, limit))
