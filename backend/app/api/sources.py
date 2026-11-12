import json
from pathlib import Path

from fastapi import APIRouter, Query, Request, Response
from quantilo.ingestion.browse import Listing
from quantilo.ingestion.original import Original
from quantilo.ingestion.preview import Preview
from quantilo.ingestion.sniff import sniff_path
from quantilo.ingestion.sources import detect_format

from app.api.deps import Datasets, Projects
from app.core.errors import AppError
from app.schemas.datasets import (
    ImportResponse,
    JobOut,
    LinkOut,
    OriginalRequest,
    OriginOut,
    PreviewRequest,
    SampleOut,
    SourceImportRequest,
    SourceImportResponse,
    SourceOut,
    UploadOut,
)
from app.services.datasets.service import DatasetService
from app.services.datasets.sources import SourceRecord

router = APIRouter(tags=["sources"])


def source_out(record: SourceRecord, dataset_ids: list[str]) -> SourceOut:
    return SourceOut(
        id=record.id,
        kind=record.kind,
        name=record.name,
        source_path=record.source_path,
        size_bytes=record.size_bytes,
        created_at=record.created_at,
        dataset_ids=dataset_ids,
    )


@router.post("/sources/preview")
def preview_source(body: PreviewRequest, service: Datasets) -> Preview:
    return service.preview(body.path, body.options)


@router.post("/sources/original")
def source_original(body: OriginalRequest, service: Datasets) -> Original:
    """080b / 085: the left pane of the import screen (a sheet's cells, a document's lines)."""
    return service.original(body.path, body.sheet)


@router.post("/sources/import", status_code=201)
def import_tables(
    body: SourceImportRequest, service: Datasets, projects: Projects
) -> SourceImportResponse:
    """080b / 085: the chosen tables of one file, imported as one source (one dataset each)."""
    from app.api.datasets import to_out

    if body.project_id is not None:
        projects.get(body.project_id)  # 404 before anything is imported
    source, started = service.import_tables(
        body.path, [(table.id, table.name) for table in body.tables], body.options, body.name
    )
    if body.project_id is not None:
        projects.move([record.id for record, _ in started], body.project_id)
    imports = [
        ImportResponse(
            dataset=to_out(service, service.get(record.id)),
            job=JobOut.of(service.job(job.id)) if job else None,
            reused=job is None,
        )
        for record, job in started
    ]
    return SourceImportResponse(
        source=source_out(source, [record.id for record, _ in started]), imports=imports
    )


@router.get("/sources")
def list_sources(service: Datasets) -> list[SourceOut]:
    return [source_out(record, ids) for record, ids in service.list_sources()]


@router.get("/sources/{source_id}")
def get_source(source_id: str, service: Datasets) -> SourceOut:
    return source_out(service.get_source(source_id), service.source_datasets(source_id))


@router.delete("/sources/{source_id}", status_code=204)
def delete_source(source_id: str, service: Datasets) -> Response:
    service.delete_source(source_id)
    return Response(status_code=204)


def origin_out(service: DatasetService, dataset_id: str) -> OriginOut:
    source, siblings, record, links = service.origin(dataset_id)
    return OriginOut(
        source=source_out(source, siblings) if source else None,
        locator=record.source_locator,
        layout=json.loads(record.layout) if record.layout else None,
        links=[LinkOut(**link.__dict__) for link in links],
    )


@router.get("/datasets/{dataset_id}/origin")
def dataset_origin(dataset_id: str, service: Datasets) -> OriginOut:
    return origin_out(service, dataset_id)


@router.get("/browse")
def browse(service: Datasets, path: str | None = None) -> Listing:
    return service.browse(path)


@router.get("/samples")
def list_samples(service: Datasets) -> list[SampleOut]:
    return [
        SampleOut(name=file.path.stem, path=str(file.path), format=file.format, size=file.size)
        for file in service.samples()
    ]


@router.put("/uploads", status_code=201)
async def upload(
    request: Request, service: Datasets, filename: str = Query(min_length=1)
) -> UploadOut:
    """Receive a file as the raw request body (no multipart), stored in the workspace."""
    if detect_format(Path(filename)) is None:
        raise AppError(f"Unsupported file type: {filename}", code="unsupported", status_code=415)
    limit = service.settings.max_upload_mb * 1024 * 1024
    target = service.new_upload_path(filename)
    size = 0
    with target.open("wb") as handle:
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                handle.close()
                target.unlink(missing_ok=True)
                raise AppError(
                    f"File too large (limit {service.settings.max_upload_mb} MB).",
                    code="too_large",
                    status_code=413,
                )
            handle.write(chunk)
    if size == 0:
        target.unlink(missing_ok=True)
        raise AppError("The file is empty.", code="empty", status_code=422)
    found = sniff_path(target)
    if found.kind == "unsupported" and found.format != "xls":
        # 082, R-DET-03: named by its first bytes, never kept (a legacy .xls is explained later)
        target.unlink(missing_ok=True)
        raise AppError(f"{filename}: {found.reason}", code="unsupported", status_code=415)
    return UploadOut(path=str(target), size=size)
