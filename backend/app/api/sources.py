from pathlib import Path

from fastapi import APIRouter, Query, Request
from quantilo.ingestion.browse import Listing
from quantilo.ingestion.preview import Preview
from quantilo.ingestion.sources import detect_format

from app.api.deps import Datasets
from app.core.errors import AppError
from app.schemas.datasets import PreviewRequest, SampleOut, UploadOut

router = APIRouter(tags=["sources"])


@router.post("/sources/preview")
def preview_source(body: PreviewRequest, service: Datasets) -> Preview:
    return service.preview(body.path, body.options)


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
    return UploadOut(path=str(target), size=size)
