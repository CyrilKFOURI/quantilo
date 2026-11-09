from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse
from quantilo.changes.compare import Changes
from quantilo.changes.kpis import KpiChange
from starlette.background import BackgroundTask

from app.services.versions.service import RelationsAnswer, RowsAnswer, VersionInfo, VersionService

router = APIRouter(tags=["changes"])


def get_version_service(request: Request) -> VersionService:
    return request.app.state.versions


Service = Annotated[VersionService, Depends(get_version_service)]
Versus = Annotated[
    str | None,
    Query(description="Another dataset to compare with (default: the previous version)."),
]


@router.get("/datasets/{dataset_id}/versions")
def versions(dataset_id: str, service: Service) -> list[VersionInfo]:
    """Every imported version, newest first."""
    return service.versions(dataset_id)


@router.get("/datasets/{dataset_id}/changes")
def changes(
    dataset_id: str,
    service: Service,
    vs: Versus = None,
    focus: Annotated[
        str | None, Query(description="Comma-separated columns to list first.")
    ] = None,
) -> Changes:
    """What changed since the previous version (or since dataset `vs`), from the two photos."""
    chosen = [name for name in focus.split(",") if name] if focus else []
    return service.changes(dataset_id, vs, chosen)


@router.get("/datasets/{dataset_id}/changes/rows")
def rows(
    dataset_id: str,
    service: Service,
    vs: Versus = None,
    key: Annotated[str | None, Query(min_length=1)] = None,
) -> RowsAnswer:
    """Rows added, removed and modified, joined on `key` (default: the first identifier)."""
    return service.rows(dataset_id, vs, key)


@router.get("/datasets/{dataset_id}/changes/rows/export", response_class=FileResponse)
def export_rows(
    dataset_id: str,
    service: Service,
    key: Annotated[str, Query(min_length=1)],
    vs: Versus = None,
) -> FileResponse:
    file = service.export_rows(dataset_id, vs, key)
    return FileResponse(
        file.path,
        media_type="text/csv",
        filename=file.filename,
        background=BackgroundTask(file.remove),
    )


@router.get("/datasets/{dataset_id}/changes/kpis")
def kpis(dataset_id: str, service: Service, vs: Versus = None) -> list[KpiChange]:
    return service.kpis(dataset_id, vs)


@router.get("/datasets/{dataset_id}/changes/relations")
def relations(dataset_id: str, service: Service, vs: Versus = None) -> RelationsAnswer:
    return service.relations(dataset_id, vs)
