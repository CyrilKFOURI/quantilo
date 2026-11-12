from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse
from quantilo.changes.compare import Changes
from quantilo.changes.kpis import KpiChange
from starlette.background import BackgroundTask

from app.services.versions.service import (
    RelationsAnswer,
    RowsAnswer,
    VersionInfo,
    VersionService,
    WordsAnswer,
)

router = APIRouter(tags=["changes"])


def get_version_service(request: Request) -> VersionService:
    return request.app.state.versions


Service = Annotated[VersionService, Depends(get_version_service)]
Versus = Annotated[
    str | None,
    Query(description="Another dataset to compare with (default: the previous version)."),
]
Version = Annotated[
    int | None,
    Query(
        ge=1,
        description="An earlier import of this dataset to compare with (default: the previous "
        "one). Ignored with `vs`.",
    ),
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
    version: Version = None,
) -> Changes:
    """What changed since the previous version (or since dataset `vs`), from the two photos."""
    chosen = [name for name in focus.split(",") if name] if focus else []
    return service.changes(dataset_id, vs, chosen, version)


@router.get("/datasets/{dataset_id}/changes/words")
def words(
    dataset_id: str,
    service: Service,
    column: Annotated[str, Query(min_length=1)],
    vs: Versus = None,
    version: Version = None,
) -> WordsAnswer:
    """059: the words that rise in each version of a text column (both versions' rows needed)."""
    return service.words(dataset_id, column, vs, version)


@router.get("/datasets/{dataset_id}/changes/rows")
def rows(
    dataset_id: str,
    service: Service,
    vs: Versus = None,
    key: Annotated[str | None, Query(min_length=1)] = None,
    version: Version = None,
) -> RowsAnswer:
    """Rows added, removed and modified, joined on `key` (default: the cleaning's row map between
    an original and its cleaned versions, otherwise the first identifier)."""
    return service.rows(dataset_id, vs, key, version)


@router.get("/datasets/{dataset_id}/changes/rows/export", response_class=FileResponse)
def export_rows(
    dataset_id: str,
    service: Service,
    key: Annotated[str | None, Query(min_length=1)] = None,
    vs: Versus = None,
    version: Version = None,
) -> FileResponse:
    file = service.export_rows(dataset_id, vs, key, version)
    return FileResponse(
        file.path,
        media_type="text/csv",
        filename=file.filename,
        background=BackgroundTask(file.remove),
    )


@router.get("/datasets/{dataset_id}/changes/kpis")
def kpis(
    dataset_id: str, service: Service, vs: Versus = None, version: Version = None
) -> list[KpiChange]:
    return service.kpis(dataset_id, vs, version)


@router.get("/datasets/{dataset_id}/changes/relations")
def relations(
    dataset_id: str, service: Service, vs: Versus = None, version: Version = None
) -> RelationsAnswer:
    return service.relations(dataset_id, vs, version)
