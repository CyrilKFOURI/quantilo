from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse
from quantilo.rows.export import Format
from quantilo.rows.service import ExportFile, RowService
from quantilo.rows.view import MAX_LIMIT, Mode, RowsPage, View
from starlette.background import BackgroundTask

from app.api.filters import Filters

router = APIRouter(tags=["rows"])

MEDIA_TYPES = {
    "csv": "text/csv",
    "parquet": "application/vnd.apache.parquet",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def get_row_service(request: Request) -> RowService:
    return request.app.state.rows


def view_params(
    request: Request,
    dataset_id: str,
    mode: Mode = "first",
    search: Annotated[str, Query(max_length=200)] = "",
    sort: Annotated[str | None, Query(min_length=1)] = None,
    desc: bool = False,
    pinned: Annotated[
        bool, Query(description="Only the pinned rows found (099 bis, R-PIN-17).")
    ] = False,
    pin_label: Annotated[str | None, Query(max_length=60)] = None,
) -> View:
    rows = None
    if pinned:  # R-PIN-17: the pins found, its own and those followed from its parent
        found = request.app.state.pins.pins(dataset_id)
        rows = tuple(
            p.row for p in found.pins if p.found and (pin_label is None or p.label == pin_label)
        )
    return View(mode, search, sort, desc, rows)


Service = Annotated[RowService, Depends(get_row_service)]
ViewParams = Annotated[View, Depends(view_params)]


def _download(file: ExportFile, media_type: str) -> FileResponse:
    return FileResponse(
        file.path,
        media_type=media_type,
        filename=file.filename,
        background=BackgroundTask(file.remove),
    )


@router.get("/datasets/{dataset_id}/rows")
def rows(
    dataset_id: str,
    service: Service,
    filters: Filters,
    view: ViewParams,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=0, le=MAX_LIMIT)] = 100,
) -> RowsPage:
    return service.rows(dataset_id, view, offset, limit, filters)


@router.get("/datasets/{dataset_id}/export", response_class=FileResponse)
def export(
    dataset_id: str,
    service: Service,
    filters: Filters,
    view: ViewParams,
    format: Format = "csv",
    columns: Annotated[
        str | None, Query(description="Comma-separated column names (default: every column).")
    ] = None,
) -> FileResponse:
    chosen = [name for name in columns.split(",") if name] if columns else None
    file = service.export(dataset_id, format, view, chosen, filters)
    return _download(file, MEDIA_TYPES[format])


@router.get("/datasets/{dataset_id}/export/summary", response_class=FileResponse)
def export_summary(dataset_id: str, service: Service, filters: Filters) -> FileResponse:
    return _download(service.summary(dataset_id, filters), MEDIA_TYPES["csv"])
