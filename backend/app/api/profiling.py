from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from quantilo.charts.prepare import Chart, ChartParams
from quantilo.profiling.overview import Overview
from quantilo.profiling.service import ProfilingService

from app.api.filters import Filters
from app.schemas.columns import ColumnSummary

router = APIRouter(tags=["profiling"])


def get_profiling_service(request: Request) -> ProfilingService:
    return request.app.state.profiling


Profiling = Annotated[ProfilingService, Depends(get_profiling_service)]


@router.get("/datasets/{dataset_id}/overview")
def overview(dataset_id: str, service: Profiling, filters: Filters) -> Overview:
    return service.overview(dataset_id, filters)


@router.get("/datasets/{dataset_id}/columns/chart")
def chart(
    dataset_id: str,
    service: Profiling,
    filters: Filters,
    column: Annotated[str, Query(min_length=1)],
    view: str | None = None,
    top_n: Annotated[int, Query(ge=1, le=50)] = 10,
    full: bool = False,
    step: str | None = None,
    bins: Annotated[int | None, Query(ge=5, le=200)] = None,
) -> Chart:
    return service.chart(dataset_id, column, ChartParams(view, top_n, full, step, bins), filters)


@router.get("/datasets/{dataset_id}/columns/summaries")
def summaries(dataset_id: str, service: Profiling, filters: Filters) -> dict[str, ColumnSummary]:
    """Column summaries over the filtered rows (types and detection are not affected)."""
    return {
        name: ColumnSummary.model_validate(summary)
        for name, summary in service.summaries(dataset_id, filters).items()
    }
