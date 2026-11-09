from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from quantilo.correlations.matrix import Method
from quantilo.correlations.service import Correlations, CorrelationService

from app.api.filters import Filters

router = APIRouter(tags=["correlations"])


def get_correlation_service(request: Request) -> CorrelationService:
    return request.app.state.correlations


Service = Annotated[CorrelationService, Depends(get_correlation_service)]
Columns = Annotated[
    str | None, Query(description="Comma-separated column names (default: every column).")
]


@router.get("/datasets/{dataset_id}/correlations")
def correlations(
    dataset_id: str,
    service: Service,
    filters: Filters,
    method: Method = "pearson",
    columns: Columns = None,
) -> Correlations:
    chosen = [name for name in columns.split(",") if name] if columns else None
    return service.correlations(dataset_id, method, chosen, filters)
