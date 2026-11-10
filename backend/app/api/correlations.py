from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from quantilo.compare.multiple import Correction
from quantilo.correlations.matrix import Method
from quantilo.correlations.service import Correlations, CorrelationService
from quantilo.correlations.structure import Factors, Structure

from app.api.filters import Filters

router = APIRouter(tags=["correlations"])


def get_correlation_service(request: Request) -> CorrelationService:
    return request.app.state.correlations


Service = Annotated[CorrelationService, Depends(get_correlation_service)]
Factor = Annotated[
    int | None, Query(ge=1, le=30, description="Factors (default: the parallel analysis' count).")
]
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
    correction: Correction = "auto",
) -> Correlations:
    """The association matrix; `correction` adjusts the pairs' p-values (auto: Benjamini-Hochberg
    from 10 pairs, Holm below)."""
    chosen = [name for name in columns.split(",") if name] if columns else None
    return service.correlations(dataset_id, method, chosen, filters, correction)


@router.get("/datasets/{dataset_id}/correlations/structure")
def structure(
    dataset_id: str, service: Service, filters: Filters, columns: Columns = None
) -> Structure:
    """How many real dimensions the numeric columns carry (078): standardised PCA, components
    kept by Horn's parallel analysis, loadings, scree and a sampled biplot."""
    chosen = [name for name in columns.split(",") if name] if columns else None
    return service.structure(dataset_id, chosen, filters)


@router.get("/datasets/{dataset_id}/correlations/factors")
def factors(
    dataset_id: str,
    service: Service,
    filters: Filters,
    columns: Columns = None,
    k: Factor = None,
) -> Factors:
    """Varimax factor analysis, communalities, KMO and Bartlett's test (078, Advanced > More)."""
    chosen = [name for name in columns.split(",") if name] if columns else None
    return service.factors(dataset_id, chosen, filters, k)
