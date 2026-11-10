"""Regression (067, 068): the first plane at once, each Advanced section on its own route so the
page asks for it only when the section is opened (onion rule)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from quantilo.regression.analyze import (
    InfluenceTable,
    MoreChecks,
    RegressionReport,
    StandardErrorTable,
    TermsTable,
)
from quantilo.regression.rules import RegressionSuggestion, SeRequest
from quantilo.regression.service import RegressionService
from quantilo.regression.service import Request as RegressionRequest

from app.api.filters import Filters

router = APIRouter(tags=["regression"])


def get_regression_service(request: Request) -> RegressionService:
    return request.app.state.regression


Service = Annotated[RegressionService, Depends(get_regression_service)]


def regression_request(
    target: Annotated[
        str | None, Query(description="Column to explain (default: R-REG-20).")
    ] = None,
    columns: Annotated[
        str | None,
        Query(description="Comma-separated explanatory columns (absent: R-REG-21; empty: none)."),
    ] = None,
    weights: Annotated[
        str | None,
        Query(description="Weight column (absent: a column of role weight; empty: none)."),
    ] = None,
    cluster: Annotated[str | None, Query(description="Grouped by: errors clustered on it.")] = None,
    se: SeRequest = "auto",
    focus: Annotated[str | None, Query(description="The column the conclusion speaks of.")] = None,
) -> RegressionRequest:
    chosen = tuple(name for name in columns.split(",") if name) if columns is not None else None
    return RegressionRequest(target or None, chosen, weights, cluster or None, se, focus or None)


Asked = Annotated[RegressionRequest, Depends(regression_request)]


@router.get("/datasets/{dataset_id}/regression")
def regression(
    dataset_id: str, service: Service, filters: Filters, asked: Asked
) -> RegressionReport:
    """Coefficients with intervals (standard errors chosen by rule), adjusted p-values, quality,
    checks in sentences and the residual plots."""
    return service.regression(dataset_id, asked, filters)


@router.get("/datasets/{dataset_id}/regression/suggestion")
def suggestion(
    dataset_id: str, service: Service, filters: Filters, target: str | None = None
) -> RegressionSuggestion:
    """The target (R-REG-20) and up to 5 columns (R-REG-21) the page starts from, with reasons."""
    return service.suggestion(dataset_id, target or None, filters)


@router.get("/datasets/{dataset_id}/regression/standard-errors")
def standard_errors(
    dataset_id: str, service: Service, filters: Filters, asked: Asked
) -> StandardErrorTable:
    """Advanced > Standard errors: every kind side by side, with what it allows for."""
    found = service.section(dataset_id, "standard_errors", asked, filters)
    assert isinstance(found, StandardErrorTable)
    return found


@router.get("/datasets/{dataset_id}/regression/influence")
def influence(dataset_id: str, service: Service, filters: Filters, asked: Asked) -> InfluenceTable:
    """Advanced > Influence: the 20 most influential rows, with DFBETAS."""
    found = service.section(dataset_id, "influence", asked, filters)
    assert isinstance(found, InfluenceTable)
    return found


@router.get("/datasets/{dataset_id}/regression/terms")
def terms(dataset_id: str, service: Service, filters: Filters, asked: Asked) -> TermsTable:
    """Advanced > Terms: reference levels, columns left out and why, VIF, condition number."""
    found = service.section(dataset_id, "terms", asked, filters)
    assert isinstance(found, TermsTable)
    return found


@router.get("/datasets/{dataset_id}/regression/more")
def more(dataset_id: str, service: Service, filters: Filters, asked: Asked) -> MoreChecks:
    """Advanced > More (computed only when opened): White's test, the fit without the
    influential rows."""
    found = service.section(dataset_id, "more", asked, filters)
    assert isinstance(found, MoreChecks)
    return found
