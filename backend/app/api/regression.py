"""Regression (067, 068): the first plane at once, each Advanced section on its own route so the
page asks for it only when the section is opened (onion rule)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from quantilo.regression.analyze import (
    FamilyAlternatives,
    InfluenceTable,
    MoreChecks,
    RegressionReport,
    StandardErrorTable,
    TermsTable,
)
from quantilo.regression.families import FamilyRequest
from quantilo.regression.iv import IvTable
from quantilo.regression.panel import PanelTable
from quantilo.regression.robust import QuantileProcess, RobustTable
from quantilo.regression.rules import RegressionSuggestion, SeRequest
from quantilo.regression.service import RegressionService
from quantilo.regression.service import Request as RegressionRequest
from quantilo.regression.terms import Coding, PartialPlot

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
    family: Annotated[
        FamilyRequest, Query(description="Model family (default: chosen from the target, 069).")
    ] = "auto",
    exposure: Annotated[
        str | None,
        Query(description="Exposure of a count, its log the offset (absent: role exposure)."),
    ] = None,
    terms: Annotated[
        str,
        Query(
            description=(
                "070: comma-separated terms: spline:x, square:x, cube:x, log:x, interaction:a|b, "
                "trend:date, month:date, weekday:date, reference:column|level."
            )
        ),
    ] = "",
    center: Annotated[
        bool, Query(description="070: numbers centred at their mean (R-REG-TERM-05).")
    ] = False,
    coding: Annotated[
        Coding,
        Query(description="070: categories against a reference level, or their average."),
    ] = "reference",
) -> RegressionRequest:
    chosen = tuple(name for name in columns.split(",") if name) if columns is not None else None
    return RegressionRequest(
        target or None,
        chosen,
        weights,
        cluster or None,
        se,
        focus or None,
        family,
        exposure,
        tuple(term for term in terms.split(",") if term),
        center,
        coding,
    )


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


@router.get("/datasets/{dataset_id}/regression/families")
def families(
    dataset_id: str, service: Service, filters: Filters, asked: Asked
) -> FamilyAlternatives:
    """Advanced > More > Other families (069, computed only when opened): probit, Box-Cox,
    fractional and multinomial logit on the same rows, or why they do not apply."""
    found = service.section(dataset_id, "families", asked, filters)
    assert isinstance(found, FamilyAlternatives)
    return found


@router.get("/datasets/{dataset_id}/regression/panel")
def panel(dataset_id: str, service: Service, filters: Filters, asked: Asked) -> PanelTable:
    """Advanced > Panel (071): fixed effects (entity, period, two-way), first differences,
    between and random effects, clustered by entity; Mundlak chooses, Hausman informs."""
    found = service.section(dataset_id, "panel", asked, filters)
    assert isinstance(found, PanelTable)
    return found


@router.get("/datasets/{dataset_id}/regression/partial")
def partial(
    dataset_id: str,
    service: Service,
    filters: Filters,
    asked: Asked,
    column: Annotated[str, Query(description="The explanatory column to show.")],
) -> PartialPlot:
    """Advanced > Terms > Shape (070, asked for once a column is chosen): the added-variable
    and component + residual plots of one column."""
    return service.partial(dataset_id, asked, column, filters)


@router.get("/datasets/{dataset_id}/regression/robust")
def robust(
    dataset_id: str,
    service: Service,
    filters: Filters,
    asked: Asked,
    quantile: Annotated[
        float, Query(description="072: the quantile, one of 0.1, 0.25, 0.5, 0.75, 0.9.")
    ] = 0.5,
) -> RobustTable:
    """Advanced > Robust & quantile (072): Huber, Tukey and one quantile side by side with least
    squares, and the law and tail of the residuals."""
    return service.robust(dataset_id, asked, quantile, filters)


@router.get("/datasets/{dataset_id}/regression/quantiles")
def quantiles(dataset_id: str, service: Service, filters: Filters, asked: Asked) -> QuantileProcess:
    """Advanced > More > Along the distribution (072, computed only when opened): the
    coefficients at τ = 0.1…0.9, the test of 0.1 against 0.9, Student-t errors."""
    found = service.section(dataset_id, "quantiles", asked, filters)
    assert isinstance(found, QuantileProcess)
    return found


@router.get("/datasets/{dataset_id}/regression/instruments")
def instrumental(
    dataset_id: str,
    service: Service,
    filters: Filters,
    asked: Asked,
    endogenous: Annotated[
        str, Query(description="Comma-separated endogenous columns, among `columns`.")
    ] = "",
    instruments: Annotated[
        str, Query(description="Comma-separated instruments, outside `columns`.")
    ] = "",
) -> IvTable:
    """Advanced > More > Instruments (071): two-stage least squares, only with the endogenous
    columns and instruments the user names (R-REG-IV-01)."""
    return service.instruments(
        dataset_id,
        asked,
        tuple(name for name in endogenous.split(",") if name),
        tuple(name for name in instruments.split(",") if name),
        filters,
    )
