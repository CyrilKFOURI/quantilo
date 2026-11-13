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
from quantilo.regression.crosssection import ComponentsTable, CrossSectionTable, OutputsTable
from quantilo.regression.families import FamilyRequest
from quantilo.regression.iv import IvTable
from quantilo.regression.methods import MethodKind, MoreMethods
from quantilo.regression.mixed import MixedTable
from quantilo.regression.panel import PanelTable
from quantilo.regression.physics import CurvesTable, MeasurementTable
from quantilo.regression.resampling import ResamplingTable
from quantilo.regression.robust import QuantileProcess, RobustTable
from quantilo.regression.rules import RegressionSuggestion, SeRequest
from quantilo.regression.selection import SelectionTable
from quantilo.regression.service import MethodOptions, MethodResult, RegressionService
from quantilo.regression.service import Request as RegressionRequest
from quantilo.regression.stability import StabilityTable
from quantilo.regression.survival import CoxTable, TobitTable
from quantilo.regression.terms import Coding, PartialPlot
from quantilo.regression.varying import RegimesTable, VaryingTable

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


@router.get("/datasets/{dataset_id}/regression/stability")
def stability(dataset_id: str, service: Service, filters: Filters, asked: Asked) -> StabilityTable:
    """Advanced > Stability over time (073): CUSUM and CUSUM of squares of the recursive
    residuals, a Chow test at the strongest date, rolling coefficients, the factor reading."""
    found = service.section(dataset_id, "stability", asked, filters)
    assert isinstance(found, StabilityTable)
    return found


@router.get("/datasets/{dataset_id}/regression/resampling")
def resampling(
    dataset_id: str, service: Service, filters: Filters, asked: Asked
) -> ResamplingTable:
    """Advanced > Resampling (077): bootstrap of rows, clusters or blocks, wild cluster
    bootstrap with few clusters, Freedman-Lane permutation of one coefficient."""
    found = service.section(dataset_id, "resampling", asked, filters)
    assert isinstance(found, ResamplingTable)
    return found


@router.get("/datasets/{dataset_id}/regression/selection")
def selection(dataset_id: str, service: Service, filters: Filters, asked: Asked) -> SelectionTable:
    """Advanced > More > Selection (077, computed only when opened): plug-in lasso, post-lasso,
    double selection of the focus column, stability selection."""
    found = service.section(dataset_id, "selection", asked, filters)
    assert isinstance(found, SelectionTable)
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


# --- Advanced > More: 074 (cross-section), 075 (physics), 076 (health) ---------------------------


@router.get("/datasets/{dataset_id}/regression/methods")
def methods(dataset_id: str, service: Service, filters: Filters, asked: Asked) -> MoreMethods:
    """Advanced > More: the methods offered for these data, each with the rule that proposes it
    and its reason, and the columns the rules picked (uncertainty, event, groups). Cheap: each
    method is computed by its own route when its fold is opened."""
    return service.methods(dataset_id, asked, filters)


def _method(
    service: RegressionService,
    dataset_id: str,
    kind: MethodKind,
    asked: RegressionRequest,
    filters: Filters,
    options: MethodOptions | None = None,
) -> MethodResult:
    return service.method(dataset_id, kind, asked, options, filters)


@router.get("/datasets/{dataset_id}/regression/cross-section")
def cross_section(
    dataset_id: str, service: Service, filters: Filters, asked: Asked
) -> CrossSectionTable:
    """Advanced > More > Fama-MacBeth and IC (074): one regression a period, the mean of the
    coefficients with Newey-West errors, the IC and ICIR of each column."""
    found = _method(service, dataset_id, "cross_section", asked, filters)
    assert isinstance(found, CrossSectionTable)
    return found


@router.get("/datasets/{dataset_id}/regression/outputs")
def outputs(
    dataset_id: str,
    service: Service,
    filters: Filters,
    asked: Asked,
    outputs: Annotated[
        str, Query(description="074: comma-separated other outcomes, numeric, at most 10.")
    ] = "",
) -> OutputsTable:
    """Advanced > More > Several outcomes (074): the coefficient matrix, Wilks and Pillai, the GRS
    test of the alphas for dated returns."""
    options = MethodOptions(outputs=tuple(name for name in outputs.split(",") if name))
    found = _method(service, dataset_id, "outputs", asked, filters, options)
    assert isinstance(found, OutputsTable)
    return found


@router.get("/datasets/{dataset_id}/regression/components")
def components(
    dataset_id: str, service: Service, filters: Filters, asked: Asked
) -> ComponentsTable:
    """Advanced > More > Principal components (074): the numeric columns through their principal
    components, the coefficients back in the units of the columns."""
    found = _method(service, dataset_id, "components", asked, filters)
    assert isinstance(found, ComponentsTable)
    return found


Sigma = Annotated[
    str | None,
    Query(
        description="075: the uncertainty of the target (absent: role uncertainty; empty: none)."
    ),
]
SigmaX = Annotated[
    str | None,
    Query(description="075: the uncertainty of x (absent: role uncertainty; empty: none)."),
]


@router.get("/datasets/{dataset_id}/regression/measurement")
def measurement(
    dataset_id: str,
    service: Service,
    filters: Filters,
    asked: Asked,
    sigma: Sigma = None,
    sigma_x: SigmaX = None,
    rho: Annotated[
        str | None,
        Query(description="075: the correlation of each row's errors on x and y (York)."),
    ] = None,
) -> MeasurementTable:
    """Advanced > More > Uncertainties and errors on x (075): the fit by 1/sigma^2, chi^2, pulls;
    Deming, York (errors correlated by `rho`) and total least squares."""
    options = MethodOptions(sigma=sigma, sigma_x=sigma_x, rho=rho)
    found = _method(service, dataset_id, "measurement", asked, filters, options)
    assert isinstance(found, MeasurementTable)
    return found


@router.get("/datasets/{dataset_id}/regression/curves")
def curves(
    dataset_id: str,
    service: Service,
    filters: Filters,
    asked: Asked,
    sigma: Sigma = None,
    formula: Annotated[
        str | None,
        Query(description="075: a curve of x typed by the user, e.g. a * exp(b * x) + c."),
    ] = None,
) -> CurvesTable:
    """Advanced > More > Curve (075): exponential, power, logistic, Michaelis-Menten and Gaussian
    peak, ranked by AIC with the straight line, and the user's formula."""
    options = MethodOptions(sigma=sigma, formula=formula or None)
    found = _method(service, dataset_id, "curves", asked, filters, options)
    assert isinstance(found, CurvesTable)
    return found


@router.get("/datasets/{dataset_id}/regression/survival")
def survival(
    dataset_id: str,
    service: Service,
    filters: Filters,
    asked: Asked,
    event: Annotated[
        str | None, Query(description="076: the yes/no event column (absent: R-SRV-01).")
    ] = None,
    strata: Annotated[
        str | None, Query(description="076: a category whose levels get their own baseline.")
    ] = None,
    start: Annotated[
        str | None,
        Query(description="076: the start of each row's interval (absent: R-REG-SURV-04)."),
    ] = None,
) -> CoxTable:
    """Advanced > More > Survival (076): the Cox model of the target as a time to the event,
    hazard ratios, the Schoenfeld test of proportional hazards, strata, intervals and adjusted
    survival curves."""
    options = MethodOptions(event=event, strata=strata, start=start)
    found = _method(service, dataset_id, "survival", asked, filters, options)
    assert isinstance(found, CoxTable)
    return found


@router.get("/datasets/{dataset_id}/regression/tobit")
def tobit(dataset_id: str, service: Service, filters: Filters, asked: Asked) -> TobitTable:
    """Advanced > More > Censored (076): the Tobit model of a target held at a bound."""
    found = _method(service, dataset_id, "tobit", asked, filters)
    assert isinstance(found, TobitTable)
    return found


@router.get("/datasets/{dataset_id}/regression/mixed")
def mixed(
    dataset_id: str,
    service: Service,
    filters: Filters,
    asked: Asked,
    slope: Annotated[
        str | None, Query(description="076: a numeric term whose slope varies by group.")
    ] = None,
) -> MixedTable:
    """Advanced > More > Mixed model (076): a random intercept by Grouped by (or the entity of the
    panel), its share of the variance, the test against least squares."""
    found = _method(service, dataset_id, "mixed", asked, filters, MethodOptions(slope=slope))
    assert isinstance(found, MixedTable)
    return found


@router.get("/datasets/{dataset_id}/regression/varying")
def varying(dataset_id: str, service: Service, filters: Filters, asked: Asked) -> VaryingTable:
    """Advanced > More > Coefficients that drift (073): each coefficient a random walk, Kalman
    smoother, tested against fixed coefficients."""
    found = _method(service, dataset_id, "varying", asked, filters)
    assert isinstance(found, VaryingTable)
    return found


@router.get("/datasets/{dataset_id}/regression/regimes")
def regimes(
    dataset_id: str,
    service: Service,
    filters: Filters,
    asked: Asked,
    regimes: Annotated[int, Query(ge=2, le=3, description="073: 2 or 3 regimes.")] = 2,
) -> RegimesTable:
    """Advanced > More > Regimes (073): Markov-switching regression, each regime its own
    coefficients (and spread of the errors when the BIC says so)."""
    found = _method(service, dataset_id, "regimes", asked, filters, MethodOptions(regimes=regimes))
    assert isinstance(found, RegimesTable)
    return found
