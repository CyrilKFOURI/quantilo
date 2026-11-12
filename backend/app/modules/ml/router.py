import json
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import Response
from pydantic import TypeAdapter, ValidationError
from quantilo.filters.model import Condition, parse_filters
from quantilo.ml import catalog
from quantilo.ml.classes import Classes, TargetDistribution
from quantilo.ml.engine import Catalogue, NeutralizeRequest, Targets
from quantilo.ml.evaluate import Validation
from quantilo.ml.panel import Neutralized, PanelCheck
from quantilo.ml.preparation import Choices
from quantilo.ml.target import Task
from quantilo.ml.train import Estimate, PreparationPreview

from app.modules.ml import errors, interpret
from app.modules.ml.service import (
    CompareJob,
    CompareRequest,
    ExplanationState,
    MlError,
    MlService,
    Run,
    RunSummary,
    TrainingJob,
    TuneJob,
    TuneStart,
)

router = APIRouter(tags=["module:ml"])


def get_service(request: Request) -> MlService:
    return request.app.state.ml


def active_filters(
    filters: Annotated[str | None, Query(max_length=100_000)] = None,
) -> list[Condition]:
    return parse_filters(filters)


Service = Annotated[MlService, Depends(get_service)]
Filters = Annotated[list[Condition], Depends(active_filters)]


@router.get("/datasets/{dataset_id}/targets")
def targets(dataset_id: str, service: Service, filters: Filters) -> Targets:
    """Every column as a possible target: the problem it defines, or why it cannot be one."""
    return service.targets(dataset_id, filters)


@router.get("/datasets/{dataset_id}/panel")
def panel(
    dataset_id: str,
    service: Service,
    filters: Filters,
    target: str | None = None,
    period: str | None = None,
    entity: str | None = None,
) -> PanelCheck:
    """Whether the rows look like a panel (entities by periods): its period and entity, found or
    as given, and the embargo the target calls for."""
    return service.panel(dataset_id, filters, target, period, entity)


@router.get("/catalog")
def models(
    service: Service,
    task: Task,
    rows: Annotated[int, Query(ge=0, description="Training rows.")] = 1_000,
    minority: Annotated[float | None, Query(ge=0, le=1)] = None,
    features: Annotated[int, Query(ge=0)] = 10,
    categorical_share: Annotated[float, Query(ge=0, le=1)] = 0.0,
    missing_share: Annotated[float, Query(ge=0, le=1)] = 0.0,
) -> Catalogue:
    """The models that handle a task: settings with data-dependent defaults, rules between
    settings, badges, limits, and two or three recommendations for this data."""
    context = catalog.Context(rows, task, minority, features, categorical_share, missing_share)
    return service.catalogue(context)


@router.post("/datasets/{dataset_id}/runs", status_code=202)
def train(
    dataset_id: str,
    service: Service,
    filters: Filters,
    target: Annotated[str, Query(min_length=1)],
    model: str | None = None,
    settings: Annotated[str | None, Query(description="JSON object of settings.")] = None,
    choices: Annotated[Choices | None, Body(description="Preparation choices.")] = None,
    classes: Annotated[str | None, Query(description="JSON: a number cut into classes.")] = None,
    validation: Annotated[str | None, Query(description="JSON: how the model is judged.")] = None,
) -> TrainingJob:
    """Train a model in the background (defaults for everything not given)."""
    return service.start(
        dataset_id,
        target,
        model,
        _settings(settings),
        filters,
        choices,
        _classes(classes),
        _validation(validation),
    )


@router.post("/datasets/{dataset_id}/preparation")
def preparation(
    dataset_id: str,
    service: Service,
    filters: Filters,
    target: Annotated[str, Query(min_length=1)],
    model: str | None = None,
    settings: Annotated[str | None, Query(description="JSON object of settings.")] = None,
    choices: Annotated[Choices | None, Body(description="Preparation choices.")] = None,
    classes: Annotated[str | None, Query(description="JSON: a number cut into classes.")] = None,
    validation: Annotated[str | None, Query(description="JSON: how the model is judged.")] = None,
) -> PreparationPreview:
    """The decisions the next training would take, and its first 20 training rows prepared."""
    return service.preview(
        dataset_id,
        target,
        model,
        _settings(settings),
        filters,
        choices,
        _classes(classes),
        _validation(validation),
    )


@router.get("/datasets/{dataset_id}/targets/{column}/distribution")
def distribution(
    dataset_id: str, column: str, service: Service, filters: Filters
) -> TargetDistribution:
    """Quantiles, histogram and sample of a numeric target, to cut it into classes."""
    return service.distribution(dataset_id, column, filters)


CLASSES = TypeAdapter(Classes)
VALIDATION = TypeAdapter(Validation)


def _validation(validation: str | None) -> Validation | None:
    if not validation:
        return None
    try:
        return VALIDATION.validate_json(validation)
    except ValidationError as error:
        message = error.errors()[0].get("msg", "Invalid validation.")
        raise MlError(f"Validation: {message}", "bad_validation") from None


def _classes(classes: str | None) -> Classes | None:
    if not classes:
        return None
    try:
        return CLASSES.validate_json(classes)
    except ValidationError as error:
        message = error.errors()[0].get("msg", "Invalid classes.")
        raise MlError(f"Classes: {message}", "bad_classes") from None


def _settings(settings: str | None) -> dict[str, Any] | None:
    try:
        values: dict[str, Any] | None = json.loads(settings) if settings else None
    except json.JSONDecodeError:
        raise MlError("Settings must be a JSON object.", "bad_settings") from None
    if values is not None and not isinstance(values, dict):
        raise MlError("Settings must be a JSON object.", "bad_settings")
    return values


@router.post("/datasets/{dataset_id}/compare", status_code=202)
def compare(
    dataset_id: str,
    service: Service,
    filters: Filters,
    request: CompareRequest,
    target: Annotated[str, Query(min_length=1)],
    classes: Annotated[str | None, Query(description="JSON: a number cut into classes.")] = None,
    validation: Annotated[str | None, Query(description="JSON: how the models are judged.")] = None,
) -> CompareJob:
    """Train several models one after the other on the same rows and validation."""
    return service.compare(
        dataset_id, target, request, filters, _classes(classes), _validation(validation)
    )


@router.get("/compare-jobs/{job_id}")
def comparison(job_id: str, service: Service) -> CompareJob:
    return service.comparison(job_id)


@router.post("/datasets/{dataset_id}/tune", status_code=202)
def tune(
    dataset_id: str,
    service: Service,
    filters: Filters,
    request: TuneStart,
    target: Annotated[str, Query(min_length=1)],
    classes: Annotated[str | None, Query(description="JSON: a number cut into classes.")] = None,
    validation: Annotated[str | None, Query(description="JSON: how the model is judged.")] = None,
) -> TuneJob:
    """Search the best settings of a model within a budget, on the training rows only."""
    return service.tune(
        dataset_id, target, request, filters, _classes(classes), _validation(validation)
    )


@router.get("/tune-jobs/{job_id}")
def tuning(job_id: str, service: Service) -> TuneJob:
    return service.tuning(job_id)


@router.delete("/tune-jobs/{job_id}")
def stop_tuning(job_id: str, service: Service) -> TuneJob:
    """Stop the search after the current trial and keep the best one found."""
    return service.stop_tuning(job_id)


@router.get("/jobs/{job_id}")
def job(job_id: str, service: Service) -> TrainingJob:
    return service.job(job_id)


@router.delete("/jobs/{job_id}")
def cancel(job_id: str, service: Service) -> TrainingJob:
    """Stop a training at its next checkpoint; nothing is saved."""
    return service.cancel(job_id)


@router.post("/datasets/{dataset_id}/estimate")
def estimate(
    dataset_id: str,
    service: Service,
    filters: Filters,
    target: Annotated[str, Query(min_length=1)],
    model: str | None = None,
    settings: Annotated[str | None, Query(description="JSON object of settings.")] = None,
    choices: Annotated[Choices | None, Body(description="Preparation choices.")] = None,
    classes: Annotated[str | None, Query(description="JSON: a number cut into classes.")] = None,
    validation: Annotated[str | None, Query(description="JSON: how the model is judged.")] = None,
) -> Estimate:
    """How long training would take (fits on 500 and 2 000 rows, extrapolated)."""
    return service.estimate(
        dataset_id,
        target,
        model,
        _settings(settings),
        filters,
        choices,
        _classes(classes),
        _validation(validation),
    )


@router.get("/datasets/{dataset_id}/runs")
def runs(dataset_id: str, service: Service, target: str | None = None) -> list[RunSummary]:
    """Saved runs, most recent first (for one target when given)."""
    return service.runs(dataset_id, target)


@router.get("/datasets/{dataset_id}/runs/{run_id}")
def run(dataset_id: str, run_id: str, service: Service) -> Run:
    return service.run(dataset_id, run_id)


@router.get("/datasets/{dataset_id}/runs/{run_id}/explanation")
def explanation(dataset_id: str, run_id: str, service: Service) -> ExplanationState:
    """Why the model predicts what it predicts: computed in the background on first call (status
    "running": call again), then kept with the run."""
    return service.explanation(dataset_id, run_id)


@router.post("/datasets/{dataset_id}/runs/{run_id}/neutralize")
def neutralize(
    dataset_id: str, run_id: str, request: NeutralizeRequest, service: Service
) -> Neutralized:
    """A panel run's held-out predictions without the part some columns explain linearly,
    period by period, and every period scored again."""
    return service.neutralize(dataset_id, run_id, request)


@router.get("/datasets/{dataset_id}/runs/{run_id}/periods.csv")
def periods(dataset_id: str, run_id: str, service: Service) -> Response:
    """A panel run's score per period, as CSV."""
    return Response(
        service.periods_csv(dataset_id, run_id),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="periods-{run_id}.csv"'},
    )


@router.get("/datasets/{dataset_id}/runs/{run_id}/predictions.csv")
def predictions(dataset_id: str, run_id: str, service: Service) -> Response:
    """Predictions on the test rows (sample), as CSV."""
    return Response(
        service.predictions_csv(dataset_id, run_id),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="predictions-{run_id}.csv"'},
    )


router.include_router(errors.router)  # where a model is wrong (036)
router.include_router(interpret.router)  # interpretability on demand (125)
