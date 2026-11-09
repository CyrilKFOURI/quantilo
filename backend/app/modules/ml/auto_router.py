"""Routes of the Auto (038): start it, follow it step by step, stop it and keep the best."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from app.modules.ml.auto import AutoJob, AutoRequest, AutoService
from app.modules.ml.router import Filters, _classes, _validation

router = APIRouter(tags=["module:ml"])


def get_auto(request: Request) -> AutoService:
    return request.app.state.ml_auto


Auto = Annotated[AutoService, Depends(get_auto)]


@router.post("/datasets/{dataset_id}/auto", status_code=202)
def start(
    dataset_id: str,
    service: Auto,
    filters: Filters,
    request: AutoRequest,
    target: Annotated[str, Query(min_length=1)],
    classes: Annotated[str | None, Query(description="JSON: a number cut into classes.")] = None,
    validation: Annotated[str | None, Query(description="JSON: how the models are judged.")] = None,
) -> AutoJob:
    """Find the best model within a time budget: baselines, one model per family, successive
    elimination, tuning of the 3 best families, then an ensemble kept only if clearly better."""
    return service.start(
        dataset_id, target, request, filters, _classes(classes), _validation(validation)
    )


@router.get("/auto-jobs/{job_id}")
def follow(job_id: str, service: Auto) -> AutoJob:
    return service.job(job_id)


@router.delete("/auto-jobs/{job_id}")
def stop(job_id: str, service: Auto) -> AutoJob:
    """Stop the Auto at its next checkpoint; the best model found so far is saved as a run."""
    return service.stop(job_id)
