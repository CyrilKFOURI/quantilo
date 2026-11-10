from typing import Annotated

from fastapi import APIRouter, Depends, Request
from quantilo.cleaning.model import Steps
from quantilo.cleaning.preview import CleanPreview

from app.core.errors import AppError
from app.modules.clean.service import (
    CleanDraft,
    CleanFamily,
    CleanJob,
    CleanLineageItem,
    CleanPreviewRequest,
    CleanService,
    CleanSettings,
    CleanSuggestions,
    SaveRequest,
)

router = APIRouter(tags=["module:clean"])


def get_service(request: Request) -> CleanService:
    """404 while `QUANTILO_CLEAN_ENABLED` is off (S8)."""
    service: CleanService | None = request.app.state.clean
    if service is None:
        raise AppError("Cleaning is turned off.", code="clean_disabled", status_code=404)
    return service


Service = Annotated[CleanService, Depends(get_service)]


@router.get("/settings")
def settings(service: Service) -> CleanSettings:
    return service.settings()


@router.put("/settings")
def update_settings(settings: CleanSettings, service: Service) -> CleanSettings:
    return service.update_settings(settings)


@router.get("/lineage")
def lineage(service: Service) -> list[CleanLineageItem]:
    """Every cleaned version and the dataset it was made from."""
    return service.lineage()


@router.get("/datasets/{dataset_id}/suggestions")
def suggestions(dataset_id: str, service: Service) -> CleanSuggestions:
    """The Basic suggestions, each with its reason, numbers and step (cached per import)."""
    return service.suggestions(dataset_id)


@router.get("/datasets/{dataset_id}/draft")
def draft(dataset_id: str, service: Service) -> CleanDraft:
    """The saved draft, or the automatic suggestions' steps when there is none."""
    return service.draft(dataset_id)


@router.put("/datasets/{dataset_id}/draft")
def save_draft(dataset_id: str, body: Steps, service: Service) -> CleanDraft:
    return service.save_draft(dataset_id, body.steps)


@router.delete("/datasets/{dataset_id}/draft")
def reset_draft(dataset_id: str, service: Service) -> CleanDraft:
    """Back to the automatic steps."""
    return service.reset_draft(dataset_id)


@router.post("/datasets/{dataset_id}/preview")
def preview(dataset_id: str, body: CleanPreviewRequest, service: Service) -> CleanPreview:
    """Before / after of every step (or one): counts on the whole file, first rows touched."""
    return service.preview(dataset_id, body)


@router.post("/datasets/{dataset_id}/versions", status_code=202)
def save(dataset_id: str, body: SaveRequest, service: Service) -> CleanJob:
    """Save the steps as a new cleaned version (a dataset); the original is not changed."""
    return service.save(dataset_id, body.steps)


@router.get("/datasets/{dataset_id}/versions")
def family(dataset_id: str, service: Service) -> CleanFamily:
    """The original of this dataset and its cleaned versions, v1 first."""
    return service.family(dataset_id)


@router.get("/jobs/{job_id}")
def job(job_id: str, service: Service) -> CleanJob:
    return service.job(job_id)


@router.delete("/jobs/{job_id}")
def cancel(job_id: str, service: Service) -> CleanJob:
    return service.cancel(job_id)
