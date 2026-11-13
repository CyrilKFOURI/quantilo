"""The model of a project (088): relations between its datasets, layout, analysis datasets."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.schemas.model import (
    AggregateIn,
    BuildIn,
    CheckIn,
    CheckOut,
    CreatedOut,
    DatesIn,
    LayoutIn,
    ModelOut,
    PreviewOut,
    RelationIn,
    RelationPatch,
)
from app.services.model.service import ModelService

router = APIRouter(prefix="/projects/{project_id}/model", tags=["model"])


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model


Service = Annotated[ModelService, Depends(get_model_service)]


@router.get("")
def get_model(project_id: str, service: Service) -> ModelOut:
    """Tables, accepted and proposed relations (detected now), layout, analysis datasets."""
    return service.model(project_id)


@router.post("/relations")
def save_relation(project_id: str, body: RelationIn, service: Service) -> ModelOut:
    """Accept or reject a proposal, or add a relation by hand; returns the whole model."""
    return service.save_relation(project_id, body)


@router.patch("/relations/{relation_id}")
def update_relation(
    project_id: str, relation_id: str, body: RelationPatch, service: Service
) -> ModelOut:
    return service.update_relation(project_id, relation_id, body)


@router.delete("/relations/{relation_id}")
def delete_relation(project_id: str, relation_id: str, service: Service) -> ModelOut:
    """The relation can be proposed again if the data still says so."""
    return service.delete_relation(project_id, relation_id)


@router.post("/check")
def check_relation(project_id: str, body: CheckIn, service: Service) -> CheckOut:
    """Exact counts of a pair of columns before saving it."""
    return service.check(project_id, body)


@router.put("/layout")
def save_layout(project_id: str, body: LayoutIn, service: Service) -> ModelOut:
    """Card positions and the version each family uses."""
    return service.save_layout(project_id, body)


@router.post("/preview")
def preview_build(project_id: str, body: BuildIn, service: Service) -> PreviewOut:
    """The analysis dataset, counted and not written."""
    return service.preview(project_id, body)


@router.post("/build", status_code=201)
def build(project_id: str, body: BuildIn, service: Service) -> CreatedOut:
    """Write the analysis dataset and import it in the project."""
    return service.build(project_id, body)


@router.post("/builds/{dataset_id}/rebuild")
def rebuild(project_id: str, dataset_id: str, service: Service) -> CreatedOut:
    return service.rebuild(project_id, dataset_id)


@router.post("/dates", status_code=201)
def dates(project_id: str, body: DatesIn, service: Service) -> CreatedOut:
    """C1 (R-BLD-08): a date table, one row per day of a date column, related to its table."""
    return service.dates(project_id, body)


@router.post("/aggregate", status_code=201)
def aggregate(project_id: str, body: AggregateIn, service: Service) -> CreatedOut:
    """Advanced: a child table summed up to its parent's rows, as a new dataset."""
    return service.aggregate(project_id, body)
