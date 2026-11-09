from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import BaseModel, Field
from quantilo.filters.model import Condition, parse_filters
from quantilo.kpi.compute import KpiResult
from quantilo.kpi.definition import Definition
from quantilo.kpi.service import Suggestions

from app.modules.kpi.service import KpiService, SavedKpi

router = APIRouter(tags=["module:kpi"])


def get_service(request: Request) -> KpiService:
    return request.app.state.kpi


def active_filters(
    filters: Annotated[str | None, Query(max_length=100_000)] = None,
) -> list[Condition]:
    return parse_filters(filters)


Service = Annotated[KpiService, Depends(get_service)]
Filters = Annotated[list[Condition], Depends(active_filters)]


class SaveKpi(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    definition: Definition


class UpdateKpi(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    definition: Definition | None = None


@router.get("/datasets/{dataset_id}/suggestions")
def suggestions(dataset_id: str, service: Service) -> Suggestions:
    return service.suggestions(dataset_id)


@router.post("/datasets/{dataset_id}/compute")
def compute(
    dataset_id: str, definition: Definition, service: Service, filters: Filters
) -> KpiResult:
    return service.compute(dataset_id, definition, filters)


@router.get("/datasets/{dataset_id}/definitions")
def list_kpis(dataset_id: str, service: Service) -> list[SavedKpi]:
    return service.all(dataset_id)


@router.post("/datasets/{dataset_id}/definitions", status_code=201)
def create(dataset_id: str, body: SaveKpi, service: Service) -> SavedKpi:
    return service.create(dataset_id, body.name, body.definition)


@router.get("/datasets/{dataset_id}/definitions/{kpi_id}")
def get_kpi(dataset_id: str, kpi_id: str, service: Service) -> SavedKpi:
    return service.get(dataset_id, kpi_id)


@router.put("/datasets/{dataset_id}/definitions/{kpi_id}")
def update(dataset_id: str, kpi_id: str, body: UpdateKpi, service: Service) -> SavedKpi:
    return service.update(dataset_id, kpi_id, body.name, body.definition)


@router.post("/datasets/{dataset_id}/definitions/{kpi_id}/duplicate", status_code=201)
def duplicate(dataset_id: str, kpi_id: str, service: Service) -> SavedKpi:
    return service.duplicate(dataset_id, kpi_id)


@router.delete("/datasets/{dataset_id}/definitions/{kpi_id}", status_code=204)
def delete(dataset_id: str, kpi_id: str, service: Service) -> Response:
    service.delete(dataset_id, kpi_id)
    return Response(status_code=204)
