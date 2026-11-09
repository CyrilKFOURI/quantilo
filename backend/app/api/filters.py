from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import BaseModel, Field
from quantilo.filters.model import Condition, parse_filters

from app.services.filters.saved import SavedFilter, SavedFilterService

router = APIRouter(tags=["filters"])


def active_filters(
    filters: Annotated[
        str | None,
        Query(description="JSON list of conditions, combined with AND.", max_length=100_000),
    ] = None,
) -> list[Condition]:
    return parse_filters(filters)


Filters = Annotated[list[Condition], Depends(active_filters)]


def get_saved_filters(request: Request) -> SavedFilterService:
    return request.app.state.saved_filters


Saved = Annotated[SavedFilterService, Depends(get_saved_filters)]


class SaveFilterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    conditions: list[Condition] = Field(min_length=1, max_length=50)


@router.get("/datasets/{dataset_id}/saved-filters")
def list_saved(dataset_id: str, service: Saved) -> list[SavedFilter]:
    return service.all(dataset_id)


@router.post("/datasets/{dataset_id}/saved-filters", status_code=201)
def save(dataset_id: str, body: SaveFilterRequest, service: Saved) -> SavedFilter:
    return service.save(dataset_id, body.name, body.conditions)


@router.delete("/datasets/{dataset_id}/saved-filters/{filter_id}", status_code=204)
def delete(dataset_id: str, filter_id: str, service: Saved) -> Response:
    service.delete(dataset_id, filter_id)
    return Response(status_code=204)
