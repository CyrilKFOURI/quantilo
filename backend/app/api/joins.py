"""Join the open dataset with another one (089): proposed keys, preview, then the new dataset."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.schemas.joins import JoinIn, JoinPreviewOut, JoinSuggestIn, JoinSuggestOut
from app.schemas.model import CreatedOut
from app.services.joins import JoinService

router = APIRouter(prefix="/datasets/{dataset_id}/join", tags=["joins"])


def get_join_service(request: Request) -> JoinService:
    return request.app.state.joins


Service = Annotated[JoinService, Depends(get_join_service)]


@router.post("/suggest")
def suggest_join(dataset_id: str, body: JoinSuggestIn, service: Service) -> JoinSuggestOut:
    """The keys the rules propose between this dataset and `other_id` (R-JN-01 to 03)."""
    return service.suggest(dataset_id, body)


@router.post("/preview")
def preview_join(dataset_id: str, body: JoinIn, service: Service) -> JoinPreviewOut:
    """Counts on every row and the first rows of the result; nothing is written."""
    return service.preview(dataset_id, body)


@router.post("", status_code=201)
def create_join(dataset_id: str, body: JoinIn, service: Service) -> CreatedOut:
    """Write the joined dataset (after the user confirmed the preview) and import it."""
    return service.create(dataset_id, body)
