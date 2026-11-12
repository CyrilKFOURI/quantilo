"""Pinned rows (099): pin rows with a label and a colour, manage the labels, and see what
distinguishes the pinned rows from the others (computed when asked, R-PIN-05)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from quantilo.segments.pinned import PinnedComparison

from app.services.pins.service import (
    LabelChange,
    PinnedRows,
    PinRequest,
    PinService,
    UnpinRequest,
)

router = APIRouter(prefix="/datasets/{dataset_id}/pins", tags=["pins"])


def get_pin_service(request: Request) -> PinService:
    return request.app.state.pins


Service = Annotated[PinService, Depends(get_pin_service)]


@router.get("")
def pins(dataset_id: str, service: Service) -> PinnedRows:
    """Every pin; a row that moved after a re-import is found again by its key (R-PIN-02)."""
    return service.pins(dataset_id)


@router.put("")
def pin(dataset_id: str, body: PinRequest, service: Service) -> PinnedRows:
    """Pin rows (numbers in the file, from 0) with a label; the label's rows take its colour."""
    return service.pin(dataset_id, body)


@router.post("/remove")
def unpin(dataset_id: str, body: UnpinRequest, service: Service) -> PinnedRows:
    return service.unpin(dataset_id, body)


@router.patch("/labels")
def change_label(dataset_id: str, body: LabelChange, service: Service) -> PinnedRows:
    """Rename a label, or give it another colour."""
    return service.change_label(dataset_id, body)


@router.get("/compare")
def compare(
    dataset_id: str,
    service: Service,
    label: Annotated[str | None, Query(min_length=1, max_length=60)] = None,
) -> PinnedComparison:
    """What distinguishes the pinned rows (all, or one label) from every other row."""
    return service.compare(dataset_id, label)
