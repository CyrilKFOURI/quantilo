from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from quantilo.typing.knowledge_base import load
from quantilo.typing.types import AnalysisType

from app.api.datasets import to_out
from app.api.deps import Datasets
from app.schemas.columns import (
    ColumnChangesRequest,
    ColumnOut,
    ColumnSummary,
    ConversionOut,
    RoleOut,
)
from app.schemas.datasets import DatasetOut
from app.services.columns.service import ColumnService

router = APIRouter(tags=["columns"])


def get_column_service(request: Request) -> ColumnService:
    return request.app.state.columns


Columns = Annotated[ColumnService, Depends(get_column_service)]


@router.get("/datasets/{dataset_id}/columns")
def list_columns(dataset_id: str, service: Columns) -> list[ColumnOut]:
    return [ColumnOut.of(view) for view in service.columns(dataset_id)]


@router.patch("/datasets/{dataset_id}/columns")
def update_columns(
    dataset_id: str, body: ColumnChangesRequest, service: Columns
) -> list[ColumnOut]:
    return [ColumnOut.of(view) for view in service.update(dataset_id, body.changes)]


@router.post("/datasets/{dataset_id}/columns/detect")
def detect_columns(dataset_id: str, service: Columns) -> list[ColumnOut]:
    return [ColumnOut.of(view) for view in service.detect(dataset_id)]


@router.get("/datasets/{dataset_id}/columns/conversion")
def conversion_preview(
    dataset_id: str,
    service: Columns,
    column: Annotated[str, Query(min_length=1)],
    type: Annotated[AnalysisType, Query()],
) -> ConversionOut:
    parser, summary = service.conversion(dataset_id, column, type)
    return ConversionOut(type=type, parser=parser, summary=ColumnSummary.model_validate(summary))


@router.post("/datasets/{dataset_id}/columns/review")
def review_types(dataset_id: str, service: Columns, datasets: Datasets) -> DatasetOut:
    return to_out(datasets, service.mark_reviewed(dataset_id))


@router.get("/column-roles")
def list_roles() -> list[RoleOut]:
    return [RoleOut(id=role.id, label=role.label) for role in load().roles]
