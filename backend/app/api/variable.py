from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from quantilo.variable.analyze import VariableProfile
from quantilo.variable.fitting import Fit
from quantilo.variable.outliers import OutlierMethod, Outliers
from quantilo.variable.service import VariableService

from app.api.filters import Filters

router = APIRouter(tags=["variable"])


def get_variable_service(request: Request) -> VariableService:
    return request.app.state.variable


Variable = Annotated[VariableService, Depends(get_variable_service)]
ColumnName = Annotated[str, Query(min_length=1)]


@router.get("/datasets/{dataset_id}/variable")
def profile(
    dataset_id: str, column: ColumnName, service: Variable, filters: Filters
) -> VariableProfile:
    return service.profile(dataset_id, column, filters)


@router.get("/datasets/{dataset_id}/variable/fit")
def fit(dataset_id: str, column: ColumnName, service: Variable, filters: Filters) -> Fit:
    return service.fit(dataset_id, column, filters)


@router.get("/datasets/{dataset_id}/variable/outliers")
def outliers(
    dataset_id: str,
    column: ColumnName,
    service: Variable,
    filters: Filters,
    method: OutlierMethod = "iqr",
) -> Outliers:
    return service.outliers(dataset_id, column, method, filters)
