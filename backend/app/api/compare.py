from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request
from quantilo.compare.analyze import Comparison
from quantilo.compare.durations import Survival, Transitions
from quantilo.compare.service import CompareService
from quantilo.timeseries.series import Aggregate, Align, Options, Period

from app.api.filters import Filters

router = APIRouter(tags=["compare"])


def get_compare_service(request: Request) -> CompareService:
    return request.app.state.compare


Service = Annotated[CompareService, Depends(get_compare_service)]
Name = Annotated[str, Query(min_length=1)]


@router.get("/datasets/{dataset_id}/compare")
def compare(
    dataset_id: str,
    service: Service,
    filters: Filters,
    x: Name,
    y: Name,
    z: Annotated[str | None, Query(min_length=1)] = None,
    color: Annotated[str | None, Query(min_length=1)] = None,
    period: Period | Literal["auto"] = "auto",
    agg: Aggregate | Literal["auto"] = "auto",
    window: Annotated[int | None, Query(ge=2, le=365)] = None,
    align: Align = "trailing",
    numbers: Annotated[
        str | None, Query(description="Whole-number columns to keep as numbers, comma-separated.")
    ] = None,
) -> Comparison:
    """Chart data and tests; period, agg, window and align shape a date-and-number series. A whole
    number with at most 12 values reads as categories unless listed in `numbers`."""
    options = Options(period, agg, window, align)
    kept = [n for n in (numbers or "").split(",") if n]
    return service.compare(dataset_id, x, y, z, color, filters, options, kept)


Column = Annotated[str | None, Query(min_length=1)]


@router.get("/datasets/{dataset_id}/survival")
def survival(
    dataset_id: str,
    service: Service,
    filters: Filters,
    duration: Column = None,
    event: Column = None,
    start: Column = None,
    end: Column = None,
    group: Column = None,
    event_value: Annotated[str | None, Query(max_length=200)] = None,
) -> Survival:
    """Kaplan-Meier curves (95 % log-log intervals), median survival and the log-rank test.
    Without columns, the reading a rule detects (R-SRV-01 duration + event, R-SRV-02 start + end
    dates); `setup` is null when no rule fires (the section stays hidden)."""
    return service.survival(dataset_id, filters, duration, event, start, end, group, event_value)


@router.get("/datasets/{dataset_id}/transitions")
def transitions(
    dataset_id: str,
    service: Service,
    filters: Filters,
    entity: Column = None,
    time: Column = None,
    status: Column = None,
    detect_only: bool = False,
) -> Transitions:
    """Status transitions as a Markov chain: matrix, stationary shares, absorbing states and the
    expected steps before reaching them. Without columns, the reading R-MKV-01 detects;
    `detect_only` returns it without reading the rows."""
    return service.transitions(dataset_id, filters, entity, time, status, detect_only)
