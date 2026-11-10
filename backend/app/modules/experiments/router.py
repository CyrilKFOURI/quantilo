from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from quantilo.experiments.analyze import ExperimentDesign, ExperimentResult
from quantilo.experiments.bandit import BanditSimulation
from quantilo.experiments.plan import SampleSize, SampleSizeRequest
from quantilo.experiments.service import (
    ExperimentCandidates,
    ExperimentEngine,
    MetricBaseline,
)
from quantilo.filters.model import Condition, parse_filters

router = APIRouter(tags=["module:experiments"])


def get_service(request: Request) -> ExperimentEngine:
    return request.app.state.experiments


def active_filters(
    filters: Annotated[str | None, Query(max_length=100_000)] = None,
) -> list[Condition]:
    return parse_filters(filters)


Service = Annotated[ExperimentEngine, Depends(get_service)]
Filters = Annotated[list[Condition], Depends(active_filters)]


@router.get("/datasets/{dataset_id}/candidates")
def candidates(dataset_id: str, service: Service, filters: Filters) -> ExperimentCandidates:
    return service.candidates(dataset_id, filters)


@router.post("/datasets/{dataset_id}/analyze")
def analyze(
    dataset_id: str, design: ExperimentDesign, service: Service, filters: Filters
) -> ExperimentResult:
    return service.analyze(dataset_id, design, filters)


@router.get("/datasets/{dataset_id}/baseline")
def baseline(
    dataset_id: str,
    column: Annotated[str, Query(min_length=1)],
    service: Service,
    filters: Filters,
) -> MetricBaseline:
    return service.baseline(dataset_id, column, filters)


@router.post("/sample-size")
def sample_size(request: SampleSizeRequest, service: Service) -> SampleSize:
    return service.sample_size(request)


@router.post("/bandit-simulation")
def bandit_simulation(request: SampleSizeRequest, service: Service) -> BanditSimulation:
    """The planned test run with a fixed split and with a bandit (Plan, 049)."""
    return service.bandit_simulation(request)
