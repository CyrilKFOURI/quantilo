from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Query, Request
from pydantic import BaseModel, Field
from quantilo.experiments.analyze import ExperimentDesign, ExperimentError, ExperimentResult
from quantilo.experiments.bandit import BanditSimulation
from quantilo.experiments.offpolicy import DecisionsRequest, LoggedDecisions, PolicyEvaluation
from quantilo.experiments.plan import SampleSize, SampleSizeRequest
from quantilo.experiments.policy import PolicyLearning
from quantilo.experiments.service import (
    ExperimentCandidates,
    ExperimentEngine,
    MetricBaseline,
    PolicyModel,
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


class PolicyRequest(BaseModel):
    design: ExperimentDesign
    covariates: list[str] | None = Field(
        default=None,
        max_length=50,
        description="Columns describing the units before the test; every usable one if unset.",
    )
    qini: bool = Field(
        default=False,
        description='Add the Qini curves (the page\'s "Advanced > More", asked when opened).',
    )


@router.post("/datasets/{dataset_id}/policy")
def policy(
    dataset_id: str, body: PolicyRequest, service: Service, filters: Filters
) -> PolicyLearning:
    """Who should get which variant (055 § B): a policy tree on doubly robust scores, its
    cross-validated value against everyone on each variant, the Qini curve."""
    return service.policy(dataset_id, body.design, body.covariates, filters, body.qini)


@router.get("/datasets/{dataset_id}/decisions")
def decisions(dataset_id: str, service: Service, filters: Filters) -> LoggedDecisions:
    """Does the dataset look like logged decisions (an action, its reward, covariates)?"""
    return service.decisions(dataset_id, filters)


def policy_models(app: FastAPI, dataset_id: str, run_ids: list[str]) -> list[PolicyModel]:
    """The saved ML models (runs of this dataset) to evaluate as policies, as Predict loads them
    (the pipeline is a file this app wrote in its own workspace)."""
    if not run_ids:
        return []
    predictions = getattr(app.state, "ml_predictions", None)
    if predictions is None:
        raise ExperimentError("Saved models need the ML module.", "models")
    found = []
    for run_id in run_ids:
        run = predictions.ml.run(dataset_id, run_id)
        card = predictions.card(dataset_id, run_id)
        label = f"{card.model_label or run.model} → {run.target}"
        found.append(PolicyModel(label, predictions._pipeline(dataset_id, run_id), card))
    return found


@router.post("/datasets/{dataset_id}/evaluate-policy")
def evaluate_policy(
    dataset_id: str,
    body: DecisionsRequest,
    request: Request,
    service: Service,
    filters: Filters,
) -> PolicyEvaluation:
    """Evaluate policies from logged decisions (055 § C): diagnostics, then DM, IPS, SNIPS and
    DR for the logged decisions, each constant policy, the learnt rule and the saved models."""
    models = policy_models(request.app, dataset_id, body.models)
    return service.evaluate_policy(dataset_id, body, filters, models)
