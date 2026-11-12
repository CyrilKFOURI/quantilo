"""Interpretability under Explain, Advanced (125), each computed only when asked for: the
accumulated local effects of one column, the global surrogate, the check of the monotonic
constraints, and what would change one row's prediction. ALE and the surrogate are cached beside
the run (its rows and filters never change); a counterfactual depends on the user's choices and
is computed each time."""

import hashlib
from typing import Annotated, Any

import joblib
from fastapi import APIRouter, Depends, Query, Request
from pydantic import TypeAdapter
from quantilo.ml import ale as effects
from quantilo.ml import counterfactual as whatif
from quantilo.ml import surrogate as imitation
from quantilo.ml.engine import CounterfactualRequest

from app.modules.ml.service import EXPLANATION, MlService

ALE = TypeAdapter(effects.Ale)
SURROGATE = TypeAdapter(imitation.Surrogate)


def _pipeline(service: MlService, dataset_id: str, run_id: str) -> Any:
    # a file this app wrote itself in its own local workspace, never one received from outside
    return joblib.load(service._folder(dataset_id) / f"{run_id}.joblib")


def _rule(service: MlService, dataset_id: str, run_id: str, column: str) -> str | None:
    """R-XAI-01, read from the run's explanation when it was computed."""
    cached = service._folder(dataset_id) / f"{run_id}.explain.json"
    if not cached.is_file():
        return None
    found = EXPLANATION.validate_json(cached.read_bytes())
    return next((d.extrapolates for d in found.dependence if d.column == column), None)


def ale(
    service: MlService, dataset_id: str, run_id: str, column: str, bins: int = effects.BINS
) -> effects.Ale:
    run = service.run(dataset_id, run_id)
    key = hashlib.sha256(f"{column}\x00{bins}".encode()).hexdigest()[:16]
    cached = service._folder(dataset_id) / f"{run_id}.ale-{key}.json"
    if cached.is_file():
        return ALE.validate_json(cached.read_bytes())
    found = service._ale(
        dataset_id,
        _pipeline(service, dataset_id, run_id),
        run,
        service._conditions(run_id),
        column,
        bins,
        _rule(service, dataset_id, run_id, column),
    )
    cached.write_bytes(ALE.dump_json(found))
    return found


def surrogate(
    service: MlService, dataset_id: str, run_id: str, depth: int = imitation.DEPTH
) -> imitation.Surrogate:
    run = service.run(dataset_id, run_id)
    cached = service._folder(dataset_id) / f"{run_id}.surrogate-{int(depth)}.json"
    if cached.is_file():
        return SURROGATE.validate_json(cached.read_bytes())
    found = service._surrogate(
        dataset_id, _pipeline(service, dataset_id, run_id), run, service._conditions(run_id), depth
    )
    cached.write_bytes(SURROGATE.dump_json(found))
    return found


def monotonic(service: MlService, dataset_id: str, run_id: str) -> list[effects.Monotonic]:
    run = service.run(dataset_id, run_id)
    return service._monotonic(
        dataset_id, _pipeline(service, dataset_id, run_id), run, service._conditions(run_id)
    )


def counterfactual(
    service: MlService, dataset_id: str, run_id: str, request: CounterfactualRequest
) -> whatif.Counterfactual:
    run = service.run(dataset_id, run_id)
    return service._counterfactual(
        dataset_id,
        _pipeline(service, dataset_id, run_id),
        run,
        service._conditions(run_id),
        request,
    )


# --- Routes --------------------------------------------------------------------------------------

router = APIRouter()


def get_service(request: Request) -> MlService:
    return request.app.state.ml


Service = Annotated[MlService, Depends(get_service)]


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/ale")
def run_ale(
    dataset_id: str,
    run_id: str,
    service: Service,
    column: Annotated[str, Query(min_length=1, max_length=500)],
    bins: Annotated[int, Query(ge=2, le=effects.MAX_BINS)] = effects.BINS,
) -> effects.Ale:
    """Accumulated local effects of one column (numbers, ordered categories), with the check of
    its monotonic constraint when it has one."""
    return ale(service, dataset_id, run_id, column, bins)


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/surrogate")
def run_surrogate(
    dataset_id: str,
    run_id: str,
    service: Service,
    depth: Annotated[int, Query(ge=1, le=imitation.MAX_DEPTH)] = imitation.DEPTH,
) -> imitation.Surrogate:
    """A shallow tree trained on the model's predictions, with its fidelity (R-XAI-04)."""
    return surrogate(service, dataset_id, run_id, depth)


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/monotonic")
def run_monotonic(dataset_id: str, run_id: str, service: Service) -> list[effects.Monotonic]:
    """Each monotonic constraint of the run, checked row by row (empty: none was set)."""
    return monotonic(service, dataset_id, run_id)


@router.post("/datasets/{dataset_id}/runs/{run_id}/explain/counterfactual")
def run_counterfactual(
    dataset_id: str, run_id: str, request: CounterfactualRequest, service: Service
) -> whatif.Counterfactual:
    """What would change one row's prediction: at most three changes on the columns allowed to
    change (R-XAI-03 locks identifiers, dates, text and personal attributes by default), and
    the closest real row with that outcome."""
    return counterfactual(service, dataset_id, run_id, request)
