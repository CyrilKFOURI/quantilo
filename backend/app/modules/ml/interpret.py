"""Interpretability under Explain, Advanced (125), each computed only when asked for: the
accumulated local effects of one column, the global surrogate, the check of the monotonic
constraints, and what would change one row's prediction. ALE and the surrogate are cached beside
the run (its rows and filters never change); a counterfactual depends on the user's choices and
is computed each time.

Under Explain, Advanced, More (126), each asked for by its own button: SHAP by permutations for
any model (R-XAI-12), Friedman's H (R-XAI-13), one row's anchor (R-XAI-14), and the prototypes and
criticisms of each class (R-XAI-15). Each is cached beside the run, per setting.

Models readable by construction, in the Scorecard tab of a two-class run (121, 122): FasterRisk's
scorecard, the sparse CART tree, and under More GOSDT's optimal tree and TreeFARMS's Rashomon set
(extra `interpret`, R-XAI-10). Each is cached beside the run, per setting."""

import hashlib
from collections.abc import Callable
from typing import Annotated, Any

import joblib
from fastapi import APIRouter, Depends, Query, Request
from pydantic import TypeAdapter
from quantilo.interpret import anchors as anchoring
from quantilo.interpret import extras
from quantilo.interpret import interactions as pairing
from quantilo.interpret import kernel_shap as sampling
from quantilo.interpret import rashomon as optimal
from quantilo.interpret import risk_score as scores
from quantilo.interpret import sparse_tree as sparse
from quantilo.interpret import tabular_prototypes as examples
from quantilo.ml import ale as effects
from quantilo.ml import counterfactual as whatif
from quantilo.ml import surrogate as imitation
from quantilo.ml.engine import CounterfactualRequest

from app.modules.ml.service import EXPLANATION, MlService, load_held_out

ALE = TypeAdapter(effects.Ale)
SURROGATE = TypeAdapter(imitation.Surrogate)
SCORES = TypeAdapter(scores.RiskScores)
CLASS_CARDS = TypeAdapter(scores.ClassCards)
ALE_2D = TypeAdapter(effects.Ale2d)
SPARSE = TypeAdapter(sparse.SparseTree)
OPTIMAL = TypeAdapter(optimal.OptimalTree)
RASHOMON = TypeAdapter(optimal.RashomonSet)
PERMUTATION_SHAP = TypeAdapter(sampling.PermutationShap)
INTERACTIONS = TypeAdapter(pairing.Interactions)
ANCHOR = TypeAdapter(anchoring.Anchor)
PROTOTYPES = TypeAdapter(examples.Prototypes)


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


def ale_2d(service: MlService, dataset_id: str, run_id: str, first: str, second: str) -> Any:
    """R-XAI-20 (125 bis): kept next to the run, like the one-column ALE."""
    run = service.run(dataset_id, run_id)
    key = hashlib.sha256(f"{first}\x00{second}".encode()).hexdigest()[:16]
    return _cached(
        service,
        dataset_id,
        run_id,
        f"ale2d-{key}",
        ALE_2D,
        lambda: service._ale_2d(
            dataset_id,
            _pipeline(service, dataset_id, run_id),
            run,
            service._conditions(run_id),
            first,
            second,
        ),
    )


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


# --- Advanced, More (126) ---------------------------------------------------------------------


def _ranked(service: MlService, dataset_id: str, run_id: str) -> list[str]:
    """The inputs from the most to the least important (permutation importance of the run's
    explanation when it was computed, else the run's own importances)."""
    cached = service._folder(dataset_id) / f"{run_id}.explain.json"
    if cached.is_file():
        found = EXPLANATION.validate_json(cached.read_bytes())
        if found.permutation is not None:
            return [item.column for item in found.permutation.items]
    run = service.run(dataset_id, run_id)
    ordered = sorted(run.importances, key=lambda i: -i.importance)
    return [i.feature for i in ordered]


def permutation_shap(
    service: MlService, dataset_id: str, run_id: str, pairs: int, background: int
) -> sampling.PermutationShap:
    run = service.run(dataset_id, run_id)
    return _cached(
        service,
        dataset_id,
        run_id,
        f"permutation-shap-{int(pairs)}-{int(background)}",
        PERMUTATION_SHAP,
        lambda: service._permutation_shap(
            dataset_id,
            _pipeline(service, dataset_id, run_id),
            run,
            service._conditions(run_id),
            pairs,
            background,
        ),
    )


def interactions(
    service: MlService, dataset_id: str, run_id: str, columns: int
) -> pairing.Interactions:
    run = service.run(dataset_id, run_id)
    return _cached(
        service,
        dataset_id,
        run_id,
        f"interactions-{int(columns)}",
        INTERACTIONS,
        lambda: service._interactions(
            dataset_id,
            _pipeline(service, dataset_id, run_id),
            run,
            service._conditions(run_id),
            _ranked(service, dataset_id, run_id),
            columns,
        ),
    )


def anchor(
    service: MlService, dataset_id: str, run_id: str, row: int, threshold: float, conditions: int
) -> anchoring.Anchor:
    run = service.run(dataset_id, run_id)
    return _cached(
        service,
        dataset_id,
        run_id,
        f"anchor-{int(row)}-{round(threshold * 100)}-{int(conditions)}",
        ANCHOR,
        lambda: service._anchor(
            dataset_id,
            _pipeline(service, dataset_id, run_id),
            run,
            service._conditions(run_id),
            row,
            _ranked(service, dataset_id, run_id),
            threshold,
            conditions,
        ),
    )


def prototypes(
    service: MlService, dataset_id: str, run_id: str, count: int, criticisms: int
) -> examples.Prototypes:
    run = service.run(dataset_id, run_id)
    return _cached(
        service,
        dataset_id,
        run_id,
        f"prototypes-{int(count)}-{int(criticisms)}",
        PROTOTYPES,
        lambda: service._prototypes(
            dataset_id,
            _pipeline(service, dataset_id, run_id),
            run,
            service._conditions(run_id),
            _ranked(service, dataset_id, run_id),
            count,
            criticisms,
        ),
    )


# --- Models readable by construction (121, 122) -------------------------------------------------


def _held_out(service: MlService, dataset_id: str, run_id: str) -> Any:
    held = service._folder(dataset_id) / f"{run_id}.heldout.npz"
    return load_held_out(held) if held.is_file() else None


def _cached(
    service: MlService,
    dataset_id: str,
    run_id: str,
    name: str,
    adapter: TypeAdapter[Any],
    compute: Callable[[], Any],
) -> Any:
    cached = service._folder(dataset_id) / f"{run_id}.{name}.json"
    if cached.is_file():
        return adapter.validate_json(cached.read_bytes())
    found = compute()
    cached.write_bytes(adapter.dump_json(found))
    return found


def scorecard(
    service: MlService,
    dataset_id: str,
    run_id: str,
    conditions: int,
    points: int,
    monotone: bool = False,
) -> scores.RiskScores:
    run = service.run(dataset_id, run_id)
    return _cached(
        service,
        dataset_id,
        run_id,
        f"scorecard-{int(conditions)}-{int(points)}{'-monotone' if monotone else ''}",
        SCORES,
        lambda: service._risk_scores(
            dataset_id,
            _pipeline(service, dataset_id, run_id),
            run,
            service._conditions(run_id),
            _held_out(service, dataset_id, run_id),
            conditions,
            points,
            monotone,
        ),
    )


def class_scorecards(
    service: MlService,
    dataset_id: str,
    run_id: str,
    conditions: int,
    points: int,
    monotone: bool = False,
) -> scores.ClassCards:
    """121 bis (R-XAI-17): one card per class, kept next to the run."""
    run = service.run(dataset_id, run_id)
    return _cached(
        service,
        dataset_id,
        run_id,
        f"classcards-{int(conditions)}-{int(points)}{'-monotone' if monotone else ''}",
        CLASS_CARDS,
        lambda: service._class_cards(
            dataset_id,
            _pipeline(service, dataset_id, run_id),
            run,
            service._conditions(run_id),
            _held_out(service, dataset_id, run_id),
            conditions,
            points,
            monotone,
        ),
    )


def sparse_tree(
    service: MlService, dataset_id: str, run_id: str, depth: int, leaves: int
) -> sparse.SparseTree:
    run = service.run(dataset_id, run_id)
    return _cached(
        service,
        dataset_id,
        run_id,
        f"sparse-{int(depth)}-{int(leaves)}",
        SPARSE,
        lambda: service._sparse_tree(
            dataset_id,
            _pipeline(service, dataset_id, run_id),
            run,
            service._conditions(run_id),
            _held_out(service, dataset_id, run_id),
            depth,
            leaves,
        ),
    )


def optimal_tree(service: MlService, dataset_id: str, run_id: str, kind: str, depth: int) -> Any:
    """GOSDT (`kind` "gosdt") or TreeFARMS ("treefarms"), refused with R-XAI-10 when absent."""
    run = service.run(dataset_id, run_id)
    return _cached(
        service,
        dataset_id,
        run_id,
        f"{kind}-{int(depth)}",
        OPTIMAL if kind == "gosdt" else RASHOMON,
        lambda: service._optimal(
            dataset_id,
            _pipeline(service, dataset_id, run_id),
            run,
            service._conditions(run_id),
            _held_out(service, dataset_id, run_id),
            kind,
            depth,
        ),
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


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/ale-2d")
def run_ale_2d(
    dataset_id: str,
    run_id: str,
    service: Service,
    first: Annotated[str, Query(min_length=1, max_length=500)],
    second: Annotated[str, Query(min_length=1, max_length=500)],
) -> effects.Ale2d:
    """The two-column ALE of two numbers (R-XAI-20): what the pair adds on top of each column's
    own effect, on a 10 by 10 grid of quantiles."""
    return ale_2d(service, dataset_id, run_id, first, second)


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


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/permutation-shap")
def run_permutation_shap(
    dataset_id: str,
    run_id: str,
    service: Service,
    pairs: Annotated[int, Query(ge=1, le=50)] = sampling.PAIRS,
    background: Annotated[int, Query(ge=5, le=200)] = sampling.BACKGROUND,
) -> sampling.PermutationShap:
    """SHAP values for any model, estimated by sampling `pairs` antithetic pairs of orders against
    `background` rows, each value with its 95 % interval; base + sum = the model's output exactly
    (R-XAI-12)."""
    return permutation_shap(service, dataset_id, run_id, pairs, background)


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/interactions")
def run_interactions(
    dataset_id: str,
    run_id: str,
    service: Service,
    columns: Annotated[int, Query(ge=2, le=pairing.MAX_COLUMNS)] = pairing.COLUMNS,
) -> pairing.Interactions:
    """Friedman's H for every pair of the `columns` most important inputs; pairs whose joint
    effect is too weak to judge are said so (R-XAI-13)."""
    return interactions(service, dataset_id, run_id, columns)


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/anchor")
def run_anchor(
    dataset_id: str,
    run_id: str,
    service: Service,
    row: Annotated[int, Query(ge=0)],
    threshold: Annotated[
        float, Query(ge=anchoring.MIN_THRESHOLD, le=anchoring.MAX_THRESHOLD)
    ] = anchoring.THRESHOLD,
    conditions: Annotated[
        int, Query(ge=1, le=anchoring.LIMIT_CONDITIONS)
    ] = anchoring.MAX_CONDITIONS,
) -> anchoring.Anchor:
    """One row's anchor: IF conditions THEN the model's outcome, on at least `threshold` of the
    similar rows (lower end of the 95 % interval), and the share of rows it covers (R-XAI-14)."""
    return anchor(service, dataset_id, run_id, row, threshold, conditions)


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/prototypes")
def run_prototypes(
    dataset_id: str,
    run_id: str,
    service: Service,
    prototypes_: Annotated[
        int, Query(alias="prototypes", ge=1, le=examples.MAX_PROTOTYPES)
    ] = examples.PROTOTYPES,
    criticisms: Annotated[int, Query(ge=0, le=examples.MAX_CRITICISMS)] = examples.CRITICISMS,
) -> examples.Prototypes:
    """Real rows that summarise each class (prototypes) and those the summary misses most
    (criticisms), by MMD-critic (R-XAI-15)."""
    return prototypes(service, dataset_id, run_id, prototypes_, criticisms)


@router.get("/interpret/available")
def interpret_available() -> list[extras.Availability]:
    """The optional methods of the `interpret` extra (GOSDT, TreeFARMS, PaCMAP, TriMAP) and,
    for each one that cannot run here, why (R-XAI-10) and the command that brings it."""
    return extras.available()


@router.get("/datasets/{dataset_id}/runs/{run_id}/readable/scorecard")
def run_scorecard(
    dataset_id: str,
    run_id: str,
    service: Service,
    conditions: Annotated[int, Query(ge=1, le=scores.MAX_CONDITIONS)] = scores.CONDITIONS,
    points: Annotated[int, Query(ge=3, le=10)] = scores.BOUND,
    monotone: bool = False,
) -> scores.RiskScores:
    """Two classes (R-XAI-05): FasterRisk's scorecard, with small integer points per condition,
    the table total -> risk, other cards almost as good, and the gap to the run's model on the
    same held-out rows (R-XAI-07, R-XAI-08); `monotone`: one direction per number (R-XAI-16)."""
    return scorecard(service, dataset_id, run_id, conditions, points, monotone)


@router.get("/datasets/{dataset_id}/runs/{run_id}/readable/class-scorecards")
def run_class_scorecards(
    dataset_id: str,
    run_id: str,
    service: Service,
    conditions: Annotated[int, Query(ge=1, le=scores.MAX_CONDITIONS)] = scores.CONDITIONS,
    points: Annotated[int, Query(ge=3, le=10)] = scores.BOUND,
    monotone: bool = False,
) -> scores.ClassCards:
    """Three classes or more (R-XAI-17): one card per class, that class against the others,
    compared with the run's model on the same held-out rows."""
    return class_scorecards(service, dataset_id, run_id, conditions, points, monotone)


@router.get("/datasets/{dataset_id}/runs/{run_id}/readable/sparse-tree")
def run_sparse_tree(
    dataset_id: str,
    run_id: str,
    service: Service,
    depth: Annotated[int, Query(ge=1, le=sparse.MAX_DEPTH)] = sparse.DEPTH,
    leaves: Annotated[int, Query(ge=2, le=sparse.MAX_LEAVES)] = sparse.LEAVES,
) -> sparse.SparseTree:
    """A sparse CART tree learnt on the target, pruned by cross-validation with the 1-SE rule
    (R-XAI-09), and its gap to the run's model."""
    return sparse_tree(service, dataset_id, run_id, depth, leaves)


@router.get("/datasets/{dataset_id}/runs/{run_id}/readable/optimal-tree")
def run_optimal_tree(
    dataset_id: str,
    run_id: str,
    service: Service,
    depth: Annotated[int, Query(ge=1, le=sparse.MAX_DEPTH)] = optimal.DEPTH,
) -> optimal.OptimalTree:
    """GOSDT's optimal sparse tree (extra `interpret`; 409 with the reason when it cannot run
    here, R-XAI-10)."""
    return optimal_tree(service, dataset_id, run_id, "gosdt", depth)


@router.get("/datasets/{dataset_id}/runs/{run_id}/readable/rashomon")
def run_rashomon(
    dataset_id: str,
    run_id: str,
    service: Service,
    depth: Annotated[int, Query(ge=1, le=sparse.MAX_DEPTH)] = optimal.DEPTH,
) -> optimal.RashomonSet:
    """TreeFARMS's Rashomon set: the trees within 5 % of the best, and the columns they share
    (extra `interpret`; 409 with the reason when it cannot run here, R-XAI-10)."""
    return optimal_tree(service, dataset_id, run_id, "treefarms", depth)
