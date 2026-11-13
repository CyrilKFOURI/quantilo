"""Routes of 118: the set's concepts, and what explains the Advanced > More models and TCAV.

Nothing here trains a model: the More models train in a run (`POST /runs` with `more`); their
evidence is read as the run kept it, and TCAV is computed when it is asked for, then cached."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response

from app.modules.vision import concepts as concept_store
from app.modules.vision import more_explain as more
from app.modules.vision.concepts import ConceptGroupIn, ConceptOut
from app.modules.vision.explain import ExplainService
from app.modules.vision.service import VisionService

router = APIRouter(tags=["module:vision"])


def get_vision(request: Request) -> VisionService:
    return request.app.state.vision


def get_explain(request: Request) -> ExplainService:
    return request.app.state.vision_explain


Vision = Annotated[VisionService, Depends(get_vision)]
Explain = Annotated[ExplainService, Depends(get_explain)]


@router.get("/datasets/{dataset_id}/concepts")
def concepts(dataset_id: str, service: Vision) -> list[ConceptOut]:
    """The set's concepts (R-VCB-01, R-VTC-01): metadata columns, saved groups of images, and
    the classes (TCAV only)."""
    record = service._record(dataset_id)
    return concept_store.concept_list(service, record, service._table(record))


@router.post("/datasets/{dataset_id}/concepts", status_code=201)
def save_concept(dataset_id: str, body: ConceptGroupIn, service: Vision) -> ConceptOut:
    """Save a group of images (the gallery's selection) as a named concept."""
    return concept_store.save_group(service, dataset_id, body)


@router.delete("/datasets/{dataset_id}/concepts/{group_id}", status_code=204)
def delete_concept(dataset_id: str, group_id: str, service: Vision) -> Response:
    concept_store.delete_group(service, dataset_id, group_id)
    return Response(status_code=204)


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/images/{image_id}/looks")
def looks_like(dataset_id: str, run_id: str, image_id: str, service: Explain) -> more.WhyLooksLike:
    """ProtoPNet-lite: "this patch looks like that prototype", as the run kept it (R-VPP-06)."""
    run = service._run(dataset_id, run_id)
    return more.looks_like(service, run, run.row_of(image_id))


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/images/{image_id}/concepts")
def image_concepts(
    dataset_id: str, run_id: str, image_id: str, service: Explain
) -> more.WhyConcepts:
    """Concept bottleneck: the concepts read in the image and the class head (R-VCB-05)."""
    run = service._run(dataset_id, run_id)
    return more.concepts_of(service, run, run.row_of(image_id))


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/prototypes")
def prototypes(dataset_id: str, run_id: str, service: Explain) -> more.ProtoOverview | None:
    """ProtoPNet-lite's prototypes, by class (null when the run has none)."""
    return more.prototypes_overview(service, service._run(dataset_id, run_id))


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/tcav/options")
def tcav_options(dataset_id: str, run_id: str, service: Explain) -> more.TcavOptions:
    """Whether TCAV can test this run, on which concepts and layers (cheap: nothing computed)."""
    return more.tcav_options(service, service._run(dataset_id, run_id))


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/tcav")
def tcav(
    dataset_id: str,
    run_id: str,
    service: Explain,
    concept: Annotated[str, Query(max_length=120, description="A concept key.")],
    layer: Annotated[str | None, Query(max_length=40)] = None,
) -> more.WhyTcav:
    """TCAV of one concept for every class, with its test against random concepts (R-VTC-05)."""
    return more.tcav(service, service._run(dataset_id, run_id), concept, layer)
