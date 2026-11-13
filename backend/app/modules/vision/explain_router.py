"""Routes of the explanations (117): "Why?" on one image of a kept run, and the Explain tab.

Everything is computed on demand and cached next to the run (`app.modules.vision.explain`)."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request

from app.modules.vision.explain import (
    ExplainOverview,
    ExplainService,
    WhyExplanation,
    WhyMap,
    WhySanity,
)

router = APIRouter(tags=["module:vision"])

MapMethod = Literal[
    "kept",
    "occlusion",
    "grad_cam",
    "rollout",
    "saliency",
    "smoothgrad",
    "integrated_gradients",
    "patch_pca",
    "patchcore",
]
SanityMethod = Literal["grad_cam", "saliency", "integrated_gradients"]


def get_explain(request: Request) -> ExplainService:
    return request.app.state.vision_explain


Service = Annotated[ExplainService, Depends(get_explain)]
Model = Annotated[str | None, Query(description="A model of the run (default: the run's own).")]
Target = Annotated[str | None, Query(description="The class to explain (default: predicted).")]


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain")
def explain_run(
    dataset_id: str, run_id: str, service: Service, model: Model = None
) -> ExplainOverview:
    """The Explain tab: prototypes and criticisms of each class in the model's space, and, with
    a classic logistic model, the importance of each descriptor family."""
    return service.overview(dataset_id, run_id, model)


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/images/{image_id}")
def explain_image(
    dataset_id: str, run_id: str, image_id: str, service: Service, model: Model = None
) -> WhyExplanation:
    """ "Why?": the prediction, the nearest training images, the nearest prototype, the
    counterfactual example, and the maps that can be asked for (with their level and cost)."""
    return service.image(dataset_id, run_id, image_id, model)


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/images/{image_id}/map")
def explain_map(
    dataset_id: str,
    run_id: str,
    image_id: str,
    method: MapMethod,
    service: Service,
    model: Model = None,
    target: Target = None,
) -> WhyMap:
    """One map, as an 8-bit grey PNG and its range: the page colours it, sets its opacity and
    threshold and draws its contours without another call."""
    return service.map(dataset_id, run_id, image_id, method, model, target)


@router.get("/datasets/{dataset_id}/runs/{run_id}/explain/images/{image_id}/sanity")
def explain_sanity(
    dataset_id: str,
    run_id: str,
    image_id: str,
    service: Service,
    method: SanityMethod = "grad_cam",
    model: Model = None,
    target: Target = None,
) -> WhySanity:
    """The cascading randomisation test (Adebayo et al., 2018) of one map (R-VEX-10)."""
    return service.sanity(dataset_id, run_id, image_id, method, model, target)
