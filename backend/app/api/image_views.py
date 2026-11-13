"""The Map (109) and Prepare (110) tabs of an image set: the map of the images, the weights of the
deep methods, the previews of each preparation step and the saved recipe."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from app.schemas.image_views import (
    ImageMapBuildIn,
    ImageMapMethodsOut,
    ImageMapStateOut,
    ImagePrepIn,
    ImagePrepOut,
    ImagePrepPreviewIn,
    ImagePrepPreviewOut,
    ImageSemanticIn,
    ImageSemanticOut,
)
from app.services.image_views import CLASSIC, ImageViewService

router = APIRouter(tags=["images"])


def get_image_views(request: Request) -> ImageViewService:
    return request.app.state.image_views


Views = Annotated[ImageViewService, Depends(get_image_views)]


@router.get("/images/map/methods")
def map_methods(service: Views) -> ImageMapMethodsOut:
    """Which ways of describing images can be chosen here, and why the others are greyed."""
    return ImageMapMethodsOut.model_validate(service.methods())


@router.post("/images/weights/{key}/download", status_code=202)
def download_weights(key: str, service: Views) -> ImageMapMethodsOut:
    """R-VDP-02: the page showed the size and licence and the user agreed; runs in the
    background."""
    return ImageMapMethodsOut.model_validate(service.download(key))


@router.get("/datasets/{dataset_id}/images/map")
def image_map(
    dataset_id: str,
    service: Views,
    method: str = CLASSIC,
    projection: Annotated[str, Query(pattern="^(pca|tsne|pacmap|trimap)$")] = "pca",
    clustering: Annotated[str, Query(pattern="^(kmeans|hdbscan)$")] = "kmeans",
    k: Annotated[int | None, Query(ge=2, le=30)] = None,
) -> ImageMapStateOut:
    """The map when its space is built; otherwise where its build stands (R-VMAP-10). PaCMAP and
    TriMAP (R-VMAP-12) answer 409 with their reason when the extra `interpret` cannot run here."""
    return ImageMapStateOut.model_validate(
        service.map(dataset_id, method, projection, clustering, k)
    )


@router.post("/datasets/{dataset_id}/images/map", status_code=202)
def build_image_map(dataset_id: str, body: ImageMapBuildIn, service: Views) -> ImageMapStateOut:
    """Build the space of a method in the background (a second call while it runs does nothing)."""
    return ImageMapStateOut.model_validate(
        service.build(dataset_id, body.method, body.confirm_long)
    )


@router.delete("/datasets/{dataset_id}/images/map")
def cancel_image_map(dataset_id: str, service: Views, method: str = CLASSIC) -> ImageMapStateOut:
    return ImageMapStateOut.model_validate(service.cancel(dataset_id, method))


@router.get("/datasets/{dataset_id}/images/semantic")
def semantic_duplicates(dataset_id: str, service: Views) -> ImageSemanticOut:
    """R-VQ-17 (Quality > Advanced): where the search for semantic near-copies stands, and its
    groups once done. Never starts it."""
    return ImageSemanticOut.model_validate(service.semantic(dataset_id))


@router.post("/datasets/{dataset_id}/images/semantic", status_code=202)
def find_semantic_duplicates(
    dataset_id: str, service: Views, body: ImageSemanticIn | None = None
) -> ImageSemanticOut:
    """Compute the DINOv2 vectors missing from the cache, then compare them (in the background);
    409 with the reason when the extra or the weights are missing (R-VDP-01, 02)."""
    confirm = body.confirm_long if body is not None else False
    return ImageSemanticOut.model_validate(service.find_semantic(dataset_id, confirm))


@router.delete("/datasets/{dataset_id}/images/semantic")
def cancel_semantic_duplicates(dataset_id: str, service: Views) -> ImageSemanticOut:
    return ImageSemanticOut.model_validate(service.cancel_semantic(dataset_id))


@router.get("/datasets/{dataset_id}/images/prepare")
def prepare_recipe(dataset_id: str, service: Views) -> ImagePrepOut:
    """The saved recipe (else the default for the task, R-VPR-07) and the border suggestion."""
    return ImagePrepOut.model_validate(service.recipe(dataset_id))


@router.put("/datasets/{dataset_id}/images/prepare")
def save_prepare_recipe(dataset_id: str, body: ImagePrepIn, service: Views) -> ImagePrepOut:
    """Check and keep the recipe (settings only, R-VPR-03); 422 on an unknown step or a bound."""
    return ImagePrepOut.model_validate(service.save_recipe(dataset_id, body.recipe))


@router.post("/datasets/{dataset_id}/images/prepare/preview")
def prepare_preview(
    dataset_id: str, body: ImagePrepPreviewIn, service: Views
) -> ImagePrepPreviewOut:
    """One step on one image: what reaches the step, what it makes of it, their measures."""
    return ImagePrepPreviewOut.model_validate(
        service.preview(dataset_id, body.image_id, body.recipe, body.step, body.seed)
    )
