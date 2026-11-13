"""Vision models (112): the task rule (R-VML), training and evaluation jobs, kept runs.

The pages (Model, Results) come with 113; the import and the image files are 106
(`app.services.image_sets`); the explanations of a run ("Why?", Explain) are 117; the
Advanced > More models, the concepts and TCAV are 118."""

from fastapi import FastAPI

from app.modules.vision.explain import ExplainService
from app.modules.vision.explain_router import router as explain_router
from app.modules.vision.more_router import router as more_router
from app.modules.vision.router import router
from app.modules.vision.service import VisionService

router.include_router(explain_router)  # 117
router.include_router(more_router)  # 118: concepts, ProtoPNet, bottleneck, TCAV


def setup(app: FastAPI) -> None:
    settings = getattr(app.state, "settings", None)

    def allow_gpu() -> bool:
        """Settings > Compute: "Use the GPU when available" (unset: when a card is ready)."""
        if settings is None:
            return True
        choice = settings.get().compute.use_gpu
        return True if choice is None else bool(choice)

    app.state.vision = VisionService(
        app.state.workspace, app.state.datasets, app.state.image_sets, allow_gpu
    )
    app.state.vision_explain = ExplainService(app.state.vision)
    views = getattr(app.state, "image_views", None)
    if views is not None:  # b2: the semantic near-copies (R-VQ-17) follow the same setting
        views.allow_gpu = allow_gpu


__all__ = ["router", "setup"]
