"""Machine learning page (docs/pages/11-machine-learning.md)."""

from typing import Any

from fastapi import FastAPI

from app.modules.ml.agent_tools import agent_tools as training_tools
from app.modules.ml.auto import AutoService
from app.modules.ml.auto_router import router as auto_router
from app.modules.ml.predict_router import router as predict_router
from app.modules.ml.predict_tools import prediction_tools
from app.modules.ml.predictions import PredictionService
from app.modules.ml.router import router
from app.modules.ml.service import MlService

router.include_router(predict_router)  # predict on new data, export the model (037)
router.include_router(auto_router)  # the Auto (038), on the same prefix


def setup(app: FastAPI) -> None:
    app.state.ml = MlService(app.state.columns, app.state.workspace)
    app.state.ml_predictions = PredictionService(app.state.ml, getattr(app.state, "versions", None))
    app.state.datasets.after_import.append(app.state.ml_predictions.after_import)
    app.state.ml_auto = AutoService(app.state.ml)


def agent_tools(app: FastAPI) -> list[Any]:
    return [*training_tools(app), *prediction_tools(app)]


__all__ = ["agent_tools", "router", "setup"]
