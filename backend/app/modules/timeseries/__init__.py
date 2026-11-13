"""Time series page: one series, analysed section by section.

See docs/pages/10-series-temporelles.md.
"""

import os

from fastapi import FastAPI
from quantilo.timeseries import ai
from quantilo.timeseries.service import TimeSeriesService

from app.modules.timeseries import analyses, forecast_ai
from app.modules.timeseries.agent_tools import agent_tools
from app.modules.timeseries.router import router

router.include_router(analyses.router)  # analyses kept with "Save analysis" (030)
router.include_router(forecast_ai.router)  # AI forecasting models and their weights (028 B1)
MODELS_DIR = os.environ.get("QUANTILO_MODELS_DIR")  # set before the app starts: it wins


def setup(app: FastAPI) -> None:
    app.state.timeseries = TimeSeriesService(app.state.columns)
    app.state.ts_analyses = analyses.AnalysisService(app.state.workspace, app.state.datasets)
    clearers = getattr(app.state, "settings", None)
    if clearers is not None:
        clearers.cache_clearers.append(app.state.timeseries.clear)
    # 028 B1: the AI models' weights live in <home>/models, never in the global cache
    workspace = getattr(app.state, "workspace", None)
    if workspace is not None and not MODELS_DIR:
        os.environ["QUANTILO_MODELS_DIR"] = str(workspace.root / "models")
    app.state.forecast_ai = forecast_ai.ForecastAiService()

    def allow_gpu() -> bool:
        """Settings > Compute: "Use the GPU when available" (unset: when a card is ready)."""
        if clearers is None:
            return True
        choice = clearers.get().compute.use_gpu
        return True if choice is None else bool(choice)

    ai.set_gpu_policy(allow_gpu)


__all__ = ["agent_tools", "router", "setup"]
