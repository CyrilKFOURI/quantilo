"""Time series page: one series, analysed section by section.

See docs/pages/10-series-temporelles.md.
"""

from fastapi import FastAPI
from quantilo.timeseries.service import TimeSeriesService

from app.modules.timeseries import analyses
from app.modules.timeseries.agent_tools import agent_tools
from app.modules.timeseries.router import router

router.include_router(analyses.router)  # analyses kept with "Save analysis" (030)


def setup(app: FastAPI) -> None:
    app.state.timeseries = TimeSeriesService(app.state.columns)
    app.state.ts_analyses = analyses.AnalysisService(app.state.workspace, app.state.datasets)
    clearers = getattr(app.state, "settings", None)
    if clearers is not None:
        clearers.cache_clearers.append(app.state.timeseries.clear)


__all__ = ["agent_tools", "router", "setup"]
