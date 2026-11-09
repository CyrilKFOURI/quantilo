"""KPI builder page (docs/pages/12-kpi-builder.md)."""

from fastapi import FastAPI

from app.modules.kpi.agent_tools import agent_tools
from app.modules.kpi.router import router
from app.modules.kpi.service import KpiService


def setup(app: FastAPI) -> None:
    app.state.kpi = KpiService(app.state.columns, app.state.workspace)
    clearers = getattr(app.state, "settings", None)
    if clearers is not None:
        clearers.cache_clearers.append(app.state.kpi.clear)
    versions = getattr(app.state, "versions", None)
    if versions is not None:  # what changed between two versions (031) knows the saved KPIs
        versions.usages.append(app.state.kpi.usage)
        versions.saved_kpis.append(app.state.kpi.definitions)


__all__ = ["agent_tools", "router", "setup"]
