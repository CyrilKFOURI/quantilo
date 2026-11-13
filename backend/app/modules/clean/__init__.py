"""Clean page (docs/pages/17-nettoyage.md, feature 065): suggestions, steps, cleaned versions."""

from fastapi import FastAPI

from app.modules.clean.agent_tools import agent_tools
from app.modules.clean.router import router
from app.modules.clean.service import CleanService


def setup(app: FastAPI) -> None:
    """`QUANTILO_CLEAN_ENABLED=false` turns the module off: its routes answer 404 (S8)."""
    if not app.state.datasets.settings.clean_enabled:
        app.state.clean = None
        return
    app.state.clean = CleanService(app.state.columns, app.state.workspace)
    app.state.datasets.after_import.append(app.state.clean.after_import)
    app.state.versions.lineages.append(app.state.clean.row_link)  # rows paired in Changes (095)
    app.state.pins.lineages.append(app.state.clean.row_link)  # pins followed (099 bis, R-PIN-16)


__all__ = ["agent_tools", "router", "setup"]
