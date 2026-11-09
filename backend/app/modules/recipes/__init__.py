"""Recipes (030): keep the work done on a dataset and replay it on another file.

See docs/pages/16-recettes.md. The module reaches the other modules' services through app.state
(never by importing them); set up after them in modules/registry.py.
"""

from fastapi import FastAPI

from app.modules.recipes.agent_tools import agent_tools
from app.modules.recipes.router import router
from app.modules.recipes.service import RecipeService
from app.modules.recipes.target import Services


def setup(app: FastAPI) -> None:
    state = app.state
    services = Services(
        datasets=state.datasets,
        columns=state.columns,
        saved_filters=state.saved_filters,
        reports=state.reports,
        rows=state.rows,
        versions=state.versions,
        kpi=getattr(state, "kpi", None),
        timeseries=getattr(state, "timeseries", None),
        analyses=getattr(state, "ts_analyses", None),
        ml=getattr(state, "ml", None),
    )
    state.recipes = RecipeService(state.workspace, services, lite=state.datasets.settings.lite)
    state.versions.usages.append(state.recipes.usage)  # a column a recipe reads: Breaking if gone


__all__ = ["agent_tools", "router", "setup"]
