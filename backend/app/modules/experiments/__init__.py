"""Experiments page: analyze an A/B test from the dataset, or plan one (sample size).

See docs/pages/15-experiments.md. Nothing is stored: an experiment is described by the page's URL.
"""

from fastapi import FastAPI
from quantilo.experiments.service import ExperimentEngine

from app.modules.experiments.agent_tools import agent_tools
from app.modules.experiments.router import router


def setup(app: FastAPI) -> None:
    app.state.experiments = ExperimentEngine(app.state.columns)
    clearers = getattr(app.state, "settings", None)
    if clearers is not None:
        clearers.cache_clearers.append(app.state.experiments.clear)


__all__ = ["agent_tools", "router", "setup"]
