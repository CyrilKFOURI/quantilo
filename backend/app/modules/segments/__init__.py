"""Segments page (docs/pages/14-segments.md): segments (032) and unusual rows (033)."""

from fastapi import FastAPI

from app.modules.segments.agent_tools import agent_tools
from app.modules.segments.router import router
from app.modules.segments.service import SegmentsService


def setup(app: FastAPI) -> None:
    app.state.segments = SegmentsService(app.state.columns, app.state.workspace)
    app.state.datasets.after_import.append(app.state.segments.after_import)


__all__ = ["agent_tools", "router", "setup"]
