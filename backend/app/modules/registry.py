"""Optional modules. Adding a module = creating its package and listing it here."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, FastAPI

from app.modules import agent, experiments, kpi, ml, recipes, segments, synthetic, timeseries


@dataclass(frozen=True)
class Module:
    id: str
    router: APIRouter
    setup: Callable[[FastAPI], None] | None = None  # creates the module's services on app.state
    # Tools the module offers to the AI agent (app.core.tools contract), built after every setup.
    agent_tools: Callable[[FastAPI], list[Any]] | None = None


MODULES: list[Module] = [
    Module(
        id="timeseries",
        router=timeseries.router,
        setup=timeseries.setup,
        agent_tools=timeseries.agent_tools,
    ),
    Module(id="ml", router=ml.router, setup=ml.setup, agent_tools=ml.agent_tools),
    Module(
        id="segments",
        router=segments.router,
        setup=segments.setup,
        agent_tools=segments.agent_tools,
    ),
    Module(id="kpi", router=kpi.router, setup=kpi.setup, agent_tools=kpi.agent_tools),
    Module(
        id="synthetic",
        router=synthetic.router,
        setup=synthetic.setup,
        agent_tools=synthetic.agent_tools,
    ),
    Module(
        id="experiments",
        router=experiments.router,
        setup=experiments.setup,
        agent_tools=experiments.agent_tools,
    ),
    Module(id="agent", router=agent.router, setup=agent.setup),
    # after the modules whose services it replays (030)
    Module(
        id="recipes", router=recipes.router, setup=recipes.setup, agent_tools=recipes.agent_tools
    ),
]


def mount_modules(app: FastAPI, prefix: str, modules: list[Module] | None = None) -> None:
    chosen = MODULES if modules is None else modules
    for module in chosen:
        if module.setup is not None:
            module.setup(app)
        app.include_router(module.router, prefix=f"{prefix}/modules/{module.id}")
    # every service exists now: collect the tools modules offer to the agent
    app.state.agent_tools = [
        tool for module in chosen if module.agent_tools for tool in module.agent_tools(app)
    ]
