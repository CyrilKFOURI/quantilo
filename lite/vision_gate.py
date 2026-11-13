"""Vision in the browser version, loaded on demand (119, R-VIS-WEB-04).

The Vision routes import Pillow (about 5 MB), which the page does not load when it starts: the
start stays as fast as before. Until then `app.modules.vision` is a stand-in with no route; the
worker (frontend/src/lite/worker.ts) loads Pillow before the first request that needs it, then
`enable()` mounts the real module where the registry would have (`/api/v1/modules/vision`).
"""

import importlib
import importlib.util
import sys
import types
from typing import Any

VISION = "app.modules.vision"


class StandIn(types.ModuleType):
    """`app.modules.vision` until Pillow is loaded: no route, nothing set up, no agent tool."""

    def __init__(self) -> None:
        super().__init__(VISION, "Vision, loaded with Pillow when first needed (R-VIS-WEB-04).")
        from fastapi import APIRouter

        self.router = APIRouter()

    def setup(self, app: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return None  # an optional hook of the registry (agent_tools...): none yet


def pillow_loaded() -> bool:
    return importlib.util.find_spec("PIL") is not None


def defer() -> bool:
    """Put the stand-in in place when Pillow is not loaded (before the app is imported)."""
    if pillow_loaded() or VISION in sys.modules:
        return False
    sys.modules[VISION] = StandIn()
    return True


def deferred() -> bool:
    return isinstance(sys.modules.get(VISION), StandIn)


def enable(app: Any, prefix: str) -> bool:
    """Mount the real Vision module, once Pillow is loaded: what `mount_modules` does for it."""
    stand_in = sys.modules.get(VISION)
    if not isinstance(stand_in, StandIn):
        return False
    if not pillow_loaded():
        raise RuntimeError("Pillow is not loaded: Vision cannot start (R-VIS-WEB-04).")
    del sys.modules[VISION]
    package = sys.modules.get("app.modules")
    if package is not None and getattr(package, "vision", None) is stand_in:
        delattr(package, "vision")
    real = importlib.import_module(VISION)
    if getattr(real, "setup", None) is not None:
        real.setup(app)
    app.include_router(real.router, prefix=f"{prefix}/modules/vision")
    tools = getattr(real, "agent_tools", None)
    if tools is not None:
        app.state.agent_tools = [*getattr(app.state, "agent_tools", []), *tools(app)]
    return True
