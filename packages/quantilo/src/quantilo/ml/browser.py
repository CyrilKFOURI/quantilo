"""What machine learning does differently in the browser version (102): Python runs in the page
(Pyodide, WebAssembly), on one core, with little memory. The ML module is the same; these rules,
each with its code and reason, are read where the module would otherwise assume a computer.

- R-WEB-01: at most 50 000 training rows (a reproducible sample beyond), not 200 000 or every row.
- R-WEB-02: one core (`n_jobs = 1`): WebAssembly has no threads and no processes; so a random
  forest or extra trees grow 100 trees by default instead of 300 (the setting says why).
- R-WEB-03: the Auto spends at most 5 minutes: the page waits for it and cannot stop it midway.
- R-WEB-04: XGBoost, LightGBM, CatBoost and tuning (Optuna) are left to the desktop app.
"""

import contextlib
import sys
from typing import Any

IN_BROWSER = sys.platform == "emscripten"
MAX_ROWS = 50_000  # R-WEB-01
AUTO_MINUTES = 5.0  # R-WEB-03
FOREST_TREES = 100  # R-WEB-02: a forest's trees by default on one core (300 on a computer)
DESKTOP_ONLY = "available in the desktop app"


def active() -> bool:
    """True when the engine runs in the browser (read at call time, so tests can simulate it)."""
    return IN_BROWSER


def n_jobs() -> int:
    """R-WEB-02: every core on a computer, one in the browser."""
    return 1 if active() else -1


def max_rows(limit: int | None) -> int | None:
    """R-WEB-01: the training rows asked for, held to 50 000 in the browser."""
    if not active():
        return limit
    return MAX_ROWS if limit is None else min(limit, MAX_ROWS)


def auto_minutes(minutes: float) -> float:
    """R-WEB-03: the Auto's budget, held to 5 minutes in the browser."""
    return min(minutes, AUTO_MINUTES) if active() else minutes


def desktop_only(label: str) -> str:
    """The sentence said wherever something left to the desktop app is asked for (R-WEB-04)."""
    return f"{label} is {DESKTOP_ONLY}."


def single_core(pipeline: Any) -> Any:
    """R-WEB-02 for a model trained elsewhere (an exported zip): every `n_jobs` set to 1."""
    if not active() or not hasattr(pipeline, "get_params"):
        return pipeline
    found = {key: 1 for key in pipeline.get_params(deep=True) if key.split("__")[-1] == "n_jobs"}
    if found:
        with contextlib.suppress(ValueError):  # a nested estimator refuses: leave it as it is
            pipeline.set_params(**found)
    return pipeline
