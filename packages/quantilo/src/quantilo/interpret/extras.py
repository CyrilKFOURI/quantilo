"""R-XAI-10: the optional methods of the `interpret` extra, and why one cannot run here.

Their packages are heavy (C++ solvers, numba, faiss) and left to the owner's choice: they are
never installed by default and never imported when the engine starts. Each method asks
`require()` at call time; the interface greys out what `available()` refuses, with its reason.
"""

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from types import ModuleType

from quantilo.ml import browser

INSTALL = 'pip install "quantilo[interpret]"'
# gosdt publishes wheels for CPython 3.9 to 3.12 only (the Docker image runs 3.12)
GOSDT_LAST_PYTHON = (3, 12)


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    package: str  # the module imported
    license: str


METHODS = {
    "gosdt": Method("gosdt", "Optimal tree (GOSDT)", "gosdt", "BSD-3"),
    "treefarms": Method("treefarms", "Rashomon set (TreeFARMS)", "treefarms", "BSD-3"),
    "pacmap": Method("pacmap", "PaCMAP", "pacmap", "Apache-2.0"),
    "trimap": Method("trimap", "TriMAP", "trimap", "Apache-2.0"),
}


@dataclass
class Availability:
    key: str
    label: str
    available: bool
    reason: str | None  # R-XAI-10, when it is not
    install: str | None  # the command that brings it


class ExtraMissing(RuntimeError):
    """An optional method asked for while its package is absent: the message is the reason."""


def _installed(package: str) -> bool:
    """Without importing it (find_spec only reads the import path)."""
    if package in sys.modules:
        return True
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def check(key: str) -> Availability:
    method = METHODS[key]
    if browser.active():
        reason = f"{method.label} needs the desktop app: its solver is not in the browser version."
        return Availability(key, method.label, False, f"R-XAI-10: {reason}", None)
    if _installed(method.package):
        return Availability(key, method.label, True, None, None)
    if key == "gosdt" and sys.version_info[:2] > GOSDT_LAST_PYTHON:
        version = f"{sys.version_info[0]}.{sys.version_info[1]}"
        reason = (
            f"R-XAI-10: gosdt has no wheel for Python {version} (only 3.9 to 3.12): use the "
            "Docker image (Python 3.12), or a Python 3.12 environment with the interpret extra."
        )
        return Availability(key, method.label, False, reason, INSTALL)
    if key == "treefarms" and sys.platform == "win32" and sys.version_info[:2] > (3, 7):
        # its Windows wheel (0.2.4) links python37.dll: it installs but cannot load
        reason = (
            "R-XAI-10: treefarms' Windows build only loads on Python 3.7: use the Docker image "
            "(Linux), or Linux / macOS with the interpret extra."
        )
        return Availability(key, method.label, False, reason, None)
    reason = f"R-XAI-10: {method.package} is not installed (optional extra `interpret`)."
    return Availability(key, method.label, False, reason, INSTALL)


def available() -> list[Availability]:
    """Every optional method, and whether it can run here."""
    return [check(key) for key in METHODS]


def require(key: str) -> ModuleType:
    """Import the method's package, or raise `ExtraMissing` with the reason."""
    found = check(key)
    if not found.available:
        message = found.reason or f"{found.label} is not available."
        raise ExtraMissing(f"{message} {found.install}".strip() if found.install else message)
    return importlib.import_module(METHODS[key].package)
