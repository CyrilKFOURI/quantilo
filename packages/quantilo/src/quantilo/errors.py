"""Errors carrying a message a user can act on, an error code and the HTTP status the app uses."""

import difflib
import unicodedata
from typing import Any


class ExplorerError(Exception):
    def __init__(self, message: str, *, code: str = "app_error", status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        # Extra fields of the error body (e.g. the field to fix), added by subclasses.
        self.details: dict[str, Any] = {}


class NotFoundError(ExplorerError):
    def __init__(self, what: str) -> None:
        super().__init__(f"{what} not found", code="not_found", status_code=404)


class ColumnNotFound(NotFoundError):
    """An unknown column, with the closest names to suggest."""

    def __init__(self, column: str, available: list[str] | None = None) -> None:
        super().__init__(f"Column {column!r}")
        self.column = column
        self.suggestions = close_names(column, available or [])
        if self.suggestions:
            hint = ", ".join(repr(name) for name in self.suggestions)
            self.message = f"{self.message}; did you mean {hint}?"
            self.args = (self.message,)
            self.details["suggestions"] = self.suggestions


def _plain(name: str) -> str:
    """Lower case, no accents, no spaces, `_` or `-`: "Recommended IND" and "recommended_ind"
    agree (the same normalisation as recipes/matching.py)."""
    decomposed = unicodedata.normalize("NFKD", name)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c for c in plain.casefold() if not c.isspace() and c not in "_-")


def close_names(column: str, available: list[str], n: int = 3) -> list[str]:
    """The names to suggest for an unknown column (094, ML-7): the same name once normalised
    first, then the closest spellings, of the raw names and of the normalised ones."""
    wanted = _plain(column)
    found = [name for name in available if _plain(name) == wanted]
    plain = {}
    for name in available:
        plain.setdefault(_plain(name), name)
    close = difflib.get_close_matches(column, available, n=n, cutoff=0.7)
    close += [plain[p] for p in difflib.get_close_matches(wanted, list(plain), n=n, cutoff=0.7)]
    for name in close:
        if name not in found:
            found.append(name)
    return found[:n]
