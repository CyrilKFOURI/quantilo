"""Errors carrying a message a user can act on, an error code and the HTTP status the app uses."""

import difflib
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
        self.suggestions = difflib.get_close_matches(column, available or [], n=3, cutoff=0.7)
        if self.suggestions:
            hint = ", ".join(repr(name) for name in self.suggestions)
            self.message = f"{self.message}; did you mean {hint}?"
            self.args = (self.message,)
            self.details["suggestions"] = self.suggestions
