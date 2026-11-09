"""Errors carrying a message a user can act on, an error code and an HTTP status.

The engine's error class: the app turns any of them into an HTTP response (app.api.errors).
"""

from quantilo.errors import ExplorerError as AppError

__all__ = ["AppError"]
