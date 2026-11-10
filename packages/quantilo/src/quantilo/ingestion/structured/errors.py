"""The one error of the structured reader: a message a user can act on."""

from quantilo.errors import ExplorerError


class StructuredError(ExplorerError):
    """A document that can't be read into tables (malformed, too big, too deep, unsafe XML)."""

    def __init__(self, message: str, *, rule: str | None = None) -> None:
        super().__init__(message, code="structured_error", status_code=422)
        self.rule = rule
        if rule:
            self.details["rule"] = rule
