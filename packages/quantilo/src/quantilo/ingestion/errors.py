import zipfile
from collections.abc import Iterator
from contextlib import contextmanager

import duckdb
from openpyxl.utils.exceptions import InvalidFileException

from quantilo.errors import ExplorerError


class IngestionError(ExplorerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="ingestion_error", status_code=422)


def describe_duckdb_error(message: str) -> str:
    first_line = message.strip().splitlines()[0] if message.strip() else "Unknown error"
    for prefix in ("Invalid Input Error: ", "Conversion Error: ", "IO Error: ", "Binder Error: "):
        first_line = first_line.removeprefix(prefix)
    if "Invalid unicode" in message:
        return "This file is not UTF-8 encoded: try another encoding (latin-1)."
    if "Could not convert" in message or "CSV Error" in message:
        return f"Could not read this CSV ({first_line}). Check the delimiter and decimal separator."
    return first_line


@contextmanager
def readable_errors() -> Iterator[None]:
    """Turn library exceptions into IngestionError with a message a user can act on."""
    try:
        yield
    except IngestionError:
        raise
    except duckdb.Error as error:
        raise IngestionError(describe_duckdb_error(str(error))) from error
    except (zipfile.BadZipFile, InvalidFileException) as error:
        raise IngestionError(
            "This Excel file can't be read: it may be password-protected or corrupted."
        ) from error
    except UnicodeDecodeError as error:
        raise IngestionError("This file is not UTF-8 encoded: try another encoding.") from error
