"""Open an Excel workbook whatever its name (082, R-DET-01).

openpyxl refuses a path whose extension is not .xlsx/.xlsm/.xltx/.xltm; a workbook renamed
`.xls` or `.dat` is still a workbook, so it is opened through a file handle, closed with it.
"""

from pathlib import Path
from typing import Any

import openpyxl

OPENPYXL_SUFFIXES = (".xlsx", ".xlsm", ".xltx", ".xltm")


def load_workbook(path: Path | str, **options: Any) -> Any:
    path = Path(path)
    if path.suffix.lower() in OPENPYXL_SUFFIXES:
        return openpyxl.load_workbook(path, **options)
    handle = path.open("rb")
    try:
        workbook = openpyxl.load_workbook(handle, **options)
    except Exception:
        handle.close()
        raise
    close = workbook.close

    def close_both() -> None:
        try:
            close()
        finally:
            handle.close()

    workbook.close = close_both
    return workbook
