"""A JSON, JSON Lines or XML document read once for its preview and the import of its tables (085).

The structured reader (083) parses the whole document to give every linked table; the preview and
then the import of each selected table reuse the same parse, kept here for a few documents.
"""

import threading
from collections import OrderedDict
from pathlib import Path

from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.structured import StructuredError, StructuredLayout, StructuredOptions
from quantilo.ingestion.structured import read_structured as read

_CACHE: OrderedDict[tuple[object, ...], StructuredLayout] = OrderedDict()
_CACHE_SIZE = 4
_LOCK = threading.Lock()


def read_document(
    path: Path, records_path: str | None = None, one_table: bool = False
) -> StructuredLayout:
    """The document's linked tables; the structured errors become import errors (their rule)."""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, records_path, one_table)
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
        try:
            layout = read(path, StructuredOptions(records_path=records_path, one_table=one_table))
        except StructuredError as error:
            rule = f" ({error.rule})" if getattr(error, "rule", None) else ""
            raise IngestionError(f"{path.name}: {error}{rule}") from error
        except RecursionError as error:
            raise IngestionError(f"{path.name}: nesting too deep (R-LIM-02).") from error
        _CACHE[key] = layout
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
        return layout


def main_table(layout: StructuredLayout) -> str:
    """The table shown and imported first: the first selected table that is not metadata."""
    for table in layout.tables:
        if table.kind != "meta" and table.selected:
            return table.name
    for table in layout.tables:
        if table.kind != "meta":
            return table.name
    if not layout.tables:
        raise IngestionError(f"No table found in {Path(layout.source).name}.")
    return layout.tables[0].name
