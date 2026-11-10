"""JSON, JSON Lines and XML documents → linked tables, every rule explained (083).

    layout = read_structured("orders.json")
    layout.summary()          # "3 tables from orders.json: orders (4 rows), orders.items (8), …"
    layout.arrow("orders.items")
    layout.write_parquet("out/")

Standard library (json, xml.parsers.expat) and pyarrow only: it runs in the browser (Pyodide).
"""

from quantilo.ingestion.structured.engine import read_structured
from quantilo.ingestion.structured.errors import StructuredError
from quantilo.ingestion.structured.model import (
    RecordsCandidate,
    Relation,
    RuleHit,
    StructuredColumn,
    StructuredLayout,
    StructuredOptions,
    StructuredTable,
)

__all__ = [
    "RecordsCandidate",
    "Relation",
    "RuleHit",
    "StructuredColumn",
    "StructuredError",
    "StructuredLayout",
    "StructuredOptions",
    "StructuredTable",
    "read_structured",
]
