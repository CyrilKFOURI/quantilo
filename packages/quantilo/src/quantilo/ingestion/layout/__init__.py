"""Table detection at import (feature 080): every real table of a sheet, its header and its plan.

from quantilo.ingestion.layout import detect_layout

layout = detect_layout("workbook.xlsx")
for table in layout.tables:
    print(table.id, table.header_rows, table.confidence, table.summary())
layout.read(layout.tables[0])                 # a pyarrow Table, clean-ups applied
layout.read(layout.tables[0], reshape=True)   # with the proposed long format
"""

from quantilo.ingestion.layout.detect import detect_layout, detect_range, find_table
from quantilo.ingestion.layout.model import (
    ColumnPlan,
    DetectedTable,
    Layout,
    RuleHit,
    SheetLayout,
    Transform,
)

__all__ = [
    "ColumnPlan",
    "DetectedTable",
    "Layout",
    "RuleHit",
    "SheetLayout",
    "Transform",
    "detect_layout",
    "detect_range",
    "find_table",
]
