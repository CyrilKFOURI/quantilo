"""The source as it is, for the left pane of the import screen (080b, 085).

A sheet gives its top-left window of cells (200 rows x 50 columns, values as text, merges kept);
a document gives its first lines. Nothing is interpreted here: the right pane shows the result.
"""

import contextlib
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.layout.grid import Limits, load_grids

MAX_ROWS = 200
MAX_COLS = 50
MAX_LINES = 200
MAX_TEXT_BYTES = 64 * 1024


@dataclass
class Original:
    kind: Literal["sheet", "text"]
    sheet: str | None = None
    cells: list[list[str | None]] = field(default_factory=list)  # row 1 first, column A first
    merges: list[list[int]] = field(default_factory=list)  # [top, left, bottom, right], 1-based
    text: str | None = None
    truncated: bool = False


def _shown(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.isoformat(sep=" ").removesuffix(" 00:00:00")
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return str(value)


def sheet_cells(path: Path, sheet: str) -> Original:
    grids, _ = load_grids(path, [sheet], Limits(max_rows=MAX_ROWS, max_cols=MAX_COLS))
    if not grids:
        raise IngestionError(f"No sheet {sheet!r} in {path.name}.")
    grid = grids[0]
    return Original(
        kind="sheet",
        sheet=sheet,
        cells=[[_shown(value) for value in row] for row in grid.values],
        merges=[
            [top + 1, left + 1, bottom + 1, right + 1] for top, left, bottom, right in grid.merges
        ],
        truncated=grid.truncated or grid.n_cols >= MAX_COLS,
    )


def text_head(path: Path) -> Original:
    with path.open("rb") as handle:
        head = handle.read(MAX_TEXT_BYTES + 1)
    truncated = len(head) > MAX_TEXT_BYTES
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = head[:MAX_TEXT_BYTES].decode("utf-16", errors="replace")
    else:
        text = head[:MAX_TEXT_BYTES].decode("utf-8", errors="replace").removeprefix("﻿")
    lines = text.splitlines()
    if len(lines) <= 2 and not truncated:  # a minified JSON document: shown indented, unchanged
        with contextlib.suppress(ValueError, RecursionError):
            lines = json.dumps(json.loads(text), indent=2, ensure_ascii=False).splitlines()
    if len(lines) > MAX_LINES:
        lines, truncated = lines[:MAX_LINES], True
    return Original(kind="text", text="\n".join(lines), truncated=truncated)
