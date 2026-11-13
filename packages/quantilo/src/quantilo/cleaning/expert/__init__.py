"""Clean's Expert layer (101, M6, study § 2.3), shown as *Advanced › More checks*: learnt methods,
run one at a time on demand (C-29), cancellable (C-31), whose results are suggestions in mode S,
risk "review", never ticked (C-30).

- E-06 `unusual_rows`: Isolation Forest, as in Unusual rows (033).
- E-03 `suspicious_labels`: confident learning on out-of-sample probabilities.
- E-07 `fill_empty`: KNN or iterative imputation (MICE), each value with its band.
- E-01 / E-02 `cell_errors`: errors learnt from a few marks (Raha / Baran in light).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Literal

import duckdb
from pydantic import BaseModel, ConfigDict, Field

from quantilo.catalog import ColumnView
from quantilo.cleaning.expert import cells as cell_errors
from quantilo.cleaning.expert import data as d
from quantilo.cleaning.expert import impute, labels, unusual
from quantilo.cleaning.model import CleanError
from quantilo.cleaning.suggestion import CleanSuggestion

MethodId = Literal["unusual_rows", "suspicious_labels", "fill_empty", "cell_errors"]
METHOD_IDS: tuple[MethodId, ...] = (
    "unusual_rows",
    "suspicious_labels",
    "fill_empty",
    "cell_errors",
)
RULES: dict[MethodId, str] = {
    "unusual_rows": "E-06",
    "suspicious_labels": "E-03",
    "fill_empty": "E-07",
    "cell_errors": "E-01",
}


class ExpertMethod(BaseModel):
    """A method of *More checks* for this dataset, read from its columns only (C-29)."""

    id: MethodId
    rule: str
    available: bool
    why_not: str | None = None  # when it does not apply here
    columns: list[str] = Field(default_factory=list)  # what it can be run on (a target, a column)


class ExpertMark(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row: int = Field(ge=0)  # in the file, from 0
    error: bool


class ExpertOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str | None = Field(default=None, max_length=500)  # labels: target; cells: column
    columns: list[str] | None = Field(default=None, max_length=50)  # fill_empty
    imputer: impute.Imputer = "knn"
    marks: list[ExpertMark] | None = Field(default=None, max_length=200)


class CellToMark(BaseModel):
    row: int
    value: str | None
    hint: str  # what sets its group apart ("a rare value")
    alike: int  # cells of its group


class ExpertResult(BaseModel):
    method: MethodId
    rule: str
    suggestions: list[CleanSuggestion] = Field(default_factory=list)
    note: str | None = None  # why nothing was found, or what was left out
    to_mark: list[CellToMark] = Field(default_factory=list)  # cell_errors before any mark
    rows: int = 0  # rows read
    sampled: bool = False
    seconds: float = 0.0


def methods(views: Sequence[ColumnView]) -> list[ExpertMethod]:
    """The four methods, whether each applies to these columns and on what (no row is read)."""
    found = [
        ExpertMethod(
            id="unusual_rows", rule="E-06", available=False, why_not=unusual.why_not(views),
        ),
        ExpertMethod(
            id="suspicious_labels", rule="E-03", available=False,
            why_not=labels.why_not(views), columns=labels.targets(views),
        ),
        ExpertMethod(
            id="fill_empty", rule="E-07", available=False,
            why_not=impute.why_not(views), columns=impute.targets(views),
        ),
        ExpertMethod(
            id="cell_errors", rule="E-01", available=False,
            why_not=cell_errors.why_not(views), columns=cell_errors.columns(views),
        ),
    ]  # fmt: skip
    for method in found:
        method.available = method.why_not is None
    return found


def run(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    views: Sequence[ColumnView],
    method: MethodId,
    options: ExpertOptions | None = None,
    stopped: d.Stopped = lambda: False,
) -> ExpertResult:
    """Run one method (C-29); `stopped` is asked between its stages (C-31)."""
    if method not in METHOD_IDS:
        raise CleanError(f"Unknown check {method!r}.", status_code=404)
    options = options or ExpertOptions()
    started = time.monotonic()
    with d.one_thread():
        found, note, frame, marks = _dispatch(
            connection, parquet_path, views, method, options, stopped
        )
    return ExpertResult(
        method=method,
        rule=RULES[method],
        suggestions=found,
        note=note,
        to_mark=marks,
        rows=len(frame) if frame is not None else d.rows_of(views),
        sampled=frame.sampled if frame is not None else False,
        seconds=round(time.monotonic() - started, 2),
    )


def _dispatch(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    views: Sequence[ColumnView],
    method: MethodId,
    options: ExpertOptions,
    stopped: d.Stopped,
) -> tuple[list[CleanSuggestion], str | None, d.Frame | None, list[CellToMark]]:
    marks: list[CellToMark] = []
    frame: d.Frame | None = None
    if method == "unusual_rows":
        found, note, frame = unusual.run(connection, parquet_path, views, stopped)
    elif method == "suspicious_labels":
        found, note, frame = labels.run(connection, parquet_path, views, options.column, stopped)
    elif method == "fill_empty":
        found, note = impute.run(
            connection, parquet_path, views, options.imputer, options.columns, stopped
        )
    else:
        given = [cell_errors.Mark(m.row, m.error) for m in options.marks or []]
        found, note, cells = cell_errors.run(
            connection, parquet_path, views, options.column, given or None, stopped
        )
        marks = [CellToMark(row=c.row, value=c.value, hint=c.hint, alike=c.alike) for c in cells]
    return found, note, frame, marks


__all__ = [
    "METHOD_IDS",
    "CellToMark",
    "ExpertMark",
    "ExpertMethod",
    "ExpertOptions",
    "ExpertResult",
    "MethodId",
    "methods",
    "run",
]
