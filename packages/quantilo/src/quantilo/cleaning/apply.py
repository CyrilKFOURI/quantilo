"""Write the cleaned child: its Parquet (parent's row order) and `rowmap.parquet` (child row →
parent row, plus the parent rows removed and the step that removed them). The parent is only
read; both files are written beside their target then renamed, so a failure leaves nothing."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb

from quantilo.cleaning.compile import ROW, compile_steps
from quantilo.cleaning.model import CleanError
from quantilo.cleaning.preview import compare
from quantilo.cleaning.report import CleanReport, StepReport
from quantilo.ingestion.sql import ident, lit


class CleanCancelled(CleanError):
    def __init__(self) -> None:
        super().__init__("Stopped: nothing was saved.", code="cancelled", status_code=409)


def apply_steps(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    steps: list[Any],
    target: Path,
    rowmap: Path,
    stopped: Callable[[], bool] = lambda: False,
) -> CleanReport:
    started = time.monotonic()
    compiled = compile_steps(connection, parquet_path, steps, strict=True)
    if stopped():
        raise CleanCancelled
    overall = compare(connection, compiled, 0, -1)
    if stopped():
        raise CleanCancelled
    target.parent.mkdir(parents=True, exist_ok=True)
    rowmap.parent.mkdir(parents=True, exist_ok=True)
    data_partial = target.with_name(target.name + ".partial")
    map_partial = rowmap.with_name(rowmap.name + ".partial")
    row = ident(ROW)
    try:
        query = f"{compiled.sql(-1, compiled.output_columns())} ORDER BY {row}"
        connection.execute(
            f"COPY ({query}) TO {lit(str(data_partial))} (FORMAT parquet, COMPRESSION zstd)"
        )
        if stopped():
            raise CleanCancelled
        final = compiled.final.name
        parts = [
            f"SELECT CAST(row_number() OVER (ORDER BY {row}) - 1 AS BIGINT) AS child_row, "
            f"CAST({row} AS BIGINT) AS parent_row, CAST(NULL AS VARCHAR) AS step FROM {final}"
        ]
        for index, stage in enumerate(compiled.stages):
            if not stage.removes_rows:
                continue
            previous = compiled.stages[index - 1].name
            parts.append(
                f"SELECT CAST(NULL AS BIGINT), CAST(r AS BIGINT), {lit(stage.step.id)} FROM "
                f"(SELECT {row} AS r FROM {previous} EXCEPT SELECT {row} FROM {stage.name})"
            )
        rowmap_sql = f"WITH {', '.join(compiled.ctes)} {' UNION ALL '.join(parts)}"
        connection.execute(
            f"COPY ({rowmap_sql} ORDER BY parent_row) TO {lit(str(map_partial))} "
            "(FORMAT parquet, COMPRESSION zstd)"
        )
        removed_by = dict(
            connection.execute(
                f"SELECT step, count(*) FROM read_parquet({lit(str(map_partial))}) "
                "WHERE step IS NOT NULL GROUP BY step"
            ).fetchall()
        )
        if stopped():
            raise CleanCancelled
        data_partial.replace(target)
        map_partial.replace(rowmap)
    finally:
        data_partial.unlink(missing_ok=True)
        map_partial.unlink(missing_ok=True)
    return CleanReport(
        rows_before=overall.rows_before,
        rows_after=overall.rows_after,
        rows_removed=overall.rows_removed,
        cells_changed=overall.cells_changed,
        columns_before=len(compiled.stages[0].columns),
        columns_after=len(compiled.final.columns),
        cells_by_column={c.column: c.cells for c in overall.columns if c.cells},
        steps=[
            StepReport(
                stage.step.id,
                stage.step.op,
                stage.step.rule,
                stage.step.origin,
                int(removed_by.get(stage.step.id, 0)),
            )
            for stage in compiled.stages[1:]
        ],
        added=overall.added,
        dropped=overall.dropped,
        seconds=round(time.monotonic() - started, 2),
    )
