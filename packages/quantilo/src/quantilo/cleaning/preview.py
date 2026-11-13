"""Before / after: counts on the whole file in one query, and the first rows touched (C-10)."""

from dataclasses import dataclass, field

import duckdb

from quantilo.cleaning.compile import ROW, Compiled, Stage, as_text
from quantilo.cleaning.model import CleanError
from quantilo.ingestion.sql import ident

SAMPLE_ROWS = 50
SHOWN_COLUMNS = 12


@dataclass
class ColumnDiff:
    column: str  # its name after
    before: str | None  # its name before (None: added)
    cells: int  # cells whose shown value changed (kept rows)
    dtype_before: str | None = None
    dtype_after: str | None = None


@dataclass
class PreviewRow:
    row: int  # the parent's row number (0-based)
    removed: bool
    before: list[str | None]
    after: list[str | None]
    changed: list[bool]


@dataclass
class CleanPreview:
    rows_before: int
    rows_after: int
    rows_removed: int
    cells_changed: int
    columns: list[ColumnDiff]  # every column after, with its change count
    added: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    shown: list[str] = field(default_factory=list)  # the columns of `sample`
    sample: list[PreviewRow] = field(default_factory=list)
    problems: dict[str, str] = field(default_factory=dict)
    # C-59: the step that reshaped the table between the two stages (its id), when one did: the
    # rows no longer pair one to one, `sample` shows the first rows of the result as they are
    reshaped: str | None = None
    reshaped_op: str | None = None


def _origins(compiled: Compiled, start: int, end: int) -> dict[str, str | None]:
    """Column after stage `end` → its name at stage `start` (renames followed)."""
    mapping: dict[str, str | None] = {name: name for name in compiled.stages[end].names}
    for index in range(end, start, -1):
        origin = compiled.stages[index].origin
        mapping = {
            after: (origin.get(current) if current is not None else None)
            for after, current in mapping.items()
        }
    return mapping


def compare(
    connection: duckdb.DuckDBPyConnection, compiled: Compiled, start: int, end: int
) -> CleanPreview:
    """Stage `start` against stage `end` (0 = the parent; -1 = after every step)."""
    end = end % len(compiled.stages)
    before: Stage = compiled.stages[start]
    after: Stage = compiled.stages[end]
    origins = _origins(compiled, start, end)
    reshaping = compiled.reshaped(start, end)
    if reshaping is not None:
        return _reshaped(connection, compiled, start, end, origins, reshaping)
    pairs = [(a, b) for a, b in origins.items() if b is not None]
    kept = set(b for _, b in pairs)
    dropped = [name for name in before.names if name not in kept]
    added = [a for a, b in origins.items() if b is None]

    def differs(a: str, b: str) -> str:
        left = as_text(b, before.dtype(b)).replace(ident(b), f"b.{ident(b)}")
        right = as_text(a, after.dtype(a)).replace(ident(a), f"a.{ident(a)}")
        return f"({left}) IS DISTINCT FROM ({right})"

    with_clause = ", ".join(compiled.ctes[: end + 1])  # stage i is the CTE t{i}
    joined = f"{before.name} AS b LEFT JOIN {after.name} AS a ON a.{ident(ROW)} = b.{ident(ROW)}"
    present = f"a.{ident(ROW)} IS NOT NULL"
    counts = [f"count(*) FILTER (WHERE {present} AND {differs(a, b)})" for a, b in pairs]
    query = (
        f"WITH {with_clause} SELECT count(*), count(*) FILTER (WHERE NOT ({present}))"
        + "".join(f", {c}" for c in counts)
        + f" FROM {joined}"
    )
    try:
        row = connection.execute(query).fetchone()
    except duckdb.Error as error:
        message = str(error).splitlines()[0]
        raise CleanError(f"The preview could not be computed: {message}") from None
    assert row is not None
    rows_before, removed = int(row[0]), int(row[1])
    changes = {a: int(n) for (a, _), n in zip(pairs, row[2:], strict=True)}
    columns = [
        ColumnDiff(
            a,
            b,
            changes.get(a, 0),
            before.dtype(b) if b is not None else None,
            after.dtype(a),
        )
        for a, b in origins.items()
    ]
    preview = CleanPreview(
        rows_before=rows_before,
        rows_after=rows_before - removed,
        rows_removed=removed,
        cells_changed=sum(changes.values()),
        columns=columns,
        added=added,
        dropped=dropped,
        problems=dict(compiled.problems),
    )
    touched = [(a, b) for a, b in pairs if changes.get(a)][:SHOWN_COLUMNS]
    if not touched and not removed:
        return preview
    shown = touched or [(a, b) for a, b in pairs][:6]
    preview.shown = [a for a, _ in shown]
    tests = [f"NOT ({present})", *(differs(a, b) for a, b in touched)]
    values = [
        *(as_text(b, before.dtype(b)).replace(ident(b), f"b.{ident(b)}") for _, b in shown),
        *(as_text(a, after.dtype(a)).replace(ident(a), f"a.{ident(a)}") for a, _ in shown),
    ]
    sample = connection.execute(
        f"WITH {with_clause} SELECT b.{ident(ROW)}, NOT ({present}), {', '.join(values)} "
        f"FROM {joined} WHERE {' OR '.join(tests)} ORDER BY b.{ident(ROW)} LIMIT {SAMPLE_ROWS}"
    ).fetchall()
    width = len(shown)
    for record in sample:
        old = [None if v is None else str(v) for v in record[2 : 2 + width]]
        new = [None if v is None else str(v) for v in record[2 + width :]]
        gone = bool(record[1])
        preview.sample.append(
            PreviewRow(
                row=int(record[0]),
                removed=gone,
                before=old,
                after=[None] * width if gone else new,
                changed=[False] * width
                if gone
                else [o != n for o, n in zip(old, new, strict=True)],
            )
        )
    return preview


def _reshaped(
    connection: duckdb.DuckDBPyConnection,
    compiled: Compiled,
    start: int,
    end: int,
    origins: dict[str, str | None],
    reshaping: Stage,
) -> CleanPreview:
    """C-59: the rows before and after, the columns, and the first rows of the result; no cell is
    compared (a row of the result does not come from one row before)."""
    before, after = compiled.stages[start], compiled.stages[end]
    with_clause = ", ".join(compiled.ctes[: end + 1])
    try:
        row = connection.execute(
            f"WITH {with_clause} SELECT (SELECT count(*) FROM {before.name}), "
            f"(SELECT count(*) FROM {after.name})"
        ).fetchone()
    except duckdb.Error as error:
        message = str(error).splitlines()[0]
        raise CleanError(f"The preview could not be computed: {message}") from None
    assert row is not None
    kept = {b for b in origins.values() if b is not None}
    shown = after.names[:SHOWN_COLUMNS]
    preview = CleanPreview(
        rows_before=int(row[0]),
        rows_after=int(row[1]),
        rows_removed=0,
        cells_changed=0,
        columns=[
            ColumnDiff(a, b, 0, before.dtype(b) if b is not None else None, after.dtype(a))
            for a, b in origins.items()
        ],
        added=[a for a, b in origins.items() if b is None],
        dropped=[name for name in before.names if name not in kept],
        shown=shown,
        problems=dict(compiled.problems),
        reshaped=reshaping.step.id,
        reshaped_op=reshaping.step.op,
    )
    if not shown:
        return preview
    values = ", ".join(as_text(name, after.dtype(name)) for name in shown)
    sample = connection.execute(
        f"WITH {with_clause} SELECT {ident(ROW)}, {values} FROM {after.name} "
        f"ORDER BY {ident(ROW)} LIMIT {SAMPLE_ROWS}"
    ).fetchall()
    for record in sample:
        cells = [None if v is None else str(v) for v in record[1:]]
        preview.sample.append(PreviewRow(int(record[0]), False, cells, cells, [False] * len(cells)))
    return preview


def preview(
    connection: duckdb.DuckDBPyConnection, compiled: Compiled, step_id: str | None = None
) -> CleanPreview:
    """Every step (the parent against the result), or one step against the table before it."""
    if step_id is None:
        return compare(connection, compiled, 0, -1)
    index = compiled.stage_of(step_id)
    if index is None:
        problem = compiled.problems.get(step_id)
        raise CleanError(problem or f"Step {step_id} is off or unknown.")
    return compare(connection, compiled, index - 1, index)
