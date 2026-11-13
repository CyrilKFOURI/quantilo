"""N35: the groups of a removal of duplicates, side by side, for the review group by group.

Like Splink's and dedupe's review of pairs (without the learning) and Informatica's survivorship:
each group is the rows the step takes for one record (equal on its keys, or listed by A-16), as
the draft leaves them just before the step; per group, the row each rule would keep (the first,
the most complete, the latest by each date column), the rule's own order (its first row is kept,
the others give their values when the user asks to merge) and the columns whose values differ.
The step's review fields (`survivors`, `not_duplicates`, `merge`) are not read: the groups are the
rule's, the page says what the review changes in them. Nothing is written.
"""

from dataclasses import dataclass, field
from typing import Any

import duckdb

from quantilo.cleaning import model as m
from quantilo.cleaning.compile import ROW, Compiled, as_text, duplicate_groups_sql, is_temporal
from quantilo.ingestion.sql import ident

MAX_GROUPS = 200  # groups shown at most in one call
CELL = 200  # characters of a cell shown
DATES = 5  # date columns a group says its latest row by
GROUP_ROWS = 50  # rows of one group shown (its first ones, and the row the rule keeps)


@dataclass
class DuplicateRow:
    row: int  # its number in the file (from 0)
    filled: int  # its cells with a value (`most_complete` counts them)
    cells: list[str | None]  # as the user reads them, one per column of `columns`


@dataclass
class DuplicateGroup:
    rows: list[DuplicateRow]  # in the file's order (the first GROUP_ROWS, and the rule's row)
    size: int  # its rows on the whole file
    order: list[int]  # the rows in the rule's order: the first is the one the rule keeps
    first: int
    most_complete: int
    latest: dict[str, int] = field(default_factory=dict)  # by each date column of `dates`
    differing: list[int] = field(default_factory=list)  # columns whose values differ


@dataclass
class DuplicateGroups:
    step_id: str
    columns: list[str]
    dates: list[str]  # the columns a group can keep its latest row by
    keep: str  # the step's rule: first, last, most_complete, latest ("listed": A-16's groups)
    order_by: str | None
    total_groups: int  # groups of two rows or more, on the whole file
    total_rows: int  # their rows
    removed: int  # rows the rule removes (one kept per group)
    offset: int
    groups: list[DuplicateGroup] = field(default_factory=list)


def _stage(compiled: Compiled, step_id: str) -> int:
    for index, stage in enumerate(compiled.stages[1:], start=1):
        if stage.step.id == step_id:
            if stage.step.op != "drop_duplicates":
                raise m.CleanError("This step does not remove duplicates.")
            return index
    if step_id in compiled.problems:
        raise m.CleanError(f"Step {step_id} cannot run: {compiled.problems[step_id]}")
    raise m.CleanError(f"No step {step_id} in the cleaning (or it is turned off).")


def _latest(rows: list[tuple[int, Any]]) -> int:
    """The row with the latest value, empty ones last, then the first (compile's `latest`)."""
    present = [(row, value) for row, value in rows if value is not None]
    if not present:
        return min(row for row, _ in rows)
    best = max(value for _, value in present)
    return min(row for row, value in present if value == best)


def duplicate_groups(
    connection: duckdb.DuckDBPyConnection,
    compiled: Compiled,
    step_id: str,
    offset: int = 0,
    limit: int = 20,
) -> DuplicateGroups:
    """The groups of step `step_id` (a `drop_duplicates`), the first `limit` after `offset`, by
    their first row. A group of more than GROUP_ROWS rows shows its first ones and the row the
    rule keeps: its most complete and latest rows are read among them."""
    index = _stage(compiled, step_id)
    step: m.DropDuplicates = compiled.stages[index].step
    prev = compiled.stages[index - 1]
    base = step.model_copy(update={"survivors": None, "not_duplicates": None, "merge": False})
    source, partition, order = duplicate_groups_sql(base, prev)
    names = prev.names
    dates = [name for name, dtype in prev.columns if is_temporal(dtype)][:DATES]
    if step.order_by and step.order_by in names and step.order_by not in dates:
        dates.insert(0, step.order_by)
    filled = " + ".join(f"CASE WHEN {ident(n)} IS NULL THEN 0 ELSE 1 END" for n in names) or "0"
    window = f"PARTITION BY {partition}"
    marked = (
        f"SELECT {prev.name}.*, count(*) OVER ({window}) AS __quantilo_size, "
        f"min({ident(ROW)}) OVER ({window}) AS __quantilo_first, "
        f"row_number() OVER ({window} ORDER BY {order}) AS __quantilo_place, "
        f"row_number() OVER ({window} ORDER BY {ident(ROW)}) AS __quantilo_seq, "
        f"{filled} AS __quantilo_filled FROM {source}"
    )
    with_clause = ", ".join(compiled.ctes[:index])
    connection.execute(
        f"CREATE OR REPLACE TEMP TABLE quantilo_duplicates AS WITH {with_clause} "
        f"SELECT * FROM ({marked}) WHERE __quantilo_size > 1"
    )
    try:
        total_groups, total_rows = connection.execute(
            "SELECT count(DISTINCT __quantilo_first), count(*) FROM quantilo_duplicates"
        ).fetchone() or (0, 0)
        limit = max(1, min(int(limit), MAX_GROUPS))
        offset = max(0, int(offset))
        texts = ", ".join(
            f"CASE WHEN {ident(n)} IS NULL THEN NULL ELSE left({as_text(n, d)}, {CELL}) END"
            for n, d in prev.columns
        )
        natives = "".join(f", {ident(name)}" for name in dates)
        found = connection.execute(
            f"SELECT __quantilo_first, {ident(ROW)}, __quantilo_place, __quantilo_filled, "
            f"__quantilo_size{', ' + texts if texts else ''}{natives} FROM quantilo_duplicates "
            "WHERE __quantilo_first IN (SELECT DISTINCT __quantilo_first FROM quantilo_duplicates "
            f"ORDER BY 1 LIMIT {limit} OFFSET {offset}) AND (__quantilo_seq <= {GROUP_ROWS} OR "
            f"__quantilo_place = 1) ORDER BY __quantilo_first, {ident(ROW)}"
        ).fetchall()
    finally:
        connection.execute("DROP TABLE IF EXISTS quantilo_duplicates")
    by_first: dict[int, list[tuple[Any, ...]]] = {}
    for record in found:
        by_first.setdefault(int(record[0]), []).append(record)
    width = len(names)
    groups = []
    for records in by_first.values():
        rows = [
            DuplicateRow(
                int(r[1]), int(r[3]), [None if v is None else str(v) for v in r[5 : 5 + width]]
            )
            for r in records
        ]
        places = sorted(records, key=lambda r: int(r[2]))
        latest = {
            name: _latest([(int(r[1]), r[5 + width + at]) for r in records])
            for at, name in enumerate(dates)
        }
        fullest = max(row.filled for row in rows)
        differing = [at for at in range(width) if len({row.cells[at] for row in rows}) > 1]
        groups.append(
            DuplicateGroup(
                rows=rows,
                size=int(records[0][4]),
                order=[int(r[1]) for r in places],
                first=rows[0].row,
                most_complete=next(row.row for row in rows if row.filled == fullest),
                latest=latest,
                differing=differing,
            )
        )
    return DuplicateGroups(
        step_id=step_id,
        columns=names,
        dates=dates,
        keep="listed" if step.groups is not None else step.keep,
        order_by=step.order_by,
        total_groups=int(total_groups),
        total_rows=int(total_rows),
        removed=int(total_rows) - int(total_groups),
        offset=offset,
        groups=groups,
    )
