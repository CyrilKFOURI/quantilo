"""N36: the clusters of an entity number (A-20's `entity_column`), side by side, for the review
cluster by cluster (N35's panel).

Each cluster is the records the step makes one entity (a name and its anchor, as the step reads
them), as the draft leaves the rows just before the step: per record its most frequent spelling,
its rows, its first row in the file and that row's cells; per cluster the columns whose values
differ between those first rows. The step's review fields (`not_entities`, `split`) are not read:
the clusters are the rule's, the page says what the review changes in them. Nothing is written.
"""

from dataclasses import dataclass, field

import duckdb

from quantilo.cleaning import model as m
from quantilo.cleaning.compile import ROW, Compiled, as_text, entity_keys_sql
from quantilo.ingestion.sql import ident, lit

MAX_CLUSTERS = 200  # clusters shown at most in one call
CELL = 200  # characters of a cell shown


@dataclass
class EntityRecordView:
    index: int  # in its cluster, the step's order
    anchor: str
    name: str  # as the step reads it (folded)
    spelling: str | None  # its most frequent spelling at this step (None: no row has it now)
    rows: int
    first: int | None  # its first row in the file (from 0)
    cells: list[str | None] = field(default_factory=list)  # that row, one per column


@dataclass
class EntityCluster:
    index: int  # in the step's clusters
    rows: int
    records: list[EntityRecordView]
    differing: list[int] = field(default_factory=list)  # columns whose first rows differ


@dataclass
class EntityClusters:
    step_id: str
    column: str
    anchor: str | None
    to: str
    columns: list[str]
    total_clusters: int
    total_records: int
    total_rows: int  # rows of the clusters' records at this step
    offset: int
    clusters: list[EntityCluster] = field(default_factory=list)


def _stage(compiled: Compiled, step_id: str) -> int:
    for index, stage in enumerate(compiled.stages[1:], start=1):
        if stage.step.id == step_id:
            if stage.step.op != "entity_column":
                raise m.CleanError("This step does not number entities.")
            return index
    if step_id in compiled.problems:
        raise m.CleanError(f"Step {step_id} cannot run: {compiled.problems[step_id]}")
    raise m.CleanError(f"No step {step_id} in the cleaning (or it is turned off).")


def entity_clusters(
    connection: duckdb.DuckDBPyConnection,
    compiled: Compiled,
    step_id: str,
    offset: int = 0,
    limit: int = 20,
) -> EntityClusters:
    """The clusters of step `step_id` (an `entity_column`), the first `limit` after `offset`, in
    the step's order."""
    index = _stage(compiled, step_id)
    step: m.EntityColumn = compiled.stages[index].step
    prev = compiled.stages[index - 1]
    limit = max(1, min(int(limit), MAX_CLUSTERS))
    offset = max(0, int(offset))
    result = EntityClusters(
        step_id=step_id,
        column=step.column,
        anchor=step.anchor,
        to=step.to,
        columns=prev.names,
        total_clusters=len(step.groups),
        total_records=sum(len(group) for group in step.groups),
        total_rows=0,
        offset=offset,
    )
    if not step.groups:
        return result
    name, anchor = entity_keys_sql(step, prev)
    every = ", ".join(
        f"({cluster}, {at}, {lit(record.anchor)}, {lit(record.name)})"
        for cluster, records in enumerate(step.groups)
        for at, record in enumerate(records)
    )
    table = "(VALUES " + every + ") AS e(e_cluster, e_record, e_anchor, e_name)"
    spelling = as_text(step.column, prev.dtype(step.column))
    with_clause = ", ".join(compiled.ctes[:index])
    keyed = (
        f"SELECT {name} AS k_name, {anchor} AS k_anchor, {spelling} AS k_spelling, "
        f"{ident(ROW)} AS k_row FROM {prev.name}"
    )
    found = connection.execute(
        f"WITH {with_clause}, k AS ({keyed}) SELECT e_cluster, e_record, count(k_row), "
        f"mode(k_spelling), min(k_row) FROM {table} LEFT JOIN k ON k_name = e_name AND "
        "k_anchor IS NOT DISTINCT FROM e_anchor GROUP BY e_cluster, e_record"
    ).fetchall()
    stats = {(int(c), int(r)): (int(n), s, f) for c, r, n, s, f in found}
    result.total_rows = sum(n for n, _, _ in stats.values())
    shown = list(range(len(step.groups)))[offset : offset + limit]
    firsts = sorted(
        {
            int(stats[(c, r)][2])
            for c in shown
            for r in range(len(step.groups[c]))
            if stats.get((c, r), (0, None, None))[2] is not None
        }
    )
    texts = ", ".join(
        f"CASE WHEN {ident(n)} IS NULL THEN NULL ELSE left({as_text(n, d)}, {CELL}) END"
        for n, d in prev.columns
    )
    cells: dict[int, list[str | None]] = {}
    if firsts:
        rows = connection.execute(
            f"WITH {with_clause} SELECT {ident(ROW)}, {texts} FROM {prev.name} WHERE "
            f"{ident(ROW)} IN ({', '.join(map(str, firsts))})"
        ).fetchall()
        cells = {int(r[0]): [None if v is None else str(v) for v in r[1:]] for r in rows}
    for cluster in shown:
        records = []
        for at, record in enumerate(step.groups[cluster]):
            count, text, first = stats.get((cluster, at), (0, None, None))
            records.append(
                EntityRecordView(
                    index=at,
                    anchor=record.anchor,
                    name=record.name,
                    spelling=None if text is None else str(text),
                    rows=count,
                    first=None if first is None else int(first),
                    cells=cells.get(int(first), []) if first is not None else [],
                )
            )
        present = [r.cells for r in records if r.cells]
        differing = [at for at in range(len(prev.names)) if len({row[at] for row in present}) > 1]
        result.clusters.append(
            EntityCluster(
                index=cluster,
                rows=sum(r.rows for r in records),
                records=records,
                differing=differing,
            )
        )
    return result
