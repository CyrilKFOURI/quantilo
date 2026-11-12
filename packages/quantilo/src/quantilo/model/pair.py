"""Join two datasets into a new one (089, R-JN): the keys proposed by rules, the join checked on
every row, the result shown before anything is saved.

The model of a project (088) builds a star from many tables; this is the shorter path of the
Data page, between the open dataset ("left") and one other ("right"), project or not:

- **lookup**: the right table has one row per key; each left row gets the right columns it
  points to. The left rows are kept, never repeated (LEFT JOIN, R-JOIN-01 to 03).
- **expand**: the left table has one row per key and the right one several (an order and its
  lines): the result has one row per right row, with the left columns (R-JN-05).
- **summarise**: same keys, but the left rows are kept: the right rows are counted (and their
  measures summed and averaged) per key (R-BLD-06).

Pure engine: DuckDB in, dataclasses out (Pyodide-safe)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import duckdb

from quantilo.ingestion.sql import lit
from quantilo.model.build import (
    JoinType,
    Link,
    LinkCheck,
    Measure,
    ModelError,
    OutputColumn,
    aggregate_sql,
    check,
    default_measures,
    flat_sql,
    plan,
    warnings,
)
from quantilo.model.keys import Reason, TableInput, link_kinds, profile_table
from quantilo.model.relations import detect

Mode = Literal["lookup", "expand", "summarise"]
SAME_NAME_INCLUDED = 0.5  # R-JN-03: same-name columns proposed down to this match
SAMPLE_ROWS = 20  # rows of the result shown before saving
DOCUMENT_CHILD = "_parent_id"  # R-JN-02: the keys a JSON/XML document's tables carry (083)
DOCUMENT_PARENT = "_id"


@dataclass
class PairKey:
    """A proposed pair of key columns between the two tables."""

    left_columns: list[str]
    right_columns: list[str]
    mode: Mode  # the mode the counts recommend
    cardinality: str  # seen from the left: "n:1", "1:1", "1:n", "n:m"
    confidence: str
    matched_share: float  # share of the rows on the "many" side whose key is found
    reasons: list[Reason] = field(default_factory=list)


@dataclass
class ModeOption:
    mode: Mode
    reason: str | None = None  # why it cannot be chosen; None = available


@dataclass
class PairPreview:
    mode: Mode
    join_type: JoinType
    rows_before: int  # rows of the left table
    rows_after: int
    columns: list[OutputColumn]
    added: list[str]  # the new columns' names
    rows: list[list[str | None]]  # first rows of the result, as text
    check: LinkCheck
    modes: list[ModeOption]
    reasons: list[Reason] = field(default_factory=list)
    warnings: list[Reason] = field(default_factory=list)
    refused: list[Reason] = field(default_factory=list)


def _count(connection: duckdb.DuckDBPyConnection, table: TableInput) -> int:
    row = connection.execute(f"SELECT count(*) FROM read_parquet({lit(table.storage_path)})")
    found = row.fetchone()
    return int(found[0]) if found else 0


def _names(table: TableInput) -> set[str]:
    return {view.name for view in table.columns}


def suggest(
    connection: duckdb.DuckDBPyConnection, left: TableInput, right: TableInput
) -> list[PairKey]:
    """R-JN-01 to 03: the relations the 088 rules find between the two tables (either way), the
    known link of a document's tables, then same-name columns; best first, one per column."""
    profiles = [profile_table(connection, left), profile_table(connection, right)]
    found: list[PairKey] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

    def add(key: PairKey) -> None:
        pair = (tuple(key.left_columns), tuple(key.right_columns))
        if pair in seen:
            return
        seen.add(pair)
        found.append(key)

    for relation in detect(connection, [left, right], profiles):
        to_right = relation.child_id == left.id
        left_cols = relation.child_columns if to_right else relation.parent_columns
        right_cols = relation.parent_columns if to_right else relation.child_columns
        mode: Mode = "lookup" if to_right else "expand"
        cardinality = relation.cardinality if to_right else _flip(relation.cardinality)
        reasons = [Reason("R-JN-01", "found by the rules of the project model (R-REL)")]
        add(
            PairKey(
                left_cols,
                right_cols,
                mode,
                cardinality,
                relation.confidence,
                relation.inclusion,
                reasons + relation.reasons,
            )
        )
    for child, parent, to_right in ((left, right, True), (right, left, False)):
        if DOCUMENT_CHILD in _names(child) and DOCUMENT_PARENT in _names(parent):
            key = _checked(connection, child, [DOCUMENT_CHILD], parent, [DOCUMENT_PARENT])
            if key is None or key.matched_share < SAME_NAME_INCLUDED:
                continue
            reason = Reason(
                "R-JN-02",
                f"{child.name}.{DOCUMENT_CHILD} points to {parent.name}.{DOCUMENT_PARENT}: the "
                "link the document's tables were split along",
            )
            add(_oriented(key, to_right, "high", [reason, *key.reasons]))
    if not found:
        right_names = _names(right)
        for view in sorted(left.columns, key=lambda item: item.position):
            if view.name not in right_names or view.name in (DOCUMENT_CHILD, DOCUMENT_PARENT):
                continue
            if isinstance(link_kinds(left, [view.name], right, [view.name]), Reason):
                continue
            forward = _checked(connection, left, [view.name], right, [view.name])
            if forward is None or forward.matched_share < SAME_NAME_INCLUDED:
                continue
            reason = Reason(
                "R-JN-03",
                f"`{view.name}` is in both tables and {forward.matched_share:.0%} of its "
                f"{left.name} values are found in {right.name}",
            )
            to_right = forward.cardinality != "n:m"
            key = forward
            if not to_right:
                backward = _checked(connection, right, [view.name], left, [view.name])
                if backward is not None and backward.cardinality != "n:m":
                    key = backward
                else:
                    to_right = True
            add(_oriented(key, to_right, "low", [reason, *key.reasons]))
    return found


def _flip(cardinality: str) -> str:
    return {"n:1": "1:n", "1:n": "n:1"}.get(cardinality, cardinality)


@dataclass
class _Checked:
    child_columns: list[str]
    parent_columns: list[str]
    cardinality: str
    matched_share: float
    reasons: list[Reason]


def _checked(
    connection: duckdb.DuckDBPyConnection,
    child: TableInput,
    child_columns: list[str],
    parent: TableInput,
    parent_columns: list[str],
) -> _Checked | None:
    try:
        found = check(connection, child, child_columns, parent, parent_columns)
    except ModelError:
        return None
    return _Checked(
        child_columns, parent_columns, found.cardinality, found.matched_share, found.reasons
    )


def _oriented(key: _Checked, to_right: bool, confidence: str, reasons: list[Reason]) -> PairKey:
    if to_right:
        return PairKey(
            key.child_columns,
            key.parent_columns,
            "lookup",
            key.cardinality,
            confidence,
            key.matched_share,
            reasons,
        )
    return PairKey(
        key.parent_columns,
        key.child_columns,
        "expand",
        _flip(key.cardinality),
        confidence,
        key.matched_share,
        reasons,
    )


def modes(
    forward: LinkCheck, backward: LinkCheck | None, left: str, right: str
) -> list[ModeOption]:
    """R-JN-04: which ways of joining the counts allow. `forward`: left rows looked up in right;
    `backward`: right rows looked up in left."""
    lookup = None
    if forward.duplicate_keys:
        lookup = (
            f"{forward.duplicate_keys:,} key values repeat in {right}: each {left} row would be "
            "repeated (R-JN-04)."
        )
    other = None
    if backward is None or backward.duplicate_keys:
        other = (
            f"key values repeat in {left}: {right} rows can't be attached to one {left} row "
            "(R-JN-04)."
        )
    return [
        ModeOption("lookup", lookup),
        ModeOption("expand", other),
        ModeOption("summarise", other),
    ]


def recommended(options: list[ModeOption], forward: LinkCheck) -> Mode:
    """R-JN-05: look up when the right table has one row per key (the left rows stay as they
    are); otherwise one row per right row (expand)."""
    available = {option.mode for option in options if option.reason is None}
    if "lookup" in available and forward.cardinality in ("n:1", "1:1"):
        return "lookup"
    if "expand" in available:
        return "expand"
    return "lookup"


def _sql(
    left: TableInput,
    right: TableInput,
    left_columns: list[str],
    right_columns: list[str],
    mode: Mode,
    join_type: JoinType,
    picks: list[str] | None,
    measures: list[Measure] | None,
) -> tuple[str, list[OutputColumn], list[str]]:
    tables = {left.id: left, right.id: right}
    if mode == "lookup":
        link = Link(left.id, left_columns, right.id, right_columns, join_type)
        built = plan(left.id, tables, [link], {right.id: picks} if picks is not None else None)
        return flat_sql(built), built.columns, built.steps[0].out_names if built.steps else []
    if mode == "expand":
        link = Link(right.id, right_columns, left.id, left_columns, join_type)
        built = plan(right.id, tables, [link], {left.id: picks} if picks is not None else None)
        return flat_sql(built), built.columns, built.steps[0].out_names if built.steps else []
    link = Link(right.id, right_columns, left.id, left_columns, "left")
    chosen = measures if measures is not None else default_measures(right, set(right_columns))
    sql, added = aggregate_sql(right, link, left, chosen)
    columns = [OutputColumn(view.name, left.id, view.name) for view in left.columns]
    return sql, columns + added, [column.name for column in added]


def preview_pair(
    connection: duckdb.DuckDBPyConnection,
    left: TableInput,
    right: TableInput,
    left_columns: list[str],
    right_columns: list[str],
    mode: Mode | None = None,
    join_type: JoinType | None = None,
    picks: list[str] | None = None,
    measures: list[Measure] | None = None,
) -> PairPreview:
    """Counts on every row and the first rows of the result; nothing is written. `mode` and
    `join_type` left out: the recommended ones (R-JN-05, R-JOIN-01 to 03)."""
    forward = check(connection, left, left_columns, right, right_columns)
    try:
        backward: LinkCheck | None = check(connection, right, right_columns, left, left_columns)
    except ModelError:
        backward = None
    options = modes(forward, backward, left.name, right.name)
    chosen = mode or recommended(options, forward)
    used = forward if chosen == "lookup" else (backward or forward)
    child, parent = (left, right) if chosen == "lookup" else (right, left)
    join: JoinType = join_type or ("inner" if used.recommended == "inner" else "left")
    if chosen == "summarise":
        join = "left"
    reasons = list(used.reasons)
    if chosen == "expand":
        reasons.insert(
            0,
            Reason(
                "R-JN-05",
                f"{left.name} has one row per key and {right.name} several: one row per "
                f"{right.name} row, with its {left.name} columns",
            ),
        )
    elif chosen == "summarise":
        reasons.insert(
            0,
            Reason(
                "R-BLD-06",
                f"the {left.name} rows are kept: {right.name} rows counted (and their measures "
                "summed and averaged) per key",
            ),
        )
    refused = [
        Reason("R-JN-04", option.reason)
        for option in options
        if option.mode == chosen and option.reason
    ]
    before = _count(connection, left)
    result = PairPreview(
        mode=chosen,
        join_type=join,
        rows_before=before,
        rows_after=0,
        columns=[],
        added=[],
        rows=[],
        check=used,
        modes=options,
        reasons=reasons,
        warnings=[] if chosen == "summarise" else warnings(used, join, child.name, parent.name),
        refused=refused,
    )
    if refused:
        return result
    sql, columns, added = _sql(
        left, right, left_columns, right_columns, chosen, join, picks, measures
    )
    counted = connection.execute(f"SELECT count(*) FROM ({sql})").fetchone()
    result.rows_after = int(counted[0]) if counted else 0
    result.columns, result.added = columns, added
    result.rows = [
        list(row)
        for row in connection.execute(
            f"SELECT COLUMNS(*)::VARCHAR FROM ({sql}) LIMIT {SAMPLE_ROWS}"
        ).fetchall()
    ]
    expected = before if chosen in ("lookup", "summarise") else _count(connection, right)
    if join in ("left", "inner") and result.rows_after > expected:
        result.refused.append(
            Reason(
                "R-BLD-03",
                f"the join gives {result.rows_after:,} rows for {expected:,}: a key became "
                "duplicated since it was checked",
            )
        )
    return result


def write_pair(
    connection: duckdb.DuckDBPyConnection,
    left: TableInput,
    right: TableInput,
    left_columns: list[str],
    right_columns: list[str],
    target: Path,
    mode: Mode | None = None,
    join_type: JoinType | None = None,
    picks: list[str] | None = None,
    measures: list[Measure] | None = None,
) -> PairPreview:
    """The same checks as `preview_pair`, then the result written to Parquet."""
    result = preview_pair(
        connection, left, right, left_columns, right_columns, mode, join_type, picks, measures
    )
    if result.refused:
        raise ModelError(" ".join(f"{r.text} ({r.code})." for r in result.refused))
    sql, _, _ = _sql(
        left, right, left_columns, right_columns, result.mode, result.join_type, picks, measures
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(f"COPY ({sql}) TO {lit(str(target))} (FORMAT parquet, COMPRESSION zstd)")
    return result
