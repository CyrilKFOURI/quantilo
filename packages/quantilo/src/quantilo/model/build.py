"""Check a relation, recommend its join, build the analysis dataset (088, R-JOIN, R-BLD).

The analysis dataset is one flat table: the fact table, then along the accepted relations (out
from the fact, snowflake up to `MAX_DEPTH` links), the chosen columns of each dimension, named
`<dimension>_<column>`. Each joined table is read in a sub-query that adds its canonical key
(`keys.key_expression`), so the join compares exactly what the detection compared."""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import duckdb

from quantilo.catalog import ColumnView
from quantilo.errors import ExplorerError
from quantilo.ingestion.sql import ident, lit
from quantilo.model import names
from quantilo.model.keys import (
    KEY_ROLES,
    MAX_KEY_LENGTH,
    Kind,
    Match,
    Reason,
    TableInput,
    TableProfile,
    key_expression,
    link_kinds,
    source,
)
from quantilo.typing.parsers import typed_sql

JoinType = Literal["left", "inner", "right", "full"]
Recommended = Literal["left", "inner", "right", "full", "none"]
Cardinality = Literal["1:1", "n:1", "n:m"]
AggregateFunction = Literal["sum", "mean", "min", "max"]

MAX_DEPTH = 3  # R-BLD-01: snowflake followed up to 3 links from the fact
EXAMPLES = 5  # R-JOIN-06
_SQL_JOIN = {"left": "LEFT JOIN", "inner": "JOIN", "right": "RIGHT JOIN", "full": "FULL OUTER JOIN"}


class ModelError(ExplorerError):
    """A relation or a build the rules refuse (the message says which rule and why)."""

    def __init__(self, message: str, code: str = "model_refused") -> None:
        super().__init__(message, code=code, status_code=422)


@dataclass
class Link:
    """An accepted relation, as the build uses it."""

    child_id: str
    child_columns: list[str]
    parent_id: str
    parent_columns: list[str]
    join_type: JoinType = "left"
    id: str | None = None
    active: bool = True  # C1 (R-REL-13): an inactive relation is kept but not built
    match: Match = "exact"  # C1 (R-REL-14): "normalized" ignores case, accents, punctuation


@dataclass
class Example:
    value: str
    rows: int


@dataclass
class LinkCheck:
    """Counts of one relation on all rows (R-JOIN-06)."""

    child_rows: int
    child_empty: int  # child rows whose key is empty
    matched_rows: int  # child rows whose key is found
    orphan_rows: int  # child rows whose key is not found
    orphans: list[Example]
    child_distinct: int
    parent_rows: int
    parent_distinct: int
    duplicate_keys: int  # key values that repeat in the parent
    duplicates: list[Example]
    unused_keys: int  # parent keys no child row uses
    cardinality: Cardinality
    recommended: Recommended
    reasons: list[Reason] = field(default_factory=list)
    # C1: child rows an exact match misses that a normalized one finds (R-REL-14), and orphan
    # values one letter away from a key (R-REL-15: a slip to fix in Clean, never joined silently)
    normalized_gain: int = 0
    near_misses: list[tuple[str, str]] = field(default_factory=list)

    @property
    def matched_share(self) -> float:
        return self.matched_rows / self.child_rows if self.child_rows else 0.0


def check(
    connection: duckdb.DuckDBPyConnection,
    child: TableInput,
    child_columns: list[str],
    parent: TableInput,
    parent_columns: list[str],
    match: Match = "exact",
) -> LinkCheck:
    """Exact counts on every row; the recommendation and its reason (R-JOIN-01 to 04)."""
    kinds = link_kinds(child, child_columns, parent, parent_columns)
    if isinstance(kinds, Reason):
        raise ModelError(f"{kinds.text} ({kinds.code}).", code="incompatible")
    child_kinds, parent_kinds = kinds
    child_value = key_expression(child, child_columns, child_kinds, match)
    parent_value = key_expression(parent, parent_columns, parent_kinds, match)
    connection.execute(
        f"CREATE OR REPLACE TEMP TABLE check_c AS SELECT {child_value} AS v FROM {source(child)}"
    )
    connection.execute(
        f"CREATE OR REPLACE TEMP TABLE check_p AS SELECT {parent_value} AS v, count(*) AS n "
        f"FROM {source(parent)} GROUP BY 1"
    )
    row = connection.execute(
        "SELECT (SELECT count(*) FROM check_c), "
        "(SELECT count(*) FROM check_c WHERE v IS NULL), "
        "(SELECT count(*) FROM check_c JOIN check_p USING (v)), "
        "(SELECT count(DISTINCT v) FROM check_c), "
        "(SELECT coalesce(sum(n), 0) FROM check_p), "
        "(SELECT count(*) FROM check_p WHERE v IS NOT NULL), "
        "(SELECT count(*) FROM check_p WHERE v IS NOT NULL AND n > 1), "
        "(SELECT count(*) FROM check_p WHERE v IS NOT NULL AND v NOT IN "
        "(SELECT v FROM check_c WHERE v IS NOT NULL))"
    ).fetchone()
    values = [int(value or 0) for value in (row or (0,) * 8)]
    child_rows, empty, matched, child_distinct, parent_rows, parent_distinct, dup, unused = values
    orphans = [
        Example(str(value), int(count))
        for value, count in connection.execute(
            "SELECT v, count(*) FROM check_c WHERE v IS NOT NULL AND v NOT IN "
            "(SELECT v FROM check_p WHERE v IS NOT NULL) GROUP BY v "
            f"ORDER BY 2 DESC, 1 LIMIT {EXAMPLES}"
        ).fetchall()
    ]
    duplicates = [
        Example(str(value), int(count))
        for value, count in connection.execute(
            "SELECT v, n FROM check_p WHERE v IS NOT NULL AND n > 1 "
            f"ORDER BY n DESC, v LIMIT {EXAMPLES}"
        ).fetchall()
    ]
    texts = any(kind == "text" for kind in child_kinds.values())
    gain = 0
    near: list[tuple[str, str]] = []
    orphan_rows = child_rows - empty - matched
    if texts and orphan_rows and match == "exact":
        gain = _normalized_gain(connection, child, child_columns, parent, parent_columns, kinds)
        if len(child_columns) == 1:
            near = _near_misses(connection)
    connection.execute("DROP TABLE IF EXISTS check_c")
    connection.execute("DROP TABLE IF EXISTS check_p")
    child_unique = child_distinct == child_rows - empty
    if dup:
        cardinality: Cardinality = "n:m"
    else:
        cardinality = "1:1" if child_unique else "n:1"
    result = LinkCheck(
        child_rows=child_rows,
        child_empty=empty,
        matched_rows=matched,
        orphan_rows=child_rows - empty - matched,
        orphans=orphans,
        child_distinct=child_distinct,
        parent_rows=parent_rows,
        parent_distinct=parent_distinct,
        duplicate_keys=dup,
        duplicates=duplicates,
        unused_keys=unused,
        cardinality=cardinality,
        recommended="left",
        normalized_gain=gain,
        near_misses=near,
    )
    result.recommended, result.reasons = recommend(result, child.name, parent.name)
    if gain:
        result.reasons.append(
            Reason(
                "R-REL-14",
                f"{_n(gain)} more {child.name} rows find their match when case, accents and "
                "punctuation are ignored: choose that comparison if they mean the same thing",
            )
        )
    if near:
        shown = ", ".join(f"'{a}' → '{b}'" for a, b in near[:3])
        result.reasons.append(
            Reason(
                "R-REL-15",
                f"{len(near)} unmatched values are one letter away from a key ({shown}): fix them "
                "in Clean (they are never joined by guess)",
            )
        )
    return result


MAX_NEAR = 2_000  # R-REL-15: orphan values compared at most
MAX_NEAR_KEYS = 20_000  # ...against at most this many keys


def _normalized_gain(
    connection: duckdb.DuckDBPyConnection,
    child: TableInput,
    child_columns: list[str],
    parent: TableInput,
    parent_columns: list[str],
    kinds: tuple[dict[str, Kind], dict[str, Kind]],
) -> int:
    """R-REL-14: child rows found by a normalized comparison that the exact one misses."""
    child_value = key_expression(child, child_columns, kinds[0], "normalized")
    parent_value = key_expression(parent, parent_columns, kinds[1], "normalized")
    exact_child = key_expression(child, child_columns, kinds[0])
    row = connection.execute(
        f"SELECT count(*) FROM (SELECT {child_value} AS n, {exact_child} AS v FROM "
        f"{source(child)}) AS c WHERE c.v IS NOT NULL AND c.v NOT IN "
        "(SELECT v FROM check_p WHERE v IS NOT NULL) AND c.n IN "
        f"(SELECT {parent_value} FROM {source(parent)})"
    ).fetchone()
    return int(row[0]) if row else 0


def _near_misses(connection: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """R-REL-15: orphan values (5 characters or more) at Damerau-Levenshtein distance 1 of exactly
    one key, bounded (2,000 orphans by 20,000 keys); shown, never applied."""
    row = connection.execute("SELECT count(*) FROM check_p WHERE v IS NOT NULL").fetchone()
    if not row or int(row[0]) > MAX_NEAR_KEYS:
        return []
    found = connection.execute(
        "WITH o AS (SELECT DISTINCT v FROM check_c WHERE v IS NOT NULL AND length(v) >= 5 AND v "
        f"NOT IN (SELECT v FROM check_p WHERE v IS NOT NULL) LIMIT {MAX_NEAR}), "
        "pairs AS (SELECT o.v AS a, p.v AS b FROM o JOIN check_p AS p ON p.v IS NOT NULL AND "
        "abs(length(o.v) - length(p.v)) <= 1 AND lower(o.v) <> lower(p.v) AND "
        "damerau_levenshtein(o.v, p.v) = 1) "
        "SELECT a, min(b) FROM pairs GROUP BY a HAVING count(*) = 1 ORDER BY a"
    ).fetchall()
    return [(str(a), str(b)) for a, b in found]


def _n(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def recommend(check: LinkCheck, child: str, parent: str) -> tuple[Recommended, list[Reason]]:
    """R-JOIN-01 to 04: the join that keeps every fact row and never multiplies one."""
    if check.cardinality == "n:m":
        return "none", [
            Reason(
                "R-JOIN-04",
                f"{_n(check.duplicate_keys)} key values repeat in {parent}: a join would multiply "
                f"{child} rows (and their sums). Aggregate {child} to the key first, or use a "
                "table with one row per key.",
            )
        ]
    if check.cardinality == "1:1":
        return "left", [
            Reason("R-JOIN-03", f"one to one: LEFT JOIN keeps every {child} row, none is repeated")
        ]
    complete = check.orphan_rows == 0 and check.child_empty == 0 and check.unused_keys == 0
    if complete:
        return "inner", [
            Reason(
                "R-JOIN-02",
                f"every {child} row finds its {parent} row and every {parent} row is used: "
                "INNER JOIN gives the same rows as LEFT JOIN",
            )
        ]
    missing = check.orphan_rows + check.child_empty
    text = f"LEFT JOIN keeps all {_n(check.child_rows)} {child} rows"
    if missing:
        text += f"; {_n(missing)} without a {parent} match keep empty {parent} columns"
    return "left", [Reason("R-JOIN-01", text)]


def warnings(check: LinkCheck, join_type: JoinType, child: str, parent: str) -> list[Reason]:
    """What the chosen join does that the user may not expect (R-JOIN-01, 05, 06)."""
    out: list[Reason] = []
    missing = check.orphan_rows + check.child_empty
    if join_type == "inner" and missing:
        out.append(
            Reason("R-JOIN-01", f"{_n(missing)} {child} rows would be dropped (no {parent} match)")
        )
    if join_type in ("right", "full") and check.unused_keys:
        out.append(
            Reason(
                "R-JOIN-05",
                f"{_n(check.unused_keys)} {parent} rows without a {child} row are added; their "
                f"{child} columns are empty",
            )
        )
    if check.duplicate_keys:
        out.append(
            Reason(
                "R-JOIN-06",
                f"{_n(check.duplicate_keys)} key values repeat in {parent}: the build is refused",
            )
        )
    return out


# --- The flat table -------------------------------------------------------------------------


@dataclass
class Step:
    link: Link
    columns: list[str]  # picked parent columns
    prefix: str
    child_alias: str
    parent_alias: str
    depth: int
    out_names: list[str] = field(default_factory=list)  # the columns' names in the flat table


@dataclass
class OutputColumn:
    name: str
    table_id: str
    column: str


@dataclass
class Filter:
    """C1 (R-BLD-07): a condition on a column of one table of the build. On a dimension it keeps
    the fact rows whose dimension row meets it (one → many, as Power BI's filter direction)."""

    table_id: str
    column: str
    op: Literal["eq", "ne", "in", "not_in", "lt", "le", "gt", "ge", "empty", "not_empty"]
    values: list[str] = field(default_factory=list)


@dataclass
class Plan:
    fact: TableInput
    tables: dict[str, TableInput]
    steps: list[Step]
    columns: list[OutputColumn]
    skipped: list[Reason] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)


# R-BLD-02: contact details describe a person, not the thing analysed
CONTACT_ROLES = frozenset({"phone", "email", "url", "address"})


def default_columns(
    table: TableInput, keys: set[str], profile: TableProfile | None = None
) -> list[str]:
    """R-BLD-02: the descriptive columns of a dimension: not its keys, its free or long text
    (over 64 characters), nor contact details."""
    long = {
        column.name
        for column in (profile.columns if profile else [])
        if (column.max_length or 0) > MAX_KEY_LENGTH
    }
    picked: list[str] = []
    for view in sorted(table.columns, key=lambda item: item.position):
        if _descriptive(view) and view.name not in keys and view.name not in long:
            picked.append(view.name)
    return picked


def _descriptive(view: ColumnView) -> bool:
    if view.excluded or view.type in ("ignored", "identifier"):
        return False
    return not (view.role in KEY_ROLES or view.role == "free_text" or view.role in CONTACT_ROLES)


def plan(
    fact_id: str,
    tables: dict[str, TableInput],
    links: list[Link],
    picks: dict[str, list[str]] | None = None,
    profiles: dict[str, TableProfile] | None = None,
    filters: list[Filter] | None = None,
) -> Plan:
    """The joins out from the fact, breadth first (R-BLD-01). `picks`: parent id → columns.
    Inactive relations are left out (R-REL-13); `filters` restrict the rows (R-BLD-07)."""
    if fact_id not in tables:
        raise ModelError("The fact table is not in this project.", code="not_in_project")
    fact = tables[fact_id]
    by_child: dict[str, list[Link]] = {}
    inactive: list[Reason] = []
    for link in links:
        if not link.active:
            parent = tables.get(link.parent_id)
            inactive.append(
                Reason(
                    "R-REL-13",
                    f"the inactive relation {'+'.join(link.child_columns)} → "
                    f"{parent.name if parent else link.parent_id} is not used",
                )
            )
            continue
        by_child.setdefault(link.child_id, []).append(link)
    links = [link for link in links if link.active]
    reached: dict[str, int] = {}
    order: list[tuple[Link, str, int]] = []  # link, child alias, depth
    queue: deque[tuple[str, str, int]] = deque([(fact_id, "t0", 0)])
    seen = {fact_id}
    skipped: list[Reason] = []
    while queue:
        table_id, alias, depth = queue.popleft()
        for link in sorted(
            by_child.get(table_id, []), key=lambda item: (item.parent_id, item.child_columns)
        ):
            parent = tables.get(link.parent_id)
            if parent is None:
                continue
            if depth + 1 > MAX_DEPTH:
                skipped.append(
                    Reason("R-BLD-01", f"{parent.name} is more than {MAX_DEPTH} links away")
                )
                continue
            if link.parent_id in seen and link.parent_id == fact_id:
                continue  # never back to the fact
            reached[link.parent_id] = reached.get(link.parent_id, 0) + 1
            parent_alias = f"t{len(order) + 1}"
            order.append((link, alias, depth + 1))
            if link.parent_id not in seen:
                seen.add(link.parent_id)
                queue.append((link.parent_id, parent_alias, depth + 1))
    link_columns = {(link.child_id, column) for link in links for column in link.child_columns}
    steps: list[Step] = []
    taken = {view.name for view in fact.columns}
    columns = [
        OutputColumn(view.name, fact.id, view.name)
        for view in sorted(fact.columns, key=lambda v: v.position)
    ]
    for index, (link, child_alias, depth) in enumerate(order, start=1):
        parent = tables[link.parent_id]
        own_keys = set(link.parent_columns) | {c for t, c in link_columns if t == parent.id}
        wanted = (picks or {}).get(parent.id)
        valid = {view.name for view in parent.columns}
        chosen = (
            [c for c in wanted if c in valid and c not in link.parent_columns]
            if wanted is not None
            else default_columns(parent, own_keys, (profiles or {}).get(parent.id))
        )
        if reached[parent.id] > 1:
            label = names.prefix("_".join(link.child_columns))
        else:
            label = names.prefix(parent.name)
        out_names: list[str] = []
        for column in chosen:
            name = _free(f"{label}_{column}", taken)
            taken.add(name)
            out_names.append(name)
            columns.append(OutputColumn(name, parent.id, column))
        steps.append(Step(link, chosen, label, child_alias, f"t{index}", depth, out_names))
    reached_ids = {fact.id, *(step.link.parent_id for step in steps)}
    kept_filters: list[Filter] = []
    for item in filters or []:
        table = tables.get(item.table_id)
        if table is None or item.table_id not in reached_ids:
            skipped.append(Reason("R-BLD-07", "a filter on a table the build does not reach"))
            continue
        if item.column not in {view.name for view in table.columns}:
            raise ModelError(
                f"`{item.column}` is not a column of {table.name}.", code="unknown_column"
            )
        kept_filters.append(item)
    return Plan(fact, tables, steps, columns, skipped + inactive, kept_filters)


def _free(name: str, taken: set[str]) -> str:
    if name not in taken:
        return name
    number = 2
    while f"{name} {number}" in taken:
        number += 1
    return f"{name} {number}"


def filter_sql(table: TableInput, item: Filter) -> str:
    """R-BLD-07: the condition on the column as the app reads it (its parser), numbers compared
    as numbers when the values are numbers."""
    view = next(v for v in table.columns if v.name == item.column)
    typed = typed_sql(view.name, view.parser)
    text = f"CAST({typed} AS VARCHAR)"
    if item.op == "empty":
        return f"({typed} IS NULL OR trim({text}) = '')"
    if item.op == "not_empty":
        return f"({typed} IS NOT NULL AND trim({text}) <> '')"
    values = list(item.values) or [""]
    if item.op in ("in", "not_in", "eq", "ne"):
        listed = ", ".join(lit(value) for value in values)
        test = f"{text} IN ({listed})"
        return (
            f"coalesce({test}, false)"
            if item.op in ("in", "eq")
            else f"NOT coalesce({test}, false)"
        )
    symbol = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}[item.op]
    number = f"TRY_CAST({lit(values[0])} AS DOUBLE)"
    return (
        f"coalesce(CASE WHEN {number} IS NOT NULL THEN TRY_CAST({typed} AS DOUBLE) {symbol} "
        f"{number} ELSE {text} {symbol} {lit(values[0])} END, false)"
    )


def flat_sql(plan_: Plan) -> str:
    """The SELECT of the flat table, ordered as the fact table's rows."""
    extras: dict[str, list[str]] = {"t0": []}
    tables_of = {"t0": plan_.fact}
    for step in plan_.steps:
        tables_of[step.parent_alias] = plan_.tables[step.link.parent_id]
        extras.setdefault(step.parent_alias, [])
    joins: list[str] = []
    for index, step in enumerate(plan_.steps, start=1):
        child = tables_of[step.child_alias]
        parent = tables_of[step.parent_alias]
        kinds = link_kinds(child, step.link.child_columns, parent, step.link.parent_columns)
        if isinstance(kinds, Reason):
            raise ModelError(f"{kinds.text} ({kinds.code}).", code="incompatible")
        child_kinds, parent_kinds = kinds
        match = step.link.match
        extras[step.child_alias].append(
            f"{key_expression(child, step.link.child_columns, child_kinds, match)} AS __f{index}"
        )
        extras[step.parent_alias].append(
            f"{key_expression(parent, step.link.parent_columns, parent_kinds, match)} AS __k{index}"
        )
        joins.append(
            f"{_SQL_JOIN[step.link.join_type]} {{{step.parent_alias}}} AS {step.parent_alias} "
            f"ON {step.child_alias}.__f{index} = {step.parent_alias}.__k{index}"
        )

    def subquery(alias: str) -> str:
        table = tables_of[alias]
        added = ", ".join(extras[alias])
        numbered = ", file_row_number = true" if alias == "t0" else ""
        body = f"read_parquet({lit(table.storage_path)}{numbered})"
        tests = [filter_sql(table, item) for item in plan_.filters if item.table_id == table.id]
        where = f" WHERE {' AND '.join(tests)}" if tests else ""
        return f"(SELECT *{', ' + added if added else ''} FROM {body}{where})"

    # R-BLD-07: a filtered dimension keeps the fact rows that reach one of its kept rows
    filtered = {item.table_id for item in plan_.filters}
    reach = [
        f"{step.parent_alias}.__k{index} IS NOT NULL"
        for index, step in enumerate(plan_.steps, start=1)
        if step.link.parent_id in filtered
    ]

    fact_extra = ["file_row_number"] + [
        f"__f{i}" for i, s in enumerate(plan_.steps, 1) if s.child_alias == "t0"
    ]
    selects = [f"t0.* EXCLUDE ({', '.join(fact_extra)})"]
    for step in plan_.steps:
        for column, name in zip(step.columns, step.out_names, strict=True):
            selects.append(f"{step.parent_alias}.{ident(column)} AS {ident(name)}")
    body = f"SELECT {', '.join(selects)} FROM {subquery('t0')} AS t0"
    for join, step in zip(joins, plan_.steps, strict=True):
        body += " " + join.replace("{" + step.parent_alias + "}", subquery(step.parent_alias))
    if reach:
        body += f" WHERE {' AND '.join(reach)}"
    return body + " ORDER BY t0.file_row_number NULLS LAST"


@dataclass
class StepPreview:
    link: Link
    prefix: str
    columns: int
    check: LinkCheck


@dataclass
class Preview:
    rows_before: int
    rows_after: int
    columns: list[OutputColumn]
    steps: list[StepPreview]
    refused: list[Reason]
    skipped: list[Reason]
    filtered_out: int = 0  # C1 (R-BLD-07): fact rows the filters leave out


def preview(connection: duckdb.DuckDBPyConnection, plan_: Plan) -> Preview:
    """Counts before writing anything: per relation, then the rows the join gives."""
    steps: list[StepPreview] = []
    refused: list[Reason] = []
    for step in plan_.steps:
        child = plan_.tables.get(step.link.child_id, plan_.fact)
        parent = plan_.tables[step.link.parent_id]
        found = check(
            connection,
            child,
            step.link.child_columns,
            parent,
            step.link.parent_columns,
            step.link.match,
        )
        steps.append(StepPreview(step.link, step.prefix, len(step.columns), found))
        if found.duplicate_keys:
            refused.append(
                Reason(
                    "R-BLD-03",
                    f"{_n(found.duplicate_keys)} key values repeat in {parent.name} "
                    f"(`{', '.join(step.link.parent_columns)}`): rows would multiply",
                )
            )
    row = connection.execute(f"SELECT count(*) FROM {source(plan_.fact)}").fetchone()
    before = int(row[0]) if row else 0
    after = before
    filtered_out = 0
    if not refused:
        counted = connection.execute(f"SELECT count(*) FROM ({flat_sql(plan_)})").fetchone()
        after = int(counted[0]) if counted else 0
        widening = any(step.link.join_type in ("right", "full") for step in plan_.steps)
        if plan_.filters and not widening:  # R-BLD-07: the rows the filters keep
            unfiltered = Plan(plan_.fact, plan_.tables, plan_.steps, plan_.columns)
            total = connection.execute(f"SELECT count(*) FROM ({flat_sql(unfiltered)})").fetchone()
            filtered_out = (int(total[0]) if total else 0) - after
            before = before - filtered_out
        if after > before and not widening:
            refused.append(
                Reason("R-BLD-03", f"the join gives {_n(after)} rows for {_n(before)} fact rows")
            )
    return Preview(before, after, plan_.columns, steps, refused, plan_.skipped, filtered_out)


def write_flat(connection: duckdb.DuckDBPyConnection, plan_: Plan, target: Path) -> Preview:
    """Write the flat table to Parquet after the same checks as `preview` (R-BLD-03)."""
    result = preview(connection, plan_)
    if result.refused:
        raise ModelError(" ".join(f"{r.text} ({r.code})." for r in result.refused))
    target.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"COPY ({flat_sql(plan_)}) TO {lit(str(target))} (FORMAT parquet, COMPRESSION zstd)"
    )
    return result


# --- Aggregate a child table to its parent's level (Advanced) ---------------------------------


@dataclass
class Measure:
    column: str
    function: AggregateFunction


def default_measures(child: TableInput, keys: set[str]) -> list[Measure]:
    """R-BLD-06: sum and mean of the child's numeric columns that are not keys."""
    out: list[Measure] = []
    for view in sorted(child.columns, key=lambda item: item.position):
        if view.excluded or view.name in keys or view.role in KEY_ROLES:
            continue
        if view.type in ("numeric_continuous", "numeric_discrete"):
            out += [Measure(view.name, "sum"), Measure(view.name, "mean")]
    return out


def aggregate_sql(
    child: TableInput, link: Link, parent: TableInput, measures: list[Measure]
) -> tuple[str, list[OutputColumn]]:
    """The SELECT of `aggregate` (in the parent's row order) and the columns it adds."""
    kinds = link_kinds(child, link.child_columns, parent, link.parent_columns)
    if isinstance(kinds, Reason):
        raise ModelError(f"{kinds.text} ({kinds.code}).", code="incompatible")
    child_kinds, parent_kinds = kinds
    views = {view.name: view for view in child.columns}
    label = names.prefix(child.name)
    taken = {view.name for view in parent.columns}
    count_name = _free(f"{label}_count", taken)
    taken.add(count_name)
    added = [OutputColumn(count_name, child.id, "*")]
    inner = [f"{key_expression(child, link.child_columns, child_kinds)} AS __k", "count(*) AS __n"]
    outer = [f"coalesce(a.__n, 0) AS {ident(count_name)}"]
    sql_function = {"sum": "sum", "mean": "avg", "min": "min", "max": "max"}
    for index, measure in enumerate(measures):
        view = views.get(measure.column)
        if view is None:
            raise ModelError(
                f"`{measure.column}` is not a column of {child.name}.", code="unknown_column"
            )
        value = f"TRY_CAST({typed_sql(view.name, view.parser)} AS DOUBLE)"
        inner.append(f"{sql_function[measure.function]}({value}) AS __m{index}")
        name = _free(f"{label}_{names.prefix(view.name)}_{measure.function}", taken)
        taken.add(name)
        outer.append(f"a.__m{index} AS {ident(name)}")
        added.append(OutputColumn(name, child.id, view.name))
    parent_key = key_expression(parent, link.parent_columns, parent_kinds)
    sql = (
        f"SELECT p.* EXCLUDE (file_row_number, __k), {', '.join(outer)} "
        f"FROM (SELECT *, {parent_key} AS __k FROM read_parquet({lit(parent.storage_path)}, "
        "file_row_number = true)) AS p "
        f"LEFT JOIN (SELECT {', '.join(inner)} FROM {source(child)} GROUP BY 1) AS a "
        "ON p.__k = a.__k ORDER BY p.file_row_number"
    )
    return sql, added


def aggregate(
    connection: duckdb.DuckDBPyConnection,
    child: TableInput,
    link: Link,
    parent: TableInput,
    measures: list[Measure],
    target: Path | None = None,
) -> tuple[int, list[OutputColumn]]:
    """The parent's rows with `<child>_count` and one column per measure (R-BLD-06). Writes
    `target` when given; returns the row count and the added columns."""
    sql, added = aggregate_sql(child, link, parent, measures)
    found = check(connection, child, link.child_columns, parent, link.parent_columns)
    if found.duplicate_keys:
        raise ModelError(
            f"{_n(found.duplicate_keys)} key values repeat in {parent.name}: one row per key is "
            "needed to aggregate (R-BLD-06)."
        )
    counted = connection.execute(f"SELECT count(*) FROM ({sql})").fetchone()
    rows = int(counted[0]) if counted else 0
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        connection.execute(f"COPY ({sql}) TO {lit(str(target))} (FORMAT parquet, COMPRESSION zstd)")
    return rows, added
