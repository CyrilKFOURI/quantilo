"""Relations between the tables of a project (088, R-REL-01 to 10).

For every column of a table that can link (`ColumnProfile.linkable`) and every candidate key of
another table: names (R-REL-02), types (R-REL-03), then the share of the column's distinct values
found in the key (inclusion, R-REL-04), measured on at most `SAMPLE` distinct values chosen by
hash (reproducible, R-REL-10). Pairs that can't reach the inclusion threshold are dropped from
the profiles alone (distinct counts, integer ranges) before any query."""

import hashlib
from dataclasses import dataclass, field
from typing import Literal

import duckdb

from quantilo.model import names
from quantilo.model.keys import (
    APPROX_ERROR,
    ColumnProfile,
    Kind,
    Reason,
    TableInput,
    TableProfile,
    key_expression,
    source,
)

Cardinality = Literal["1:1", "n:1", "n:m"]
Confidence = Literal["high", "medium", "low"]

SAMPLE = 50_000  # distinct child values compared at most (R-REL-10)
INCLUDED = 0.95  # R-REL-04
NAMED_INCLUDED = 0.5  # R-REL-04: with identical names, proposed down to here (low confidence)
SPECIFIC_TEXT = 20  # R-REL-05: distinct text codes that make values alone convincing
DENSE_MAX = 1000  # R-REL-05: integer keys 1..N up to here, at least half filled, are "dense"


@dataclass
class Relation:
    """A child column (or two) whose values are found in a key of the parent."""

    child_id: str
    child_columns: list[str]
    parent_id: str
    parent_columns: list[str]
    cardinality: Cardinality
    name_score: float
    inclusion: float  # share of the child's distinct values found in the key
    coverage: float  # share of the key's values the child uses
    confidence: Confidence
    score: float
    reasons: list[Reason] = field(default_factory=list)
    sampled: bool = False

    @property
    def key(self) -> str:
        """Stable id of a proposal (the same pair of columns gives the same key)."""
        text = "|".join(
            [
                self.child_id,
                ",".join(self.child_columns),
                self.parent_id,
                ",".join(self.parent_columns),
            ]
        )
        return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def compatible(child: Kind, parent: Kind) -> bool:
    """R-REL-03 on kinds; a text holding integers meets an integer through its canonical form."""
    return child == parent or {child, parent} == {"integer", "text"}


def dense(profile: ColumnProfile) -> bool:
    """R-REL-05: 1..N with N small and the range at least half filled (row numbers, codes 1-9)."""
    if profile.kind != "integer" or profile.minimum is None or profile.maximum is None:
        return False
    span = profile.maximum - profile.minimum + 1
    return profile.maximum <= DENSE_MAX and span > 0 and profile.distinct >= span * 0.5


def specific(child: ColumnProfile, parent: ColumnProfile) -> bool:
    """R-REL-05: values that rarely meet by chance (text codes, sparse integers)."""
    if parent.kind == "date" or child.kind == "date":
        return False
    if parent.kind == "text":
        return child.distinct >= SPECIFIC_TEXT
    return not dense(parent) and child.distinct >= SPECIFIC_TEXT


SAME_COLUMNS = 0.8  # R-REL-12: share of column names two pieces of one table have in common


def pieces(profiles: list[TableProfile]) -> list[tuple[str, str]]:
    """R-REL-12: pairs of tables with the same columns (retail_2010 and retail_2011): pieces of
    one table to stack, not two tables to join."""
    out: list[tuple[str, str]] = []
    for index, left in enumerate(profiles):
        for right in profiles[index + 1 :]:
            a = {column.name.lower() for column in left.columns}
            b = {column.name.lower() for column in right.columns}
            if min(len(a), len(b)) >= 3 and len(a & b) / len(a | b) >= SAME_COLUMNS:
                out.append((left.id, right.id))
    return out


def _can_link(child: ColumnProfile) -> bool:
    """R-REL-06: never on flags of two values (booleans and the rest are not `linkable`)."""
    return child.linkable and child.distinct > 2


def detect(
    connection: duckdb.DuckDBPyConnection,
    tables: list[TableInput],
    profiles: list[TableProfile],
    families: dict[str, str] | None = None,
) -> list[Relation]:
    """Every proposal between different tables (and families, R-REL-01), best first."""
    family = families or {}
    by_id = {table.id: table for table in tables}
    found: list[Relation] = []
    key_sets: dict[tuple[object, ...], str] = {}
    apart = {frozenset(pair) for pair in pieces(profiles)}
    for parent in profiles:
        for key in parent.keys:
            for child in profiles:
                if child.id == parent.id or family.get(child.id, child.id) == family.get(
                    parent.id, parent.id
                ):
                    continue
                if frozenset((child.id, parent.id)) in apart:
                    continue
                found += _try(connection, by_id, child, parent, key.columns, key.primary, key_sets)
    found += _many_to_many(connection, by_id, profiles, found, family, key_sets, apart)
    return _best_per_column(found, {p.id: p for p in profiles})


def _try(
    connection: duckdb.DuckDBPyConnection,
    tables: dict[str, TableInput],
    child: TableProfile,
    parent: TableProfile,
    key_columns: list[str],
    primary: bool,
    key_sets: dict[tuple[object, ...], str],
) -> list[Relation]:
    """Every child column (or pair of columns) that may refer to this key of the parent."""
    parts = [parent.column(name) for name in key_columns]
    if len(parts) == 2:
        # a two-column key is matched by names only (R-KEY-03): the best column for each part
        matched: list[ColumnProfile] = []
        scores: list[float] = []
        for part in parts:
            options = [
                (names.similarity(column.name, child.name, part.name, parent.name), column)
                for column in child.columns
                if _can_link(column) and column not in matched and _compatible(column, part)
            ]
            options = [(score, column) for score, column in options if score >= 0.8]
            if not options:
                return []
            score, column = max(options, key=lambda option: option[0])
            scores.append(score)
            matched.append(column)
        found = _measure_pair(
            connection, tables, child, parent, matched, parts, min(scores), key_sets
        )
        return [found] if found else []
    part = parts[0]
    out: list[Relation] = []
    for column in child.columns:
        if not _can_link(column) or not _compatible(column, part):
            continue
        score = names.similarity(column.name, child.name, part.name, parent.name)
        if score == 0 and not (primary and specific(column, part)):
            continue  # R-REL-05: without a shared name, only to the primary key, specific values
        found = _measure_pair(connection, tables, child, parent, [column], [part], score, key_sets)
        if found:
            out.append(found)
    return out


def _compatible(column: ColumnProfile, part: ColumnProfile) -> bool:
    return bool(part.kind and column.kind and compatible(column.kind, part.kind))


def _measure_pair(
    connection: duckdb.DuckDBPyConnection,
    tables: dict[str, TableInput],
    child: TableProfile,
    parent: TableProfile,
    matched: list[ColumnProfile],
    parts: list[ColumnProfile],
    name_score: float,
    key_sets: dict[tuple[object, ...], str],
) -> Relation | None:
    lone, part = matched[0], parts[0]
    if (lone.kind == "date" or part.kind == "date") and name_score == 0:
        return None
    floor = NAMED_INCLUDED if name_score == 1 else INCLUDED
    # cheap bounds first: a query that can't pass is not run
    too_many = lone.distinct * (1 - APPROX_ERROR) > part.distinct / floor + 1
    if len(parts) == 1 and (not _range_fits(lone, part) or too_many):
        return None
    inclusion, coverage, sampled = _inclusion(
        connection,
        tables[child.id],
        [c.name for c in matched],
        matched,
        tables[parent.id],
        [p.name for p in parts],
        parts,
        key_sets,
    )
    if inclusion < floor:
        return None
    cardinality: Cardinality = "1:1" if len(matched) == 1 and lone.unique else "n:1"
    return _relation(
        child,
        parent,
        matched,
        [p.name for p in parts],
        cardinality,
        name_score,
        inclusion,
        coverage,
        sampled,
    )


def _range_fits(child: ColumnProfile, key: ColumnProfile) -> bool:
    """Integers outside the key's range can't be in it: skip a query that can't pass."""
    if child.kind != "integer" or key.kind != "integer":
        return True
    if child.minimum is None or child.maximum is None or key.minimum is None or key.maximum is None:
        return True
    return max(child.minimum, key.minimum) <= min(child.maximum, key.maximum)


def _inclusion(
    connection: duckdb.DuckDBPyConnection,
    child_table: TableInput,
    child_columns: list[str],
    child_profiles: list[ColumnProfile],
    parent_table: TableInput,
    key_columns: list[str],
    key_profiles: list[ColumnProfile],
    key_sets: dict[tuple[object, ...], str],
) -> tuple[float, float, bool]:
    # a code read as text on one side and as a number on the other meets as a number (R-REL-03)
    child_kinds: dict[str, Kind] = {}
    key_kinds: dict[str, Kind] = {}
    for child, key in zip(child_profiles, key_profiles, strict=True):
        if child.kind is None or key.kind is None:
            return 0.0, 0.0, False
        shared: Kind = "integer" if child.kind != key.kind else key.kind
        child_kinds[child.name], key_kinds[key.name] = shared, shared
    store = (parent_table.id, tuple(key_columns), tuple(sorted(key_kinds.items())))
    if store not in key_sets:
        name = f"keys_{len(key_sets)}"
        expression = key_expression(parent_table, key_columns, key_kinds)
        connection.execute(
            f"CREATE OR REPLACE TEMP TABLE {name} AS SELECT DISTINCT {expression} AS v "
            f"FROM {source(parent_table)} WHERE ({expression}) IS NOT NULL"
        )
        key_sets[store] = name
    keys = key_sets[store]
    expression = key_expression(child_table, child_columns, child_kinds)
    row = connection.execute(
        f"WITH values_ AS (SELECT DISTINCT {expression} AS v FROM {source(child_table)} "
        f"WHERE ({expression}) IS NOT NULL), "
        f"picked AS (SELECT v FROM values_ ORDER BY hash(v) LIMIT {SAMPLE}) "
        "SELECT count(*), count(k.v), (SELECT count(*) FROM values_), "
        f"(SELECT count(*) FROM {keys}) "
        f"FROM picked LEFT JOIN {keys} AS k USING (v)"
    ).fetchone()
    picked, found, total, key_count = (int(value or 0) for value in (row or (0, 0, 0, 0)))
    if picked == 0:
        return 0.0, 0.0, False
    inclusion = found / picked
    sampled = total > picked
    used = inclusion * total  # estimate of the keys the child uses
    coverage = min(used / key_count, 1.0) if key_count else 0.0
    return inclusion, coverage, sampled


def _confidence(name_score: float, inclusion: float) -> Confidence:
    """R-REL-08."""
    if name_score >= 0.8 and inclusion >= 0.99:
        return "high"
    if (name_score >= 0.6 and inclusion >= INCLUDED) or (name_score == 0 and inclusion >= 0.99):
        return "medium"
    return "low"


def _percent(value: float) -> str:
    if value >= 1:
        return "100%"
    if value >= 0.995:
        return "99.5%+"
    return f"{value * 100:.1f}%".replace(".0%", "%")


def _relation(
    child: TableProfile,
    parent: TableProfile,
    matched: list[ColumnProfile],
    key_columns: list[str],
    cardinality: Cardinality,
    name_score: float,
    inclusion: float,
    coverage: float,
    sampled: bool,
) -> Relation:
    reasons: list[Reason] = []
    child_label = " + ".join(f"`{c.name}`" for c in matched)
    parent_label = " + ".join(f"`{name}`" for name in key_columns)
    if name_score == 1:
        reasons.append(Reason("R-REL-02", f"{child_label} and {parent_label} name the same thing"))
    elif name_score >= 0.6:
        reasons.append(Reason("R-REL-02", f"{child_label} and {parent_label} have close names"))
    else:
        reasons.append(
            Reason("R-REL-05", "no shared name, but the values are specific codes found in the key")
        )
    found = f"{_percent(inclusion)} of the values of {child_label} are found in {parent.name}"
    if sampled:
        found += f" (on {SAMPLE:,} values picked at random)".replace(",", " ")
    reasons.append(Reason("R-REL-04", found))
    if inclusion < INCLUDED:
        reasons.append(
            Reason(
                "R-REL-04",
                "below 95%: proposed only because the names are identical; check the orphans",
            )
        )
    if cardinality == "1:1":
        reasons.append(Reason("R-REL-07", f"{child_label} is unique too: one to one"))
    else:
        reasons.append(
            Reason(
                "R-REL-07",
                f"{parent_label} is unique in {parent.name}: "
                f"one {parent.name} row for many {child.name} rows",
            )
        )
    confidence = _confidence(name_score, inclusion)
    score = 0.5 * name_score + 0.4 * inclusion + 0.1 * coverage
    return Relation(
        child_id=child.id,
        child_columns=[c.name for c in matched],
        parent_id=parent.id,
        parent_columns=list(key_columns),
        cardinality=cardinality,
        name_score=name_score,
        inclusion=inclusion,
        coverage=coverage,
        confidence=confidence,
        score=round(score, 4),
        reasons=reasons,
        sampled=sampled,
    )


def _many_to_many(
    connection: duckdb.DuckDBPyConnection,
    tables: dict[str, TableInput],
    profiles: list[TableProfile],
    found: list[Relation],
    family: dict[str, str],
    key_sets: dict[tuple[object, ...], str],
    apart: set[frozenset[str]],
) -> list[Relation]:
    """R-REL-07: two tables sharing a key-named column that neither has unique, and no table
    where it is a key (a dimension would carry both as n:1)."""
    covered = {
        names.base(column, tables[r.child_id].name) for r in found for column in r.child_columns
    }
    out: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()
    for left in profiles:
        for right in profiles:
            if left.id >= right.id or family.get(left.id, left.id) == family.get(
                right.id, right.id
            ):
                continue
            if frozenset((left.id, right.id)) in apart:
                continue
            for a in left.columns:
                if not (_can_link(a) and not a.unique and names.has_key_word(a.name)):
                    continue
                for b in right.columns:
                    if not (_can_link(b) and not b.unique and a.kind and b.kind):
                        continue
                    if not compatible(a.kind, b.kind):
                        continue
                    if names.similarity(a.name, left.name, b.name, right.name) < 1:
                        continue
                    base = names.base(a.name, left.name)
                    if base in covered or (left.id, right.id, base) in seen:
                        continue
                    # the smaller set of values is the child (its values found in the other)
                    child, parent, c_col, p_col = (
                        (left, right, a, b) if a.distinct <= b.distinct else (right, left, b, a)
                    )
                    inclusion, coverage, sampled = _inclusion(
                        connection,
                        tables[child.id],
                        [c_col.name],
                        [c_col],
                        tables[parent.id],
                        [p_col.name],
                        [p_col],
                        key_sets,
                    )
                    if inclusion < INCLUDED:
                        continue
                    seen.add((left.id, right.id, base))
                    relation = _relation(
                        child,
                        parent,
                        [c_col],
                        [p_col.name],
                        "n:m",
                        1.0,
                        inclusion,
                        coverage,
                        sampled,
                    )
                    relation.reasons[-1] = Reason(
                        "R-REL-07",
                        f"`{c_col.name}` repeats in both tables: many to many, rows would multiply",
                    )
                    relation.confidence = "low"
                    out.append(relation)
    return out


def _best_per_column(found: list[Relation], profiles: dict[str, TableProfile]) -> list[Relation]:
    """R-REL-09: one parent per child column; a 1:1 pair is kept once, from the side where the
    column is not the primary key (then the bigger table) to the other."""
    best: dict[tuple[str, tuple[str, ...]], Relation] = {}
    for relation in sorted(found, key=lambda r: -r.score):
        slot = (relation.child_id, tuple(relation.child_columns))
        if slot not in best:
            best[slot] = relation
    kept = list(best.values())
    out: list[Relation] = []
    for relation in kept:
        if relation.cardinality == "1:1":
            twin = next(
                (
                    other
                    for other in kept
                    if other.child_id == relation.parent_id
                    and other.parent_id == relation.child_id
                    and other.child_columns == relation.parent_columns
                ),
                None,
            )
            if twin is not None and _parent_side(relation, profiles) != relation.parent_id:
                continue
        out.append(relation)
    return sorted(out, key=lambda r: (-r.score, r.child_id, r.child_columns))


def _parent_side(relation: Relation, profiles: dict[str, TableProfile]) -> str:
    """In 1:1, the parent is where the column is the primary key, then the smaller table."""
    child, parent = profiles[relation.child_id], profiles[relation.parent_id]
    child_pk = child.primary_key and child.primary_key.columns == relation.child_columns
    parent_pk = parent.primary_key and parent.primary_key.columns == relation.parent_columns
    if parent_pk and not child_pk:
        return parent.id
    if child_pk and not parent_pk:
        return child.id
    if parent.rows != child.rows:
        return parent.id if parent.rows < child.rows else child.id
    return min(parent.id, child.id)  # a tie: one side, always the same
