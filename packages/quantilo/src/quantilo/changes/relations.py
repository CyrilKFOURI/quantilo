"""The strongest relations (007) before and after: a link that appears or disappears is flagged."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import duckdb

from quantilo.catalog import ColumnView
from quantilo.correlations.matrix import Matrix, MatrixColumn, association_matrix
from quantilo.filters.scope import Where, scoped_source

TOP = 10
WEAK = 0.1  # |value| below: no relation
STRONG = 0.3  # |value| above: a relation worth naming
MOVED = 0.1


@dataclass
class RelationChange:
    a: str
    b: str
    metric: str
    before: float | None  # None: the pair was not computed in that version
    after: float | None
    status: Literal["appeared", "disappeared", "changed", "stable"]


def _matrix(
    connection: duckdb.DuckDBPyConnection, storage_path: str, views: Sequence[ColumnView]
) -> Matrix:
    columns = [MatrixColumn(v.name, v.type, v.parser) for v in views if not v.excluded]
    source = scoped_source(storage_path, Where(), numbered=True)
    matrix, _loaded = association_matrix(connection, source, columns, "pearson", None)
    return matrix


def compare_relations(before: Matrix, after: Matrix, top: int = TOP) -> list[RelationChange]:
    """The `top` strongest pairs of each version, each with its value in the other."""

    def key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    old = {key(p.a, p.b): p for p in before.pairs}
    new = {key(p.a, p.b): p for p in after.pairs}
    chosen = [key(p.a, p.b) for p in after.pairs[:top]]
    chosen += [k for k in (key(p.a, p.b) for p in before.pairs[:top]) if k not in chosen]
    found: list[RelationChange] = []
    for pair in chosen:
        was, now = old.get(pair), new.get(pair)
        b = abs(was.value) if was else 0.0
        a = abs(now.value) if now else 0.0
        if a >= STRONG and b < WEAK:
            status = "appeared"
        elif b >= STRONG and a < WEAK:
            status = "disappeared"
        elif abs(a - b) >= MOVED:
            status = "changed"
        else:
            status = "stable"
        metric = (now or was).metric  # type: ignore[union-attr]
        found.append(
            RelationChange(
                pair[0],
                pair[1],
                metric,
                was.value if was else None,
                now.value if now else None,
                status,  # type: ignore[arg-type]
            )
        )
    order = {"appeared": 0, "disappeared": 1, "changed": 2, "stable": 3}
    found.sort(key=lambda r: (order[r.status], -max(abs(r.before or 0), abs(r.after or 0))))
    return found


def relation_changes(
    connection: duckdb.DuckDBPyConnection,
    before_path: str,
    before_views: Sequence[ColumnView],
    after_path: str,
    after_views: Sequence[ColumnView],
) -> list[RelationChange]:
    return compare_relations(
        _matrix(connection, before_path, before_views),
        _matrix(connection, after_path, after_views),
    )
