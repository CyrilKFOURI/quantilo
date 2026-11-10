"""What the rows say about the suggested KPIs (048): dependencies and correlations, on a sample.

A reproducible sample of at most 50,000 rows (`ORDER BY hash(file_row_number + seed)`, CLAUDE.md)
is read once into a temporary table; every pair is then a small GROUP BY on it. One connection, no
thread: the browser version (Pyodide) runs it as is.
"""

from collections.abc import Sequence
from itertools import combinations
from typing import Protocol

import duckdb

from quantilo.compare.analyze import label_sql, number_sql
from quantilo.kpi.suggest import Facts
from quantilo.typing.types import AnalysisType, Parser

SAMPLE_ROWS = 50_000
SEED = 48
MIN_ROWS = 30  # fewer rows with both values: no dependency claimed
TABLE = "kpi_probe"


class Typed(Protocol):
    name: str
    type: AnalysisType
    parser: Parser


def measure(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    labels: Sequence[Typed],
    numbers: Sequence[Typed],
    keys: Sequence[Typed] = (),
) -> Facts:
    """`source` keeps `file_row_number`. `labels` are read as categories (A → B shares),
    `numbers` as numbers (Pearson r of each pair); 093: each key → number share (R-SUG-29)."""
    if len(labels) < 2 and len(numbers) < 2 and not (keys and numbers):
        return Facts()
    expressions = [f"{label_sql(c)} AS l{i}" for i, c in enumerate(labels)]
    expressions += [f"{label_sql(c)} AS k{i}" for i, c in enumerate(keys)]
    for i, column in enumerate(numbers):
        value = number_sql(column)
        expressions.append(f"(CASE WHEN isfinite({value}) THEN {value} END) AS n{i}")
    (total,) = connection.execute(f"SELECT count(*) FROM {source}").fetchone() or (0,)
    query = f"CREATE OR REPLACE TEMP TABLE {TABLE} AS SELECT {', '.join(expressions)} FROM {source}"
    if total > SAMPLE_ROWS:
        picked = (
            f"SELECT file_row_number FROM {source} "
            f"ORDER BY hash(file_row_number + {SEED}) LIMIT {SAMPLE_ROWS}"
        )
        query += f" WHERE file_row_number IN ({picked})"
    connection.execute(query)
    try:
        facts = Facts(rows=min(int(total), SAMPLE_ROWS))
        for (i, a), (j, b) in combinations(enumerate(labels), 2):
            both, a_fixes, b_fixes = connection.execute(
                f"WITH g AS (SELECT l{i} AS a, l{j} AS b, count(*) AS n FROM {TABLE} "
                f"WHERE l{i} IS NOT NULL AND l{j} IS NOT NULL GROUP BY ALL) "
                "SELECT (SELECT sum(n) FROM g), "
                "(SELECT sum(m) FROM (SELECT max(n) AS m FROM g GROUP BY a)), "
                "(SELECT sum(m) FROM (SELECT max(n) AS m FROM g GROUP BY b))"
            ).fetchone() or (None, None, None)
            if not both or both < MIN_ROWS or a_fixes is None or b_fixes is None:
                continue
            facts.dependency[(a.name, b.name)] = float(a_fixes) / float(both)
            facts.dependency[(b.name, a.name)] = float(b_fixes) / float(both)
        for i, key in enumerate(keys):
            for j, number in enumerate(numbers):
                both, fixes = connection.execute(
                    f"WITH g AS (SELECT k{i} AS a, n{j} AS b, count(*) AS n FROM {TABLE} "
                    f"WHERE k{i} IS NOT NULL AND n{j} IS NOT NULL GROUP BY ALL) "
                    "SELECT (SELECT sum(n) FROM g), "
                    "(SELECT sum(m) FROM (SELECT max(n) AS m FROM g GROUP BY a))"
                ).fetchone() or (None, None)
                if both and both >= MIN_ROWS and fixes is not None:
                    facts.dependency[(key.name, number.name)] = float(fixes) / float(both)
        if numbers:
            parts = [
                f"min(n{i}), max(n{i}), count(*) FILTER (WHERE n{i} <> trunc(n{i}))"
                for i in range(len(numbers))
            ]
            row = connection.execute(f"SELECT {', '.join(parts)} FROM {TABLE}").fetchone() or ()
            for i, column in enumerate(numbers):
                low, high, fractional = row[3 * i : 3 * i + 3] if row else (None, None, 1)
                facts.ranges[column.name] = (
                    None if low is None else float(low),
                    None if high is None else float(high),
                    int(fractional or 0) == 0,
                )
        pairs = list(combinations(range(len(numbers)), 2))
        if pairs:
            row = connection.execute(
                f"SELECT {', '.join(f'corr(n{i}, n{j})' for i, j in pairs)} FROM {TABLE}"
            ).fetchone()
            for (i, j), r in zip(pairs, row or (), strict=False):
                if r is not None and r == r:  # NaN: a constant column
                    facts.correlation[frozenset((numbers[i].name, numbers[j].name))] = float(r)
        return facts
    finally:
        connection.execute(f"DROP TABLE IF EXISTS {TABLE}")
