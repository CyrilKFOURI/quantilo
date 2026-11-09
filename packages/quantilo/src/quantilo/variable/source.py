"""One column of a dataset, read through its parser, with the helpers every analysis needs.

The source must expose `file_row_number` (read_parquet(..., file_row_number=true)) so that
rows can be pointed at; statistics run on every row, tests and fits on a seeded sample.
"""

import math
from dataclasses import dataclass

import duckdb
import numpy as np

from quantilo.typing.parsers import typed_sql
from quantilo.typing.types import AnalysisType, Parser

SEED = 42
PREFETCH = 20_000


@dataclass
class VariableInput:
    name: str
    type: AnalysisType
    parser: Parser
    order: list[str] | None = None


def clean(value: object) -> float | None:
    """SQL/numpy number → JSON-safe float (NaN and infinities become None)."""
    if value is None:
        return None
    number = float(value)  # type: ignore[arg-type]
    return number if math.isfinite(number) else None


class VariableSource:
    def __init__(self, connection: duckdb.DuckDBPyConnection, source: str, column: VariableInput):
        self.connection = connection
        self.column = column
        typed = typed_sql(column.name, column.parser)
        self.values = f"(SELECT {typed} AS t, file_row_number AS row FROM {source})"
        # Finite numbers only: infinities are counted apart and kept out of the moments.
        self.numbers = (
            f"(SELECT CAST(t AS DOUBLE) AS x, row FROM {self.values} "
            "WHERE t IS NOT NULL AND isfinite(CAST(t AS DOUBLE)))"
        )
        self._sampled: np.ndarray | None = None
        self._complete = False  # the cached sample holds every value

    def one(self, sql: str) -> tuple:
        return self.connection.execute(sql).fetchone() or ()

    def all(self, sql: str) -> list[tuple]:
        return self.connection.execute(sql).fetchall()

    def sample(self, size: int) -> np.ndarray:
        """Finite values, at most `size` of them, drawn reproducibly.

        Rows are ordered by a hash of their row number, so a smaller sample is the start of a
        larger one: one query serves every size up to PREFETCH.
        """
        if self._sampled is None or (len(self._sampled) < size and not self._complete):
            limit = max(size, PREFETCH)
            rows = self.all(
                f"SELECT x FROM {self.numbers} ORDER BY hash(row + {SEED}) LIMIT {limit}"
            )
            self._sampled = np.array([value for (value,) in rows], dtype=float)
            self._complete = len(self._sampled) < limit
        return self._sampled[:size]
