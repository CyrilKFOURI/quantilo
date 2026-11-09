"""A numeric target predicted as ordered classes (018): its distribution, for the browser to place
the limits live, and the limits applied to every row at training time."""

import math
from dataclasses import dataclass
from typing import Literal, cast

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from quantilo.ml import data as ml_data
from quantilo.variable.source import SEED

QUANTILE_LEVELS = 1_000  # 1 001 quantiles: one per 0.1 % of the rows
HISTOGRAM_BINS = 80
SAMPLE_SIZE = 5_000
EXACT_VALUES = 50  # distinct values listed with their counts up to this number
EXACT_QUANTILES_UP_TO = 200_000  # beyond, approx_quantile (CLAUDE.md: quantile cost)


class Classes(BaseModel):
    """How the numeric target is cut: k - 1 inner limits, k names, intervals (a, b]."""

    method: Literal["quantile", "width", "jenks", "custom", "values"] = "quantile"
    edges: list[float] = Field(min_length=1, max_length=9)
    names: list[str] = Field(min_length=2, max_length=10)
    counts: list[int] = Field(default_factory=list)  # rows per class (filled by the server)

    @model_validator(mode="after")
    def _check(self) -> "Classes":
        if len(self.names) != len(self.edges) + 1:
            raise ValueError("One more name than limits is needed.")
        if any(not math.isfinite(e) for e in self.edges):
            raise ValueError("Limits must be numbers.")
        if any(b <= a for a, b in zip(self.edges, self.edges[1:], strict=False)):
            raise ValueError("Limits must increase.")
        names = [n.strip() for n in self.names]
        if any(not n or len(n) > 40 for n in names) or len(set(names)) != len(names):
            raise ValueError("Class names must be distinct, 1 to 40 characters.")
        self.names = names
        return self


def apply(values: pd.Series, classes: Classes) -> pd.Series:
    """The class of every value (None where the value is missing)."""
    cut = pd.cut(
        values.astype(float),
        bins=[-np.inf, *classes.edges, np.inf],
        labels=classes.names,
        right=True,
    )
    return cast(pd.Series, pd.Series(cut, index=values.index).astype(object))


@dataclass
class ValueCount:
    value: float
    count: int


@dataclass
class TargetDistribution:
    """What the browser needs to cut the target: exact quantiles, a fine histogram, a sorted
    sample (natural breaks) and, for few distinct values, each value with its count."""

    column: str
    rows: int
    distinct: int
    min: float | None
    max: float | None
    p1: float | None
    p99: float | None
    quantiles: list[float]
    equal_counts: list[list[float]]  # [k - 2]: exact limits for k = 2 … 10 equal-count classes
    histogram_start: float | None
    histogram_width: float | None
    histogram: list[int]
    sample: list[float]
    values: list[ValueCount] | None


def distribution(
    connection: duckdb.DuckDBPyConnection, source: str, field: ml_data.Field
) -> TargetDistribution:
    expression = ml_data._expression(field)
    values = (
        f"(SELECT x, r FROM (SELECT {expression} AS x, file_row_number AS r FROM {source}) "
        "WHERE x IS NOT NULL)"
    )
    row = connection.execute(
        f"SELECT count(*), count(DISTINCT x), min(x), max(x) FROM {values}"
    ).fetchone()
    rows, distinct, low, high = row if row else (0, 0, None, None)
    if not rows:
        return TargetDistribution(
            field.name, 0, 0, None, None, None, None, [], [], None, None, [], [], None
        )
    low, high = float(cast(float, low)), float(cast(float, high))
    levels = [i / QUANTILE_LEVELS for i in range(QUANTILE_LEVELS + 1)]
    function = "quantile_disc" if rows <= EXACT_QUANTILES_UP_TO else "approx_quantile"
    (grid,) = connection.execute(f"SELECT {function}(x, {levels}) FROM {values}").fetchone() or (
        [],
    )
    quantiles = [float(v) for v in grid]
    fractions = sorted({i / k for k in range(2, 11) for i in range(1, k)})
    (cuts,) = connection.execute(f"SELECT {function}(x, {fractions}) FROM {values}").fetchone() or (
        [],
    )
    at = dict(zip(fractions, (float(v) for v in cuts), strict=True))
    equal_counts = [[at[i / k] for i in range(1, k)] for k in range(2, 11)]
    p1, p99 = quantiles[10], quantiles[990]
    width = (float(high) - float(low)) / HISTOGRAM_BINS if high > low else 1.0
    counts = connection.execute(
        f"SELECT least(floor((x - {float(low)}) / {width}), {HISTOGRAM_BINS - 1})::INT AS b, "
        f"count(*) FROM {values} GROUP BY b ORDER BY b"
    ).fetchall()
    histogram = [0] * HISTOGRAM_BINS
    for bucket, count in counts:
        histogram[int(bucket)] = int(count)
    sample = sorted(
        float(v)
        for (v,) in connection.execute(
            f"SELECT x FROM {values} ORDER BY hash(r + {SEED}) LIMIT {SAMPLE_SIZE}"
        ).fetchall()
    )
    exact = None
    if distinct <= EXACT_VALUES:
        exact = [
            ValueCount(float(v), int(c))
            for v, c in connection.execute(
                f"SELECT x, count(*) FROM {values} GROUP BY x ORDER BY x"
            ).fetchall()
        ]
    return TargetDistribution(
        field.name,
        int(rows),
        int(distinct),
        float(low),
        float(high),
        p1,
        p99,
        quantiles,
        equal_counts,
        float(low),
        width,
        histogram,
        sample,
        exact,
    )
