"""What is left once trend and seasonality are removed, and how it is distributed.

Reuses the variable page (describe, normality tests, density, QQ-plot, law fitting) on an
in-memory table, so the residuals get exactly the same analysis as any column.
"""

from dataclasses import dataclass
from typing import Literal

import duckdb
import numpy as np
import pyarrow as pa

from quantilo.charts.prepare import Bin
from quantilo.timeseries.decomposition import stl
from quantilo.typing.types import Parser
from quantilo.variable.fitting import Fit, fit
from quantilo.variable.numeric import (
    QQ,
    Curve,
    Describe,
    Normality,
    describe,
    kde,
    normality,
    qq,
)
from quantilo.variable.source import VariableInput, VariableSource

Source = Literal["residuals", "series", "returns"]
SOURCES: tuple[Source, ...] = ("residuals", "series", "returns")
BINS = 30


@dataclass
class Remainder:
    """Residuals of the series and how they were obtained."""

    values: np.ndarray
    fitted: np.ndarray  # series minus residuals (trend + seasonality)
    method: str  # "STL remainder" or "straight-line residuals"


@dataclass
class Distribution:
    source: Source
    method: str | None  # how residuals were obtained (residuals only)
    describe: Describe
    normality: Normality
    density: Curve | None
    qq: QQ | None
    bins: list[Bin]
    fit: Fit


class DistributionError(ValueError):
    pass


def remainder(series: np.ndarray, season: int | None) -> Remainder:
    """STL remainder when two cycles fit (trend and seasonality removed), else residuals of a
    straight line."""
    if season and season >= 2 and len(series) >= 2 * season:
        found = stl(series, season)
        label = "STL remainder" if found.robust else "STL remainder (non-robust: few cycles)"
        return Remainder(found.resid, series - found.resid, label)
    positions = np.arange(len(series), dtype=float)
    slope, intercept = np.polyfit(positions, series, 1)
    line = slope * positions + intercept
    return Remainder(series - line, line, "straight-line residuals")


def deseasonalized(series: np.ndarray, season: int | None) -> tuple[np.ndarray, bool]:
    """The series without its seasonal component (STL), when two cycles fit."""
    if not season or season < 2 or len(series) < 2 * season:
        return series, False
    return series - stl(series, season).seasonal, True


def values_for(
    series: np.ndarray, source: Source, season: int | None
) -> tuple[np.ndarray, str | None]:
    if source == "series":
        return series, None
    if source == "returns":
        previous = series[:-1]
        if (previous == 0).any():
            raise DistributionError("Returns need values that are never 0.")
        return series[1:] / previous - 1, None
    found = remainder(series, season)
    return found.values, found.method


def histogram(values: np.ndarray) -> list[Bin]:
    counts, edges = np.histogram(values, bins=min(BINS, max(5, len(values) // 4)))
    return [
        Bin(start=float(edges[i]), end=float(edges[i + 1]), count=int(c))
        for i, c in enumerate(counts)
    ]


def distribution(series: np.ndarray, source: Source, season: int | None) -> Distribution:
    values, method = values_for(series, source, season)
    values = values[np.isfinite(values)]
    if len(values) < 8:
        raise DistributionError("Needs at least 8 values.")
    connection = duckdb.connect()
    try:
        connection.register(
            "values_table",
            pa.table({"file_row_number": pa.array(range(len(values))), "value": values}),
        )
        column = VariableInput("value", "numeric_continuous", Parser(kind="raw"))
        variable = VariableSource(connection, "values_table", column)
        summary = describe(variable)
        return Distribution(
            source,
            method,
            summary,
            normality(variable, summary),
            kde(variable, summary),
            qq(variable, summary),
            histogram(values),
            fit(variable, discrete=False),
        )
    finally:
        connection.close()
