"""Association between two coded columns, with the measures of the Correlations page:
Pearson or Spearman r (two numbers), corrected Cramér's V (two categories), η (mixed)."""

from typing import Literal

import numpy as np

from quantilo.correlations.matrix import correlation, cramers_v, eta, ranks

Measure = Literal["pearson", "spearman", "cramers_v", "eta"]


def as_groups(codes: np.ndarray) -> np.ndarray:
    """Level indexes as integers, -1 for empty values."""
    return np.where(np.isnan(codes), -1, codes).astype(np.int64)


def association(
    a: np.ndarray,
    a_categorical: bool,
    b: np.ndarray,
    b_categorical: bool,
    method: Literal["pearson", "spearman"] = "pearson",
) -> tuple[float | None, Measure]:
    """Signed r for two numbers, otherwise a strength between 0 and 1 (None if undefined)."""
    if a_categorical and b_categorical:
        value, _p, _n = cramers_v(as_groups(a), as_groups(b))
        return value, "cramers_v"
    if a_categorical or b_categorical:
        values, groups = (b, a) if a_categorical else (a, b)
        value, _p, _n = eta(values.astype(float), as_groups(groups))
        return value, "eta"
    if method == "spearman":
        value, _n = correlation(ranks(a.astype(float)), ranks(b.astype(float)))
        return value, "spearman"
    value, _n = correlation(a.astype(float), b.astype(float))
    return value, "pearson"


def strength_matrix(columns: list[tuple[np.ndarray, bool]]) -> np.ndarray:
    """|association| of every pair (0 when undefined), diagonal 0."""
    size = len(columns)
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            value, _ = association(columns[i][0], columns[i][1], columns[j][0], columns[j][1])
            matrix[i, j] = matrix[j, i] = abs(value) if value is not None else 0.0
    return matrix
