# ruff: noqa: RUF002, RUF003  (Δ, −, × are the usual notation)
"""Accumulated local effects (125, Apley & Zhu 2020): the effect of one column read only on real
neighbouring rows. Partial dependence sets a column to every value of a grid for every row, so
when the column is strongly linked to another (R-XAI-01, |r|, Cramér's V or η > 0.8) it scores
combinations that never occur: it extrapolates. ALE cuts the column into intervals of equal
counts (quantiles), moves each row only to the two edges of its own interval, averages the
difference and adds the intervals up. A number: intervals; ordered categories: steps from one
category to the next, measured on the rows of both. Centred: the average effect is 0.

Also the check of a monotonic constraint (019): along a grid of the column, every row's
prediction must only rise (+1) or only fall (−1)."""

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from quantilo.ml.target import Task

SEED = 42
ROWS = 2_000  # rows the effects are averaged on
BINS = 20  # intervals of a number (quantiles)
MAX_BINS = 50
TOP_CATEGORIES = 30
CHECK_ROWS = 200  # rows whose curve a monotonic constraint is checked on
CHECK_GRID = 25
TOLERANCE = 1e-9  # relative: floating-point noise is not a violation


class AleError(ValueError):
    """The column cannot get accumulated local effects (with the reason)."""


@dataclass
class Monotonic:
    column: str
    sign: int  # +1: may only rise; −1: may only fall
    rows: int  # rows checked
    violations: int  # rows whose curve goes the other way somewhere
    largest: float  # the largest step the wrong way, on the output's scale
    sentence: str


@dataclass
class Ale:
    column: str
    kind: str  # "number" or "category"
    grid: list[str | float]  # edges of the intervals (number) or the categories in order
    curves: dict[str, list[float]]  # one centred curve per output, at each grid point
    counts: list[int]  # rows per interval (number: one fewer than the grid) or per category
    output: str
    rows: int
    order: str | None  # how ordered categories were put in order
    sentence: str
    rule: str | None = None  # R-XAI-01 when a strong link made ALE the reading to trust
    monotonic: Monotonic | None = None


Predict = Callable[[pd.DataFrame], np.ndarray]


def predictor(pipeline: Pipeline, task: Task) -> tuple[Predict, list[str], str]:
    """The model's output as a matrix (rows × outputs), its names and what it measures."""
    classes = [str(c) for c in getattr(pipeline, "classes_", [])]
    if task == "regression":
        return (
            lambda frame: np.asarray(pipeline.predict(frame), dtype=float)[:, None],
            ["prediction"],
            "Prediction",
        )
    if len(classes) == 2:
        return (
            lambda frame: np.asarray(pipeline.predict_proba(frame), dtype=float)[:, [1]],
            [classes[1]],
            f"Probability of “{classes[1]}”",
        )
    return (
        lambda frame: np.asarray(pipeline.predict_proba(frame), dtype=float),
        classes,
        "Probability of each class",
    )


def _sample(n: int, limit: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(SEED).choice(n, limit, replace=False))


def _numeric(raw: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_bool_dtype(raw)


def _with(frame: pd.DataFrame, column: str, values: Any) -> pd.DataFrame:
    changed = frame.copy()
    changed[column] = values
    return changed


def number_effects(
    predict: Predict, x: pd.DataFrame, column: str, bins: int = BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edges, centred effects at the edges (edges × outputs) and rows per interval."""
    values = cast(pd.Series, x[column]).to_numpy(dtype=float)
    keep = np.isfinite(values)
    frame, values = x.loc[keep], values[keep]
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        raise AleError(f"{column} takes a single value on these rows: no effect to measure.")
    k = np.clip(np.searchsorted(edges, values, side="left"), 1, len(edges) - 1)
    high = predict(_with(frame, column, edges[k]))
    low = predict(_with(frame, column, edges[k - 1]))
    differences = high - low
    intervals = len(edges) - 1
    counts = np.bincount(k - 1, minlength=intervals)
    local = np.zeros((intervals, differences.shape[1]))
    for j in range(differences.shape[1]):
        local[:, j] = np.bincount(
            k - 1, weights=differences[:, j], minlength=intervals
        ) / np.maximum(counts, 1)
    accumulated = np.vstack([np.zeros((1, local.shape[1])), np.cumsum(local, axis=0)])
    # centred so that the average effect over the rows is 0 (each interval at its middle)
    middle = (accumulated[1:] + accumulated[:-1]) / 2
    accumulated -= (counts[:, None] * middle).sum(axis=0) / max(counts.sum(), 1)
    return edges, accumulated, counts


def category_effects(
    predict: Predict, x: pd.DataFrame, column: str, order: list[str]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Categories in order, centred effects (categories × outputs) and rows per category: each
    step from one category to the next is measured on the rows of both."""
    labels = cast(pd.Series, x[column]).astype(str).where(cast(pd.Series, x[column]).notna(), None)
    present = [c for c in order if (labels == c).any()]
    if len(present) < 2:
        raise AleError(f"{column} has fewer than two of its categories on these rows.")
    original = cast(pd.Series, x[column])
    values = {str(v): v for v in original.dropna().unique()}
    counts = np.array([int((labels == c).sum()) for c in present])
    steps = []
    for previous, current in itertools.pairwise(present):
        frame = x.loc[(labels == previous) | (labels == current)]
        size = len(frame)
        up = predict(_with(frame, column, [values[current]] * size))
        down = predict(_with(frame, column, [values[previous]] * size))
        steps.append((up - down).mean(axis=0))
    accumulated = np.vstack([np.zeros((1, len(steps[0]))), np.cumsum(np.array(steps), axis=0)])
    accumulated -= (counts[:, None] * accumulated).sum(axis=0) / max(counts.sum(), 1)
    return present, accumulated, counts


def _order(raw: pd.Series, column: str, orders: dict[str, list[str]]) -> tuple[list[str], str]:
    if pd.api.types.is_bool_dtype(raw) or set(raw.dropna().unique()) <= {True, False}:
        return ["False", "True"], "False, then True"
    if orders.get(column):
        return [str(v) for v in orders[column]], "the order set in Columns"
    if isinstance(raw.dtype, pd.CategoricalDtype) and raw.cat.ordered:
        return [str(v) for v in raw.cat.categories], "the column's own order"
    raise AleError(
        f"{column} has no order: accumulated local effects step from one category to the next, "
        "so they need one (set it in Columns), or read the partial dependence."
    )


def _sentence(column: str, kind: str, curve: np.ndarray, grid: list[Any], output: str) -> str:
    span = float(curve.max() - curve.min())
    high, low = grid[int(np.argmax(curve))], grid[int(np.argmin(curve))]
    show = (lambda v: f"{v:.4g}") if kind == "number" else str
    return (
        f"Over its real values, {column} moves the {output.lower()} by {span:.3g} at most: "
        f"highest near {show(high)}, lowest near {show(low)}."
    )


def ale(
    pipeline: Pipeline,
    x: pd.DataFrame,
    task: Task,
    column: str,
    bins: int = BINS,
    orders: dict[str, list[str]] | None = None,
    rule: str | None = None,
    sign: int | None = None,
) -> Ale:
    if column not in x.columns:
        raise AleError(f"{column} is not an input of this model.")
    raw = cast(pd.Series, x[column])
    if pd.api.types.is_datetime64_any_dtype(raw):
        raise AleError(f"{column} is a date: read its partial dependence.")
    predict, names, output = predictor(pipeline, task)
    xs = x.iloc[_sample(len(x), ROWS)]
    bins = int(min(max(bins, 2), MAX_BINS))
    if _numeric(raw):
        edges, accumulated, counts = number_effects(predict, xs, column, bins)
        grid: list[str | float] = [float(v) for v in edges]
        kind, order = "number", None
    else:
        wanted, order = _order(raw, column, orders or {})
        labels, accumulated, counts = category_effects(predict, xs, column, wanted[:TOP_CATEGORIES])
        grid, kind = cast(list[str | float], list(labels)), "category"
    curves = {names[j]: [float(v) for v in accumulated[:, j]] for j in range(len(names))}
    main = accumulated[:, 0] if len(names) else np.zeros(1)
    checked = monotonic(pipeline, x, task, column, sign) if sign else None
    return Ale(
        column,
        kind,
        grid,
        curves,
        [int(c) for c in counts],
        output,
        len(xs),
        order,
        _sentence(column, kind, main, grid, names[0] if len(names) == 1 else output),
        rule,
        checked,
    )


def monotonic(pipeline: Pipeline, x: pd.DataFrame, task: Task, column: str, sign: int) -> Monotonic:
    """Each checked row's prediction along a grid of the column (5th to 95th percentile) must
    only rise (+1) or only fall (−1): what a monotonic constraint promises, row by row."""
    predict, _, output = predictor(pipeline, task)
    raw = cast(pd.Series, x[column])
    values = raw.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    rows = x.iloc[_sample(len(x), CHECK_ROWS)]
    grid = np.unique(np.quantile(values, np.linspace(0.05, 0.95, CHECK_GRID)))
    repeated = rows.loc[rows.index.repeat(len(grid))].copy()
    repeated[column] = np.tile(grid, len(rows))
    found = predict(repeated)[:, -1].reshape(len(rows), len(grid))  # the last output (binary: 1)
    steps = np.diff(found, axis=1) * sign
    scale = max(float(np.abs(found).max()), 1.0) * TOLERANCE
    wrong = (steps < -scale).any(axis=1)
    largest = float(-steps.min()) if steps.size and steps.min() < -scale else 0.0
    direction = "rise" if sign > 0 else "fall"
    if wrong.any():
        sentence = (
            f"{column}: {int(wrong.sum())} of {len(rows)} rows go the wrong way somewhere (by up "
            f"to {largest:.3g}): the constraint is not held."
        )
    else:
        sentence = (
            f"{column}: checked on {len(rows)} rows, the {output.lower()} only {direction}s as "
            f"{column} grows."
        )
    return Monotonic(column, sign, len(rows), int(wrong.sum()), largest, sentence)
