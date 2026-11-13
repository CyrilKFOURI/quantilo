"""SHAP values for any model, by sampling permutations (126, Explain › Advanced › More).

Exact SHAP exists for linear models and trees (020, 125). For the others (k-NN, SVM, MLP, a
stacked ensemble…) the Shapley value of a column is estimated as in Štrumbelj and Kononenko
(2014): walk from a background row to the explained row, switching the columns one by one in a
random order, and credit each column with the change of the prediction when it switches. Averaged
over the orders and the background rows, that is the interventional Shapley value.

R-XAI-12: 10 pairs of antithetic orders (an order and its reverse: they cancel much of each
other's noise), 50 background rows drawn from the evaluation rows; rows explained =
min(200, budget / (orders × columns × background)), the shown rows first; budget 3·10⁶
predictions on a computer, 3·10⁵ in the browser; fewer than 10 rows: refused, with what to
change. Two classes: the probability of the second class; several classes: refused (one base
value per explanation).

Additivity is exact, not approximate: along one order the credits add up to f(x) − f(background
row), so base (the mean prediction of the background rows) + Σ φ = f(x) for every row and any
number of orders. A 95 % interval per value comes from the spread over the pairs of orders.
numpy and the model's own `predict`: no thread, no `shap` package, so it runs in Pyodide."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from quantilo.ml import browser
from quantilo.ml import explain as explainer
from quantilo.ml.ale import predictor
from quantilo.ml.target import Task

SEED = 42
PAIRS = 10  # antithetic pairs of orders
BACKGROUND = 50
ROWS = 200
MIN_ROWS = 10
BUDGET = 3_000_000  # model predictions (rows) on a computer
BROWSER_BUDGET = 300_000
CHUNK = 20_000  # rows predicted at once
Z = 1.96  # 95 % interval
METHOD = "Permutation sampling (any model, estimated)"


class PermutationShapError(ValueError):
    """Permutation SHAP cannot run on this model or these rows (with the reason)."""


@dataclass
class PermutationShap:
    shap: explainer.Shap  # the same shape as Explain's SHAP: beeswarm and waterfall read it
    half_width: list[list[float]]  # rows × columns of `shap.columns`: the 95 % interval's half
    orders: int  # permutations used (pairs × 2)
    background: int
    additivity: float  # the largest |base + Σ φ − f(x)| over the rows (floating-point only)
    widest: float  # the largest half width, on the output's scale
    sentence: str


def budget() -> int:
    return BROWSER_BUDGET if browser.active() else BUDGET


def rows_within(columns: int, pairs: int = PAIRS, background: int = BACKGROUND) -> int:
    """R-XAI-12: how many rows the budget explains."""
    cost = max(1, 2 * pairs * max(columns, 1) * background)
    return int(min(ROWS, budget() // cost))


def _predict_all(predict, frame: pd.DataFrame) -> np.ndarray:
    parts = [predict(frame.iloc[i : i + CHUNK]) for i in range(0, len(frame), CHUNK)]
    return np.concatenate(parts)[:, 0] if parts else np.zeros(0)


def _walk(
    predict,
    x: pd.DataFrame,
    background: pd.DataFrame,
    order: np.ndarray,
    start: np.ndarray,
) -> np.ndarray:
    """Credits (rows × columns) along one order, averaged over the background rows: each block
    of the frame is the background, whose columns take the row's values one by one."""
    n, b, m = len(x), len(background), x.shape[1]
    frame = pd.concat([background] * n, ignore_index=True)
    previous = start
    credits = np.zeros((n, m))
    for j in order:
        column = x.columns[j]
        frame[column] = np.repeat(cast(pd.Series, x[column]).to_numpy(), b)
        current = _predict_all(predict, frame)
        credits[:, j] = (current - previous).reshape(n, b).mean(axis=1)
        previous = current
    return credits


def permutation_shap(
    pipeline: Pipeline,
    x: pd.DataFrame,
    rows: np.ndarray,
    task: Task,
    shown: list[int],
    texts: set[str] | None = None,
    pairs: int = PAIRS,
    background: int = BACKGROUND,
) -> PermutationShap:
    """`x`: the evaluation rows (their file rows in `rows`); `shown`: file rows explained first."""
    if task == "multiclass":
        raise PermutationShapError(
            "Permutation SHAP explains one output: with several classes, read permutation "
            "importance and the partial dependence."
        )
    if x.empty:
        raise PermutationShapError("No rows to explain.")
    pairs = int(min(max(pairs, 1), 50))
    background = int(min(max(background, 5), 200, len(x)))
    m = x.shape[1]
    limit = min(rows_within(m, pairs, background), len(x))
    if limit < MIN_ROWS:
        raise PermutationShapError(
            f"R-XAI-12: with {m} columns, {2 * pairs} orders and {background} background rows, "
            f"the budget explains fewer than {MIN_ROWS} rows: lower the orders or the background "
            "rows, or keep fewer columns."
        )
    predict, _, output = predictor(pipeline, task)
    keep = [int(i) for i in np.flatnonzero(np.isin(rows, shown))]
    picked = explainer._sample(len(x), limit, keep)
    xs = x.iloc[picked].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    reference = x.iloc[np.sort(rng.choice(len(x), background, replace=False))]
    reference = reference.reset_index(drop=True)
    base_rows = _predict_all(predict, reference)
    base = float(base_rows.mean())
    start = np.tile(base_rows, len(xs))
    estimates = np.zeros((pairs, len(xs), m))
    for p in range(pairs):
        order = rng.permutation(m)
        forward = _walk(predict, xs, reference, order, start)
        backward = _walk(predict, xs, reference, order[::-1], start)
        estimates[p] = (forward + backward) / 2
    phi = estimates.mean(axis=0)
    spread = estimates.std(axis=0, ddof=1) if pairs > 1 else np.zeros_like(phi)
    half = Z * spread / np.sqrt(pairs)
    exact = _predict_all(predict, xs)
    additivity = float(np.max(np.abs(base + phi.sum(axis=1) - exact))) if len(xs) else 0.0
    note = None
    if len(xs) < len(x):
        note = (
            f"R-XAI-12: {len(xs):,} of {len(x):,} rows explained (the shown rows first): each "
            f"costs {2 * pairs} orders × {m} columns × {background} background rows of "
            "predictions."
        )
    columns = [str(c) for c in x.columns]
    scale = "prediction" if task == "regression" else "probability"
    summary = explainer.summarise(
        phi, columns, xs, [int(v) for v in rows[picked]], texts or set(), METHOD, scale, base, note
    )
    classes = [str(c) for c in getattr(pipeline, "classes_", [])]
    summary.positive = classes[1] if task == "binary" and len(classes) == 2 else None
    widest = float(half.max()) if half.size else 0.0
    said = output[:1].lower() + output[1:]  # “Probability of “Yes”” keeps its class's case
    top = summary.features[0] if summary.features else None
    sentence = (
        f"Estimated on {len(xs):,} rows from {2 * pairs} orders and {background} background "
        f"rows: {top.column if top else 'no column'} moves the {said} most "
        f"({top.mean_abs:.3g} on average)"
        if top
        else "No column moves the prediction."
    )
    sentence += f"; every value is within ±{widest:.2g} (95 %)." if top else ""
    return PermutationShap(
        summary,
        [[float(v) for v in row] for row in half],
        2 * pairs,
        background,
        additivity,
        widest,
        sentence,
    )
