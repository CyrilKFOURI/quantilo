"""Gaussian copula.

Each column is mapped to a standard normal through its ranks (numbers, dates) or the cumulative
shares of its levels (categories, where empty is one more level). A correlation matrix is learned
on the normal scores (pairwise, over rows where both values exist); new scores are drawn from it
and mapped back through the real quantiles or level shares. Empty numbers are drawn at the real
rate.
"""

import numpy as np
import pandas as pd
from scipy import stats

from quantilo.synthetic.encode import Coded, Learned
from quantilo.synthetic.methods import Cancelled, Fitted, Options, StopCheck

EDGE = 1e-6


def _levels(values: np.ndarray) -> np.ndarray:
    """Level indexes with empty as one more level."""
    present = values[~np.isnan(values)]
    top = int(present.max()) + 1 if present.size else 0
    return np.where(np.isnan(values), top, values).astype(np.int64)


def _to_uniform(values: np.ndarray, categorical: bool, rng: np.random.Generator) -> np.ndarray:
    """Uniform scores; NaN where a number is empty."""
    if categorical:
        levels = _levels(values)
        shares = np.bincount(levels) / len(levels)
        lows = np.concatenate([[0.0], np.cumsum(shares)[:-1]])
        u = lows[levels] + rng.random(len(levels)) * shares[levels]
        return np.clip(u, EDGE, 1 - EDGE)
    u = np.full(len(values), np.nan)
    present = ~np.isnan(values)
    if present.any():
        u[present] = (stats.rankdata(values[present]) - 0.5) / int(present.sum())
    return np.clip(u, EDGE, 1 - EDGE)


def _from_uniform(
    u: np.ndarray, values: np.ndarray, categorical: bool, rng: np.random.Generator
) -> np.ndarray:
    if categorical:
        levels = _levels(values)
        bounds = np.cumsum(np.bincount(levels) / len(levels))
        drawn = np.minimum(np.searchsorted(bounds, u, side="right"), len(bounds) - 1).astype(float)
        top = len(bounds) - 1 if np.isnan(values).any() else len(bounds)
        drawn[drawn == top] = np.nan
        return drawn
    present = values[~np.isnan(values)]
    if present.size == 0:
        return np.full(len(u), np.nan)
    result = np.quantile(present, np.clip(u, 0, 1))
    result[rng.random(len(u)) < np.isnan(values).mean()] = np.nan
    return result


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues so the matrix can be sampled, keeping a unit diagonal."""
    matrix = np.nan_to_num((matrix + matrix.T) / 2)
    np.fill_diagonal(matrix, 1.0)
    values, vectors = np.linalg.eigh(matrix)
    fixed = vectors @ np.diag(np.maximum(values, 1e-6)) @ vectors.T
    scale = np.sqrt(np.diag(fixed))
    return fixed / np.outer(scale, scale)


def generate(
    coded: Coded, learned: list[Learned], options: Options, should_stop: StopCheck
) -> tuple[Coded, Fitted]:
    rng = np.random.default_rng(options.seed)
    scores = []
    for item in learned:
        if should_stop():
            raise Cancelled
        u = _to_uniform(coded[item.name], item.categorical, rng)
        scores.append(stats.norm.ppf(u))
    normal = pd.DataFrame(np.column_stack(scores))
    correlation = (
        _nearest_correlation(normal.corr().to_numpy()) if normal.shape[1] > 1 else np.ones((1, 1))
    )
    if should_stop():
        raise Cancelled
    draws = rng.multivariate_normal(np.zeros(len(learned)), correlation, size=options.rows)
    uniform = stats.norm.cdf(draws)
    result: Coded = {}
    for i, item in enumerate(learned):
        result[item.name] = _from_uniform(uniform[:, i], coded[item.name], item.categorical, rng)
    return result, Fitted("copula")
