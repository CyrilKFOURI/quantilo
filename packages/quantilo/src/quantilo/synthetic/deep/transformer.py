"""Mode-specific normalization (CTGAN paper, section 4.2), from the coded columns of 023.

A number (or a date in seconds) becomes a mode, one-hot, chosen among the components of a
Bayesian Gaussian mixture, and a scalar alpha = (x - mean) / (4 std) in [-0.99, 0.99]. An empty
value is one more mode (an addition: the paper has no empty values). A category becomes a
one-hot, empty being one more level.
"""

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture

from quantilo.synthetic.encode import Coded, Learned

MAX_MODES = 10
MIN_WEIGHT = 0.005
FIT_SAMPLE = 5_000
ALPHA_LIMIT = 0.99


@dataclass
class Span:
    """A block of output columns and the activation that produces it."""

    size: int
    activation: Literal["tanh", "softmax"]
    discrete: bool = False  # a category: part of the conditional vector (CTGAN)


@dataclass
class ColumnTransform:
    name: str
    categorical: bool
    means: np.ndarray = field(default_factory=lambda: np.zeros(0))  # per kept mode
    stds: np.ndarray = field(default_factory=lambda: np.zeros(0))
    levels: int = 0  # categories: one-hot size without the empty level
    empty: bool = False  # an "empty" mode or level is appended
    mixture: BayesianGaussianMixture | None = None  # numbers with several values
    kept: np.ndarray | None = None  # components of the mixture heavy enough to be modes

    @property
    def spans(self) -> list[Span]:
        if self.categorical:
            return [Span(self.levels + int(self.empty), "softmax", True)]
        return [Span(1, "tanh"), Span(len(self.means) + int(self.empty), "softmax")]

    @property
    def width(self) -> int:
        return sum(s.size for s in self.spans)


def fit_column(values: np.ndarray, item: Learned, seed: int) -> ColumnTransform:
    missing = np.isnan(values)
    empty = bool(missing.any())
    if item.categorical:
        return ColumnTransform(item.name, True, levels=max(len(item.levels), 1), empty=empty)
    present = values[~missing]
    if present.size == 0:
        return ColumnTransform(item.name, False, np.zeros(1), np.ones(1), empty=empty)
    distinct = np.unique(present)
    if distinct.size == 1:
        return ColumnTransform(item.name, False, distinct[:1], np.ones(1), empty=empty)
    rng = np.random.default_rng(seed)
    sample = (
        present if present.size <= FIT_SAMPLE else rng.choice(present, FIT_SAMPLE, replace=False)
    )
    mixture = BayesianGaussianMixture(
        n_components=min(MAX_MODES, distinct.size),
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=0.001,
        max_iter=100,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mixture.fit(sample.reshape(-1, 1))
    kept = np.asarray(mixture.weights_) > MIN_WEIGHT
    means = np.asarray(mixture.means_).reshape(-1)[kept]
    stds = np.sqrt(np.asarray(mixture.covariances_).reshape(-1))[kept]
    return ColumnTransform(item.name, False, means, np.maximum(stds, 1e-6), 0, empty, mixture, kept)


@dataclass
class Transformer:
    columns: list[ColumnTransform]

    @property
    def spans(self) -> list[Span]:
        return [s for c in self.columns for s in c.spans]

    @property
    def width(self) -> int:
        return sum(c.width for c in self.columns)


def fit(coded: Coded, learned: list[Learned], seed: int) -> Transformer:
    return Transformer([fit_column(coded[i.name], i, seed + k) for k, i in enumerate(learned)])


def _modes(column: ColumnTransform, present: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A mode per value, drawn from its membership probabilities (as the paper)."""
    if column.mixture is None or column.kept is None or len(column.means) == 1:
        return np.zeros(len(present), dtype=np.int64)
    probabilities = column.mixture.predict_proba(present.reshape(-1, 1))[:, column.kept]
    probabilities = probabilities + 1e-12
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    cumulative = probabilities.cumsum(axis=1)
    draws = rng.random((len(present), 1))
    return np.minimum((cumulative < draws).sum(axis=1), len(column.means) - 1)


def transform(transformer: Transformer, coded: Coded, seed: int) -> np.ndarray:
    """Rows as the networks see them: alpha and one-hots, float32."""
    rng = np.random.default_rng(seed)
    rows = len(next(iter(coded.values())))
    blocks = []
    for column in transformer.columns:
        values = coded[column.name]
        missing = np.isnan(values)
        if column.categorical:
            block = np.zeros((rows, column.width), dtype=np.float32)
            index = np.where(missing, column.levels, np.nan_to_num(values)).astype(np.int64)
            block[np.arange(rows), np.clip(index, 0, column.width - 1)] = 1
            blocks.append(block)
            continue
        modes = np.full(rows, len(column.means), dtype=np.int64)  # the empty mode
        alpha = np.zeros(rows, dtype=np.float32)
        present = ~missing
        if present.any():
            picked = _modes(column, values[present], rng)
            modes[present] = picked
            scaled = (values[present] - column.means[picked]) / (4 * column.stds[picked])
            alpha[present] = np.clip(scaled, -ALPHA_LIMIT, ALPHA_LIMIT)
        one_hot = np.zeros((rows, column.width - 1), dtype=np.float32)
        one_hot[np.arange(rows), np.clip(modes, 0, column.width - 2)] = 1
        blocks.append(np.column_stack([alpha, one_hot]))
    return np.column_stack(blocks).astype(np.float32)


def inverse(transformer: Transformer, data: np.ndarray) -> Coded:
    """Network outputs back to coded columns: the most likely mode or level, then the value."""
    result: Coded = {}
    start = 0
    for column in transformer.columns:
        if column.categorical:
            choice = data[:, start : start + column.width].argmax(axis=1)
            result[column.name] = np.where(choice >= column.levels, np.nan, choice).astype(float)
        else:
            alpha = np.clip(data[:, start], -1, 1)
            mode = data[:, start + 1 : start + column.width].argmax(axis=1)
            kept = len(column.means)
            safe = np.minimum(mode, kept - 1)
            values = alpha * 4 * column.stds[safe] + column.means[safe]
            result[column.name] = np.where(mode >= kept, np.nan, values)
        start += column.width
    return result
