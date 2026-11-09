"""Coherent totals: forecasts of a hierarchy (stores → regions → total) that add up (039).

Every node is forecast on its own (base forecasts ŷ), then made coherent by ỹ = S G ŷ, where S
sums the bottom series into every node and G maps the base forecasts to bottom ones:

- bottom-up: G keeps the bottom forecasts;
- top-down by historical proportions: G spreads the total by each leaf's share of the total over
  the whole history ("proportions of the historical averages", Gross & Sohl 1990);
- MinT with a shrunk covariance (Wickramasuriya, Athanasopoulos & Hyndman, 2019):
  G = (S'W⁻¹S)⁻¹S'W⁻¹, W = covariance of the in-sample residuals shrunk towards its diagonal
  (Schäfer & Strimmer, 2005), as in R's `hts` and `fable`.

Pure numpy: no DuckDB, no model.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

Method = Literal["bottom_up", "top_down", "mint_shrink"]
METHODS: tuple[Method, ...] = ("bottom_up", "top_down", "mint_shrink")
TOTAL = "Total"
MIN_RESIDUALS = 20  # aligned residuals per node for MinT; fewer: bottom-up
TOLERANCE = 1e-6  # children must add up to their parent within this


@dataclass
class Tree:
    """Nodes from the top (the total) down to the leaves, level by level."""

    labels: list[str]
    levels: list[str]  # the level (column name, or "Total") of each node
    parents: list[int | None]  # index of each node's parent
    bottom: list[int]  # indices of the leaves, in the order they were given

    def summing(self) -> np.ndarray:
        """S: one row per node, one column per leaf; 1 where the leaf belongs to the node."""
        matrix = np.zeros((len(self.labels), len(self.bottom)))
        for column, leaf in enumerate(self.bottom):
            node: int | None = leaf
            while node is not None:
                matrix[node, column] = 1.0
                node = self.parents[node]
        return matrix


def build_tree(leaves: list[str], leaf_level: str, ancestors: list[tuple[str, list[str]]]) -> Tree:
    """`ancestors`: for each level above the leaves (finest first), its name and the label of
    each leaf's node at that level. The total is added as the root."""
    labels: list[str] = [TOTAL]
    levels: list[str] = [TOTAL]
    parents: list[int | None] = [None]
    above: dict[tuple[int, str], int] = {}  # (level depth, label) → node
    chain = list(reversed(ancestors))  # coarsest first
    for depth, (name, of_leaf) in enumerate(chain):
        for i, label in enumerate(of_leaf):
            if (depth, label) in above:
                continue
            parent = 0 if depth == 0 else above[(depth - 1, chain[depth - 1][1][i])]
            above[(depth, label)] = len(labels)
            labels.append(label)
            levels.append(name)
            parents.append(parent)
    bottom = []
    for i, label in enumerate(leaves):
        parent = above[(len(chain) - 1, chain[-1][1][i])] if chain else 0
        bottom.append(len(labels))
        labels.append(label)
        levels.append(leaf_level)
        parents.append(parent)
    return Tree(labels, levels, parents, bottom)


def is_function_of(pairs: list[tuple[str, str]]) -> bool:
    """Whether each child label goes with a single parent label (a functional dependency)."""
    seen: dict[str, str] = {}
    return all(seen.setdefault(child, parent) == parent for child, parent in pairs)


def shrink_covariance(residuals: np.ndarray) -> tuple[np.ndarray, float]:
    """Covariance of the residuals (periods × nodes, not centred: they average about zero),
    shrunk towards its diagonal by the Schäfer-Strimmer intensity λ ∈ [0, 1]."""
    n = residuals.shape[0]
    sample = residuals.T @ residuals / n
    variance = np.diag(sample).copy()
    sd = np.sqrt(variance)
    sd[sd == 0] = 1.0
    scaled = residuals / sd
    correlation = scaled.T @ scaled / n
    spread = (scaled**2).T @ (scaled**2) - (scaled.T @ scaled) ** 2 / n
    spread *= 1 / (n * (n - 1)) if n > 1 else 0.0
    np.fill_diagonal(spread, 0.0)
    off = correlation - np.eye(len(variance))
    np.fill_diagonal(off, 0.0)
    total = float(np.sum(off**2))
    shrink = 1.0 if total == 0 else float(np.clip(np.sum(spread) / total, 0.0, 1.0))
    covariance = shrink * np.diag(variance) + (1 - shrink) * sample
    # A constant node (no error) would make W singular: a tiny floor on the diagonal.
    floor = max(float(variance.max()) * 1e-9, 1e-12)
    covariance[np.diag_indices_from(covariance)] = np.maximum(np.diag(covariance), floor)
    return covariance, shrink


def projection(
    tree: Tree,
    method: Method,
    residuals: np.ndarray | None = None,
    history: np.ndarray | None = None,
) -> np.ndarray:
    """G (leaves × nodes) for the method. `residuals` (nodes × periods) for MinT, `history`
    (nodes × periods) for top-down."""
    summing = tree.summing()
    nodes, leaves = summing.shape
    if method == "bottom_up":
        matrix = np.zeros((leaves, nodes))
        matrix[np.arange(leaves), tree.bottom] = 1.0
        return matrix
    if method == "top_down":
        if history is None:
            raise ValueError("Top-down needs the history of the leaves.")
        sums = np.nansum(history[tree.bottom], axis=1)
        whole = float(sums.sum())
        shares = sums / whole if whole else np.full(leaves, 1 / leaves)
        matrix = np.zeros((leaves, nodes))
        matrix[:, 0] = shares
        return matrix
    if residuals is None:
        raise ValueError("MinT needs the in-sample residuals of every node.")
    covariance, _ = shrink_covariance(residuals.T)
    inverse = np.linalg.pinv(covariance)
    return np.linalg.solve(summing.T @ inverse @ summing, summing.T @ inverse)


def reconcile(tree: Tree, base: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Coherent forecasts S G ŷ for base forecasts (nodes × steps)."""
    return tree.summing() @ (matrix @ base)


def incoherence(tree: Tree, forecasts: np.ndarray) -> float:
    """Largest gap between a node and the sum of its children, over every node and step."""
    gap = 0.0
    for node in range(len(tree.labels)):
        children = [i for i, parent in enumerate(tree.parents) if parent == node]
        if children:
            difference = forecasts[node] - forecasts[children].sum(axis=0)
            gap = max(gap, float(np.max(np.abs(difference))))
    return gap


def aligned_residuals(residuals: list[np.ndarray | None]) -> np.ndarray | None:
    """Residuals of every node aligned on their last periods (models drop a few at the start);
    None when one node has none."""
    if any(r is None or not len(r) for r in residuals):
        return None
    usable = [np.asarray(r, dtype=float) for r in residuals if r is not None]
    length = min(len(r) for r in usable)
    stacked = np.vstack([r[len(r) - length :] for r in usable])
    return np.nan_to_num(stacked)


def auto_method(residuals: np.ndarray | None) -> Method:
    """MinT when every node has enough aligned residuals, bottom-up otherwise."""
    return (
        "mint_shrink"
        if residuals is not None and residuals.shape[1] >= MIN_RESIDUALS
        else "bottom_up"
    )
