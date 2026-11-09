"""Sequential CART, in the manner of synthpop.

Columns are generated one after the other. Each one is predicted by a decision tree learned on
the columns already generated; every synthetic row then takes a real value from its leaf. A leaf
holds at least 5 real rows, so it never points to a single real row.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from quantilo.synthetic.associations import strength_matrix
from quantilo.synthetic.encode import Coded, Learned
from quantilo.synthetic.methods import (
    Cancelled,
    Edge,
    Fitted,
    Options,
    StopCheck,
    draw_in_groups,
    marginal,
)

MIN_LEAF = 5
SMOOTH_MIN_DISTINCT = 20  # values repeated on a few spikes are not smoothed
ORDER_SAMPLE = 5_000


def auto_order(coded: Coded, learned: list[Learned], seed: int) -> tuple[list[str], list[Edge]]:
    """The column most linked to the others first, then by decreasing link with the columns
    already placed."""
    names = [item.name for item in learned]
    if len(names) < 2:
        return names, []
    rows = len(coded[names[0]])
    pick = np.random.default_rng(seed).permutation(rows)[:ORDER_SAMPLE]
    matrix = strength_matrix([(coded[i.name][pick], i.categorical) for i in learned])
    first = int(np.argmax(matrix.sum(axis=1)))
    placed, edges = [first], []
    while len(placed) < len(names):
        remaining = [i for i in range(len(names)) if i not in placed]
        links = matrix[np.ix_(remaining, placed)]
        best = int(np.argmax(links.sum(axis=1)))
        chosen = remaining[best]
        edges.append(
            Edge(
                names[placed[int(np.argmax(links[best]))]], names[chosen], float(links[best].max())
            )
        )
        placed.append(chosen)
    return [names[i] for i in placed], edges


def smoothable(item: Learned, values: np.ndarray) -> bool:
    """Only continuous numbers with many distinct values: noise on whole numbers, dates or a
    few repeated values would create values that give the synthetic data away."""
    return item.kind == "number" and len(np.unique(values)) >= SMOOTH_MIN_DISTINCT


def _leaves(tree: DecisionTreeClassifier | DecisionTreeRegressor, data: np.ndarray) -> np.ndarray:
    return tree.apply(data).astype(np.int64)


def _smooth(
    values: np.ndarray,
    leaves: np.ndarray,
    train_values: np.ndarray,
    train_leaves: np.ndarray,
    bounds: tuple[float | None, float | None],
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian noise with the leaf's spread * Silverman's factor, kept inside the bounds."""
    present = ~np.isnan(train_values)
    if not present.any():
        return values
    unique, inverse = np.unique(train_leaves[present], return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=train_values[present])
    squares = np.bincount(inverse, weights=train_values[present] ** 2)
    means = sums / counts
    spread = np.sqrt(np.maximum(squares / counts - means**2, 0))
    width = spread * 1.06 * counts ** (-1 / 5)
    position = np.clip(np.searchsorted(unique, leaves), 0, len(unique) - 1)
    noise_scale = np.where(unique[position] == leaves, width[position], 0.0)
    noisy = values + rng.standard_normal(len(values)) * noise_scale
    low, high = bounds
    if low is not None and high is not None:
        noisy = np.clip(noisy, low, high)
    return np.where(np.isnan(values), np.nan, noisy)


def generate(
    coded: Coded, learned: list[Learned], options: Options, should_stop: StopCheck
) -> tuple[Coded, Fitted]:
    rng = np.random.default_rng(options.seed)
    by_name = {item.name: item for item in learned}
    if options.order:
        order = [n for n in options.order if n in by_name]
        order += [n for n in by_name if n not in order]
        edges: list[Edge] = []
    else:
        order, edges = auto_order(coded, learned, options.seed)
    rows = options.rows
    result: Coded = {}
    for position, name in enumerate(order):
        if should_stop():
            raise Cancelled
        item, target = by_name[name], coded[name]
        if position == 0:
            first = marginal(target, rows, rng)
            if options.smooth and smoothable(item, target[~np.isnan(target)]):
                no_leaf = np.zeros(len(target), dtype=np.int64)
                first = _smooth(
                    first,
                    np.zeros(rows, dtype=np.int64),
                    target,
                    no_leaf,
                    (item.minimum, item.maximum),
                    rng,
                )
            result[name] = first
            continue
        previous = order[:position]
        train = np.column_stack([coded[n] for n in previous])
        synthetic = np.column_stack([result[n] for n in previous])
        result[name] = _column(item, target, train, synthetic, options, rng)
    return result, Fitted("cart", order=order, edges=edges)


def _column(
    item: Learned,
    target: np.ndarray,
    train: np.ndarray,
    synthetic: np.ndarray,
    options: Options,
    rng: np.random.Generator,
) -> np.ndarray:
    seed = options.seed
    if item.categorical:
        # Empty is one more class: the tree learns when values are missing too.
        classes = np.where(np.isnan(target), -1, target).astype(np.int64)
        tree = DecisionTreeClassifier(min_samples_leaf=MIN_LEAF, random_state=seed)
        tree.fit(train, classes)
        drawn = draw_in_groups(_leaves(tree, train), target, _leaves(tree, synthetic), rng)
        return drawn
    missing = np.isnan(target)
    values = np.full(len(synthetic), np.nan)
    keep = np.ones(len(synthetic), dtype=bool)
    if missing.any():
        empty = DecisionTreeClassifier(min_samples_leaf=MIN_LEAF, random_state=seed)
        empty.fit(train, missing.astype(np.int64))
        flags = draw_in_groups(
            _leaves(empty, train), missing.astype(float), _leaves(empty, synthetic), rng
        )
        keep = flags == 0
        if missing.all() or not keep.any():
            return values
    present = ~missing
    tree = DecisionTreeRegressor(min_samples_leaf=MIN_LEAF, random_state=seed)
    tree.fit(train[present], target[present])
    train_leaves = _leaves(tree, train[present])
    wanted = _leaves(tree, synthetic[keep])
    drawn = draw_in_groups(train_leaves, target[present], wanted, rng)
    if options.smooth and smoothable(item, target[present]):
        drawn = _smooth(
            drawn, wanted, target[present], train_leaves, (item.minimum, item.maximum), rng
        )
    values[keep] = drawn
    return values
