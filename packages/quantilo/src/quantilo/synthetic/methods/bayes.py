"""Bayesian network with a Chow-Liu tree.

Columns are discretised (numbers and dates into up to 10 quantile intervals; empty is a state of
its own) and linked by the maximum spanning tree of their mutual information. Each column is drawn
given its parent's state; numbers then take a real value of their interval.
"""

import numpy as np

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

BINS = 10


def discretise(values: np.ndarray, categorical: bool) -> np.ndarray:
    """States 0..k-1; empty values get state k (one past the last)."""
    missing = np.isnan(values)
    if categorical:
        states = np.where(missing, -1, values).astype(np.int64)
        top = int(states.max()) + 1 if states.size else 0
        return np.where(missing, top, states)
    present = values[~missing]
    if present.size == 0:
        return np.zeros(len(values), dtype=np.int64)
    edges = np.unique(np.quantile(present, np.linspace(0, 1, BINS + 1))[1:-1])
    states = np.searchsorted(edges, np.where(missing, 0, values), side="right")
    return np.where(missing, len(edges) + 1, states).astype(np.int64)


def mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    rows, columns = int(a.max()) + 1, int(b.max()) + 1
    table = np.bincount(a * columns + b, minlength=rows * columns).reshape(rows, columns)
    joint = table / table.sum()
    pa, pb = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    used = joint > 0
    return float((joint[used] * np.log(joint[used] / (pa @ pb)[used])).sum())


def chow_liu(states: list[np.ndarray]) -> tuple[int, list[tuple[int, int, float]]]:
    """Root (largest total information) and the maximum spanning tree as (parent, child, mi)."""
    size = len(states)
    info = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            info[i, j] = info[j, i] = mutual_information(states[i], states[j])
    root = int(np.argmax(info.sum(axis=1))) if size else 0
    placed, edges = {root}, []
    while len(placed) < size:
        best = max(
            ((p, c) for p in placed for c in range(size) if c not in placed),
            key=lambda pair: info[pair[0], pair[1]],
        )
        edges.append((best[0], best[1], float(info[best])))
        placed.add(best[1])
    return root, edges


def generate(
    coded: Coded, learned: list[Learned], options: Options, should_stop: StopCheck
) -> tuple[Coded, Fitted]:
    rng = np.random.default_rng(options.seed)
    names = [item.name for item in learned]
    states = [discretise(coded[i.name], i.categorical) for i in learned]
    root, edges = chow_liu(states)
    if should_stop():
        raise Cancelled
    drawn: dict[int, np.ndarray] = {root: marginal(states[root], options.rows, rng)}
    for parent, child, _ in edges:
        if should_stop():
            raise Cancelled
        drawn[child] = draw_in_groups(states[parent], states[child], drawn[parent], rng)
    result: Coded = {}
    for i, item in enumerate(learned):
        result[item.name] = draw_in_groups(states[i], coded[item.name], drawn[i], rng)
    return result, Fitted(
        "bayes", edges=[Edge(names[p], names[c], strength) for p, c, strength in edges]
    )
