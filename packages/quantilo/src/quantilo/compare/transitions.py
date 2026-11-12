"""Status transitions as a Markov chain (062): the transition matrix between consecutive
observations of the same entity, its stationary distribution, and, when some states are never
left (churned, closed), the absorbing chain: expected steps before reaching one of them."""

from dataclasses import dataclass, field

import numpy as np

ABSORBING_STAY = 1.0 - 1e-12  # a state whose every observed exit stays in it


@dataclass
class Chain:
    states: list[str]
    counts: list[list[int]]  # counts[i][j]: transitions observed from state i to state j
    probabilities: list[list[float]]  # row-normalised counts (a row with no exit stays in place)
    absorbing: list[str]  # states never left in the data
    never_left: list[str]  # absorbing states with no exit observed at all (only last seen)
    stay_steps: list[float | None]  # expected steps in each state before leaving it: 1 / (1 - p_ii)
    stationary: list[float] | None  # long-run share of time in each state (no absorbing state)
    stationary_note: str | None
    steps_to_absorption: dict[str, float] = field(default_factory=dict)  # transient state → steps
    absorption: dict[str, dict[str, float]] = field(default_factory=dict)  # from → absorbing → p
    transitions: int = 0


def count_transitions(entities: np.ndarray, states: np.ndarray, levels: list[str]) -> np.ndarray:
    """Rows already sorted by entity then time: each pair of consecutive rows of one entity is a
    transition."""
    index = {level: i for i, level in enumerate(levels)}
    codes = np.array([index.get(str(s), -1) for s in states])
    same = entities[1:] == entities[:-1]
    keep = same & (codes[1:] >= 0) & (codes[:-1] >= 0)
    counts = np.zeros((len(levels), len(levels)), dtype=int)
    np.add.at(counts, (codes[:-1][keep], codes[1:][keep]), 1)
    return counts


def stationary_distribution(p: np.ndarray) -> np.ndarray:
    """pi P = pi with pi summing to 1, solved by least squares (unique for an irreducible chain)."""
    k = len(p)
    system = np.vstack([p.T - np.eye(k), np.ones(k)])
    target = np.concatenate([np.zeros(k), [1.0]])
    pi, *_ = np.linalg.lstsq(system, target, rcond=None)
    pi = np.clip(pi, 0, None)
    return pi / pi.sum()


def _reachable(p: np.ndarray, start: int) -> set[int]:
    seen, todo = {start}, [start]
    while todo:
        i = todo.pop()
        for j in np.flatnonzero(p[i] > 0):
            if int(j) not in seen:
                seen.add(int(j))
                todo.append(int(j))
    return seen


def chain(counts: np.ndarray, levels: list[str]) -> Chain:
    counts = np.asarray(counts, dtype=int)
    k = len(levels)
    exits = counts.sum(axis=1)
    p = np.zeros((k, k))
    for i in range(k):
        if exits[i] > 0:
            p[i] = counts[i] / exits[i]
        else:
            p[i, i] = 1.0  # R-MKV-03: no exit seen, the state is read as absorbing
    absorbing = [i for i in range(k) if p[i, i] >= ABSORBING_STAY]
    never_left = [levels[i] for i in range(k) if exits[i] == 0]
    stay = [None if p[i, i] >= ABSORBING_STAY else float(1 / (1 - p[i, i])) for i in range(k)]
    result = Chain(
        states=list(levels),
        counts=counts.tolist(),
        probabilities=p.tolist(),
        absorbing=[levels[i] for i in absorbing],
        never_left=never_left,
        stay_steps=stay,
        stationary=None,
        stationary_note=None,
        transitions=int(counts.sum()),
    )
    if not absorbing:
        whole = all(len(_reachable(p, i)) == k for i in range(k))
        result.stationary = [float(x) for x in stationary_distribution(p)]
        if not whole:
            result.stationary_note = (
                "Some states cannot be reached from the others: the long-run shares depend on "
                "the starting state; the shares shown are one possible balance."
            )
        return result
    result.stationary_note = (
        "Some states are never left: in the long run every entity ends in one of them, so the "
        "long-run shares are replaced by the expected steps before reaching them."
    )
    transient = [i for i in range(k) if i not in absorbing]
    if not transient:
        return result
    q = p[np.ix_(transient, transient)]
    r = p[np.ix_(transient, absorbing)]
    try:
        fundamental = np.linalg.inv(np.eye(len(transient)) - q)
    except np.linalg.LinAlgError:
        return result
    steps = fundamental.sum(axis=1)
    probabilities = fundamental @ r
    for row, i in enumerate(transient):
        if not np.isfinite(steps[row]) or steps[row] < 0:
            continue
        result.steps_to_absorption[levels[i]] = float(steps[row])
        result.absorption[levels[i]] = {
            levels[a]: float(probabilities[row, col]) for col, a in enumerate(absorbing)
        }
    return result
