"""k-prototypes (Huang, 1998): k-means for rows with numbers and categories, without one-hot.

The distance between a row and a prototype is the squared Euclidean distance on the numbers plus
γ times the number of categories that differ. A prototype is the mean of its rows' numbers and the
mode of each of their categories. γ balances the two parts; Huang suggests a value between 1/3 and
2/3 of the average standard deviation of the numbers: γ = 0.5 × that average by default.

The iteration is the batch one (Lloyd): every row moves to its nearest prototype, then every
prototype is recomputed, until no row moves. The seeding is k-means++ on the same mixed distance.
"""

from dataclasses import dataclass

import numpy as np

MAX_ITER = 100


@dataclass
class KPrototypes:
    labels: np.ndarray  # (n,) prototype of each row
    numbers: np.ndarray  # (k, p) numeric part of each prototype
    categories: np.ndarray  # (k, q) category codes of each prototype
    gamma: float
    cost: float  # sum of each row's distance to its prototype
    iterations: int

    def predict(self, numbers: np.ndarray, categories: np.ndarray) -> np.ndarray:
        return nearest(numbers, categories, self.numbers, self.categories, self.gamma)[0]


def default_gamma(numbers: np.ndarray) -> float:
    """0.5 × the mean standard deviation of the numbers (1 without numbers: k-modes)."""
    if numbers.shape[1] == 0:
        return 1.0
    spread = float(np.mean(np.std(numbers, axis=0)))
    return 0.5 * spread if spread > 0 else 1.0


def distances(
    numbers: np.ndarray,
    categories: np.ndarray,
    centres: np.ndarray,
    modes: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """(n, k): squared distance on the numbers + γ × mismatching categories."""
    found = np.empty((len(numbers), len(centres)))
    for j in range(len(centres)):
        numeric = ((numbers - centres[j]) ** 2).sum(axis=1) if numbers.shape[1] else 0.0
        mismatches = (categories != modes[j]).sum(axis=1) if categories.shape[1] else 0.0
        found[:, j] = numeric + gamma * mismatches
    return found


def nearest(
    numbers: np.ndarray,
    categories: np.ndarray,
    centres: np.ndarray,
    modes: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The nearest prototype of each row and its distance."""
    found = distances(numbers, categories, centres, modes, gamma)
    labels = found.argmin(axis=1)
    return labels, found[np.arange(len(found)), labels]


def _seed(
    numbers: np.ndarray, categories: np.ndarray, k: int, gamma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """k-means++: the first prototype at random, each next one drawn with a probability
    proportional to the distance to the nearest prototype already chosen."""
    n = len(numbers)
    chosen = [int(rng.integers(n))]
    closest = distances(numbers, categories, numbers[chosen], categories[chosen], gamma).min(axis=1)
    while len(chosen) < k:
        total = float(closest.sum())
        pick = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=closest / total))
        chosen.append(pick)
        closest = np.minimum(
            closest,
            distances(numbers, categories, numbers[[pick]], categories[[pick]], gamma)[:, 0],
        )
    return numbers[chosen].astype(float).copy(), categories[chosen].copy()


def _modes(categories: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """The most frequent code of each category column within each prototype's rows (the
    smallest code on a tie, as np.argmax does)."""
    modes = np.zeros((k, categories.shape[1]), dtype=categories.dtype)
    for a in range(categories.shape[1]):
        width = int(categories[:, a].max()) + 1 if len(categories) else 1
        counts = np.zeros((k, width), dtype=np.int64)
        np.add.at(counts, (labels, categories[:, a]), 1)
        modes[:, a] = counts.argmax(axis=1)
    return modes


def _update(
    numbers: np.ndarray, categories: np.ndarray, labels: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Means and modes of each prototype's rows; also the size of each prototype."""
    sizes = np.bincount(labels, minlength=k)
    centres = np.zeros((k, numbers.shape[1]))
    if numbers.shape[1]:
        np.add.at(centres, labels, numbers)
        centres /= np.maximum(sizes, 1)[:, None]
    return centres, _modes(categories, labels, k), sizes


def fit_once(
    numbers: np.ndarray,
    categories: np.ndarray,
    k: int,
    gamma: float,
    rng: np.random.Generator,
    start: tuple[np.ndarray, np.ndarray] | None = None,
    max_iter: int = MAX_ITER,
) -> KPrototypes:
    """One run from a seeding (or the given starting prototypes)."""
    centres, modes = start if start is not None else _seed(numbers, categories, k, gamma, rng)
    centres, modes = centres.astype(float).copy(), modes.copy()
    labels = np.full(len(numbers), -1)
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        moved, gaps = nearest(numbers, categories, centres, modes, gamma)
        if np.array_equal(moved, labels):
            break
        labels = moved
        centres, modes, sizes = _update(numbers, categories, labels, k)
        for empty in np.flatnonzero(sizes == 0):  # an empty prototype takes the farthest row
            far = int(gaps.argmax())
            centres[empty], modes[empty] = numbers[far], categories[far]
            labels[far], gaps[far] = empty, 0.0
    labels, gaps = nearest(numbers, categories, centres, modes, gamma)
    return KPrototypes(labels, centres, modes, gamma, float(gaps.sum()), iterations)


def fit(
    numbers: np.ndarray,
    categories: np.ndarray,
    k: int,
    seed: int = 42,
    n_init: int = 4,
    gamma: float | None = None,
) -> KPrototypes:
    """The lowest-cost of `n_init` runs. `numbers`: (n, p) floats, already scaled; `categories`:
    (n, q) non-negative integer codes (missing values as their own code)."""
    numbers = np.asarray(numbers, dtype=float).reshape(len(numbers), -1)
    categories = np.asarray(categories, dtype=np.int64).reshape(len(numbers), -1)
    if len(numbers) < k:
        raise ValueError(f"{len(numbers)} rows cannot make {k} prototypes.")
    weight = default_gamma(numbers) if gamma is None else gamma
    rng = np.random.default_rng(seed)
    best: KPrototypes | None = None
    for _ in range(max(1, n_init)):
        found = fit_once(numbers, categories, k, weight, rng)
        if best is None or found.cost < best.cost:
            best = found
    assert best is not None
    return best
