"""Explanation by cases (117; study 2026-11-11 §§ 6.4, 8, 13.4): the training examples nearest
to a prediction, the prototypes of each class with their criticisms, and the counterfactual
example. Shared by vision and, later, the tabular ML (126).

Faithful by construction: every comparison is made in the space the model itself reads (the
frozen network's vectors for a probe, the PCA components for a classic model), so what is shown
is the model's own evidence, not a story told about it afterwards (Rudin, 2019).

numpy only (no thread, no process): it runs in the browser version too.

- R-VEX-01: nearest examples = the k training rows with the largest cosine similarity to the
  query in the model's space (k = 5), with their labels; the sentence says how many share the
  prediction ("4 of the 5 nearest training images are labelled healthy").
- R-VEX-02: prototypes = k-medoids of each class (3 per class, cosine distance, k-medoids++
  start then alternating updates, seed 0; at most 2,000 rows per class, drawn with the seed);
  each medoid comes with the share of its class it stands for.
- R-VEX-03: criticisms = the rows the prototypes represent worst, by the MMD witness function
  (Kim, Khanna, Koyejo, 2016, "Examples are not enough, learn to criticize!"): RBF kernel on
  the normalised vectors, gamma = 1 / mean squared distance between the class's rows (the median
  heuristic let a tight majority hide a small uncovered mode: tests); the 3 rows of largest positive
  witness that are not prototypes: the images of a part of the class the prototypes leave
  uncovered (an isolated outlier has a witness near 0, it is not a criticism).
  `mmd_prototypes` gives MMD-critic's own greedy prototypes.
- R-VEX-04: counterfactual example = the nearest training row of another class (the most
  probable other class when the probabilities are given), in the same space.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

SEED = 0
NEAREST = 5  # R-VEX-01
PER_CLASS = 3  # R-VEX-02
CRITICISMS = 3  # R-VEX-03
MAX_PER_CLASS = 2_000
MEDOID_ITERATIONS = 50


@dataclass
class Neighbour:
    index: int  # row of the bank
    similarity: float  # cosine, in [-1, 1] (euclidean: minus the distance)
    label: str | None = None
    distance: float | None = None  # the model's own distance, when it reads one


@dataclass
class NearestExamples:
    neighbours: list[Neighbour]
    agreeing: int  # neighbours whose label is `label`
    label: str | None
    sentence: str


@dataclass
class ClassPrototypes:
    label: str
    size: int  # rows of the class (before any draw)
    prototypes: list[int]  # rows of the bank, most representative first
    shares: list[float]  # share of the class nearest to each prototype
    criticisms: list[int]  # rows the prototypes represent worst
    witness: list[float]  # their witness value (> 0: under-represented by the prototypes)
    notes: list[str] = field(default_factory=list)


def normalise(x: np.ndarray) -> np.ndarray:
    """Rows divided by their Euclidean norm (a zero row stays zero)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def cosine(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """`(m, n)` cosine similarities between the rows of `query` and those of `bank`."""
    return normalise(query) @ normalise(bank).T


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def nearest_examples(
    query: np.ndarray,
    bank: np.ndarray,
    labels: Sequence[str | None] | None = None,
    *,
    k: int = NEAREST,
    label: str | None = None,
    exclude: Sequence[int] = (),
    noun: str = "training image",
    metric: str = "cosine",
) -> NearestExamples:
    """R-VEX-01: the `k` rows of `bank` most similar to `query` (cosine; or nearest by
    Euclidean distance with `metric="euclidean"`, when that is what the model reads), most
    similar first. `label` is the prediction the sentence compares the neighbours' labels with;
    `exclude` rows (the query itself, its copies) are never returned."""
    if metric == "euclidean":
        q = np.asarray(query, dtype=np.float64).reshape(1, -1)
        diff = np.asarray(bank, dtype=np.float64) - q
        sims = -np.sqrt(np.einsum("ij,ij->i", diff, diff))
    elif metric == "cosine":
        sims = cosine(np.asarray(query), np.asarray(bank))[0]
    else:
        raise ValueError(f"Unknown metric {metric!r}: cosine or euclidean.")
    sims[~np.isfinite(sims)] = -np.inf
    if len(exclude):
        sims[np.asarray(list(exclude), dtype=int)] = -np.inf
    k = int(min(k, int(np.isfinite(sims).sum())))
    if k <= 0:
        return NearestExamples([], 0, label, f"No {noun} to compare with.")
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top], kind="mergesort")]
    found = [
        Neighbour(
            int(i),
            float(sims[i]),
            None if labels is None else _text(labels[int(i)]),
            float(-sims[i]) if metric == "euclidean" else None,
        )
        for i in top
    ]
    agreeing = sum(1 for n in found if label is not None and n.label == label)
    if metric == "euclidean":
        span = f"distance {-found[0].similarity:.3g} to {-found[-1].similarity:.3g}"
    else:
        span = f"similarity {found[0].similarity:.2f} to {found[-1].similarity:.2f}"
    if label is None or labels is None:
        sentence = f"The {_plural(k, noun)} the model finds most similar ({span}, R-VEX-01)."
    else:
        others = Counter(n.label for n in found if n.label != label and n.label is not None)
        rest = ", ".join(f"{name} {count}" for name, count in others.most_common())
        sentence = (
            f"{agreeing} of the {_plural(k, f'nearest {noun}')} are labelled {label}"
            + (f"; the others: {rest}" if rest else "")
            + f" ({span}, R-VEX-01)."
        )
    return NearestExamples(found, agreeing, label, sentence)


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _distances(x: np.ndarray) -> np.ndarray:
    """Cosine distances (1 - similarity) between the rows of `x`, clipped at 0."""
    z = normalise(x)
    return np.clip(1.0 - z @ z.T, 0.0, 2.0)


def kmedoids(x: np.ndarray, k: int, *, seed: int = SEED) -> tuple[list[int], np.ndarray]:
    """R-VEX-02: `k` medoids of the rows of `x` (cosine distance): k-medoids++ start, then each
    cluster's medoid is moved to the row of least total distance until nothing moves. Returns the
    medoids, ordered by cluster size (largest first), and each row's cluster in that order."""
    n = len(x)
    if n == 0:
        return [], np.zeros(0, dtype=int)
    k = int(max(1, min(k, n)))
    d = _distances(x)
    rng = np.random.default_rng(seed)
    medoids = [int(np.argmin(d.sum(axis=1)))]  # the most central row first: stable
    while len(medoids) < k:
        nearest = d[:, medoids].min(axis=1)
        weights = nearest**2
        total = weights.sum()
        if total <= 0:
            rest = [i for i in range(n) if i not in medoids]
            medoids.append(int(rng.choice(rest)))
            continue
        medoids.append(int(rng.choice(n, p=weights / total)))
    assign = np.argmin(d[:, medoids], axis=1)
    for _ in range(MEDOID_ITERATIONS):
        moved = False
        for c in range(k):
            members = np.flatnonzero(assign == c)
            if len(members) == 0:
                continue
            best = int(members[np.argmin(d[np.ix_(members, members)].sum(axis=1))])
            if best != medoids[c]:
                medoids[c], moved = best, True
        new = np.argmin(d[:, medoids], axis=1)
        if not moved and np.array_equal(new, assign):
            break
        assign = new
    sizes = np.bincount(assign, minlength=k)
    order = np.argsort(-sizes, kind="mergesort")
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(k)
    return [medoids[c] for c in order], rank[assign]


def mean_gamma(x: np.ndarray) -> float:
    """R-VEX-03: RBF bandwidth, gamma = 1 / mean of the squared distances between the rows."""
    z = normalise(x)
    sq = np.clip(2.0 - 2.0 * (z @ z.T), 0.0, None)
    upper = sq[np.triu_indices(len(z), 1)]
    mean = float(np.mean(upper)) if len(upper) else 1.0
    return 1.0 / max(mean, 1e-12)


def _kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    a, b = normalise(x), normalise(y)
    sq = np.clip(2.0 - 2.0 * (a @ b.T), 0.0, None)
    return np.exp(-gamma * sq)


def witness(x: np.ndarray, prototypes: Sequence[int], gamma: float | None = None) -> np.ndarray:
    """R-VEX-03: MMD witness of every row, mean kernel to the data minus mean kernel to the
    prototypes; > 0 where the prototypes under-represent the data."""
    gamma = mean_gamma(x) if gamma is None else gamma
    k = _kernel(x, x, gamma)
    return k.mean(axis=1) - k[:, list(prototypes)].mean(axis=1)


def criticisms(
    x: np.ndarray, prototypes: Sequence[int], m: int = CRITICISMS, gamma: float | None = None
) -> tuple[list[int], list[float]]:
    """R-VEX-03: the `m` rows of largest positive witness that are not prototypes."""
    w = witness(x, prototypes, gamma)
    order = np.argsort(-w, kind="mergesort")
    taken = set(prototypes)
    chosen = [int(i) for i in order if int(i) not in taken and w[i] > 0][:m]
    return chosen, [float(w[i]) for i in chosen]


def mmd_prototypes(x: np.ndarray, m: int = PER_CLASS, gamma: float | None = None) -> list[int]:
    """MMD-critic's greedy prototypes (Kim et al., 2016): each step adds the row that most
    lowers the MMD between the data and the prototypes."""
    n = len(x)
    m = int(max(0, min(m, n)))
    gamma = mean_gamma(x) if gamma is None else gamma
    k = _kernel(x, x, gamma)
    column_mean = k.mean(axis=0)
    chosen: list[int] = []
    for _ in range(m):
        best, best_value = -1, -np.inf
        for c in range(n):
            if c in chosen:
                continue
            s = [*chosen, c]
            value = 2.0 * column_mean[s].sum() / len(s) - k[np.ix_(s, s)].sum() / len(s) ** 2
            if value > best_value:
                best, best_value = c, value
        chosen.append(best)
    return chosen


def class_prototypes(
    x: np.ndarray,
    labels: Sequence[str | None],
    *,
    per_class: int = PER_CLASS,
    critics: int = CRITICISMS,
    seed: int = SEED,
    max_per_class: int = MAX_PER_CLASS,
) -> list[ClassPrototypes]:
    """R-VEX-02 and R-VEX-03 for every class of `labels` (rows of `x` with a NaN or no label are
    left out). Indices are rows of `x`."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x).all(axis=1)
    text = np.array([_text(v) for v in labels], dtype=object)
    rng = np.random.default_rng(seed)
    out = []
    for label in sorted({v for v in text if v is not None}):
        rows = np.flatnonzero((text == label) & finite)
        if len(rows) == 0:
            continue
        notes = []
        size = len(rows)
        if size > max_per_class:
            rows = np.sort(rng.choice(rows, max_per_class, replace=False))
            notes.append(
                f"Prototypes of {label} drawn from {max_per_class:,} of its {size:,} images "
                "(seed 0)."
            )
        sub = x[rows]
        medoids, assign = kmedoids(sub, per_class, seed=seed)
        shares = [float(np.mean(assign == c)) for c in range(len(medoids))]
        if len(rows) > len(medoids):
            found, values = criticisms(sub, medoids, critics)
        else:
            found, values = [], []
        out.append(
            ClassPrototypes(
                label=label,
                size=size,
                prototypes=[int(rows[i]) for i in medoids],
                shares=shares,
                criticisms=[int(rows[i]) for i in found],
                witness=values,
                notes=notes,
            )
        )
    return out


def counterfactual_example(
    query: np.ndarray,
    bank: np.ndarray,
    labels: Sequence[str | None],
    own: str | None,
    *,
    target: str | None = None,
    exclude: Sequence[int] = (),
) -> Neighbour | None:
    """R-VEX-04: the row of `bank` nearest to `query` among those labelled `target` (or, without
    one, labelled anything but `own`). None when no such row exists."""
    text = np.array([_text(v) for v in labels], dtype=object)
    if target is not None:
        allowed = text == target
    else:
        allowed = np.array([v is not None and v != own for v in text], dtype=bool)
    if len(exclude):
        allowed[np.asarray(list(exclude), dtype=int)] = False
    if not allowed.any():
        return None
    sims = cosine(np.asarray(query), np.asarray(bank))[0]
    sims[~allowed | ~np.isfinite(sims)] = -np.inf
    best = int(np.argmax(sims))
    if not np.isfinite(sims[best]):
        return None
    return Neighbour(best, float(sims[best]), _text(text[best]))


__all__ = [
    "ClassPrototypes",
    "NearestExamples",
    "Neighbour",
    "class_prototypes",
    "cosine",
    "counterfactual_example",
    "criticisms",
    "kmedoids",
    "mean_gamma",
    "mmd_prototypes",
    "nearest_examples",
    "normalise",
    "witness",
]
