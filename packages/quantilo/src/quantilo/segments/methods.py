"""The clustering methods, the scores that choose k, and the stability check (032 § 3 to 5).

Every score is computed on the same prepared space (one-hot categories, robust-scaled numbers),
whatever the method, so runs compare. Slow methods fit on a sample and every other row goes to its
nearest group."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.cluster import hierarchy
from sklearn.cluster import HDBSCAN, KMeans, MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

from quantilo.segments import kprototypes as kp

Method = Literal["kmeans", "kprototypes", "gmm", "ward", "hdbscan"]
METHODS: dict[str, str] = {
    "kmeans": "k-means",
    "kprototypes": "k-prototypes",
    "gmm": "Gaussian mixture",
    "ward": "Hierarchical (Ward)",
    "hdbscan": "HDBSCAN",
}
K_RANGE = range(2, 11)
MINIBATCH_ROWS = 100_000  # k-means becomes MiniBatch k-means beyond
GMM_ROWS = 50_000  # a Gaussian mixture is fitted on at most this many rows
WARD_MAX = 20_000  # Ward is offered up to this many rows…
WARD_ROWS = 5_000  # …and its tree built on at most this many (memory grows with n²)
HDBSCAN_ROWS = 20_000
SILHOUETTE_ROWS = 3_000
SCORE_ROWS = 50_000  # Calinski–Harabasz, Davies–Bouldin and inertia on at most this many rows
SEARCH_ROWS = 50_000  # k = 2 to 10 are tried on at most this many rows; the chosen k on all
STABILITY_RUNS = 10
STABILITY_SHARE = 0.8
STABILITY_ROWS = 50_000  # subsamples are drawn from at most this many rows
STABLE = 0.6
WEAK_SILHOUETTE = 0.25
NOT_STABLE = "These segments are not stable: the data may not have clear groups."


class Cancelled(Exception):
    """The user stopped the work."""


@dataclass
class Space:
    """Prepared rows: one-hot space (x), and the numbers and category codes of k-prototypes."""

    x: np.ndarray
    numbers: np.ndarray
    codes: np.ndarray

    def take(self, index: np.ndarray) -> "Space":
        return Space(self.x[index], self.numbers[index], self.codes[index])

    def __len__(self) -> int:
        return len(self.x)


Predict = Callable[[Space], np.ndarray]


@dataclass
class Fitted:
    labels: np.ndarray  # on the rows it was fitted on (−1: not grouped)
    predict: Predict  # the group of new rows
    bic: float | None = None
    membership: np.ndarray | None = None  # Gaussian mixture: probability of the chosen group


@dataclass
class SegmentScore:
    """How well k groups separate (higher silhouette and Calinski–Harabasz, lower Davies–Bouldin,
    inertia and BIC are better)."""

    k: int
    silhouette: float | None
    calinski_harabasz: float | None
    davies_bouldin: float | None
    inertia: float | None
    bic: float | None = None


@dataclass
class SegmentKChoice:
    k: int
    forced: bool
    why: str
    votes: dict[str, int] = field(default_factory=dict)  # criterion → its best k


@dataclass
class SegmentStability:
    """Mean adjusted Rand index between the segmentation and the same method refitted on 10
    subsamples of 80 % of the rows."""

    ari: float
    runs: list[float]
    rows: int  # rows per subsample
    stable: bool
    warning: str | None


def sample(n: int, limit: int, seed: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))


def _nearest_centre(space: Space, labels: np.ndarray) -> Predict:
    groups = np.unique(labels[labels >= 0])
    centres = np.vstack([space.x[labels == g].mean(axis=0) for g in groups])

    def predict(rows: Space) -> np.ndarray:
        gaps = np.column_stack([((rows.x - centre) ** 2).sum(axis=1) for centre in centres])
        return groups[gaps.argmin(axis=1)]

    return predict


def _kmeans(space: Space, k: int, seed: int) -> Fitted:
    model = (
        MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096)  # type: ignore[arg-type]
        if len(space) > MINIBATCH_ROWS
        else KMeans(n_clusters=k, random_state=seed, n_init=3)  # type: ignore[arg-type]
    )
    labels = np.asarray(model.fit_predict(space.x))
    return Fitted(labels, lambda rows: np.asarray(model.predict(rows.x)))


def _kprototypes(space: Space, k: int, seed: int) -> Fitted:
    model = kp.fit(space.numbers, space.codes, k, seed, n_init=4 if len(space) <= 20_000 else 2)
    return Fitted(model.labels, lambda rows: model.predict(rows.numbers, rows.codes))


def _gmm(space: Space, k: int, seed: int) -> Fitted:
    index = sample(len(space), GMM_ROWS, seed)
    covariance = "full" if space.x.shape[1] <= 20 else "diag"
    model = GaussianMixture(
        n_components=k, covariance_type=covariance, reg_covar=1e-3, random_state=seed
    )
    model.fit(space.x[index])
    chances = model.predict_proba(space.x)
    return Fitted(
        chances.argmax(axis=1),
        lambda rows: np.asarray(model.predict(rows.x)),
        bic=float(model.bic(space.x[index])),
        membership=chances.max(axis=1),
    )


@dataclass
class WardTree:
    index: np.ndarray  # rows the tree was built on
    linkage: np.ndarray


def ward_tree(space: Space, seed: int) -> WardTree:
    index = sample(len(space), WARD_ROWS, seed)
    return WardTree(index, hierarchy.linkage(space.x[index], method="ward"))


def _ward(space: Space, k: int, seed: int, tree: WardTree | None = None) -> Fitted:
    tree = tree or ward_tree(space, seed)
    cut = hierarchy.fcluster(tree.linkage, k, criterion="maxclust") - 1
    built = space.take(tree.index)
    predict = _nearest_centre(built, cut)
    labels = predict(space)
    labels[tree.index] = cut
    return Fitted(labels, predict)


def _hdbscan(space: Space, seed: int) -> Fitted:
    index = sample(len(space), HDBSCAN_ROWS, seed)
    model = HDBSCAN(min_cluster_size=max(10, len(index) // 100), copy=True)  # type: ignore[arg-type]
    found = np.asarray(model.fit_predict(space.x[index]))
    neighbours = KNeighborsClassifier(n_neighbors=1).fit(space.x[index], found)

    def predict(rows: Space) -> np.ndarray:
        return np.asarray(neighbours.predict(rows.x))

    labels = predict(space) if len(index) < len(space) else found.copy()
    labels[index] = found
    return Fitted(labels, predict)


def fit(
    space: Space, method: Method, k: int, seed: int = 42, tree: WardTree | None = None
) -> Fitted:
    if method == "kmeans":
        return _kmeans(space, k, seed)
    if method == "kprototypes":
        return _kprototypes(space, k, seed)
    if method == "gmm":
        return _gmm(space, k, seed)
    if method == "ward":
        return _ward(space, k, seed, tree)
    return _hdbscan(space, seed)


# --- Scores and k --------------------------------------------------------------------------------


def inertia(x: np.ndarray, labels: np.ndarray) -> float:
    """Sum of squared distances of each grouped row to its group's mean."""
    total = 0.0
    for group in np.unique(labels[labels >= 0]):
        members = x[labels == group]
        total += float(((members - members.mean(axis=0)) ** 2).sum())
    return total


def score(
    x: np.ndarray, labels: np.ndarray, k: int, seed: int, bic: float | None = None
) -> SegmentScore:
    if len(x) > SCORE_ROWS:
        index = sample(len(x), SCORE_ROWS, seed)
        x, labels = x[index], labels[index]
    grouped = labels >= 0
    rows, found = x[grouped], labels[grouped]
    groups = len(np.unique(found))
    if groups < 2 or groups >= len(rows):
        return SegmentScore(k, None, None, None, inertia(x, labels), bic)
    try:
        silhouette = float(
            silhouette_score(
                rows, found, sample_size=min(len(rows), SILHOUETTE_ROWS), random_state=seed
            )
        )
    except ValueError:  # the sample drew a single group
        silhouette = None
    return SegmentScore(
        k,
        silhouette,
        float(calinski_harabasz_score(rows, found)),
        float(davies_bouldin_score(rows, found)),
        inertia(x, labels),
        bic,
    )


def elbow(scores: list[SegmentScore]) -> int | None:
    """Where the inertia curve bends most: the k farthest below the chord from its first to its
    last point (both normalised to [0, 1])."""
    found = [(s.k, s.inertia) for s in scores if s.inertia is not None]
    if len(found) < 3:
        return None
    ks = np.array([k for k, _ in found], dtype=float)
    values = np.array([v for _, v in found], dtype=float)
    span = values.max() - values.min()
    if span <= 0:
        return None
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (values - values.min()) / span
    return int(ks[int(np.argmax((1 - x) - y))])


def _best(scores: list[SegmentScore], key: str, lowest: bool = False) -> int | None:
    found = [(s.k, getattr(s, key)) for s in scores if getattr(s, key) is not None]
    if not found:
        return None
    return (min if lowest else max)(found, key=lambda item: item[1])[0]


CRITERIA = {
    "silhouette": "best silhouette",
    "calinski_harabasz": "best Calinski–Harabasz",
    "davies_bouldin": "best Davies–Bouldin",
}


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def choose_k(scores: list[SegmentScore], method: Method, forced: int | None) -> SegmentKChoice:
    """The k the scores agree on (or the one the user set), with the sentence that says why."""
    if method == "gmm":
        best = _best(scores, "bic", lowest=True)
        votes = {"bic": best} if best is not None else {}
        if forced is not None:
            return SegmentKChoice(forced, True, f"{forced} segments: your choice.", votes)
        k = best or scores[0].k
        return SegmentKChoice(k, False, f"{k} segments: the lowest BIC.", votes)
    votes = {
        "silhouette": _best(scores, "silhouette"),
        "calinski_harabasz": _best(scores, "calinski_harabasz"),
        "davies_bouldin": _best(scores, "davies_bouldin", lowest=True),
        "elbow": elbow(scores),
    }
    cast_votes = {key: value for key, value in votes.items() if value is not None}
    if forced is not None:
        return SegmentKChoice(forced, True, f"{forced} segments: your choice.", cast_votes)
    if not cast_votes:
        k = scores[0].k
        return SegmentKChoice(k, False, f"{k} segments: no score could choose.", cast_votes)
    counts: dict[int, int] = {}
    for value in cast_votes.values():
        counts[value] = counts.get(value, 0) + 1
    most = max(counts.values())
    tied = sorted(k for k, c in counts.items() if c == most)
    silhouette_k = cast_votes.get("silhouette")
    k = silhouette_k if silhouette_k in tied else tied[0]
    agreeing = [CRITERIA[key] for key in CRITERIA if cast_votes.get(key) == k]
    elbow_k = cast_votes.get("elbow")
    if agreeing:
        why = f"{k} segments: {_join(agreeing)}"
        if elbow_k == k:
            why += f", and the elbow is at {k}"
        elif elbow_k is not None:
            why += f" (the elbow is at {elbow_k})"
    else:
        why = f"{k} segments: the elbow is at {k}"
        if silhouette_k is not None:
            why += f" (best silhouette at {silhouette_k})"
    return SegmentKChoice(k, False, why + ".", cast_votes)


# --- Stability -----------------------------------------------------------------------------------


def stability(
    space: Space,
    method: Method,
    k: int,
    reference: np.ndarray,
    seed: int = 42,
    should_stop: Callable[[], bool] = lambda: False,
    progress: Callable[[int], None] = lambda _done: None,
) -> SegmentStability:
    """The same method refitted on STABILITY_RUNS subsamples; each compared with the
    segmentation on the rows they share (adjusted Rand index)."""
    rng = np.random.default_rng(seed)
    pool = sample(len(space), STABILITY_ROWS, seed)
    size = max(k + 1, round(STABILITY_SHARE * len(pool)))
    runs = []
    for i in range(STABILITY_RUNS):
        if should_stop():
            raise Cancelled
        index = np.sort(rng.choice(pool, size, replace=False))
        refit = fit(space.take(index), method, k, seed + i + 1)
        runs.append(float(adjusted_rand_score(reference[index], refit.labels)))
        progress(i + 1)
    mean = float(np.mean(runs))
    stable = mean >= STABLE
    return SegmentStability(mean, runs, size, stable, None if stable else NOT_STABLE)


def relabel(labels: np.ndarray) -> dict[int, int]:
    """Groups numbered by size, the largest first; "not grouped" (−1) stays −1."""
    groups, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = sorted(zip(groups.tolist(), counts.tolist(), strict=True), key=lambda g: (-g[1], g[0]))
    mapping = {int(g): i for i, (g, _c) in enumerate(order)}
    mapping[-1] = -1
    return mapping


def apply(mapping: dict[int, int], labels: np.ndarray) -> np.ndarray:
    table = np.full(max(mapping) + 2, -1, dtype=np.int64)  # shifted by one: −1 → index 0
    for old, new in mapping.items():
        table[old + 1] = new
    return table[np.asarray(labels, dtype=np.int64) + 1]
