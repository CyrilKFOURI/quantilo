"""The map of the segments (032 § 7): the prepared rows in 2D (PCA, or t-SNE on demand), coloured
by segment, with each segment's centre; and Ward's tree, cut where the segments are. Under More
(122), PaCMAP or TriMAP when the optional `interpret` extra is there (R-XAI-10)."""

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

MAP_POINTS = 5_000  # points drawn (and t-SNE's rows)
DENDROGRAM_LEAVES = 30


@dataclass
class SegmentMapCentre:
    segment: int
    x: float
    y: float


@dataclass
class SegmentMap:
    projection: Literal["pca", "tsne", "pacmap", "trimap"]
    x: list[float]
    y: list[float]
    segments: list[int]  # segment of each point
    rows: list[int]  # row number of each point in the file
    centres: list[SegmentMapCentre]
    explained: list[float] | None  # PCA: share of the variance each axis explains
    points: int  # points drawn
    of: int  # rows they were drawn from


@dataclass
class SegmentDendrogram:
    """Ward's tree, its last merges only (scipy's dendrogram coordinates)."""

    icoord: list[list[float]]
    dcoord: list[list[float]]
    leaves: list[str]  # "(120)": a branch of 120 rows
    cut: float  # height where the tree splits into the segments
    rows: int  # rows the tree was built on


def sample(n: int, seed: int, limit: int = MAP_POINTS) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))


def _rounded(values: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in values]


def _centres(coordinates: np.ndarray, labels: np.ndarray) -> list[SegmentMapCentre]:
    return [
        SegmentMapCentre(
            int(g),
            round(float(coordinates[labels == g, 0].mean()), 4),
            round(float(coordinates[labels == g, 1].mean()), 4),
        )
        for g in np.unique(labels)
    ]


def pca_map(x: np.ndarray, labels: np.ndarray, rows: np.ndarray, seed: int) -> SegmentMap:
    """Every fit row projected (for the centres); MAP_POINTS of them drawn."""
    components = min(2, x.shape[1], len(x))
    pca = PCA(n_components=components, random_state=seed)
    coordinates = pca.fit_transform(x)
    if components < 2:
        coordinates = np.column_stack([coordinates, np.zeros(len(x))])
    explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    drawn = sample(len(x), seed)
    return SegmentMap(
        "pca",
        _rounded(coordinates[drawn, 0]),
        _rounded(coordinates[drawn, 1]),
        [int(v) for v in labels[drawn]],
        [int(v) for v in rows[drawn]],
        _centres(coordinates, labels),
        explained,
        len(drawn),
        len(x),
    )


def tsne_map(x: np.ndarray, labels: np.ndarray, rows: np.ndarray, of: int, seed: int) -> SegmentMap:
    """t-SNE of the drawn rows (at most MAP_POINTS): neighbours stay neighbours, distances
    between groups do not mean much."""
    perplexity = float(min(30, max(2, (len(x) - 1) // 3)))
    coordinates = TSNE(
        n_components=2, perplexity=perplexity, init="pca", random_state=seed
    ).fit_transform(x)
    return SegmentMap(
        "tsne",
        _rounded(coordinates[:, 0]),
        _rounded(coordinates[:, 1]),
        [int(v) for v in labels],
        [int(v) for v in rows],
        _centres(coordinates, labels),
        None,
        len(x),
        of,
    )


def extra_map(
    x: np.ndarray, labels: np.ndarray, rows: np.ndarray, of: int, seed: int, method: str
) -> SegmentMap:
    """PaCMAP or TriMAP of the drawn rows (122): local and global structure kept together."""
    from quantilo.interpret import maps

    coordinates = maps.embed(x, cast(maps.Method, method), seed)
    return SegmentMap(
        cast(Literal["pacmap", "trimap"], method),
        _rounded(coordinates[:, 0]),
        _rounded(coordinates[:, 1]),
        [int(v) for v in labels],
        [int(v) for v in rows],
        _centres(coordinates, labels),
        None,
        len(x),
        of,
    )


def dendrogram(linkage: np.ndarray, k: int) -> SegmentDendrogram:
    found = hierarchy.dendrogram(
        linkage, truncate_mode="lastp", p=DENDROGRAM_LEAVES, no_plot=True, show_contracted=False
    )
    heights = linkage[:, 2]
    # the cut lies between the merge that leaves k groups and the next one up
    n = len(heights)
    low = float(heights[n - k]) if 0 < k <= n else 0.0
    high = float(heights[n - k + 1]) if 0 < k - 1 < n else low
    return SegmentDendrogram(
        [[round(float(v), 4) for v in line] for line in found["icoord"]],
        [[round(float(v), 4) for v in line] for line in found["dcoord"]],
        [str(label) for label in found["ivl"]],
        round((low + high) / 2, 4),
        n + 1,
    )
