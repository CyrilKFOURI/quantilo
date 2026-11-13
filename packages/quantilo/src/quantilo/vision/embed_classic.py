"""The map of an image set (109; study 2026-11-11 vision-et-xai § 4.4).

Every image becomes a point of a plane where images that look alike are close:

1. **Space** (R-VMAP-01, R-VMAP-02): the classic descriptor of 103 (HSV histogram 72 ‖ HOG 1,764 ‖
   uniform LBP 236) of each image, each column standardised, each family weighted by
   1/sqrt(its columns) so that HOG does not drown colour and texture, then a PCA to 50
   dimensions. A deep embedding (114) goes through the same standardisation and PCA.
2. **Map** (R-VMAP-03): the first two components of that PCA (their share of the variance is
   said); t-SNE on demand, with the code of Segments (032).
3. **Groups** (R-VMAP-04): k-means on the 50 components, k from 2 to 12 by the best silhouette,
   searched on 5,000 images at most, then every image goes to its nearest centre (the methods and
   scores of Segments); HDBSCAN on demand.
4. **Reading** (R-VMAP-05): adjusted Rand index between groups and labels, and the share of
   images whose 10 nearest neighbours mostly share their label, against chance.

numpy, scipy and scikit-learn only: the same numbers in Pyodide."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image

from quantilo.segments import methods
from quantilo.segments.projection import sample, tsne_map
from quantilo.vision.decode import DecodedImage, DecodeFailure, try_decode
from quantilo.vision.descriptors import (
    CLASSIC_SIZE,
    HOG_SIZE,
    HSV_SIZE,
    LBP_SIZE,
    classic_descriptor,
)
from quantilo.vision.measures import MEASURE_SIDE, working_copy

if TYPE_CHECKING:
    from quantilo.vision.ingest import ImageSet

SPACE_DIMS = 50
MAP_POINTS = 5_000  # R-VMAP-06
K_RANGE = range(2, 13)  # R-VMAP-04
SEARCH_ROWS = 5_000
NEIGHBOURS = 10
MATCH_ARI = 0.5
SEPARATE_MARGIN = 0.25
FAMILIES = (("hsv", HSV_SIZE), ("hog", HOG_SIZE), ("lbp", LBP_SIZE))
CLASSIC_TITLE = "Classic descriptors (colour HSV, HOG, LBP)"

Projection = Literal["pca", "tsne", "pacmap", "trimap"]  # c3, R-VMAP-12: the last two in More
Clustering = Literal["kmeans", "hdbscan"]
Source = str | Path | bytes | Image.Image | DecodedImage
Progress = Callable[[int, int], None]


class Cancelled(RuntimeError):
    """The user stopped the build."""


# --- descriptors ---------------------------------------------------------------------------------


def _image(source: Source) -> Image.Image | None:
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, DecodedImage):
        return source.image
    decoded = try_decode(source, reduce_to=MEASURE_SIDE)
    return None if isinstance(decoded, DecodeFailure) else decoded.image


def classic_vectors(
    sources: Sequence[Source | None],
    *,
    progress: Progress | None = None,
    cancel: Callable[[], bool] | None = None,
) -> np.ndarray:
    """(n, 2,072) float32: the classic descriptor of each image, on its 512-px working copy
    (R-VMAP-01); a row of NaN for an image that is None or cannot be decoded."""
    n = len(sources)
    out = np.full((n, CLASSIC_SIZE), np.nan, dtype=np.float32)
    for i, source in enumerate(sources):
        if cancel is not None and cancel():
            raise Cancelled("Map cancelled.")
        image = None if source is None else _image(source)
        if image is not None:
            out[i] = classic_descriptor(working_copy(image)).vector
        if progress is not None and (i % 20 == 0 or i + 1 == n):
            progress(i + 1, n)
    return out


def image_set_sources(images: ImageSet) -> list[Path | None]:
    """The stored file of each row of an imported set (None when it was not decoded at import)."""
    table = images.table
    stored = table["stored_path"].to_pylist()
    decoded = table["decode_ok"].to_pylist()
    return [
        images.store.path(stored[i]) if decoded[i] and stored[i] is not None else None
        for i in range(table.num_rows)
    ]


# --- space ---------------------------------------------------------------------------------------


@dataclass
class Space:
    """The 50 components of each image (NaN rows: no vector) and how they were made."""

    vectors: np.ndarray  # (n, d) float32, d <= 50
    explained: list[float]  # share of the standardised variance of each component
    method: str  # "classic", "dinov2_s14"...
    title: str
    source_dims: int

    @property
    def ok(self) -> np.ndarray:
        return np.isfinite(self.vectors).all(axis=1)

    @property
    def dims(self) -> int:
        return int(self.vectors.shape[1])


def _weights(dims: int, families: Sequence[tuple[str, int]] | None) -> np.ndarray:
    """R-VMAP-02: every column of a family weighs 1/sqrt(its size), so each family brings the same
    total variance after standardisation. Without families (a deep vector), every column is 1."""
    if not families or sum(size for _, size in families) != dims:
        return np.ones(dims)
    return np.concatenate([np.full(size, 1.0 / np.sqrt(size)) for _, size in families])


def reduce(
    vectors: np.ndarray,
    *,
    method: str = "classic",
    title: str = CLASSIC_TITLE,
    families: Sequence[tuple[str, int]] | None = FAMILIES,
    dims: int = SPACE_DIMS,
    seed: int = 0,
) -> Space:
    """Standardise each column, weigh the families, keep `dims` principal components."""
    from sklearn.decomposition import PCA

    raw = np.asarray(vectors, dtype=np.float64)
    ok = np.isfinite(raw).all(axis=1)
    n_ok = int(ok.sum())
    width = min(dims, max(1, n_ok - 1), raw.shape[1])
    if n_ok < 3:
        empty = np.full((len(raw), width), np.nan, dtype=np.float32)
        return Space(empty, [0.0] * width, method, title, raw.shape[1])
    data = raw[ok]
    spread = data.std(axis=0)
    kept = spread > 1e-9  # a constant column says nothing
    z = (data[:, kept] - data[:, kept].mean(axis=0)) / spread[kept]
    z *= _weights(raw.shape[1], families)[kept]
    pca = PCA(n_components=min(width, z.shape[1]), random_state=seed)
    scores = pca.fit_transform(z)
    out = np.full((len(raw), scores.shape[1]), np.nan, dtype=np.float32)
    out[ok] = scores.astype(np.float32)
    explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    return Space(out, explained, method, title, raw.shape[1])


def classic_space(
    sources: Sequence[Source | None],
    *,
    progress: Progress | None = None,
    cancel: Callable[[], bool] | None = None,
    seed: int = 0,
) -> Space:
    """Descriptors then `reduce` (R-VMAP-01, R-VMAP-02)."""
    return reduce(classic_vectors(sources, progress=progress, cancel=cancel), seed=seed)


def save_space(path: Path, keys: Sequence[str], space: Space) -> None:
    """A Parquet table: `rel_path` + `vector` (FixedSizeList<float32>, decision 090); the
    description of the space in the schema's metadata."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    flat = pa.array(space.vectors.ravel(), type=pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(flat, space.dims)
    ok = pa.array(space.ok.tolist(), type=pa.bool_())
    table = pa.table(
        {"rel_path": pa.array(list(keys), type=pa.string()), "vector": vectors, "ok": ok}
    )
    meta = {
        "explained": space.explained,
        "method": space.method,
        "title": space.title,
        "source_dims": space.source_dims,
    }
    table = table.replace_schema_metadata({"quantilo.space": json.dumps(meta)})
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    pq.write_table(table, partial, compression="zstd")
    partial.replace(path)


def load_space(path: Path, keys: Sequence[str]) -> Space:
    """The saved space, aligned on `keys` (an image no longer in the table is dropped, a new one
    has no vector)."""
    import json

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    meta = json.loads((table.schema.metadata or {})[b"quantilo.space"])
    dims = table.schema.field("vector").type.list_size
    flat = table["vector"].combine_chunks().flatten().to_numpy(zero_copy_only=False)
    stored = np.asarray(flat, dtype=np.float32).reshape(-1, dims)
    ok = np.asarray(table["ok"].to_pylist(), dtype=bool)
    position = {str(k): i for i, k in enumerate(table["rel_path"].to_pylist())}
    vectors = np.full((len(keys), dims), np.nan, dtype=np.float32)
    for row, key in enumerate(keys):
        at = position.get(key)
        if at is not None and ok[at]:
            vectors[row] = stored[at]
    return Space(
        vectors, list(meta["explained"]), meta["method"], meta["title"], meta["source_dims"]
    )


# --- map -----------------------------------------------------------------------------------------


@dataclass
class KScore:
    k: int
    silhouette: float | None


@dataclass
class LabelReading:
    """How well the labels separate in this space (R-VMAP-05)."""

    ari: float | None  # groups vs labels
    neighbours_same: float | None  # share of images whose 10 neighbours mostly share their label
    chance: float | None  # the same share expected by chance (sum of the squared label shares)
    labels: int


@dataclass
class ImageMap:
    method: str
    title: str
    projection: Projection
    clustering: Clustering
    rows: list[int]  # table row of each drawn point
    x: list[float]
    y: list[float]
    clusters: list[int]  # group of EVERY row of the table (-1: no vector, or noise for HDBSCAN)
    k: int
    k_forced: bool
    scores: list[KScore]
    silhouette: float | None
    explained: list[float] | None  # the two axes (PCA)
    reading: LabelReading | None
    sentence: str
    notes: list[str] = field(default_factory=list)
    points: int = 0
    of: int = 0  # images with a vector


def _rounded(values: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in values]


def _groups(
    x: np.ndarray, clustering: Clustering, k: int | None, seed: int
) -> tuple[np.ndarray, int, bool, list[KScore], float | None, str | None]:
    """Group of each row of `x` (R-VMAP-04), with the k search."""
    n = len(x)
    empty = np.empty((n, 0))
    space = methods.Space(x, empty, empty)
    if clustering == "hdbscan":
        fitted = methods.fit(space, "hdbscan", 0, seed)
        labels = np.asarray(fitted.labels)
        found = len(np.unique(labels[labels >= 0]))
        scored = methods.score(x, labels, found, seed)
        noise = int((labels < 0).sum())
        note = f"HDBSCAN found {found} groups; {noise:,} images belong to none." if noise else None
        return labels, found, False, [], scored.silhouette, note
    search = sample(n, seed, SEARCH_ROWS)
    probe = space.take(search)
    scores: list[KScore] = []
    best_k, best_silhouette = 2, -2.0
    upper = min(max(K_RANGE), max(2, len(search) - 1))
    if k is None:
        for candidate in range(min(K_RANGE), upper + 1):
            fitted = methods.fit(probe, "kmeans", candidate, seed)
            silhouette = methods.score(probe.x, fitted.labels, candidate, seed).silhouette
            scores.append(KScore(candidate, None if silhouette is None else round(silhouette, 4)))
            if silhouette is not None and silhouette > best_silhouette:
                best_k, best_silhouette = candidate, silhouette
    chosen = k if k is not None else best_k
    chosen = max(2, min(chosen, n - 1))
    fitted = methods.fit(probe, "kmeans", chosen, seed)
    labels = np.asarray(fitted.predict(space))
    silhouette = methods.score(x, labels, chosen, seed).silhouette
    note = None if len(search) == n else f"k searched on {len(search):,} of {n:,} images."
    return labels, chosen, k is not None, scores, silhouette, note


def label_reading(
    x: np.ndarray, labels: Sequence[str | None], clusters: np.ndarray, seed: int = 0
) -> LabelReading | None:
    """R-VMAP-05 on the rows that have a label (at least 2 labels, each of 2 images or more)."""
    from sklearn.metrics import adjusted_rand_score
    from sklearn.neighbors import NearestNeighbors

    given = np.array([v is not None and str(v) != "" for v in labels])
    names = np.array([str(v) if v is not None else "" for v in labels])
    if given.sum() < 4 or len(set(names[given])) < 2:
        return None
    rows = sample(int(given.sum()), seed, SEARCH_ROWS)
    index = np.flatnonzero(given)[rows]
    xs, ys = x[index], names[index]
    grouped = clusters[index] >= 0
    ari = (
        round(float(adjusted_rand_score(ys[grouped], clusters[index][grouped])), 4)
        if grouped.sum() >= 2
        else None
    )
    k = min(NEIGHBOURS, len(xs) - 1)
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(xs).kneighbors(xs, return_distance=False)
    same = (ys[neighbours[:, 1:]] == ys[:, None]).mean(axis=1)
    _, counts = np.unique(ys, return_counts=True)
    shares = counts / counts.sum()
    return LabelReading(
        ari,
        round(float((same > 0.5).mean()), 4),
        round(float((shares**2).sum()), 4),
        len(counts),
    )


def sentence_of(
    reading: LabelReading | None, k: int, silhouette: float | None, clustering: Clustering
) -> str:
    """R-VMAP-05: what the map shows, in one sentence."""
    groups = f"{k} groups" + (f" (silhouette {silhouette:.2f})" if silhouette is not None else "")
    if reading is None:
        how = "HDBSCAN" if clustering == "hdbscan" else "k-means, k chosen by silhouette"
        return f"No labels to compare with: {groups} found by {how}."
    if reading.ari is not None and reading.ari >= MATCH_ARI:
        return f"Clusters match labels: adjusted Rand {reading.ari:.2f}."
    same, chance = reading.neighbours_same or 0.0, reading.chance or 0.0
    rand = "" if reading.ari is None else f" (adjusted Rand {reading.ari:.2f})"
    if same >= chance + SEPARATE_MARGIN:
        return (
            f"Labels separate in this view: {same:.0%} of images have mostly same-label "
            f"neighbours (chance: {chance:.0%}), though the {groups} do not follow them{rand}."
        )
    return (
        "Labels overlap in this view: a linear model on these features may struggle "
        f"({same:.0%} of images have mostly same-label neighbours; chance: {chance:.0%})."
    )


EXTRA_NOTES = {
    "pacmap": "PaCMAP (Wang, Huang, Rudin, Shaposhnik, 2021) keeps both the neighbours and the "
    "layout of the groups better than t-SNE; axes have no unit (R-VMAP-12).",
    "trimap": "TriMAP (Amid, Warmuth, 2019) keeps the global layout through triplets of images; "
    "axes have no unit (R-VMAP-12).",
}


def image_map(
    space: Space,
    labels: Sequence[str | None] | None = None,
    *,
    projection: Projection = "pca",
    clustering: Clustering = "kmeans",
    k: int | None = None,
    seed: int = 0,
    max_points: int = MAP_POINTS,
) -> ImageMap:
    """The map of a space (R-VMAP-03 to R-VMAP-06). `labels`: one per row of the table."""
    ok = space.ok
    present = np.flatnonzero(ok)
    n_all = len(space.vectors)
    clusters_all = np.full(n_all, -1, dtype=np.int64)
    notes: list[str] = []
    if len(present) < 3:
        return ImageMap(
            space.method, space.title, projection, clustering, [], [], [], clusters_all.tolist(),
            0, False, [], None, None, None, "Too few images to draw a map.", notes, 0,
            len(present),
        )  # fmt: skip
    x = space.vectors[present].astype(np.float64)
    groups, chosen, forced, scores, silhouette, note = _groups(x, clustering, k, seed)
    if note:
        notes.append(note)
    if silhouette is not None and silhouette < methods.WEAK_SILHOUETTE:
        notes.append(
            f"The groups overlap (silhouette {silhouette:.2f}, below {methods.WEAK_SILHOUETTE}): "
            "read them as regions of the map, not as classes."
        )
    clusters_all[present] = groups
    drawn = sample(len(present), seed, max_points)
    if len(drawn) < len(present):
        notes.append(
            f"{len(drawn):,} of {len(present):,} images drawn (fixed seed); groups cover them all."
        )
    explained: list[float] | None
    if projection == "tsne":
        found = tsne_map(x[drawn], groups[drawn], present[drawn], len(present), seed)
        xs, ys = found.x, found.y
        explained = None
        notes.append("t-SNE keeps neighbours together; distances between groups mean little.")
    elif projection in ("pacmap", "trimap"):
        from quantilo.interpret import maps as extra_maps

        # R-VMAP-12: the optional extra `interpret` (numba), on the points drawn only
        xy = extra_maps.embed(x[drawn], projection, seed=seed)
        xs, ys = _rounded(xy[:, 0]), _rounded(xy[:, 1])
        explained = None
        notes.append(EXTRA_NOTES[projection])
    else:
        second = x[:, 1] if x.shape[1] > 1 else np.zeros(len(x))
        xs, ys = _rounded(x[drawn, 0]), _rounded(second[drawn])
        explained = space.explained[:2]
    if len(present) < n_all:
        notes.append(f"{n_all - len(present):,} images have no vector (not decoded).")
    reading = (
        label_reading(x, [labels[int(i)] for i in present], groups, seed)
        if labels is not None
        else None
    )
    return ImageMap(
        method=space.method,
        title=space.title,
        projection=projection,
        clustering=clustering,
        rows=[int(v) for v in present[drawn]],
        x=xs,
        y=ys,
        clusters=[int(v) for v in clusters_all],
        k=chosen,
        k_forced=forced,
        scores=scores,
        silhouette=None if silhouette is None else round(silhouette, 4),
        explained=explained,
        reading=reading,
        sentence=sentence_of(reading, chosen, silhouette, clustering),
        notes=notes,
        points=len(drawn),
        of=len(present),
    )


def image_set_map(images: ImageSet, **options: object) -> ImageMap:
    """Classic space and map of an imported set (library use: `dx.image_set(...)`)."""
    space = classic_space(image_set_sources(images))
    labels = images.table["label"].to_pylist()
    return image_map(space, labels, **options)  # type: ignore[arg-type]


__all__ = [
    "CLASSIC_TITLE",
    "MAP_POINTS",
    "SPACE_DIMS",
    "Cancelled",
    "ImageMap",
    "KScore",
    "LabelReading",
    "Space",
    "classic_space",
    "classic_vectors",
    "image_map",
    "image_set_map",
    "image_set_sources",
    "label_reading",
    "load_space",
    "reduce",
    "save_space",
    "sentence_of",
]
