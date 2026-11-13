"""109: the classic map of an image set (embed_classic)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from quantilo.vision import embed_classic as ec
from quantilo.vision.descriptors import CLASSIC_SIZE
from tests.vision_fixtures import checker, encode, flat, noise


def two_kinds(n: int = 12) -> tuple[list[Image.Image], list[str]]:
    """Red smooth scenes against grey checkerboards: two kinds any descriptor separates."""
    images: list[Image.Image] = []
    labels: list[str] = []
    rng = np.random.default_rng(0)
    for _ in range(n):
        red = np.full((96, 96, 3), (200.0, 40.0, 40.0)) + rng.normal(0, 6, (96, 96, 3))
        images.append(Image.fromarray(np.clip(red, 0, 255).astype(np.uint8), "RGB"))
        labels.append("red")
        grey = np.asarray(checker(96, 8), dtype=np.float64) * 0.6 + rng.normal(0, 6, (96, 96, 3))
        images.append(Image.fromarray(np.clip(grey, 0, 255).astype(np.uint8), "RGB"))
        labels.append("checker")
    return images, labels


def test_classic_vectors_have_the_descriptor_size_and_nan_for_broken() -> None:
    sources = [noise(), b"not an image", None, flat()]
    vectors = ec.classic_vectors(sources)
    assert vectors.shape == (4, CLASSIC_SIZE)
    assert np.isfinite(vectors[0]).all() and np.isfinite(vectors[3]).all()
    assert np.isnan(vectors[1]).all() and np.isnan(vectors[2]).all()


def test_classic_vectors_read_paths_and_report_progress(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    path.write_bytes(encode(noise()))
    seen: list[tuple[int, int]] = []
    vectors = ec.classic_vectors([path], progress=lambda i, n: seen.append((i, n)))
    assert np.isfinite(vectors).all()
    assert seen[-1] == (1, 1)


def test_classic_vectors_can_be_cancelled() -> None:
    with pytest.raises(ec.Cancelled):
        ec.classic_vectors([noise()] * 3, cancel=lambda: True)


def test_reduce_weighs_families_and_keeps_nan_rows() -> None:
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(30, CLASSIC_SIZE)).astype(np.float32)
    raw[5] = np.nan
    space = ec.reduce(raw)
    assert space.dims == 28  # at most n_ok - 1 components
    assert not space.ok[5] and space.ok.sum() == 29
    assert space.explained == sorted(space.explained, reverse=True)
    assert 0 < sum(space.explained) <= 1.0001


def test_family_weights_balance_hog() -> None:
    weights = ec._weights(CLASSIC_SIZE, ec.FAMILIES)
    shares = [
        float((weights[a:b] ** 2).sum())
        for a, b in ((0, 72), (72, 72 + 1764), (72 + 1764, CLASSIC_SIZE))
    ]
    assert shares == pytest.approx([1.0, 1.0, 1.0])  # R-VMAP-02: same total variance per family
    assert (ec._weights(768, ec.FAMILIES) == 1).all()  # a deep vector: no families


def test_map_of_two_kinds_finds_two_groups_matching_the_labels() -> None:
    images, labels = two_kinds()
    space = ec.classic_space(images)
    found = ec.image_map(space, labels)
    assert found.k == 2
    assert found.reading is not None and found.reading.ari == pytest.approx(1.0)
    assert found.sentence == "Clusters match labels: adjusted Rand 1.00."
    assert len(found.x) == len(found.y) == len(found.rows) == len(images)
    assert len(found.clusters) == len(images)
    assert [s.k for s in found.scores] == list(range(2, 13))
    assert found.explained is not None and len(found.explained) == 2


def test_overlapping_labels_are_said() -> None:
    images, _ = two_kinds()
    labels = ["a", "b"] * (len(images) // 2)  # labels cut across both kinds
    labels = [labels[i] if i % 4 < 2 else labels[i + 1 - 2 * (i % 2)] for i in range(len(images))]
    found = ec.image_map(ec.classic_space(images), labels)
    assert found.sentence.startswith("Labels overlap in this view")


def test_no_labels_says_the_groups() -> None:
    images, _ = two_kinds(6)
    found = ec.image_map(ec.classic_space(images), None)
    assert found.reading is None
    assert found.sentence.startswith("No labels to compare with: 2 groups")


def test_forced_k_and_hdbscan_and_tsne() -> None:
    images, labels = two_kinds(8)
    space = ec.classic_space(images)
    forced = ec.image_map(space, labels, k=3)
    assert forced.k == 3 and forced.k_forced and forced.scores == []
    tsne = ec.image_map(space, labels, projection="tsne")
    assert tsne.explained is None and len(tsne.x) == len(images)
    assert any("t-SNE" in note for note in tsne.notes)
    dense = ec.image_map(space, labels, clustering="hdbscan")
    assert dense.clustering == "hdbscan" and dense.scores == []


def test_sampling_draws_at_most_max_points_but_groups_every_image() -> None:
    images, labels = two_kinds(10)
    space = ec.classic_space(images)
    found = ec.image_map(space, labels, max_points=7)
    assert found.points == 7 and found.of == len(images)
    assert all(c >= 0 for c in found.clusters)
    assert any("7 of 20 images drawn" in note for note in found.notes)


def test_too_few_images() -> None:
    found = ec.image_map(ec.classic_space([noise(), noise(seed=1)]), ["a", "b"])
    assert found.sentence == "Too few images to draw a map." and found.x == []


def test_space_round_trips_by_rel_path(tmp_path: Path) -> None:
    images, _ = two_kinds(4)
    space = ec.classic_space(images)
    keys = [f"img/{i}.png" for i in range(len(images))]
    path = tmp_path / "embeddings" / "classic.parquet"
    ec.save_space(path, keys, space)
    back = ec.load_space(path, [*reversed(keys), "new.png"])
    assert np.allclose(back.vectors[0], space.vectors[-1])
    assert np.isnan(back.vectors[-1]).all()  # an image added later has no vector
    assert back.explained == space.explained and back.method == "classic"


def test_pacmap_and_trimap_draw_the_points_through_the_extra(monkeypatch) -> None:
    """c3, R-VMAP-12: the optional maps of 122, on the points drawn only."""
    import sys
    import types

    images, labels = two_kinds(8)
    space = ec.classic_space(images)

    class Reducer:
        def __init__(self, **_: object) -> None:
            pass

        def fit_transform(self, x, **_: object):
            return np.asarray(x)[:, :2] * 3

    monkeypatch.setitem(sys.modules, "pacmap", types.SimpleNamespace(PaCMAP=Reducer))
    monkeypatch.setitem(sys.modules, "trimap", types.SimpleNamespace(TRIMAP=Reducer))
    for method in ("pacmap", "trimap"):
        found = ec.image_map(space, labels, projection=method)
        assert found.projection == method and found.explained is None
        assert len(found.x) == len(images) and "R-VMAP-12" in found.notes[-1]
