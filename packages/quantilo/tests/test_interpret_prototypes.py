"""Explanation by cases (117): nearest examples, prototypes, criticisms, counterfactual example."""

import numpy as np
import pytest

from quantilo.interpret import prototypes as pt


def blobs(seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Two classes on two directions of the plane (cosine space), 40 rows each, plus one stray
    row of class a pointing the other way."""
    rng = np.random.default_rng(seed)
    a = np.array([1.0, 0.0, 0.0]) + rng.normal(0, 0.05, (40, 3))
    b = np.array([0.0, 1.0, 0.0]) + rng.normal(0, 0.05, (40, 3))
    stray = np.array([[0.0, 0.0, 1.0]])
    return np.vstack([a, b, stray]), ["a"] * 40 + ["b"] * 40 + ["a"]


def test_cosine_and_normalise():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    z = pt.normalise(x)
    assert np.allclose(z[0], [0.6, 0.8])
    assert np.allclose(z[1], 0.0)
    assert pt.cosine(np.array([1.0, 0.0]), np.array([[2.0, 0.0], [0.0, 5.0]])).tolist() == [
        [1.0, 0.0]
    ]


def test_nearest_examples_are_the_most_similar_and_say_how_many_agree():
    x, labels = blobs()
    query = np.array([0.9, 0.1, 0.0])
    found = pt.nearest_examples(query, x, labels, k=5, label="a")
    assert len(found.neighbours) == 5
    sims = [n.similarity for n in found.neighbours]
    assert sims == sorted(sims, reverse=True)
    assert all(n.label == "a" for n in found.neighbours)
    assert found.agreeing == 5
    assert "5 of the 5 nearest training images are labelled a" in found.sentence
    brute = np.argsort(-pt.cosine(query, x)[0])[:5]
    assert [n.index for n in found.neighbours] == brute.tolist()


def test_nearest_examples_exclude_rows_and_nan():
    x, labels = blobs()
    x[3] = np.nan
    query = x[0]
    found = pt.nearest_examples(query, x, labels, k=80, exclude=[0])
    indices = [n.index for n in found.neighbours]
    assert 0 not in indices and 3 not in indices
    assert len(indices) == len(x) - 2
    empty = pt.nearest_examples(query, x[:1], labels[:1], exclude=[0])
    assert empty.neighbours == [] and "No training image" in empty.sentence


def test_kmedoids_finds_one_medoid_per_blob_and_orders_by_size():
    x, _ = blobs()
    medoids, assign = pt.kmedoids(x[:80], 2)
    assert sorted(m // 40 for m in medoids) == [0, 1]
    assert assign.shape == (80,)
    assert np.bincount(assign).tolist() == [40, 40]
    # deterministic
    again, _ = pt.kmedoids(x[:80], 2)
    assert again == medoids
    one, assign1 = pt.kmedoids(x[:5], 9)
    assert len(one) == 5 and len(set(one)) == 5 and assign1.shape == (5,)


def test_medoid_is_the_row_of_least_total_distance_in_its_cluster():
    x, _ = blobs()
    medoids, assign = pt.kmedoids(x[:40], 1)
    d = 1 - pt.cosine(x[:40], x[:40])
    assert medoids == [int(np.argmin(d.sum(axis=1)))]
    assert (assign == 0).all()


def test_criticisms_come_from_the_part_of_the_class_no_prototype_covers():
    x, labels = blobs()
    rng = np.random.default_rng(3)
    corner = np.array([0.0, 0.0, 1.0]) + rng.normal(0, 0.05, (6, 3))  # a small mode of class a
    x = np.vstack([x[:80], corner])
    labels = labels[:80] + ["a"] * 6
    found = pt.class_prototypes(x, labels, per_class=1, critics=3)
    by = {c.label: c for c in found}
    assert set(by) == {"a", "b"}
    assert by["a"].size == 46 and by["b"].size == 40
    assert by["a"].prototypes[0] < 40  # the medoid sits in the main blob
    assert all(c >= 80 for c in by["a"].criticisms)  # the uncovered mode is criticised
    assert len(by["a"].criticisms) == 3 and all(w > 0 for w in by["a"].witness)
    assert pytest.approx(sum(by["a"].shares)) == 1.0


def test_witness_is_zero_when_prototypes_are_all_rows():
    x, _ = blobs()
    w = pt.witness(x[:10], list(range(10)))
    assert np.allclose(w, 0.0)


def test_mmd_prototypes_cover_both_blobs():
    x, _ = blobs()
    chosen = pt.mmd_prototypes(x[:80], 2)
    assert sorted(c // 40 for c in chosen) == [0, 1]


def test_class_prototypes_draws_large_classes_and_skips_missing_labels():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 4))
    labels = ["a"] * 50 + [None] * 10
    found = pt.class_prototypes(x, labels, per_class=3, max_per_class=20)
    assert [c.label for c in found] == ["a"]
    assert found[0].size == 50
    assert "drawn from 20 of its 50" in found[0].notes[0]
    assert all(p < 50 for p in found[0].prototypes + found[0].criticisms)


def test_counterfactual_is_the_nearest_of_another_class():
    x, labels = blobs()
    query = np.array([0.7, 0.3, 0.0])
    found = pt.counterfactual_example(query, x, labels, "a")
    assert found is not None and found.label == "b"
    sims = pt.cosine(query, x)[0]
    b = [i for i, v in enumerate(labels) if v == "b"]
    assert found.index == b[int(np.argmax(sims[b]))]
    targeted = pt.counterfactual_example(query, x, labels, "b", target="a", exclude=[0])
    assert targeted is not None and targeted.label == "a" and targeted.index != 0
    assert pt.counterfactual_example(query, x, ["a"] * len(x), "a") is None


def test_nearest_examples_by_euclidean_distance():
    bank = np.array([[0.0, 0.0], [10.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    found = pt.nearest_examples(np.array([0.5, 0.0]), bank, k=3, metric="euclidean")
    assert [n.index for n in found.neighbours] == [0, 2, 3]
    assert [n.distance for n in found.neighbours][:2] == [0.5, 0.5]
    assert "distance" in found.sentence
    with pytest.raises(ValueError):
        pt.nearest_examples(bank[0], bank, metric="manhattan")


def test_nearest_sentence_names_the_other_labels():
    x, labels = blobs()
    found = pt.nearest_examples(np.array([0.5, 0.5, 0.0]), x, labels, k=6, label="a")
    assert found.agreeing < 6
    assert "; the others: b" in found.sentence
