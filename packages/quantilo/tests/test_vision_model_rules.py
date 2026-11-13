"""b2: temperature scaling applied out of fold (R-VM-06), possibly mislabelled images (R-VQ-16)
and semantic near-copies (R-VQ-17)."""

import numpy as np
import pytest

from quantilo.vision import metrics as vm
from quantilo.vision.duplicates import DuplicateGroup, semantic_groups, semantic_pairs
from quantilo.vision.quality import MODEL_RULES, RULES, possibly_mislabelled, semantic_duplicates


def overconfident(n: int = 3000, k: int = 3, sharpen: float = 3.0, seed: int = 11):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, k)) * 1.5
    true_p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in true_p])
    sharp = np.exp(sharpen * logits) / np.exp(sharpen * logits).sum(axis=1, keepdims=True)
    return y, sharp, true_p


def test_parts_keep_copies_together_and_are_balanced():
    groups = np.repeat(np.arange(50), 2)  # 50 pairs of copies
    part = vm.calibration_parts(len(groups), groups)
    assert set(part.tolist()) == {0, 1, 2, 3, 4}
    assert all(part[2 * g] == part[2 * g + 1] for g in range(50))
    assert np.bincount(part).tolist() == [20] * 5
    assert vm.calibration_parts(100, groups).tolist() == part.tolist()  # seeded


def test_temperature_applied_out_of_fold():
    y, sharp, _ = overconfident()
    found = vm.calibration(y, sharp)
    assert found.applied and len(found.temperatures) == 5
    assert all(v == pytest.approx(3.0, rel=0.15) for v in found.temperatures)
    assert found.ece_after is not None and found.ece_after < 0.03 < found.reliability.ece
    assert found.reliability_after is not None
    assert found.reliability_after.ece == found.ece_after
    assert found.log_loss_after is not None and found.log_loss_after < vm.log_loss(y, sharp)
    assert found.brier_after is not None and found.brier_after < vm.brier(y, sharp)
    assert "R-VM-06" in found.reason and "->" in found.reason
    calibrated, temps = vm.crossfit_temperature(y, sharp)
    assert (calibrated.argmax(axis=1) == sharp.argmax(axis=1)).all()  # predictions unchanged
    assert temps == pytest.approx(found.temperatures, abs=1e-4)
    assert np.allclose(vm.calibrated_proba(y, sharp, None, found), calibrated)


def test_reliable_or_small_sets_are_not_rescaled():
    y, sharp, true_p = overconfident()
    reliable = vm.calibration(y, true_p)
    assert not reliable.applied and not reliable.suggest_temperature
    assert np.array_equal(vm.calibrated_proba(y, true_p, None, reliable), true_p)
    few = vm.calibration(y[:20], sharp[:20])
    assert few.suggest_temperature and not few.applied
    assert "proposed, not applied" in few.reason


def test_report_carries_the_applied_temperature():
    y, sharp, _ = overconfident(n=600)
    report = vm.classification_report(y, proba=sharp, bootstrap=False)
    assert report.calibration is not None and report.calibration.applied
    assert report.log_loss == pytest.approx(vm.log_loss(y, sharp))  # the model as it is


def test_possibly_mislabelled_by_the_rule():
    classes = ["cat", "dog", "fox"]
    proba = np.array(
        [
            [0.05, 0.90, 0.05],  # labelled cat: dog 0.90 -> flagged
            [0.08, 0.12, 0.80],  # cat 0.08, fox exactly 0.80 -> not (more than 0.8 needed)
            [0.90, 0.05, 0.05],  # labelled cat: fine
            [0.02, 0.97, 0.01],  # labelled fox: dog 0.97 -> flagged, first (surest)
            [0.15, 0.85, 0.00],  # labelled cat 0.15 -> not (less than 0.1 needed)
        ]
    )
    labels = ["cat", "cat", "cat", "fox", "cat"]
    found = possibly_mislabelled(labels, proba, classes, [10, 11, 12, 13, 14], model="Probe")
    assert found.code == "R-VQ-16" and found.effect == "flagged"
    assert found.rows == (13, 10) and found.count == 2
    assert found.details["images"][0] == {
        "row": 13,
        "label": "fox",
        "suggested": "dog",
        "suggested_proba": 0.97,
        "label_proba": 0.01,
    }
    assert "check their label" in found.message and "by Probe" in found.message
    assert found.details["checked"] == 5
    clean = possibly_mislabelled(["cat"], proba[2:3], classes, [0])
    assert clean.count == 0 and "No image looks mislabelled" in clean.message


def test_model_rules_stay_out_of_the_import_catalogue():
    assert set(MODEL_RULES) == {"R-VQ-16", "R-VQ-17"}
    assert not set(MODEL_RULES) & set(RULES)


def brute_pairs(x: np.ndarray, threshold: float) -> set[tuple[int, int]]:
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    sims = unit @ unit.T
    n = len(x)
    return {(i, j) for i in range(n) for j in range(i + 1, n) if sims[i, j] >= threshold}


def test_semantic_pairs_match_brute_force_by_blocks():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(40, 16)).astype(np.float32)
    noisy = base[:10] + 0.05 * rng.normal(size=(10, 16)).astype(np.float32)
    x = np.vstack([base, noisy])
    found = semantic_pairs(x, 0.95, block=7)
    assert {(i, j) for i, j, _ in found} == brute_pairs(x, 0.95)
    assert len(found) >= 10 and all(c >= 0.95 for _, _, c in found)


def test_semantic_groups_skip_what_the_hashes_found_and_nan_rows():
    x = np.array(
        [[1, 0], [0.99, 0.01], [0, 1], [0.01, 0.99], [1, 1], [np.nan, 1]], dtype=np.float32
    )
    pairs = semantic_pairs(x, 0.95)
    assert {(i, j) for i, j, _ in pairs} == {(0, 1), (2, 3)}
    known = [DuplicateGroup(0, "near", (2, 3), None)]
    assert [g.rows for g in semantic_groups(len(x), pairs, known)] == [(0, 1)]


def test_semantic_duplicates_finding():
    x = np.array([[1, 0, 0], [0.98, 0.02, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    found = semantic_duplicates(
        x, labels=["a", "b", "a", "c"], split=["train", "test", None, None], network="DINOv2"
    )
    assert found.code == "R-VQ-17" and found.rows == (0, 1) and found.count == 2
    group = found.details["groups"][0]
    assert group["two_labels"] and group["leak"] and group["cosine"] >= 0.95
    assert found.details["leaks"] == 1 and found.details["compared"] == 4
    assert "1 group" in found.message and "DINOv2" in found.message
    none = semantic_duplicates(np.eye(3, dtype=np.float32))
    assert none.count == 0 and "No semantic near-copy" in none.message


def test_a_set_of_look_alikes_is_its_style():
    """MVTec bottle: one object shot the same way, almost every pair above 0.95: nothing flagged."""
    rng = np.random.default_rng(5)
    x = (np.ones((20, 8)) + 0.01 * rng.normal(size=(20, 8))).astype(np.float32)
    found = semantic_duplicates(x, network="DINOv2")
    assert found.count == 0 and found.details["groups"] == []
    assert found.details["style"] and "this set's style" in found.message
