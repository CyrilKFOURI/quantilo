"""105: quality rules R-VQ-01 to 15, near duplicates by bands, leaks, groups for the split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from quantilo.vision.duplicates import (
    duplicate_groups,
    exact_copies,
    group_ids,
    label_conflicts,
    leaks,
    near_pairs,
)
from quantilo.vision.hashes import hamming
from quantilo.vision.quality import (
    RULES,
    QualityThresholds,
    assess,
    knn_distances,
    robust_z,
    thresholds_from,
)
from tests.vision_fixtures import encode, flat, scene


def table(n: int = 20, **overrides: Any) -> dict[str, list[Any]]:
    """n ordinary, distinct, decoded images; `overrides` = {column: {row: value}}."""
    rng = np.random.default_rng(1)
    data: dict[str, list[Any]] = {
        "rel_path": [f"img/{i}.png" for i in range(n)],
        "decode_ok": [True] * n,
        "decode_error": [None] * n,
        "width": [500 + int(rng.integers(-20, 20)) for _ in range(n)],
        "height": [400 + int(rng.integers(-20, 20)) for _ in range(n)],
        "sharpness": [float(rng.uniform(800, 1200)) for _ in range(n)],
        "luma_p05": [0.1] * n,
        "luma_p95": [0.9] * n,
        "clipped_dark": [0.0] * n,
        "clipped_light": [0.0] * n,
        "contrast": [0.2] * n,
        "entropy": [7.0] * n,
        "is_grayscale": [False] * n,
        "brightness": [float(rng.uniform(0.45, 0.55)) for _ in range(n)],
        "colourfulness": [float(rng.uniform(30, 40)) for _ in range(n)],
        "pixel_sha": [f"{i:064x}" for i in range(n)],
        # far apart hashes: random 64-bit values
        "phash": [int(v) for v in rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64)],
        "dhash": [int(v) for v in rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64)],
        "label": ["a" if i % 2 else "b" for i in range(n)],
        "split": [None] * n,
    }
    for column, values in overrides.items():
        for row, value in values.items():
            data[column][row] = value
    return data


def codes_of(report, row: int) -> list[str]:
    return report.issues[row]


def test_catalogue_has_every_rule_with_a_reason() -> None:
    assert [f"R-VQ-{i:02d}" for i in range(1, 16)] == list(RULES)
    for rule in RULES.values():
        assert rule.reason and rule.title
        for name in rule.parameters:
            assert hasattr(QualityThresholds(), name)


def test_clean_set_has_no_finding() -> None:
    report = assess(table())
    assert report.findings == []
    assert report.sentence == "No quality issue found by the R-VQ rules."
    assert not any(report.excluded)


def test_r01_broken_file_is_excluded_and_listed() -> None:
    report = assess(table(decode_ok={3: False}, decode_error={3: "truncated file"}))
    finding = report["R-VQ-01"]
    assert finding is not None and finding.rows == (3,)
    assert finding.details["errors"] == {"truncated file": 1}
    assert report.excluded[3] and codes_of(report, 3) == ["R-VQ-01"]
    assert finding.effect == "excluded"


def test_r02_too_small_absolute_and_relative() -> None:
    report = assess(table(width={0: 20}, height={1: 100}))
    finding = report["R-VQ-02"]
    assert finding is not None
    # 20 px is under 32; 500 x 100 is a quarter of the median area, not under 1/16
    assert finding.rows == (0,)
    report = assess(table(width={2: 100}, height={2: 100}))  # 10,000 < 200,000 / 16
    assert report["R-VQ-02"].rows == (2,)
    assert not report.excluded[2]


def test_r03_odd_aspect_ratio() -> None:
    report = assess(table(width={4: 2000}, height={4: 400}))
    assert report["R-VQ-03"].rows == (4,)
    assert "padded" in report["R-VQ-03"].message


def test_r04_blurry_needs_both_conditions() -> None:
    # far below the median AND under 100
    report = assess(table(sharpness={5: 20.0}))
    assert report["R-VQ-04"].rows == (5,)
    # a set of smooth images (all under 100) has no blurry one: the relative part fails
    smooth = table(sharpness={i: 60.0 + i for i in range(20)})
    assert assess(smooth).finding("R-VQ-04") is None
    # the absolute threshold can be changed
    report = assess(table(sharpness={5: 150.0}), thresholds=QualityThresholds(blur_sharpness=200))
    assert report["R-VQ-04"].rows == (5,)
    assert report["R-VQ-04"].parameters["blur_sharpness"] == 200


def test_r05_r06_r07_exposure_and_information() -> None:
    report = assess(
        table(
            luma_p95={0: 0.1},
            clipped_dark={1: 0.6},
            luma_p05={2: 0.9},
            clipped_light={3: 0.3},
            contrast={4: 0.01},
            entropy={5: 2.0},
        )
    )
    assert report["R-VQ-05"].rows == (0, 1)
    assert report["R-VQ-06"].rows == (2, 3)
    assert report["R-VQ-07"].rows == (4, 5)


def test_r08_mixed_colour_modes_both_ways() -> None:
    report = assess(table(is_grayscale={7: True}))
    assert report["R-VQ-08"].rows == (7,)
    assert "95% colour" in report["R-VQ-08"].message
    grey = table(is_grayscale={i: True for i in range(20) if i != 9})
    assert assess(grey)["R-VQ-08"].rows == (9,)
    half = table(is_grayscale={i: True for i in range(10)})
    assert assess(half).finding("R-VQ-08") is None


def test_r09_exact_copies_keep_the_first() -> None:
    sha = "f" * 64
    report = assess(table(pixel_sha={2: sha, 6: sha, 11: sha}))
    finding = report["R-VQ-09"]
    assert finding.rows == (2, 6, 11) and finding.count == 2
    assert [report.excluded[r] for r in (2, 6, 11)] == [False, True, True]
    assert finding.details["kept"] == [2]


def test_r10_near_duplicates_by_bands_and_groups() -> None:
    base = table()
    p, d = base["phash"][0], base["dhash"][0]
    # 4 bits off in pHash, 8 in dHash: near; another 3 further bits from row 1: chained group
    p1, d1 = p ^ 0b1111, d ^ 0xFF
    p2, d2 = p1 ^ (0b111 << 40), d1
    data = table(phash={1: p1, 2: p2}, dhash={1: d1, 2: d2})
    data["phash"][0], data["dhash"][0] = p, d
    assert hamming(p, p1) == 4 and hamming(p1, p2) == 3 and hamming(p, p2) == 7
    report = assess(data)
    finding = report["R-VQ-10"]
    assert finding.rows == (0, 1, 2)  # union-find: 0 ~ 1 ~ 2 even though 0 and 2 are 7 apart
    assert not any(report.excluded[:3])
    assert report.groups[0].kind == "near"
    # a stricter threshold separates them
    strict = assess(data, thresholds=QualityThresholds(near_phash=3))
    assert strict["R-VQ-10"].rows == (1, 2)


def test_near_pairs_agree_with_brute_force() -> None:
    rng = np.random.default_rng(7)
    base = rng.integers(-(2**63), 2**63 - 1, 40, dtype=np.int64)
    phash = list(base)
    dhash = list(base)
    for i in range(40, 140):  # 100 noisy copies of the 40 bases
        src = int(base[i % 40])
        flips = rng.choice(64, int(rng.integers(0, 10)), replace=False)
        value = src & ((1 << 64) - 1)  # unsigned, flipped, then signed again
        for bit in flips:
            value ^= 1 << int(bit)
        value = value - (1 << 64) if value >= 1 << 63 else value
        phash.append(value)
        dhash.append(src)
    phash = [int(v) for v in phash]
    dhash = [int(v) for v in dhash]
    for t in (2, 6, 9):
        expected = {
            (i, j)
            for i in range(len(phash))
            for j in range(i + 1, len(phash))
            if hamming(phash[i], phash[j]) <= t and hamming(dhash[i], dhash[j]) <= 10
        }
        assert set(near_pairs(phash, dhash, max_phash=t)) == expected


def test_near_pairs_skip_missing_hashes() -> None:
    assert near_pairs([5, None, 5], [1, None, 1]) == [(0, 2)]


def test_r11_leak_across_train_and_test() -> None:
    sha = "e" * 64
    data = table(
        pixel_sha={0: sha, 1: sha, 2: "d" * 64, 3: "d" * 64},
        split={0: "train", 1: "test", 2: "train", 3: "train", 4: "test"},
    )
    report = assess(data)
    finding = report["R-VQ-11"]
    assert finding.rows == (1,)
    assert "1 test image has a near-copy in train" in finding.message
    assert report.leaks[0].train_rows == (0,)


def test_r12_same_image_two_labels() -> None:
    sha = "c" * 64
    report = assess(table(pixel_sha={0: sha, 1: sha}, label={0: "cat", 1: "dog"}))
    assert report["R-VQ-12"].rows == (0, 1)
    assert report["R-VQ-12"].count == 1


def test_r13_imbalance_sets_the_metric() -> None:
    data = table(40, label={i: "rare" if i < 6 else "common" for i in range(40)})
    report = assess(data)
    finding = report["R-VQ-13"]
    assert finding is not None and finding.details["ratio"] == pytest.approx(34 / 6)
    assert report.recommended_metric == "balanced_accuracy"
    assert report.class_weight == "balanced"
    assert finding.rows == () and finding.effect == "dataset"
    assert assess(table()).recommended_metric == "accuracy"


def test_r14_small_classes_warned_and_excluded() -> None:
    labels = {i: "big" for i in range(30)}
    labels.update({i: "small" for i in range(30, 37)})  # 7: warned
    labels.update({i: "tiny" for i in range(37, 40)})  # 3: excluded
    report = assess(table(40, label=labels))
    finding = report["R-VQ-14"]
    assert finding.details["classes"] == {"small": 7, "tiny": 3}
    assert finding.details["dropped"] == ["tiny"]
    assert not report.excluded[30] and report.excluded[38]
    # R-VQ-13 is read without the dropped class: 30 / 7
    assert report["R-VQ-13"].details["ratio"] == pytest.approx(30 / 7)


def test_r15_unlike_the_others_by_size_and_content() -> None:
    report = assess(table(width={3: 4000}, height={3: 3000}))
    finding = report["R-VQ-15"]
    assert 3 in finding.rows and finding.details["by"]["width"] >= 1
    # content: one image far from 59 others in the descriptor space
    rng = np.random.default_rng(0)
    vectors = rng.normal(0, 1, (60, 16))
    vectors[10] += 25
    report = assess(table(60), descriptors=vectors)
    assert report["R-VQ-15"].rows == (10,)
    assert report["R-VQ-15"].details["by"] == {"content": 1}
    # without an outlier the quantile alone never fires (the z part)
    assert assess(table(60), descriptors=rng.normal(0, 1, (60, 16))).finding("R-VQ-15") is None


def test_rules_can_be_ignored_and_small_sets_are_said() -> None:
    sha = "b" * 64
    report = assess(table(pixel_sha={0: sha, 1: sha}), ignore={"R-VQ-09"})
    assert report.finding("R-VQ-09") is None and not report.excluded[1]
    tiny = assess(table(3))
    assert any("need 8 decoded images" in note for note in tiny.notes)


def test_groups_for_the_split() -> None:
    groups = duplicate_groups(["x", "y", "x", "z"], [1, 2, 3, 4], [9, 8, 7, 6], near=False)
    assert [g.rows for g in groups] == [(0, 2)]
    ids = group_ids(4, groups)
    assert ids[0] == ids[2] and len(set(ids.tolist())) == 3
    assert exact_copies(["x", None, "x"]) == [[0, 2]]
    assert leaks(groups, ["train", None, "test", None])[0].eval_rows == (2,)
    assert label_conflicts(groups, ["a", None, "b", None]) == groups


def test_apply_writes_issues_excluded_and_group() -> None:
    sha = "a" * 64
    data = table(30, pixel_sha={0: sha, 1: sha}, label={0: "a", 1: "a"}, sharpness={5: 10.0})
    report = assess(data)
    arrow = pa.table(data | {"issues": [[]] * 30, "excluded": [False] * 30})
    out = report.apply(arrow)
    assert out["issues"].to_pylist()[1] == ["R-VQ-09"]
    assert out["issues"].to_pylist()[5] == ["R-VQ-04"]
    assert out["excluded"].to_pylist()[:2] == [False, True]
    assert out["duplicate_group"].to_pylist()[:3] == [0, 0, None]
    summary = report.to_dict()
    assert summary["excluded"] == 1 and summary["findings"][0]["code"] == "R-VQ-04"


def test_thresholds_from_json() -> None:
    assert thresholds_from({"blur_sharpness": 50}).blur_sharpness == 50.0
    assert thresholds_from({"min_side": 64.0}).min_side == 64
    with pytest.raises(ValueError, match="Unknown"):
        thresholds_from({"sharp": 1})
    with pytest.raises(ValueError, match="positive"):
        thresholds_from({"min_side": -1})


def test_robust_z_and_knn() -> None:
    z = robust_z(np.array([1.0] * 9 + [9.0]))  # MAD 0: the mean deviation takes over
    assert z[-1] > 4 and z[0] == 0
    assert (robust_z(np.ones(4)) == 0).all()
    x = np.array([[0.0], [1.0], [2.0], [10.0]])
    d = knn_distances(x, 1)
    assert d.tolist()[-1] == pytest.approx(8 / x.std())


def test_import_carries_the_quality(tmp_path: Path) -> None:
    from quantilo.vision.ingest import image_set

    root = tmp_path / "set"
    for label in ("cat", "dog"):
        (root / label).mkdir(parents=True)
        for i in range(6):
            seed = i + (10 if label == "dog" else 0)
            (root / label / f"{i}.png").write_bytes(encode(scene(seed=seed)))
    # an exact copy, re-encoded as BMP, and a black frame
    (root / "cat" / "copy.bmp").write_bytes(encode(scene(seed=0), "BMP"))
    (root / "dog" / "black.png").write_bytes(encode(flat((0, 0, 0), (160, 120))))
    images = image_set(root, workspace=tmp_path / "ws", dataset_id="q")
    rows = {r["rel_path"]: r for r in images.table.to_pylist()}
    assert "R-VQ-09" in rows["cat/copy.bmp"]["issues"]
    assert rows["cat/copy.bmp"]["excluded"] is True
    assert rows["cat/copy.bmp"]["duplicate_group"] == rows["cat/0.png"]["duplicate_group"]
    assert "R-VQ-05" in rows["dog/black.png"]["issues"]  # R-VQ-07 is these flat scenes' style
    assert images.quality is not None and "R-VQ-09" in images.quality.codes
    off = image_set(root, workspace=tmp_path / "ws", dataset_id="q2", quality=False)
    assert off.quality is None and off.table["issues"].to_pylist()[0] == []


def test_the_sets_style_is_not_a_defect() -> None:
    # a white studio background on every image: R-VQ-06 says so instead of flagging all
    report = assess(table(clipped_light={i: 0.35 for i in range(20)}))
    assert report.finding("R-VQ-06") is None
    assert any(note.startswith("R-VQ-06 (overexposed) would flag 100%") for note in report.notes)
    # a few overexposed images are still flagged
    assert assess(table(clipped_light={0: 0.35, 1: 0.4}))["R-VQ-06"].rows == (0, 1)
    # inspection photos that all look alike: no near group, exact copies still found
    same = table(phash={i: 12345 for i in range(20)}, dhash={i: 678 for i in range(20)})
    same["pixel_sha"][3] = same["pixel_sha"][4]
    report = assess(same, thresholds=QualityThresholds())
    assert report.finding("R-VQ-10") is None and report.finding("R-VQ-11") is None
    assert report["R-VQ-09"].rows == (3, 4)
    assert [g.rows for g in report.groups] == [(3, 4)]
