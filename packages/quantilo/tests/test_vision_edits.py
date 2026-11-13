"""c3: the user's edits of an image set (decisions, labels, pins, thresholds: R-VQT, R-VED) and the
parallel reading of an import (R-VIS-ING-07)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from quantilo.vision import parallel
from quantilo.vision.edits import (
    DERIVED,
    NO_DESCRIPTORS,
    Edits,
    decided_table,
    edited_table,
    labelled_table,
    load_descriptors,
    pinned_table,
)
from quantilo.vision.ingest import image_set
from quantilo.vision.quality import QualityThresholds
from tests.vision_fixtures import encode, image_folder, scene


def _table(n: int = 6) -> pa.Table:
    return pa.table(
        {
            "rel_path": [f"img{i}.png" for i in range(n)],
            "label": ["a"] * (n // 2) + ["b"] * (n - n // 2),
            "excluded": [i == 0 for i in range(n)],
            "decode_ok": [True] * n,
        }
    )


def test_decisions_keep_the_rules_answer() -> None:
    table = decided_table(_table(), {"img0.png": "keep", "img1.png": "exclude"})
    assert table["excluded"].to_pylist()[:3] == [False, True, False]
    assert table["rule_excluded"].to_pylist()[:2] == [True, False]
    again = decided_table(table, {})  # back to the rules: read from rule_excluded
    assert again["excluded"].to_pylist()[:2] == [True, False]


def test_labels_keep_the_original_once() -> None:
    first = labelled_table(_table(), {"img0.png": "b"})
    assert first["label"].to_pylist()[0] == "b"
    assert first["label_original"].to_pylist()[0] == "a"
    second = labelled_table(first, {"img0.png": "c"})
    assert second["label"].to_pylist()[0] == "c"
    assert second["label_original"].to_pylist()[0] == "a"  # R-VED-02: never changed after
    back = labelled_table(second, {})
    assert back["label"].to_pylist()[0] == "a"
    assert labelled_table(_table(), {}).column_names == _table().column_names


def test_pins() -> None:
    table = pinned_table(_table(), {"img2.png": True})
    assert table["pinned"].to_pylist() == [False, False, True, False, False, False]
    assert "pinned" not in pinned_table(_table(), {}).column_names
    assert {"decision", "rule_excluded", "label_original", "pinned"} == set(DERIVED)


def _set(tmp_path: Path, n: int = 24) -> Path:
    layout = {f"{('a', 'b')[i % 2]}/x{i}.png": encode(scene(i)) for i in range(n)}
    return image_folder(tmp_path / "set", layout)


def test_thresholds_and_labels_run_the_rules_again(tmp_path: Path) -> None:
    images = image_set(_set(tmp_path), workspace=tmp_path / "ws", dataset_id="s")
    table = images.table
    keys = [str(v) for v in table["rel_path"].to_pylist()]
    descriptors = load_descriptors(images.root, keys)
    assert descriptors is not None and descriptors.shape == (24, 64)  # R-VQT-03
    assert np.isfinite(descriptors).all()
    assert images.quality is not None and "R-VQ-04" not in images.quality.codes
    # thresholds that every image crosses (both conditions of R-VQ-04): the rule fires on all
    strict = QualityThresholds(blur_sharpness=1e12, blur_z=-1e9)
    _, report = edited_table(table, Edits(), thresholds=strict, descriptors=descriptors)
    assert report is not None and report["R-VQ-04"].count == 24
    assert report.thresholds.blur_sharpness == 1e12
    # labels: one class reduced to 3 images is left out of models by R-VQ-14
    b_keys = [k for k in keys if k.startswith("b/")]
    relabel = {k: "a" for k in b_keys[3:]}
    edited, report = edited_table(table, Edits(labels=relabel), descriptors=descriptors)
    assert report is not None and "R-VQ-14" in report.codes
    labels = edited["label"].to_pylist()
    assert labels.count("b") == 3
    assert edited["rule_excluded"].to_pylist() == edited["excluded"].to_pylist()
    assert sum(edited["excluded"].to_pylist()) == 3
    # a decision still wins over the rule
    kept = {b_keys[0]: "keep"}
    edited, _ = edited_table(table, Edits(decisions=kept, labels=relabel), descriptors=descriptors)
    assert sum(edited["excluded"].to_pylist()) == 2


def test_without_descriptors_the_note_says_so(tmp_path: Path) -> None:
    n = 60
    table = pa.table(
        {
            "rel_path": [f"x{i}.png" for i in range(n)],
            "decode_ok": [True] * n,
            "width": [100] * n,
            "height": [100] * n,
            "excluded": [False] * n,
        }
    )
    _, report = edited_table(table, Edits())
    assert report is not None and NO_DESCRIPTORS in report.notes
    _, quiet = edited_table(table, Edits(), ignore={"R-VQ-15"})
    assert quiet is not None and NO_DESCRIPTORS not in quiet.notes


# --- R-VIS-ING-07 -------------------------------------------------------------------------------


def test_workers_follow_the_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    gib = parallel.GIB
    assert parallel.import_workers(50, free=32 * gib, cpus=16).count == 1  # small set
    assert parallel.import_workers(1_000, free=32 * gib, cpus=16).count == 4
    assert parallel.import_workers(1_000, free=4 * gib, cpus=16).count == 2  # memory bound
    low = parallel.import_workers(1_000, free=int(2.5 * gib), cpus=16)
    assert low.count == 1 and "GiB" in low.reason
    assert parallel.import_workers(1_000, free=None, cpus=3).count == 2  # cores - 1
    assert parallel.import_workers(1_000, requested=1).count == 1
    from quantilo.ml import browser

    monkeypatch.setattr(browser, "active", lambda: True)
    assert "browser" in parallel.import_workers(1_000, free=32 * gib, cpus=16).reason


def test_free_memory_is_read_here() -> None:
    found = parallel.free_memory()
    assert found is None or found > 0


def test_parallel_import_gives_the_same_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the count is bounded by the free memory: fixed here, or a loaded machine reads in one process
    monkeypatch.setattr(parallel, "free_memory", lambda: 16 * parallel.GIB)
    folder = _set(tmp_path, 10)
    one = image_set(folder, workspace=tmp_path / "w1", dataset_id="one", workers=1)
    two = image_set(folder, workspace=tmp_path / "w2", dataset_id="two", workers=2)
    assert "R-VIS-ING-07" in {r.code for r in two.rules}
    assert "R-VIS-ING-07" not in {r.code for r in one.rules}
    drop = {"stored_path", "thumb_path"}
    a = [{k: v for k, v in r.items() if k not in drop} for r in one.table.to_pylist()]
    b = [{k: v for k, v in r.items() if k not in drop} for r in two.table.to_pylist()]
    assert a == b
    assert [r["thumb_path"] for r in one.table.to_pylist()] == [
        r["thumb_path"] for r in two.table.to_pylist()
    ]
    assert len(list((two.root / "thumbs").glob("*.webp"))) == 10
