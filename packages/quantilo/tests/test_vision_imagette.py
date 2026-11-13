"""120 T8: the quality imagette. Every R-VQ rule that the pixels or the files can trigger finds
exactly its cases, through the real import (decode, measures, hashes, descriptors, rules), from a
folder and from a ZIP with traps (§ 12 of the study: "chaque R-VQ détecte exactement ses cas")."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantilo.vision.ingest import ImageSet, image_set
from quantilo.vision.quality import RULES
from quantilo.vision.zipsafe import UnsafeArchive
from tests.vision_fixtures import (
    IMAGETTE_RULES,
    IMAGETTE_ZIP_HIDDEN,
    IMAGETTE_ZIP_SKIPPED,
    Imagette,
    imagette_folder,
    imagette_zip,
    quality_imagette,
    rule_confusion,
)


@pytest.fixture(scope="module")
def imagette() -> Imagette:
    return quality_imagette()


def _import(source: Path, workspace: Path) -> ImageSet:
    return image_set(
        source, workspace=workspace, dataset_id="imagette", thumbnails=False, workers=1
    )


@pytest.fixture(scope="module")
def from_folder(imagette: Imagette, tmp_path_factory: pytest.TempPathFactory) -> ImageSet:
    root = tmp_path_factory.mktemp("imagette")
    return _import(imagette_folder(root / "set", imagette), root / "ws")


def _issues(images: ImageSet) -> dict[str, list[str]]:
    table = images.table
    return dict(zip(table["rel_path"].to_pylist(), table["issues"].to_pylist(), strict=True))


def _excluded(images: ImageSet) -> set[str]:
    table = images.table
    paths, excluded = table["rel_path"].to_pylist(), table["excluded"].to_pylist()
    return {p for p, e in zip(paths, excluded, strict=True) if e}


def _misfires(table: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in table if row["missed"] or row["extra"]]


def test_the_imagette_covers_every_import_rule(imagette: Imagette) -> None:
    assert list(IMAGETTE_RULES) == list(RULES)  # R-VQ-16 and 17 need a model (MODEL_RULES)
    covered = {code for codes in imagette.expected.values() for code in codes} | imagette.dataset
    assert covered == set(RULES)
    clean = [p for p, codes in imagette.expected.items() if not codes]
    assert len(imagette.files) == 78 and len(clean) == 53
    assert quality_imagette().files == imagette.files  # deterministic


def test_every_rule_finds_exactly_its_cases(imagette: Imagette, from_folder: ImageSet) -> None:
    report = from_folder.quality
    assert report is not None
    table = rule_confusion(imagette.expected, _issues(from_folder), IMAGETTE_RULES)
    assert _misfires(table) == []
    # R-VQ-13 has no row: it changes the set's metric (a 57, b 13 once R-VQ-14 drops c)
    assert {f.code for f in report.findings if not f.rows} == imagette.dataset
    assert report.recommended_metric == "balanced_accuracy"
    assert report["R-VQ-13"].details["counts"] == {"a": 57, "b": 13}
    assert report["R-VQ-14"].details["dropped"] == ["c"]
    assert _excluded(from_folder) == imagette.excluded
    # every part of R-VQ-15 has its own case: size (huge), brightness (dark, bright), colour
    # (vivid), content (the scanned page, through the descriptors)
    assert report["R-VQ-15"].details["by"] == {
        "width": 1, "height": 1, "brightness": 2, "colour": 1, "content": 1,
    }  # fmt: skip
    assert report.notes == []  # no rule read as the set's style, no size note


def test_the_broken_files_say_why(from_folder: ImageSet) -> None:
    rows = {r["rel_path"]: r for r in from_folder.table.to_pylist()}
    assert "too large to decode safely" in rows["train/a/bomb.png"]["decode_error"]
    assert (rows["train/a/bomb.png"]["width"], rows["train/a/bomb.png"]["height"]) == (
        20_000,
        20_000,
    )
    for name in ("truncated", "corrupt"):
        assert rows[f"train/a/{name}.png"]["decode_ok"] is False
    report = from_folder.quality
    assert report is not None and sum(report["R-VQ-01"].details["errors"].values()) == 3


def test_the_zip_gives_the_same_findings_and_skips_its_traps(
    imagette: Imagette, tmp_path: Path
) -> None:
    images = _import(imagette_zip(tmp_path / "imagette.zip", imagette), tmp_path / "ws")
    assert images.source_kind == "zip"
    assert {s.rel_path: s.code for s in images.skipped} == IMAGETTE_ZIP_SKIPPED
    hidden = next(r for r in images.rules if r.code == "R-VIS-ING-04")
    assert hidden.count == len(IMAGETTE_ZIP_HIDDEN)
    # the bomb never reaches the decoder: the ZIP's ratio rule skips it first (R-VIS-ZIP-04)
    expected = {p: c for p, c in imagette.expected.items() if p not in IMAGETTE_ZIP_SKIPPED}
    table = rule_confusion(expected, _issues(images), IMAGETTE_RULES)
    assert _misfires(table) == []
    assert _excluded(images) == imagette.excluded - set(IMAGETTE_ZIP_SKIPPED)


def test_a_zip_slip_refuses_the_whole_imagette(imagette: Imagette, tmp_path: Path) -> None:
    path = imagette_zip(tmp_path / "slip.zip", imagette, slip=True)
    with pytest.raises(UnsafeArchive) as raised:
        _import(path, tmp_path / "ws")
    assert raised.value.rule == "R-VIS-ZIP-01"
    assert not (tmp_path / "evil.png").exists()
