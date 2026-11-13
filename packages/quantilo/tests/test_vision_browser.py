"""119: vision in the browser version (R-VIS-WEB-01 to 03). The browser is simulated by the switch
of 102 (`quantilo.ml.browser.IN_BROWSER`), as the ML tests do."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from quantilo.ml import browser as ml_browser
from quantilo.vision import browser, deep
from quantilo.vision import classic_models as cm
from quantilo.vision.ingest import image_set, preview_image_set
from quantilo.vision.measures import measure
from quantilo.vision.quality import QualityThresholds
from tests.vision_fixtures import encode, image_folder, scene, zip_of


@pytest.fixture
def in_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ml_browser, "IN_BROWSER", True)


def _folder(tmp_path: Path, n: int = 6, size: tuple[int, int] = (640, 480)) -> Path:
    layout: dict[str, Image.Image | bytes] = {
        f"{label}/{label}_{i}.png": scene(k * 10 + i, size)
        for k, label in enumerate(("healthy", "rust"))
        for i in range(n // 2)
    }
    return image_folder(tmp_path / "set", layout)


def _codes(found) -> set[str]:
    return {rule.code for rule in found.rules}


# --- R-VIS-WEB-01: at most 2,000 images and 300 MB -------------------------------------------


def test_budget_keeps_everything_on_a_computer() -> None:
    found = browser.budget([10**9] * 3_000)
    assert found.kept == 3_000 and found.note is None


def test_budget_holds_the_browser_to_2000_images(in_browser: None) -> None:
    found = browser.budget([1_000] * 2_500)
    assert found.kept == 2_000
    assert found.note is not None and found.note.code == "R-VIS-WEB-01"
    assert found.note.count == 500
    assert "first 2,000 of 2,500 images" in found.note.message
    assert "50,000" in found.note.reason


def test_budget_holds_the_browser_to_300_mb(in_browser: None) -> None:
    found = browser.budget([100 * 2**20] * 5)
    assert found.kept == 3 and found.note is not None and "300 MB" in found.note.message
    assert browser.budget([1_000] * 10).note is None


def test_import_takes_the_first_images_and_says_so(
    in_browser: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(browser, "MAX_IMAGES", 4)
    source = _folder(tmp_path, n=6)
    preview = preview_image_set(source)
    assert preview.n_images == 4 and "R-VIS-WEB-01" in _codes(preview)
    images = image_set(source, workspace=tmp_path / "ws", dataset_id="web")
    assert images.n_images == 4 and "R-VIS-WEB-01" in _codes(images)


def test_zip_import_is_held_too(in_browser: None, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(browser, "MAX_IMAGES", 2)
    archive = zip_of(tmp_path / "set.zip", {f"a/{i}.png": encode(scene(i)) for i in range(5)})
    images = image_set(archive, workspace=tmp_path / "ws", dataset_id="webzip")
    assert images.n_images == 2 and "R-VIS-WEB-01" in _codes(images)


# --- R-VIS-WEB-02: measured at 256 pixels, blur threshold moved -----------------------------


def test_measures_at_256_pixels_in_the_browser(in_browser: None, tmp_path: Path) -> None:
    assert browser.measure_side() == 256 and browser.measure_side(512) == 512
    found = measure(scene(1, (640, 480)))
    assert max(found.measured_width, found.measured_height) == 256
    images = image_set(_folder(tmp_path), workspace=tmp_path / "ws", dataset_id="web256")
    widths = images.table.column("width").to_pylist()
    assert widths == [640] * 6  # the file's own size is kept
    note = next(r for r in images.rules if r.code == "R-VIS-WEB-02")
    assert "256 pixels" in note.message and "300" in note.message


def test_measures_stay_at_512_on_a_computer(tmp_path: Path) -> None:
    assert browser.measure_side() == 512
    found = measure(scene(1, (640, 480)))
    assert max(found.measured_width, found.measured_height) == 512
    images = image_set(_folder(tmp_path), workspace=tmp_path / "ws", dataset_id="web512")
    assert not {"R-VIS-WEB-01", "R-VIS-WEB-02"} & _codes(images)


def test_blur_threshold_follows_the_scale() -> None:
    assert browser.quality_thresholds(None, 512) is None
    moved = browser.quality_thresholds(None, 256)
    assert moved is not None and moved.blur_sharpness == browser.BLUR_SHARPNESS == 300
    by_hand = browser.quality_thresholds(QualityThresholds(blur_sharpness=42), 256)
    assert by_hand is not None and by_hand.blur_sharpness == 42  # set by hand: kept


# --- R-VIS-WEB-03: classic models only, every deep option greyed -----------------------------


def test_every_deep_option_says_desktop_only(in_browser: None) -> None:
    status = deep.status()
    assert not status.available and status.reason == browser.DESKTOP_REASON
    assert all(not b.available and b.reason == browser.DESKTOP_REASON for b in status.backbones)
    assert all(not d.available and d.reason == browser.DESKTOP_REASON for d in status.detectors)
    assert status.reason.startswith("Desktop only")
    assert "R-VDP-01" in status.reason  # the page hides the download button on this code
    assert cm.deep_option(None).reason == browser.DESKTOP_REASON
    assert cm.choose_detector("patchcore", None, []) == (None, browser.DESKTOP_REASON)
    assert cm.BROWSER_MAX_IMAGES == browser.MAX_IMAGES
