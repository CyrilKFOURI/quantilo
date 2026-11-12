"""103: measures at a fixed scale, pHash/dHash, and the classic descriptors (HSV, HOG, LBP)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageFilter

from quantilo.vision.decode import decode_image
from quantilo.vision.descriptors import (
    CLASSIC_SIZE,
    HOG_SIZE,
    HSV_SIZE,
    LBP_SIZE,
    classic_descriptor,
    hog,
    hsv_histogram,
    lbp,
    lbp_codes,
)
from quantilo.vision.hashes import dhash, hamming, hamming_many, phash, pixel_sha, to_hex
from quantilo.vision.measures import (
    MEASURE_SIDE,
    colourfulness,
    dominant_colours,
    laplacian_variance,
    measure,
    working_copy,
)
from tests.vision_fixtures import checker, encode, flat, gradient, noise, scene

# --- measures -----------------------------------------------------------------------------------


def test_flat_grey_image() -> None:
    m = measure(flat((128, 128, 128)))
    assert m.brightness == pytest.approx(128 / 255, abs=1e-3)
    assert m.contrast == pytest.approx(0.0, abs=1e-9)
    assert m.entropy == 0.0
    assert m.sharpness == 0.0
    assert m.saturation == 0.0
    assert m.colourfulness == pytest.approx(0.0, abs=1e-9)
    assert m.is_grayscale
    assert m.dominant_colour == "#808080"
    assert m.clipped_dark == 0.0 and m.clipped_light == 0.0


def test_black_and_white_images_are_clipped() -> None:
    assert measure(flat((0, 0, 0))).clipped_dark == 1.0
    assert measure(flat((255, 255, 255))).clipped_light == 1.0
    assert measure(flat((0, 0, 0))).luma_p95 == 0.0


def test_pure_red_is_saturated_and_colourful() -> None:
    m = measure(flat((255, 0, 0)))
    assert m.saturation == pytest.approx(1.0)
    assert not m.is_grayscale
    assert m.colourfulness == pytest.approx(0.3 * np.hypot(255, 127.5))  # no spread, only mean
    assert m.brightness == pytest.approx(0.299, abs=1e-3)  # Rec. 601


def test_hasler_suesstrunk_on_known_values() -> None:
    rgb = np.zeros((1, 2, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    rgb[0, 1] = (0, 0, 255)
    rg = np.array([255.0, 0.0])
    yb = np.array([127.5, -255.0])
    expected = np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    assert colourfulness(rgb) == pytest.approx(expected)


def test_entropy_of_two_equal_levels_is_one_bit() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, 5:] = 255
    assert measure(Image.fromarray(image)).entropy == pytest.approx(1.0)


def test_blur_lowers_sharpness() -> None:
    sharp = scene(3, (400, 300))
    blurred = sharp.filter(ImageFilter.GaussianBlur(4))
    assert measure(blurred).sharpness < measure(sharp).sharpness / 5


def test_laplacian_of_a_linear_ramp_is_zero() -> None:
    ramp = np.tile(np.arange(20, dtype=np.float64), (20, 1))
    assert laplacian_variance(ramp) == pytest.approx(0.0)


def test_measures_use_the_fixed_scale_so_sizes_compare() -> None:
    big = checker(2048, 64)
    small = big.resize((1024, 1024), Image.Resampling.BOX)
    m_big, m_small = measure(big), measure(small)
    assert (m_big.measured_width, m_big.measured_height) == (MEASURE_SIDE, MEASURE_SIDE)
    assert m_big.sharpness == pytest.approx(m_small.sharpness, rel=0.05)
    # without the fixed scale the variance of the Laplacian differs a lot
    raw_big = laplacian_variance(np.asarray(big.convert("L"), dtype=np.float64))
    raw_small = laplacian_variance(np.asarray(small.convert("L"), dtype=np.float64))
    assert raw_big != pytest.approx(raw_small, rel=0.05)


def test_small_images_are_never_enlarged() -> None:
    copy = working_copy(noise((100, 50)))
    assert copy.size == (100, 50)
    assert working_copy(noise((1024, 256))).size == (512, 128)


def test_dominant_colours_are_deterministic_and_ordered() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:, :30] = (200, 20, 20)
    image[:, 30:] = (20, 20, 200)
    first = dominant_colours(image, seed=0)
    assert first == dominant_colours(image, seed=0)
    assert first[0][0] == "#c81414"
    assert first[0][1] == pytest.approx(0.75)
    assert measure(Image.fromarray(image)).dominant_colour == "#c81414"


def test_slightly_noisy_grey_jpeg_is_still_greyscale() -> None:
    grey = gradient((200, 100))
    decoded = decode_image(encode(grey, "JPEG", quality=70))
    assert measure(decoded.image).is_grayscale
    assert not measure(scene(1)).is_grayscale


# --- hashes -------------------------------------------------------------------------------------


def test_dhash_of_a_left_to_right_ramp_is_all_ones() -> None:
    assert dhash(gradient((90, 80))) == -1  # 64 bits set, as a signed int64
    assert to_hex(dhash(gradient((90, 80)))) == "f" * 16


def test_hashes_are_signed_int64() -> None:
    for image in (scene(0), noise(), checker()):
        for value in (phash(image), dhash(image)):
            assert -(2**63) <= value < 2**63


def test_near_copies_have_close_hashes_and_others_far() -> None:
    original = scene(7, (320, 240))
    recompressed = decode_image(encode(original, "JPEG", quality=60)).image
    resized = original.resize((160, 120), Image.Resampling.BILINEAR)
    other = scene(8, (320, 240))
    assert hamming(phash(original), phash(recompressed)) <= 6
    assert hamming(dhash(original), dhash(recompressed)) <= 10
    assert hamming(phash(original), phash(resized)) <= 6
    assert hamming(phash(original), phash(other)) > 10


def test_hamming_many_matches_hamming() -> None:
    values = np.array([phash(scene(i)) for i in range(5)], dtype=np.int64)
    distances = hamming_many(int(values[0]), values)
    assert distances.tolist() == [hamming(int(values[0]), int(v)) for v in values]
    assert distances[0] == 0


def test_pixel_sha_ignores_the_file_format() -> None:
    image = noise((30, 20), seed=4)
    as_png = decode_image(encode(image, "PNG")).image
    as_bmp = decode_image(encode(image, "BMP")).image
    assert pixel_sha(as_png) == pixel_sha(as_bmp)
    assert encode(image, "PNG") != encode(image, "BMP")
    assert pixel_sha(as_png) != pixel_sha(decode_image(encode(noise((30, 20), 5), "PNG")).image)


# --- descriptors --------------------------------------------------------------------------------


def test_descriptor_sizes() -> None:
    d = classic_descriptor(scene(2))
    assert d.hsv.shape == (HSV_SIZE,) == (72,)
    assert d.hog.shape == (HOG_SIZE,) == (1_764,)
    assert d.lbp.shape == (LBP_SIZE,) == (236,)
    assert d.vector.shape == (CLASSIC_SIZE,) == (2_072,)
    assert d.vector.dtype == np.float32


def test_hsv_histogram_of_pure_red() -> None:
    hist = hsv_histogram(flat((255, 0, 0)))
    assert hist.sum() == pytest.approx(1.0)
    assert hist[8] == pytest.approx(1.0)  # hue bin 0, saturation bin 2, value bin 2


def test_hog_of_vertical_stripes_points_across_them() -> None:
    stripes = np.zeros((128, 128, 3), dtype=np.uint8)
    stripes[:, ::8] = 255
    stripes[:, 1::8] = 255
    h = hog(Image.fromarray(stripes)).reshape(-1, 9)
    # horizontal gradients (angle 0) fall in the first and last bins, split at the 0/180 seam
    assert (h[:, 0] + h[:, 8]).sum() > 0.9 * h.sum()
    assert hog(flat()).sum() == 0.0


def test_lbp_of_a_flat_image_is_one_uniform_pattern() -> None:
    hist = lbp(flat()).reshape(4, 59)
    assert np.allclose(hist.sum(axis=1), 1.0)
    assert np.allclose(hist[:, 57], 1.0)  # code 255: every neighbour >= the centre
    codes = lbp_codes(np.full((5, 5), 7.0))
    assert (codes == 255).all()


def test_descriptors_tell_textures_apart() -> None:
    a, b = classic_descriptor(checker(256, 8)).vector, classic_descriptor(noise((256, 256))).vector
    a2 = classic_descriptor(checker(256, 8).rotate(0)).vector
    assert np.linalg.norm(a - a2) == pytest.approx(0.0)
    assert np.linalg.norm(a - b) > 1.0
