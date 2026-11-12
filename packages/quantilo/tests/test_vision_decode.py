"""103: safe decoding with Pillow (bombs, truncated files, SVG and HEIC refused, EXIF orientation,
no GPS kept, every pixel transformation announced)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from PIL import Image, ImageFile

from quantilo.vision.decode import (
    MAX_IMAGE_PIXELS,
    DecodedImage,
    DecodeFailure,
    ImageDecodeError,
    decode_image,
    try_decode,
)
from tests.vision_fixtures import (
    HEIC,
    SVG,
    encode,
    jpeg_with_exif,
    noise,
    png_bomb,
    truncated,
    write,
)


def test_png_decodes_to_rgb_without_metadata() -> None:
    decoded = decode_image(encode(noise((40, 30)), "PNG"))
    assert decoded.format == "png"
    assert (decoded.width, decoded.height) == (40, 30)
    assert decoded.image.mode == "RGB"
    assert decoded.image.info == {}
    assert decoded.array().shape == (30, 40, 3)
    assert decoded.notes == ()


def test_format_comes_from_the_bytes_not_the_name(tmp_path) -> None:
    path = write(tmp_path / "logo.jpg", encode(noise(), "PNG"))
    assert decode_image(path).format == "png"


def test_exif_orientation_applied_and_no_gps_kept() -> None:
    data = jpeg_with_exif(noise((40, 20)), orientation=6, gps=True)
    decoded = decode_image(data)
    # rotated 90 degrees: shown 20 wide and 40 high, as a viewer shows it
    assert (decoded.width, decoded.height) == (20, 40)
    assert decoded.image.size == (20, 40)
    assert decoded.notes[0].startswith("R-VIS-DEC-01")
    assert decoded.info.exif is not None and decoded.info.exif.has_location is True
    assert "exif" not in decoded.image.info and "icc_profile" not in decoded.image.info
    # nothing of the GPS block survives in what can be saved from the decoded image
    resaved = encode(decoded.image, "JPEG")
    assert b"Exif" not in resaved


def test_transparency_flattened_on_white() -> None:
    rgba = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
    decoded = decode_image(encode(rgba, "PNG"))
    assert decoded.has_alpha
    assert decoded.array()[0, 0].tolist() == [255, 255, 255]
    assert any(n.startswith("R-VIS-DEC-02") for n in decoded.notes)


def test_sixteen_bit_pixels_scaled_to_eight() -> None:
    values = np.full((10, 10), 65535, dtype=np.uint16)
    values[:5] = 0
    decoded = decode_image(encode(Image.fromarray(values), "PNG"))
    assert decoded.bit_depth == 16
    pixels = decoded.array()
    assert pixels[0, 0, 0] == 0 and pixels[9, 9, 0] == 255
    assert any(n.startswith("R-VIS-DEC-03") for n in decoded.notes)


def test_animated_gif_first_frame_and_frame_count() -> None:
    frames = [Image.new("RGB", (10, 10), c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    buffer = encode(frames[0], "GIF", save_all=True, append_images=frames[1:], duration=50)
    decoded = decode_image(buffer)
    assert decoded.frames == 3
    assert any(n.startswith("R-VIS-DEC-04") for n in decoded.notes)
    red = decoded.array()[0, 0]
    assert red[0] > 200 and red[1] < 50


def test_cmyk_converted_to_rgb_and_said() -> None:
    decoded = decode_image(encode(Image.new("CMYK", (8, 8), (0, 0, 0, 0)), "JPEG"))
    assert decoded.source_mode == "CMYK"
    assert any(n.startswith("R-VIS-DEC-06") for n in decoded.notes)


def test_large_jpeg_decoded_at_reduced_scale_keeps_its_true_size() -> None:
    data = encode(noise((2048, 1536)), "JPEG", quality=80)
    decoded = decode_image(data, reduce_to=512)
    assert (decoded.width, decoded.height) == (2048, 1536)
    assert decoded.scale >= 2
    assert max(decoded.image.size) < 2048
    assert any(n.startswith("R-VIS-DEC-05") for n in decoded.notes)


def test_decompression_bomb_refused_before_decoding() -> None:
    bomb = png_bomb(20_000, 20_000)  # 400 megapixels in a few kilobytes
    assert len(bomb) < 200_000
    start = time.perf_counter()
    failure = try_decode(bomb)
    assert time.perf_counter() - start < 2.0  # refused from the header, nothing decoded
    assert isinstance(failure, DecodeFailure)
    assert failure.kind == "too_large"
    assert failure.code == "R-VQ-01"
    assert "too large to decode safely" in failure.reason
    assert Image.MAX_IMAGE_PIXELS == MAX_IMAGE_PIXELS


def test_lower_pixel_limit_is_honoured() -> None:
    failure = try_decode(encode(noise((100, 100)), "PNG"), max_pixels=5_000)
    assert isinstance(failure, DecodeFailure) and failure.kind == "too_large"


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "BMP", "TIFF"])
def test_truncated_file_is_an_error_not_a_grey_image(fmt: str) -> None:
    data = truncated(encode(noise((200, 150)), fmt), 0.5)
    failure = try_decode(data)
    assert isinstance(failure, DecodeFailure)
    assert failure.kind in ("truncated", "corrupt")
    assert failure.code == "R-VQ-01"


def test_truncated_refused_even_if_the_caller_allowed_it() -> None:
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        failure = try_decode(truncated(encode(noise((200, 150)), "PNG"), 0.5))
        assert isinstance(failure, DecodeFailure)
        assert ImageFile.LOAD_TRUNCATED_IMAGES is True  # put back as the caller had it
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def test_svg_refused_with_its_reason() -> None:
    failure = try_decode(SVG)
    assert isinstance(failure, DecodeFailure)
    assert failure.kind == "svg"
    assert "scripts" in failure.reason


def test_heic_read_from_headers_only() -> None:
    failure = try_decode(HEIC)
    assert isinstance(failure, DecodeFailure)
    assert failure.kind == "unsupported"
    assert "convert to JPEG" in failure.reason


@pytest.mark.parametrize(
    "data",
    [b"", b"name,age\n", b"8BPS\x00\x01" + b"\x00" * 40, b"%!PS-Adobe-3.0 EPSF-3.0\n"],
    ids=["empty", "csv", "psd", "eps"],
)
def test_other_files_are_not_images(data: bytes) -> None:
    failure = try_decode(data)
    assert isinstance(failure, DecodeFailure)
    assert failure.kind == "not_image"


def test_damaged_header_is_corrupt() -> None:
    data = bytearray(encode(noise(), "PNG"))
    data[16:24] = b"\xff" * 8  # width and height of IHDR
    assert isinstance(try_decode(bytes(data)), DecodeFailure)


def test_decode_image_raises_with_the_failure() -> None:
    with pytest.raises(ImageDecodeError) as raised:
        decode_image(SVG)
    assert raised.value.failure.kind == "svg"


def test_file_too_big_is_not_read(tmp_path, monkeypatch) -> None:
    import quantilo.vision.decode as decode

    monkeypatch.setattr(decode, "MAX_FILE_BYTES", 100)
    path = write(tmp_path / "a.png", encode(noise(), "PNG"))
    failure = try_decode(path)
    assert isinstance(failure, DecodeFailure) and failure.kind == "file_too_big"


def test_decoded_image_is_a_decoded_image() -> None:
    assert isinstance(try_decode(encode(noise(), "BMP")), DecodedImage)
