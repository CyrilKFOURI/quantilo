"""Image detection and metadata (084): tiny binary fixtures written here, one per format, plus
truncated, hostile and GPS-carrying files. No pixel is ever decoded, no GPS value ever leaves."""

import json
import struct
import zlib
from pathlib import Path

import pytest

import quantilo as dx
from quantilo.errors import NotFoundError
from quantilo.ingestion import images
from quantilo.ingestion.images import MAX_BYTES_READ, ImageInfo, inspect_image, sniff_image

# --- builders -----------------------------------------------------------------------------------


def png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def png(
    width: int = 640,
    height: int = 480,
    bit_depth: int = 8,
    color_type: int = 6,
    chunks: tuple[tuple[bytes, bytes], ...] = (),
    idat: bytes = b"\x78\x9c\x03\x00\x00\x00\x00\x01",
) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    body = png_chunk(b"IHDR", ihdr)
    for kind, data in chunks:
        body += png_chunk(kind, data)
    body += png_chunk(b"IDAT", idat) + png_chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + body


TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1}
Entry = tuple[int, int, object]  # tag, type, value(s)


def _encode(order: str, kind: int, value: object) -> tuple[bytes, int]:
    if kind == 2:
        raw = str(value).encode() + b"\0"
        return raw, len(raw)
    values = value if isinstance(value, list | tuple) else [value]
    if kind == 5:
        raw = b"".join(struct.pack(order + "II", n, d) for n, d in values)  # type: ignore[misc]
        return raw, len(values)
    code = {1: "B", 3: "H", 4: "I", 7: "B"}[kind]
    return b"".join(struct.pack(order + code, v) for v in values), len(values)  # type: ignore[arg-type]


def _ifd(order: str, entries: list[Entry], offset: int, next_ifd: int = 0) -> bytes:
    """One IFD placed at `offset`, its out-of-line values right after it."""
    entries = sorted(entries, key=lambda e: e[0])
    data_at = offset + 2 + 12 * len(entries) + 4
    table, data = b"", b""
    for tag, kind, value in entries:
        raw, count = _encode(order, kind, value)
        if len(raw) <= 4:
            field = raw.ljust(4, b"\0")
        else:
            field = struct.pack(order + "I", data_at + len(data))
            data += raw + (b"\0" if len(raw) % 2 else b"")
        table += struct.pack(order + "HHI", tag, kind, count) + field
    return (
        struct.pack(order + "H", len(entries)) + table + struct.pack(order + "I", next_ifd) + data
    )


def tiff(
    ifd0: list[Entry],
    exif: list[Entry] | None = None,
    gps: list[Entry] | None = None,
    pages: int = 1,
    order: str = "<",
) -> bytes:
    """A TIFF stream (a .tif file or the body of an EXIF segment) with optional Exif and GPS IFDs
    and `pages` chained copies of IFD0."""

    def layout(pointers: dict[int, int]) -> list[bytes]:
        top = [*ifd0, *((tag, 4, at) for tag, at in pointers.items())]
        blocks, at = [], 8
        for page in range(pages):
            size = len(_ifd(order, top, at))
            nxt = at + size if page < pages - 1 else 0
            blocks.append(_ifd(order, top, at, nxt))
            at += size
        return blocks

    pointers = {}
    if exif is not None:
        pointers[34665] = 0
    if gps is not None:
        pointers[34853] = 0
    blocks = layout(pointers)
    at = 8 + sum(len(b) for b in blocks)
    sub = b""
    if exif is not None:
        pointers[34665] = at
        block = _ifd(order, exif, at)
        sub += block
        at += len(block)
    if gps is not None:
        pointers[34853] = at
        sub += _ifd(order, gps, at)
    blocks = layout(pointers)
    head = (b"II*\0" if order == "<" else b"MM\0*") + struct.pack(order + "I", 8)
    return head + b"".join(blocks) + sub


GPS = [  # Paris, distinctive numbers that must never appear in a result
    (1, 2, "N"),
    (2, 5, [(48, 1), (51, 1), (241337, 10000)]),
    (3, 2, "E"),
    (4, 5, [(2, 1), (17, 1), (409911, 10000)]),
]
CAMERA = [
    (271, 2, "Canon"),
    (272, 2, "Canon EOS R5"),
    (274, 3, 6),
    (282, 5, [(300, 1)]),
    (283, 5, [(300, 1)]),
    (296, 3, 2),
]


def jpeg_segment(marker: int, data: bytes) -> bytes:
    return bytes([0xFF, marker]) + struct.pack(">H", len(data) + 2) + data


def jpeg(
    width: int = 4000,
    height: int = 3000,
    components: int = 3,
    exif: bytes | None = None,
    jfif_units: int | None = None,
    density: int = 72,
    sof: int = 0xC0,
) -> bytes:
    body = b"\xff\xd8"
    if jfif_units is not None:
        body += jpeg_segment(
            0xE0,
            b"JFIF\0\x01\x02"
            + bytes([jfif_units])
            + struct.pack(">HH", density, density)
            + b"\0\0",
        )
    if exif is not None:
        body += jpeg_segment(0xE1, b"Exif\0\0" + exif)
    body += jpeg_segment(0xDB, bytes(65))  # a quantisation table
    comps = b"".join(bytes([i + 1, 0x11, 0]) for i in range(components))
    body += jpeg_segment(
        sof, bytes([8]) + struct.pack(">HH", height, width) + bytes([components]) + comps
    )
    body += jpeg_segment(0xDA, bytes([components]) + bytes(2 * components) + b"\x00\x3f\x00")
    return body + b"\x12\x34\x56\x78" + b"\xff\xd9"


def gif(
    width: int = 32, height: int = 16, frames: int = 1, loop: bool = False, block: int = 2
) -> bytes:
    body = b"GIF89a" + struct.pack("<HHBBB", width, height, 0xF1, 0, 0) + bytes(3 * 4)
    if loop:
        body += b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00"
    frame = b"\x21\xf9\x04\x04\x0a\x00\x00\x00"  # graphic control: delay
    frame += b"\x2c" + struct.pack("<HHHHB", 0, 0, width, height, 0)
    frame += b"\x02" + bytes([block]) + bytes(block) + b"\x00"
    return body + frame * frames + b"\x3b"


def riff_chunk(kind: bytes, data: bytes) -> bytes:
    return kind + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")


def webp(chunks: list[bytes]) -> bytes:
    body = b"WEBP" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def vp8(width: int, height: int) -> bytes:
    return riff_chunk(
        b"VP8 ", b"\x30\x01\x00\x9d\x01\x2a" + struct.pack("<HH", width, height) + bytes(8)
    )


def vp8x(width: int, height: int, flags: int) -> bytes:
    return riff_chunk(
        b"VP8X",
        bytes([flags, 0, 0, 0])
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little"),
    )


def bmp(width: int = 20, height: int = -10, bpp: int = 24, ppm: int = 2835) -> bytes:
    dib = struct.pack("<IiiHHIIiiII", 40, width, height, 1, bpp, 0, 0, ppm, ppm, 0, 0)
    return b"BM" + struct.pack("<IHHI", 54 + len(dib), 0, 0, 54) + dib + bytes(16)


def box(kind: bytes, payload: bytes, full: bool = False) -> bytes:
    body = (b"\0\0\0\0" if full else b"") + payload
    return struct.pack(">I", 8 + len(body)) + kind + body


def heif(major: bytes, compatible: tuple[bytes, ...], sizes: tuple[tuple[int, int], ...]) -> bytes:
    ftyp = box(b"ftyp", major + b"\0\0\0\0" + b"".join(compatible))
    props = b"".join(box(b"ispe", struct.pack(">II", w, h), full=True) for w, h in sizes)
    props += box(b"pixi", bytes([3, 10, 10, 10]), full=True)
    meta = box(
        b"meta", box(b"hdlr", bytes(20), full=True) + box(b"iprp", box(b"ipco", props)), full=True
    )
    return ftyp + meta + box(b"mdat", bytes(16))


def look(data: bytes, name: str | None = None) -> ImageInfo:
    return inspect_image(data, name=name)


# --- formats ------------------------------------------------------------------------------------


def test_png_reads_dimensions_mode_depth_and_dpi() -> None:
    info = look(png(chunks=((b"pHYs", struct.pack(">IIB", 3780, 3780, 1)),)), "chart.png")
    assert info.is_image and info.format == "png" and info.mime == "image/png"
    assert (info.width, info.height) == (640, 480)
    assert info.color_mode == "RGBA" and info.channels == 4
    assert info.bit_depth == 8 and info.bits_per_pixel == 32
    assert info.frames == 1 and not info.animated
    assert info.dpi == (96.01, 96.01)
    assert info.aspect_ratio == "4:3"
    assert info.sentence.startswith("PNG image, 640 × 480 pixels (0.31 megapixels, 4:3), RGBA")


@pytest.mark.parametrize(
    ("color_type", "bit_depth", "mode", "channels"),
    [(0, 16, "grayscale", 1), (2, 8, "RGB", 3), (3, 4, "palette", 1), (4, 8, "grayscale+alpha", 2)],
)
def test_png_colour_types(color_type: int, bit_depth: int, mode: str, channels: int) -> None:
    info = look(png(bit_depth=bit_depth, color_type=color_type))
    assert (info.color_mode, info.channels, info.bit_depth) == (mode, channels, bit_depth)


def test_apng_frames_come_from_actl() -> None:
    info = look(png(chunks=((b"acTL", struct.pack(">II", 24, 0)),)))
    assert info.animated and info.frames == 24
    assert "Animated: 24 frames." in info.sentence


def test_png_exif_after_a_large_idat_is_found_without_reading_the_pixels() -> None:
    exif = tiff([(271, 2, "Pixel"), (272, 2, "Pixel 8")])
    data = png(idat=bytes(3 * MAX_BYTES_READ), chunks=())
    data = data[:-12] + png_chunk(b"eXIf", exif) + data[-12:]  # eXIf after IDAT, before IEND
    info = look(data)
    assert info.exif is not None and info.exif.model == "Pixel 8"
    assert info.bytes_read < 4096


def test_jpeg_with_exif_camera_date_orientation_and_resolution() -> None:
    exif = tiff(CAMERA, exif=[(36867, 2, "2024:05:01 14:03:27")])
    info = look(jpeg(exif=exif), "IMG_0042.jpg")
    assert info.format == "jpeg" and (info.width, info.height) == (4000, 3000)
    assert info.color_mode == "RGB" and info.bit_depth == 8 and info.bits_per_pixel == 24
    assert info.dpi == (300.0, 300.0)
    assert info.megapixels == 12.0
    assert info.exif is not None
    assert (info.exif.make, info.exif.model) == ("Canon", "Canon EOS R5")
    assert info.exif.date_taken == "2024-05-01T14:03:27"
    assert info.exif.orientation == 6 and info.exif.orientation_label == "rotated 90° clockwise"
    assert (info.display_width, info.display_height) == (3000, 4000)
    assert not info.exif.has_location
    assert "Taken on 2024-05-01 at 14:03 with a Canon EOS R5." in info.sentence
    assert "shown 3,000 × 4,000" in info.sentence


def test_jpeg_gps_is_detected_and_never_exposed() -> None:
    exif = tiff(CAMERA, exif=[(36867, 2, "2024:05:01 14:03:27")], gps=GPS)
    info = look(jpeg(exif=exif), "home.jpg")
    assert info.exif is not None and info.exif.has_location
    assert "Contains GPS location (not shown)." in info.sentence
    dumped = json.dumps(dx.to_dict(info)) + repr(info)
    for leak in ("241337", "24.1337", "409911", "40.9911", "48.85", "2.29", "[48", "(48"):
        assert leak not in dumped
    assert not any("gps" in key.lower() and key != "has_location" for key in dx.to_dict(info))


def test_big_endian_exif_and_date_fallback() -> None:
    exif = tiff(
        [(271, 2, "NIKON CORPORATION"), (272, 2, "D850"), (306, 2, "2023:12:31 23:59:00")],
        order=">",
    )
    info = look(jpeg(exif=exif))
    assert info.exif is not None
    assert info.exif.date_taken == "2023-12-31T23:59:00"
    assert "with a NIKON CORPORATION D850" in info.sentence


def test_jfif_density_in_centimetres_and_without_unit() -> None:
    assert look(jpeg(jfif_units=2, density=118)).dpi == (299.72, 299.72)
    assert look(jpeg(jfif_units=0, density=1)).dpi is None


def test_jpeg_grayscale_cmyk_and_progressive() -> None:
    assert look(jpeg(components=1)).color_mode == "grayscale"
    assert look(jpeg(components=4)).color_mode == "CMYK"
    assert look(jpeg(sof=0xC2)).width == 4000  # progressive
    assert look(jpeg(sof=0xC4)).width is None  # C4 is a Huffman table, not a frame


def test_gif_still_and_animated() -> None:
    still = look(gif())
    assert (still.width, still.height, still.frames, still.animated) == (32, 16, 1, False)
    assert still.color_mode == "palette" and still.bit_depth == 2
    moving = look(gif(frames=3, loop=True))
    assert moving.frames == 3 and moving.animated and not moving.frames_at_least


def test_webp_lossy_lossless_and_animated() -> None:
    lossy = look(webp([vp8(300, 200)]))
    assert (lossy.format, lossy.width, lossy.height, lossy.color_mode) == ("webp", 300, 200, "RGB")
    bits = (100 - 1) | ((50 - 1) << 14) | (1 << 28)
    lossless = look(webp([riff_chunk(b"VP8L", b"\x2f" + struct.pack("<I", bits) + bytes(4))]))
    assert (lossless.width, lossless.height, lossless.color_mode) == (100, 50, "RGBA")
    frame = riff_chunk(b"ANMF", bytes(16) + vp8(10, 10))
    animated = look(webp([vp8x(400, 300, 0x12), riff_chunk(b"ANIM", bytes(6)), frame, frame]))
    assert (animated.width, animated.height) == (400, 300)
    assert animated.animated and animated.frames == 2 and animated.color_mode == "RGBA"


def test_webp_exif_chunk() -> None:
    exif = b"Exif\0\0" + tiff([(272, 2, "iPhone 15")], gps=GPS)
    info = look(webp([vp8x(8, 8, 0x08), vp8(8, 8), riff_chunk(b"EXIF", exif)]))
    assert info.exif is not None and info.exif.model == "iPhone 15" and info.exif.has_location


def test_bmp_top_down_and_dpi() -> None:
    info = look(bmp())
    assert (info.format, info.width, info.height) == ("bmp", 20, 10)
    assert info.color_mode == "RGB" and info.bits_per_pixel == 24 and info.dpi == (72.01, 72.01)
    assert look(bmp(bpp=8)).color_mode == "palette"


def test_text_starting_with_bm_is_not_a_bmp() -> None:
    assert sniff_image(b"BMW,Audi,Renault\n1,2,3\n") is None


def test_tiff_file_dimensions_mode_and_pages() -> None:
    ifd0 = [
        (256, 4, 1200),
        (257, 3, 800),
        (258, 3, [8, 8, 8]),
        (262, 3, 2),
        (277, 3, 3),
        *CAMERA[:2],
    ]
    info = look(tiff(ifd0, pages=3), "scan.tif")
    assert (info.format, info.width, info.height) == ("tiff", 1200, 800)
    assert info.color_mode == "RGB" and info.bit_depth == 8 and info.bits_per_pixel == 24
    assert info.frames == 3 and not info.animated
    assert "3 pages." in info.sentence
    assert info.exif is not None and info.exif.make == "Canon"
    big = look(tiff(ifd0, order=">"))
    assert (big.width, big.color_mode) == (1200, "RGB")


def test_heic_and_avif_take_the_largest_ispe() -> None:
    info = look(heif(b"heic", (b"mif1", b"heic"), ((512, 384), (4032, 3024))), "IMG_1.HEIC")
    assert (info.format, info.mime) == ("heic", "image/heic")
    assert (info.width, info.height, info.bit_depth, info.channels) == (4032, 3024, 10, 3)
    assert any("EXIF in HEIF" in note for note in info.notes)
    avif = look(heif(b"avif", (b"mif1", b"avif"), ((64, 64),)))
    assert (avif.format, avif.width, avif.animated) == ("avif", 64, False)
    assert look(heif(b"avis", (b"avif", b"msf1"), ((64, 64),))).animated


def test_video_ftyp_is_named_not_an_image() -> None:
    info = look(box(b"ftyp", b"isom\0\0\0\0isomavc1") + box(b"mdat", bytes(8)))
    assert not info.is_image and "video" in (info.reason or "")


# --- refusals -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "svg",
    [
        b'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
        b"\xef\xbb\xbf  \n<svg width='10'/>",
        b'<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "x">\n<SVG></SVG>',
    ],
)
def test_svg_is_refused_with_a_reason(svg: bytes) -> None:
    info = look(svg, "logo.svg")
    assert not info.is_image and info.format == "svg"
    assert info.reason == "SVG is XML and can carry scripts: it is not treated as an image."
    assert info.sentence == info.reason
    assert info.width is None and info.vision == ()


def test_other_files_are_not_images() -> None:
    csv = look(b"a,b\n1,2\n", "data.csv")
    assert not csv.is_image and csv.format is None
    assert "not a recognised image" in (csv.reason or "")
    assert "PDF" in (look(b"%PDF-1.7\n").reason or "")
    assert look(b"").reason == "The file is empty."


# --- damaged and hostile files ------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        png()[:20],
        png()[:8],
        jpeg()[:30],
        b"\xff\xd8\xff\xe1\xff\xff" + b"Exif\0\0II",  # APP1 claims 64 KiB, the file stops
        gif()[:10],
        gif(frames=2)[:-12],
        webp([vp8(300, 200)])[:24],
        bmp()[:20],
        tiff([(256, 4, 10)])[:12],
        b"II*\0" + struct.pack("<I", 10**9),  # IFD0 far outside the file
        heif(b"heic", (b"mif1",), ((10, 10),))[:40],
        b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 0xFFFFFFF0) + b"IHDR",
    ],
)
def test_damaged_files_never_raise(data: bytes) -> None:
    info = look(data)
    assert info.is_image
    assert isinstance(info.sentence, str) and info.sentence
    if info.width is None:
        assert info.notes
        assert "could not be read" in info.sentence


def _fixtures() -> list[bytes]:
    exif = tiff(CAMERA, exif=[(36867, 2, "2024:05:01 14:03:27")], gps=GPS)
    frame = riff_chunk(b"ANMF", bytes(16) + vp8(10, 10))
    return [
        png(chunks=((b"acTL", struct.pack(">II", 2, 0)), (b"eXIf", exif))),
        jpeg(exif=exif, jfif_units=1),
        gif(frames=3, loop=True),
        webp([vp8x(40, 30, 0x1A), riff_chunk(b"ANIM", bytes(6)), frame, riff_chunk(b"EXIF", exif)]),
        bmp(),
        tiff([(256, 4, 12), (257, 3, 8), (258, 3, [8, 8, 8]), (262, 3, 2), *CAMERA], gps=GPS),
        heif(b"heic", (b"mif1",), ((64, 48),)),
    ]


def test_every_truncation_and_random_damage_never_raises() -> None:
    import random

    rng = random.Random(84)
    for data in _fixtures():
        for cut in range(len(data)):
            info = look(data[:cut])
            assert info.sentence
            if info.exif is not None:
                assert "241337" not in repr(info)
        for _ in range(300):
            damaged = bytearray(data)
            for _ in range(rng.randint(1, 6)):
                damaged[rng.randrange(len(damaged))] = rng.randrange(256)
            assert look(bytes(damaged)).sentence


def test_truncated_png_says_why() -> None:
    info = look(png()[:20])
    assert info.format == "png" and info.width is None
    assert "The header is truncated or damaged: dimensions unknown." in info.notes


def test_truncated_gif_counts_at_least() -> None:
    info = look(gif(frames=3)[:-14])
    assert info.frames_at_least and info.frames == 2


def test_exif_bounds_entries_and_pointers() -> None:
    # An IFD that claims 60 000 entries, one value pointing outside the segment, a loop to itself.
    order = "<"
    entries = struct.pack(order + "HHI", 271, 2, 50) + struct.pack(order + "I", 10**6)
    entries += struct.pack(order + "HHI", 274, 3, 1) + struct.pack(order + "HH", 3, 0)
    ifd = struct.pack(order + "H", 60000) + entries + struct.pack(order + "I", 8)
    exif = b"II*\0" + struct.pack(order + "I", 8) + ifd
    info = look(jpeg(exif=exif))
    assert info.exif is not None
    assert info.exif.make is None and info.exif.orientation == 3
    assert info.width == 4000


def test_tiff_page_loop_is_cut() -> None:
    data = bytearray(tiff([(256, 3, 5), (257, 3, 5)], pages=2))
    # point the second page back to the first
    second = 8 + 2 + 12 * 2 + 4
    next_of_second = second + 2 + 12 * 2
    data[next_of_second : next_of_second + 4] = struct.pack("<I", 8)
    info = look(bytes(data))
    assert info.frames == 2


def test_reads_are_bounded_on_a_huge_gif(tmp_path: Path) -> None:
    path = tmp_path / "huge.gif"
    path.write_bytes(gif(frames=40_000, block=60))  # ~3 MB, ~40 000 sub-blocks
    info = inspect_image(path)
    assert info.animated and info.frames_at_least and info.frames is not None
    assert info.bytes_read <= MAX_BYTES_READ
    assert any("limit" in note for note in info.notes)
    assert info.size_bytes == path.stat().st_size


def test_over_100_megapixels_is_not_previewable() -> None:
    info = look(png(width=20_000, height=10_000))
    assert not info.previewable and info.megapixels == 200.0
    assert "no preview" in info.sentence


# --- sentence, extension, vision, facade ----------------------------------------------------------


def test_extension_that_lies_is_named() -> None:
    info = look(png(), "logo.jpg")
    assert info.format == "png"
    assert "The name says .jpg but the content is PNG." in info.sentence
    assert "The name says" not in look(png(), "logo.PNG").sentence


def test_aspect_ratio_falls_back_to_a_decimal() -> None:
    assert look(png(width=1920, height=1080)).aspect_ratio == "16:9"
    assert look(png(width=1366, height=768)).aspect_ratio == "1.78:1"


def test_vision_tools_are_listed_for_later() -> None:
    info = look(png())
    assert [tool.key for tool in info.vision] == [
        "classification",
        "object_detection",
        "ocr",
        "embeddings",
    ]
    assert all(tool.status == "coming_later" for tool in info.vision)


def test_sniff_image_formats() -> None:
    assert sniff_image(png()) == "png"
    assert sniff_image(jpeg()) == "jpeg"
    assert sniff_image(gif()) == "gif"
    assert sniff_image(webp([vp8(2, 2)])) == "webp"
    assert sniff_image(bmp()) == "bmp"
    assert sniff_image(tiff([(256, 3, 1)])) == "tiff"
    assert sniff_image(heif(b"heic", (), ((1, 1),))) == "heic"
    assert sniff_image(b"<svg/>") is None
    assert images.MIME_TYPES["webp"] == "image/webp"


def test_facade_reads_a_path(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpeg"
    path.write_bytes(jpeg(exif=tiff(CAMERA, gps=GPS)))
    info = dx.inspect_image(path)
    assert info.name == "photo.jpeg" and info.format == "jpeg"
    assert info.size_bytes == path.stat().st_size
    assert dx.to_dict(info)["exif"]["has_location"] is True
    with pytest.raises(NotFoundError):
        dx.inspect_image(tmp_path / "missing.png")
