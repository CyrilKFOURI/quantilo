"""120 - the vision bench scripts, without network: download guards, file checks, PNG reader."""

import hashlib
import importlib.util
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


datasets = load("vision_datasets")
bench = load("vision_bench")


def test_destination_inside_the_repository_is_refused():
    with pytest.raises(datasets.DatasetError, match="never committed"):
        datasets.check_destination(datasets.ROOT / "data" / "vision")
    assert not datasets.DEFAULT_DEST.resolve().is_relative_to(datasets.ROOT)


def test_noncommercial_set_needs_explicit_acceptance(tmp_path):
    with pytest.raises(datasets.DatasetError, match="accept-noncommercial"):
        datasets.get(datasets.MVTEC_BOTTLE, tmp_path)


def test_mislabelled_mirror_is_refused():
    fake = datasets.Dataset(
        name="x", repo="MSherbinii/mvtec-ad-metal-nut", revision="0", licence="mit",
        licence_name="MIT", task="", citation="", use="",
    )  # fmt: skip
    with pytest.raises(datasets.DatasetError, match="wrong licence"):
        datasets.check_licence(fake)


def test_size_limit():
    with pytest.raises(datasets.DatasetError, match="1 GB"):
        datasets.check_space(Path("."), 2_000_000_000)


def test_file_checks_by_sha256_and_git_blob(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")
    sha = hashlib.sha256(b"hello").hexdigest()
    blob = hashlib.sha1(b"blob 5\0hello").hexdigest()
    assert datasets._matches(path, datasets.FileSpec("f.bin", 5, sha256=sha))
    assert datasets._matches(path, datasets.FileSpec("f.bin", 5, git_sha1=blob))
    assert not datasets._matches(path, datasets.FileSpec("f.bin", 5, sha256="0" * 64))
    assert not datasets._matches(path, datasets.FileSpec("f.bin", 6, sha256=sha))


def test_pinned_sets_stay_under_the_limit():
    for d in datasets.DATASETS.values():
        assert (d.expected_bytes or sum(f.size for f in d.files)) < datasets.MAX_BYTES
        assert len(d.revision) == 40


def encode_png(pixels: np.ndarray, filters: list[int]) -> bytes:
    """A PNG written by hand with the given filter per row (reference: the PNG specification)."""
    height, width = pixels.shape[:2]
    bpp = 1 if pixels.ndim == 2 else pixels.shape[2]
    colour = {1: 0, 3: 2, 4: 6}[bpp]
    rows = pixels.reshape(height, width * bpp).astype(int)
    raw = b""
    previous = np.zeros(width * bpp, dtype=int)
    for y, kind in enumerate(filters):
        line = rows[y]
        a = np.r_[np.zeros(bpp, dtype=int), line[:-bpp]]
        c = np.r_[np.zeros(bpp, dtype=int), previous[:-bpp]]
        if kind == 0:
            out = line
        elif kind == 1:
            out = line - a
        elif kind == 2:
            out = line - previous
        elif kind == 3:
            out = line - (a + previous) // 2
        else:
            p = a + previous - c
            pa, pb, pc = abs(p - a), abs(p - previous), abs(p - c)
            pred = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, previous, c))
            out = line - pred
        raw += bytes([kind]) + bytes((out % 256).astype(np.uint8))
        previous = line

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, colour, 0, 0, 0)
    return (
        bench.PNG_SIGNATURE
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_png_reader_all_filters(channels):
    rng = np.random.default_rng(channels)
    shape = (5, 7) if channels == 1 else (5, 7, channels)
    pixels = rng.integers(0, 256, size=shape).astype(np.uint8)
    data = encode_png(pixels, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(bench.png_decode(data), pixels)
    assert bench.png_header(data)[:2] == (7, 5)
