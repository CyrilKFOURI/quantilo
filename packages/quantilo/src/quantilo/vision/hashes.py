"""Image fingerprints (103, § 4.1 of the study): perceptual pHash and dHash, and the pixel SHA.

- **pHash**, the classic method: the luma resized to 32 x 32, a 2-D DCT (`scipy.fft.dctn`, type II,
  orthonormal), the 8 x 8 block of the lowest frequencies, each coefficient compared with the median
  of the block **without the DC term** (the mean brightness says nothing of the content). 64 bits.
- **dHash**: the luma resized to 9 x 8, each pixel compared with its right-hand neighbour. 64 bits.

Both are stored as signed int64 (the Parquet type), bit 63 first. Two images are near copies when
their Hamming distances are small (R-VQ-10 in 105: pHash <= 6 and dHash <= 10).

- **pixel_sha**: the SHA-256 of the decoded pixels (mode, size, then the bytes), so a PNG saved
  again as BMP is still an exact duplicate (R-VQ-09), whatever the file bytes.
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image
from scipy.fft import dctn

HASH_BITS = 64
_MASK = (1 << HASH_BITS) - 1


def _grey(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    grey = image if image.mode == "L" else image.convert("L")
    return np.asarray(grey.resize(size, Image.Resampling.LANCZOS), dtype=np.float64)


def _pack(bits: np.ndarray) -> int:
    """64 booleans (bit 63 first) as a signed int64."""
    value = 0
    for bit in np.asarray(bits, dtype=bool).ravel():
        value = (value << 1) | int(bit)
    return value - (1 << HASH_BITS) if value >= 1 << (HASH_BITS - 1) else value


def phash(image: Image.Image) -> int:
    """The perceptual hash (DCT) of an image, as a signed int64."""
    coefficients = np.asarray(dctn(_grey(image, (32, 32)), type=2, norm="ortho"))
    block = coefficients[:8, :8]
    median = float(np.median(block.ravel()[1:]))  # the DC term left out
    return _pack(block > median)


def dhash(image: Image.Image) -> int:
    """The difference hash of an image (9 x 8 luma, left < right), as a signed int64."""
    pixels = _grey(image, (9, 8))
    return _pack(pixels[:, 1:] > pixels[:, :-1])


def hamming(a: int, b: int) -> int:
    """The number of bits that differ between two 64-bit hashes."""
    return ((a ^ b) & _MASK).bit_count()


def hamming_many(value: int, others: np.ndarray) -> np.ndarray:
    """Hamming distances between one hash and an array of int64 hashes (numpy popcount)."""
    x = np.bitwise_xor(np.asarray(others, dtype=np.int64), np.int64(value)).view(np.uint64)
    return np.bitwise_count(x).astype(np.int64)


def to_hex(value: int) -> str:
    """A hash as 16 hexadecimal digits (the unsigned reading of the int64)."""
    return f"{value & _MASK:016x}"


def pixel_sha(image: Image.Image) -> str:
    """SHA-256 of the decoded pixels: mode, size and raw bytes."""
    digest = hashlib.sha256(f"{image.mode}:{image.size[0]}x{image.size[1]}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


__all__ = ["HASH_BITS", "dhash", "hamming", "hamming_many", "phash", "pixel_sha", "to_hex"]
