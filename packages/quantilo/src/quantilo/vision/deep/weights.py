"""Pretrained weights for the deep vision models (114): a frozen manifest, downloaded only when
the caller confirms, checked by SHA-256, read as safetensors.

- R-VDP-02: nothing is downloaded unless `download(key, confirm=True)`; without it,
  `ConfirmationRequired` carries the name, size and licence to show the user first.
- R-VDP-03: HTTPS only, from the URL of the manifest (a pinned Hugging Face revision); the size is
  capped while reading, the SHA-256 checked before the file takes its final name, and again on
  the first load in a process. Only safetensors files are read: no pickle, no `torch.hub`, no
  downloaded code is ever executed.

Standard library only at import time (no torch): `status` works without the extra."""

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

HF = "https://huggingface.co"
ATTEMPTS = 8  # Hugging Face often resets long connections: resume with Range
TIMEOUT = 60
CHUNK = 1 << 20
USER_AGENT = "quantilo-vision-weights/1"


@dataclass(frozen=True)
class WeightSpec:
    key: str
    title: str
    repo: str  # Hugging Face model id
    revision: str  # pinned commit: the file cannot change under us
    filename: str
    size: int  # bytes
    sha256: str  # Git LFS object id = SHA-256 of the file
    licence: str
    note: str = ""  # conditions shown next to the licence

    @property
    def url(self) -> str:
        return f"{HF}/{self.repo}/resolve/{self.revision}/{self.filename}"

    @property
    def size_text(self) -> str:
        return size_text(self.size)

    @property
    def local_name(self) -> str:
        return f"{self.key}-{self.sha256[:12]}.safetensors"


_IMAGENET = "trained on ImageNet: ImageNet terms apply (non-commercial research use)"

MANIFEST: dict[str, WeightSpec] = {
    spec.key: spec
    for spec in (
        WeightSpec(
            "dinov2_s14",
            "DINOv2 ViT-S/14",
            "timm/vit_small_patch14_dinov2.lvd142m",
            "4610ca143709d58a633b6397a74412c2c3842454",
            "model.safetensors",
            88_240_510,
            "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081",
            "Apache-2.0",
        ),
        WeightSpec(
            "dinov2_b14",
            "DINOv2 ViT-B/14",
            "timm/vit_base_patch14_dinov2.lvd142m",
            "4685c99dabffe5affac90bd99dbffd25801ae58d",
            "model.safetensors",
            346_334_872,
            "55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b",
            "Apache-2.0",
        ),
        WeightSpec(
            "resnet18",
            "ResNet-18 (torchvision IMAGENET1K_V1)",
            "timm/resnet18.tv_in1k",
            "bbd144b3e5565108aad885f145491d11bc6ce807",
            "model.safetensors",
            46_807_446,
            "694f673df6520a3158624e8a89af086f59923ee4cd7436fe5bc3bc71d295ad81",
            "BSD-3-Clause",
            _IMAGENET,
        ),
        WeightSpec(
            "resnet50",
            "ResNet-50 (torchvision IMAGENET1K_V2)",
            "timm/resnet50.tv2_in1k",
            "10eea9779fc8a753af919714889bd7f60fba412d",
            "model.safetensors",
            102_469_840,
            "30f57ada2f100011a4bd38000df906d6354688a8ca599be580337151c3c5f6a3",
            "BSD-3-Clause",
            _IMAGENET,
        ),
    )
}


class WeightsError(RuntimeError):
    """A download or a weights file that cannot be trusted (R-VDP-03)."""


class ConfirmationRequired(RuntimeError):
    """R-VDP-02: the user has to agree to the download first (name, size, licence)."""

    def __init__(self, spec: WeightSpec) -> None:
        self.spec = spec
        super().__init__(
            f"Downloading {spec.title} needs your confirmation: {spec.size_text} "
            f"({spec.licence}) from {spec.url} (R-VDP-02)."
        )


class WeightsMissing(RuntimeError):
    """The weights are not on this machine (R-VDP-02): nothing was downloaded."""

    def __init__(self, spec: WeightSpec) -> None:
        self.spec = spec
        super().__init__(missing_reason(spec))


def size_text(size: int) -> str:
    mb = size / 1_000_000
    return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.0f} MB"


def missing_reason(spec: WeightSpec) -> str:
    return (
        f"Download the weights first: {spec.title}, {spec.size_text} ({spec.licence}). "
        "Needs a connection once (R-VDP-02)."
    )


def spec_of(key: str | WeightSpec) -> WeightSpec:
    if isinstance(key, WeightSpec):
        return key
    try:
        return MANIFEST[key]
    except KeyError:
        known = ", ".join(MANIFEST)
        raise KeyError(f"Unknown weights {key!r}: one of {known}.") from None


def root() -> Path:
    """`QUANTILO_VISION_WEIGHTS`, else next to the app's embeddings cache (its workspace), else
    the user's cache folder."""
    configured = os.environ.get("QUANTILO_VISION_WEIGHTS")
    if configured:
        return Path(configured)
    embeddings = os.environ.get("QUANTILO_EMBEDDINGS_CACHE")
    if embeddings:
        return Path(embeddings).parent / "models" / "vision"
    return Path.home() / ".cache" / "quantilo" / "models" / "vision"


def path(key: str | WeightSpec) -> Path:
    return root() / spec_of(key).local_name


def file_sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def downloaded(key: str | WeightSpec) -> bool:
    """The file is there with the right size (cheap; the SHA-256 is checked on load)."""
    spec = spec_of(key)
    file = path(spec)
    return file.is_file() and file.stat().st_size == spec.size


_VERIFIED: dict[Path, tuple[int, int]] = {}  # path -> (size, mtime_ns) checked in this process


def verify(key: str | WeightSpec) -> Path:
    """The local file, after checking its size and SHA-256 (once per process and version of the
    file). Raises `WeightsMissing` or `WeightsError`."""
    spec = spec_of(key)
    file = path(spec)
    if not file.is_file():
        raise WeightsMissing(spec)
    stat = file.stat()
    stamp = (stat.st_size, stat.st_mtime_ns)
    if _VERIFIED.get(file) == stamp:
        return file
    if stat.st_size != spec.size or file_sha256(file) != spec.sha256:
        raise WeightsError(
            f"{file.name} does not match the manifest of {spec.title} (size or SHA-256): "
            "delete it and download again (R-VDP-03)."
        )
    _VERIFIED[file] = stamp
    return file


Opener = Callable[[urllib.request.Request, float], Any]


def _open(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def _copy(response: BinaryIO, out: BinaryIO, digest: Any, have: int, limit: int, progress) -> int:
    while block := response.read(CHUNK):
        have += len(block)
        if have > limit:
            raise WeightsError(
                f"The server sent more than the {limit:,} bytes expected (R-VDP-03)."
            )
        digest.update(block)
        out.write(block)
        if progress is not None:
            progress(have, limit)
    return have


def download(
    key: str | WeightSpec,
    *,
    confirm: bool = False,
    progress: Callable[[int, int], None] | None = None,
    opener: Opener | None = None,
    attempts: int = ATTEMPTS,
    pause: float = 2.0,
) -> Path:
    """Fetch the weights once (R-VDP-02, R-VDP-03) and return the local file. `confirm=True` is
    the user's agreement, after seeing the size and licence; already there and valid: no
    connection. `opener` replaces `urllib.request.urlopen` (tests)."""
    spec = spec_of(key)
    if not spec.url.startswith("https://"):
        raise WeightsError(f"{spec.title}: only HTTPS downloads are allowed (R-VDP-03).")
    if sys.platform == "emscripten":
        raise WeightsError("Weights cannot be downloaded in the browser version.")
    file = path(spec)
    if downloaded(spec):
        try:
            return verify(spec)
        except WeightsError:
            file.unlink()
    if not confirm:
        raise ConfirmationRequired(spec)
    file.parent.mkdir(parents=True, exist_ok=True)
    part = file.with_suffix(".part")
    part.unlink(missing_ok=True)
    digest = hashlib.sha256()
    have = 0
    fetch = opener or _open
    last: BaseException | None = None
    for attempt in range(attempts):
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(spec.url, headers=headers)
        try:
            with fetch(request, TIMEOUT) as response, part.open("ab") as out:
                status = getattr(response, "status", 200)
                if have and status != 206:  # the server ignored Range: start again
                    out.truncate(0)
                    have, digest = 0, hashlib.sha256()
                have = _copy(response, out, digest, have, spec.size, progress)
            if have == spec.size:
                break
            last = WeightsError(f"Connection closed after {have:,} of {spec.size:,} bytes.")
        except WeightsError:
            part.unlink(missing_ok=True)
            raise
        except (OSError, urllib.error.URLError) as error:
            last = error
            # what reached the disk is kept: resume from there, with its digest
            have = part.stat().st_size if part.exists() else 0
            digest = hashlib.sha256()
            if have:
                with part.open("rb") as kept:
                    while block := kept.read(CHUNK):
                        digest.update(block)
        if attempt + 1 < attempts:
            time.sleep(pause)
    else:
        part.unlink(missing_ok=True)
        raise WeightsError(
            f"Download of {spec.title} failed after {attempts} attempts ({last}). "
            "Check the connection and try again."
        )
    if digest.hexdigest() != spec.sha256:
        part.unlink(missing_ok=True)
        raise WeightsError(
            f"{spec.title}: the SHA-256 of the downloaded file does not match the manifest: "
            "file deleted (R-VDP-03)."
        )
    os.replace(part, file)
    stat = file.stat()
    _VERIFIED[file] = (stat.st_size, stat.st_mtime_ns)
    return file


def load_state_dict(key: str | WeightSpec) -> dict[str, Any]:
    """The tensors of a verified safetensors file (R-VDP-03). Needs the extra."""
    file = verify(key)
    from safetensors.torch import load_file  # pyright: ignore[reportMissingImports]

    return dict(load_file(str(file), device="cpu"))
