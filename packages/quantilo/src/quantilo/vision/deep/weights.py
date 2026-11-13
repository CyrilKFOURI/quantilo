"""Pretrained weights for the deep vision models (114): a frozen manifest, downloaded only when
the caller confirms, checked by SHA-256, read as safetensors.

- R-VDP-02: nothing is downloaded unless `download(key, confirm=True)`; without it,
  `ConfirmationRequired` carries the name, size and licence to show the user first.
- R-VDP-03: HTTPS only, from the URL of the manifest (a pinned Hugging Face revision); the size is
  capped while reading, the SHA-256 checked before the file takes its final name, and again on
  the first load in a process. Only safetensors files are read: no pickle, no `torch.hub`, no
  downloaded code is ever executed.

Standard library only at import time (no torch): `status` works without the extra."""

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantilo import hub

HF = hub.HF
ATTEMPTS = hub.ATTEMPTS
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
        # PatchCore and PaDiM (115): the torchvision IMAGENET1K_V1 weights of the papers.
        WeightSpec(
            "wide_resnet50_2",
            "Wide ResNet-50-2 (torchvision IMAGENET1K_V1)",
            "timm/wide_resnet50_2.tv_in1k",
            "343c5fa59ca8a131e0cdf42144a641b36a0387c2",
            "model.safetensors",
            275_835_296,
            "df6fb6c4824769769de18e14088475fd6ee94236849aa4e5d8022ba9d9a9a16c",
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


size_text = hub.size_text


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


file_sha256 = hub.file_sha256


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


Opener = hub.Opener


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
    try:
        hub.fetch(
            spec.url,
            spec.size,
            spec.sha256,
            file,
            title=spec.title,
            user_agent=USER_AGENT,
            progress=progress,
            opener=opener,
            attempts=attempts,
            pause=pause,
        )
    except hub.DownloadError as error:
        raise WeightsError(f"{error} (R-VDP-03)") from error
    stat = file.stat()
    _VERIFIED[file] = (stat.st_size, stat.st_mtime_ns)
    return file


def load_state_dict(key: str | WeightSpec) -> dict[str, Any]:
    """The tensors of a verified safetensors file (R-VDP-03). Needs the extra."""
    file = verify(key)
    from safetensors.torch import load_file  # pyright: ignore[reportMissingImports]

    return dict(load_file(str(file), device="cpu"))
