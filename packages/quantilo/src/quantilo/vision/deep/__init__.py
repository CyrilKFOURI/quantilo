"""Deep vision on the desktop (114; study 2026-11-11 vision-et-xai §§ 2, 4.4, 6, 10, 11).

Optional extra `vision` (`pip install "quantilo[vision]"`: torch, torchvision, timm,
safetensors). Nothing in this package imports them at module level: without the extra the rest
of quantilo imports and runs, and `status()` says why each deep option is greyed (R-VDP-01).

- `weights`: frozen manifest, download on explicit confirmation, SHA-256, safetensors.
- `backbones`: DINOv2 ViT-S/14 (default), DINOv2 ViT-B/14, ResNet-18/50, frozen.
- `budget`: GPU (6 GB, fp16) or CPU, batch halved on out-of-memory, time estimate.
- `embed`: one vector per image (`embed_image_set`), 2D map coordinates.
- `probe`: linear probe and cosine kNN, scored by `vision.metrics` (numpy and scikit-learn only).

    from quantilo.vision import deep
    deep.status()                                   # what is possible here, and why not
    deep.weights.download("dinov2_s14", confirm=True)  # 88 MB, once
    vectors = deep.embed_image_set(images)
    result = deep.probe_image_set(images, vectors)
"""

import importlib.util
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

EXTRA = "vision"
INSTALL = 'pip install "quantilo[vision]"'
PACKAGES = ("torch", "torchvision", "timm", "safetensors")
BROWSER = "Not available in the browser version: deep models need the desktop app (R-VDP-01)."

if TYPE_CHECKING:
    from quantilo.vision.deep.embed import (
        Embeddings,
        embed_image_set,
        embed_images,
        map_coordinates,
    )
    from quantilo.vision.deep.probe import ProbeReport, probe_embeddings, probe_image_set


def missing() -> list[str]:
    """Packages of the extra that are not installed (never imported here)."""
    return [name for name in PACKAGES if importlib.util.find_spec(name) is None]


def installed() -> bool:
    return not missing()


@dataclass
class BackboneStatus:
    key: str
    title: str
    level: str  # "first" | "advanced"
    dims: int
    available: bool
    downloaded: bool
    size: str
    licence: str
    note: str
    reason: str | None


@dataclass
class DeepStatus:
    available: bool  # the extra is there (each backbone still needs its weights)
    installed: bool
    reason: str | None
    missing: list[str] = field(default_factory=list)
    backbones: list[BackboneStatus] = field(default_factory=list)

    def backbone(self, key: str) -> BackboneStatus:
        return next(b for b in self.backbones if b.key == key)


def _extra_reason(absent: list[str]) -> str:
    return f"Needs the optional extra vision ({INSTALL}); missing: {', '.join(absent)} (R-VDP-01)."


def status() -> DeepStatus:
    """Whether each deep option can be chosen here, and if not, why (R-VDP-01, R-VDP-02). Cheap:
    no torch import, no network, no hashing."""
    from quantilo.vision.deep import weights
    from quantilo.vision.deep.backbones import BACKBONES

    browser = sys.platform == "emscripten"
    absent = [] if browser else missing()
    extra_reason = BROWSER if browser else (_extra_reason(absent) if absent else None)
    rows = []
    for spec in BACKBONES.values():
        w = spec.weight_spec
        assert w is not None
        on_disk = not browser and weights.downloaded(w)
        reason = extra_reason or (None if on_disk else weights.missing_reason(w))
        rows.append(
            BackboneStatus(
                spec.key,
                spec.title,
                spec.level,
                spec.dims,
                reason is None,
                on_disk,
                w.size_text,
                w.licence,
                w.note,
                reason,
            )
        )
    return DeepStatus(extra_reason is None, not browser and not absent, extra_reason, absent, rows)


_LAZY = {
    "Embeddings": "embed",
    "embed_images": "embed",
    "embed_image_set": "embed",
    "map_coordinates": "embed",
    "ProbeReport": "probe",
    "probe_embeddings": "probe",
    "probe_image_set": "probe",
}


def __getattr__(name: str) -> Any:
    """`deep.embed_image_set`, `deep.probe`... imported on first use (lazy, R-VDP-01)."""
    if name in _LAZY:
        module = importlib.import_module(f"quantilo.vision.deep.{_LAZY[name]}")
        return getattr(module, name)
    if name in {"weights", "backbones", "budget", "embed", "probe"}:
        return importlib.import_module(f"quantilo.vision.deep.{name}")
    raise AttributeError(f"module 'quantilo.vision.deep' has no attribute {name!r}")


__all__ = [
    "BROWSER",
    "EXTRA",
    "INSTALL",
    "PACKAGES",
    "BackboneStatus",
    "DeepStatus",
    "Embeddings",
    "ProbeReport",
    "embed_image_set",
    "embed_images",
    "installed",
    "map_coordinates",
    "missing",
    "probe_embeddings",
    "probe_image_set",
    "status",
]
