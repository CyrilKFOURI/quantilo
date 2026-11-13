"""Deep vision on the desktop (114; study 2026-11-11 vision-et-xai §§ 2, 4.4, 6, 10, 11).

Optional extra `vision` (`pip install "quantilo[vision]"`: torch, torchvision, timm,
safetensors). Nothing in this package imports them at module level: without the extra the rest
of quantilo imports and runs, and `status()` says why each deep option is greyed (R-VDP-01).

- `weights`: frozen manifest, download on explicit confirmation, SHA-256, safetensors.
- `backbones`: DINOv2 ViT-S/14 (default), DINOv2 ViT-B/14, ResNet-18/50, frozen.
- `budget`: GPU (6 GB, fp16) or CPU, batch halved on out-of-memory, time estimate.
- `embed`: one vector per image (`embed_image_set`), 2D map coordinates.
- `probe`: linear probe and cosine kNN, scored by `vision.metrics` (numpy and scikit-learn only).
- 118, Advanced › More (classification): `finetune` (the top blocks, AMP, early stopping),
  `protopnet` (ProtoPNet-lite, "this looks like that"), `concepts` (concept bottleneck),
  `tcav` (concept activation vectors and their test), `more` (their place in a run).
- `patchcore` (115, the default anomaly detector), `padim` (115, Advanced), `autoencoder` (116,
  Advanced: convolutional autoencoder and VAE); `anomaly`: their shared pieces and
  `anomaly_image_set` (fit on the normal training images, score and evaluate the test images).

    from quantilo.vision import deep
    deep.status()                                   # what is possible here, and why not
    deep.weights.download("dinov2_s14", confirm=True)  # 88 MB, once
    vectors = deep.embed_image_set(images)
    result = deep.probe_image_set(images, vectors)
    run = deep.anomaly_image_set(images, "patchcore")  # Wide ResNet-50-2 or ResNet-18 weights
"""

import importlib.util
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from quantilo.vision import browser as web

EXTRA = "vision"
INSTALL = 'pip install "quantilo[vision]"'
PACKAGES = ("torch", "torchvision", "timm", "safetensors")
BROWSER = web.DESKTOP_REASON  # R-VIS-WEB-03: every deep option in the browser version (119)

if TYPE_CHECKING:
    from quantilo.vision.deep.anomaly import AnomalyResult, AnomalyRun, anomaly_image_set
    from quantilo.vision.deep.autoencoder import ConvAutoencoder
    from quantilo.vision.deep.embed import (
        Embeddings,
        embed_image_set,
        embed_images,
        map_coordinates,
    )
    from quantilo.vision.deep.padim import PaDiM
    from quantilo.vision.deep.patchcore import PatchCore
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
class DetectorStatus:
    """An anomaly detector of 115 / 116 and whether it can run here."""

    key: str  # "patchcore", "padim", "autoencoder", "vae"
    title: str
    level: str  # "first" | "advanced" | "more" (c3: the ensemble)
    weights: list[str]  # keys of weights.MANIFEST it may use (none for the autoencoders)
    available: bool
    reason: str | None
    note: str = ""


@dataclass
class DeepStatus:
    available: bool  # the extra is there (each backbone still needs its weights)
    installed: bool
    reason: str | None
    missing: list[str] = field(default_factory=list)
    backbones: list[BackboneStatus] = field(default_factory=list)
    detectors: list[DetectorStatus] = field(default_factory=list)

    def backbone(self, key: str) -> BackboneStatus:
        return next(b for b in self.backbones if b.key == key)

    def detector(self, key: str) -> DetectorStatus:
        return next(d for d in self.detectors if d.key == key)


def _extra_reason(absent: list[str]) -> str:
    return f"Needs the optional extra vision ({INSTALL}); missing: {', '.join(absent)} (R-VDP-01)."


def status() -> DeepStatus:
    """Whether each deep option can be chosen here, and if not, why (R-VDP-01, R-VDP-02). Cheap:
    no torch import, no network, no hashing."""
    from quantilo.vision.deep import weights
    from quantilo.vision.deep.backbones import BACKBONES

    browser = web.active()
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
    return DeepStatus(
        extra_reason is None,
        not browser and not absent,
        extra_reason,
        absent,
        rows,
        _detectors(extra_reason, browser),
    )


def _detectors(extra_reason: str | None, browser: bool) -> list[DetectorStatus]:
    """115 / 116: PatchCore needs Wide ResNet-50-2 (GPU) or ResNet-18 (CPU), PaDiM ResNet-18,
    the autoencoders no weights (they are trained here)."""
    from quantilo.vision.deep import weights

    def need(keys: list[str]) -> str | None:
        if extra_reason or any(not browser and weights.downloaded(k) for k in keys):
            return extra_reason
        return weights.missing_reason(weights.MANIFEST[keys[0]])

    wrn, r18 = "wide_resnet50_2", "resnet18"
    patchcore, padim = need([wrn, r18]), need([r18])
    trained = "Trained here on the normal images: no download, minutes on a GPU (R-VAD-08)."
    return [
        DetectorStatus(
            "patchcore",
            "PatchCore",
            "first",
            [wrn, r18],
            patchcore is None,
            patchcore,
            "Wide ResNet-50-2 on a GPU, ResNet-18 on the CPU (R-VAD-04).",
        ),
        DetectorStatus("padim", "PaDiM", "advanced", [r18], padim is None, padim),
        DetectorStatus(
            "autoencoder",
            "Convolutional autoencoder",
            "advanced",
            [],
            extra_reason is None,
            extra_reason,
            trained,
        ),
        DetectorStatus(
            "vae",
            "Variational autoencoder",
            "advanced",
            [],
            extra_reason is None,
            extra_reason,
            trained,
        ),
        # c3, R-VAN-07: Advanced > More, both on the same folds, their ranks averaged
        DetectorStatus(
            "patchcore_ae",
            "PatchCore + autoencoder (average rank)",
            "more",
            [wrn, r18],
            patchcore is None and extra_reason is None,
            patchcore or extra_reason,
            "A memory of normal patches and a reconstruction fail differently: their percentile "
            "ranks are averaged, and so are their maps (R-VAN-07).",
        ),
    ]


_LAZY = {
    "Embeddings": "embed",
    "embed_images": "embed",
    "embed_image_set": "embed",
    "map_coordinates": "embed",
    "ProbeReport": "probe",
    "probe_embeddings": "probe",
    "probe_image_set": "probe",
    "AnomalyResult": "anomaly",
    "AnomalyRun": "anomaly",
    "anomaly_image_set": "anomaly",
    "PatchCore": "patchcore",
    "PaDiM": "padim",
    "ConvAutoencoder": "autoencoder",
}
_MODULES = {
    "weights",
    "backbones",
    "budget",
    "embed",
    "probe",
    "anomaly",
    "patchcore",
    "padim",
    "autoencoder",
    "finetune",
    "protopnet",
    "concepts",
    "tcav",
    "more",
}


def __getattr__(name: str) -> Any:
    """`deep.embed_image_set`, `deep.probe`, `deep.PatchCore`... imported on first use (lazy,
    R-VDP-01)."""
    if name in _LAZY:
        module = importlib.import_module(f"quantilo.vision.deep.{_LAZY[name]}")
        return getattr(module, name)
    if name in _MODULES:
        return importlib.import_module(f"quantilo.vision.deep.{name}")
    raise AttributeError(f"module 'quantilo.vision.deep' has no attribute {name!r}")


__all__ = [
    "BROWSER",
    "EXTRA",
    "INSTALL",
    "PACKAGES",
    "AnomalyResult",
    "AnomalyRun",
    "BackboneStatus",
    "ConvAutoencoder",
    "DeepStatus",
    "DetectorStatus",
    "Embeddings",
    "PaDiM",
    "PatchCore",
    "ProbeReport",
    "anomaly_image_set",
    "embed_image_set",
    "embed_images",
    "installed",
    "map_coordinates",
    "missing",
    "probe_embeddings",
    "probe_image_set",
    "status",
]
