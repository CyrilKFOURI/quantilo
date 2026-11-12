"""Frozen pretrained networks that turn an image into a vector (114; study § 4.4, § 6.3).

| Key | Network | Vector | Level |
|---|---|---|---|
| `dinov2_s14` | DINOv2 ViT-S/14 | 768 = CLS token ‖ mean of patch tokens | first plan (desktop) |
| `dinov2_b14` | DINOv2 ViT-B/14 | 1 536, same pooling | Advanced |
| `resnet18` | ResNet-18 | 512 = global average of the last stage | Advanced (CPU fallback of 115) |
| `resnet50` | ResNet-50 | 2 048 = penultimate layer | Advanced |

Built with timm (`pretrained=False`: timm downloads nothing), then filled from our own verified
safetensors file (`weights.py`). The specs are plain data: this module imports torch and timm
only inside `load`, so it can be read without the extra."""

from dataclasses import dataclass, field
from typing import Any, Literal

from quantilo.vision.deep import weights as weights_manager

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIDE = 224

Kind = Literal["vit", "cnn"]
Level = Literal["first", "advanced"]


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    title: str
    architecture: str  # timm name
    kind: Kind
    dims: int
    pooling: str
    level: Level
    gpu_batch: int  # fp16 on a 6 GB card (study § 10)
    cpu_batch: int
    weights: str | None = None  # key in weights.MANIFEST (None: a custom network)
    side: int = SIDE
    crop: float = 1.0  # side / crop is the shorter side before the centre crop
    interpolation: Literal["bicubic", "bilinear"] = "bicubic"
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    @property
    def weight_spec(self) -> weights_manager.WeightSpec | None:
        return None if self.weights is None else weights_manager.MANIFEST[self.weights]


BACKBONES: dict[str, BackboneSpec] = {
    spec.key: spec
    for spec in (
        BackboneSpec(
            "dinov2_s14",
            "DINOv2 ViT-S/14",
            "vit_small_patch14_dinov2",
            "vit",
            768,
            "CLS token ‖ mean of the patch tokens",
            "first",
            gpu_batch=64,
            cpu_batch=16,
            weights="dinov2_s14",
        ),
        BackboneSpec(
            "dinov2_b14",
            "DINOv2 ViT-B/14",
            "vit_base_patch14_dinov2",
            "vit",
            1536,
            "CLS token ‖ mean of the patch tokens",
            "advanced",
            gpu_batch=32,
            cpu_batch=8,
            weights="dinov2_b14",
        ),
        BackboneSpec(
            "resnet18",
            "ResNet-18",
            "resnet18",
            "cnn",
            512,
            "global average of the last stage",
            "advanced",
            gpu_batch=128,
            cpu_batch=32,
            weights="resnet18",
            crop=0.875,
            interpolation="bilinear",
        ),
        BackboneSpec(
            "resnet50",
            "ResNet-50",
            "resnet50",
            "cnn",
            2048,
            "global average of the last stage (penultimate layer)",
            "advanced",
            gpu_batch=64,
            cpu_batch=16,
            weights="resnet50",
            crop=0.875,
            interpolation="bilinear",
        ),
    )
}
DEFAULT = "dinov2_s14"


def spec_of(key: str | BackboneSpec) -> BackboneSpec:
    if isinstance(key, BackboneSpec):
        return key
    try:
        return BACKBONES[key]
    except KeyError:
        raise KeyError(f"Unknown backbone {key!r}: one of {', '.join(BACKBONES)}.") from None


@dataclass
class Backbone:
    """A network in evaluation mode, gradients off, with the way its output becomes a vector.

    `module` follows timm: `forward_features(x)` gives tokens `(n, prefix + patches, d)` for a
    ViT, a map `(n, c, h, w)` for a CNN. Any module that does the same works (tests use a tiny
    one)."""

    spec: BackboneSpec
    module: Any  # torch.nn.Module
    weights_sha: str = "custom"
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @property
    def model_key(self) -> str:
        """Identifies the vectors: same key, comparable vectors."""
        s = self.spec
        return f"{s.key}|{self.weights_sha[:12]}|{s.side}px|crop={s.crop}|{s.pooling}"

    def features(self, x: Any) -> Any:
        """`(n, 3, side, side)` normalised images -> `(n, dims)` float32 vectors."""
        out = self.module.forward_features(x)
        if self.spec.kind == "vit":
            prefix = int(getattr(self.module, "num_prefix_tokens", 1))
            cls = out[:, 0]
            patches = out[:, prefix:].mean(dim=1)
            vectors = (cls, patches)
            import torch

            return torch.cat(vectors, dim=1).float()
        return out.mean(dim=(2, 3)).float()


def _resample_positions(state: dict[str, Any], module: Any) -> None:
    """DINOv2's position embedding is stored for 518 px (37 x 37 patches); built at 224 px
    (16 x 16), it is resampled once here, bicubic, as timm does when it loads its own weights."""
    from timm.layers.pos_embed import resample_abs_pos_embed  # pyright: ignore[reportMissingImports]

    stored = state.get("pos_embed")
    target = getattr(module, "pos_embed", None)
    if stored is None or target is None or stored.shape == target.shape:
        return
    prefix = 0 if getattr(module, "no_embed_class", False) else int(module.num_prefix_tokens)
    grid = module.patch_embed.grid_size
    state["pos_embed"] = resample_abs_pos_embed(
        stored, new_size=list(grid), num_prefix_tokens=prefix
    )


def load(key: str | BackboneSpec = DEFAULT) -> Backbone:
    """The network with its verified weights (R-VDP-03). Never downloads: raises
    `WeightsMissing` (R-VDP-02) or, without the extra, `RuntimeError` with the reason."""
    from quantilo.vision.deep import INSTALL, missing

    spec = spec_of(key)
    if absent := missing():
        raise RuntimeError(
            f"{spec.title} needs the optional extra vision ({INSTALL}); missing: "
            f"{', '.join(absent)} (R-VDP-01)."
        )
    weight_spec = spec.weight_spec
    if weight_spec is None:
        raise ValueError(f"{spec.key} has no weights in the manifest: build Backbone yourself.")
    state = weights_manager.load_state_dict(weight_spec)  # verified, safetensors only
    import timm  # pyright: ignore[reportMissingImports]

    options: dict[str, Any] = {"pretrained": False, "num_classes": 0}
    if spec.kind == "vit":
        options["img_size"] = spec.side
    module = timm.create_model(spec.architecture, **options)
    for head in ("fc.weight", "fc.bias", "head.weight", "head.bias"):
        state.pop(head, None)  # the ImageNet classifier: unused, the head is removed
    _resample_positions(state, module)
    result = module.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"The weights of {spec.title} do not fit the network (missing "
            f"{list(result.missing_keys)[:3]}, unexpected {list(result.unexpected_keys)[:3]})."
        )
    return Backbone(spec, module, weight_spec.sha256)
