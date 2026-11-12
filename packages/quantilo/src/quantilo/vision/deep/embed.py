"""One vector per image from a frozen network (114; study § 4.4 and § 6.3).

Images are decoded by `vision.decode` (103: bombs, truncation, formats refused, no metadata),
the shorter side brought to `side / crop` (bicubic for DINOv2, bilinear for ResNet), centre
cropped to 224 x 224 and normalised as ImageNet. An image that cannot be decoded keeps a row of
NaN and is listed in `failures`. The run follows its budget (R-VDP-04, R-VDP-05).

torch is imported on first use only (R-VDP-01)."""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from quantilo.vision.decode import DecodedImage, DecodeFailure, try_decode
from quantilo.vision.deep import budget as budgets
from quantilo.vision.deep.backbones import DEFAULT, Backbone, BackboneSpec
from quantilo.vision.deep.backbones import load as load_backbone

if TYPE_CHECKING:
    from quantilo.vision.ingest import ImageSet

Source = str | Path | bytes | Image.Image | DecodedImage


class Cancelled(RuntimeError):
    pass


@dataclass
class Embeddings:
    vectors: np.ndarray  # (n, dims) float32; NaN rows for the images not embedded
    ok: np.ndarray  # (n,) bool
    backbone: str
    title: str
    model_key: str
    device: str  # "NVIDIA GeForce RTX 4050 Laptop GPU" or "CPU"
    batch: int  # batch at the end of the run (after any halving)
    seconds: float
    estimate: str | None = None  # R-VDP-05, from the first batch
    notes: list[str] = field(default_factory=list)
    failures: dict[int, str] = field(default_factory=dict)  # row -> reason
    image_ids: list[str] | None = None

    @property
    def dims(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def sentence(self) -> str:
        n = int(self.ok.sum())
        text = (
            f"{n:,} images embedded with {self.title} ({self.dims} numbers each) on "
            f"{self.device} in {budgets.duration_text(self.seconds)}."
        )
        if self.failures:
            text += f" {len(self.failures):,} left out (not decoded)."
        return text


def preprocess(image: Image.Image, spec: BackboneSpec) -> np.ndarray:
    """RGB image -> `(3, side, side)` float32, normalised."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    side = spec.side
    short = round(side / spec.crop)
    w, h = image.size
    scale = short / min(w, h)
    size = (max(side, round(w * scale)), max(side, round(h * scale)))
    resample = (
        Image.Resampling.BICUBIC if spec.interpolation == "bicubic" else Image.Resampling.BILINEAR
    )
    resized = image.resize(size, resample)
    left = (size[0] - side) // 2
    top = (size[1] - side) // 2
    cropped = resized.crop((left, top, left + side, top + side))
    x = np.asarray(cropped, dtype=np.float32) / 255.0
    x = (x - np.asarray(spec.mean, dtype=np.float32)) / np.asarray(spec.std, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _decode(source: Source, spec: BackboneSpec) -> Image.Image | str:
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, DecodedImage):
        return source.image
    decoded = try_decode(source, reduce_to=round(spec.side / spec.crop))
    if isinstance(decoded, DecodeFailure):
        return f"{decoded.code}: {decoded.reason}"
    return decoded.image


def _resolve(backbone: str | BackboneSpec | Backbone) -> Backbone:
    return backbone if isinstance(backbone, Backbone) else load_backbone(backbone)


def embed_images(
    images: Sequence[Source],
    backbone: str | BackboneSpec | Backbone = DEFAULT,
    *,
    device: budgets.Choice = "auto",
    allow_gpu: bool = True,
    batch: int | None = None,
    confirm_long: bool = False,
    progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Embeddings:
    """Vectors of `images` (paths, bytes, PIL images or decoded images), in order.

    Raises `WeightsMissing` when the weights are not downloaded (never downloads), `LongRun`
    after the first batch when the run would exceed 15 minutes and `confirm_long` is False."""
    import torch

    net = _resolve(backbone)
    spec = net.spec
    plan = budgets.plan(spec, device, allow_gpu, batch)
    n = len(images)
    vectors = np.full((n, spec.dims), np.nan, dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    failures: dict[int, str] = {}
    started = time.perf_counter()
    estimate: budgets.Estimate | None = None
    module = net.module

    def place(target: budgets.Budget) -> None:
        module.to(torch.device(target.device))
        if target.half:
            module.half()
        else:
            module.float()
        if target.threads:
            torch.set_num_threads(target.threads)

    def forward(chunk: Sequence[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack(chunk)).to(plan.device)
        x = x.half() if plan.half else x.float()
        with torch.inference_mode():
            return net.features(x).cpu().numpy().astype(np.float32)

    previous_threads = torch.get_num_threads()
    place(plan)
    try:
        done = 0
        start = 0
        while start < n:
            if cancel is not None and cancel():
                raise Cancelled("Embedding cancelled.")
            rows = list(range(start, min(n, start + plan.batch)))
            start = rows[-1] + 1
            pixels: list[np.ndarray] = []
            kept: list[int] = []
            for row in rows:
                image = _decode(images[row], spec)
                if isinstance(image, str):
                    failures[row] = image
                    continue
                pixels.append(preprocess(image, spec))
                kept.append(row)
            if pixels:
                first = time.perf_counter()
                try:
                    parts = budgets.run_in_batches(forward, pixels, plan)
                except budgets.GpuExhausted:
                    plan = budgets.cpu_budget(
                        spec, "GPU ran out of memory with a batch of 1: finished on the CPU."
                    )
                    place(plan)
                    parts = budgets.run_in_batches(forward, pixels, plan)
                vectors[kept] = np.concatenate(parts, axis=0)
                ok[kept] = True
                if estimate is None:
                    estimate = budgets.estimate(
                        time.perf_counter() - first, len(kept), n, plan.label
                    )
                    if estimate.needs_confirmation and not confirm_long:
                        raise budgets.LongRun(estimate)
            done = start
            if progress is not None:
                progress(done, n)
    finally:
        module.to("cpu").float()
        torch.set_num_threads(previous_threads)
        if plan.on_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
    notes = list(net.notes) + plan.notes
    notes.insert(0, f"Ran on {plan.sentence}.")
    return Embeddings(
        vectors,
        ok,
        spec.key,
        spec.title,
        net.model_key,
        plan.label,
        plan.batch,
        time.perf_counter() - started,
        estimate.sentence if estimate else None,
        notes,
        failures,
    )


def embed_image_set(
    images: "ImageSet",
    backbone: str | BackboneSpec | Backbone = DEFAULT,
    **options: Any,
) -> Embeddings:
    """Vectors of every row of an imported image set (104), aligned with its table. Rows not
    decoded at import (R-VQ-01) or excluded keep NaN and are listed in `failures`."""
    table = images.table
    stored = table["stored_path"].to_pylist()
    decoded = table["decode_ok"].to_pylist()
    excluded = table["excluded"].to_pylist()
    rows = [
        i for i in range(table.num_rows) if decoded[i] and not excluded[i] and stored[i] is not None
    ]
    sources: list[Source] = [images.store.path(stored[i]) for i in rows]
    inner = embed_images(sources, backbone, **options)
    n = table.num_rows
    vectors = np.full((n, inner.dims), np.nan, dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    vectors[rows] = inner.vectors
    ok[rows] = inner.ok
    failures = {rows[i]: reason for i, reason in inner.failures.items()}
    embedded = set(rows)
    for i in range(n):
        if i not in embedded:
            failures[i] = (
                "R-VQ-01: not decoded at import" if not decoded[i] else "excluded by the user"
            )
    inner.vectors, inner.ok, inner.failures = vectors, ok, failures
    inner.image_ids = [str(v) for v in table["image_id"].to_pylist()]
    return inner


@dataclass
class MapCoordinates:
    x: np.ndarray  # (n,) NaN where the image has no vector
    y: np.ndarray
    explained: tuple[float, float]  # share of variance of each axis
    sentence: str


def map_coordinates(embeddings: Embeddings | np.ndarray, seed: int = 0) -> MapCoordinates:
    """2D map of the vectors (PCA on the rows that have one), for the *Map* tab (109)."""
    from sklearn.decomposition import PCA

    vectors = embeddings.vectors if isinstance(embeddings, Embeddings) else embeddings
    valid = np.isfinite(vectors).all(axis=1)
    x = np.full(len(vectors), np.nan)
    y = np.full(len(vectors), np.nan)
    if valid.sum() < 3:
        return MapCoordinates(x, y, (0.0, 0.0), "Too few images to draw a map.")
    data = vectors[valid].astype(np.float64)
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    pca = PCA(2, random_state=seed).fit(data)
    points = pca.transform(data)
    x[valid], y[valid] = points[:, 0], points[:, 1]
    share = (float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1]))
    title = embeddings.title if isinstance(embeddings, Embeddings) else "the vectors"
    sentence = (
        f"Principal components of {title}: the two axes keep "
        f"{100 * (share[0] + share[1]):.0f} % of the variance."
    )
    return MapCoordinates(x, y, share, sentence)
