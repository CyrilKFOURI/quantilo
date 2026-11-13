"""PaDiM (115, Advanced): a Gaussian per position of the patch grid (Defard et al., "PaDiM: a
Patch Distribution Modeling Framework for Anomaly Detection and Localization", ICPR 2021; study
§ 6.3).

- R-VAD-07: ResNet-18 frozen, stages 1, 2 and 3 brought to the grid of stage 1 (nearest, as the
  paper's concatenation) and concatenated (448 numbers), 100 of them drawn at random (seed 0;
  550 of 1,792 with Wide ResNet-50-2). At each position of the grid, the mean and covariance of
  the normal images (+ 0.01 I, as the paper), accumulated image by image in float64 so the set
  can be large. A patch scores its Mahalanobis distance to its position's Gaussian; the map is
  upsampled (bilinear) and smoothed (Gaussian, sigma = 4 px); image score = the map's maximum.
  The inverse covariances are kept under 1 GB in fp32, otherwise fewer numbers are drawn, said.

PaDiM has no case to show: its map is its explanation. torch is imported on first use only."""

import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from quantilo.vision.deep import budget as budgets
from quantilo.vision.deep.anomaly import (
    SEED,
    SIDE,
    SIGMA,
    AnomalyResult,
    Runner,
    Source,
    batches,
    intermediates,
    normalise,
    resolve,
    smooth,
)
from quantilo.vision.deep.backbones import Backbone, BackboneSpec

LAYERS = (1, 2, 3)
EPSILON = 0.01
COVARIANCE_MB = 1024
DIMS = {"resnet18": 100, "wide_resnet50_2": 550}
DEFAULT_DIMS = 100
_POOLING = "stages 1 + 2 + 3, random dimensions"

BACKBONES: dict[str, BackboneSpec] = {
    spec.key: spec
    for spec in (
        BackboneSpec(
            "resnet18",
            "ResNet-18",
            "resnet18",
            "cnn",
            448,
            _POOLING,
            "advanced",
            gpu_batch=64,
            cpu_batch=16,
            weights="resnet18",
            side=SIDE,
            interpolation="bilinear",
        ),
        BackboneSpec(
            "wide_resnet50_2",
            "Wide ResNet-50-2",
            "wide_resnet50_2",
            "cnn",
            1792,
            _POOLING,
            "advanced",
            gpu_batch=16,
            cpu_batch=4,
            weights="wide_resnet50_2",
            side=SIDE,
            interpolation="bilinear",
        ),
    )
}


def stage_embedding(maps: Sequence[Any]) -> Any:
    """Stage maps -> `(n, channels, h, w)` on the grid of the first one (nearest upsampling)."""
    import torch
    import torch.nn.functional as F

    size = maps[0].shape[-2:]
    grid = [maps[0].float()] + [
        F.interpolate(m.float(), size=size, mode="nearest") for m in maps[1:]
    ]
    return torch.cat(grid, dim=1)


def _mb(mb: float) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.3g} MB"


class PaDiM:
    method = "padim"

    def __init__(
        self,
        backbone: str | BackboneSpec | Backbone = "resnet18",
        *,
        side: int = SIDE,
        dims: int | None = None,
        sigma: float = SIGMA,
        epsilon: float = EPSILON,
        covariance_mb: float = COVARIANCE_MB,
        device: budgets.Choice = "auto",
        allow_gpu: bool = True,
        batch: int | None = None,
        confirm_long: bool = False,
        progress: Callable[[int, int], None] | None = None,
        cancel: Callable[[], bool] | None = None,
        seed: int = SEED,
    ) -> None:
        self.backbone = backbone
        self.side = side
        self.dims = dims
        self.sigma = sigma
        self.epsilon = epsilon
        self.covariance_mb = covariance_mb
        self.device: budgets.Choice = device
        self.allow_gpu = allow_gpu
        self.batch = batch
        self.confirm_long = confirm_long
        self.progress = progress
        self.cancel = cancel
        self.seed = seed
        self.net: Backbone | None = None
        self.index: Any = None  # the numbers drawn (torch long tensor)
        self.mean: Any = None  # (positions, d) float32, CPU
        self.inverse: Any = None  # (positions, d, d) float32, CPU
        self.grid: tuple[int, int] = (1, 1)
        self.images = 0
        self.notes: list[str] = []
        self.fit_failures: dict[int, str] = {}
        self.estimate: str | None = None

    @property
    def title(self) -> str:
        name = self.net.spec.title if self.net is not None else "frozen network"
        return f"PaDiM ({name})"

    @property
    def model_key(self) -> str:
        assert self.net is not None
        d = 0 if self.index is None else len(self.index)
        return f"padim|{self.net.model_key}|side={self.side}|dims={d}|seed={self.seed}"

    def _draw(self, channels: int, positions: int) -> Any:
        import torch

        key = self.net.spec.key if self.net is not None else ""
        d = min(channels, self.dims or DIMS.get(key, DEFAULT_DIMS))
        cap = int(math.sqrt(self.covariance_mb * 2**20 / (4 * positions)))
        if d > cap:
            self.notes.append(
                f"{d} numbers per patch would need more than {_mb(self.covariance_mb)} "
                f"of covariances: {cap} drawn instead (R-VAD-07)."
            )
            d = max(1, cap)
        generator = torch.Generator().manual_seed(self.seed)
        return torch.randperm(channels, generator=generator)[:d].sort().values

    def fit(self, sources: Sequence[Source]) -> "PaDiM":
        """Mean and covariance of each position over the normal images (R-VAD-07)."""
        import torch

        self.net = net = resolve(self.backbone, BACKBONES)
        n = len(sources)
        failures: dict[int, str] = {}
        sizes: list[tuple[int, int] | None] = [None] * n
        sums: dict[str, Any] = {}

        with Runner(
            net, self.device, self.allow_gpu, self.batch, confirm_long=self.confirm_long
        ) as runner:

            def forward(chunk: Sequence[np.ndarray]) -> tuple[Any, Any, int]:
                x = runner.tensor(chunk)
                with torch.inference_mode():
                    emb = stage_embedding(intermediates(net, x, LAYERS))
                    b, c, h, w = emb.shape
                    if self.index is None:
                        self.index = self._draw(c, h * w)
                        self.grid = (h, w)
                    e = emb[:, self.index.to(emb.device)].flatten(2).permute(0, 2, 1).double()
                    return e.sum(dim=0), torch.einsum("bpd,bpe->pde", e, e), b

            done = 0
            for rows, pixels in batches(
                sources, self.side, lambda: runner.plan.batch, failures, sizes, self.cancel
            ):
                if pixels:
                    first = time.perf_counter()
                    for s1, s2, b in runner.run(forward, list(normalise(pixels))):
                        if "s1" not in sums:
                            sums["s1"], sums["s2"], sums["n"] = s1, s2, 0
                        else:
                            sums["s1"] = sums["s1"].to(s1.device) + s1
                            sums["s2"] = sums["s2"].to(s2.device) + s2
                        sums["n"] += b
                    runner.timed(time.perf_counter() - first, len(pixels), n)
                done = rows[-1] + 1 if rows else done
                if self.progress is not None:
                    self.progress(done, n)
            count = int(sums.get("n", 0))
            if count < 2:
                raise ValueError("PaDiM needs at least 2 decoded normal images.")
            mean = sums["s1"] / count
            outer = torch.einsum("pd,pe->pde", mean, mean)
            cov = (sums["s2"] - count * outer) / (count - 1)
            d = cov.shape[-1]
            cov = cov + self.epsilon * torch.eye(d, dtype=cov.dtype, device=cov.device)
            self.inverse = torch.linalg.inv(cov).float().cpu()
            self.mean = mean.float().cpu()
            del sums
            self.notes.extend(runner.notes)
            if runner.estimate is not None:
                self.estimate = runner.estimate.sentence
        self.images = count
        self.fit_failures = failures
        h, w = self.grid
        self.notes.append(
            f"A Gaussian of {len(self.index)} numbers at each of the {h} x {w} positions, from "
            f"{count:,} normal images (R-VAD-07)."
        )
        return self

    def score(self, sources: Sequence[Source]) -> AnomalyResult:
        """Mahalanobis maps and scores (R-VAD-07)."""
        import torch
        import torch.nn.functional as F

        if self.mean is None or self.net is None:
            raise RuntimeError("Fit PaDiM on normal images first.")
        net = self.net
        n = len(sources)
        side = self.side
        started = time.perf_counter()
        failures: dict[int, str] = {}
        sizes: list[tuple[int, int] | None] = [None] * n
        raw = np.full((n, side, side), np.nan, dtype=np.float32)
        ok = np.zeros(n, dtype=bool)
        cache: dict[str, Any] = {}

        def stats_on(device: Any) -> tuple[Any, Any, Any]:
            key = str(device)
            if key not in cache:
                cache.clear()
                cache[key] = (
                    self.mean.to(device),
                    self.inverse.to(device),
                    self.index.to(device),
                )
            return cache[key]

        with Runner(
            net, self.device, self.allow_gpu, self.batch, confirm_long=self.confirm_long
        ) as runner:

            def forward(chunk: Sequence[np.ndarray]) -> np.ndarray:
                x = runner.tensor(chunk)
                mean, inverse, index = stats_on(x.device)
                with torch.inference_mode():
                    emb = stage_embedding(intermediates(net, x, LAYERS))
                    b, _, h, w = emb.shape
                    delta = emb[:, index].flatten(2).permute(0, 2, 1) - mean
                    left = torch.einsum("bpd,pde->bpe", delta, inverse)
                    dist = (left * delta).sum(dim=-1).clamp_(min=0).sqrt().view(b, 1, h, w)
                    up = F.interpolate(dist, (side, side), mode="bilinear", align_corners=False)
                    return up[:, 0].cpu().numpy()

            done = 0
            try:
                for rows, pixels in batches(
                    sources, side, lambda: runner.plan.batch, failures, sizes, self.cancel
                ):
                    if pixels:
                        first = time.perf_counter()
                        parts = runner.run(forward, list(normalise(pixels)))
                        index = np.asarray(rows)
                        raw[index] = np.concatenate(parts)
                        ok[index] = True
                        runner.timed(time.perf_counter() - first, len(pixels), n)
                    done = rows[-1] + 1 if rows else done
                    if self.progress is not None:
                        self.progress(done, n)
            finally:
                cache.clear()
            notes = list(self.notes) + [s for s in runner.notes if s not in self.notes]
            estimate = runner.estimate.sentence if runner.estimate is not None else None
            label = runner.plan.label
        maps = raw.copy()
        scores = np.full(n, np.nan)
        if ok.any():
            maps[ok] = smooth(raw[ok], self.sigma)
            scores[ok] = maps[ok].reshape(int(ok.sum()), -1).max(axis=1)
        return AnomalyResult(
            self.method,
            self.title,
            scores,
            maps,
            ok,
            sizes,
            side,
            label,
            time.perf_counter() - started,
            self.model_key,
            estimate,
            notes,
            failures,
        )


__all__ = ["BACKBONES", "PaDiM", "stage_embedding"]
