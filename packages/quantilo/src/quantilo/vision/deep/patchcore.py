"""PatchCore-lite (115): the default anomaly detector of the desktop app (Roth et al., "Towards
Total Recall in Industrial Anomaly Detection", CVPR 2022; study §§ 6.3, 8, 10).

No training: a frozen ImageNet network describes every patch of the normal images; a memory bank
keeps a well-spread tenth of them; a new patch is as unusual as its distance to the nearest
patch of the bank. Each score is explained by a case: the normal patch closest to the most
unusual area, shown next to it (§ 8, in the spirit of Rudin 2019).

- R-VAD-04: Wide ResNet-50-2 on a GPU, ResNet-18 on the CPU (§ 10); stages 2 and 3, each
  averaged over its 3 x 3 neighbourhood, stage 3 brought to the grid of stage 2 (bilinear), the
  two concatenated (1,536 numbers per patch for WRN-50-2, 384 for ResNet-18).
- R-VAD-05: memory bank = greedy k-center coreset of 10 % of the patches, distances measured
  after a random projection to 128 dimensions (seed 0). The bank is kept under 1.5 GB in fp16,
  the patches before the coreset under 3 GB (otherwise a random share of each image's patches),
  each reduction said in the notes.
- R-VAD-06: score of a patch = Euclidean distance to its nearest bank patch (`torch.cdist` by
  blocks, in fp32; no faiss); image score = the largest patch score; map = patch scores
  upsampled to the image (bilinear) and smoothed (Gaussian, sigma = 4 px).

torch is imported on first use only (R-VDP-01)."""

import math
import time
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass, field
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
    free_gpu,
    intermediates,
    normalise,
    resolve,
    scale_box,
    smooth,
)
from quantilo.vision.deep.backbones import Backbone, BackboneSpec

LAYERS = (2, 3)
CORESET = 0.10
PROJECTION = 128
NEIGHBOURHOOD = 3
BANK_MB = 1536  # § 10: the bank stays under 1.5 GB in fp16
STORE_MB = 3072  # patches kept before the coreset, in fp16
BLOCK = 4096  # rows of the query and of the bank per `torch.cdist` call
GPU_KEY, CPU_KEY = "wide_resnet50_2", "resnet18"
_POOLING = "stages 2 + 3, 3 x 3 average"

BACKBONES: dict[str, BackboneSpec] = {
    spec.key: spec
    for spec in (
        BackboneSpec(
            "wide_resnet50_2",
            "Wide ResNet-50-2",
            "wide_resnet50_2",
            "cnn",
            1536,
            _POOLING,
            "first",
            gpu_batch=16,  # § 10: batches of 16 at 256 px, about 2 GB
            cpu_batch=4,
            weights="wide_resnet50_2",
            side=SIDE,
            interpolation="bilinear",
        ),
        BackboneSpec(
            "resnet18",
            "ResNet-18",
            "resnet18",
            "cnn",
            384,
            _POOLING,
            "first",
            gpu_batch=64,
            cpu_batch=16,
            weights="resnet18",
            side=SIDE,
            interpolation="bilinear",
        ),
    )
}


def default_backbone(device: budgets.Choice = "auto", allow_gpu: bool = True) -> str:
    """R-VAD-04: Wide ResNet-50-2 where a GPU will run it, ResNet-18 on the CPU."""
    from quantilo.synthetic.deep.device import pick

    return GPU_KEY if pick(device, allow_gpu).device.type != "cpu" else CPU_KEY


def patch_embedding(maps: Sequence[Any], neighbourhood: int = NEIGHBOURHOOD) -> Any:
    """Stage maps -> `(n, channels, h, w)` patch features on the grid of the first map: each map
    averaged over its neighbourhood, the others upsampled (bilinear), all concatenated."""
    import torch
    import torch.nn.functional as F

    pad = neighbourhood // 2
    pooled = [
        F.avg_pool2d(m.float(), neighbourhood, stride=1, padding=pad) if neighbourhood > 1 else m
        for m in maps
    ]
    size = pooled[0].shape[-2:]
    grid = [pooled[0]] + [
        F.interpolate(p, size=size, mode="bilinear", align_corners=False) for p in pooled[1:]
    ]
    return torch.cat(grid, dim=1)


def random_projection(channels: int, dims: int, seed: int = SEED) -> Any:
    """A Gaussian projection to `dims` numbers (Johnson-Lindenstrauss), identity when the patches
    are already that small."""
    import torch

    if channels <= dims:
        return torch.eye(channels)
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(channels, dims, generator=generator) / math.sqrt(dims)


def greedy_coreset(z: Any, k: int, seed: int = SEED) -> np.ndarray:
    """Greedy k-center (Sener and Savarese 2018, as in PatchCore): start from a random patch,
    then add the patch farthest from those already chosen, `k` times. Indices, in the order
    chosen."""
    import torch

    n = int(z.shape[0])
    if k >= n:
        return np.arange(n)
    z = z.float()
    norms = (z * z).sum(dim=1)
    generator = torch.Generator().manual_seed(seed)
    first = int(torch.randint(n, (1,), generator=generator))
    chosen = torch.empty(k, dtype=torch.long, device=z.device)
    chosen[0] = first

    def distances(row: Any) -> Any:
        centre = z[row]
        return (norms - 2 * torch.mv(z, centre) + norms[row]).clamp_(min=0)

    closest = distances(first)
    for i in range(1, k):
        far = torch.argmax(closest)
        chosen[i] = far
        torch.minimum(closest, distances(far), out=closest)
    return chosen.cpu().numpy()


def nearest(query: Any, bank: Any, block: int = BLOCK) -> tuple[Any, Any]:
    """Distance of each query row to its nearest bank row, and that row (`torch.cdist` by
    blocks of `block` x `block`, in fp32, so the memory stays bounded)."""
    import torch

    q = int(query.shape[0])
    best = torch.full((q,), math.inf, device=query.device)
    where = torch.zeros(q, dtype=torch.long, device=query.device)
    for qs in range(0, q, block):
        part = query[qs : qs + block].float()
        part_best = torch.full((len(part),), math.inf, device=query.device)
        part_where = torch.zeros(len(part), dtype=torch.long, device=query.device)
        for bs in range(0, int(bank.shape[0]), block):
            chunk = bank[bs : bs + block].float()
            value, index = torch.cdist(part, chunk).min(dim=1)
            better = value < part_best
            part_best = torch.where(better, value, part_best)
            part_where = torch.where(better, index + bs, part_where)
        best[qs : qs + block] = part_best
        where[qs : qs + block] = part_where
    return best, where


@dataclass
class PatchExplanation:
    """The case behind a score (§ 8): an area of the image and the normal patch closest to it."""

    image: int  # in the order given to `score`
    cell: tuple[int, int]  # (row, column) on the patch grid
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1) in the image's own pixels
    distance: float
    normal_image: int  # in the order given to `fit`
    normal_box: tuple[int, int, int, int]  # in that normal image's own pixels
    normal_row: int | None = None  # its row in the image set's table, when known
    sentence: str = ""


@dataclass
class PatchCoreResult(AnomalyResult):
    grid: tuple[int, int] = (0, 0)  # (rows, columns) of the patch grid
    patch_scores: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), np.float32))
    nearest: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), np.int32))
    top: list[PatchExplanation | None] = field(default_factory=list)  # the most unusual area
    bank_origins: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int32))
    fit_sizes: list[tuple[int, int] | None] = field(default_factory=list)
    fit_rows: list[int] | None = None  # table rows of the fitting images (image sets)
    neighbourhood: int = NEIGHBOURHOOD

    def _box(
        self, cell: tuple[int, int], size: tuple[int, int] | None
    ) -> tuple[int, int, int, int]:
        rows, cols = self.grid
        r = self.neighbourhood // 2
        sy, sx = self.side / rows, self.side / cols
        gy, gx = cell
        box = (
            max(0.0, (gx - r) * sx),
            max(0.0, (gy - r) * sy),
            min(float(self.side), (gx + 1 + r) * sx),
            min(float(self.side), (gy + 1 + r) * sy),
        )
        return scale_box(box, self.side, size)

    def explain(self, i: int, x: float | None = None, y: float | None = None) -> PatchExplanation:
        """The normal patch closest to the area at `(x, y)` of image `i` (its own pixels), or to
        its most unusual area when no point is given."""
        if not self.ok[i]:
            raise ValueError(f"Image {i} was not scored: {self.failures.get(i, 'not decoded')}.")
        rows, cols = self.grid
        if x is None or y is None:
            flat = int(np.argmax(self.patch_scores[i]))
            gy, gx = divmod(flat, cols)
        else:
            w, h = self.sizes[i] or (self.side, self.side)
            gx = min(cols - 1, max(0, int(x / w * cols)))
            gy = min(rows - 1, max(0, int(y / h * rows)))
        bank_row = int(self.nearest[i, gy, gx])
        image, ny, nx = (int(v) for v in self.bank_origins[bank_row])
        distance = float(self.patch_scores[i, gy, gx])
        box = self._box((gy, gx), self.sizes[i])
        normal_box = self._box((ny, nx), self.fit_sizes[image] if self.fit_sizes else None)
        normal_row = None if self.fit_rows is None else self.fit_rows[image]
        which = f"row {normal_row}" if normal_row is not None else f"image {image}"
        lead = "This area looks least like" if x is None else "This area is closest to"
        sentence = (
            f"{lead} the normal images: the nearest normal patch (training {which}, "
            f"box {normal_box}) is at distance {distance:.3g} (R-VAD-06)."
        )
        return PatchExplanation(i, (gy, gx), box, distance, image, normal_box, normal_row, sentence)

    def subset(self, rows: Sequence[int]) -> "PatchCoreResult":
        index = np.asarray(rows, dtype=int)
        out = super().subset(rows)
        assert isinstance(out, PatchCoreResult)
        out.patch_scores = self.patch_scores[index]
        out.nearest = self.nearest[index]
        out.top = []
        for j, i in enumerate(index):
            found = self.top[i] if self.top else None
            out.top.append(None if found is None else _renumber(found, j))
        return out


def _renumber(found: PatchExplanation, image: int) -> PatchExplanation:
    moved = copy(found)
    moved.image = image
    return moved


def _mb(mb: float) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.3g} MB"


class PatchCore:
    """Fit on normal images (`fit`), then score any image (`score`)."""

    method = "patchcore"

    def __init__(
        self,
        backbone: str | BackboneSpec | Backbone = "auto",
        *,
        side: int = SIDE,
        coreset: float = CORESET,
        projection: int = PROJECTION,
        neighbourhood: int = NEIGHBOURHOOD,
        sigma: float = SIGMA,
        device: budgets.Choice = "auto",
        allow_gpu: bool = True,
        batch: int | None = None,
        confirm_long: bool = False,
        bank_mb: float = BANK_MB,
        store_mb: float = STORE_MB,
        progress: Callable[[int, int], None] | None = None,
        cancel: Callable[[], bool] | None = None,
        seed: int = SEED,
    ) -> None:
        if not 0 < coreset <= 1:
            raise ValueError("The coreset share is in (0, 1].")
        self.backbone = backbone
        self.side = side
        self.coreset = coreset
        self.projection = projection
        self.neighbourhood = neighbourhood
        self.sigma = sigma
        self.device: budgets.Choice = device
        self.allow_gpu = allow_gpu
        self.batch = batch
        self.confirm_long = confirm_long
        self.bank_mb = bank_mb
        self.store_mb = store_mb
        self.progress = progress
        self.cancel = cancel
        self.seed = seed
        self.net: Backbone | None = None
        self.bank: Any = None  # (m, channels) fp16 on the CPU
        self.origins = np.zeros((0, 3), dtype=np.int32)  # (fit image, grid row, grid column)
        self.fit_sizes: list[tuple[int, int] | None] = []
        self.fit_failures: dict[int, str] = {}
        self.notes: list[str] = []
        self.patches = 0  # before the coreset
        self.fit_seconds = 0.0
        self.estimate: str | None = None
        self.grid: tuple[int, int] = (1, 1)

    @property
    def title(self) -> str:
        name = self.net.spec.title if self.net is not None else "frozen network"
        return f"PatchCore ({name})"

    @property
    def model_key(self) -> str:
        assert self.net is not None
        return (
            f"patchcore|{self.net.model_key}|side={self.side}|coreset={self.coreset}|"
            f"projection={self.projection}|neighbourhood={self.neighbourhood}"
        )

    def _network(self) -> Backbone:
        if self.net is not None:
            return self.net
        key = self.backbone
        if key == "auto":
            key = default_backbone(self.device, self.allow_gpu)
            self.notes.append(
                f"{BACKBONES[key].title}: Wide ResNet-50-2 on a GPU, ResNet-18 on the CPU "
                "(R-VAD-04)."
            )
        self.net = resolve(key, BACKBONES)
        return self.net

    def fit(self, sources: Sequence[Source]) -> "PatchCore":
        """Build the memory bank from normal images (R-VAD-04, R-VAD-05). Raises `WeightsMissing`
        without the weights (never downloads), `LongRun` beyond 15 minutes without
        `confirm_long`."""
        import torch

        net = self._network()
        n = len(sources)
        if not n:
            raise ValueError("PatchCore needs normal images to learn from.")
        started = time.perf_counter()
        failures: dict[int, str] = {}
        sizes: list[tuple[int, int] | None] = [None] * n
        rng = np.random.default_rng(self.seed)
        kept_patches: list[Any] = []
        kept_z: list[Any] = []
        kept_origins: list[np.ndarray] = []
        state: dict[str, Any] = {"proj": None, "keep": None, "channels": 0}

        with Runner(
            net, self.device, self.allow_gpu, self.batch, confirm_long=self.confirm_long
        ) as runner:

            def forward(chunk: Sequence[np.ndarray]) -> tuple[Any, Any, tuple[int, int]]:
                x = runner.tensor(chunk)
                with torch.inference_mode():
                    emb = patch_embedding(intermediates(net, x, LAYERS), self.neighbourhood)
                    b, c, h, w = emb.shape
                    flat = emb.permute(0, 2, 3, 1).reshape(b, h * w, c)
                    if state["proj"] is None:
                        state["proj"] = random_projection(c, self.projection, self.seed)
                        state["channels"] = c
                    z = flat.float() @ state["proj"].to(flat.device)
                    return flat.half().cpu(), z.cpu(), (h, w)

            done = 0
            for rows, pixels in batches(
                sources, self.side, lambda: runner.plan.batch, failures, sizes, self.cancel
            ):
                if pixels:
                    first = time.perf_counter()
                    parts = runner.run(forward, list(normalise(pixels)))
                    position = 0
                    for flat, z, (h, w) in parts:
                        for j in range(flat.shape[0]):
                            image = rows[position]
                            position += 1
                            keep = self._keep(state, h * w, n, flat.shape[2])
                            picked = (
                                np.sort(rng.choice(h * w, keep, replace=False))
                                if keep < h * w
                                else np.arange(h * w)
                            )
                            index = torch.from_numpy(picked)
                            kept_patches.append(flat[j, index])
                            kept_z.append(z[j, index])
                            kept_origins.append(
                                np.stack(
                                    [np.full(len(picked), image), picked // w, picked % w], axis=1
                                ).astype(np.int32)
                            )
                        self.grid = (h, w)
                    runner.timed(time.perf_counter() - first, len(pixels), n)
                done = rows[-1] + 1 if rows else done
                if self.progress is not None:
                    self.progress(done, n)
            if not kept_patches:
                raise ValueError("No normal image could be decoded: nothing to learn from.")
            patches = torch.cat(kept_patches)
            z = torch.cat(kept_z)
            origins = np.concatenate(kept_origins)
            del kept_patches, kept_z
            self.patches = int(patches.shape[0])
            k = max(1, math.ceil(self.coreset * self.patches))
            cap = max(1, int(self.bank_mb * 2**20 // (patches.shape[1] * 2)))
            if k > cap:
                self.notes.append(
                    f"Memory bank capped at {_mb(self.bank_mb)}: {cap:,} patches instead "
                    f"of {k:,} ({100 * cap / self.patches:.1f} % instead of "
                    f"{100 * self.coreset:.0f} %) (R-VAD-05)."
                )
                k = cap
            chosen = self._coreset(z, k, runner.plan.device)
            self.bank = patches[torch.from_numpy(chosen)].contiguous()
            self.origins = origins[chosen]
            self.notes.extend(runner.notes)
            if runner.estimate is not None:
                self.estimate = runner.estimate.sentence
        self.fit_sizes = sizes
        self.fit_failures = failures
        self.fit_seconds = time.perf_counter() - started
        self.notes.append(
            f"Memory bank: {len(chosen):,} of {self.patches:,} normal patches "
            f"({100 * len(chosen) / self.patches:.1f} %, greedy coreset after a projection to "
            f"{self.projection} dimensions) from {n - len(failures):,} images (R-VAD-05)."
        )
        if failures:
            self.notes.append(f"{len(failures):,} normal images not decoded: left out.")
        return self

    def _keep(self, state: dict[str, Any], per_image: int, images: int, channels: int) -> int:
        """Patches kept per image before the coreset (R-VAD-05: at most `store_mb` in fp16)."""
        if state["keep"] is None:
            cap = int(self.store_mb * 2**20 // (channels * 2))
            keep = min(per_image, max(1, cap // max(images, 1)))
            state["keep"] = keep
            if keep < per_image:
                self.notes.append(
                    f"{images:,} images x {per_image:,} patches exceed "
                    f"{_mb(self.store_mb)}: {keep:,} random patches kept per image "
                    "before the coreset (R-VAD-05)."
                )
        return state["keep"]

    def _coreset(self, z: Any, k: int, device: str) -> np.ndarray:
        import torch

        from quantilo.synthetic.deep.device import out_of_memory

        try:
            return greedy_coreset(z.to(torch.device(device)), k, self.seed)
        except RuntimeError as error:
            if device == "cpu" or not out_of_memory(error):
                raise
            free_gpu()
            self.notes.append("GPU ran out of memory for the coreset: chosen on the CPU.")
            return greedy_coreset(z, k, self.seed)
        finally:
            free_gpu()

    def score(self, sources: Sequence[Source]) -> PatchCoreResult:
        """Scores, smoothed maps and the nearest normal patch of every image (R-VAD-06)."""
        import torch
        import torch.nn.functional as F

        if self.bank is None or self.net is None:
            raise RuntimeError("Fit PatchCore on normal images first.")
        net = self.net
        n = len(sources)
        side = self.side
        started = time.perf_counter()
        failures: dict[int, str] = {}
        sizes: list[tuple[int, int] | None] = [None] * n
        rows_, cols_ = self.grid
        patch_scores = np.full((n, rows_, cols_), np.nan, dtype=np.float32)
        nearest_rows = np.zeros((n, rows_, cols_), dtype=np.int32)
        raw = np.full((n, side, side), np.nan, dtype=np.float32)
        ok = np.zeros(n, dtype=bool)
        cache: dict[str, Any] = {}

        def bank_on(device: Any) -> Any:
            key = str(device)
            if key not in cache:
                cache.clear()
                bank = self.bank.to(device)
                cache[key] = bank if device.type == "cuda" else bank.float()
            return cache[key]

        with Runner(
            net, self.device, self.allow_gpu, self.batch, confirm_long=self.confirm_long
        ) as runner:

            def forward(chunk: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                x = runner.tensor(chunk)
                with torch.inference_mode():
                    emb = patch_embedding(intermediates(net, x, LAYERS), self.neighbourhood)
                    b, c, h, w = emb.shape
                    query = emb.permute(0, 2, 3, 1).reshape(-1, c)
                    dist, where = nearest(query, bank_on(x.device))
                    grid = dist.view(b, 1, h, w)
                    up = F.interpolate(grid, (side, side), mode="bilinear", align_corners=False)
                    return (
                        grid[:, 0].cpu().numpy(),
                        where.view(b, h, w).cpu().numpy().astype(np.int32),
                        up[:, 0].cpu().numpy(),
                    )

            done = 0
            try:
                for rows, pixels in batches(
                    sources, side, lambda: runner.plan.batch, failures, sizes, self.cancel
                ):
                    if pixels:
                        first = time.perf_counter()
                        parts = runner.run(forward, list(normalise(pixels)))
                        scores_ = np.concatenate([p[0] for p in parts])
                        index = np.asarray(rows)
                        patch_scores[index] = scores_
                        nearest_rows[index] = np.concatenate([p[1] for p in parts])
                        raw[index] = np.concatenate([p[2] for p in parts])
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
        if ok.any():
            maps[ok] = smooth(raw[ok], self.sigma)
        scores = np.full(n, np.nan)
        if ok.any():
            scores[ok] = patch_scores[ok].reshape(int(ok.sum()), -1).max(axis=1)
        result = PatchCoreResult(
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
            grid=(rows_, cols_),
            patch_scores=patch_scores,
            nearest=nearest_rows,
            bank_origins=self.origins,
            fit_sizes=self.fit_sizes,
            neighbourhood=self.neighbourhood,
        )
        result.top = [result.explain(i) if ok[i] else None for i in range(n)]
        return result


__all__ = [
    "BACKBONES",
    "PatchCore",
    "PatchCoreResult",
    "PatchExplanation",
    "default_backbone",
    "greedy_coreset",
    "nearest",
    "patch_embedding",
]
