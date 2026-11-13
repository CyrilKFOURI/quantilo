"""Convolutional autoencoder and VAE (116, Advanced): input -> reconstruction -> error (study
§ 6.3; Bergmann et al., "Improving Unsupervised Defect Segmentation by Applying Structural
Similarity to Autoencoders", VISAPP 2019; Kingma and Welling 2014 for the VAE).

Trained on the normal images only: what the network never saw, it reconstructs badly, and the
difference is the map. Much weaker than PatchCore (the MVTec paper), and slower, but its
reconstruction speaks: the before -> after component (110) shows the image, what the model
expected, and where they differ.

- R-VAD-08: 5 blocks down (stride 2, 32 -> 256 channels), a latent of 128 numbers, 5 blocks up;
  loss = mean squared error + (1 - SSIM) (11 x 11 Gaussian window, sigma 1.5); Adam, rate 2e-4,
  weight decay 1e-5, 100 epochs, batches of 16, images at 256 px in [0, 1], seed 0. The VAE adds
  the Kullback-Leibler term (per pixel, weight 1) and reconstructs from the mean at scoring.
  Error map = 1 - SSIM (or the squared error) averaged over the colours, smoothed (Gaussian,
  sigma = 4 px); image score = the map's maximum.
- R-VDP-04, R-VDP-05: GPU when allowed (fp32), the batch halved and the training restarted on an
  out-of-memory error, the CPU at a batch of 1; the first epoch is timed and a training longer
  than 15 minutes asks first.

torch is imported on first use only."""

import itertools
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image

from quantilo.vision.deep import budget as budgets
from quantilo.vision.deep.anomaly import (
    SEED,
    SIDE,
    SIGMA,
    AnomalyResult,
    Cancelled,
    Source,
    decode_rgb,
    free_gpu,
    png,
    smooth,
)
from quantilo.vision.deep.backbones import BackboneSpec

Error = Literal["ssim", "l2"]
CHANNELS = (32, 64, 128, 256, 256)
LATENT = 128
EPOCHS = 100
BATCH = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_WEIGHT = 1.0
KL_WEIGHT = 1.0
MAX_IMAGES = 10_000  # decoded images kept in memory for training (256 px: about 2 GB)

# Plain data for the budget (no weights: the network is trained here).
SPEC = BackboneSpec(
    "conv_autoencoder",
    "Convolutional autoencoder",
    "none",
    "cnn",
    LATENT,
    "trained on the normal images",
    "advanced",
    gpu_batch=BATCH,
    cpu_batch=BATCH,
    side=SIDE,
)


def _network(side: int, latent: int, variational: bool) -> Any:
    import torch
    from torch import nn

    if side % 32:
        raise ValueError(f"The autoencoder needs a side divisible by 32 (5 halvings), not {side}.")
    bottom = side // 32

    def down(cin: int, cout: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 4, stride=2, padding=1),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def up(cin: int, cout: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    widths = (3, *CHANNELS)

    class ConvAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.variational = variational
            self.encoder = nn.Sequential(*(down(a, b) for a, b in itertools.pairwise(widths)))
            self.to_latent = nn.Conv2d(CHANNELS[-1], latent * (2 if variational else 1), bottom)
            self.from_latent = nn.Sequential(
                nn.ConvTranspose2d(latent, CHANNELS[-1], bottom), nn.ReLU(inplace=True)
            )
            back = (*CHANNELS[::-1], 32)
            self.decoder = nn.Sequential(
                *(up(a, b) for a, b in itertools.pairwise(back[:-1])),
                nn.ConvTranspose2d(back[-2], 3, 4, stride=2, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, x: Any, sample: bool = True) -> tuple[Any, Any]:
            code = self.to_latent(self.encoder(x))
            kl = torch.zeros((), device=x.device)
            if self.variational:
                mu, logvar = code.chunk(2, dim=1)
                logvar = logvar.clamp(-10, 10)
                code = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=(1, 2, 3)).mean()
            return self.decoder(self.from_latent(code)), kl

    return ConvAE()


def _window(channels: int, device: Any) -> Any:
    import torch

    r = SSIM_WINDOW // 2
    g = torch.exp(-(torch.arange(-r, r + 1, dtype=torch.float32) ** 2) / (2 * SSIM_SIGMA**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, SSIM_WINDOW, SSIM_WINDOW).to(device)


def ssim_map(x: Any, y: Any) -> Any:
    """Structural similarity per pixel and channel (Wang et al. 2004) of images in [0, 1]:
    Gaussian window 11 x 11, sigma 1.5, borders reflected. `(n, c, h, w)`."""
    import torch.nn.functional as F

    c = x.shape[1]
    w = _window(c, x.device).to(x.dtype)
    pad = SSIM_WINDOW // 2

    def blur(t: Any) -> Any:
        return F.conv2d(F.pad(t, (pad, pad, pad, pad), mode="reflect"), w, groups=c)

    c1, c2 = 0.01**2, 0.03**2
    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    cxy = blur(x * y) - mx * my
    return ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def error_map(x: Any, y: Any, kind: Error = "ssim") -> Any:
    """`(n, h, w)`: 1 - SSIM, or the squared error, averaged over the colours."""
    if kind == "ssim":
        return (1 - ssim_map(x, y)).mean(dim=1)
    return ((x - y) ** 2).mean(dim=1)


@dataclass
class BeforeAfter:
    """Image -> what the model expected -> where they differ (for the component of 110)."""

    input: Image.Image
    reconstruction: Image.Image
    error: Image.Image  # grey, 0 = no error, 255 = `scale` or more
    scale: float
    sentence: str

    def pngs(self) -> tuple[bytes, bytes, bytes]:
        return png(self.input), png(self.reconstruction), png(self.error)


@dataclass
class ReconstructionResult(AnomalyResult):
    inputs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0, 3), np.uint8))
    reconstructions: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0, 3), np.uint8))
    error: str = "ssim"

    def before_after(self, i: int, scale: float | None = None) -> BeforeAfter:
        """Image `i` at the model's side, its reconstruction, and its error map in grey, scaled
        to `scale` (default: the largest error among the images scored)."""
        if not self.ok[i]:
            raise ValueError(f"Image {i} was not scored: {self.failures.get(i, 'not decoded')}.")
        top = scale or float(np.nanmax(self.maps[self.ok])) or 1.0
        grey = np.clip(self.maps[i] / top, 0, 1)
        what = "1 - SSIM" if self.error == "ssim" else "squared error"
        return BeforeAfter(
            Image.fromarray(self.inputs[i]),
            Image.fromarray(self.reconstructions[i]),
            Image.fromarray((grey * 255).round().astype(np.uint8), mode="L"),
            top,
            f"Left: the image; middle: what {self.title} expected from normal images; right: "
            f"where they differ ({what}, white = {top:.3g} or more) (R-VAD-08).",
        )

    def subset(self, rows: Sequence[int]) -> "ReconstructionResult":
        index = np.asarray(rows, dtype=int)
        out = super().subset(rows)
        assert isinstance(out, ReconstructionResult)
        out.inputs = self.inputs[index]
        out.reconstructions = self.reconstructions[index]
        return out


class ConvAutoencoder:
    """Fit on normal images (`fit`, trains the network), then score any image (`score`)."""

    def __init__(
        self,
        *,
        variational: bool = False,
        side: int = SIDE,
        latent: int = LATENT,
        epochs: int = EPOCHS,
        batch: int = BATCH,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        ssim_weight: float = SSIM_WEIGHT,
        kl_weight: float = KL_WEIGHT,
        error: Error = "ssim",
        sigma: float = SIGMA,
        device: budgets.Choice = "auto",
        allow_gpu: bool = True,
        confirm_long: bool = False,
        progress: Callable[[int, int], None] | None = None,
        cancel: Callable[[], bool] | None = None,
        seed: int = SEED,
    ) -> None:
        if side % 32:
            raise ValueError(f"The autoencoder needs a side divisible by 32, not {side}.")
        self.variational = variational
        self.side = side
        self.latent = latent
        self.epochs = epochs
        self.batch = batch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ssim_weight = ssim_weight
        self.kl_weight = kl_weight
        self.error: Error = error
        self.sigma = sigma
        self.device: budgets.Choice = device
        self.allow_gpu = allow_gpu
        self.confirm_long = confirm_long
        self.progress = progress
        self.cancel = cancel
        self.seed = seed
        self.module: Any = None
        self.losses: list[float] = []
        self.notes: list[str] = []
        self.fit_failures: dict[int, str] = {}
        self.fit_seconds = 0.0
        self.estimate: str | None = None
        self.label = "CPU"

    @property
    def method(self) -> str:
        return "vae" if self.variational else "autoencoder"

    @property
    def title(self) -> str:
        return "Variational autoencoder" if self.variational else "Convolutional autoencoder"

    @property
    def model_key(self) -> str:
        return (
            f"{self.method}|side={self.side}|latent={self.latent}|epochs={self.epochs}|"
            f"ssim={self.ssim_weight}|kl={self.kl_weight}|seed={self.seed}"
        )

    def _pixels(self, sources: Sequence[Source]) -> tuple[np.ndarray, dict[int, str], list]:
        if len(sources) > MAX_IMAGES:
            raise ValueError(
                f"{len(sources):,} images: the autoencoder trains on {MAX_IMAGES:,} at most."
            )
        failures: dict[int, str] = {}
        sizes: list[tuple[int, int] | None] = [None] * len(sources)
        kept: list[np.ndarray] = []
        for i, source in enumerate(sources):
            decoded = decode_rgb(source, self.side)
            if isinstance(decoded, str):
                failures[i] = decoded
                continue
            kept.append(decoded[0])
            sizes[i] = decoded[1]
        pixels = np.stack(kept) if kept else np.zeros((0, self.side, self.side, 3), np.uint8)
        return pixels, failures, sizes

    def fit(self, sources: Sequence[Source]) -> "ConvAutoencoder":
        """Train on normal images (R-VAD-08). Raises `LongRun` when the first epoch says the
        training would exceed 15 minutes and `confirm_long` is False."""
        from quantilo.synthetic.deep.device import out_of_memory, pick

        started = time.perf_counter()
        pixels, failures, _ = self._pixels(sources)
        if len(pixels) < 2:
            raise ValueError("The autoencoder needs at least 2 decoded normal images.")
        chosen = pick(self.device, self.allow_gpu)
        device, label = chosen.device, chosen.label
        notes = [chosen.note] if chosen.note else []
        batch = max(1, min(self.batch, len(pixels)))
        while True:
            try:
                self._train(pixels, device, batch, label)
                break
            except RuntimeError as error:
                if device.type == "cpu" or not out_of_memory(error):
                    raise
                free_gpu()
                if batch > 1:
                    batch //= 2
                    notes.append(
                        f"GPU ran out of memory: batch halved to {batch}, training restarted "
                        "(R-VDP-04)."
                    )
                else:
                    import torch

                    device, label = torch.device("cpu"), "CPU"
                    notes.append("GPU ran out of memory with a batch of 1: trained on the CPU.")
        self.label = label
        self.notes = [
            f"Trained on {label} for {self.epochs} epochs, batches of {batch}, fp32 (R-VAD-08).",
            *notes,
        ]
        self.fit_failures = failures
        if failures:
            self.notes.append(f"{len(failures):,} normal images not decoded: left out.")
        self.fit_seconds = time.perf_counter() - started
        return self

    def _train(self, pixels: np.ndarray, device: Any, batch: int, label: str) -> None:
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        module = _network(self.side, self.latent, self.variational).to(device)
        optimiser = torch.optim.Adam(
            module.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        data = torch.from_numpy(pixels).permute(0, 3, 1, 2)  # uint8 on the CPU
        n = len(data)
        steps = math.ceil(n / batch)
        self.losses = []
        self.estimate = None
        module.train()
        try:
            for epoch in range(self.epochs):
                first = time.perf_counter()
                order = torch.from_numpy(rng.permutation(n))
                total = 0.0
                for step in range(steps):
                    if self.cancel is not None and self.cancel():
                        raise Cancelled("Training cancelled.")
                    rows = order[step * batch : (step + 1) * batch]
                    if len(rows) < 2 and n >= 2:  # batch norm needs 2 images
                        continue
                    x = data[rows].to(device).float() / 255.0
                    out, kl = module(x)
                    loss = torch.nn.functional.mse_loss(out, x)
                    if self.ssim_weight:
                        loss = loss + self.ssim_weight * (1 - ssim_map(out, x).mean())
                    if self.variational:
                        loss = loss + self.kl_weight * kl / x[0].numel()
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    optimiser.step()
                    total += float(loss.detach()) * len(rows)
                self.losses.append(total / n)
                if epoch == 0:
                    seconds = (time.perf_counter() - first) * self.epochs
                    estimate = budgets.Estimate(seconds, n, label)
                    self.estimate = (
                        f"≈ {budgets.duration_text(seconds)} for {self.epochs} epochs on "
                        f"{n:,} images on {label}"
                    )
                    if estimate.needs_confirmation and not self.confirm_long:
                        raise budgets.LongRun(estimate)
                if self.progress is not None:
                    self.progress(epoch + 1, self.epochs)
        except BaseException:
            module.to("cpu")
            raise
        module.eval()
        self.module = module.to("cpu")

    def score(self, sources: Sequence[Source]) -> ReconstructionResult:
        """Reconstruction, error map and score of every image (R-VAD-08)."""
        import torch

        from quantilo.synthetic.deep.device import out_of_memory, pick

        if self.module is None:
            raise RuntimeError("Fit the autoencoder on normal images first.")
        started = time.perf_counter()
        n = len(sources)
        side = self.side
        inputs = np.zeros((n, side, side, 3), dtype=np.uint8)
        outputs = np.zeros((n, side, side, 3), dtype=np.uint8)
        raw = np.full((n, side, side), np.nan, dtype=np.float32)
        ok = np.zeros(n, dtype=bool)
        failures: dict[int, str] = {}
        sizes: list[tuple[int, int] | None] = [None] * n
        rows: list[int] = []
        for i, source in enumerate(sources):
            decoded = decode_rgb(source, side)
            if isinstance(decoded, str):
                failures[i] = decoded
                continue
            inputs[i] = decoded[0]
            sizes[i] = decoded[1]
            rows.append(i)
        chosen = pick(self.device, self.allow_gpu)
        device, label = chosen.device, chosen.label
        batch = max(1, self.batch)
        start = 0
        while start < len(rows):
            part = rows[start : start + batch]
            try:
                self.module.to(device)
                with torch.inference_mode():
                    x = torch.from_numpy(inputs[part]).permute(0, 3, 1, 2).to(device).float() / 255
                    out, _ = self.module(x, sample=False)
                    err = error_map(out, x, self.error)
                pictures = (out.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1)
                outputs[part] = pictures.cpu().numpy()
                raw[part] = err.cpu().numpy()
                ok[part] = True
            except RuntimeError as error:
                if device.type == "cpu" or not out_of_memory(error):
                    raise
                free_gpu()
                if batch > 1:
                    batch //= 2
                else:
                    device, label = torch.device("cpu"), "CPU"
                continue
            start += len(part)
            if self.progress is not None:
                self.progress(min(start, len(rows)), len(rows))
        self.module.to("cpu")
        free_gpu()
        maps = raw.copy()
        scores = np.full(n, np.nan)
        if ok.any():
            maps[ok] = smooth(raw[ok], self.sigma)
            scores[ok] = maps[ok].reshape(int(ok.sum()), -1).max(axis=1)
        notes = list(self.notes)
        notes.append(
            "Error map: "
            + ("1 - SSIM" if self.error == "ssim" else "squared error")
            + f", averaged over the colours, smoothed (sigma = {self.sigma:g} px); score = its "
            "maximum (R-VAD-08)."
        )
        return ReconstructionResult(
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
            self.estimate,
            notes,
            failures,
            inputs=inputs,
            reconstructions=outputs,
            error=self.error,
        )


__all__ = [
    "BeforeAfter",
    "ConvAutoencoder",
    "ReconstructionResult",
    "error_map",
    "ssim_map",
]
