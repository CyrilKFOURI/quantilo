"""Fine-tuning the top of a pretrained backbone (118; study §§ 6.1, 6.3 Advanced › More, 10).

The frozen probe (114) already holds most of the gain; fine-tuning is the last step, worth it
only when the gap to the probe says so (§ 6.1). It is never run before the user asks.

- R-VFT-01 what learns: the top of the network and a new linear head. A ViT: its last 2
  blocks and its final norm; a ResNet: its last stage. Everything below stays frozen, so no
  gradient (and no activation kept for one) goes below the first trainable block: memory stays
  small. Batch norms keep their ImageNet statistics (the whole network stays in evaluation mode):
  batches of 32 or 64 images estimate them badly.
- R-VFT-02 discriminative learning rates (Howard and Ruder, 2018): the head 1e-3, the last block
  1e-4, each block below half the one above; AdamW (weight decay 0.05), cosine decay, label
  smoothing 0.1; classes weighted when the largest is more than 3 times the smallest (R-VCL-05).
- R-VFT-03 images: the centre square the probe reads, kept once at 256 px (decoded by
  `vision.decode`); training sees a random crop of 60 to 100 % of it, flipped half the time;
  scoring sees the probe's own view (DINOv2: the whole square; ResNet: its centre 87.5 %).
- R-VFT-04 early stopping: 15 % of the training images, grouped by copies and stratified, are
  held out; training stops when their loss has not improved for 3 epochs (at most 15), and the
  weights of the best epoch are kept.
- R-VFT-05 mixed precision (AMP, fp16 with a loss scaler) on a GPU; fp32 on the CPU. A GPU
  out-of-memory error halves the batch and starts the epoch again, said; at a batch of 1 the
  run moves to the CPU, said (R-VDP-04).
- R-VFT-06 the first epoch is timed: "≈ 2 min for 15 epochs on ..."; beyond 15 minutes the run
  stops and asks first (`LongRun`, R-VDP-05).

torch is imported inside the functions only (extra `vision`, R-VDP-01).
"""

import math
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from quantilo.vision.deep import budget as budgets
from quantilo.vision.deep.backbones import Backbone, BackboneSpec

BANK_SIDE = 256  # R-VFT-03
BANK_BUDGET_MB = 1_536  # above it, images are decoded again at each batch (said)
TOP_BLOCKS = {"vit": 2, "cnn": 1}  # R-VFT-01
LR_HEAD = 1e-3  # R-VFT-02
LR_TOP = 1e-4
LR_DECAY = 0.5
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
CROP_SCALE = (0.6, 1.0)  # R-VFT-03
MAX_EPOCHS = 15  # R-VFT-04
PATIENCE = 3
VAL_SHARE = 0.15
MIN_VALIDATION = 20  # fewer training images: no early stopping, a fixed number of epochs
FIXED_EPOCHS = 5
IMBALANCE = 3.0  # R-VCL-05
# study § 10: batches of the fine-tuning on a 6 GB card, with AMP
GPU_BATCH = {"vit": 64, "cnn": 32}
CPU_BATCH = 16

Source = str | Path | bytes | Image.Image | None
Progress = Callable[[float, str], None]
Cancel = Callable[[], bool]


class Cancelled(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# The images, once


def square(image: Image.Image, side: int = BANK_SIDE, bicubic: bool = True) -> np.ndarray:
    """The centre square of the image (the region a 224-px network reads after its short-side
    resize), resized to `side`: `(side, side, 3)` uint8."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    resample = Image.Resampling.BICUBIC if bicubic else Image.Resampling.BILINEAR
    cropped = image.crop((left, top, left + s, top + s)).resize((side, side), resample)
    return np.asarray(cropped, dtype=np.uint8)


def _decode(source: Source, side: int) -> Image.Image | None:
    if source is None:
        return None
    if isinstance(source, Image.Image):
        return source
    from quantilo.vision.decode import DecodeFailure, try_decode

    decoded = try_decode(source, reduce_to=side)
    return None if isinstance(decoded, DecodeFailure) else decoded.image


class ImageBank:
    """The centre square of each image at 256 px (R-VFT-03), decoded once and kept in memory
    when the whole set fits in `BANK_BUDGET_MB`, decoded again at each batch otherwise."""

    def __init__(
        self,
        sources: Sequence[Source],
        spec: BackboneSpec,
        rows: Sequence[int] | np.ndarray | None = None,
        *,
        side: int = BANK_SIDE,
        progress: Progress | None = None,
        cancel: Cancel | None = None,
    ) -> None:
        self.sources = list(sources)
        self.side = side
        self.bicubic = spec.interpolation == "bicubic"
        wanted = sorted(set(range(len(self.sources)) if rows is None else map(int, rows)))
        self.kept = len(wanted) * side * side * 3 <= BANK_BUDGET_MB * 1024 * 1024
        self.notes: list[str] = []
        if not self.kept:
            self.notes.append(
                f"{len(wanted):,} images do not fit in {BANK_BUDGET_MB:,} MB at {side} px: they "
                "are decoded again at each batch (slower)."
            )
        self._pixels: dict[int, np.ndarray] = {}
        self.failed: set[int] = set()
        for i, row in enumerate(wanted):
            if cancel is not None and cancel():
                raise Cancelled("Stopped: nothing was saved.")
            pixels = self._load(row)
            if pixels is None:
                self.failed.add(row)
            elif self.kept:
                self._pixels[row] = pixels
            if progress is not None and (i % 64 == 0 or i == len(wanted) - 1):
                progress((i + 1) / max(1, len(wanted)), f"Images: {i + 1:,} / {len(wanted):,}")

    def _load(self, row: int) -> np.ndarray | None:
        image = _decode(self.sources[row], self.side)
        return None if image is None else square(image, self.side, self.bicubic)

    def ok(self, rows: Sequence[int] | np.ndarray) -> np.ndarray:
        return np.array([int(r) not in self.failed for r in rows], dtype=bool)

    def pixels(self, rows: Sequence[int] | np.ndarray) -> np.ndarray:
        """`(len(rows), side, side, 3)` uint8 (a row that failed is black)."""
        out = np.zeros((len(rows), self.side, self.side, 3), dtype=np.uint8)
        for i, r in enumerate(rows):
            r = int(r)
            if r in self.failed:
                continue
            found = self._pixels.get(r) if self.kept else self._load(r)
            if found is not None:
                out[i] = found
        return out


def eval_view(x: Any, spec: BackboneSpec) -> Any:
    """`(n, 3, 256, 256)` float in [0, 1] -> the probe's own view at `spec.side` (R-VFT-03): the
    centre `crop` of the square (DINOv2: all of it), resized."""
    import torch.nn.functional as functional

    side = x.shape[-1]
    keep = round(side * spec.crop)
    start = (side - keep) // 2
    x = x[..., start : start + keep, start : start + keep]
    if keep != spec.side:
        x = functional.interpolate(x, size=(spec.side, spec.side), mode="bilinear", antialias=True)
    return x


def augment(x: Any, spec: BackboneSpec, generator: np.random.Generator) -> Any:
    """R-VFT-03: a random crop of 60 to 100 % of the square's area (aspect 3/4 to 4/3), resized
    to `spec.side`, flipped half the time."""
    import torch
    import torch.nn.functional as functional

    n, _, side, _ = x.shape
    out = []
    for i in range(n):
        area = generator.uniform(*CROP_SCALE) * side * side
        ratio = math.exp(generator.uniform(math.log(3 / 4), math.log(4 / 3)))
        w = int(min(side, round(math.sqrt(area * ratio))))
        h = int(min(side, round(math.sqrt(area / ratio))))
        top = int(generator.integers(0, side - h + 1))
        left = int(generator.integers(0, side - w + 1))
        part = x[i : i + 1, :, top : top + h, left : left + w]
        part = functional.interpolate(
            part, size=(spec.side, spec.side), mode="bilinear", antialias=True
        )
        if generator.random() < 0.5:
            part = torch.flip(part, dims=[3])
        out.append(part)
    return torch.cat(out, dim=0)


def to_input(pixels: np.ndarray, spec: BackboneSpec, device: Any) -> Any:
    """uint8 `(n, side, side, 3)` -> float `(n, 3, side, side)` in [0, 1] on `device`."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(pixels)).to(device)
    return x.permute(0, 3, 1, 2).float().div_(255.0)


def normalise(x: Any, spec: BackboneSpec) -> Any:
    import torch

    mean = torch.tensor(spec.mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(spec.std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def pooled(module: Any, spec: BackboneSpec, tokens: Any) -> Any:
    """The backbone's vector from `forward_features` (as `Backbone.features`, kept in the input's
    precision and differentiable)."""
    import torch

    if spec.kind == "vit":
        prefix = int(getattr(module, "num_prefix_tokens", 1))
        return torch.cat([tokens[:, 0], tokens[:, prefix:].mean(dim=1)], dim=1)
    return tokens.mean(dim=(2, 3))


# ---------------------------------------------------------------------------------------------
# What learns


def trainable_blocks(module: Any, kind: str, n: int | None = None) -> list[tuple[str, Any]]:
    """R-VFT-01: the top `n` blocks, from the output down (a ViT's `blocks`, a ResNet's
    stages)."""
    n = TOP_BLOCKS.get(kind, 1) if n is None else n
    if kind == "vit" and hasattr(module, "blocks"):
        blocks = list(module.blocks)
        picked = [(f"block {i + 1}", blocks[i]) for i in range(len(blocks) - 1, -1, -1)]
    else:
        names = [name for name in ("layer4", "layer3", "layer2", "layer1") if hasattr(module, name)]
        picked = [(name.replace("layer", "stage "), getattr(module, name)) for name in names]
    if not picked:
        raise ValueError("This network has no blocks or stages to fine-tune (R-VFT-01).")
    return picked[: max(1, n)]


def validation_rows(
    y: np.ndarray, groups: np.ndarray, seed: int = 0, share: float = VAL_SHARE
) -> np.ndarray:
    """R-VFT-04: positions (in `y`) held out for early stopping, grouped by copies and
    stratified; empty when there are too few images."""
    from sklearn.model_selection import StratifiedGroupKFold

    if len(y) < MIN_VALIDATION or len(np.unique(groups)) < 4:
        return np.zeros(0, dtype=int)
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        return np.zeros(0, dtype=int)
    folds = max(2, round(1 / share))
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    _, held = next(splitter.split(np.zeros(len(y)), y, groups))
    return np.sort(held)


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float | None
    val_balanced_accuracy: float | None
    seconds: float


@dataclass
class TrainingLog:
    """What happened during training, said on the Results tab."""

    device: str
    batch: int
    precision: str  # "fp16 (AMP)" or "fp32"
    epochs: int  # epochs run
    best_epoch: int
    stopped: str  # why training stopped, in words
    history: list[EpochLog] = field(default_factory=list)
    seconds: float = 0.0
    estimate: str | None = None  # R-VFT-06, after the first epoch
    trained: str = ""  # what learnt, in words
    notes: list[str] = field(default_factory=list)


def _balanced(y: np.ndarray, pred: np.ndarray) -> float:
    classes = np.unique(y)
    return float(np.mean([np.mean(pred[y == c] == c) for c in classes]))


class FineTuned:
    """The network after fine-tuning: its probabilities and vectors for any rows of the bank."""

    def __init__(
        self,
        net: Backbone,
        head: Any,
        bank: ImageBank,
        device: Any,
        log: TrainingLog,
        batch: int,
    ) -> None:
        self.net = net
        self.head = head
        self.bank = bank
        self.device = device
        self.log = log
        self.batch = batch

    def _forward(self, rows: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        spec = self.net.spec
        module = self.net.module
        vectors, scores = [], []
        with torch.inference_mode():
            for start in range(0, len(rows), self.batch):
                chunk = list(rows[start : start + self.batch])
                x = normalise(
                    eval_view(to_input(self.bank.pixels(chunk), spec, self.device), spec), spec
                )
                with torch.autocast(
                    self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
                ):
                    v = pooled(module, spec, module.forward_features(x))
                    s = self.head(v)
                vectors.append(v.float().cpu().numpy())
                scores.append(s.float().cpu().numpy())
        if not rows:
            k = int(self.head.out_features)
            return np.zeros((0, spec.dims), np.float32), np.zeros((0, k))
        return np.concatenate(vectors).astype(np.float32), np.concatenate(scores).astype(np.float64)

    def predict(self, rows: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(probabilities `(n, k)` in float64, fine-tuned vectors `(n, dims)`)."""
        vectors, scores = self._forward(rows)
        scores -= scores.max(axis=1, keepdims=True)
        e = np.exp(scores)
        return e / e.sum(axis=1, keepdims=True), vectors

    def head_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """The head as numpy `(weight (k, d), bias (k,))`: probabilities from the kept vectors."""
        weight = self.head.weight.detach().float().cpu().numpy().astype(np.float64)
        bias = self.head.bias.detach().float().cpu().numpy().astype(np.float64)
        return weight, bias

    def release(self) -> None:
        import torch

        self.net.module.to("cpu").float()
        self.head.to("cpu").float()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def finetune(
    net: Backbone,
    bank: ImageBank,
    train_rows: Sequence[int] | np.ndarray,
    y: Sequence[int] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    n_classes: int,
    *,
    allow_gpu: bool = True,
    confirm_long: bool = False,
    seed: int = 0,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch: int | None = None,
    blocks: int | None = None,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
) -> FineTuned:
    """Fine-tune the top of `net` (R-VFT-01 to 06) on `train_rows` of the bank, labels `y`
    coded 0..k-1 (aligned with `train_rows`). `net` is modified in place: load a fresh one for
    each fold. Raises `LongRun` after the first epoch when the run would pass 15 minutes."""
    import torch

    spec = net.spec
    module = net.module
    rows = np.asarray(train_rows, dtype=int)
    codes = np.asarray(y, dtype=int)
    group_arr = np.asarray([str(g) for g in groups], dtype=object)
    good = bank.ok(rows)
    rows, codes, group_arr = rows[good], codes[good], group_arr[good]
    if len(np.unique(codes)) < 2:
        raise ValueError("Fine-tuning needs at least 2 classes among the training images.")
    held = validation_rows(codes, group_arr, seed)
    learn = np.setdiff1d(np.arange(len(rows)), held)
    notes = list(bank.notes)
    epochs_cap = max_epochs if len(held) else min(max_epochs, FIXED_EPOCHS)
    if not len(held):
        notes.append(
            f"Too few training images to hold some out: {epochs_cap} epochs, no early stopping "
            "(R-VFT-04)."
        )

    from quantilo.synthetic.deep.device import out_of_memory, pick

    device = pick("auto", allow_gpu)
    notes += [device.note] if device.note else []
    on_gpu = device.device.type == "cuda"
    if device.device.type == "cpu":
        size = batch or CPU_BATCH
    else:
        memory = budgets._gpu_memory_mb(device.device)
        size = budgets.scaled_batch(GPU_BATCH.get(spec.kind, 32), memory)
        size = min(batch, size) if batch else size
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)

    # R-VFT-01: freeze everything, then open the top blocks; the head is new
    for p in module.parameters():
        p.requires_grad_(False)
    top = trainable_blocks(module, spec.kind, blocks)
    groups_lr: list[dict[str, Any]] = []
    for depth, (_name, block) in enumerate(top):
        params = list(block.parameters())
        for p in params:
            p.requires_grad_(True)
        groups_lr.append({"params": params, "lr": LR_TOP * LR_DECAY**depth})
    norm = getattr(module, "norm", None) if spec.kind == "vit" else None
    if norm is not None and any(True for _ in norm.parameters()):
        params = list(norm.parameters())
        for p in params:
            p.requires_grad_(True)
        groups_lr.append({"params": params, "lr": LR_TOP})
    head = torch.nn.Linear(spec.dims, n_classes)
    groups_lr.append({"params": list(head.parameters()), "lr": LR_HEAD})
    trained = (
        f"the last {len(top)} {'blocks' if spec.kind == 'vit' else 'stage'}"
        f"{' and the final norm' if norm is not None else ''} of {spec.title}, and a new linear "
        f"head (learning rates: head {LR_HEAD:g}, top {LR_TOP:g}, halved per block below; "
        "R-VFT-02)"
    )

    counts = np.bincount(codes[learn], minlength=n_classes).astype(np.float64)
    weight = None
    present = counts[counts > 0]
    if present.max() / present.min() > IMBALANCE:
        w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
        weight = torch.tensor(w, dtype=torch.float32)
        notes.append(
            "Classes weighted by their inverse frequency: the largest is more than 3 times the "
            "smallest (R-VCL-05)."
        )

    def place(where: Any) -> None:
        module.to(where).float()
        head.to(where).float()

    place(device.device)
    module.eval()  # R-VFT-01: batch norms keep their statistics; no dropout in the frozen part
    optimiser = torch.optim.AdamW(groups_lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, math.ceil(len(learn) / size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, epochs_cap * steps_per_epoch)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=on_gpu)  # pyright: ignore[reportPrivateImportUsage]
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=None if weight is None else weight.to(device.device),
        label_smoothing=LABEL_SMOOTHING,
    )
    plain_loss = torch.nn.CrossEntropyLoss(
        weight=None if weight is None else weight.to(device.device)
    )
    trainable = [p for g in groups_lr for p in g["params"]]

    def run_epoch() -> float:
        order = generator.permutation(learn)
        total, seen = 0.0, 0
        for start in range(0, len(order), size):
            if cancel is not None and cancel():
                raise Cancelled("Stopped: nothing was saved.")
            chunk = order[start : start + size]
            x = to_input(bank.pixels(rows[chunk]), spec, device.device)
            x = normalise(augment(x, spec, generator), spec)
            target = torch.from_numpy(codes[chunk]).to(device.device)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device.device.type, dtype=torch.float16, enabled=on_gpu):
                scores = head(pooled(module, spec, module.forward_features(x)))
                loss = loss_fn(scores.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimiser)
            scaler.update()
            with warnings.catch_warnings():
                # the loss scaler skips a step when fp16 overflows: the schedule moves on anyway
                warnings.simplefilter("ignore", UserWarning)
                scheduler.step()
            total += float(loss.detach()) * len(chunk)
            seen += len(chunk)
        return total / max(1, seen)

    def evaluate_held() -> tuple[float, float]:
        scores = []
        with torch.inference_mode():
            for start in range(0, len(held), size):
                chunk = held[start : start + size]
                x = normalise(
                    eval_view(to_input(bank.pixels(rows[chunk]), spec, device.device), spec), spec
                )
                with torch.autocast(device.device.type, dtype=torch.float16, enabled=on_gpu):
                    scores.append(head(pooled(module, spec, module.forward_features(x))).float())
        s = torch.cat(scores)
        target = torch.from_numpy(codes[held]).to(device.device)
        loss = float(plain_loss(s, target))
        pred = s.argmax(dim=1).cpu().numpy()
        return loss, _balanced(codes[held], pred)

    def snapshot() -> list[Any]:
        return [p.detach().to("cpu", copy=True) for p in trainable]

    history: list[EpochLog] = []
    best_loss, best_epoch, best_state = math.inf, 0, snapshot()
    waited = 0
    started = time.perf_counter()
    estimate: budgets.Estimate | None = None
    stopped = ""
    epoch = 0
    label = device.label
    try:
        while epoch < epochs_cap:
            t0 = time.perf_counter()
            try:
                train_loss = run_epoch()
            except RuntimeError as error:
                if device.device.type == "cpu" or not out_of_memory(error):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if size > 1:
                    size = max(1, size // 2)
                    notes.append(
                        f"GPU ran out of memory: batch halved to {size}, epoch {epoch + 1} started "
                        "again (R-VFT-05)."
                    )
                else:
                    device = pick("cpu", False)
                    on_gpu, label = False, "CPU"
                    place(device.device)
                    size = CPU_BATCH
                    notes.append("GPU ran out of memory with a batch of 1: finished on the CPU.")
                optimiser.zero_grad(set_to_none=True)
                continue
            val_loss = val_bacc = None
            if len(held):
                val_loss, val_bacc = evaluate_held()
            epoch += 1
            history.append(
                EpochLog(
                    epoch,
                    round(train_loss, 5),
                    None if val_loss is None else round(val_loss, 5),
                    None if val_bacc is None else round(val_bacc, 5),
                    round(time.perf_counter() - t0, 2),
                )
            )
            if estimate is None:
                estimate = budgets.Estimate(
                    (time.perf_counter() - t0) * epochs_cap, len(learn), label
                )
                if estimate.needs_confirmation and not confirm_long:
                    raise budgets.LongRun(estimate)
            if progress is not None:
                text = f"Fine-tuning: epoch {epoch} / {epochs_cap}"
                if val_loss is not None:
                    text += f", held-out loss {val_loss:.3f}"
                progress(epoch / epochs_cap, text)
            if val_loss is None:
                best_epoch, best_state = epoch, snapshot()
                continue
            if val_loss < best_loss - 1e-4:
                best_loss, best_epoch, best_state, waited = val_loss, epoch, snapshot(), 0
            else:
                waited += 1
                if waited >= patience:
                    stopped = (
                        f"Stopped after epoch {epoch}: the held-out loss had not improved for "
                        f"{patience} epochs; the weights of epoch {best_epoch} are kept (R-VFT-04)."
                    )
                    break
        if not stopped:
            stopped = (
                f"Ran all {epochs_cap} epochs; the weights of epoch {best_epoch} (lowest held-out "
                "loss) are kept (R-VFT-04)."
                if len(held)
                else f"Ran {epochs_cap} epochs (R-VFT-04)."
            )
        with torch.no_grad():
            for p, kept in zip(trainable, best_state, strict=True):
                p.copy_(kept.to(p.device))
    except BaseException:
        module.to("cpu").float()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    for p in module.parameters():
        p.requires_grad_(False)
    head.requires_grad_(False)
    log = TrainingLog(
        device=label,
        batch=size,
        precision="fp16 (AMP)" if on_gpu else "fp32",
        epochs=epoch,
        best_epoch=best_epoch,
        stopped=stopped,
        history=history,
        seconds=round(time.perf_counter() - started, 2),
        estimate=None
        if estimate is None
        else f"≈ {budgets.duration_text(estimate.seconds)} for at most {epochs_cap} epochs on "
        f"{estimate.label} (timed on the first epoch, R-VFT-06)",
        trained=trained,
        notes=notes,
    )
    if len(held):
        log.notes.insert(
            0,
            f"{len(held):,} of the {len(rows):,} training images held out for early stopping, "
            "grouped by copies (R-VFT-04).",
        )
    return FineTuned(net, head, bank, device.device, log, size)


__all__ = [
    "BANK_SIDE",
    "Cancelled",
    "EpochLog",
    "FineTuned",
    "ImageBank",
    "TrainingLog",
    "augment",
    "eval_view",
    "finetune",
    "normalise",
    "pooled",
    "square",
    "to_input",
    "trainable_blocks",
    "validation_rows",
]
