"""Compute budget of the deep vision models (114; study § 10).

- R-VDP-04: the GPU when it is ready and allowed (the app's *Use the GPU when available*), in
  fp16, with the batch of the study's table for a 6 GB card (smaller cards: in proportion);
  otherwise the CPU in fp32, the CPU batch, threads = physical cores. A GPU out-of-memory error
  halves the batch and tries again, said in the notes; at a batch of 1 the run finishes on the
  CPU, said too.
- R-VDP-05: the first batch is timed; the estimate is shown ("≈ 3 min for 1,200 images on ...")
  and a run longer than 15 minutes asks for confirmation first.

torch is imported inside the functions that need it, never at module level."""

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from quantilo.vision.deep.backbones import BackboneSpec

Choice = Literal["auto", "cpu", "gpu"]

REFERENCE_MB = 6 * 1024  # the batches of § 10 are sized for a 6 GB card (RTX 4050 Laptop)
REFERENCE_TOLERANCE = 0.97  # what the driver reports is a little under the nominal size
LONG_RUN_SECONDS = 15 * 60  # R-VDP-05
CPU_BATCH_CAP = 32


class GpuExhausted(RuntimeError):
    """Out of GPU memory even with a batch of 1: the caller moves to the CPU (R-VDP-04)."""


class LongRun(RuntimeError):
    """R-VDP-05: the run would take more than 15 minutes; ask before going on."""

    def __init__(self, estimate: "Estimate") -> None:
        self.estimate = estimate
        super().__init__(
            f"{estimate.sentence}: more than {LONG_RUN_SECONDS // 60} minutes, confirm to go on "
            "(R-VDP-05)."
        )


@dataclass
class Budget:
    device: str  # "cuda", "mps" or "cpu"
    label: str  # "NVIDIA GeForce RTX 4050 Laptop GPU" or "CPU"
    batch: int
    half: bool  # fp16 weights and inputs (GPU only)
    threads: int | None = None  # CPU threads (None on a GPU)
    memory_mb: int | None = None  # total memory of the card
    notes: list[str] = field(default_factory=list)

    @property
    def on_gpu(self) -> bool:
        return self.device != "cpu"

    @property
    def sentence(self) -> str:
        precision = "fp16" if self.half else "fp32"
        where = self.label if self.on_gpu else f"CPU, {self.threads} threads"
        return f"{where}, batches of {self.batch}, {precision} (R-VDP-04)"


@dataclass
class Estimate:
    seconds: float
    images: int
    label: str

    @property
    def needs_confirmation(self) -> bool:
        return self.seconds > LONG_RUN_SECONDS

    @property
    def sentence(self) -> str:
        return f"≈ {duration_text(self.seconds)} for {self.images:,} images on {self.label}"


def duration_text(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def physical_cores() -> int:
    """Physical cores, approximated as half the logical ones on a machine with hyper-threading
    (the standard library does not tell them apart); at least 1."""
    logical = os.cpu_count() or 1
    return max(1, logical // 2) if logical >= 4 else logical


def scaled_batch(base: int, memory_mb: int | None) -> int:
    """The batch of a 6 GB card, scaled down to the card's memory (power of 2, at least 1). A
    card within 3 % of 6 GB counts as one: the RTX 4050 Laptop reports 6,141 MiB."""
    if not memory_mb or memory_mb >= REFERENCE_MB * REFERENCE_TOLERANCE:
        return base
    share = base * memory_mb / REFERENCE_MB
    return max(1, 2 ** math.floor(math.log2(max(share, 1))))


def _gpu_memory_mb(device: Any) -> int | None:
    import torch

    if device.type != "cuda":
        return None
    try:
        return int(torch.cuda.get_device_properties(device).total_memory // (1024 * 1024))
    except (RuntimeError, AssertionError):
        return None


def plan(
    spec: BackboneSpec,
    choice: Choice = "auto",
    allow_gpu: bool = True,
    batch: int | None = None,
) -> Budget:
    """Where and how the network runs (R-VDP-04). `batch` overrides the table (capped by it)."""
    from quantilo.synthetic.deep.device import pick

    device = pick(choice, allow_gpu)
    notes = [device.note] if device.note else []
    if device.device.type == "cpu":
        size = min(batch or spec.cpu_batch, max(spec.cpu_batch, CPU_BATCH_CAP))
        return Budget("cpu", "CPU", size, False, physical_cores(), None, notes)
    memory = _gpu_memory_mb(device.device)
    table = scaled_batch(spec.gpu_batch, memory)
    size = min(batch, table) if batch else table
    half = device.device.type == "cuda"  # MPS: fp32 (fp16 support varies)
    return Budget(device.device.type, device.label, size, half, None, memory, notes)


def cpu_budget(spec: BackboneSpec, note: str | None = None) -> Budget:
    notes = [note] if note else []
    return Budget("cpu", "CPU", spec.cpu_batch, False, physical_cores(), None, notes)


def run_in_batches[T](
    fn: Callable[[Sequence[Any]], T],
    items: Sequence[Any],
    budget: Budget,
) -> list[T]:
    """`fn` on `items` cut to the budget's batch; on a GPU out-of-memory error, the batch is
    halved (said once in `budget.notes`) and the same items tried again. Raises `GpuExhausted`
    when even one item does not fit."""
    from quantilo.synthetic.deep.device import out_of_memory

    out: list[T] = []
    start = 0
    while start < len(items):
        chunk = items[start : start + budget.batch]
        try:
            out.append(fn(chunk))
        except RuntimeError as error:
            if not budget.on_gpu or not out_of_memory(error):
                raise
            _free_gpu()
            if budget.batch == 1:
                raise GpuExhausted("Out of GPU memory with a batch of 1.") from error
            budget.batch = max(1, budget.batch // 2)
            budget.notes.append(
                f"GPU ran out of memory: batch halved to {budget.batch} and tried again (R-VDP-04)."
            )
            continue
        start += len(chunk)
    return out


def _free_gpu() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def estimate(seconds: float, done: int, total: int, label: str) -> Estimate:
    """R-VDP-05: the whole run, extrapolated from the first `done` images."""
    per_image = seconds / max(done, 1)
    return Estimate(per_image * total, total, label)
