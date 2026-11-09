"""Where a deep model trains: the GPU when it is ready and allowed, otherwise the CPU; back to
the CPU if the card runs out of memory."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

Choice = Literal["auto", "cpu", "gpu"]


@dataclass
class Device:
    device: torch.device
    label: str  # "NVIDIA GeForce RTX 4050 Laptop GPU" or "CPU"
    note: str | None = None


def gpu_name() -> str | None:
    """The ready card PyTorch sees (CUDA, or Apple's MPS), None otherwise."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "Apple GPU (MPS)"
    return None


def pick(choice: Choice, allow_gpu: bool) -> Device:
    """`allow_gpu` is the app setting; it only decides "auto"."""
    cpu = Device(torch.device("cpu"), "CPU")
    if choice == "cpu" or (choice == "auto" and not allow_gpu):
        return cpu
    name = gpu_name()
    if name is None:
        if choice == "gpu":
            cpu.note = "No GPU ready: trained on the CPU."
        return cpu
    return Device(torch.device("cuda" if torch.cuda.is_available() else "mps"), name)


def out_of_memory(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def with_fallback[T](fit: Callable[[torch.device], T], device: Device) -> tuple[T, Device]:
    """Run `fit` on the device; on a GPU memory error, run it again on the CPU and say so."""
    try:
        return fit(device.device), device
    except RuntimeError as error:
        if device.device.type == "cpu" or not out_of_memory(error):
            raise
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cpu = Device(torch.device("cpu"), "CPU", "GPU ran out of memory: finished on the CPU.")
    return fit(cpu.device), cpu
