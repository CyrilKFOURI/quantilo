"""Which graphics card the deep models (023b) could use. Without PyTorch, `nvidia-smi` tells
whether an NVIDIA card is there; with PyTorch, `torch.cuda` tells whether it is usable."""

import importlib.util
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

State = Literal["ready", "no_support", "not_found"]


@dataclass
class GpuStatus:
    state: State
    name: str | None = None
    memory_mb: int | None = None
    driver: str | None = None
    torch: bool = False  # PyTorch is installed
    message: str = ""


def _nvidia_smi() -> tuple[str, int | None, str] | None:
    command = shutil.which("nvidia-smi")
    if command is None:
        return None
    try:
        output = subprocess.run(
            [
                command,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = next((line for line in output.splitlines() if line.strip()), None)
    if first is None:
        return None
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        return None
    memory = int(float(parts[1])) if parts[1].replace(".", "", 1).isdigit() else None
    return parts[0], memory, parts[2]


def _torch_cuda() -> bool | None:
    """True / False when PyTorch is installed, None otherwise (never imported needlessly)."""
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        import torch  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def detect() -> GpuStatus:
    card = _nvidia_smi()
    cuda = _torch_cuda()
    if card is None:
        return GpuStatus("not_found", torch=cuda is not None, message="No GPU found")
    name, memory, driver = card
    size = f" · {memory / 1024:.0f} GB" if memory else ""
    if cuda is False:
        return GpuStatus(
            "no_support",
            name,
            memory,
            driver,
            True,
            f"{name}{size} found, but this build has no GPU support (CPU PyTorch)",
        )
    return GpuStatus("ready", name, memory, driver, cuda is not None, f"{name}{size} · ready")
