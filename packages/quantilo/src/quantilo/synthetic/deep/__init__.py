"""Deep generators (feature 023b): CTGAN and TVAE, written from the paper (Xu et al., 2019).

PyTorch is an optional extra (`uv sync --extra deep`): nothing here imports it at module level,
so the app starts without it.
"""

import importlib.util
from dataclasses import dataclass

BATCH = 500  # rows per training step (the paper's)
GPU_BATCH = 2_000  # CTGAN on a GPU: twice the rows per second of batches of 500
GPU_ROWS = 6_000_000  # rows seen in training at most (epochs * rows): ~3 minutes on a laptop GPU
CPU_ROWS = 1_500_000  # ~2 minutes on a laptop CPU
MIN_EPOCHS, MAX_EPOCHS = 30, 300
TIMEGAN_GPU_STEPS = 1_000  # TimeGAN steps per phase: about 2 minutes on a laptop GPU
TIMEGAN_CPU_STEPS = 150  # recurrent networks are about 5x slower on a CPU: short and rough


@dataclass
class DeepStatus:
    """Whether the deep methods can run here, and on what."""

    installed: bool
    gpu: str | None = None  # name of a ready card, None = CPU only


def installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def batch_size(gpu: bool) -> int:
    return GPU_BATCH if gpu else BATCH


def auto_epochs(rows: int, gpu: bool) -> int:
    """300 epochs (as the paper) unless training would see more rows than the budget of the
    device; never fewer than 30."""
    budget = GPU_ROWS if gpu else CPU_ROWS
    return int(min(MAX_EPOCHS, max(MIN_EPOCHS, budget // max(rows, 1))))
