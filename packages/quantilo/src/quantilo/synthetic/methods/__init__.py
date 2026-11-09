"""Generation methods. Each one learns from coded training columns and returns coded synthetic
columns of the same kind (numbers, seconds, level indexes; NaN = empty)."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from quantilo.synthetic.encode import Coded, Learned
from quantilo.synthetic.plan import Method


class Cancelled(Exception):
    """The user stopped the run."""


@dataclass
class SequenceSettings:
    """TimeGAN (023c): how rows form histories, and what to generate."""

    time: str | None
    entity: str | None = None
    length: int = 24
    sequences: int = 1
    steps: int | None = None  # per training phase; None = automatic
    step_seconds: float | None = None  # time between two steps; None = the real median


@dataclass
class Options:
    rows: int
    seed: int = 42
    smooth: bool = True  # CART: Gaussian noise on numbers, scaled to the leaf's spread
    order: list[str] | None = None  # CART: column order (None = automatic)
    drop_copies: bool = True  # replace synthetic rows identical to a training row
    epochs: int | None = None  # deep methods: None = automatic
    device: Literal["auto", "cpu", "gpu"] = "auto"  # deep methods
    allow_gpu: bool = True  # the app setting "Use the GPU when available" (decides "auto")
    progress: Callable[[float, str], None] | None = None  # deep methods: per epoch
    sequence: SequenceSettings | None = None  # TimeGAN


@dataclass
class Edge:
    parent: str
    child: str
    strength: float  # mutual information (Bayesian network) or association (CART order)


@dataclass
class Fitted:
    """What the method found, shown with the run."""

    method: Method
    order: list[str] = field(default_factory=list)  # CART: columns in generation order
    edges: list[Edge] = field(default_factory=list)  # Bayesian network: the Chow-Liu tree
    notes: list[str] = field(default_factory=list)
    dropped: int = 0  # synthetic rows identical to a training row, replaced
    device: str | None = None  # deep methods: the card or "CPU"
    epochs: int | None = None  # deep methods
    steps: int | None = None  # TimeGAN: training steps per phase


StopCheck = Callable[[], bool]


def draw_in_groups(
    groups: np.ndarray, values: np.ndarray, wanted: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """For each wanted group, a random training value from the same group (a real value of the
    same leaf or interval). Unknown groups draw from all values."""
    order = np.argsort(groups, kind="stable")
    sorted_groups, sorted_values = groups[order], values[order]
    unique, starts, counts = np.unique(sorted_groups, return_index=True, return_counts=True)
    position = np.searchsorted(unique, wanted)
    position = np.clip(position, 0, len(unique) - 1)
    known = unique[position] == wanted
    start = np.where(known, starts[position], 0)
    count = np.where(known, counts[position], len(sorted_values))
    picked = start + np.floor(rng.random(len(wanted)) * count).astype(np.int64)
    return sorted_values[np.minimum(picked, len(sorted_values) - 1)]


def marginal(values: np.ndarray, rows: int, rng: np.random.Generator) -> np.ndarray:
    """Real values drawn at random, empty ones included."""
    return values[rng.integers(0, len(values), size=rows)]


def generate(
    method: Method,
    coded: Coded,
    learned: list[Learned],
    options: Options,
    should_stop: StopCheck = lambda: False,
) -> tuple[Coded, Fitted]:
    from quantilo.synthetic.methods import bayes, cart, copula, independent

    if method in ("ctgan", "tvae"):
        from quantilo.synthetic.methods import deep  # needs PyTorch: imported on use

        return deep.generate(method, coded, learned, options, should_stop)
    runners = {
        "cart": cart.generate,
        "copula": copula.generate,
        "bayes": bayes.generate,
        "independent": independent.generate,
    }
    return runners[method](coded, learned, options, should_stop)
