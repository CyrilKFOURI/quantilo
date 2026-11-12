"""How the deep generators' networks are trained (066), in Advanced > More of the deep methods.

Without settings (`None`), every network is trained as its paper says (Adam, a constant rate):
the default stays the published method. When the user asks for it:

- optimizer: AdamW ★ (Adam with the weight decay applied apart from the gradient, Loshchilov and
  Hutter, 2019), Adam, SGD with Nesterov momentum, or RAdam (Adam with a rectified warm-up of its
  variance, Liu et al., 2020); all are in torch.
- schedule of the rate: constant ★, cosine (decays to zero), or one-cycle (rises then decays,
  Smith, 2018), after a linear warm-up over a share of the steps.
- clipping: the gradient's norm held to a bound, which calms a GAN's critic.
- EMA: an exponential moving average of the weights, used to generate (smoother samples).

Rules, each with its reason:
- R-SYN-NET-01: SGD without a rate given starts at 10 times the paper's rate: plain momentum moves
  far slower than Adam at the same rate.
- R-SYN-NET-02: AdamW without a weight decay given uses 0.01, torch's default; the papers' decay
  (1e-5, 1e-6) was an L2 penalty, which AdamW applies differently.
- R-SYN-NET-03: one-cycle without a warm-up rises over the first 30 % of the steps, as in Smith's
  paper; with a warm-up, over that share.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import torch
    from torch import nn

Optimizer = Literal["adamw", "adam", "sgd_nesterov", "radam"]
Schedule = Literal["constant", "cosine", "one_cycle"]
SGD_FACTOR = 10.0  # R-SYN-NET-01
ADAMW_DECAY = 0.01  # R-SYN-NET-02
ONE_CYCLE_RISE = 0.3  # R-SYN-NET-03
ONE_CYCLE_START = 0.04  # one-cycle starts at the rate / 25 (Smith; torch's div_factor)
MOMENTUM = 0.9


class Training(BaseModel):
    """The very advanced training settings of the deep generators (CTGAN, TVAE, TimeGAN)."""

    optimizer: Optimizer = "adamw"
    learning_rate: float | None = Field(default=None, gt=0, le=1.0)  # None: the paper's
    weight_decay: float | None = Field(default=None, ge=0, le=1.0)  # None: see R-SYN-NET-02
    schedule: Schedule = "constant"
    warmup: float = Field(default=0.0, ge=0.0, le=0.5)  # share of the steps
    clip_norm: float | None = Field(default=None, gt=0, le=100.0)  # None: no clipping
    ema: float | None = Field(default=None, ge=0.9, le=0.9999)  # decay; None: no average


@dataclass
class Paper:
    """The optimizer a paper uses for one network (what `None` settings keep)."""

    learning_rate: float
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0


def factor(step: int, total: int, schedule: Schedule, warmup: float) -> float:
    """The rate's multiplier at `step` (0-based) of `total` steps."""
    total = max(total, 1)
    if schedule == "one_cycle":
        rise = max(1, round(total * (warmup or ONE_CYCLE_RISE)))
        if step < rise:
            return ONE_CYCLE_START + (1 - ONE_CYCLE_START) * step / rise
        progress = (step - rise) / max(1, total - rise)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    ramp = round(total * warmup)
    if step < ramp:
        return (step + 1) / ramp
    if schedule == "cosine":
        progress = (step - ramp) / max(1, total - ramp)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return 1.0


def make_optimizer(
    parameters: list[torch.nn.Parameter], training: Training | None, paper: Paper
) -> torch.optim.Optimizer:
    import torch

    if training is None:
        return torch.optim.Adam(
            parameters, lr=paper.learning_rate, betas=paper.betas, weight_decay=paper.weight_decay
        )
    rate = training.learning_rate
    if training.optimizer == "sgd_nesterov":
        return torch.optim.SGD(
            parameters,
            lr=rate if rate is not None else paper.learning_rate * SGD_FACTOR,
            momentum=MOMENTUM,
            nesterov=True,
            weight_decay=training.weight_decay or paper.weight_decay,
        )
    rate = rate if rate is not None else paper.learning_rate
    if training.optimizer == "adamw":
        decay = training.weight_decay if training.weight_decay is not None else ADAMW_DECAY
        return torch.optim.AdamW(parameters, lr=rate, betas=paper.betas, weight_decay=decay)
    decay = training.weight_decay if training.weight_decay is not None else paper.weight_decay
    kind = torch.optim.RAdam if training.optimizer == "radam" else torch.optim.Adam
    return kind(parameters, lr=rate, betas=paper.betas, weight_decay=decay)


class Stepper:
    """One network's optimizer, rate schedule and clipping, stepped as the paper's optimizer."""

    def __init__(
        self,
        parameters: list[torch.nn.Parameter],
        training: Training | None,
        paper: Paper,
        total_steps: int,
    ) -> None:
        import torch

        self.parameters = list(parameters)
        self.training = training
        self.optimizer = make_optimizer(self.parameters, training, paper)
        self.schedule = None
        if training is not None and (training.schedule != "constant" or training.warmup > 0):
            chosen, share = training.schedule, training.warmup
            self.schedule = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda step: factor(step, total_steps, chosen, share)
            )

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        import torch

        if self.training is not None and self.training.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters, self.training.clip_norm)
        self.optimizer.step()
        if self.schedule is not None:
            self.schedule.step()


class Ema:
    """An exponential moving average of a network's weights, swapped in to generate."""

    def __init__(self, network: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(network).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    def update(self, network: nn.Module) -> None:
        import torch

        with torch.no_grad():
            for average, current in zip(
                self.shadow.state_dict().values(), network.state_dict().values(), strict=True
            ):
                if average.dtype.is_floating_point:
                    average.mul_(self.decay).add_(current, alpha=1 - self.decay)
                else:
                    average.copy_(current)

    def copy_to(self, network: nn.Module) -> None:
        network.load_state_dict(self.shadow.state_dict())


def ema_for(network: nn.Module, training: Training | None) -> Ema | None:
    return Ema(network, training.ema) if training is not None and training.ema else None


def describe(training: Training | None) -> str | None:
    """What was used, said with the run."""
    if training is None:
        return None
    names = {"adamw": "AdamW", "adam": "Adam", "sgd_nesterov": "SGD (Nesterov)", "radam": "RAdam"}
    parts = [names[training.optimizer]]
    if training.learning_rate is not None:
        parts.append(f"rate {training.learning_rate:g}")
    if training.schedule != "constant":
        parts.append(training.schedule.replace("_", "-"))
    if training.warmup:
        parts.append(f"warm-up {training.warmup:.0%}")
    if training.clip_norm is not None:
        parts.append(f"gradient clipped at {training.clip_norm:g}")
    if training.ema:
        parts.append(f"EMA {training.ema:g}")
    return "Networks trained with " + ", ".join(parts) + "."
