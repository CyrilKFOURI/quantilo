"""Plan an experiment before it runs: units per group to detect an effect, the duration at a given
daily traffic, and the curve "units per group ↔ smallest detectable lift".

The sizes are statsmodels' `NormalIndPower` on a standardised effect: Cohen's h for a rate
(`proportion_effectsize`), the difference ÷ standard deviation for a mean. With more than two
variants, α is shared between the comparisons to the control (Bonferroni, the first step of the
Holm correction the analysis applies).
"""

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from quantilo.errors import ExplorerError
from quantilo.experiments.stats import _quiet

CURVE_POINTS = 41
CURVE_SPAN = 10.0  # the curve runs from size ÷ 10 to size × 10


class PlanError(ExplorerError):
    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message, code="experiment_plan", status_code=422)
        self.details = {"field": field}


class SampleSizeRequest(BaseModel):
    kind: Literal["rate", "mean"] = "rate"
    baseline: float = Field(description="The current rate (0.10 = 10 %) or mean.")
    sd: float | None = Field(default=None, gt=0, description="Standard deviation of a mean metric.")
    effect: float = Field(description="Smallest effect worth detecting.")
    relative: bool = Field(
        default=True, description="The effect is a lift (0.05 = +5 %), else absolute."
    )
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    power: float = Field(default=0.8, gt=0.05, lt=1)
    variants: int = Field(default=2, ge=2, le=5, description="Groups, the control included.")
    ratio: float = Field(
        default=1.0, gt=0, le=20, description="Units of each variant per control unit."
    )
    daily: int | None = Field(default=None, gt=0, description="Units entering the test per day.")

    @property
    def target(self) -> float:
        return self.baseline * (1 + self.effect) if self.relative else self.baseline + self.effect


@dataclass
class CurvePoint:
    per_group: int  # control units
    lift: float | None  # smallest detectable relative effect
    absolute: float | None


@dataclass
class SampleSize:
    kind: str
    baseline: float
    target: float  # the rate or mean to detect
    effect_absolute: float
    effect_relative: float | None
    effect_size: float  # Cohen's h or d
    alpha: float
    alpha_per_comparison: float
    power: float
    variants: int
    ratio: float
    control: int  # units in the control group
    per_variant: int  # units in each other variant
    total: int
    days: int | None
    sentence: str
    curve: list[CurvePoint] = field(default_factory=list)


def _effect_size(request: SampleSizeRequest, target: float) -> float:
    if request.kind == "rate":
        return float(proportion_effectsize(target, request.baseline))
    assert request.sd is not None
    return (target - request.baseline) / request.sd


def _detectable(request: SampleSizeRequest, size: float, alpha: float, n: int) -> float | None:
    """The smallest target detectable with n control units: invert Cohen's h or d."""
    effect = abs(size)
    if request.kind == "mean":
        assert request.sd is not None
        return request.baseline + math.copysign(effect * request.sd, request.effect)
    start = 2 * math.asin(math.sqrt(request.baseline))
    angle = start + math.copysign(effect, request.effect)
    if not 0 <= angle <= math.pi:
        return None
    return math.sin(angle / 2) ** 2


def _pct(value: float) -> str:
    return f"{value * 100:.3g} %"


def _check(request: SampleSizeRequest) -> None:
    if request.effect == 0:
        raise PlanError("The effect to detect cannot be zero.", "effect")
    if request.kind == "rate" and not 0 < request.baseline < 1:
        raise PlanError("A rate is between 0 and 1 (0.10 for 10 %).", "baseline")
    if request.kind == "mean" and request.sd is None:
        raise PlanError("A mean metric needs its standard deviation.", "sd")
    if request.kind == "rate" and not 0 < request.target < 1:
        raise PlanError("The rate to detect falls outside 0–1: choose a smaller effect.", "effect")


def sample_size(request: SampleSizeRequest) -> SampleSize:
    _check(request)
    target = request.target
    comparisons = request.variants - 1
    alpha = request.alpha / comparisons
    size = _effect_size(request, target)
    analysis = NormalIndPower()
    n = _quiet(
        lambda: analysis.solve_power(
            effect_size=abs(size), alpha=alpha, power=request.power, ratio=request.ratio
        )
    )
    control = math.ceil(float(n))
    per_variant = math.ceil(control * request.ratio)
    total = control + comparisons * per_variant
    days = math.ceil(total / request.daily) if request.daily else None

    curve = []
    for units in np.geomspace(control / CURVE_SPAN, control * CURVE_SPAN, CURVE_POINTS):
        n_units = max(2, round(float(units)))
        found = _quiet(
            lambda n_units=n_units: analysis.solve_power(
                effect_size=None,
                nobs1=n_units,
                alpha=alpha,
                power=request.power,
                ratio=request.ratio,
            )
        )
        detectable = _detectable(request, float(found), alpha, n_units)
        if detectable is None:
            curve.append(CurvePoint(n_units, None, None))
            continue
        gap = detectable - request.baseline
        relative = gap / request.baseline if request.baseline else None
        curve.append(CurvePoint(n_units, relative, gap))

    effect_absolute = target - request.baseline
    effect_relative = effect_absolute / request.baseline if request.baseline else None
    shown = (
        f"{_pct(request.baseline)} → {_pct(target)}"
        if request.kind == "rate"
        else f"{request.baseline:.4g} → {target:.4g}"
    )
    groups = (
        f"{control:,} units per group"
        if request.ratio == 1
        else f"{control:,} in the control and {per_variant:,} in each variant"
    )
    sentence = (
        f"To detect {shown} with {request.power * 100:.0f} % power at α = {request.alpha:g}, "
        "you need "
        f"{groups}, {total:,} in total"
        + (f": about {days:,} days at {request.daily:,} units a day." if days else ".")
    )
    if comparisons > 1:
        sentence += f" α is shared by {comparisons} comparisons to the control ({alpha:.3g} each)."
    return SampleSize(
        request.kind,
        request.baseline,
        target,
        effect_absolute,
        effect_relative,
        size,
        request.alpha,
        alpha,
        request.power,
        request.variants,
        request.ratio,
        control,
        per_variant,
        total,
        days,
        sentence,
        curve,
    )
