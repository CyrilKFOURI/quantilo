"""One experiment of a dataset, for the library: the app's Experiments page as methods.

exp = ds.experiment(variant="group", metrics=["converted"], unit="user_id")
exp.analyze()                         # SRM, a comparison per metric and variant, Holm, power
exp.bandit()                          # P(best), expected loss, what a bandit would have done
exp.policy()                          # who should get which variant: a rule, its value, Qini
dx.sample_size(baseline=0.10, effect=0.05)   # units per group before running one
dx.bandit_simulation(baseline=0.10, effect=0.05, daily=1_000)   # fixed split vs bandit
"""

from typing import TYPE_CHECKING, Any

from quantilo.experiments.analyze import (
    ExperimentDesign,
    ExperimentMetric,
    ExperimentResult,
)
from quantilo.experiments.bandit import BanditSimulation, ExperimentBandit, simulate
from quantilo.experiments.plan import SampleSize, SampleSizeRequest
from quantilo.experiments.plan import sample_size as plan
from quantilo.experiments.service import ExperimentEngine
from quantilo.filters.model import Condition

if TYPE_CHECKING:  # scikit-learn loads on first use
    from quantilo.experiments.policy import PolicyLearning

MetricLike = str | dict[str, Any] | ExperimentMetric


def metric_of(value: MetricLike) -> ExperimentMetric:
    """ "converted", {"column": "revenue", "denominator": "orders"} or an ExperimentMetric."""
    if isinstance(value, ExperimentMetric):
        return value
    if isinstance(value, str):
        return ExperimentMetric(column=value)
    return ExperimentMetric.model_validate(value)


class Experiment:
    """An experiment described on a dataset; `analyze()` runs it on the (filtered) rows."""

    def __init__(
        self,
        engine: ExperimentEngine,
        dataset_id: str,
        design: ExperimentDesign,
        conditions: list[Condition],
    ) -> None:
        self._engine = engine
        self._id = dataset_id
        self.design = design
        self._conditions = conditions

    def analyze(self) -> ExperimentResult:
        """Sample ratio check, then each metric against the control: values, difference and lift
        with their interval, p-value (Holm-adjusted), power reached, smallest detectable effect,
        and a sentence; subgroups and the cumulative difference when asked."""
        return self._engine.analyze(self._id, self.design, self._conditions)

    def bandit(self) -> ExperimentBandit | None:
        """The primary metric's bandit view: P(each variant is best) and the expected loss of
        shipping it (Beta or Normal posteriors, seeded Monte Carlo), and with `time=` the replay
        of what a Thompson-sampling bandit would have done. None for a ratio (see
        `analyze().bandit_reason`). It complements the test, it does not replace it."""
        return self.analyze().bandit

    def policy(
        self, covariates: list[str] | None = None, *, qini: bool = False
    ) -> "PolicyLearning":
        """Who should get which variant (055): a depth 1-3 tree grown on doubly robust scores
        of the primary metric, its cross-validated value against everyone on each variant (with
        bootstrap intervals) and the Qini curve. `covariates`: the columns describing the units
        before the test (every usable one when unset); `qini=True` adds the Qini curves. A
        hypothesis to confirm with a new test."""
        return self._engine.policy(self._id, self.design, covariates, self._conditions, qini)

    def __repr__(self) -> str:
        metrics = ", ".join(m.column for m in self.design.metrics)
        return f"<Experiment {self.design.variant!r}: {metrics}>"


def sample_size(
    baseline: float,
    effect: float,
    *,
    kind: str = "rate",
    relative: bool = True,
    sd: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    variants: int = 2,
    ratio: float = 1.0,
    daily: int | None = None,
) -> SampleSize:
    """Units per group to detect `effect` (a lift of 0.05 = +5 % when `relative`, else an absolute
    difference) on a rate (`baseline` 0.10 = 10 %) or a mean (`kind="mean"`, with its `sd`);
    the duration when `daily` units enter the test each day."""
    request = SampleSizeRequest.model_validate(
        {
            "kind": kind,
            "baseline": baseline,
            "sd": sd,
            "effect": effect,
            "relative": relative,
            "alpha": alpha,
            "power": power,
            "variants": variants,
            "ratio": ratio,
            "daily": daily,
        }
    )
    return plan(request)


def bandit_simulation(
    baseline: float,
    effect: float,
    *,
    kind: str = "rate",
    relative: bool = True,
    sd: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    variants: int = 2,
    ratio: float = 1.0,
    daily: int | None = None,
) -> BanditSimulation:
    """The planned test (same arguments as `sample_size`) run 200 times with a fixed split and
    with a Thompson-sampling bandit: units gained, share of the traffic sent to the best variant,
    how often each ends sure of it."""
    request = SampleSizeRequest.model_validate(
        {
            "kind": kind,
            "baseline": baseline,
            "sd": sd,
            "effect": effect,
            "relative": relative,
            "alpha": alpha,
            "power": power,
            "variants": variants,
            "ratio": ratio,
            "daily": daily,
        }
    )
    return simulate(request)
