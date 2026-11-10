"""What the Experiments module lets the AI agent do (specs 034, 049): analyze an A/B test of the
dataset, read its bandit view, and plan one (sample size), through this module's own engine."""

from typing import Any, ClassVar, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from quantilo.experiments.analyze import (
    ExperimentDesign,
    ExperimentMetric,
    ExperimentResult,
)
from quantilo.experiments.plan import SampleSizeRequest
from quantilo.experiments.service import ExperimentEngine

from app.core.errors import AppError
from app.core.tools import AppLink, Tool, ToolContext, ToolError, ToolOutput, compact


class AnalyzeInput(BaseModel):
    variant: str = Field(description="The column naming each unit's variant (2 to 5 values).")
    metrics: list[str | ExperimentMetric] = Field(
        min_length=1,
        max_length=10,
        description=(
            "Metric columns (0/1 or yes/no = a rate, numbers = a mean, whole numbers = a count). "
            "A ratio of sums: {column: 'revenue', denominator: 'orders'}; CUPED: {column: "
            "'spend', pre: 'spend_before'}. The first is the primary metric."
        ),
    )
    control: str | None = Field(None, description="The control variant; guessed if unset.")
    unit: str | None = Field(
        None, description="When a user has several rows, the column naming the user."
    )
    segment: str | None = Field(None, description="A column to split the effect by.")
    time: str | None = Field(None, description="A date column for the cumulative difference.")
    expected: dict[str, float] | None = Field(
        None, description="Planned share of each variant, e.g. {A: 50, B: 50} (equal if unset)."
    )

    def design(self) -> ExperimentDesign:
        return ExperimentDesign(
            variant=self.variant,
            metrics=[
                m if isinstance(m, ExperimentMetric) else ExperimentMetric(column=m)
                for m in self.metrics
            ],
            control=self.control,
            unit=self.unit,
            segment=self.segment,
            time=self.time,
            expected=self.expected,
        )


def _content(result: ExperimentResult) -> dict[str, Any]:
    content: dict[str, Any] = {
        "control": result.design.control,
        "rows": result.rows,
        "units": result.units,
        "variants": [{"label": v.label, "units": v.units} for v in result.variants],
        "sample_ratio_check": {"p": result.srm.p_value, "alarm": result.srm.alarm},
        "correction": result.correction,
        "warnings": result.warnings,
        "metrics": [
            {
                "metric": m.label,
                "kind": m.kind,
                "primary": m.primary,
                "aggregation": m.aggregation,
                "cuped": m.cuped.sentence if m.cuped else None,
                "comparisons": [
                    {
                        "variant": c.variant,
                        "test": c.test,
                        "sentence": c.sentence,
                        "diff": c.diff,
                        "ci": [c.diff_low, c.diff_high],
                        "lift": c.lift,
                        "p": c.p_value,
                        "p_adjusted": c.p_adjusted,
                        "power": c.power,
                        "detectable_lift": c.mde_relative,
                        "verdict": c.verdict,
                        "checks": [check.note for check in c.checks],
                    }
                    for c in m.comparisons
                ],
            }
            for m in result.metrics
        ],
    }
    if result.segments:
        content["segments"] = {
            "column": result.segments.column,
            "warning": result.segments.warning,
            "levels": [
                {
                    "level": s.level,
                    "units": s.units,
                    "sentences": [c.sentence for c in s.comparisons],
                }
                for s in result.segments.levels
            ],
        }
    if result.cumulative:
        content["cumulative_warning"] = result.cumulative.warning
    return compact(content, max_items=12)


class AnalyzeExperiment(Tool[AnalyzeInput]):
    name: ClassVar[str] = "analyze_experiment"
    description: ClassVar[str] = (
        "Analyze an A/B test already run on the filtered rows: sample ratio check (SRM), each "
        "metric against the control with the test that suits it (z-test of proportions, Welch, "
        "Poisson, delta method), difference and lift with their 95 % interval, Holm-adjusted "
        "p-values, power reached and the smallest detectable lift, CUPED, subgroups. Quote its "
        "sentences: a non-significant, underpowered result means 'not enough data', never 'no "
        "effect'."
    )
    Input: ClassVar[type[BaseModel]] = AnalyzeInput

    def __init__(self, engine: ExperimentEngine) -> None:
        self.engine = engine

    def label(self, args: AnalyzeInput) -> str:
        return f"Analyzing the experiment on {args.variant}"

    def run(self, ctx: ToolContext, args: AnalyzeInput) -> ToolOutput:
        try:
            result = self.engine.analyze(ctx.dataset_id, args.design(), ctx.conditions)
        except AppError as error:
            raise ToolError(f"This experiment can't be analyzed: {error.message}") from error
        primary = next(m for m in result.metrics if m.primary)
        first = primary.comparisons[0] if primary.comparisons else None
        summary = (
            "Sample ratio mismatch: results may be biased"
            if result.srm.alarm
            else (first.sentence.split(":")[0] + f": {first.verdict}" if first else "Analyzed")
        )
        params = {"variant": args.variant, "metric": primary.column}
        if args.unit:
            params["unit"] = args.unit
        return ToolOutput(
            summary, _content(result), AppLink("experiments", params, "Open in Experiments")
        )


def _bandit_content(result: ExperimentResult) -> dict[str, Any]:
    bandit = result.bandit
    if bandit is None:
        return {"bandit_view": None, "why": result.bandit_reason}
    content: dict[str, Any] = {
        "metric": bandit.metric,
        "prior": bandit.prior,
        "variants": [
            {
                "variant": arm.variant,
                "p_best": arm.p_best,
                "expected_loss": arm.expected_loss,
                "sentence": arm.sentence,
            }
            for arm in bandit.arms
        ],
        "guardrail": bandit.guardrail,
    }
    if bandit.replay:
        content["replay"] = {
            "sentence": bandit.replay.sentence,
            "gained": bandit.replay.gained,
            "gained_relative": bandit.replay.gained_relative,
            "regret_fixed": bandit.replay.regret_fixed[-1],
            "regret_bandit": bandit.replay.regret_bandit[-1],
            "note": bandit.replay.note,
        }
    else:
        content["replay"] = bandit.replay_reason
    return compact(content, max_items=12)


class BanditView(Tool[AnalyzeInput]):
    name: ClassVar[str] = "bandit_view"
    description: ClassVar[str] = (
        "The bandit view of an A/B test already run (primary metric: a rate, a mean or a count): "
        "P(each variant is best) and the expected loss of shipping it (Beta or Normal "
        "posteriors, seeded Monte Carlo) and, with a date column (time=), what a Thompson-"
        "sampling bandit would have done: units gained or lost against the fixed split and the "
        "cumulative regret. Use it after analyze_experiment, never instead of it: always quote "
        "its guardrail (after an adaptive allocation, classical intervals are biased)."
    )
    Input: ClassVar[type[BaseModel]] = AnalyzeInput

    def __init__(self, engine: ExperimentEngine) -> None:
        self.engine = engine

    def label(self, args: AnalyzeInput) -> str:
        return f"Reading the bandit view of {args.variant}"

    def run(self, ctx: ToolContext, args: AnalyzeInput) -> ToolOutput:
        try:
            result = self.engine.analyze(ctx.dataset_id, args.design(), ctx.conditions)
        except AppError as error:
            raise ToolError(f"This experiment can't be analyzed: {error.message}") from error
        summary = result.bandit.sentence if result.bandit else "No bandit view"
        primary = next(m for m in result.metrics if m.primary)
        params = {"variant": args.variant, "metric": primary.column}
        if args.unit:
            params["unit"] = args.unit
        return ToolOutput(
            summary, _bandit_content(result), AppLink("experiments", params, "Open in Experiments")
        )


class SampleSizeInput(BaseModel):
    kind: Literal["rate", "mean"] = Field("rate", description="A rate (0/1) or a mean metric.")
    baseline: float = Field(description="Current rate (0.10 = 10 %) or mean.")
    sd: float | None = Field(None, gt=0, description="Standard deviation (mean metrics only).")
    effect: float = Field(description="Smallest effect worth detecting: 0.05 = +5 % if relative.")
    relative: bool = Field(True, description="False: the effect is an absolute difference.")
    alpha: float = Field(0.05, gt=0, lt=0.5)
    power: float = Field(0.8, gt=0.05, lt=1)
    variants: int = Field(2, ge=2, le=5, description="Groups, the control included.")
    daily: int | None = Field(None, gt=0, description="Units entering the test per day.")


class SampleSizeTool(Tool[SampleSizeInput]):
    name: ClassVar[str] = "sample_size"
    description: ClassVar[str] = (
        "Plan an A/B test: units per group (and total, and days at a daily traffic) to detect "
        "an effect on a rate or a mean, with the given α and power (statsmodels' NormalIndPower). "
        "Use analyze_experiment's or the dataset's current value as the baseline."
    )
    Input: ClassVar[type[BaseModel]] = SampleSizeInput

    def __init__(self, engine: ExperimentEngine) -> None:
        self.engine = engine

    def label(self, args: SampleSizeInput) -> str:
        return "Computing the sample size"

    def run(self, ctx: ToolContext, args: SampleSizeInput) -> ToolOutput:
        try:
            plan = self.engine.sample_size(SampleSizeRequest(**args.model_dump()))
        except AppError as error:
            raise ToolError(f"No sample size: {error.message}") from error
        content = compact(plan, max_items=6)
        return ToolOutput(
            f"{plan.control:,} units per group, {plan.total:,} in total",
            content,
            AppLink("experiments", {"tab": "plan"}, "Open the Plan section"),
        )


def agent_tools(app: FastAPI) -> list[Tool[Any]]:
    engine: ExperimentEngine = app.state.experiments
    return [AnalyzeExperiment(engine), BanditView(engine), SampleSizeTool(engine)]
