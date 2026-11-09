"""The Auto (038) as a tool of the AI agent: the best model within a time budget the user gives."""

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from app.core.tools import Tool, ToolContext, ToolError, ToolOutput, wait_for
from app.modules.ml.agent_tools import VALIDATION_HELP, _failed, _link, _r, _validation
from app.modules.ml.auto import AutoRequest, AutoService
from app.modules.ml.service import MlError, MlService


class AutoInput(BaseModel):
    target: str = Field(description="The column to predict.")
    minutes: float = Field(
        gt=0,
        le=60,
        description="The time budget the USER chose (ask them first: 2, 5, 10 or 30 minutes).",
    )
    stacking: bool = Field(False, description="Also try a linear stack of the best models.")
    validation: dict[str, Any] | None = Field(None, description=VALIDATION_HELP)


class AutoMl(Tool[AutoInput]):
    name: ClassVar[str] = "auto_ml"
    kind = "compute"
    description: ClassVar[str] = (
        "Find the best model for a target within a time budget, on its own: baselines (constant "
        "and linear), one model per family, successive elimination, tuning of the 3 best "
        "families, then an ensemble kept only if it beats the best single model by more than one "
        "standard deviation. Saves the kept model as a run and returns the leaderboard. Before "
        "calling, ASK THE USER how many minutes it may take (2, 5, 10 or 30): never choose the "
        "budget yourself."
    )
    Input: ClassVar[type[BaseModel]] = AutoInput

    def __init__(self, service: MlService, auto: AutoService) -> None:
        self.service = service
        self.auto = auto

    def label(self, args: AutoInput) -> str:
        return f"Finding the best model for {args.target} in {args.minutes:g} min"

    def run(self, ctx: ToolContext, args: AutoInput) -> ToolOutput:
        try:
            job = self.auto.start(
                ctx.dataset_id,
                args.target,
                AutoRequest(minutes=args.minutes, stacking=args.stacking),
                ctx.conditions,
                validation=_validation(args.validation),
            )
        except (MlError, ValueError) as error:
            raise _failed(error) from error
        budget = args.minutes * 60
        job = wait_for(
            lambda: self.auto.job(job.id),
            lambda j: j.status != "running",
            ctx,
            describe=lambda j: (
                f"Auto: {j.progress.step or 'starting'}" if j.progress else "Auto: loading",
                min((j.progress.seconds if j.progress else 0) / budget, 1.0),
            ),
            cancel=lambda: self.auto.stop(job.id),
            timeout=budget * 1.2 + 300,
        )
        progress = job.progress
        if job.status == "failed" or progress is None:
            raise ToolError(f"The Auto failed: {job.error or 'no model was trained'}")
        leaderboard = [
            {
                "id": e.id,
                "model": e.model,
                "label": e.label,
                "family": e.family,
                "score": _r(e.score),
                "std": _r(e.std, 3),
                "seconds": _r(e.seconds, 3),
                "status": e.status,
                **({"note": e.note} if e.note else {}),
                **({"settings": e.settings} if e.settings else {}),
            }
            for e in progress.ranking()[:20]
        ]
        content: dict[str, Any] = {
            "metric": progress.metric,
            "greater_is_better": progress.greater_is_better,
            "compared_on": progress.compared_on,
            "kept": progress.best,
            "run_id": job.run_id,
            "gain": progress.gain,
            "test_score": _r(progress.test_score),
            "seconds": _r(progress.seconds, 3),
            "stopped": progress.stopped,
            "leaderboard": leaderboard,
            "steps": [
                {"step": s.title, "status": s.status, "note": s.note} for s in progress.steps
            ],
        }
        if progress.ensemble is not None:
            content["ensemble"] = {
                "kept": progress.ensemble.kept,
                "reason": progress.ensemble.reason,
                "members": {m.label: m.weight for m in progress.ensemble.members},
            }
        best = next((e for e in progress.entries if e.id == progress.best), None)
        summary = (
            f"Kept {best.label}: {progress.metric} {_r(best.score, 3)}"
            + (f" ({progress.gain})" if progress.gain else "")
            if best
            else "No model kept"
        )
        return ToolOutput(summary, content, _link(job.target, job.run_id))
