"""What changed since the previous version of the dataset, or against another dataset (031)."""

from typing import Any, ClassVar

from pydantic import BaseModel, Field
from quantilo.changes.compare import Changes, Shift

from app.core.errors import AppError
from app.modules.agent.engine.tools import AppLink, Tool, ToolContext, ToolError, ToolOutput
from app.modules.agent.tools import AppServices
from app.modules.agent.tools.dataset import _r

MAX_SHIFTS = 10


class ChangesInput(BaseModel):
    versus: str | None = Field(
        default=None,
        description="Another dataset to compare with (its name or id). Leave empty for the "
        "previous version of this dataset (after a reimport).",
    )


def _shift(s: Shift) -> dict[str, Any]:
    found: dict[str, Any] = {"column": s.column, "kind": s.kind, "psi": _r(s.psi, 3)}
    if s.drift:
        found["drift"] = s.drift
    if s.wasserstein is not None:
        found["wasserstein_iqr"] = _r(s.wasserstein, 3)
    if s.before and s.after:
        found["mean"] = [_r(s.before.mean), _r(s.after.mean)]
        found["median"] = [_r(s.before.median), _r(s.after.median)]
    if s.appeared:
        found["appeared"] = {a.value: _r(a.after, 3) for a in s.appeared[:5]}
    if s.disappeared:
        found["disappeared"] = {d.value: _r(d.before, 3) for d in s.disappeared[:5]}
    if s.used_by:
        found["used_by"] = s.used_by
    return found


def changes_content(c: Changes) -> dict[str, Any]:
    return {
        "headline": c.headline,
        "level": c.level_label,
        "reasons": c.reasons,
        "rows": {"before": c.before.rows, "after": c.after.rows},
        "schema": [
            {
                k: v
                for k, v in (
                    ("kind", s.kind),
                    ("column", s.column),
                    ("was", s.before),
                    ("now", s.after),
                    ("detail", s.detail),
                    ("used_by", s.used_by),
                )
                if v
            }
            for s in c.schema
        ],
        "shifts": [_shift(s) for s in c.distributions[:MAX_SHIFTS]],
        "quality": [
            {
                "column": q.column,
                "missing_pct": [_r(q.missing_before, 3), _r(q.missing_after, 3)],
                "outliers_pct": [_r(q.outliers_before, 3), _r(q.outliers_after, 3)],
            }
            for q in c.quality
            if q.notable
        ],
        "dates": [
            {
                "column": d.column,
                "range": [d.before_min, d.before_max, d.after_min, d.after_max],
                "future": d.future,
                "gaps": d.gaps,
            }
            for d in c.dates
            if d.notable
        ],
        "psi_reading": c.thresholds,
    }


class WhatChanged(Tool[ChangesInput]):
    name: ClassVar[str] = "what_changed"
    description: ClassVar[str] = (
        "What changed in this dataset since its previous version (after a reimport; for a "
        "cleaned version, against its original), or against another dataset (e.g. last "
        "month's file, or another cleaned version): columns added, removed, probably "
        "renamed or retyped, missing values, dates, distributions ranked by PSI (< 0.1 stable, "
        "0.1-0.25 moderate, > 0.25 strong), new categories, and a level: No notable change, "
        "Check, Breaking. Use it before trusting numbers on a new file."
    )
    Input: ClassVar[type[BaseModel]] = ChangesInput

    def __init__(self, services: AppServices) -> None:
        self.services = services

    def label(self, args: ChangesInput) -> str:
        return (
            f"Comparing with {args.versus}"
            if args.versus
            else "Comparing with the previous version"
        )

    def _other(self, ctx: ToolContext, versus: str | None) -> str | None:
        if not versus:
            return None
        datasets = self.services.columns.datasets.list_all()
        match = next((d for d in datasets if versus in (d.id, d.name)), None)
        if match is None:
            names = ", ".join(sorted(d.name for d in datasets if d.id != ctx.dataset_id))
            raise ToolError(f"No dataset called {versus!r}. Datasets: {names or 'none'}.")
        return match.id

    def run(self, ctx: ToolContext, args: ChangesInput) -> ToolOutput:
        versions = self.services.versions
        if versions is None:
            raise ToolError("Version comparison is not available in this app.")
        other = self._other(ctx, args.versus) or versions.original_of(ctx.dataset_id)
        try:
            found = versions.changes(ctx.dataset_id, other)
        except AppError as error:
            raise ToolError(error.message) from error
        params = {"vs": other} if other else {}
        return ToolOutput(
            f"{found.level_label}: {found.headline}",
            changes_content(found),
            AppLink("changes", params, "Open the Changes page"),
        )
