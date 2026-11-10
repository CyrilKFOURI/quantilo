"""Tools on the dataset and its columns: overview, column list, one column's profile, counts."""

from difflib import get_close_matches
from typing import Any, ClassVar

from pydantic import BaseModel, Field
from quantilo.charts.prepare import ChartParams
from quantilo.filters.model import Condition
from quantilo.variable.analyze import VariableProfile

from app.modules.agent.engine.tools import AppLink, Tool, ToolContext, ToolError, ToolOutput
from app.modules.agent.tools import AppServices
from app.services.columns.service import ColumnView

ALPHA = 0.05
NUMERIC = ("numeric_continuous", "numeric_discrete")


def _r(value: float | None, digits: int = 4) -> float | None:
    """Numbers the model reads: a few significant digits, never NaN."""
    if value is None or value != value:
        return None
    return float(f"{value:.{digits}g}")


def find_column(services: AppServices, ctx: ToolContext, name: str) -> ColumnView:
    """The column, or an error that names the close matches (R-AG-T3)."""
    views = [v for v in services.columns.columns(ctx.dataset_id) if not v.excluded]
    for view in views:
        if view.name == name:
            return view
    names = [v.name for v in views]
    by_case = {n.lower(): n for n in names}
    if name.lower() in by_case:
        raise ToolError(f"Column {name!r} not found; did you mean {by_case[name.lower()]!r}?")
    close = get_close_matches(name, names, n=3)
    hint = f" Close matches: {', '.join(close)}." if close else f" Columns: {', '.join(names)}."
    raise ToolError(f"Column {name!r} not found.{hint}")


class NoInput(BaseModel):
    pass


class GetDatasetOverview(Tool[NoInput]):
    name: ClassVar[str] = "get_dataset_overview"
    description: ClassVar[str] = (
        "Size, column types, missing values, duplicates and quality alerts of the dataset, over "
        "the filtered rows. Start here when the question is about the dataset as a whole or its "
        "quality. Not for one column's distribution (use profile_column)."
    )
    Input: ClassVar[type[BaseModel]] = NoInput

    def __init__(self, services: AppServices) -> None:
        self.services = services

    def label(self, args: NoInput) -> str:
        return "Reading the dataset overview"

    def run(self, ctx: ToolContext, args: NoInput) -> ToolOutput:
        o = self.services.profiling.overview(ctx.dataset_id, ctx.conditions)
        content = {
            "rows": o.rows,
            "total_rows": o.total_rows,
            "columns": o.used_columns,
            "types": o.types,
            "missing_cells_pct": o.missing_pct,
            "duplicate_rows": o.duplicate_rows,
            "duplicate_columns": o.duplicate_columns,
            "alerts": [
                {"severity": a.severity, "message": a.message, "columns": a.columns}
                for a in o.alerts
            ],
        }
        summary = f"{o.rows:,} rows × {o.used_columns} columns · {len(o.alerts)} alerts"
        return ToolOutput(summary, content, AppLink("overview", {}, "Open the overview"))


class ListColumns(Tool[NoInput]):
    name: ClassVar[str] = "list_columns"
    description: ClassVar[str] = (
        "Every column with its detected type, semantic role, detection confidence, missing share "
        "and number of distinct values. Use it to pick columns for an analysis."
    )
    Input: ClassVar[type[BaseModel]] = NoInput

    def __init__(self, services: AppServices) -> None:
        self.services = services

    def label(self, args: NoInput) -> str:
        return "Listing the columns"

    def run(self, ctx: ToolContext, args: NoInput) -> ToolOutput:
        views = self.services.columns.columns(ctx.dataset_id)
        stats = self.services.profiling.overview(ctx.dataset_id, ctx.conditions).column_stats
        columns = []
        for v in views:
            s = stats.get(v.name)
            columns.append(
                {
                    "name": v.name,
                    "type": v.type,
                    "role": v.role,
                    "confidence": v.confidence,
                    "excluded": v.excluded,
                    "missing_pct": s.missing_pct if s else None,
                    "distinct": s.distinct if s else None,
                }
            )
        used = sum(1 for v in views if not v.excluded)
        return ToolOutput(
            f"{used} columns", {"columns": columns}, AppLink("columns", {}, "Open the columns")
        )


class ColumnInput(BaseModel):
    column: str = Field(description="Exact column name, as list_columns gives it.")


class ProfileColumn(Tool[ColumnInput]):
    name: ClassVar[str] = "profile_column"
    description: ClassVar[str] = (
        "One column in depth: key statistics, distribution shape, normality tests, outliers "
        "(IQR rule) for numbers; top categories and balance for categories; range and gaps for "
        "dates. Use it before choosing a test that assumes normality, or to describe a variable."
    )
    Input: ClassVar[type[BaseModel]] = ColumnInput

    def __init__(self, services: AppServices) -> None:
        self.services = services

    def label(self, args: ColumnInput) -> str:
        return f"Profiling {args.column}"

    def run(self, ctx: ToolContext, args: ColumnInput) -> ToolOutput:
        view = find_column(self.services, ctx, args.column)
        profile = self.services.variable.profile(ctx.dataset_id, view.name, ctx.conditions)
        content: dict[str, Any] = {"column": view.name, "type": view.type, "role": view.role}
        summary = f"{view.name}: {view.type}"
        if profile.numeric is not None:
            content |= self._numeric(profile)
            outliers = self.services.variable.outliers(
                ctx.dataset_id, view.name, "iqr", ctx.conditions
            )
            rule = next((r for r in outliers.rules if r.method == "iqr"), None)
            if rule is not None:
                content["outliers_iqr"] = {
                    "count": rule.count,
                    "pct": _r(rule.pct, 3),
                    "low_fence": _r(rule.low),
                    "high_fence": _r(rule.high),
                }
            d = profile.numeric.describe
            summary = (
                f"{view.name}: mean {_r(d.mean)}, median {_r(d.median)}, skew {_r(d.skewness, 3)}"
            )
        elif profile.categorical is not None:
            p = profile.categorical
            content |= {
                "rows": p.rows,
                "missing": p.missing,
                "distinct": p.distinct,
                "top": p.top,
                "top_share": _r(p.top_share, 3),
                "imbalanced": p.imbalanced,
                "rare_categories": p.rare_count,
                "evenness": _r(p.normalized_entropy, 3),
            }
            chart = self.services.profiling.chart(
                ctx.dataset_id, view.name, ChartParams(top_n=10), ctx.conditions
            )
            content["most_frequent"] = [
                {"label": b["label"], "count": b["count"]} for b in chart.bars
            ]
            if chart.others:
                content["others"] = chart.others
            summary = f"{view.name}: {p.distinct} categories"
        elif profile.date is not None:
            p = profile.date
            content |= {
                "rows": p.rows,
                "missing": p.missing,
                "from": p.min,
                "to": p.max,
                "step": p.step or p.granularity,
                "periods": p.periods,
                "empty_periods": p.gap_count,
            }
            summary = f"{view.name}: {p.min} → {p.max}"
        elif profile.identifier is not None:
            p = profile.identifier
            content |= {
                "rows": p.rows,
                "missing": p.missing,
                "distinct": p.distinct,
                "unique_pct": _r(p.unique_pct, 3),
                "duplicated_values": p.duplicated_values,
            }
        elif profile.text is not None:
            p = profile.text
            content |= {
                "rows": p.rows,
                "missing": p.missing,
                "distinct": p.distinct,
                "length": [p.min_length, p.max_length],
                "mean_length": _r(p.mean_length, 3),
                # 050: what the texts say and in which language
                "kind": {"label": "short text", "address": "address"}.get(
                    view.role or "", "free text"
                ),
                "language": p.language_label,
                "median_words": p.median_words,
                "distinct_words": p.distinct_words,
                "top_words": [b["label"] for b in p.words[:10]],
                "top_expressions": [b["label"] for b in p.bigrams[:10]],
                "repeated_texts": p.repeated_count,
            }
        link = AppLink("variable", {"column": view.name}, f"Open {view.name}")
        return ToolOutput(summary, content, link)

    @staticmethod
    def _numeric(profile: VariableProfile) -> dict[str, Any]:
        assert profile.numeric is not None
        d = profile.numeric.describe
        normality = profile.numeric.normality
        tests = [
            {"test": t.key, "statistic": _r(t.statistic), "p_value": _r(t.p_value, 3)}
            for t in normality.tests
            if t.p_value is not None
        ]
        rejected = sum(1 for t in normality.tests if t.p_value is not None and t.p_value < ALPHA)
        return {
            "rows": d.rows,
            "count": d.count,
            "missing": d.missing,
            "distinct": d.distinct,
            "zeros": d.zeros,
            "negatives": d.negatives,
            "mean": _r(d.mean),
            "std": _r(d.std),
            "min": _r(d.min),
            "p5": _r(d.p5),
            "q1": _r(d.q1),
            "median": _r(d.median),
            "q3": _r(d.q3),
            "p95": _r(d.p95),
            "max": _r(d.max),
            "skewness": _r(d.skewness, 3),
            "kurtosis": _r(getattr(d, "kurtosis", None), 3),
            "normality": {
                "tests": tests,
                "rejected_at_0.05": f"{rejected} of {len(tests)}",
                "tested_on": normality.sample,
                "note": normality.reason,
            },
        }


class CountInput(BaseModel):
    conditions: list[Condition] = Field(
        description=(
            "Extra conditions combined with AND, on top of the user's filters. Examples: "
            '{"column": "Sex", "op": "in", "values": ["female"]}, '
            '{"column": "Age", "op": "between", "low": 20, "high": 40}, '
            '{"column": "Cabin", "op": "is_null"}, '
            '{"column": "Survived", "op": "in", "values": ["True"]} (a boolean column).'
        ),
        max_length=10,
    )
    within: list[Condition] = Field(
        default_factory=list,
        description=(
            "Optional: count among the rows meeting these conditions only, so the share is a "
            "rate within that group. Example: conditions Survived in True, within Sex in female "
            "→ the survival rate of women. Rows with an empty value in a 'within' column are "
            "left out of the group."
        ),
        max_length=10,
    )


class CountRows(Tool[CountInput]):
    name: ClassVar[str] = "count_rows"
    description: ClassVar[str] = (
        "How many rows meet some conditions, and their share of the filtered rows, or of a group "
        "('within'). Shows no row. Use it for questions like “how many passengers under 10 "
        "survived?” or “what share of women survived?”: the app computes every share, so never "
        "divide counts yourself."
    )
    Input: ClassVar[type[BaseModel]] = CountInput

    def __init__(self, services: AppServices) -> None:
        self.services = services

    def label(self, args: CountInput) -> str:
        return "Counting rows"

    def run(self, ctx: ToolContext, args: CountInput) -> ToolOutput:
        for condition in [*args.conditions, *args.within]:
            find_column(self.services, ctx, condition.column)
        group = ctx.conditions + args.within
        try:
            base = self.services.profiling.overview(ctx.dataset_id, group).rows
            matched = self.services.profiling.overview(ctx.dataset_id, group + args.conditions).rows
        except Exception as error:
            raise ToolError(f"These conditions can't be applied: {error}") from error
        share = matched / base if base else 0.0
        content: dict[str, Any] = {
            "rows": matched,
            "of_rows": base,
            "share": _r(share, 3),
            "pct": round(100 * share, 1),
        }
        if args.within:
            content["of"] = "the rows meeting 'within'"
        return ToolOutput(f"{matched:,} of {base:,} rows ({share:.1%})", content)
