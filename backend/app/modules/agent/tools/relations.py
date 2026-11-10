"""Tools on relationships: compare 2 or 3 columns, the association matrix, links with a target."""

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field
from quantilo.compare.analyze import Comparison
from quantilo.compare.stats import BivariateTest
from quantilo.correlations.matrix import Pair

from app.modules.agent.engine.tools import AppLink, Tool, ToolContext, ToolError, ToolOutput
from app.modules.agent.tools import AppServices
from app.modules.agent.tools.dataset import _r, find_column

MAX_GROUPS = 15
MAX_PAIRS = 15
METRICS = {"pearson": "Pearson r", "spearman": "Spearman ρ", "cramers_v": "Cramér's V", "eta": "η"}


def _test(t: BivariateTest) -> dict[str, Any]:
    found: dict[str, Any] = {
        "test": t.key,
        "statistic": _r(t.statistic),
        "p_value": _r(t.p_value, 3),
        "n": t.n,
        "recommended": t.recommended,
    }
    if t.effect is not None:
        found["effect"] = {
            "name": t.effect.name,
            "value": _r(t.effect.value, 3),
            "magnitude": t.effect.magnitude,
        }
    if t.note:
        found["note"] = t.note
    return found


def comparison_content(c: Comparison) -> dict[str, Any]:
    content: dict[str, Any] = {
        "kind": c.kind,
        "x": c.x,
        "y": c.y,
        "rows_with_both": c.rows,
        "rows_after_filters": c.total,
        "tests": [_test(t) for t in c.tests],
    }
    if c.z:
        content["z"] = c.z
    if c.color:
        content["color"] = c.color
    if c.regression is not None:
        content["regression"] = {
            "slope": _r(c.regression.slope),
            "intercept": _r(c.regression.intercept),
            "r2": _r(c.regression.r2, 3),
        }
    if c.groups:
        content["groups"] = [
            {
                "group": g.label,
                **({"color": g.color} if g.color else {}),
                "n": g.n,
                "mean": _r(g.mean),
                "median": _r(g.median),
                "q1": _r(g.q1),
                "q3": _r(g.q3),
                "mean_95ci": [_r(g.ci_low), _r(g.ci_high)],
                "mean_ci_method": g.ci_method,
                "mean_ci_reason": g.ci_reason,
                "median_95ci": [_r(g.median_low), _r(g.median_high)],
            }
            for g in c.groups[:MAX_GROUPS]
        ]
        if c.intervals:
            content["intervals"] = c.intervals
    if c.differences:
        # most significant first, after the correction (051)
        ranked = sorted(c.differences, key=lambda d: (d.p_adjusted is None, d.p_adjusted or 0))
        content["pairwise_differences"] = [
            {
                "b_minus_a": f"{d.b} − {d.a}",
                "diff": _r(d.diff),
                "ci_95": [_r(d.low), _r(d.high)],
                "interval": d.interval,
                "p_value": _r(d.p_value, 4),
                "p_adjusted": _r(d.p_adjusted, 4),
            }
            for d in ranked[:MAX_PAIRS]
        ]
    if c.multiple is not None:
        content["multiple_testing"] = c.multiple.sentence
    if c.contingency is not None:
        t = c.contingency
        # totals and shares given ready-made: the model must not compute figures (R-AG-G1)
        column_totals = [sum(row[j] for row in t.counts) for j in range(len(t.columns))]
        content["crosstab"] = {
            "rows_are": c.y,
            "columns_are": c.x,
            "columns": t.columns,
            "column_totals": column_totals,
            "rows": [
                {
                    "category": label,
                    "counts": counts,
                    "total": sum(counts),
                    "pct_of_column": [
                        _pct(n, total) for n, total in zip(counts, column_totals, strict=False)
                    ],
                    "pct_of_row": [_pct(n, sum(counts)) for n in counts],
                }
                for label, counts in list(zip(t.rows, t.counts, strict=False))[:MAX_GROUPS]
            ],
        }
    if c.trend is not None:
        content["trend"] = {
            k: (_r(v, 4) if isinstance(v, float) else v) for k, v in vars(c.trend).items()
        }
    if c.message:
        content["note"] = c.message
    return content


def _pct(part: float, whole: float) -> float | None:
    return round(100 * part / whole, 1) if whole else None


def _verdict(c: Comparison) -> str:
    chosen = next((t for t in c.tests if t.recommended), c.tests[0] if c.tests else None)
    if chosen is None:
        return f"{c.x} × {c.y}: no test applies"
    text = f"{c.x} × {c.y}: {chosen.key}"
    if chosen.p_value is not None:
        text += ", p < 0.001" if chosen.p_value < 0.001 else f", p = {chosen.p_value:.3f}"
    if chosen.effect is not None and chosen.effect.value is not None:
        text += f", {chosen.effect.name} = {chosen.effect.value:.2f}"
        if chosen.effect.magnitude:
            text += f" ({chosen.effect.magnitude})"
    return text


class CompareInput(BaseModel):
    x: str = Field(description="First column.")
    y: str = Field(description="Second column.")
    z: str | None = Field(None, description="Third number, for a 3-D view of three numbers.")
    color: str | None = Field(None, description="A category that splits the chart (optional).")


class CompareColumns(Tool[CompareInput]):
    name: ClassVar[str] = "compare_columns"
    description: ClassVar[str] = (
        "Cross two columns (optionally a third, or a colour category). The app picks the right "
        "visual and tests from the types: correlation tests for two numbers, t-test / ANOVA / "
        "rank tests with effect sizes for a number by a category, χ² with Cramér's V for two "
        "categories, trend tests for a date and a number. Returns every test with its p-value, "
        "effect size and whether it is the recommended one, plus group summaries (mean and median "
        "intervals, bootstrap when a group is small or skewed) and, for groups, every pairwise "
        "difference with its interval and p adjusted for multiple tests, or the crosstab."
    )
    Input: ClassVar[type[BaseModel]] = CompareInput

    def __init__(self, services: AppServices) -> None:
        self.services = services

    def label(self, args: CompareInput) -> str:
        return f"Comparing {args.x} × {args.y}"

    def run(self, ctx: ToolContext, args: CompareInput) -> ToolOutput:
        names = [find_column(self.services, ctx, n).name for n in (args.x, args.y)]
        z = find_column(self.services, ctx, args.z).name if args.z else None
        color = find_column(self.services, ctx, args.color).name if args.color else None
        try:
            c = self.services.compare.compare(
                ctx.dataset_id, names[0], names[1], z, color, ctx.conditions
            )
        except Exception as error:
            raise ToolError(f"These columns can't be compared: {error}") from error
        params = {"x": names[0], "y": names[1]}
        if z:
            params["z"] = z
        if color:
            params["color"] = color
        link = AppLink("compare", params, "Open in Compare")
        return ToolOutput(_verdict(c), comparison_content(c), link)


def _pair(p: Pair) -> dict[str, Any]:
    return {
        "a": p.a,
        "b": p.b,
        "measure": METRICS.get(p.metric, p.metric),
        "value": _r(p.value, 3),
        "p_value": _r(p.p_value, 4),
        "q_value": _r(p.q_value, 3),
        "evidence": p.evidence,
        "n": p.n,
    }


class CorrelationsInput(BaseModel):
    method: Literal["pearson", "spearman"] = Field(
        "pearson", description="Between two numbers: pearson (linear) or spearman (monotonic)."
    )


class GetCorrelations(Tool[CorrelationsInput]):
    name: ClassVar[str] = "get_correlations"
    description: ClassVar[str] = (
        "The mixed association matrix of the dataset (Pearson or Spearman between numbers, "
        "bias-corrected Cramér's V between categories, η between a number and a category), as the "
        "strongest pairs with their evidence level (effect size and p adjusted over every pair: "
        "Benjamini-Hochberg from 10 pairs, Holm below; q_value is the adjusted p), a sentence on "
        "how many links survive the correction, plus the rule-based insights of the Correlations "
        "page. Use it for an overview of what is linked to what."
    )
    Input: ClassVar[type[BaseModel]] = CorrelationsInput

    def __init__(self, services: AppServices) -> None:
        self.services = services

    def label(self, args: CorrelationsInput) -> str:
        return "Computing the association matrix"

    def run(self, ctx: ToolContext, args: CorrelationsInput) -> ToolOutput:
        found = self.services.correlations.correlations(
            ctx.dataset_id, args.method, None, ctx.conditions
        )
        m = found.matrix
        content = {
            "rows": m.rows,
            "sampled": m.sampled,
            "columns": [c.name for c in m.columns],
            "left_out": [{"column": e.name, "reason": e.reason} for e in m.excluded],
            "strongest_pairs": [_pair(p) for p in m.pairs[:MAX_PAIRS]],
            "multiple_testing": m.multiple.sentence if m.multiple else None,
            "insights": [i.text for i in sorted(found.insights, key=lambda i: -i.score)[:10]],
        }
        strong = sum(1 for p in m.pairs if p.evidence == "strong")
        summary = f"{len(m.columns)} columns · {strong} strong links"
        return ToolOutput(summary, content, AppLink("correlations", {}, "Open the correlations"))


class TargetInput(BaseModel):
    target: str = Field(description="The column to explain, e.g. the outcome of interest.")


class RankAssociationsWith(Tool[TargetInput]):
    name: ClassVar[str] = "rank_associations_with"
    description: ClassVar[str] = (
        "Every other column ranked by the strength of its link with one target column, with the "
        "measure, value and evidence level. Use it first for “what drives X?” questions, then "
        "compare_columns on the top ones to see how."
    )
    Input: ClassVar[type[BaseModel]] = TargetInput

    def __init__(self, services: AppServices) -> None:
        self.services = services

    def label(self, args: TargetInput) -> str:
        return f"Ranking what is linked to {args.target}"

    def run(self, ctx: ToolContext, args: TargetInput) -> ToolOutput:
        target = find_column(self.services, ctx, args.target).name
        found = self.services.correlations.correlations(
            ctx.dataset_id, "pearson", None, ctx.conditions
        )
        m = found.matrix
        if target not in {c.name for c in m.columns}:
            reason = next((e.reason for e in m.excluded if e.name == target), "not comparable")
            raise ToolError(f"{target} is not in the association matrix: {reason}.")
        pairs = sorted((p for p in m.pairs if target in (p.a, p.b)), key=lambda p: -abs(p.value))
        ranked = [{**_pair(p), "column": p.b if p.a == target else p.a} for p in pairs[:MAX_PAIRS]]
        top = ranked[0] if ranked else None
        summary = (
            f"Strongest link with {target}: {top['column']} ({top['measure']} {top['value']})"
            if top
            else f"No column is linked to {target}"
        )
        return ToolOutput(
            summary,
            {"target": target, "ranked": ranked},
            AppLink("correlations", {}, "Open the correlations"),
        )


class StructureInput(BaseModel):
    factors: bool = Field(
        False,
        description="Also run the factor analysis (varimax), KMO and Bartlett's test: slower, "
        "for questions about latent factors or whether the columns can be factored.",
    )


class GetStructure(Tool[StructureInput]):
    name: ClassVar[str] = "get_structure"
    description: ClassVar[str] = (
        "How many real dimensions the numeric columns carry (the Structure section of the "
        "Correlations page): a standardised PCA whose number of components is chosen by Horn's "
        "parallel analysis, the share of variance each keeps and the columns that load on it, "
        "with the app's sentence. With factors=true, also the varimax factor analysis, its "
        "communalities, KMO (overall and per column) and Bartlett's sphericity test. Use it for "
        "“are these columns redundant?”, “how many underlying factors?” questions."
    )
    Input: ClassVar[type[BaseModel]] = StructureInput

    def __init__(self, services: AppServices) -> None:
        self.services = services

    def label(self, args: StructureInput) -> str:
        return "Analysing the structure of the numeric columns"

    def run(self, ctx: ToolContext, args: StructureInput) -> ToolOutput:
        found = self.services.correlations.structure(ctx.dataset_id, None, ctx.conditions)
        link = AppLink("correlations", {}, "Open the correlations")
        if found.reason:
            return ToolOutput(found.reason, {"reason": found.reason}, link)
        content: dict[str, Any] = {
            "conclusion": found.conclusion,
            "columns": found.columns,
            "left_out": [{"column": c.name, "reason": c.reason} for c in found.left_out],
            "rows": found.rows,
            "sampled": found.sampled,
            "components_kept": found.retained,
            "kaiser_would_keep": found.kaiser,
            "components": [
                {
                    "component": f"PC{c.index}",
                    "eigenvalue": _r(c.eigenvalue, 3),
                    "variance_pct": round(100 * c.explained, 1),
                    "cumulative_pct": round(100 * c.cumulative, 1),
                    "kept": c.retained,
                    "columns": c.label,
                }
                for c in found.components[: max(found.retained, 2)]
            ],
            "notes": found.notes,
        }
        if args.factors:
            f = self.services.correlations.factors(ctx.dataset_id, None, ctx.conditions)
            content["factor_analysis"] = {
                "conclusion": f.conclusion,
                "factors": f.k,
                "rotation": f.rotation,
                "kmo": f.kmo,
                "kmo_reading": f.kmo_label,
                "kmo_per_column": dict(zip(f.columns, f.msa, strict=False)),
                "bartlett_chi2": f.bartlett_chi2,
                "bartlett_df": f.bartlett_df,
                "bartlett_p": _r(f.bartlett_p, 4),
                "variance_pct": [round(100 * v, 1) for v in f.variance],
                "factor_columns": {f"F{i + 1}": label for i, label in enumerate(f.labels)},
                "communalities": dict(zip(f.columns, f.communalities, strict=False)),
                "notes": f.notes,
            }
        return ToolOutput(found.conclusion, content, link)
