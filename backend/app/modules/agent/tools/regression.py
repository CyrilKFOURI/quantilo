"""Tool on the Regression tab (068): a regression with inference, as the page computes it."""

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field
from quantilo.errors import ExplorerError
from quantilo.regression.analyze import RegressionCoefficient, RegressionReport
from quantilo.regression.service import Request

from app.modules.agent.engine.tools import AppLink, Tool, ToolContext, ToolError, ToolOutput
from app.modules.agent.tools import AppServices
from app.modules.agent.tools.dataset import _r, find_column

MAX_TERMS = 30


class RegressionInput(BaseModel):
    target: str | None = Field(
        None, description="The number or yes/no column to explain (default: picked by rule)."
    )
    columns: list[str] | None = Field(
        None,
        description=(
            "Explanatory columns (default: the target's 5 strongest links). Include the "
            "controls: a coefficient reads 'other columns held fixed'."
        ),
        max_length=60,
    )
    cluster: str | None = Field(
        None, description="Rows that share this column share their shocks (store, patient…)."
    )
    se: Literal[
        "auto", "classic", "hc0", "hc1", "hc2", "hc3", "hac", "cluster", "cluster2", "dk"
    ] = Field("auto", description="Standard errors; auto follows the rules (keep it).")
    family: Literal[
        "auto", "ols", "logit", "probit", "poisson", "negbin", "gamma", "log_ols", "boxcox",
        "fractional", "mnlogit",
    ] = Field(
        "auto",
        description=(
            "Model family; auto follows the target (logit for yes/no, Poisson for counts, "
            "multinomial for 3-10 classes, least squares otherwise). Force negbin, log_ols, "
            "gamma or fractional only when a check proposes it or the user asks."
        ),
    )  # fmt: skip
    terms: list[str] | None = Field(
        None,
        description=(
            "070: extra terms: 'spline:x' (a curve, when the straight-lines check proposes it), "
            "'square:x', 'log:x', 'interaction:a|b' (only for a hypothesis the user states), "
            "'trend:date', 'month:date', 'weekday:date', 'reference:column|level'."
        ),
        max_length=20,
    )
    center: bool = Field(
        False, description="070: centre the numbers (main effects read at the mean of the others)."
    )


def _term(t: RegressionCoefficient) -> dict[str, Any]:
    found: dict[str, Any] = {
        "term": t.term,
        "estimate": _r(t.estimate),
        "ci95": [_r(t.ci_low), _r(t.ci_high)],
        "p_value": _r(t.p_value, 3),
        "p_adjusted": _r(t.q_value, 3),
        "evidence": t.evidence,
        "standardized": _r(t.standardized, 3),
    }
    if t.marginal is not None:
        found["average_marginal_effect"] = _r(t.marginal)
        found["odds_ratio"] = _r(t.odds_ratio)
    if t.ratio is not None:
        found["ratio"] = _r(t.ratio)
        found["ratio_ci95"] = [_r(t.ratio_low), _r(t.ratio_high)]
    if t.outcome is not None:
        found["outcome"] = t.outcome
    return found


READS = {
    "units": "units of the target",
    "points": "points of probability (average marginal effects)",
    "ratio": "ratios exp(b): % change of the target",
}


def _family(r: RegressionReport) -> dict[str, Any] | None:
    f = r.family
    if f is None:
        return None
    found: dict[str, Any] = {
        "kind": f.kind,
        "label": f.label,
        "rule": f.rule,
        "reason": f.reason,
        "effects_read_as": READS[f.reads],
    }
    extra = {
        "exposure": f.exposure,
        "negative_binomial_alpha": _r(f.alpha, 3),
        "variance_over_mean": _r(f.dispersion, 2),
        "duan_smearing": _r(f.smearing, 3),
        "boxcox_lambda": _r(f.boxcox_lambda, 3),
        "classes": f.outcomes or None,
        "note": f.note,
    }
    found.update({key: value for key, value in extra.items() if value is not None})
    return found


def regression_content(r: RegressionReport) -> dict[str, Any]:
    q = r.quality
    return {
        "target": r.target,
        "model": r.model_label,
        "model_reason": r.model_reason,
        "family": _family(r),
        "standard_errors": {
            "kind": r.se.label,
            "reason": r.se.reason,
            "rule": r.se.rule,
            "lag": r.se.lag,
            "clusters": r.se.clusters,
            "warning": r.se.warning,
        },
        "rows_used": q.rows,
        "rows_after_filters": q.total,
        "sampled": q.sampled,
        "r2": _r(q.r2, 3),
        "adjusted_r2": _r(q.adj_r2, 3),
        "out_of_sample_r2": _r(q.oos_r2, 3),
        "pseudo_r2": _r(q.pseudo_r2, 3),
        "conclusion": r.sentence,
        "multiple_testing": r.multiple.sentence,
        "terms": [_term(t) for t in r.terms[:MAX_TERMS]],
        "checks": [
            {"check": c.title, "status": c.status, "sentence": c.sentence}
            | ({"try_family": c.family} if c.family else {})
            | ({"try_term": c.term} if c.term else {})
            for c in r.checks
        ],
        "left_out": [{"term": item.term, "reason": item.reason} for item in r.left_out],
        "structure": r.structure.message,
        "hints": r.hints,
    }


class FitRegression(Tool[RegressionInput]):
    name: ClassVar[str] = "fit_regression"
    description: ClassVar[str] = (
        "A regression with inference, as on the Regression tab. The family follows the target "
        "by rule, with its reason: least squares (weighted when a column has the role weight), "
        "logistic for a yes/no target with average marginal effects, Poisson for counts (rate "
        "ratios, an exposure column as offset, overdispersion checked), multinomial logit for "
        "3 to 10 classes; log(y), Gamma, negative binomial, probit, Box-Cox and fractional "
        "logit on request. The standard errors are chosen by rule and named with their reason "
        "(HC3 by default, clustered by entity for panel data, HAC for one dated series). Gives "
        "every coefficient with its 95 % interval, p and p adjusted over the coefficients, "
        "evidence, the out-of-sample R², and the checks in sentences (spread, time, straight "
        "lines, normality, repeated columns, influential rows). Use it for 'what drives X, other "
        "things equal?' after rank_associations_with."
    )
    Input: ClassVar[type[BaseModel]] = RegressionInput

    def __init__(self, services: AppServices) -> None:
        self.services = services

    def label(self, args: RegressionInput) -> str:
        return f"Fitting a regression of {args.target}" if args.target else "Fitting a regression"

    def run(self, ctx: ToolContext, args: RegressionInput) -> ToolOutput:
        service = self.services.regression
        if service is None:
            raise ToolError("Regression is not available in this app.")
        target = find_column(self.services, ctx, args.target).name if args.target else None
        columns = (
            [find_column(self.services, ctx, name).name for name in args.columns]
            if args.columns is not None
            else None
        )
        cluster = find_column(self.services, ctx, args.cluster).name if args.cluster else None
        request = Request(
            target,
            tuple(columns) if columns is not None else None,
            None,
            cluster,
            args.se,
            family=args.family,
            terms=tuple(args.terms or ()),
            center=args.center,
        )
        try:
            found = service.regression(ctx.dataset_id, request, ctx.conditions)
        except ExplorerError as error:
            raise ToolError(error.message) from error
        clear = sum(1 for t in found.terms[1:] if t.q_value is not None and t.q_value < 0.05)
        summary = (
            f"{found.target} ~ {len(found.terms) - 1} terms · {found.model_label} · "
            f"{found.se.label} · {clear} clear effects"
        )
        params = {"target": found.target, "columns": ",".join(found.columns)}
        if cluster:
            params["cluster"] = cluster
        if args.se != "auto":
            params["se"] = args.se
        if args.family != "auto":
            params["family"] = args.family
        if args.terms:
            params["terms"] = ",".join(args.terms)
        if args.center:
            params["center"] = "1"
        return ToolOutput(
            summary, regression_content(found), AppLink("regression", params, "Open the regression")
        )
