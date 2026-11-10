"""Assemble a report from what the app already computes (overview, profiles, charts, matrix,
comparisons), and write the findings (R-REP-14) with rules, never with a model."""

import math
import re
from dataclasses import dataclass, field
from datetime import datetime

from quantilo.catalog import ColumnView
from quantilo.charts.prepare import Chart
from quantilo.compare.analyze import Comparison
from quantilo.compare.stats import BivariateTest
from quantilo.correlations.matrix import Matrix, Pair
from quantilo.correlations.service import Correlations
from quantilo.filters.model import Condition
from quantilo.profiling.overview import Overview
from quantilo.report.model import (
    Fact,
    Finding,
    Missing,
    PairRow,
    PairView,
    Related,
    Report,
    Share,
    VariableCard,
)
from quantilo.variable.analyze import VariableProfile
from quantilo.variable.outliers import OutlierRule

ALPHA = 0.05
TOP_PAIRS = 10
PAIR_VIEWS = 6  # R-REP-15
KEY_FINDINGS = 12  # R-REP-14
RELATED = 3

TYPE_LABELS = {
    "numeric_continuous": "Continuous",
    "numeric_discrete": "Discrete",
    "categorical_nominal": "Category",
    "categorical_ordinal": "Ordered category",
    "boolean": "Boolean",
    "datetime": "Date / time",
    "text": "Free text",
    "identifier": "Identifier",
    "ignored": "Ignored",
}
METRIC_LABELS = {
    "pearson": "Pearson r",
    "spearman": "Spearman ρ",
    "cramers_v": "Cramér's V",
    "eta": "η (correlation ratio)",
}
SHORT_METRICS = {"pearson": "r", "spearman": "ρ", "cramers_v": "V", "eta": "η"}
TEST_LABELS = {
    "pearson": "Pearson correlation",
    "spearman": "Spearman rank correlation",
    "kendall": "Kendall's tau",
    "welch_t": "Welch's t-test",
    "mann_whitney": "Mann–Whitney U",
    "anova": "One-way ANOVA",
    "welch_anova": "Welch's ANOVA",
    "kruskal": "Kruskal–Wallis",
    "levene": "Levene",
    "chi2": "χ² test of independence",
    "fisher": "Fisher's exact test",
}
NO_CHART = ("identifier", "high_cardinality", "empty")  # R-REP-06, R-REP-11
NUMERIC = ("numeric_continuous", "numeric_discrete")
SEVERITY_WEIGHT = {"warning": 3.0, "link": 2.0, "notice": 1.0}

METHODS = [
    Fact(
        "Normality tests",
        "Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling, Jarque-Bera and Lilliefors each ask "
        "whether the values could come from a normal distribution; p < α = 0.05 rejects it. With "
        "many values, tiny departures are detected: the histogram and the QQ-plot tell whether "
        "they matter.",
    ),
    Fact(
        "QQ-plot",
        "Sorted values against the quantiles a normal distribution would give. On the line: "
        "normal. Bending away at the ends: heavier or lighter tails.",
    ),
    Fact(
        "Outliers (IQR)",
        "Values below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR, where IQR = Q3 − Q1 "
        "(Tukey's fences), as drawn by the box plot whiskers.",
    ),
    Fact(
        "Skewness",
        "Asymmetry of the distribution: 0 for a symmetric one, above 1 (or below −1) clearly "
        "lopsided. A long right tail pulls the mean above the median.",
    ),
    Fact(
        "Pearson r / Spearman ρ",
        "Linear (r) or monotonic (ρ, on ranks) link between two numbers, from −1 to 1.",
    ),
    Fact(
        "Cramér's V",
        "Strength of the link between two categories, from 0 to 1, from the χ² test, corrected "
        "for small tables (Bergsma).",
    ),
    Fact(
        "η (correlation ratio)",
        "Share of a number's variation explained by a category, from 0 to 1 (√ of the ANOVA η²).",
    ),
    Fact(
        "Effect sizes",
        "Cohen's d (gap between two means in standard deviations), rank-biserial r, η² and ε² "
        "(share of variance explained by groups), Cramér's V. Read as negligible, small, medium "
        "or large with Cohen's conventions: the size of an effect matters more than its p-value.",
    ),
    Fact(
        "q-value and evidence",
        "p-values adjusted over every pair (Benjamini-Hochberg). Evidence combines the size of the "
        "effect and the adjusted p: strong, moderate, weak or none.",
    ),
]


@dataclass
class ColumnResult:
    """What the service gathered for one column."""

    view: ColumnView
    profile: VariableProfile | None = None
    chart: Chart | None = None
    extra: list[tuple[str, Chart]] = field(default_factory=list)
    outliers: OutlierRule | None = None
    error: str | None = None


def fmt(value: float | int | None, digits: int = 4) -> str:
    """1234567 → "1,234,567"; 0.123456 → "0.1235"; None → "—"."""
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    if value != value:  # NaN
        return "—"
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    if 1e-3 <= abs(value) < 1e6:
        return f"{value:,.{digits}g}" if abs(value) >= 1 else f"{value:.{digits}g}"
    return f"{value:.3g}"


def pct(value: float) -> str:
    return f"{value:.1f} %" if value < 10 else f"{value:.0f} %"


def size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def anchor(name: str, used: set[str]) -> str:
    base = "v-" + (re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "column")
    found, i = base, 2
    while found in used:
        found, i = f"{base}-{i}", i + 1
    used.add(found)
    return found


def _value(value: str | float | None) -> str:
    return fmt(value) if isinstance(value, float | int) else str(value)


def describe_condition(condition: Condition) -> str:
    """A filter in words: "Sex is female", "Age between 20 and 40"."""
    c, op = condition.column, condition.op
    no = condition.negate
    if op == "eq":
        text = f"{c} {'is not' if no else 'is'} {_value(condition.value)}"
    elif op == "in":
        text = f"{c} {'is not' if no else 'is'} {', '.join(condition.values)}"
    elif op in ("gt", "gte", "lt", "lte"):
        sign = {"gt": ">", "gte": "≥", "lt": "<", "lte": "≤"}[op]
        text = f"{'not ' if no else ''}{c} {sign} {_value(condition.value)}"
    elif op == "between":
        word = "not between" if no else "between"
        text = f"{c} {word} {_value(condition.low)} and {_value(condition.high)}"
    elif op == "contains":
        text = f"{c} {'does not contain' if no else 'contains'} “{condition.value}”"
    elif op == "starts_with":
        text = f"{c} {'does not start' if no else 'starts'} with “{condition.value}”"
    elif op == "is_null":
        text = f"{c} {'is not empty' if no else 'is empty'}"
    elif op == "last_days":
        word = "not in" if no else "in"
        text = f"{c} {word} the last {_value(condition.value)} days up to {condition.high}"
    elif op == "part_in":
        part = (condition.part or "part").replace("_", " ")
        text = f"{part} of {c} {'is not' if no else 'is'} {', '.join(condition.values)}"
    elif op == "unique":
        keys = ", ".join([c, *condition.columns])
        text = f"one row per {keys} (the {condition.keep})"
    else:  # duplicated
        keys = ", ".join([c, *condition.columns])
        text = f"rows whose {keys} appear{' once' if no else ' more than once'}"
    return text


def normality_note(profile: VariableProfile) -> str | None:
    numeric = profile.numeric
    if numeric is None:
        return None
    normality = numeric.normality
    ran = [t for t in normality.tests if t.p_value is not None]
    if not ran:
        return f"Normality tests don't apply. {normality.reason or ''}".strip()
    rejected = sum(1 for t in ran if (t.p_value or 0) < ALPHA)
    if rejected:
        verdict = f"{rejected} of {len(ran)} tests reject normality at α = {ALPHA}."
    else:
        verdict = f"No test rejects normality at α = {ALPHA}."
    if normality.sample < normality.n:
        verdict += f" Tests ran on a sample of {normality.sample:,} of the {normality.n:,} values."
    return verdict


# --- Findings (R-REP-14) --------------------------------------------------------------------


def _missing_finding(name: str, missing: int, missing_pct: float) -> Finding | None:
    if missing_pct >= 20:
        return Finding(
            "warning",
            "Many missing values",
            f"{missing:,} rows ({pct(missing_pct)}) have no {name}: analyses that need it drop "
            "them, and models must impute them.",
            [name],
            score=missing_pct / 10,
        )
    if missing_pct >= 5:
        return Finding(
            "notice",
            "Missing values",
            f"{missing:,} rows ({pct(missing_pct)}) have no {name}.",
            [name],
            score=missing_pct / 20,
        )
    return None


def numeric_findings(
    name: str, profile: VariableProfile, outliers: OutlierRule | None, discrete: bool
) -> list[Finding]:
    assert profile.numeric is not None
    d = profile.numeric.describe
    found: list[Finding] = []
    if d.distinct == 1:
        found.append(
            Finding(
                "warning",
                "Constant column",
                f"Every value of {name} is {fmt(d.min)}: it carries no information.",
                [name],
                score=4,
            )
        )
        return found
    skew = d.skewness
    if skew is not None and abs(skew) > 1:
        side = "right" if skew > 0 else "left"
        pull = "above" if skew > 0 else "below"
        advice = (
            " A log or Box-Cox transform makes it more symmetric for linear models."
            if skew > 0 and (d.min or 0) >= 0
            else ""
        )
        found.append(
            Finding(
                "warning" if abs(skew) > 2 else "notice",
                f"Skewed to the {side}",
                f"Skewness {skew:.2f}: a long {side} tail pulls the mean ({fmt(d.mean)}) {pull} "
                f"the median ({fmt(d.median)}).{advice}",
                [name],
                score=min(abs(skew), 6) / 2,
            )
        )
    if (
        outliers is not None
        and outliers.count
        and outliers.low is not None
        and outliers.high is not None
        and outliers.high > outliers.low  # IQR = 0: the rule would flag every other value
    ):
        found.append(
            Finding(
                "warning" if outliers.pct >= 5 else "notice",
                "Outliers",
                f"{outliers.count:,} values ({pct(outliers.pct)}) fall outside "
                f"[{fmt(outliers.low)}, {fmt(outliers.high)}] by the IQR rule; check them before "
                "averaging or fitting.",
                [name],
                score=1 + outliers.pct / 5,
            )
        )
    if d.count and d.zeros / d.count >= 0.3:
        share = 100 * d.zeros / d.count
        found.append(
            Finding(
                "notice",
                "Many zeros",
                f"{pct(share)} of the values are 0: a zero-inflated variable, often better split "
                "into “zero or not” and “how much”.",
                [name],
                score=share / 30,
            )
        )
    note = normality_note(profile)
    ran = [t for t in profile.numeric.normality.tests if t.p_value is not None]
    if note and ran and any((t.p_value or 0) < ALPHA for t in ran):
        found.append(
            Finding(
                "notice",
                "Not normally distributed",
                f"{note} Prefer rank-based tests (Mann–Whitney, Kruskal–Wallis, Spearman) when "
                "comparing it across groups.",
                [name],
                score=0.5,
            )
        )
    if discrete and d.distinct <= 12:
        found.append(
            Finding(
                "notice",
                "Few distinct values",
                f"Only {d.distinct} whole numbers: Compare reads it as categories, and a model may "
                "do better with it encoded as one.",
                [name],
                score=0.4,
            )
        )
    return found


def categorical_findings(name: str, profile: VariableProfile, rows: int) -> list[Finding]:
    assert profile.categorical is not None
    p = profile.categorical
    found: list[Finding] = []
    present = p.rows - p.missing
    if p.distinct == 1:
        found.append(
            Finding(
                "warning",
                "Constant column",
                f"{name} takes a single value: it carries no information.",
                [name],
                score=4,
            )
        )
        return found
    if p.imbalanced and p.top is not None and p.top_share is not None:
        found.append(
            Finding(
                "warning",
                "One dominant category",
                f"“{p.top['label']}” holds {pct(p.top_share * 100)} of the rows: the other "
                "categories have little data, and accuracy can look high by always answering it.",
                [name],
                score=2 + p.top_share,
            )
        )
    if p.rare_count:
        found.append(
            Finding(
                "notice",
                "Rare categories",
                f"{p.rare_count:,} of the {p.distinct:,} categories each hold under 1 % of the "
                "rows; grouping them as “Other” keeps models and charts stable.",
                [name],
                score=min(p.rare_count, 20) / 10,
            )
        )
    if present and p.distinct > 50 and p.distinct / present > 0.5:
        found.append(
            Finding(
                "notice",
                "Almost unique",
                f"{p.distinct:,} distinct values for {present:,} rows: it behaves like an "
                "identifier more than a category.",
                [name],
                score=1.5,
            )
        )
    return found


def card(
    result: ColumnResult,
    alerts: list[str],
    missing_pct: float,
    distinct: int,
    rows: int,
    used: set[str],
) -> VariableCard:
    view, profile = result.view, result.profile
    found = VariableCard(
        name=view.name,
        anchor=anchor(view.name, used),
        type=view.type,
        type_label=TYPE_LABELS.get(view.type, view.type),
        role=view.role,
        missing_pct=missing_pct,
        distinct=distinct,
        profile=profile
        if profile and profile.numeric and profile.numeric.describe.distinct > 12
        else None,
        outliers=result.outliers,
    )
    facts, findings = found.facts, found.findings
    if result.error:
        findings.append(Finding("warning", "Not analysed", result.error, [view.name], score=1))
    missing = 0
    if profile and profile.numeric:
        d = profile.numeric.describe
        missing = d.missing
        facts += [
            Fact("Mean", fmt(d.mean)),
            Fact("Median", fmt(d.median)),
            Fact("Std", fmt(d.std)),
            Fact("IQR", fmt(d.iqr)),
            Fact("Min", fmt(d.min)),
            Fact("Max", fmt(d.max)),
            Fact("Skewness", fmt(d.skewness, 3)),
            Fact("Zeros", fmt(d.zeros)),
        ]
        if d.negatives:
            facts.append(Fact("Negatives", fmt(d.negatives)))
        found.quantiles = [
            Fact(label, fmt(value))
            for label, value in (
                ("1 %", d.p1),
                ("5 %", d.p5),
                ("25 %", d.q1),
                ("50 %", d.median),
                ("75 %", d.q3),
                ("95 %", d.p95),
                ("99 %", d.p99),
            )
        ]
        findings += numeric_findings(
            view.name, profile, result.outliers, view.type == "numeric_discrete"
        )
    elif profile and profile.categorical:
        p = profile.categorical
        missing = p.missing
        facts += [Fact("Distinct", fmt(p.distinct))]
        if p.top is not None and p.top_share is not None:
            facts.append(Fact("Mode", f"{p.top['label']} ({pct(p.top_share * 100)})"))
        if p.normalized_entropy is not None:
            facts.append(Fact("Evenness", fmt(p.normalized_entropy, 2)))
        if p.rare_count:
            facts.append(Fact("Rare categories", fmt(p.rare_count)))
        findings += categorical_findings(view.name, profile, rows)
    elif profile and profile.date:
        p = profile.date
        missing = p.missing
        facts += [
            Fact("From", p.min or "—"),
            Fact("To", p.max or "—"),
            Fact("Step", p.step or p.granularity),
            Fact("Periods", fmt(p.periods)),
        ]
        if p.gap_count:
            findings.append(
                Finding(
                    "notice",
                    "Gaps in time",
                    f"{p.gap_count:,} empty periods between the first and last date"
                    + (f", from {p.gaps[0]}" if p.gaps else "")
                    + ": a time series will need them filled or flagged.",
                    [view.name],
                    score=1,
                )
            )
    elif profile and profile.identifier:
        p = profile.identifier
        missing = p.missing
        facts += [
            Fact("Unique", pct(p.unique_pct)),
            Fact("Duplicated values", fmt(p.duplicated_values)),
            Fact("Length", f"{fmt(p.min_length)} – {fmt(p.max_length)}"),
        ]
        if p.duplicated_values:
            findings.append(
                Finding(
                    "warning",
                    "Identifiers repeat",
                    f"{p.duplicated_values:,} values of {view.name} appear more than once: either "
                    "rows are duplicated or it is not a unique key.",
                    [view.name],
                    score=2.5,
                )
            )
    elif profile and profile.text:
        p = profile.text
        missing = p.missing
        facts += [
            Fact("Length", f"{fmt(p.min_length)} – {fmt(p.max_length)}"),
            Fact("Mean length", fmt(p.mean_length, 3)),
        ]
    if found_missing := _missing_finding(view.name, missing, missing_pct):
        findings.insert(0, found_missing)
    if "to_review" in alerts:
        findings.append(
            Finding(
                "notice",
                "Type to confirm",
                f"Detected as {found.type_label.lower()} with {view.confidence} confidence: "
                "check it on the Columns page.",
                [view.name],
                score=0.6,
            )
        )
    invalid = int((view.summary or {}).get("invalid_count") or 0)
    if "invalid_values" in alerts and invalid:
        findings.append(
            Finding(
                "warning",
                "Unreadable values",
                f"{invalid:,} values can't be read as {found.type_label.lower()}: they count as "
                "missing in every analysis.",
                [view.name],
                score=2,
            )
        )
    for f in findings:
        f.anchor = found.anchor
    if result.chart is not None:
        if result.chart.kind not in NO_CHART and view.type != "text":
            found.chart = result.chart
        if result.chart.kind == "bars" and result.chart.rows:
            total = sum(b["count"] for b in result.chart.bars) + (
                result.chart.others["count"] if result.chart.others else 0
            )
            found.top = [
                Share(b["label"], b["count"], b["count"] / total if total else 0)
                for b in result.chart.bars[:10]
            ]
    found.extra = result.extra
    return found


def related(matrix: Matrix | None, anchors: dict[str, str]) -> dict[str, list[Related]]:
    """The few columns most linked with each column, from the association matrix."""
    found: dict[str, list[Related]] = {}
    if matrix is None:
        return found
    names = [c.name for c in matrix.columns]
    for i, name in enumerate(names):
        links = [
            (names[j], matrix.metrics[i][j], value)
            for j, value in enumerate(matrix.values[i])
            if j != i and value is not None and not math.isnan(value)
        ]
        links.sort(key=lambda link: -abs(link[2]))
        found[name] = [
            Related(other, anchors.get(other), SHORT_METRICS.get(metric or "", metric or ""), value)
            for other, metric, value in links[:RELATED]
            if abs(value) >= 0.1
        ]
    return found


def test_verdict(tests: list[BivariateTest]) -> str:
    chosen = next((t for t in tests if t.recommended), tests[0] if tests else None)
    if chosen is None:
        return "No test applies."
    parts = [TEST_LABELS.get(chosen.key, chosen.key)]
    if chosen.p_value is not None:
        parts.append("p < 0.001" if chosen.p_value < 0.001 else f"p = {chosen.p_value:.3f}")
    if chosen.effect is not None and chosen.effect.value is not None:
        effect = f"{chosen.effect.name} = {chosen.effect.value:.2f}"
        if chosen.effect.magnitude:
            effect += f" ({chosen.effect.magnitude})"
        parts.append(effect)
    return " · ".join(parts)


def _insight_finding(kind: str, severity: str, text: str, columns: list[str], score: float):
    links = ("correlated", "redundant", "group_difference", "associated")
    titles = {
        "skewed": "Skewed variable",
        "imbalanced": "Imbalanced category",
        "missing": "Missing values",
        "correlated": "Strong correlation",
        "redundant": "Redundant columns",
        "group_difference": "Groups differ",
        "associated": "Linked categories",
    }
    level = "link" if kind in links else ("warning" if severity == "warning" else "notice")
    return Finding(level, titles.get(kind, "Insight"), text, columns, score=2 + score)  # type: ignore[arg-type]


def build_report(
    *,
    title: str,
    source: str,
    version: str,
    overview: Overview,
    results: list[ColumnResult],
    excluded: list[str],
    correlations: Correlations | None,
    comparisons: list[tuple[Pair, Comparison]],
    conditions: list[Condition],
    now: datetime,
) -> Report:
    by_column: dict[str, list[str]] = {}  # alert kinds per column
    for alert in overview.alerts:
        for name in alert.columns:
            by_column.setdefault(name, []).append(alert.kind)
    used: set[str] = set()
    stats = overview.column_stats
    variables = [
        card(
            r,
            by_column.get(r.view.name, []),
            stats[r.view.name].missing_pct if r.view.name in stats else 0.0,
            stats[r.view.name].distinct if r.view.name in stats else 0,
            overview.rows,
            used,
        )
        for r in results
    ]
    anchors = {v.name: v.anchor for v in variables}
    matrix = (
        correlations.matrix
        if correlations is not None and len(correlations.matrix.columns) >= 2
        else None
    )
    links = related(matrix, anchors)
    for v in variables:
        v.related = links.get(v.name, [])

    # dataset-level findings, then the correlation insights, then every column's
    dataset_findings: list[Finding] = []
    if overview.duplicate_rows:
        share = 100 * overview.duplicate_rows / overview.rows if overview.rows else 0
        dataset_findings.append(
            Finding(
                "warning",
                "Duplicate rows",
                f"{overview.duplicate_rows:,} rows ({pct(share)}) are exact copies of another "
                "row: they weigh twice in every statistic.",
                score=3 + share,
            )
        )
    for group in overview.duplicate_columns:
        dataset_findings.append(
            Finding(
                "warning",
                "Duplicate columns",
                f"{', '.join(group)} hold the same values: keep one.",
                list(group),
                anchors.get(group[0]),
                score=3,
            )
        )
    insights = (
        [
            _insight_finding(i.kind, i.severity, i.text, i.columns, i.score)
            for i in sorted(correlations.insights, key=lambda i: -i.score)
        ]
        if correlations is not None
        else []
    )
    for finding in insights:
        finding.anchor = next((anchors[c] for c in finding.columns if c in anchors), None)
    column_findings = [f for v in variables for f in v.findings]
    every = dataset_findings + insights + column_findings
    ranked = sorted(every, key=lambda f: -(SEVERITY_WEIGHT[f.severity] + f.score))
    key: list[Finding] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for finding in ranked:  # one finding per (title, columns), so the list stays varied
        mark = (finding.title, tuple(finding.columns))
        if mark not in seen:
            seen.add(mark)
            key.append(finding)
        if len(key) == KEY_FINDINGS:
            break

    duplicate_columns = sum(len(group) for group in overview.duplicate_columns)
    rows_text = f"{overview.rows:,}"
    if overview.rows != overview.total_rows:
        rows_text += f" of {overview.total_rows:,}"
    columns_text = f"{overview.used_columns:,}"
    if overview.used_columns != overview.columns:
        columns_text += f" of {overview.columns:,}"
    summary = [
        Fact("Rows", rows_text),
        Fact("Columns", columns_text),
        Fact("Missing cells", pct(overview.missing_pct)),
        Fact("Duplicate rows", f"{overview.duplicate_rows:,}"),
        Fact("Duplicate columns", f"{duplicate_columns:,}"),
        Fact("Size", size(overview.size_bytes)),
    ]
    types = [
        (TYPE_LABELS.get(kind, kind), count)
        for kind, count in sorted(overview.types.items(), key=lambda item: -item[1])
    ]
    missing = sorted(
        (Missing(v.name, v.missing_pct) for v in variables if v.missing_pct > 0),
        key=lambda m: -m.pct,
    )
    pairs = (
        [
            PairRow(p.a, p.b, METRIC_LABELS.get(p.metric, p.metric), p.value, p.q_value, p.evidence)
            for p in matrix.pairs[:TOP_PAIRS]
        ]
        if matrix is not None
        else []
    )
    views = [
        PairView(
            pair.a,
            pair.b,
            SHORT_METRICS.get(pair.metric, pair.metric),
            pair.value,
            pair.evidence,
            comparison,
            test_verdict(comparison.tests),
        )
        for pair, comparison in comparisons
    ]
    return Report(
        title=title,
        source=source,
        generated_at=now.strftime("%Y-%m-%d %H:%M"),
        version=version,
        rows=overview.rows,
        total_rows=overview.total_rows,
        columns=overview.used_columns,
        filters=[describe_condition(c) for c in conditions],
        summary=summary,
        types=types,
        key_findings=key,
        findings_total=len(every),
        variables=variables,
        excluded=excluded,
        pairs_views=views,
        matrix=matrix,
        pairs=pairs,
        insights=insights,
        missing=missing,
        complete=len(variables) - len(missing),
        methods=METHODS,
    )


def pairs_to_draw(matrix: Matrix | None) -> list[Pair]:
    """R-REP-15: the strongest pairs, strong and moderate evidence first."""
    if matrix is None:
        return []
    order = {"strong": 0, "moderate": 1, "weak": 2, "none": 3}
    ranked = sorted(matrix.pairs, key=lambda p: (order.get(p.evidence, 3), -abs(p.value)))
    return [p for p in ranked if p.evidence != "none"][:PAIR_VIEWS]


def file_name(title: str) -> str:
    """R-REP-13: "<dataset>-report.html", without characters a file system refuses."""
    safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", title).strip(" .") or "dataset"
    return f"{safe[:120]}-report.html"
