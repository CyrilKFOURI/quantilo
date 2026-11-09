"""Rule-based insights: what deserves attention, in one sentence each, with a link to the proof."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from quantilo.correlations.evidence import SIGNIFICANT
from quantilo.correlations.matrix import ROLES, Loaded, Matrix, MatrixColumn, Pair

SKEWED = 2.0
DOMINANT = 0.7
MISSING = 0.2
REDUNDANT = 0.95
MAX_INSIGHTS = 30  # column insights, and links per level of evidence

Kind = Literal[
    "skewed", "imbalanced", "missing", "correlated", "redundant", "group_difference", "associated"
]


@dataclass
class InsightLink:
    page: Literal["variable", "compare"]
    x: str
    y: str | None = None


@dataclass
class Insight:
    id: str  # stable across recomputations
    kind: Kind
    severity: Literal["warning", "info"]
    text: str
    columns: list[str]
    link: InsightLink
    score: float  # importance, for sorting
    evidence: Literal["strong", "moderate", "weak"] | None = None  # links only


def _pct(share: float) -> str:
    return f"{share * 100:.0f}%"


def _p(p: float | None) -> str:
    if p is None:
        return ""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def column_insights(
    loaded: Loaded,
    columns: list[MatrixColumn],
    top_shares: dict[str, tuple[str, float]],
) -> list[Insight]:
    """Skewness, missing values and dominant categories, on the rows the matrix was built on."""
    insights: list[Insight] = []
    for c in columns:
        data = loaded.arrays.get(c.name)
        if data is None or not len(data):
            continue
        skew = None
        if ROLES.get(c.type) == "num":
            present_values = data[~np.isnan(data)]
            missing = 1 - len(present_values) / len(data)
            if len(present_values) > 2 and np.ptp(present_values) > 0:
                skew = float(stats.skew(present_values, bias=False))
        else:
            missing = float((data < 0).mean())
        if missing >= MISSING:
            insights.append(
                Insight(
                    f"missing:{c.name}",
                    "missing",
                    "warning",
                    f"{_pct(missing)} of the values of {c.name} are missing.",
                    [c.name],
                    InsightLink("variable", c.name),
                    0.5 + missing,
                )
            )
        if skew is not None and skew == skew and abs(skew) > SKEWED:
            side = "right: a few very large values" if skew > 0 else "left: a few very small values"
            insights.append(
                Insight(
                    f"skewed:{c.name}",
                    "skewed",
                    "info",
                    f"{c.name} is strongly skewed to the {side} (skewness {skew:.1f}). "
                    "Medians describe it better than means; a log transform may help.",
                    [c.name],
                    InsightLink("variable", c.name),
                    0.4 + min(abs(skew), 10) / 20,
                )
            )
        if c.name in top_shares:
            label, share = top_shares[c.name]
            if share > DOMINANT:
                insights.append(
                    Insight(
                        f"imbalanced:{c.name}",
                        "imbalanced",
                        "info",
                        f'{c.name} is imbalanced: "{label}" makes up {_pct(share)} of the rows.',
                        [c.name],
                        InsightLink("variable", c.name),
                        0.3 + share / 2,
                    )
                )
    return insights


# Wording per level; "chance" = notable effect but the adjusted p-value does not rule out luck.
CORRELATED = {
    "strong": ("{b} rises strongly with {a}", "{b} falls strongly as {a} rises"),
    "moderate": ("{b} tends to rise with {a}", "{b} tends to fall as {a} rises"),
    "small": ("{b} rises slightly with {a}", "{b} falls slightly as {a} rises"),
    "chance": (
        "{b} may rise with {a}, but this could be chance",
        "{b} may fall as {a} rises, but this could be chance",
    ),
}
GROUPS = {
    "strong": "{num} differs strongly across the groups of {groups}",
    "moderate": "{num} differs clearly across the groups of {groups}",
    "small": "{num} differs slightly across the groups of {groups}",
    "chance": "{num} seems to differ across the groups of {groups}, but this could be chance",
}
ASSOCIATIONS = {
    "strong": "{a} and {b} are strongly associated",
    "moderate": "{a} and {b} are associated",
    "small": "{a} and {b} are slightly associated",
    "chance": "{a} and {b} seem associated, but this could be chance",
}
LEVEL_SCORE = {"strong": 3.0, "moderate": 2.0, "weak": 1.0}


def _wording(pair: Pair) -> str:
    if pair.evidence != "weak":
        return pair.evidence
    return "chance" if pair.q_value is None or pair.q_value >= SIGNIFICANT else "small"


def _adj(q: float | None) -> str:
    if q is None:
        return ""
    return "adj. p < 0.001" if q < 0.001 else f"adj. p = {q:.3f}"


def pair_insight(pair: Pair, roles: dict[str, str]) -> Insight | None:
    """One sentence per link worth a look, worded by its level of evidence."""
    if pair.evidence == "none":
        return None
    a, b, value, level = pair.a, pair.b, pair.value, pair.evidence
    link = InsightLink("compare", a, b)
    key = f"{a}|{b}"
    reliable = pair.q_value is not None and pair.q_value < SIGNIFICANT
    score = LEVEL_SCORE[level] + abs(value)
    wording = _wording(pair)
    if pair.metric in ("pearson", "spearman"):
        symbol = "r" if pair.metric == "pearson" else "ρ"
        if abs(value) > REDUNDANT and reliable:
            return Insight(
                f"redundant:{key}",
                "redundant",
                "warning",
                f"{a} and {b} are almost identical ({symbol} = {value:.3f}): probably redundant, "
                "keep one of them in models.",
                [a, b],
                link,
                4 + abs(value),
                level,
            )
        rising, falling = CORRELATED[wording]
        sentence = (rising if value > 0 else falling).format(a=a, b=b)
        return Insight(
            f"correlated:{key}",
            "correlated",
            "info",
            f"{sentence} ({symbol} = {value:.2f}, {_adj(pair.q_value)}).",
            [a, b],
            link,
            score,
            level,
        )
    if pair.metric == "eta":
        numeric, groups = (a, b) if roles.get(a) == "num" else (b, a)
        sentence = GROUPS[wording].format(num=numeric, groups=groups)
        return Insight(
            f"group_difference:{key}",
            "group_difference",
            "info",
            f"{sentence} (η = {value:.2f}, {_adj(pair.q_value)}).",
            [a, b],
            InsightLink("compare", groups, numeric),
            score,
            level,
        )
    if value > REDUNDANT and reliable:
        return Insight(
            f"redundant:{key}",
            "redundant",
            "warning",
            f"{a} and {b} carry nearly the same information (Cramér's V = {value:.2f}): "
            "one probably determines the other.",
            [a, b],
            link,
            4 + value,
            level,
        )
    sentence = ASSOCIATIONS[wording].format(a=a, b=b)
    return Insight(
        f"associated:{key}",
        "associated",
        "info",
        f"{sentence} (Cramér's V = {value:.2f}, {_adj(pair.q_value)}).",
        [a, b],
        link,
        score,
        level,
    )


def top_shares(loaded: Loaded) -> dict[str, tuple[str, float]]:
    """Most frequent category and its share among present values (code 0 = most frequent)."""
    shares: dict[str, tuple[str, float]] = {}
    for name, labels in loaded.labels.items():
        codes = loaded.arrays[name]
        present = codes[codes >= 0]
        if len(present) and labels:
            shares[name] = (labels[0], float((present == 0).mean()))
    return shares


def insights_for(columns: list[MatrixColumn], matrix: Matrix, loaded: Loaded) -> list[Insight]:
    """Column insights, then links: every level is sent, the page chooses which to show."""
    found = sorted(column_insights(loaded, columns, top_shares(loaded)), key=lambda i: -i.score)
    found = found[:MAX_INSIGHTS]
    roles = {column.name: column.role for column in matrix.columns}
    links = sorted(
        (insight for pair in matrix.pairs if (insight := pair_insight(pair, roles))),
        key=lambda insight: -insight.score,
    )
    for level in LEVEL_SCORE:
        found += [insight for insight in links if insight.evidence == level][:MAX_INSIGHTS]
    found.sort(key=lambda insight: -insight.score)
    return found
