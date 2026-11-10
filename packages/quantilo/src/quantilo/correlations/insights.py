"""Rule-based insights: what deserves attention, in one sentence each, with a link to the proof."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from quantilo.correlations.evidence import SIGNIFICANT
from quantilo.correlations.matrix import ROLES, Loaded, Matrix, MatrixColumn, Pair
from quantilo.ml import codes

SKEWED = 2.0
DOMINANT = 0.7
MISSING = 0.2
REDUNDANT = 0.95
MAX_INSIGHTS = 30  # column insights, and links per level of evidence
CODE_ROLES = {"identifier", "postal_code", "entity", "customer_key", "order_key"}  # R-COR-01
SPIKE_LINK = 0.5  # R-COR-02: links this strong are checked for a spike of identical values
SPIKE_SHARE = 0.005  # a value repeated on this share of the rows (and 5 rows at least)…
SPIKE_ROWS = 5
SPIKE_OUTSIDE = (1, 99)  # the spread of the other values: percentiles…
SPIKE_GAP = 0.25  # …and the spike lies this share of it beyond all of them
COORDINATES = {"latitude", "longitude"}
SPIKE_DROP = 0.2  # the link loses this much without the spike's rows (or changes sign)
MIN_CLEAN_ROWS = 10

Kind = Literal[
    "skewed",
    "imbalanced",
    "missing",
    "correlated",
    "redundant",
    "group_difference",
    "associated",
    "codes",
    "spike",
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


def code_columns(columns: list[MatrixColumn], loaded: Loaded) -> dict[str, str]:
    """R-COR-01 (094, COR-1): numbers that are codes, by their role (identifier, postal code,
    entity) or by their name and whole values (R-CODE-01: bin__, block, patient_nbr). Their
    order means nothing: a link between two codes is a fact of the numbering, not a finding.
    Returns each one's reason."""
    found: dict[str, str] = {}
    for column in columns:
        values = loaded.arrays.get(column.name)
        if ROLES.get(column.type) != "num" or values is None:
            continue
        if column.role in CODE_ROLES:
            found[column.name] = f"A code (its role: {column.role.replace('_', ' ')})."
        elif (word := codes.numeric_code(column.name, values)) is not None:
            found[column.name] = f"A code: its name says so ({word}) and its values are whole."
    return found


def codes_insight(found: dict[str, str]) -> Insight | None:
    names = list(found)
    if not names:
        return None
    listed = ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else "")
    verb = "looks like a code" if len(names) == 1 else "look like codes"
    return Insight(
        "codes:" + "|".join(names),
        "codes",
        "warning",
        f"{listed} {verb} (R-COR-01): their order means nothing, so their links are not "
        "listed as findings. If they are real numbers, change their type in Columns.",
        names,
        InsightLink("variable", names[0]),
        4.8,
    )


def _spikes(values: np.ndarray, role: str | None) -> list[float]:
    """The extreme values (the lowest, the highest) repeated on many rows and far from all the
    others: -200, -100000, 9999, coordinates at 0. A 0 at the bottom of a quantity that is
    otherwise positive is a real zero (no second floor), except for a coordinate."""
    present = values[~np.isnan(values)]
    if len(present) < SPIKE_ROWS * 4:
        return []
    found = []
    for value in {float(present.min()), float(present.max())}:
        count = int((present == value).sum())
        if count < max(SPIKE_ROWS, SPIKE_SHARE * len(present)):
            continue
        others = present[present != value]
        if len(others) < SPIKE_ROWS:
            continue
        low, high = np.percentile(others, SPIKE_OUTSIDE)
        gap = SPIKE_GAP * max(float(high - low), 1e-12)
        if not (value < others.min() - gap or value > others.max() + gap):
            continue
        if value == 0 and others.min() > 0 and role not in COORDINATES:
            continue
        found.append(value)
    return sorted(found)


def _link(a: np.ndarray, b: np.ndarray, spearman: bool) -> float | None:
    if len(a) < MIN_CLEAN_ROWS or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    if spearman:
        a, b = stats.rankdata(a), stats.rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def spike_insight(
    pair: Pair, loaded: Loaded, spikes: dict[str, list[float]], roles: dict[str, str | None]
) -> Insight | None:
    """R-COR-02 (094, COR-2): a strong link carried by a spike of identical values (a missing-
    value code such as -200 at the same rows of both columns, coordinates at 0): without those
    rows the link falls apart. The finding becomes a pointer to the Clean page."""
    if pair.metric not in ("pearson", "spearman") or abs(pair.value) < SPIKE_LINK:
        return None
    for name in (pair.a, pair.b):
        if name not in spikes:
            spikes[name] = _spikes(loaded.arrays[name], roles.get(name))
    peaks = {name: spikes[name] for name in (pair.a, pair.b) if spikes[name]}
    if not peaks:
        return None
    a, b = loaded.arrays[pair.a], loaded.arrays[pair.b]
    keep = ~np.isnan(a) & ~np.isnan(b)
    for name, values in peaks.items():
        keep &= ~np.isin(loaded.arrays[name], values)
    clean = _link(a[keep], b[keep], pair.metric == "spearman")
    if clean is not None and abs(clean) >= abs(pair.value) - SPIKE_DROP and clean * pair.value > 0:
        return None
    symbol = "r" if pair.metric == "pearson" else "ρ"
    said = " and ".join(
        f"{' / '.join(f'{v:g}' for v in values)} in {name}" for name, values in peaks.items()
    )
    without = (
        f"without those rows, {symbol} = {clean:.2f}"
        if clean is not None
        else "without those rows, too little is left"
    )
    return Insight(
        f"spike:{pair.a}|{pair.b}",
        "spike",
        "warning",
        f"The link between {pair.a} and {pair.b} ({symbol} = {pair.value:.3f}) is carried by a "
        f"repeated value ({said}): {without}. Check those values on the Clean page: they may "
        "stand for missing values (R-COR-02).",
        [pair.a, pair.b],
        InsightLink("compare", pair.a, pair.b),
        4.5 + abs(pair.value),
    )


def insights_for(columns: list[MatrixColumn], matrix: Matrix, loaded: Loaded) -> list[Insight]:
    """Column insights, then links: every level is sent, the page chooses which to show. Codes
    give one note instead of links (R-COR-01); a link carried by a spike of identical values
    gives a pointer to cleaning instead of a finding (R-COR-02)."""
    found = sorted(column_insights(loaded, columns, top_shares(loaded)), key=lambda i: -i.score)
    found = found[:MAX_INSIGHTS]
    roles = {column.name: column.role for column in matrix.columns}
    coded = code_columns(columns, loaded)
    note = codes_insight(coded)
    if note is not None:
        found.append(note)
    spikes: dict[str, list[float]] = {}
    column_roles = {column.name: column.role for column in columns}
    checked: list[Insight] = []
    for pair in matrix.pairs:
        if pair.a in coded or pair.b in coded:
            continue
        spike = spike_insight(pair, loaded, spikes, column_roles)
        if spike is not None:
            found.append(spike)
            continue
        insight = pair_insight(pair, roles)
        if insight is not None:
            checked.append(insight)
    links = sorted(checked, key=lambda insight: -insight.score)
    for level in LEVEL_SCORE:
        found += [insight for insight in links if insight.evidence == level][:MAX_INSIGHTS]
    found.sort(key=lambda insight: -insight.score)
    return found
