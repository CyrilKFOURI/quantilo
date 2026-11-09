"""The portrait of each segment, in the columns' own units (032 § 6 and 8): what sets it apart,
a proposed name, simple rules that describe who belongs, and the profile table."""

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from quantilo.ml.data import NUMERIC
from quantilo.ml.preparation import MISSING, Decision

TRAITS = 5  # traits kept per segment
NAMED = 0.3  # a trait names the segment from this effect on
STRONG = 0.8  # a profile cell stands out from this effect on (Cohen's "large")
RULE_ROWS = 20_000  # the rules' tree learns on at most this many rows
RULE_DEPTH = 3
RULE_LEAVES = 3  # rules kept per segment
RULE_CATEGORIES = 10  # categories of a column the rules may name
NOT_GROUPED = "Not grouped"


@dataclass
class SegmentTrait:
    column: str
    kind: Literal["number", "category"]
    text: str  # "income: 2.1× the average", "plan: 78 % are Premium, vs 21 % overall"
    effect: float  # standardised gap with the whole (signed for numbers)
    value: str | None  # the category (categories only)
    segment: float | None  # the segment's mean, or the category's share in it
    overall: float | None  # the same on every row


@dataclass
class SegmentRulePart:
    column: str
    op: Literal["<=", ">", "=", "!="]
    value: float | str
    text: str


@dataclass
class SegmentRule:
    parts: list[SegmentRulePart]
    text: str
    rows: int  # rows of the segment it covers
    precision: float  # share of the rows it covers that are in the segment


@dataclass
class SegmentRules:
    rules: list[SegmentRule]
    recall: float  # share of the segment the rules describe
    precision: float
    text: str


@dataclass
class SegmentProfileCell:
    value: float | str | None  # mean, or most frequent category
    share: float | None  # the category's share (categories)
    effect: float | None
    strong: bool


@dataclass
class SegmentProfileRow:
    column: str
    kind: Literal["number", "category"]
    overall: SegmentProfileCell
    cells: list[SegmentProfileCell]  # one per segment, in the segments' order


def number(value: float) -> str:
    """1234.5 → "1,234.5", 0.000123 → "0.000123": four significant digits, thousands separated."""
    if not math.isfinite(value):
        return str(value)
    if value != 0 and (abs(value) >= 1e15 or abs(value) < 1e-4):
        return f"{value:.3g}"
    digits = max(0, 3 - math.floor(math.log10(abs(value)))) if value else 0
    text = f"{value:,.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def percent(share: float) -> str:
    """0.78 → "78 %", 0.002 → "0.2 %", 0.00004 → "< 0.01 %"."""
    value = 100 * share
    if value == 0:
        return "0 %"
    if value < 0.01:
        return "< 0.01 %"
    if value < 1:
        return f"{value:.2g} %"
    return f"{value:.0f} %"


def _labels(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING).astype(str)


def _numbers(decisions: list[Decision]) -> list[Decision]:
    return [d for d in decisions if d.type in NUMERIC]


def _categories(decisions: list[Decision]) -> list[Decision]:
    return [d for d in decisions if d.type not in NUMERIC and d.type != "datetime"]


def _number_trait(column: str, inside: pd.Series, values: pd.Series) -> SegmentTrait | None:
    present = values.dropna()
    mine = inside.dropna()
    if len(present) < 2 or mine.empty:
        return None
    mean, spread = float(np.mean(present)), float(np.std(present, ddof=1))
    own = float(np.mean(mine))
    if spread <= 0:
        return None
    effect = (own - mean) / spread
    if float(present.min()) >= 0 and mean > 0:
        text = f"{column}: {own / mean:.1f}× the average"
    else:
        text = f"{column}: {number(own)} on average, vs {number(mean)} overall"
    return SegmentTrait(column, "number", text, effect, None, own, mean)


def _category_trait(column: str, inside: pd.Series, values: pd.Series) -> SegmentTrait | None:
    """The category most over-represented in the segment, by standardised difference of shares."""
    overall = _labels(values).value_counts(normalize=True)
    mine = _labels(inside).value_counts(normalize=True)
    best: SegmentTrait | None = None
    for category, share in mine.items():
        p = float(overall.get(category) or 0.0)
        if p <= 0 or p >= 1 or category == MISSING:
            continue
        effect = (float(share) - p) / math.sqrt(p * (1 - p))
        if effect <= 0 or (best is not None and effect <= best.effect):
            continue
        text = f"{column}: {percent(float(share))} are {category}, vs {percent(p)} overall"
        best = SegmentTrait(column, "category", text, effect, str(category), float(share), p)
    return best


def traits(
    frame: pd.DataFrame, labels: np.ndarray, segment: int, decisions: list[Decision]
) -> list[SegmentTrait]:
    """The columns that set a segment apart from all the rows, strongest first."""
    inside = labels == segment
    found: list[SegmentTrait] = []
    for d in _numbers(decisions):
        values = cast(pd.Series, frame[d.column])
        trait = _number_trait(d.column, cast(pd.Series, values[inside]), values)
        if trait:
            found.append(trait)
    for d in _categories(decisions):
        values = cast(pd.Series, frame[d.column])
        trait = _category_trait(d.column, cast(pd.Series, values[inside]), values)
        if trait:
            found.append(trait)
    found.sort(key=lambda t: -abs(t.effect))
    return found[:TRAITS]


def _short(trait: SegmentTrait) -> str:
    if trait.kind == "number":
        return f"{'High' if trait.effect > 0 else 'Low'} {trait.column}"
    value = trait.value or ""
    plain = value.replace(".", "", 1).lstrip("-").isdigit() or value in ("True", "False")
    return f"{trait.column} = {value}" if plain or len(value) > 24 else value


def name(found: list[SegmentTrait]) -> str:
    """ "High income · Premium": the two strongest traits, when they are marked enough."""
    marked = [t for t in found if abs(t.effect) >= NAMED][:2]
    return " · ".join(_short(t) for t in marked) or "Close to average"


def unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for given in names:
        seen[given] = seen.get(given, 0) + 1
        out.append(given if seen[given] == 1 else f"{given} ({seen[given]})")
    return out


# --- Rules ---------------------------------------------------------------------------------------


@dataclass
class _Feature:
    column: str
    category: str | None  # one-hot feature "column = category"; None: a number


def _features(frame: pd.DataFrame, decisions: list[Decision]) -> tuple[np.ndarray, list[_Feature]]:
    """Numbers as they are (missing → median) and the main categories as 0 / 1 columns."""
    columns: list[np.ndarray] = []
    features: list[_Feature] = []
    for d in _numbers(decisions):
        values = cast(pd.Series, frame[d.column])
        median = float(values.median()) if values.notna().any() else 0.0
        columns.append(values.fillna(median).to_numpy(dtype=float))
        features.append(_Feature(d.column, None))
    for d in _categories(decisions):
        labels = _labels(cast(pd.Series, frame[d.column]))
        for category in labels.value_counts().index[:RULE_CATEGORIES]:
            columns.append((labels == category).to_numpy(dtype=float))
            features.append(_Feature(d.column, str(category)))
    if not columns:
        return np.zeros((len(frame), 0)), features
    return np.column_stack(columns), features


def _part(feature: _Feature, upper: bool, threshold: float) -> SegmentRulePart:
    if feature.category is not None:
        op: Literal["<=", ">", "=", "!="] = "!=" if upper else "="
        sign = "≠" if upper else "="
        return SegmentRulePart(
            feature.column, op, feature.category, f"{feature.column} {sign} {feature.category}"
        )
    op = "<=" if upper else ">"
    sign = "≤" if upper else ">"
    return SegmentRulePart(
        feature.column, op, threshold, f"{feature.column} {sign} {number(threshold)}"
    )


def _simplify(parts: list[tuple[_Feature, bool, float]]) -> list[SegmentRulePart]:
    """One bound of each side per number (the tightest), and "= a" makes "≠ b" redundant."""
    lows: dict[str, float] = {}
    highs: dict[str, float] = {}
    equal: dict[str, str] = {}
    different: dict[str, list[str]] = {}
    order: list[str] = []
    for feature, upper, threshold in parts:
        if feature.column not in order:
            order.append(feature.column)
        if feature.category is not None:
            if upper:
                different.setdefault(feature.column, []).append(feature.category)
            else:
                equal[feature.column] = feature.category
        elif upper:
            highs[feature.column] = min(highs.get(feature.column, math.inf), threshold)
        else:
            lows[feature.column] = max(lows.get(feature.column, -math.inf), threshold)
    found = []
    for column in order:
        if column in equal:
            found.append(_part(_Feature(column, equal[column]), False, 0))
            continue
        for category in different.get(column, []):
            found.append(_part(_Feature(column, category), True, 0))
        if column in lows:
            found.append(_part(_Feature(column, None), False, lows[column]))
        if column in highs:
            found.append(_part(_Feature(column, None), True, highs[column]))
    return found


def rules(
    frame: pd.DataFrame,
    labels: np.ndarray,
    segment: int,
    decisions: list[Decision],
    seed: int = 42,
) -> SegmentRules | None:
    """A depth-3 decision tree learns to tell the segment from the rest; its leaves that say
    "in the segment" are the rules, with the share of the segment they describe."""
    x, features = _features(frame, decisions)
    y = labels == segment
    positives = int(y.sum())
    if x.shape[1] == 0 or positives == 0 or positives == len(y):
        return None
    if len(y) > RULE_ROWS:
        index = np.sort(np.random.default_rng(seed).choice(len(y), RULE_ROWS, replace=False))
        x, y = x[index], y[index]
        positives = int(y.sum())
        if positives == 0 or positives == len(y):
            return None
    tree = DecisionTreeClassifier(
        max_depth=RULE_DEPTH, min_samples_leaf=max(5, len(y) // 200), random_state=seed
    ).fit(x, y)
    inner = tree.tree_
    leaves: list[tuple[list[tuple[_Feature, bool, float]], int, int]] = []

    def walk(node: int, path: list[tuple[_Feature, bool, float]]) -> None:
        left, right = int(inner.children_left[node]), int(inner.children_right[node])
        if left == -1:
            members = np.asarray(tree.apply(x)) == node
            inside = int(y[members].sum())
            if inside * 2 > int(members.sum()):
                leaves.append((path, inside, int(members.sum())))
            return
        feature = features[int(inner.feature[node])]
        threshold = float(inner.threshold[node])
        walk(left, [*path, (feature, True, threshold)])
        walk(right, [*path, (feature, False, threshold)])

    walk(0, [])
    if not leaves:
        return None
    leaves.sort(key=lambda leaf: -leaf[1])
    kept = leaves[:RULE_LEAVES]
    found = []
    for path, inside, members in kept:
        parts = _simplify(path)
        text = " and ".join(p.text for p in parts) or "every row"
        found.append(SegmentRule(parts, text, inside, inside / members if members else 0.0))
    covered = sum(inside for _, inside, _ in kept)
    matching = sum(members for _, _, members in kept)
    recall = covered / positives
    precision = covered / matching if matching else 0.0
    text = (
        f"These rules describe {percent(recall)} of the segment "
        f"({percent(precision)} of the rows they match are in it)."
    )
    return SegmentRules(found, recall, precision, text)


# --- Profile table -------------------------------------------------------------------------------


def profiles(
    frame: pd.DataFrame, labels: np.ndarray, segments: list[int], decisions: list[Decision]
) -> list[SegmentProfileRow]:
    """One row per column: the mean (numbers) or the most frequent category in each segment."""
    found: list[SegmentProfileRow] = []
    for d in _numbers(decisions):
        values = cast(pd.Series, frame[d.column])
        mean = float(np.nanmean(values)) if values.notna().any() else math.nan
        spread = float(np.nanstd(values, ddof=1)) if values.notna().sum() > 1 else math.nan
        cells = []
        for segment in segments:
            own = cast(pd.Series, values[labels == segment]).mean()
            own = None if pd.isna(own) else float(own)
            effect = (own - mean) / spread if own is not None and spread > 0 else None
            strong = effect is not None and abs(effect) >= STRONG
            cells.append(SegmentProfileCell(own, None, effect, strong))
        overall = SegmentProfileCell(None if math.isnan(mean) else mean, None, None, False)
        found.append(SegmentProfileRow(d.column, "number", overall, cells))
    for d in _categories(decisions):
        labels_ = _labels(cast(pd.Series, frame[d.column]))
        shares = labels_.value_counts(normalize=True)
        top = str(shares.index[0]) if len(shares) else None
        cells = []
        for segment in segments:
            mine = cast(pd.Series, labels_[labels == segment]).value_counts(normalize=True)
            if mine.empty:
                cells.append(SegmentProfileCell(None, None, None, False))
                continue
            mode, share = str(mine.index[0]), float(mine.iloc[0])
            p = float(shares.get(mode) or 0.0)
            effect = (share - p) / math.sqrt(p * (1 - p)) if 0 < p < 1 else None
            strong = effect is not None and effect >= STRONG
            cells.append(SegmentProfileCell(mode, share, effect, strong))
        overall = SegmentProfileCell(top, float(shares.iloc[0]) if top else None, None, False)
        found.append(SegmentProfileRow(d.column, "category", overall, cells))
    return found
