"""099 — What distinguishes pinned rows: the rows a user pinned, set against the other rows, column
by column, in the columns' own units (rules R-PIN-06 to R-PIN-14 of the spec).

Pure: it reads two frames (the pinned rows, the other rows) already typed; the engine loads them."""

import math
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import stats

from quantilo.compare.multiple import adjust
from quantilo.ml.preparation import MISSING, Decision
from quantilo.segments.portrait import SegmentRules, number, percent, rules

MIN_PINNED = 2  # R-PIN-06
MIN_OTHERS = 2
MIN_SHARED = 2  # R-PIN-08: a category counts when at least 2 pinned rows share it
FEW_PINNED = 10  # R-PIN-11
RULES_FROM = 5  # R-PIN-12
NUMBER_GAP = 0.5  # R-PIN-07: |Glass's delta| kept from here
CATEGORY_GAP = 0.5  # R-PIN-08
MISSING_GAP = 0.3  # R-PIN-09: 30 points
ALPHA = 0.05  # R-PIN-10
KEPT = 8  # R-PIN-14


@dataclass
class PinnedTrait:
    column: str
    kind: Literal["number", "category", "missing"]
    rule: str  # R-PIN-07, R-PIN-08 or R-PIN-09
    text: str  # "amount: 9.8× the others' average (1,960 vs 200)"
    effect: float  # standardised gap, signed (numbers, empty cells) or not (categories)
    pinned: float  # mean, share of the category, or share of empty cells among the pinned rows
    others: float  # the same among the other rows
    value: str | None  # the category (categories only)
    p_value: float | None  # after Holm's correction over every column tested (R-PIN-10)
    chance: bool  # p_value is 0.05 or more: could be chance


@dataclass
class PinnedComparison:
    label: str | None  # the label compared; None: every pinned row
    pinned: int  # pinned rows found and compared
    others: int  # other rows in the dataset
    compared: int  # other rows read (a sample beyond FIT_ROWS, R-PIN-13)
    sampled: bool
    columns: list[str]  # the columns compared (R-PIN-14)
    traits: list[PinnedTrait]
    rules: SegmentRules | None
    reading: str  # the conclusion, in one sentence
    warnings: list[str] = field(default_factory=list)


def _test_numbers(inside: np.ndarray, outside: np.ndarray) -> float | None:
    try:
        return float(cast(Any, stats.mannwhitneyu(inside, outside, alternative="two-sided")).pvalue)
    except ValueError:
        return None


def _fisher(a: int, b: int, c: int, d: int) -> float | None:
    """Two-sided Fisher exact test on [[a, b], [c, d]]."""
    try:
        return float(cast(Any, stats.fisher_exact([[a, b], [c, d]])).pvalue)
    except ValueError:
        return None


def number_trait(column: str, pinned: pd.Series, others: pd.Series) -> PinnedTrait | None:
    """R-PIN-07: Glass's delta (the spread of the other rows), with a Mann–Whitney test."""
    mine = pinned.dropna().to_numpy(dtype=float)
    rest = others.dropna().to_numpy(dtype=float)
    if len(mine) < MIN_PINNED or len(rest) < MIN_OTHERS:
        return None
    spread = float(np.std(rest, ddof=1))
    if not math.isfinite(spread) or spread <= 0:
        return None
    own, mean = float(np.mean(mine)), float(np.mean(rest))
    effect = (own - mean) / spread
    if abs(effect) < NUMBER_GAP:
        return None
    if float(min(rest.min(), mine.min())) >= 0 and mean > 0:
        text = f"{column}: {own / mean:.1f}× the others' average ({number(own)} vs {number(mean)})"
    else:
        text = f"{column}: {number(own)} on average, vs {number(mean)} for the others"
    return PinnedTrait(
        column, "number", "R-PIN-07", text, effect, own, mean, None,
        _test_numbers(mine, rest), False,
    )  # fmt: skip


def _labels(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING).astype(str)


def category_trait(column: str, pinned: pd.Series, others: pd.Series) -> PinnedTrait | None:
    """R-PIN-08: the category most over-represented among the pinned rows, with Fisher's test."""
    mine = _labels(pinned)
    rest = _labels(others)
    mine = cast(pd.Series, mine[mine != MISSING])
    rest = cast(pd.Series, rest[rest != MISSING])
    if len(mine) < MIN_PINNED or len(rest) < MIN_OTHERS:
        return None
    counts_in = mine.value_counts()
    counts_out = rest.value_counts()
    best: PinnedTrait | None = None
    for category, count in counts_in.items():
        if int(count) < MIN_SHARED:
            continue  # a value of one pinned row says nothing about the group
        share = float(count) / len(mine)
        outside = int(counts_out.get(category) or 0)
        p = max(outside, 0.5) / len(rest)  # a category absent elsewhere stays measurable
        if p >= 1:
            continue
        effect = (share - p) / math.sqrt(p * (1 - p))
        if effect < CATEGORY_GAP or (best is not None and effect <= best.effect):
            continue
        text = (
            f"{column}: {percent(share)} of pinned rows are {category}, "
            f"vs {percent(outside / len(rest))} of the others"
        )
        p_value = _fisher(int(count), len(mine) - int(count), outside, len(rest) - outside)
        best = PinnedTrait(
            column, "category", "R-PIN-08", text, effect, share, outside / len(rest),
            str(category), p_value, False,
        )  # fmt: skip
    return best


def missing_trait(column: str, pinned: pd.Series, others: pd.Series) -> PinnedTrait | None:
    """R-PIN-09: empty cells far more (or far less) frequent among the pinned rows."""
    if len(pinned) < MIN_PINNED or len(others) < MIN_OTHERS:
        return None
    empty_in, empty_out = int(pinned.isna().sum()), int(others.isna().sum())
    share_in, share_out = empty_in / len(pinned), empty_out / len(others)
    if abs(share_in - share_out) < MISSING_GAP:
        return None
    q = min(max(empty_out, 0.5) / len(others), 1 - 0.5 / len(others))
    effect = (share_in - q) / math.sqrt(q * (1 - q))
    text = (
        f"{column} is empty in {percent(share_in)} of pinned rows, "
        f"vs {percent(share_out)} of the others"
    )
    p_value = _fisher(empty_in, len(pinned) - empty_in, empty_out, len(others) - empty_out)
    return PinnedTrait(
        column, "missing", "R-PIN-09", text, effect, share_in, share_out, None, p_value, False
    )


def _reading(label: str | None, pinned: int, traits: list[PinnedTrait]) -> str:
    who = f"The {pinned} rows pinned “{label}”" if label else f"The {pinned} pinned rows"
    if not traits:
        return f"{who} look like the other rows on every column compared."
    firm = [t for t in traits if not t.chance]
    lead = (firm or traits)[0]
    if not firm:
        return f"{who} differ most on {lead.column}, but every difference could be chance."
    columns = ", ".join(t.column for t in firm[:3])
    return f"{who} differ most on {columns}."


def compare(
    pinned: pd.DataFrame,
    others: pd.DataFrame,
    numbers: list[str],
    categories: list[str],
    label: str | None = None,
    others_total: int | None = None,
    seed: int = 42,
) -> PinnedComparison:
    """The pinned rows against the other rows (R-PIN-06 to R-PIN-14)."""
    total = len(others) if others_total is None else others_total
    columns = [*numbers, *categories]
    sampled = total > len(others)
    base = PinnedComparison(label, len(pinned), total, len(others), sampled, columns, [], None, "")
    if len(pinned) < MIN_PINNED:
        base.reading = (
            f"Pin at least {MIN_PINNED} rows to compare them with the others ({len(pinned)} found)."
        )
        return base
    if len(others) < MIN_OTHERS:
        base.reading = f"Fewer than {MIN_OTHERS} other rows: nothing to compare the pinned rows to."
        return base
    found: list[PinnedTrait] = []
    for column in numbers:
        inside, outside = cast(pd.Series, pinned[column]), cast(pd.Series, others[column])
        found += [t for t in (number_trait(column, inside, outside),) if t]
        found += [t for t in (missing_trait(column, inside, outside),) if t]
    for column in categories:
        inside, outside = cast(pd.Series, pinned[column]), cast(pd.Series, others[column])
        found += [t for t in (category_trait(column, inside, outside),) if t]
        found += [t for t in (missing_trait(column, inside, outside),) if t]
    # R-PIN-10: Holm over every column tested (one test per column and kind)
    tests = len(columns) * 2
    known = [t.p_value for t in found]
    padded = known + [1.0] * max(0, tests - len(known))  # the columns that kept no trait
    for trait, adjusted in zip(found, adjust(padded, "holm")[: len(found)], strict=True):
        trait.p_value = adjusted
        trait.chance = adjusted is None or adjusted >= ALPHA
    found.sort(key=lambda t: (t.chance, -abs(t.effect)))
    base.traits = found[:KEPT]
    if len(pinned) < FEW_PINNED:
        base.warnings.append(f"{len(pinned)} pinned rows: read these differences as hints.")
    if sampled:
        base.warnings.append(
            f"The other rows are a reproducible sample of {len(others):,} of {total:,}."
        )
    if len(pinned) >= RULES_FROM and columns:
        frame = cast(pd.DataFrame, pd.concat([pinned[columns], others[columns]], ignore_index=True))
        labels = np.r_[np.ones(len(pinned), dtype=np.int64), np.zeros(len(others), dtype=np.int64)]
        decisions = [Decision(c, "numeric_continuous", True) for c in numbers]
        decisions += [Decision(c, "categorical_nominal", True) for c in categories]
        found_rules = rules(frame, labels, 1, decisions, seed)
        if found_rules is not None:
            found_rules.text = found_rules.text.replace("of the segment", "of the pinned rows")
            found_rules.text = found_rules.text.replace("are in it", "are pinned")
        base.rules = found_rules
    base.reading = _reading(label, len(pinned), base.traits)
    return base
