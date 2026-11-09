"""Resemblance, computed as SDMetrics' Quality report (comparable with published scores), plus
coverage and empty values shown beside it.

- Column shapes: KSComplement (numbers, dates), TVComplement (categories).
- Column pair trends: CorrelationSimilarity (two numbers), ContingencySimilarity (categories;
  a number is first cut into 10 equal intervals computed on the real data). Only pairs where the
  real data shows a real link count: |r| > 0.5, or Cramér's V > 0.3.
"""

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import pandas as pd
from scipy import stats

from quantilo.correlations.matrix import correlation, cramers_v, ranks
from quantilo.synthetic.encode import Learned, column_of, labels, numbers, seconds

STRONG_R = 0.5
STRONG_V = 0.3
PAIR_BINS = 10
MISSING = "\x00empty"


@dataclass
class ShapeScore:
    column: str
    metric: Literal["KSComplement", "TVComplement"]
    score: float | None


@dataclass
class PairScore:
    a: str
    b: str
    metric: Literal["CorrelationSimilarity", "ContingencySimilarity"]
    score: float | None
    real: float | None  # real-data link: r for two numbers, Cramér's V otherwise
    synthetic: float | None
    meets: bool  # strong enough in the real data to count


@dataclass
class Coverage:
    column: str
    range: float | None  # RangeCoverage (numbers, dates)
    category: float | None  # CategoryCoverage (categories)
    missing: float  # MissingValueSimilarity
    outliers: float | None  # OutlierCoverage (None: no real outlier)


@dataclass
class Quality:
    score: float  # mean of the two properties (SDMetrics' overall score)
    shapes: float
    trends: float | None  # None: no strong link in the real data to keep
    shape_scores: list[ShapeScore] = field(default_factory=list)
    pair_scores: list[PairScore] = field(default_factory=list)
    coverage: list[Coverage] = field(default_factory=list)


# --- Formulas --------------------------------------------------------------------------------


def ks_complement(real: np.ndarray, synthetic: np.ndarray) -> float | None:
    real, synthetic = real[~np.isnan(real)], synthetic[~np.isnan(synthetic)]
    if real.size == 0 or synthetic.size == 0:
        return None
    statistic = cast(float, stats.ks_2samp(real, synthetic)[0])  # the KS distance
    return float(1 - statistic)


def shares(values: pd.Series) -> pd.Series:
    """Share of each level, empty counting as a level."""
    return labels(values).fillna(MISSING).value_counts(normalize=True)


def tv_complement(real: pd.Series, synthetic: pd.Series) -> float:
    r, s = shares(real), shares(synthetic)
    levels = r.index.union(s.index)
    return float(
        1 - 0.5 * np.abs(r.reindex(levels, fill_value=0) - s.reindex(levels, fill_value=0)).sum()
    )


def correlation_similarity(
    real_x: np.ndarray,
    real_y: np.ndarray,
    synthetic_x: np.ndarray,
    synthetic_y: np.ndarray,
    method: Literal["pearson", "spearman"] = "pearson",
) -> tuple[float | None, float | None, float | None]:
    """(1 - |r_synthetic - r_real| / 2, r_real, r_synthetic)."""

    def r(x: np.ndarray, y: np.ndarray) -> float | None:
        if method == "spearman":
            x, y = ranks(x), ranks(y)
        return correlation(x, y)[0]

    real_r, synthetic_r = r(real_x, real_y), r(synthetic_x, synthetic_y)
    if real_r is None or synthetic_r is None:
        return None, real_r, synthetic_r
    return 1 - abs(synthetic_r - real_r) / 2, real_r, synthetic_r


def contingency_similarity(
    real_a: np.ndarray, real_b: np.ndarray, synthetic_a: np.ndarray, synthetic_b: np.ndarray
) -> float:
    """1 - ½ ΣΣ |real share - synthetic share| over the cells of the two crossed tables.
    Inputs are shared integer codes (≥ 0), empty already a code of its own."""
    columns = int(max(real_b.max(initial=0), synthetic_b.max(initial=0))) + 1
    size = (int(max(real_a.max(initial=0), synthetic_a.max(initial=0))) + 1) * columns
    real = np.bincount(real_a * columns + real_b, minlength=size) / max(len(real_a), 1)
    synthetic = np.bincount(synthetic_a * columns + synthetic_b, minlength=size) / max(
        len(synthetic_a), 1
    )
    return float(1 - 0.5 * np.abs(real - synthetic).sum())


def range_coverage(real: np.ndarray, synthetic: np.ndarray) -> float | None:
    real, synthetic = real[~np.isnan(real)], synthetic[~np.isnan(synthetic)]
    if real.size == 0 or synthetic.size == 0:
        return None
    low, high = real.min(), real.max()
    if high == low:
        return 1.0
    missed_low = max(0.0, (synthetic.min() - low) / (high - low))
    missed_high = max(0.0, (high - synthetic.max()) / (high - low))
    return float(max(0.0, 1 - (missed_low + missed_high)))


def category_coverage(real: pd.Series, synthetic: pd.Series) -> float | None:
    found = set(labels(real).dropna())
    if not found:
        return None
    return len(found & set(labels(synthetic).dropna())) / len(found)


def missing_similarity(real: pd.Series, synthetic: pd.Series) -> float:
    return float(1 - abs(real.isna().mean() - synthetic.isna().mean()))


def outlier_coverage(real: np.ndarray, synthetic: np.ndarray) -> float | None:
    real, synthetic = real[~np.isnan(real)], synthetic[~np.isnan(synthetic)]
    if real.size < 4 or synthetic.size == 0:
        return None
    q1, q3 = np.quantile(real, [0.25, 0.75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    real_share = float(((real < low) | (real > high)).mean())
    if real_share == 0:
        return None
    synthetic_share = float(((synthetic < low) | (synthetic > high)).mean())
    return min(1.0, synthetic_share / real_share)


# --- Columns as the pair formulas need them --------------------------------------------------


def as_numbers(series: pd.Series, item: Learned) -> np.ndarray:
    return seconds(series) if item.kind == "date" else numbers(series)


def shared_codes(real: pd.Series, synthetic: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Category labels as codes shared by both sets; empty is a code too."""
    joined = pd.concat([labels(real).fillna(MISSING), labels(synthetic).fillna(MISSING)])
    codes, _ = pd.factorize(joined, sort=True)
    return codes[: len(real)].astype(np.int64), codes[len(real) :].astype(np.int64)


def binned(
    real: np.ndarray, synthetic: np.ndarray, bins: int = PAIR_BINS
) -> tuple[np.ndarray, np.ndarray]:
    """Equal intervals computed on the real values, applied to both; empty is interval `bins`."""
    present = real[~np.isnan(real)]
    low, high = (present.min(), present.max()) if present.size else (0.0, 1.0)
    inner = np.linspace(low, high, bins + 1)[1:-1]

    def cut(values: np.ndarray) -> np.ndarray:
        codes = np.searchsorted(inner, np.nan_to_num(values), side="right")
        return np.where(np.isnan(values), bins, np.clip(codes, 0, bins - 1)).astype(np.int64)

    return cut(real), cut(synthetic)


def _without_empty(
    a: np.ndarray, b: np.ndarray, empty_a: int | None, empty_b: int | None
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.ones(len(a), dtype=bool)
    if empty_a is not None:
        keep &= a != empty_a
    if empty_b is not None:
        keep &= b != empty_b
    return np.where(keep, a, -1), np.where(keep, b, -1)


def quality(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    learned: list[Learned],
    method: Literal["pearson", "spearman"] = "pearson",
) -> Quality:
    shape_scores, coverage = [], []
    codes: dict[str, tuple[np.ndarray, np.ndarray, int | None]] = {}
    values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for item in learned:
        r, s = column_of(real, item.name), column_of(synthetic, item.name)
        if item.categorical:
            shape_scores.append(ShapeScore(item.name, "TVComplement", tv_complement(r, s)))
            real_codes, synthetic_codes = shared_codes(r, s)
            joined = sorted(set(labels(r).fillna(MISSING)) | set(labels(s).fillna(MISSING)))
            empty = joined.index(MISSING) if MISSING in joined else None
            codes[item.name] = (real_codes, synthetic_codes, empty)
            coverage.append(
                Coverage(item.name, None, category_coverage(r, s), missing_similarity(r, s), None)
            )
            continue
        rn, sn = as_numbers(r, item), as_numbers(s, item)
        values[item.name] = (rn, sn)
        shape_scores.append(ShapeScore(item.name, "KSComplement", ks_complement(rn, sn)))
        real_bins, synthetic_bins = binned(rn, sn)
        codes[item.name] = (real_bins, synthetic_bins, PAIR_BINS)
        coverage.append(
            Coverage(
                item.name,
                range_coverage(rn, sn),
                None,
                missing_similarity(r, s),
                outlier_coverage(rn, sn),
            )
        )
    pair_scores = []
    for i, a in enumerate(learned):
        for b in learned[i + 1 :]:
            if not a.categorical and not b.categorical:
                (rx, sx), (ry, sy) = values[a.name], values[b.name]
                score, real_r, synthetic_r = correlation_similarity(rx, ry, sx, sy, method)
                meets = real_r is not None and abs(real_r) > STRONG_R
                pair_scores.append(
                    PairScore(
                        a.name, b.name, "CorrelationSimilarity", score, real_r, synthetic_r, meets
                    )
                )
                continue
            ra, sa, empty_a = codes[a.name]
            rb, sb, empty_b = codes[b.name]
            score = contingency_similarity(ra, rb, sa, sb)
            real_v = cramers_v(*_without_empty(ra, rb, empty_a, empty_b))[0]
            synthetic_v = cramers_v(*_without_empty(sa, sb, empty_a, empty_b))[0]
            meets = real_v is not None and real_v > STRONG_V
            pair_scores.append(
                PairScore(
                    a.name, b.name, "ContingencySimilarity", score, real_v, synthetic_v, meets
                )
            )
    shape_values = [s.score for s in shape_scores if s.score is not None]
    shapes = float(np.mean(shape_values)) if shape_values else 0.0
    counted = [p.score for p in pair_scores if p.meets and p.score is not None]
    trends = float(np.mean(counted)) if counted else None
    score = shapes if trends is None else (shapes + trends) / 2
    return Quality(score, shapes, trends, shape_scores, pair_scores, coverage)


def verdict(score: float) -> str:
    if score >= 0.85:
        return "Very close"
    if score >= 0.7:
        return "Close"
    if score >= 0.5:
        return "Rough"
    return "Far from the real data"
