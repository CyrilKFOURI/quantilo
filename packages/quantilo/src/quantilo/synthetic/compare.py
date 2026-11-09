"""Real and synthetic side by side, in the manner of Sweetviz but with gap measures:
counts also as % of each set, shared intervals, aligned levels, paired statistics with Δ."""

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

from quantilo.synthetic.associations import Measure, association
from quantilo.synthetic.encode import Learned, column_of, labels
from quantilo.synthetic.quality import MISSING, Quality, as_numbers, shared_codes

MINI_BINS = 12
TOP_LEVELS = 20
TOP_VALUES = 10
SCATTER_POINTS = 600
OTHERS = "Others"


# --- Summary list ----------------------------------------------------------------------------


@dataclass
class ColumnLine:
    column: str
    type: str
    kind: str
    metric: str  # "KS" or "TV"
    score: float | None
    real: list[float]  # mini overlaid histogram (or top levels): shares of the real rows
    synthetic: list[float]
    biggest: str  # the largest difference, in words


def _pct(value: float) -> str:
    return f"{value:.0%}" if abs(value) >= 0.01 or value == 0 else f"{value:.1%}"


def _biggest_numbers(real: np.ndarray, synthetic: np.ndarray) -> str:
    real, synthetic = real[~np.isnan(real)], synthetic[~np.isnan(synthetic)]
    if real.size == 0 or synthetic.size == 0:
        return "No values to compare"
    span = float(real.max() - real.min()) or 1.0
    gaps = {
        "mean": (real.mean(), synthetic.mean()),
        "median": (np.median(real), np.median(synthetic)),
        "spread": (real.std(), synthetic.std()),
    }
    name, (r, s) = max(gaps.items(), key=lambda item: abs(item[1][1] - item[1][0]) / span)
    if r == 0:
        return f"{name} {s - r:+.3g}"
    return f"{name} {(s - r) / abs(r):+.0%}"


def _get(counts: pd.Series, level: object) -> float:
    """A count or share by level, 0 for a level that is not there."""
    return float(cast(float, counts.get(level, 0.0)))


def _biggest_levels(real: pd.Series, synthetic: pd.Series) -> str:
    r = labels(real).fillna("(empty)").value_counts(normalize=True)
    s = labels(synthetic).fillna("(empty)").value_counts(normalize=True)
    levels = r.index.union(s.index)
    gaps = (s.reindex(levels, fill_value=0) - r.reindex(levels, fill_value=0)).abs()
    if gaps.empty:
        return "No values to compare"
    level = gaps.idxmax()
    before, after = _get(r, level), _get(s, level)
    if before == 0:
        return f"'{level}' invented ({_pct(after)})"
    return f"'{level}' {_pct(before)} → {_pct(after)}"


def summaries(
    real: pd.DataFrame, synthetic: pd.DataFrame, learned: list[Learned], quality: Quality
) -> list[ColumnLine]:
    """One line per column, most damaged first."""
    scores = {s.column: s.score for s in quality.shape_scores}
    found = []
    for item in learned:
        r, s = column_of(real, item.name), column_of(synthetic, item.name)
        if item.categorical:
            rs = labels(r).fillna("(empty)").value_counts(normalize=True)
            ss = labels(s).fillna("(empty)").value_counts(normalize=True)
            top = list(rs.head(MINI_BINS).index)
            mini_real = [_get(rs, v) for v in top]
            mini_synthetic = [_get(ss, v) for v in top]
            biggest, metric = _biggest_levels(r, s), "TV"
        else:
            rn, sn = as_numbers(r, item), as_numbers(s, item)
            edges = _edges(rn, MINI_BINS)
            mini_real, mini_synthetic = _shares(rn, edges), _shares(sn, edges)
            biggest, metric = _biggest_numbers(rn, sn), "KS"
        found.append(
            ColumnLine(
                item.name,
                item.type,
                item.kind,
                metric,
                scores.get(item.name),
                mini_real,
                mini_synthetic,
                biggest,
            )
        )
    found.sort(key=lambda c: c.score if c.score is not None else 2.0)
    return found


# --- One column in detail --------------------------------------------------------------------


@dataclass
class Count:
    label: str  # "Values", "Empty", "Distinct", "Zeros"
    real: int
    real_share: float
    synthetic: int
    synthetic_share: float


@dataclass
class Stat:
    label: str
    real: float | None
    synthetic: float | None
    delta: float | None
    delta_share: float | None  # Δ / |real|


@dataclass
class ValueRow:
    value: str
    real: int
    real_share: float
    synthetic: int
    synthetic_share: float
    delta: float  # synthetic share - real share
    status: Literal["missing in synthetic", "invented"] | None = None


@dataclass
class SharedHistogram:
    edges: list[float]
    real: list[float]  # share of each set's values per interval
    synthetic: list[float]


@dataclass
class Ecdf:
    real_x: list[float]
    synthetic_x: list[float]
    levels: list[float]  # cumulative share at each x


@dataclass
class TargetOverlay:
    target: str
    measure: str  # "Mean of revenue" or "Share of 'True'"
    groups: list[str]
    real: list[float | None]
    synthetic: list[float | None]


@dataclass
class ColumnDetail:
    column: str
    type: str
    kind: str
    score: float | None
    counts: list[Count] = field(default_factory=list)
    histogram: SharedHistogram | None = None
    ecdf: Ecdf | None = None
    stats: list[Stat] = field(default_factory=list)
    frequent_real: list[ValueRow] = field(default_factory=list)
    frequent_synthetic: list[ValueRow] = field(default_factory=list)
    smallest: list[ValueRow] = field(default_factory=list)
    largest: list[ValueRow] = field(default_factory=list)
    levels: list[ValueRow] = field(default_factory=list)  # categories: aligned table
    target: TargetOverlay | None = None


def _edges(real: np.ndarray, bins: int | None) -> np.ndarray:
    present = real[~np.isnan(real)]
    if present.size == 0:
        return np.array([0.0, 1.0])
    low, high = float(present.min()), float(present.max())
    if high == low:
        return np.array([low - 0.5, high + 0.5])
    if bins is None:  # Freedman-Diaconis, between 10 and 50 intervals
        q1, q3 = np.quantile(present, [0.25, 0.75])
        width = 2 * (q3 - q1) * present.size ** (-1 / 3)
        bins = int(np.clip(np.ceil((high - low) / width), 10, 50)) if width > 0 else 20
    return np.linspace(low, high, bins + 1)


def _shares(values: np.ndarray, edges: np.ndarray) -> list[float]:
    present = values[~np.isnan(values)]
    if present.size == 0:
        return [0.0] * (len(edges) - 1)
    counts = np.histogram(np.clip(present, edges[0], edges[-1]), edges)[0]
    return (counts / present.size).tolist()


def _counts(
    real: pd.Series, synthetic: pd.Series, numbers: tuple[np.ndarray, np.ndarray] | None
) -> list[Count]:
    def row(label: str, r: int, s: int) -> Count:
        return Count(
            label,
            r,
            r / len(real) if len(real) else 0.0,
            s,
            s / len(synthetic) if len(synthetic) else 0.0,
        )

    rows = [
        row("Values", int(real.notna().sum()), int(synthetic.notna().sum())),
        row("Empty", int(real.isna().sum()), int(synthetic.isna().sum())),
        row("Distinct", int(real.nunique()), int(synthetic.nunique())),
    ]
    if numbers is not None:
        rows.append(row("Zeros", int((numbers[0] == 0).sum()), int((numbers[1] == 0).sum())))
    return rows


STAT_LEVELS = [
    ("Min", 0.0),
    ("5 %", 0.05),
    ("Q1", 0.25),
    ("Median", 0.5),
    ("Q3", 0.75),
    ("95 %", 0.95),
    ("Max", 1.0),
]


def _stats(real: np.ndarray, synthetic: np.ndarray) -> list[Stat]:
    real, synthetic = real[~np.isnan(real)], synthetic[~np.isnan(synthetic)]

    def describe(values: np.ndarray) -> dict[str, float | None]:
        if values.size == 0:
            return {}
        quantiles = np.quantile(values, [q for _, q in STAT_LEVELS])
        found: dict[str, float | None] = {
            label: float(v) for (label, _), v in zip(STAT_LEVELS, quantiles, strict=True)
        }
        found["Mean"] = float(values.mean())
        found["Std. deviation"] = float(values.std(ddof=1)) if values.size > 1 else None
        found["IQR"] = float(quantiles[4] - quantiles[2])
        constant = values.size < 3 or values.std() == 0
        found["Skewness"] = None if constant else float(stats.skew(values))
        found["Kurtosis"] = None if constant else float(stats.kurtosis(values))
        return found

    r, s = describe(real), describe(synthetic)
    labels_ = [label for label, _ in STAT_LEVELS] + [
        "Mean",
        "Std. deviation",
        "IQR",
        "Skewness",
        "Kurtosis",
    ]
    rows = []
    for label in labels_:
        a, b = r.get(label), s.get(label)
        delta = b - a if a is not None and b is not None else None
        share = delta / abs(a) if delta is not None and a else None
        rows.append(Stat(label, a, b, delta, share))
    return rows


def _value_rows(real: pd.Series, synthetic: pd.Series, values: list, fmt) -> list[ValueRow]:
    rc, sc = real.value_counts(), synthetic.value_counts()
    rn, sn = max(len(real), 1), max(len(synthetic), 1)
    rows = []
    for v in values:
        r, s = int(_get(rc, v)), int(_get(sc, v))
        status = "invented" if r == 0 else "missing in synthetic" if s == 0 else None
        rows.append(ValueRow(fmt(v), r, r / rn, s, s / sn, s / sn - r / rn, status))  # type: ignore[arg-type]
    return rows


def _format_number(item: Learned):
    if item.kind == "date":
        return lambda v: str(pd.Timestamp(v)).replace(" 00:00:00", "")
    return lambda v: f"{v:g}" if isinstance(v, float) else str(v)


def _overlay(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    item: Learned,
    target: Learned,
    edges: np.ndarray | None,
) -> TargetOverlay | None:
    if target.name == item.name or target.kind == "date":
        return None

    def outcome(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
        if target.categorical:
            text = labels(column_of(frame, target.name))
            positive = (
                "True" if "True" in target.levels else target.levels[-1] if target.levels else ""
            )
            return np.where(
                text.isna(), np.nan, (text == positive).astype(float)
            ), f"Share of '{positive}'"
        return as_numbers(column_of(frame, target.name), target), f"Mean of {target.name}"

    def groups(frame: pd.DataFrame) -> np.ndarray:
        if item.categorical:
            return labels(column_of(frame, item.name)).fillna("(empty)").to_numpy()
        values = as_numbers(column_of(frame, item.name), item)
        assert edges is not None
        bins = np.clip(
            np.searchsorted(edges[1:-1], np.nan_to_num(values), side="right"), 0, len(edges) - 2
        )
        return np.where(np.isnan(values), -1, bins)

    if target.categorical and len(target.levels) > 2 and "True" not in target.levels:
        return None
    (real_y, measure), (synthetic_y, _) = outcome(real), outcome(synthetic)
    real_g, synthetic_g = groups(real), groups(synthetic)
    if item.categorical:
        keys = list(pd.Series(real_g).value_counts().head(TOP_LEVELS).index)
        names = [str(k) for k in keys]
    else:
        assert edges is not None
        keys = list(range(len(edges) - 1))
        fmt = _format_number(item)
        names = [f"{fmt(edges[k])} - {fmt(edges[k + 1])}" for k in keys]

    def means(y: np.ndarray, g: np.ndarray) -> list[float | None]:
        out: list[float | None] = []
        for k in keys:
            chosen = y[(g == k) & ~np.isnan(y)]
            out.append(float(chosen.mean()) if chosen.size else None)
        return out

    return TargetOverlay(
        target.name, measure, names, means(real_y, real_g), means(synthetic_y, synthetic_g)
    )


def column_detail(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    item: Learned,
    score: float | None,
    bins: int | None = None,
    target: Learned | None = None,
) -> ColumnDetail:
    r, s = column_of(real, item.name), column_of(synthetic, item.name)
    detail = ColumnDetail(item.name, item.type, item.kind, score)
    if item.categorical:
        rl, sl = labels(r), labels(s)
        detail.counts = _counts(rl, sl, None)
        order = list(rl.value_counts().index)
        invented = [v for v in sl.value_counts().index if v not in set(order)]
        everything = _value_rows(rl.dropna(), sl.dropna(), order + invented, str)
        shown = everything[:TOP_LEVELS]
        rest = everything[TOP_LEVELS:]
        if rest:
            real_n = sum(x.real for x in rest)
            synthetic_n = sum(x.synthetic for x in rest)
            real_share = sum(x.real_share for x in rest)
            synthetic_share = sum(x.synthetic_share for x in rest)
            shown.append(
                ValueRow(
                    OTHERS,
                    real_n,
                    real_share,
                    synthetic_n,
                    synthetic_share,
                    synthetic_share - real_share,
                )
            )
        detail.levels = shown
        if target is not None:
            detail.target = _overlay(real, synthetic, item, target, None)
        return detail
    rn, sn = as_numbers(r, item), as_numbers(s, item)
    detail.counts = _counts(r, s, (rn, sn))
    edges = _edges(rn, bins)
    detail.histogram = SharedHistogram(edges.tolist(), _shares(rn, edges), _shares(sn, edges))
    levels = np.linspace(0, 1, 101)
    rp, sp = rn[~np.isnan(rn)], sn[~np.isnan(sn)]
    if rp.size and sp.size:
        detail.ecdf = Ecdf(
            np.quantile(rp, levels).tolist(), np.quantile(sp, levels).tolist(), levels.tolist()
        )
    detail.stats = _stats(rn, sn)
    fmt = _format_number(item)
    real_values, synthetic_values = pd.Series(rp), pd.Series(sp)
    detail.frequent_real = _value_rows(
        real_values, synthetic_values, list(real_values.value_counts().head(TOP_VALUES).index), fmt
    )
    detail.frequent_synthetic = _value_rows(
        real_values,
        synthetic_values,
        list(synthetic_values.value_counts().head(TOP_VALUES).index),
        fmt,
    )
    union = np.unique(np.concatenate([rp, sp]))
    detail.smallest = _value_rows(real_values, synthetic_values, union[:TOP_VALUES].tolist(), fmt)
    detail.largest = _value_rows(
        real_values, synthetic_values, union[::-1][:TOP_VALUES].tolist(), fmt
    )
    if target is not None:
        overlay_edges = _edges(rn, 10)
        detail.target = _overlay(real, synthetic, item, target, overlay_edges)
    return detail


# --- Links between columns -------------------------------------------------------------------


@dataclass
class LinkPair:
    a: str
    b: str
    measure: Measure
    real: float | None
    synthetic: float | None
    gap: float | None


@dataclass
class Links:
    columns: list[str]
    real: list[list[float | None]]
    synthetic: list[list[float | None]]
    gap: list[list[float | None]]
    measures: list[list[str]]
    worst: list[LinkPair] = field(default_factory=list)  # the 5 pairs that changed most


def links(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    learned: list[Learned],
    method: Literal["pearson", "spearman"] = "pearson",
) -> Links:
    coded = {}
    for item in learned:
        if item.categorical:
            rc, sc = shared_codes(column_of(real, item.name), column_of(synthetic, item.name))
            joined = sorted(
                set(labels(column_of(real, item.name)).fillna(MISSING))
                | set(labels(column_of(synthetic, item.name)).fillna(MISSING))
            )
            empty = joined.index(MISSING) if MISSING in joined else -2
            coded[item.name] = (
                np.where(rc == empty, np.nan, rc).astype(float),
                np.where(sc == empty, np.nan, sc).astype(float),
            )
        else:
            coded[item.name] = (
                as_numbers(column_of(real, item.name), item),
                as_numbers(column_of(synthetic, item.name), item),
            )
    size = len(learned)
    names = [i.name for i in learned]
    r_matrix: list[list[float | None]] = [[None] * size for _ in range(size)]
    s_matrix: list[list[float | None]] = [[None] * size for _ in range(size)]
    g_matrix: list[list[float | None]] = [[None] * size for _ in range(size)]
    m_matrix = [[""] * size for _ in range(size)]
    pairs = []
    for i, a in enumerate(learned):
        r_matrix[i][i] = s_matrix[i][i] = 1.0
        g_matrix[i][i] = 0.0
        for j in range(i + 1, size):
            b = learned[j]
            rv, measure = association(
                coded[a.name][0], a.categorical, coded[b.name][0], b.categorical, method
            )
            sv, _ = association(
                coded[a.name][1], a.categorical, coded[b.name][1], b.categorical, method
            )
            gap = abs(sv - rv) if rv is not None and sv is not None else None
            r_matrix[i][j] = r_matrix[j][i] = rv
            s_matrix[i][j] = s_matrix[j][i] = sv
            g_matrix[i][j] = g_matrix[j][i] = gap
            m_matrix[i][j] = m_matrix[j][i] = measure
            pairs.append(LinkPair(a.name, b.name, measure, rv, sv, gap))
    worst = sorted((p for p in pairs if p.gap is not None), key=lambda p: -(p.gap or 0))[:5]
    return Links(names, r_matrix, s_matrix, g_matrix, m_matrix, worst)


@dataclass
class PairBox:
    group: str
    real: list[float | None]  # whisker low, Q1, median, Q3, whisker high
    synthetic: list[float | None]


@dataclass
class PairDetail:
    a: str
    b: str
    kind: Literal["numbers", "mixed", "categories"]
    real_points: list[list[float]] = field(default_factory=list)  # [x, y]
    synthetic_points: list[list[float]] = field(default_factory=list)
    boxes: list[PairBox] = field(default_factory=list)
    rows: list[str] = field(default_factory=list)  # crossed table levels
    columns: list[str] = field(default_factory=list)
    real_table: list[list[float]] = field(default_factory=list)  # share of all rows per cell
    synthetic_table: list[list[float]] = field(default_factory=list)


def _box(values: np.ndarray) -> list[float | None]:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return [None] * 5
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    low = values[values >= q1 - 1.5 * (q3 - q1)].min()
    high = values[values <= q3 + 1.5 * (q3 - q1)].max()
    return [float(low), float(q1), float(median), float(q3), float(high)]


def pair_detail(
    real: pd.DataFrame, synthetic: pd.DataFrame, a: Learned, b: Learned, seed: int = 42
) -> PairDetail:
    if not a.categorical and not b.categorical:
        detail = PairDetail(a.name, b.name, "numbers")
        for frame, target in ((real, detail.real_points), (synthetic, detail.synthetic_points)):
            x, y = as_numbers(column_of(frame, a.name), a), as_numbers(column_of(frame, b.name), b)
            keep = np.flatnonzero(~np.isnan(x) & ~np.isnan(y))
            pick = np.random.default_rng(seed).permutation(keep)[:SCATTER_POINTS]
            target.extend([[float(x[k]), float(y[k])] for k in pick])
        return detail
    if a.categorical and b.categorical:
        detail = PairDetail(a.name, b.name, "categories")
        detail.rows = list(labels(column_of(real, a.name)).value_counts().head(8).index)
        detail.columns = list(labels(column_of(real, b.name)).value_counts().head(8).index)
        for frame, table in ((real, detail.real_table), (synthetic, detail.synthetic_table)):
            crossed = pd.crosstab(
                labels(column_of(frame, a.name)), labels(column_of(frame, b.name))
            ) / max(len(frame), 1)
            crossed = crossed.reindex(index=detail.rows, columns=detail.columns, fill_value=0)
            table.extend(crossed.to_numpy(dtype=float).tolist())
        return detail
    category, number = (a, b) if a.categorical else (b, a)
    detail = PairDetail(category.name, number.name, "mixed")
    groups = list(labels(column_of(real, category.name)).value_counts().head(10).index)
    real_groups, synthetic_groups = (
        labels(column_of(real, category.name)),
        labels(column_of(synthetic, category.name)),
    )
    real_values, synthetic_values = (
        as_numbers(column_of(real, number.name), number),
        as_numbers(column_of(synthetic, number.name), number),
    )
    for group in groups:
        detail.boxes.append(
            PairBox(
                str(group),
                _box(real_values[(real_groups == group).to_numpy()]),
                _box(synthetic_values[(synthetic_groups == group).to_numpy()]),
            )
        )
    return detail


# --- Overview --------------------------------------------------------------------------------


@dataclass
class MeanSpread:
    column: str
    real_mean: float
    real_std: float
    synthetic_mean: float
    synthetic_std: float


@dataclass
class Projection:
    real: list[list[float]]
    synthetic: list[list[float]]
    explained: list[float]


@dataclass
class SyntheticOverview:
    mean_spread: list[MeanSpread] = field(default_factory=list)
    pca: Projection | None = None


def overview(
    real: pd.DataFrame, synthetic: pd.DataFrame, learned: list[Learned], seed: int = 42
) -> SyntheticOverview:
    numeric = [i for i in learned if not i.categorical]
    found = SyntheticOverview()
    columns_real, columns_synthetic = [], []
    for item in numeric:
        rn, sn = (
            as_numbers(column_of(real, item.name), item),
            as_numbers(column_of(synthetic, item.name), item),
        )
        if np.isnan(rn).all() or np.isnan(sn).all():
            continue
        found.mean_spread.append(
            MeanSpread(
                item.name,
                float(np.nanmean(rn)),
                float(np.nanstd(rn)),
                float(np.nanmean(sn)),
                float(np.nanstd(sn)),
            )
        )
        columns_real.append(rn)
        columns_synthetic.append(sn)
    if len(columns_real) >= 2:
        r, s = np.column_stack(columns_real), np.column_stack(columns_synthetic)
        mean, std = np.nanmean(r, axis=0), np.nanstd(r, axis=0)
        std[std == 0] = 1
        r, s = np.nan_to_num((r - mean) / std), np.nan_to_num((s - mean) / std)
        rng = np.random.default_rng(seed)
        r = r[rng.permutation(len(r))[:2_000]]
        s = s[rng.permutation(len(s))[:2_000]]
        pca = PCA(n_components=2, random_state=seed).fit(r)
        found.pca = Projection(
            np.round(pca.transform(r), 4).tolist(),
            np.round(pca.transform(s), 4).tolist(),
            [float(v) for v in pca.explained_variance_ratio_],
        )
    return found
