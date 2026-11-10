"""Unusual rows across columns (033): a score per row, a threshold where the sorted scores break,
and for each unusual row the columns that make it so, in a sentence.

Every score grows with how unusual a row is. A column's contribution is how much the score falls
when the row's value is replaced by the column's median (numbers, dates) or mode (categories) —
within the row's segment for the "distance to its segment" method."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from quantilo.ml.data import NUMERIC
from quantilo.ml.preparation import MISSING, Decision
from quantilo.segments.portrait import number, percent

UnusualMethod = Literal["isolation_forest", "mahalanobis", "lof", "segment"]
METHODS: dict[str, str] = {
    "isolation_forest": "Isolation Forest",
    "mahalanobis": "Robust Mahalanobis distance",
    "lof": "Local Outlier Factor",
    "segment": "Distance to its segment",
}
TREES = 200
TREE_ROWS = 2_048
MCD_ROWS = 20_000
LOF_ROWS = 50_000
LOF_NEIGHBOURS = 20
MCD_COLUMNS = 30
MCD_MIN_NUMBERS = 3  # among categories, the distance reads the numbers alone from 3 of them
ELLIPTICAL_SKEW = 1.0
LIST_ROWS = 500  # rows listed (with their reasons), most unusual first
TOP_SHARE = 0.01  # without a clear break, the 1 % most unusual rows
REASONS = 3
RARE_COMBINATION = 0.01
BATCH_ROWS = 20_000  # prepared rows scored in one call by the reasons


@dataclass
class UnusualReason:
    column: str
    contribution: float  # how much the score falls when the value becomes typical
    text: str  # "price is in the top 0.2 % (412,000)"


@dataclass
class UnusualRow:
    row: int  # row number in the file
    key: str  # what a mark is kept under: the identifier, or the row's fingerprint
    rank: int  # 1 = the most unusual
    score: float
    reasons: list[UnusualReason]
    combination: str | None  # "orders = 0 while revenue = 3,012: this combination appears in …"
    values: dict[str, Any]  # every column of the row
    segment: str | None = None  # "distance to its segment": the row's segment


@dataclass
class UnusualThreshold:
    value: float
    rows: int  # rows at or above it
    rule: Literal["break", "top_share"]
    why: str


@dataclass
class UnusualCurve:
    """Scores sorted from the most unusual row down: every listed rank, then a sampled tail."""

    ranks: list[int]
    scores: list[float]


Score = Callable[[np.ndarray, np.ndarray | None], np.ndarray]  # prepared rows, segments → score


@dataclass
class Scorer:
    method: UnusualMethod
    score: Score  # for any rows
    fit_scores: np.ndarray  # the fit rows' own scores


def elliptical(x: np.ndarray) -> bool:
    """Roughly elliptical: every prepared column has |skewness| ≤ ELLIPTICAL_SKEW."""
    centred = x - x.mean(axis=0)
    spread = x.std(axis=0)
    spread[spread == 0] = 1
    skew = np.abs(((centred / spread) ** 3).mean(axis=0))
    return bool(np.all(skew <= ELLIPTICAL_SKEW))


def auto_method(used: list[Decision], x: np.ndarray) -> tuple[UnusualMethod, str]:
    numbers = all(d.type in NUMERIC for d in used)
    if numbers and len(used) <= MCD_COLUMNS and elliptical(x):
        return "mahalanobis", (
            "Robust Mahalanobis distance: every column is a number, at most 30 of them, "
            "roughly elliptical once prepared."
        )
    if not numbers:
        why = "categories among the columns"
    elif len(used) > MCD_COLUMNS:
        why = f"{len(used)} columns (more than {MCD_COLUMNS})"
    else:
        why = "skewed columns even once prepared"
    return "isolation_forest", f"Isolation Forest (200 trees): {why}."


def numbers_only(used: list[Decision]) -> list[str]:
    """R-UNU-MCD-1 (056): the robust Mahalanobis distance asked for on numbers mixed with other
    columns reads the numbers alone when there are at least MCD_MIN_NUMBERS of them. The other
    columns are left out with their reason; returns their names (none when nothing changes)."""
    numbers = [d for d in used if d.type in NUMERIC]
    if len(numbers) == len(used) or len(numbers) < MCD_MIN_NUMBERS:
        return []
    left = []
    for decision in used:
        if decision.type in NUMERIC:
            continue
        decision.use = False
        decision.apply(
            "R-UNU-MCD-1",
            f"Left out: the robust Mahalanobis distance reads numbers only "
            f"({len(numbers)} numbers kept).",
        )
        left.append(decision.column)
    return left


def _sample(n: int, limit: int, seed: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))


def isolation_forest(x: np.ndarray, seed: int) -> Scorer:
    """Trees cut along the axes, so a row normal on each column but off a correlation would slip
    through: they also see the principal components, which follow the correlations. Each tree
    learns on TREE_ROWS rows (scikit-learn's 256 leave such a row out of most trees)."""
    rotation = PCA(random_state=seed).fit(x) if x.shape[1] > 1 else None

    def seen(rows: np.ndarray) -> np.ndarray:
        return rows if rotation is None else np.hstack([rows, rotation.transform(rows)])

    model = IsolationForest(
        n_estimators=TREES,
        max_samples=min(len(x), TREE_ROWS),  # type: ignore[arg-type]
        random_state=seed,
    ).fit(seen(x))

    def score(rows: np.ndarray, _groups: np.ndarray | None) -> np.ndarray:
        return -np.asarray(model.score_samples(seen(rows)))

    return Scorer("isolation_forest", score, score(x, None))


def mahalanobis(x: np.ndarray, seed: int) -> Scorer:
    model = MinCovDet(random_state=seed).fit(x[_sample(len(x), MCD_ROWS, seed)])

    def score(rows: np.ndarray, _groups: np.ndarray | None) -> np.ndarray:
        return np.sqrt(np.maximum(np.asarray(model.mahalanobis(rows)), 0))

    return Scorer("mahalanobis", score, score(x, None))


def local_outlier_factor(x: np.ndarray, seed: int) -> Scorer:
    index = _sample(len(x), LOF_ROWS, seed)
    neighbours = min(LOF_NEIGHBOURS, max(1, len(index) - 1))
    model = LocalOutlierFactor(n_neighbors=neighbours, novelty=True).fit(x[index])

    def score(rows: np.ndarray, _groups: np.ndarray | None) -> np.ndarray:
        return -np.asarray(model.score_samples(rows))

    own = score(x, None)
    own[index] = -np.asarray(model.negative_outlier_factor_)  # without counting itself
    return Scorer("lof", score, own)


def segment_distance(x: np.ndarray, groups: np.ndarray) -> Scorer:
    """Distance to the centre of the row's own segment, over that segment's median distance;
    rows not grouped are measured from the nearest centre."""
    ids = np.unique(groups[groups >= 0])
    if len(ids) == 0:
        raise ValueError("The segmentation has no segment.")
    centres = {int(g): x[groups == g].mean(axis=0) for g in ids}

    def raw(rows: np.ndarray, own: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gaps = np.column_stack([np.sqrt(((rows - centres[int(g)]) ** 2).sum(axis=1)) for g in ids])
        mine = np.array([np.searchsorted(ids, g) if g >= 0 else -1 for g in own])
        nearest = gaps.argmin(axis=1)
        pick = np.where(mine >= 0, mine, nearest)
        return gaps[np.arange(len(rows)), pick], ids[pick]

    distances, fit_groups = raw(x, groups)
    scales = {}
    for g in ids:
        median = float(np.median(distances[fit_groups == g])) if (fit_groups == g).any() else 0.0
        scales[int(g)] = median if median > 0 else 1.0

    def score(rows: np.ndarray, own: np.ndarray | None) -> np.ndarray:
        assert own is not None
        found, used = raw(rows, own)
        return found / np.array([scales[int(g)] for g in used])

    return Scorer("segment", score, score(x, groups))


def fit(method: UnusualMethod, x: np.ndarray, groups: np.ndarray | None, seed: int) -> Scorer:
    if method == "mahalanobis":
        return mahalanobis(x, seed)
    if method == "lof":
        return local_outlier_factor(x, seed)
    if method == "segment":
        assert groups is not None
        return segment_distance(x, groups)
    return isolation_forest(x, seed)


# --- Threshold -----------------------------------------------------------------------------------


def propose(scores: np.ndarray) -> UnusualThreshold:
    """At the break of the sorted scores (steep, then flat) within the 10 % most unusual rows;
    without a clear break, the 1 % most unusual."""
    ordered = np.sort(np.asarray(scores, dtype=float))[::-1]
    n = len(ordered)
    window = min(n, max(20, n // 10))
    top = ordered[:window]
    if window >= 5 and top[0] > top[-1]:
        x = np.arange(window) / (window - 1)
        y = (top - top[-1]) / (top[0] - top[-1])
        gap = (1 - x) - y
        knee = int(np.argmax(gap))
        steep = (top[0] - top[knee]) / max(knee, 1)
        flat = (top[knee] - top[-1]) / max(window - 1 - knee, 1)
        if knee >= 1 and gap[knee] >= 0.3 and steep >= 3 * max(flat, 1e-12):
            value = float(top[knee - 1])
            rows = int((ordered >= value).sum())
            return UnusualThreshold(
                value,
                rows,
                "break",
                f"The scores drop sharply over the {rows:,} most unusual "
                f"row{'s' if rows > 1 else ''}, then level off.",
            )
    rows = max(1, math.ceil(TOP_SHARE * n))
    value = float(ordered[rows - 1]) if n else 0.0
    return UnusualThreshold(
        value,
        int((ordered >= value).sum()),
        "top_share",
        f"No clear break in the scores: the {percent(TOP_SHARE)} most unusual rows.",
    )


def curve(scores: np.ndarray, listed: int = LIST_ROWS, tail: int = 200) -> UnusualCurve:
    ordered = np.sort(np.asarray(scores, dtype=float))[::-1]
    n = len(ordered)
    exact = list(range(1, min(n, listed) + 1))
    sampled = (
        np.unique(np.geomspace(min(n, listed) + 1, n, tail).astype(int)).tolist()
        if n > listed
        else []
    )
    ranks = [*exact, *(r for r in sampled if r > len(exact))]
    return UnusualCurve(ranks, [round(float(ordered[r - 1]), 6) for r in ranks])


# --- Reasons -------------------------------------------------------------------------------------


def _labels(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING).astype(str)


def typical(frame: pd.DataFrame, decision: Decision) -> Any:
    """The value a column's rows have most typically: median (numbers, dates) or mode."""
    values = cast(pd.Series, frame[decision.column]).dropna()
    if values.empty:
        return None
    if decision.type in NUMERIC or decision.type == "datetime":
        return values.quantile(0.5, interpolation="nearest")
    return _labels(values).value_counts().index[0]


def display(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "empty"
    if isinstance(value, float | int | np.number) and not isinstance(value, bool):
        return number(float(value))
    if isinstance(value, pd.Timestamp):
        return value.isoformat().replace("T00:00:00", "")
    return str(value)


def _side(values: pd.Series, value: Any) -> pd.Series:
    """Rows as extreme as `value` on its side of the median (numbers), or equal (categories)."""
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_datetime64_any_dtype(values):
        if value is None or pd.isna(value):
            return values.isna()
        median = values.median()
        return values <= value if value <= median else values >= value
    return _labels(values) == ("Missing" if value is None or pd.isna(value) else str(value))


def _sentence(column: str, value: Any, reference: pd.Series, suffix: str) -> str:
    present = reference.dropna()
    numeric = pd.api.types.is_numeric_dtype(reference) or pd.api.types.is_datetime64_any_dtype(
        reference
    )
    if value is None or pd.isna(value):
        share = float(reference.isna().mean()) if len(reference) else 0.0
        return f"{column} is empty, as in {percent(share)} of rows{suffix}"
    if numeric and len(present):
        median = present.median()
        if value >= median:
            share = float((present >= value).mean())
            where = "top"
        else:
            share = float((present <= value).mean())
            where = "bottom"
        if share <= 0.1:
            return f"{column} is in the {where} {percent(share)}{suffix} ({display(value)})"
        return f"{column} = {display(value)} (typical{suffix}: {display(median)})"
    share = float((_labels(reference) == str(value)).mean()) if len(reference) else 0.0
    if share < 0.05:
        return f"{column} = {value} is rare{suffix} ({percent(share)} of rows)"
    return f"{column} = {value} ({percent(share)} of rows{suffix})"


def reasons(
    top: pd.DataFrame,
    scores: np.ndarray,
    groups: np.ndarray | None,
    used: list[Decision],
    reference: pd.DataFrame,
    reference_groups: np.ndarray | None,
    names: dict[int, str],
    transform: Callable[[pd.DataFrame], np.ndarray],
    score: Score,
    one_column: Callable[[pd.DataFrame, np.ndarray, str], np.ndarray | None] | None = None,
) -> list[tuple[list[UnusualReason], str | None]]:
    """For each row of `top`, the REASONS columns whose typical value lowers its score most.
    `one_column(changed, base, column)` prepares rows that differ from the prepared `base` in
    one column only, faster than `transform` (None: not possible)."""
    contributions = np.zeros((len(top), len(used)))
    base = transform(top) if one_column is not None else None
    batch: list[tuple[int, np.ndarray]] = []

    def flush() -> None:
        # one call to the scorer for several columns: a forest's per-call cost dominates
        stacked = np.vstack([x for _, x in batch])
        tiled = np.tile(groups, len(batch)) if groups is not None else None
        found = score(stacked, tiled).reshape(len(batch), len(top))
        for (j, _), changed_scores in zip(batch, found, strict=True):
            contributions[:, j] = scores - changed_scores
        batch.clear()

    for j, decision in enumerate(used):
        changed = top.copy()
        if groups is not None and reference_groups is not None:
            for g in np.unique(groups):
                inside = reference.loc[reference_groups == g]
                value = typical(inside if len(inside) else reference, decision)
                changed.loc[groups == g, decision.column] = value
        else:
            changed[decision.column] = typical(reference, decision)
        prepared = (
            one_column(changed, base, decision.column)
            if one_column is not None and base is not None
            else None
        )
        batch.append((j, prepared if prepared is not None else transform(changed)))
        if len(batch) * len(top) >= BATCH_ROWS:
            flush()
    if batch:
        flush()
    found = []
    for i in range(len(top)):
        order = [j for j in np.argsort(-contributions[i]) if contributions[i, j] > 0][:REASONS]
        group = int(groups[i]) if groups is not None else None
        if group is not None and reference_groups is not None:
            rows = reference.loc[reference_groups == group]
            suffix = f" for segment = {names.get(group, str(group))}"
        else:
            rows, suffix = reference, ""
        rows = rows if len(rows) else reference
        items = []
        for j in order:
            column = used[j].column
            value = top[column].iloc[i]
            text = _sentence(column, value, cast(pd.Series, rows[column]), suffix)
            items.append(UnusualReason(column, round(float(contributions[i, j]), 6), text))
        found.append((items, _combination(top.iloc[i], items, rows, suffix)))
    return found


def _combination(
    row: pd.Series, items: list[UnusualReason], rows: pd.DataFrame, suffix: str
) -> str | None:
    """When the two main columns' values are rare together (under RARE_COMBINATION of rows)."""
    if len(items) < 2 or rows.empty:
        return None
    a, b = items[0].column, items[1].column
    first = _side(cast(pd.Series, rows[a]), row[a])
    second = _side(cast(pd.Series, rows[b]), row[b])
    joint = float((first & second).mean())
    if joint >= RARE_COMBINATION:
        return None
    return (
        f"{a} = {display(row[a])} while {b} = {display(row[b])}: this combination appears in "
        f"{percent(joint)} of rows{suffix}"
    )
