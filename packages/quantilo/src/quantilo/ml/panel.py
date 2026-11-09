"""Panel data (035): the same entities seen at successive periods (ids by eras, shops by weeks,
customers by months). Rows are not independent: an entity comes back every period and the rows of
one period share its context, so a shuffled split scores too well. Here: recognising a panel,
walk-forward folds made of whole periods with an embargo, the score of every period and their
summary, the leaks proper to a panel, and how much the predictions follow single columns
(exposure) with the way to remove it (neutralization)."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, cast

import numpy as np
import pandas as pd

from quantilo.ml import evaluate
from quantilo.variable.source import clean

PERIOD_RANK = "__period"  # the period's rank among every period of the data (set by the loader)
MIN_PERIODS = 10  # a column with fewer distinct values is not taken for a period
MIN_ROWS_PER_PERIOD = 5  # each period comes back on several rows
MANY_PERIODS = 200  # beyond: train on 1 period in 4 by default
DEFAULT_STEP = 4
MAX_LAG = 20  # periods looked at to find how far ahead the target looks
LAG_CORRELATION = 0.05  # an entity's target "still overlaps" a later one above this…
EXPOSURE_CELLS = 20_000_000  # rows x columns read for exposures (whole periods beyond)
TOP_EXPOSURES = 5
FUTURE_CORRELATION = 0.5  # a column follows the next period's target at least this much
FUTURE_MARGIN = 0.2  # … and this much more than the current target
LATE_SHARE = 0.8  # a column that only starts in the last 20 % of the periods
SUSPICIOUS_SKILL = 0.5
SUSPICIOUS_GAP = 0.3
PERIOD_HINTS = ("era", "period", "date", "month", "week", "year", "quarter", "time", "day")
ENTITY_HINTS = ("id", "ticker", "symbol", "entity", "customer", "client", "store", "shop", "user")
# a period counts as good above these (correlations above 0, an AUC above chance)
NATURAL_THRESHOLDS = {"pearson": 0.0, "spearman": 0.0, "r2": 0.0, "mcc": 0.0, "kappa": 0.0}
NATURAL_THRESHOLDS["roc_auc"] = 0.5

Criterion = Literal["mean", "sharpe", "worst"]


# --- Periods in order ----------------------------------------------------------------------------


def _label(value: object) -> str:
    if isinstance(value, datetime):
        moment = cast(pd.Timestamp, pd.Timestamp(value))
        return moment.date().isoformat() if moment == moment.normalize() else moment.isoformat(" ")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _sort_key(values: pd.Series) -> pd.Series:
    """Numbers (and digit strings such as "0001") by value, dates in time, other labels in
    alphabetical order."""
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_datetime64_any_dtype(values):
        return values
    numbers = cast(pd.Series, pd.to_numeric(values, errors="coerce"))
    if int(numbers.notna().sum()) == int(values.notna().sum()):
        return numbers
    return cast(pd.Series, values.astype(str).where(values.notna()))


def period_ranks(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, dict[int, str]]:
    """Each row's period rank (-1 without a period) and each rank's label. The loader's ranks
    (`__period`, over every period of the data) are used when present, so that an embargo counts
    real periods even when only some periods were read."""
    values = cast(pd.Series, frame[column])
    if PERIOD_RANK in frame.columns:
        found = cast(pd.Series, pd.to_numeric(cast(pd.Series, frame[PERIOD_RANK]), errors="coerce"))
        ranks = found.where(values.notna()).fillna(-1).to_numpy(dtype=int)
    else:
        key = _sort_key(values)
        distinct = pd.Index(key.dropna().unique()).sort_values()
        ranks = np.asarray(distinct.get_indexer(key), dtype=int)
    first = pd.DataFrame({"rank": ranks, "value": values.tolist()}).drop_duplicates("rank")
    labels = {int(r): _label(v) for r, v in zip(first["rank"], first["value"], strict=True)}
    labels.pop(-1, None)
    return ranks, labels


def _chunks(ranks: np.ndarray) -> list[np.ndarray]:
    """Row positions grouped by period, periods in order."""
    order = np.argsort(ranks, kind="stable")
    cuts = np.flatnonzero(np.diff(ranks[order])) + 1
    return [chunk for chunk in np.split(order, cuts) if len(chunk)]


# --- Walk-forward folds --------------------------------------------------------------------------


@dataclass
class FoldSpan:
    """One fold on the time axis: what it trained on, the gap left out, what it was tested on."""

    fold: int
    train_start: int  # period ranks
    train_end: int
    test_start: int
    test_end: int
    train_periods: int  # periods trained on (after sampling)
    test_periods: int
    train_rows: int
    test_rows: int
    train_from: str  # period labels
    train_to: str
    test_from: str
    test_to: str


@dataclass
class Plan:
    folds: list[tuple[np.ndarray, np.ndarray]]  # training and test positions of every fold
    spans: list[FoldSpan]
    final: np.ndarray  # positions the saved model is trained on
    step: int  # trained on 1 period in `step`


def _every(periods: np.ndarray, step: int) -> np.ndarray:
    """One period in `step`, counting back from the most recent: whole periods or none."""
    return periods[::-1][::step][::-1]


def plan(
    ranks: np.ndarray,
    labels: dict[int, str],
    folds: int,
    embargo: int,
    window: str = "expanding",
    window_periods: int | None = None,
    sample_periods: int | None = None,
) -> Plan:
    """Periods sorted; the last ones cut into `folds` test blocks. Each fold trains on the periods
    before its block, minus the `embargo` periods just before it (all of them: expanding window;
    the last N: rolling window), then is tested on the block."""
    present = np.unique(ranks[ranks >= 0])
    m = len(present)
    if m < folds + 1:
        raise evaluate.ValidationError(
            f"Panel walk-forward with {folds} folds needs at least {folds + 1} periods "
            f"({m} here): use fewer folds."
        )
    step = sample_periods or (DEFAULT_STEP if m > MANY_PERIODS else 1)
    block = m // (folds + 1)
    starts = [m - (folds - i) * block for i in range(folds)]
    first_window = int((present < present[starts[0]] - embargo).sum())
    length = window_periods or first_window
    found: list[tuple[np.ndarray, np.ndarray]] = []
    spans: list[FoldSpan] = []
    for i, start in enumerate(starts):
        tested = present[start : starts[i + 1] if i + 1 < folds else m]
        before = present[present < tested[0] - embargo]
        if window == "rolling":
            before = before[-length:]
        trained = _every(before, step)
        if not len(trained):
            raise evaluate.ValidationError(
                f"Fold {i + 1} has no period left to train on with an embargo of {embargo}: "
                "use fewer folds or a shorter embargo."
            )
        fit_at = np.flatnonzero(np.isin(ranks, trained))
        held_at = np.flatnonzero(np.isin(ranks, tested))
        found.append((fit_at, held_at))
        spans.append(
            FoldSpan(
                i + 1,
                int(before[0]),
                int(before[-1]),
                int(tested[0]),
                int(tested[-1]),
                len(trained),
                len(tested),
                len(fit_at),
                len(held_at),
                labels.get(int(before[0]), str(before[0])),
                labels.get(int(before[-1]), str(before[-1])),
                labels.get(int(tested[0]), str(tested[0])),
                labels.get(int(tested[-1]), str(tested[-1])),
            )
        )
    final_periods = present[-length:] if window == "rolling" else present
    final = np.flatnonzero(np.isin(ranks, _every(final_periods, step)))
    return Plan(found, spans, final, step)


# --- Scores per period ---------------------------------------------------------------------------


@dataclass
class PeriodScore:
    period: str
    rank: int
    fold: int
    rows: int
    score: float | None  # the primary metric on this period's rows
    pearson: float | None = None  # numbers: correlation of predictions and actual values
    spearman: float | None = None  # … of their ranks


@dataclass
class PeriodSummary:
    """How regular the model is from one period to the next."""

    metric: str
    periods: int
    mean: float | None
    std: float | None
    sharpe: float | None  # mean / std: the "Sharpe ratio" of the scores
    threshold: float | None  # a good period scores above it (below for an error)
    share_good: float | None
    worst: float | None
    worst_period: str | None
    longest_bad_streak: int | None  # periods in a row on the wrong side of the threshold
    lower_is_better: bool


Scorer = Callable[[np.ndarray], dict[str, float | None]]


def regression_scorer(metric: str, actual: np.ndarray, predicted: np.ndarray) -> Scorer:
    def score(at: np.ndarray) -> dict[str, float | None]:
        a, p = actual[at], predicted[at]
        return {
            "score": evaluate.regression_metric(metric, a, p),
            "pearson": evaluate.regression_metric("pearson", a, p),
            "spearman": evaluate.regression_metric("spearman", a, p),
        }

    return score


def classification_scorer(
    metric: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    positive: str | None,
) -> Scorer:
    def score(at: np.ndarray) -> dict[str, float | None]:
        value = evaluate.classification_metric(
            metric, actual[at], predicted[at], probabilities[at], classes, positive
        )
        return {"score": value}

    return score


def score_periods(
    ranks: np.ndarray, folds: np.ndarray, labels: dict[int, str], score: Scorer
) -> list[PeriodScore]:
    found = []
    for at in _chunks(ranks):
        rank = int(ranks[at[0]])
        values = score(at) if len(at) >= 2 else {}
        found.append(
            PeriodScore(
                labels.get(rank, str(rank)),
                rank,
                int(folds[at[0]]),
                len(at),
                values.get("score"),
                values.get("pearson"),
                values.get("spearman"),
            )
        )
    return found


def summarise(
    periods: list[PeriodScore], metric: str, threshold: float | None = None
) -> PeriodSummary:
    lower = metric in evaluate.LOWER_IS_BETTER
    limit = threshold if threshold is not None else NATURAL_THRESHOLDS.get(metric)
    scored = [p for p in periods if p.score is not None]
    values = np.array([p.score for p in scored], dtype=float)
    if not len(values):
        return PeriodSummary(
            metric, len(periods), None, None, None, limit, None, None, None, None, lower
        )
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else None
    worst_at = int(values.argmax() if lower else values.argmin())
    share = streak = None
    if limit is not None:
        good = values < limit if lower else values > limit
        share = float(good.mean())
        streak, run = 0, 0
        for ok in good:
            run = 0 if ok else run + 1
            streak = max(streak, run)
    return PeriodSummary(
        metric,
        len(periods),
        clean(mean),
        clean(std) if std is not None else None,
        clean(mean / std) if std else None,
        limit,
        clean(share) if share is not None else None,
        clean(float(values[worst_at])),
        scored[worst_at].period,
        streak,
        lower,
    )


def criterion_value(summary: PeriodSummary, criterion: str) -> float | None:
    """The run's score: the mean of the periods, their mean / std, or the worst period."""
    if criterion == "sharpe":
        return summary.sharpe
    if criterion == "worst":
        return summary.worst
    return summary.mean


# --- Exposure and neutralization -----------------------------------------------------------------


@dataclass
class Exposure:
    column: str
    mean: float  # mean over the periods of the correlation of predictions and column (signed)


def exposures(scores: np.ndarray, features: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Correlation of `scores` with each column of `features`, period by period, averaged over
    the periods (empty values count as the period's mean; a constant column counts 0)."""
    total = np.zeros(features.shape[1])
    counted = np.zeros(features.shape[1])
    for at in _chunks(ranks):
        if len(at) < 3:
            continue
        s = scores[at] - scores[at].mean()
        if not s.any():
            continue
        x = features[at]
        x = np.nan_to_num(x - np.nanmean(x, axis=0, keepdims=True), nan=0.0)
        norms = np.sqrt((x * x).sum(axis=0)) * math.sqrt(float(s @ s))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.where(norms > 0, (x.T @ s) / norms, 0.0)
        total += corr
        counted += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counted > 0, total / np.maximum(counted, 1), 0.0)


def neutralize(
    scores: np.ndarray, features: np.ndarray, ranks: np.ndarray, proportion: float
) -> np.ndarray:
    """Remove from the predictions `proportion` of the part a linear fit on `features` explains,
    period by period (least squares), then give each period back its mean and spread so the
    scores stay on the same scale."""
    out = scores.astype(float).copy()
    for at in _chunks(ranks):
        s = scores[at].astype(float)
        x = features[at]
        x = np.nan_to_num(x - np.nanmean(x, axis=0, keepdims=True), nan=0.0)
        centred = s - s.mean()
        beta = np.linalg.lstsq(x, centred, rcond=None)[0]
        neutral = centred - proportion * (x @ beta)
        spread, before = float(neutral.std()), float(centred.std())
        if spread > 0:
            neutral = neutral / spread * before
        out[at] = neutral + s.mean()
    return out


def _thinned(ranks: np.ndarray, columns: int) -> np.ndarray:
    """Row positions of whole periods, evenly spread, within the cells read for exposures."""
    present = np.unique(ranks)
    rows_per_period = len(ranks) / max(len(present), 1)
    keep = max(1, int(EXPOSURE_CELLS / max(columns * rows_per_period, 1)))
    if keep >= len(present):
        return np.arange(len(ranks))
    chosen = present[np.linspace(0, len(present) - 1, keep).round().astype(int)]
    return np.flatnonzero(np.isin(ranks, chosen))


def top_exposures(
    scores: np.ndarray, frame: pd.DataFrame, columns: list[str], ranks: np.ndarray
) -> tuple[list[Exposure], float | None]:
    """The columns the predictions follow most, and the largest exposure."""
    if not columns:
        return [], None
    at = _thinned(ranks, len(columns))
    features = frame[columns].to_numpy(dtype=float)[at]
    found = exposures(scores[at], features, ranks[at])
    order = np.argsort(-np.abs(found))[:TOP_EXPOSURES]
    shown = [Exposure(columns[i], float(found[i])) for i in order]
    return shown, clean(float(np.abs(found).max())) if len(found) else None


# --- Leaks proper to a panel ---------------------------------------------------------------------


@dataclass
class PanelAlert:
    kind: Literal["future", "late", "suspicious", "embargo"]
    message: str
    column: str | None = None


def numeric_target(target: pd.Series) -> np.ndarray | None:
    numbers = cast(pd.Series, pd.to_numeric(target, errors="coerce"))
    if int(numbers.notna().sum()) == int(target.notna().sum()):
        return numbers.to_numpy(dtype=float)
    labels = target.astype(str)
    if labels.nunique() == 2:  # two classes: the rarer one is 1
        rare = str(labels.value_counts().index[-1])
        return (labels == rare).to_numpy(dtype=float)
    return None


def _plural(name: str) -> str:
    return name if name.endswith("s") else f"{name}s"


def future_columns(
    frame: pd.DataFrame,
    columns: list[str],
    target: np.ndarray,
    ranks: np.ndarray,
    entity: str | None,
    period: str,
    labels: dict[int, str],
) -> list[PanelAlert]:
    """Columns that seem to come from the future: much more linked to the entity's next target
    than to the current one, or only filled in the last periods."""
    alerts: list[PanelAlert] = []
    if not columns:
        return alerts
    data = cast(
        pd.DataFrame, frame[columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    )
    if entity is not None:
        keys = pd.DataFrame(
            {"entity": frame[entity].astype(str).to_numpy(), "rank": ranks, "y": target}
        )
        order = keys.sort_values(["entity", "rank"], kind="stable").index
        following = keys.loc[order].groupby("entity")["y"].shift(-1)
        upcoming = cast(pd.Series, pd.Series(following, index=order).reindex(keys.index))
        now = pd.Series(target)
        has_next = upcoming.notna().to_numpy()
        if int(has_next.sum()) >= 30:
            known = cast(pd.DataFrame, data[has_next])
            r_now = known.corrwith(cast(pd.Series, now[has_next])).abs()
            r_next = known.corrwith(cast(pd.Series, upcoming[has_next])).abs()
            for column in columns:
                ahead, current = float(r_next.get(column, 0) or 0), float(r_now.get(column, 0) or 0)
                if ahead >= FUTURE_CORRELATION and ahead >= current + FUTURE_MARGIN:
                    alerts.append(
                        PanelAlert(
                            "future",
                            f"{column} looks like a column from the future: it follows the "
                            f"next {period}'s target (r = {ahead:.2f}) far more than the current "
                            f"one (r = {current:.2f}).",
                            column,
                        )
                    )
    present = np.unique(ranks[ranks >= 0])
    if len(present) >= MIN_PERIODS:
        late = present[int(len(present) * LATE_SHARE)]
        for column in columns:
            filled = cast(pd.Series, data[column]).notna().to_numpy()
            if not filled.any() or filled.mean() > 0.5:
                continue
            start = int(ranks[filled].min())
            if start >= late:
                alerts.append(
                    PanelAlert(
                        "late",
                        f"{column} only exists from {period} {labels.get(start, start)} on, in "
                        "the last periods: it may have been filled after the facts.",
                        column,
                    )
                )
    return alerts


def baseline_score(
    target: np.ndarray, frame: pd.DataFrame, columns: list[str], ranks: np.ndarray
) -> np.ndarray | None:
    """What the 3 columns most linked to the target give alone: a least-squares fit on them."""
    if not columns:
        return None
    at = _thinned(ranks, len(columns))
    features = frame[columns].to_numpy(dtype=float)
    linked = np.abs(exposures(target[at], features[at], ranks[at]))
    best = np.argsort(-linked)[:3]
    x = features[:, best]
    x = np.nan_to_num((x - np.nanmean(x, axis=0)) / (np.nanstd(x, axis=0) + 1e-12), nan=0.0)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    return design @ beta


def _skill(metric: str, value: float) -> float:
    return 2 * value - 1 if metric == "roc_auc" else value


def suspicious(
    metric: str, model_mean: float | None, baseline_mean: float | None, period: str
) -> PanelAlert | None:
    """A mean score per period far above what the best 3 columns give alone."""
    if model_mean is None or baseline_mean is None:
        return None
    model, baseline = _skill(metric, model_mean), _skill(metric, baseline_mean)
    if model >= SUSPICIOUS_SKILL and model - baseline >= SUSPICIOUS_GAP:
        return PanelAlert(
            "suspicious",
            f"The mean score per {period} ({model_mean:.3f}) is far above what the 3 best "
            f"columns give alone ({baseline_mean:.3f}). Check that no column is computed from "
            "the target.",
        )
    return None


# --- How far ahead the target looks --------------------------------------------------------------


def horizon(
    entities: pd.Series, ranks: np.ndarray, target: np.ndarray, entity: str
) -> tuple[int, str] | None:
    """The embargo the target calls for. A target that looks h periods ahead (a 20-day return on
    weekly rows) overlaps the next h - 1 targets of the same entity: once each period's mean is
    removed, its correlation with itself stays significant up to lag h - 1, then falls. One that
    never falls is persistence (an entity effect), not a horizon: no suggestion then. The
    entity's mean is kept: over a short panel, removing it would bias every lag downwards."""
    data = pd.DataFrame(
        {"entity": entities.astype(str).to_numpy(), "rank": ranks, "y": target}
    ).dropna()
    data = cast(pd.DataFrame, data[data["rank"] >= 0]).drop_duplicates(subset=["entity", "rank"])
    if len(data) < 30:
        return None
    data["y"] = data["y"] - data.groupby("rank")["y"].transform("mean")
    last, r_last = 0, 0.0
    for lag in range(1, MAX_LAG + 1):
        later = data.assign(rank=data["rank"] - lag)
        pairs = data.merge(later, on=["entity", "rank"], suffixes=("", "_later"))
        if len(pairs) < 30:
            break
        r = float(np.corrcoef(pairs["y"], pairs["y_later"])[0, 1])
        # … and above chance for this many pairs
        if not math.isfinite(r) or r < max(LAG_CORRELATION, 3 / math.sqrt(len(pairs))):
            break
        last, r_last = lag, r
    else:
        return None
    if last == 0:
        return None
    ahead = last + 1
    return ahead, (
        f"The target of an {entity} stays correlated with its next ones for {last} "
        f"period{'s' if last > 1 else ''} (r = {r_last:.2f} at lag {last}): it likely looks "
        f"{ahead} periods ahead. An embargo of {ahead} keeps these overlapping targets out of "
        "training."
    )


# --- Recognising a panel -------------------------------------------------------------------------


@dataclass
class PanelCheck:
    """What the ML page says about the rows: a panel or not, its period and entity."""

    panel: bool
    period: str | None
    entity: str | None
    periods: int
    per_period: int  # rows per period, on average (≈ entities per period)
    entities: int | None  # distinct entities among the rows read
    duplicates: int  # rows read that repeat an (entity, period) pair
    embargo: int | None  # suggested: how far ahead the target looks, in periods
    embargo_reason: str | None
    period_candidates: list[str] = field(default_factory=list)
    entity_candidates: list[str] = field(default_factory=list)
    message: str | None = None


def _hinted(name: str, hints: tuple[str, ...]) -> bool:
    lowered = name.lower()
    words = lowered.replace("-", "_").replace(" ", "_").split("_")
    return any(h == lowered or h in words for h in hints)


def orderable(values: pd.Series, name: str) -> bool:
    """Periods have an order: numbers (whole ones), dates, digit strings, or a period's name."""
    present = values.dropna()
    if not len(present):
        return False
    if pd.api.types.is_datetime64_any_dtype(values):
        return True
    numbers = cast(pd.Series, pd.to_numeric(present, errors="coerce"))
    if numbers.notna().all():
        return bool((numbers == numbers.round()).all())
    return _hinted(name, PERIOD_HINTS)


def rank_period_candidates(names: list[str], types: dict[str, str]) -> list[str]:
    """Named like a period first, then dates, then the others."""
    return sorted(
        names,
        key=lambda n: (not _hinted(n, PERIOD_HINTS), types.get(n) != "datetime", names.index(n)),
    )


def rank_entity_candidates(names: list[str], types: dict[str, str]) -> list[str]:
    return sorted(
        names,
        key=lambda n: (not _hinted(n, ENTITY_HINTS), types.get(n) != "identifier", names.index(n)),
    )


def _entity_fit(frame: pd.DataFrame, period: str, entity: str) -> float | None:
    """Mean number of periods an entity is seen in, or None when the column is no entity (values
    repeated within a period, or never seen twice)."""
    pairs = frame[[period, entity]].dropna().astype(str)
    if len(pairs) < 10:
        return None
    unique = len(pairs.drop_duplicates())
    if unique / len(pairs) < 0.9:
        return None
    seen = unique / max(int(cast(pd.Series, pairs[entity]).nunique()), 1)
    return seen if seen >= 2 else None


def _entity(frame: pd.DataFrame, period: str, candidates: list[str]) -> str | None:
    fits = {c: fit for c in candidates if (fit := _entity_fit(frame, period, c)) is not None}
    return max(fits, key=lambda c: fits[c]) if fits else None


def recognise(
    frame: pd.DataFrame,
    period_candidates: list[str],
    entity_candidates: list[str],
    rows: int,
    distinct: dict[str, int],
    target: str | None = None,
    period: str | None = None,
    entity: str | None = None,
) -> PanelCheck:
    """Find the period (a column whose values come back on many rows, in an order) and the entity
    (a column unique within a period that comes back period after period), or check the ones the
    user declared."""
    declared = period is not None
    if period is not None:
        others = [c for c in entity_candidates if c != period]
        if entity is None:
            entity = _entity(frame, period, others)
    else:
        # an unnamed number column (an age, a count) is a period only when an entity comes back
        # at each of its values
        for candidate in period_candidates:
            if not orderable(cast(pd.Series, frame[candidate]), candidate):
                continue
            others = [c for c in entity_candidates if c != candidate]
            found = entity or _entity(frame, candidate, others)
            if found is not None or _hinted(candidate, PERIOD_HINTS):
                period, entity = candidate, found
                break
    if period is None:
        return PanelCheck(False, None, None, 0, 0, None, 0, None, None, period_candidates, [])
    others = [c for c in entity_candidates if c != period]
    ranks, _ = period_ranks(frame, period)
    periods = distinct.get(period) or len(np.unique(ranks[ranks >= 0]))
    per_period = round(rows / periods) if periods else 0
    duplicates = entities = 0
    if entity is not None:
        pairs = frame[[period, entity]].dropna().astype(str)
        duplicates = int(len(pairs) - len(pairs.drop_duplicates()))
        entities = int(cast(pd.Series, pairs[entity]).nunique())
    embargo = reason = None
    if entity is not None and target is not None and target in frame.columns:
        numbers = numeric_target(cast(pd.Series, frame[target]))
        if numbers is not None:
            found = horizon(cast(pd.Series, frame[entity]), ranks, numbers, entity)
            if found is not None:
                embargo, reason = found
    panel = declared or (periods >= MIN_PERIODS and per_period >= MIN_ROWS_PER_PERIOD)
    message = None
    if panel:
        what = (
            f"{periods:,} {_plural(period)} × ~{per_period:,} {_plural(entity)}"  # noqa: RUF001
            if entity
            else f"{periods:,} {_plural(period)} of ~{per_period:,} rows"
        )
        message = f"This looks like panel data: {what}. Random splits would overestimate the score."
    return PanelCheck(
        panel,
        period,
        entity,
        periods,
        per_period,
        entities if entity is not None else None,
        duplicates,
        embargo,
        reason,
        period_candidates,
        others,
        message,
    )


# --- The panel part of a run ---------------------------------------------------------------------


@dataclass
class PanelResult:
    period_column: str
    entity_column: str | None
    window: str
    embargo: int
    step: int  # trained on 1 period in `step`
    criterion: str
    value: float | None  # the criterion over the periods: the run's score
    periods: list[PeriodScore]
    summary: PeriodSummary
    folds: list[FoldSpan]
    exposures: list[Exposure]  # the columns the predictions follow most
    max_exposure: float | None
    exposure_columns: int  # numeric columns the exposure was measured on
    alerts: list[PanelAlert] = field(default_factory=list)
    suggested_embargo: int | None = None
    embargo_reason: str | None = None


@dataclass
class OutOfFold:
    """The held-out predictions of a panel run, kept beside it to neutralize them later."""

    rows: np.ndarray  # file row numbers
    ranks: np.ndarray
    folds: np.ndarray
    scores: np.ndarray  # numbers: predictions; two classes: probability of `positive`
    actual: np.ndarray
    positive: str | None = None
    negative: str | None = None
    threshold: float = 0.5


@dataclass
class Neutralized:
    columns: list[str]
    proportion: float
    value: float | None  # the criterion after neutralization
    periods: list[PeriodScore]
    summary: PeriodSummary
    before: list[Exposure]  # exposure to the chosen columns, before and after
    after: list[Exposure]


class PanelError(ValueError):
    pass


def neutralized(
    oof: OutOfFold,
    frame: pd.DataFrame,
    columns: list[str],
    proportion: float,
    metric: str,
    labels: dict[int, str],
    criterion: str,
    threshold: float | None = None,
) -> Neutralized:
    """The run's held-out predictions with their linear part on `columns` removed, and every
    period scored again. `frame`: those columns, one row per held-out prediction, in order."""
    if not columns:
        raise PanelError("Choose at least one column to neutralize on.")
    if not 0 <= proportion <= 1:
        raise PanelError("The proportion goes from 0 to 100 %.")
    features = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    scores = neutralize(oof.scores, features, oof.ranks, proportion)
    if oof.positive is None:
        scorer = regression_scorer(metric, oof.actual.astype(float), scores)
    else:
        negative, positive = str(oof.negative), oof.positive
        chances = np.clip(scores, 0.0, 1.0)
        predicted = np.where(chances >= oof.threshold, positive, negative)
        scorer = classification_scorer(
            metric,
            oof.actual.astype(str),
            predicted,
            np.column_stack([1 - chances, chances]),
            [negative, positive],
            positive,
        )
    periods = score_periods(oof.ranks, oof.folds, labels, scorer)
    summary = summarise(periods, metric, threshold)
    before = exposures(oof.scores, features, oof.ranks)
    after = exposures(scores, features, oof.ranks)
    return Neutralized(
        columns,
        proportion,
        criterion_value(summary, criterion),
        periods,
        summary,
        [Exposure(c, float(v)) for c, v in zip(columns, before, strict=True)],
        [Exposure(c, float(v)) for c, v in zip(columns, after, strict=True)],
    )
