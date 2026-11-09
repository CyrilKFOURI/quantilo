"""Sequences for TimeGAN (feature 023c): the automatic choices (rules R-SEQ-01 to 07), sliding
windows and their way back to rows, and the sequence measures of the TimeGAN paper.

A sequence is the history of one entity (a store, a customer) sorted by time; without an entity,
the whole data is one long series. Windows of a fixed length are cut from each history.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantilo.synthetic.encode import Coded, Learned, column_of, labels, seconds

MIN_LENGTH, MAX_LENGTH, DEFAULT_LENGTH = 8, 100, 24
MAX_WINDOWS = 20_000
MAX_SEQUENCES = 10_000
MAX_LEVELS = 20  # categories: the rest becomes one "other" slot
DUPLICATE_TIMES = 0.05  # an entity whose dates repeat more often than this is not a series
ENTITY_TYPES = {"categorical_nominal", "categorical_ordinal", "identifier", "numeric_discrete"}
STEPS = [(86_400.0, "daily"), (7 * 86_400.0, "weekly"), (3_600.0, "hourly"), (60.0, "every minute")]


@dataclass
class EntityCandidate:
    name: str
    sequences: int  # histories with at least MIN_LENGTH steps
    median_length: int
    reason: str


@dataclass
class SequencePlan:
    time: str | None
    entity: str | None
    candidates: list[EntityCandidate] = field(default_factory=list)
    step_seconds: float | None = None
    step: str = ""  # "daily", "weekly", "monthly" or a duration
    length: int = DEFAULT_LENGTH
    sequences: int = 1
    reason: str = ""
    error: str | None = None  # why TimeGAN cannot run on this data
    warning: str | None = None  # e.g. dates that repeat without an entity to tell histories apart


def step_label(step: float) -> str:
    for value, name in STEPS:
        if abs(step - value) <= 0.02 * value:
            return name
    if 28 * 86_400 <= step <= 31 * 86_400:
        return "monthly"
    if 365 * 86_400 <= step <= 366 * 86_400:
        return "yearly"
    if step >= 86_400:
        return f"every {step / 86_400:g} days"
    return f"every {step / 3_600:g} hours" if step >= 3_600 else f"every {step:g} s"


def _histories(frame: pd.DataFrame, time: str, entity: str | None) -> list[np.ndarray]:
    """Times of each history, sorted."""
    stamps = seconds(column_of(frame, time))
    if entity is None:
        present = stamps[~np.isnan(stamps)]
        return [np.sort(present)]
    keys = labels(column_of(frame, entity)).to_numpy()
    out = []
    for key in pd.unique(keys):
        if key is None:
            continue
        times = stamps[(keys == key) & ~np.isnan(stamps)]
        out.append(np.sort(times))
    return out


def _median_step(histories: list[np.ndarray]) -> float | None:
    gaps = np.concatenate([np.diff(h) for h in histories if len(h) > 1] or [np.zeros(0)])
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if gaps.size else None


def plan_sequences(
    frame: pd.DataFrame,
    times: list[str],
    entities: list[str],
    length: int | None = None,
) -> SequencePlan:
    """`times`: date columns; `entities`: columns that may identify a history, best first by the
    caller's order of preference."""
    if not times:
        return SequencePlan(None, None, error="Needs a date column.")
    time = times[0]
    candidates = []
    for name in entities:
        if name == time:
            continue
        histories = _histories(frame, time, name)
        long_enough = [h for h in histories if len(h) >= MIN_LENGTH]
        if len(long_enough) < 2:
            continue
        repeated = np.mean([1 - len(np.unique(h)) / len(h) for h in long_enough])
        if repeated > DUPLICATE_TIMES:
            continue
        median = int(np.median([len(h) for h in long_enough]))
        candidates.append(
            EntityCandidate(
                name,
                len(long_enough),
                median,
                f"{len(long_enough)} histories of about {median} dates, one value per date.",
            )
        )
    candidates.sort(key=lambda c: (-c.sequences * min(c.median_length, DEFAULT_LENGTH), c.name))
    entity = candidates[0].name if candidates else None
    histories = _histories(frame, time, entity)
    shortest = min((len(h) for h in histories if len(h) >= MIN_LENGTH), default=0)
    if shortest < MIN_LENGTH:
        return SequencePlan(
            time, entity, candidates, error=f"Needs histories of at least {MIN_LENGTH} dates."
        )
    step = _median_step(histories)
    chosen = length or min(DEFAULT_LENGTH, shortest)
    chosen = int(min(MAX_LENGTH, max(MIN_LENGTH, chosen)))
    warning = None
    if entity is not None:
        count = sum(1 for h in histories if len(h) >= chosen)
        reason = f"One sequence per {entity}, {chosen} steps each."
    else:
        count = max(1, len(histories[0]) // chosen)
        reason = f"One long series cut into {count} windows of {chosen} steps."
        series = histories[0]
        per_date = len(series) / max(len(np.unique(series)), 1)
        if per_date > 1 + DUPLICATE_TIMES:
            warning = (
                f"Dates repeat (about {per_date:.1f} rows per date) and no column tells the "
                "histories apart: rows of a date follow each other in file order. Pick an entity "
                "if there is one."
            )
    return SequencePlan(
        time,
        entity,
        candidates,
        step,
        step_label(step) if step else "",
        chosen,
        int(min(MAX_SEQUENCES, max(1, count))),
        reason,
        warning=warning,
    )


# --- Windows ---------------------------------------------------------------------------------


@dataclass
class Slot:
    """Where a column sits in a time step."""

    name: str
    kind: str  # "number" or "category"
    start: int
    width: (
        int  # numbers: 1 (+1 for the empty indicator); categories: levels kept (+ others, + empty)
    )
    low: float = 0.0
    high: float = 1.0
    fill: float = 0.5  # scaled median, for empty numbers
    empty: bool = False
    kept: int = 0  # categories: levels with their own slot
    others: bool = False  # categories: one slot for the rest
    rest: list[float] = field(default_factory=list)  # shares of the rest, to draw a level back


@dataclass
class Layout:
    slots: list[Slot]

    @property
    def width(self) -> int:
        return sum(s.width for s in self.slots)


def layout_for(coded: Coded, learned: list[Learned]) -> Layout:
    slots, start = [], 0
    for item in learned:
        values = coded[item.name]
        missing = np.isnan(values)
        if item.categorical:
            present = values[~missing].astype(np.int64)
            counts = (
                np.bincount(present, minlength=len(item.levels)) if present.size else np.zeros(1)
            )
            kept = min(len(item.levels), MAX_LEVELS)
            others = len(item.levels) > MAX_LEVELS
            rest = (
                counts[kept:] / counts[kept:].sum()
                if others and counts[kept:].sum()
                else np.zeros(0)
            )
            width = kept + int(others) + int(missing.any())
            slots.append(
                Slot(
                    item.name,
                    "category",
                    start,
                    width,
                    empty=bool(missing.any()),
                    kept=kept,
                    others=others,
                    rest=rest.tolist(),
                )
            )
        else:
            present = values[~missing]
            low, high = (float(present.min()), float(present.max())) if present.size else (0.0, 1.0)
            span = high - low or 1.0
            fill = float((np.median(present) - low) / span) if present.size else 0.5
            width = 1 + int(missing.any())
            slots.append(
                Slot(item.name, "number", start, width, low, high, fill, bool(missing.any()))
            )
        start += slots[-1].width
    return Layout(slots)


def encode_steps(coded: Coded, layout: Layout) -> np.ndarray:
    """Rows (already sorted) as time steps in [0, 1]."""
    rows = len(next(iter(coded.values()))) if coded else 0
    out = np.zeros((rows, layout.width), dtype=np.float32)
    for slot in layout.slots:
        values = coded[slot.name]
        missing = np.isnan(values)
        if slot.kind == "number":
            span = slot.high - slot.low or 1.0
            out[:, slot.start] = np.where(
                missing, slot.fill, (np.nan_to_num(values) - slot.low) / span
            )
            if slot.empty:
                out[:, slot.start + 1] = missing
            continue
        index = np.nan_to_num(values).astype(np.int64)
        index = np.where(index >= slot.kept, slot.kept, index)  # the "other" slot
        if slot.empty:
            index = np.where(missing, slot.width - 1, index)
        out[np.arange(rows), slot.start + np.clip(index, 0, slot.width - 1)] = 1
    return out


def decode_steps(steps: np.ndarray, layout: Layout, rng: np.random.Generator) -> Coded:
    """Time steps back to coded columns: numbers rescaled, empty drawn from its indicator,
    levels drawn from the outputs."""
    result: Coded = {}
    rows = len(steps)
    for slot in layout.slots:
        if slot.kind == "number":
            span = slot.high - slot.low or 1.0
            values = np.clip(steps[:, slot.start], 0, 1) * span + slot.low
            if slot.empty:
                values = np.where(
                    rng.random(rows) < np.clip(steps[:, slot.start + 1], 0, 1), np.nan, values
                )
            result[slot.name] = values
            continue
        block = np.clip(steps[:, slot.start : slot.start + slot.width], 1e-6, None)
        cumulative = (block / block.sum(axis=1, keepdims=True)).cumsum(axis=1)
        drawn = (cumulative < rng.random((rows, 1))).sum(axis=1)
        drawn = np.minimum(drawn, slot.width - 1).astype(float)
        if slot.others:
            other = drawn == slot.kept
            if other.any() and slot.rest:
                shares = np.cumsum(slot.rest)
                drawn[other] = slot.kept + np.minimum(
                    np.searchsorted(shares, rng.random(int(other.sum()))), len(slot.rest) - 1
                )
        if slot.empty:
            drawn[drawn == slot.width - 1] = np.nan
        result[slot.name] = drawn
    return result


@dataclass
class Windows:
    data: np.ndarray  # windows * length * width, in [0, 1]
    layout: Layout
    starts: np.ndarray  # first time (seconds) of each history, to date new sequences


def order_rows(
    frame: pd.DataFrame, time: str, entity: str | None
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Row positions of each history, sorted by time (rows without a date are left out)."""
    stamps = seconds(column_of(frame, time))
    keys = (
        labels(column_of(frame, entity)).to_numpy()
        if entity
        else np.zeros(len(frame), dtype=object)
    )
    groups = []
    for key in pd.unique(keys):
        if key is None:
            continue
        rows = np.flatnonzero((keys == key) & ~np.isnan(stamps))
        groups.append(rows[np.argsort(stamps[rows], kind="stable")])
    return stamps, groups


def windows(
    frame: pd.DataFrame,
    coded: Coded,
    layout: Layout,
    time: str,
    entity: str | None,
    length: int,
    seed: int,
    stride: int = 1,
    limit: int = MAX_WINDOWS,
) -> Windows:
    steps = encode_steps(coded, layout)
    stamps, groups = order_rows(frame, time, entity)
    cut = []
    for rows in groups:
        for begin in range(0, len(rows) - length + 1, stride):
            cut.append(rows[begin : begin + length])
    if not cut:
        return Windows(np.zeros((0, length, layout.width), dtype=np.float32), layout, np.zeros(0))
    index = np.stack(cut)
    if len(index) > limit:
        index = index[np.sort(np.random.default_rng(seed).choice(len(index), limit, replace=False))]
    starts = np.array([stamps[rows[0]] for rows in groups if len(rows)])
    return Windows(steps[index], layout, starts)


def windows_in_blocks(coded: Coded, layout: Layout, length: int) -> Windows:
    """Synthetic rows in generation order, `length` rows per history (how TimeGAN writes them)."""
    steps = encode_steps(coded, layout)
    count = len(steps) // length
    return Windows(
        steps[: count * length].reshape(count, length, layout.width), layout, np.zeros(0)
    )


def split_histories(
    frame: pd.DataFrame, time: str, entity: str | None, seed: int, share: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Held-out real rows that keep whole histories: 1 entity in 5 (at least one), or the last
    20 % of a single series."""
    _, groups = order_rows(frame, time, entity)
    if entity is None or len(groups) < 2:
        rows = groups[0] if groups else np.zeros(0, dtype=np.int64)
        cut = int(len(rows) * (1 - share))
        return frame.iloc[rows[:cut]].reset_index(drop=True), frame.iloc[rows[cut:]].reset_index(
            drop=True
        )
    rng = np.random.default_rng(seed)
    held = max(1, round(len(groups) * share))
    chosen = set(rng.choice(len(groups), held, replace=False).tolist())
    train = np.concatenate([g for i, g in enumerate(groups) if i not in chosen])
    holdout = np.concatenate([g for i, g in enumerate(groups) if i in chosen])
    return frame.iloc[train].reset_index(drop=True), frame.iloc[holdout].reset_index(drop=True)


# --- Measures --------------------------------------------------------------------------------


@dataclass
class ColumnAutocorrelation:
    column: str
    real: list[float]  # mean autocorrelation at lags 1..L
    synthetic: list[float]
    similarity: float  # 1 - mean |real - synthetic| over the lags, at least 0


@dataclass
class SampleSequences:
    column: str
    real: list[list[float]]  # a few windows, in the column's units
    synthetic: list[list[float]]


@dataclass
class SequenceReport:
    time: str
    entity: str | None
    length: int
    step: str
    sequences: int  # synthetic sequences
    train_windows: int
    autocorrelation: float | None  # mean similarity over numeric columns
    columns: list[ColumnAutocorrelation] = field(default_factory=list)
    discriminative: float | None = None  # 1 - 2 |accuracy - 0.5|
    discriminative_accuracy: float | None = None
    predictive_real: float | None = None  # MAE, trained on real windows (TRTR)
    predictive_synthetic: float | None = None  # MAE, trained on synthetic windows (TSTR)
    predictive_ratio: float | None = None  # real MAE / synthetic MAE
    samples: list[SampleSequences] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)


def autocorrelation(series: np.ndarray, lags: int) -> np.ndarray:
    """Mean autocorrelation of windows (windows * length) at lags 1..lags; flat windows skipped."""
    centred = series - series.mean(axis=1, keepdims=True)
    variance = (centred**2).sum(axis=1)
    keep = variance > 1e-12
    if not keep.any():
        return np.zeros(lags)
    centred, variance = centred[keep], variance[keep]
    return np.array(
        [
            float(np.mean((centred[:, :-lag] * centred[:, lag:]).sum(axis=1) / variance))
            for lag in range(1, lags + 1)
        ]
    )


def autocorrelations(real: Windows, synthetic: Windows) -> list[ColumnAutocorrelation]:
    length = real.data.shape[1]
    lags = max(1, min(10, length - 1))
    out = []
    for slot in real.layout.slots:
        if slot.kind != "number" or not len(real.data) or not len(synthetic.data):
            continue
        r = autocorrelation(real.data[:, :, slot.start], lags)
        s = autocorrelation(synthetic.data[:, :, slot.start], lags)
        out.append(
            ColumnAutocorrelation(
                slot.name, r.tolist(), s.tolist(), float(max(0.0, 1 - np.mean(np.abs(r - s))))
            )
        )
    return out


def samples(real: Windows, synthetic: Windows, seed: int, count: int = 5) -> list[SampleSequences]:
    rng = np.random.default_rng(seed)
    out = []
    for slot in real.layout.slots:
        if slot.kind != "number":
            continue
        span = slot.high - slot.low or 1.0

        def pick(data: np.ndarray, slot: Slot = slot, span: float = span) -> list[list[float]]:
            if not len(data):
                return []
            chosen = rng.choice(len(data), min(count, len(data)), replace=False)
            return np.round(data[chosen, :, slot.start] * span + slot.low, 6).tolist()

        out.append(SampleSequences(slot.name, pick(real.data), pick(synthetic.data)))
    return out
