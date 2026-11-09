"""Are real rows protected? Measured against real rows held out from the generator.

Distance between two rows (SDMetrics' definition): |a - b| / real range, capped at 1, for numbers
and dates; 0 or 1 for categories and empty values; the row distance is the mean over columns.
DCR (distance to closest record) = the distance to the nearest row of a reference set.
"""

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import pandas as pd

from quantilo.synthetic.encode import Learned, column_of, labels
from quantilo.synthetic.plan import RARE
from quantilo.synthetic.quality import as_numbers

MAX_QUERY = 5_000  # synthetic rows whose DCR is computed
MAX_REFERENCE = 5_000  # real rows per reference set (training and held-out, same size)
BLOCK = 512
COPY_TOLERANCE = 0.01  # share of the range under which two numbers count as equal
MEMORISATION = 0.8
HISTOGRAM_BINS = 30


class PrivacyError(ValueError):
    pass


@dataclass
class Rows:
    """Rows coded for distances: numbers scaled by the real range, categories as shared codes."""

    numbers: np.ndarray  # rows * numeric columns (NaN = empty)
    codes: np.ndarray  # rows * categorical columns (-1 = empty)

    def __len__(self) -> int:
        return len(self.numbers)

    def take(self, index: np.ndarray) -> "Rows":
        return Rows(self.numbers[index], self.codes[index])


@dataclass
class RareValue:
    column: str
    value: str
    real: int
    synthetic: int


@dataclass
class DcrHistogram:
    edges: list[float]
    synthetic: list[float]  # share of synthetic rows per bin (DCR to training rows)
    holdout: list[float]  # share of held-out real rows per bin (DCR to training rows)


@dataclass
class Privacy:
    score: float  # min(1 - share of copied unique rows, overfitting protection)
    unique_copies: int  # synthetic rows equal to a training row that is unique in the data
    unique_copy_share: float
    copies: int  # synthetic rows exactly equal to any training row
    copy_share: float
    near_copies: int  # numbers within 1 % of the range (SDMetrics' NewRowSynthesis)
    new_rows: float  # NewRowSynthesis = 1 - near copy share
    overfitting: float | None  # DCROverfittingProtection
    closer_to_train: float | None  # share of synthetic rows closer to training than held-out rows
    baseline: float | None  # DCRBaselineProtection
    median_dcr: float | None
    rare: list[RareValue] = field(default_factory=list)
    histogram: DcrHistogram | None = None
    warnings: list[str] = field(default_factory=list)
    verdict: str = ""


@dataclass
class CapVariant:
    name: Literal["CAP", "Zero CAP", "Generalized CAP"]
    attack: float  # attacker accuracy
    protection: float


@dataclass
class Disclosure:
    known: list[str]
    sensitive: str
    baseline: float  # accuracy of a random guess
    variants: list[CapVariant] = field(default_factory=list)
    score: float = 0.0  # protection of the CAP variant
    rows: int = 0


def code_rows(frames: list[pd.DataFrame], learned: list[Learned]) -> list[Rows]:
    """Codes for distances; ranges and levels come from the first frame (the real rows)."""
    numeric = [i for i in learned if not i.categorical]
    categorical = [i for i in learned if i.categorical]
    numbers: list[list[np.ndarray]] = [[] for _ in frames]
    for item in numeric:
        values = [as_numbers(column_of(f, item.name), item) for f in frames]
        present = values[0][~np.isnan(values[0])]
        low = float(present.min()) if present.size else 0.0
        span = float(present.max()) - low if present.size else 0.0
        for slot, v in enumerate(values):
            numbers[slot].append(
                (v - low) / span if span > 0 else np.where(np.isnan(v), np.nan, 0.0)
            )
    codes: list[list[np.ndarray]] = [[] for _ in frames]
    for item in categorical:
        texts = [labels(column_of(f, item.name)) for f in frames]
        index: dict[str, int] = {}
        for text in texts:
            for value in text.dropna().unique():
                index.setdefault(value, len(index))
        for slot, text in enumerate(texts):
            codes[slot].append(
                np.array([-1 if v is None else index[v] for v in text], dtype=np.int64)
            )
    return [
        Rows(
            np.column_stack(numbers[s]) if numbers[s] else np.empty((len(f), 0)),
            np.column_stack(codes[s]) if codes[s] else np.empty((len(f), 0), dtype=np.int64),
        )
        for s, f in enumerate(frames)
    ]


def closest(query: Rows, reference: Rows) -> np.ndarray:
    """DCR of each query row to the reference rows."""
    columns = query.numbers.shape[1] + query.codes.shape[1]
    result = np.empty(len(query))
    if columns == 0 or len(reference) == 0:
        result.fill(np.nan)
        return result
    numbers = query.numbers.astype(np.float32)
    known = reference.numbers.astype(np.float32)
    known_missing = np.isnan(known)
    known_filled = np.nan_to_num(known)
    total = np.empty((BLOCK, len(reference)), dtype=np.float32)
    gap = np.empty_like(total)
    for start in range(0, len(query), BLOCK):
        stop = min(start + BLOCK, len(query))
        rows = stop - start
        block, buffer = total[:rows], gap[:rows]
        block.fill(0)
        for j in range(numbers.shape[1]):
            q = numbers[start:stop, j]
            q_missing = np.isnan(q)
            np.subtract(np.nan_to_num(q)[:, None], known_filled[None, :, j], out=buffer)
            np.abs(buffer, out=buffer)
            np.minimum(buffer, 1.0, out=buffer)
            if q_missing.any() or known_missing[:, j].any():
                # one side empty: 1; both empty: 0
                either = q_missing[:, None] | known_missing[None, :, j]
                buffer[either] = (q_missing[:, None] != known_missing[None, :, j])[either]
            block += buffer
        for j in range(query.codes.shape[1]):
            block += query.codes[start:stop, j][:, None] != reference.codes[None, :, j]
        result[start:stop] = block.min(axis=1) / columns
    return result


def _copy_keys(rows: Rows, tolerance: float | None) -> pd.MultiIndex | pd.Index:
    """Numbers rounded to `tolerance` * the range (near-equal rows share a key), or exact."""
    filled = np.nan_to_num(rows.numbers, nan=-999.0)
    rounded = np.round(filled / tolerance) if tolerance else np.round(filled, 9)
    data = np.column_stack([rounded, rows.codes]) if rows.codes.size else rounded
    frame = pd.DataFrame(data)
    if frame.shape[1] == 1:
        return pd.Index(frame.iloc[:, 0])
    return pd.MultiIndex.from_frame(frame)


def unique_copies(synthetic: Rows, train: Rows) -> int:
    """Synthetic rows equal to a training row whose values no other training row shares: the
    copies that can point to one person."""
    if synthetic.numbers.shape[1] + synthetic.codes.shape[1] == 0:
        return 0
    keys = _copy_keys(train, None)
    counts = pd.Series(1, index=keys).groupby(level=list(range(keys.nlevels))).size()
    single = cast(pd.Series, counts[counts == 1]).index
    return int(_copy_keys(synthetic, None).isin(single).sum())


def copies(synthetic: Rows, train: Rows, tolerance: float | None = None) -> int:
    """Synthetic rows equal to a training row: exactly, or with numbers within `tolerance` * the
    range (SDMetrics' NewRowSynthesis uses 1 %)."""
    if synthetic.numbers.shape[1] + synthetic.codes.shape[1] == 0:
        return 0
    return int(_copy_keys(synthetic, tolerance).isin(_copy_keys(train, tolerance)).sum())


def overfitting_protection(to_train: np.ndarray, to_holdout: np.ndarray) -> tuple[float, float]:
    """(score, share of synthetic rows closer to training rows); ties count half."""
    closer = float(np.mean((to_train < to_holdout) + 0.5 * (to_train == to_holdout)))
    return min(1.0, 2 * (1 - closer)), closer


def uniform_rows(train: Rows, count: int, rng: np.random.Generator) -> Rows:
    """Random rows inside the real ranges and levels (the baseline)."""
    numbers = rng.random((count, train.numbers.shape[1]))
    codes = np.empty((count, train.codes.shape[1]), dtype=np.int64)
    for j in range(train.codes.shape[1]):
        levels = np.unique(train.codes[:, j])
        codes[:, j] = levels[rng.integers(0, len(levels), size=count)]
    return Rows(numbers, codes)


def rare_values(
    real: pd.DataFrame, synthetic: pd.DataFrame, learned: list[Learned]
) -> list[RareValue]:
    found = []
    for item in learned:
        if not item.categorical:
            continue
        real_counts = labels(column_of(real, item.name)).value_counts()
        rare = cast(pd.Series, real_counts[real_counts < RARE])
        if rare.empty:
            continue
        synthetic_counts = labels(column_of(synthetic, item.name)).value_counts()
        for value, count in rare.items():
            if value in synthetic_counts.index:
                found.append(
                    RareValue(
                        item.name, str(value), int(count), int(cast(int, synthetic_counts[value]))
                    )
                )
    found.sort(key=lambda r: (r.real, -r.synthetic))
    return found[:50]


def _sample(rows: Rows, count: int, rng: np.random.Generator) -> Rows:
    if len(rows) <= count:
        return rows
    return rows.take(rng.choice(len(rows), size=count, replace=False))


def privacy(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    learned: list[Learned],
    seed: int = 42,
) -> Privacy:
    rng = np.random.default_rng(seed)
    train_rows, holdout_rows, synthetic_rows = code_rows([train, holdout, synthetic], learned)
    copy_count = copies(synthetic_rows, train_rows)
    copy_share = copy_count / len(synthetic) if len(synthetic) else 0.0
    unique_count = unique_copies(synthetic_rows, train_rows)
    unique_share = unique_count / len(synthetic) if len(synthetic) else 0.0
    near = copies(synthetic_rows, train_rows, COPY_TOLERANCE)
    query = _sample(synthetic_rows, MAX_QUERY, rng)
    size = min(len(train_rows), len(holdout_rows), MAX_REFERENCE)
    overfitting = closer = baseline = median = None
    histogram = None
    if size >= 10 and len(query):
        reference = _sample(train_rows, size, rng)
        held = _sample(holdout_rows, size, rng)
        to_train = closest(query, reference)
        to_holdout = closest(query, held)
        overfitting, closer = overfitting_protection(to_train, to_holdout)
        random = closest(uniform_rows(train_rows, len(query), rng), reference)
        median = float(np.median(to_train))
        random_median = float(np.median(random))
        baseline = min(1.0, median / random_median) if random_median > 0 else None
        holdout_to_train = closest(held, reference)
        edges = np.linspace(0, 1, HISTOGRAM_BINS + 1)
        histogram = DcrHistogram(
            [float(e) for e in edges],
            (np.histogram(to_train, edges)[0] / len(to_train)).tolist(),
            (np.histogram(holdout_to_train, edges)[0] / len(holdout_to_train)).tolist(),
        )
    new_rows = 1 - (near / len(synthetic) if len(synthetic) else 0.0)
    score = min(1 - unique_share, overfitting if overfitting is not None else 1.0)
    warnings = []
    if unique_count:
        warnings.append(
            f"{unique_count:,} synthetic rows ({unique_share:.1%}) copy a real row that is unique "
            "in the data."
        )
    if overfitting is not None and overfitting < MEMORISATION:
        warnings.append(
            f"Memorisation: {closer:.0%} of synthetic rows are closer to the training rows than "
            "to new real rows (50 % expected)."
        )
    verdict = (
        " ".join(warnings)
        if warnings
        else "No unique real row copied, and synthetic rows are no closer to the real ones than "
        "new real rows would be."
    )
    return Privacy(
        score,
        unique_count,
        unique_share,
        copy_count,
        copy_share,
        near,
        new_rows,
        overfitting,
        closer,
        baseline,
        median,
        rare_values(train, synthetic, learned),
        histogram,
        warnings,
        verdict,
    )


# --- Disclosure (CAP) ------------------------------------------------------------------------


def _keys(
    frame: pd.DataFrame,
    columns: list[str],
    learned: dict[str, Learned],
    edges: dict[str, np.ndarray],
) -> list[tuple]:
    parts = []
    for name in columns:
        item = learned[name]
        if item.categorical:
            parts.append(labels(column_of(frame, name)).fillna("\x00").to_numpy())
        else:
            values = as_numbers(column_of(frame, name), item)
            bins = np.searchsorted(edges[name], np.nan_to_num(values), side="right")
            parts.append(np.where(np.isnan(values), -1, bins))
    return list(zip(*parts, strict=True)) if parts else []


def disclosure(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    learned: list[Learned],
    known: list[str],
    sensitive: str,
    seed: int = 42,
) -> Disclosure:
    """Correct Attribution Probability: knowing some columns of a real row, how often do the
    matching synthetic rows reveal its sensitive value? Numbers are cut into 10 quantile
    intervals of the real data. Protection = min(1, (1 - attack) / (1 - random guess))."""
    by_name = {i.name: i for i in learned}
    unknown = [c for c in [*known, sensitive] if c not in by_name]
    if unknown:
        raise PrivacyError(f"Not modelled: {', '.join(unknown)}.")
    if not known or sensitive in known:
        raise PrivacyError("Pick at least one known column and a different sensitive column.")
    real = real.sample(min(len(real), MAX_QUERY), random_state=seed)
    edges = {}
    for name in [*known, sensitive]:
        item = by_name[name]
        if not item.categorical:
            values = as_numbers(column_of(real, name), item)
            present = values[~np.isnan(values)]
            edges[name] = (
                np.unique(np.quantile(present, np.linspace(0, 1, 11))[1:-1])
                if present.size
                else np.array([])
            )
    real_known = _keys(real, known, by_name, edges)
    synthetic_known = _keys(synthetic, known, by_name, edges)
    real_secret = [k[0] for k in _keys(real, [sensitive], by_name, edges)]
    synthetic_secret = [k[0] for k in _keys(synthetic, [sensitive], by_name, edges)]
    votes: dict[tuple, dict] = {}
    for key, secret in zip(synthetic_known, synthetic_secret, strict=True):
        bucket = votes.setdefault(key, {})
        bucket[secret] = bucket.get(secret, 0) + 1
    keys = list(votes)
    key_array = np.array([[str(v) for v in k] for k in keys], dtype=object) if keys else None
    matched, zero, general = [], [], []
    for key, secret in zip(real_known, real_secret, strict=True):
        bucket = votes.get(key)
        if bucket is None and key_array is not None:
            distance = (key_array != np.array([str(v) for v in key], dtype=object)).sum(axis=1)
            nearest = [keys[i] for i in np.flatnonzero(distance == distance.min())]
            merged: dict = {}
            for k in nearest:
                for s, c in votes[k].items():
                    merged[s] = merged.get(s, 0) + c
            general.append(merged.get(secret, 0) / sum(merged.values()))
            zero.append(0.0)
            continue
        if bucket is None:
            zero.append(0.0)
            general.append(0.0)
            continue
        share = bucket.get(secret, 0) / sum(bucket.values())
        matched.append(share)
        zero.append(share)
        general.append(share)
    distinct = len(set(real_secret))
    baseline = 1 / distinct if distinct else 1.0

    def protect(attack: float) -> float:
        return 1.0 if baseline >= 1 else min(1.0, (1 - attack) / (1 - baseline))

    variants = []
    for name, shares in (("CAP", matched), ("Zero CAP", zero), ("Generalized CAP", general)):
        attack = float(np.mean(shares)) if shares else 0.0
        variants.append(CapVariant(name, attack, protect(attack)))  # type: ignore[arg-type]
    return Disclosure(known, sensitive, baseline, variants, variants[0].protection, len(real))
