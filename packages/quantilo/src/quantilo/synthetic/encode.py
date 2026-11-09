"""Learn each modelled column (bounds, decimals, levels) and go between typed values and the
numeric codes the methods work on.

Typed frame: numbers as floats, dates as datetime64, categories as text; empty = NaN / None.
Codes: numbers as they are, dates as seconds, categories as level indexes; empty = NaN.
"""

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import pandas as pd

from quantilo.typing.types import AnalysisType

NUMERIC: set[AnalysisType] = {"numeric_continuous", "numeric_discrete"}
CATEGORICAL: set[AnalysisType] = {"categorical_nominal", "categorical_ordinal", "boolean"}
RARE = 5  # a level seen fewer times is rare

Kind = Literal["number", "integer", "date", "category"]
OTHER = "Other"
EPOCH = pd.Timestamp("1970-01-01")
DAY = 86_400.0

Coded = dict[str, np.ndarray]


@dataclass
class Learned:
    name: str
    type: AnalysisType
    kind: Kind
    missing: float  # share of empty values
    minimum: float | None = None  # numbers: the value; dates: seconds
    maximum: float | None = None
    decimals: int = 0
    day_only: bool = False
    levels: list[str] = field(default_factory=list)  # code i = levels[i]
    grouped: bool = False  # rare levels were grouped into "Other" (R-SYN-08)

    @property
    def categorical(self) -> bool:
        return self.kind == "category"


def kind_of(analysis_type: AnalysisType) -> Kind:
    if analysis_type == "numeric_discrete":
        return "integer"
    if analysis_type in NUMERIC:
        return "number"
    if analysis_type == "datetime":
        return "date"
    if analysis_type in CATEGORICAL:
        return "category"
    raise ValueError(f"A {analysis_type} column cannot be modelled.")


def column_of(frame: pd.DataFrame, name: str) -> pd.Series:
    """One column of the frame, typed as a series."""
    return cast(pd.Series, frame[name])


def seconds(values: pd.Series) -> np.ndarray:
    """Dates as seconds since 1970 (NaN for empty values)."""
    stamps = cast(pd.Series, pd.to_datetime(values, errors="coerce"))
    return ((stamps - EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def numbers(values: pd.Series) -> np.ndarray:
    return cast(pd.Series, pd.to_numeric(values, errors="coerce")).to_numpy(dtype=float)


def labels(values: pd.Series) -> pd.Series:
    """Category values as text; empty values stay None."""
    return pd.Series(
        [None if pd.isna(v) else str(v) for v in values], index=values.index, dtype=object
    )


def decimals(values: np.ndarray) -> int:
    """Fewest decimals that write every value exactly (at most 6)."""
    sample = values[~np.isnan(values)][:2_000]
    if sample.size == 0:
        return 0
    tolerance = 1e-9 * max(1.0, float(np.abs(sample).max()))
    for places in range(7):
        if np.all(np.abs(np.round(sample, places) - sample) <= tolerance):
            return places
    return 6


def group_rare(values: pd.Series, rare: int = RARE) -> pd.Series:
    """Levels seen fewer than `rare` times become "Other" (R-SYN-08)."""
    text = labels(values)
    counts = text.value_counts()
    keep = set(cast(pd.Series, counts[counts >= rare]).index)
    return pd.Series(
        [v if v is None or v in keep else OTHER for v in text], index=text.index, dtype=object
    )


def learn_column(
    name: str, values: pd.Series, analysis_type: AnalysisType, order: list[str] | None = None
) -> Learned:
    kind = kind_of(analysis_type)
    missing = float(values.isna().mean()) if len(values) else 0.0
    learned = Learned(name, analysis_type, kind, missing)
    if kind == "category":
        text = labels(values)
        counts = text.value_counts()
        found = [str(v) for v in counts.index]
        ordered = [v for v in (order or []) if v in counts.index]
        learned.levels = ordered + [v for v in found if v not in ordered]
        learned.grouped = OTHER in counts.index and OTHER not in (order or [])
        return learned
    raw = seconds(values) if kind == "date" else numbers(values)
    present = raw[~np.isnan(raw)]
    if present.size:
        learned.minimum, learned.maximum = float(present.min()), float(present.max())
    if kind == "date":
        learned.day_only = bool(present.size) and bool(np.all(np.mod(present, DAY) == 0))
    else:
        learned.decimals = 0 if kind == "integer" else decimals(present)
    return learned


def encode_column(values: pd.Series, learned: Learned) -> np.ndarray:
    if learned.kind == "category":
        index = {level: i for i, level in enumerate(learned.levels)}
        text = labels(values)
        return np.array([np.nan if v is None else index.get(v, np.nan) for v in text], dtype=float)
    return seconds(values) if learned.kind == "date" else numbers(values)


def encode(frame: pd.DataFrame, learned: list[Learned]) -> Coded:
    return {item.name: encode_column(column_of(frame, item.name), item) for item in learned}


def decode_column(
    codes: np.ndarray,
    learned: Learned,
    bounds: tuple[float | None, float | None] = (None, None),
    clip: bool = True,
) -> pd.Series:
    """Codes back to typed values, inside the bounds (unless `clip` is off: the dates of new
    sequences), with the column's decimals."""
    if learned.kind == "category":
        levels = np.array([*learned.levels, None], dtype=object)
        index = np.where(np.isnan(codes), len(learned.levels), codes).astype(int)
        return pd.Series(levels[np.clip(index, 0, len(learned.levels))], dtype=object)
    low = bounds[0] if bounds[0] is not None else learned.minimum
    high = bounds[1] if bounds[1] is not None else learned.maximum
    if not clip:
        low = high = None
    values = codes.astype(float).copy()
    if learned.kind == "date":
        if learned.day_only:
            values = np.round(values / DAY) * DAY
            low = None if low is None else np.ceil(low / DAY) * DAY
            high = None if high is None else np.floor(high / DAY) * DAY
        values = np.clip(values, low, high) if low is not None and high is not None else values
        stamps = EPOCH + pd.to_timedelta(values, unit="s")
        return pd.Series(pd.DatetimeIndex(stamps)).dt.round("s")
    places = 0 if learned.kind == "integer" else learned.decimals
    values = np.round(values, places)
    if low is not None and high is not None:
        step = 10.0**-places
        low_edge = np.ceil(round(low / step, 6)) * step
        high_edge = np.floor(round(high / step, 6)) * step
        values = np.round(np.clip(values, low_edge, high_edge), places)
    return pd.Series(values, dtype=float)
