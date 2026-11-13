"""R-XAI-06: the true / false conditions a scorecard (121) or an optimal tree (122) is built on,
read on the dataset's columns in their units.

- a number: "x ≤ t" at its deciles, each threshold rounded to 2 significant digits, distinct,
  with at least 1 % of the rows on each side; a number with 2 values or fewer: "x = v";
- a category (booleans read "True" / "False"): "x = c" for its 10 most frequent values with at
  least 1 % of the rows; one condition only for two values (True, or the rarer one);
- "x is empty" when at least 1 % of the rows are empty;
- dates, free text and identifiers are left out: a card adds points of conditions anyone can
  check by hand, and a round threshold reads and applies better than 12.37.

The same conditions are then evaluated on other rows (`matrix`), so the rows a card is judged on
never shape its thresholds."""

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd

QUANTILES = np.linspace(0.1, 0.9, 9)
MIN_SHARE = 0.01
TOP_CATEGORIES = 10
SKIPPED_TYPES = {"datetime", "text", "identifier"}

Kind = Literal["below", "above", "equals", "differs", "empty"]


@dataclass
class Condition:
    column: str
    kind: Kind
    value: str | None  # the threshold or the category, as shown
    number: float | None = None  # the threshold as a number

    @property
    def text(self) -> str:
        if self.kind == "empty":
            return f"{self.column} is empty"
        sign = {"below": "≤", "above": ">", "equals": "=", "differs": "≠"}[self.kind]
        return f"{self.column} {sign} {self.value}"

    def flipped(self) -> "Condition":
        """The opposite condition (a threshold read the other way)."""
        other = {"below": "above", "above": "below", "equals": "differs", "differs": "equals"}
        if self.kind == "empty":
            raise ValueError("'is empty' has no readable opposite")
        return Condition(self.column, cast(Kind, other[self.kind]), self.value, self.number)

    def holds(self, frame: pd.DataFrame) -> np.ndarray:
        raw = cast(pd.Series, frame[self.column])
        if self.kind == "empty":
            return raw.isna().to_numpy()
        if self.kind in ("below", "above"):
            values = _floats(raw)
            threshold = float(self.number if self.number is not None else 0.0)
            with np.errstate(invalid="ignore"):
                found = values <= threshold if self.kind == "below" else values > threshold
            return found & ~np.isnan(values)
        labels = _labels(raw)
        same = (labels == self.value).to_numpy() & raw.notna().to_numpy()
        return same if self.kind == "equals" else ~same & raw.notna().to_numpy()


def _floats(raw: pd.Series) -> np.ndarray:
    return np.asarray(cast(pd.Series, pd.to_numeric(raw, errors="coerce")).to_numpy(), dtype=float)


def _labels(raw: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(raw):
        return raw.map({True: "True", False: "False"}).astype(str)
    if pd.api.types.is_numeric_dtype(raw):
        return raw.map(_number_label).astype(str)
    return raw.astype(str)


def _number_label(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    number = float(cast(float, value))
    return str(int(number)) if number.is_integer() else f"{number:.4g}"


def _round(value: float) -> float:
    return float(f"{value:.2g}")


def _shown(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.4g}"


def _numbers(column: str, raw: pd.Series) -> list[Condition]:
    values = _floats(raw)
    present = values[~np.isnan(values)]
    if not len(present):
        return []
    distinct = np.unique(present)
    if len(distinct) <= 2:
        high = float(distinct[-1])
        if len(distinct) == 1:
            return []
        return [Condition(column, "equals", _number_label(high))]
    found: list[Condition] = []
    seen: set[float] = set()
    for q in np.quantile(present, QUANTILES):
        threshold = _round(float(q))
        if threshold in seen:
            continue
        below = int((present <= threshold).sum())
        above = len(present) - below
        if min(below, above) >= MIN_SHARE * len(values):
            seen.add(threshold)
            found.append(Condition(column, "below", _shown(threshold), threshold))
    return found


def _categories(column: str, raw: pd.Series) -> list[Condition]:
    labels = cast(pd.Series, _labels(raw)[raw.notna()])
    if not len(labels):
        return []
    counts = labels.value_counts()
    if len(counts) < 2:
        return []
    if len(counts) == 2:
        chosen = "True" if "True" in counts.index else str(counts.index[-1])
        return [Condition(column, "equals", chosen)]
    counts = cast(pd.Series, counts[counts >= MIN_SHARE * len(raw)])
    return [Condition(column, "equals", str(c)) for c in list(counts.index)[:TOP_CATEGORIES]]


def conditions_for(
    frame: pd.DataFrame, skip: set[str] | None = None, types: dict[str, str] | None = None
) -> list[Condition]:
    """The candidate conditions of every usable column (R-XAI-06), learnt on `frame`."""
    skip = skip or set()
    types = types or {}
    found: list[Condition] = []
    for column in [str(c) for c in frame.columns]:
        if column in skip or types.get(column) in SKIPPED_TYPES:
            continue
        raw = cast(pd.Series, frame[column])
        if pd.api.types.is_datetime64_any_dtype(raw):
            continue
        if pd.api.types.is_bool_dtype(raw) or not pd.api.types.is_numeric_dtype(raw):
            found += _categories(column, raw)
        else:
            found += _numbers(column, raw)
        empty = float(raw.isna().mean()) if len(raw) else 0.0
        if MIN_SHARE <= empty <= 1 - MIN_SHARE:
            found.append(Condition(column, "empty", None))
    return found


def matrix(frame: pd.DataFrame, conditions: list[Condition]) -> np.ndarray:
    """1.0 where each condition holds, row by row (rows × conditions)."""
    if not conditions:
        return np.zeros((len(frame), 0))
    return np.column_stack([c.holds(frame) for c in conditions]).astype(float)
