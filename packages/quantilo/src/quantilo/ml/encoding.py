"""Encoders of the preparation (docs/features/063-ml-encodage/spec.md).

- `FoldPlan`: the internal folds of the target encoding, aligned with the validation (R-CAT-08).
- `FrequencyEncoder`: a category → its share of the training rows (R-CAT-09).
- `RareGrouper`: rare categories → "Other" before ordinal codes or a target encoding (R-CAT-01).
- `Cycle`: a number that goes round → sine and cosine (R-DATE-04).
- `PastSmooth`: the mean or EWMA of a column's past values, per entity, shifted (R-LAG-01).

Every class is pickled with the model (037); none needs anything beyond numpy, pandas and
scikit-learn, so all of them run in Pyodide.
"""

import copy
import math
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

FOLDS = 5
SEED = 42
OTHER = "Other"

FoldKind = Literal["random", "periods", "time", "groups"]

# --- R-CAT-08: the folds of the target encoding ---------------------------------------------------


@dataclass
class TargetFolds:
    """How the target encoding is cross-fitted, and why (R-CAT-08)."""

    kind: FoldKind
    column: str | None
    why: str


def target_folds(
    method: str, period: str | None, group: str | None, time: str | None
) -> TargetFolds:
    """The folds that follow the validation: whole periods, moments or groups never on both
    sides of an internal fold, like the rows the model is judged on."""
    if method == "panel_walk_forward" and period:
        return TargetFolds(
            "periods",
            period,
            f"Target encoding cross-fitted on {FOLDS} blocks of consecutive whole periods of "
            f"{period} (panel validation): a row is never encoded with the target of its own "
            "period, as for the periods the model is judged on.",
        )
    if method == "time_split" and time:
        return TargetFolds(
            "time",
            time,
            f"Target encoding cross-fitted on {FOLDS} blocks of consecutive values of {time} "
            "(time-based split): a row is never encoded with rows of its own moment, as for the "
            "most recent rows the model is judged on.",
        )
    if method == "group_kfold" and group:
        return TargetFolds(
            "groups",
            group,
            f"Target encoding cross-fitted on {FOLDS} folds of whole groups of {group} "
            "(group k-fold): a row is never encoded with the target of its own group, as for "
            "the groups the model is judged on.",
        )
    return TargetFolds(
        "random",
        None,
        f"Target encoding cross-fitted on {FOLDS} random folds of the training rows: the test "
        "rows are drawn at random too.",
    )


def order_key(values: pd.Series) -> pd.Series:
    """A key that sorts the values in time: dates as nanoseconds, numbers as they are, digit
    strings by value, other labels alphabetically."""
    if pd.api.types.is_datetime64_any_dtype(values):
        stamps = cast(pd.Series, pd.to_datetime(values))
        return cast(
            pd.Series,
            pd.Series(stamps.astype("int64").astype(float), index=values.index).where(
                stamps.notna()
            ),
        )
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float)
    numbers = cast(pd.Series, pd.to_numeric(values, errors="coerce"))
    if int(numbers.notna().sum()) == int(values.notna().sum()):
        return numbers.astype(float)
    return cast(pd.Series, values.astype(str).where(values.notna()))


def fold_keys(frame: pd.DataFrame, folds: TargetFolds) -> pd.Series | None:
    """Each row's fold key (period, moment or group), indexed like the frame; None when the
    rows cannot be told apart by their index."""
    if folds.kind == "random" or folds.column not in frame.columns:
        return None
    if not frame.index.is_unique:
        return None
    values = cast(pd.Series, frame[folds.column])
    if folds.kind == "groups":
        return cast(pd.Series, values.astype(str).where(values.notna(), "\x00missing"))
    key = order_key(values)
    if not pd.api.types.is_numeric_dtype(key):  # labels: their rank in alphabetical order
        distinct = pd.Index(key.dropna().unique()).sort_values()
        key = pd.Series(distinct.get_indexer(key), index=frame.index, dtype=float)
    return cast(pd.Series, key.fillna(-np.inf))  # rows without a date count as the oldest


class FoldPlan:
    """A scikit-learn splitter for `TargetEncoder(cv=…)`: it reads the fold key of every row
    through the index of the DataFrame it receives, so that whole periods, moments or groups
    stay on one side of every internal fold. The keys are never pickled (a saved model only
    transforms), and a copy shares them. Without readable keys: folds of contiguous rows."""

    def __init__(self, kind: FoldKind, keys: pd.Series | None = None, folds: int = FOLDS) -> None:
        self.kind: FoldKind = kind
        self.keys = keys
        self.folds = folds

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["keys"] = None
        return state

    def __deepcopy__(self, memo: dict[int, Any]) -> "FoldPlan":
        return FoldPlan(self.kind, self.keys, copy.deepcopy(self.folds, memo))

    def __repr__(self) -> str:
        return f"FoldPlan(kind={self.kind!r}, folds={self.folds})"

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.folds

    def _keys(self, X) -> np.ndarray | None:
        if self.keys is None or not isinstance(X, pd.DataFrame | pd.Series):
            return None
        found = self.keys.reindex(X.index)
        if bool(found.isna().any()):
            return None
        return found.to_numpy()

    def split(self, X, y=None, groups=None):
        n = len(X)
        keys = self._keys(X)
        found = None if keys is None else self._grouped(keys)
        if found is None:  # contiguous rows, never shuffled: rows often come in time order
            yield from KFold(min(self.folds, n)).split(np.zeros((n, 1)))
            return
        yield from found

    def _grouped(self, keys: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]] | None:
        codes, uniques = pd.factorize(keys, sort=self.kind != "groups")
        if len(uniques) < 2:
            return None
        if self.kind == "groups":
            splitter = GroupKFold(min(self.folds, len(uniques)))
            return list(splitter.split(np.zeros((len(keys), 1)), groups=codes))
        # consecutive keys in blocks of about equal rows, a key never cut in two
        counts = np.bincount(codes, minlength=len(uniques))
        before = np.cumsum(counts) - counts
        folds = min(self.folds, len(uniques))
        block_of = np.minimum((before * folds) // len(keys), folds - 1)
        block = block_of[codes]
        found = []
        for b in np.unique(block):
            test = np.flatnonzero(block == b)
            found.append((np.flatnonzero(block != b), test))
        return found if len(found) >= 2 else None


def random_folds(classes: bool) -> KFold | StratifiedKFold:
    """The folds used before 063, kept for random validations (scores unchanged)."""
    return (StratifiedKFold if classes else KFold)(FOLDS, shuffle=True, random_state=SEED)


# --- R-CAT-09 and R-CAT-01: frequency, rare categories --------------------------------------------


def _labels(X) -> np.ndarray:
    values = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=object)[:, 0]
    return np.asarray(pd.Series(values).astype(str), dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """A category → its share of the training rows; a category never seen → 0 (R-CAT-09)."""

    def fit(self, X, y=None):
        labels = _labels(X)
        counts = pd.Series(labels).value_counts()
        self.categories_ = [np.asarray(sorted(counts.index), dtype=object)]
        self.shares_ = {str(k): float(v) / max(len(labels), 1) for k, v in counts.items()}
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        labels = _labels(X)
        return np.asarray([self.shares_.get(v, 0.0) for v in labels], dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        name = input_features[0] if input_features is not None and len(input_features) else "x0"
        return np.asarray([name], dtype=object)


class RareGrouper(BaseEstimator, TransformerMixin):
    """Categories with fewer than `max(rare_rows, rare_share * training rows)` rows → "Other";
    when some were grouped, a category never seen becomes "Other" too. A DataFrame comes out
    as a DataFrame with its index (the target-encoding folds read it)."""

    def __init__(self, rare_rows: int = 10, rare_share: float = 0.01) -> None:
        self.rare_rows = rare_rows
        self.rare_share = rare_share

    def fit(self, X, y=None):
        labels = _labels(X)
        counts = pd.Series(labels).value_counts()
        least = max(self.rare_rows, math.ceil(self.rare_share * len(labels)))
        self.categories_ = [np.asarray(sorted(counts.index), dtype=object)]
        self.kept_ = {str(k) for k, v in counts.items() if v >= least}
        self.other_ = len(self.kept_) < len(counts)
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        labels = _labels(X)
        known = set(str(c) for c in self.categories_[0])
        out = np.asarray(
            [
                v if v in self.kept_ or (not self.other_ and v not in known) else OTHER
                for v in labels
            ],
            dtype=object,
        )
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame({X.columns[0]: out}, index=X.index)
        return out.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        name = input_features[0] if input_features is not None and len(input_features) else "x0"
        return np.asarray([name], dtype=object)


# --- R-DATE-04: numbers that go round -------------------------------------------------------------


class Cycle(BaseEstimator, TransformerMixin):
    """The first column → its sine and cosine over `period` (after `start`: month 1 → angle 0);
    any further column (a "was missing" flag) passes through."""

    def __init__(self, period: float = 24.0, start: float = 0.0) -> None:
        self.period = period
        self.start = start

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        angle = 2 * np.pi * (values[:, 0] - self.start) / self.period
        return np.column_stack([np.sin(angle), np.cos(angle), values[:, 1:]])

    def get_feature_names_out(self, input_features=None):
        names = (
            list(input_features)
            if input_features is not None
            else [f"x{i}" for i in range(self.n_features_in_)]
        )
        return np.asarray([f"{names[0]}_sin", f"{names[0]}_cos", *names[1:]], dtype=object)


# --- R-LAG-01: causal smoothed lags -------------------------------------------------------------


class PastSmooth(BaseEstimator, TransformerMixin):
    """Columns in: the value, the entity (when `entity`), the period. Out: the value and the
    mean (`kind="mean"`, last `window` observations) or EWMA (span `window`) of the entity's
    values **shifted by `shift` observations**: the row's own period and the `shift - 1` before
    it never count. Past values come from the rows transformed and from the history kept at
    fitting (the last `window + shift` periods of each entity). `keep_value=False` (063 bis,
    R-LAG-03, the target's own past): only the shifted mean comes out, never the value."""

    keep_value = True  # models saved before 063 bis

    def __init__(
        self,
        kind: str = "mean",
        window: int = 5,
        shift: int = 1,
        entity: bool = True,
        keep_value: bool = True,
    ) -> None:
        self.kind = kind
        self.window = window
        self.shift = shift
        self.entity = entity
        self.keep_value = keep_value

    def _parts(self, X) -> tuple[np.ndarray, pd.MultiIndex]:
        """Each row's value, and its (entity, period key)."""
        data = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=object))
        value = cast(pd.Series, pd.to_numeric(data.iloc[:, 0], errors="coerce"))
        entity = (
            data.iloc[:, 1].astype(str).to_numpy()
            if self.entity
            else np.full(len(data), "", dtype=object)
        )
        key = order_key(cast(pd.Series, data.iloc[:, -1].reset_index(drop=True)))
        rows = pd.MultiIndex.from_arrays([entity, key.to_numpy()], names=["entity", "key"])
        return value.to_numpy(dtype=float), rows

    @staticmethod
    def _by_period(value: np.ndarray, rows: pd.MultiIndex) -> pd.Series:
        """One value per (entity, period): the mean of its rows; rows without a period left out."""
        found = pd.Series(value, index=rows)
        found = found[np.asarray(pd.notna(rows.get_level_values("key")))]
        return found.groupby(level=["entity", "key"], sort=True).mean()

    def fit(self, X, y=None):
        value, rows = self._parts(X)
        series = self._by_period(value, rows)
        keep = self.window + self.shift
        self.history_ = cast(pd.Series, series.groupby(level="entity").tail(keep))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        value, rows = self._parts(X)
        series = self._by_period(value, rows)
        history = self.history_[~self.history_.index.isin(series.index)]
        combined = pd.concat([history, series]).sort_index()
        shifted = combined.groupby(level="entity").shift(self.shift)
        grouped = shifted.groupby(level="entity")
        if self.kind == "ewm":
            past = grouped.ewm(span=self.window, min_periods=1, ignore_na=True).mean()
        else:
            past = grouped.rolling(self.window, min_periods=1).mean()
        past = cast(pd.Series, past).droplevel(0)
        shifted_past = past.reindex(rows).to_numpy(dtype=float)
        if not self.keep_value:
            return shifted_past[:, None]
        return np.column_stack([value, shifted_past])

    def get_feature_names_out(self, input_features=None):
        name = input_features[0] if input_features is not None and len(input_features) else "x0"
        if not self.keep_value:
            return np.asarray([f"{name}_past_{self.kind}"], dtype=object)
        return np.asarray([name, f"{name}_past_{self.kind}"], dtype=object)
