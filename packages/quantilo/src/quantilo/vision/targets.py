"""What an image model can predict besides a class (c3; study § 6.2, R-VML-05 and R-VML-06).

- **A number** (R-VML-05): a numeric metadata column (one the import did not make, nor our edits,
  R-VED-04) with at least `MIN_KNOWN` known values and `MIN_DISTINCT` distinct ones, typically a
  column of the labels file (weight, count, age). Offered in Advanced, never the rule's task: a
  numeric column is not always what one wants to predict.
- **Several labels** (R-VML-06): labels joined by `|` (`cat|outdoor`) on at least 2 usable images,
  or 2 true/false metadata columns or more. Each label needs `MIN_PER_LABEL` images with it and as
  many without it (R-VMLB-02), the others are left out and said.

Pure functions on the image table: the browser version reads them too.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from quantilo.vision.task import Table, column, n_rows

MIN_KNOWN = 20  # R-VML-05: known values of the column, among usable images
MIN_DISTINCT = 10  # R-VML-05: fewer distinct values read better as classes
MIN_PER_LABEL = 5  # R-VMLB-02: images with the label, and without it
MIN_PIPE_IMAGES = 2  # R-VML-06: images whose label holds several labels
SEPARATOR = "|"


def _made() -> set[str]:
    """Columns written by the import (§ 3.4) and by the user's edits (R-VED-04)."""
    from quantilo.vision.edits import DERIVED
    from quantilo.vision.ingest import COLUMNS

    return set(COLUMNS) | set(DERIVED) | {"label", "split", "is_normal", "mask_path"}


def _names(data: Table) -> list[str]:
    if isinstance(data, Mapping):
        return list(data)
    return list(getattr(data, "column_names", []))


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def numeric_targets(data: Table, usable: np.ndarray) -> list[str]:
    """R-VML-05: the numeric metadata columns a regression can learn, in table order."""
    made = _made()
    out = []
    for name in _names(data):
        if name in made or name.startswith("_"):
            continue
        values = column(data, name)
        known = [_number(v) for i, v in enumerate(values) if usable[i]]
        numbers = [v for v in known if v is not None]
        raw = [v for i, v in enumerate(values) if usable[i] and v is not None]
        if not numbers or len(numbers) < len(raw):  # text or booleans mixed in
            continue
        if len(numbers) >= MIN_KNOWN and len(set(numbers)) >= MIN_DISTINCT:
            out.append(name)
    return out


_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


def _flag(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        return None
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def binary_targets(data: Table, usable: np.ndarray) -> list[str]:
    """R-VML-06: true/false metadata columns (booleans, 0/1, yes/no), in table order."""
    made = _made()
    out = []
    for name in _names(data):
        if name in made or name.startswith("_"):
            continue
        values = [v for i, v in enumerate(column(data, name)) if usable[i] and v is not None]
        if not values:
            continue
        flags = [_flag(v) for v in values]
        if any(f is None for f in flags):
            continue
        if any(flags) and not all(flags):
            out.append(name)
    return out


def pipe_images(labels: Sequence[str | None], usable: np.ndarray) -> int:
    """Usable images whose label holds several labels joined by `|`."""
    return sum(
        1 for i, lab in enumerate(labels) if usable[i] and lab is not None and SEPARATOR in lab
    )


@dataclass
class LabelSets:
    """The multi-label target: one row per table row, one column per label kept."""

    source: Literal["pipe", "columns"]
    names: list[str]  # the labels kept (columns of `y`)
    y: np.ndarray  # (n_rows, len(names)) bool
    known: np.ndarray  # (n_rows,) bool: the row has a known set of labels
    left_out: dict[str, int]  # labels with too few images (with or without), and their images


def label_sets(
    data: Table,
    usable: np.ndarray,
    labels: Sequence[str | None],
    columns: Sequence[str] = (),
) -> LabelSets:
    """R-VML-06: the labels of each image, from `|` in the labels (when some image has several)
    or from the true/false `columns`; rare labels left out (R-VMLB-02)."""
    n = n_rows(data)
    known = np.zeros(n, dtype=bool)
    if pipe_images(labels, usable) >= MIN_PIPE_IMAGES or not columns:
        source: Literal["pipe", "columns"] = "pipe"
        sets: list[set[str]] = []
        for i, lab in enumerate(labels):
            tokens = (
                {t.strip() for t in lab.split(SEPARATOR) if t.strip()} if lab is not None else set()
            )
            known[i] = bool(usable[i] and tokens)
            sets.append(tokens)
        names = sorted({t for i, s in enumerate(sets) if known[i] for t in s})
        y = np.array([[name in s for name in names] for s in sets], dtype=bool).reshape(
            n, len(names)
        )
    else:
        source = "columns"
        names = list(columns)
        flags = [[_flag(v) for v in column(data, name)] for name in names]
        y = np.zeros((n, len(names)), dtype=bool)
        for i in range(n):
            row = [f[i] for f in flags]
            known[i] = bool(usable[i]) and all(v is not None for v in row)
            y[i] = [bool(v) for v in row]
    positives = y[known].sum(axis=0) if len(names) else np.zeros(0)
    negatives = known.sum() - positives
    keep = [
        j
        for j in range(len(names))
        if positives[j] >= MIN_PER_LABEL and negatives[j] >= MIN_PER_LABEL
    ]
    left_out = {names[j]: int(positives[j]) for j in range(len(names)) if j not in keep}
    return LabelSets(source, [names[j] for j in keep], y[:, keep], known, left_out)


def regression_values(data: Table, target: str) -> np.ndarray:
    """R-VML-05: the target of each row, NaN when unknown."""
    return np.array(
        [np.nan if (v := _number(x)) is None else v for x in column(data, target)], dtype=float
    )


__all__ = [
    "MIN_DISTINCT",
    "MIN_KNOWN",
    "MIN_PER_LABEL",
    "MIN_PIPE_IMAGES",
    "SEPARATOR",
    "LabelSets",
    "binary_targets",
    "label_sets",
    "numeric_targets",
    "pipe_images",
    "regression_values",
]
