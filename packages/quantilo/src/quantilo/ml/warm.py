"""Warm start of a search (066): the first trials are the best settings already found for this
model on this dataset, its cleaned versions or a dataset that looks alike.

Resemblance is measured on three meta-features, each with its reason:
- the kind of target (number or classes): settings tuned for one rarely suit the other, so a
  different kind is never used (R-ML-WARM-01);
- the number of rows, on a log scale: depth, leaves and learning rates scale with the rows;
- the share of category columns: it changes how much trees split on encoded levels.

Rules:
- R-ML-WARM-01: same model, same kind of target, else not a candidate.
- R-ML-WARM-02: distance = |log10 rows - log10 rows'| + 2 x |categorical share - share'|; a dataset
  other than this one and its versions counts only within 1.0 (ten times the rows, or half the
  columns changing kind, is too far).
- R-ML-WARM-03: at most 3 starting points, this dataset first, then its versions, then the
  closest datasets; a setting outside the search space is brought back to its bounds, a choice
  outside the space's choices is dropped.
- R-ML-WARM-04: the grid sampler walks its own grid: no warm start.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from quantilo.ml.tune import SearchParam

MAX_STARTS = 3  # R-ML-WARM-03
MAX_DISTANCE = 1.0  # R-ML-WARM-02
CATEGORICAL_WEIGHT = 2.0

Source = Literal["this dataset", "a cleaned version", "a similar dataset"]
ORDER: dict[str, int] = {"this dataset": 0, "a cleaned version": 1, "a similar dataset": 2}


@dataclass
class Meta:
    """What a dataset looks like to a model."""

    rows: int
    categorical_share: float
    kind: Literal["number", "classes"]


@dataclass
class Candidate:
    """Settings already found: a saved run or the best of a finished search."""

    model: str
    settings: dict[str, Any]
    score: float | None
    greater_is_better: bool
    meta: Meta
    source: Source
    dataset: str = ""  # its name, said with the trial


@dataclass
class Start:
    """A starting point of the search, and why it was chosen."""

    params: dict[str, Any]
    reason: str
    distance: float = 0.0
    source: Source = "this dataset"
    changed: list[str] = field(default_factory=list)  # settings brought back to the space


def kind_of(task: str) -> Literal["number", "classes"]:
    return "number" if task == "regression" else "classes"


def distance(a: Meta, b: Meta) -> float:
    """R-ML-WARM-02; infinite when the kind of target differs (R-ML-WARM-01)."""
    if a.kind != b.kind:
        return math.inf
    rows = abs(math.log10(max(a.rows, 1)) - math.log10(max(b.rows, 1)))
    return rows + CATEGORICAL_WEIGHT * abs(a.categorical_share - b.categorical_share)


def _within(params: dict[str, Any], space: list[SearchParam]) -> tuple[dict[str, Any], list[str]]:
    """The candidate's searched settings, held inside the space (R-ML-WARM-03)."""
    kept: dict[str, Any] = {}
    changed: list[str] = []
    for param in space:
        if param.fixed is not None or param.key not in params:
            continue
        value = params[param.key]
        if param.kind == "categorical":
            if str(value) in param.choices:
                kept[param.key] = str(value)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        low, high = float(param.low or 0), float(param.high or 1)
        bounded = min(max(number, low), high)
        if bounded != number:
            changed.append(param.key)
        kept[param.key] = round(bounded) if param.kind == "int" else bounded
    return kept, changed


def starts(
    model: str, meta: Meta, candidates: list[Candidate], space: list[SearchParam]
) -> list[Start]:
    """The starting points of a search of `model` on a dataset described by `meta`."""
    ranked: list[tuple[int, float, float, Candidate]] = []
    for c in candidates:
        if c.model != model:
            continue
        gap = 0.0 if c.source != "a similar dataset" else distance(meta, c.meta)
        if c.meta.kind != meta.kind or gap > MAX_DISTANCE:
            continue
        signed = c.score if c.score is not None and math.isfinite(c.score) else -math.inf
        quality = signed if c.greater_is_better or signed == -math.inf else -signed
        ranked.append((ORDER[c.source], gap, -quality, c))
    ranked.sort(key=lambda item: item[:3])
    found: list[Start] = []
    seen: set[str] = set()
    for _, gap, _, c in ranked:
        params, changed = _within(c.settings, space)
        if not params:
            continue
        key = repr(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        found.append(Start(params, _reason(c, gap), gap, c.source, changed))
        if len(found) == MAX_STARTS:
            break
    return found


def _reason(candidate: Candidate, gap: float) -> str:
    where = f" ({candidate.dataset})" if candidate.dataset else ""
    if candidate.source == "a similar dataset":
        return (
            f"Best settings found on a similar dataset{where}: {candidate.meta.rows:,} rows, "
            f"{candidate.meta.categorical_share:.0%} category columns (distance {gap:.2f})."
        )
    return f"Best settings found on {candidate.source}{where}."


def meta_of(types: dict[str, str], target: str, rows: int, task: str) -> Meta:
    """The meta-features of a dataset from its column types (no data read): category columns
    among the numbers and categories other than the target."""
    from quantilo.ml.data import CATEGORICAL, NUMERIC

    usable = [
        t for name, t in types.items() if name != target and (t in NUMERIC or t in CATEGORICAL)
    ]
    categorical = [t for t in usable if t in CATEGORICAL]
    share = len(categorical) / len(usable) if usable else 0.0
    return Meta(rows, share, kind_of(task))
