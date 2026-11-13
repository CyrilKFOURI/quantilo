# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false
# pyright: reportReturnType=false
# (pandas Series and their masks: the stubs cannot follow them)
"""Anchors (126, Explain › Advanced › More; Ribeiro, Singh and Guestrin 2018): for one row, a rule
"IF tenure ≤ 9 AND Contract = Month-to-month THEN the model predicts Yes" that holds on at least
95 % of the rows that share its conditions, and the share of the rows it covers. A written rule,
read without a chart: the closest to the app's own way of saying things.

R-XAI-14:

- conditions: one per column, the row's own interval among the column's quartiles for a number
  (`a ≤ q1`, `q1 < a ≤ q2`, …, `a > q3`), its value for a category or a yes/no, "is empty" for a
  missing value; free text, dates and identifiers are left out;
- "similar rows": real rows drawn from the evaluation rows, whose anchored columns are drawn again
  among the real values that meet the condition (the other columns keep their own values), as
  the tabular sampler of the paper;
- search: a beam of 3 rules grown one condition at a time, up to 4 conditions (1 to 6); 400
  samples per candidate, 1,600 more for those whose precision reaches the threshold; a rule is
  kept when the lower end of its 95 % Wilson interval reaches the threshold (0.95; 0.80 to 0.99),
  so 400 lucky draws do not make a rule; among the kept rules of the smallest size, the one that
  covers the most rows wins;
- a number predicted: "the same outcome" is a prediction within ± 10 % of the spread (5th to 95th
  percentile) of the predictions.

When no rule of that size reaches the threshold, the most precise one is returned with
`reached = False`, and the sentence says so. numpy and the model's `predict`: Pyodide."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from quantilo.ml.data import Field
from quantilo.ml.target import Task

SEED = 42
THRESHOLD = 0.95
MIN_THRESHOLD, MAX_THRESHOLD = 0.80, 0.99
MAX_CONDITIONS = 4
LIMIT_CONDITIONS = 6
BEAM = 3
SAMPLES = 400
MORE_SAMPLES = 1_600
MAX_COLUMNS = 30
BAND = 0.10  # regression: ± this share of the predictions' 5th-95th percentile spread
SPREAD_ROWS = 2_000
Z = 1.96


class AnchorError(ValueError):
    """No anchor can be searched for this row (with the reason)."""


@dataclass
class AnchorCondition:
    column: str
    text: str  # "tenure ≤ 9"
    value: str  # the row's own value


@dataclass
class Anchor:
    row: int
    predicted: str  # what the model gives the row
    outcome: str  # what "the same outcome" means
    conditions: list[AnchorCondition]
    rule: str
    precision: float
    precision_low: float  # lower end of the 95 % Wilson interval
    coverage: float  # share of the evaluation rows that meet every condition
    samples: int  # samples the precision was measured on
    reached: bool
    threshold: float
    evaluated: int  # rules tried
    sentence: str
    left_out: list[str] = field(default_factory=list)


@dataclass
class _Condition:
    column: str
    text: str
    value: str
    holds: Callable[[pd.Series], np.ndarray]
    pool: np.ndarray  # real values that meet the condition


def wilson_low(successes: int, n: int, z: float = Z) -> float:
    if n == 0:
        return 0.0
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / (1 + z * z / n))


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _label(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "(empty)"
    if isinstance(value, float | np.floating):
        return _fmt(float(value))
    return str(value)[:40]


def _number(raw: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_bool_dtype(raw)


def _condition(raw: pd.Series, column: str, value: Any) -> _Condition | None:
    """The row's condition on one column (R-XAI-14), or None when the column says nothing."""
    if pd.isna(value):
        mask = raw.isna().to_numpy()
        if not mask.any():
            return None
        return _Condition(
            column, f"{column} is empty", "(empty)", lambda s: s.isna().to_numpy(), raw[mask].values
        )
    if _number(raw):
        values = raw.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        edges = np.unique(np.quantile(finite, [0.25, 0.5, 0.75])) if len(finite) else []
        if len(edges) == 0 or float(finite.min()) == float(finite.max()):
            return None
        v = float(value)
        if v <= edges[0]:
            low, high, text = -np.inf, float(edges[0]), f"{column} ≤ {_fmt(edges[0])}"
        elif v > edges[-1]:
            low, high, text = float(edges[-1]), np.inf, f"{column} > {_fmt(edges[-1])}"
        else:
            k = int(np.searchsorted(edges, v, side="left"))
            low, high = float(edges[k - 1]), float(edges[k])
            text = f"{_fmt(low)} < {column} ≤ {_fmt(high)}"

        def holds(s: pd.Series, low: float = low, high: float = high) -> np.ndarray:
            numbers = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            return (numbers > low) & (numbers <= high)

        pool = raw[holds(raw)].to_numpy()
        return _Condition(column, text, _label(value), holds, pool) if len(pool) else None
    wanted = str(value)

    def same(s: pd.Series, wanted: str = wanted) -> np.ndarray:
        return (s.astype(str) == wanted).to_numpy() & s.notna().to_numpy()

    pool = raw[same(raw)].to_numpy()
    return _Condition(column, f"{column} = {wanted[:40]}", wanted[:40], same, pool)


def _skipped(x: pd.DataFrame, fields: list[Field], texts: set[str]) -> dict[str, str]:
    by_name = {f.name: f for f in fields}
    left: dict[str, str] = {}
    for column in map(str, x.columns):
        info = by_name.get(column)
        raw = cast(pd.Series, x[column])
        if column in texts or (info is not None and info.type == "text"):
            left[column] = "free text"
        elif pd.api.types.is_datetime64_any_dtype(raw) or (info and info.type == "datetime"):
            left[column] = "a date"
        elif info is not None and info.type == "identifier":
            left[column] = "an identifier"
    return left


class _Outcome:
    """Whether the model gives a frame's rows the same outcome as the explained row."""

    def __init__(
        self,
        pipeline: Pipeline,
        task: Task,
        row: pd.DataFrame,
        x: pd.DataFrame,
        threshold: float | None,
        positive: str | None,
    ) -> None:
        self.pipeline, self.task = pipeline, task
        self.classes = [str(c) for c in getattr(pipeline, "classes_", [])]
        self.threshold, self.positive = threshold, positive
        if task == "regression":
            self.value = float(np.asarray(pipeline.predict(row), dtype=float)[0])
            sample = x.iloc[: min(len(x), SPREAD_ROWS)]
            spread = np.quantile(np.asarray(pipeline.predict(sample), dtype=float), [0.05, 0.95])
            self.band = max(BAND * float(spread[1] - spread[0]), 1e-12)
            self.predicted = _fmt(self.value)
            self.outcome = f"a prediction within {_fmt(self.value)} ± {self.band:.3g}"
            return
        self.label = self._decide(row)[0]
        self.predicted = self.label
        self.outcome = f"the class “{self.label}”"

    def _decide(self, frame: pd.DataFrame) -> np.ndarray:
        if len(self.classes) == 2 and self.threshold is not None and self.positive in self.classes:
            p = np.asarray(self.pipeline.predict_proba(frame), dtype=float)
            p = p[:, self.classes.index(self.positive)]
            other = next(c for c in self.classes if c != self.positive)
            return np.where(p >= self.threshold, self.positive, other)
        return np.asarray(self.pipeline.predict(frame)).astype(str)

    def same(self, frame: pd.DataFrame) -> np.ndarray:
        if self.task == "regression":
            predicted = np.asarray(self.pipeline.predict(frame), dtype=float)
            return np.abs(predicted - self.value) <= self.band
        return self._decide(frame) == self.label


class _Search:
    """Precision of candidate rules on shared draws (common random numbers: two rules differ by
    their conditions, not by their luck)."""

    def __init__(
        self, x: pd.DataFrame, conditions: list[_Condition], outcome: _Outcome, seed: int = SEED
    ):
        rng = np.random.default_rng(seed)
        total = SAMPLES + MORE_SAMPLES
        self.base = x.iloc[rng.integers(0, len(x), total)].reset_index(drop=True)
        self.draws = [c.pool[rng.integers(0, len(c.pool), total)] for c in conditions]
        self.conditions, self.outcome, self.x = conditions, outcome, x
        self.hits: dict[tuple[int, ...], tuple[int, int]] = {}

    def _frame(self, rule: tuple[int, ...], start: int, stop: int) -> pd.DataFrame:
        frame = self.base.iloc[start:stop].copy()
        for i in rule:
            frame[self.conditions[i].column] = self.draws[i][start:stop]
        return frame

    def measure(self, rules: list[tuple[int, ...]], start: int, stop: int) -> None:
        if not rules:
            return
        frame = pd.concat([self._frame(r, start, stop) for r in rules], ignore_index=True)
        same = self.outcome.same(frame).reshape(len(rules), stop - start)
        for rule, row in zip(rules, same, strict=True):
            hits, n = self.hits.get(rule, (0, 0))
            self.hits[rule] = (hits + int(row.sum()), n + stop - start)

    def precision(self, rule: tuple[int, ...]) -> float:
        hits, n = self.hits[rule]
        return hits / n if n else 0.0

    def low(self, rule: tuple[int, ...]) -> float:
        return wilson_low(*self.hits[rule])

    def coverage(self, rule: tuple[int, ...]) -> float:
        mask = np.ones(len(self.x), dtype=bool)
        for i in rule:
            c = self.conditions[i]
            mask &= c.holds(cast(pd.Series, self.x[c.column]))
        return float(mask.mean()) if len(mask) else 0.0


def anchor(
    pipeline: Pipeline,
    x: pd.DataFrame,
    rows: np.ndarray,
    task: Task,
    row: int,
    fields: list[Field],
    texts: set[str] | None = None,
    ranked: list[str] | None = None,
    threshold: float = THRESHOLD,
    max_conditions: int = MAX_CONDITIONS,
    decision: float | None = None,
    positive: str | None = None,
) -> Anchor:
    """`x`: the evaluation rows (their file rows in `rows`); `row`: the file row to explain;
    `ranked`: the inputs by importance (the beam looks at the 30 first); `decision` and
    `positive`: a two-class run's own threshold."""
    where = np.flatnonzero(rows == row)
    if not len(where):
        raise AnchorError("This row is not among the rows the model was scored on.")
    threshold = float(min(max(threshold, MIN_THRESHOLD), MAX_THRESHOLD))
    max_conditions = int(min(max(max_conditions, 1), LIMIT_CONDITIONS))
    original = x.iloc[[int(where[0])]].reset_index(drop=True)
    left = _skipped(x, fields, texts or set())
    order = [
        str(c) for c in dict.fromkeys([*(ranked or []), *map(str, x.columns)]) if c in x.columns
    ]
    conditions: list[_Condition] = []
    for column in order:
        if column in left or len(conditions) >= MAX_COLUMNS:
            continue
        found = _condition(cast(pd.Series, x[column]), column, original.iloc[0][column])
        if found is not None:
            conditions.append(found)
    if not conditions:
        raise AnchorError("No column of this row can make a condition (text, dates, constants).")
    outcome = _Outcome(pipeline, task, original, x, decision, positive)
    search = _Search(x, conditions, outcome)
    beam: list[tuple[int, ...]] = [()]
    best: tuple[int, ...] | None = None
    for _size in range(1, max_conditions + 1):
        candidates = sorted(
            {tuple(sorted((*b, i))) for b in beam for i in range(len(conditions)) if i not in b}
        )
        if not candidates:
            break
        search.measure(candidates, 0, SAMPLES)
        close = [r for r in candidates if search.precision(r) >= threshold]
        search.measure(close, SAMPLES, SAMPLES + MORE_SAMPLES)
        valid = [r for r in close if search.low(r) >= threshold]
        if valid:
            best = max(valid, key=lambda r: (search.coverage(r), search.precision(r)))
            break
        ranked_now = sorted(
            candidates, key=lambda r: (-search.precision(r), -search.coverage(r), r)
        )
        beam = ranked_now[:BEAM]
        if best is None or search.precision(beam[0]) > search.precision(best):
            best = beam[0]
    assert best is not None
    reached = search.low(best) >= threshold
    chosen = [conditions[i] for i in sorted(best, key=lambda i: order.index(conditions[i].column))]
    rule = "IF " + " AND ".join(c.text for c in chosen) + f" THEN {outcome.outcome}"
    precision, low, coverage = search.precision(best), search.low(best), search.coverage(best)
    _, samples = search.hits[best]
    if reached:
        sentence = (
            f"{rule}: the model gives {outcome.outcome} to {precision:.0%} of similar rows (95 % "
            f"sure of at least {low:.0%}), and the rule covers {coverage:.0%} of the rows."
        )
    else:
        sentence = (
            f"No rule of at most {max_conditions} conditions reaches {threshold:.0%}: the closest, "
            f"{rule}, holds on {precision:.0%} of similar rows and covers {coverage:.0%} of the "
            "rows. This row sits where the model's outcome depends on many columns at once."
        )
    return Anchor(
        row,
        outcome.predicted,
        outcome.outcome,
        [AnchorCondition(c.column, c.text, c.value) for c in chosen],
        rule,
        precision,
        low,
        coverage,
        samples,
        reached,
        threshold,
        len(search.hits),
        sentence,
        [f"{c} ({why})" for c, why in left.items()],
    )
