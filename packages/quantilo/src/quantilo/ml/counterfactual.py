# ruff: noqa: RUF002, RUF003  (→, ≥, ≤, − are the usual notation)
"""What would change this prediction? (125) For one row: at most three changes, on the columns
the user lets change, that bring the model to another class or past a value. Greedy: each step
tries every allowed column at every candidate value (the column's real values: 19 quantiles of a
number, the 20 most frequent categories), keeps the change that reaches the goal at the smallest
cost (or, failing that, comes closest), then each number changed is moved back towards the row's
own value as far as the goal still holds.

R-XAI-03: identifiers, dates, free text and personal attributes (age, gender, a customer or
order key, e-mail, phone, postcode: their role, 002) are locked by default: a decision cannot
change them. The user may unlock any of them.

Plausibility (in the spirit of NICE, Brughmans et al. 2023): the closest real row that the model
already gives the wanted outcome, by the Gower distance (numbers: gap / range; categories: 0 or
1; averaged). The change is plausible when it lands as close to that row as most real rows are to
their nearest neighbour (90th percentile); otherwise the combination it asks for is unusual."""

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from quantilo.ml.data import Field
from quantilo.ml.target import Task

SEED = 42
MAX_CHANGES = 3
QUANTILES = np.linspace(0.05, 0.95, 19)
TOP_CATEGORIES = 20
NEIGHBOUR_ROWS = 300  # rows whose nearest-neighbour distance sets what "close" means
LOCKED_ROLES = {
    "identifier": "an identifier",
    "age": "a personal attribute (age)",
    "gender": "a personal attribute (gender)",
    "customer_key": "an identifier (customer)",
    "order_key": "an identifier (order)",
    "email": "a personal contact (e-mail)",
    "phone": "a personal contact (phone)",
    "postal_code": "where the person lives (postcode)",
}


class CounterfactualError(ValueError):
    pass


@dataclass
class Editable:
    column: str
    editable: bool  # what the search may change
    locked: str | None  # R-XAI-03: why it is locked by default (None: free by default)


@dataclass
class Change:
    column: str
    before: str
    after: str
    output: float  # the model's output after this change (and the ones before it)


@dataclass
class Compared:
    column: str
    row: str
    changed: str
    nearest: str | None  # the closest real row with the wanted outcome


@dataclass
class Counterfactual:
    row: int
    goal: str
    output: str  # what `before` and `after` measure
    before: float
    after: float
    reached: bool
    changes: list[Change]
    sentence: str
    columns: list[Editable]
    table: list[Compared]
    nearest_row: int | None
    distance: float | None  # Gower distance of the changed row to that real row
    typical: float | None  # 90th percentile of the real rows' nearest-neighbour distance
    plausible: bool | None
    plausibility: str
    target_class: str | None = None
    target_value: float | None = None
    classes: list[str] = field(default_factory=list)


def _kind(raw: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(raw):
        return "date"
    if pd.api.types.is_bool_dtype(raw):
        return "category"
    if pd.api.types.is_numeric_dtype(raw):
        return "number"
    return "category"


def columns(x: pd.DataFrame, fields: list[Field], texts: set[str]) -> list[Editable]:
    """Every input of the model, editable or locked by default (R-XAI-03)."""
    by_name = {f.name: f for f in fields}
    found = []
    for column in [str(c) for c in x.columns]:
        info = by_name.get(column)
        reason = None
        if column in texts or (info is not None and info.type == "text"):
            reason = "free text"
        elif info is not None and info.type == "identifier":
            reason = "an identifier"
        elif _kind(cast(pd.Series, x[column])) == "date" or (info and info.type == "datetime"):
            reason = "a date"
        elif info is not None and info.role in LOCKED_ROLES:
            reason = LOCKED_ROLES[info.role]
        found.append(Editable(column, reason is None, reason))
    return found


def _label(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "(empty)"
    if isinstance(value, float | np.floating):
        return f"{float(value):.4g}"
    return str(value)[:40]


class _Goal:
    """The model's output as a score to raise, and whether a row reaches the goal."""

    def __init__(
        self,
        pipeline: Pipeline,
        task: Task,
        row: pd.DataFrame,
        target_class: str | None,
        target_value: float | None,
        threshold: float | None,
        positive: str | None,
        reference: pd.DataFrame,
    ) -> None:
        self.pipeline, self.task = pipeline, task
        self.classes = [str(c) for c in getattr(pipeline, "classes_", [])]
        self.target_class: str | None = None
        self.target_value: float | None = None
        if task == "regression":
            now = float(np.asarray(pipeline.predict(row), dtype=float)[0])
            if target_value is None:  # the other side of the typical predictions
                spread = np.asarray(pipeline.predict(reference), dtype=float)
                low, middle, high = np.quantile(spread, [0.25, 0.5, 0.75])
                target_value = float(high if now <= middle else low)
            self.target_value = float(target_value)
            self.up = self.target_value >= now
            self.output = "prediction"
            way = "at least" if self.up else "at most"
            self.goal = f"a prediction of {way} {self.target_value:.4g}"
            return
        probabilities = np.asarray(pipeline.predict_proba(row), dtype=float)[0]
        current = self._decide(probabilities[None], threshold, positive)[0]
        if target_class is None or target_class not in self.classes:
            ranked = [self.classes[i] for i in np.argsort(-probabilities)]
            target_class = next(c for c in ranked if c != current)
        self.target_class = target_class
        self.index = self.classes.index(str(target_class))
        self.threshold, self.positive = threshold, positive
        self.output = f"probability of “{target_class}”"
        self.goal = f"the class “{target_class}”"

    def _decide(self, probabilities: np.ndarray, threshold, positive) -> list[str]:
        if len(self.classes) == 2 and threshold is not None and positive in self.classes:
            p = probabilities[:, self.classes.index(positive)]
            other = next(c for c in self.classes if c != positive)
            return [positive if v >= threshold else other for v in p]
        return [self.classes[i] for i in probabilities.argmax(axis=1)]

    def evaluate(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Output shown, score to raise, goal reached — one per row of `frame`."""
        if self.task == "regression":
            predicted = np.asarray(self.pipeline.predict(frame), dtype=float)
            assert self.target_value is not None
            score = predicted if self.up else -predicted
            reached = predicted >= self.target_value if self.up else predicted <= self.target_value
            return predicted, score, reached
        probabilities = np.asarray(self.pipeline.predict_proba(frame), dtype=float)
        decided = np.array(self._decide(probabilities, self.threshold, self.positive))
        wanted = probabilities[:, self.index]
        return wanted, wanted, decided == self.target_class


def _candidates(x: pd.DataFrame, column: str) -> list[Any]:
    raw = cast(pd.Series, x[column])
    if _kind(raw) == "number":
        values = raw.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return []
        grid = np.unique(np.quantile(values, QUANTILES))
        if pd.api.types.is_integer_dtype(raw) or np.allclose(values, np.round(values)):
            grid = np.unique(np.round(grid))
        return [float(v) for v in grid]
    return list(raw.dropna().value_counts().index[:TOP_CATEGORIES])


class _Distance:
    """Gower distance between rows: numbers |a − b| / range, categories 0 or 1, averaged."""

    def __init__(self, x: pd.DataFrame, used: list[str]) -> None:
        self.used = used
        self.numbers = [c for c in used if _kind(cast(pd.Series, x[c])) == "number"]
        self.categories = [c for c in used if c not in self.numbers]
        values = x[self.numbers].to_numpy(dtype=float) if self.numbers else np.zeros((len(x), 0))
        spread = np.nanmax(values, axis=0) - np.nanmin(values, axis=0) if len(x) else np.zeros(0)
        self.range = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)

    def matrix(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numbers = frame[self.numbers].to_numpy(dtype=float) / self.range
        labels = (
            frame[self.categories].astype(str).to_numpy()
            if self.categories
            else np.zeros((len(frame), 0))
        )
        return numbers, labels

    def to_rows(self, one: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
        a_num, a_cat = self.matrix(one)
        b_num, b_cat = self.matrix(rows)
        gaps = np.abs(b_num - a_num)
        gaps = np.where(np.isnan(gaps), 1.0, gaps)  # an empty value differs from any value
        mismatch = (b_cat != a_cat).astype(float)
        total = gaps.sum(axis=1) + mismatch.sum(axis=1)
        return total / max(len(self.used), 1)


def counterfactual(
    pipeline: Pipeline,
    x: pd.DataFrame,
    rows: np.ndarray,
    task: Task,
    row: int,
    fields: list[Field],
    texts: set[str],
    editable: list[str] | None = None,
    target_class: str | None = None,
    target_value: float | None = None,
    max_changes: int = MAX_CHANGES,
    threshold: float | None = None,
    positive: str | None = None,
) -> Counterfactual:
    """`x`: the evaluation rows (their file rows in `rows`); `row`: the file row to change."""
    where = np.flatnonzero(rows == row)
    if not len(where):
        raise CounterfactualError("This row is not among the rows the model was scored on.")
    listed = columns(x, fields, texts)
    if editable is not None:
        allowed = set(editable)
        listed = [
            Editable(c.column, c.column in allowed and c.column not in texts, c.locked)
            for c in listed
        ]
    free = [c.column for c in listed if c.editable]
    max_changes = int(min(max(max_changes, 1), MAX_CHANGES))
    original = x.iloc[[int(where[0])]].reset_index(drop=True)
    goal = _Goal(pipeline, task, original, target_class, target_value, threshold, positive, x)
    before, _, reached_now = goal.evaluate(original)
    if bool(reached_now[0]) or not free:
        said = (
            f"This row already gets {goal.goal}: nothing to change."
            if bool(reached_now[0])
            else "No column may change: unlock at least one."
        )
        return _result(
            goal,
            x,
            rows,
            row,
            original,
            original,
            [],
            float(before[0]),
            float(before[0]),
            bool(reached_now[0]),
            listed,
            said,
        )
    numbers = {c for c in free if _kind(cast(pd.Series, x[c])) == "number"}
    spread = {c: _scale(cast(pd.Series, x[c])) for c in numbers}
    candidates = {c: _candidates(x, c) for c in free}
    current = original.copy()
    changed: list[str] = []
    steps: list[Change] = []
    output, reached = float(before[0]), False
    for _ in range(max_changes):
        trials, labels = [], []
        for column in free:
            if column in changed:
                continue
            for value in candidates[column]:
                if _label(value) == _label(current.iloc[0][column]):
                    continue
                trial = current.copy()
                trial[column] = [value]
                trials.append(trial)
                labels.append((column, value))
        if not trials:
            break
        frame = pd.concat(trials, ignore_index=True)
        frame = frame.astype(
            {c: x[c].dtype for c in frame.columns if c in x.columns}, errors="ignore"
        )
        shown, score, done = goal.evaluate(frame)
        costs = np.array([_cost(c, v, original, spread) for c, v in labels])
        if done.any():
            pick = int(np.flatnonzero(done)[np.argmin(costs[done])])
        else:
            pick = int(np.argmax(score - 1e-6 * costs))
            if score[pick] <= goal.evaluate(current)[1][0] + 1e-12:
                break  # nothing moves the model the right way any more
        column, value = labels[pick]
        current[column] = [value]
        current = current.astype({column: x[column].dtype}, errors="ignore")
        changed.append(column)
        output, reached = float(shown[pick]), bool(done[pick])
        steps.append(Change(column, _label(original.iloc[0][column]), _label(value), output))
        if reached:
            break
    if reached:
        current, steps, output = _closer(goal, original, current, steps, candidates, x)
    return _result(
        goal, x, rows, row, original, current, steps, float(before[0]), output, reached, listed
    )


def _scale(raw: pd.Series) -> float:
    values = raw.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 1.0
    low, high = np.quantile(values, [0.25, 0.75])
    width = float(high - low) or float(values.max() - values.min())
    return width or 1.0


def _cost(column: str, value: Any, original: pd.DataFrame, spread: dict[str, float]) -> float:
    """How big a change is: a number in interquartile ranges, a category counts 1."""
    if column in spread:
        before = original.iloc[0][column]
        if pd.isna(before):
            return 1.0
        return abs(float(value) - float(before)) / spread[column]
    return 1.0


def _closer(goal, original, current, steps, candidates, x):
    """Move each changed number back towards the row's own value while the goal holds."""
    for step in steps:
        column = step.column
        if column not in x.columns or _kind(cast(pd.Series, x[column])) != "number":
            continue
        start = original.iloc[0][column]
        if pd.isna(start):
            continue
        target = float(current.iloc[0][column])
        between = [
            v
            for v in candidates[column]
            if min(float(start), target) <= v <= max(float(start), target) and v != target
        ]
        if not between:
            continue
        trials = []
        for value in between:
            trial = current.copy()
            trial[column] = [value]
            trials.append(trial)
        _, _, done = goal.evaluate(pd.concat(trials, ignore_index=True))
        if not done.any():
            continue
        kept = [v for v, ok in zip(between, done, strict=True) if ok]
        best = min(kept, key=lambda v: abs(v - float(start)))
        current = current.copy()
        current[column] = [best]
        step.after = _label(best)
    final = goal.evaluate(current)[0]
    output = float(final[0])
    if steps:
        steps[-1].output = output
    return current, steps, output


def _result(
    goal, x, rows, row, original, current, steps, before, output, reached, listed, said=None
):
    used = [str(c) for c in x.columns if _kind(cast(pd.Series, x[c])) != "date"]
    distance = _Distance(x, used)
    _, _, meets = goal.evaluate(x)
    nearest_row = gap = typical = None
    plausible: bool | None = None
    candidates_at = np.flatnonzero(meets)
    if len(candidates_at):
        gaps = distance.to_rows(current, x.iloc[candidates_at])
        best = int(np.argmin(gaps))
        nearest_at = int(candidates_at[best])
        nearest_row, gap = int(rows[nearest_at]), float(gaps[best])
        typical = _typical(distance, x)
        plausible = bool(reached and typical is not None and gap <= typical)
    if said is not None:
        sentence = said
    elif not reached:
        sentence = (
            f"No change of at most {len(steps) or 1} allowed column(s) brings this row to "
            f"{goal.goal}; the closest it gets: {goal.output} {output:.3g} (from {before:.3g})."
        )
    else:
        moves = "; ".join(f"{s.column} {s.before} → {s.after}" for s in steps)
        sentence = (
            f"{moves}: the model gives {goal.goal} ({goal.output} {before:.3g} → {output:.3g})."
        )
    if nearest_row is None:
        plausibility = f"No real row gets {goal.goal} from this model: nothing to compare with."
    elif plausible:
        plausibility = (
            f"Plausible: the closest real row with that outcome (row {nearest_row + 1}) is at "
            f"distance {gap:.2f}, as close as most real rows are to their nearest neighbour "
            f"(90 % are within {typical:.2f})."
        )
    elif reached:
        plausibility = (
            f"Unusual: the closest real row with that outcome (row {nearest_row + 1}) is at "
            f"distance {gap:.2f}, farther than 90 % of real rows are from their nearest "
            f"neighbour ({typical:.2f}): the combination asked for is rare in the data."
        )
    else:
        plausibility = (
            f"The closest real row with that outcome is row {nearest_row + 1} (distance "
            f"{gap:.2f}): see what differs in the table."
        )
    changed = {s.column for s in steps}
    nearest = (
        x.iloc[int(np.flatnonzero(rows == nearest_row)[0])] if nearest_row is not None else None
    )
    table = [
        Compared(
            c.column,
            _label(original.iloc[0][c.column]),
            _label(current.iloc[0][c.column]),
            _label(nearest[c.column]) if nearest is not None else None,
        )
        for c in sorted(listed, key=lambda c: (c.column not in changed, not c.editable))
    ]
    return Counterfactual(
        int(row),
        f"Reach {goal.goal}",
        goal.output,
        before,
        output,
        reached,
        steps,
        sentence,
        listed,
        table,
        nearest_row,
        gap,
        typical,
        plausible,
        plausibility,
        goal.target_class,
        goal.target_value,
        goal.classes,
    )


def _typical(distance: _Distance, x: pd.DataFrame) -> float | None:
    """90th percentile of the distance of a real row to its nearest other real row."""
    if len(x) < 3:
        return None
    pick = np.random.default_rng(SEED).choice(len(x), min(NEIGHBOUR_ROWS, len(x)), replace=False)
    found = []
    for i in pick:
        gaps = distance.to_rows(x.iloc[[int(i)]], x)
        gaps[int(i)] = np.inf
        found.append(float(gaps.min()))
    return float(np.quantile(found, 0.9))
