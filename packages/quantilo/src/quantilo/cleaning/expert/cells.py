"""E-01 / E-02: cell errors learnt from a few marks, in the spirit of Raha and Baran (C-37).

Each cell of the chosen column becomes a vector (dBoost and Auto-Detect's signals: how far a number
is, in value and in order of magnitude; how rare the value and its pattern `A/a/9` are; its length;
whether it is empty, close to a much more frequent spelling, or against the value another column
gives). Cells are grouped (k-means); one cell per group is offered to mark *Error* or *OK*. Marks
spread to their group, a random forest learns from them, and every cell like the errors is
proposed, each with a correction (Baran-lite): the value a column that determines this one gives,
else a frequent close spelling, else empty."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as default

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from quantilo.catalog import ColumnView
from quantilo.cleaning import model as m
from quantilo.cleaning import phonetic as ph
from quantilo.cleaning.expert import data as d
from quantilo.cleaning.suggestion import CleanSuggestion

RULE = "E-01"
TO_MARK = 20
MIN_ROWS = 20
FD_SHARE = 0.95  # C-37: a column that determines this one on this share of the rows
FD_GAIN = 0.8  # … and removes this share of the errors of guessing the most frequent value
FD_GROUP = 3
FD_DISTINCT = 5_000
NEAR_RATIO = 10
NEAR_DISTINCT = 2_000
TRAIN_CELLS = 20_000
FEATURES = ("empty", "rare", "pattern", "length", "far", "magnitude", "near", "dependency")
HINTS = {
    "empty": "empty",
    "rare": "a rare value",
    "pattern": "an unusual pattern",
    "length": "an unusual length",
    "far": "far from the other values",
    "magnitude": "another order of magnitude",
    "near": "close to a much more frequent value",
    "dependency": "disagrees with {column}",
}
RUNS = re.compile(r"(.)\1+")


@dataclass
class Mark:
    row: int
    error: bool


@dataclass
class ToMark:
    row: int
    value: str | None
    hint: str
    alike: int  # cells of its group


def columns(views: Sequence[ColumnView]) -> list[str]:
    return [f.name for f in d.usable(views, max_levels=None)]


def why_not(views: Sequence[ColumnView]) -> str | None:
    if d.rows_of(views) < MIN_ROWS:
        return f"Needs at least {MIN_ROWS} rows."
    if not columns(views):
        return "Needs a column of numbers or categories."
    return None


def pattern(value: str) -> str:
    """Auto-Detect's generalization: A for a capital, a for a small letter, 9 for a digit; runs
    of one class written once ('Paris' → 'Aa', '75011' → '9')."""
    out = []
    for char in value:
        if char.isdigit():
            out.append("9")
        elif char.isalpha():
            out.append("A" if char.isupper() else "a")
        else:
            out.append(char)
    return RUNS.sub(r"\1", "".join(out))


@dataclass
class Cells:
    """The chosen column's cells (keys, as written) and their vectors."""

    field: d.Field
    frame: d.Frame
    keys: list[str | None]  # what is compared: the folded category, or the number as text
    shown: list[str | None]  # as written
    x: np.ndarray
    determinant: str | None = None
    majority: dict[str, str] | None = None  # determinant's value → this column's usual key
    spelling: dict[str, str] | None = None  # key → its most frequent spelling
    near: dict[str, str] = default(default_factory=dict)  # rare key → close frequent key

    def __len__(self) -> int:
        return len(self.keys)


def _keys(field: d.Field, frame: d.Frame) -> tuple[list[str | None], list[str | None]]:
    values = frame.data[field.name]
    if field.kind == "number":
        keys = [None if math.isnan(v) else d.number_text(v) for v in values.to_numpy(dtype=float)]
        return keys, list(keys)
    shown = [None if v is None else str(v) for v in frame.text[field.name]]
    return [None if v is None else str(v) for v in values], shown


def _determinant(
    keys: list[str | None], frame: d.Frame, others: Sequence[d.Field]
) -> tuple[str | None, dict[str, str] | None]:
    """The category that determines this column best (C-37), with its usual value per key."""
    target = pd.Series(keys, dtype=object)
    present = target.notna()
    given = target.loc[present]
    if present.sum() < MIN_ROWS or given.nunique() > FD_DISTINCT:
        return None, None
    base = float(given.value_counts(normalize=True).iloc[0])
    best: tuple[float, str, dict[str, str]] | None = None
    for other in others:
        if other.kind != "category":
            continue
        source = frame.data[other.name]
        both = present & source.notna()
        if both.sum() < MIN_ROWS:
            continue
        pairs = pd.DataFrame({"a": source.loc[both], "b": target.loc[both]})
        counts = pairs.value_counts(["a", "b"]).reset_index()  # a, b, count
        ordered = counts.sort_values(["a", "count", "b"], ascending=[True, False, True])
        top = ordered.drop_duplicates("a")
        groups = len(top)
        if groups < 2 or both.sum() / groups < FD_GROUP:
            continue
        share = float(top["count"].sum() / both.sum())
        if share < FD_SHARE or (1 - share) > (1 - FD_GAIN) * (1 - base):
            continue
        if best is None or share > best[0]:
            best = (share, other.name, dict(zip(top["a"], top["b"], strict=True)))
    if best is None:
        return None, None
    return best[1], best[2]


def _near(keys: list[str | None]) -> dict[str, str]:
    """A rare value → the much more frequent value one or two edits away (A-03's distance)."""
    counts = Counter(k for k in keys if k)
    if len(counts) > NEAR_DISTINCT:
        return {}
    frequent = [(k, c) for k, c in counts.most_common() if len(k) >= 4]
    found = {}
    for value, count in counts.items():
        if len(value) < 4:
            continue
        for other, rows in frequent:
            if rows < NEAR_RATIO * count:
                break
            if other != value and ph.damerau(value, other, 2) <= 2:
                found[value] = other
                break
    return found


def cells_of(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    views: Sequence[ColumnView],
    column: str,
    marked: Sequence[int] = (),
) -> Cells:
    fields = d.usable(views, max_levels=None)
    field = next(f for f in fields if f.name == column)
    others = d.usable(views, skip={column})
    frame = d.load(connection, parquet_path, [field, *others], extra_rows=marked)
    keys, shown = _keys(field, frame)
    determinant, majority = _determinant(keys, frame, others)
    near = _near(keys) if field.kind == "category" else {}
    present = [k for k in keys if k is not None]
    value_share = Counter(present)
    patterns = [pattern(s) if s is not None else "" for s in shown]
    pattern_share = Counter(p for p, k in zip(patterns, keys, strict=True) if k is not None)
    total = max(len(present), 1)
    lengths = np.array([len(s) if s is not None else np.nan for s in shown], dtype=float)
    length_median, length_spread = d.robust_scale(lengths)
    vectors = np.zeros((len(keys), len(FEATURES)))
    numbers = frame.data[field.name].to_numpy(dtype=float) if field.kind == "number" else None
    median, spread, log_median, log_spread = 0.0, 1.0, 0.0, 1.0
    magnitude = np.zeros(len(keys))
    if numbers is not None:
        median, spread = d.robust_scale(numbers)
        magnitude = np.log10(np.abs(numbers) + 1e-9)
        log_median, log_spread = d.robust_scale(magnitude)
    source = frame.data[determinant] if determinant else None
    for i, key in enumerate(keys):
        if key is None:
            vectors[i, 0] = 1.0
            continue
        vectors[i, 1] = -math.log10(value_share[key] / total)
        vectors[i, 2] = -math.log10(pattern_share[patterns[i]] / total)
        vectors[i, 3] = abs(lengths[i] - length_median) / length_spread
        if numbers is not None:
            vectors[i, 4] = min(abs(numbers[i] - median) / spread, 1e3)
            vectors[i, 5] = min(abs(magnitude[i] - log_median) / log_spread, 1e3)
        vectors[i, 6] = 1.0 if key in near else 0.0
        if source is not None and majority is not None:
            usual = majority.get(source.iloc[i]) if source.iloc[i] is not None else None
            vectors[i, 7] = 1.0 if usual is not None and usual != key else 0.0
    spelling = None
    if field.kind == "category":
        spelling = d.display_of(frame, column)
    return Cells(field, frame, keys, shown, vectors, determinant, majority, spelling, near)


def _groups(cells: Cells) -> tuple[np.ndarray, np.ndarray]:
    """k-means over the distinct vectors (weighted by their cells): each cell's group, and the
    centres."""
    unique, inverse, counts = np.unique(cells.x, axis=0, return_inverse=True, return_counts=True)
    k = min(TO_MARK, len(unique))
    if k <= 1:
        return np.zeros(len(cells), dtype=int), unique[:1]
    model = KMeans(n_clusters=k, n_init="auto", random_state=d.SEED)
    model.fit(unique, sample_weight=counts)
    return np.asarray(model.labels_)[inverse.ravel()], np.asarray(model.cluster_centers_)


def _hint(cells: Cells, centre: np.ndarray) -> str:
    mean, spread = cells.x.mean(axis=0), cells.x.std(axis=0)
    spread[spread == 0] = 1
    lift = (centre - mean) / spread
    strongest = int(np.argmax(lift))
    if lift[strongest] < 0.5:
        return "a typical cell"
    return HINTS[FEATURES[strongest]].format(column=cells.determinant or "another column")


def to_mark(cells: Cells) -> list[ToMark]:
    """One cell per group, the groups that stand out first (S7)."""
    labels, centres = _groups(cells)
    found = []
    for group in range(len(centres)):
        members = np.flatnonzero(labels == group)
        if len(members) == 0:
            continue
        gaps = ((cells.x[members] - centres[group]) ** 2).sum(axis=1)
        pick = int(members[int(np.argmin(gaps))])
        oddness = float(centres[group].sum())
        found.append((oddness, pick, len(members), _hint(cells, centres[group])))
    found.sort(key=lambda item: (-item[0], int(cells.frame.rows[item[1]])))
    return [
        ToMark(int(cells.frame.rows[pick]), cells.shown[pick], hint, alike)
        for _, pick, alike, hint in found[:TO_MARK]
    ]


def _correction(cells: Cells, index: int) -> tuple[str | None, str]:
    """Baran-lite (C-37): the determinant's usual value, a close frequent spelling, else empty."""
    key = cells.keys[index]
    if cells.determinant and cells.majority:
        source = cells.frame.data[cells.determinant].iloc[index]
        usual = cells.majority.get(source) if source is not None else None
        if usual is not None and usual != key:
            return usual, "dependency"
    if key is not None and key in cells.near:
        return cells.near[key], "near"
    return None, "empty"


def _written(cells: Cells, key: str | None) -> str | None:
    if key is None or cells.spelling is None:
        return key
    return cells.spelling.get(key, key)


def learn(cells: Cells, marks: Sequence[Mark], stopped: d.Stopped) -> CleanSuggestion | str:
    errors = {mark.row for mark in marks if mark.error}
    fine = {mark.row for mark in marks if not mark.error}
    if not errors or not fine:
        return "Mark at least one cell as Error and one as OK, then learn again."
    position = {int(row): i for i, row in enumerate(cells.frame.rows)}
    labels, _ = _groups(cells)
    target = np.full(len(cells), -1)
    for rows, value in ((fine, 0), (errors, 1)):
        for row in rows:
            if row in position:
                target[position[row]] = value
    marked = target.copy()
    spread = 0
    for group in np.unique(labels):
        members = labels == group
        seen = set(marked[members & (marked >= 0)].tolist())
        if len(seen) == 1:
            value = seen.pop()
            newly = members & (target < 0)
            spread += int(newly.sum())
            target[newly] = value
    d.check(stopped)
    train = np.flatnonzero(target >= 0)
    if len(train) > TRAIN_CELLS:
        train = np.sort(np.random.default_rng(d.SEED).choice(train, TRAIN_CELLS, replace=False))
    forest = RandomForestClassifier(
        n_estimators=100, random_state=d.SEED, class_weight="balanced", n_jobs=1
    )
    forest.fit(cells.x[train], target[train])
    error_column = list(forest.classes_).index(1)
    probability = np.asarray(forest.predict_proba(cells.x))[:, error_column]
    flagged = (probability >= 0.5) & (marked != 0)
    flagged |= marked == 1
    picked = np.flatnonzero(flagged)
    picked = picked[np.argsort(-probability[picked], kind="stable")]
    fixes: dict[int, str | None] = {}
    sources: Counter[str] = Counter()
    examples = []
    for index in picked:
        if len(fixes) >= m.MAX_MAP:
            break
        new, source = _correction(cells, int(index))
        if new == cells.keys[index]:
            continue
        row = int(cells.frame.rows[index])
        written = _written(cells, new)
        fixes[row] = written
        sources[source] += 1
        examples.append((f"{d.row_label(row)}: {cells.shown[index]}", written))
    name = cells.field.name
    if not fixes:
        return f"{name}: no cell looks like the errors you marked."
    said = {
        "dependency": f"{d.n(sources['dependency'])} from {cells.determinant}",
        "near": f"{d.n(sources['near'])} to a close frequent spelling",
        "empty": f"{d.n(sources['empty'])} left empty",
    }
    parts = [text for source, text in said.items() if sources[source]]
    marks_text = f"{d.plural(len(errors), 'Error mark')} and {d.plural(len(fine), 'OK mark')}"
    reason = (
        f"Learnt from your {marks_text} (spread to {d.plural(spread, 'similar cell')}), a random "
        f"forest on what sets each cell apart (rarity of its value and pattern, length, distance "
        f"from the others{', agreement with ' + cells.determinant if cells.determinant else ''}), "
        f"on {d.n(len(cells))} rows{cells.frame.sample_note()}. Corrections: "
        f"{', '.join(parts)}. A cell you marked OK is never changed."
    )
    count = len(fixes)
    return d.suggestion(
        RULE,
        name,
        [cells.field],
        f"Fix {d.plural(count, 'cell')} of {name} like the errors you marked",
        reason,
        m.SetCells(id="s0", column=name, cells=fixes),
        cells=count,
        examples=examples,
    )


def run(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    views: Sequence[ColumnView],
    column: str | None,
    marks: Sequence[Mark] | None,
    stopped: d.Stopped,
) -> tuple[list[CleanSuggestion], str | None, list[ToMark]]:
    problem = why_not(views)
    if problem:
        return [], problem, []
    choices = columns(views)
    if column is None:
        return [], "Pick the column to look for errors in.", []
    if column not in choices:
        return [], f"'{column}' is not a column of numbers or categories.", []
    cells = cells_of(connection, parquet_path, views, column, [mark.row for mark in marks or []])
    d.check(stopped)
    if not marks:
        return [], None, to_mark(cells)
    result = learn(cells, marks, stopped)
    if isinstance(result, str):
        return [], result, to_mark(cells)
    return [result], None, []
