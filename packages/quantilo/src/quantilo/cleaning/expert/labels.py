"""E-03: suspicious labels of a target column (confident learning): out-of-sample probabilities
over 5 stratified folds (`HistGradientBoosting`, which runs in Pyodide), a threshold per class (the
mean probability of the class on its own rows), and the rows whose label the model contradicts with
confidence (C-35). Relabel them or remove them; never ticked (C-30)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from quantilo.catalog import ColumnView
from quantilo.cleaning import model as m
from quantilo.cleaning.expert import data as d
from quantilo.cleaning.suggestion import CleanSuggestion

RULE = "E-03"
TRAIN_ROWS = 50_000  # C-32
FOLDS = 5
MIN_CLASS = 5  # C-34: a class needs this many rows to be checked over 5 folds
MIN_CLASSES = 2
# C-35: the model at least halves the errors of always guessing the most frequent label, or it
# says nothing (Telco's Churn: 20 % of errors against 27 %, a weak model whose doubts are no errors)
ERROR_CUT = 0.5
CONFIDENT = 0.5
EARLY_STOP: Any = True  # the stubs only know "auto", which stops early from 10,000 rows only


def targets(views: Sequence[ColumnView]) -> list[str]:
    """Categories of 2 to 50 values, with at least one other usable column."""
    found = []
    for view in sorted(views, key=lambda v: v.position):
        field = d.field_of(view)
        count = d.levels(view)
        if field is None or field.kind != "category" or (count is not None and count < 2):
            continue
        if d.usable(views, skip={view.name}):
            found.append(view.name)
    return found


def why_not(views: Sequence[ColumnView]) -> str | None:
    if not targets(views):
        return "Needs a category of 2 to 50 values (the labels) and other columns to learn from."
    return None


def _model() -> HistGradientBoostingClassifier:
    """Early stopping on a tenth of the rows keeps the probabilities honest (without it, 100
    rounds on a noisy label give 0 % and 100 %)."""
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        early_stopping=EARLY_STOP,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=d.SEED,
        categorical_features="from_dtype",
    )


def run(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    views: Sequence[ColumnView],
    target: str | None,
    stopped: d.Stopped,
) -> tuple[list[CleanSuggestion], str | None, d.Frame | None]:
    choices = targets(views)
    if not choices:
        return [], why_not(views), None
    if target is None:
        return [], "Pick the column whose labels to check.", None
    if target not in choices:
        return [], f"'{target}' is not a category of 2 to 50 values.", None
    view = next(v for v in views if v.name == target)
    label = d.field_of(view)
    assert label is not None
    fields = d.usable(views, skip={target})
    frame = d.load(connection, parquet_path, [label, *fields], limit=TRAIN_ROWS)
    d.check(stopped)
    y_all = frame.data[target]
    counts = y_all.value_counts()
    kept = [str(k) for k, c in counts.items() if c >= MIN_CLASS]
    small = int(counts[counts < MIN_CLASS].sum()) if len(counts) else 0
    if len(kept) < MIN_CLASSES:
        return [], f"{target} has fewer than two labels of {MIN_CLASS} rows or more.", frame
    keep = y_all.isin(kept).to_numpy()
    index = np.flatnonzero(keep)
    classes = sorted(kept)
    position = {c: i for i, c in enumerate(classes)}
    y = np.array([position[v] for v in y_all.iloc[index]])
    x, mask = d.codes(frame, fields)
    x = x[index]
    table = pd.DataFrame(
        {f"f{i}": (pd.Categorical(x[:, i]) if mask[i] else x[:, i]) for i in range(x.shape[1])}
    )
    proba = np.zeros((len(index), len(classes)))
    folds = min(FOLDS, int(np.bincount(y).min()))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=d.SEED)
    for train, test in splitter.split(table, y):
        d.check(stopped)
        model = _model().fit(table.iloc[train], y[train])
        fitted = list(model.classes_)
        found = model.predict_proba(table.iloc[test])
        for column, klass in enumerate(fitted):
            proba[test, int(klass)] = found[:, column]
    accuracy = float((proba.argmax(axis=1) == y).mean())
    majority = float(np.bincount(y).max() / len(y))
    names = d.display_of(frame, target)
    shown = [names.get(c, c) for c in classes]
    left = (
        f"; {d.plural(small, 'row')} of labels with fewer than {MIN_CLASS} rows left out"
        if small
        else ""
    )
    learnt = (
        f"Gradient boosting on {len(fields)} other columns, {FOLDS} folds out of sample, "
        f"{d.n(len(index))} rows{frame.sample_note()}{left}: it finds {target} on "
        f"{accuracy:.0%} of the rows (always guessing the most frequent label: {majority:.0%})."
    )
    if (1 - accuracy) > ERROR_CUT * (1 - majority):
        quiet = (
            f"The other columns do not tell {target} well enough (the model does not halve the "
            "errors of the most frequent label): no label looks wrong for a reason."
        )
        return [], f"{learnt} {quiet}", frame
    thresholds = np.array([proba[y == j, j].mean() for j in range(len(classes))])
    given = proba[np.arange(len(y)), y]
    others = proba.copy()
    others[np.arange(len(y)), y] = -1
    confident = (others >= thresholds) & (others >= CONFIDENT)
    others[~confident] = -1
    proposed = others.argmax(axis=1)
    suspect = (others.max(axis=1) >= 0) & (given < thresholds[y])
    picked = np.flatnonzero(suspect)
    margin = others.max(axis=1) - given
    picked = picked[np.argsort(-margin[picked], kind="stable")][: m.MAX_MAP]
    if len(picked) == 0:
        return [], f"{learnt} No label is contradicted with confidence.", frame
    rows = [int(frame.rows[index[i]]) for i in picked]
    cells: dict[int, str | None] = {
        row: shown[int(proposed[i])] for row, i in zip(rows, picked, strict=True)
    }
    examples = [
        (
            f"{d.row_label(row)}: {shown[int(y[i])]} ({given[i]:.0%})",
            f"{shown[int(proposed[i])]} ({others[i].max():.0%})",
        )
        for row, i in zip(rows, picked, strict=True)
    ]
    count = len(rows)
    why = (
        f"{learnt} These rows carry a label the model gives a low probability, while it is "
        "confident of another (above that label's own threshold, the mean probability on its rows, "
        "and above 50 %). A rare case can be right. Cleaning labels helps most when the rows you "
        "test a model on are cleaned too."
    )
    relabel = m.SetCells(id="s0", column=target, cells=cells)
    remove = m.DropRows(id="s0", rows=sorted(rows))
    return (
        [
            d.suggestion(
                RULE,
                f"{target}:relabel",
                [label],
                f"Relabel {d.plural(count, 'row')} of {target} as the other columns say",
                why,
                relabel,
                cells=count,
                examples=examples,
            ),
            d.suggestion(
                RULE,
                f"{target}:remove",
                [label],
                f"Remove {d.plural(count, 'row')} whose {target} looks wrong",
                why,
                remove,
                rows=count,
                examples=[(b, "removed") for b, _ in examples],
                number=1,
            ),
        ],
        None,
        frame,
    )
