"""E-06: unusual rows across columns, with the Isolation Forest and the threshold of Unusual rows
(033, `segments/unusual.py`): mark them in a column, or remove them. Never ticked (C-30)."""

from __future__ import annotations

from collections.abc import Sequence

import duckdb
import numpy as np

from quantilo.catalog import ColumnView
from quantilo.cleaning import model as m
from quantilo.cleaning.expert import data as d
from quantilo.cleaning.suggestion import CleanSuggestion
from quantilo.segments.unusual import isolation_forest, propose

RULE = "E-06"
MIN_ROWS = 50
MIN_FIELDS = 2
LABEL = "unusual"
COLUMN = "unusual_row"


def why_not(views: Sequence[ColumnView]) -> str | None:
    if d.rows_of(views) < MIN_ROWS:
        return f"Needs at least {MIN_ROWS} rows."
    if len(d.usable(views)) < MIN_FIELDS:
        return "Needs at least two columns of numbers or categories."
    return None


def _reason_of(x: np.ndarray, frame: d.Frame, fields: list[d.Field], index: int) -> str:
    """The column that makes the row most unusual, in words: its value and how far it is."""
    strongest = int(np.argmax(np.abs(x[index])))
    field = fields[strongest]
    if field.kind == "number":
        value = frame.data[field.name].iloc[index]
        if np.isnan(value):
            return f"{field.name} empty"
        return f"{field.name} {d.number_text(value)}"
    spelled = frame.text[field.name].iloc[index]
    return f"{field.name} '{spelled}'" if spelled is not None else f"{field.name} empty"


def run(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    views: Sequence[ColumnView],
    stopped: d.Stopped,
) -> tuple[list[CleanSuggestion], str | None, d.Frame | None]:
    problem = why_not(views)
    if problem:
        return [], problem, None
    fields = d.usable(views)
    frame = d.load(connection, parquet_path, fields)
    d.check(stopped)
    encoder = d.Encoder.fit(frame, fields)
    x = encoder.transform(frame)
    # how far each value is, column by column, for the sentence: |z| for numbers, rarity else
    far = np.column_stack(
        [
            np.abs(x[:, i]) if f.kind == "number" else -np.log(np.maximum(x[:, i], 1e-9))
            for i, f in enumerate(fields)
        ]
    )
    scorer = isolation_forest(x, d.SEED)
    d.check(stopped)
    scores = scorer.fit_scores
    threshold = propose(scores)
    flagged = np.flatnonzero(scores >= threshold.value)
    flagged = flagged[np.argsort(-scores[flagged], kind="stable")][: m.MAX_MAP]
    if len(flagged) == 0:
        return [], "No row stands out.", frame
    rows = [int(frame.rows[i]) for i in flagged]
    reasons = {int(frame.rows[i]): _reason_of(far, frame, fields, int(i)) for i in flagged}
    count = len(rows)
    capped = " (the 5,000 most unusual)" if count == m.MAX_MAP else ""
    why = (
        f"Isolation Forest (200 trees, the method of Unusual rows) on {len(fields)} columns, "
        f"{d.n(len(frame))} rows{frame.sample_note()}: {threshold.why}{capped} An unusual row may "
        "be right: look at it before you remove it."
    )
    examples = [(f"{d.row_label(r)}: {reasons[r]}", LABEL) for r in rows]
    to = d.free_name(views, COLUMN)
    mark = m.LabelRows(id="s0", to=to, labels=dict.fromkeys(rows, LABEL))
    remove = m.DropRows(id="s0", rows=sorted(rows))
    first = fields[:1]
    return (
        [
            d.suggestion(
                RULE,
                "mark",
                first,
                f"Mark {d.plural(count, 'unusual row')} in a column ({to})",
                why,
                mark,
                cells=count,
                examples=examples,
            ),
            d.suggestion(
                RULE,
                "remove",
                first,
                f"Remove {d.plural(count, 'unusual row')}",
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
