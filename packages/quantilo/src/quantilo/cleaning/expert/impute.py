"""E-07: empty cells of a number column filled from similar rows, each with its uncertainty band,
for the analysis only (C-36): KNN (the mean of the 5 nearest rows, band = their min–max) or
iterative imputation (MICE: 5 draws of scikit-learn's `IterativeImputer`, band = their min–max)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import duckdb
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (makes it importable)
from sklearn.impute import IterativeImputer
from sklearn.metrics.pairwise import nan_euclidean_distances

from quantilo.catalog import ColumnView
from quantilo.cleaning import model as m
from quantilo.cleaning.expert import data as d
from quantilo.cleaning.suggestion import CleanSuggestion

RULE = "E-07"
Imputer = Literal["knn", "iterative"]
NAMES = {"knn": "KNN, the 5 most similar rows", "iterative": "iterative imputation (MICE), 5 draws"}
SHORT = {"knn": "KNN", "iterative": "MICE"}
MAX_EMPTY = 0.6  # C-36: a column emptier than this is better left as it is
NEIGHBOURS = 5
DRAWS = 5
REFERENCE_ROWS = 20_000  # C-32: rows the neighbours are looked for among
MAX_TARGETS = 10
CHUNK = 250  # query rows per distance block (250 × 20,000 distances)
LEAK = (
    "For the analysis only: for machine learning, leave the empty cells to the ML preparation, "
    "which fills them inside each fold (no leak between training and test rows)."
)


def targets(views: Sequence[ColumnView]) -> list[str]:
    """Number columns with empty cells, 60 % of them at most."""
    found = []
    for view in sorted(views, key=lambda v: v.position):
        field = d.field_of(view)
        if field is None or field.kind != "number":
            continue
        if 0 < d.empty_share(view) <= MAX_EMPTY:
            found.append(view.name)
    return found


def why_not(views: Sequence[ColumnView]) -> str | None:
    if not targets(views):
        return "Needs a column of numbers with empty cells (60 % of them at most)."
    if len(d.usable(views)) < 2:
        return "Needs other columns to find similar rows."
    return None


def _nearest(queries: np.ndarray, reference: np.ndarray, stopped: d.Stopped) -> np.ndarray:
    """The NEIGHBOURS nearest reference rows of each query, by a distance over the columns both
    have (an empty cell is left out, as in scikit-learn's KNNImputer), in chunks."""
    found = []
    for start in range(0, len(queries), CHUNK):
        d.check(stopped)
        gaps = np.asarray(nan_euclidean_distances(queries[start : start + CHUNK], reference))
        gaps = np.nan_to_num(gaps, nan=np.inf)
        order = np.argpartition(gaps, NEIGHBOURS - 1, axis=1)[:, :NEIGHBOURS]
        found.append(order)
    return np.vstack(found)


def _fill(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    views: Sequence[ColumnView],
    target: d.Field,
    imputer: Imputer,
    stopped: d.Stopped,
) -> CleanSuggestion | str:
    features = d.usable(views, skip={target.name})
    if not features:
        return f"{target.name}: no other column to find similar rows."
    fields = [target, *features]
    known = d.load(connection, parquet_path, fields, limit=REFERENCE_ROWS, where="v0 IS NOT NULL")
    empty = d.load(connection, parquet_path, fields, limit=m.MAX_MAP + 1, where="v0 IS NULL")
    if empty.total > m.MAX_MAP:
        return (
            f"{target.name}: {d.n(empty.total)} empty cells, more than {d.n(m.MAX_MAP)} for one "
            "step: let the ML preparation fill them."
        )
    # a row with nothing else filled (an empty row) has nothing to compare: left as it is
    alone = np.asarray(empty.data[[f.name for f in features]].isna().all(axis=1), dtype=bool)
    if alone.any():
        empty = d.Frame(
            empty.data.loc[~alone].reset_index(drop=True),
            empty.text.loc[~alone].reset_index(drop=True),
            empty.rows[~alone],
            empty.total,
            empty.sampled,
        )
    left = ""
    if alone.any():
        left = (
            f" {d.plural(int(alone.sum()), 'empty cell')} left: nothing else filled on their rows."
        )
    if len(empty) == 0 or len(known) < NEIGHBOURS + 1:
        return f"{target.name}: no row to compare with.{left}"
    d.check(stopped)
    y = known.data[target.name].to_numpy(dtype=float)
    low, high = float(np.nanmin(y)), float(np.nanmax(y))
    if imputer == "knn":
        encoder = d.Encoder.fit(known, features, onehot=True)
        nearest = _nearest(
            encoder.transform(empty, missing=np.nan),
            encoder.transform(known, missing=np.nan),
            stopped,
        )
        values = y[nearest]
        guess, band_low, band_high = values.mean(axis=1), values.min(axis=1), values.max(axis=1)
    else:
        encoder = d.Encoder.fit(known, features)
        x_known = np.column_stack([y, encoder.transform(known)])
        x_empty = np.column_stack([np.full(len(empty), np.nan), encoder.transform(empty)])
        stacked = np.vstack([x_known, x_empty])
        draws = []
        for draw in range(DRAWS):
            d.check(stopped)
            imputer_model = IterativeImputer(
                sample_posterior=True, max_iter=10, random_state=d.SEED + draw
            )
            imputed = np.asarray(imputer_model.fit_transform(stacked))
            draws.append(imputed[len(x_known) :, 0])
        found = np.clip(np.column_stack(draws), low, high)
        guess, band_low, band_high = found.mean(axis=1), found.min(axis=1), found.max(axis=1)
    if target.integer:
        guess, band_low, band_high = np.round(guess), np.round(band_low), np.round(band_high)
    cells: dict[int, str | None] = {}
    examples = []
    widths = band_high - band_low
    for index, row in enumerate(empty.rows):
        value = d.number_text(guess[index])
        cells[int(row)] = value
        band = f"{d.number_text(band_low[index])}–{d.number_text(band_high[index])}"
        examples.append((f"{d.row_label(int(row))}: empty", f"{value} ({band})"))
    count = len(cells)
    reason = (
        f"Filled from {NAMES[imputer]}, on {len(features)} other columns and "
        f"{d.n(len(known))} rows with a value{known.sample_note()}. Each value comes with its band "
        f"(the values it was drawn from); half of the bands are narrower than "
        f"{d.number_text(float(np.median(widths)))} ({target.name} goes from {d.number_text(low)} "
        f"to {d.number_text(high)}).{left} {LEAK}"
    )
    return d.suggestion(
        RULE,
        f"{target.name}:{imputer}",
        [target],
        f"Fill {d.plural(count, 'empty cell')} of {target.name} from similar rows "
        f"({SHORT[imputer]})",
        reason,
        m.SetCells(id="s0", column=target.name, cells=cells),
        cells=count,
        examples=examples,
        number=target.position % 50 + (50 if imputer == "iterative" else 0),
    )


def run(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    views: Sequence[ColumnView],
    imputer: Imputer,
    columns: list[str] | None,
    stopped: d.Stopped,
) -> tuple[list[CleanSuggestion], str | None]:
    problem = why_not(views)
    if problem:
        return [], problem
    choices = targets(views)
    wanted = [c for c in (columns or choices) if c in choices][:MAX_TARGETS]
    if not wanted:
        return [], "Pick number columns with empty cells."
    found, notes = [], []
    by_name = {v.name: v for v in views}
    for name in wanted:
        field = d.field_of(by_name[name])
        assert field is not None
        result = _fill(connection, parquet_path, views, field, imputer, stopped)
        if isinstance(result, str):
            notes.append(result)
        else:
            found.append(result)
    return found, " ".join(notes) or None
