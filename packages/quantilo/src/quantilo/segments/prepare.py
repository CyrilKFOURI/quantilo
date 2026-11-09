"""Columns and preparation shared by Segments (032) and Unusual rows (033): the ML rules engine
(017) with its "distance" profile, then the matrices the methods read.

Everything that learns from data (medians, scales, categories kept) is fitted on the fit rows
(at most FIT_ROWS, a reproducible sample beyond) and then applied to every row."""

import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any, cast

import duckdb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from quantilo.ml import data as ml_data
from quantilo.ml import preparation as prep
from quantilo.ml.data import CATEGORICAL, NUMERIC, Field

IN_BROWSER = sys.platform == "emscripten"  # the lite version (Pyodide): one core, little memory
FIT_ROWS = 50_000 if IN_BROWSER else 200_000
CHUNK = 100_000  # rows assigned at a time beyond the fit rows
MIN_ROWS = 20
OTHER = "Other"


@dataclass
class Rows:
    """The fit rows: typed columns and "__row" (row number in the file)."""

    data: pd.DataFrame
    rows: int  # rows after the filters
    sampled: bool  # the fit rows are a sample of them


def load(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    fields: list[Field],
    limit: int | None = None,
) -> Rows:
    frame = ml_data.load(connection, source, fields, None, limit or FIT_ROWS)
    return Rows(frame.data, frame.rows, frame.sampled)


def every_row(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    fields: list[Field],
    chunk: int | None = None,
) -> Iterator[pd.DataFrame]:
    """All the rows after the filters, typed, in file order, `chunk` rows at a time."""
    aliases = {field.name: f"c{i}" for i, field in enumerate(fields)}
    selected = ", ".join(f"{ml_data._expression(f)} AS {aliases[f.name]}" for f in fields)
    reader = connection.sql(
        f"SELECT {selected}, file_row_number AS __row FROM {source} ORDER BY file_row_number"
    ).to_arrow_reader(chunk or CHUNK)
    for batch in cast(Any, reader):
        data = batch.to_pandas()
        data.columns = [*(f.name for f in fields), "__row"]
        yield typed(data, fields)


def typed(data: pd.DataFrame, fields: list[Field]) -> pd.DataFrame:
    """Numbers as floats, as `ml_data.load` reads them."""
    for field in fields:
        if field.type in NUMERIC and field.name in data:
            data[field.name] = pd.Series(
                pd.to_numeric(data[field.name], errors="coerce"), index=data.index, dtype=float
            )
    return data


def choices_for(
    fields: list[Field], columns: list[str] | None, choices: prep.Choices | None
) -> prep.Choices:
    """Columns given by name are used (if they can be) and every other one is left out; the
    preparation choices apply on top."""
    found = choices.model_copy(deep=True) if choices else prep.Choices()
    if columns is None:
        return found
    wanted = set(columns)
    for field in fields:
        current = found.columns.get(field.name, prep.ColumnChoice())
        found.columns[field.name] = current.model_copy(update={"use": field.name in wanted})
    return found


def decide(
    frame: pd.DataFrame,
    fields: list[Field],
    columns: list[str] | None,
    choices: prep.Choices | None,
) -> prep.Preparation:
    return prep.decide_distance(frame, fields, choices_for(fields, columns, choices))


def is_categorical(decision: prep.Decision) -> bool:
    """A category for k-prototypes: nominal, boolean, or ordinal without an order."""
    return decision.type in CATEGORICAL and decision.encoding != "ordinal"


def categorical_share(preparation: prep.Preparation) -> float:
    used = preparation.used
    return sum(is_categorical(d) for d in used) / len(used) if used else 0.0


@dataclass
class Matrix:
    """The prepared rows: one-hot and robust-scaled (`x`), and for k-prototypes the numbers
    alone (`numbers`) with the category codes (`codes`)."""

    pipeline: Any  # fitted: original columns → x
    names: list[str]
    numeric: Any | None  # fitted: original columns → numbers (None without numbers)
    categories: dict[str, list[str]]  # per category column: codes 0…n-1 (Other and Missing last)
    x: np.ndarray
    numbers: np.ndarray
    codes: np.ndarray

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return _finite(self.pipeline.transform(frame))

    def mixed(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numbers = (
            _finite(self.numeric.transform(frame))
            if self.numeric is not None
            else np.zeros((len(frame), 0))
        )
        return numbers, encode(frame, self.categories)


def _finite(values: Any) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _pipeline(preparation: prep.Preparation, rows: int) -> Pipeline:
    return Pipeline(prep.steps(preparation, rows, "regression"))


def _labels(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(prep.MISSING).astype(str)


def category_levels(series: pd.Series) -> list[str]:
    """The most frequent categories (at most DISTANCE_CATEGORIES), then Other and Missing."""
    counts = _labels(series).value_counts()
    kept = [str(c) for c in counts.index[: prep.DISTANCE_CATEGORIES] if c != prep.MISSING]
    return [*kept, OTHER, prep.MISSING]


def encode(frame: pd.DataFrame, categories: dict[str, list[str]]) -> np.ndarray:
    """Category codes: a category not kept is Other."""
    columns = []
    for name, levels in categories.items():
        index = {level: i for i, level in enumerate(levels)}
        codes = _labels(cast(pd.Series, frame[name])).map(index).fillna(index[OTHER])
        columns.append(codes.to_numpy(dtype=np.int64))
    if not columns:
        return np.zeros((len(frame), 0), dtype=np.int64)
    return np.column_stack(columns).astype(np.int64)


def matrix(preparation: prep.Preparation, frame: pd.DataFrame) -> Matrix:
    """Fit the preparation on `frame` and prepare it."""
    used = preparation.used
    if not used:
        raise ValueError("No column can be used: pick at least one number or category.")
    rows = len(frame)
    pipeline = _pipeline(preparation, rows)
    x = _finite(pipeline.fit_transform(frame))
    last: Any = pipeline[-1]
    names = [str(n) for n in last.get_feature_names_out()]
    numeric_part = [d for d in used if not is_categorical(d)]
    numeric = None
    numbers = np.zeros((rows, 0))
    if numeric_part:
        numeric = _pipeline(replace(preparation, decisions=numeric_part), rows)
        numbers = _finite(numeric.fit_transform(frame))
    categories = {
        d.column: category_levels(cast(pd.Series, frame[d.column]))
        for d in used
        if is_categorical(d)
    }
    return Matrix(pipeline, names, numeric, categories, x, numbers, encode(frame, categories))
