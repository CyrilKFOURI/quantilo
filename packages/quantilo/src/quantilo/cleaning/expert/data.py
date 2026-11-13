"""What the Expert checks read (101, C-32): the usable columns, a reproducible sample of rows in
their analysis type, and the suggestion each method writes (layer "expert", mode S, risk "review",
C-30)."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import duckdb
import numpy as np
import pandas as pd

from quantilo.catalog import ColumnView
from quantilo.cleaning import model as m
from quantilo.cleaning import phonetic as ph
from quantilo.cleaning.apply import CleanCancelled
from quantilo.cleaning.compile import INTEGERS
from quantilo.cleaning.rules import basic
from quantilo.cleaning.suggestion import CleanSuggestion, Example
from quantilo.ingestion.sql import ident, lit
from quantilo.typing.parsers import typed_sql
from quantilo.typing.types import KEY_ROLES, NUMERIC_TYPES

SAMPLE_ROWS = 200_000  # C-32
SEED = 101
MAX_COLUMNS = 30
MAX_LEVELS = 50  # a category used as a feature, or a label to check, has at most this many values
LISTED = 10  # examples of a suggestion
CATEGORIES = ("categorical_nominal", "categorical_ordinal", "boolean")
NULL_TOKENS = basic.SURE_TOKENS | basic.WEAK_TOKENS
# C-38: after Basic (≤ 170) and Advanced (200 to 330).
ORDER = {"E-06": 400, "E-03": 410, "E-07": 420, "E-01": 430}
# 096 contract: every rule says how it changes the dataset, by step when it has two.
IMPACT: dict[str, m.Impact] = {"E-03": "values", "E-07": "values", "E-01": "values"}
STEP_IMPACT: dict[tuple[str, str], m.Impact] = {
    ("E-06", "label_rows"): "columns",
    ("E-06", "drop_rows"): "rows",
    ("E-03", "set_cells"): "values",
    ("E-03", "drop_rows"): "rows",
}
# Step ids of each method, apart from the Basic and Advanced ones (s1, s2…).
FIRST_ID = {"E-06": 5_001, "E-03": 5_101, "E-07": 5_201, "E-01": 5_301}

Stopped = Callable[[], bool]


def one_thread() -> Any:
    """OpenMP held to one thread: scikit-learn's default oversubscribes a busy machine (a 3,000-row
    gradient boosting took 47 s instead of 3); Pyodide has one thread anyway."""
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError:  # pragma: no cover - threadpoolctl comes with scikit-learn
        return contextlib.nullcontext()
    return threadpool_limits(limits=1)


def check(stopped: Stopped) -> None:
    """A point where a run stops when cancelled (C-31)."""
    if stopped():
        raise CleanCancelled


def n(count: int | float) -> str:
    return f"{count:,}"


def plural(count: int, noun: str) -> str:
    return f"{n(count)} {noun}" + ("" if count == 1 else "s")


def number_text(value: float) -> str:
    rounded = round(float(value), 6)
    if rounded == int(rounded) and abs(rounded) < 1e15:
        return str(int(rounded))
    return format(rounded, ".6g")


def row_label(row: int) -> str:
    return f"row {row + 1:,}"  # from 1, the first data row (097, decision 3)


# --- Columns -------------------------------------------------------------------------------------


@dataclass
class Field:
    name: str
    kind: Literal["number", "category"]
    integer: bool
    view: ColumnView

    @property
    def position(self) -> int:
        return self.view.position


def _summary(view: ColumnView, key: str) -> int | None:
    value = (view.summary or {}).get(key)
    return int(value) if isinstance(value, int | float) else None


def rows_of(views: Sequence[ColumnView]) -> int:
    return max((_summary(v, "rows") or 0 for v in views), default=0)


def empty_share(view: ColumnView) -> float:
    rows = _summary(view, "rows") or 0
    if not rows:
        return 0.0
    return ((_summary(view, "null_count") or 0) + (_summary(view, "invalid_count") or 0)) / rows


def levels(view: ColumnView) -> int | None:
    return _summary(view, "distinct_count")


def field_of(view: ColumnView, max_levels: int | None = MAX_LEVELS) -> Field | None:
    """A column the methods read: a number or a category of few values; never an identifier, a
    free text, a date or a column left out."""
    if view.excluded or (view.role or "") in KEY_ROLES or view.role == "free_text":
        return None
    if view.type in NUMERIC_TYPES:
        integer = view.type == "numeric_discrete" or view.storage_dtype in INTEGERS
        return Field(view.name, "number", integer, view)
    if view.type in CATEGORIES:
        count = levels(view)
        if max_levels is not None and count is not None and count > max_levels:
            return None
        return Field(view.name, "category", False, view)
    return None


def usable(
    views: Sequence[ColumnView], skip: set[str] | None = None, max_levels: int | None = MAX_LEVELS
) -> list[Field]:
    found = []
    for view in sorted(views, key=lambda v: v.position):
        if skip and view.name in skip:
            continue
        field = field_of(view, max_levels)
        if field is not None and empty_share(view) <= 0.6:
            found.append(field)
    return found[:MAX_COLUMNS]


def free_name(views: Sequence[ColumnView], wanted: str) -> str:
    names = {v.name for v in views}
    name, number = wanted, 2
    while name in names:
        name, number = f"{wanted}_{number}", number + 1
    return name


# --- Rows ----------------------------------------------------------------------------------------


@dataclass
class Frame:
    """Rows read in their analysis type: numbers as floats, categories folded (`fold`, C-34) with
    their spellings kept in `text`. `rows`: their number in the file, in file order."""

    data: pd.DataFrame
    text: pd.DataFrame
    rows: np.ndarray
    total: int  # rows in the dataset
    sampled: bool

    def __len__(self) -> int:
        return len(self.rows)

    def sample_note(self) -> str:
        if not self.sampled:
            return ""
        return f" (a reproducible sample of {n(len(self))} of the {n(self.total)} rows)"


def _expr(field: Field) -> str:
    parser = field.view.parser
    value = typed_sql(field.name, parser) if parser is not None else ident(field.name)
    if field.kind == "number":
        return f"TRY_CAST(({value}) AS DOUBLE)"
    return f"CAST(({value}) AS VARCHAR)"


def folded(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    key = ph.fold(str(value))
    return None if not key or key in NULL_TOKENS else key


def load(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    fields: Sequence[Field],
    limit: int = SAMPLE_ROWS,
    where: str | None = None,
    extra_rows: Sequence[int] = (),
) -> Frame:
    """`fields` of a reproducible sample of `limit` rows (CLAUDE.md: `ORDER BY hash(row + seed)`),
    or of every row; `where` on the analysis values (`v0`, `v1`… for the fields); `extra_rows`
    always read (the rows a user marked)."""
    source = f"read_parquet({lit(parquet_path)}, file_row_number = true)"
    columns = ", ".join(f"{_expr(f)} AS v{i}" for i, f in enumerate(fields))
    typed = f"SELECT file_row_number AS row_number, {columns} FROM {source}"
    condition = f" WHERE {where}" if where else ""
    total_row = connection.execute(f"SELECT count(*) FROM ({typed}){condition}").fetchone()
    total = int(total_row[0]) if total_row else 0
    sampled = total > limit
    order = f"hash(row_number + {SEED})" if sampled else "row_number"
    sql = f"SELECT * FROM ({typed}){condition} ORDER BY {order} LIMIT {int(limit)}"
    if extra_rows:
        wanted = ", ".join(str(int(r)) for r in sorted(set(extra_rows)))
        sql = f"({sql}) UNION SELECT * FROM ({typed}) WHERE row_number IN ({wanted})"
    raw = connection.execute(f"SELECT * FROM ({sql}) ORDER BY row_number").df()
    data, text = {}, {}
    for index, field in enumerate(fields):
        values = pd.Series(raw[f"v{index}"])
        if field.kind == "number":
            data[field.name] = pd.Series(pd.to_numeric(values, errors="coerce"), dtype=float)
        else:
            spelled = values.astype(object).where(values.notna(), None)
            text[field.name] = spelled
            data[field.name] = spelled.map(folded)
    index = pd.RangeIndex(len(raw))
    return Frame(
        pd.DataFrame(data, index=index),
        pd.DataFrame(text, index=index),
        raw["row_number"].to_numpy(dtype=np.int64),
        total,
        sampled,
    )


def display_of(frame: Frame, name: str) -> dict[str, str]:
    """Each folded value of a category → its most frequent spelling (C-34)."""
    if name not in frame.text:
        return {}
    pairs = pd.DataFrame({"key": frame.data[name], "spelling": frame.text[name]}).dropna()
    if pairs.empty:
        return {}
    counts = pairs.value_counts(["key", "spelling"]).reset_index()  # key, spelling, count
    counts = counts.sort_values(["key", "count", "spelling"], ascending=[True, False, True])
    return dict(counts.drop_duplicates("key")[["key", "spelling"]].itertuples(index=False))


# --- Features ------------------------------------------------------------------------------------


def robust_scale(values: np.ndarray) -> tuple[float, float]:
    """Median and a robust spread (IQR / 1.349, else MAD × 1.4826, else the standard deviation)."""
    present = values[~np.isnan(values)]
    if present.size == 0:
        return 0.0, 1.0
    median = float(np.median(present))
    q1, q3 = np.percentile(present, [25, 75])
    spread = float(q3 - q1) / 1.349
    if spread <= 0:
        spread = float(np.median(np.abs(present - median))) * 1.4826
    if spread <= 0:
        spread = float(present.std())
    return median, spread if spread > 0 else 1.0


@dataclass
class Encoder:
    """Numbers centred on their median and divided by a robust spread (empty: 0, the median);
    categories as the share of their value, or one column per value (`onehot`, ≤ 10 values)."""

    fields: list[Field]
    scales: dict[str, tuple[float, float]]
    shares: dict[str, dict[Any, float]]
    onehot: dict[str, list[str]]

    @classmethod
    def fit(cls, frame: Frame, fields: Sequence[Field], onehot: bool = False) -> Encoder:
        scales, shares, hot = {}, {}, {}
        for field in fields:
            values = frame.data[field.name]
            if field.kind == "number":
                scales[field.name] = robust_scale(values.to_numpy(dtype=float))
                continue
            counts = values.fillna("\x00").value_counts(normalize=True)
            shares[field.name] = counts.to_dict()
            if onehot and len(counts) <= 10:
                hot[field.name] = [str(k) for k in counts.index]
        return cls(list(fields), scales, shares, hot)

    def transform(self, frame: Frame, missing: float = 0.0) -> np.ndarray:
        """`missing`: what an empty number becomes (0, the median; NaN for a distance that leaves
        it out)."""
        parts = []
        for field in self.fields:
            values = frame.data[field.name]
            if field.kind == "number":
                median, spread = self.scales[field.name]
                x = (values.to_numpy(dtype=float) - median) / spread
                parts.append(np.nan_to_num(np.clip(x, -1e6, 1e6), nan=missing)[:, None])
            elif field.name in self.onehot:
                keys = values.fillna("\x00").astype(str).to_numpy()
                parts.append(np.column_stack([keys == k for k in self.onehot[field.name]]) * 1.0)
            else:
                share = self.shares[field.name]
                keys = values.fillna("\x00").tolist()
                parts.append(np.array([share.get(k, 0.0) for k in keys], dtype=float)[:, None])
        if not parts:
            return np.zeros((len(frame), 0))
        return np.hstack(parts).astype(float)


def codes(frame: Frame, fields: Sequence[Field]) -> tuple[np.ndarray, np.ndarray]:
    """Numbers as they are (empty: NaN) and categories as codes, with the mask of the categories
    (for HistGradientBoosting's native categories)."""
    columns, mask = [], []
    for field in fields:
        values = frame.data[field.name]
        if field.kind == "number":
            columns.append(values.to_numpy(dtype=float))
            mask.append(False)
        else:
            categories = pd.Categorical(values)
            code = categories.codes.astype(float)
            code[code < 0] = np.nan
            columns.append(code)
            mask.append(True)
    if not columns:
        return np.zeros((len(frame), 0)), np.zeros(0, dtype=bool)
    return np.column_stack(columns), np.array(mask, dtype=bool)


# --- Suggestions ---------------------------------------------------------------------------------


def suggestion(
    rule: str,
    key: str,
    fields: Sequence[Field],
    title: str,
    reason: str,
    step: Any = None,
    cells: int = 0,
    rows: int = 0,
    examples: Sequence[tuple[str | None, str | None]] = (),
    number: int = 0,
) -> CleanSuggestion:
    """An Expert suggestion: mode S, risk "review", never in the default draft (C-30)."""
    op = step.op if step is not None else ""
    impact = STEP_IMPACT.get((rule, op)) or IMPACT.get(rule)
    assert impact is not None, f"{rule}: no impact in the catalogue"
    if step is not None:
        step.origin, step.rule = "suggested", rule
        step.id = f"s{FIRST_ID[rule] + number}"
    found = CleanSuggestion(
        id=f"{rule}:{key}",
        rule=rule,
        layer="expert",
        mode="S",
        impact=impact,
        risk="review",
        order=ORDER[rule] * 10_000 + (fields[0].position if fields else 0) * 10 + number,
        title=title,
        reason=reason,
        columns=[f.name for f in fields],
        cells=cells,
        rows=rows,
        examples=[Example(before=b, after=a) for b, a in list(examples)[:LISTED]],
        step=step,
    )
    if step is not None:
        step.suggestion = found.id
    return found
