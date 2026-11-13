"""Prototypes and criticisms of a table's classes (126, Explain › Advanced › More; MMD-critic,
Kim, Khanna and Koyejo 2016). The vision module keeps its own prototypes (`prototypes.py`, 117);
these are rows of a table.

Prototypes: a few real rows whose distribution is closest to the class's, by the maximum mean
discrepancy (MMD²) under a Gaussian kernel, chosen greedily. Criticisms: the real rows that the
prototypes stand for worst, where the witness function

    w(x) = mean over the class of k(x, ·) − mean over the prototypes of k(x, ·)

is largest in size, chosen greedily with a log-determinant term so that two criticisms are not
the same row twice.

R-XAI-15: by true class (a number predicted: all rows, one group); at most 1,000 rows per class
(seeded draw); numbers centred and scaled (a missing value: the median), categories as
indicators of their 20 most frequent values × 1/√2 (two categories that differ are then as far
apart as numbers one standard deviation apart), free text, dates and identifiers left out; the
kernel's width by the median heuristic (the median squared distance between rows); 4 prototypes
(1 to 8), 2 criticisms (0 to 5). numpy only: Pyodide."""

import math
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from quantilo.ml.data import Field
from quantilo.ml.target import Task

SEED = 42
PROTOTYPES = 4
MAX_PROTOTYPES = 8
CRITICISMS = 2
MAX_CRITICISMS = 5
ROWS = 1_000  # per class
TOP_CATEGORIES = 20
SHOWN_COLUMNS = 6
MAX_GROUPS = 10


class PrototypeError(ValueError):
    """No prototypes can be chosen (with the reason)."""


@dataclass
class ExampleRow:
    row: int  # file row
    values: list[str]  # the shown columns' values
    predicted: str  # what the model gives the row
    share: float | None = None  # prototypes: the share of the class's rows closest to it
    witness: float | None = None  # criticisms: the witness value (positive: under-represented)


@dataclass
class ClassExamples:
    label: str
    rows: int  # rows of the class the choice was made on
    prototypes: list[ExampleRow]
    criticisms: list[ExampleRow]
    mmd: list[float]  # MMD² after each prototype added


@dataclass
class Prototypes:
    columns: list[str]  # the columns shown for each example (the most important first)
    groups: list[ClassExamples]
    width: float  # the kernel's width (median heuristic), in scaled units
    sentence: str
    left_out: list[str] = field(default_factory=list)


def encode(x: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """R-XAI-15: one row per row of `x`, numbers centred and scaled, categories as indicators."""
    parts: list[np.ndarray] = []
    for column in columns:
        raw = cast(pd.Series, x[column])
        if pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_bool_dtype(raw):
            values = raw.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue
            values = np.where(np.isfinite(values), values, float(np.median(finite)))
            spread = float(values.std())
            parts.append(((values - values.mean()) / (spread if spread > 0 else 1.0))[:, None])
        else:
            labels = raw.astype(str).where(raw.notna(), "(empty)")
            top = labels.value_counts().index[:TOP_CATEGORIES]
            block = np.stack([(labels == v).to_numpy(dtype=float) for v in top], axis=1)
            parts.append(block / math.sqrt(2))
    if not parts:
        return np.zeros((len(x), 0))
    return np.concatenate(parts, axis=1)


def squared_distances(z: np.ndarray) -> np.ndarray:
    norms = (z**2).sum(axis=1)
    return np.maximum(norms[:, None] + norms[None, :] - 2 * z @ z.T, 0.0)


def width_of(distances: np.ndarray) -> float:
    """The median heuristic: the median squared distance between two different rows."""
    n = len(distances)
    off = distances[~np.eye(n, dtype=bool)] if n > 1 else np.array([1.0])
    positive = off[off > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def mmd2(kernel: np.ndarray, chosen: list[int]) -> float:
    """MMD² between the class (every row of `kernel`) and the rows `chosen`."""
    if not chosen:
        return float(kernel.mean())
    return float(
        kernel.mean() - 2 * kernel[:, chosen].mean() + kernel[np.ix_(chosen, chosen)].mean()
    )


def choose_prototypes(kernel: np.ndarray, count: int) -> tuple[list[int], list[float]]:
    """Greedy: each step adds the row that lowers MMD² the most."""
    n = len(kernel)
    column_mean = kernel.mean(axis=0)
    chosen: list[int] = []
    trail: list[float] = []
    cross = 0.0  # Σ over chosen of the class mean similarity
    inner = 0.0  # Σ over chosen pairs (both orders and the diagonal) of the kernel
    to_chosen = np.zeros(n)  # Σ over chosen of k(c, j), for each candidate c
    for _ in range(min(count, n)):
        m = len(chosen) + 1
        gain = 2 * (cross + column_mean) / m - (inner + 2 * to_chosen + np.diag(kernel)) / m**2
        gain[chosen] = -np.inf
        best = int(np.argmax(gain))
        cross += float(column_mean[best])
        inner += 2 * float(to_chosen[best]) + float(kernel[best, best])
        to_chosen += kernel[:, best]
        chosen.append(best)
        trail.append(mmd2(kernel, chosen))
    return chosen, trail


def choose_criticisms(
    kernel: np.ndarray, prototypes: list[int], count: int
) -> tuple[list[int], np.ndarray]:
    """Greedy on Σ |w(c)| + log det K_CC (the second term keeps criticisms apart)."""
    witness = kernel.mean(axis=1) - kernel[:, prototypes].mean(axis=1)
    chosen: list[int] = []
    free = [i for i in range(len(kernel)) if i not in set(prototypes)]
    for _ in range(min(count, len(free))):
        best, score = -1, -np.inf
        for c in free:
            if c in chosen:
                continue
            rows = [*chosen, c]
            sign, logdet = np.linalg.slogdet(kernel[np.ix_(rows, rows)] + 1e-9 * np.eye(len(rows)))
            if sign <= 0:
                continue
            value = float(np.abs(witness[rows]).sum() + logdet)
            if value > score:
                best, score = c, value
        if best < 0:
            break
        chosen.append(best)
    return chosen, witness


def _label(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "(empty)"
    if isinstance(value, float | np.floating):
        return f"{float(value):.4g}"
    return str(value)[:40]


def _left_out(x: pd.DataFrame, fields: list[Field], texts: set[str]) -> dict[str, str]:
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


def prototypes(
    pipeline: Pipeline | None,
    x: pd.DataFrame,
    y: pd.Series | np.ndarray,
    rows: np.ndarray,
    task: Task,
    fields: list[Field],
    texts: set[str] | None = None,
    ranked: list[str] | None = None,
    count: int = PROTOTYPES,
    criticisms: int = CRITICISMS,
) -> Prototypes:
    """`x`, `y`: the evaluation rows and their true values (file rows in `rows`); `ranked`: the
    inputs by importance (the first six are shown for each example)."""
    count = int(min(max(count, 1), MAX_PROTOTYPES))
    criticisms = int(min(max(criticisms, 0), MAX_CRITICISMS))
    left = _left_out(x, fields, texts or set())
    order = [
        str(c) for c in dict.fromkeys([*(ranked or []), *map(str, x.columns)]) if c in x.columns
    ]
    used = [c for c in order if c not in left]
    if not used or x.empty:
        raise PrototypeError("No column can be compared between rows (text, dates, identifiers).")
    labels = np.asarray(y).astype(str) if task != "regression" else np.full(len(x), "All rows")
    groups_found = pd.Series(labels).value_counts().index[:MAX_GROUPS]
    rng = np.random.default_rng(SEED)
    shown = used[:SHOWN_COLUMNS]
    groups: list[ClassExamples] = []
    widths: list[float] = []
    scaled = encode(x, used)  # one scale for every class: a class's own spread would inflate it
    for label in groups_found:
        where = np.flatnonzero(labels == label)
        if len(where) > ROWS:
            where = np.sort(rng.choice(where, ROWS, replace=False))
        part = x.iloc[where].reset_index(drop=True)
        z = scaled[where]
        distances = squared_distances(z)
        width = width_of(distances)
        widths.append(width)
        kernel = np.exp(-distances / width)
        chosen, trail = choose_prototypes(kernel, count)
        critics, witness = (
            choose_criticisms(kernel, chosen, criticisms) if criticisms else ([], None)
        )
        nearest = np.argmax(kernel[:, chosen], axis=1)
        picked = [*chosen, *critics]
        if pipeline is not None and picked:
            predicted = np.asarray(pipeline.predict(part.iloc[picked]))
        else:
            predicted = np.array([""] * len(picked))

        def example(
            i: int,
            k: int,
            share: float | None,
            w: float | None,
            part: pd.DataFrame = part,
            predicted: np.ndarray = predicted,
            where: np.ndarray = where,
        ) -> ExampleRow:
            values = [_label(part.iloc[i][c]) for c in shown]
            p = predicted[k]
            text = _label(float(p)) if task == "regression" and str(p) else str(p)
            return ExampleRow(int(rows[where[i]]), values, text, share, w)

        protos = [example(i, k, float((nearest == k).mean()), None) for k, i in enumerate(chosen)]
        crits = [
            example(i, len(chosen) + k, None, float(witness[i]) if witness is not None else None)
            for k, i in enumerate(critics)
        ]
        groups.append(ClassExamples(str(label), len(where), protos, crits, trail))
    first = groups[0]
    share = first.prototypes[0].share if first.prototypes else 0.0
    sentence = (
        f"{len(first.prototypes)} real rows summarise "
        f"{'the rows' if task == 'regression' else f'the class “{first.label}”'}: the first "
        f"stands for {share:.0%} of them"
    )
    sentence += (
        f"; {len(first.criticisms)} rows are what this summary misses most."
        if first.criticisms
        else "."
    )
    return Prototypes(
        shown,
        groups,
        float(np.median(widths)) if widths else 1.0,
        sentence,
        [f"{c} ({why})" for c, why in left.items()],
    )
