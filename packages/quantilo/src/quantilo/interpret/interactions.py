"""Which pairs of columns the model combines: Friedman's H statistic (126, Explain › Advanced ›
More; Friedman and Popescu 2008).

For two columns j and k, PD_jk is the partial dependence on both, PD_j and PD_k on each alone,
all centred and read at the rows of a sample. Without interaction PD_jk = PD_j + PD_k; H² is the
share of the variance of PD_jk that the sum does not give:

    H²_jk = Σ_i (PD_jk(x_ij, x_ik) − PD_j(x_ij) − PD_k(x_ik))² / Σ_i PD_jk(x_ij, x_ik)²

0: the model adds the two effects; 1: the joint effect is all interaction.

R-XAI-13: the 10 most important columns (permutation importance; text, dates and identifiers
left out), 45 pairs; each partial dependence costs n² predictions, so the sample has
n = min(300, √(budget / (pairs + columns))) rows, budget 4·10⁶ predictions on a computer, 4·10⁵ in
the browser. H² is a ratio: when the joint effect is almost nothing (its variance under 1 % of the
predictions' variance) it becomes noise, and the pair is said to be too weak to judge rather than
ranked (Molnar, *Interpretable Machine Learning*, § 8.3). numpy only: Pyodide."""

import itertools
import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from quantilo.ml import browser
from quantilo.ml.ale import predictor
from quantilo.ml.target import Task

SEED = 42
COLUMNS = 10
MAX_COLUMNS = 15
MAX_ROWS = 300
MIN_ROWS = 30
BUDGET = 4_000_000
BROWSER_BUDGET = 400_000
WEAK = 0.01  # joint effect's variance, as a share of the predictions' variance
CHUNK = 50_000


class InteractionError(ValueError):
    """The interactions cannot be measured (with the reason)."""


@dataclass
class PairH:
    first: str
    second: str
    h: float  # √H², in [0, 1] once clipped
    share: float  # H²: the share of the joint effect that is interaction
    joint: float  # variance of the joint partial dependence / variance of the predictions
    weak: bool  # R-XAI-13: too weak to judge


@dataclass
class Interactions:
    columns: list[str]
    pairs: list[PairH]  # strongest first; weak pairs last
    rows: int
    output: str
    sentence: str
    skipped: list[str]  # columns left out, with their reason in `note`
    note: str | None = None


def budget() -> int:
    return BROWSER_BUDGET if browser.active() else BUDGET


def rows_within(columns: int) -> int:
    pairs = columns * (columns - 1) // 2
    return int(min(MAX_ROWS, math.isqrt(max(budget() // max(pairs + columns, 1), 0))))


def _predict(predict, frame: pd.DataFrame) -> np.ndarray:
    return np.concatenate([predict(frame.iloc[i : i + CHUNK]) for i in range(0, len(frame), CHUNK)])


def _dependence(predict, xs: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """The partial dependence on `columns` at each row of `xs` (rows × outputs), centred: row i's
    values of those columns given to every row, the predictions averaged."""
    n = len(xs)
    frame = pd.concat([xs] * n, ignore_index=True)
    for column in columns:
        frame[column] = np.repeat(cast(pd.Series, xs[column]).to_numpy(), n)
    found = _predict(predict, frame).reshape(n, n, -1).mean(axis=1)
    return found - found.mean(axis=0)


def interactions(
    pipeline: Pipeline,
    x: pd.DataFrame,
    task: Task,
    ranked: list[str],
    skip: dict[str, str] | None = None,
    count: int = COLUMNS,
) -> Interactions:
    """`ranked`: the inputs from the most to the least important; `skip`: column → why left out."""
    skip = skip or {}
    count = int(min(max(count, 2), MAX_COLUMNS))
    order = [str(c) for c in dict.fromkeys([*ranked, *map(str, x.columns)]) if c in x.columns]
    usable = [c for c in order if c not in skip]
    columns = usable[:count]
    if len(columns) < 2:
        raise InteractionError("Interactions need two columns the model can be read on.")
    n = min(rows_within(len(columns)), len(x))
    if n < MIN_ROWS:
        raise InteractionError(
            f"R-XAI-13: {len(columns)} columns leave fewer than {MIN_ROWS} rows within the budget: "
            "keep fewer columns."
        )
    predict, _, output = predictor(pipeline, task)
    rng = np.random.default_rng(SEED)
    xs = x.iloc[np.sort(rng.choice(len(x), n, replace=False))].reset_index(drop=True)
    variance = float(_predict(predict, xs).var(axis=0).sum())
    single = {c: _dependence(predict, xs, [c]) for c in columns}
    pairs: list[PairH] = []
    for first, second in itertools.combinations(columns, 2):
        joint = _dependence(predict, xs, [first, second])
        denominator = float((joint**2).sum())
        rest = joint - single[first] - single[second]
        share = float((rest**2).sum()) / denominator if denominator > 0 else 0.0
        spread = denominator / n / variance if variance > 0 else 0.0
        weak = spread < WEAK
        if weak:  # R-XAI-13: a ratio of two near-zero sums is noise, not an interaction
            share = 0.0
        pairs.append(
            PairH(first, second, math.sqrt(min(max(share, 0.0), 1.0)), share, spread, weak)
        )
    pairs.sort(key=lambda p: (p.weak, -p.h))
    strong = [p for p in pairs if not p.weak]
    said = output[:1].lower() + output[1:]  # “Probability of “Yes”” keeps its class's case
    if strong:
        top = strong[0]
        sentence = (
            f"The strongest interaction is {top.first} × {top.second} (H = {top.h:.2f}): "
            f"{min(top.share, 1.0):.0%} of their joint effect on the {said} is not the "
            "sum of their own effects."
        )
        if top.h < 0.1:
            sentence += " That is small: the model mostly adds its columns' effects."
    else:
        sentence = "No pair has a joint effect strong enough to judge (R-XAI-13)."
    left = [c for c in order if c in skip]
    note = None
    if left:
        note = "Left out: " + "; ".join(f"{c} ({skip[c]})" for c in left[:6])
        note += ", ..." if len(left) > 6 else ""
    return Interactions(columns, pairs, n, output, sentence, left, note)
