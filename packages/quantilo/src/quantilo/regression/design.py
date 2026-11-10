"""The design matrix (067): numbers as they are, categories as indicators against their most
frequent level (Top 20 + Others), incomplete rows left out and counted, and the columns that
cannot be estimated (constant, exact combination of others) left out with their reason."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

MAX_LEVELS = 20  # R-REG-22: beyond, the levels go to "Others"
MAX_TERMS = 200  # columns of the design, indicators included (study § 3.5)
COLLINEAR = 1e-9  # residual share of a column's variance below this: an exact combination
OTHERS = "Others"
INTERCEPT = "(Intercept)"

Kind = Literal["num", "cat"]


class DesignError(ValueError):
    pass


@dataclass
class Variable:
    """A column as loaded: floats with NaN, or category codes (most frequent first, −1 missing)."""

    name: str
    kind: Kind
    values: np.ndarray
    labels: list[str] = field(default_factory=list)  # by code, for categories


@dataclass
class ReferenceLevel:
    column: str
    level: str
    levels: int  # levels kept as indicators (the reference not counted)
    others: int  # levels merged into "Others"
    reason: str


@dataclass
class LeftOutTerm:
    term: str
    column: str
    reason: str


@dataclass
class MissingRows:
    column: str
    rows: int  # rows of the read sample without a value in this column


@dataclass
class Design:
    x: np.ndarray  # n_used × k, intercept first
    y: np.ndarray
    terms: list[str]
    columns: list[str]  # the source column of each term
    levels: list[str | None]  # the level of an indicator
    used: np.ndarray  # mask over the rows read
    references: list[ReferenceLevel]
    left_out: list[LeftOutTerm]
    missing: list[MissingRows]


def codes(labels: list[str | None]) -> tuple[np.ndarray, list[str]]:
    """Labels → codes by frequency (most frequent first; ties by first appearance), −1 missing."""
    present = [label for label in labels if label is not None]
    distinct, first, counts = np.unique(
        np.array(present, dtype=object), return_index=True, return_counts=True
    )
    order = sorted(range(len(distinct)), key=lambda i: (-counts[i], first[i]))
    rank = {str(distinct[i]): r for r, i in enumerate(order)}
    values = np.array(
        [rank[str(label)] if label is not None else -1 for label in labels], dtype=int
    )
    return values, [str(distinct[i]) for i in order]


def _indicators(
    variable: Variable, used: np.ndarray
) -> tuple[np.ndarray, list[str], ReferenceLevel]:
    """Indicator columns of a category over the used rows; the most frequent level is the
    reference. Levels beyond MAX_LEVELS share one "Others" column."""
    values = variable.values[used]
    counts = np.bincount(values, minlength=len(variable.labels))
    order = [int(c) for c in np.argsort(-counts, kind="stable") if counts[c] > 0]
    reference, rest = order[0], order[1:]
    kept, merged = rest[:MAX_LEVELS], rest[MAX_LEVELS:]
    names = [variable.labels[c] for c in kept]
    columns = [(values == c).astype(float) for c in kept]
    if merged:
        columns.append(np.isin(values, merged).astype(float))
        names.append(OTHERS)
    label = variable.labels[reference]
    share = counts[reference] / max(len(values), 1)
    why = f"most frequent level ({share:.0%} of the rows): the narrowest intervals"
    found = ReferenceLevel(variable.name, label, len(kept), len(merged), why)
    matrix = np.column_stack(columns) if columns else np.empty((len(values), 0))
    return matrix, names, found


def _combination(names: list[str], coefficients: np.ndarray, scale: np.ndarray) -> str:
    """ "`a` + `b`", "2 × `a`": the columns a left-out one is made of."""
    weight = np.abs(coefficients) * scale
    keep = [i for i in np.argsort(-weight) if weight[i] > 1e-6 * max(weight.max(), 1e-300)][:3]
    parts = []
    for i in keep:
        c = coefficients[i]
        sign = "−" if c < 0 else "+"
        size = "" if abs(abs(c) - 1) < 1e-6 else f"{abs(c):.3g} × "
        parts.append(f"{sign} {size}`{names[i]}`")
    text = " ".join(parts).removeprefix("+ ")
    return text + (" + …" if len(keep) < int((weight > 1e-6 * weight.max()).sum()) else "")


def build(
    target: np.ndarray,
    variables: list[Variable],
    extra: dict[str, np.ndarray] | None = None,
) -> Design:
    """`extra`: other arrays a used row needs (weights, groups): NaN or −1 marks them missing."""
    used = np.isfinite(target)
    missing: list[MissingRows] = []
    named: list[tuple[str, np.ndarray]] = [
        (v.name, np.isfinite(v.values) if v.kind == "num" else v.values >= 0) for v in variables
    ]
    for name, values in (extra or {}).items():
        present = values >= 0 if values.dtype.kind in "iu" else np.isfinite(values)
        named.append((name, present))
    for name, present in named:
        if (lost := int((~present).sum())) > 0:
            missing.append(MissingRows(name, lost))
        used &= present
    if (lost := int((~np.isfinite(target)).sum())) > 0:
        missing.insert(0, MissingRows("target", lost))
    if int(used.sum()) < 3:
        raise DesignError("Fewer than 3 complete rows: nothing to estimate.")

    blocks: list[np.ndarray] = [np.ones((int(used.sum()), 1))]
    terms, columns, levels = [INTERCEPT], [INTERCEPT], [None]
    references: list[ReferenceLevel] = []
    for variable in variables:
        if variable.kind == "num":
            blocks.append(variable.values[used][:, None])
            terms.append(variable.name)
            columns.append(variable.name)
            levels.append(None)
            continue
        matrix, names, reference = _indicators(variable, used)
        references.append(reference)
        blocks.append(matrix)
        terms += [f"{variable.name} = {name}" for name in names]
        columns += [variable.name] * len(names)
        levels += names
    x = np.hstack(blocks)
    if x.shape[1] > MAX_TERMS:
        raise DesignError(
            f"{x.shape[1]} columns once categories become indicators: at most {MAX_TERMS}. "
            "Leave out a category with many levels."
        )
    keep, left_out = _estimable(x, terms, columns)
    return Design(
        x[:, keep],
        target[used],
        [terms[i] for i in keep],
        [columns[i] for i in keep],
        [levels[i] for i in keep],
        used,
        references,
        left_out,
        missing,
    )


def _estimable(
    x: np.ndarray, terms: list[str], columns: list[str]
) -> tuple[list[int], list[LeftOutTerm]]:
    """Walk the columns in order; keep one unless it is constant or an exact combination of the
    columns kept before it (R-REG-13). Works on the centred Gram matrix (k × k)."""
    n, k = x.shape
    mean = x.mean(axis=0)
    gram = (x.T @ x) / n - np.outer(mean, mean)  # covariance (biased) of every column pair
    variance = np.diag(gram).copy()
    scale = np.sqrt(np.clip(variance, 0, None))
    keep = [0]
    left_out: list[LeftOutTerm] = []
    others: list[int] = []  # kept non-intercept columns
    for j in range(1, k):
        if variance[j] <= 1e-12 * max(abs(mean[j]) ** 2, 1e-300) or variance[j] == 0:
            reason = "constant on the rows used" if np.ptp(x[:, j]) == 0 else "almost constant"
            if columns[j] != terms[j]:
                reason = "no row left at this level" if not x[:, j].any() else reason
            left_out.append(LeftOutTerm(terms[j], columns[j], reason))
            continue
        if others:
            sub = gram[np.ix_(others, others)]
            cross = gram[others, j]
            try:
                beta = np.linalg.solve(sub, cross)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(sub, cross, rcond=None)[0]
            residual = variance[j] - cross @ beta
            if residual <= COLLINEAR * variance[j]:
                made = _combination([terms[i] for i in others], beta, scale[others] / scale[j])
                left_out.append(
                    LeftOutTerm(
                        terms[j],
                        columns[j],
                        f"equals {made} (with a constant): perfectly collinear",
                    )
                )
                continue
        keep.append(j)
        others.append(j)
    return keep, left_out
