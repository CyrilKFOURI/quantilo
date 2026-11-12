"""The design matrix (067): numbers as they are, categories as indicators against their most
frequent level (Top 20 + Others), incomplete rows left out and counted, and the columns that
cannot be estimated (constant, exact combination of others) left out with their reason."""

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from quantilo.regression import terms as tm

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
    source: str | None = None  # 070: the dataset column of a log term or a date part


@dataclass
class ReferenceLevel:
    column: str
    level: str
    levels: int  # levels kept as indicators (the reference not counted)
    others: int  # levels merged into "Others"
    reason: str
    coding: str = "reference"  # 070: "deviation": each level against the average of the levels


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
    # 070: the term each column comes from, how it is computed, what the terms did
    groups: list[str] = field(default_factory=list)
    recipes: list[tm.Recipe] = field(default_factory=list)
    notes: list[tm.TermNote] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)  # display name → dataset column
    kinds: dict[str, tuple[str, list[str]]] = field(default_factory=dict)  # variable → kind, labels
    values: dict[str, np.ndarray] = field(default_factory=dict)  # over the used rows
    centres: dict[str, float] = field(default_factory=dict)  # centred numbers (R-REG-TERM-05)


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
    variable: Variable, used: np.ndarray, chosen: str | None = None, coding: str = "reference"
) -> tuple[np.ndarray, list[str], ReferenceLevel, list[tm.Recipe], str | None]:
    """Indicator columns of a category over the used rows; the most frequent level is the
    reference unless one is chosen (R-REG-TERM-06). Levels beyond MAX_LEVELS share one "Others"
    column. Deviation coding: the reference rows are −1 in every column."""
    values = variable.values[used]
    counts = np.bincount(values, minlength=len(variable.labels))
    order = [int(c) for c in np.argsort(-counts, kind="stable") if counts[c] > 0]
    refused = None
    picked = variable.labels.index(chosen) if chosen in variable.labels else None
    if chosen is not None and (picked is None or counts[picked] == 0):
        refused = f"`{chosen}` is not a level of `{variable.name}` on the rows used"
        picked = None
    reference = picked if picked is not None else order[0]
    rest = [c for c in order if c != reference]
    kept, merged = rest[:MAX_LEVELS], rest[MAX_LEVELS:]
    names = [variable.labels[c] for c in kept]
    deviation = coding == "deviation"
    sign = (values == reference).astype(float) if deviation else 0.0
    columns = [(values == c).astype(float) - sign for c in kept]
    off = reference if deviation else -1
    recipes = [tm.Recipe("ind", (variable.name,), levels=(c,), reference=off) for c in kept]
    if merged:
        columns.append(np.isin(values, merged).astype(float) - sign)
        names.append(OTHERS)
        recipes.append(tm.Recipe("ind", (variable.name,), levels=tuple(merged), reference=off))
    label = variable.labels[reference]
    share = counts[reference] / max(len(values), 1)
    if picked is not None:
        why = "chosen in the settings (R-REG-TERM-06)"
    else:
        why = f"most frequent level ({share:.0%} of the rows): the narrowest intervals"
    if deviation:
        why += (
            "; each level reads against the average of the levels (deviation coding), this one "
            "by contrast"
        )
    found = ReferenceLevel(
        variable.name,
        label,
        len(kept),
        len(merged),
        why,
        "deviation" if deviation else "reference",
    )
    matrix = np.column_stack(columns) if columns else np.empty((len(values), 0))
    return matrix, names, found, recipes, refused


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
    plan: tm.Plan | None = None,
) -> Design:
    """`extra`: other arrays a used row needs (weights, groups): NaN or −1 marks them missing.
    `plan` (070): curves, logs, interactions, contrasts and centring."""
    plan = plan or tm.Plan()
    prepared = tm.prepare_logs(
        [(v.name, v.kind, v.values if v.kind == "num" else None) for v in variables], plan
    )
    for v in variables:
        if v.source:
            prepared.sources[v.name] = v.source
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
    if prepared.transforms:
        variables = [
            replace(
                v,
                name=tm.log_name(v.name, how),
                values=tm.apply_log(v.values, how),
                source=v.name,
            )
            if (how := prepared.transforms.get(v.name))
            else v
            for v in variables
        ]

    n = int(used.sum())
    blocks: list[np.ndarray] = [np.ones((n, 1))]
    terms, columns, levels = [INTERCEPT], [INTERCEPT], [None]
    groups, recipes = [""], [tm.Recipe("const")]
    references: list[ReferenceLevel] = []
    notes = list(prepared.notes)
    main: dict[str, tuple[str, list[tm.Recipe], np.ndarray, list[str]]] = {}
    kinds: dict[str, tuple[str, list[str]]] = {}
    values_used: dict[str, np.ndarray] = {}
    centres: dict[str, float] = {}
    for variable in variables:
        kinds[variable.name] = (variable.kind, variable.labels)
        if variable.kind == "num":
            values = variable.values[used]
            values_used[variable.name] = values
            centre = float(values.mean()) if plan.center else 0.0
            if not plan.center and variable.source and variable.name.endswith(" › Trend"):
                centre = float(np.floor(values.min()))  # R-REG-TERM-08: years since a round year
            if plan.center:
                centres[variable.name] = centre
            recipe = tm.Recipe("num", (variable.name,), centre=centre)
            column = (values - centre)[:, None]
            blocks.append(column)
            main[variable.name] = ("num", [recipe], column, [])
            terms.append(variable.name)
            columns.append(variable.name)
            levels.append(None)
            groups.append(variable.name)
            recipes.append(recipe)
            continue
        values_used[variable.name] = variable.values[used]
        chosen = plan.reference(variable.name)
        matrix, names, reference, made, refused = _indicators(variable, used, chosen, plan.coding)
        if refused:
            notes.append(
                tm.TermNote(
                    f"reference:{variable.name}|{chosen}",
                    f"reference of `{variable.name}`",
                    "refused",
                    refused + "; the most frequent level stays the reference",
                    "R-REG-TERM-06",
                )
            )
        references.append(reference)
        blocks.append(matrix)
        main[variable.name] = ("cat", made, matrix, names)
        terms += [f"{variable.name} = {name}" for name in names]
        columns += [variable.name] * len(names)
        levels += names
        groups += [variable.name] * len(names)
        recipes += made
    notes += tm.hierarchy_notes(plan, {v.source or v.name for v in variables})
    for v in variables:  # R-REG-TERM-08: a date as a trend, a month or a day of the week
        kind = next(
            (k for k in tm.DATE_KINDS if v.source and v.name == tm.date_name(v.source, k)), None
        )
        if kind is not None and v.source is not None:
            what = {
                "trend": "the date in decimal years (2014.5 = mid-2014): effects read per year",
                "month": "the month of the year, each against the most frequent one",
                "weekday": "the day of the week, each against the most frequent one",
            }[kind]
            notes.append(
                tm.TermNote(f"{kind}:{v.source}", f"`{v.name}`", "used", what, "R-REG-TERM-08")
            )
    for block in tm.blocks(plan, main, values_used, notes, prepared.sources):
        blocks.append(block.matrix)
        terms += block.names
        columns += block.columns
        levels += [None] * len(block.names)
        groups += [block.group] * len(block.names)
        recipes += block.recipes
    for spec in plan.specs:
        if spec.kind == "reference" and spec.columns[0] not in kinds:
            notes.append(
                tm.TermNote(
                    spec.key,
                    f"reference of `{spec.columns[0]}`",
                    "refused",
                    f"`{spec.columns[0]}` is not a category of the model",
                    "R-REG-TERM-06",
                )
            )
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
        [groups[i] for i in keep],
        [recipes[i] for i in keep],
        notes,
        prepared.sources,
        kinds,
        values_used,
        centres,
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
