"""Terms of a regression (070): curves (restricted cubic splines, squares and cubes taken around
the mean), the log of a column, chosen interactions, dates as terms (trend, month, day of the
week), contrasts of the categories (a chosen reference, deviation from the average of the
levels) and centring. Each term carries its reason (R-REG-TERM-01…11); a term made of several
columns is read through a joint test and an effect curve, not coefficient by coefficient.

Everything here is numpy (Pyodide-compatible). `design.build` asks this module for the columns
of each term; the Terms section and the partial plots read the fit back through the same
recipes, so a curve is drawn with exactly the columns that were estimated."""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

TermKind = Literal[
    "spline", "square", "cube", "log", "interaction", "trend", "month", "weekday", "reference"
]
KINDS: tuple[TermKind, ...] = (
    "spline",
    "square",
    "cube",
    "log",
    "interaction",
    "trend",
    "month",
    "weekday",
    "reference",
)
DATE_KINDS: tuple[TermKind, ...] = ("trend", "month", "weekday")
DATE_LABELS = {"trend": "Trend", "month": "Month", "weekday": "Day of week"}
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
Coding = Literal["reference", "deviation"]
KNOTS_SMALL = (0.10, 0.50, 0.90)  # R-REG-TERM-02, under SMALL_N rows (Harrell)
KNOTS = (0.05, 0.275, 0.50, 0.725, 0.95)
SMALL_N = 100
CURVE_POINTS = 40  # R-REG-TERM-10
CURVE_RANGE = (0.01, 0.99)
PARTNER_CURVES = 5
PARTIAL_POINTS = 2_000  # R-REG-TERM-11
SEED = 42


class TermError(ValueError):
    pass


# --- The terms asked for ------------------------------------------------------------------------


@dataclass(frozen=True)
class TermSpec:
    kind: TermKind
    columns: tuple[str, ...]
    level: str | None = None  # the reference level of `reference:column|level`

    @property
    def key(self) -> str:
        if self.kind == "reference":
            return f"reference:{self.columns[0]}|{self.level}"
        return f"{self.kind}:{'|'.join(self.columns)}"


def parse(text: str) -> TermSpec:
    """`kind:column`, `interaction:a|b`, `reference:column|level`."""
    kind, sep, rest = text.strip().partition(":")
    if not sep or kind not in KINDS or not rest:
        raise TermError(
            f"`{text}` is not a term: write kind:column, the kind being one of {', '.join(KINDS)}."
        )
    if kind == "interaction":
        first, sep, second = rest.partition("|")
        if not sep or not first or not second or first == second:
            raise TermError("An interaction names two different columns: interaction:a|b.")
        return TermSpec("interaction", (first, second))
    if kind == "reference":
        column, sep, level = rest.partition("|")
        if not sep or not column or not level:
            raise TermError("A reference names a column and one of its levels: reference:c|level.")
        return TermSpec("reference", (column,), level)
    return TermSpec(kind, (rest,))  # type: ignore[arg-type]


@dataclass(frozen=True)
class Plan:
    """The terms of a regression beyond the columns as they are."""

    specs: tuple[TermSpec, ...] = ()
    center: bool = False
    coding: Coding = "reference"
    asked: tuple[str, ...] = ()  # the explanatory columns chosen (to say which ones a term adds)

    def logs(self) -> set[str]:
        return {s.columns[0] for s in self.specs if s.kind == "log"}

    def reference(self, column: str) -> str | None:
        return next(
            (s.level for s in self.specs if s.kind == "reference" and s.columns[0] == column),
            None,
        )

    def columns(self) -> list[str]:
        """Every dataset column a term reads as it is (dates apart)."""
        found: list[str] = []
        for spec in self.specs:
            if spec.kind in DATE_KINDS:
                continue
            for name in spec.columns:
                if name not in found:
                    found.append(name)
        return found

    def dates(self) -> list[TermSpec]:
        return [s for s in self.specs if s.kind in DATE_KINDS]


def plan_of(
    texts: tuple[str, ...] | list[str],
    center: bool = False,
    coding: Coding = "reference",
    asked: tuple[str, ...] = (),
) -> Plan:
    found: dict[str, TermSpec] = {}
    for text in texts:
        if not text.strip():
            continue
        spec = parse(text)
        if spec.kind == "reference":  # one reference per column, the last one wins
            found = {
                k: v
                for k, v in found.items()
                if not (v.kind == "reference" and v.columns == spec.columns)
            }
        found[spec.key] = spec
    specs = list(found.values())
    cubes = {s.columns[0] for s in specs if s.kind == "cube"}
    specs = [s for s in specs if not (s.kind == "square" and s.columns[0] in cubes)]
    return Plan(tuple(specs), center, coding, asked)


def date_name(column: str, kind: str) -> str:
    return f"{column} › {DATE_LABELS[kind]}"


@dataclass
class TermNote:
    """What a term did, or why it could not (shown in Advanced › Terms)."""

    key: str
    label: str
    status: Literal["used", "added", "refused"]
    reason: str
    rule: str


# --- Recipes: how each column of the design is computed -----------------------------------------


@dataclass(frozen=True)
class Recipe:
    """One column of the design as a function of the variables' values, so that a curve or a
    partial plot recomputes exactly the columns that were estimated."""

    kind: Literal["const", "num", "ind", "spline", "power", "product"]
    variables: tuple[str, ...] = ()
    centre: float = 0.0  # num: subtracted; power: the mean the power is taken around
    power: int = 1
    knots: tuple[float, ...] = ()
    basis: int = 0  # spline: which nonlinear column
    levels: tuple[int, ...] = ()  # ind: the codes this column is 1 for
    reference: int = -1  # ind, deviation coding: the code that is −1
    parts: tuple["Recipe", ...] = ()  # product


def rcs(x: np.ndarray, knots: tuple[float, ...] | np.ndarray) -> np.ndarray:
    """The k − 2 nonlinear columns of a restricted cubic spline (Harrell's rcspline.eval, norm 2):
    cubic between the knots, linear beyond the outer ones; x itself is the linear part."""
    t = np.asarray(knots, dtype=float)
    k = len(t)
    scale = (t[-1] - t[0]) ** 2 or 1.0
    cols = []
    for j in range(k - 2):
        col = (
            np.maximum(x - t[j], 0) ** 3
            - np.maximum(x - t[k - 2], 0) ** 3 * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
            + np.maximum(x - t[k - 1], 0) ** 3 * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
        )
        cols.append(col / scale)
    return np.column_stack(cols) if cols else np.empty((len(x), 0))


def evaluate(recipe: Recipe, values: dict[str, np.ndarray], n: int) -> np.ndarray:
    if recipe.kind == "const":
        return np.ones(n)
    if recipe.kind == "num":
        return values[recipe.variables[0]] - recipe.centre
    if recipe.kind == "ind":
        codes = values[recipe.variables[0]]
        column = np.isin(codes, recipe.levels).astype(float)
        if recipe.reference >= 0:
            column -= (codes == recipe.reference).astype(float)
        return column
    if recipe.kind == "spline":
        return rcs(values[recipe.variables[0]], recipe.knots)[:, recipe.basis]
    if recipe.kind == "power":
        return (values[recipe.variables[0]] - recipe.centre) ** recipe.power
    out = np.ones(n)
    for part in recipe.parts:
        out = out * evaluate(part, values, n)
    return out


def uses(recipe: Recipe) -> set[str]:
    if recipe.kind == "product":
        return set().union(*(uses(p) for p in recipe.parts))
    return set(recipe.variables)


# --- Building the terms (called by design.build) ------------------------------------------------


@dataclass
class Block:
    """Columns of the design added by one term."""

    matrix: np.ndarray
    names: list[str]
    columns: list[str]  # the display column of each name
    group: str
    recipes: list[Recipe]


@dataclass
class Prepared:
    transforms: dict[str, str]  # variable name → "log" or "log1p"
    sources: dict[str, str]  # display name → dataset column
    notes: list[TermNote] = field(default_factory=list)


def log_name(column: str, kind: str) -> str:
    return f"log({column})" if kind == "log" else f"log(1 + {column})"


def prepare_logs(names_kinds: list[tuple[str, str, np.ndarray | None]], plan: Plan) -> Prepared:
    """R-REG-TERM-04: which numeric columns become log(x) or log(1 + x), or why not.
    `names_kinds`: (name, kind, values) of every variable."""
    found = Prepared({}, {})
    wanted = plan.logs()
    for name, kind, values in names_kinds:
        found.sources[name] = name
        if name not in wanted:
            continue
        key = f"log:{name}"
        if kind != "num" or values is None:
            found.notes.append(
                TermNote(key, f"log({name})", "refused", "a category has no log", "R-REG-TERM-04")
            )
            continue
        present = values[np.isfinite(values)]
        low = float(present.min()) if len(present) else math.nan
        if not math.isfinite(low) or low < 0:
            found.notes.append(
                TermNote(
                    key,
                    f"log({name})",
                    "refused",
                    f"`{name}` has values below 0 (down to {low:.3g}): no log; kept as it is",
                    "R-REG-TERM-04",
                )
            )
            continue
        how = "log" if low > 0 else "log1p"
        found.transforms[name] = how
        label = log_name(name, how)
        found.sources[label] = name
        reason = (
            f"`{label}` in place of `{name}`: effects read per 1 % rise"
            if how == "log"
            else f"`{name}` has zeros: `{label}` in place of it (the log of 0 does not exist)"
        )
        found.notes.append(TermNote(key, label, "used", reason, "R-REG-TERM-04"))
    return found


def apply_log(values: np.ndarray, how: str) -> np.ndarray:
    return np.log(values) if how == "log" else np.log1p(values)


def knots_of(x: np.ndarray) -> tuple[float, ...]:
    """R-REG-TERM-02: Harrell's knot quantiles, duplicates dropped."""
    levels = KNOTS_SMALL if len(x) < SMALL_N else KNOTS
    found = np.unique(np.quantile(x, levels))
    return tuple(float(v) for v in found)


def blocks(
    plan: Plan,
    main: dict[str, tuple[str, list[Recipe], np.ndarray, list[str]]],
    values: dict[str, np.ndarray],
    notes: list[TermNote],
    sources: dict[str, str] | None = None,
) -> list[Block]:
    """The columns of the curves and interactions over the used rows.

    `main`: variable name → (kind "num"/"cat", recipes, matrix, level names) of its main effect;
    `values`: transformed values (num) or codes (cat) over the used rows."""
    found: list[Block] = []
    sources = sources or {}
    for spec in plan.specs:
        if spec.kind in ("log", "reference", *DATE_KINDS):
            continue
        names = [_display(c, main, sources) for c in spec.columns]
        missing = [c for c, name in zip(spec.columns, names, strict=True) if name not in main]
        if missing:
            notes.append(
                TermNote(
                    spec.key,
                    spec.key,
                    "refused",
                    f"`{missing[0]}` is not in the model",
                    "R-REG-TERM-01",
                )
            )
            continue
        if spec.kind == "interaction":
            block = _interaction(spec, names, main)
            if block is not None:
                found.append(block)
                notes.append(_interaction_note(spec, names, main, plan))
            continue
        name = names[0]
        kind = main[name][0]
        if kind != "num":
            notes.append(
                TermNote(
                    spec.key,
                    spec.key,
                    "refused",
                    f"`{name}` is a category: its levels already take any shape",
                    "R-REG-TERM-02",
                )
            )
            continue
        x = values[name]
        if spec.kind == "spline":
            knots = knots_of(x)
            if len(knots) < 3:
                notes.append(
                    TermNote(
                        spec.key,
                        f"curve of `{name}`",
                        "refused",
                        f"too few distinct values in `{name}` for a curve: use it as a "
                        "category or a square",
                        "R-REG-TERM-02",
                    )
                )
                continue
            basis = rcs(x, knots)
            label = [f"{name} · curve {j + 1}" for j in range(basis.shape[1])]
            recipe = [
                Recipe("spline", (name,), knots=knots, basis=j) for j in range(basis.shape[1])
            ]
            found.append(Block(basis, label, [name] * len(label), spec.key, recipe))
            shown = ", ".join(_short(k) for k in knots)
            notes.append(
                TermNote(
                    spec.key,
                    f"curve of `{name}`",
                    "used",
                    f"restricted cubic spline, {len(knots)} knots at {shown} "
                    "(quantiles); straight beyond the outer knots",
                    "R-REG-TERM-02",
                )
            )
            continue
        mean = float(x.mean())
        powers = (2,) if spec.kind == "square" else (2, 3)
        sup = {2: "²", 3: "³"}
        matrix = np.column_stack([(x - mean) ** p for p in powers])
        label = [f"{name}{sup[p]}" for p in powers]
        recipe = [Recipe("power", (name,), centre=mean, power=p) for p in powers]
        found.append(Block(matrix, label, [name] * len(label), spec.key, recipe))
        notes.append(
            TermNote(
                spec.key,
                " and ".join(f"`{t}`" for t in label),
                "used",
                f"taken around the mean of `{name}` ({_short(mean)}): the slope of "
                f"`{name}` reads at its mean",
                "R-REG-TERM-03",
            )
        )
    return found


def _display(column: str, names: Iterable[str], sources: dict[str, str]) -> str:
    """A term names a dataset column; its main effect may be log(column) or a date's trend."""
    names = list(names)
    if column in names:
        return column
    return next((name for name in names if sources.get(name) == column), column)


def _is_log(name: str) -> bool:
    return name.startswith("log(")


def _interaction(
    spec: TermSpec,
    names: list[str],
    main: dict[str, tuple[str, list[Recipe], np.ndarray, list[str]]],
) -> Block | None:
    """R-REG-TERM-07: products of the main-effect columns of the two variables."""
    a, b = names
    kind_a, rec_a, mat_a, lev_a = main[a]
    kind_b, rec_b, mat_b, lev_b = main[b]
    if kind_a == "cat" and kind_b == "num":  # the number first: "slope of a by level of b"
        a, b = b, a
        kind_a, rec_a, mat_a, lev_a, kind_b, rec_b, mat_b, lev_b = (
            kind_b,
            rec_b,
            mat_b,
            lev_b,
            kind_a,
            rec_a,
            mat_a,
            lev_a,
        )
    if mat_a.shape[1] == 0 or mat_b.shape[1] == 0:
        return None
    columns, names_out, recipes = [], [], []
    for i in range(mat_a.shape[1]):
        for j in range(mat_b.shape[1]):
            columns.append(mat_a[:, i] * mat_b[:, j])
            left = a if kind_a == "num" else f"{a} = {lev_a[i]}"
            right = b if kind_b == "num" else f"{b} = {lev_b[j]}"
            names_out.append(f"{left} × {right}")
            recipes.append(Recipe("product", parts=(rec_a[i], rec_b[j])))
    label = f"{a} × {b}"
    return Block(np.column_stack(columns), names_out, [label] * len(names_out), spec.key, recipes)


def _interaction_note(
    spec: TermSpec,
    names: list[str],
    main: dict[str, tuple[str, list[Recipe], np.ndarray, list[str]]],
    plan: Plan,
) -> TermNote:
    kinds = [main[n][0] for n in names]
    if kinds == ["num", "num"]:
        how = "the slope of each changes with the other"
    elif "num" in kinds:
        number = names[kinds.index("num")]
        category = names[1 - kinds.index("num")]
        how = f"one slope of `{number}` per level of `{category}`"
    else:
        how = "one effect per combination of levels"
    added = [c for c in spec.columns if c not in plan.asked]
    if added:
        how += "; main effect " + ", ".join(f"`{c}`" for c in added) + " added (R-REG-TERM-01)"
    return TermNote(spec.key, f"`{names[0]}` × `{names[1]}`", "used", how, "R-REG-TERM-07")


def _short(value: float) -> str:
    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else "—"
    magnitude = abs(value)
    if magnitude >= 1e15 or magnitude < 1e-4:
        return f"{value:.2e}"
    decimals = max(0, 2 - math.floor(math.log10(magnitude)))
    return f"{value:,.{decimals}f}".replace("-", "−")


def hierarchy_notes(plan: Plan, present: set[str]) -> list[TermNote]:
    """R-REG-TERM-01: the columns a curve, a square or an interaction brought in."""
    notes: list[TermNote] = []
    seen: set[str] = set()
    for spec in plan.specs:
        if spec.kind not in ("spline", "square", "cube", "interaction"):
            continue
        for column in spec.columns:
            if column in plan.asked or column in seen or column not in present:
                continue
            seen.add(column)
            notes.append(
                TermNote(
                    f"main:{column}",
                    f"`{column}`",
                    "added",
                    f"added: {_what(spec.kind)} needs its main effect",
                    "R-REG-TERM-01",
                )
            )
    return notes


def _what(kind: str) -> str:
    return {"spline": "a curve", "square": "a square", "cube": "a cube"}.get(kind, "an interaction")


# --- Dates as terms (R-REG-TERM-08) -------------------------------------------------------------


def date_values(values: pd.Series, kind: str) -> tuple[np.ndarray, list[str] | None, str]:
    """The trend in decimal years (floats, NaN missing) or month / day codes (−1 missing) with their
    labels, and the reason."""
    dates = pd.to_datetime(values, errors="coerce")
    if kind == "trend":
        length = np.where(dates.dt.is_leap_year.fillna(False).to_numpy(dtype=bool), 366.0, 365.0)
        start = dates.dt.day_of_year.to_numpy(dtype=float, na_value=np.nan) - 1
        hours = dates.dt.hour.to_numpy(dtype=float, na_value=np.nan) / 24
        year = dates.dt.year.to_numpy(dtype=float, na_value=np.nan)
        return year + (start + hours) / length, None, "decimal years (2014.5 = mid-2014)"
    if kind == "month":
        parts = dates.dt.month.to_numpy(dtype=float, na_value=np.nan)
        labels: list[str] = list(MONTHS)
        why = "12 months, each against the most frequent one"
    else:
        parts = dates.dt.day_of_week.to_numpy(dtype=float, na_value=np.nan) + 1
        labels = list(WEEKDAYS)
        why = "7 days, each against the most frequent one"
    codes = np.where(np.isfinite(parts), np.nan_to_num(parts, nan=1) - 1, -1).astype(int)
    return codes, labels, why


# --- Reading the terms back (Terms section, partial plots) --------------------------------------


@dataclass
class TermTest:
    key: str  # the group tested
    question: str
    columns: int  # coefficients tested
    statistic: float | None  # F
    df_num: int
    df_den: int
    p_value: float | None
    status: Literal["ok", "info"]
    sentence: str


@dataclass
class CurveLine:
    label: str
    y: list[float]
    low: list[float]
    high: list[float]


@dataclass
class EffectCurve:
    column: str
    x: list[float]
    lines: list[CurveLine]
    sentence: str
    scale: str  # what y is: the target, or its log / Box-Cox


@dataclass
class SimpleSlope:
    term: str  # the interaction
    column: str  # whose slope
    at: str  # "`waterfront` = 1", "`grade` at its 25th percentile (7)"
    slope: float
    low: float | None
    high: float | None
    p_value: float | None


@dataclass
class LevelEffect:
    column: str
    level: str
    estimate: float
    low: float | None
    high: float | None
    p_value: float | None
    derived: bool  # the level without a column (deviation coding): −Σ of the others


@dataclass
class TermsReading:
    tests: list[TermTest] = field(default_factory=list)
    curves: list[EffectCurve] = field(default_factory=list)
    slopes: list[SimpleSlope] = field(default_factory=list)
    contrasts: list[LevelEffect] = field(default_factory=list)


@dataclass
class Estimates:
    """What reading the terms needs from a fit."""

    beta: np.ndarray
    cov: np.ndarray
    df: int
    least_squares: bool
    scale: str  # "`price`", "log(`price`)"


def _wald(
    beta: np.ndarray, cov: np.ndarray, idx: list[int], df: int
) -> tuple[float | None, float | None]:
    if not idx:
        return None, None
    b = beta[idx]
    v = cov[np.ix_(idx, idx)]
    try:
        f = float(b @ np.linalg.pinv(v) @ b) / len(idx)
    except np.linalg.LinAlgError:
        return None, None
    if not math.isfinite(f):
        return None, None
    return f, float(stats.f.sf(f, len(idx), max(df, 1)))


def _contrast(
    g: np.ndarray, est: Estimates
) -> tuple[float, float | None, float | None, float | None]:
    value = float(g @ est.beta)
    var = float(g @ est.cov @ g)
    if var <= 0 or not math.isfinite(var):
        return value, None, None, None
    se = math.sqrt(var)
    half = float(stats.t.ppf(0.975, max(est.df, 1))) * se
    p = float(2 * stats.t.sf(abs(value / se), max(est.df, 1)))
    return value, value - half, value + half, p


def _p(p: float | None) -> str:
    if p is None:
        return "p = —"
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}"


def read(
    groups: list[str],
    terms: list[str],
    recipes: list[Recipe],
    sources: dict[str, str],
    variables: dict[str, tuple[str, list[str]]],
    values: dict[str, np.ndarray],
    x: np.ndarray,
    plan: Plan,
    est: Estimates,
) -> TermsReading:
    """Joint tests (R-REG-TERM-09), effect curves (R-REG-TERM-10), the slopes of an interaction
    and the levels under deviation coding.

    `variables`: name → (kind, labels) ; `values`: transformed values / codes over the used rows."""
    found = TermsReading()
    index: dict[str, list[int]] = {}
    for j, g in enumerate(groups):
        index.setdefault(g, []).append(j)
    by_name = {name: [j for j, r in enumerate(recipes) if name in uses(r)] for name in variables}
    # does a variable with a curve or several columns matter, is it a curve
    for name, (kind, _) in variables.items():
        own = [j for j in by_name[name] if groups[j] in (name, *_curve_groups(plan, name, sources))]
        nonlinear = [j for j in own if groups[j] != name]
        if kind == "cat" and len(own) > 1:
            f, p = _wald(est.beta, est.cov, own, est.df)
            found.tests.append(
                _test(
                    name,
                    f"Does `{name}` matter?",
                    own,
                    f,
                    p,
                    est.df,
                    f"`{name}` matters",
                    f"`{name}` shows no clear effect",
                )
            )
        if nonlinear:
            f, p = _wald(est.beta, est.cov, own, est.df)
            found.tests.append(
                _test(
                    name,
                    f"Does `{name}` matter?",
                    own,
                    f,
                    p,
                    est.df,
                    f"`{name}` matters",
                    f"`{name}` shows no clear effect",
                )
            )
            f, p = _wald(est.beta, est.cov, nonlinear, est.df)
            found.tests.append(
                _test(
                    groups[nonlinear[0]],
                    f"Is the effect of `{name}` a curve?",
                    nonlinear,
                    f,
                    p,
                    est.df,
                    f"`{name}` is a curve",
                    f"a straight line is enough for `{name}`",
                )
            )
    for spec in plan.specs:
        if spec.kind != "interaction" or spec.key not in index:
            continue
        idx = index[spec.key]
        a, b = (_display(c, variables, sources) for c in spec.columns)
        f, p = _wald(est.beta, est.cov, idx, est.df)
        found.tests.append(
            _test(
                spec.key,
                f"Does the effect of `{a}` depend on `{b}`?",
                idx,
                f,
                p,
                est.df,
                f"the effect of `{a}` depends on `{b}`",
                f"no sign that the effect of `{a}` depends on `{b}`",
            )
        )
        found.slopes.extend(_slopes(spec, a, b, variables, values, recipes, groups, est))
    found.contrasts = _deviation_levels(variables, recipes, est, plan)
    if est.least_squares:
        found.curves = _curves(variables, values, recipes, groups, sources, x, plan, est)
    return found


def _curve_groups(plan: Plan, name: str, sources: dict[str, str]) -> list[str]:
    source = sources.get(name, name)
    return [
        s.key
        for s in plan.specs
        if s.kind in ("spline", "square", "cube") and s.columns[0] in (name, source)
    ]


def _test(
    key: str,
    question: str,
    idx: list[int],
    f: float | None,
    p: float | None,
    df: int,
    yes: str,
    no: str,
) -> TermTest:
    clear = p is not None and p < 0.05
    text = f"{yes[0].upper()}{yes[1:]}" if clear else f"{no[0].upper()}{no[1:]}"
    stat = f"F = {f:.3g}, {_p(p)}" if f is not None else "not testable"
    return TermTest(
        key,
        question,
        len(idx),
        f,
        len(idx),
        max(df, 1),
        p,
        "ok" if clear else "info",
        f"{text} ({stat}).",
    )


def _slopes(
    spec: TermSpec,
    a: str,
    b: str,
    variables: dict[str, tuple[str, list[str]]],
    values: dict[str, np.ndarray],
    recipes: list[Recipe],
    groups: list[str],
    est: Estimates,
) -> list[SimpleSlope]:
    """The slope of the number of an interaction at each level (or quartile) of the other: the
    derivative of the linear predictor, a linear combination of the coefficients."""
    kinds = {a: variables[a][0], b: variables[b][0]}
    numbers = [v for v in (a, b) if kinds[v] == "num"]
    if not numbers:
        return []
    number = numbers[0]
    partner = b if number == a else a
    own = [j for j, r in enumerate(recipes) if r.kind == "num" and r.variables == (number,)]
    if not own:
        return []
    inter = [j for j, g in enumerate(groups) if g == spec.key]
    found: list[SimpleSlope] = []
    for label, value in _partner_points(partner, variables, values):
        g = np.zeros(len(est.beta))
        g[own[0]] = 1.0
        for j in inter:
            part = next(p for p in recipes[j].parts if partner in uses(p))
            g[j] = float(evaluate(part, {partner: np.array([value])}, 1)[0])
        slope, low, high, p = _contrast(g, est)
        found.append(SimpleSlope(spec.key, number, label, slope, low, high, p))
    return found


def _partner_points(
    partner: str, variables: dict[str, tuple[str, list[str]]], values: dict[str, np.ndarray]
) -> list[tuple[str, float]]:
    kind, labels = variables[partner]
    data = values[partner]
    if kind == "cat":
        counts = np.bincount(data[data >= 0], minlength=len(labels))
        order = [int(c) for c in np.argsort(-counts, kind="stable") if counts[c] > 0]
        return [(f"`{partner}` = {labels[c]}", float(c)) for c in order[:PARTNER_CURVES]]
    points = np.quantile(data, [0.25, 0.5, 0.75])
    names = ("25th percentile", "median", "75th percentile")
    return [
        (f"`{partner}` at its {name} ({_short(float(v))})", float(v))
        for name, v in zip(names, points, strict=True)
    ]


def _deviation_levels(
    variables: dict[str, tuple[str, list[str]]],
    recipes: list[Recipe],
    est: Estimates,
    plan: Plan,
) -> list[LevelEffect]:
    """R-REG-TERM-06: under deviation coding, every level against the average of the levels,
    the one without a column by −Σ of the others."""
    if plan.coding != "deviation":
        return []
    found: list[LevelEffect] = []
    for name, (kind, labels) in variables.items():
        if kind != "cat":
            continue
        idx = [j for j, r in enumerate(recipes) if r.kind == "ind" and r.variables == (name,)]
        if not idx:
            continue
        for j in idx:
            g = np.zeros(len(est.beta))
            g[j] = 1.0
            value, low, high, p = _contrast(g, est)
            level = _level_label(recipes[j], labels)
            found.append(LevelEffect(name, level, value, low, high, p, False))
        g = np.zeros(len(est.beta))
        g[idx] = -1.0
        value, low, high, p = _contrast(g, est)
        reference = recipes[idx[0]].reference
        found.append(LevelEffect(name, labels[reference], value, low, high, p, True))
    return found


def _level_label(recipe: Recipe, labels: list[str]) -> str:
    if len(recipe.levels) == 1:
        return labels[recipe.levels[0]]
    return "Others"


def _curves(
    variables: dict[str, tuple[str, list[str]]],
    values: dict[str, np.ndarray],
    recipes: list[Recipe],
    groups: list[str],
    sources: dict[str, str],
    x: np.ndarray,
    plan: Plan,
    est: Estimates,
) -> list[EffectCurve]:
    """R-REG-TERM-10: the fitted target along a number that has a curve, a log or an
    interaction; other columns at their design mean; one line per partner level."""
    found: list[EffectCurve] = []
    means = x.mean(axis=0)
    interactions = [s for s in plan.specs if s.kind == "interaction"]
    for name, (kind, _) in variables.items():
        if kind != "num":
            continue
        curved = bool(_curve_groups(plan, name, sources))
        logged = _is_log(name)
        partners = [
            _display(c, variables, sources)
            for s in interactions
            for c in s.columns
            if name
            in (
                _display(s.columns[0], variables, sources),
                _display(s.columns[1], variables, sources),
            )
        ]
        partners = [p for p in dict.fromkeys(partners) if p != name]
        if not (curved or logged or partners):
            continue
        label = sources.get(name, name) if logged else name
        raw = _raw(values[name], name)
        low, high = np.quantile(raw, CURVE_RANGE)
        if not high > low:
            continue
        grid = np.linspace(low, high, CURVE_POINTS)
        transformed = _transform(grid, name)
        partner = partners[0] if partners else None
        points = _partner_points(partner, variables, values) if partner else [("", math.nan)]
        lines: list[CurveLine] = []
        for level, value in points:
            point_values = {name: transformed}
            if partner is not None:
                point_values[partner] = np.full(CURVE_POINTS, value)
            rows = np.tile(means, (CURVE_POINTS, 1))
            for j, recipe in enumerate(recipes):
                if uses(recipe) and uses(recipe) <= set(point_values):
                    rows[:, j] = evaluate(recipe, point_values, CURVE_POINTS)
            fitted = rows @ est.beta
            se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", rows, est.cov, rows), 0, None))
            half = float(stats.t.ppf(0.975, max(est.df, 1))) * se
            lines.append(
                CurveLine(
                    level or "fitted",
                    [float(v) for v in fitted],
                    [float(v) for v in fitted - half],
                    [float(v) for v in fitted + half],
                )
            )
        first = lines[0]
        lowest, highest = int(np.argmin(first.y)), int(np.argmax(first.y))
        dated = name.endswith(" › Trend")  # a year reads 1997.4, not 1,997

        def at(value: float, dated: bool = dated) -> str:
            return f"{value:.1f}" if dated else _short(value)

        sentence = (
            f"Along `{label}`, the fitted {est.scale} goes from {_short(first.y[0])} to "
            f"{_short(first.y[-1])}"
            + (
                f", highest at {at(float(grid[highest]))}"
                if 0 < highest < CURVE_POINTS - 1
                else f", lowest at {at(float(grid[lowest]))}"
                if 0 < lowest < CURVE_POINTS - 1
                else ""
            )
            + ", other columns at their mean."
        )
        if partner is not None and len(lines) > 1:
            each = "level" if variables[partner][0] == "cat" else "value"
            sentence += f" One line per {each} of `{partner}`."
        found.append(EffectCurve(label, [float(v) for v in grid], lines, sentence, est.scale))
    return found


def _raw(values: np.ndarray, name: str) -> np.ndarray:
    """A log column back in the units of the data."""
    if not _is_log(name):
        return values
    return np.expm1(values) if name.startswith("log(1 +") else np.exp(values)


def _transform(grid: np.ndarray, name: str) -> np.ndarray:
    if not _is_log(name):
        return grid
    return np.log1p(grid) if name.startswith("log(1 +") else np.log(grid)


# --- Partial plots (R-REG-TERM-11) --------------------------------------------------------------


@dataclass
class PartialPlot:
    column: str
    available: bool
    reason: str
    # added-variable plot: residuals of the target and of the column on the other terms
    x_resid: list[float] = field(default_factory=list)
    y_resid: list[float] = field(default_factory=list)
    slope: float | None = None
    # component + residual: the column's estimated contribution plus the residuals
    x: list[float] = field(default_factory=list)
    component_resid: list[float] = field(default_factory=list)
    line_x: list[float] = field(default_factory=list)
    line_y: list[float] = field(default_factory=list)
    rows: list[int] = field(default_factory=list)
    points: int = 0
    of: int = 0
    sentence: str = ""


def partial(
    column: str,
    recipes: list[Recipe],
    sources: dict[str, str],
    variables: dict[str, tuple[str, list[str]]],
    values: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    resid: np.ndarray,
    weights: np.ndarray | None,
    rows: np.ndarray,
    scale: str,
) -> PartialPlot:
    name = _display(column, variables, sources)
    if name not in variables:
        return PartialPlot(column, False, f"`{column}` is not in the model.")
    if variables[name][0] != "num":
        return PartialPlot(column, False, f"`{column}` is a category: see its levels in the table.")
    own = [j for j, r in enumerate(recipes) if uses(r) == {name}]
    linear = [j for j in own if recipes[j].kind == "num"]
    if not linear:
        return PartialPlot(column, False, f"`{column}` has no straight term left in the model.")
    others = [j for j in range(x.shape[1]) if j not in own and name not in uses(recipes[j])]
    root = np.sqrt(weights) if weights is not None else np.ones(len(y))
    z = x[:, others] * root[:, None]

    def residual(v: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(z, v * root, rcond=None)
        return v * root - z @ coef

    ex = residual(x[:, linear[0]])
    ey = residual(y)
    slope = float(ex @ ey / (ex @ ex)) if float(ex @ ex) > 0 else None
    n = len(y)
    positions = np.arange(n)
    if n > PARTIAL_POINTS:
        positions = np.sort(np.random.default_rng(SEED).choice(n, PARTIAL_POINTS, replace=False))
    source = sources.get(name, name) if _is_log(name) else name
    raw = _raw(values[name], name)
    component = x[:, own] @ beta[own]
    component = component - component.mean()
    grid = np.linspace(float(raw.min()), float(raw.max()), CURVE_POINTS)
    grid_values = {name: _transform(grid, name)}
    line = (
        np.column_stack([evaluate(recipes[j], grid_values, CURVE_POINTS) for j in own]) @ beta[own]
    )
    line = line - float((x[:, own] @ beta[own]).mean())
    added = [j for j in own if j != linear[0]]
    shown = _short(slope) if slope is not None else "—"
    if added:  # its curve's columns are taken out with the others: the straight part only
        text = (
            f"Once the other columns and its curve are taken out, the straight part of `{source}` "
            f"has a slope of {shown}. The second plot shows the whole effect, curve included."
        )
    else:
        text = (
            f"Once the other columns are taken out, the slope of `{name}` is {shown} (its "
            "coefficient)."
        )
        if _is_log(name):
            text += f" The second plot shows it in the units of `{source}`."
    if any(name in uses(r) and uses(r) != {name} for r in recipes):
        text += " Its interactions are left out of the component."
    return PartialPlot(
        source,
        True,
        "",
        [float(v) for v in ex[positions]],
        [float(v) for v in ey[positions]],
        slope,
        [float(v) for v in raw[positions]],
        [float(v) for v in (component + resid)[positions]],
        [float(v) for v in grid],
        [float(v) for v in line],
        [int(v) for v in rows[positions]],
        len(positions),
        n,
        text,
    )
