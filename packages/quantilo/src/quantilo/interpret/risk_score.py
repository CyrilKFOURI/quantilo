"""Sparse risk scores with small integer points (121): FasterRisk, re-implemented in numpy.

Method of FasterRisk (Liu, Zhong, Li, Seltzer and Rudin, "FasterRisk: Fast and Accurate
Interpretable Risk Scores", NeurIPS 2022), BSD-3-Clause, Copyright (c) 2022 Jiachang Liu.
Re-implemented here (no code copied) because the package pins scikit-learn 1.2, pandas 1.5 and
numpy < 2; in numpy only, so the browser version (Pyodide) runs it too.

1. Sparse beam search: a logistic regression on true / false conditions with coefficients in
   [−B, B]; at each step every parent of the beam tries the conditions of largest gradient
   (one coordinate each), and the best children are refined by coordinate descent.
2. Diverse sparse solutions: each condition of the best solution is swapped for another one;
   the solutions within 5 % of the best loss are kept (a pool of near-equals to choose from).
3. Star ray search: for multipliers m from 1 to B / max|w|, m·w is rounded sequentially (each
   step rounds the coordinate, up or down, that costs the least), then the intercept and the
   scale are fitted on the integer total: risk = σ(a + b·total), which chooses FasterRisk's
   intercept and multiplier at once.
"""

import itertools
import time
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from scipy.special import expit

from quantilo.interpret import compare
from quantilo.interpret.conditions import Condition, conditions_for, matrix

CONDITIONS = 5  # conditions on the card, by default
MAX_CONDITIONS = 10
BOUND = 5  # points from −5 to 5, by default
BOUNDS = (3, 5, 10)
BEAM = 10
PARENTS = 10
POOL = 20
GAP = 0.05  # diverse solutions within 5 % of the best loss
MULTIPLIERS = 20
RIDGE = 1e-8
OTHERS = 5  # other cards shown
MAX_CLASSES = 8  # R-XAI-17: one card per class, up to 8 classes
ROWS = 10_000  # R-XAI-08: rows learnt and rows judged (desktop)
BROWSER_ROWS = 5_000
SWEEPS = 50
TOLERANCE = 1e-7


class RiskScoreError(ValueError):
    pass


# --- The continuous problem ----------------------------------------------------------------------


def _softplus(margin: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, margin)


@dataclass
class _Problem:
    x: np.ndarray  # rows × conditions, 0 / 1
    y: np.ndarray  # 0 / 1
    bound: float
    on: list[np.ndarray] = field(default_factory=list)  # rows where each condition holds
    # R-XAI-16: each coefficient's own interval (a monotone column forbids one sign)
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.on = [np.flatnonzero(self.x[:, j]) for j in range(self.x.shape[1])]
        width = self.x.shape[1]
        if self.lo is None:
            self.lo = np.full(width, -self.bound)
        if self.hi is None:
            self.hi = np.full(width, self.bound)

    @property
    def n(self) -> int:
        return len(self.y)

    def loss(self, margin: np.ndarray, w: np.ndarray) -> float:
        return float(np.mean(_softplus(margin) - self.y * margin) + RIDGE * float(w @ w))


@dataclass
class _Solution:
    support: tuple[int, ...]
    w: np.ndarray  # one coefficient per condition (zeros off the support)
    w0: float
    margin: np.ndarray
    loss: float


def _step_intercept(problem: _Problem, s: _Solution) -> float:
    p = expit(s.margin)
    g = float(np.mean(p - problem.y))
    h = float(np.mean(p * (1 - p))) + 1e-12
    delta = -g / h
    base = float(np.mean(_softplus(s.margin) - problem.y * s.margin))
    for _ in range(20):
        moved = s.margin + delta
        if float(np.mean(_softplus(moved) - problem.y * moved)) <= base + 1e-15:
            s.margin, s.w0 = moved, s.w0 + delta
            return abs(delta)
        delta /= 2
    return 0.0


def _step(problem: _Problem, s: _Solution, j: int) -> float:
    """One Newton step on coefficient j inside [−B, B], halved until the loss goes down. Only
    the rows where condition j holds move: the step costs their count, not every row."""
    rows = problem.on[j]
    if not len(rows):
        return 0.0
    n = problem.n
    m, y = s.margin[rows], problem.y[rows]
    p = expit(m)
    wj = float(s.w[j])
    g = float((p - y).sum()) / n + 2 * RIDGE * wj
    h = float((p * (1 - p)).sum()) / n + 2 * RIDGE + 1e-12
    assert problem.lo is not None and problem.hi is not None
    target = float(np.clip(wj - g / h, problem.lo[j], problem.hi[j]))
    delta = target - wj
    before = float((_softplus(m) - y * m).sum()) / n + RIDGE * wj * wj
    for _ in range(20):
        if abs(delta) < 1e-12:
            return 0.0
        moved = m + delta
        after = float((_softplus(moved) - y * moved).sum()) / n + RIDGE * (wj + delta) ** 2
        if after <= before + 1e-15:
            s.margin[rows] = moved
            s.w[j] = wj + delta
            return abs(delta)
        delta /= 2
    return 0.0


def _refine(problem: _Problem, s: _Solution) -> _Solution:
    """Coordinate descent on the intercept and the support until the loss stops moving."""
    last = s.loss
    for _ in range(SWEEPS):
        _step_intercept(problem, s)
        for j in s.support:
            _step(problem, s, j)
        s.loss = problem.loss(s.margin, s.w)
        if last - s.loss < TOLERANCE:
            break
        last = s.loss
    return s


def _start(problem: _Problem) -> _Solution:
    rate = float(np.clip(problem.y.mean(), 1e-6, 1 - 1e-6))
    w0 = float(np.log(rate / (1 - rate)))
    margin = np.full(problem.n, w0)
    p = problem.x.shape[1]
    return _Solution((), np.zeros(p), w0, margin, problem.loss(margin, np.zeros(p)))


def _copy(s: _Solution) -> _Solution:
    return _Solution(s.support, s.w.copy(), s.w0, s.margin.copy(), s.loss)


def _gradient(problem: _Problem, s: _Solution) -> np.ndarray:
    return problem.x.T @ (expit(s.margin) - problem.y) / problem.n


def _with(problem: _Problem, parent: _Solution, j: int, refine: bool) -> _Solution:
    child = _copy(parent)
    child.support = tuple(sorted((*parent.support, j)))
    for _ in range(3):  # FasterRisk's screening: the new coordinate alone first
        _step(problem, child, j)
    child.loss = problem.loss(child.margin, child.w)
    return _refine(problem, child) if refine else child


def _candidates(problem: _Problem, s: _Solution, exclude: set[int], count: int) -> list[int]:
    raw = _gradient(problem, s)
    grad = np.abs(raw)
    assert problem.lo is not None and problem.hi is not None
    # R-XAI-16: a coefficient held at or above 0 cannot start downwards, nor one at or below 0 up
    grad[(problem.lo >= 0) & (raw > 0)] = -1.0
    grad[(problem.hi <= 0) & (raw < 0)] = -1.0
    grad[list(exclude | set(s.support))] = -1.0
    order = np.argsort(-grad, kind="stable")[:count]
    return [int(j) for j in order if grad[j] > 1e-12]


def beam_search(problem: _Problem, k: int, beam: int = BEAM, parents: int = PARENTS):
    """The beam of solutions with k conditions at most, best first."""
    frontier = [_refine(problem, _start(problem))]
    for _ in range(k):
        screened: dict[tuple[int, ...], _Solution] = {}
        for parent in frontier:
            for j in _candidates(problem, parent, set(), parents):
                key = tuple(sorted((*parent.support, j)))
                if key in screened:
                    continue
                screened[key] = _with(problem, parent, j, refine=False)
        if not screened:
            break
        best = sorted(screened.values(), key=lambda s: s.loss)[:beam]
        frontier = sorted((_refine(problem, s) for s in best), key=lambda s: s.loss)
    return frontier


def _without(problem: _Problem, s: _Solution, j: int) -> _Solution:
    reduced = _copy(s)
    reduced.margin[problem.on[j]] -= reduced.w[j]
    reduced.w[j] = 0.0
    reduced.support = tuple(i for i in s.support if i != j)
    reduced.loss = problem.loss(reduced.margin, reduced.w)
    return _refine(problem, reduced)


def diverse(
    problem: _Problem,
    best: _Solution,
    parents: int = PARENTS,
    gap: float = GAP,
    pool: int = POOL,
) -> list[_Solution]:
    """Solutions that swap one condition of the best one and lose at most `gap` of its loss."""
    found = {best.support: best}
    for j in best.support:
        reduced = _without(problem, best, j)
        for c in _candidates(problem, reduced, {j}, parents):
            key = tuple(sorted((*reduced.support, c)))
            if key in found:
                continue
            child = _with(problem, reduced, c, refine=True)
            if child.loss <= best.loss * (1 + gap):
                found[key] = child
    return sorted(found.values(), key=lambda s: s.loss)[:pool]


# --- Integer points ------------------------------------------------------------------------------


@dataclass
class IntegerCard:
    support: list[int]  # condition indices, in the order shown
    points: list[int]
    intercept: float  # risk = σ(intercept + scale · total)
    scale: float
    loss: float  # mean logistic loss on the learnt rows


def calibrate(total: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """σ(a + b·total) fitted by Newton's method (two parameters), b kept ≥ 0."""
    rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    a, b = float(np.log(rate / (1 - rate))), 0.0
    design = np.column_stack([np.ones(len(total)), total.astype(float)])

    def loss(a: float, b: float) -> float:
        margin = a + b * total
        return float(np.mean(_softplus(margin) - y * margin))

    current = loss(a, b)
    for _ in range(50):
        p = expit(a + b * total)
        g = design.T @ (p - y) / len(y)
        h = (design * (p * (1 - p))[:, None]).T @ design / len(y) + np.eye(2) * 1e-9
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while t > 1e-6:
            na, nb = a - t * step[0], max(b - t * step[1], 0.0)
            found = loss(na, nb)
            if found <= current + 1e-15:
                break
            t /= 2
        else:
            break
        moved = abs(na - a) + abs(nb - b)
        a, b, current = na, nb, found
        if moved < 1e-10:
            break
    return a, b, current


def _sequential_round(
    xs: np.ndarray,
    y: np.ndarray,
    gamma: np.ndarray,
    intercept: float,
    m: float,
    bound: int,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
) -> np.ndarray:
    """Round m·w one coordinate at a time: each step takes the coordinate and the direction
    (down or up) whose rounding costs the least loss, the others still continuous."""
    gamma = gamma.copy()
    base = intercept + xs @ gamma
    left = [j for j in range(len(gamma)) if not float(gamma[j]).is_integer()]
    on = [np.flatnonzero(xs[:, j]) for j in range(xs.shape[1])]
    while left:
        best: tuple[float, int, float] | None = None
        for j in left:
            rows = on[j]
            now = base[rows] / m
            kept = float((_softplus(now) - y[rows] * now).sum())
            for target in {float(np.floor(gamma[j])), float(np.ceil(gamma[j]))}:
                low = -bound if lo is None else max(-bound, float(lo[j]))
                high = bound if hi is None else min(bound, float(hi[j]))
                target = float(np.clip(target, low, high))
                moved = (base[rows] + (target - gamma[j]) * 1.0) / m
                cost = float((_softplus(moved) - y[rows] * moved).sum()) - kept
                if best is None or cost < best[0]:
                    best = (cost, j, target)
        assert best is not None
        _, j, target = best
        base[on[j]] += target - gamma[j]
        gamma[j] = target
        left.remove(j)
    return np.round(gamma).astype(int)


def star_ray(
    problem: _Problem, s: _Solution, bound: int, multipliers: int = MULTIPLIERS
) -> IntegerCard | None:
    support = list(s.support)
    if not support:
        return None
    w = s.w[support]
    peak = float(np.abs(w).max())
    if peak <= 0:
        return None
    xs = problem.x[:, support]
    best: IntegerCard | None = None
    top = max(bound / peak, 1.0)
    for m in np.linspace(1.0, top, multipliers) if top > 1 else [1.0]:
        assert problem.lo is not None and problem.hi is not None
        points = _sequential_round(
            xs, problem.y, m * w, m * s.w0, float(m), bound,
            problem.lo[support], problem.hi[support],
        )  # fmt: skip
        kept = [i for i, v in enumerate(points) if v != 0]
        if not kept:
            continue
        total = xs[:, kept] @ points[kept]
        a, b, loss = calibrate(total, problem.y)
        if best is None or loss < best.loss:
            best = IntegerCard(
                [support[i] for i in kept], [int(points[i]) for i in kept], a, b, loss
            )
    return best


def fit(
    x: np.ndarray,
    y: np.ndarray,
    k: int = CONDITIONS,
    bound: int = BOUND,
    beam: int = BEAM,
    parents: int = PARENTS,
    pool: int = POOL,
    gap: float = GAP,
    multipliers: int = MULTIPLIERS,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
) -> list[IntegerCard]:
    """FasterRisk on a 0 / 1 matrix: the integer cards found, best (lowest loss) first, one per
    set of conditions. `lo` / `hi`: each condition's interval of points (R-XAI-16)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[1] == 0:
        raise RiskScoreError("No condition to build a scorecard on.")
    if y.min() == y.max():
        raise RiskScoreError("A scorecard needs both classes among the rows it learns from.")
    problem = _Problem(
        x,
        y,
        float(bound),
        lo=None if lo is None else np.asarray(lo, dtype=float),
        hi=None if hi is None else np.asarray(hi, dtype=float),
    )
    frontier = beam_search(problem, k, beam, parents)
    solutions = {s.support: s for s in frontier if s.loss <= frontier[0].loss * (1 + gap)}
    for s in diverse(problem, frontier[0], parents, gap, pool):
        solutions.setdefault(s.support, s)
    ranked = sorted(solutions.values(), key=lambda s: s.loss)[:pool]
    cards: dict[tuple[int, ...], IntegerCard] = {}
    for s in ranked:
        card = star_ray(problem, s, bound, multipliers)
        if card is None:
            continue
        key = tuple(sorted(card.support))
        if key not in cards or card.loss < cards[key].loss:
            cards[key] = card
    return sorted(cards.values(), key=lambda c: c.loss)


# --- The card, as read -----------------------------------------------------------------------


@dataclass
class CardLine:
    condition: str
    column: str
    points: int
    share: float  # share of the learnt rows where it holds
    # 121 bis: the condition itself, to apply the card to other rows (Predict on new data)
    kind: str | None = None
    value: str | None = None
    number: float | None = None

    def as_condition(self) -> Condition:
        if self.kind is None:
            raise RiskScoreError("This card was made before 121 bis: open the Scorecard again.")
        return Condition(self.column, self.kind, self.value, self.number)  # type: ignore[arg-type]


@dataclass
class RiskStep:
    total: int
    risk: float  # the card's probability of the positive class
    rows: int  # checked rows with this total
    observed: float | None  # share of positives among them


@dataclass
class Scorecard:
    lines: list[CardLine]
    steps: list[RiskStep]
    intercept: float  # risk = σ(intercept + scale · total)
    scale: float
    auc: float  # on the checked rows
    learnt_auc: float
    ece: float
    brier: float
    log_loss: float
    columns: int  # distinct columns used
    sentence: str


@dataclass
class RiskScores:
    positive: str
    conditions: int  # at most, as asked
    bound: int
    best: Scorecard
    others: list[Scorecard]  # other cards almost as good (different conditions)
    gap: compare.Gap
    learnt: int
    checked: int
    candidates: int  # conditions tried (R-XAI-06)
    source: str  # which rows (R-XAI-08)
    notes: list[str]
    seconds: float
    monotone: bool = False  # R-XAI-16: points kept monotone in each number
    shape: list[str] = field(default_factory=list)  # R-XAI-16: numbers whose points go up and down


def _readable(conditions: list[Condition], card: IntegerCard) -> tuple[list[Condition], list[int]]:
    """Thresholds read so that their points add risk: "x ≤ t: −3" becomes "x > t: +3" (the
    total moves by a constant, the risk of each row does not)."""
    shown, points = [], []
    for j, value in zip(card.support, card.points, strict=True):
        condition = conditions[j]
        if value < 0 and condition.kind in ("below", "above"):
            condition, value = condition.flipped(), -value
        shown.append(condition)
        points.append(int(value))
    order = sorted(range(len(points)), key=lambda i: (-abs(points[i]), shown[i].text))
    return [shown[i] for i in order], [points[i] for i in order]


def _totals(points: list[int]) -> list[int]:
    reachable = {0}
    for value in points:
        reachable |= {t + value for t in reachable}
    return sorted(reachable)


def card(
    conditions: list[Condition],
    found: IntegerCard,
    learn: pd.DataFrame,
    y_learn: np.ndarray,
    check: pd.DataFrame,
    y_check: np.ndarray,
) -> tuple[Scorecard, np.ndarray, np.ndarray]:
    """The card as read, and its total and risk on the checked rows."""
    shown, points = _readable(conditions, found)
    learnt = matrix(learn, shown)
    checked = matrix(check, shown)
    total_learn = learnt @ np.array(points, dtype=float)
    a, b, _ = calibrate(total_learn, y_learn)
    total = checked @ np.array(points, dtype=float)
    risk = expit(a + b * total)
    steps = []
    for value in _totals(points):
        inside = total == value
        rows = int(inside.sum())
        steps.append(
            RiskStep(
                value,
                float(expit(a + b * value)),
                rows,
                float(y_check[inside].mean()) if rows else None,
            )
        )
    lines = [
        CardLine(
            c.text,
            c.column,
            p,
            float(learnt[:, i].mean()) if len(learnt) else 0.0,
            c.kind,
            c.value,
            c.number,
        )
        for i, (c, p) in enumerate(zip(shown, points, strict=True))
    ]
    columns = len({c.column for c in shown})
    score = compare.auc(total, y_check)
    sentence = (
        f"{len(lines)} condition{'s' if len(lines) != 1 else ''} on {columns} "
        f"column{'s' if columns != 1 else ''}, points from {min(points):+d} to {max(points):+d}: "
        f"ROC AUC {score:.3f} on {len(y_check):,} rows it did not learn from."
    )
    read = Scorecard(
        lines,
        steps,
        float(a),
        float(b),
        score,
        compare.auc(total_learn, y_learn),
        compare.ece(risk, y_check),
        compare.brier(risk, y_check),
        compare.log_loss(risk, y_check),
        columns,
        sentence,
    )
    return read, total, risk


def risk_scores(
    learn: pd.DataFrame,
    y_learn: np.ndarray,
    check: pd.DataFrame,
    y_check: np.ndarray,
    model_probability: np.ndarray,
    positive: str,
    model_label: str,
    source: str,
    k: int = CONDITIONS,
    bound: int = BOUND,
    skip: set[str] | None = None,
    types: dict[str, str] | None = None,
    monotone: bool = False,
) -> RiskScores:
    """The best card and the others almost as good, compared with the run's model on the
    checked rows (`model_probability`: its probability of `positive` there). `monotone`: the
    points of each number keep one direction (R-XAI-16)."""
    started = time.perf_counter()
    k = int(min(max(k, 1), MAX_CONDITIONS))
    bound = int(bound) if int(bound) in BOUNDS else BOUND
    y_learn = np.asarray(y_learn, dtype=float)
    y_check = np.asarray(y_check, dtype=float)
    candidates = conditions_for(learn, skip, types)
    if not candidates:
        raise RiskScoreError(
            "No column gives a condition a card can use (R-XAI-06: numbers, categories and "
            "flags are used; dates, free text and identifiers are left out)."
        )
    lo, hi = monotone_bounds(candidates, learn, y_learn, bound) if monotone else (None, None)
    found = fit(matrix(learn, candidates), y_learn, k, bound, lo=lo, hi=hi)
    if not found:
        raise RiskScoreError("No condition separates the classes: no scorecard to show.")
    cards = [card(candidates, f, learn, y_learn, check, y_check) for f in found]
    best, total, risk = cards[0]
    seen = {tuple(sorted(line.condition for line in best.lines))}
    others = []
    for other, _, _ in cards[1:]:
        key = tuple(sorted(line.condition for line in other.lines))
        if key in seen:
            continue
        seen.add(key)
        others.append(other)
        if len(others) >= OTHERS:
            break
    what = f"A {len(best.lines)}-condition scorecard"
    comparison = compare.gap(
        np.asarray(model_probability, dtype=float),
        total,
        y_check,
        what,
        model_label,
        np.asarray(model_probability, dtype=float),
        risk,
    )
    notes = [
        f"R-XAI-06: {len(candidates)} conditions tried, from the deciles of the numbers (rounded "
        "to 2 significant digits) and the frequent categories; dates, free text and identifiers "
        "left out.",
        f"Rows {source}.",
    ]
    if monotone:
        notes.append(
            "R-XAI-16: monotone points, the risk moves one way along each number (the way it "
            "goes with the target on the learnt rows)."
        )
    return RiskScores(
        positive,
        k,
        bound,
        best,
        others,
        comparison,
        len(y_learn),
        len(y_check),
        len(candidates),
        source,
        notes,
        round(time.perf_counter() - started, 2),
        monotone,
        shape(best.lines),
    )


# --- 121 bis: monotone points (R-XAI-16) -------------------------------------------------------


def directions(conditions: list[Condition], learn: pd.DataFrame, y: np.ndarray) -> dict[str, int]:
    """Each number's direction: +1 when the target rises with it on the learnt rows (Spearman),
    -1 when it falls, 0 when flat or unknown."""
    found: dict[str, int] = {}
    for column in {c.column for c in conditions if c.kind in ("below", "above")}:
        values = np.asarray(pd.to_numeric(cast(pd.Series, learn[column]), errors="coerce"), float)
        known = ~np.isnan(values)
        if known.sum() < 3 or np.ptp(values[known]) == 0:
            found[column] = 0
            continue
        ranks = pd.Series(values[known]).rank().to_numpy()
        target = np.asarray(y, dtype=float)[known]
        r = np.corrcoef(ranks, target)[0, 1] if np.ptp(target) else 0.0
        found[column] = 0 if not np.isfinite(r) or r == 0 else (1 if r > 0 else -1)
    return found


def monotone_bounds(
    conditions: list[Condition], learn: pd.DataFrame, y: np.ndarray, bound: int
) -> tuple[np.ndarray, np.ndarray]:
    """R-XAI-16: the points of a number's conditions keep one sign, so the risk moves one way
    along it: when the target rises with x, "x ≤ t" never adds points and "x > t" never takes
    any away (and the reverse). Categories, flags and "is empty" stay free."""
    way = directions(conditions, learn, y)
    lo = np.full(len(conditions), -float(bound))
    hi = np.full(len(conditions), float(bound))
    for j, c in enumerate(conditions):
        d = way.get(c.column, 0) if c.kind in ("below", "above") else 0
        if d == 0:
            continue
        adds = (c.kind == "above") == (d > 0)  # the condition holds where the risk is higher
        if adds:
            lo[j] = 0.0
        else:
            hi[j] = 0.0
    return lo, hi


def shape(lines: list[CardLine]) -> list[str]:
    """R-XAI-16: the numbers whose points go up then down along the values (a hump or a dip):
    said, so that the user can ask for monotone points when such a shape makes no sense."""
    said = []
    for column in dict.fromkeys(line.column for line in lines):
        cuts = [
            line
            for line in lines
            if line.column == column and line.kind in ("below", "above") and line.number is not None
        ]
        if len(cuts) < 2:
            continue
        edges = sorted({float(c.number) for c in cuts if c.number is not None})
        probes = [edges[0] - 1.0, *[(a + b) / 2 for a, b in itertools.pairwise(edges)]]
        probes.append(edges[-1] + 1.0)
        totals = []
        for x in probes:
            total = 0
            for c in cuts:
                assert c.number is not None
                holds = x <= c.number if c.kind == "below" else x > c.number
                total += c.points if holds else 0
            totals.append(total)
        steps = np.sign(np.diff(totals))
        if (steps > 0).any() and (steps < 0).any():
            shown = ", ".join(f"{p:+d}" for p in totals)
            said.append(
                f"R-XAI-16 · {column}: the points go up and down along its values ({shown} from "
                "the lowest values to the highest). Ask for monotone points if the risk should "
                "move one way."
            )
    return said


# --- 121 bis: one card per class (R-XAI-17) ----------------------------------------------------


@dataclass
class ClassCard:
    label: str
    card: Scorecard
    auc: float  # one class against the others, on the checked rows
    model_auc: float


@dataclass
class ClassCards:
    classes: list[ClassCard]
    accuracy: float  # the class whose card gives the highest risk, on the checked rows
    model_accuracy: float
    conditions: int
    bound: int
    learnt: int
    checked: int
    source: str
    sentence: str
    notes: list[str]
    seconds: float
    monotone: bool = False


def class_cards(
    learn: pd.DataFrame,
    y_learn: np.ndarray,
    check: pd.DataFrame,
    y_check: np.ndarray,
    model_probabilities: np.ndarray,
    classes: list[str],
    model_label: str,
    source: str,
    k: int = CONDITIONS,
    bound: int = BOUND,
    skip: set[str] | None = None,
    types: dict[str, str] | None = None,
    monotone: bool = False,
) -> ClassCards:
    """R-XAI-17: three classes or more, one card per class (that class against the others); a
    row goes to the class whose card gives the highest risk. Compared with the run's model by
    each class's ROC AUC and by the accuracy on the checked rows."""
    started = time.perf_counter()
    if len(classes) > MAX_CLASSES:
        raise RiskScoreError(
            f"R-XAI-17: one card per class is read up to {MAX_CLASSES} classes; this run has "
            f"{len(classes)}. Group the classes first (Settings › Classes)."
        )
    k = int(min(max(k, 1), MAX_CONDITIONS))
    bound = int(bound) if int(bound) in BOUNDS else BOUND
    labels_learn = np.asarray(y_learn).astype(str)
    labels_check = np.asarray(y_check).astype(str)
    candidates = conditions_for(learn, skip, types)
    if not candidates:
        raise RiskScoreError("No column gives a condition a card can use (R-XAI-06).")
    x = matrix(learn, candidates)
    found: list[ClassCard] = []
    risks = np.full((len(labels_check), len(classes)), -np.inf)
    for index, label in enumerate(classes):
        y01 = (labels_learn == label).astype(float)
        if y01.min() == y01.max():
            continue
        lo, hi = monotone_bounds(candidates, learn, y01, bound) if monotone else (None, None)
        cards = fit(x, y01, k, bound, lo=lo, hi=hi)
        if not cards:
            continue
        truth = (labels_check == label).astype(float)
        read, total, risk = card(candidates, cards[0], learn, y01, check, truth)
        risks[:, index] = risk
        found.append(
            ClassCard(
                label,
                read,
                compare.auc(total, truth),
                compare.auc(model_probabilities[:, index], truth),
            )
        )
    if not found:
        raise RiskScoreError("No class could be told apart from the others: no card to show.")
    guessed = np.asarray(classes)[np.argmax(risks, axis=1)]
    accuracy = float(np.mean(guessed == labels_check))
    model_accuracy = float(
        np.mean(np.asarray(classes)[np.argmax(model_probabilities, axis=1)] == labels_check)
    )
    sentence = (
        f"{len(found)} cards of {k} conditions at most, one per class: a row goes to the class "
        f"whose card gives the highest risk. Accuracy {accuracy:.0%} against {model_accuracy:.0%} "
        f"for the {model_label}, on {len(labels_check):,} rows neither learnt from (R-XAI-17)."
    )
    notes = [
        f"R-XAI-06: {len(candidates)} conditions tried for each class.",
        f"Rows {source}.",
    ]
    return ClassCards(
        found,
        accuracy,
        model_accuracy,
        k,
        bound,
        len(labels_learn),
        len(labels_check),
        source,
        sentence,
        notes,
        round(time.perf_counter() - started, 2),
        monotone,
    )


# --- 121 bis: the card on other rows (R-XAI-18) ------------------------------------------------


@dataclass
class CardRow:
    """One row read by the card: the conditions that hold, their points, the total and risk."""

    row: int
    lines: list[CardLine]  # the card's lines that hold on this row
    total: int
    risk: float
    positive: str
    sentence: str


def apply_lines(lines: list[CardLine], frame: pd.DataFrame) -> np.ndarray:
    """The card's total for each row of `frame`, from the lines alone (no learnt rows needed)."""
    shown = [line.as_condition() for line in lines]
    return matrix(frame, shown) @ np.array([line.points for line in lines], dtype=float)


def card_row(found: RiskScores, frame: pd.DataFrame, row: int) -> CardRow:
    """R-XAI-18: the best card applied to one row of new data, said in a sentence."""
    lines = found.best.lines
    held = matrix(frame, [line.as_condition() for line in lines])[0].astype(bool)
    total = int(sum(line.points for line, on in zip(lines, held, strict=True) if on))
    risk = float(expit(found.best.intercept + found.best.scale * total))
    kept = [line for line, on in zip(lines, held, strict=True) if on]
    parts = ", ".join(f"{line.condition} {line.points:+d}" for line in kept) or "no condition holds"
    sentence = (
        f"The scorecard reads {parts}: total {total:+d}, a risk of {risk:.0%} for "
        f"{found.positive} (R-XAI-18). It was learnt on the run's rows: check that this row is "
        "like them."
    )
    return CardRow(row, kept, total, risk, found.positive, sentence)


def _condition(candidates: list[Condition], line: CardLine) -> Condition:
    for c in candidates:
        if c.text == line.condition:
            return c
        if c.kind in ("below", "above") and c.flipped().text == line.condition:
            return c.flipped()
    raise RiskScoreError(f"Unknown condition {line.condition}")  # pragma: no cover


def apply(lines: list[CardLine], frame: pd.DataFrame, candidates: list[Condition]) -> np.ndarray:
    """The card's total for each row of `frame`."""
    shown = [_condition(candidates, line) for line in lines]
    return matrix(frame, shown) @ np.array([line.points for line in lines], dtype=float)


__all__ = [
    "CardLine",
    "CardRow",
    "ClassCard",
    "ClassCards",
    "IntegerCard",
    "RiskScoreError",
    "RiskScores",
    "RiskStep",
    "Scorecard",
    "apply",
    "apply_lines",
    "calibrate",
    "card_row",
    "class_cards",
    "fit",
    "monotone_bounds",
    "risk_scores",
    "shape",
]
