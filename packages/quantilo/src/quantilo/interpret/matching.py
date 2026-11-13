"""Matched groups (123): FLAME, almost-exact matching (Wang, Morucci, Awan, Liu, Roy, Rudin &
Volfovsky, JMLR 2021), re-implemented in DuckDB SQL.

On observational data with a binary treatment, units are matched **exactly** on their
covariates (each one discrete, or cut into bins: R-MAT-02): a GROUP BY on the covariates gives
cells, and a cell holding both treated and control units is a matched group. The units left
unmatched are matched again on fewer covariates: at each level the covariate least useful to
predict the outcome is dropped (R-MAT-04), measured on held-out rows that are never matched
(R-MAT-03), so that the groups stay as fine as possible on what matters. Each group is
readable ("device = mobile · 25 < age ≤ 40"), has its own effect (CATE, R-MAT-06) and its rows.

No scikit-learn here: matching and prediction errors are GROUP BY queries, the rest numpy.
"""

import itertools
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal, cast

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

TREATED, OUTCOME, ROW = "__treated", "__outcome", "__row"
BINS = 4  # R-MAT-02: quartiles of a number with many values
KEEP_VALUES = 8  # a number with at most this many values keeps them as they are
TOP_CATEGORIES = 10
EMPTY, OTHER = "(empty)", "Other"
HOLDOUT = 0.2  # R-MAT-03
TRADEOFF = 0.1  # R-MAT-04: C, the weight of the balancing factor against the prediction error
STOP_PE = 0.05  # R-MAT-05: stop when the prediction error rises by more than 5 %
MIN_ARM = 25  # R-MAT-01: units per arm
MAX_COVARIATES = 20
MAX_UNITS = 50_000
GROUPS_SHOWN = 50
ROWS_SHOWN = 20
BALANCED = 0.1  # R-MAT-07: standardized difference under which a covariate is balanced
SEED = 123
MAX_ARMS = 10  # 123 bis (R-MAT-10): one pair of values at a time, out of at most 10
DAME_SETS = 200  # R-MAT-09: candidate sets tried at each DAME step, at most
DAME_LEVELS = 60

OBSERVATIONAL = (
    "Observational: matching compares like with like on the columns given, it does not remove "
    "a hidden factor that drove both the decision and the outcome. Use columns measured before "
    "the decision, never ones it changed."
)
METHOD = (
    "FLAME (almost-exact matching, Wang, Rudin et al. 2021): units are grouped exactly on their "
    "covariates (numbers cut into bins); the unmatched ones are matched again after dropping the "
    "covariate that least helps to predict the outcome, measured on {holdout} held-out rows "
    "never matched. Intervals: normal approximation, given the groups."
)
DAME_METHOD = (
    "DAME (dynamic almost-exact matching, Liu, Rudin et al. 2019): units are grouped exactly on "
    "their covariates (numbers cut into bins); each next level matches on the set of covariates, "
    "among those whose larger sets are done, that best predicts the outcome on {holdout} held-out "
    "rows never matched, so a covariate can come back (R-MAT-09). Intervals: normal "
    "approximation, given the groups."
)


# --- What the app asks and what it gets ---------------------------------------------------------


class MatchingRequest(BaseModel):
    treatment: str = Field(description="The column with two values: treated or not.")
    outcome: str = Field(description="The outcome measured after it (a number or yes/no).")
    treated: str | None = Field(
        default=None,
        description="The treated value; the one that is not the control when unset.",
    )
    covariates: list[str] | None = Field(
        default=None,
        max_length=50,
        description="Columns describing the units before the treatment; every usable one if unset.",
    )
    bins: int = Field(default=BINS, ge=2, le=10, description="Bins for a number with many values.")
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    # 123 bis: the pair compared among more values (R-MAT-10); the order of drops (R-MAT-09)
    control: str | None = Field(
        default=None, description="The control value; guessed when unset (R-MAT-10)."
    )
    method: Literal["flame", "dame"] = Field(
        default="flame", description="FLAME drops one covariate for good; DAME searches subsets."
    )


@dataclass
class MatchingOffer:
    """Whether matched groups can be computed (R-MAT-01), and why."""

    available: bool
    reason: str


@dataclass
class MatchingCovariate:
    column: str
    used: bool
    how: str  # how it was made discrete, or why it was left out
    levels: list[str]  # each level as a condition ("25 < age ≤ 40")
    dropped_at: int | None  # the first level that no longer matches on it


@dataclass
class MatchingLevel:
    level: int  # 1: every covariate
    covariates: list[str]  # matched exactly on these
    dropped: str | None  # the covariate dropped to reach this level
    prediction_error: float  # share of the outcome's variation left unexplained (held out)
    balancing: float  # matched share of the unmatched controls + of the unmatched treated
    groups: int
    treated: int  # treated units matched at this level
    control: int


@dataclass
class MatchedGroup:
    id: int
    level: int
    conditions: list[str]
    treated: int
    control: int
    treated_mean: float
    control_mean: float
    effect: float  # treated mean − control mean (CATE)
    low: float | None
    high: float | None
    rows: list[int]  # row numbers in the file, the first ones


@dataclass
class MatchingEffect:
    key: Literal["naive", "ate", "att"]
    label: str
    value: float | None
    low: float | None
    high: float | None
    units: int
    sentence: str


@dataclass
class BalanceRow:
    column: str
    before: float | None  # standardized difference, treated vs control
    after: float | None  # on the matched units, weighted by group
    balanced: bool
    dropped_at: int | None


@dataclass
class MatchedGroups:
    treatment: str
    treated: str
    control: str
    outcome: str
    kind: Literal["rate", "mean"]
    units: int  # units matched or not (the held-out ones apart)
    held_out: int
    sampled: bool
    treated_units: int
    control_units: int
    matched_treated: int
    matched_control: int
    effects: list[MatchingEffect]  # naive, average (ATE), on the treated (ATT)
    group_count: int
    groups: list[MatchedGroup]  # the largest ones
    levels: list[MatchingLevel]
    covariates: list[MatchingCovariate]
    balance: list[BalanceRow]
    stopped: str
    sentence: str
    warnings: list[str]
    method: str


@dataclass
class MatchingUnavailable:
    reason: str


# --- 123 bis: DAME, the dynamic order of drops (R-MAT-09) -------------------------------------


def _dame(
    engine: "_Flame",
    used: list[str],
    infos: list[MatchingCovariate],
    levels: list[MatchingLevel],
    full: float,
) -> str:
    """DAME (Liu, Dieng, Roy, Rudin, Volfovsky, 2019): instead of dropping covariates one after the
    other for good, each step picks, among the sets whose every superset one covariate larger was
    already matched on (the downward closure), the one of best MQ = 0.1 x BF - PE; it may come back
    to a covariate FLAME has dropped. Same stops as FLAME (R-MAT-05)."""
    everything = frozenset(range(len(used)))
    processed: set[frozenset[int]] = {everything}

    def ready(candidate: frozenset[int]) -> bool:
        return all(candidate | {k} in processed for k in everything - candidate)

    while len(levels) < DAME_LEVELS:
        left_t, left_c = engine.unmatched()
        if not left_t or not left_c:
            return (
                "Every treated unit is matched."
                if not left_t
                else (f"Every control unit is matched; {left_t:,} treated units are left.")
            )
        pool = sorted(
            {
                s - {j}
                for s in processed
                for j in s
                if len(s) > 1 and s - {j} not in processed and ready(s - {j})
            },
            key=lambda c: (-len(c), sorted(c)),
        )[:DAME_SETS]
        if not pool:
            return (
                f"No set of covariates is left to match on: {left_t:,} treated units are left "
                "unmatched."
            )
        tried = []
        for candidate in pool:
            columns = sorted(candidate)
            error = engine.prediction_error(columns)
            tried.append((TRADEOFF * engine.balancing(columns) - error, -error, columns, error))
        _, _, best, error = max(tried, key=lambda item: (item[0], item[1], [-c for c in item[2]]))
        if error > full * (1 + STOP_PE) + 1e-12:
            missing = ", ".join(used[c] for c in sorted(everything - set(best)))
            return (
                f"The best next set (without {missing}) would raise the prediction error by more "
                f"than {STOP_PE:.0%} (R-MAT-05): {left_t:,} treated units are left unmatched "
                "rather than matched on too little."
            )
        processed.add(frozenset(best))
        level = len(levels) + 1
        for c in everything - set(best):
            info = infos[[i.column for i in infos].index(used[c])]
            if info.dropped_at is None:
                info.dropped_at = level
        groups, nt, nc = engine.match(best, level)
        dropped = ", ".join(used[c] for c in sorted(everything - set(best)))
        levels.append(
            MatchingLevel(
                level, [used[c] for c in best], dropped, error, engine.balancing(best), groups,
                nt, nc,
            )
        )  # fmt: skip
    return f"Stopped after {DAME_LEVELS} levels (R-MAT-09)."


# --- The offer (R-MAT-01), instant ---------------------------------------------------------------


def offer(actions: list[str], covariates: list[str], counts: dict[str, int]) -> MatchingOffer:
    """R-MAT-01: two values (a treatment and its control), at least one covariate, at least
    MIN_ARM units of each. 123 bis (R-MAT-10): with 3 to 10 values, one pair at a time (each
    treatment against the control), the two largest arms being the smallest that must hold."""
    if len(actions) < 2 or len(actions) > MAX_ARMS:
        return MatchingOffer(
            False,
            f"Matched groups compare a treatment with its control: {len(actions)} actions here, "
            f"two to {MAX_ARMS} are needed.",
        )
    if len(actions) > 2:
        if not covariates:
            return MatchingOffer(False, "Matched groups need a column describing the units.")
        big = [a for a in actions if counts.get(a, 0) >= MIN_ARM]
        if len(big) < 2:
            return MatchingOffer(
                False, f"Fewer than two actions with {MIN_ARM} units or more: too few to match."
            )
        return MatchingOffer(
            True,
            f"{len(actions)} actions ({', '.join(actions[:5])}{'…' if len(actions) > 5 else ''}): "
            "matched groups compare one treatment with the control at a time; choose the pair "
            "(R-MAT-10).",
        )
    if not covariates:
        return MatchingOffer(False, "Matched groups need a column describing the units.")
    small = [a for a in actions if counts.get(a, 0) < MIN_ARM]
    if small:
        return MatchingOffer(
            False, f"Fewer than {MIN_ARM} units with {', '.join(small)}: too few to match."
        )
    shown = ", ".join(covariates[:4]) + ("…" if len(covariates) > 4 else "")
    return MatchingOffer(
        True,
        f"Two actions ({actions[0]}, {actions[1]}) and {len(covariates)} column"
        f"{'s' if len(covariates) > 1 else ''} describing the units ({shown}): the units can be "
        "matched exactly on them, to compare like with like (R-MAT-01).",
    )


# --- Discrete covariates (R-MAT-02) --------------------------------------------------------------


def _fmt(value: float) -> str:
    text = f"{value:.4g}"
    return text.replace("-", "−") if value < 0 else text


def discrete(column: str, values: pd.Series, bins: int = BINS) -> tuple[np.ndarray, list[str], str]:
    """Integer codes, the condition each code stands for, and how they were made. A number with
    many values is cut at its quantiles; a category keeps its most frequent values."""
    if pd.api.types.is_bool_dtype(values) or (
        values.dtype == object and set(values.dropna().unique()) <= {True, False}
    ):
        labels = values.map({True: "True", False: "False"}).fillna(EMPTY).astype(str)
        return _coded(column, labels, "yes / no")
    if pd.api.types.is_numeric_dtype(values):
        numbers = values.astype(float)
        seen = np.unique(numbers.dropna().to_numpy())
        if len(seen) <= KEEP_VALUES:
            labels = numbers.map(lambda v: EMPTY if pd.isna(v) else _fmt(float(v)))
            order = [_fmt(float(v)) for v in seen] + [EMPTY]
            return _coded(column, labels, f"its {len(seen)} values", order)
        cuts = np.unique(np.quantile(numbers.dropna().to_numpy(), np.arange(1, bins) / bins))
        codes = np.digitize(numbers.to_numpy(), cuts, right=True)
        texts = (
            [f"{column} ≤ {_fmt(cuts[0])}"]
            + [f"{_fmt(a)} < {column} ≤ {_fmt(b)}" for a, b in itertools.pairwise(cuts)]
            + [f"{column} > {_fmt(cuts[-1])}"]
        )
        missing = numbers.isna().to_numpy()
        if missing.any():
            codes = np.where(missing, len(texts), codes)
            texts.append(f"{column} {EMPTY}")
        return (
            codes.astype(np.int64),
            texts,
            f"{len(cuts) + 1} bins at its quantiles",
        )
    labels = values.astype(str).where(values.notna(), EMPTY)
    counts = labels.value_counts()
    kept = list(counts.index[:TOP_CATEGORIES])
    how = "its values"
    if len(counts) > TOP_CATEGORIES:
        labels = labels.where(labels.isin(kept), OTHER)
        how = f"its {TOP_CATEGORIES} most frequent values, the rest as {OTHER}"
    return _coded(column, labels, how)


def _coded(
    column: str, labels: pd.Series, how: str, order: list[str] | None = None
) -> tuple[np.ndarray, list[str], str]:
    seen = list(dict.fromkeys(order or [])) or sorted(labels.unique(), key=str)
    seen = [s for s in seen if s in set(labels.unique())]
    position = {label: i for i, label in enumerate(seen)}
    codes = labels.map(position).to_numpy(dtype=np.int64)
    texts = [f"{column} {label}" if label == EMPTY else f"{column} = {label}" for label in seen]
    return codes, texts, how


# --- FLAME in SQL ---------------------------------------------------------------------------------


def _k(j: int) -> str:
    return f"k{j}"


class _Flame:
    """The matching set `m` and the held-out set `h` in an in-memory DuckDB."""

    def __init__(self, connection: duckdb.DuckDBPyConnection, p: int) -> None:
        self.db = connection
        self.p = p
        self.next_id = 1
        h = self.db.execute("SELECT t, var_pop(y) FROM h GROUP BY t ORDER BY t").fetchall()
        self.spread = {int(t): float(v or 0.0) for t, v in h}

    def prediction_error(self, columns: list[int]) -> float:
        """R-MAT-03: on the held-out rows, each arm's outcome predicted by the mean of its cell
        (leave one out, the arm's mean for a cell of one), as a share of the arm's variance;
        the mean over both arms."""
        keys = "".join(f", {_k(j)}" for j in columns)
        rows = self.db.execute(
            f"""
            WITH cells AS (
                SELECT t{keys}, count(*) AS n, sum(y) AS s, sum(y * y) AS q FROM h GROUP BY t{keys}
            ), arms AS (SELECT t, sum(n) AS an, sum(s) AS asum FROM cells GROUP BY t)
            SELECT c.t, sum(CASE
                WHEN c.n > 1 THEN c.n * greatest(c.n * c.q - c.s * c.s, 0) / ((c.n - 1) * (c.n - 1))
                ELSE power(c.s - (a.asum - c.s) / (a.an - 1), 2) END) / max(a.an)
            FROM cells c JOIN arms a USING (t) GROUP BY c.t ORDER BY c.t
            """
        ).fetchall()
        shares = [
            float(mse) / self.spread[int(t)] if self.spread.get(int(t), 0) > 0 else 0.0
            for t, mse in rows
        ]
        return float(np.mean(shares)) if shares else 0.0

    def _cells(self, columns: list[int]) -> str:
        """The cells on `columns` that hold a treated and a control unit (matched before or
        not, R-MAT-04) and at least one unit still unmatched: those units are matched now."""
        keys = ", ".join(_k(j) for j in columns)
        return (
            f"SELECT {keys}, sum(t) AS nt, count(*) - sum(t) AS nc, "
            "sum(CASE WHEN grp IS NULL THEN t ELSE 0 END) AS new_t, "
            "sum(CASE WHEN grp IS NULL THEN 1 - t ELSE 0 END) AS new_c "
            f"FROM m GROUP BY {keys} "
            "HAVING sum(t) > 0 AND count(*) > sum(t) AND count(*) FILTER (WHERE grp IS NULL) > 0"
        )

    def unmatched(self) -> tuple[int, int]:
        t, c = self.db.execute(
            "SELECT coalesce(sum(t), 0), count(*) - coalesce(sum(t), 0) FROM m WHERE grp IS NULL"
        ).fetchone() or (0, 0)
        return int(t), int(c)

    def balancing(self, columns: list[int]) -> float:
        """R-MAT-04: the share of the unmatched controls plus the share of the unmatched
        treated that matching on `columns` would match now."""
        left_t, left_c = self.unmatched()
        nt, nc = self.db.execute(
            f"SELECT coalesce(sum(new_t), 0), coalesce(sum(new_c), 0) FROM ({self._cells(columns)})"
        ).fetchone() or (0, 0)
        return (float(nc) / left_c if left_c else 0.0) + (float(nt) / left_t if left_t else 0.0)

    def match(self, columns: list[int], level: int) -> tuple[int, int, int]:
        """Group the units exactly on `columns`: in a cell holding treated and control units,
        the unmatched ones get this group as theirs, the ones matched before serve again as
        matches (FLAME's repeats). Returns (groups, treated, control) newly matched."""
        keys = ", ".join(_k(j) for j in columns)
        self.db.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE g AS
            SELECT *, {self.next_id - 1} + row_number() OVER (ORDER BY nt + nc DESC, {keys}) AS gid
            FROM ({self._cells(columns)})
            """
        )
        found, nt, nc = self.db.execute(
            "SELECT count(*), coalesce(sum(new_t), 0), coalesce(sum(new_c), 0) FROM g"
        ).fetchone() or (0, 0, 0)
        if found:
            joined = " AND ".join(f"m.{_k(j)} = g.{_k(j)}" for j in columns)
            self.db.execute(
                f"INSERT INTO members SELECT g.gid, {level}, m.i FROM m JOIN g ON {joined}"
            )
            self.db.execute(
                f"UPDATE m SET grp = g.gid, lvl = {level} FROM g WHERE m.grp IS NULL AND {joined}"
            )
            self.next_id += int(found)
        return int(found), int(nt), int(nc)


# --- Effects and balance -----------------------------------------------------------------------


def _pooled(y: np.ndarray, groups: np.ndarray) -> float:
    """Within-group variance of the outcome (R-MAT-06): pooled over the groups with two units or
    more; the plain variance when there is none."""
    frame = pd.DataFrame({"g": groups, "y": y})
    stats = frame.groupby("g")["y"].agg(["count", "var"])
    counts = np.asarray(stats["count"], dtype=float)
    spreads = np.asarray(stats["var"], dtype=float)
    kept = counts >= 2
    df = float((counts[kept] - 1).sum())
    if df > 0:
        return float(((counts[kept] - 1) * spreads[kept]).sum() / df)
    return float(np.var(y, ddof=1)) if len(y) > 1 else 0.0


def _smd(
    values: np.ndarray, numeric: bool, t: np.ndarray, weights: np.ndarray | None = None
) -> float | None:
    """Standardized difference (R-MAT-07): a number's mean gap ÷ the pooled spread; a
    category's largest gap of share ÷ √(p(1 − p))."""
    w = np.ones(len(values)) if weights is None else weights
    treated, control = t == 1, t == 0
    if not treated.any() or not control.any():
        return None
    if numeric:
        ok = ~np.isnan(values)
        a, b = ok & treated, ok & control
        if w[a].sum() <= 0 or w[b].sum() <= 0:
            return None
        ma = np.average(values[a], weights=w[a])
        mb = np.average(values[b], weights=w[b])
        va = np.average((values[a] - ma) ** 2, weights=w[a])
        vb = np.average((values[b] - mb) ** 2, weights=w[b])
        spread = math.sqrt((va + vb) / 2)
        return float(abs(ma - mb) / spread) if spread > 0 else 0.0
    worst = 0.0
    for level in np.unique(values):
        on = values == level
        pa = w[treated & on].sum() / w[treated].sum()
        pb = w[control & on].sum() / w[control].sum()
        pooled = (pa + pb) / 2
        if 0 < pooled < 1:
            worst = max(worst, abs(pa - pb) / math.sqrt(pooled * (1 - pooled)))
    return float(worst)


def _value(kind: str, value: float | None) -> str:
    from quantilo.experiments.analyze import _value as shown

    return shown(cast(Any, kind), value)


def _gap(kind: str, value: float | None) -> str:
    from quantilo.experiments.analyze import _gap as shown

    return shown(cast(Any, kind), value)


def _effect(kind: str, value: float | None) -> str:
    """An effect with its unit: points for a rate, per unit otherwise."""
    if value is None:
        return "—"
    return f"{_gap(kind, value)} {'points' if kind == 'rate' else 'per unit'}"


# --- FLAME ---------------------------------------------------------------------------------------


def flame(
    frame: pd.DataFrame,
    covariates: list[str],
    treatment: str,
    treated: str,
    control: str,
    outcome: str,
    bins: int = BINS,
    alpha: float = 0.05,
    sampled: bool = False,
    method: Literal["flame", "dame"] = "flame",
) -> MatchedGroups | MatchingUnavailable:
    """`frame`: TREATED (1 / 0), OUTCOME (a number), ROW (row number in the file, optional) and
    the covariates by name, typed (numbers as floats)."""
    data = frame.dropna(subset=[TREATED, OUTCOME]).reset_index(drop=True)
    t_all = data[TREATED].astype(int).to_numpy()
    y_all = data[OUTCOME].astype(float).to_numpy()
    for label, arm in ((treated, 1), (control, 0)):
        if int((t_all == arm).sum()) < MIN_ARM:
            return MatchingUnavailable(
                f"Fewer than {MIN_ARM} units with {label}: too few to match (R-MAT-01)."
            )
    kind: Literal["rate", "mean"] = "rate" if np.isin(y_all, [0.0, 1.0]).all() else "mean"
    t_name, c_name = f"{treatment} = {treated}", f"{treatment} = {control}"

    # R-MAT-02: each covariate made discrete
    infos: list[MatchingCovariate] = []
    codes: list[np.ndarray] = []
    used: list[str] = []
    for i, column in enumerate(covariates):
        if i >= MAX_COVARIATES:
            infos.append(
                MatchingCovariate(
                    column, False, f"left out: at most {MAX_COVARIATES} covariates", [], None
                )
            )
            continue
        found, texts, how = discrete(column, cast(pd.Series, data[column]), bins)
        if len(texts) < 2:
            infos.append(MatchingCovariate(column, False, "left out: a single value", texts, None))
            continue
        infos.append(MatchingCovariate(column, True, how, texts, None))
        codes.append(found)
        used.append(column)
    if not used:
        return MatchingUnavailable(
            "No covariate with two values or more: nothing to match the units on."
        )

    # R-MAT-03: a held-out share of each arm, never matched, measures the prediction error
    rng = np.random.default_rng(SEED)
    held = np.zeros(len(data), dtype=bool)
    for arm in (0, 1):
        rows = np.flatnonzero(t_all == arm)
        held[rng.choice(rows, max(5, round(HOLDOUT * len(rows))), replace=False)] = True
    table = pd.DataFrame({"i": np.arange(len(data)), "t": t_all, "y": y_all})
    for j, found in enumerate(codes):
        table[_k(j)] = found
    connection = duckdb.connect()
    try:
        connection.register("frame_m", table[~held])
        connection.register("frame_h", table[held])
        connection.execute(
            "CREATE TABLE m AS SELECT *, NULL::BIGINT AS grp, NULL::INTEGER AS lvl FROM frame_m"
        )
        connection.execute("CREATE TABLE h AS SELECT * FROM frame_h")
        connection.execute("CREATE TABLE members (gid BIGINT, lvl INTEGER, i BIGINT)")
        engine = _Flame(connection, len(used))
        active = list(range(len(used)))
        levels: list[MatchingLevel] = []
        full = engine.prediction_error(active)
        balance_now = engine.balancing(active)
        groups, nt, nc = engine.match(active, 1)
        levels.append(MatchingLevel(1, list(used), None, full, balance_now, groups, nt, nc))
        # R-MAT-09 (123 bis): DAME searches the subsets; FLAME drops one covariate for good
        stopped = _dame(engine, used, infos, levels, full) if method == "dame" else ""
        while method == "flame":
            left_t, left_c = engine.unmatched()
            if not left_t or not left_c:
                stopped = (
                    "Every treated unit is matched."
                    if not left_t
                    else f"Every unit with {c_name} is matched; {left_t:,} treated units are left."
                )
                break
            if len(active) == 1:
                stopped = (
                    f"One covariate left ({used[active[0]]}): matching on none would compare "
                    f"everyone with everyone. {left_t:,} treated units are left unmatched."
                )
                break
            tried = []
            for j in active:
                rest = [c for c in active if c != j]
                error = engine.prediction_error(rest)
                tried.append((TRADEOFF * engine.balancing(rest) - error, -error, -j, j, error))
            _, _, _, drop, error = max(tried)
            if error > full * (1 + STOP_PE) + 1e-12:
                stopped = (
                    f"Dropping another covariate (at best {used[drop]}) would raise the prediction "
                    f"error by more than {STOP_PE:.0%} (R-MAT-05): {left_t:,} treated units are "
                    "left unmatched rather than matched on too little."
                )
                break
            active = [c for c in active if c != drop]
            infos[[c.column for c in infos].index(used[drop])].dropped_at = len(levels) + 1
            balance_now = engine.balancing(active)
            groups, nt, nc = engine.match(active, len(levels) + 1)
            levels.append(
                MatchingLevel(
                    len(levels) + 1,
                    [used[c] for c in active],
                    used[drop],
                    error,
                    balance_now,
                    groups,
                    nt,
                    nc,
                )
            )
        assigned = connection.execute("SELECT i, grp FROM m ORDER BY i").df()
        memberships = connection.execute("SELECT gid, lvl, i FROM members ORDER BY gid, i").df()
    finally:
        connection.close()

    # R-MAT-06: effects per group, on average and on the treated
    m_index = assigned["i"].to_numpy(dtype=np.int64)
    t = t_all[m_index]
    y = y_all[m_index]
    group = assigned["grp"].to_numpy(dtype=float)  # each unit's own group
    on = ~np.isnan(group)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    member_gid = memberships["gid"].to_numpy(dtype=np.int64)
    member_i = memberships["i"].to_numpy(dtype=np.int64)
    member_level = memberships["lvl"].to_numpy(dtype=np.int64)
    member_t = t_all[member_i]
    member_y = y_all[member_i]
    var_t = _pooled(member_y[member_t == 1], member_gid[member_t == 1]) if len(member_i) else 0.0
    var_c = _pooled(member_y[member_t == 0], member_gid[member_t == 0]) if len(member_i) else 0.0
    ids = np.unique(member_gid)
    owners = pd.Series(group[on], dtype=float).value_counts()
    owners_t = pd.Series(group[on & (t == 1)], dtype=float).value_counts()
    rows_of = data[ROW].to_numpy() if ROW in data else np.arange(len(data))
    level_columns = {lv.level: lv.covariates for lv in levels}
    starts = np.searchsorted(member_gid, ids, side="left")
    ends = np.searchsorted(member_gid, ids, side="right")
    stats: list[MatchedGroup] = []
    owned, effects, variances, owned_t = [], [], [], []
    weights = np.zeros(len(data))  # R-MAT-07: what each unit weighs in the average effect
    for gid, start, end in zip(ids, starts, ends, strict=True):
        units = member_i[start:end]
        arm = member_t[start:end]
        values = member_y[start:end]
        n_t, n_c = int((arm == 1).sum()), int((arm == 0).sum())
        mean_t, mean_c = float(values[arm == 1].mean()), float(values[arm == 0].mean())
        effect = mean_t - mean_c
        variance = var_t / n_t + var_c / n_c
        level = int(member_level[start])
        first = int(units[0])
        conditions = [
            infos[[c.column for c in infos].index(name)].levels[int(codes[used.index(name)][first])]
            for name in level_columns[level]
        ]
        spread = z * math.sqrt(variance) if variance > 0 else None
        stats.append(
            MatchedGroup(
                int(gid),
                level,
                conditions,
                n_t,
                n_c,
                mean_t,
                mean_c,
                effect,
                effect - spread if spread is not None else None,
                effect + spread if spread is not None else None,
                sorted(int(r) for r in rows_of[units])[:ROWS_SHOWN],
            )
        )
        own = float(owners.get(float(gid)) or 0)
        owned.append(own)
        effects.append(effect)
        variances.append(variance)
        owned_t.append(float(owners_t.get(float(gid)) or 0))
        weights[units[arm == 1]] += own / n_t
        weights[units[arm == 0]] += own / n_c

    total_t, total_c = int((t == 1).sum()), int((t == 0).sum())
    matched_t, matched_c = int((on & (t == 1)).sum()), int((on & (t == 0)).sum())
    level_text = f"{(1 - alpha) * 100:.0f} %"

    def combined(weights: np.ndarray) -> tuple[float | None, float | None, float | None]:
        if not len(weights) or weights.sum() <= 0:
            return None, None, None
        w = weights / weights.sum()
        value = float(np.dot(w, effects))
        spread = z * math.sqrt(float(np.dot(w**2, variances)))
        return value, value - spread, value + spread

    naive = float(y[t == 1].mean() - y[t == 0].mean())
    naive_spread = z * math.sqrt(
        float(np.var(y[t == 1], ddof=1)) / total_t + float(np.var(y[t == 0], ddof=1)) / total_c
    )
    ate = combined(np.asarray(owned, dtype=float))
    att = combined(np.asarray(owned_t, dtype=float))

    def sentence(label: str, value: tuple[float | None, float | None, float | None]) -> str:
        return (
            f"{label}: {_effect(kind, value[0])} on {outcome}, {level_text} CI "
            f"[{_gap(kind, value[1])} ; {_gap(kind, value[2])}]."
        )

    found_effects = [
        MatchingEffect(
            "naive",
            "Naive gap (no matching)",
            naive,
            naive - naive_spread,
            naive + naive_spread,
            total_t + total_c,
            sentence(
                f"Naive gap, {t_name} vs {c_name} without matching",
                (naive, naive - naive_spread, naive + naive_spread),
            ),
        ),
        MatchingEffect(
            "ate",
            "Average effect (matched units)",
            *ate,
            matched_t + matched_c,
            sentence(f"Average effect of {t_name} over the matched units", ate),
        ),
        MatchingEffect(
            "att",
            "Effect on the treated (matched)",
            *att,
            matched_t,
            sentence(f"Effect on the matched units with {t_name}", att),
        ),
    ]

    # R-MAT-07: balance before and after
    weight = weights[m_index]
    kept = weight > 0
    balance: list[BalanceRow] = []
    for info in infos:
        raw = cast(pd.Series, data[info.column]).iloc[m_index]
        numeric = pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_bool_dtype(raw)
        values = (
            raw.astype(float).to_numpy()
            if numeric
            else raw.astype(str).where(raw.notna(), EMPTY).to_numpy()
        )
        before = _smd(values, numeric, t)
        after = _smd(values[kept], numeric, t[kept], weight[kept]) if kept.any() else None
        balance.append(
            BalanceRow(
                info.column,
                before,
                after,
                after is not None and after <= BALANCED,
                info.dropped_at,
            )
        )

    stats.sort(key=lambda g: (-(g.treated + g.control), g.id))
    share = matched_t / total_t if total_t else 0.0
    ate_value = found_effects[1]
    summary = (
        f"{matched_t:,} of {total_t:,} units with {t_name} ({share:.0%}) matched exactly with "
        f"{matched_c:,} with {c_name} in {len(ids):,} group{'s' if len(ids) != 1 else ''}, on "
        f"{len(levels)} level{'s' if len(levels) != 1 else ''}. Average effect of {t_name} on "
        f"{outcome}: {_effect(kind, ate_value.value)} ({level_text} CI "
        f"[{_gap(kind, ate_value.low)} ; {_gap(kind, ate_value.high)}]), against a naive gap of "
        f"{_effect(kind, naive)}."
        if len(ids)
        else f"No group holds units with {t_name} and with {c_name}: no unit could be matched."
    )
    warnings = [OBSERVATIONAL]
    if total_t and share < 0.5:
        warnings.append(
            f"Only {share:.0%} of the units with {t_name} found a match: the effect speaks for "
            "them, not for every treated unit."
        )
    unbalanced = [row.column for row in balance if row.after is not None and not row.balanced]
    if unbalanced:
        warnings.append(
            f"Still unbalanced after matching (standardized difference above {BALANCED:g}): "
            f"{', '.join(unbalanced[:5])}. Units matched on the later levels no longer share it."
        )
    if sampled:
        warnings.append(f"Matched on a reproducible sample of {MAX_UNITS:,} units.")
    return MatchedGroups(
        treatment,
        treated,
        control,
        outcome,
        kind,
        len(t),
        int(held.sum()),
        sampled,
        total_t,
        total_c,
        matched_t,
        matched_c,
        found_effects,
        len(ids),
        stats[:GROUPS_SHOWN],
        levels,
        infos,
        balance,
        stopped,
        summary,
        warnings,
        (DAME_METHOD if method == "dame" else METHOD).format(holdout=f"{HOLDOUT:.0%}"),
    )
