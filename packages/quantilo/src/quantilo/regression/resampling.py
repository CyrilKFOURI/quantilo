"""Resampling (077): intervals that do not lean on a formula. The bootstrap resamples what is
independent (rows, whole clusters, or blocks of a series), the wild cluster bootstrap tests each
coefficient when the clusters are few, and the Freedman–Lane permutation tests one coefficient
without assuming normal errors. 999 draws, seeded; beyond 5·10⁷ resampled values, "m out of n"
as in `compare/bootstrap.py`. Rules R-REG-RS-01…05.

Everything is numpy / scipy on one thread: Pyodide-compatible."""

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats

from quantilo.compare import bootstrap as bs
from quantilo.regression import analyze as an

DRAWS = 999
SEED = 42
BUDGET = 50_000_000  # resampled values (rows × draws)
CHUNK = 2_000_000  # values held at once
WILD_CLUSTERS = 100  # R-REG-RS-03: beyond, the clustered errors are reliable enough
FEW_CLUSTERS = 40  # R-REG-SE-03
WEBB_BELOW = 12  # R-REG-RS-03
MAX_TERMS = 20
LEAST_SQUARES = ("ols", "wls", "log_ols", "boxcox")
WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])

Scheme = Literal["pairs", "cluster", "block"]


@dataclass
class ResampledTerm:
    term: str
    estimate: float
    low: float | None  # the first plane's interval
    high: float | None
    p_value: float | None
    boot_se: float | None
    boot_low: float | None
    boot_high: float | None
    wild_p: float | None = None


@dataclass
class Permutation:
    term: str
    statistic: float
    p_value: float
    draws: int
    reason: str
    sentence: str


@dataclass
class ResamplingTable:
    available: bool
    reason: str
    scheme: Scheme = "pairs"
    scheme_reason: str = ""
    draws: int = DRAWS
    rows: int = 0
    resampled: int = 0  # rows (or clusters) drawn each time
    terms: list[ResampledTerm] = field(default_factory=list)
    wild: bool = False
    wild_reason: str = ""
    weights: str | None = None  # "Rademacher" or "Webb"
    clusters: int | None = None
    permutation: Permutation | None = None
    permutation_reason: str = ""
    sentence: str = ""
    notes: list[str] = field(default_factory=list)


def _unavailable(fit: an.Fitted) -> str | None:
    if fit.ls is None or fit.model not in LEAST_SQUARES:
        label = fit.family.label if fit.family is not None else fit.model
        return (
            "Resampling is offered for a number explained by least squares; the "
            f"{label} model keeps its robust standard errors."
        )
    n, k = fit.design.x.shape
    if n < max(20, 2 * k + 5):
        return f"{n} rows for {k} terms: too few to resample."
    return None


def _codes(fit: an.Fitted) -> np.ndarray | None:
    grouping = fit.grouping
    if grouping.cluster_codes is not None:
        return grouping.cluster_codes
    if fit.inputs.shape.kind == "panel":
        return grouping.entity_codes
    return None


def scheme_of(fit: an.Fitted) -> tuple[Scheme, str]:
    """R-REG-RS-01: resample what is independent."""
    codes = _codes(fit)
    if fit.se.kind in ("cluster", "cluster2", "dk") and codes is not None:
        who = fit.inputs.cluster or fit.inputs.shape.entity or "cluster"
        return "cluster", (
            f"R-REG-RS-01: whole `{who}` groups are drawn, as the standard errors are clustered: "
            "rows of one group are not independent."
        )
    if fit.inputs.shape.kind == "series":
        n = fit.design.x.shape[0]
        return "block", (
            f"R-REG-RS-01: blocks of {block_length(n)} consecutive rows (n^⅓) are drawn: close "
            "dates are alike, single rows would break that."
        )
    return "pairs", "R-REG-RS-01: rows are drawn with their target (pairs bootstrap)."


def block_length(n: int) -> int:
    return max(2, round(n ** (1 / 3)))


def _solve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _counted(x: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Least squares on the drawn rows, each counted as often as drawn: (X'CX)⁻¹ X'Cy, no copy
    of the rows; a draw that leaves a level out falls back to the pseudo-inverse."""
    c = np.bincount(rows, minlength=len(y)).astype(float)
    xc = x * c[:, None]
    gram = xc.T @ x
    try:
        return np.linalg.solve(gram, xc.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ (xc.T @ y)


def bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    scheme: Scheme,
    codes: np.ndarray | None = None,
    draws: int = DRAWS,
    seed: int = SEED,
) -> tuple[np.ndarray, int]:
    """`draws` × k coefficients, the deviations from the full fit scaled by √(m/n) when only m
    of the n rows (or clusters) are drawn each time (R-REG-RS-02)."""
    n = len(y)
    beta = _solve(x, y)
    rng = np.random.default_rng(seed)
    found = np.empty((draws, x.shape[1]))
    if scheme == "cluster" and codes is not None:
        order = np.argsort(codes, kind="stable")
        starts = np.flatnonzero(np.r_[True, np.diff(codes[order]) != 0])
        members = np.split(order, starts[1:])
        g = len(members)
        m = min(g, max(2, BUDGET // (draws * max(n // g, 1))))
        for b in range(draws):
            rows = np.concatenate([members[i] for i in rng.integers(0, g, m)])
            found[b] = _counted(x, y, rows)
        units = g
    elif scheme == "block":
        length = block_length(n)
        m = min(n, BUDGET // draws)
        blocks = math.ceil(m / length)
        for b in range(draws):
            first = rng.integers(0, n - length + 1, blocks)
            rows = (first[:, None] + np.arange(length)).ravel()[:m]
            found[b] = _counted(x, y, rows)
        units = n
    else:
        m = min(n, BUDGET // draws)
        for b in range(draws):
            rows = rng.integers(0, n, m)
            found[b] = _counted(x, y, rows)
        units = n
    if m < units:
        found = beta + (found - beta) * math.sqrt(m / units)
    return found, m


def wild_cluster(
    x: np.ndarray,
    y: np.ndarray,
    codes: np.ndarray,
    columns: list[int],
    draws: int = DRAWS,
    seed: int = SEED,
) -> tuple[list[float | None], str]:
    """Wild cluster restricted bootstrap-t (Cameron, Gelbach & Miller 2008) of β_j = 0 for each
    j in `columns`, CR1 errors. Exact and fast: under the restricted fit, β*_j and the cluster
    scores are linear in the cluster weights v (G × draws), so no row is refitted (Roodman,
    MacKinnon, Nielsen & Webb 2019). Webb's six-point weights below 12 clusters, Rademacher
    above (R-REG-RS-03)."""
    n, k = x.shape
    _, codes = np.unique(codes, return_inverse=True)
    g = int(codes.max()) + 1
    bread = np.linalg.pinv(x.T @ x)
    p_matrix = bread @ x.T  # k × n: β̂ = P y
    beta = p_matrix @ y
    resid = y - x @ beta
    correction = g / max(g - 1, 1) * (n - 1) / max(n - k, 1)
    rng = np.random.default_rng(seed)
    if g < WEBB_BELOW:
        v = WEBB[rng.integers(0, 6, (g, draws))]
        label = "Webb"
    else:
        v = rng.choice(np.array([-1.0, 1.0]), (g, draws))
        label = "Rademacher"

    def sums(values: np.ndarray) -> np.ndarray:  # per-cluster sums of the rows of `values`
        flat = values.reshape(n, -1)
        out = np.zeros((g, flat.shape[1]))
        np.add.at(out, codes, flat)
        return out

    found: list[float | None] = []
    for j in columns:
        c = p_matrix[j]
        score = sums((c * resid)[:, None])[:, 0]
        se = math.sqrt(correction * float(score @ score))
        if se <= 0:
            found.append(None)
            continue
        t = beta[j] / se
        rest = np.delete(x, j, axis=1)
        restricted = y - rest @ _solve(rest, y)  # residuals of the fit without j
        a = sums((c * restricted)[:, None])[:, 0]  # β*_j = a · v
        cx = sums(c[:, None] * x)  # G × k: Σ_{i∈g} c_i x_i
        q = (sums(p_matrix.T * restricted[:, None])).T  # k × G: Σ_{i∈h} p_i e_i
        kernel = np.diag(a) - cx @ q  # cluster scores of the bootstrap residuals: K v
        stars = a @ v
        spread = np.sqrt(correction * ((kernel @ v) ** 2).sum(axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_star = np.where(spread > 0, stars / spread, 0.0)
        beyond = int((np.abs(t_star) >= abs(t) - 1e-12).sum())
        found.append((beyond + 1) / (draws + 1))
    return found, label


def freedman_lane(
    x: np.ndarray, y: np.ndarray, j: int, draws: int = DRAWS, seed: int = SEED
) -> tuple[float, float]:
    """Permutation p-value of β_j = 0 (Freedman & Lane 1983): permute the residuals of the
    fit without j, add them back to its fitted values, refit, compare the t statistics."""
    n, k = x.shape
    bread = np.linalg.pinv(x.T @ x)
    p_matrix = bread @ x.T
    rest = np.delete(x, j, axis=1)
    fitted = rest @ _solve(rest, y)
    resid = y - fitted

    def t_of(ys: np.ndarray) -> np.ndarray:  # ys: n × b
        beta = p_matrix @ ys
        e = ys - x @ beta
        s2 = (e**2).sum(axis=0) / max(n - k, 1)
        return beta[j] / np.sqrt(np.clip(s2 * bread[j, j], 1e-300, None))

    observed = float(t_of(y[:, None])[0])
    rng = np.random.default_rng(seed)
    chunk = max(1, CHUNK // n)
    beyond = 0
    done = 0
    while done < draws:
        b = min(chunk, draws - done)
        perms = rng.permuted(np.broadcast_to(np.arange(n), (b, n)), axis=1)
        ys = fitted[:, None] + resid[perms].T
        beyond += int((np.abs(t_of(ys)) >= abs(observed) - 1e-12).sum())
        done += b
    return observed, (beyond + 1) / (draws + 1)


def _focus(fit: an.Fitted, own: list[int]) -> int:
    focus = fit.inputs.focus
    if focus is not None:
        for j in own:
            if fit.design.columns[j] == focus:
                return j
    se = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
    return min(own, key=lambda j: -abs(fit.ls.beta[j] / se[j]) if se[j] > 0 else 0)  # type: ignore[union-attr]


def section(fit: an.Fitted) -> ResamplingTable:
    """Advanced › Resampling (077)."""
    missing = _unavailable(fit)
    if missing is not None:
        return ResamplingTable(False, missing)
    assert fit.ls is not None
    a = fit.ls.arrays
    x, y = a.x, a.x @ fit.ls.beta + a.e  # weighted rows: WLS is OLS on them
    n, k = x.shape
    scheme, why = scheme_of(fit)
    codes = _codes(fit)
    reps, m = bootstrap(x, y, scheme, codes if scheme == "cluster" else None)
    se = np.sqrt(np.clip(np.diag(fit.covariance), 0, None))
    crit = float(stats.t.ppf(0.975, fit.df))
    terms: list[ResampledTerm] = []
    for j in range(min(k, MAX_TERMS + 1)):
        b = float(fit.ls.beta[j])
        lo, hi = np.quantile(reps[:, j], [0.025, 0.975])
        has = se[j] > 0
        terms.append(
            ResampledTerm(
                fit.design.terms[j],
                b,
                b - crit * se[j] if has else None,
                b + crit * se[j] if has else None,
                float(2 * stats.t.sf(abs(b / se[j]), fit.df)) if has else None,
                float(np.std(reps[:, j], ddof=1)),
                float(lo),
                float(hi),
            )
        )
    table = ResamplingTable(
        True, "", scheme, why, DRAWS, n, m, terms, notes=_notes(fit, scheme, m, n, codes)
    )
    own = list(range(1, min(k, MAX_TERMS + 1)))
    if codes is not None and fit.se.kind == "cluster":
        g = len(np.unique(codes))
        table.clusters = g
        if g <= WILD_CLUSTERS:
            p_values, label = wild_cluster(x, y, codes, own)
            for term, p in zip(terms[1:], p_values, strict=True):
                term.wild_p = p
            table.wild, table.weights = True, label
            table.wild_reason = (
                f"R-REG-RS-03: {g} clusters"
                + (", fewer than 40: clustered errors are too narrow" if g < FEW_CLUSTERS else "")
                + f"; the wild cluster bootstrap-t ({label} weights"
                + (", as there are fewer than 12 clusters" if label == "Webb" else "")
                + ") gives each coefficient a p-value that holds with few clusters."
            )
        else:
            table.wild_reason = (
                f"{g} clusters: enough for the clustered standard errors; no wild bootstrap."
            )
    else:
        table.wild_reason = "The wild cluster bootstrap is for clustered standard errors."
    if own and scheme == "pairs":
        j = _focus(fit, own)
        rows = np.arange(n)
        if n * DRAWS > BUDGET:  # R-REG-RS-02: a reproducible sample of rows
            rows = np.sort(np.random.default_rng(SEED).choice(n, BUDGET // DRAWS, replace=False))
        statistic, p = freedman_lane(x[rows], y[rows], j)
        reason = (
            "R-REG-RS-04: the focus column"
            if fit.inputs.focus == fit.design.columns[j]
            else "R-REG-RS-04: the strongest effect"
        )
        sample = f" on a sample of {len(rows):,} rows" if len(rows) < n else ""
        table.permutation = Permutation(
            fit.design.terms[j],
            statistic,
            p,
            DRAWS,
            reason,
            f"Permuting the residuals of the model without `{fit.design.terms[j]}` "
            f"{DRAWS} times{sample}, a t as large as {abs(statistic):.2f} comes "
            f"{'rarely' if p < 0.05 else 'often'} by chance ({an.fmt_p(p)}, Freedman–Lane): "
            "no normal errors assumed.",
        )
    else:
        table.permutation_reason = (
            "Permutation assumes rows that can be exchanged: not with clusters or dates."
        )
    table.sentence = _sentence(table)
    return table


def _notes(fit: an.Fitted, scheme: Scheme, m: int, n: int, codes: np.ndarray | None) -> list[str]:
    found: list[str] = []
    units = len(np.unique(codes)) if scheme == "cluster" and codes is not None else n
    if m < units:
        found.append(
            "R-REG-RS-02: " + bs.sample_note(m, units).strip(" ()") + ", to stay within budget."
        )
    if fit.ls is not None and fit.ls.weights is not None:
        found.append(f"Rows keep their weights (`{fit.inputs.weights}`).")
    if fit.design.x.shape[1] > MAX_TERMS + 1:
        found.append(f"The first {MAX_TERMS} terms are shown.")
    return found


def _sentence(table: ResamplingTable) -> str:
    """R-REG-RS-05: say which coefficients the resampling sees differently."""
    differ: list[str] = []
    for term in table.terms[1:]:
        if None in (term.p_value, term.boot_low, term.boot_high):
            continue
        assert term.p_value is not None and term.boot_low is not None
        assert term.boot_high is not None
        clear = term.p_value < 0.05
        boot_clear = term.boot_low > 0 or term.boot_high < 0
        if clear != boot_clear:
            differ.append(term.term)
    wild = [t.term for t in table.terms[1:] if t.wild_p is not None and t.p_value is not None]
    wild_differ = [
        t.term
        for t in table.terms[1:]
        if t.term in wild
        and t.wild_p is not None
        and t.p_value is not None
        and (t.wild_p < 0.05) != (t.p_value < 0.05)
    ]
    if not differ and not wild_differ:
        return (
            f"{table.draws} bootstrap draws agree with the first plane: every effect that is "
            "clear there is clear here, and none other (R-REG-RS-05)."
        )
    names = ", ".join(f"`{name}`" for name in dict.fromkeys([*differ, *wild_differ]))
    return (
        f"The resampled intervals disagree with the first plane on {names}: trust the "
        "resampling there, it leans on no formula (R-REG-RS-05)."
    )
