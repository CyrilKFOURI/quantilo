"""Bivariate tests with effect sizes.

Tests that only need means, variances or counts (Pearson, Welch t, ANOVA, Welch-ANOVA, χ²) run on
every row through those summaries; rank tests run on a seeded sample.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats

from quantilo.variable.source import clean

Magnitude = Literal["negligible", "small", "medium", "large"]


@dataclass
class Effect:
    name: str  # r, rho, tau, R², d, r_rb, η², ε², V
    value: float | None
    magnitude: Magnitude | None


@dataclass
class BivariateTest:
    # pearson, spearman, kendall, welch_t, mann_whitney, anova, welch_anova, kruskal, levene,
    # chi2, fisher
    key: str
    statistic: float | None
    p_value: float | None
    n: int
    effect: Effect | None = None
    recommended: bool = False
    note: str | None = None


@dataclass
class GroupStats:
    n: int
    mean: float
    var: float  # sample variance


def magnitude(value: float | None, small: float, medium: float, large: float) -> Magnitude | None:
    if value is None:
        return None
    size = abs(value)
    if size < small:
        return "negligible"
    if size < medium:
        return "small"
    if size < large:
        return "medium"
    return "large"


def correlation_effect(name: str, value: float | None) -> Effect:
    return Effect(name, value, magnitude(value, 0.1, 0.3, 0.5))


def _quiet(compute) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute()


# --- num × num ---------------------------------------------------------------------------------


def pearson(r: float | None, n: int) -> BivariateTest:
    """Pearson's r from its value on every row; p-value from Student's t with n - 2 df."""
    if r is None or n < 3:
        return BivariateTest("pearson", None, None, n, note="Needs at least 3 rows with variation.")
    r = max(-1.0, min(1.0, r))
    if abs(r) == 1:
        p = 0.0
    else:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        p = float(2 * stats.t.sf(abs(t), n - 2))
    return BivariateTest("pearson", r, p, n, correlation_effect("r", r))


def rank_correlations(x: np.ndarray, y: np.ndarray) -> list[BivariateTest]:
    n = len(x)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        note = "Needs at least 3 rows with variation."
        return [
            BivariateTest("spearman", None, None, n, note=note),
            BivariateTest("kendall", None, None, n, note=note),
        ]
    rho = _quiet(lambda: stats.spearmanr(x, y))
    tau = _quiet(lambda: stats.kendalltau(x, y))
    return [
        BivariateTest(
            "spearman",
            clean(rho.statistic),
            clean(rho.pvalue),
            n,
            correlation_effect("rho", clean(rho.statistic)),
        ),
        BivariateTest(
            "kendall",
            clean(tau.statistic),
            clean(tau.pvalue),
            n,
            correlation_effect("tau", clean(tau.statistic)),
        ),
    ]


# --- num × groups -------------------------------------------------------------------------------


def welch_t(a: GroupStats, b: GroupStats) -> BivariateTest:
    n = a.n + b.n
    if a.n < 2 or b.n < 2 or (a.var == 0 and b.var == 0):
        return BivariateTest(
            "welch_t", None, None, n, note="Each group needs 2 values and some variation."
        )
    result = _quiet(
        lambda: stats.ttest_ind_from_stats(
            a.mean, math.sqrt(a.var), a.n, b.mean, math.sqrt(b.var), b.n, equal_var=False
        )
    )
    pooled = math.sqrt(((a.n - 1) * a.var + (b.n - 1) * b.var) / (n - 2))
    d = (a.mean - b.mean) / pooled if pooled else None
    return BivariateTest(
        "welch_t",
        clean(result.statistic),
        clean(result.pvalue),
        n,
        Effect("d", clean(d), magnitude(d, 0.2, 0.5, 0.8)),
        recommended=True,
        note="Welch's version: it does not assume equal variances.",
    )


def welch_interval(
    a: GroupStats, b: GroupStats, alpha: float = 0.05
) -> tuple[float | None, float | None]:
    """Interval of b's mean minus a's, with Welch–Satterthwaite degrees of freedom."""
    if a.n < 2 or b.n < 2:
        return None, None
    va, vb = a.var / a.n, b.var / b.n
    se2 = va + vb
    if se2 <= 0:
        return None, None
    df = se2**2 / (va**2 / (a.n - 1) + vb**2 / (b.n - 1))
    margin = float(stats.t.ppf(1 - alpha / 2, df)) * math.sqrt(se2)
    diff = b.mean - a.mean
    return diff - margin, diff + margin


def mann_whitney(a: np.ndarray, b: np.ndarray) -> BivariateTest:
    n = len(a) + len(b)
    if len(a) < 1 or len(b) < 1:
        return BivariateTest("mann_whitney", None, None, n, note="Each group needs values.")
    result = _quiet(lambda: stats.mannwhitneyu(a, b, alternative="two-sided"))
    rank_biserial = 2 * float(result.statistic) / (len(a) * len(b)) - 1
    return BivariateTest(
        "mann_whitney",
        clean(result.statistic),
        clean(result.pvalue),
        n,
        correlation_effect("r_rb", rank_biserial),
    )


def anovas(groups: list[GroupStats]) -> list[BivariateTest]:
    """Classic one-way ANOVA and Welch's ANOVA, from each group's n, mean and variance."""
    usable = [g for g in groups if g.n >= 2]
    n = sum(g.n for g in usable)
    k = len(usable)
    if k < 2 or n - k < 1:
        note = "Needs 2 groups with at least 2 values each."
        return [
            BivariateTest("anova", None, None, n, note=note),
            BivariateTest("welch_anova", None, None, n, note=note),
        ]
    grand = sum(g.n * g.mean for g in usable) / n
    between = sum(g.n * (g.mean - grand) ** 2 for g in usable)
    within = sum((g.n - 1) * g.var for g in usable)
    results = []
    if within > 0:
        f = (between / (k - 1)) / (within / (n - k))
        p = float(stats.f.sf(f, k - 1, n - k))
        eta = between / (between + within)
        results.append(
            BivariateTest("anova", f, p, n, Effect("η²", eta, magnitude(eta, 0.01, 0.06, 0.14)))
        )
    else:
        results.append(
            BivariateTest("anova", None, None, n, note="No variation inside the groups.")
        )
    if all(g.var > 0 for g in usable):
        weights = [g.n / g.var for g in usable]
        total = sum(weights)
        weighted = sum(w * g.mean for w, g in zip(weights, usable, strict=True)) / total
        a = sum(w * (g.mean - weighted) ** 2 for w, g in zip(weights, usable, strict=True)) / (
            k - 1
        )
        spread = sum((1 - w / total) ** 2 / (g.n - 1) for w, g in zip(weights, usable, strict=True))
        b = 1 + 2 * (k - 2) / (k * k - 1) * spread
        f = a / b
        df2 = (k * k - 1) / (3 * spread)
        results.append(BivariateTest("welch_anova", f, float(stats.f.sf(f, k - 1, df2)), n))
    else:
        results.append(
            BivariateTest("welch_anova", None, None, n, note="A group has no variation.")
        )
    return results


def kruskal(samples: list[np.ndarray]) -> BivariateTest:
    usable = [s for s in samples if len(s)]
    n = sum(len(s) for s in usable)
    if len(usable) < 2 or n < 3:
        return BivariateTest("kruskal", None, None, n, note="Needs 2 groups with values.")
    result = _quiet(lambda: stats.kruskal(*usable))
    h = clean(result.statistic)
    epsilon = h * (n + 1) / (n * n - 1) if h is not None else None
    return BivariateTest(
        "kruskal",
        h,
        clean(result.pvalue),
        n,
        Effect("ε²", epsilon, magnitude(epsilon, 0.01, 0.06, 0.14)),
    )


def levene(samples: list[np.ndarray]) -> BivariateTest:
    usable = [s for s in samples if len(s) >= 2]
    n = sum(len(s) for s in usable)
    if len(usable) < 2:
        return BivariateTest("levene", None, None, n, note="Needs 2 groups with at least 2 values.")
    result = _quiet(lambda: stats.levene(*usable, center="median"))
    return BivariateTest(
        "levene",
        clean(result.statistic),
        clean(result.pvalue),
        n,
        note="Equal variances? (Brown–Forsythe, median-centred)",
    )


def group_tests(groups: list[GroupStats], samples: list[np.ndarray]) -> list[BivariateTest]:
    """The tests for a numeric variable across groups, with the recommended one flagged."""
    if len(groups) == 2:
        return [welch_t(groups[0], groups[1]), mann_whitney(samples[0], samples[1])]
    spread = levene(samples)
    classic, welch = anovas(groups)
    unequal = spread.p_value is not None and spread.p_value < 0.05
    if unequal:
        welch.recommended = True
        welch.note = (
            "Recommended: Levene's test finds unequal variances, which classic ANOVA assumes away."
        )
    else:
        classic.recommended = True
        classic.note = "Recommended: Levene's test finds no evidence of unequal variances."
    return [classic, welch, kruskal(samples), spread]


# --- cat × cat ----------------------------------------------------------------------------------


def contingency_tests(table: np.ndarray) -> list[BivariateTest]:
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    n = int(table.sum())
    if table.shape[0] < 2 or table.shape[1] < 2:
        return [
            BivariateTest("chi2", None, None, n, note="Needs at least 2 categories on each side.")
        ]
    result = _quiet(lambda: stats.chi2_contingency(table, correction=False))
    chi2 = float(result.statistic)
    smaller = min(table.shape) - 1
    v = math.sqrt(chi2 / (n * smaller)) if n else None
    scale = math.sqrt(smaller)
    effect = Effect("V", v, magnitude(v, 0.1 / scale, 0.3 / scale, 0.5 / scale))
    low_expected = float((result.expected_freq < 5).mean())
    note = (
        f"{low_expected:.0%} of expected counts are below 5: the χ² approximation is shaky."
        if low_expected > 0.2
        else None
    )
    tests = [
        BivariateTest("chi2", chi2, float(result.pvalue), n, effect, recommended=True, note=note)
    ]
    if table.shape == (2, 2):
        fisher = _quiet(lambda: stats.fisher_exact(table))
        exact = BivariateTest(
            "fisher",
            clean(fisher.statistic),
            clean(fisher.pvalue),
            n,
            note="Exact test for 2 × 2 tables; statistic = odds ratio.",
        )
        if low_expected > 0:
            exact.recommended = True
            tests[0].recommended = False
        tests.append(exact)
    return tests
