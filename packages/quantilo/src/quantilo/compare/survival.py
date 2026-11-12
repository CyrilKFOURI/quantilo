"""Survival curves without a survival library (062): Kaplan-Meier with Greenwood intervals,
median survival with its interval, and the log-rank test, in numpy and scipy only."""

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

Z95 = 1.959963984540054


@dataclass
class KaplanMeier:
    """The step curve: at each distinct time, the share still without the event."""

    times: list[float]  # 0, then every distinct time where something happened
    survival: list[float]
    lower: list[float]  # 95 % interval, log-log transform (stays within 0 and 1)
    upper: list[float]
    at_risk: list[int]  # still followed just before each time
    events: list[int]  # events at each time
    n: int
    total_events: int
    median: float | None  # first time the curve reaches 0.5 or below; None if it never does
    median_lower: float | None
    median_upper: float | None  # None: the interval stays open (the curve stays above 0.5)


def _first_at_or_below(times: np.ndarray, curve: np.ndarray, level: float = 0.5) -> float | None:
    hit = np.flatnonzero(curve <= level + 1e-12)
    return float(times[hit[0]]) if hit.size else None


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> KaplanMeier:
    """`events`: True where the event happened, False where the row was censored (still followed
    when the data ends). Ties: events count before censorings at the same time (the usual rule)."""
    durations = np.asarray(durations, dtype=float)
    happened = np.asarray(events, dtype=bool)
    n = len(durations)
    distinct = np.unique(durations)
    # at risk just before t: every row whose duration is >= t
    order = np.sort(durations)
    at_risk = n - np.searchsorted(order, distinct, side="left")
    dead = np.bincount(np.searchsorted(distinct, durations[happened]), minlength=len(distinct))
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(at_risk > 0, 1 - dead / at_risk, 1.0)
        survival = np.cumprod(factor)
        # Greenwood: var(log S) = sum d / (n (n - d)); infinite once S reaches 0
        terms = np.where((at_risk - dead) > 0, dead / (at_risk * (at_risk - dead)), np.inf)
        terms = np.where(dead == 0, 0.0, terms)
        var_log = np.cumsum(terms)
        log_s = np.log(survival)
        # log-log interval: S ** exp(± z se / log S), se = sqrt(var_log)
        se = np.sqrt(var_log) / np.abs(log_s)
        lower = np.exp(-np.exp(np.log(-log_s) + Z95 * se))
        upper = np.exp(-np.exp(np.log(-log_s) - Z95 * se))
    lower = np.where(survival >= 1, 1.0, np.where(survival <= 0, 0.0, lower))
    upper = np.where(survival >= 1, 1.0, np.where(survival <= 0, 0.0, upper))
    lower, upper = np.nan_to_num(lower, nan=0.0), np.nan_to_num(upper, nan=1.0)
    times = np.concatenate([[0.0], distinct]) if distinct.size == 0 or distinct[0] > 0 else distinct
    pad = len(times) - len(distinct)
    survival_all = np.concatenate([[1.0] * pad, survival])
    lower_all = np.concatenate([[1.0] * pad, lower])
    upper_all = np.concatenate([[1.0] * pad, upper])
    at_risk_all = np.concatenate([[n] * pad, at_risk]).astype(int)
    events_all = np.concatenate([[0] * pad, dead]).astype(int)
    return KaplanMeier(
        times=[float(t) for t in times],
        survival=[float(s) for s in survival_all],
        lower=[float(s) for s in lower_all],
        upper=[float(s) for s in upper_all],
        at_risk=[int(r) for r in at_risk_all],
        events=[int(d) for d in events_all],
        n=n,
        total_events=int(happened.sum()),
        median=_first_at_or_below(times, survival_all),
        # R's rule: the median's interval runs from where the lower band crosses 0.5 to where
        # the upper band does
        median_lower=_first_at_or_below(times, lower_all),
        median_upper=_first_at_or_below(times, upper_all),
    )


def survival_at(curve: KaplanMeier, time: float) -> float:
    """The curve read at a time (a step function: the last value at or before it)."""
    index = int(np.searchsorted(np.asarray(curve.times), time, side="right")) - 1
    return curve.survival[max(index, 0)]


@dataclass
class LogRank:
    statistic: float
    df: int
    p_value: float
    observed: list[int]  # events per group
    expected: list[float]  # events expected per group if every group had the same curve


def log_rank(durations: np.ndarray, events: np.ndarray, groups: np.ndarray) -> LogRank | None:
    """Mantel-Haenszel log-rank test of K groups: (O - E)' V^-1 (O - E) on K - 1 groups, a
    chi-square with K - 1 degrees of freedom. None with fewer than 2 groups or no event."""
    durations = np.asarray(durations, dtype=float)
    happened = np.asarray(events, dtype=bool)
    labels, index = np.unique(np.asarray(groups), return_inverse=True)
    k = len(labels)
    if k < 2 or not happened.any():
        return None
    times = np.unique(durations[happened])
    at_risk = np.zeros((len(times), k))
    dead = np.zeros((len(times), k))
    for g in range(k):
        own = np.sort(durations[index == g])
        at_risk[:, g] = len(own) - np.searchsorted(own, times, side="left")
        own_dead = durations[(index == g) & happened]
        dead[:, g] = np.bincount(np.searchsorted(times, own_dead), minlength=len(times))
    n = at_risk.sum(axis=1)
    d = dead.sum(axis=1)
    expected = at_risk * (d / n)[:, None]
    observed_minus = (dead - expected).sum(axis=0)
    # hypergeometric variance-covariance, summed over the event times
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(n > 1, d * (n - d) / (n**2 * (n - 1)), 0.0)
    v = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            share = at_risk[:, i] * ((i == j) * n - at_risk[:, j])
            v[i, j] = float((factor * share).sum())
    head, cov = observed_minus[:-1], v[:-1, :-1]
    try:
        statistic = float(head @ np.linalg.solve(cov, head))
    except np.linalg.LinAlgError:
        statistic = float(head @ np.linalg.pinv(cov) @ head)
    df = k - 1
    p = float(stats.chi2.sf(statistic, df)) if math.isfinite(statistic) else 1.0
    return LogRank(
        statistic=statistic,
        df=df,
        p_value=p,
        observed=[int(x) for x in dead.sum(axis=0)],
        expected=[float(x) for x in expected.sum(axis=0)],
    )
