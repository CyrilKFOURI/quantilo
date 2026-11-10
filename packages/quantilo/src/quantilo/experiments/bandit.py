"""The bandit view of an experiment (spec 049): which variant is probably best, what shipping it
would cost if it is not, and what a Thompson-sampling bandit would have done with the traffic.

- `view()`: Bayesian posteriors of each variant's value (Beta for a rate, Normal for a mean or a
  count), P(each variant is best) and the expected loss of shipping it, by Monte Carlo with a fixed
  seed; with a date, the replay of a batched Thompson sampling over the randomized test.
- `simulate()`: the Plan section's "fixed split vs bandit", from the planned effect and traffic.

It complements the frequentist test, it does not replace it: after an adaptive allocation, the
classical intervals are biased (Hadad et al., PNAS 2021). Pure numpy, seeded: same numbers on
every call, in the app, the library and the browser version.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from quantilo.experiments.plan import SampleSizeRequest, sample_size

BanditKind = Literal["rate", "mean", "count"]

DRAWS = 20_000  # P(best) within ±0.4 point
SEED = 49
REPLAY_DRAWS = 2_000  # per period of the replay
SIM_RUNS = 200
SIM_DRAWS = 200  # per period and run of the simulation
SIM_FINAL_DRAWS = 1_000
SIM_BATCHES = 20  # without a daily traffic
SIM_MAX_PERIODS = 60
SURE = 0.95
LABELS = ("control", "B", "C", "D", "E")
GUARDRAIL = (
    "The bandit view complements the A/B test above, it does not replace it. After an adaptive "
    "allocation, classical confidence intervals and p-values are biased: run a bandit to earn, "
    "a fixed split to learn."
)
REPLAY_NOTE = (
    "Valid only because the test was randomized: in each period, a variant's mean estimates what "
    "it would have given to all the traffic of that period. The bandit updates once per period "
    "(Thompson sampling, the first period at the planned split)."
)


# --- What the view returns ----------------------------------------------------------------------


@dataclass
class BanditArm:
    variant: str
    control: bool
    units: int
    mean: float  # posterior mean
    low: float  # 95 % credible interval
    high: float
    p_best: float
    expected_loss: float  # E[best − this], in the metric's unit (a share for a rate)
    sentence: str


@dataclass
class BanditShare:
    variant: str
    shares: list[float]  # the bandit's share of each period's traffic


@dataclass
class BanditReplay:
    column: str
    grain: str  # day, week, month
    periods: list[str]
    units: list[int]  # per period
    shares: list[BanditShare]
    fixed: list[float]  # the planned split
    winner: str  # best overall mean: the policy "ship it from day one" regret is measured against
    value_fixed: float  # the metric summed over the traffic, fixed split
    value_bandit: float
    gained: float  # bandit − fixed
    gained_relative: float | None
    regret_fixed: list[float]  # cumulative, against shipping the winner from the first period
    regret_bandit: list[float]
    sentence: str
    note: str


@dataclass
class ExperimentBandit:
    metric: str
    kind: BanditKind
    prior: str
    draws: int
    seed: int
    arms: list[BanditArm]
    leader: str  # highest P(best)
    sentence: str
    replay: BanditReplay | None
    replay_reason: str | None
    guardrail: str


@dataclass
class Timeline:
    """The units that have a date, per variant (the control first): their values and the index
    of their period."""

    column: str
    grain: str
    periods: list[str]
    values: list[np.ndarray]
    index: list[np.ndarray]


# --- Words --------------------------------------------------------------------------------------


def _prob(p: float) -> str:
    if p >= 0.9995:
        return "> 99.9 %"
    if p < 0.0005:
        return "< 0.1 %"
    shown = p * 100
    return f"{shown:.0f} %" if 1 <= shown <= 99 else f"{shown:.1f} %"


def _compared(text: str) -> str:
    """ "= 94 %", or "> 99.9 %" without an equals sign."""
    return text if text[0] in "<>" else f"= {text}"


def _figure(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.2g}"


def loss_text(kind: str, loss: float) -> str:
    """0.1 pt for a rate (percentage points), the metric's unit otherwise."""
    shown = loss * 100 if kind == "rate" else loss
    unit = " pt" if kind == "rate" else ""
    if shown < 0.005:
        return f"< 0.01{unit}"
    return f"{_figure(shown)}{unit}"


def _signed(value: float) -> str:
    text = _figure(abs(value))
    return f"+{text}" if value >= 0 else f"−{text}"


def _amount(kind: str, metric: str, value: float) -> str:
    """Conversions for a rate, the metric summed otherwise: +38 converted · −1,204 revenue."""
    if kind == "rate":
        return f"{'+' if value >= 0 else '−'}{abs(value):,.0f} {metric}"
    return f"{_signed(value)} {metric}"


def _percent(value: float) -> str:
    """Rounded half up, as the page does (Math.round)."""
    return f"{math.floor(value * 100 + 0.5)} %"


# --- Posteriors ---------------------------------------------------------------------------------


def _draws(
    rng: np.random.Generator,
    kind: str,
    n: np.ndarray,
    total: np.ndarray,
    squares: np.ndarray,
    size: int,
) -> np.ndarray:
    """`size` draws of each variant's value: (size, k). Beta(1 + s, 1 + n − s) for a rate; for a
    mean, Normal(mean, s² / n) with the pooled spread when a variant has fewer than 2 units."""
    k = len(n)
    if kind == "rate":
        return rng.beta(1 + total, 1 + np.maximum(n - total, 0), size=(size, k))
    all_n = n.sum()
    pooled_mean = total.sum() / all_n if all_n else 0.0
    pooled_var = (squares.sum() - all_n * pooled_mean**2) / (all_n - 1) if all_n > 1 else 1.0
    pooled_sd = math.sqrt(max(pooled_var, 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(n > 0, total / np.maximum(n, 1), pooled_mean)
        var = np.where(n > 1, (squares - n * mean**2) / np.maximum(n - 1, 1), pooled_sd**2)
    # a constant variant keeps a sliver of spread, so that ties are broken at random
    se = np.maximum(np.sqrt(np.maximum(var, 0) / np.maximum(n, 1)), 1e-9 * (1 + np.abs(mean)))
    return mean + se * rng.standard_normal((size, k))


def _credible(
    kind: str, n: np.ndarray, total: np.ndarray, squares: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each posterior's mean and 95 % credible interval, exactly."""
    if kind == "rate":
        a, b = 1 + total, 1 + np.maximum(n - total, 0)
        return a / (a + b), stats.beta.ppf(0.025, a, b), stats.beta.ppf(0.975, a, b)
    mean = total / n
    se = np.sqrt(np.maximum((squares - n * mean**2) / (n - 1), 0) / n)
    z = float(stats.norm.ppf(0.975))
    return mean, mean - z * se, mean + z * se


def best_and_loss(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P(each column is the highest) and E[max − column] over the draws."""
    k = theta.shape[1]
    best = np.bincount(np.argmax(theta, axis=1), minlength=k) / theta.shape[0]
    loss = (theta.max(axis=1, keepdims=True) - theta).mean(axis=0)
    return best, loss


def _sums(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.array([len(v) for v in values], float)
    total = np.array([float(v.sum()) for v in values])
    squares = np.array([float((v * v).sum()) for v in values])
    return n, total, squares


# --- The view -----------------------------------------------------------------------------------


def view(
    metric: str,
    kind: BanditKind,
    labels: list[str],
    values: list[np.ndarray],
    planned: list[float],
    timeline: Timeline | None,
) -> ExperimentBandit:
    """`labels` and `values` (one value per unit) start with the control; `planned` is the
    planned split; `timeline`, when a date is chosen, the period of each unit."""
    rng = np.random.default_rng(SEED)
    n, total, squares = _sums(values)
    theta = _draws(rng, kind, n, total, squares, DRAWS)
    p_best, loss = best_and_loss(theta)
    mean, low, high = _credible(kind, n, total, squares)
    arms = []
    for i, label in enumerate(labels):
        sentence = (
            f"P({label} is best) {_compared(_prob(float(p_best[i])))}, expected loss if you "
            "ship it: "
            f"{loss_text(kind, float(loss[i]))}."
        )
        arms.append(
            BanditArm(
                label,
                i == 0,
                int(n[i]),
                float(mean[i]),
                float(low[i]),
                float(high[i]),
                float(p_best[i]),
                float(loss[i]),
                sentence,
            )
        )
    leader = arms[int(np.argmax(p_best))]
    prior = (
        "Beta(1, 1): a uniform prior on each rate"
        if kind == "rate"
        else "Normal(mean, s² / n): a flat prior, the large-sample posterior of each mean"
    )
    replay, reason = None, None
    if timeline is None:
        reason = (
            "Choose a date under More ▸ Over time to replay what a bandit would have done, "
            "period by period."
        )
    elif len(timeline.periods) < 2:
        reason = "The test lasts one period: there is nothing to replay."
    elif any(len(v) == 0 for v in timeline.values):
        reason = "A variant has no unit with a date: there is nothing to replay."
    else:
        replay = _replay(metric, kind, labels, planned, timeline)
    return ExperimentBandit(
        metric,
        kind,
        prior,
        DRAWS,
        SEED,
        arms,
        leader.variant,
        leader.sentence,
        replay,
        reason,
        GUARDRAIL,
    )


def _replay(
    metric: str,
    kind: BanditKind,
    labels: list[str],
    planned: list[float],
    timeline: Timeline,
) -> BanditReplay:
    """A Thompson sampling updated once per period: its share of period t is P(best) given the
    units of the periods before t; each variant is valued at its mean in that period."""
    periods, k = len(timeline.periods), len(labels)
    n = np.zeros((periods, k))
    total = np.zeros((periods, k))
    squares = np.zeros((periods, k))
    for j, (v, index) in enumerate(zip(timeline.values, timeline.index, strict=True)):
        where = index.astype(int)
        np.add.at(n[:, j], where, 1.0)
        np.add.at(total[:, j], where, v)
        np.add.at(squares[:, j], where, v * v)
    before_n = np.cumsum(n, axis=0) - n
    before_total = np.cumsum(total, axis=0) - total
    before_squares = np.cumsum(squares, axis=0) - squares
    overall = total.sum(axis=0) / np.maximum(n.sum(axis=0), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        so_far = np.where(before_n > 0, before_total / before_n, overall)
        means = np.where(n > 0, total / n, so_far)

    rng = np.random.default_rng(SEED)
    fixed = np.asarray(planned, float)
    shares = np.empty((periods, k))
    shares[0] = fixed
    for t in range(1, periods):
        theta = _draws(rng, kind, before_n[t], before_total[t], before_squares[t], REPLAY_DRAWS)
        shares[t] = best_and_loss(theta)[0]

    traffic = n.sum(axis=1)
    winner = int(np.argmax(overall))
    bandit_values = traffic * (shares * means).sum(axis=1)
    fixed_values = traffic * (means @ fixed)
    oracle = traffic * means[:, winner]
    regret_bandit = np.cumsum(oracle - bandit_values)
    regret_fixed = np.cumsum(oracle - fixed_values)
    value_bandit, value_fixed = float(bandit_values.sum()), float(fixed_values.sum())
    gained = value_bandit - value_fixed
    relative = gained / abs(value_fixed) if value_fixed else None

    best_share = float((shares[:, winner] * traffic).sum() / traffic.sum())
    word = {"day": "days", "week": "weeks", "month": "months"}.get(timeline.grain, "periods")
    verb = "gained" if gained >= 0 else "lost"
    sentence = (
        f"Over {periods} {word}, a bandit would have sent {_percent(best_share)} of the units to "
        f"{labels[winner]} ({_percent(float(fixed[winner]))} with the fixed split) and {verb} "
        f"about {_amount(kind, metric, gained).lstrip('+−')}"
        + (f" ({'+' if gained >= 0 else '−'}{abs(relative) * 100:.1f} %)" if relative else "")
        + f". Cumulative regret against shipping {labels[winner]} from the first {timeline.grain}: "
        f"{_figure(float(regret_bandit[-1]))} for the bandit, "
        f"{_figure(float(regret_fixed[-1]))} for the fixed split."
    )
    return BanditReplay(
        timeline.column,
        timeline.grain,
        timeline.periods,
        [int(t) for t in traffic],
        [BanditShare(label, [float(s) for s in shares[:, j]]) for j, label in enumerate(labels)],
        [float(s) for s in fixed],
        labels[winner],
        value_fixed,
        value_bandit,
        gained,
        relative,
        [float(r) for r in regret_fixed],
        [float(r) for r in regret_bandit],
        sentence,
        REPLAY_NOTE,
    )


# --- Plan: fixed split vs bandit ----------------------------------------------------------------


@dataclass
class BanditSimulation:
    kind: str
    baseline: float
    target: float
    labels: list[str]  # the control first; B carries the planned effect
    values: list[float]  # each variant's true value in the simulation
    best: str
    units: int  # the horizon: the fixed design's total sample size
    grain: Literal["day", "batch"]
    per_period: list[int]
    runs: int
    seed: int
    cumulative_units: list[int]  # at the end of each period (the chart's x)
    regret_fixed: list[float]  # cumulative, expected, against always sending the best variant
    regret_bandit: list[float]  # mean over the runs
    regret_bandit_low: list[float]  # 10th percentile
    regret_bandit_high: list[float]  # 90th percentile
    gained: float  # mean, in conversions (rate) or the metric summed (mean)
    gained_low: float
    gained_high: float
    best_share_fixed: float
    best_share_bandit: float  # mean share of the traffic sent to the best variant
    sure_fixed: float  # runs ending with P(best) ≥ 95 % on the right variant
    sure_bandit: float
    sentence: str
    guardrail: str
    assumptions: str


def _traffic(total: int, daily: int | None) -> tuple[list[int], Literal["day", "batch"]]:
    if daily:
        days = math.ceil(total / daily)
        per = [daily] * (days - 1) + [total - daily * (days - 1)]
        if days <= SIM_MAX_PERIODS:
            return per, "day"
        count = SIM_MAX_PERIODS
    else:
        count = SIM_BATCHES
    edges = np.linspace(0, total, count + 1).round().astype(int)
    return [int(width) for width in np.diff(edges)], "batch"


def _posterior(
    rng: np.random.Generator, kind: str, n: np.ndarray, s: np.ndarray, sd: float, size: int
) -> np.ndarray:
    """Draws (runs, size, k) of each variant's value; a rate's Beta approximated by a Normal."""
    if kind == "rate":
        mean = (s + 1) / (n + 2)
        se = np.sqrt(mean * (1 - mean) / (n + 3))
    else:
        mean = np.where(n > 0, s / np.maximum(n, 1), 0.0)
        se = np.where(n > 0, sd / np.sqrt(np.maximum(n, 1)), 1e6 * sd)
    noise = rng.standard_normal((n.shape[0], size, n.shape[1]))
    return mean[:, None, :] + se[:, None, :] * noise


def _p_best(theta: np.ndarray) -> np.ndarray:
    k = theta.shape[2]
    winners = np.argmax(theta, axis=2)
    return np.stack([(winners == j).mean(axis=1) for j in range(k)], axis=1)


def _allocate(shares: np.ndarray, units: int) -> np.ndarray:
    counts = np.floor(shares * units).astype(int)
    top = np.argmax(shares, axis=1)
    counts[np.arange(len(counts)), top] += units - counts.sum(axis=1)
    return counts


def simulate(request: SampleSizeRequest) -> BanditSimulation:
    """Run the planned test SIM_RUNS times with a fixed split and with a batched Thompson
    sampling, on the planned traffic and duration; one variant (B) has the planned effect."""
    plan = sample_size(request)
    k = request.variants
    labels: list[str] = list(LABELS[:k])
    truth = np.full(k, request.baseline, float)
    truth[1] = plan.target
    best = int(np.argmax(truth))
    sd = (
        request.sd
        if request.kind == "mean" and request.sd
        else math.sqrt(request.baseline * (1 - request.baseline))
    )
    per_period, grain = _traffic(plan.total, request.daily)
    weights = np.array([1.0] + [request.ratio] * (k - 1))
    fixed = weights / weights.sum()

    rng = np.random.default_rng(SEED)
    runs = SIM_RUNS
    n_bandit = np.zeros((runs, k))
    s_bandit = np.zeros((runs, k))
    n_fixed = np.zeros((runs, k))
    s_fixed = np.zeros((runs, k))
    gap = truth[best] - truth
    regret_bandit = np.zeros((runs, len(per_period)))
    regret_fixed = np.zeros(len(per_period))
    sent_best = np.zeros(runs)

    def outcomes(counts: np.ndarray) -> np.ndarray:
        if request.kind == "rate":
            return rng.binomial(counts, truth[None, :]).astype(float)
        return counts * truth[None, :] + np.sqrt(counts) * sd * rng.standard_normal(counts.shape)

    for t, units in enumerate(per_period):
        if t == 0:
            shares = np.tile(fixed, (runs, 1))
        else:
            theta = _posterior(rng, request.kind, n_bandit, s_bandit, sd, SIM_DRAWS)
            shares = _p_best(theta)
        counts = _allocate(shares, units)
        s_bandit += outcomes(counts)
        n_bandit += counts
        sent_best += counts[:, best]
        regret_bandit[:, t] = counts @ gap
        fixed_counts = _allocate(np.tile(fixed, (runs, 1)), units)
        s_fixed += outcomes(fixed_counts)
        n_fixed += fixed_counts
        regret_fixed[t] = units * float(fixed @ gap)
    regret_bandit = np.cumsum(regret_bandit, axis=1)
    regret_fixed = np.cumsum(regret_fixed)
    gains = regret_fixed[-1] - regret_bandit[:, -1]

    def sure(n: np.ndarray, s: np.ndarray) -> float:
        final = _p_best(_posterior(rng, request.kind, n, s, sd, SIM_FINAL_DRAWS))
        return float((final[:, best] >= SURE).mean())

    sure_fixed, sure_bandit = sure(n_fixed, s_fixed), sure(n_bandit, s_bandit)
    best_share_bandit = float(sent_best.mean() / plan.total)
    gained, low, high = float(gains.mean()), *(float(q) for q in np.quantile(gains, [0.1, 0.9]))

    def amount(value: float) -> str:
        if request.kind == "rate":
            return f"{'+' if value >= 0 else '−'}{abs(value):,.0f}"
        return _signed(value)

    what = "conversions" if request.kind == "rate" else "on the metric's total"
    if grain == "day":
        span = f"Over {len(per_period):,} days at {request.daily:,} units a day"
    elif request.daily:
        span = (
            f"Over {plan.days:,} days at {request.daily:,} units a day "
            f"(grouped in {len(per_period)} batches)"
        )
    else:
        span = f"Over {len(per_period)} equal batches"
    sentence = (
        f"{span}, {plan.total:,} units (the size the plan needs), a bandit would send "
        f"{_percent(best_share_bandit)} of the traffic to {labels[best]} against "
        f"{_percent(float(fixed[best]))} with the fixed split, earning about "
        f"{amount(gained)} {what} (10–90 %: {amount(low)} to {amount(high)}). "
        f"At the end, the fixed split is sure of the best variant (P(best) ≥ 95 %) in "
        f"{_percent(sure_fixed)} of the runs, the bandit in {_percent(sure_bandit)}."
    )
    assumptions = (
        f"{runs} simulated tests (seed {SEED}): {labels[1]} at the planned effect, "
        + ("the other variants like the control, " if k > 2 else "")
        + "the bandit updated once per "
        + ("day" if grain == "day" else "batch")
        + ", the first one at the fixed split."
    )
    cumulative_units = [int(u) for u in np.cumsum(per_period)]
    regret_low, regret_high = np.quantile(regret_bandit, [0.1, 0.9], axis=0)
    return BanditSimulation(
        request.kind,
        request.baseline,
        plan.target,
        labels,
        [float(v) for v in truth],
        labels[best],
        plan.total,
        grain,
        per_period,
        runs,
        SEED,
        cumulative_units,
        [float(r) for r in regret_fixed],
        [float(r) for r in regret_bandit.mean(axis=0)],
        [float(r) for r in regret_low],
        [float(r) for r in regret_high],
        gained,
        low,
        high,
        float(fixed[best]),
        best_share_bandit,
        sure_fixed,
        sure_bandit,
        sentence,
        GUARDRAIL,
        assumptions,
    )
