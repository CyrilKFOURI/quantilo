"""A structural model in the manner of Prophet (028 A6), without its library (CmdStan).

y = piecewise-linear trend + Fourier seasons + drivers, fitted like Prophet's MAP: the series
divided by max |y|, time from 0 to 1 over the history, 25 candidate changepoints over the first
80 % of the history whose slope changes get a Laplace prior (L1, scale 0.05), seasonal terms a
weak normal prior (scale 10), drivers none. Its own intervals simulate Gaussian noise plus new
changepoints in the future, as Prophet does (Taylor & Letham, 2018); like Prophet's they ignore
the drift of the level, so the final ones are conformal (`approximate`, see `fc.run_model`).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from quantilo.timeseries import forecast as fc
from quantilo.timeseries import series as ts

CHANGEPOINTS = 25  # candidate changepoints (Prophet's n_changepoints)
CHANGEPOINT_RANGE = 0.8  # share of the history they are spread over
CHANGEPOINT_SCALE = 0.05  # Laplace prior scale of a slope change (changepoint_prior_scale)
SEASONALITY_SCALE = 10.0  # normal prior scale of a Fourier coefficient
CUSTOM_ORDER = 5  # Fourier pairs of a cycle chosen by hand
SIMULATIONS = 500  # trajectories behind the intervals
REFITS = 3  # times the noise level sigma (and so the L1 penalty sigma² / scale) is re-estimated


@dataclass(frozen=True)
class Cycle:
    """A seasonal cycle as Fourier terms: `length` in periods, `order` sine / cosine pairs."""

    name: str
    length: float
    order: int


# The cycles each period can have, with Prophet's default orders (daily 4, weekly 3, yearly 10).
CYCLES: dict[ts.Period, tuple[Cycle, ...]] = {
    "hour": (Cycle("daily", 24, 4), Cycle("weekly", 168, 3), Cycle("yearly", 8766, 10)),
    "day": (Cycle("weekly", 7, 3), Cycle("yearly", 365.25, 10)),
    "week": (Cycle("yearly", 52.18, 10),),
    "month": (Cycle("yearly", 12, 6),),
    "quarter": (Cycle("yearly", 4, 2),),
    "year": (),
}
# The usual season of each period: a model given only the season infers the period from it.
_PERIOD_OF_SEASON: dict[int, ts.Period] = {length: period for period, length in ts.SEASON.items()}


def cycles(period: ts.Period | None, season: int | None, n: int) -> tuple[list[Cycle], list[str]]:
    """The cycles a series of `n` periods supports, and why the others are left out. Rule: a
    cycle the period allows is used when the history covers it at least twice (Prophet's rule for
    the yearly season); its order is capped at half its length. A season set by hand that is
    none of them is added. Without the period, it is inferred from the usual season."""
    if period is None and season is not None:
        period = _PERIOD_OF_SEASON.get(season)
    allowed = list(CYCLES.get(period, ())) if period else []
    if season and season >= 2 and not any(abs(c.length - season) < 0.5 for c in allowed):
        allowed.append(Cycle(f"cycle of {season}", season, CUSTOM_ORDER))
    used, skipped = [], []
    for cycle in allowed:
        if n >= 2 * cycle.length:
            used.append(
                Cycle(cycle.name, cycle.length, max(1, min(cycle.order, int(cycle.length // 2))))
            )
        else:
            skipped.append(f"{cycle.name} needs 2 cycles")
    return used, skipped


def fourier(t: np.ndarray, cycle: Cycle) -> np.ndarray:
    """sin and cos of 2πkt / length for k = 1 … order (t = position from the series' start)."""
    angles = 2 * np.pi * np.outer(t, np.arange(1, cycle.order + 1)) / cycle.length
    return np.column_stack([np.sin(angles), np.cos(angles)])


def changepoint_positions(n: int, candidates: int = CHANGEPOINTS) -> np.ndarray:
    """Prophet's candidates: evenly spaced over the first 80 % of the history (fewer when the
    series is short: at most one per period, never the first one)."""
    span = math.floor(n * CHANGEPOINT_RANGE)
    count = min(candidates, span - 1)
    if count < 1:
        return np.empty(0, dtype=int)
    return np.unique(np.round(np.linspace(0, span - 1, count + 1)).astype(int)[1:])


def _lasso(gram: np.ndarray, corr: np.ndarray, penalty: float) -> np.ndarray:
    """min ½ b'Gb - c'b + penalty·|b|₁ by coordinate descent on the Gram matrix (few columns)."""
    beta = np.zeros(len(corr))
    diagonal = np.diag(gram).copy()
    for _ in range(1_000):
        moved = 0.0
        for j in range(len(corr)):
            if diagonal[j] <= 1e-12:
                continue
            rho = corr[j] - gram[j] @ beta + diagonal[j] * beta[j]
            new = math.copysign(max(abs(rho) - penalty, 0.0), rho) / diagonal[j]
            moved = max(moved, abs(new - beta[j]))
            beta[j] = new
        if moved <= 1e-9 * max(1.0, float(np.max(np.abs(beta)))):
            break
    return beta


@dataclass
class _Fit:
    free: np.ndarray  # intercept, slope, Fourier, drivers
    deltas: np.ndarray  # slope changes at the candidates
    sigma: float  # noise of a new value, on the scaled series


def _fit(
    y: np.ndarray,
    free: np.ndarray,
    hinges: np.ndarray,
    ridge: np.ndarray,
    prior: float = CHANGEPOINT_SCALE,
) -> _Fit:
    """MAP of y = free·a + hinges·δ with a normal prior on some `free` columns (`ridge` holds
    1 / prior scale², 0 = none) and a Laplace prior on δ, sigma re-estimated REFITS times. The free
    block is profiled out exactly (Frisch-Waugh on the data stacked with the prior rows)."""
    n, k = free.shape
    a = np.linalg.lstsq(free, y, rcond=None)[0]
    deltas = np.zeros(hinges.shape[1])
    sigma2 = float(np.mean((y - free @ a) ** 2)) or 1e-12
    for _ in range(REFITS if hinges.shape[1] else 1):
        weights = np.sqrt(ridge * sigma2)
        stacked = np.vstack([free, np.diag(weights)[weights > 0]])
        target = np.concatenate([y, np.zeros(stacked.shape[0] - n)])
        bends = np.vstack([hinges, np.zeros((stacked.shape[0] - n, hinges.shape[1]))])
        u, s, _ = np.linalg.svd(stacked, full_matrices=False)
        basis = u[:, s > s.max() * 1e-10]
        rest_y = target - basis @ (basis.T @ target)
        rest_b = bends - basis @ (basis.T @ bends)
        if hinges.shape[1]:
            deltas = _lasso(rest_b.T @ rest_b, rest_b.T @ rest_y, sigma2 / prior)
        a = np.linalg.lstsq(stacked, target - bends @ deltas, rcond=None)[0]
        sigma2 = float(np.mean((y - free @ a - hinges @ deltas) ** 2)) or 1e-12
    # Noise of a new value: the residual variance corrected for the parameters fitted (RSS over
    # n - p), then for the error of the fitted curve itself (about p / n of it, on average).
    used = min(n - 1, k + int(np.sum(np.abs(deltas) > 1e-6)))
    noise = sigma2 * n / (n - used) * (1 + used / n)
    return _Fit(a, deltas, math.sqrt(noise))


def _future_bends(
    t_future: np.ndarray, rate: float, scale: float, rng: np.random.Generator
) -> np.ndarray:
    """SIMULATIONS trajectories of the trend changes after the history (t > 1): changepoints
    arrive at `rate` per unit of time, their slope change drawn from Laplace(0, scale)."""
    paths = np.zeros((SIMULATIONS, len(t_future)))
    end = float(t_future[-1]) if len(t_future) else 1.0
    if end <= 1 or rate <= 0 or scale <= 0:
        return paths
    for path in paths:
        count = rng.poisson(rate * (end - 1))
        if not count:
            continue
        at = 1 + rng.random(count) * (end - 1)
        changes = rng.laplace(0, scale, count)
        path += (np.maximum(t_future[:, None] - at[None, :], 0) * changes).sum(axis=1)
    return paths


def model(
    period: ts.Period | None = None,
    changepoints: int = CHANGEPOINTS,
    changepoint_scale: float = CHANGEPOINT_SCALE,
) -> Callable[..., fc.Prediction]:
    """The Prophet-like model for a period (None: inferred from the season). Drivers (028 A4)
    are extra columns without any prior: same contract as the linear regression; public holidays
    (028 A5) come as one column per holiday, so each gets its own effect, as in Prophet (whose
    prior of scale 10 on them is almost flat on the scaled series). `changepoints` candidates and
    their Laplace prior scale are Prophet's `n_changepoints` and `changepoint_prior_scale`
    (028 A7). Its components (trend, each season, each driver column) come with the forecast."""

    def fit(
        y: np.ndarray,
        h: int,
        season: int | None,
        x: np.ndarray | None = None,
        x_future: np.ndarray | None = None,
    ) -> fc.Prediction:
        n = len(y)
        if n < 10:
            raise fc.ModelError("Needs at least 10 periods.")
        used, skipped = cycles(period, season, n)
        scale = float(np.max(np.abs(y))) or 1.0
        position = np.arange(n + h, dtype=float)
        t = position / (n - 1)
        points = changepoint_positions(n, changepoints)
        hinges = np.maximum(t[:, None] - t[points][None, :], 0)
        seasonal = [fourier(position, cycle) for cycle in used]
        k = 0 if x is None or x_future is None else x.shape[1]
        drivers = (
            np.vstack([x, x_future[:h]]) if k and x is not None and x_future is not None else None
        )
        parts = [np.ones((n + h, 1)), t[:, None], *seasonal]
        if drivers is not None:
            parts.append(drivers)
        free = np.column_stack(parts)
        widths = [2, *(block.shape[1] for block in seasonal), k]
        ridge = np.concatenate(
            [np.zeros(widths[0]), np.full(sum(widths[1:-1]), 1 / SEASONALITY_SCALE**2), np.zeros(k)]
        )
        found = _fit(y[:n] / scale, free[:n], hinges[:n], ridge, changepoint_scale)
        mean = free[n:] @ found.free + hinges[n:] @ found.deltas
        fitted = free[:n] @ found.free + hinges[:n] @ found.deltas
        rng = np.random.default_rng(0)
        bends = _future_bends(
            t[n:],
            len(points),
            float(np.mean(np.abs(found.deltas))) + 1e-8 if len(points) else 0.0,
            rng,
        )
        paths = mean[None, :] + bends + rng.normal(0, found.sigma, (SIMULATIONS, h))
        low80, high80, low95, high95 = np.quantile(paths, [0.1, 0.9, 0.025, 0.975], axis=0)
        kept = int(np.sum(np.abs(found.deltas) > 1e-6))
        seasons = " + ".join(f"{c.name} (K={c.order})" for c in used)
        label = f"Piecewise trend ({kept} changepoint{'' if kept == 1 else 's'})"
        label += f" + {seasons}" if seasons else " (no season)"
        if skipped:
            label += f" · {', '.join(skipped)}"
        prediction = fc.Prediction(
            mean * scale,
            low80 * scale,
            high80 * scale,
            low95 * scale,
            high95 * scale,
            label,
            (y - fitted * scale),
        )
        prediction.approximate = True  # its own intervals covered ~45-75 % on rolling origins
        if k:
            prediction.coef = found.free[-k:] * scale
            prediction.exog_effects = free[:, -k:] * found.free[-k:] * scale
        trend = (free[:, :2] @ found.free[:2] + hinges @ found.deltas) * scale
        prediction.components = [fc.Component("Trend", "trend", [float(v) for v in trend])]
        at = 2
        for cycle, block in zip(used, seasonal, strict=True):
            width = block.shape[1]
            values = block @ found.free[at : at + width] * scale
            prediction.components.append(
                fc.Component(cycle.name, "season", [float(v) for v in values])
            )
            at += width
        return prediction

    return fit
