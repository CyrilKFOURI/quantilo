"""N-HiTS and N-BEATS trained here (028 B4): small PyTorch networks, CPU, no weights to download.

Both read the last L periods (a window) and output the next h. N-BEATS (Oreshkin et al., 2020,
the generic variant) is a stack of fully connected blocks, each removing what it explains from
the window (backcast) and adding its part of the forecast. N-HiTS (Challu et al., 2023) gives
each block a pooled view of the window (coarse to fine) and predicts a few points of the
forecast, interpolated to h: fewer parameters, smoother long horizons.

Every window is standardised by its own mean and standard deviation (series of any level look
alike), the loss is the mean absolute error, the weights start from a fixed seed. Training stops
after `STEPS` batches or at the time budget, whichever comes first.

Rule (coded, with its reason): offered only for a long series (at least `MIN_PERIODS`) or for
many groups (`MIN_GROUPS`, one network trained on every group's windows, 039): with fewer windows
a network does worse than the statistical models (M4 competition).
"""

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from quantilo.timeseries import forecast as fc

MIN_PERIODS = 500
MIN_GROUPS = 10
STEPS = 400  # batches per training
BATCH = 64
HIDDEN = 128
FIT_SECONDS = 20.0  # per training, when the forecast has no time limit
MIN_WINDOWS = 50  # training windows needed at an origin
KINDS = {"nhits": "N-HiTS", "nbeats": "N-BEATS"}
THREADS = 4  # small networks: more threads only add overhead (16 threads ran ~8× slower)


@contextmanager
def threads() -> Iterator[None]:
    """At most `THREADS` CPU threads for torch while it trains or forecasts, restored after."""
    import torch  # pyright: ignore[reportMissingImports]

    before = torch.get_num_threads()
    torch.set_num_threads(min(THREADS, before))
    try:
        yield
    finally:
        torch.set_num_threads(before)


def refusal(n: int) -> str | None:
    """Why a series of n periods is not forecast by a trained network (None: it is)."""
    if n >= MIN_PERIODS:
        return None
    return (
        f"Needs at least {MIN_PERIODS} periods (this series has {n}), or {MIN_GROUPS} groups or "
        "more in Forecast every group: with fewer windows to learn from, a network does worse "
        "than the statistical models."
    )


def window(h: int, season: int | None) -> int:
    """Input length: four horizons or two cycles, between 16 and 256 periods."""
    return int(min(256, max(16, 4 * h, 2 * (season or 0))))


def _network(kind: str, length: int, h: int) -> Any:
    import torch  # pyright: ignore[reportMissingImports]
    from torch import nn  # pyright: ignore[reportMissingImports]
    from torch.nn import functional  # pyright: ignore[reportMissingImports]

    def mlp(size: int, layers: int) -> nn.Sequential:
        parts: list[nn.Module] = [nn.Linear(size, HIDDEN), nn.ReLU()]
        for _ in range(layers - 1):
            parts += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        return nn.Sequential(*parts)

    class Generic(nn.Module):
        """An N-BEATS block: fully connected layers, then a backcast and a forecast."""

        def __init__(self) -> None:
            super().__init__()
            self.body = mlp(length, 4)
            self.back = nn.Linear(HIDDEN, length)
            self.ahead = nn.Linear(HIDDEN, h)

        def forward(self, x):
            z = self.body(x)
            return self.back(z), self.ahead(z)

    class Hierarchical(nn.Module):
        """An N-HiTS block: max-pooled input, a few knots interpolated to the window and h."""

        def __init__(self, pool: int, rate: int) -> None:
            super().__init__()
            self.pool = pool
            self.body = mlp(math.ceil(length / pool), 2)
            self.back = nn.Linear(HIDDEN, max(2, math.ceil(length / rate)))
            self.ahead = nn.Linear(HIDDEN, max(2, math.ceil(h / rate)))

        def forward(self, x):
            pooled = functional.max_pool1d(
                x.unsqueeze(1), self.pool, self.pool, ceil_mode=True
            ).squeeze(1)
            z = self.body(pooled)

            def stretch(knots, size: int):
                return functional.interpolate(
                    knots.unsqueeze(1), size=size, mode="linear", align_corners=True
                ).squeeze(1)

            return stretch(self.back(z), length), stretch(self.ahead(z), h)

    class Stack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if kind == "nhits":
                sizes = [(min(8, max(1, length // 8)), min(8, max(1, h // 2))), (2, 2), (1, 1)]
                self.blocks = nn.ModuleList(Hierarchical(p, r) for p, r in sizes)
            else:
                self.blocks = nn.ModuleList(Generic() for _ in range(4))

        def forward(self, x):
            rest, total = x, torch.zeros(x.shape[0], h, dtype=x.dtype)
            for block in self.blocks:
                back, ahead = block(rest)
                rest, total = rest - back, total + ahead
            return total

    return Stack()


def _windows(series: list[np.ndarray], length: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Every (input, target) pair of every series, standardised by the input's mean and spread."""
    inputs, targets = [], []
    for y in series:
        for end in range(length, len(y) - h + 1):
            inputs.append(y[end - length : end])
            targets.append(y[end : end + h])
    x, t = np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)
    return x, t


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=1, keepdims=True)
    spread = x.std(axis=1, keepdims=True)
    return mean, np.where(spread > 1e-8, spread, 1.0)


def train(
    kind: str, series: list[np.ndarray], h: int, season: int | None, budget: float
) -> tuple[Any, int, np.ndarray]:
    """A network fitted on the windows of `series`; returns it, its window length and the
    in-sample RMSE at each step ahead (validation intervals)."""
    longest = max(len(y) for y in series)
    length = min(window(h, season), max(8, longest - h - MIN_WINDOWS))
    x, target = _windows(series, length, h)
    if len(x) < MIN_WINDOWS:
        raise fc.ModelError(
            f"Needs at least {length + h + MIN_WINDOWS} periods to learn from its windows."
        )
    with threads():
        return _fit(kind, x, target, length, h, budget)


def _fit(
    kind: str, x: np.ndarray, target: np.ndarray, length: int, h: int, budget: float
) -> tuple[Any, int, np.ndarray]:
    import torch  # pyright: ignore[reportMissingImports]

    mean, spread = _standardise(x)
    inputs = torch.from_numpy((x - mean) / spread)
    outputs = torch.from_numpy((target - mean) / spread)
    torch.manual_seed(0)
    network = _network(kind, length, h)
    optimiser = torch.optim.Adam(network.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(0)
    started = time.monotonic()
    network.train()
    for _ in range(STEPS):
        pick = torch.randint(0, len(inputs), (min(BATCH, len(inputs)),), generator=generator)
        loss = torch.mean(torch.abs(network(inputs[pick]) - outputs[pick]))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if time.monotonic() - started > budget:
            break
    network.eval()
    with torch.inference_mode():
        last = slice(max(0, len(inputs) - 200), len(inputs))
        fitted = network(inputs[last]).numpy() * spread[last] + mean[last]
    rmse = np.sqrt(np.mean((fitted - target[last]) ** 2, axis=0))
    return network, length, rmse


def predict(network: Any, length: int, contexts: list[np.ndarray]) -> np.ndarray:
    """(series, h) forecasts from the last `length` periods of each context."""
    import torch  # pyright: ignore[reportMissingImports]

    x = np.array([np.asarray(c, dtype=np.float32)[-length:] for c in contexts])
    mean, spread = _standardise(x)
    with threads(), torch.inference_mode():
        out = network(torch.from_numpy((x - mean) / spread)).numpy()
    return out * spread + mean


def _require_torch() -> None:
    from quantilo.timeseries import ai

    missing = ai.missing_packages("nhits")
    if ai.in_browser() or missing:
        raise fc.Unavailable(ai.status("nhits").reason or "Needs PyTorch.")


def budget(tuning: fc.Tuning) -> float:
    """Seconds per training: the time limit (028 A7) shared by the origins and the final fit."""
    if tuning.time_limit is None:
        return FIT_SECONDS
    return max(1.0, min(FIT_SECONDS, tuning.time_limit / 6))


def model(kind: str, tuning: fc.Tuning = fc.DEFAULT_TUNING) -> fc.Model:
    """N-HiTS or N-BEATS as a forecast model: trained at each rolling origin on what came before,
    then on the whole series. Its intervals (in-sample errors) only serve the validation: the
    final ones are conformal (`run_model`)."""
    title = KINDS[kind]

    def fit(y: np.ndarray, h: int, season: int | None) -> fc.Prediction:
        _require_torch()
        network, length, rmse = train(kind, [np.asarray(y, dtype=float)], h, season, budget(tuning))
        mean = predict(network, length, [y])[0].astype(float)
        prediction = fc._band(mean, rmse.astype(float), f"{title} on {length}-period windows")
        prediction.approximate = True
        return prediction

    fit.refusal = refusal  # type: ignore[attr-defined]
    return fit


def forecast_many(kind: str, values: np.ndarray, h: int, season: int | None) -> list[Any]:
    """One network trained on every series' windows (many groups, 039), retrained at each
    rolling origin; the same shape as the global model's result."""
    from quantilo.timeseries import global_model as gm

    _require_torch()
    count, n = values.shape
    runs: list[list[fc.OriginRun]] = [[] for _ in range(count)]
    nan = np.full(h, np.nan)
    for size in fc.origins(n, h, season):
        steps = min(h, n - size)
        try:
            network, length, _ = train(kind, list(values[:, :size]), steps, season, FIT_SECONDS)
        except fc.ModelError:
            continue
        found = predict(network, length, list(values[:, :size]))
        for i in range(count):
            runs[i].append(
                fc.OriginRun(
                    size,
                    fc.Prediction(found[i], nan[:steps], nan[:steps], nan[:steps], nan[:steps]),
                )
            )
    network, length, _ = train(kind, list(values), h, season, FIT_SECONDS)
    final = predict(network, length, list(values))
    out = []
    for i in range(count):
        y = values[i]
        try:
            scores = fc.score(y, runs[i], h, season) if runs[i] else None
        except fc.ModelError:
            scores = None
        mean = final[i].astype(float)
        bands = fc.conformal(y, runs[i], mean) or (mean, mean, mean, mean)
        out.append(
            gm.GlobalSeries(
                scores,
                [float(v) for v in mean],
                *[[float(v) for v in band] for band in bands],
                backtests=[
                    fc.Backtest(
                        run.size,
                        [float(v) for v in run.prediction.mean],
                        [None] * len(run.prediction.mean),
                        [None] * len(run.prediction.mean),
                    )
                    for run in runs[i]
                ],
            )
        )
    return out
