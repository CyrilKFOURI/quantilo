"""Forecast every group of a series at once (039): the best model per series, a global model,
coherent totals, in the background with progress and cancel.

Series are forecast in parallel, one process per core (`ProcessPoolExecutor`); in the browser
(Pyodide: neither processes nor threads) one after the other. Pure arrays in, dataclasses out:
the DuckDB part lives in `service.py`.
"""

import math
import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    BrokenExecutor,
    Future,
    ThreadPoolExecutor,
    wait,
)
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:  # the browser (Pyodide) has no processes: imported only where used
    from concurrent.futures import ProcessPoolExecutor

from quantilo.changes.rows import cell
from quantilo.timeseries import forecast as fc
from quantilo.timeseries import global_model as gm
from quantilo.timeseries import hierarchy as hi
from quantilo.variable.source import clean

IN_BROWSER = sys.platform == "emscripten"
MAX_GROUPS = 200  # series forecast on their own; the others are pooled into "Others"
BROWSER_GROUPS = 30  # in the browser version, one series after the other
LIMIT_RANGE = (1, 1_000)
DEFAULT_MODELS = ("seasonal_naive", "ets_auto", "combination")
SLOW_MODELS = ("arima_auto", "sarima", "sarimax")  # left out unless asked for
INTERMITTENT_ADDED = ("croston", "sba", "tsb")  # added to series with intermittent demand
SHORT_PERIODS = 12  # fewer periods (or fewer than two cycles): seasonal naive only
PARALLEL_FROM = 4  # fewer series: not worth starting processes
MAX_WORKERS = 16  # processes at most, whatever the number of cores (memory: ~250 MB each)
SHOWN = 12  # small multiples
KEPT = 8  # finished jobs remembered

Reconcile = Literal["auto", "bottom_up", "top_down", "mint_shrink"]
Status = Literal["running", "done", "cancelled", "failed"]


# --- One series ----------------------------------------------------------------------------------


@dataclass
class SeriesTask:
    label: str
    values: list[float]  # complete (no hole)
    keys: list[str]
    horizon: int
    season: int | None
    start: int = 0  # first period with rows: history before it does not count


@dataclass
class SeriesOutcome:
    label: str
    best: str | None
    info: str | None
    scores: fc.Scores | None
    forecast: list[float]
    low80: list[float]
    high80: list[float]
    low95: list[float]
    high95: list[float]
    backtests: list[fc.Backtest]
    residuals: list[float]  # in-sample, of the best model (MinT)
    short_history: bool
    demand: fc.DemandPattern | None
    models: list[str]  # the models tried
    error: str | None = None
    seconds: float = 0.0


def _recording(model: fc.Model, n: int, key: str, found: dict[str, np.ndarray]) -> fc.Model:
    """The model, keeping the in-sample residuals of its fit on the whole series."""

    def fit(y: np.ndarray, h: int, season: int | None) -> fc.Prediction:
        prediction = model(y, h, season)
        if len(y) == n and prediction.resid is not None:
            found[key] = np.asarray(prediction.resid, dtype=float)
        return prediction

    return fit


def is_short(n: int, start: int, season: int | None) -> bool:
    history = n - start
    return history < SHORT_PERIODS or bool(season and history < 2 * season)


def series_models(keys: list[str], y: np.ndarray, start: int, season: int | None) -> list[str]:
    """The models one series is forecast with."""
    if is_short(len(y), start, season):
        return ["seasonal_naive"] if season and len(y) >= season + 2 else ["naive"]
    chosen = [k for k in keys if season or k not in ("seasonal_naive", "holt_winters")]
    demand = fc.demand_pattern(y)
    if demand and demand.intermittent:
        chosen += [k for k in INTERMITTENT_ADDED if k not in chosen]
    return chosen or ["naive"]


def _mase(result: fc.ModelResult) -> float:
    return result.scores.mase if result.scores and result.scores.mase is not None else math.inf


def forecast_series(task: SeriesTask) -> SeriesOutcome:
    """Every model on one series, scored on its own rolling origins; the best is kept."""
    started = time.monotonic()
    y = np.asarray(task.values, dtype=float)
    h, season = task.horizon, task.season
    keys = series_models(task.keys, y, task.start, season)
    residuals: dict[str, np.ndarray] = {}
    results: list[fc.ModelResult] = []
    for key in keys:
        if key != "combination":
            model = _recording(fc.MODELS[key], len(y), key, residuals)
            results.append(fc.run_model(key, model, y, h, season))
    if "combination" in keys:
        results.append(fc.combine(results, y, h, season))
    scored = sorted((r for r in results if r.scores), key=_mase)
    short = is_short(len(y), task.start, season)
    demand = fc.demand_pattern(y)
    if scored:
        best = scored[0]
        if best.key == "combination":
            members = sorted((r for r in scored if r.key != "combination"), key=_mase)
            kept = [residuals[r.key] for r in members[: fc.COMBINED] if r.key in residuals]
            length = min((len(r) for r in kept), default=0)
            resid = np.mean([r[len(r) - length :] for r in kept], axis=0) if length else []
        else:
            resid = residuals.get(best.key, np.array([]))
        return SeriesOutcome(
            task.label,
            best.key,
            best.info,
            best.scores,
            best.forecast,
            best.low80,
            best.high80,
            best.low95,
            best.high95,
            best.backtests,
            [float(v) for v in resid],
            short,
            demand,
            keys,
            seconds=time.monotonic() - started,
        )
    # Too short to validate: the simplest forecast that can be made, without scores.
    for key in (*keys, "naive"):
        try:
            final = fc.MODELS[key](y, h, season)
        except (fc.ModelError, ValueError, IndexError):
            continue
        mean, low80, high80, low95, high95 = ([float(v) for v in b] for b in _bands(final))
        return SeriesOutcome(
            task.label,
            key,
            final.info,
            None,
            mean,
            low80,
            high80,
            low95,
            high95,
            [],
            [float(v) for v in (final.resid if final.resid is not None else [])],
            short,
            demand,
            keys,
            "Too short to validate: not scored.",
            time.monotonic() - started,
        )
    flat = [float(y[-1]) if len(y) else 0.0] * h
    return SeriesOutcome(
        task.label, None, None, None, flat, flat, flat, flat, flat, [], [], short, demand, keys,
        "No model could be fitted.", time.monotonic() - started,
    )  # fmt: skip


def _bands(p: fc.Prediction) -> tuple[np.ndarray, ...]:
    return p.mean, p.low80, p.high80, p.low95, p.high95


def _one_thread() -> None:
    """Each process keeps to its core: no extra BLAS / OpenMP threads."""
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:  # pragma: no cover - threadpoolctl comes with scikit-learn
        pass


def default_workers(count: int) -> int:
    """One process per core, never more than the series nor MAX_WORKERS (each process loads
    statsmodels and scikit-learn: about 250 MB)."""
    if IN_BROWSER:
        return 1
    return max(1, min(os.cpu_count() or 1, count, MAX_WORKERS))


_pool: "ProcessPoolExecutor | None" = None
_pool_size = 0
_pool_lock = threading.Lock()


def shared_pool(workers: int) -> "ProcessPoolExecutor":
    """One pool of processes kept for the life of the program: starting a process and importing
    the models costs seconds, paid once instead of at every forecast."""
    global _pool, _pool_size
    with _pool_lock:
        if _pool is None or _pool_size < workers:
            if _pool is not None:
                _pool.shutdown(wait=False, cancel_futures=True)
            from concurrent.futures import ProcessPoolExecutor

            _pool = ProcessPoolExecutor(max_workers=workers, initializer=_one_thread)
            _pool_size = workers
        return _pool


@contextmanager
def _without_main_script() -> Iterator[None]:
    """Processes started inside do not re-run the caller's script. On Windows and macOS a new
    process imports the main module first, so a script without `if __name__ == "__main__":`
    would run again in each one (and fail when it starts its own forecast). The tasks live in
    this package: the processes do not need the script. A process starts inside `submit`."""
    main = sys.modules.get("__main__")
    if main is None:
        yield
        return
    saved = {key: main.__dict__[key] for key in ("__file__", "__spec__") if key in main.__dict__}
    with _pool_lock:
        main.__dict__.pop("__file__", None)
        main.__dict__["__spec__"] = None
        try:
            yield
        finally:
            main.__dict__.pop("__spec__", None)
            main.__dict__.update(saved)


def _drop_pool(broken: "ProcessPoolExecutor") -> None:
    """Forget a pool whose process died (out of memory…): the next forecast starts a new one."""
    global _pool, _pool_size
    with _pool_lock:
        if _pool is broken:
            _pool, _pool_size = None, 0
    broken.shutdown(wait=False, cancel_futures=True)


def warm_up() -> list[Future]:
    """Start the processes ahead of a forecast (when a group is chosen); each one imports the
    models while the user looks at the options. Returns the start-up tasks."""
    if IN_BROWSER:
        return []
    workers = default_workers(os.cpu_count() or 1)
    if workers <= 1:
        return []
    pool = shared_pool(workers)
    with _without_main_script():
        return [pool.submit(_one_thread) for _ in range(workers)]


def forecast_all(
    tasks: list[SeriesTask],
    workers: int,
    cancelled: Callable[[], bool] = lambda: False,
    finished: Callable[[SeriesOutcome], None] = lambda _: None,
) -> list[SeriesOutcome] | None:
    """Every task, in parallel when it pays; the outcomes in the tasks' order, None when
    cancelled (series not started yet are dropped)."""
    if workers <= 1 or len(tasks) < PARALLEL_FROM or IN_BROWSER:
        done = []
        for task in tasks:
            if cancelled():
                return None
            outcome = forecast_series(task)
            finished(outcome)
            done.append(outcome)
        return done
    outcomes: dict[int, SeriesOutcome] = {}
    pool = shared_pool(workers)
    try:
        with _without_main_script():
            pending = {pool.submit(forecast_series, t): i for i, t in enumerate(tasks)}
    except BrokenExecutor:  # a process died (BrokenProcessPool)
        _drop_pool(pool)
        return forecast_all(tasks, 1, cancelled, finished)
    try:
        while pending:
            if cancelled():
                return None
            ready, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in ready:
                index = pending.pop(future)
                try:
                    outcomes[index] = future.result()
                except BrokenExecutor:  # a process died (BrokenProcessPool)
                    # A process died: the series left run here, one after the other.
                    _drop_pool(pool)
                    pending.clear()
                    break
                finished(outcomes[index])
    finally:
        for future in pending:
            future.cancel()  # series not started yet; running ones end on their own
    for i, task in enumerate(tasks):
        if i not in outcomes:
            if cancelled():
                return None
            outcomes[i] = forecast_series(task)
            finished(outcomes[i])
    return [outcomes[i] for i in range(len(tasks))]


# --- Every group ---------------------------------------------------------------------------------


@dataclass
class Node:
    """One series to forecast: a leaf (a group, or the pooled "Others"), or, with a hierarchy,
    a parent or the total."""

    label: str
    level: str
    values: list[float]  # complete, on the history periods
    start: int = 0
    other: bool = False


@dataclass
class GroupRow:
    label: str
    level: str  # the group column, a parent column, or "Total"
    parent: str | None
    volume: float  # sum of the values over the history
    demand: str | None  # smooth, erratic, intermittent, lumpy (series with zeros)
    best: str | None
    info: str | None
    mase: float | None
    wape: float | None
    total: float  # forecast summed over the horizon
    change: float | None  # vs the same periods a year earlier, %
    short_history: bool
    other: bool  # the pooled "Others" series
    history: list[float | None]
    forecast: list[float]
    low80: list[float]
    high80: list[float]
    low95: list[float]
    high95: list[float]
    base: list[float]  # before reconciliation (= forecast without it)
    global_mase: float | None = None
    error: str | None = None


@dataclass
class BestShare:
    model: str
    series: int
    share: float  # of the group's series, in %


@dataclass
class GlobalSummary:
    library: str  # lightgbm or scikit-learn
    info: str
    wins: int  # series where it has the lowest MASE
    series: int


@dataclass
class BatchSummary:
    """An AI model run on every series at once (028 B3, B4): pretrained in batches (Chronos-2
    learning across the series), or one network trained on every series' windows."""

    model: str
    info: str
    wins: int  # series where it has the lowest MASE
    series: int
    error: str | None = None  # why it did not run (extra, weights, too few groups)


@dataclass
class LevelError:
    level: str
    nodes: int
    mase_before: float | None
    mase_after: float | None


@dataclass
class HierarchyResult:
    levels: list[str]  # top → bottom: Total, region, store
    method: hi.Method
    method_auto: bool
    gap: float  # largest |parent - Σ children| after reconciliation
    errors: list[LevelError]
    shrinkage: float | None = None  # MinT's λ
    note: str | None = None


@dataclass
class GroupForecast:
    group: str
    horizon: int
    periods: list[str]
    future: list[str]
    models: list[str]
    rows: list[GroupRow]  # leaves by volume (Others last), then parents, then the total
    groups: int  # labels of the group column
    kept: int  # forecast on their own
    pooled: int  # labels in "Others"
    limit: int
    short: int  # leaves with a short history
    best_models: list[BestShare]
    global_model: GlobalSummary | None
    hierarchy: HierarchyResult | None
    notes: list[str]
    workers: int
    seconds: float = 0.0
    batch_models: list[BatchSummary] = field(default_factory=list)


@dataclass
class GroupJob:
    id: str
    status: Status
    done: int  # series finished
    total: int
    stage: Literal["series", "global", "ai", "reconcile"] = "series"
    current: str | None = None
    result: GroupForecast | None = None
    error: str | None = None


def _change(values: list[float], forecast: list[float], per_year: int) -> float | None:
    """Forecast total against the same periods one year earlier, in %."""
    n, h = len(values), len(forecast)
    if per_year < h or n - per_year < 0 or n - per_year + h > n:
        return None
    before = float(np.sum(values[n - per_year : n - per_year + h]))
    return clean((sum(forecast) - before) / abs(before) * 100) if before else None


def _level_errors(
    tree: hi.Tree,
    outcomes: list[SeriesOutcome],
    history: np.ndarray,
    matrix: np.ndarray,
    season: int | None,
) -> list[LevelError]:
    """Mean MASE of each level on the rolling origins every node shares, before and after the
    forecasts of each origin are reconciled with the same G."""
    by_origin = [{b.origin: b.forecast for b in o.backtests} for o in outcomes]
    shared = set(by_origin[0]) if by_origin else set()
    for found in by_origin[1:]:
        shared &= set(found)
    before = np.zeros(len(outcomes))
    after = np.zeros(len(outcomes))
    counted = 0
    for origin in sorted(shared):
        steps = min(len(f[origin]) for f in by_origin)
        steps = min(steps, history.shape[1] - origin)
        if steps <= 0:
            continue
        base = np.array([f[origin][:steps] for f in by_origin])
        coherent = hi.reconcile(tree, base, matrix)
        actual = history[:, origin : origin + steps]
        scales = np.array([fc._scale(row[:origin], season)[0] for row in history])
        before += np.mean(np.abs(base - actual), axis=1) / scales
        after += np.mean(np.abs(coherent - actual), axis=1) / scales
        counted += 1
    errors = []
    for level in dict.fromkeys(tree.levels):
        nodes = [i for i, name in enumerate(tree.levels) if name == level]
        errors.append(
            LevelError(
                level,
                len(nodes),
                clean(float(np.mean(before[nodes])) / counted) if counted else None,
                clean(float(np.mean(after[nodes])) / counted) if counted else None,
            )
        )
    return errors


@dataclass
class Reconciled:
    forecasts: np.ndarray  # nodes × h
    result: HierarchyResult


def reconcile_outcomes(
    tree: hi.Tree,
    outcomes: list[SeriesOutcome],
    history: np.ndarray,
    method: Reconcile,
    season: int | None,
) -> Reconciled:
    """Make the base forecasts of every node coherent; `outcomes` and `history` follow the
    tree's nodes."""
    residuals = hi.aligned_residuals(
        [np.asarray(o.residuals) if o.residuals else None for o in outcomes]
    )
    chosen = hi.auto_method(residuals) if method == "auto" else method
    note = None
    if chosen == "mint_shrink" and residuals is None:
        chosen, note = "bottom_up", "A node has no in-sample residuals: bottom-up instead of MinT."
    elif method == "auto" and chosen == "bottom_up":
        note = f"Fewer than {hi.MIN_RESIDUALS} residuals per series: bottom-up instead of MinT."
    shrinkage = None
    if chosen == "mint_shrink" and residuals is not None:
        shrinkage = hi.shrink_covariance(residuals.T)[1]
    matrix = hi.projection(tree, chosen, residuals, history)
    base = np.array([o.forecast for o in outcomes])
    coherent = hi.reconcile(tree, base, matrix)
    return Reconciled(
        coherent,
        HierarchyResult(
            list(dict.fromkeys(tree.levels)),
            chosen,
            method == "auto",
            hi.incoherence(tree, coherent),
            _level_errors(tree, outcomes, history, matrix, season),
            clean(shrinkage) if shrinkage is not None else None,
            note,
        ),
    )


def _with_global(
    outcome: SeriesOutcome, found: gm.GlobalSeries, info: str, key: str = gm.KEY
) -> SeriesOutcome:
    """The global model (or an AI model run on every series, `key`) replaces a series' best
    model where its MASE is lower."""
    mine = outcome.scores.mase if outcome.scores and outcome.scores.mase is not None else math.inf
    theirs = found.scores.mase if found.scores and found.scores.mase is not None else math.inf
    if theirs >= mine or not found.forecast:
        return outcome
    return SeriesOutcome(
        outcome.label,
        key,
        info,
        found.scores,
        found.forecast,
        found.low80,
        found.high80,
        found.low95,
        found.high95,
        found.backtests,
        found.residuals,
        outcome.short_history,
        outcome.demand,
        [*outcome.models, key],
        seconds=outcome.seconds,
    )


def batch_refusal(key: str, series: int, periods: int) -> str | None:
    """Why an AI model is not run on every group (None: it is): a trained network needs many
    groups or long series (028 B4); a pretrained one needs the extra and its weights."""
    from quantilo.timeseries import ai, neural

    few = series < neural.MIN_GROUPS and periods < neural.MIN_PERIODS
    if key in fc.NEURAL_MODELS and few:
        return (
            f"{neural.KINDS[key]} needs {neural.MIN_GROUPS} groups or more, or "
            f"{neural.MIN_PERIODS} periods: {series} series of {periods} periods are too "
            "few windows to learn from."
        )
    try:
        ai.require(key)
    except ai.AiError as error:
        return str(error)
    return None


def run_batch_model(key: str, values: np.ndarray, h: int, season: int | None) -> list[Any]:
    if key in fc.NEURAL_MODELS:
        from quantilo.timeseries import neural

        return neural.forecast_many(key, values, h, season)
    from quantilo.timeseries import pretrained

    return pretrained.forecast_many(key, values, h, season)


def batch_info(key: str, values: np.ndarray, season: int | None) -> str:
    from quantilo.timeseries import ai, neural

    title = ai.spec_of(key).title
    if key in fc.NEURAL_MODELS:
        length = neural.window(1, season)
        return f"{title} trained on every series' windows (up to {length} periods)"
    if key == "chronos_2":
        return f"{title}, zero-shot, learning across the {len(values)} series"
    return f"{title}, zero-shot, every series"


def forecast_groups(
    job: GroupJob,
    nodes: list[Node],
    tree: hi.Tree | None,
    keys: list[str],
    horizon: int,
    season: int | None,
    per_year: int,
    periods: list[str],
    future: list[str],
    meta: dict[str, Any],
    cancelled: Callable[[], bool],
    *,
    use_global: bool = False,
    reconcile: Reconcile | None = None,
    workers: int | None = None,
) -> GroupForecast | None:
    """Forecast every node (leaves first, in `nodes`' order; with a tree, `nodes` follows its
    nodes), then the global model on the leaves, then reconcile. None when cancelled."""
    started = time.monotonic()
    # AI models (028 B3, B4) run on every series at once, in this process, after the others:
    # never loaded once per worker process.
    batched = [k for k in keys if k in fc.AI_MODELS]
    own = [k for k in keys if k not in fc.AI_MODELS] or ["seasonal_naive", "naive"]
    tasks = [SeriesTask(n.label, n.values, own, horizon, season, n.start) for n in nodes]
    chosen_workers = workers or default_workers(len(tasks))

    def finished(outcome: SeriesOutcome) -> None:
        job.done += 1
        job.current = outcome.label

    outcomes = forecast_all(tasks, chosen_workers, cancelled, finished)
    if outcomes is None:
        return None
    leaves = tree.bottom if tree else list(range(len(nodes)))
    summary = None
    global_scores: dict[int, float | None] = {}
    if use_global and len(leaves) >= 2:
        job.stage, job.current = "global", None
        values = np.array([nodes[i].values for i in leaves])
        found = gm.forecast(values, horizon, season)
        if cancelled():
            return None
        kind = gm.library()
        name = "LightGBM" if kind == "lightgbm" else "scikit-learn gradient boosting"
        info = f"Global {name} on {gm.lags_for(values.shape[1], season)} lags, every series"
        wins = 0
        for leaf, result in zip(leaves, found, strict=True):
            global_scores[leaf] = result.scores.mase if result.scores else None
            replaced = _with_global(outcomes[leaf], result, info)
            wins += replaced.best == gm.KEY
            outcomes[leaf] = replaced
        summary = GlobalSummary(kind, info, wins, len(leaves))
    batches: list[BatchSummary] = []
    if batched and leaves:
        job.stage = "ai"
        values = np.array([nodes[i].values for i in leaves])
        for key in batched:
            job.current = key
            refused = batch_refusal(key, len(leaves), values.shape[1])
            found_ai: list[Any] = []
            if refused is None:
                try:
                    found_ai = run_batch_model(key, values, horizon, season)
                except (fc.ModelError, ValueError, RuntimeError) as error:
                    refused = str(error)
            if refused is not None:
                batches.append(BatchSummary(key, "", 0, len(leaves), refused))
                continue
            if cancelled():
                return None
            info = batch_info(key, values, season)
            wins = 0
            for leaf, result in zip(leaves, found_ai, strict=True):
                replaced = _with_global(outcomes[leaf], result, info, key)
                wins += replaced.best == key
                outcomes[leaf] = replaced
            batches.append(BatchSummary(key, info, wins, len(leaves)))
    forecasts = np.array([o.forecast for o in outcomes])
    hierarchy = None
    notes = list(meta.get("notes", []))
    if tree is not None and reconcile is not None:
        job.stage, job.current = "reconcile", None
        history = np.array([n.values for n in nodes])
        done = reconcile_outcomes(tree, outcomes, history, reconcile, season)
        forecasts, hierarchy = done.forecasts, done.result
        if hierarchy.note:
            notes.append(hierarchy.note)
    parents = (
        {i: tree.labels[p] for i, p in enumerate(tree.parents) if p is not None} if tree else {}
    )
    rows = []
    for i, (node, outcome) in enumerate(zip(nodes, outcomes, strict=True)):
        shift = forecasts[i] - np.asarray(outcome.forecast)

        def moved(band: list[float], shift: np.ndarray = shift) -> list[float]:
            return [float(v) for v in np.asarray(band) + shift]

        rows.append(
            GroupRow(
                node.label,
                node.level,
                parents.get(i),
                float(np.sum(node.values)),
                outcome.demand.kind if outcome.demand else None,
                outcome.best,
                outcome.info,
                outcome.scores.mase if outcome.scores else None,
                outcome.scores.wape if outcome.scores else None,
                float(np.sum(forecasts[i])),
                _change(node.values, [float(v) for v in forecasts[i]], per_year),
                outcome.short_history,
                node.other,
                [float(v) for v in node.values],
                [float(v) for v in forecasts[i]],
                moved(outcome.low80),
                moved(outcome.high80),
                moved(outcome.low95),
                moved(outcome.high95),
                list(outcome.forecast),
                global_scores.get(i),
                outcome.error,
            )
        )
    leaf_rows = [rows[i] for i in leaves]
    depth = list(dict.fromkeys(n.level for n in nodes))
    above = [i for i in range(len(nodes)) if i not in set(leaves)]
    order = [*leaves, *sorted(above, key=lambda i: -depth.index(nodes[i].level))]
    rows = [rows[i] for i in order]
    counted: dict[str, int] = {}
    for row in leaf_rows:
        if row.best:
            counted[row.best] = counted.get(row.best, 0) + 1
    catalogue = [*fc.MODELS, gm.KEY]
    shares = [
        BestShare(key, count, round(count / len(leaf_rows) * 100, 1))
        for key, count in sorted(counted.items(), key=lambda kv: (-kv[1], catalogue.index(kv[0])))
    ]
    return GroupForecast(
        meta["group"],
        horizon,
        periods,
        future,
        keys,
        rows,
        meta["groups"],
        meta["kept"],
        meta["pooled"],
        meta["limit"],
        sum(r.short_history for r in leaf_rows),
        shares,
        summary,
        hierarchy,
        notes,
        chosen_workers,
        time.monotonic() - started,
        batches,
    )


# --- Jobs ----------------------------------------------------------------------------------------


Work = Callable[[GroupJob, Callable[[], bool]], GroupForecast | None]


class GroupJobs:
    """Background jobs of `forecast_groups`, reused for the same request."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="groups")
        self._jobs: OrderedDict[str, GroupJob] = OrderedDict()
        self._by_key: dict[tuple[Any, ...], str] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._by_key.clear()

    def get(self, job_id: str) -> GroupJob | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> GroupJob | None:
        job = self._jobs.get(job_id)
        if job and job.status == "running":
            self._cancels[job_id].set()
        return job

    def start(self, key: tuple[Any, ...], total: int, work: Work) -> GroupJob:
        with self._lock:
            known = self._by_key.get(key)
            if known and (job := self._jobs.get(known)) and job.status in ("running", "done"):
                return job
            job = GroupJob(uuid.uuid4().hex[:12], "running", 0, total)
            self._jobs[job.id] = job
            self._cancels[job.id] = threading.Event()
            self._by_key[key] = job.id
            while len(self._jobs) > KEPT:
                old, _ = self._jobs.popitem(last=False)
                self._cancels.pop(old, None)
                self._by_key = {k: v for k, v in self._by_key.items() if v != old}
        self._executor.submit(self._run, job, work)
        return job

    def _run(self, job: GroupJob, work: Work) -> None:
        cancelled = self._cancels[job.id].is_set
        try:
            result = work(job, cancelled)
            if result is None:
                job.status = "cancelled"
                return
            job.result, job.status, job.current = result, "done", None
        except Exception as error:
            job.status, job.error = "failed", str(error)


# --- Export --------------------------------------------------------------------------------------


def long_csv(result: GroupForecast) -> str:
    """One line per series and forecast period; labels a spreadsheet would run as a formula are
    quoted (negative numbers stay numbers)."""

    def text(value: str) -> str:
        escaped = cell(value).replace('"', '""')
        return f'"{escaped}"' if any(c in escaped for c in ',"\n\r') else escaped

    lines = ["group,level,period,forecast,low80,high80,low95,high95,model"]
    for row in result.rows:
        for i, period in enumerate(result.future):
            numbers = (row.forecast[i], row.low80[i], row.high80[i], row.low95[i], row.high95[i])
            lines.append(
                ",".join(
                    [
                        text(row.label),
                        text(row.level),
                        period,
                        *(repr(float(v)) for v in numbers),
                        text(row.best or ""),
                    ]
                )
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "BestShare",
    "GlobalSummary",
    "GroupForecast",
    "GroupJob",
    "GroupJobs",
    "GroupRow",
    "HierarchyResult",
    "LevelError",
    "Node",
    "SeriesOutcome",
    "SeriesTask",
    "forecast_all",
    "forecast_groups",
    "forecast_series",
    "long_csv",
    "reconcile_outcomes",
]
