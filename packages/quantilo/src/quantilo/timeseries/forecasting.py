"""Forecast jobs: models run one after the other in the background, with progress and cancel.

Results are kept by (series, horizon, models) so running the same forecast again is instant.
"""

import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal

import numpy as np

from quantilo.timeseries import ai
from quantilo.timeseries import drivers as dv
from quantilo.timeseries import forecast as fc
from quantilo.timeseries import prophet_like as pl
from quantilo.timeseries import series as ts
from quantilo.timeseries.seasonality import period_end, starts

Status = Literal["running", "done", "cancelled", "failed"]
KEPT = 16


@dataclass
class SeriesForecast:
    horizon: int
    periods: list[str]  # history (complete periods)
    history: list[float | None]
    future: list[str]  # the forecast periods
    reference: str  # the naive model each model is compared with
    origins: int  # rolling origins used for validation
    results: list[fc.ModelResult]  # sorted by MASE, failures last
    best: str | None
    demand: fc.DemandPattern | None = None  # set when the series has zeros
    drivers: list[dv.DriverInfo] = field(default_factory=list)  # 028 A4
    driver_notes: list[str] = field(default_factory=list)  # assumptions about future values
    # Seasonal cycles the history covers at least twice ("weekly", "yearly"): with two or more,
    # models with several seasons (Prophet-like, STLF) are suggested (028 A6).
    cycles: list[str] = field(default_factory=list)
    # Coded rules on the length of the series (028 B2, B4): a pretrained model for a short one,
    # a trained network for a long one, each with its reason.
    suggestions: list[ai.ModelSuggestion] = field(default_factory=list)


@dataclass
class ForecastJob:
    id: str
    status: Status
    done: int  # models finished
    total: int
    current: str | None = None  # the model being fitted
    result: SeriesForecast | None = None
    error: str | None = None


def future_periods(last: str, period: ts.Period, horizon: int) -> list[str]:
    moment = starts([last], period)[0]
    labels = []
    for _ in range(horizon):
        moment = period_end(moment, period) + timedelta(seconds=1)
        labels.append(moment.strftime(ts.FORMATS[period]))
    return labels


class ForecastJobs:
    def __init__(self, workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="forecast")
        self._jobs: OrderedDict[str, ForecastJob] = OrderedDict()
        self._by_key: dict[tuple[Any, ...], str] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._by_key.clear()

    def get(self, job_id: str) -> ForecastJob | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> ForecastJob | None:
        job = self._jobs.get(job_id)
        if job and job.status == "running":
            self._cancels[job_id].set()
        return job

    def start(
        self,
        key: tuple[Any, ...],
        values: list[float | None],
        periods: list[str],
        period: ts.Period,
        season: int | None,
        horizon: int,
        models: dict[str, fc.Model],
        drivers: dv.DriverSet | None = None,
        limit: float | None = None,
    ) -> ForecastJob:
        with self._lock:
            known = self._by_key.get(key)
            if known and (job := self._jobs.get(known)) and job.status in ("running", "done"):
                return job
            extra = dv.extra_runs(drivers) if drivers else 0
            total = len(models) + extra * sum(key in fc.DRIVER_MODELS for key in models)
            job = ForecastJob(uuid.uuid4().hex[:12], "running", 0, total)
            self._jobs[job.id] = job
            self._cancels[job.id] = threading.Event()
            self._by_key[key] = job.id
            while len(self._jobs) > KEPT:
                old_id, _ = self._jobs.popitem(last=False)
                self._cancels.pop(old_id, None)
                self._by_key = {k: v for k, v in self._by_key.items() if v != old_id}
        self._executor.submit(
            self._run, job, values, periods, period, season, horizon, models, drivers, limit
        )
        return job

    def _run(
        self,
        job: ForecastJob,
        values: list[float | None],
        periods: list[str],
        period: ts.Period,
        season: int | None,
        horizon: int,
        models: dict[str, fc.Model],
        drivers: dv.DriverSet | None = None,
        limit: float | None = None,
    ) -> None:
        cancelled = self._cancels[job.id].is_set

        def step() -> None:
            job.done += 1

        try:
            series, _ = ts.complete(values)
            results: list[fc.ModelResult] = []
            for key, model in models.items():
                if key == "combination":
                    continue  # needs the others: computed last
                if self._cancels[job.id].is_set():
                    job.status = "cancelled"
                    return
                job.current = key
                refused = ai.refusal(key, len(series))
                if refused:  # 028 B1, B4: extra or weights missing, a network on a short series
                    results.append(fc.ModelResult(key, None, None, None, error=refused))
                    if drivers is not None and drivers.infos and key in fc.DRIVER_MODELS:
                        job.done += dv.extra_runs(drivers)  # its comparisons will not run
                elif drivers is not None and drivers.infos and key in fc.DRIVER_MODELS:
                    results.append(
                        dv.run(key, model, series, horizon, season, drivers, cancelled, step, limit)
                    )
                else:
                    results.append(fc.run_model(key, model, series, horizon, season, limit))
                job.done += 1
            if "combination" in models:
                job.current = "combination"
                results.append(fc.combine(results, series, horizon, season))
                job.done += 1
            job.current = None
            reference = self._reference(results, series, horizon, season)
            ranked = sorted(
                results,
                key=lambda r: (
                    r.scores is None,
                    r.scores.mase if r.scores and r.scores.mase is not None else np.inf,
                ),
            )
            best = next((r.key for r in ranked if r.scores), None)
            job.result = SeriesForecast(
                horizon,
                periods,
                [None if v is None else float(v) for v in values],
                future_periods(periods[-1], period, horizon),
                reference,
                len(fc.origins(len(series), horizon, season)),
                ranked,
                best,
                fc.demand_pattern(series),
                drivers.infos if drivers else [],
                drivers.notes if drivers else [],
                [cycle.name for cycle in pl.cycles(period, season, len(series))[0]],
                ai.suggestions(len(series), season, list(models)),
            )
            job.status = "done"
        except Exception as error:
            job.status, job.error = "failed", str(error)

    @staticmethod
    def _reference(results: list[fc.ModelResult], series, horizon: int, season: int | None) -> str:
        """The naive reference is always scored, even when it was not among the chosen models."""
        wanted = "seasonal_naive" if season and len(series) > 2 * season + 2 else "naive"
        if not any(r.key == wanted and r.scores for r in results):
            extra = fc.run_model(wanted, fc.MODELS[wanted], series, horizon, season)
            fc.compare_to_reference([*results, extra], season)
            return wanted
        return fc.compare_to_reference(results, season)


ModelFactory = Callable[[str], fc.Model]
