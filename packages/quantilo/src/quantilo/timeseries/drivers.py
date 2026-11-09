"""Drivers of a forecast (028 A4): outside variables that may explain the series.

A driver is a column of the file aggregated to the series' period, a calendar variable, or a 0/1
flag (holidays, events). It is either **known in advance** (a planned price: its future values
are given) or **past only** (visits: it enters lagged by the horizon h, so the forecast of a
period only uses driver values at least h periods older; no leakage from the future).

SARIMAX, the linear regression (TSLM) and gradient boosting on lags use them; every driver model
is also scored without any driver and without each one, on the same rolling origins.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np

from quantilo.errors import ExplorerError
from quantilo.timeseries import forecast as fc
from quantilo.timeseries import series as ts
from quantilo.variable.source import clean

Kind = Literal["column", "flag", "calendar"]
Future = Literal["file", "constant", "last"]
CalendarPart = Literal["weekday", "month", "month_end"]

MAX_DRIVERS = 6
HELPS = 0.95  # a driver helps when the MASE with it is at least 5 % lower
SUGGEST = 0.5  # correlation of the period-to-period changes that suggests a driver
CALENDAR: dict[str, tuple[ts.Period, ...]] = {
    "weekday": ("hour", "day"),
    "month": ("day", "week"),
    "month_end": ("day",),
}
CALENDAR_LABELS = {"weekday": "Day of week", "month": "Month", "month_end": "End of month"}


class DriverError(ExplorerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="bad_driver", status_code=422)


@dataclass(frozen=True)
class Driver:
    """One driver as asked for. `column`: a column of the file, or for a calendar driver one of
    `weekday`, `month`, `month_end`. Calendar drivers and flags are always known in advance."""

    column: str
    kind: Kind = "column"
    known: bool = False  # known in advance; past only otherwise (lagged by the horizon)
    future: Future = "file"  # where the future values come from, when known in advance
    value: float | None = None  # the constant, when future = "constant"

    @property
    def in_advance(self) -> bool:
        return self.known or self.kind != "column"


@dataclass
class RawDriver:
    """A column aggregated per period: on the series' periods, then on the forecast periods
    (None where the file has no row)."""

    history: list[float | None]
    future: list[float | None]


@dataclass
class DriverInfo:
    """How a driver was used."""

    column: str
    kind: Kind
    known: bool
    source: Literal["lag", "file", "constant", "last", "calendar"]
    lag: int  # periods; 0 when known in advance
    label: str
    future_values: list[float | None]  # over the horizon (one-column drivers; empty otherwise)


@dataclass
class DriverSet:
    """The drivers as model inputs: one row per period of the series, then per forecast period."""

    infos: list[DriverInfo]
    history: np.ndarray  # (periods, columns)
    future: np.ndarray  # (horizon, columns)
    groups: list[list[int]]  # the columns of each driver
    repeat_last: list[int] = field(default_factory=list)  # validation repeats their last value
    notes: list[str] = field(default_factory=list)
    lag: int = 0  # of the past-only drivers

    def window(self, size: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Inputs of a model trained on the first `size` periods and forecasting `steps`."""
        seen = self.history[:size]
        if size >= len(self.history):
            return seen, self.future[:steps]
        ahead = self.history[size : size + steps].copy()
        for column in self.repeat_last:
            ahead[:, column] = self.history[size - 1, column]
        return seen, ahead

    def with_future(self, values: dict[str, list[float]]) -> "DriverSet":
        """The same drivers with other future values (a scenario): only known-in-advance
        one-column drivers can be changed, with one value per forecast period."""
        future = self.future.copy()
        infos = list(self.infos)
        for name, given in values.items():
            at = next((i for i, info in enumerate(self.infos) if info.column == name), None)
            if at is None:
                raise DriverError(f"{name!r} is not one of the drivers.")
            info = self.infos[at]
            if info.kind == "calendar" or info.lag:
                what = "a calendar variable" if info.kind == "calendar" else "past only"
                raise DriverError(f"{name!r} is {what}: its future values cannot be changed.")
            if len(given) != len(future):
                raise DriverError(f"{name!r} needs {len(future)} future values, got {len(given)}.")
            future[:, self.groups[at][0]] = np.asarray(given, dtype=float)
            infos[at] = dataclasses.replace(info, future_values=[float(v) for v in given])
        return dataclasses.replace(self, infos=infos, future=future)


# --- Building the inputs -------------------------------------------------------------------------


def calendar_columns(part: str, moments: list[datetime], period: ts.Period) -> np.ndarray:
    """Indicators of a calendar part: weekday (Tuesday … Sunday, Monday is the base), month
    (February … December), end of month (the last day of a month)."""
    if part not in CALENDAR:
        raise DriverError(f"Unknown calendar variable {part!r}: weekday, month or month_end.")
    if period not in CALENDAR[part]:
        allowed = " or ".join(CALENDAR[part])
        raise DriverError(f"{CALENDAR_LABELS[part]} needs {allowed} periods, not {period}.")
    if part == "weekday":
        days = np.array([m.weekday() for m in moments])
        return np.column_stack([(days == d).astype(float) for d in range(1, 7)])
    if part == "month":
        months = np.array([m.month for m in moments])
        return np.column_stack([(months == k).astype(float) for k in range(2, 13)])
    last = [float((m + timedelta(days=1)).month != m.month) for m in moments]
    return np.array(last, dtype=float).reshape(-1, 1)


def build(
    drivers: list[Driver],
    raw: dict[str, RawDriver],
    moments: list[datetime],
    future_moments: list[datetime],
    period: ts.Period,
) -> DriverSet:
    """The model inputs of the drivers: `raw` holds the aggregated columns (flags included),
    `moments` the starts of the series' periods, `future_moments` those of the forecast."""
    if len(drivers) > MAX_DRIVERS:
        raise DriverError(f"At most {MAX_DRIVERS} drivers.")
    if len({(d.kind == "calendar", d.column) for d in drivers}) < len(drivers):
        raise DriverError("A driver is listed twice.")
    n, h = len(moments), len(future_moments)
    history: list[np.ndarray] = []
    future: list[np.ndarray] = []
    infos: list[DriverInfo] = []
    groups: list[list[int]] = []
    repeat_last: list[int] = []
    notes: list[str] = []
    width = 0

    def add(past: np.ndarray, ahead: np.ndarray, info: DriverInfo, repeat: bool = False) -> None:
        nonlocal width
        history.append(past.reshape(n, -1))
        future.append(ahead.reshape(h, -1))
        columns = list(range(width, width + history[-1].shape[1]))
        groups.append(columns)
        if repeat:
            repeat_last.extend(columns)
        width += len(columns)
        infos.append(info)

    for driver in drivers:
        if driver.kind == "calendar":
            add(
                calendar_columns(driver.column, moments, period),
                calendar_columns(driver.column, future_moments, period),
                DriverInfo(
                    driver.column,
                    "calendar",
                    True,
                    "calendar",
                    0,
                    CALENDAR_LABELS[driver.column],
                    [],
                ),
            )
            continue
        if driver.column not in raw:
            raise DriverError(f"No values for the driver {driver.column!r}.")
        values = raw[driver.column]
        if driver.kind == "flag":
            past = np.array([0.0 if v is None else v for v in values.history])
            ahead_raw = values.future
            missing = sum(v is None for v in ahead_raw)
            if missing:
                notes.append(
                    f"{driver.column}: {missing} of {h} forecast periods have no value in the "
                    "file; they are taken as 0 (no event)."
                )
            ahead = np.array([0.0 if v is None else v for v in ahead_raw])
            info = DriverInfo(
                driver.column, "flag", True, "file", 0, driver.column, _numbers(ahead)
            )
            add(past, ahead, info)
            continue
        past, _ = ts.complete(values.history)
        if len(past) != n:
            raise DriverError(f"{driver.column!r} has no value on the series' periods.")
        if not driver.known:
            if n <= h:
                raise DriverError("The series is too short to lag a past-only driver.")
            lagged = np.concatenate([np.full(h, past[0]), past[: n - h]])
            ahead = past[n - h :]
            info = DriverInfo(
                driver.column, "column", False, "lag", h, driver.column, _numbers(ahead)
            )
            add(lagged, ahead, info)
            continue
        last = float(past[-1])
        source: Literal["file", "constant", "last"] = driver.future
        repeat = False
        if driver.future == "constant":
            if driver.value is None or not np.isfinite(driver.value):
                raise DriverError(f"{driver.column}: give the constant future value.")
            ahead = np.full(h, float(driver.value))
            notes.append(
                f"{driver.column}: held at {driver.value:g} over the forecast; the validation "
                "used its actual past values, so its scores are an upper bound."
            )
        elif driver.future == "last":
            ahead, repeat = np.full(h, last), True
            notes.append(
                f"{driver.column}: its last value ({last:g}) is repeated over the forecast, which "
                "assumes it stays flat."
            )
        else:
            known = [v for v in values.future if v is not None]
            if not known:
                ahead, repeat, source = np.full(h, last), True, "last"
                notes.append(
                    f"{driver.column}: the file has no future values; its last value ({last:g}) "
                    "is repeated over the forecast, which assumes it stays flat."
                )
            else:
                filled: list[float] = []
                carried = last
                for v in values.future:
                    carried = carried if v is None else float(v)
                    filled.append(carried)
                ahead = np.array(filled)
                missing = h - len(known)
                if missing:
                    notes.append(
                        f"{driver.column}: {missing} of {h} forecast periods have no value in the "
                        "file; the previous value is repeated for them."
                    )
        info = DriverInfo(driver.column, "column", True, source, 0, driver.column, _numbers(ahead))
        add(past, ahead, info, repeat)

    return DriverSet(
        infos,
        np.column_stack(history) if history else np.empty((n, 0)),
        np.column_stack(future) if future else np.empty((h, 0)),
        groups,
        repeat_last,
        notes,
        h if any(info.lag for info in infos) else 0,
    )


def _numbers(values: np.ndarray) -> list[float | None]:
    return [clean(v) for v in values]


# --- Models with drivers -------------------------------------------------------------------------


class Bound:
    """A driver model seen as a plain model `(y, h, season)`: the training length tells which
    driver rows it may see. `keep` = the drivers given to it (all by default). Columns that do
    not vary over the training window are left out; past-only drivers drop the first h periods
    (their lagged values do not exist there)."""

    def __init__(
        self, model: Callable[..., fc.Prediction], drivers: DriverSet, keep: list[int] | None = None
    ) -> None:
        self.model = model
        self.drivers = drivers
        self.keep = list(range(len(drivers.infos))) if keep is None else keep
        self.final: fc.Prediction | None = None

    def __call__(self, y: np.ndarray, h: int, season: int | None) -> fc.Prediction:
        size = len(y)
        seen, ahead = self.drivers.window(size, h)
        trim = self.drivers.lag if any(self.drivers.infos[i].lag for i in self.keep) else 0
        if trim and size - trim < 10:
            raise fc.ModelError("Too few periods once the past-only drivers are lagged.")
        y_fit, seen = y[trim:], seen[trim:]
        columns = [c for i in self.keep for c in self.drivers.groups[i]]
        active = [c for c in columns if np.ptp(seen[:, c]) > 0]
        if active:
            prediction = self.model(y_fit, h, season, seen[:, active], ahead[:, active])
            used = sum(
                any(c in active for c in self.drivers.groups[i])
                for i in range(len(self.drivers.infos))
            )
            prediction.info = f"{prediction.info} + {used} driver{'s' if used > 1 else ''}"
            prediction.exog = active
        else:
            prediction = self.model(y_fit, h, season)
        if size >= len(self.drivers.history):
            self.final = prediction
        return prediction


def _effect(
    prediction: fc.Prediction | None, group: list[int]
) -> tuple[float | None, float | None, float | None]:
    if prediction is None or not prediction.exog:
        return None, None, None
    at = [prediction.exog.index(c) for c in group if c in prediction.exog]
    if not at:
        return None, None, None
    coefficient = p_value = importance = None
    if len(group) == 1 and prediction.coef is not None:
        coefficient = clean(prediction.coef[at[0]])
        p_value = clean(prediction.pvalues[at[0]]) if prediction.pvalues is not None else None
    if prediction.importance is not None:
        importance = clean(np.sum(prediction.importance[at]))
    return coefficient, p_value, importance


def run(
    key: str,
    model: Callable[..., fc.Prediction],
    y: np.ndarray,
    h: int,
    season: int | None,
    drivers: DriverSet,
    cancelled: Callable[[], bool] = lambda: False,
    step: Callable[[], None] = lambda: None,
) -> fc.ModelResult:
    """Score a driver model with every driver, then without any and without each one (same
    origins), and report each driver's coefficient or importance. `step` is called after each
    of the `extra_runs` comparisons (progress)."""
    bound = Bound(model, drivers)
    result = fc.run_model(key, bound, y, h, season)
    result.uses_drivers = True
    if result.scores is None or result.scores.mase is None or not drivers.infos:
        return result

    def mase(keep: list[int]) -> float | None:
        try:
            return fc.evaluate(y, Bound(model, drivers, keep), h, season).mase
        except (fc.ModelError, ValueError, np.linalg.LinAlgError):
            return None
        finally:
            step()

    result.mase_without_drivers = mase([])
    count = len(drivers.infos)
    for i, info in enumerate(drivers.infos):
        if cancelled():
            break
        without = (
            result.mase_without_drivers if count == 1 else mase([j for j in range(count) if j != i])
        )
        coefficient, p_value, importance = _effect(bound.final, drivers.groups[i])
        result.drivers.append(
            fc.DriverEffect(
                info.label,
                coefficient,
                p_value,
                importance,
                without,
                verdict(result.scores.mase, without),
            )
        )
    return result


def extra_runs(drivers: DriverSet) -> int:
    """Validations a driver model runs on top of its own: without any driver, then without each
    one (when there are several)."""
    count = len(drivers.infos)
    return 0 if not count else 1 + (count if count > 1 else 0)


def verdict(with_it: float, without: float | None) -> Literal["helps", "unclear", "no_help"] | None:
    if without is None:
        return None
    if with_it <= without * HELPS:
        return "helps"
    return "no_help" if with_it >= without else "unclear"


def forecast_with_drivers(
    y: np.ndarray,
    h: int,
    season: int | None,
    model: Callable[..., fc.Prediction],
    drivers: DriverSet,
    future: dict[str, list[float]] | None = None,
) -> fc.Prediction:
    """The final forecast of a driver model with given future driver values (scenarios, spec 039
    story 6): `future` maps a known-in-advance driver column to one value per forecast period;
    drivers left out keep their values. Intervals are the model's own (approximate for gradient
    boosting: no validation is run)."""
    chosen = drivers.with_future(future) if future else drivers
    return Bound(model, chosen)(y, h, season)


MAX_SCENARIOS = 3  # named scenarios kept with an analysis


@dataclass(frozen=True)
class ScenarioChange:
    """One change of a what-if scenario (039 story 6): a known-in-advance driver moved by
    `amount` % (`how="percent"`, -10 = ten percent lower) or set to `amount` (`how="value"`),
    on the forecast periods from `start` to `end` (period labels such as "2025-12"; None = from
    the first, or to the last, forecast period)."""

    column: str
    how: Literal["percent", "value"] = "percent"
    amount: float = 0.0
    start: str | None = None
    end: str | None = None


def scenario_future(
    drivers: DriverSet, periods: list[str], changes: list[ScenarioChange]
) -> dict[str, list[float]]:
    """The future values of the changed drivers once the changes are applied, in order; the
    periods outside a change's range keep their base values."""
    future: dict[str, list[float]] = {}
    for change in changes:
        info = next((i for i in drivers.infos if i.column == change.column), None)
        if info is None:
            raise DriverError(f"{change.column!r} is not one of the drivers.")
        for bound in (change.start, change.end):
            if bound is not None and bound not in periods:
                raise DriverError(
                    f"{bound!r} is not a forecast period ({periods[0]} to {periods[-1]})."
                )
        at = drivers.infos.index(info)
        base = future.get(change.column) or [
            float(v) for v in drivers.future[:, drivers.groups[at][0]]
        ]
        first = periods.index(change.start) if change.start else 0
        last = periods.index(change.end) if change.end else len(periods) - 1
        future[change.column] = [
            (v * (1 + change.amount / 100) if change.how == "percent" else change.amount)
            if first <= i <= last
            else v
            for i, v in enumerate(base)
        ]
    return future


# --- Suggestions ---------------------------------------------------------------------------------


@dataclass
class DriverCandidate:
    """A column that could be a driver."""

    column: str
    flag: bool  # only 0 / 1 values (holidays, promotions): suggested as a flag
    correlation: float | None  # of the period-to-period changes with the series
    suggested: bool  # |correlation| ≥ 0.5
    future_periods: int  # forecast periods with a value in the file


def correlation_of_changes(y: np.ndarray, values: list[float | None]) -> float | None:
    """Pearson correlation of the first differences: two series that both grow are not linked
    for that reason alone; two that move together from one period to the next may be."""
    x, _ = ts.complete(values)
    if len(x) != len(y) or len(y) < 5:
        return None
    dy, dx = np.diff(y), np.diff(x)
    if np.std(dy) == 0 or np.std(dx) == 0:
        return None
    return clean(np.corrcoef(dy, dx)[0, 1])


def candidate(column: str, flag: bool, y: np.ndarray, values: RawDriver) -> DriverCandidate:
    r = correlation_of_changes(y, values.history)
    return DriverCandidate(
        column,
        flag,
        r,
        r is not None and abs(r) >= SUGGEST,
        sum(v is not None for v in values.future),
    )


def calendar_parts(period: ts.Period) -> list[str]:
    """The calendar variables that make sense for the period."""
    return [part for part, periods in CALENDAR.items() if period in periods]
