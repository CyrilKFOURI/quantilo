"""Pretrained forecasting models (028 B2, B3, B5): Chronos-Bolt Tiny / Small / Base, Chronos-2 and
TimesFM 2.5, zero-shot (no training), scored on the same rolling origins as the other models:
at each origin the model only sees the history before it (its context).

Intervals come from the models' own quantiles. Chronos-Bolt and TimesFM were trained on the
deciles 0.1 to 0.9: their 80 % interval is native, their 95 % one is the 80 % one widened by the
normal ratio z(0.975) / z(0.9) on each side (it says so in the row). Chronos-2 has the 0.01 and
0.05 quantiles too: both of its intervals are its own.

Imports of torch, chronos and timesfm are lazy: without the extra, every model raises a
`ModelError` carrying the reason (`ai.status`), and nothing else changes.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from quantilo.timeseries import ai
from quantilo.timeseries import forecast as fc

LEVELS = (0.025, 0.1, 0.5, 0.9, 0.975)  # rows of `quantiles`
WIDEN = fc.Z95 / fc.Z80  # 95 % half-width from the 80 % one, normal tails
CONTEXT = {"chronos_bolt": 2048, "chronos_2": 8192, "timesfm": 1024}  # periods seen at most
TIMESFM_HORIZON = 256  # compiled once; longer horizons are refused with the reason
CROSS_BATCH = 100  # series forecast together by Chronos-2 (its technical report's group size)


def _native(spec: ai.ModelSpec) -> tuple[float, ...]:
    """Levels asked of the model itself: Chronos-2 knows the tails, the others the deciles."""
    return LEVELS if spec.family == "chronos_2" else (0.1, 0.5, 0.9)


def quantiles(
    key: str,
    contexts: list[np.ndarray],
    h: int,
    covariates: list[tuple[np.ndarray, np.ndarray]] | None = None,
    cross: bool = False,
) -> np.ndarray:
    """(series, h, 5): the `LEVELS` quantiles of each series' next h periods. `covariates`, one
    (past (n, k), future (h, k)) per series, are Chronos-2's known-in-advance drivers; `cross`
    lets Chronos-2 learn across the series of a batch."""
    try:
        spec = ai.require(key)
        model = ai.pipeline(key)
    except ai.AiError as error:
        raise fc.Unavailable(str(error)) from error
    limit = CONTEXT[spec.family]
    inputs = [np.asarray(c, dtype=np.float32)[-limit:] for c in contexts]
    try:
        from quantilo.timeseries.neural import threads

        with warnings.catch_warnings(), threads():
            warnings.simplefilter("ignore")  # a horizon past the model's own: it says so, unrolls
            raw = _predict(spec, model, inputs, h, covariates, cross, limit)
    except fc.ModelError:
        raise
    except Exception as error:  # a model that fails on this input: its row says why
        raise fc.ModelError(f"{spec.title} could not forecast this series: {error}") from error
    if raw.shape[-1] == 3:  # deciles only: widen the 80 % band to 95 %
        low, mid, high = raw[..., 0], raw[..., 1], raw[..., 2]
        raw = np.stack(
            [mid - (mid - low) * WIDEN, low, mid, high, mid + (high - mid) * WIDEN], axis=-1
        )
    return np.sort(raw, axis=-1)  # no crossing


def _predict(
    spec: ai.ModelSpec,
    model: Any,
    inputs: list[np.ndarray],
    h: int,
    covariates: list[tuple[np.ndarray, np.ndarray]] | None,
    cross: bool,
    limit: int,
) -> np.ndarray:
    import torch  # pyright: ignore[reportMissingImports]

    levels = list(_native(spec))
    with torch.inference_mode():
        if spec.family == "timesfm":
            if h > TIMESFM_HORIZON:
                raise fc.ModelError(f"TimesFM 2.5 forecasts {TIMESFM_HORIZON} periods at most.")
            _compile_timesfm(model, max(len(x) for x in inputs))
            _, found = model.forecast(horizon=h, inputs=[x.astype(float) for x in inputs])
            found = np.asarray(found, dtype=float)  # (series, h, 10): mean, then 0.1 … 0.9
            return found[..., [1, 5, 9]]
        if spec.family == "chronos_2":
            if covariates:
                items: list[Any] = []
                for target, (past, future) in zip(inputs, covariates, strict=True):
                    past = np.asarray(past, dtype=np.float32)[-limit:]
                    names = [f"driver_{j}" for j in range(past.shape[1])]
                    items.append(
                        {
                            "target": target,
                            "past_covariates": {n: past[:, j] for j, n in enumerate(names)},
                            "future_covariates": {
                                n: np.asarray(future[:h, j], dtype=np.float32)
                                for j, n in enumerate(names)
                            },
                        }
                    )
            else:
                items = [torch.from_numpy(x) for x in inputs]
            found, _ = model.predict_quantiles(
                items,
                prediction_length=h,
                quantile_levels=levels,
                batch_size=CROSS_BATCH if cross else 256,
                cross_learning=cross,
            )
            return np.stack([np.asarray(q[0].float().cpu(), dtype=float) for q in found])
        found, _ = model.predict_quantiles(
            [torch.from_numpy(x) for x in inputs], prediction_length=h, quantile_levels=levels
        )
        return np.asarray(found.float().cpu(), dtype=float)


def _compile_timesfm(model: Any, longest: int) -> None:
    """TimesFM decodes for a fixed context and horizon: set them once per context size class."""
    import timesfm  # pyright: ignore[reportMissingImports]

    patch = 32
    context = min(CONTEXT["timesfm"], max(patch, -(-longest // patch) * patch))
    config = getattr(model, "forecast_config", None)
    if config is not None and config.max_context >= context:
        return
    model.compile(
        timesfm.ForecastConfig(
            max_context=context,
            max_horizon=TIMESFM_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
        )
    )


def _prediction(q: np.ndarray, info: str) -> fc.Prediction:
    return fc.Prediction(q[:, 2], q[:, 1], q[:, 3], q[:, 0], q[:, 4], info)


def _label(spec: ai.ModelSpec, n: int) -> str:
    """The row name: Chronos-Bolt Tiny, zero-shot on 156 periods (Bound adds the drivers)."""
    text = f"{spec.title}, zero-shot on {min(n, CONTEXT[spec.family])} periods"
    if spec.family != "chronos_2":
        text += " · 95 % interval widened from its 80 % quantiles"
    return text


def model(key: str) -> Callable[..., fc.Prediction]:
    """A pretrained model as a forecast model `(y, h, season)`; Chronos-2 also takes the drivers
    `(y, h, season, x, x_future)` (028 A4 contract, through `drivers.Bound`)."""
    spec = ai.spec_of(key)

    def fit(
        y: np.ndarray,
        h: int,
        season: int | None,
        x: np.ndarray | None = None,
        x_future: np.ndarray | None = None,
    ) -> fc.Prediction:
        given = (
            [(x, x_future)]
            if spec.drivers and x is not None and x_future is not None and x.shape[1]
            else None
        )
        return _prediction(quantiles(key, [y], h, given)[0], _label(spec, len(y)))

    fit.pretrained = True  # type: ignore[attr-defined]
    return fit


# --- Several series at once (039 every-group forecast, 028 B3) -----------------------------------


def forecast_many(key: str, values: np.ndarray, h: int, season: int | None) -> list[Any]:
    """Every series of `values` (series × periods) forecast in batches at each rolling origin,
    then from the whole history: Chronos-2 learns across the series of a batch (`cross`); the
    others forecast each series on its own, batched for speed. Same shape as the global model's
    result (`global_model.GlobalSeries`)."""
    from quantilo.timeseries import global_model as gm

    spec = ai.spec_of(key)
    cross = spec.family == "chronos_2"
    count, n = values.shape
    runs: list[list[fc.OriginRun]] = [[] for _ in range(count)]
    for size in fc.origins(n, h, season):
        steps = min(h, n - size)
        found = quantiles(key, list(values[:, :size]), steps, cross=cross)
        for i in range(count):
            runs[i].append(fc.OriginRun(size, _prediction(found[i], "")))
    final = quantiles(key, list(values), h, cross=cross)
    out = []
    for i in range(count):
        y = values[i]
        try:
            scores = fc.score(y, runs[i], h, season) if runs[i] else None
        except fc.ModelError:
            scores = None
        q = final[i]
        out.append(
            gm.GlobalSeries(
                scores,
                [float(v) for v in q[:, 2]],
                [float(v) for v in q[:, 1]],
                [float(v) for v in q[:, 3]],
                [float(v) for v in q[:, 0]],
                [float(v) for v in q[:, 4]],
                [
                    fc.Backtest(
                        run.size,
                        [float(v) for v in run.prediction.mean],
                        [float(v) for v in run.prediction.low95],
                        [float(v) for v in run.prediction.high95],
                    )
                    for run in runs[i]
                ],
            )
        )
    return out
