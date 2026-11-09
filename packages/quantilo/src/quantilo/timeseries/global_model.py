"""One gradient boosting model for every series at once (039 story 4), as in the M5 competition.

Each series is divided by its mean over the training window (series of different sizes look
alike), then one model learns the next value from: lags 1 to L, rolling means over 3 and over L
periods, the position in the cycle and the series' identity (both categories). It forecasts one
step at a time (recursive), and the forecast is scaled back.

LightGBM when it is installed (the `[ml]` extra); otherwise scikit-learn's histogram gradient
boosting with the same features (the library without `[ml]`, the browser version).
"""

import importlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quantilo.timeseries import forecast as fc

KEY = "global"
MIN_ROWS = 5  # training rows per series below which an origin is skipped


@dataclass
class GlobalSeries:
    """The global model's result for one series, scored on the same origins as the others."""

    scores: fc.Scores | None
    forecast: list[float] = field(default_factory=list)
    low80: list[float] = field(default_factory=list)
    high80: list[float] = field(default_factory=list)
    low95: list[float] = field(default_factory=list)
    high95: list[float] = field(default_factory=list)
    backtests: list[fc.Backtest] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)  # in-sample one-step errors


def library() -> str:
    try:
        importlib.import_module("lightgbm")
        return "lightgbm"
    except ImportError:
        return "scikit-learn"


def lags_for(n: int, season: int | None) -> int:
    return max(3, min(season or 12, 24, n // 4))


def _row(z: np.ndarray, t: int, lags: int, season: int | None, series: int) -> list[float]:
    """Features predicting z[t] from z[:t]."""
    past = z[t - lags : t]
    return [
        *past[::-1],
        float(np.mean(z[max(0, t - 3) : t])),
        float(np.mean(past)),
        float(t % season) if season else 0.0,
        float(series),
    ]


def _model(kind: str, series: int, width: int) -> tuple[Any, list[int]]:
    categorical = [width - 2, width - 1]  # position in the cycle, series identity
    if kind == "lightgbm":
        lightgbm = importlib.import_module("lightgbm")
        return lightgbm.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            random_state=42,
            verbose=-1,
        ), categorical
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Its categories must stay under 255 values: past that, the identity stays a number.
    settings: dict[str, Any] = {
        "categorical_features": categorical if series < 250 else categorical[:1]
    }
    return HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        random_state=42,
        **settings,
    ), categorical


def fit_predict(
    values: np.ndarray, h: int, season: int | None, kind: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Forecasts (series × h) and in-sample one-step residuals (series × rows), or None when the
    series are too short to learn from."""
    count, n = values.shape
    lags = lags_for(n, season)
    if n - lags < MIN_ROWS:
        return None
    scales = np.array([float(np.mean(np.abs(v))) or 1.0 for v in values])
    scaled = values / scales[:, None]
    rows = [_row(scaled[i], t, lags, season, i) for i in range(count) for t in range(lags, n)]
    target = np.array([scaled[i, t] for i in range(count) for t in range(lags, n)])
    inputs = np.array(rows)
    model, categorical = _model(kind, count, inputs.shape[1])
    if kind == "lightgbm":
        model.fit(inputs, target, categorical_feature=categorical)
    else:
        model.fit(inputs, target)
    fitted = np.asarray(model.predict(inputs), dtype=float).reshape(count, n - lags)
    residuals = (target.reshape(count, n - lags) - fitted) * scales[:, None]
    history = np.hstack([scaled, np.zeros((count, h))])
    for step in range(h):
        t = n + step
        batch = np.array([_row(history[i], t, lags, season, i) for i in range(count)])
        history[:, t] = np.asarray(model.predict(batch), dtype=float)
    return history[:, n:] * scales[:, None], residuals


def forecast(values: np.ndarray, h: int, season: int | None) -> list[GlobalSeries]:
    """Score the global model on the rolling origins of the series (retrained at each origin on
    what came before), then forecast from the whole history with conformal intervals."""
    kind = library()
    count, n = values.shape
    runs: list[list[fc.OriginRun]] = [[] for _ in range(count)]
    nan = np.full(h, np.nan)
    for size in fc.origins(n, h, season):
        steps = min(h, n - size)
        found = fit_predict(values[:, :size], steps, season, kind)
        if found is None:
            continue
        for i in range(count):
            mean = found[0][i]
            runs[i].append(
                fc.OriginRun(
                    size, fc.Prediction(mean, nan[:steps], nan[:steps], nan[:steps], nan[:steps])
                )
            )
    final = fit_predict(values, h, season, kind)
    out = []
    for i in range(count):
        y = values[i]
        try:
            scores = fc.score(y, runs[i], h, season) if runs[i] else None
        except fc.ModelError:
            scores = None
        if final is None:
            out.append(GlobalSeries(scores))
            continue
        mean = final[0][i]
        bands = fc.conformal(y, runs[i], mean) or (mean, mean, mean, mean)
        out.append(
            GlobalSeries(
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
                residuals=[float(v) for v in final[1][i] if math.isfinite(v)],
            )
        )
    return out
