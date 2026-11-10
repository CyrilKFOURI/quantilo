# ruff: noqa: RUF001, RUF002, RUF003  (α, σ, λ and en dashes are the methods' usual notation)
"""Smoothing a series (064): moving average, EWMA, LOWESS, Savitzky-Golay, Kalman, Hodrick-Prescott.

Every method returns a curve with one value per period, its automatic parameters (each with the
reason it was chosen) and whether it is *causal*: the value at t uses data up to t only. Non-causal
curves look ahead and are for reading only, never an input to a model or a forecast.

Kalman = a structural model (`UnobservedComponents`): local level or local linear trend, with a
fixed seasonal pattern when the series has two full cycles, the form with the lowest AIC kept.
Missing periods are handled natively (the update step is skipped and the uncertainty grows), so
the same fit gives a filtered level (causal), a smoothed level (non-causal), estimates of the
missing periods with a band, and one-step-ahead prediction errors for anomalies.
"""

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from quantilo.timeseries import series as ts

SmoothKey = Literal[
    "moving_average", "ewma", "kalman_filtered", "kalman_smoothed", "lowess", "savgol", "hp"
]
SMOOTHERS: tuple[SmoothKey, ...] = (
    "moving_average",
    "ewma",
    "kalman_filtered",
    "kalman_smoothed",
    "lowess",
    "savgol",
    "hp",
)
DEFAULT_SMOOTHERS: tuple[SmoothKey, ...] = ("moving_average",)
KalmanForm = Literal["llevel", "lltrend", "llevel_seasonal", "lltrend_seasonal"]
KALMAN_FORMS: tuple[KalmanForm, ...] = ("llevel", "lltrend", "llevel_seasonal", "lltrend_seasonal")
FORM_NAMES: dict[KalmanForm, str] = {
    "llevel": "local level",
    "lltrend": "local linear trend",
    "llevel_seasonal": "local level + seasonal",
    "lltrend_seasonal": "local linear trend + seasonal",
}
# Ravn & Uhlig (2002): 1600 for quarters, scaled by the 4th power of the frequency ratio.
HP_LAMBDA: dict[ts.Period, float] = {"month": 129_600.0, "quarter": 1_600.0, "year": 6.25}
HP_WARNING = (
    "The last points will move when new data arrives: the filter's end points are unreliable "
    "and it can create cycles that are not in the data (Hamilton 2018)."
)
MAX_FIT = 600  # Kalman variances are estimated on the last periods at most (cost), then applied
DUMMY_SEASON_MAX = 53  # longer cycles use trigonometric terms instead of one dummy per period
HARMONICS = 6
MIN_KNOWN = 8  # periods with a value needed by every model-based method
Z95 = 1.959964
OUTLIER_Z = 3.5  # robust z beyond which a period counts as an outlier (LOWESS suggestion)
OUTLIER_SHARE = 0.05  # beyond this share of the periods, they are not outliers but the series
NOISE_RATIO = 0.5  # EWMA suggestion: σ(differences) / (√2 σ(values)); 1 = white noise
LOWESS_DELTA_ABOVE = 500  # points: interpolate between close x beyond this many (cost)


@dataclass
class Param:
    key: str  # "alpha", "frac", "window", "order", "model", "lambda"
    value: float | int | str
    auto: bool
    reason: str


@dataclass
class Curve:
    key: SmoothKey
    causal: bool
    values: list[float | None]
    low: list[float | None] = field(default_factory=list)  # 95 % band, when the method has one
    high: list[float | None] = field(default_factory=list)
    params: list[Param] = field(default_factory=list)
    info: str | None = None  # what was chosen, in a sentence
    notes: list[str] = field(default_factory=list)
    error: str | None = None  # why it could not be computed


@dataclass
class Suggestion:
    key: SmoothKey
    reason: str


@dataclass
class SmoothParams:
    """Overrides of the automatic parameters (None / "auto" = the rule)."""

    window: int | None = None
    align: ts.Align = "trailing"
    ewma_alpha: float | None = None
    lowess_frac: float | None = None
    savgol_window: int | None = None
    savgol_order: int | None = None
    kalman_model: KalmanForm | Literal["auto"] = "auto"
    hp_lambda: float | None = None


def _array(values: Sequence[float | None] | np.ndarray) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _listed(values: np.ndarray) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in values]


def number(value: float) -> str:
    """12 400, 0.31, 3.2e-05: short and readable."""
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    if abs(value) >= 1_000:
        return f"{value:,.0f}"
    return f"{value:.3g}"


def unavailable(period: ts.Period) -> dict[SmoothKey, str]:
    """Methods that do not apply to this period, with the reason."""
    if period in HP_LAMBDA:
        return {}
    return {"hp": "λ is defined for monthly, quarterly and yearly data only."}


# --- EWMA ----------------------------------------------------------------------------------------


def _ewma_pass(x: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Levels after each update and the sum of squared one-step errors; gaps carry the level."""
    out = np.full(len(x), np.nan)
    level = math.nan
    sse = 0.0
    for i, value in enumerate(x):
        if np.isfinite(value):
            if math.isnan(level):
                level = float(value)
            else:
                error = float(value) - level
                sse += error * error
                level += alpha * error
        out[i] = level
    return out, sse


def ewma_alpha(x: np.ndarray) -> float:
    """α of simple exponential smoothing by conditional maximum likelihood: with Gaussian errors
    it is the α with the smallest sum of squared one-step errors."""
    from scipy.optimize import minimize_scalar

    found = minimize_scalar(
        lambda a: _ewma_pass(x, float(a))[1],
        bounds=(0.01, 0.99),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(found.x)


def _ewma(x: np.ndarray, params: SmoothParams) -> Curve:
    if np.isfinite(x).sum() < 3:
        return Curve("ewma", True, _listed(np.full(len(x), np.nan)), error="Needs 3 values.")
    auto = params.ewma_alpha is None
    alpha = ewma_alpha(x) if auto else float(params.ewma_alpha or 0.3)
    levels, _ = _ewma_pass(x, alpha)
    reason = "by maximum likelihood: the smallest one-step errors" if auto else "set in the options"
    return Curve(
        "ewma",
        True,
        _listed(levels),
        params=[Param("alpha", round(alpha, 4), auto, reason)],
        info=f"α = {alpha:.2f}: each period moves the level {alpha:.0%} of the way to its value.",
    )


# --- LOWESS, Savitzky-Golay, HP ------------------------------------------------------------------


def _cycle(period: ts.Period, season: int | None, n: int) -> int:
    return season or ts.default_window(period, n)


def lowess_frac(period: ts.Period, season: int | None, n: int) -> float:
    return min(1.0, max(0.1, 2 * _cycle(period, season, n) / max(n, 1)))


def _lowess(x: np.ndarray, period: ts.Period, season: int | None, params: SmoothParams) -> Curve:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    n = len(x)
    known = np.isfinite(x)
    if known.sum() < 5:
        return Curve("lowess", False, _listed(np.full(n, np.nan)), error="Needs 5 values.")
    auto = params.lowess_frac is None
    frac = lowess_frac(period, season, n) if auto else float(params.lowess_frac or 0.3)
    t = np.arange(n, dtype=float)
    delta = 0.01 * n if known.sum() > LOWESS_DELTA_ABOVE else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = lowess(x[known], t[known], frac=frac, it=3, delta=delta, return_sorted=False)
    curve = np.interp(t, t[known], np.asarray(fitted, dtype=float))
    cycle = _cycle(period, season, n)
    reason = (
        f"two cycles ({2 * cycle} periods) of the {n} in each neighbourhood, 10 % at least"
        if auto
        else "set in the options"
    )
    return Curve(
        "lowess",
        False,
        _listed(curve),
        params=[Param("frac", round(frac, 4), auto, reason), Param("it", 3, True, "robust passes")],
        info=f"Local lines over {frac:.0%} of the series, outliers down-weighted (3 passes).",
    )


def savgol_window(n: int) -> int:
    window = max(5, round(n / 20))
    return window if window % 2 else window + 1


def _savgol(x: np.ndarray, params: SmoothParams) -> Curve:
    from scipy.signal import savgol_filter

    n = len(x)
    if n < 7:
        return Curve("savgol", False, _listed(np.full(n, np.nan)), error="Needs 7 periods.")
    filled, missing = ts.complete(_listed(x))
    auto_window = params.savgol_window is None
    window = savgol_window(n) if auto_window else int(params.savgol_window or 5)
    order = 2 if params.savgol_order is None else int(params.savgol_order)
    window = min(window if window % 2 else window + 1, n if n % 2 else n - 1)
    order = min(order, window - 2)
    values = np.asarray(savgol_filter(filled, window, order, mode="interp"), dtype=float)
    notes = [f"{missing} empty periods interpolated before filtering."] if missing else []
    return Curve(
        "savgol",
        False,
        _listed(values),
        params=[
            Param(
                "window",
                window,
                auto_window,
                "about a twentieth of the series, odd" if auto_window else "set in the options",
            ),
            Param(
                "order",
                order,
                params.savgol_order is None,
                "a parabola keeps peaks and slopes"
                if params.savgol_order is None
                else "set in the options",
            ),
        ],
        info=f"A degree-{order} polynomial fitted over {window} periods around each one.",
        notes=notes,
    )


def _hp(x: np.ndarray, period: ts.Period, params: SmoothParams) -> Curve:
    from statsmodels.tsa.filters.hp_filter import hpfilter

    n = len(x)
    reason = unavailable(period).get("hp")
    if reason and params.hp_lambda is None:
        return Curve("hp", False, _listed(np.full(n, np.nan)), error=reason)
    if n < 6:
        return Curve("hp", False, _listed(np.full(n, np.nan)), error="Needs 6 periods.")
    filled, missing = ts.complete(_listed(x))
    auto = params.hp_lambda is None
    lam = HP_LAMBDA.get(period, 1_600.0) if auto else float(params.hp_lambda or 1_600.0)
    _, trend = hpfilter(filled, lam)  # type: ignore[arg-type]  # λ is a float (6.25)
    notes = [HP_WARNING]
    if missing:
        notes.append(f"{missing} empty periods interpolated before filtering.")
    return Curve(
        "hp",
        False,
        _listed(np.asarray(trend, dtype=float)),
        params=[
            Param(
                "lambda",
                lam,
                auto,
                f"Ravn & Uhlig (2002) for {period}ly data" if auto else "set in the options",
            )
        ],
        info=f"Trend of the Hodrick–Prescott filter, λ = {number(lam)}.",
        notes=notes,
    )


# --- Kalman --------------------------------------------------------------------------------------


@dataclass
class KalmanFit:
    form: KalmanForm
    season: int | None  # the cycle in the model (None without a seasonal component)
    label: str  # "local level + seasonal (52)"
    auto: bool  # the form was chosen by AIC
    params: np.ndarray  # variances, in the units of the series squared
    sigmas: dict[str, float]  # standard deviations: noise, level, trend
    aic: float
    compared: dict[str, float]  # AIC of every candidate form, on the same periods
    fitted_on: int  # periods used to estimate the variances
    burn: int  # first periods dominated by the diffuse start
    level_filtered: np.ndarray
    level_filtered_sd: np.ndarray
    level_smoothed: np.ndarray
    level_smoothed_sd: np.ndarray
    signal: np.ndarray  # smoothed level + seasonal, the estimate of every period
    signal_sd: np.ndarray  # its standard deviation, observation noise included
    predicted: np.ndarray  # one-step-ahead predictions (data up to t - 1)
    predicted_sd: np.ndarray
    results: Any = None  # statsmodels results on the whole (standardized) series
    center: float = 0.0
    scale: float = 1.0


class KalmanError(ValueError):
    """The structural model cannot be fitted on this series."""


def seasonal_fits(known: int, season: int | None) -> bool:
    return bool(season and season >= 2 and known >= 2 * season)


def _spec(form: KalmanForm, season: int | None) -> dict[str, Any]:
    spec: dict[str, Any] = {"level": "llevel" if form.startswith("llevel") else "lltrend"}
    if form.endswith("seasonal") and season:
        if season <= DUMMY_SEASON_MAX:
            spec.update(seasonal=season, stochastic_seasonal=False)
        else:
            harmonics = min(HARMONICS, season // 2)
            spec.update(
                freq_seasonal=[{"period": season, "harmonics": harmonics}],
                stochastic_freq_seasonal=[False],
            )
    return spec


def _model(z: np.ndarray, form: KalmanForm, season: int | None) -> Any:
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    return UnobservedComponents(z, **_spec(form, season))


def _quiet(compute: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute()


def _candidates(known: int, season: int | None) -> list[KalmanForm]:
    forms: list[KalmanForm] = ["llevel", "lltrend"]
    if seasonal_fits(known, season):
        forms.extend(("llevel_seasonal", "lltrend_seasonal"))
    return forms


def _choose(
    z: np.ndarray, forms: list[KalmanForm], season: int | None
) -> tuple[KalmanForm, Any, dict[str, float]]:
    """Fit every form on the same periods and keep the lowest AIC, each log-likelihood summed
    over the periods left after the longest diffuse start of the candidates (otherwise the form
    that skips more periods looks better)."""
    fits: list[tuple[KalmanForm, Any]] = []
    for form in forms:
        try:
            model = _model(z, form, season)
            fits.append((form, _quiet(lambda m=model: m.fit(disp=False, maxiter=200))))
        except Exception:  # a form that does not converge is skipped
            continue
    if not fits:
        raise KalmanError("No structural model could be fitted.")
    burn = max(int(fit.loglikelihood_burn) for _, fit in fits)
    common = len(z) - burn >= 10
    scores: dict[str, float] = {}
    for form, fit in fits:
        if common:
            ll = float(np.nansum(np.asarray(fit.llf_obs, dtype=float)[burn:]))
            scores[form] = -2 * ll + 2 * len(fit.params)
        else:
            scores[form] = float(fit.aic)
    finite = {k: v for k, v in scores.items() if math.isfinite(v)}
    if not finite:
        raise KalmanError("No structural model could be fitted.")
    best = min(finite, key=lambda k: finite[k])
    return best, dict(fits)[best], finite  # type: ignore[return-value]


def _design(results: Any) -> np.ndarray:
    design = np.asarray(results.model["design"], dtype=float)
    return design[:, :, 0] if design.ndim == 3 else design


def kalman(
    values: Sequence[float | None] | np.ndarray,
    season: int | None,
    form: KalmanForm | Literal["auto"] = "auto",
    params: np.ndarray | None = None,
) -> KalmanFit:
    """Fit the structural model (form by AIC unless given) and run the filter and the smoother.
    `params` (variances, from an earlier fit) skips the estimation."""
    y = _array(values)
    known = np.isfinite(y)
    count = int(known.sum())
    if count < MIN_KNOWN:
        raise KalmanError(f"Needs at least {MIN_KNOWN} periods with a value.")
    center = float(np.mean(y[known]))
    scale = float(np.std(y[known])) or 1.0
    z = (y - center) / scale
    fit_part = z[-MAX_FIT:]
    used_season = season if seasonal_fits(int(np.isfinite(fit_part).sum()), season) else None
    compared: dict[str, float] = {}
    if params is not None:
        chosen: KalmanForm = "llevel" if form == "auto" else form
        scaled = np.asarray(params, dtype=float) / scale**2
    else:
        forms: list[KalmanForm] = _candidates(int(np.isfinite(fit_part).sum()), used_season)
        if form != "auto":
            # a seasonal form asked on a series too short for it: the same form without
            similar: list[KalmanForm] = [f for f in forms if f.startswith(form[:7])]
            forms = [cast(KalmanForm, form)] if form in forms else similar[:1]
        chosen, fitted, compared = _choose(fit_part, forms, used_season)
        scaled = np.asarray(fitted.params, dtype=float)
    if not chosen.endswith("seasonal"):
        used_season = None
    model = _model(z, chosen, used_season)
    results = _quiet(lambda: model.smooth(scaled))
    names = list(model.param_names)
    variances = dict(zip(names, scaled * scale**2, strict=True))
    sigmas = {
        key: math.sqrt(max(float(variances[name]), 0.0))
        for key, name in (
            ("noise", "sigma2.irregular"),
            ("level", "sigma2.level"),
            ("trend", "sigma2.trend"),
        )
        if name in variances
    }
    design = _design(results)
    smoothed = np.asarray(results.smoothed_state, dtype=float)
    smoothed_cov = np.asarray(results.smoothed_state_cov, dtype=float)
    filtered = np.asarray(results.filtered_state, dtype=float)
    filtered_cov = np.asarray(results.filtered_state_cov, dtype=float)
    obs = float(np.asarray(results.model["obs_cov"], dtype=float).ravel()[0])
    signal = (design @ smoothed)[0]
    signal_var = np.einsum("ik,kjt,jl->t", design, smoothed_cov, design.T) + obs
    forecasts = np.asarray(results.forecasts, dtype=float)[0]
    forecast_var = np.asarray(results.forecasts_error_cov, dtype=float)[0, 0]
    name = FORM_NAMES[chosen] + (f" ({used_season})" if used_season else "")
    return KalmanFit(
        chosen,
        used_season,
        name,
        params is None and form == "auto",
        scaled * scale**2,
        sigmas,
        compared.get(chosen, math.nan),
        compared,
        len(fit_part),
        int(results.loglikelihood_burn),
        filtered[0] * scale + center,
        np.sqrt(np.maximum(filtered_cov[0, 0], 0)) * scale,
        smoothed[0] * scale + center,
        np.sqrt(np.maximum(smoothed_cov[0, 0], 0)) * scale,
        signal * scale + center,
        np.sqrt(np.maximum(signal_var, 0)) * scale,
        forecasts * scale + center,
        np.sqrt(np.maximum(forecast_var, 0)) * scale,
        results,
        center,
        scale,
    )


def kalman_info(fit: KalmanFit) -> str:
    how = "chosen by AIC" if fit.auto else "set in the options"
    parts = [f"{fit.label[0].upper()}{fit.label[1:]}, {how}"]
    parts += [f"{key} σ = {number(value)}" for key, value in fit.sigmas.items()]
    return " · ".join(parts)


def _kalman_notes(fit: KalmanFit, n: int) -> list[str]:
    if fit.fitted_on < n:
        return [
            f"Variances estimated on the last {fit.fitted_on} periods, then applied to all {n}."
        ]
    return []


def _kalman_curve(key: SmoothKey, fit: KalmanFit, n: int, form_auto: bool) -> Curve:
    filtered = key == "kalman_filtered"
    level = fit.level_filtered if filtered else fit.level_smoothed
    sd = fit.level_filtered_sd.copy() if filtered else fit.level_smoothed_sd
    if filtered:
        sd[: fit.burn] = np.nan  # the diffuse start: no meaningful band yet
    reason = (
        "lowest AIC among " + ", ".join(FORM_NAMES[f] for f in fit.compared)  # type: ignore[index]
        if form_auto
        else "set in the options"
    )
    return Curve(
        key,
        filtered,
        _listed(level),
        _listed(level - Z95 * sd),
        _listed(level + Z95 * sd),
        params=[
            Param("model", fit.form, form_auto, reason),
            *[
                Param(f"sigma_{k}", round(v, 6), True, "by maximum likelihood")
                for k, v in fit.sigmas.items()
            ],
        ],
        info=kalman_info(fit),
        notes=_kalman_notes(fit, n),
    )


@dataclass
class Filled:
    values: list[float | None]
    low: list[float | None]  # 95 % band at the filled periods only
    high: list[float | None]
    info: str | None


def kalman_fill(values: Sequence[float | None], season: int | None) -> Filled:
    """Empty periods between the first and the last value get the smoothed estimate of the
    structural model (level + seasonal) with a 95 % band (state and observation noise)."""
    n = len(values)
    none: list[float | None] = [None] * n
    known = [i for i, v in enumerate(values) if v is not None]
    if not known or len(known) == n:
        return Filled(list(values), none, list(none), None)
    try:
        fit = kalman(values, season)
    except KalmanError as error:
        return Filled(
            ts.fill_gaps(list(values), "interpolate"),
            none,
            list(none),
            f"{error} Linear interpolation used instead.",
        )
    filled: list[float | None] = list(values)
    low: list[float | None] = list(none)
    high: list[float | None] = list(none)
    for i in range(known[0], known[-1] + 1):
        if values[i] is None:
            v, sd = float(fit.signal[i]), float(fit.signal_sd[i])
            filled[i], low[i], high[i] = v, v - Z95 * sd, v + Z95 * sd
    return Filled(filled, low, high, f"Kalman: {fit.label}, chosen by AIC")


def innovation_scores(fit: KalmanFit) -> np.ndarray:
    """|robust z| of the one-step-ahead errors divided by their predicted standard deviation;
    0 for empty periods and the diffuse start."""
    y = np.asarray(fit.results.model.endog, dtype=float).ravel() * fit.scale + fit.center
    spread = np.where(fit.predicted_sd > 0, fit.predicted_sd, np.nan)
    standard = (y - fit.predicted) / spread
    standard[: fit.burn] = np.nan
    usable = np.isfinite(standard)
    scores = np.zeros(len(y))
    if usable.sum() < 3:
        return scores
    median = float(np.median(standard[usable]))
    mad = float(np.median(np.abs(standard[usable] - median)))
    if not mad:
        return scores
    scores[usable] = np.abs(standard[usable] - median) / (1.4826 * mad)
    return scores


# --- Every method --------------------------------------------------------------------------------


def smooth(
    values: Sequence[float | None],
    keys: Sequence[SmoothKey],
    period: ts.Period,
    season: int | None,
    params: SmoothParams | None = None,
) -> list[Curve]:
    """The chosen methods' curves, in the catalogue's order; one Kalman fit serves both outputs."""
    params = params or SmoothParams()
    x = _array(values)
    n = len(x)
    out: list[Curve] = []
    fit: KalmanFit | str | None = None
    chosen: list[SmoothKey] = [k for k in SMOOTHERS if k in keys]
    for key in chosen:
        if key == "moving_average":
            window = ts.resolve_window(params.window, period, n)
            means, _ = ts.rolling(list(values), window, params.align)
            out.append(
                Curve(
                    key,
                    params.align == "trailing",
                    means,
                    params=[
                        Param(
                            "window",
                            window,
                            params.window is None,
                            "a natural cycle of the period, a quarter of the series at most"
                            if params.window is None
                            else "set in the options",
                        )
                    ],
                    info=f"Mean of {window} periods ({params.align}).",
                )
            )
        elif key == "ewma":
            out.append(_ewma(x, params))
        elif key == "lowess":
            out.append(_lowess(x, period, season, params))
        elif key == "savgol":
            out.append(_savgol(x, params))
        elif key == "hp":
            out.append(_hp(x, period, params))
        else:
            if fit is None:
                try:
                    fit = kalman(x, season, params.kalman_model)
                except KalmanError as error:
                    fit = str(error)
            if isinstance(fit, str):
                out.append(
                    Curve(key, key == "kalman_filtered", _listed(np.full(n, np.nan)), error=fit)
                )
            else:
                out.append(_kalman_curve(key, fit, n, params.kalman_model == "auto"))
    return out


def _outliers(x: np.ndarray, period: ts.Period, season: int | None) -> int:
    """Periods further than OUTLIER_Z robust z-scores from a centred rolling median."""
    known = np.isfinite(x)
    n = int(known.sum())
    if n < 10:
        return 0
    values = x[known]
    width = max(3, min(_cycle(period, season, n), n // 4) | 1)
    half = width // 2
    medians = np.array(
        [np.median(values[max(0, i - half) : i + half + 1]) for i in range(n)], dtype=float
    )
    resid = values - medians
    mad = float(np.median(np.abs(resid - np.median(resid))))
    if not mad:
        return 0
    return int((np.abs(resid - np.median(resid)) / (1.4826 * mad) > OUTLIER_Z).sum())


def noise_ratio(x: np.ndarray) -> float | None:
    """σ of the period-to-period changes / (√2 σ of the values): 1 for white noise, near 0 for a
    random walk or a smooth curve."""
    known = x[np.isfinite(x)]
    pairs = x[1:] - x[:-1]
    pairs = pairs[np.isfinite(pairs)]
    if len(known) < 5 or len(pairs) < 3:
        return None
    spread = float(np.std(known))
    return float(np.std(pairs)) / (math.sqrt(2) * spread) if spread else None


def suggest(
    values: Sequence[float | None], period: ts.Period, season: int | None
) -> list[Suggestion]:
    """Methods worth adding to the moving average, each with the rule that triggered it."""
    x = _array(values)
    n = len(x)
    if n < ts.MIN_PERIODS:
        return []
    found: list[Suggestion] = []
    missing = int((~np.isfinite(x)).sum())
    cycles = n / season if season else 0.0
    ratio = noise_ratio(x)
    if cycles < 2 and ratio is not None and ratio >= NOISE_RATIO:
        found.append(
            Suggestion(
                "ewma",
                f"No two full cycles to model and the noise dominates (change-to-level ratio "
                f"{ratio:.2f}): an EWMA follows the level, its weight chosen by likelihood.",
            )
        )
    if missing and n - missing >= MIN_KNOWN:
        found.append(
            Suggestion(
                "kalman_smoothed",
                f"{missing} of {n} periods are empty ({missing / n:.0%}): Kalman estimates them "
                "with an uncertainty band.",
            )
        )
    outliers = _outliers(x, period, season)
    # Outliers are rare by definition: many points far from the local median mean heavy tails or
    # sharp moves (a share price), which LOWESS would not fix.
    if outliers and outliers <= OUTLIER_SHARE * (n - missing):
        found.append(
            Suggestion(
                "lowess",
                f"{outliers} outlier{'s' if outliers > 1 else ''} beyond {OUTLIER_Z} robust "
                "z-scores: LOWESS down-weights them.",
            )
        )
    if n >= 50 and not missing and period in ("hour", "day"):
        found.append(
            Suggestion(
                "savgol",
                f"A dense, regular signal ({n} {period}s, none empty): Savitzky–Golay smooths it "
                "and keeps its peaks.",
            )
        )
    return found
