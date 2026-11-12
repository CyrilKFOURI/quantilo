"""Regimes of a series (062): a Markov-switching model (statsmodels `MarkovRegression`, a
Gaussian hidden Markov model with its own mean and variance in each regime). Slow for a modest
gain, so it is computed only on demand (Advanced, owner's onion rule).

Rules, each said with its reason:
- R-RG-01  what is modelled: a series with a unit root (ADF p > 0.05) is read through its period
           changes (first differences): the regimes of a random walk's level only split the early
           from the late periods, while its changes show calm and turbulent phases.
- R-RG-02  how many regimes: 2 and 3 are fitted, the lower BIC wins (BIC penalises the extra
           regime's parameters more than AIC, on short series especially).
- R-RG-03  names: the regimes are ordered by their variance on changes ("calm" to "turbulent"),
           by their mean on the level ("low" to "high").
- R-RG-04  at most the last 1,000 periods are fitted (cost), said when it cuts.
- R-RG-05  empty periods are skipped, not interpolated (service): a straight line through a
           weekend would form a fake "calm" regime.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import adfuller

On = Literal["auto", "level", "changes"]

MIN_PERIODS = 40
MAX_PERIODS = 1_000  # R-RG-04
CHOICES = (2, 3)  # R-RG-02
NAMES = {
    "changes": {2: ["calm", "turbulent"], 3: ["calm", "moderate", "turbulent"]},
    "level": {2: ["low", "high"], 3: ["low", "middle", "high"]},
}


@dataclass
class RegimeFit:
    k: int
    aic: float | None
    bic: float | None
    note: str | None = None  # why the fit failed


@dataclass
class Regime:
    name: str
    mean: float
    std: float
    share: float  # share of the periods most likely in this regime
    expected_duration: float | None  # periods: 1 / (1 - p_stay)
    stay: float  # probability to stay in it from one period to the next


@dataclass
class Regimes:
    on: Literal["level", "changes"]
    on_reason: str
    k: int | None = None
    k_reason: str = ""
    start: int = 0  # first period fitted (R-RG-04)
    fits: list[RegimeFit] = field(default_factory=list)
    regimes: list[Regime] = field(default_factory=list)
    transition: list[list[float]] = field(default_factory=list)  # [from][to]
    probabilities: list[list[float]] = field(default_factory=list)  # [regime][period], smoothed
    most_likely: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)  # what was modelled, period by period
    switches: int = 0
    sentence: str | None = None
    message: str | None = None


def _quiet(compute):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute()


def choose_on(values: np.ndarray, on: On) -> tuple[Literal["level", "changes"], str]:
    if on == "level":
        return "level", "The level, as you chose."
    if on == "changes":
        return "changes", "The period changes, as you chose."
    try:
        p = float(_quiet(lambda: adfuller(values, autolag="AIC"))[1])
    except (ValueError, np.linalg.LinAlgError):
        p = 1.0
    if p > 0.05:
        return "changes", (
            f"R-RG-01: the series has a unit root (ADF p = {p:.2g}), so its period changes are "
            "modelled: regimes of a wandering level would only split early from late periods."
        )
    return "level", (
        f"R-RG-01: the series is stationary (ADF p = {p:.2g}), so its level is modelled."
    )


def fit(values: np.ndarray, k: int) -> Any:
    model = MarkovRegression(values, k_regimes=k, trend="c", switching_variance=True)
    return _quiet(lambda: model.fit(disp=False))


def find_regimes(values: np.ndarray, k: int | None = None, on: On = "auto") -> Regimes:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < MIN_PERIODS:
        return Regimes("level", "", message=f"Needs at least {MIN_PERIODS} periods.")
    chosen_on, on_reason = choose_on(values, on)
    modelled = np.diff(values) if chosen_on == "changes" else values
    start = max(0, len(modelled) - MAX_PERIODS)
    modelled = modelled[start:]
    if start:
        on_reason += f" R-RG-04: only the last {MAX_PERIODS:,} periods are fitted (cost)."
    result = Regimes(chosen_on, on_reason, start=start + (1 if chosen_on == "changes" else 0))
    if np.ptp(modelled) == 0:
        result.message = "The series does not vary."
        return result
    scale = float(np.std(modelled)) or 1.0
    scaled = (modelled - modelled.mean()) / scale  # better conditioned; undone below
    fitted: dict[int, Any] = {}
    for candidate in [k] if k else CHOICES:
        try:
            res = fit(scaled, candidate)
            if not np.isfinite(res.llf):
                raise ValueError("the likelihood is not finite")
            fitted[candidate] = res
            result.fits.append(RegimeFit(candidate, float(res.aic), float(res.bic)))
        except (ValueError, np.linalg.LinAlgError) as error:
            result.fits.append(RegimeFit(candidate, None, None, str(error)))
    if not fitted:
        result.message = "The Markov-switching model did not fit this series."
        return result
    best = min(fitted, key=lambda c: fitted[c].bic)
    result.k = best
    if k:
        result.k_reason = f"{k} regimes, as you chose."
    else:
        bics = ", ".join(f"{f.k}: {f.bic:.1f}" for f in result.fits if f.bic is not None)
        result.k_reason = f"R-RG-02: {best} regimes, the lower BIC ({bics})."
    res = fitted[best]
    params = dict(zip(res.model.param_names, np.asarray(res.params), strict=True))
    means = np.array([params[f"const[{i}]"] for i in range(best)]) * scale + modelled.mean()
    stds = np.sqrt([params[f"sigma2[{i}]"] for i in range(best)]) * scale
    probs = np.asarray(res.smoothed_marginal_probabilities)  # periods x regimes
    # statsmodels: regime_transition[to, from]
    matrix = np.asarray(res.regime_transition)[..., 0].T
    order = np.argsort(stds if chosen_on == "changes" else means)  # R-RG-03
    names = NAMES[chosen_on][best]
    likely_raw = probs.argmax(axis=1)
    rank = {int(old): new for new, old in enumerate(order)}
    likely = np.array([rank[int(r)] for r in likely_raw])
    result.most_likely = [int(r) for r in likely]
    result.probabilities = [[float(p) for p in probs[:, old]] for old in order]
    result.transition = [[float(matrix[i, j]) for j in order] for i in order]
    result.values = [float(v) for v in modelled]
    result.switches = int((likely[1:] != likely[:-1]).sum())
    for new, old in enumerate(order):
        stay = float(matrix[old, old])
        result.regimes.append(
            Regime(
                name=names[new],
                mean=float(means[old]),
                std=float(stds[old]),
                share=float((likely == new).mean()),
                expected_duration=float(1 / (1 - stay)) if stay < 1 else None,
                stay=stay,
            )
        )
    result.sentence = _sentence(result)
    return result


def _sentence(result: Regimes) -> str:
    last = result.regimes[result.most_likely[-1]]
    longest = max(result.regimes, key=lambda r: r.expected_duration or float("inf"))
    what = "changes" if result.on == "changes" else "level"
    duration = (
        f"{longest.expected_duration:.0f} periods"
        if longest.expected_duration is not None
        else "the whole series"
    )
    return (
        f"{len(result.regimes)} regimes of the {what}, {result.switches} switches; the series "
        f"ends in the {last.name} regime. A {longest.name} phase lasts {duration} on average."
    )
