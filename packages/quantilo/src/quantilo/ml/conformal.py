"""Prediction intervals of a number (057), guaranteed by conformal prediction.

The held-out rows (test rows, or the out-of-fold rows of a cross-validation) were never used to fit
the model: their errors tell how far a new prediction may fall. Two ways:

- split conformal ★: the same width for every row, the errors' quantile at ⌈(n + 1) level⌉ / n
  (Vovk et al., 2005; Lei et al., 2018). Free: no model is fitted again.
- CQR, conformalized quantile regression (Romano et al., 2019): two quantile models (gradient
  boosting, quantile loss) give a band that widens where the target spreads; the held-out rows
  then widen or narrow it by one quantity, so the level is kept. Two more fits: Advanced > More.

The coverage shown is checked honestly: half of the held-out rows set the width, the other half,
never seen by that width, says how many fell inside. The interval used on new data is then set on
every held-out row (more errors, a steadier width).

Rules, each with its reason:
- R-ML-CONF-01: at least 20 held-out rows, and enough that each half can reach the level (38 at
  95 %, 198 at 99 %): the ⌈(n + 1) level⌉-th error must exist; below, no interval.
- R-ML-CONF-02: when the held-out rows follow time (time split, panel walk-forward), the earlier
  half sets the width and the later half checks it: a random half would hide a drift.
- R-ML-CONF-03: the coverage found is compared with the level within two binomial standard errors
  of the checked rows; below, the page warns that the rows may not be alike (exchangeability).
- R-ML-CONF-04: CQR needs test rows: under cross-validation it falls back to split conformal.
- R-ML-CONF-05: CQR's quantile models are fitted on the prepared training rows; beyond 2 000
  prepared inputs it falls back to split conformal (the band would take too long to learn).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

LEVEL = 0.9  # the default coverage (90 %)
MIN_ROWS = 20  # R-ML-CONF-01
MAX_CQR_INPUTS = 2_000  # R-ML-CONF-05
CQR_ITERATIONS = 200
SEED = 42

Method = Literal["split", "cqr"]


@dataclass
class ConformalResult:
    """A number's prediction interval and how well it held on rows it never saw."""

    method: Method
    level: float
    half_width: float  # split: ± around the prediction; CQR: added to each side of the band
    rows: int  # held-out rows the interval for new data is set on
    calibration_rows: int  # rows that set the width in the check
    check_rows: int  # rows the coverage is measured on
    coverage: float | None  # share of the check rows inside their interval
    mean_width: float | None  # mean width of the check rows' intervals (high - low)
    order: Literal["random", "time"]
    tolerance: float  # R-ML-CONF-03: two binomial standard errors at the level
    rules: list[str] = field(default_factory=list)  # the rules applied, with their reason

    @property
    def below(self) -> bool:
        """The coverage fell short of the level by more than chance allows (R-ML-CONF-03)."""
        return self.coverage is not None and self.coverage < self.level - self.tolerance


class Bands:
    """CQR's lower and upper quantile models, applied after the pipeline's preparation steps:
    kept on the fitted pipeline, so the interval of new rows is the same one."""

    def __init__(self, lower: Any, upper: Any, half_width: float) -> None:
        self.lower = lower
        self.upper = upper
        self.half_width = half_width

    def band(self, prepared: Any) -> tuple[np.ndarray, np.ndarray]:
        """(low, high) before the conformal widening, sorted so low ≤ high."""
        matrix = _dense(prepared)
        with _one_thread():
            low = np.asarray(self.lower.predict(matrix), dtype=float)
            high = np.asarray(self.upper.predict(matrix), dtype=float)
        return np.minimum(low, high), np.maximum(low, high)

    def interval(self, prepared: Any) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.band(prepared)
        return low - self.half_width, high + self.half_width


BANDS_ATTRIBUTE = "quantilo_bands_"  # where a pipeline keeps its CQR bands


def quantile(scores: np.ndarray, level: float) -> float:
    """The conformal quantile: the ⌈(n + 1) level⌉-th smallest score (infinite when n is too
    small for the level: the interval would be the whole line)."""
    n = len(scores)
    rank = math.ceil((n + 1) * level)
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


def needed(level: float) -> int:
    """Held-out rows needed at this level (R-ML-CONF-01): each half holds level / (1 - level)."""
    return max(MIN_ROWS, 2 * math.ceil(level / (1 - level) - 1e-9))


def halves(n: int, chronological: bool, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Calibration and check positions: the earlier and later halves in time (R-ML-CONF-02),
    else a random half each."""
    order = np.arange(n) if chronological else np.random.default_rng(seed).permutation(n)
    cut = n // 2
    return order[:cut], order[cut:]


def tolerance(level: float, rows: int) -> float:
    return 2 * math.sqrt(level * (1 - level) / max(rows, 1))


def split(
    actual: np.ndarray,
    predicted: np.ndarray,
    level: float = LEVEL,
    chronological: bool = False,
    seed: int = SEED,
) -> ConformalResult | None:
    """Split conformal on the held-out rows (in time order when `chronological`)."""
    keep = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[keep], predicted[keep]
    errors = np.abs(actual - predicted)
    if len(errors) < needed(level):
        return None
    calibrate, check = halves(len(errors), chronological, seed)
    width = quantile(errors[calibrate], level)
    inside = errors[check] <= width
    final = quantile(errors, level)
    rules = [_rule_rows(len(errors)), _rule_order(chronological)]
    return ConformalResult(
        "split",
        level,
        final,
        len(errors),
        len(calibrate),
        len(check),
        float(inside.mean()),
        2 * width if math.isfinite(width) else None,
        "time" if chronological else "random",
        tolerance(level, len(check)),
        rules,
    )


def fit_bands(prepared_train: Any, y_train: np.ndarray, level: float, seed: int = SEED) -> Any:
    """The two quantile models of CQR, at (1 - level) / 2 and (1 + level) / 2."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    alpha = (1 - level) / 2
    matrix = _dense(prepared_train)
    models = []
    # one thread: on small bands, OpenMP threads cost more than they save (and the browser
    # version has a single core anyway)
    with _one_thread():
        for q in (alpha, 1 - alpha):
            model = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=CQR_ITERATIONS, random_state=seed
            )
            model.fit(matrix, y_train)
            models.append(model)
    return models[0], models[1]


def cqr(
    lower: Any,
    upper: Any,
    prepared_held: Any,
    actual: np.ndarray,
    level: float = LEVEL,
    chronological: bool = False,
    seed: int = SEED,
) -> tuple[ConformalResult, Bands] | None:
    """Conformalized quantile regression on the held-out rows: the scores are how far each answer
    lies outside its band, max(low - y, y - high)."""
    bands = Bands(lower, upper, 0.0)
    low, high = bands.band(prepared_held)
    keep = np.isfinite(actual) & np.isfinite(low) & np.isfinite(high)
    actual, low, high = actual[keep], low[keep], high[keep]
    if len(actual) < needed(level):
        return None
    scores = np.maximum(low - actual, actual - high)
    calibrate, check = halves(len(scores), chronological, seed)
    width = quantile(scores[calibrate], level)
    inside = scores[check] <= width
    final = quantile(scores, level)
    bands.half_width = final
    widths = (high[check] - low[check]) + 2 * width
    rules = [
        _rule_rows(len(scores)),
        _rule_order(chronological),
        "CQR: two quantile models (gradient boosting) give a band that widens where the "
        "target spreads; the held-out rows widen or narrow it so the level is kept.",
    ]
    return (
        ConformalResult(
            "cqr",
            level,
            final,
            len(scores),
            len(calibrate),
            len(check),
            float(inside.mean()),
            float(np.mean(widths)) if math.isfinite(width) else None,
            "time" if chronological else "random",
            tolerance(level, len(check)),
            rules,
        ),
        bands,
    )


def fallback_rule(reason: Literal["cross", "inputs"]) -> str:
    if reason == "cross":
        return (
            "R-ML-CONF-04: adaptive intervals (CQR) need test rows; under cross-validation the "
            "interval has the same width for every row (split conformal)."
        )
    return (
        f"R-ML-CONF-05: more than {MAX_CQR_INPUTS:,} prepared inputs: the quantile band would "
        "take too long to learn, so the interval has the same width for every row."
    )


def _rule_rows(rows: int) -> str:
    return (
        f"R-ML-CONF-01: the width is set on {rows:,} held-out errors, rows the model never saw; "
        "half of them set it for the check, the other half measure the coverage."
    )


def _rule_order(chronological: bool) -> str:
    if chronological:
        return (
            "R-ML-CONF-02: the held-out rows follow time: the earlier half sets the width, the "
            "later half checks it, so a drift shows as a lower coverage."
        )
    return "R-ML-CONF-02: the held-out rows are split at random into the two halves."


def _one_thread() -> Any:
    """OpenMP held to one thread (threadpoolctl comes with scikit-learn, Pyodide's too)."""
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError:  # pragma: no cover
        return contextlib.nullcontext()
    return threadpool_limits(limits=1)


def _dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)
