"""Least squares (OLS, WLS) by QR, the logistic model of a yes/no target (R-REG-01), and what
every fit hands to the rest: coefficients, residuals, leverage, likelihood."""

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import linalg

from quantilo.regression.covariance import Arrays


@dataclass
class LeastSquares:
    beta: np.ndarray
    arrays: Arrays  # weighted design (√w X), weighted residuals, bread, leverage
    fitted: np.ndarray  # X b, in the target's units
    resid: np.ndarray  # y − X b (unweighted)
    weights: np.ndarray | None
    r2: float
    adj_r2: float
    llf: float
    df_resid: int

    @property
    def n(self) -> int:
        return self.arrays.n

    @property
    def k(self) -> int:
        return self.arrays.k

    @property
    def aic(self) -> float:
        return -2 * self.llf + 2 * self.k

    @property
    def bic(self) -> float:
        return -2 * self.llf + self.k * math.log(self.n)


def least_squares(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> LeastSquares:
    root = np.sqrt(weights) if weights is not None else None
    xw = x * root[:, None] if root is not None else x
    yw = y * root if root is not None else y
    q, r = np.linalg.qr(xw)
    beta = linalg.solve_triangular(r, q.T @ yw)
    inverse = linalg.solve_triangular(r, np.eye(r.shape[0]))
    bread = inverse @ inverse.T
    h = np.einsum("ij,ij->i", q, q)
    ew = yw - xw @ beta
    n, k = x.shape
    sse = float(ew @ ew)
    if weights is not None:
        centre = float(np.average(y, weights=weights))
        tss = float(weights @ (y - centre) ** 2)
    else:
        tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / tss if tss > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - 1) / max(n - k, 1) if tss > 0 else float("nan")
    llf = -n / 2 * (math.log(2 * math.pi) + math.log(max(sse, 1e-300) / n) + 1)
    if weights is not None:
        llf += 0.5 * float(np.log(weights).sum())
    fitted = x @ beta
    return LeastSquares(
        beta, Arrays(xw, ew, bread, h), fitted, y - fitted, weights, r2, adj, llf, n - k
    )


def lstsq_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, w: np.ndarray | None
) -> np.ndarray:
    """Predictions of a fold (a level absent from the training rows gives a rank-deficient
    design: least squares by SVD takes the smallest solution)."""
    root = np.sqrt(w) if w is not None else np.ones(len(y_train))
    beta = np.linalg.lstsq(x_train * root[:, None], y_train * root, rcond=None)[0]
    return x_test @ beta


@dataclass
class Logistic:
    beta: np.ndarray
    cov: np.ndarray
    fitted: np.ndarray  # probabilities
    llf: float
    llnull: float
    ame: np.ndarray  # average marginal effects (non-intercept terms)
    ame_cov: np.ndarray
    separated: bool
    n: int
    k: int

    @property
    def pseudo_r2(self) -> float:
        return 1 - self.llf / self.llnull if self.llnull else float("nan")

    @property
    def aic(self) -> float:
        return -2 * self.llf + 2 * self.k

    @property
    def bic(self) -> float:
        return -2 * self.llf + self.k * math.log(self.n)


def logistic(
    x: np.ndarray,
    y: np.ndarray,
    cov_type: str = "HC0",
    cov_kwds: Mapping[str, object] | None = None,
) -> Logistic:
    """statsmodels Logit; robust sandwich by default (HC0), cluster or HAC by the same rules as
    least squares. Perfect separation (some fitted probabilities 0 or 1) is reported."""
    import statsmodels.api as sm

    separated = False
    with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
        warnings.simplefilter("always")
        model = sm.Logit(y, x)
        result = model.fit(disp=0, maxiter=200, cov_type=cov_type, cov_kwds=dict(cov_kwds or {}))
        p = np.asarray(result.predict(x))
        try:
            margins = result.get_margeff(at="overall", method="dydx")
            ame, ame_cov = np.asarray(margins.margeff), np.asarray(margins.margeff_cov)
        except (ValueError, np.linalg.LinAlgError):
            ame = np.full(x.shape[1] - 1, np.nan)
            ame_cov = np.full((x.shape[1] - 1,) * 2, np.nan)
    for warning in caught:
        if "separation" in str(warning.message).lower():
            separated = True
    if np.mean((p < 1e-8) | (p > 1 - 1e-8)) > 0:
        separated = True
    return Logistic(
        np.asarray(result.params),
        np.asarray(result.cov_params()),
        p,
        float(result.llf),
        float(result.llnull),
        ame,
        ame_cov,
        separated,
        int(x.shape[0]),
        int(x.shape[1]),
    )


def logistic_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    import statsmodels.api as sm

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        result = sm.Logit(y_train, x_train).fit(disp=0, maxiter=100, method="bfgs")
    return np.asarray(result.predict(x_test))
