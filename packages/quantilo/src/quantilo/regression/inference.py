"""The coefficients of a linear model fitted elsewhere (079: ML's linear regression) read by the
same engine as the Regression tab: least squares, standard errors chosen by rule (HC3 here, as
ML's rows carry no structure the rules could see: R-REG-SE-05), t intervals, and p-values
adjusted over the coefficients as in 051 (Holm 2–9, BH 10 and more). A rank-deficient design
(one-hot levels that add up to the intercept) goes through the pseudo-inverse."""

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from quantilo.compare.multiple import MultipleTesting, correct
from quantilo.regression import covariance as cov
from quantilo.regression.covariance import Arrays


@dataclass
class Inference:
    beta: np.ndarray  # intercept first
    std_errors: np.ndarray
    low: np.ndarray
    high: np.ndarray
    p_values: list[float | None]
    q_values: list[float | None]  # the intercept left out (None)
    residuals: np.ndarray
    df: int
    kind: str  # "hc3"
    label: str
    reason: str
    multiple: MultipleTesting


def arrays(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, Arrays, int]:
    gram = x.T @ x
    bread = np.linalg.pinv(gram)
    beta = bread @ (x.T @ y)
    h = np.einsum("ij,jk,ik->i", x, bread, x)
    rank = int(np.linalg.matrix_rank(gram))
    return beta, Arrays(x, y - x @ beta, bread, h), rank


def coefficients(x: np.ndarray, y: np.ndarray) -> Inference:
    """x has the intercept first."""
    beta, a, rank = arrays(x, y)
    n = a.n
    df = max(n - rank, 1)
    v = cov.hc(a, "hc3")
    se = np.sqrt(np.clip(np.diag(v), 0, None))
    crit = float(stats.t.ppf(0.975, df))
    p: list[float | None] = []
    for b, s in zip(beta, se, strict=True):
        p.append(float(2 * stats.t.sf(abs(b / s), df)) if s > 0 and math.isfinite(s) else None)
    adjusted, summary = correct(p[1:], noun="coefficients")
    return Inference(
        beta,
        se,
        beta - crit * se,
        beta + crit * se,
        p,
        [None, *adjusted],
        a.e,
        df,
        "hc3",
        "HC3",
        (
            "R-REG-SE-05: HC3 standard errors, as in the Regression tab: they stay right when "
            "the spread of the errors changes, and cost nothing when it does not."
        ),
        summary,
    )
