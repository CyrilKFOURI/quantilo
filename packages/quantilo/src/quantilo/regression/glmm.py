"""Generalised mixed model (076, second part): a random intercept by group for a yes/no target
(logistic) or a count (Poisson), by maximum likelihood with adaptive Gauss–Hermite quadrature
(Pinheiro & Bates 1995): each group's integral is taken around its own mode, so large groups
stay exact where plain quadrature fails. Rule R-REG-MIX-02.

numpy / scipy on one thread: Pyodide-compatible."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize, special

Link = Literal["logit", "poisson"]
NODES = 15  # R-REG-MIX-02: quadrature points a group
NEWTON = 25  # iterations to each group's mode
STEP = 1e-4  # finite differences of the Hessian (on the log-likelihood)


@dataclass
class GlmmFit:
    beta: np.ndarray
    tau: float  # standard deviation of the random intercept
    cov: np.ndarray  # of (β, log τ)
    llf: float
    converged: bool


def _row_ll(link: Link, y: np.ndarray, eta: np.ndarray) -> np.ndarray:
    if link == "logit":
        return y * eta - np.logaddexp(0.0, eta)
    return y * eta - np.exp(np.clip(eta, -700, 700)) - special.gammaln(y + 1)


def _mean_weight(link: Link, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if link == "logit":
        mu = special.expit(eta)
        return mu, mu * (1 - mu)
    mu = np.exp(np.clip(eta, -700, 700))
    return mu, mu


def loglike(
    link: Link,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    beta: np.ndarray,
    tau: float,
    offset: np.ndarray | None = None,
    nodes: int = NODES,
) -> float:
    """Σ_g log ∫ Π_{i∈g} p(y_i | x_i′β + u) φ(u; 0, τ²) du, adaptive Gauss–Hermite; `groups` are
    codes 0…G−1."""
    g = int(groups.max()) + 1
    eta = x @ beta + (offset if offset is not None else 0.0)
    tau2 = max(tau, 1e-8) ** 2
    u = np.zeros(g)
    for _ in range(NEWTON):  # the mode of each group's integrand
        mu, w = _mean_weight(link, eta + u[groups])
        grad = np.bincount(groups, y - mu, g) - u / tau2
        hess = -np.bincount(groups, w, g) - 1 / tau2
        step = grad / hess
        u -= step
        if np.abs(step).max() < 1e-10:
            break
    _, w = _mean_weight(link, eta + u[groups])
    spread = 1 / np.sqrt(np.bincount(groups, w, g) + 1 / tau2)
    z, weights = np.polynomial.hermite.hermgauss(nodes)
    terms = np.empty((nodes, g))
    for q in range(nodes):
        at = u + math.sqrt(2) * spread * z[q]
        rows = np.bincount(groups, _row_ll(link, y, eta + at[groups]), g)
        prior = -0.5 * at**2 / tau2 - 0.5 * math.log(2 * math.pi * tau2)
        terms[q] = math.log(weights[q]) + z[q] ** 2 + rows + prior
    return float((special.logsumexp(terms, axis=0) + np.log(math.sqrt(2) * spread)).sum())


def fit(
    link: Link,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    start: np.ndarray,
    offset: np.ndarray | None = None,
) -> GlmmFit:
    """(β, log τ) by L-BFGS from the GLM's β and τ = 0.5; the covariance from the numerical
    Hessian of the log-likelihood."""
    _, groups = np.unique(groups, return_inverse=True)
    k = x.shape[1]

    def negative(theta: np.ndarray) -> float:
        value = loglike(link, x, y, groups, theta[:k], math.exp(theta[k]), offset)
        return -value if math.isfinite(value) else 1e300

    found = optimize.minimize(negative, np.r_[start, math.log(0.5)], method="L-BFGS-B")
    theta = np.asarray(found.x, dtype=float)
    hessian = _hessian(negative, theta)
    try:
        cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hessian)
    return GlmmFit(theta[:k], math.exp(theta[k]), cov, -float(found.fun), bool(found.success))


def _hessian(function, theta: np.ndarray) -> np.ndarray:
    p = len(theta)
    h = STEP * np.maximum(np.abs(theta), 1.0)
    out = np.empty((p, p))
    base = function(theta)
    for i in range(p):
        for j in range(i, p):
            if i == j:
                up, down = theta.copy(), theta.copy()
                up[i] += h[i]
                down[i] -= h[i]
                out[i, i] = (function(up) - 2 * base + function(down)) / h[i] ** 2
                continue
            values = []
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                at = theta.copy()
                at[i] += si * h[i]
                at[j] += sj * h[j]
                values.append(function(at))
            out[i, j] = out[j, i] = (values[0] - values[1] - values[2] + values[3]) / (
                4 * h[i] * h[j]
            )
    return (out + out.T) / 2
