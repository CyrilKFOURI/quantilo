"""Covariance of the coefficients (067): classic, HC0–HC3, HAC, cluster one and two ways,
Driscoll–Kraay. All sandwiches share one bread, (X'X)⁻¹, so every kind can be shown side by side
from a single fit. Each one is checked against statsmodels in the tests.

The arrays are those of the (weighted) least squares: for WLS, X and the residuals are already
multiplied by √w, which turns it into OLS.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Kind = Literal["classic", "hc0", "hc1", "hc2", "hc3", "hac", "cluster", "cluster2", "dk"]
LABELS: dict[Kind, str] = {
    "classic": "Classic",
    "hc0": "HC0 (White)",
    "hc1": "HC1",
    "hc2": "HC2",
    "hc3": "HC3",
    "hac": "HAC (Newey–West)",
    "cluster": "Clustered",
    "cluster2": "Two-way clustered",
    "dk": "Driscoll–Kraay",
}


@dataclass
class Arrays:
    """What every sandwich needs: the design, the residuals, the bread and the leverage."""

    x: np.ndarray  # n × k
    e: np.ndarray  # n
    bread: np.ndarray  # (X'X)⁻¹, k × k
    h: np.ndarray  # leverage, n

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def k(self) -> int:
        return int(self.x.shape[1])


def newey_west_lag(t: int) -> int:
    """⌊4 (T/100)^(2/9)⌋ (Newey & West 1994), at least 1."""
    return max(1, math.floor(4 * (t / 100) ** (2 / 9)))


def _sandwich(a: Arrays, meat: np.ndarray) -> np.ndarray:
    v = a.bread @ meat @ a.bread
    return (v + v.T) / 2


def classic(a: Arrays) -> np.ndarray:
    s2 = float(a.e @ a.e) / max(a.n - a.k, 1)
    return s2 * a.bread


def hc(a: Arrays, kind: Literal["hc0", "hc1", "hc2", "hc3"]) -> np.ndarray:
    e2 = a.e**2
    if kind == "hc2":
        e2 = e2 / np.clip(1 - a.h, 1e-12, None)
    elif kind == "hc3":
        e2 = e2 / np.clip(1 - a.h, 1e-12, None) ** 2
    meat = (a.x * e2[:, None]).T @ a.x
    v = _sandwich(a, meat)
    return v * a.n / max(a.n - a.k, 1) if kind == "hc1" else v


def _bartlett(u: np.ndarray, lag: int) -> np.ndarray:
    """Σ_t u_t u_t' + Σ_l (1 − l/(L+1)) (Γ_l + Γ_l'), rows of `u` in time order."""
    s = u.T @ u
    for lagged in range(1, min(lag, len(u) - 1) + 1):
        gamma = u[lagged:].T @ u[:-lagged]
        s += (1 - lagged / (lag + 1)) * (gamma + gamma.T)
    return s


def hac(a: Arrays, lag: int) -> np.ndarray:
    """Newey–West with Bartlett weights; rows must be in time order. Small-sample n/(n−k)."""
    u = a.x * a.e[:, None]
    return _sandwich(a, _bartlett(u, lag)) * a.n / max(a.n - a.k, 1)


def codes_of(groups: np.ndarray) -> tuple[np.ndarray, int]:
    _, codes = np.unique(groups, return_inverse=True)
    return codes.astype(np.int64), int(codes.max()) + 1 if len(codes) else 0


def _sums(u: np.ndarray, codes: np.ndarray, count: int) -> np.ndarray:
    """Row sums of `u` within each group (count × k), in code order."""
    out = np.empty((count, u.shape[1]))
    for j in range(u.shape[1]):
        out[:, j] = np.bincount(codes, weights=u[:, j], minlength=count)
    return out


def _cluster_raw(a: Arrays, codes: np.ndarray, count: int) -> np.ndarray:
    u = _sums(a.x * a.e[:, None], codes, count)
    v = _sandwich(a, u.T @ u)
    return v * count / max(count - 1, 1) * (a.n - 1) / max(a.n - a.k, 1)


def cluster(a: Arrays, groups: np.ndarray) -> np.ndarray:
    """CR1: G/(G−1) · (n−1)/(n−k)."""
    codes, count = codes_of(groups)
    return _cluster_raw(a, codes, count)


def _psd(v: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(v)
    if values.min() >= 0:
        return v
    return (vectors * np.clip(values, 0, None)) @ vectors.T


def cluster2(a: Arrays, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Cameron–Gelbach–Miller: V₁ + V₂ − V₁∩₂, each with its own CR1 correction; a negative
    eigenvalue (possible in small samples) is set to 0 (Cameron, Gelbach & Miller 2011)."""
    c1, g1 = codes_of(first)
    c2, g2 = codes_of(second)
    both, g12 = codes_of(c1 * (int(c2.max()) + 1) + c2)
    v = _cluster_raw(a, c1, g1) + _cluster_raw(a, c2, g2) - _cluster_raw(a, both, g12)
    return _psd((v + v.T) / 2)


def driscoll_kraay(a: Arrays, time: np.ndarray, lag: int) -> np.ndarray:
    """Sum the scores over each period, then Newey–West on those sums (periods in order).
    Small-sample n/(n−k), as statsmodels' "hac" correction."""
    codes, count = codes_of(time)
    u = _sums(a.x * a.e[:, None], codes, count)
    return _sandwich(a, _bartlett(u, lag)) * a.n / max(a.n - a.k, 1)
