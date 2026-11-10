"""Influential rows (067): leverage, Cook's distance and studentized residuals on every row used,
DFBETAS on the most influential ones only (study § 3.5)."""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from quantilo.regression.covariance import Arrays

TOP_ROWS = 20
ALPHA = 0.05


@dataclass
class Influence:
    leverage: np.ndarray
    cook: np.ndarray
    studentized: np.ndarray  # external (deleted) studentized residuals
    bonferroni: np.ndarray  # outlier p-value of each row, Bonferroni over the n rows
    cook_threshold: float  # 4 / n

    @property
    def influential(self) -> int:
        return int((self.cook > self.cook_threshold).sum())

    @property
    def outliers(self) -> int:
        return int((self.bonferroni < ALPHA).sum())


def influence(a: Arrays) -> Influence:
    n, k = a.n, a.k
    df = n - k
    s2 = float(a.e @ a.e) / max(df, 1)
    h = np.clip(a.h, 0, 1 - 1e-12)
    internal = a.e / np.sqrt(s2 * (1 - h))
    cook = internal**2 * h / (k * (1 - h))
    with np.errstate(invalid="ignore", divide="ignore"):
        external = internal * np.sqrt((df - 1) / np.clip(df - internal**2, 1e-12, None))
    p = 2 * stats.t.sf(np.abs(external), max(df - 1, 1))
    return Influence(h, cook, external, np.minimum(p * n, 1.0), 4 / n)


def top_rows(found: Influence, count: int = TOP_ROWS) -> np.ndarray:
    """Positions of the most influential rows, by Cook's distance."""
    order = np.argsort(-found.cook, kind="stable")
    return order[: min(count, len(order))]


def dfbetas(a: Arrays, found: Influence, rows: np.ndarray) -> np.ndarray:
    """(b − b₍ᵢ₎) / (s₍ᵢ₎ √[(X'X)⁻¹]ⱼⱼ) for the given rows (rows × k)."""
    df = a.n - a.k
    s2 = float(a.e @ a.e) / max(df, 1)
    h = np.clip(found.leverage[rows], 0, 1 - 1e-12)
    internal2 = a.e[rows] ** 2 / (s2 * (1 - h))
    s_deleted = np.sqrt(s2 * np.clip(df - internal2, 1e-12, None) / max(df - 1, 1))
    change = (a.bread @ a.x[rows].T).T * (a.e[rows] / (1 - h))[:, None]
    return change / (s_deleted[:, None] * np.sqrt(np.diag(a.bread))[None, :])
