"""Diagnostics of a least-squares fit (067), in numpy on the weighted arrays, each checked against
statsmodels in the tests: heteroscedasticity (Breusch–Pagan, White), autocorrelation (Durbin–Watson,
Breusch–Godfrey, Ljung–Box), specification (RESET), normality (Jarque–Bera, Shapiro–Wilk),
collinearity (VIF, condition number)."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

SHAPIRO_MAX = 5_000
WHITE_MAX_TERMS = 12  # beyond, White's auxiliary regression has too many columns


@dataclass
class DiagnosticTest:
    name: str
    statistic: float | None
    p_value: float | None
    df: float | None = None


def _r2(y: np.ndarray, x: np.ndarray) -> tuple[float, float, int]:
    """Centred R² of y on x (x holds a constant), the SSR and the rank."""
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    ssr = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    return (1 - ssr / tss if tss > 0 else 0.0), ssr, int(rank)


def breusch_pagan(e: np.ndarray, x: np.ndarray) -> DiagnosticTest:
    """Koenker's studentized version: n R² of e² on X, χ²(k − 1)."""
    r2, _, rank = _r2(e**2, x)
    lm = len(e) * r2
    df = rank - 1
    return DiagnosticTest("Breusch–Pagan", lm, float(stats.chi2.sf(lm, df)) if df > 0 else None, df)


def white(e: np.ndarray, x: np.ndarray) -> DiagnosticTest | None:
    """n R² of e² on X, its squares and cross-products; None beyond WHITE_MAX_TERMS columns."""
    k = x.shape[1]
    if k > WHITE_MAX_TERMS + 1:
        return None
    i, j = np.triu_indices(k)
    products = x[:, i] * x[:, j]  # the constant times each column gives the columns themselves
    r2, _, rank = _r2(e**2, products)
    lm = len(e) * r2
    df = rank - 1
    return DiagnosticTest("White", lm, float(stats.chi2.sf(lm, df)) if df > 0 else None, df)


def durbin_watson(e: np.ndarray) -> float:
    return float((np.diff(e) ** 2).sum() / (e @ e))


def breusch_godfrey(e: np.ndarray, x: np.ndarray, lags: int) -> DiagnosticTest:
    """n R² of e on X and its first `lags` lags (zeros before the start), χ²(lags)."""
    n = len(e)
    lagged = np.column_stack(
        [np.concatenate([np.zeros(p), e[: n - p]]) for p in range(1, lags + 1)]
    )
    r2, _, _ = _r2(e, np.column_stack([x, lagged]))
    lm = n * r2
    return DiagnosticTest("Breusch–Godfrey", lm, float(stats.chi2.sf(lm, lags)), lags)


def ljung_box(e: np.ndarray, lags: int) -> DiagnosticTest:
    n = len(e)
    centred = e - e.mean()
    denominator = centred @ centred
    rho = np.array([centred[p:] @ centred[:-p] / denominator for p in range(1, lags + 1)])
    q = float(n * (n + 2) * (rho**2 / (n - np.arange(1, lags + 1))).sum())
    return DiagnosticTest("Ljung–Box", q, float(stats.chi2.sf(q, lags)), lags)


def reset(y: np.ndarray, x: np.ndarray, fitted: np.ndarray) -> DiagnosticTest:
    """Ramsey: F test of ŷ² and ŷ³ added to X (powers of the standardized ŷ, same span)."""
    spread = fitted.std()
    if spread == 0:
        return DiagnosticTest("RESET", 0.0, None)
    z = (fitted - fitted.mean()) / spread
    _, ssr_r, rank_r = _r2(y, x)
    _, ssr_u, rank_u = _r2(y, np.column_stack([x, z**2, z**3]))
    q = rank_u - rank_r
    df = len(y) - rank_u
    if q <= 0 or df <= 0 or ssr_u <= 0:
        return DiagnosticTest("RESET", 0.0, None)
    f = (ssr_r - ssr_u) / q / (ssr_u / df)
    return DiagnosticTest("RESET", float(f), float(stats.f.sf(f, q, df)), q)


def bending(resid: np.ndarray, columns: dict[str, np.ndarray]) -> str | None:
    """The column whose square explains most of what is left (R-REG-10)."""
    best, gain = None, 0.0
    for name, values in columns.items():
        spread = values.std()
        if spread == 0 or len(np.unique(values[:2000])) < 5:
            continue
        z = (values - values.mean()) / spread
        base = np.column_stack([np.ones_like(z), z])
        r_linear, _, _ = _r2(resid, base)
        r_square, _, _ = _r2(resid, np.column_stack([base, z**2]))
        if r_square - r_linear > gain:
            best, gain = name, r_square - r_linear
    return best


@dataclass
class Normality:
    jarque_bera: DiagnosticTest
    shapiro: DiagnosticTest | None
    skewness: float
    kurtosis: float  # excess


def normality(e: np.ndarray) -> Normality:
    jb: Any = stats.jarque_bera(e)
    shapiro = None
    if 3 <= len(e) <= SHAPIRO_MAX:
        found = stats.shapiro(e)
        shapiro = DiagnosticTest("Shapiro–Wilk", float(found.statistic), float(found.pvalue))
    return Normality(
        DiagnosticTest("Jarque–Bera", float(jb.statistic), float(jb.pvalue), 2),
        shapiro,
        float(stats.skew(e)),
        float(stats.kurtosis(e)),
    )


@dataclass
class Vif:
    term: str
    vif: float | None
    partner: str | None  # the column it is most correlated with
    correlation: float | None


def collinearity(x: np.ndarray, terms: list[str]) -> tuple[list[Vif], float]:
    """VIF of every non-intercept term (diagonal of the inverse correlation matrix) and the
    condition number of the columns scaled to unit length (Belsley), intercept included."""
    found: list[Vif] = []
    body = x[:, 1:]
    if body.shape[1] >= 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.atleast_2d(np.corrcoef(body, rowvar=False))
        r = np.nan_to_num(r)
        try:
            inverse = np.linalg.inv(r)
            vifs = np.diag(inverse)
        except np.linalg.LinAlgError:
            vifs = np.full(body.shape[1], np.inf)
        for j, term in enumerate(terms[1:]):
            partner = correlation = None
            if body.shape[1] > 1:
                row = np.abs(r[j]).copy()
                row[j] = -1
                other = int(np.argmax(row))
                partner, correlation = terms[1 + other], float(r[j, other])
            value = float(vifs[j]) if np.isfinite(vifs[j]) else None
            found.append(Vif(term, value, partner, correlation))
    norms = np.linalg.norm(x, axis=0)
    scaled = x / np.where(norms > 0, norms, 1)
    eigen = np.linalg.eigvalsh(scaled.T @ scaled)
    condition = float(np.sqrt(eigen[-1] / eigen[0])) if eigen[0] > 1e-300 else float("inf")
    return found, condition
