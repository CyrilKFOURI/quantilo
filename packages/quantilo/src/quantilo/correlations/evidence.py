"""How much to trust a link: effect size (Cohen) crossed with the p-value adjusted over pairs."""

from typing import Literal

import numpy as np
from scipy import stats

Size = Literal["negligible", "small", "medium", "large"]
Evidence = Literal["strong", "moderate", "weak", "none"]

# Cohen's conventions. η² = 0.01 / 0.06 / 0.14 gives η = 0.10 / 0.24 / 0.37;
# r and Cramér's V share 0.1 / 0.3 / 0.5 (V at one degree of freedom, the strictest reading).
SIZES: dict[str, tuple[float, float, float]] = {
    "pearson": (0.1, 0.3, 0.5),
    "spearman": (0.1, 0.3, 0.5),
    "cramers_v": (0.1, 0.3, 0.5),
    "eta": (0.10, 0.24, 0.37),
}
SIGNIFICANT = 0.05
HIGHLY_SIGNIFICANT = 0.001


def effect_size(metric: str, value: float) -> Size:
    small, medium, large = SIZES[metric]
    magnitude = abs(value)
    if magnitude >= large:
        return "large"
    if magnitude >= medium:
        return "medium"
    return "small" if magnitude >= small else "negligible"


def adjusted(p_values: list[float | None]) -> list[float | None]:
    """Benjamini-Hochberg: with hundreds of pairs, about 5 % pass p < 0.05 by chance alone."""
    known = [k for k, p in enumerate(p_values) if p is not None]
    result: list[float | None] = [None] * len(p_values)
    if known:
        q = stats.false_discovery_control(np.array([p_values[k] for k in known], dtype=float))
        for k, value in zip(known, q, strict=True):
            result[k] = float(value)
    return result


def grade(size: Size, q: float | None) -> Evidence:
    """Strong: large and very unlikely by chance. Moderate: at least medium and significant.
    Weak: worth a look but either small (real yet minor) or possibly chance."""
    if q is None or size == "negligible":
        return "none"
    if size == "large" and q < HIGHLY_SIGNIFICANT:
        return "strong"
    if size in ("medium", "large"):
        return "moderate" if q < SIGNIFICANT else "weak"
    return "weak" if q < SIGNIFICANT else "none"
