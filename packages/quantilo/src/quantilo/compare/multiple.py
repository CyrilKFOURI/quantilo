"""Multiple testing (051): adjust the p-values of a view, pick the method by rule, say what changed.

With 40 tests at p < 0.05, about two pass by chance alone. Benjamini–Hochberg keeps the share of
false discoveries among the results kept under α; Holm and Bonferroni keep the chance of even one
false positive under α (Holm is uniformly more powerful than Bonferroni).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

Correction = Literal["auto", "bh", "holm", "bonferroni"]
Method = Literal["bh", "holm", "bonferroni", "none"]

MIN_TESTS = 10  # R-MT-01: from this many p-values in one view, Benjamini–Hochberg
ALPHA = 0.05
LABELS: dict[Method, str] = {
    "bh": "Benjamini–Hochberg",
    "holm": "Holm",
    "bonferroni": "Bonferroni",
    "none": "no correction",
}
WHAT: dict[Method, str] = {
    "bh": "it keeps the share of false discoveries among the significant results under 5 %",
    "holm": "it keeps the chance of even one false positive under 5 %",
    "bonferroni": "each p-value is multiplied by the number of tests, the strictest reading",
    "none": "",
}


@dataclass
class MultipleTesting:
    method: Method
    label: str  # "Benjamini–Hochberg"
    tests: int  # p-values in the view
    significant: int  # p < α before the correction
    kept: int  # adjusted p < α
    alpha: float
    auto: bool  # chosen by the rule, not by the reader
    reason: str
    sentence: str


def _bh(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(m)
    result[order] = np.minimum(ranked, 1.0)
    return result


def _holm(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    ranked = np.maximum.accumulate(p[order] * (m - np.arange(m)))
    result = np.empty(m)
    result[order] = np.minimum(ranked, 1.0)
    return result


def adjust(p_values: Sequence[float | None], method: Method) -> list[float | None]:
    """Adjusted p-values in the same order; missing ones stay missing and are not counted."""
    known = [k for k, p in enumerate(p_values) if p is not None]
    result: list[float | None] = list(p_values)
    if method == "none" or not known:
        return result
    p = np.array([p_values[k] for k in known], dtype=float)
    if method == "bh":
        adjusted = _bh(p)
    elif method == "holm":
        adjusted = _holm(p)
    else:
        adjusted = np.minimum(p * len(p), 1.0)
    for k, value in zip(known, adjusted, strict=True):
        result[k] = float(value)
    return result


def choose(tests: int, requested: Correction = "auto", alpha: float = ALPHA) -> tuple[Method, str]:
    """The method and why: R-MT-01 (BH from 10 tests), R-MT-02 (Holm below), R-MT-03 (one test)."""
    if tests <= 1:
        return "none", "One test: no correction needed."
    if requested != "auto":
        return requested, f"{LABELS[requested]} chosen: {WHAT[requested]}."
    if tests >= MIN_TESTS:
        chance = f"{tests * alpha:.0f}" if tests * alpha >= 1 else f"{tests * alpha:.1f}"
        return "bh", (
            f"{tests} tests in one view: about {chance} would pass p < {alpha:g} by chance alone. "
            f"Benjamini–Hochberg (automatic from {MIN_TESTS} tests): {WHAT['bh']}."
        )
    return "holm", (
        f"{tests} tests: Holm (automatic below {MIN_TESTS} tests): {WHAT['holm']}, "
        "at little cost with few tests."
    )


def _sentence(
    method: Method, tests: int, significant: int, kept: int, alpha: float, noun: str
) -> str:
    if method == "none":
        return "One test: no correction needed."
    label = LABELS[method]
    if significant == 0:
        return (
            f"No result reaches p < {alpha:g}, before or after correcting for {tests} tests "
            f"({label})."
        )
    if kept == significant:
        every = "The" if significant == 1 else f"All {significant}"
        single = noun[:-1] if significant == 1 and noun.endswith("s") else noun
        verb = "stays" if significant == 1 else "stay"
        return (
            f"{every} significant {single} {verb} significant after correcting for {tests} tests "
            f"({label})."
        )
    verb = "stays" if kept == 1 else "stay"
    return (
        f"{kept} of the {significant} significant {noun} {verb} significant after correcting "
        f"for {tests} tests ({label})."
    )


def correct(
    p_values: Sequence[float | None],
    requested: Correction = "auto",
    alpha: float = ALPHA,
    noun: str = "results",
) -> tuple[list[float | None], MultipleTesting]:
    """Adjust the view's p-values and describe the correction in one sentence (R-MT-06)."""
    known = [p for p in p_values if p is not None]
    method, reason = choose(len(known), requested, alpha)
    adjusted = adjust(p_values, method)
    significant = sum(1 for p in known if p < alpha)
    kept = sum(1 for q in adjusted if q is not None and q < alpha)
    summary = MultipleTesting(
        method,
        LABELS[method],
        len(known),
        significant,
        kept,
        alpha,
        requested == "auto",
        reason,
        _sentence(method, len(known), significant, kept, alpha, noun),
    )
    return adjusted, summary
