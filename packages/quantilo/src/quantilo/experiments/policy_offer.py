"""Whether "who should get which variant" (spec 055 § B, R-PL1) can be computed for an
analysis. No scikit-learn here: the analysis says it at once, the rule is learnt in policy.py."""

from dataclasses import dataclass

VALUE = "__value"
MIN_UNITS = 200  # per variant (R-PL1)
MIN_COVARIATES = 2


@dataclass
class PolicyOffer:
    """Whether "who should get which variant" can be computed for an analysis (R-PL1)."""

    available: bool
    reason: str | None
    covariates: list[str]  # candidate columns


def offer(kind: str, candidates: list[str], units: dict[str, int]) -> PolicyOffer:
    """R-PL1: a rate, mean or count; at least 2 candidate covariates; 200 units per variant."""
    if kind == "ratio":
        return PolicyOffer(
            False,
            "A ratio of sums has no value per unit: choose a rate, a mean or a count as the "
            "primary metric to learn who should get which variant.",
            candidates,
        )
    if len(candidates) < MIN_COVARIATES:
        found = f"only {candidates[0]}" if candidates else "none"
        return PolicyOffer(
            False,
            f"Learning who should get which variant needs at least {MIN_COVARIATES} columns "
            f"describing the units before the test (age, device, country…): {found} here.",
            candidates,
        )
    small = [label for label, count in units.items() if count < MIN_UNITS]
    if small:
        return PolicyOffer(
            False,
            f"Fewer than {MIN_UNITS} units in {', '.join(small)}: a rule learnt on so few units "
            "would mostly fit noise.",
            candidates,
        )
    return PolicyOffer(True, None, candidates)
