"""094 — Experiment candidates: observed categories (sex, region, smoker) are not offered as an
experiment's variants; a variant needs a telling name or a planned, balanced split."""

from pathlib import Path

import numpy as np
import pandas as pd

import quantilo as dx


def test_observed_categories_are_not_suggested_as_variants(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 1_000
    frame = pd.DataFrame(
        {
            "sex": rng.choice(["female", "male"], n),
            "smoker": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
            "region": rng.choice(["north", "south", "east", "west"], n),
            "charges": rng.gamma(2, 5_000, n),
        }
    )

    found = dx.from_pandas(frame, "insurance", cache_dir=tmp_path / "i").experiment_candidates()

    assert found.suggested is None
    assert found.variants and all(not v.planned for v in found.variants)
    smoker = next(v for v in found.variants if v.column == "smoker")
    assert "uneven" in smoker.why
    assert found.note is not None and "R-EXP-01" in found.note and "smoker" in found.note


def test_a_telling_name_or_a_planned_split_is_a_variant(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 1_000
    frame = pd.DataFrame(
        {
            "smoker": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
            "arm": rng.choice(["A", "B"], n),
            "converted": rng.random(n) < 0.1,
        }
    )

    found = dx.from_pandas(frame, "ab", cache_dir=tmp_path / "a").experiment_candidates()

    assert found.variants[0].column == "arm" and found.variants[0].planned
    assert "name" in found.variants[0].why
    assert found.suggested is not None and found.suggested.variant == "arm"
    assert found.note is None
