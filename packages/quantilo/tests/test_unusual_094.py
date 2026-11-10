"""094 — Unusual rows and segments on real-world quirks: twin columns, quality flags and codes
typed as numbers are left out of the distance, each with its rule."""

from pathlib import Path

import numpy as np
import pandas as pd

import quantilo as dx
from quantilo.ml import preparation as prep
from quantilo.ml.data import Field
from quantilo.typing.types import Parser

RAW = Parser(kind="raw")


def _frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    borough = rng.choice(["BRONX", "QUEENS", "STATEN ISLAND"], n, p=[0.5, 0.47, 0.03])
    return pd.DataFrame(
        {
            "borough": borough,
            "park_borough": [b.title() for b in borough],  # same content, other spelling
            "SNOW_ATTRIBUTES": rng.choice([",,W,2400", "T,,X,", ",,W,"], n),
            "community_code": rng.integers(101, 118, n).astype(float),
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        }
    )


FIELDS = [
    Field("borough", "categorical_nominal", RAW),
    Field("park_borough", "categorical_nominal", RAW),
    Field("SNOW_ATTRIBUTES", "categorical_nominal", RAW),
    Field("community_code", "numeric_discrete", RAW),
    Field("x", "numeric_continuous", RAW),
    Field("y", "numeric_continuous", RAW),
]


def decision(preparation: prep.Preparation, column: str) -> prep.Decision:
    return next(d for d in preparation.decisions if d.column == column)


def test_twins_flags_and_codes_leave_the_distance_with_their_rule() -> None:
    preparation = prep.decide_distance(_frame(), FIELDS)

    twin = decision(preparation, "park_borough")
    assert not twin.use and twin.reactivable and twin.rules[-1] == "R-DIST-05"
    assert "borough" in twin.why[-1]
    flag = decision(preparation, "SNOW_ATTRIBUTES")
    assert not flag.use and flag.rules == ["R-DIST-06"] and flag.reactivable
    code = decision(preparation, "community_code")
    assert not code.use and code.rules == ["R-DIST-07"] and "code" in code.why[0]
    assert all(decision(preparation, c).use for c in ("borough", "x", "y"))


def test_a_column_left_out_comes_back_when_asked() -> None:
    choices = prep.Choices(
        columns={
            "park_borough": prep.ColumnChoice(use=True),
            "SNOW_ATTRIBUTES": prep.ColumnChoice(use=True),
        }
    )
    preparation = prep.decide_distance(_frame(), FIELDS, choices)

    assert decision(preparation, "park_borough").use
    assert decision(preparation, "SNOW_ATTRIBUTES").use


def test_reasons_do_not_repeat_across_twin_columns(tmp_path: Path) -> None:
    frame = _frame(n=1_500, seed=1)
    ds = dx.from_pandas(frame, "twins", cache_dir=tmp_path / "t")

    found = ds.unusual_rows()

    for row in found.listed[:20]:
        columns = [r.column for r in row.reasons]
        assert not {"borough", "park_borough"} <= set(columns)
        assert "park_borough" not in columns
