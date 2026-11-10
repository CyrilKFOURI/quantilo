"""094 — Correlation insights and structure on real-world quirks: codes typed as numbers,
links carried by a repeated sentinel value, sparse panels."""

from pathlib import Path

import numpy as np
import pandas as pd

import quantilo as dx
from quantilo.correlations.matrix import Matrix, MatrixMember
from quantilo.correlations.structure import structure_rule


def _ds(frame: pd.DataFrame, tmp_path: Path, name: str) -> dx.Dataset:
    return dx.from_pandas(frame, name, cache_dir=tmp_path / name)


def test_codes_give_no_link_insight_but_one_note_and_stay_out_of_pca(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 800
    block = rng.integers(1, 9_000, n)
    frame = pd.DataFrame(
        {
            "bin__": block * 10 + rng.integers(0, 3, n),  # a building code, tied to the block
            "block_code": block,
            "a": rng.normal(0, 1, n),
        }
    )
    frame["b"] = frame["a"] + rng.normal(0, 0.3, n)
    frame["c"] = frame["a"] - frame["b"] + rng.normal(0, 0.3, n)
    ds = _ds(frame, tmp_path, "codes")

    found = ds.correlations()

    texts = [i.text for i in found.insights]
    linked = [i for i in found.insights if i.kind in ("correlated", "redundant")]
    assert linked and not any({"bin__", "block_code"} & set(i.columns) for i in linked)
    note = next(i for i in found.insights if i.kind == "codes")
    assert "bin__" in note.text and "block_code" in note.text and "R-COR-01" in note.text
    assert any("a" in t for t in texts)
    structure = ds.structure()
    assert "bin__" not in structure.columns
    assert any(o.name == "bin__" and "code" in o.reason for o in structure.left_out)


def test_a_link_carried_by_a_sentinel_sends_to_cleaning(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 2_000
    t = rng.normal(15, 5, n)
    ah = rng.normal(1, 0.3, n)  # unrelated to t
    missing = rng.random(n) < 0.08
    t[missing] = -200
    ah[missing] = -200
    ds = _ds(pd.DataFrame({"T": t, "AH": ah}), tmp_path, "sentinel")

    found = ds.correlations()

    assert not any(i.kind == "redundant" for i in found.insights)
    spike = next(i for i in found.insights if i.kind == "spike")
    assert "-200" in spike.text and "R-COR-02" in spike.text and "Clean" in spike.text


def test_a_real_strong_link_is_still_a_finding(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 1_000
    x = rng.normal(0, 1, n)
    ds = _ds(pd.DataFrame({"x": x, "y": x + rng.normal(0, 0.05, n)}), tmp_path, "real")

    found = ds.correlations()

    assert any(i.kind == "redundant" for i in found.insights)
    assert not any(i.kind in ("spike", "codes") for i in found.insights)


def _matrix(present: list[int], rows: int, strong: list[tuple[str, str]]) -> Matrix:
    names = [f"m{i}" for i in range(len(present))]
    size = len(names)
    n = [[0] * size for _ in range(size)]
    for i, count in enumerate(present):
        n[i][i] = count
    from quantilo.correlations.matrix import Pair

    pairs = [Pair(a, b, "pearson", 0.8, 0.0, rows) for a, b in strong]
    return Matrix(
        "pearson",
        [MatrixMember(name, "numeric_continuous", "num") for name in names],
        [[None] * size for _ in range(size)],
        [[None] * size for _ in range(size)],
        n,
        list(range(size)),
        rows,
        rows,
        False,
        pairs=pairs,
    )


def test_structure_is_not_proposed_when_most_numbers_are_too_empty() -> None:
    # 10 numbers, 8 of them 60 % empty: the PCA would keep 2 columns
    matrix = _matrix([1000, 1000] + [400] * 8, 1000, [("m2", "m3"), ("m4", "m5"), ("m6", "m7")])

    rule = structure_rule(matrix)

    assert not rule.proposed and "empty" in rule.reason
    full = _matrix([1000] * 6, 1000, [("m0", "m1"), ("m2", "m3"), ("m4", "m5")])
    assert structure_rule(full).proposed


def test_a_real_zero_is_not_a_sentinel(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 1_500
    upstairs = rng.random(n) < 0.6
    second_floor = np.where(upstairs, rng.normal(800, 200, n).clip(150), 0.0)
    half_bath = np.where(upstairs, rng.integers(0, 2, n) + 1, 0).astype(float)
    frame = pd.DataFrame({"second_floor_sf": second_floor, "half_bath": half_bath})
    ds = _ds(frame, tmp_path, "zeros")

    found = ds.correlations()

    assert not any(i.kind == "spike" for i in found.insights)
