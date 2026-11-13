"""124: Describe segments with a tree (the sparse CART of 122 predicting the segment)."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo.interpret.sparse_tree import SparseTreeError
from quantilo.segments import tree


def ruled(n: int = 3_000, seed: int = 4) -> tuple[pd.DataFrame, np.ndarray]:
    """Segments that follow two questions: income above 50, then the plan."""
    rng = np.random.default_rng(seed)
    income = rng.uniform(0, 100, n)
    plan = rng.choice(["basic", "pro"], n)
    member = rng.random(n) < 0.5
    segments = np.where(income <= 50, 2, np.where(plan == "pro", 0, 1))
    frame = pd.DataFrame(
        {"income": income, "plan": plan, "member": member, "age": rng.normal(40, 9, n)}
    )
    return frame, segments


NAMES = {0: "Rich · pro", 1: "Rich · basic", 2: "Low income"}


def test_the_tree_reads_the_rules_the_segments_follow() -> None:
    frame, segments = ruled()
    found = tree.describe(frame, segments, NAMES, ["income", "plan", "member", "age"])
    assert found.fidelity > 0.97 and found.warning is None
    assert found.nodes[0].left_rule is not None and found.nodes[0].left_rule.startswith("income ≤")
    threshold = float(re.findall(r"[\d.]+", found.nodes[0].left_rule)[0])
    assert abs(threshold - 50) < 2
    by_name = {s.name: s for s in found.segments}
    low = by_name["Low income"]
    assert low.rules and low.rules[0].text.startswith("income ≤")
    assert low.recall is not None and low.recall > 0.95
    assert low.precision is not None and low.precision > 0.95
    pro = by_name["Rich · pro"]
    assert any("plan = pro" in r.text or "plan ≠ basic" in r.text for r in pro.rules)
    assert "find" in pro.text and found.checked + found.learnt == len(frame)
    assert "R-SGT-01" in found.sentence


def test_the_tree_keeps_to_its_size_and_says_when_it_is_rough() -> None:
    frame, segments = ruled(2_000, seed=9)
    small = tree.describe(frame, segments, NAMES, ["income", "plan"], depth=1, leaves=2)
    assert small.found_depth <= 1 and small.leaves <= 2
    assert small.fidelity < tree.FIDELITY and small.warning and "R-SGT-04" in small.warning
    missing = [s for s in small.segments if not s.rules]
    assert missing and "No leaf names" in missing[0].text
    noise = np.random.default_rng(1).integers(0, 3, len(frame))
    rough = tree.describe(frame, noise, NAMES, ["income", "plan", "age"])
    assert rough.fidelity < 0.6 and rough.warning


def test_more_segments_ask_for_a_deeper_tree() -> None:
    assert tree.defaults(3) == (3, 8)
    assert tree.defaults(10) == (4, 10)
    frame, _ = ruled(200)
    with pytest.raises(SparseTreeError, match="two segments"):
        tree.describe(frame, np.zeros(len(frame), dtype=int), NAMES, ["income"])


def test_a_segmentation_describes_itself_with_a_tree(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    centres = np.array([[0, 0], [8, 8], [0, 8]])
    points = np.vstack([c + rng.normal(0, 1, (300, 2)) for c in centres])
    data = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "id": np.arange(len(points))})
    ds = dx.from_pandas(data, "blobs", cache_dir=tmp_path)
    fitted = ds.segments(["x", "y"]).fit()
    described = fitted.describe_tree()
    assert {s.name for s in described.segments} == {s.name for s in fitted.segments}
    assert described.fidelity > 0.95
    assert described.nodes[0].left_rule and described.nodes[0].left_rule.split()[0] in ("x", "y")
    assert described.depth == 3 and described.max_leaves == 8


# --- 124 bis: rules as filters, the optimal tree of one segment ---------------------------------


def test_each_rule_becomes_filter_conditions_that_find_the_segment(tmp_path: Path) -> None:
    """R-SGT-05: the rule's conditions, applied as the app's filter, give the rule's rows."""
    frame, segments = ruled()
    frame.loc[:20, "income"] = np.nan
    found = tree.describe(frame, segments, NAMES, ["income", "plan", "member", "age"])
    low = next(s for s in found.segments if s.name == "Low income")
    (first, *_) = low.filters
    assert first.rule == low.rules[0].text
    assert first.conditions[0].column == "income" and first.conditions[0].op == "lte"
    assert first.note and "R-SGT-05" in first.note  # income has empty values
    ds = dx.from_pandas(frame.assign(segment=segments), "ruled", cache_dir=tmp_path)
    filtered = ds
    for condition in first.conditions:
        filtered = filtered.where(condition)
    rows = filtered.to_pandas()
    assert float(np.mean(np.asarray(rows["segment"]) == 2)) > 0.95 and len(rows) > 1_000
    pro = next(s for s in found.segments if s.name == "Rich · pro")
    ops = {(c.column, c.op, c.negate) for f in pro.filters for c in f.conditions}
    assert ("plan", "in", False) in ops or ("plan", "in", True) in ops


def test_the_optimal_tree_of_one_segment_calls_gosdt(monkeypatch: pytest.MonkeyPatch) -> None:
    """R-SGT-06: one segment against the others, through the optional extra (faked here)."""
    import sys

    from tests.test_interpret_trees import _FakeGosdt, _module

    monkeypatch.setitem(sys.modules, "gosdt", _module("gosdt", GOSDTClassifier=_FakeGosdt))
    from quantilo.interpret import extras

    monkeypatch.setattr(extras, "require", lambda name: sys.modules[name])
    frame, segments = ruled()
    found = tree.optimal_segment(frame, segments, NAMES, ["income", "plan", "age"], 2)
    assert found.leaves >= 2 and found.score > 0.9
    assert found.nodes[0].left_rule and "income" in found.nodes[0].left_rule
    with pytest.raises(SparseTreeError, match="not in this segmentation"):
        tree.optimal_segment(frame, segments, NAMES, ["income"], 7)
