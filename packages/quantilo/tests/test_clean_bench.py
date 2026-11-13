"""N01: the cleaning precision bench, in CI on the committed corpus (`tests/clean_corpus/`).

- every truth of every case holds (a milestone not merged yet: an expected failure that turns into
  a pass once it lands);
- no case has a suggestion marked automatic that no truth judges (closed world);
- every automatic catalogue entry (N02) stays at 99 % or more, in proposals and in cells.

The audit's real files (`--real`) are measured too when they are next to the repository (slow,
local only): `uv run --no-sync python ../../scripts/clean_bench.py --real`.
"""

import pytest

from quantilo.cleaning import catalogue
from tests.clean_corpus import bench
from tests.clean_corpus.cases import ALL_CASES, Truth

pytestmark = pytest.mark.clean_bench


@pytest.fixture(scope="module")
def results() -> dict[str, bench.CaseResult]:
    return {r.case: r for r in bench.run_cases(ALL_CASES)}


# Milestones merged on master: their truths must hold (strict).
MERGED = frozenset({"N06", "N07", "N08", "N09", "N12", "N13", "N14", "N15", "N18", "N19"})


def _param(case: str, truth: Truth) -> object:
    marks = (
        [pytest.mark.xfail(strict=False, reason=f"{truth.after} is not merged yet")]
        if truth.after and truth.after not in MERGED
        else []
    )
    return pytest.param(case, truth, id=f"{case}-{truth.label}", marks=marks)


TRUTHS = [_param(case.name, truth) for case in ALL_CASES for truth in case.truth]


@pytest.mark.parametrize(("case", "truth"), TRUTHS)
def test_corpus_truth(results: dict[str, bench.CaseResult], case: str, truth: Truth) -> None:
    result = results[case]
    assert result.error is None, result.error
    found = result.matches(truth)
    if truth.verdict == "wrong":
        assert not found, f"{[s.id for s in found]}: {truth.why}"
        return
    assert found, f"not proposed: {truth.why}"
    for suggestion in found:
        if truth.mode is not None:
            assert suggestion.mode == truth.mode, suggestion.id
        if truth.auto is not None:
            assert suggestion.auto is truth.auto, suggestion.id
        if truth.verdict == "decision":
            assert suggestion.mode != "A" and not suggestion.auto, suggestion.id


@pytest.mark.parametrize("case", [c.name for c in ALL_CASES])
def test_every_automatic_suggestion_is_judged(
    results: dict[str, bench.CaseResult], case: str
) -> None:
    result = results[case]
    assert result.error is None, result.error
    unjudged = [
        j.suggestion.id for j in result.judged if j.verdict == "unjudged" and j.suggestion.auto
    ]
    assert not unjudged, f"automatic suggestions no truth judges: {unjudged}"


def test_automatic_rules_keep_99_percent(results: dict[str, bench.CaseResult]) -> None:
    totals = bench.tally(results.values())
    problems = bench.gate(totals)
    assert not problems, "\n" + bench.table(totals) + "\n" + "\n".join(problems)


def test_the_table_lists_each_entry(results: dict[str, bench.CaseResult]) -> None:
    totals = bench.tally(results.values())
    text = bench.table(totals)
    assert "| B-01 | automatic |" in text and "| B-04:sure | automatic |" in text
    assert all(
        row.proposals == row.correct + row.wrong + row.decision + row.unjudged
        for row in totals.values()
    )


def test_ames_na_is_never_automatic(results: dict[str, bench.CaseResult]) -> None:
    """N02: on Ames, no B-04 suggestion that reads 'NA' as missing is automatic."""
    for suggestion in results["ames_na"].found:
        values = {str(v).lower() for v in getattr(suggestion.step, "values", None) or []}
        if suggestion.rule == "B-04" and "na" in values:
            assert not suggestion.auto, suggestion.id


def test_judging_a_merge_by_its_pairs() -> None:
    from quantilo.cleaning import model as m
    from quantilo.cleaning.suggestion import CleanSuggestion

    merge = CleanSuggestion(
        id="A-01:labels",
        rule="A-01",
        layer="advanced",
        mode="S",
        title="Merge",
        reason="…",
        columns=["labels"],
        step=m.MapValues(id="s1", column="labels", map={"b,a": "a,b", "veg,non": "non,veg"}),
    )
    right = Truth("A-01", "labels", "correct", "order", merges=(("b,a", "a,b"),))
    wrong = Truth("A-01", "labels", "wrong", "meaning", merges=(("non,veg", "veg,non"),))
    other = Truth("A-01", "labels", "wrong", "not merged", merges=(("x", "y"),))
    judged = bench.judge([merge], [right, other])
    assert judged[0].verdict == "correct"
    judged = bench.judge([merge], [right, wrong])
    assert judged[0].verdict == "wrong"  # one wrong group makes the merge wrong


@pytest.mark.slow
@pytest.mark.skipif(not bench.real_available(), reason="the corpora are not next to the repository")
def test_real_files_keep_the_automatic_rules_at_99_percent() -> None:
    results = bench.run_real()
    totals = bench.tally(results)
    problems = bench.gate(totals, every_entry=False)
    assert not problems, "\n" + bench.table(totals, "Real files") + "\n" + "\n".join(problems)
    for entry in catalogue.automatic():
        row = totals.get(entry.id)
        if row is not None and entry.measure is not None and entry.measure.judged:
            assert row.judged >= 1, entry.id
