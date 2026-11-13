"""N02: the catalogue of automatic rules: modes, measures, which suggestions are `auto`."""

import pytest

from quantilo.cleaning import catalogue
from quantilo.cleaning import model as m
from quantilo.cleaning.catalogue import AUTO_PRECISION, CATALOGUE, Entry, Measure
from quantilo.cleaning.rules import advanced, basic, tables
from quantilo.cleaning.suggest import default_steps
from quantilo.cleaning.suggestion import CleanSuggestion


def suggestion(
    rule: str,
    column: str,
    mode: str = "A",
    step: object | None = None,
    key: str | None = None,
    layer: str = "basic",
) -> CleanSuggestion:
    return CleanSuggestion.model_validate(
        {
            "id": f"{rule}:{column}" + (f":{key}" if key else ""),
            "rule": rule,
            "layer": layer,
            "mode": mode,
            "title": "t",
            "reason": "Something was found.",
            "columns": [column],
            "cells": 10,
            "step": step if step is not None else m.NormalizeSpaces(id="s1", columns=[column]),
        }
    )


def tokens(column: str, *values: str, mode: str = "A") -> CleanSuggestion:
    step = m.NullValues(id="s1", column=column, values=list(values), match="token")
    return suggestion("B-04", column, mode, step, key=None if mode == "A" else "maybe")


def test_automatic_needs_a_measure_at_99_percent() -> None:
    with pytest.raises(ValueError, match="automatic needs a measure"):
        Entry("X-01", "X-01", "x", "automatic", None, "no measure")
    with pytest.raises(ValueError, match="automatic needs a measure"):
        Entry("X-01", "X-01", "x", "automatic", Measure(39, 30, 8_352, 7_327), "B-04 as it was")
    with pytest.raises(ValueError, match="automatic needs a measure"):
        Entry("X-01", "X-01", "x", "automatic", Measure(100, 100, 1_000, 900), "cells under 99 %")
    with pytest.raises(ValueError, match="right out of"):
        Measure(3, 4)
    assert Entry("X-01", "X-01", "x", "automatic", Measure(50, 50, 10, 10), "ok").automatic
    assert Entry("X-02", "X-02", "x", "automatic", None, "lossless", lossless=True).automatic
    assert not Entry("X-03", "X-03", "x", "suggested", Measure(39, 30), "below").automatic


def test_every_entry_is_coherent() -> None:
    ids = [e.id for e in catalogue.ENTRIES]
    assert len(ids) == len(set(ids))
    rules = set(basic.ORDER) | set(advanced.ORDER) | set(tables.ORDER)
    rules |= {"E-01", "E-03", "E-06", "E-07"}
    for entry in catalogue.ENTRIES:
        assert entry.id.split(":")[0] == entry.rule
        assert entry.rule in rules, entry.id
        assert entry.evidence and entry.title
        if entry.automatic:
            assert entry.reliable
            precision = None if entry.measure is None else entry.measure.precision
            assert entry.lossless or (precision is not None and precision >= AUTO_PRECISION)
        if entry.rule.startswith(("A-", "E-", "T-")):
            assert entry.mode in ("decision", "information")  # never automatic (C-17, C-30)
    # every Basic rule has an entry
    assert set(basic.ORDER) <= {e.rule for e in catalogue.ENTRIES}


def test_the_audits_measures() -> None:
    """§ 2.3: B-04 as a whole was 30 right of 39 (87.7 % of cells); without the bare 'NA' 100 %."""
    sure, na = CATALOGUE["B-04:sure"].measure, CATALOGUE["B-04:na"].measure
    assert sure is not None and na is not None
    assert sure.judged + na.judged == 39 and sure.correct + na.correct == 30
    assert (sure.cells or 0) + (na.cells or 0) == 8_352
    whole = ((sure.cells_correct or 0) + (na.cells_correct or 0)) / 8_352
    assert round(whole, 3) == 0.877
    assert CATALOGUE["B-04:sure"].automatic and not CATALOGUE["B-04:na"].automatic
    assert CATALOGUE["B-01"].measure == Measure(50, 50, 372_603, 372_603)


def test_classify_the_sub_rules() -> None:
    sure = tokens("status", "n/a", "")
    bare = tokens("Garage Type", "na")
    weak = tokens("region", "unknown", mode="S")
    first, second = tokens("race", "?", mode="S"), tokens("payer", "?", mode="S")
    found = [sure, bare, weak, first, second]
    assert catalogue.classify(sure, found) == "B-04:sure"
    assert catalogue.classify(bare, found) == "B-04:na"
    assert catalogue.classify(weak, found) == "B-04:weak"
    assert catalogue.classify(first, found) == "B-04:file"  # '?' in two columns
    assert catalogue.classify(first, [first]) == "B-04:weak"  # in one column only
    none = suggestion(
        "B-04",
        "Garage Type",
        "S",
        m.MapValues(id="s1", column="Garage Type", map={"NA": "None"}),
        "none",
    )
    assert catalogue.classify(none) == "B-04:none"

    def sentinel(column: str, value: str, mode: str = "S") -> CleanSuggestion:
        step = m.NullValues(id="s1", column=column, values=[value], match="exact")
        return suggestion("B-09", column, mode, step, key=value)

    sensors = [sentinel(c, "-200") for c in ("CO", "NOx", "T")]
    assert {catalogue.classify(s, sensors) for s in sensors} == {"B-09:repeated"}
    assert catalogue.classify(sensors[0], sensors[:2]) == "B-09"
    zero = sentinel("latitude", "0")
    zero.id = "B-09:latitude:0,0"
    assert catalogue.classify(zero) == "B-09:zero"
    pad = m.PadZeros(id="s1", column="zip", length=5)
    assert catalogue.classify(suggestion("B-13", "zip", "A", pad)) == "B-13:postal"
    assert catalogue.classify(suggestion("B-13", "zip", "S", pad)) == "B-13:code"
    assert catalogue.classify(suggestion("B-14", "rows", "A")) == "B-14:id"
    assert catalogue.classify(suggestion("B-14", "rows", "S")) == "B-14"
    assert catalogue.classify(suggestion("B-03", "name", "S")) == "B-03:person"
    assert catalogue.classify(suggestion("B-99", "x")) == "B-99"  # not in the catalogue


def test_annotate_marks_the_reliable_ones_and_says_why() -> None:
    sure = tokens("status", "n/a")
    bare = tokens("Garage Type", "na")
    weak = tokens("region", "unknown", mode="S")
    spaces = suggestion("B-01", "city")
    merge = suggestion("A-01", "city", "S", layer="advanced")
    found = catalogue.annotate([sure, bare, weak, spaces, merge])
    assert [s.auto for s in found] == [True, False, False, True, False]
    assert [s.catalogue for s in found] == ["B-04:sure", "B-04:na", "B-04:weak", "B-01", "A-01"]
    assert sure.reason.endswith("(n = 21). Check your data.")
    assert "reliable fix" in spaces.reason and "reliable" not in bare.reason
    again = catalogue.annotate([spaces])[0]
    assert again.reason.count("reliable fix") == 1  # annotated twice: said once
    risky = suggestion("B-01", "notes")
    risky.risk = "review"
    assert not catalogue.annotate([risky])[0].auto


def test_annotate_says_how_many_cases_and_marks_the_automatic_steps() -> None:
    """N03, N04: an `auto` suggestion carries its measure (n) and an "automatic" step; a mode A
    suggestion of an unreliable entry is only suggested."""
    sure, bare, invisible = (
        tokens("status", "n/a"),
        tokens("Garage Type", "na"),
        suggestion("B-02", "x"),
    )
    for found in (sure, bare, invisible):
        assert found.step is not None
        found.step.origin = "auto"  # as the rules write a mode A step
    catalogue.annotate([sure, bare, invisible])
    assert (sure.measured, bare.measured, invisible.measured) == (21, None, 0)
    assert [s.step.origin for s in (sure, bare, invisible)] == ["auto", "suggested", "auto"]  # type: ignore[union-attr]


def test_lossless_note() -> None:
    invisible = catalogue.annotate([suggestion("B-02", "product")])[0]
    assert invisible.auto and "cannot be seen" in invisible.reason


def test_default_steps_auto_only() -> None:
    sure, bare = tokens("status", "n/a"), tokens("Garage Type", "na")
    sure.step.id, bare.step.id = "s1", "s2"  # type: ignore[union-attr]
    catalogue.annotate([sure, bare])
    assert [s.id for s in default_steps([sure, bare])] == ["s1", "s2"]
    assert [s.id for s in default_steps([sure, bare], auto_only=True)] == ["s1"]


def test_country_guard() -> None:
    """B-13:postal is automatic once N12's country guard (the knowledge base's postal lengths) is in
    the engine; suggested before."""
    entry = CATALOGUE["B-13:postal"]
    assert entry.mode == "automatic" and entry.needs == "N12"
    assert entry.automatic == catalogue.GUARDS["N12"]()
    assert entry.effective_mode == ("automatic" if entry.automatic else "suggested")
