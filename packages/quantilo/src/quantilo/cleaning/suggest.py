"""Suggestions for a dataset (the Basic layer for now), and the draft they make by default."""

from typing import Any

import duckdb

from quantilo.catalog import ColumnView
from quantilo.cleaning import model as m
from quantilo.cleaning.rules import basic
from quantilo.cleaning.suggestion import CleanSuggestion
from quantilo.ingestion.sql import lit

RULES_VERSION = 1  # bump when a rule changes, to ignore cached suggestions


def suggest(
    connection: duckdb.DuckDBPyConnection, parquet_path: str, views: list[ColumnView]
) -> list[CleanSuggestion]:
    """Every Basic suggestion, in the order their steps run (C-06); step ids s1, s2… follow it."""
    context = basic.profile(connection, f"read_parquet({lit(parquet_path)})", views)
    found = basic.run(context)
    for number, suggestion in enumerate(found, start=1):
        if suggestion.step is not None:
            suggestion.step.id = f"s{number}"
    return found


def default_steps(suggestions: list[CleanSuggestion]) -> list[Any]:
    """The draft Clean opens with: the automatic (mode A) suggestions, ticked (C-03)."""
    return [s.step.model_copy() for s in suggestions if s.mode == "A" and s.step is not None]


def with_suggestion(
    steps: list[Any], suggestion: CleanSuggestion, all_found: list[CleanSuggestion]
) -> list[Any]:
    """`steps` with the suggestion's step added at its place among the other suggested steps
    (manual steps keep theirs); its id changes if another step has it."""
    if suggestion.step is None or any(s.suggestion == suggestion.id for s in steps):
        return steps
    step = suggestion.step.model_copy()
    if any(s.id == step.id for s in steps):
        step.id = m.next_id(steps)
    order = {s.id: s.order for s in all_found}
    position = len(steps)
    for index, existing in enumerate(steps):
        rank = order.get(existing.suggestion or "")
        if rank is not None and rank > suggestion.order:
            position = index
            break
    return [*steps[:position], step, *steps[position:]]


def without_suggestion(steps: list[Any], suggestion_id: str) -> list[Any]:
    return [s for s in steps if s.suggestion != suggestion_id]
