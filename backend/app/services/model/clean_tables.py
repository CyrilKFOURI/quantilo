"""N32 (audit 2026-11-12, § 4.4): Clean's Advanced layer with the rules between the tables of a
project (T-01 orphan keys, T-02 a key repeated in the table others point to).

The model's service knows the relations the user accepted (`ModelService.links_for`); the
engine checks them (`quantilo.cleaning.rules.tables`). Attached once the modules are set up, so
the Clean module itself does not depend on the projects: its `advanced` is wrapped, and the
suggestions it caches stay as they were. These are read each time, since they depend on the
other tables, which change apart from this one."""

from typing import Any

import duckdb
from quantilo.cleaning.rules.tables import suggest_tables
from quantilo.cleaning.suggestion import CleanSuggestion

from app.services.model.service import ModelService


def _next_number(suggestions: list[CleanSuggestion]) -> int:
    numbers = [
        int(s.step.id[1:])
        for s in suggestions
        if s.step is not None and s.step.id.startswith("s") and s.step.id[1:].isdigit()
    ]
    return max(numbers, default=0)


def table_suggestions(
    model: ModelService, dataset_id: str, taken: list[CleanSuggestion]
) -> list[CleanSuggestion]:
    """T-01 and T-02 of the dataset; their step ids follow every id in `taken`."""
    links = model.links_for(dataset_id)
    if not links:
        return []
    connection = duckdb.connect()
    try:
        found = suggest_tables(connection, dataset_id, links)
    finally:
        connection.close()
    number = _next_number(taken)
    for suggestion in found:
        if suggestion.step is not None:
            number += 1
            suggestion.step.id = f"s{number}"
    return found


def attach(clean: Any, model: ModelService) -> None:
    """Clean's `advanced` (when the module is on) also returns the dataset's table rules."""
    if clean is None:
        return
    advanced = clean.advanced

    def with_tables(dataset_id: str) -> Any:
        result = advanced(dataset_id)
        taken = [*clean.suggestions(dataset_id).suggestions, *result.suggestions]
        found = table_suggestions(model, dataset_id, taken)
        if not found:
            return result
        return result.model_copy(update={"suggestions": [*result.suggestions, *found]})

    clean.advanced = with_tables
