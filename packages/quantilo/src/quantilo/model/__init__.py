"""The model of a set of tables (088): candidate keys, relations, facts and dimensions, and the
flat analysis table. Pure engine: DuckDB in, dataclasses out (no app, no threads, Pyodide-safe).

    >>> import quantilo as dx
    >>> from quantilo import model
    >>> tables = [model.table_of(dx.open(path)) for path in paths]
    >>> found = model.detect_tables(tables)
    >>> found.roles.facts, [(r.child_id, r.parent_id, r.cardinality) for r in found.relations]
"""

from dataclasses import dataclass, field
from typing import Any

import duckdb

from quantilo.model.build import (
    Link,
    LinkCheck,
    Measure,
    ModelError,
    Plan,
    Preview,
    aggregate,
    check,
    default_columns,
    default_measures,
    plan,
    preview,
    recommend,
    warnings,
    write_flat,
)
from quantilo.model.keys import (
    CandidateKey,
    ColumnProfile,
    Reason,
    TableInput,
    TableProfile,
    profile_table,
)
from quantilo.model.pair import (
    ModeOption,
    PairKey,
    PairPreview,
    preview_pair,
    suggest,
    write_pair,
)
from quantilo.model.relations import Relation, detect, pieces
from quantilo.model.roles import RoleOf, Roles, assign

__all__ = [
    "CandidateKey",
    "ColumnProfile",
    "Detection",
    "Link",
    "LinkCheck",
    "Measure",
    "ModeOption",
    "ModelError",
    "PairKey",
    "PairPreview",
    "Plan",
    "Preview",
    "Reason",
    "Relation",
    "RoleOf",
    "Roles",
    "TableInput",
    "TableProfile",
    "aggregate",
    "assign",
    "check",
    "default_columns",
    "default_measures",
    "detect",
    "detect_tables",
    "plan",
    "preview",
    "preview_pair",
    "profile_table",
    "recommend",
    "suggest",
    "table_of",
    "warnings",
    "write_flat",
    "write_pair",
]


@dataclass
class Detection:
    profiles: list[TableProfile]
    relations: list[Relation]
    roles: Roles
    # what the detection set aside and why (R-REL-12: same columns, pieces of one table)
    notes: list[Reason] = field(default_factory=list)


def detect_tables(
    tables: list[TableInput],
    families: dict[str, str] | None = None,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> Detection:
    """Profiles, proposed relations and roles of `tables` (`families`: id → family root, so a
    dataset is never related to its own cleaned version, R-REL-01)."""
    own = connection is None
    duck = connection or duckdb.connect()
    try:
        profiles = [profile_table(duck, table) for table in tables]
        relations = detect(duck, tables, profiles, families)
    finally:
        if own:
            duck.close()
    names = {profile.id: profile.name for profile in profiles}
    notes = [
        Reason(
            "R-REL-12",
            f"{names[a]} and {names[b]} have the same columns: pieces of one table to stack, "
            "not two tables to join",
        )
        for a, b in pieces(profiles)
    ]
    return Detection(profiles, relations, assign(profiles, relations), notes)


def table_of(dataset: Any) -> TableInput:
    """A library `Dataset` (`quantilo.open`, `from_pandas`) as a table of the model."""
    return TableInput(
        id=dataset.path.stem,
        name=dataset.name,
        storage_path=str(dataset.path),
        columns=dataset.types,
    )
