"""N46: the cleaning journal (`clean_journal`, migration 0034): who changed what in a dataset's
draft and when, each version saved, each suggestion dismissed, each change of the checks. The
entries of the draft go into the version saved from it (`version_id`)."""

import json
import sqlite3
from collections.abc import Sequence

from quantilo.cleaning.journal import JournalEntry

KEPT = 5_000  # entries per dataset, the oldest dropped first
SHOWN = 500


def add(connection: sqlite3.Connection, dataset_id: str, entries: Sequence[JournalEntry]) -> None:
    for entry in entries:
        connection.execute(
            "INSERT INTO clean_journal (dataset_id, version_id, at, actor, action, step_id, op, "
            "rule, origin, columns, cells, rows, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                dataset_id,
                entry.version_id,
                entry.at,
                entry.actor,
                entry.action,
                entry.step_id,
                entry.op,
                entry.rule,
                entry.origin,
                json.dumps(entry.columns),
                entry.cells,
                entry.rows,
                (entry.detail or "")[:500] or None,
            ),
        )
    if entries:
        connection.execute(
            "DELETE FROM clean_journal WHERE dataset_id = ? AND id NOT IN (SELECT id FROM "
            "clean_journal WHERE dataset_id = ? ORDER BY id DESC LIMIT ?)",
            (dataset_id, dataset_id, KEPT),
        )


def _entry(row: sqlite3.Row) -> JournalEntry:
    return JournalEntry(
        at=row["at"],
        actor=row["actor"],
        action=row["action"],
        step_id=row["step_id"],
        op=row["op"],
        rule=row["rule"],
        origin=row["origin"],
        columns=json.loads(row["columns"] or "[]"),
        cells=row["cells"],
        rows=row["rows"],
        detail=row["detail"],
        version_id=row["version_id"],
    )


def of_draft(connection: sqlite3.Connection, dataset_id: str) -> list[JournalEntry]:
    """The changes of the draft not saved into a version yet, the oldest first."""
    rows = connection.execute(
        "SELECT * FROM (SELECT * FROM clean_journal WHERE dataset_id = ? AND version_id IS NULL "
        "ORDER BY id DESC LIMIT ?) ORDER BY id",
        (dataset_id, SHOWN),
    ).fetchall()
    return [_entry(row) for row in rows]


def of_version(connection: sqlite3.Connection, version_id: str) -> list[JournalEntry]:
    rows = connection.execute(
        "SELECT * FROM clean_journal WHERE version_id = ? ORDER BY id", (version_id,)
    ).fetchall()
    return [_entry(row) for row in rows]


def close(connection: sqlite3.Connection, dataset_id: str, version_id: str) -> None:
    """The draft's changes go into the version saved from it."""
    connection.execute(
        "UPDATE clean_journal SET version_id = ? WHERE dataset_id = ? AND version_id IS NULL",
        (version_id, dataset_id),
    )
