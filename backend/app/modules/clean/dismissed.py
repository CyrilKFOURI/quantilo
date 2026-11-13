"""N05: "Don't suggest again" (`clean_dismissed`, migration 0032). A dismissal of one suggestion or
of a rule on every column belongs to the family's original (its cleaned versions share it); one of a
rule on a column name, or of a rule everywhere, reaches every dataset. Matching is the engine's
(`quantilo.cleaning.dismiss`)."""

import sqlite3
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel
from quantilo.cleaning import dismiss as dm
from quantilo.cleaning.suggestion import CleanSuggestion

from app.core.errors import AppError


class CleanDismissal(BaseModel):
    """A dismissal as stored, with its reach and why in a sentence (`reason`)."""

    id: str
    dataset_id: str | None  # the original it belongs to; None: every dataset
    scope: dm.Scope
    rule: str
    column: str | None = None
    suggestion: str | None = None
    title: str | None = None
    note: str | None = None
    created_at: str
    reason: str

    def model(self) -> dm.Dismissal:
        return dm.Dismissal(
            scope=self.scope,
            rule=self.rule,
            column=self.column,
            suggestion=self.suggestion,
            title=self.title,
            note=self.note,
        )


class HiddenSuggestion(BaseModel):
    """A suggestion a dismissal hides, and that dismissal's id."""

    suggestion: CleanSuggestion
    dismissal: str


class Dismissed(BaseModel):
    """What "Don't suggest again" did: the dismissal, and the suggestions it hides here now (the
    page takes their steps out of its draft)."""

    dismissal: CleanDismissal
    hidden: list[str]


def _row(row: sqlite3.Row) -> CleanDismissal:
    found = CleanDismissal(
        id=row["id"],
        dataset_id=row["dataset_id"],
        scope=row["scope"],
        rule=row["rule"],
        column=row["column_name"],
        suggestion=row["suggestion"],
        title=row["title"],
        note=row["note"],
        created_at=row["created_at"],
        reason="",
    )
    found.reason = dm.reason(found.model())
    return found


def applying(connection: sqlite3.Connection, root: str) -> list[CleanDismissal]:
    """The dismissals that apply to a family: its own, then those of every dataset."""
    rows = connection.execute(
        "SELECT * FROM clean_dismissed WHERE dataset_id = ? OR dataset_id IS NULL "
        "ORDER BY dataset_id IS NULL, created_at, id",
        (root,),
    ).fetchall()
    return [_row(row) for row in rows]


def add(connection: sqlite3.Connection, root: str, body: dm.Dismissal) -> CleanDismissal:
    """Store a dismissal (the same one twice is kept once)."""
    owner = root if body.scope in dm.HERE else None
    for existing in applying(connection, root):
        if existing.dataset_id == owner and existing.model().key() == body.key():
            return existing
    count = connection.execute("SELECT count(*) FROM clean_dismissed").fetchone()[0]
    if count >= dm.MAX_DISMISSED:
        raise AppError(
            f"{dm.MAX_DISMISSED:,} suggestions are dismissed already: restore some first.",
            code="too_many_dismissed",
            status_code=409,
        )
    dismissal_id = uuid.uuid4().hex[:12]
    connection.execute(
        "INSERT INTO clean_dismissed (id, dataset_id, scope, rule, column_name, suggestion, "
        "title, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            dismissal_id,
            owner,
            body.scope,
            body.rule,
            body.column,
            body.suggestion,
            body.title,
            body.note,
            datetime.now(UTC).isoformat(),
        ),
    )
    row = connection.execute(
        "SELECT * FROM clean_dismissed WHERE id = ?", (dismissal_id,)
    ).fetchone()
    return _row(row)


def remove(connection: sqlite3.Connection, dismissal_id: str) -> CleanDismissal:
    row = connection.execute(
        "SELECT * FROM clean_dismissed WHERE id = ?", (dismissal_id,)
    ).fetchone()
    if row is None:
        raise AppError(
            "This dismissal no longer exists.", code="unknown_dismissal", status_code=404
        )
    connection.execute("DELETE FROM clean_dismissed WHERE id = ?", (dismissal_id,))
    return _row(row)


def split(
    found: list[CleanSuggestion], dismissals: list[CleanDismissal]
) -> tuple[list[CleanSuggestion], list[HiddenSuggestion]]:
    shown, hidden = dm.split(found, [d.model() for d in dismissals])
    return shown, [HiddenSuggestion(suggestion=s, dismissal=dismissals[i].id) for s, i in hidden]
