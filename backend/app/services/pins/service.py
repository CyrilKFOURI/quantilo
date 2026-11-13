"""Pinned rows (099, docs/features/099-lignes-epinglees/spec.md): rows a user pinned with a label
and a colour, kept in SQLite under the row's key (R-PIN-01), found again after a re-import
(R-PIN-02), and compared with the other rows on request (R-PIN-05)."""

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import duckdb
from pydantic import BaseModel, Field, field_validator
from quantilo.ingestion.sql import lit
from quantilo.segments.engine import SegmentsEngine
from quantilo.segments.pinned import PinnedComparison

from app.core import db
from app.core.errors import AppError
from app.core.storage import Workspace
from app.services.columns.service import ColumnService

MAX_PINS = 5_000  # R-PIN-04: a `label_rows` step carries as many
MAX_ROWS = 500  # rows pinned or unpinned at a time
COLORS = 8  # slots of the theme's categorical palette (R-PIN-03)


class PinError(AppError):
    def __init__(self, message: str, code: str = "pins", status_code: int = 422) -> None:
        super().__init__(message, code=code, status_code=status_code)


def _label(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("A label needs at least one character.")
    return text


class PinRequest(BaseModel):
    rows: list[int] = Field(min_length=1, max_length=MAX_ROWS)  # row numbers in the file, from 0
    label: str = Field(min_length=1, max_length=60)
    color: int = Field(ge=1, le=COLORS)

    _strip = field_validator("label")(_label)


class UnpinRequest(BaseModel):
    keys: list[str] = Field(min_length=1, max_length=MAX_PINS)


class LabelChange(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    to: str | None = Field(default=None, min_length=1, max_length=60)  # a new name
    color: int | None = Field(default=None, ge=1, le=COLORS)

    @field_validator("to")
    @classmethod
    def _to(cls, value: str | None) -> str | None:
        return None if value is None else _label(value)


@dataclass
class Pin:
    key: str
    row: int  # in the file, from 0 (the Data page shows row + 1)
    label: str
    color: int
    created_at: str
    found: bool  # False: the row is no longer in the dataset (R-PIN-02)
    inherited: bool = False  # 099 bis: pinned on the parent, followed through the cleaning


@dataclass
class PinLabel:
    label: str
    color: int
    rows: int  # pins with this label, found or not


@dataclass
class PinnedRows:
    pins: list[Pin]  # by row number, the rows not found last
    labels: list[PinLabel]  # in the order they were first used
    missing: int  # pins whose row was not found
    limit: int = MAX_PINS
    # R-PIN-16 (099 bis): what was taken from the parent dataset of a cleaned version, said
    parent: str | None = None
    inherited: int = 0  # the parent's pins whose row is in this version


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PinService:
    def __init__(self, workspace: Workspace, columns: ColumnService) -> None:
        self.workspace = workspace
        self.columns = columns
        self.engine = SegmentsEngine(columns)
        # 099 bis: dataset id → its parent and rowmap (a cleaned version), set by the clean module
        self.lineages: list[Callable[[str], Any]] = []

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def _stored(self, dataset_id: str) -> list[sqlite3.Row]:
        with self._db() as connection:
            return connection.execute(
                "SELECT row_key, row_number, label, color, created_at FROM pinned_rows "
                "WHERE dataset_id = ? ORDER BY created_at, row_number",
                (dataset_id,),
            ).fetchall()

    def pins(self, dataset_id: str, depth: int = 0) -> PinnedRows:
        """Every pin, each row checked against its key and found again when it moved; for a
        cleaned version, the parent's pins too, followed through its rowmap (R-PIN-16)."""
        own = self._own(dataset_id)
        inherited, said = self._inherited(dataset_id, {p.row for p in own.pins}, depth)
        if not inherited and said is None:
            return own
        pins = sorted([*own.pins, *inherited], key=lambda p: (not p.found, p.row))
        labels = {item.label: item for item in own.labels}
        for pin in inherited:
            item = labels.setdefault(pin.label, PinLabel(pin.label, pin.color, 0))
            item.rows += 1
        return PinnedRows(pins, list(labels.values()), own.missing, MAX_PINS, said, len(inherited))

    def _inherited(
        self, dataset_id: str, taken: set[int], depth: int
    ) -> tuple[list[Pin], str | None]:
        """R-PIN-16: a cleaned version shows its parent's pins on the rows it kept (child row from
        the rowmap); a row the cleaning removed is counted, never silently dropped; a parent
        re-imported since the version was made is not followed (its rows may have changed)."""
        link = next((found for get in self.lineages if (found := get(dataset_id))), None)
        if link is None or depth > 3:
            return [], None
        try:
            parent = self.columns.datasets.get(link.parent_id)
        except AppError:
            return [], None
        if link.stale:
            return [], (
                f"R-PIN-16 · {parent.name} was re-imported since this version was made: its pins "
                "are not followed here."
            )
        found = self.pins(link.parent_id, depth + 1)
        wanted = {p.row: p for p in found.pins if p.found}
        if not wanted or not link.rowmap:
            return [], None
        connection = duckdb.connect()
        try:
            rows = connection.execute(
                f"SELECT parent_row, child_row FROM read_parquet({lit(str(link.rowmap))}) "
                f"WHERE parent_row IN ({', '.join(str(int(r)) for r in wanted)})"
            ).fetchall()
        finally:
            connection.close()
        kept = {int(p): int(c) for p, c in rows if c is not None}
        inherited = [
            Pin(pin.key, kept[row], pin.label, pin.color, pin.created_at, True, inherited=True)
            for row, pin in wanted.items()
            if row in kept and kept[row] not in taken
        ]
        removed = len(wanted) - len(kept)
        gone = f"; {removed:,} were on rows the cleaning removed" if removed else ""
        said = (
            f"R-PIN-16 · {len(inherited):,} pin{'s' if len(inherited) != 1 else ''} of "
            f"{parent.name}, followed through the cleaning{gone}. Pin them again here to change "
            "them."
        )
        return inherited, said

    def _own(self, dataset_id: str) -> PinnedRows:
        self.columns.datasets.get(dataset_id)
        stored = self._stored(dataset_id)
        if not stored:
            return PinnedRows([], [], 0)
        current = self.engine.keys(dataset_id, [int(r["row_number"]) for r in stored])
        moved = [r["row_key"] for r in stored if current.get(int(r["row_number"])) != r["row_key"]]
        found = self.engine.find_keys(dataset_id, moved) if moved else {}
        if found:
            with self._db() as connection, connection:
                connection.executemany(
                    "UPDATE pinned_rows SET row_number = ? WHERE dataset_id = ? AND row_key = ?",
                    [(row, dataset_id, key) for key, row in found.items()],
                )
        lost = set(moved) - set(found)
        pins = [
            Pin(
                r["row_key"],
                found.get(r["row_key"], int(r["row_number"])),
                r["label"],
                int(r["color"]),
                r["created_at"],
                r["row_key"] not in lost,
            )
            for r in stored
        ]
        labels: dict[str, PinLabel] = {}
        for pin in pins:
            item = labels.setdefault(pin.label, PinLabel(pin.label, pin.color, 0))
            item.rows += 1
        pins.sort(key=lambda p: (not p.found, p.row))
        return PinnedRows(pins, list(labels.values()), len(lost))

    def pin(self, dataset_id: str, request: PinRequest) -> PinnedRows:
        """Pin rows with a label; a row already pinned takes the new label; every row of the
        label takes its colour (R-PIN-03)."""
        self.columns.datasets.get(dataset_id)
        rows = sorted(set(request.rows))
        keys = self.engine.keys(dataset_id, rows)
        absent = [r for r in rows if r not in keys]
        if absent:
            raise PinError(f"Row {absent[0] + 1} is not in this dataset.", "unknown_row")
        stored = {r["row_key"] for r in self._stored(dataset_id)}
        new = {keys[r] for r in rows} - stored  # inherited pins never count (R-PIN-16)
        if len(stored) + len(new) > MAX_PINS:
            raise PinError(
                f"At most {MAX_PINS:,} pinned rows per dataset ({len(stored):,} already).",
                "too_many_pins",
            )
        moment = _now()
        with self._db() as connection, connection:
            connection.executemany(
                "INSERT INTO pinned_rows (dataset_id, row_key, row_number, label, color, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (dataset_id, row_key) DO UPDATE "
                "SET row_number = excluded.row_number, label = excluded.label, "
                "color = excluded.color",
                [(dataset_id, keys[r], r, request.label, request.color, moment) for r in rows],
            )
            connection.execute(
                "UPDATE pinned_rows SET color = ? WHERE dataset_id = ? AND label = ?",
                (request.color, dataset_id, request.label),
            )
        return self.pins(dataset_id)

    def unpin(self, dataset_id: str, request: UnpinRequest) -> PinnedRows:
        self.columns.datasets.get(dataset_id)
        with self._db() as connection, connection:
            connection.executemany(
                "DELETE FROM pinned_rows WHERE dataset_id = ? AND row_key = ?",
                [(dataset_id, key) for key in request.keys],
            )
        return self.pins(dataset_id)

    def change_label(self, dataset_id: str, change: LabelChange) -> PinnedRows:
        """Rename a label (into another one: its rows take that label's colour) or recolour it."""
        self.columns.datasets.get(dataset_id)
        label = change.label.strip()
        with self._db() as connection, connection:
            exists = connection.execute(
                "SELECT 1 FROM pinned_rows WHERE dataset_id = ? AND label = ? LIMIT 1",
                (dataset_id, label),
            ).fetchone()
            if not exists:
                raise PinError(f"No pinned row is labelled “{label}”.", "unknown_label", 404)
            color = change.color
            if change.to is not None and change.to != label:
                if color is None:
                    target = connection.execute(
                        "SELECT color FROM pinned_rows WHERE dataset_id = ? AND label = ? LIMIT 1",
                        (dataset_id, change.to),
                    ).fetchone()
                    color = int(target["color"]) if target else None
                connection.execute(
                    "UPDATE pinned_rows SET label = ? WHERE dataset_id = ? AND label = ?",
                    (change.to, dataset_id, label),
                )
                label = change.to
            if color is not None:
                connection.execute(
                    "UPDATE pinned_rows SET color = ? WHERE dataset_id = ? AND label = ?",
                    (color, dataset_id, label),
                )
        return self.pins(dataset_id)

    def compare(self, dataset_id: str, label: str | None = None) -> PinnedComparison:
        """R-PIN-05: asked for. The pinned rows found (all, or one label) against the others."""
        found = self.pins(dataset_id)
        if label is not None and all(p.label != label for p in found.pins):
            raise PinError(f"No pinned row is labelled “{label}”.", "unknown_label", 404)
        rows = [p.row for p in found.pins if p.found and (label is None or p.label == label)]
        return self.engine.compare_rows(dataset_id, rows, label)
