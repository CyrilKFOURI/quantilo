"""SQLite persistence for sources (085): a file imported as several tables, and their links."""

import json
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any, Literal

SourceKind = Literal["workbook", "csv", "json", "jsonl", "xml", "image"]


@dataclass
class SourceRecord:
    id: str
    kind: SourceKind
    name: str
    source_path: str
    fingerprint: str
    size_bytes: int
    created_at: str
    meta: str = "{}"
    plan: str | None = None


@dataclass
class LinkRecord:
    parent_id: str
    child_id: str
    parent_key: str
    child_key: str
    rule: str


def insert_source(connection: sqlite3.Connection, record: SourceRecord) -> None:
    values = asdict(record)
    columns = ", ".join(values)
    placeholders = ", ".join(f":{name}" for name in values)
    connection.execute(f"INSERT INTO sources ({columns}) VALUES ({placeholders})", values)
    connection.commit()


def get_source(connection: sqlite3.Connection, source_id: str) -> SourceRecord | None:
    row = connection.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
    return SourceRecord(**dict(row)) if row else None


def list_sources(connection: sqlite3.Connection) -> list[SourceRecord]:
    rows = connection.execute("SELECT * FROM sources ORDER BY created_at DESC").fetchall()
    return [SourceRecord(**dict(row)) for row in rows]


def source_datasets(connection: sqlite3.Connection, source_id: str) -> list[str]:
    rows = connection.execute(
        "SELECT id FROM datasets WHERE source_id = ? ORDER BY created_at, rowid", (source_id,)
    ).fetchall()
    return [row[0] for row in rows]


def delete_source(connection: sqlite3.Connection, source_id: str) -> None:
    connection.execute("DELETE FROM sources WHERE id = ?", (source_id,))
    connection.commit()


def delete_if_empty(connection: sqlite3.Connection, source_id: str) -> None:
    """A source without tables left is gone with its last table."""
    connection.execute(
        "DELETE FROM sources WHERE id = ? AND NOT EXISTS "
        "(SELECT 1 FROM datasets WHERE source_id = sources.id)",
        (source_id,),
    )
    connection.commit()


def insert_links(connection: sqlite3.Connection, links: list[LinkRecord]) -> None:
    connection.executemany(
        "INSERT OR IGNORE INTO dataset_links (parent_id, child_id, parent_key, child_key, rule) "
        "VALUES (:parent_id, :child_id, :parent_key, :child_key, :rule)",
        [asdict(link) for link in links],
    )
    connection.commit()


def links_of(connection: sqlite3.Connection, dataset_id: str) -> list[LinkRecord]:
    rows = connection.execute(
        "SELECT parent_id, child_id, parent_key, child_key, rule FROM dataset_links "
        "WHERE parent_id = ? OR child_id = ?",
        (dataset_id, dataset_id),
    ).fetchall()
    return [LinkRecord(**dict(row)) for row in rows]


def loads(text: str | None) -> Any:
    return json.loads(text) if text else None
