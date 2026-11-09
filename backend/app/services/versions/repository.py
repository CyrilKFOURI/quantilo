"""SQLite persistence for dataset versions: one statistical photo per import (031)."""

import json
import sqlite3
from dataclasses import dataclass

from quantilo.changes.snapshot import Snapshot, snapshot_from_dict
from quantilo.results import to_dict


@dataclass
class VersionRecord:
    dataset_id: str
    version: int
    imported_at: str
    storage_stamp: str
    row_count: int
    snapshot: Snapshot
    parquet_path: str | None = None
    kept_bytes: int = 0


def _record(row: sqlite3.Row) -> VersionRecord:
    return VersionRecord(
        dataset_id=row["dataset_id"],
        version=row["version"],
        imported_at=row["imported_at"],
        storage_stamp=row["storage_stamp"],
        row_count=row["row_count"],
        snapshot=snapshot_from_dict(json.loads(row["snapshot"])),
        parquet_path=row["parquet_path"],
        kept_bytes=row["kept_bytes"],
    )


def list_versions(connection: sqlite3.Connection, dataset_id: str) -> list[VersionRecord]:
    rows = connection.execute(
        "SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY version", (dataset_id,)
    ).fetchall()
    return [_record(row) for row in rows]


def latest(connection: sqlite3.Connection, dataset_id: str) -> VersionRecord | None:
    row = connection.execute(
        "SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY version DESC LIMIT 1",
        (dataset_id,),
    ).fetchone()
    return _record(row) if row else None


def insert(connection: sqlite3.Connection, record: VersionRecord) -> None:
    connection.execute(
        "INSERT INTO dataset_versions (dataset_id, version, imported_at, storage_stamp, "
        "row_count, snapshot, parquet_path, kept_bytes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.dataset_id,
            record.version,
            record.imported_at,
            record.storage_stamp,
            record.row_count,
            json.dumps(to_dict(record.snapshot)),
            record.parquet_path,
            record.kept_bytes,
        ),
    )
    connection.commit()


def update_snapshot(
    connection: sqlite3.Connection, dataset_id: str, version: int, snapshot: Snapshot
) -> None:
    connection.execute(
        "UPDATE dataset_versions SET snapshot = ? WHERE dataset_id = ? AND version = ?",
        (json.dumps(to_dict(snapshot)), dataset_id, version),
    )
    connection.commit()


def set_kept(
    connection: sqlite3.Connection, dataset_id: str, version: int, path: str | None, size: int
) -> None:
    connection.execute(
        "UPDATE dataset_versions SET parquet_path = ?, kept_bytes = ? "
        "WHERE dataset_id = ? AND version = ?",
        (path, size, dataset_id, version),
    )
    connection.commit()
