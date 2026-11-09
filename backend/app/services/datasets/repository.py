"""SQLite persistence for datasets, their source files and jobs."""

import sqlite3
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from typing import Any, Literal

from quantilo.ingestion.sources import SourceFile

DatasetStatus = Literal["importing", "ready", "error"]
JobStatus = Literal["pending", "running", "done", "failed"]


def now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class DatasetRecord:
    id: str
    name: str
    source_path: str
    source_kind: Literal["file", "folder"]
    format: str
    options: str
    fingerprint: str
    storage_path: str
    status: DatasetStatus
    created_at: str
    sample_pct: float = 100
    row_count: int | None = None
    col_count: int | None = None
    size_bytes: int | None = None
    error: str | None = None
    imported_at: str | None = None
    last_opened_at: str | None = None
    types_version: int = 0
    types_reviewed_at: str | None = None


@dataclass
class JobRecord:
    id: str
    kind: str
    dataset_id: str | None
    status: JobStatus
    created_at: str
    progress: float = 0
    message: str | None = None
    finished_at: str | None = None


def _dataset(row: sqlite3.Row) -> DatasetRecord:
    names = {f.name for f in fields(DatasetRecord)}
    record = dict(row)
    return DatasetRecord(**{key: value for key, value in record.items() if key in names})


def _job(row: sqlite3.Row) -> JobRecord:
    return JobRecord(**dict(row))


def insert_dataset(connection: sqlite3.Connection, record: DatasetRecord) -> None:
    values = asdict(record)
    columns = ", ".join(values)
    placeholders = ", ".join(f":{name}" for name in values)
    connection.execute(f"INSERT INTO datasets ({columns}) VALUES ({placeholders})", values)
    connection.commit()


def update_dataset(connection: sqlite3.Connection, dataset_id: str, **changes: Any) -> None:
    assignments = ", ".join(f"{name} = :{name}" for name in changes)
    connection.execute(
        f"UPDATE datasets SET {assignments} WHERE id = :id", {**changes, "id": dataset_id}
    )
    connection.commit()


def get_dataset(connection: sqlite3.Connection, dataset_id: str) -> DatasetRecord | None:
    row = connection.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    return _dataset(row) if row else None


def list_datasets(connection: sqlite3.Connection) -> list[DatasetRecord]:
    rows = connection.execute(
        "SELECT * FROM datasets ORDER BY COALESCE(last_opened_at, created_at) DESC"
    ).fetchall()
    return [_dataset(row) for row in rows]


def find_ready_by_fingerprint(
    connection: sqlite3.Connection, fingerprint: str
) -> DatasetRecord | None:
    row = connection.execute(
        "SELECT * FROM datasets WHERE fingerprint = ? AND status = 'ready' LIMIT 1",
        (fingerprint,),
    ).fetchone()
    return _dataset(row) if row else None


def count_datasets_with_source(connection: sqlite3.Connection, source_path: str) -> int:
    row = connection.execute(
        "SELECT count(*) FROM datasets WHERE source_path = ?", (source_path,)
    ).fetchone()
    return int(row[0])


def delete_dataset(connection: sqlite3.Connection, dataset_id: str) -> None:
    connection.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    connection.commit()


def replace_files(connection: sqlite3.Connection, dataset_id: str, files: list[SourceFile]) -> None:
    connection.execute("DELETE FROM dataset_files WHERE dataset_id = ?", (dataset_id,))
    connection.executemany(
        "INSERT INTO dataset_files (dataset_id, path, size, mtime) VALUES (?, ?, ?, ?)",
        [(dataset_id, str(file.path), file.size, file.mtime) for file in files],
    )
    connection.commit()


def list_files(connection: sqlite3.Connection, dataset_id: str) -> list[tuple[str, int, float]]:
    rows = connection.execute(
        "SELECT path, size, mtime FROM dataset_files WHERE dataset_id = ?", (dataset_id,)
    ).fetchall()
    return [(row[0], row[1], row[2]) for row in rows]


def insert_job(connection: sqlite3.Connection, record: JobRecord) -> None:
    values = asdict(record)
    columns = ", ".join(values)
    placeholders = ", ".join(f":{name}" for name in values)
    connection.execute(f"INSERT INTO jobs ({columns}) VALUES ({placeholders})", values)
    connection.commit()


def update_job(connection: sqlite3.Connection, job_id: str, **changes: Any) -> None:
    assignments = ", ".join(f"{name} = :{name}" for name in changes)
    connection.execute(f"UPDATE jobs SET {assignments} WHERE id = :id", {**changes, "id": job_id})
    connection.commit()


def get_job(connection: sqlite3.Connection, job_id: str) -> JobRecord | None:
    row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _job(row) if row else None


def latest_job(connection: sqlite3.Connection, dataset_id: str) -> JobRecord | None:
    row = connection.execute(
        "SELECT * FROM jobs WHERE dataset_id = ? ORDER BY created_at DESC LIMIT 1", (dataset_id,)
    ).fetchone()
    return _job(row) if row else None


def interrupt_unfinished(connection: sqlite3.Connection) -> None:
    """After a restart, work that was in progress can never finish: mark it failed."""
    message = "Interrupted: the app was stopped during this task."
    connection.execute(
        "UPDATE jobs SET status = 'failed', message = ?, finished_at = ? "
        "WHERE status IN ('pending', 'running')",
        (message, now()),
    )
    connection.execute(
        "UPDATE datasets SET status = 'error', error = ? WHERE status = 'importing'", (message,)
    )
    connection.commit()
