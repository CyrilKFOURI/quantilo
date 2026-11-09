"""SQLite persistence of column detections and user corrections."""

import sqlite3
from dataclasses import asdict, fields
from typing import Any

from quantilo.catalog import USER_FIELDS, ColumnRecord

__all__ = ["USER_FIELDS", "ColumnRecord"]


def _record(row: sqlite3.Row) -> ColumnRecord:
    names = {f.name for f in fields(ColumnRecord)}
    return ColumnRecord(**{key: value for key, value in dict(row).items() if key in names})


def list_columns(connection: sqlite3.Connection, dataset_id: str) -> list[ColumnRecord]:
    rows = connection.execute(
        "SELECT * FROM columns WHERE dataset_id = ? ORDER BY position", (dataset_id,)
    ).fetchall()
    return [_record(row) for row in rows]


def get_column(connection: sqlite3.Connection, dataset_id: str, name: str) -> ColumnRecord | None:
    row = connection.execute(
        "SELECT * FROM columns WHERE dataset_id = ? AND name = ?", (dataset_id, name)
    ).fetchone()
    return _record(row) if row else None


def replace_columns(
    connection: sqlite3.Connection, dataset_id: str, records: list[ColumnRecord]
) -> None:
    connection.execute("DELETE FROM columns WHERE dataset_id = ?", (dataset_id,))
    if records:
        values = [asdict(record) for record in records]
        names = list(values[0])
        connection.executemany(
            f"INSERT INTO columns ({', '.join(names)}) "
            f"VALUES ({', '.join(':' + name for name in names)})",
            values,
        )
    connection.commit()


def update_column(
    connection: sqlite3.Connection, dataset_id: str, name: str, **changes: Any
) -> None:
    assignments = ", ".join(f"{field} = :{field}" for field in changes)
    connection.execute(
        f"UPDATE columns SET {assignments} WHERE dataset_id = :dataset_id AND name = :name",
        {**changes, "dataset_id": dataset_id, "name": name},
    )
    connection.commit()
