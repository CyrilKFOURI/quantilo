"""One Parquet file per table (pyarrow only: it works in the browser too)."""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')  # not allowed in a Windows file name


def file_name(table: str, used: set[str]) -> str:
    base = UNSAFE.sub("_", table).strip(" .") or "table"
    name = base
    number = 2
    while name.lower() in used:
        name = f"{base}~{number}"
        number += 1
    used.add(name.lower())
    return name + ".parquet"


def write_tables(tables: dict[str, pa.Table], folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    codec = "zstd" if pa.Codec.is_available("zstd") else "snappy"
    used: set[str] = set()
    written = {}
    for name, table in tables.items():
        target = folder / file_name(name, used)
        partial = target.with_suffix(".parquet.partial")
        pq.write_table(table, partial, compression=codec)
        partial.replace(target)
        written[name] = target
    return written
