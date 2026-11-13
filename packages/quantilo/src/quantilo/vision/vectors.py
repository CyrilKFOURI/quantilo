"""A cache of one vector per image (112; study § 3.3): `images/<id>/embeddings/<method>.parquet`.

`image_id` + `vector` (FixedSizeList<float32>), and the method's key in the file's metadata (the
descriptor's version, the network and its weights): a file written under another key is ignored,
so a new descriptor or other weights never mix with old vectors. Images already cached are read,
only the others are computed. No thread, no process: it runs in the browser version too."""

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

KEY = b"quantilo.vectors.key"


def load(path: str | os.PathLike[str], key: str) -> dict[str, np.ndarray]:
    """The cached vectors by `image_id`, empty when the file is missing, unreadable or written
    under another key."""
    file = Path(path)
    if not file.is_file():
        return {}
    try:
        table = pq.read_table(file)
    except (OSError, pa.ArrowException):
        return {}
    meta = table.schema.metadata or {}
    if meta.get(KEY) != key.encode():
        return {}
    ids = table["image_id"].to_pylist()
    column = table["vector"].combine_chunks()
    if len(ids) == 0:
        return {}
    width = column.type.list_size
    flat = column.values.to_numpy(zero_copy_only=False).astype(np.float32)
    matrix = flat.reshape(len(ids), width)
    return {str(i): matrix[n] for n, i in enumerate(ids)}


def save(
    path: str | os.PathLike[str], key: str, image_ids: Sequence[str], vectors: np.ndarray
) -> None:
    """Write the vectors (rows with NaN are left out: an image that failed is computed again)."""
    matrix = np.asarray(vectors, dtype=np.float32)
    keep = np.isfinite(matrix).all(axis=1)
    ids = [str(v) for v, k in zip(image_ids, keep, strict=True) if k]
    rows = matrix[keep]
    width = int(matrix.shape[1])
    values = pa.array(rows.ravel(), type=pa.float32())
    column = pa.FixedSizeListArray.from_arrays(values, width)
    table = pa.table({"image_id": pa.array(ids, type=pa.string()), "vector": column})
    table = table.replace_schema_metadata({KEY: key.encode()})
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    partial = file.with_name(file.name + ".partial")
    pq.write_table(table, partial)
    partial.replace(file)


__all__ = ["load", "save"]
