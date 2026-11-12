# ruff: noqa: RUF002, RUF003  (›, × are the app's notation)
"""The cache of text embeddings (091, R-EMB): one vector per distinct value and model, on disk.

    cache = EmbeddingCache(root, model_key="potion-multilingual-128M|…|256")
    vectors = cache.encode(texts, encoder)   # only the values never seen are encoded

- R-EMB-01: the key of a value is `blake2b(normalised value)` on 16 bytes. A cleaned version, a
  re-import or a sister table reuse the vectors of the values that did not change.
- R-EMB-02: one folder per `model_key` (name, weights hash, dimension, normalisation, maximum
  length): vectors of two models are never mixed.
- R-EMB-03: `<root>/<model_key hash>/part-*.parquet` (`key` binary(16), `vector`
  FixedSizeList<float32>), listed in `index.json`. No vector database: numpy is enough at this
  size (100 000 × 256 × 4 bytes = 100 MB).
- R-EMB-04: a disk budget (2 GB by default); past it, the least recently used parts go first.

Standard library, numpy and pyarrow only (the engine depends on no database)."""

import hashlib
import json
import time
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BUDGET_BYTES = 2 * 1024**3  # R-EMB-04
KEY_BYTES = 16  # R-EMB-01

Encoder = Callable[[list[str]], np.ndarray]


def normalise(text: str) -> str:
    """R-EMB-01: the form a value is keyed by (Unicode NFC, spaces trimmed and collapsed)."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def value_key(text: str) -> bytes:
    return hashlib.blake2b(normalise(text).encode("utf-8"), digest_size=KEY_BYTES).digest()


def folder_name(model_key: str) -> str:
    return hashlib.blake2b(model_key.encode("utf-8"), digest_size=8).hexdigest()


class EmbeddingCache:
    def __init__(self, root: Path, model_key: str, budget: int = BUDGET_BYTES) -> None:
        self.root = Path(root)
        self.model_key = model_key
        self.budget = budget
        self.folder = self.root / folder_name(model_key)
        self._vectors: dict[bytes, np.ndarray] | None = None

    # --- index ---------------------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.folder / "index.json"

    def _index(self) -> dict[str, object]:
        try:
            return json.loads(self._index_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"model_key": self.model_key, "parts": {}}

    def _save_index(self, index: dict[str, object]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self._index_path().write_text(json.dumps(index, indent=1), encoding="utf-8")

    # --- read / write --------------------------------------------------------------------------

    def _load(self) -> dict[bytes, np.ndarray]:
        if self._vectors is not None:
            return self._vectors
        vectors: dict[bytes, np.ndarray] = {}
        index = self._index()
        parts = index.get("parts", {})
        assert isinstance(parts, dict)
        for name in sorted(parts):
            path = self.folder / name
            if not path.exists():
                continue
            table = pq.read_table(path)
            keys = table.column("key").to_pylist()
            flat = table.column("vector").combine_chunks()
            width = flat.type.list_size  # type: ignore[attr-defined]
            values = np.asarray(flat.flatten(), dtype=np.float32).reshape(-1, width)
            for key, vector in zip(keys, values, strict=True):
                vectors[key] = vector
            parts[name]["used"] = time.time()
        if parts:
            self._save_index(index)
        self._vectors = vectors
        return vectors

    def _write(self, keys: list[bytes], vectors: np.ndarray) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        index = self._index()
        parts = index.setdefault("parts", {})
        assert isinstance(parts, dict)
        name = f"part-{len(parts) + 1:05d}-{int(time.time() * 1000)}.parquet"
        width = vectors.shape[1]
        array = pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.astype(np.float32).ravel(), type=pa.float32()), width
        )
        table = pa.table({"key": pa.array(keys, type=pa.binary(KEY_BYTES)), "vector": array})
        path = self.folder / name
        pq.write_table(table, path, compression="zstd")
        parts[name] = {"rows": len(keys), "bytes": path.stat().st_size, "used": time.time()}
        self._save_index(index)
        self._evict()

    def size(self) -> int:
        total = 0
        for folder in self.root.glob("*/"):
            for part in folder.glob("part-*.parquet"):
                total += part.stat().st_size
        return total

    def _evict(self) -> None:
        """R-EMB-04: past the budget, the least recently used parts (all models) are removed."""
        entries: list[tuple[float, Path, int]] = []
        for index_path in self.root.glob("*/index.json"):
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for name, meta in index.get("parts", {}).items():
                entries.append((float(meta.get("used", 0)), index_path.parent / name, 0))
        total = sum(path.stat().st_size for _, path, _ in entries if path.exists())
        for _, path, _ in sorted(entries, key=lambda item: item[0]):
            if total <= self.budget:
                break
            if not path.exists():
                continue
            total -= path.stat().st_size
            path.unlink()
            index_path = path.parent / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index.get("parts", {}).pop(path.name, None)
            index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
            if path.parent == self.folder:
                self._vectors = None

    # --- the one entry point -------------------------------------------------------------------

    def encode(self, texts: Sequence[str], encoder: Encoder) -> np.ndarray:
        """One vector per text; only the distinct values not in the cache are encoded, once."""
        keys = [value_key(text) for text in texts]
        known = self._load()
        missing: dict[bytes, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in known and key not in missing:
                missing[key] = normalise(text)
        if missing:
            new_keys = list(missing)
            vectors = np.asarray(encoder(list(missing.values())), dtype=np.float32)
            self._write(new_keys, vectors)
            known = self._vectors if self._vectors is not None else known
            for key, vector in zip(new_keys, vectors, strict=True):
                known[key] = vector
            self._vectors = known
        self.hits = len(texts) - sum(1 for k in keys if k in missing)
        self.encoded = len(missing)
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([known[key] for key in keys])
