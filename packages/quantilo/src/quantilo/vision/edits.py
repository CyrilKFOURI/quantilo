"""The user's edits of an image set, applied to its table in one pass (108, c3).

- **Decisions** (108): `exclude` or `keep` by `rel_path`. `rule_excluded` keeps what the rules
  said, `excluded = decision == exclude OR (rule_excluded AND decision != keep)` (R-108-05, 07).
- **Labels** (R-VED-01, R-VED-02): a new label by `rel_path`; `label_original` keeps the label read
  at the import (written at the first application, never changed after).
- **Pins** (R-VED-03): `pinned`, a mark to come back to; no effect on models.
- **Thresholds** (R-VQT-01, R-VQT-02): the R-VQ rules run again on the table (no pixel decoded: the
  measures are in it), so that new thresholds and new labels (R-VQ-12, 13, 14 read them) are seen.

`DERIVED` are the columns these edits write: never a concept (118) nor a target (R-VML-05, 06),
R-VED-04. Pure functions on pyarrow tables, no thread: the browser version runs them too.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

DERIVED = ("decision", "rule_excluded", "label_original", "pinned")
DECISIONS = ("exclude", "keep")
QUALITY_VECTORS = "quality.parquet"  # R-VQT-03: in the set's `embeddings/` folder
QUALITY_KEY = "quality-v1|hsv-hog-lbp|jl64|seed105"


@dataclass
class Edits:
    """What the user changed, by the image's `rel_path` (unique; exact copies share an image_id)."""

    decisions: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)
    pins: Mapping[str, bool] = field(default_factory=dict)


def _set(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), name, values)
    return table.append_column(name, values)


def decided_table(table: pa.Table, decisions: Mapping[str, str]) -> pa.Table:
    """The table with the decisions (108): `decision`, `rule_excluded` (what the rules said, kept
    from the first application) and `excluded` (R-108-05)."""
    keys = table["rel_path"].to_pylist()
    source = "rule_excluded" if "rule_excluded" in table.column_names else "excluded"
    by_rule = [bool(value) for value in table[source].to_pylist()]
    chosen = [decisions.get(key) for key in keys]
    excluded = [
        choice == "exclude" or (rule and choice != "keep")
        for choice, rule in zip(chosen, by_rule, strict=True)
    ]
    table = _set(table, "excluded", pa.array(excluded, type=pa.bool_()))
    table = _set(table, "rule_excluded", pa.array(by_rule, type=pa.bool_()))
    return _set(table, "decision", pa.array(chosen, type=pa.string()))


def labelled_table(table: pa.Table, labels: Mapping[str, str]) -> pa.Table:
    """R-VED-02: `label_original` is the label read at the import (kept once written); `label`
    is the user's label when there is one, else the original."""
    keys = table["rel_path"].to_pylist()
    names = table.column_names
    if "label_original" in names:
        original = table["label_original"].to_pylist()
    elif "label" in names:
        original = table["label"].to_pylist()
    else:
        original = [None] * table.num_rows
    if not labels and "label_original" not in names:
        return table
    new = [labels.get(k) or o for k, o in zip(keys, original, strict=True)]
    table = _set(table, "label_original", pa.array(original, type=pa.string()))
    return _set(table, "label", pa.array(new, type=pa.string()))


def pinned_table(table: pa.Table, pins: Mapping[str, bool]) -> pa.Table:
    """R-VED-03: `pinned`, False when not pinned (the column appears with the first pin)."""
    if not pins and "pinned" not in table.column_names:
        return table
    keys = table["rel_path"].to_pylist()
    return _set(table, "pinned", pa.array([bool(pins.get(k)) for k in keys], type=pa.bool_()))


def load_descriptors(root: str | Path, keys: list[str]) -> np.ndarray | None:
    """R-VQT-03: the 64 projected values of each image kept at the import, in the order of `keys`
    (NaN when an image has none); None when the set has none (imported before c3)."""
    from quantilo.vision import vectors

    found = vectors.load(Path(root) / "embeddings" / QUALITY_VECTORS, QUALITY_KEY)
    if not found:
        return None
    width = len(next(iter(found.values())))
    out = np.full((len(keys), width), np.nan, dtype=np.float32)
    for i, key in enumerate(keys):
        vector = found.get(key)
        if vector is not None:
            out[i] = vector
    return out


def save_descriptors(root: str | Path, keys: list[str], matrix: np.ndarray) -> None:
    """R-VQT-03: kept beside the set's other vectors, by `rel_path`."""
    from quantilo.vision import vectors

    vectors.save(Path(root) / "embeddings" / QUALITY_VECTORS, QUALITY_KEY, keys, matrix)


NO_DESCRIPTORS = (
    "R-VQ-15's content part needs the descriptors kept at the import; this set was imported "
    "before they were kept: reimport it to use them (R-VQT-03)."
)


def edited_table(
    table: pa.Table,
    edits: Edits,
    *,
    thresholds: Any = None,
    ignore: Collection[str] = (),
    descriptors: np.ndarray | None = None,
    reassess: bool = True,
) -> tuple[pa.Table, Any]:
    """Every edit on the table, in order: labels, then the R-VQ rules again (with `thresholds`,
    `ignore` and `descriptors`), then the decisions over the rules' new exclusions, then pins.
    Returns (table, the new QualityReport or None when `reassess` is False)."""
    from quantilo.vision.quality import KNN_MIN_IMAGES, assess

    table = labelled_table(table, edits.labels)
    report = None
    if reassess:
        report = assess(table, thresholds=thresholds, ignore=ignore, descriptors=descriptors)
        decoded = sum(1 for v in table["decode_ok"].to_pylist() if v)
        if descriptors is None and "R-VQ-15" not in ignore and decoded >= KNN_MIN_IMAGES:
            report.notes.append(NO_DESCRIPTORS)
        table = report.apply(table)
        # the rules' exclusions become the base the decisions are read against (R-108-07)
        table = _set(table, "rule_excluded", table["excluded"])
    table = decided_table(table, edits.decisions)
    table = pinned_table(table, edits.pins)
    return table, report


__all__ = [
    "DECISIONS",
    "DERIVED",
    "NO_DESCRIPTORS",
    "QUALITY_KEY",
    "QUALITY_VECTORS",
    "Edits",
    "decided_table",
    "edited_table",
    "labelled_table",
    "load_descriptors",
    "pinned_table",
    "save_descriptors",
]
