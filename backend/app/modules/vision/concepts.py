"""Concepts of an image set (118): what the concept bottleneck and TCAV can read.

A concept is a metadata column of the table (R-VCB-01: true/false, two values, or a number; not
a column the import made), a group of images the user saved from the gallery's selection
(`datasets/<id>/vision/concepts.json`, deleted with the dataset), or, for TCAV only, the images of
a class. Groups are kept by `image_id`: a reimport that keeps the files keeps the groups.
"""

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from quantilo.vision.deep import concepts as cb
from quantilo.vision.task import column, labels_of

from app.core.errors import AppError

if TYPE_CHECKING:
    from app.modules.vision.service import VisionService
    from app.services.datasets.repository import DatasetRecord

GROUP_ID = re.compile(r"^[0-9a-f]{12}$")
TCAV_MIN = 20  # R-VTC-01
MAX_GROUPS = 50


class MoreOptionOut(BaseModel):
    """An Advanced > More model and whether it can run here (R-VMO-01), with an estimate."""

    key: Literal["finetune", "protopnet", "concept_bottleneck"]
    title: str
    available: bool
    reason: str | None
    estimate: str | None
    seconds: float | None
    note: str


class ConceptOut(BaseModel):
    key: str  # "column:<name>", "group:<id>" or "label:<class>"
    name: str
    kind: Literal["binary", "numeric"]
    source: Literal["column", "group", "label"]
    images: int  # images that have it (binary), or whose value is known (numeric)
    note: str
    bottleneck: bool  # can be a concept of the bottleneck (R-VCB-01)
    tcav: bool  # can be tested by TCAV (R-VTC-01: a binary concept with 20+ images)
    id: str | None = None  # a saved group's id
    created_at: str | None = None


class ConceptGroupIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    image_ids: list[str] = Field(min_length=1, max_length=50_000)


def _path(service: "VisionService", dataset_id: str) -> Path:
    return service.workspace.dataset_dir(dataset_id) / "vision" / "concepts.json"


def groups(service: "VisionService", dataset_id: str) -> list[dict[str, Any]]:
    path = _path(service, dataset_id)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return []
    return [g for g in data.get("groups", []) if GROUP_ID.match(str(g.get("id", "")))]


def _write(service: "VisionService", dataset_id: str, found: list[dict[str, Any]]) -> None:
    path = _path(service, dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps({"version": 1, "groups": found}), encoding="utf-8")
    partial.replace(path)


def save_group(service: "VisionService", dataset_id: str, body: ConceptGroupIn) -> ConceptOut:
    """Save the gallery's selection as a concept (R-VCB-01); its images must be in the set."""
    record = service._record(dataset_id)
    table = service._table(record)
    ids = set(str(v) for v in column(table, "image_id"))
    kept = sorted({i for i in body.image_ids if i in ids})
    if not kept:
        raise AppError("None of these images is in the set.", code="not_found", status_code=404)
    name = body.name.strip()
    found = groups(service, dataset_id)
    if any(g["name"].lower() == name.lower() for g in found):
        raise AppError(
            f"A concept named {name!r} already exists.", code="conflict", status_code=409
        )
    if len(found) >= MAX_GROUPS:
        raise AppError(
            f"At most {MAX_GROUPS} saved concepts per set: delete one first.",
            code="too_many",
            status_code=409,
        )
    group = {
        "id": uuid.uuid4().hex[:12],
        "name": name,
        "image_ids": kept,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write(service, dataset_id, [*found, group])
    return next(c for c in concept_list(service, record, table) if c.id == group["id"])


def delete_group(service: "VisionService", dataset_id: str, group_id: str) -> None:
    service._record(dataset_id)
    found = groups(service, dataset_id)
    kept = [g for g in found if g["id"] != group_id]
    if len(kept) == len(found):
        raise AppError("Concept not found.", code="not_found", status_code=404)
    _write(service, dataset_id, kept)


def _group_concepts(
    service: "VisionService", record: "DatasetRecord", table: Any
) -> list[tuple[cb.Concept, dict[str, Any]]]:
    ids = [str(v) for v in column(table, "image_id")]
    where: dict[str, list[int]] = {}
    for row, image_id in enumerate(ids):
        where.setdefault(image_id, []).append(row)
    out = []
    for g in groups(service, record.id):
        rows = sorted({r for i in g["image_ids"] for r in where.get(i, [])})
        if rows:
            out.append((cb.group_concept(f"group:{g['id']}", g["name"], rows, len(ids)), g))
    return out


def concept_list(service: "VisionService", record: "DatasetRecord", table: Any) -> list[ConceptOut]:
    """Every concept of the set: columns and saved groups (bottleneck and TCAV), classes (TCAV)."""
    out = []
    for concept in cb.column_concepts(table):
        known = np.isfinite(concept.values)
        count = int(np.nansum(concept.values)) if concept.kind == "binary" else int(known.sum())
        out.append(
            ConceptOut(
                key=concept.key,
                name=concept.name,
                kind=concept.kind,
                source="column",
                images=count,
                note=concept.note,
                bottleneck=True,
                tcav=concept.kind == "binary" and count >= TCAV_MIN,
            )
        )
    for concept, group in _group_concepts(service, record, table):
        count = int(concept.values.sum())
        out.append(
            ConceptOut(
                key=concept.key,
                name=concept.name,
                kind="binary",
                source="group",
                images=count,
                note=concept.note,
                bottleneck=count >= cb.MIN_GROUP,
                tcav=count >= TCAV_MIN,
                id=group["id"],
                created_at=group.get("created_at"),
            )
        )
    labels = [v for v in labels_of(table) if v is not None]
    for label in sorted({str(v) for v in labels}):
        count = sum(str(v) == label for v in labels)
        out.append(
            ConceptOut(
                key=f"label:{label}",
                name=label,
                kind="binary",
                source="label",
                images=count,
                note=f"The images labelled {label!r} (for TCAV only: the class itself).",
                bottleneck=False,
                tcav=count >= TCAV_MIN,
            )
        )
    return out


def engine_concepts(
    service: "VisionService", record: "DatasetRecord", table: Any, keys: list[str] | None
) -> list[cb.Concept]:
    """The bottleneck's concepts for a run (R-VCB-01): the chosen keys, or every usable one."""
    found = [c for c in cb.column_concepts(table)]
    found += [
        c for c, _ in _group_concepts(service, record, table) if c.values.sum() >= cb.MIN_GROUP
    ]
    if keys is not None:
        unknown = sorted(set(keys) - {c.key for c in found})
        if unknown:
            raise AppError(
                f"Unknown concepts: {', '.join(unknown)}.", code="not_found", status_code=404
            )
        found = [c for c in found if c.key in set(keys)]
    return found


def concept_rows(
    service: "VisionService", record: "DatasetRecord", table: Any, key: str
) -> tuple[str, list[int]]:
    """TCAV (R-VTC-01): the name and the table rows of a binary concept."""
    if key.startswith("label:"):
        label = key.removeprefix("label:")
        rows = [i for i, v in enumerate(labels_of(table)) if v is not None and str(v) == label]
        return label, rows
    if key.startswith("group:"):
        for concept, _ in _group_concepts(service, record, table):
            if concept.key == key:
                return concept.name, [int(i) for i in np.flatnonzero(concept.values == 1)]
    for concept in cb.column_concepts(table):
        if concept.key == key:
            if concept.kind != "binary":
                raise AppError(
                    "TCAV needs a concept that images have or not: this column is a number.",
                    code="not_supported",
                    status_code=422,
                )
            return concept.name, [int(i) for i in np.flatnonzero(concept.values == 1)]
    raise AppError("Concept not found.", code="not_found", status_code=404)


__all__ = [
    "ConceptGroupIn",
    "ConceptOut",
    "MoreOptionOut",
    "concept_list",
    "concept_rows",
    "delete_group",
    "engine_concepts",
    "groups",
    "save_group",
]
