"""An image set from a folder, a ZIP, a single image or a Hugging Face Parquet file (104).

The set is a dataset: a Parquet table with **one row per image** (identity, file metadata, the
measures of 103, the fingerprints, the decoding state; § 3.4 of the study), and the files stored by
content next to it (`store.py`). Tabular pages therefore work on the metadata as they are.

    images = dx.image_set("beans/", workspace="work")
    images.sentence   # "1,295 images from a folder: 3 labels (...), splits train 1,034 ..."
    images.rules      # every reading rule that fired, with its code and reason
    images.open()     # the table as a quantilo Dataset

Every automatic reading is a written rule with a code, announced in `rules` with the sentence the
import shows and its reason: R-VIS-LAY (layout), R-VIS-LAB (a labels file), R-VIS-HF (a Parquet
image column), R-VIS-ZIP (archives), R-VIS-ING (what was left out), R-VIS-DEC (decoding). Naming a
code in `ignore_rules` undoes it. Files that are not taken are listed in `skipped`, each with its
code and reason. Nothing runs in a thread or a subprocess: the same code runs in Pyodide.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import uuid
import zipfile
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.images import SVG_REASON, ImageInfo, inspect_image, sniff_image
from quantilo.vision.decode import (
    EXTENSIONS,
    FAILURE_CODE,
    MAX_FILE_BYTES,
    DecodedImage,
    try_decode,
)
from quantilo.vision.hashes import dhash, phash, pixel_sha
from quantilo.vision.layout import (
    CHECK,
    SPLIT_NAMES,
    FileLayout,
    Layout,
    RuleNote,
    normal_flags,
    read_layout,
)
from quantilo.vision.measures import MEASURE_SIDE, measure, working_copy
from quantilo.vision.store import ImageStore, check_dataset_id, image_id, sha256
from quantilo.vision.zipsafe import Member, check_zip, is_zip, read_member

if TYPE_CHECKING:
    from quantilo.dataset import Dataset

MAX_IMAGES = 50_000  # R-VIS-ING-01
MAX_DEPTH = 6  # R-VIS-ING-05
KEY_SHARE = 0.90  # R-VIS-LAB-01
LABEL_NAMES = (
    "label",
    "labels",
    "class",
    "classes",
    "class_name",
    "category",
    "target",
    "species",
    "breed",
    "diagnosis",
    "y",
)
SourceKind = Literal["image", "folder", "zip", "parquet"]

SCHEMA = pa.schema(
    [
        # identity
        ("image_id", pa.string()),
        ("file_name", pa.string()),
        ("rel_path", pa.string()),
        ("label", pa.string()),
        ("split", pa.string()),
        ("is_normal", pa.bool_()),
        ("mask_path", pa.string()),
        ("pixel_sha", pa.string()),
        ("sha256", pa.string()),
        ("stored_path", pa.string()),
        ("thumb_path", pa.string()),
        # file (084)
        ("format", pa.string()),
        ("file_bytes", pa.int64()),
        ("width", pa.int64()),
        ("height", pa.int64()),
        ("megapixels", pa.float64()),
        ("aspect_ratio", pa.float64()),
        ("color_mode", pa.string()),
        ("bit_depth", pa.int64()),
        ("frames", pa.int64()),
        ("date_taken", pa.timestamp("s")),
        ("camera", pa.string()),
        ("has_location", pa.bool_()),
        # measures (103), at 512 pixels
        ("brightness", pa.float64()),
        ("luma_p05", pa.float64()),
        ("luma_p95", pa.float64()),
        ("contrast", pa.float64()),
        ("entropy", pa.float64()),
        ("saturation", pa.float64()),
        ("colourfulness", pa.float64()),
        ("is_grayscale", pa.bool_()),
        ("sharpness", pa.float64()),
        ("clipped_dark", pa.float64()),
        ("clipped_light", pa.float64()),
        ("dominant_colour", pa.string()),
        # fingerprints
        ("phash", pa.int64()),
        ("dhash", pa.int64()),
        # state
        ("decode_ok", pa.bool_()),
        ("decode_error", pa.string()),
        ("issues", pa.list_(pa.string())),
        ("excluded", pa.bool_()),
    ]
)
COLUMNS = tuple(SCHEMA.names)

_DECODE_RULES: dict[str, tuple[str, str]] = {
    "R-VIS-DEC-01": (
        "EXIF orientation applied to {n} images, as a viewer shows them",
        "Cameras store the pixels sideways with a flag: measures and thumbnails follow what "
        "people see.",
    ),
    "R-VIS-DEC-02": (
        "Transparency flattened on white in {n} images",
        "A measure needs one colour per pixel; transparent areas are counted as white.",
    ),
    "R-VIS-DEC-03": (
        "{n} images with 16-bit or floating-point pixels scaled to 8 bits",
        "So that brightness and contrast are on the same 0-1 scale for every image.",
    ),
    "R-VIS-DEC-04": (
        "{n} animated images: the first frame is analysed",
        "A row of the table is one picture.",
    ),
    "R-VIS-DEC-05": (
        "{n} JPEG images decoded at a reduced scale",
        "The measures are taken at 512 pixels: decoding every megapixel for them only costs time.",
    ),
    "R-VIS-DEC-06": (
        "{n} images in CMYK or another colour mode converted to RGB",
        "Every measure is defined on RGB.",
    ),
}


@dataclass(frozen=True)
class Skipped:
    """A file that was not taken into the set, with its rule and reason."""

    rel_path: str
    code: str
    reason: str


@dataclass(frozen=True)
class LabelConflict:
    """R-VIS-LAY-05: the labels file and the folder disagree; the file wins."""

    rel_path: str
    folder_label: str
    file_label: str


@dataclass
class ImageSet:
    """An imported image set: its table (one row per image), where it is stored, and what was
    read by which rule."""

    dataset_id: str
    source_kind: SourceKind
    store: ImageStore
    table: pa.Table
    rules: list[RuleNote] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    conflicts: list[LabelConflict] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return self.store.root

    @property
    def table_path(self) -> Path:
        return self.store.table_path

    @property
    def n_images(self) -> int:
        return self.table.num_rows

    @property
    def n_decoded(self) -> int:
        return sum(1 for ok in self.table["decode_ok"].to_pylist() if ok)

    @property
    def labels(self) -> dict[str, int]:
        return _counts(self.table["label"].to_pylist())

    @property
    def splits(self) -> dict[str, int]:
        return _counts(self.table["split"].to_pylist())

    @property
    def sentence(self) -> str:
        n = self.n_images
        noun = "image" if n == 1 else "images"
        parts = [f"{n:,} {noun} from {'an' if self.source_kind == 'image' else 'a'} "]
        parts[0] += {
            "image": "image file",
            "folder": "folder",
            "zip": "ZIP archive",
            "parquet": "Parquet file",
        }[self.source_kind]
        details = []
        if labels := self.labels:
            shown = ", ".join(f"{k} {v:,}" for k, v in list(labels.items())[:6])
            more = ", ..." if len(labels) > 6 else ""
            details.append(f"{len(labels)} labels ({shown}{more})")
        if splits := self.splits:
            details.append("splits " + ", ".join(f"{k} {v:,}" for k, v in splits.items()))
        text = parts[0] + (": " + "; ".join(details) if details else "") + "."
        failed = n - self.n_decoded
        if failed:
            text += f" {failed:,} could not be decoded ({FAILURE_CODE})."
        if self.skipped:
            text += f" {len(self.skipped):,} files left out (see skipped)."
        return text

    def open(self, **options: Any) -> Dataset:
        """The table as a quantilo Dataset (the tabular pages work on it as on any table)."""
        from quantilo.dataset import open as open_dataset

        options.setdefault("name", self.dataset_id)
        return open_dataset(self.table_path, **options)


def _counts(values: Sequence[object]) -> dict[str, int]:
    counter = Counter(str(v) for v in values if v is not None)
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


# --- candidates ---------------------------------------------------------------------------------


@dataclass
class _Candidate:
    rel_path: str
    size: int
    fmt: str
    read: Callable[[], bytes]
    preset: FileLayout | None = None


@dataclass
class _Collected:
    candidates: list[_Candidate] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    rules: list[RuleNote] = field(default_factory=list)
    label_files: list[Path] = field(default_factory=list)
    meta: pa.Table | None = None  # HF: the other columns, one row per candidate
    resources: list[zipfile.ZipFile] = field(default_factory=list)  # closed after the build


def _not_image(head: bytes, rel: str, collected: _Collected, others: Counter[str]) -> None:
    info = inspect_image(head)
    if info.format == "svg":
        collected.skipped.append(Skipped(rel, "R-VIS-ING-06", SVG_REASON))
    else:
        others[PurePosixPath(rel).suffix.lower() or "(none)"] += 1


def _limits(
    collected: _Collected, others: Counter[str], hidden: int, deep: int, limit: int, hit: bool
) -> None:
    if hit:
        collected.rules.append(
            RuleNote(
                "R-VIS-ING-01",
                f"Only the first {limit:,} images were taken (limit).",
                "Beyond this an import takes hours and the gallery stops being browsable: split "
                "the set.",
                limit,
            )
        )
    if others:
        total = sum(others.values())
        kinds = ", ".join(f"{k} {v:,}" for k, v in others.most_common(5))
        collected.rules.append(
            RuleNote(
                "R-VIS-ING-02",
                f"{total:,} files that are not images were left out ({kinds}).",
                "The format is read from the first bytes (R-IMG-01), never from the name: a "
                "folder holds all sorts of files.",
                total,
            )
        )
    if hidden:
        collected.rules.append(
            RuleNote(
                "R-VIS-ING-04",
                f"{hidden:,} hidden files, links and system folders ignored.",
                "Links can point outside the folder; hidden files are the system's, not images.",
                hidden,
            )
        )
    if deep:
        collected.rules.append(
            RuleNote(
                "R-VIS-ING-05",
                f"{deep:,} folders deeper than {MAX_DEPTH} levels were not opened.",
                "An image set is a few levels deep; deeper folders are usually a copied disk.",
                deep,
            )
        )


def _hidden_name(name: str) -> bool:
    return name.startswith(".") or name in ("__MACOSX", "Thumbs.db", "desktop.ini")


def _collect_folder(root: Path, max_images: int) -> _Collected:
    collected = _Collected()
    others: Counter[str] = Counter()
    hidden = deep = 0
    hit = False
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and not hit:
        folder, depth = stack.pop()
        try:
            entries = sorted(os.scandir(folder), key=lambda e: e.name)
        except OSError:
            continue
        subfolders: list[Path] = []
        for entry in entries:
            path = Path(entry.path)
            if _hidden_name(entry.name) or entry.is_symlink() or path.is_junction():
                hidden += 1
                continue
            if entry.is_dir(follow_symlinks=False):
                if depth + 1 > MAX_DEPTH:
                    deep += 1
                else:
                    subfolders.append(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                continue
            rel = path.relative_to(root).as_posix()
            size = entry.stat(follow_symlinks=False).st_size
            if depth == 0 and entry.name.lower().endswith(".csv"):
                collected.label_files.append(path)
                continue
            with path.open("rb") as handle:
                head = handle.read(8192)
            fmt = sniff_image(head[:64])
            if fmt is None:
                _not_image(head, rel, collected, others)
                continue
            if size > MAX_FILE_BYTES:
                collected.skipped.append(
                    Skipped(
                        rel,
                        "R-VIS-ING-03",
                        f"{size / 2**20:,.0f} MiB, over the {MAX_FILE_BYTES // 2**20} MiB limit "
                        "for one image.",
                    )
                )
                continue
            if len(collected.candidates) >= max_images:
                hit = True
                break
            collected.candidates.append(_Candidate(rel, size, fmt, path.read_bytes))
        stack.extend((p, depth + 1) for p in reversed(subfolders))
    _limits(collected, others, hidden, deep, max_images, hit)
    return collected


def _collect_zip(path: Path, max_images: int) -> _Collected:
    plan = check_zip(path)
    collected = _Collected()
    others: Counter[str] = Counter()
    hidden = 0
    hit = False
    for skipped in plan.skipped:
        if skipped.code == "R-VIS-ZIP-06" and "hidden" in skipped.reason:
            hidden += 1
        else:
            collected.skipped.append(Skipped(skipped.name, skipped.code, skipped.reason))
    if any(m.name.lower().endswith(".csv") and "/" not in m.name for m in plan.members):
        collected.rules.append(
            RuleNote(
                "R-VIS-LAB-03",
                "A CSV at the root of the archive was not read as labels: pass it as labels= "
                "after extracting it.",
                "Labels from a file are read only from a file you can check.",
            )
        )
    archive = zipfile.ZipFile(plan.path)
    collected.resources.append(archive)
    for member in plan.members:
        try:
            head = read_member(archive, member, limit=8192)
        except Exception as error:  # a damaged member (bad CRC, unsupported method)
            collected.skipped.append(
                Skipped(member.name, "R-VIS-ZIP-07", f"This member cannot be read: {error}.")
            )
            continue
        fmt = sniff_image(head[:64])
        if fmt is None:
            _not_image(head, member.name, collected, others)
            continue
        if len(collected.candidates) >= max_images:
            hit = True
            break

        def read(member: Member = member) -> bytes:
            return read_member(archive, member)

        collected.candidates.append(_Candidate(member.name, member.size, fmt, read))
    _limits(collected, others, hidden, 0, max_images, hit)
    return collected


# --- Hugging Face Parquet -----------------------------------------------------------------------


def _image_column(pf: pq.ParquetFile) -> tuple[str, str] | None:
    """(column, "struct" | "binary") of the image column, if any (R-VIS-HF-01)."""
    schema = pf.schema_arrow
    for name in schema.names:
        kind = schema.field(name).type
        if pa.types.is_struct(kind):
            fields = {kind.field(i).name: kind.field(i).type for i in range(kind.num_fields)}
            data = fields.get("bytes")
            if data is not None and (pa.types.is_binary(data) or pa.types.is_large_binary(data)):
                return name, "struct"
    for name in schema.names:
        kind = schema.field(name).type
        if pa.types.is_binary(kind) or pa.types.is_large_binary(kind):
            for value in pf.read_row_group(0, columns=[name])[name].to_pylist()[:20]:
                if value:
                    if sniff_image(value[:64]):
                        return name, "binary"
                    break
    return None


def _class_names(pf: pq.ParquetFile) -> dict[str, list[str]]:
    """ClassLabel names of the HF `datasets` metadata, by column."""
    raw = (pf.schema_arrow.metadata or {}).get(b"huggingface")
    if not raw:
        return {}
    try:
        meta = json.loads(raw)
    except ValueError:
        return {}
    features = meta.get("info", {}).get("features") or meta.get("features") or {}
    names: dict[str, list[str]] = {}
    if isinstance(features, dict):
        for column, spec in features.items():
            if isinstance(spec, dict) and spec.get("_type") == "ClassLabel" and spec.get("names"):
                names[column] = [str(n) for n in spec["names"]]
    return names


def _split_from_name(path: Path) -> str | None:
    for text in (path.stem, path.parent.name):
        for token in re.split(r"[-_.\s]+", text.lower()):
            if token in SPLIT_NAMES:
                return SPLIT_NAMES[token]
    return None


def _collect_parquet(path: Path, max_images: int, label_column: str | None) -> _Collected:
    pf = pq.ParquetFile(path)
    found = _image_column(pf)
    if found is None:
        raise IngestionError(
            "No column of this Parquet file holds images: open it as a table (dx.open)."
        )
    column, kind = found
    collected = _Collected()
    held = "bytes and path" if kind == "struct" else "bytes"
    collected.rules.append(
        RuleNote(
            "R-VIS-HF-01",
            f"Images read from the column {column!r} ({held}) (R-VIS-HF-01). {CHECK}",
            "Hugging Face `datasets` stores each image as its file bytes in a column.",
            pf.metadata.num_rows,
        )
    )
    others = [n for n in pf.schema_arrow.names if n != column]
    table = pf.read(columns=others) if others else pa.table({})
    n_rows = pf.metadata.num_rows
    names = _class_names(pf)
    lab = label_column if label_column in others else None
    if lab is None:
        lab = next((n for n in others if n.lower() in LABEL_NAMES), None)
    labels: list[str | None] = [None] * n_rows
    if lab is not None:
        values = table[lab].to_pylist()
        mapping = names.get(lab)
        if mapping:
            collected.rules.append(
                RuleNote(
                    "R-VIS-HF-02",
                    f"Label numbers of {lab!r} read as their class names "
                    f"({', '.join(mapping[:6])}{', ...' if len(mapping) > 6 else ''}) "
                    f"(R-VIS-HF-02). {CHECK}",
                    "The file's own metadata (ClassLabel) names each number.",
                )
            )
        for i, v in enumerate(values):
            if v is None:
                continue
            if mapping and isinstance(v, int) and 0 <= v < len(mapping):
                labels[i] = mapping[v]
            else:
                labels[i] = str(v)
        collected.rules.append(
            RuleNote(
                "R-VIS-LAB-02",
                f"Labels read from the column {lab!r} (R-VIS-LAB-02). {CHECK}",
                "Its name says it holds the class of each image.",
            )
        )
        table = table.drop_columns([lab])
    split = _split_from_name(path)
    if split:
        collected.rules.append(
            RuleNote(
                "R-VIS-HF-03",
                f"Split {split!r} read from the file name {path.name!r} (R-VIS-HF-03). {CHECK}",
                "Hugging Face names each split's file train-..., test-... or validation-....",
                n_rows,
            )
        )
    group_of: list[tuple[int, int]] = []
    for g in range(pf.metadata.num_row_groups):
        group_of.extend((g, j) for j in range(pf.metadata.row_group(g).num_rows))
    cache: dict[int, list[Any]] = {}

    def cell(i: int) -> Any:
        g, j = group_of[i]
        if g not in cache:
            cache.clear()
            cache[g] = pf.read_row_group(g, columns=[column])[column].to_pylist()
        return cache[g][j]

    used: set[str] = set()
    keep: list[int] = []
    for i in range(n_rows):
        value = cell(i)
        data = value.get("bytes") if isinstance(value, dict) else value
        given = value.get("path") if isinstance(value, dict) else None
        if not data:
            collected.skipped.append(
                Skipped(f"row {i}", "R-VIS-HF-04", "This row has no image bytes.")
            )
            continue
        fmt = sniff_image(bytes(data[:64]))
        base = PurePosixPath(str(given).replace("\\", "/")).name if given else ""
        if fmt is None:
            info = inspect_image(bytes(data[:8192]))
            code = "R-VIS-ING-06" if info.format == "svg" else "R-VIS-ING-02"
            collected.skipped.append(
                Skipped(base or f"row {i}", code, info.reason or "Not an image.")
            )
            continue
        if len(keep) >= max_images:
            collected.rules.append(
                RuleNote(
                    "R-VIS-ING-01",
                    f"Only the first {max_images:,} images were taken (limit).",
                    "Beyond this an import takes hours: split the set.",
                    max_images,
                )
            )
            break
        # only the base name of the producer's path is kept: it may be a path on their machine
        rel = base or f"row_{i:06d}{EXTENSIONS[fmt]}"
        if rel in used:
            rel = f"{i:06d}_{rel}"
        used.add(rel)
        keep.append(i)

        def read(i: int = i) -> bytes:
            item = cell(i)
            return bytes((item.get("bytes") if isinstance(item, dict) else item) or b"")

        collected.candidates.append(
            _Candidate(rel, len(data), fmt, read, FileLayout(rel, label=labels[i], split=split))
        )
    collected.meta = table.take(pa.array(keep, pa.int64())) if table.num_columns else None
    return collected


# --- a labels file ------------------------------------------------------------------------------


def _read_table(path: Path) -> pa.Table:
    try:
        if path.suffix.lower() == ".parquet":
            return pq.read_table(path)
        return pacsv.read_csv(path)
    except (pa.ArrowInvalid, OSError) as error:
        raise IngestionError(f"The labels file {path.name!r} cannot be read: {error}") from None


def _normalise(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.casefold()


def _suffix_index(paths: Sequence[str]) -> dict[str, list[int]]:
    """Every trailing part of every path, with and without the extension -> images."""
    index: dict[str, list[int]] = {}
    for i, path in enumerate(paths):
        parts = path.split("/")
        for k in range(len(parts)):
            tail = "/".join(parts[k:])
            stem = str(PurePosixPath(tail).with_suffix("")) if PurePosixPath(tail).suffix else tail
            for key in {tail.casefold(), stem.casefold()}:
                bucket = index.setdefault(key, [])
                if not bucket or bucket[-1] != i:
                    bucket.append(i)
    return index


@dataclass
class _Labels:
    labels: dict[int, str | None]  # image -> label from the file
    rows: list[int | None]  # image -> row of the file
    meta: pa.Table | None
    rules: list[RuleNote]


def _match_labels(
    path: Path, paths: Sequence[str], label_column: str | None, *, found_at_root: bool
) -> _Labels | None:
    table = _read_table(path)
    index = _suffix_index(paths)
    best: tuple[float, str, list[int | None]] | None = None
    for name in table.column_names:
        values = table[name].to_pylist()
        present = [v for v in values if v is not None and str(v).strip()]
        if not present:
            continue
        hits: list[int | None] = []
        for v in values:
            matches = index.get(_normalise(v)) if v is not None else None
            hits.append(matches[0] if matches and len(matches) == 1 else None)
        share = sum(1 for h in hits if h is not None) / len(present)
        if best is None or share > best[0]:
            best = (share, name, hits)
    if best is None or best[0] < KEY_SHARE:
        if found_at_root:
            return None  # a CSV lying in the folder that does not name the images
        shown = f" (best: {best[1]!r}, {best[0]:.0%})" if best else ""
        raise IngestionError(
            f"No column of {path.name!r} names the images: at least {KEY_SHARE:.0%} of a "
            f"column's values must be an image's name or path{shown} (R-VIS-LAB-01)."
        )
    share, key, hits = best
    rules = [
        RuleNote(
            "R-VIS-LAB-01",
            f"Images matched to {path.name!r} by its column {key!r} ({share:.0%} of its values "
            f"name an image) (R-VIS-LAB-01). {CHECK}",
            "A column whose values are the images' names or paths is the key of a labels file "
            "(the usual Kaggle layout).",
            sum(1 for h in hits if h is not None),
        )
    ]
    if found_at_root:
        rules.insert(
            0,
            RuleNote(
                "R-VIS-LAB-03",
                f"{path.name!r}, at the root of the folder, read as the labels file "
                f"(R-VIS-LAB-03). {CHECK}",
                "Its column of image names says it describes these images.",
            ),
        )
    others = [n for n in table.column_names if n != key]
    lab: str | None = None
    if label_column is not None:
        if label_column not in others:
            raise IngestionError(f"{path.name!r} has no column {label_column!r}.")
        lab = label_column
        why = "the column you chose"
    else:
        lab = next((n for n in others if n.lower() in LABEL_NAMES), None)
        why = "its name says it holds the class"
        if lab is None and len(others) == 1:
            lab, why = others[0], "the only other column"
    if lab is not None:
        rules.append(
            RuleNote(
                "R-VIS-LAB-02",
                f"Labels read from the column {lab!r} of {path.name!r} (R-VIS-LAB-02). {CHECK}",
                f"It is {why}.",
            )
        )
    labels: dict[int, str | None] = {}
    rows: list[int | None] = [None] * len(paths)
    values = table[lab].to_pylist() if lab is not None else [None] * table.num_rows
    duplicates = unmatched = 0
    for r, i in enumerate(hits):
        if i is None:
            unmatched += 1
            continue
        if rows[i] is not None:
            duplicates += 1
            continue
        rows[i] = r
        if lab is not None and values[r] is not None and str(values[r]).strip():
            labels[i] = str(values[r]).strip()
    if unmatched or duplicates:
        parts = []
        if unmatched:
            parts.append(f"{unmatched:,} rows name no image of the set")
        if duplicates:
            parts.append(f"{duplicates:,} rows repeat an image (the first one is kept)")
        rules.append(
            RuleNote(
                "R-VIS-LAB-04",
                f"In {path.name!r}: {' and '.join(parts)}.",
                "A labels file often covers more (or other) images than the folder.",
                unmatched + duplicates,
            )
        )
    meta_names = [n for n in others if n != lab]
    meta = None
    if meta_names:
        taken = table.select(meta_names).take(pa.array(rows, pa.int64()))
        meta = taken
    return _Labels(labels, rows, meta, rules)


# --- building the table -------------------------------------------------------------------------


def _iso(text: str | None) -> dt.datetime | None:
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def _camera(info: ImageInfo) -> str | None:
    exif = info.exif
    if exif is None:
        return None
    make, model = exif.make, exif.model
    if make and model:
        return model if model.lower().startswith(make.split()[0].lower()) else f"{make} {model}"
    return model or make


def _file_fields(
    info: ImageInfo, data: bytes, width: int | None, height: int | None
) -> dict[str, Any]:
    exif = info.exif
    return {
        "format": info.format,
        "file_bytes": len(data),
        "width": width,
        "height": height,
        "megapixels": round(width * height / 1e6, 4) if width and height else None,
        "aspect_ratio": round(width / height, 6) if width and height else None,
        "color_mode": info.color_mode,
        "bit_depth": info.bit_depth,
        "frames": info.frames,
        "date_taken": _iso(exif.date_taken if exif else None),
        "camera": _camera(info),
        "has_location": bool(exif and exif.has_location),
    }


@dataclass
class _Build:
    store: ImageStore
    side: int
    thumbnails: bool
    decode_notes: Counter[str] = field(default_factory=Counter)
    skipped: list[Skipped] = field(default_factory=list)


def _row(c: _Candidate, entry: FileLayout, b: _Build, mask: _Candidate | None) -> dict[str, Any]:
    data = c.read()
    digest = sha256(data)
    _, stored = b.store.put_file(data, EXTENSIONS[c.fmt], digest)
    name = PurePosixPath(c.rel_path).name
    result = try_decode(data, name=name, reduce_to=b.side)
    row: dict[str, Any] = dict.fromkeys(COLUMNS)
    row.update(
        image_id=image_id(digest),
        file_name=name,
        rel_path=c.rel_path,
        label=entry.label,
        split=entry.split,
        is_normal=entry.is_normal,
        sha256=digest,
        stored_path=stored,
    )
    if isinstance(result, DecodedImage):
        decoded = result
        for note in decoded.notes:
            b.decode_notes[note[:12]] += 1
        row.update(_file_fields(decoded.info, data, decoded.width, decoded.height))
        row["bit_depth"] = decoded.bit_depth
        row["frames"] = decoded.frames
        copy = working_copy(decoded.image, b.side)
        m = measure(copy, side=b.side, prepared=True)
        row.update(
            brightness=m.brightness,
            luma_p05=m.luma_p05,
            luma_p95=m.luma_p95,
            contrast=m.contrast,
            entropy=m.entropy,
            saturation=m.saturation,
            colourfulness=m.colourfulness,
            is_grayscale=m.is_grayscale,
            sharpness=m.sharpness,
            clipped_dark=m.clipped_dark,
            clipped_light=m.clipped_light,
            dominant_colour=m.dominant_colour,
            phash=phash(copy),
            dhash=dhash(copy),
            pixel_sha=pixel_sha(decoded.image),
            decode_ok=True,
            issues=[],
            excluded=False,
        )
        if b.thumbnails:
            row["thumb_path"] = b.store.put_thumbnail(digest, decoded.image)
    else:
        info = inspect_image(data, name=name)
        row.update(_file_fields(info, data, info.display_width, info.display_height))
        row.update(
            decode_ok=False, decode_error=result.reason, issues=[FAILURE_CODE], excluded=True
        )
    if mask is not None:
        decoded_mask = try_decode(mask.read(), name=PurePosixPath(mask.rel_path).name)
        if isinstance(decoded_mask, DecodedImage):
            row["mask_path"] = b.store.put_mask(digest, decoded_mask.image)
        else:
            b.skipped.append(Skipped(mask.rel_path, "R-VIS-LAY-04", decoded_mask.reason))
    return row


def _decode_rules(notes: Counter[str]) -> list[RuleNote]:
    rules = []
    for code in sorted(notes):
        if code in _DECODE_RULES:
            text, reason = _DECODE_RULES[code]
            n = notes[code]
            rules.append(RuleNote(code, text.format(n=f"{n:,}") + f" ({code}).", reason, n))
    return rules


def _build(
    kind: SourceKind,
    collected: _Collected,
    store: ImageStore,
    *,
    labels: Path | None,
    label_column: str | None,
    ignore: Collection[str],
    side: int,
    thumbnails: bool,
) -> ImageSet:
    rules = list(collected.rules)
    skipped = list(collected.skipped)
    by_path = {c.rel_path: c for c in collected.candidates}
    if kind in ("folder", "zip"):
        layout = read_layout(list(by_path), ignore=ignore)
    else:
        layout = Layout(files=[c.preset or FileLayout(c.rel_path) for c in collected.candidates])
    rules.extend(layout.rules)
    skipped.extend(
        Skipped(p, "R-VIS-LAY-04", "A mask without a matching image.") for p in layout.orphan_masks
    )
    entries = list(layout.files)
    paths = [e.rel_path for e in entries]

    # a labels file: given, or the only CSV at the root of a folder that names the images
    matched: _Labels | None = None
    conflicts: list[LabelConflict] = []
    if labels is not None:
        matched = _match_labels(labels, paths, label_column, found_at_root=False)
    elif (
        kind == "folder"
        and len(collected.label_files) == 1
        and "R-VIS-LAB-03" not in ignore
        and entries
    ):
        found = collected.label_files[0]
        matched = _match_labels(found, paths, label_column, found_at_root=True)
        if matched is None:
            rules.append(
                RuleNote(
                    "R-VIS-LAB-03",
                    f"{found.name!r}, at the root of the folder, does not name the images: it was "
                    "not read as labels (pass labels= to use it).",
                    f"At least {KEY_SHARE:.0%} of one of its columns must be image names.",
                )
            )
    if matched is not None:
        rules.extend(matched.rules)
        for i, file_label in matched.labels.items():
            folder_label = entries[i].label
            if folder_label is not None and file_label is not None and folder_label != file_label:
                conflicts.append(LabelConflict(entries[i].rel_path, folder_label, file_label))
            entries[i] = replace(entries[i], label=file_label)
        if conflicts:
            rules.append(
                RuleNote(
                    "R-VIS-LAY-05",
                    f"{len(conflicts):,} images have one label in the file and another in the "
                    f"folders: the file's label was kept (R-VIS-LAY-05). {CHECK}",
                    "The labels file is explicit; folder names are a guess.",
                    len(conflicts),
                )
            )
        if matched.labels:
            normal, note = normal_flags({e.rel_path: e.label for e in entries}, ignore=ignore)
            rules = [r for r in rules if r.code != "R-VIS-LAY-03"]
            if note is not None:
                rules.append(note)
            entries = [replace(e, is_normal=normal[e.rel_path]) for e in entries]

    b = _Build(store, side, thumbnails)
    rows = [
        _row(by_path[e.rel_path], e, b, by_path.get(e.mask) if e.mask else None) for e in entries
    ]
    skipped.extend(b.skipped)
    rules.extend(_decode_rules(b.decode_notes))
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    extra = []
    if matched is not None and matched.meta is not None:
        extra.append(matched.meta)
    if collected.meta is not None:
        extra.append(collected.meta)
    for meta in extra:
        for name in meta.column_names:
            column = name if name not in table.column_names else f"meta_{name}"
            table = table.append_column(column, meta[name])
    store.table_path.parent.mkdir(parents=True, exist_ok=True)
    partial = store.table_path.with_name(store.table_path.name + ".partial")
    pq.write_table(table, partial, compression="zstd")
    os.replace(partial, store.table_path)
    return ImageSet(
        dataset_id=store.dataset_id,
        source_kind=kind,
        store=store,
        table=table,
        rules=rules,
        skipped=skipped,
        conflicts=conflicts,
    )


def _head(path: Path, n: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(n)


def _default_workspace() -> Path:
    from quantilo.dataset import default_cache_dir

    return default_cache_dir() / "workspace"


def image_set(
    source: str | os.PathLike[str],
    *,
    workspace: str | os.PathLike[str] | None = None,
    dataset_id: str | None = None,
    labels: str | os.PathLike[str] | None = None,
    label_column: str | None = None,
    ignore_rules: Collection[str] = (),
    thumbnails: bool = True,
    measure_side: int = MEASURE_SIDE,
    max_images: int = MAX_IMAGES,
) -> ImageSet:
    """Import an image set (see the module) from a folder, a ZIP, a Parquet file with an image
    column (Hugging Face `datasets`) or a single image.

    `labels` is a CSV or Parquet file whose key column names the images (R-VIS-LAB-01);
    `label_column` picks its label column. `ignore_rules` undoes reading rules by code (for example
    `{"R-VIS-LAY-01"}`: ignore folder labels). Files go to `workspace/images/<dataset_id>/`, the
    table to `workspace/datasets/<dataset_id>.parquet`."""
    path = Path(source)
    if not path.exists():
        from quantilo.errors import NotFoundError

        raise NotFoundError(f"File {path.name!r}")
    store = ImageStore(
        Path(workspace) if workspace is not None else _default_workspace(),
        check_dataset_id(dataset_id) if dataset_id else f"img-{uuid.uuid4().hex[:12]}",
    )
    ignore = frozenset(ignore_rules)
    label_path = Path(labels) if labels is not None else None
    if label_path is not None and not label_path.is_file():
        from quantilo.errors import NotFoundError

        raise NotFoundError(f"File {label_path.name!r}")
    kind: SourceKind
    if path.is_dir():
        kind, collected = "folder", _collect_folder(path, max_images)
    elif is_zip(path):
        kind, collected = "zip", _collect_zip(path, max_images)
    elif path.suffix.lower() == ".parquet" or _head(path, 4) == b"PAR1":
        kind, collected = "parquet", _collect_parquet(path, max_images, label_column)
    else:
        head = _head(path, 8192)
        fmt = sniff_image(head[:64])
        if fmt is None:
            info = inspect_image(head, name=path.name)
            raise IngestionError(info.reason or f"{path.name!r} is not an image.")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise IngestionError(
                f"{size / 2**20:,.0f} MiB, over the {MAX_FILE_BYTES // 2**20} MiB limit for one "
                "image (R-VIS-ING-03)."
            )
        kind = "image"
        collected = _Collected(candidates=[_Candidate(path.name, size, fmt, path.read_bytes)])
    try:
        return _build(
            kind,
            collected,
            store,
            labels=label_path,
            label_column=label_column,
            ignore=ignore,
            side=measure_side,
            thumbnails=thumbnails,
        )
    finally:
        for resource in collected.resources:
            resource.close()


__all__ = [
    "COLUMNS",
    "LABEL_NAMES",
    "MAX_DEPTH",
    "MAX_IMAGES",
    "SCHEMA",
    "ImageSet",
    "LabelConflict",
    "Skipped",
    "image_set",
]
