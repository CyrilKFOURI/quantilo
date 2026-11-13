"""How a folder of images is laid out: labels, splits, normal images and masks (104, § 3.2).

Read from the relative paths only, by written rules (R-VIS-LAY-xx). Every rule that fires is
announced with its reason and the sentence the import preview shows, and can be undone by naming its
code in `ignore` (the "Ignore folder labels" switch is `ignore={"R-VIS-LAY-01"}`).

- R-VIS-LAY-01: >= 2 sibling folders holding images, and >= 80 % of the images under them:
  `label` = the folder name (the ImageFolder convention).
- R-VIS-LAY-02: a folder `train`, `test`, `val`, `valid` or `validation` (any case, any level):
  `split` (val, valid and validation written `val`); the label is read under it.
- R-VIS-LAY-03: labels `good`, `ok`, `normal`, `nominal`, `pass`: `is_normal = true`, every other
  label a defect type (MVTec AD, VisA).
- R-VIS-LAY-04: `ground_truth/<type>/<name>_mask.png`, or a `masks` folder with matching names: a
  pixel mask linked to its image.
- R-VIS-LAY-05: a labels file and the folders disagree: the file wins, each conflict counted and
  listed (in `ingest`).
- R-VIS-LAY-06: a folder with no image file but Parquet files holding an image column (the Hugging
  Face layout `data/train-00000-of-00001.parquet`, `validation-...`, `test-...`): every such file
  is read, and its `split` comes from its file name (`split_of_file`); a file whose name names no
  split leaves its images without one. Undone by naming the code: no split is read from the names.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

LABEL_SHARE = 0.80  # R-VIS-LAY-01
SPLIT_NAMES = {
    "train": "train",
    "training": "train",
    "test": "test",
    "testing": "test",
    "val": "val",
    "valid": "val",
    "validation": "val",
}
NORMAL_LABELS = frozenset({"good", "ok", "normal", "nominal", "pass"})
MASK_FOLDERS = frozenset({"ground_truth", "groundtruth", "masks", "mask"})
CHECK = "Check that your data was read correctly."


@dataclass(frozen=True)
class RuleNote:
    """One reading rule that fired: its code, the sentence shown, why it exists, how many files."""

    code: str
    message: str
    reason: str
    count: int | None = None


@dataclass(frozen=True)
class FileLayout:
    rel_path: str
    label: str | None = None
    split: str | None = None
    is_normal: bool | None = None
    mask: str | None = None  # the relative path of its mask (R-VIS-LAY-04)


@dataclass
class Layout:
    """What the paths say: one entry per image (masks are not images), the rules that fired."""

    files: list[FileLayout]
    rules: list[RuleNote] = field(default_factory=list)
    masks: set[str] = field(default_factory=set)  # paths read as masks, not as images
    orphan_masks: list[str] = field(default_factory=list)  # masks without their image

    def by_path(self) -> dict[str, FileLayout]:
        return {f.rel_path: f for f in self.files}


def _parts(rel_path: str) -> tuple[str, ...]:
    return PurePosixPath(rel_path.replace("\\", "/")).parts


def _split_index(dirs: tuple[str, ...]) -> int | None:
    for i, name in enumerate(dirs):
        if name.lower() in SPLIT_NAMES:
            return i
    return None


def _mask_key(parts: tuple[str, ...]) -> tuple[tuple[str, str], str] | None:
    """(key, kind) of a mask path: ground_truth/<type>/<name>_mask.png keys on (type, name);
    masks/<name>(_mask).png keys on ("", name)."""
    dirs = [p.lower() for p in parts[:-1]]
    for i, name in enumerate(dirs):
        if name in MASK_FOLDERS:
            stem = PurePosixPath(parts[-1]).stem
            stem = stem.removesuffix("_mask").removesuffix("-mask")
            below = parts[i + 1 : -1]
            kind = "ground_truth" if name in ("ground_truth", "groundtruth") else "masks"
            return ((below[-1] if below and kind == "ground_truth" else ""), stem), kind
    return None


def normal_flags(
    labels: Mapping[str, str | None], *, ignore: Collection[str] = ()
) -> tuple[dict[str, bool | None], RuleNote | None]:
    """R-VIS-LAY-03: when some label is a "normal" word, `is_normal` for every labelled image
    (true for the normal words, false for the other labels, the defect types)."""
    normal: dict[str, bool | None] = dict.fromkeys(labels)
    found = {lab for lab in labels.values() if lab is not None and lab.lower() in NORMAL_LABELS}
    if not found or "R-VIS-LAY-03" in ignore:
        return normal, None
    for p, lab in labels.items():
        normal[p] = None if lab is None else lab.lower() in NORMAL_LABELS
    n_normal = sum(1 for v in normal.values() if v)
    defects = sorted({lab for p, lab in labels.items() if lab and normal[p] is False})
    named = ", ".join(sorted(found))
    return normal, RuleNote(
        "R-VIS-LAY-03",
        f"Images labelled {named} read as normal ({n_normal:,}); the other labels as defect "
        f"types ({len(defects)}) (R-VIS-LAY-03). {CHECK}",
        "This is the layout of the anomaly-detection sets (MVTec AD, VisA): the good images, and "
        "one folder per kind of defect.",
        n_normal,
    )


def read_layout(rel_paths: Sequence[str], *, ignore: Collection[str] = ()) -> Layout:
    """The labels, splits, normal flags and masks the folder layout gives, by the rules above.
    `rel_paths` are the image files relative to the imported root, with "/" or "\\"."""
    paths = [p.replace("\\", "/") for p in rel_paths]
    rules: list[RuleNote] = []

    # R-VIS-LAY-04: masks first, they are not images of the set.
    masks: dict[str, tuple[tuple[str, str], str]] = {}
    if "R-VIS-LAY-04" not in ignore:
        for path in paths:
            key = _mask_key(_parts(path))
            if key is not None:
                masks[path] = key
    images = [p for p in paths if p not in masks]

    dirs = {p: _parts(p)[:-1] for p in images}
    splits: dict[str, str | None] = {}
    rests: dict[str, tuple[str, ...]] = {}
    use_split = "R-VIS-LAY-02" not in ignore
    for p, d in dirs.items():
        index = _split_index(d) if use_split else None
        splits[p] = SPLIT_NAMES[d[index].lower()] if index is not None else None
        rests[p] = d[index + 1 :] if index is not None else d
    n_split = sum(1 for s in splits.values() if s is not None)
    if n_split:
        counts = Counter(s for s in splits.values() if s)
        written = ", ".join(f"{name} {n:,}" for name, n in sorted(counts.items()))
        rules.append(
            RuleNote(
                "R-VIS-LAY-02",
                f"Splits read from folder names ({written}) (R-VIS-LAY-02). {CHECK}",
                "A folder named train, test or val is the producer's own split: it is kept, "
                "not redrawn.",
                n_split,
            )
        )

    # R-VIS-LAY-01: strip the folders every image shares, the next folder is the label.
    common: tuple[str, ...] = ()
    rest_values = list(rests.values())
    if rest_values:
        shortest = min(len(r) for r in rest_values)
        for i in range(shortest):
            names = {r[i] for r in rest_values}
            if len(names) != 1:
                break
            common += (rest_values[0][i],)
    candidates = {p: (r[len(common)] if len(r) > len(common) else None) for p, r in rests.items()}
    distinct = {c for c in candidates.values() if c is not None}
    under = sum(1 for c in candidates.values() if c is not None)
    labels: dict[str, str | None] = dict.fromkeys(images)
    if (
        "R-VIS-LAY-01" not in ignore
        and len(distinct) >= 2
        and images
        and under / len(images) >= LABEL_SHARE
    ):
        labels = candidates
        rules.append(
            RuleNote(
                "R-VIS-LAY-01",
                f"Labels read from folder names ({len(distinct)} labels) (R-VIS-LAY-01). {CHECK}",
                "One folder per class is the usual layout of an image set (ImageFolder): "
                f"{under:,} of {len(images):,} images are in one of {len(distinct)} sibling "
                "folders.",
                under,
            )
        )
    elif len(distinct) >= 2 and "R-VIS-LAY-01" not in ignore:
        rules.append(
            RuleNote(
                "R-VIS-LAY-01",
                f"Folder names not read as labels: only {under / len(images):.0%} of the images "
                f"are in the {len(distinct)} sibling folders (at least {LABEL_SHARE:.0%} needed).",
                "Folders that hold only some of the images are a filing habit, not classes.",
                0,
            )
        )

    normal, note = normal_flags(labels, ignore=ignore)
    if note is not None:
        rules.append(note)

    # R-VIS-LAY-04: link each mask to its image by (defect type, name), or by name alone.
    linked: dict[str, str] = {}
    orphans: list[str] = []
    if masks:
        by_type_name: dict[tuple[str, str], list[str]] = defaultdict(list)
        by_name: dict[str, list[str]] = defaultdict(list)
        for p in images:
            parts = _parts(p)
            stem = PurePosixPath(parts[-1]).stem
            parent = parts[-2] if len(parts) >= 2 else ""
            by_type_name[(parent, stem)].append(p)
            by_name[stem].append(p)
        for mask_path, ((kind_name, stem), kind) in sorted(masks.items()):
            matches = by_type_name.get((kind_name, stem), []) if kind == "ground_truth" else []
            if not matches:
                matches = by_name.get(stem, [])
            if len(matches) == 1 and matches[0] not in linked:
                linked[matches[0]] = mask_path
            else:
                orphans.append(mask_path)
        if linked:
            rules.append(
                RuleNote(
                    "R-VIS-LAY-04",
                    f"{len(linked):,} pixel masks linked to their images by folder and name "
                    f"(R-VIS-LAY-04). {CHECK}",
                    "ground_truth/<type>/<name>_mask.png (or a masks folder with the same names) "
                    "holds where the defect is: it gives pixel-level scores.",
                    len(linked),
                )
            )
        if orphans:
            rules.append(
                RuleNote(
                    "R-VIS-LAY-04",
                    f"{len(orphans):,} masks without a matching image were left out.",
                    "A mask is only useful next to its image.",
                    len(orphans),
                )
            )

    files = [
        FileLayout(
            rel_path=p,
            label=labels.get(p),
            split=splits.get(p),
            is_normal=normal.get(p),
            mask=linked.get(p),
        )
        for p in images
    ]
    return Layout(files=files, rules=rules, masks=set(masks), orphan_masks=orphans)


def split_of_file(path: PurePosixPath | str) -> str | None:
    """R-VIS-LAY-06 (and R-VIS-HF-03 for one file): the split a Parquet file's name gives, from its
    stem then its folder, cut at `-`, `_`, `.` and spaces (`train-00000-of-00001` -> train)."""
    path = PurePosixPath(str(path).replace("\\", "/"))
    for text in (path.stem, path.parent.name):
        for token in re.split(r"[-_.\s]+", text.lower()):
            if token in SPLIT_NAMES:
                return SPLIT_NAMES[token]
    return None


def parquet_folder_note(files: Sequence[tuple[str, str | None, int]]) -> RuleNote:
    """R-VIS-LAY-06: the sentence for a folder of Parquet files, from (file, split, images)."""
    per_split: Counter[str] = Counter()
    unnamed: list[str] = []
    for name, split, n in files:
        if split is None:
            unnamed.append(name)
        else:
            per_split[split] += n
    total = sum(n for _, _, n in files)
    read = ", ".join(f"{k} {v:,}" for k, v in per_split.items())
    message = f"{len(files)} Parquet files with an image column read"
    message += f", split from each file's name: {read}" if read else ""
    if unnamed:
        shown = ", ".join(unnamed[:3]) + (", ..." if len(unnamed) > 3 else "")
        message += f"; no split in the name of {shown}"
    return RuleNote(
        "R-VIS-LAY-06",
        f"{message} (R-VIS-LAY-06). {CHECK}",
        "Hugging Face stores a dataset as one Parquet file (or several shards) per split, named "
        "train-..., validation-..., test-...: the folder is the dataset and each name is the "
        "producer's own split, kept rather than redrawn.",
        total,
    )


__all__ = [
    "LABEL_SHARE",
    "NORMAL_LABELS",
    "SPLIT_NAMES",
    "FileLayout",
    "Layout",
    "RuleNote",
    "normal_flags",
    "parquet_folder_note",
    "read_layout",
    "split_of_file",
]
