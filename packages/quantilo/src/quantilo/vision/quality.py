"""Quality rules of an image set (105, § 4.2 of the study): R-VQ-01 to R-VQ-15.

Every rule is written down with its code, the sentence the app shows, its reason and its
thresholds, which can be changed (`QualityThresholds`). Nothing is learnt and nothing is decided
silently: an image is only **excluded from models** by three rules (R-VQ-01 broken file, R-VQ-09
second copy of an exact duplicate, R-VQ-14 class too small to validate); every other rule flags.

    report = assess(images.table)                          # or images.quality after an import
    report.findings                                        # one per rule that fired
    report.apply(images.table)                             # `issues`, `excluded`, `duplicate_group`
    assess(table, thresholds=QualityThresholds(blur_sharpness=50), ignore={"R-VQ-08"})

Relative rules (R-VQ-02 area, R-VQ-04, R-VQ-15) compare an image with the rest of its set, by a
robust z-score: (x - median) / (1.4826 x MAD), the scale of a normal law's standard deviation. When
the MAD is 0 (most images equal), the mean absolute deviation (x 1.2533) takes its place, so one
4000 x 3000 photo among 500 x 500 ones is still seen. They need `min_relative` decoded images.

**The set's style is not a defect.** When R-VQ-05, 06, 07 or 10 would flag more than half of the
decoded images (`common_share`), it describes the set, not a few bad images: a white studio
background (06), inspection photos of one object that all look alike to the hashes (10). The rule
then flags nothing and says so in `notes`; for R-VQ-10 only exact copies are grouped (R-VQ-11 and
12 follow). Found on MVTec AD `bottle`: 100 % "overexposed", 70 % "near duplicates".

Two rules need more than the import and live in `MODEL_RULES` (Advanced, never run at import):
- R-VQ-16, possibly mislabelled: a trained model's out-of-fold probabilities give the image's own
  label less than 0.1 and another class more than 0.8 (`possibly_mislabelled`, run by 112's
  classification and read on the Quality tab). A suggestion only: nothing is excluded.
- R-VQ-17, semantic near-copies: the cosine of two frozen-network vectors (DINOv2) is at least
  0.95 (`semantic_duplicates`); only the groups the hashes of R-VQ-10 did not already find.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal

import numpy as np
import pyarrow as pa

from quantilo.vision.duplicates import (
    SEMANTIC_COSINE,
    DuplicateGroup,
    Leak,
    duplicate_groups,
    exact_copies,
    group_ids,
    label_conflicts,
    leaks,
    semantic_groups,
    semantic_pairs,
)

Effect = Literal["excluded", "flagged", "dataset"]

_MAD_SCALE = 1.4826  # MAD -> standard deviation for a normal law
_MEAN_AD_SCALE = 1.2533  # mean absolute deviation -> standard deviation (sqrt(pi / 2))
KNN_MIN_IMAGES = 50  # R-VQ-15, the descriptor part: a 99th percentile needs a few dozen images
COMMON_RULES = frozenset({"R-VQ-05", "R-VQ-06", "R-VQ-07", "R-VQ-10"})
KNN_REFERENCE = 5_000  # neighbours are searched among at most this many images (sampled, seeded)


@dataclass(frozen=True)
class QualityThresholds:
    """Every threshold of the R-VQ rules, with the study's defaults. Change one with
    `QualityThresholds(blur_sharpness=50)` or `replace(thresholds, ...)`."""

    min_side: int = 32  # R-VQ-02: short side under this, in pixels
    min_area_share: float = 1 / 16  # R-VQ-02: or area under this share of the set's median
    max_aspect: float = 4.0  # R-VQ-03: max(w/h, h/w) over this
    blur_z: float = 3.0  # R-VQ-04: log sharpness this many robust deviations under the median
    blur_sharpness: float = 100.0  # R-VQ-04: and variance of the Laplacian under this
    dark_p95: float = 0.20  # R-VQ-05: 95th percentile of the luma under this
    dark_clipped: float = 0.5  # R-VQ-05: or this share of pixels at most 2/255
    light_p05: float = 0.85  # R-VQ-06: 5th percentile of the luma over this
    light_clipped: float = 0.25  # R-VQ-06: or this share of pixels at least 253/255
    low_contrast: float = 0.04  # R-VQ-07: standard deviation of the luma under this
    low_entropy: float = 3.0  # R-VQ-07: or entropy under this many bits
    colour_majority: float = 0.90  # R-VQ-08: share of the main colour mode
    near_phash: int = 6  # R-VQ-10: pHash distance at most this (of 64 bits)
    near_dhash: int = 10  # R-VQ-10: and dHash distance at most this
    imbalance_ratio: float = 3.0  # R-VQ-13: largest class / smallest class over this
    class_warn: int = 10  # R-VQ-14: a class under this many images is said
    class_exclude: int = 5  # R-VQ-14: under this, it is left out of training
    outlier_z: float = 4.0  # R-VQ-15: robust z over this
    knn_k: int = 10  # R-VQ-15: neighbours in the descriptor space
    knn_quantile: float = 0.99  # R-VQ-15: kNN distance over this quantile of the set (and z)
    min_relative: int = 8  # relative parts of R-VQ-02, 04, 15: at least this many images
    common_share: float = 0.5  # R-VQ-05, 06, 07, 10 on more than this share: the set's style


@dataclass(frozen=True)
class QualityRule:
    """One rule of the catalogue: code, short title, why it exists, what it does, its thresholds."""

    code: str
    title: str
    reason: str
    effect: Effect
    parameters: tuple[str, ...]


RULES: dict[str, QualityRule] = {
    r.code: r
    for r in (
        QualityRule(
            "R-VQ-01",
            "Could not be decoded",
            "A broken file makes a whole training batch fail; it is listed with its error.",
            "excluded",
            (),
        ),
        QualityRule(
            "R-VQ-02",
            "Too small to learn from",
            "Resizing a tiny image up invents pixels the model would learn from.",
            "flagged",
            ("min_side", "min_area_share"),
        ),
        QualityRule(
            "R-VQ-03",
            "Odd aspect ratio",
            "A square crop would lose most of it: square resizing pads it instead.",
            "flagged",
            ("max_aspect",),
        ),
        QualityRule(
            "R-VQ-04",
            "Blurry",
            "Two conditions: far below this set's sharpness (relative) and under the usual "
            "threshold of the variance of the Laplacian (absolute); a smooth subject alone is "
            "not enough.",
            "flagged",
            ("blur_z", "blur_sharpness"),
        ),
        QualityRule(
            "R-VQ-05",
            "Too dark",
            "Underexposed: the details the model needs are in the noise.",
            "flagged",
            ("dark_p95", "dark_clipped"),
        ),
        QualityRule(
            "R-VQ-06",
            "Overexposed",
            "Burnt highlights: the pixels hold no information any more.",
            "flagged",
            ("light_p05", "light_clipped"),
        ),
        QualityRule(
            "R-VQ-07",
            "Low information",
            "A nearly uniform image: a black screen, a cardboard, a lens cap.",
            "flagged",
            ("low_contrast", "low_entropy"),
        ),
        QualityRule(
            "R-VQ-08",
            "Mixed colour modes",
            "A few greyscale images among colour ones (or the reverse): the model learns the "
            "colour instead of the object.",
            "flagged",
            ("colour_majority",),
        ),
        QualityRule(
            "R-VQ-09",
            "Exact duplicate",
            "The same pixels twice inflate the set and bias the evaluation: one copy is kept.",
            "excluded",
            (),
        ),
        QualityRule(
            "R-VQ-10",
            "Near duplicate",
            "Crops, recompressions and bursts are the same picture: reviewed side by side.",
            "flagged",
            ("near_phash", "near_dhash"),
        ),
        QualityRule(
            "R-VQ-11",
            "Copy across train and test",
            "A test image with a copy in train is a leak: the test score will be optimistic. Our "
            "own splits keep each group of copies on one side (grouped folds).",
            "flagged",
            ("near_phash", "near_dhash"),
        ),
        QualityRule(
            "R-VQ-12",
            "Same image, two labels",
            "Copies with different labels contradict each other: one of the labels is wrong.",
            "flagged",
            (),
        ),
        QualityRule(
            "R-VQ-13",
            "Imbalanced classes",
            "Accuracy hides the rare class: the main metric becomes balanced accuracy, with "
            "class weights.",
            "dataset",
            ("imbalance_ratio",),
        ),
        QualityRule(
            "R-VQ-14",
            "Class too small",
            "A stratified 5-fold validation needs at least 5 images per class.",
            "excluded",
            ("class_warn", "class_exclude"),
        ),
        QualityRule(
            "R-VQ-15",
            "Unlike the others",
            "An image from another set, a screenshot: far from the rest in size, brightness, "
            "colour or content. The kNN part needs both the quantile and the z, or it would flag "
            "1 % of every set.",
            "flagged",
            ("outlier_z", "knn_k", "knn_quantile"),
        ),
    )
}


MISLABEL_GIVEN = 0.1  # R-VQ-16: the image's own label under this probability
MISLABEL_OTHER = 0.8  # R-VQ-16: and another class over this one

# Rules that read a trained model or a frozen network: Advanced, never run at the import.
MODEL_RULES: dict[str, QualityRule] = {
    r.code: r
    for r in (
        QualityRule(
            "R-VQ-16",
            "Possibly mislabelled",
            "Confident learning: a model that never saw the image (out of fold) gives its label "
            "less than 10 % and another class more than 80 %. Often a wrong label, sometimes a "
            "hard image: a suggestion to check, nothing is excluded.",
            "flagged",
            ("given", "other"),
        ),
        QualityRule(
            "R-VQ-17",
            "Semantic near-copies",
            "Two photos of the same thing from another angle or after an edit escape the hashes "
            "of R-VQ-10; a frozen network (DINOv2) sees them as almost the same vector (cosine "
            "at least 0.95). Reviewed side by side; a split should keep them on one side.",
            "flagged",
            ("cosine",),
        ),
    )
}


@dataclass(frozen=True)
class QualityFinding:
    """A rule that fired: the sentence shown, its reason, the rows concerned, the thresholds used.
    `details` carries what a rule adds (groups, classes, the recommended metric...)."""

    code: str
    title: str
    message: str
    reason: str
    effect: Effect
    count: int
    rows: tuple[int, ...]
    parameters: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Every finding, and per row its issue codes and whether models leave it out."""

    findings: list[QualityFinding]
    issues: list[list[str]]
    excluded: list[bool]
    groups: list[DuplicateGroup]
    group_of: np.ndarray  # one group per row, for a grouped split (GroupKFold)
    leaks: list[Leak]
    thresholds: QualityThresholds
    notes: list[str] = field(default_factory=list)  # rules that could not run, and why
    recommended_metric: str = "accuracy"
    class_weight: str | None = None

    def finding(self, code: str) -> QualityFinding | None:
        return next((f for f in self.findings if f.code == code), None)

    def __getitem__(self, code: str) -> QualityFinding:
        found = self.finding(code)
        if found is None:
            raise KeyError(f"{code} did not fire.")
        return found

    @property
    def codes(self) -> list[str]:
        return [f.code for f in self.findings]

    @property
    def sentence(self) -> str:
        flagged = sum(1 for row in self.issues if row)
        if not self.findings:
            return "No quality issue found by the R-VQ rules."
        excluded = sum(self.excluded)
        text = f"{flagged:,} images with an issue ({', '.join(self.codes)})"
        if excluded:
            text += f"; {excluded:,} left out of models"
        return text + "."

    def duplicate_column(self) -> list[int | None]:
        """The duplicate group of each row, None when it has no copy."""
        column: list[int | None] = [None] * len(self.issues)
        for g in self.groups:
            for r in g.rows:
                column[r] = g.group
        return column

    def apply(self, table: pa.Table) -> pa.Table:
        """The table with `issues`, `excluded` and `duplicate_group` set from this report."""
        updates = {
            "issues": pa.array(self.issues, type=pa.list_(pa.string())),
            "excluded": pa.array(self.excluded, type=pa.bool_()),
            "duplicate_group": pa.array(self.duplicate_column(), type=pa.int64()),
        }
        for name, values in updates.items():
            if name in table.column_names:
                table = table.set_column(table.column_names.index(name), name, values)
            else:
                table = table.append_column(name, values)
        return table

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready summary (the backend stores it with the dataset)."""
        return {
            "sentence": self.sentence,
            "findings": [asdict(f) | {"rows": list(f.rows)} for f in self.findings],
            "groups": [
                {"group": g.group, "kind": g.kind, "rows": list(g.rows), "keep": g.keep}
                for g in self.groups
            ],
            "leaks": [
                {"group": lk.group, "train": list(lk.train_rows), "eval": list(lk.eval_rows)}
                for lk in self.leaks
            ],
            "thresholds": asdict(self.thresholds),
            "notes": list(self.notes),
            "recommended_metric": self.recommended_metric,
            "class_weight": self.class_weight,
            "excluded": int(sum(self.excluded)),
            "flagged": int(sum(1 for row in self.issues if row)),
        }


# --- helpers -------------------------------------------------------------------------------------


def _column(data: pa.Table | Mapping[str, Sequence[Any]], name: str, n: int) -> list[Any]:
    if isinstance(data, Mapping):
        values = data.get(name)
        return list(values) if values is not None else [None] * n
    return data[name].to_pylist() if name in data.column_names else [None] * n


def _length(data: pa.Table | Mapping[str, Sequence[Any]]) -> int:
    if isinstance(data, Mapping):
        return max((len(v) for v in data.values()), default=0)
    return data.num_rows


def robust_z(values: np.ndarray) -> np.ndarray:
    """(x - median) / robust scale, NaN kept; 0 everywhere when the values do not vary at all."""
    x = np.asarray(values, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if len(finite) == 0:
        return np.full_like(x, np.nan)
    median = float(np.median(finite))
    scale = _MAD_SCALE * float(np.median(np.abs(finite - median)))
    if scale <= 0:
        scale = _MEAN_AD_SCALE * float(np.mean(np.abs(finite - median)))
    if scale <= 0:
        return np.where(np.isfinite(x), 0.0, np.nan)
    return (x - median) / scale


def _num(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n:,} {one if n == 1 else (many or one + 's')}"


def knn_distances(
    vectors: np.ndarray, k: int, *, reference: int = KNN_REFERENCE, seed: int = 0
) -> np.ndarray:
    """Mean Euclidean distance of each row to its k nearest other rows (standardised columns),
    searched among at most `reference` rows drawn with `seed`. Brute force by chunks: numpy only."""
    x = np.asarray(vectors, dtype=np.float64)
    std = x.std(axis=0)
    x = (x - x.mean(axis=0)) / np.where(std > 0, std, 1.0)
    n = len(x)
    ref_rows = np.arange(n)
    if n > reference:
        ref_rows = np.sort(np.random.default_rng(seed).choice(n, reference, replace=False))
    ref = x[ref_rows]
    ref_sq = (ref**2).sum(axis=1)
    k = max(1, min(k, len(ref) - 1))
    out = np.empty(n)
    for start in range(0, n, 1_024):
        block = x[start : start + 1_024]
        d2 = (block**2).sum(axis=1)[:, None] + ref_sq[None, :] - 2.0 * block @ ref.T
        np.maximum(d2, 0.0, out=d2)
        rows = np.arange(start, start + len(block))
        # a row is not its own neighbour
        mine = np.searchsorted(ref_rows, rows)
        hit = (mine < len(ref_rows)) & (ref_rows[np.minimum(mine, len(ref_rows) - 1)] == rows)
        d2[np.flatnonzero(hit), mine[hit]] = np.inf
        nearest = np.partition(d2, k - 1, axis=1)[:, :k]
        out[start : start + len(block)] = np.sqrt(nearest).mean(axis=1)
    return out


# --- the rules -----------------------------------------------------------------------------------


def assess(
    data: pa.Table | Mapping[str, Sequence[Any]],
    *,
    thresholds: QualityThresholds | None = None,
    ignore: Collection[str] = (),
    descriptors: np.ndarray | None = None,
) -> QualityReport:
    """Run R-VQ-01 to 15 on an image table (the columns of § 3.4: `decode_ok`, `width`, `height`,
    measures, `phash`, `dhash`, `pixel_sha`, `label`, `split`). `descriptors` (one row per image,
    NaN for images not decoded) adds the content part of R-VQ-15. `ignore` names rules to skip.
    `excluded` is recomputed from the rules (01, 09, 14) only."""
    th = thresholds or QualityThresholds()
    skip = frozenset(ignore)
    n = _length(data)
    col = {name: _column(data, name, n) for name in (
        "decode_ok", "decode_error", "rel_path", "width", "height", "sharpness", "luma_p05",
        "luma_p95", "clipped_dark", "clipped_light", "contrast", "entropy", "is_grayscale",
        "brightness", "colourfulness", "pixel_sha", "phash", "dhash", "label", "split",
    )}  # fmt: skip
    ok = np.asarray([bool(v) for v in col["decode_ok"]], dtype=bool)
    issues: list[set[str]] = [set() for _ in range(n)]
    excluded = np.zeros(n, dtype=bool)
    findings: list[QualityFinding] = []
    notes: list[str] = []

    def params(code: str) -> dict[str, float]:
        return {p: float(getattr(th, p)) for p in RULES[code].parameters}

    def add(
        code: str,
        rows: Sequence[int] | np.ndarray,
        message: str,
        count: int | None = None,
        exclude: Sequence[int] | None = None,
        **details: Any,
    ) -> None:
        """Record a finding; `exclude` = the rows left out of models (all of them by default
        for an "excluded" rule)."""
        rows = tuple(int(r) for r in rows)
        rule = RULES[code]
        for r in rows:
            issues[r].add(code)
        if rule.effect == "excluded":
            excluded[list(rows if exclude is None else exclude)] = True
        findings.append(
            QualityFinding(
                code,
                rule.title,
                f"{message} ({code}).",
                rule.reason,
                rule.effect,
                len(rows) if count is None else count,
                rows,
                params(code),
                details,
            )
        )

    def common(code: str, count: int) -> bool:
        """True (and said) when a style rule fires on most of the set (see the module)."""
        if code not in COMMON_RULES or n_ok < th.min_relative or count <= th.common_share * n_ok:
            return False
        notes.append(
            f"{code} ({RULES[code].title.lower()}) would flag {count / n_ok:.0%} of the images: "
            "read as this set's style, not flagged. Check a few images."
        )
        return True

    def flag(code: str, mask: np.ndarray, message: str, **details: Any) -> None:
        rows = np.flatnonzero(mask & ok)
        if code not in skip and len(rows) and not common(code, len(rows)):
            add(code, rows, message.format(n=_plural(len(rows), "image")), **details)

    # R-VQ-01: not decoded
    if "R-VQ-01" not in skip and (~ok).any():
        broken = np.flatnonzero(~ok)
        errors = Counter(str(col["decode_error"][r] or "unknown error") for r in broken)
        add(
            "R-VQ-01",
            broken,
            f"{_plural(len(broken), 'file')} could not be decoded: left out of models",
            errors=dict(errors.most_common(5)),
        )

    width, height = _num(col["width"]), _num(col["height"])
    n_ok = int(ok.sum())
    relative = n_ok >= th.min_relative
    if not relative and n_ok:
        notes.append(
            f"Relative parts of R-VQ-02, 04 and 15 need {th.min_relative} decoded images "
            f"({n_ok} here): only the absolute thresholds apply."
        )

    # R-VQ-02: too small
    with np.errstate(invalid="ignore"):
        small = np.fmin(width, height) < th.min_side
        if relative:
            area = width * height
            median_area = float(np.nanmedian(area[ok]))
            small |= area < th.min_area_share * median_area
    flag(
        "R-VQ-02",
        small,
        f"{{n}} too small to learn from (short side under {th.min_side} px, or under "
        f"1/{round(1 / th.min_area_share)} of the median area)",
    )

    # R-VQ-03: odd aspect ratio
    with np.errstate(invalid="ignore", divide="ignore"):
        aspect = np.fmax(width / height, height / width)
    flag(
        "R-VQ-03",
        aspect > th.max_aspect,
        f"{{n}} with an odd aspect ratio (over {th.max_aspect:g}:1): square resizing would "
        "distort them, so they are padded",
    )

    # R-VQ-04: blurry (relative and absolute)
    sharpness = _num(col["sharpness"])
    if relative:
        z = np.full(n, np.nan)
        z[ok] = robust_z(np.log1p(np.clip(sharpness[ok], 0, None)))
        with np.errstate(invalid="ignore"):
            blurry = (z < -th.blur_z) & (sharpness < th.blur_sharpness)
        median_sharp = float(np.nanmedian(sharpness[ok]))
        flag(
            "R-VQ-04",
            blurry,
            f"{{n}} blurry: sharpness under {th.blur_sharpness:g} and far below the set's median "
            f"({median_sharp:,.0f})",
            median_sharpness=median_sharp,
        )

    # R-VQ-05 / 06 / 07: exposure and information
    with np.errstate(invalid="ignore"):
        dark = (_num(col["luma_p95"]) < th.dark_p95) | (_num(col["clipped_dark"]) > th.dark_clipped)
        light = (_num(col["luma_p05"]) > th.light_p05) | (
            _num(col["clipped_light"]) > th.light_clipped
        )
        flat = (_num(col["contrast"]) < th.low_contrast) | (_num(col["entropy"]) < th.low_entropy)
    flag("R-VQ-05", dark, "{n} too dark")
    flag("R-VQ-06", light, "{n} overexposed")
    flag("R-VQ-07", flat, "{n} with low information (nearly uniform)")

    # R-VQ-08: mixed colour modes
    grey = np.asarray([bool(v) for v in col["is_grayscale"]], dtype=bool)
    if n_ok and "R-VQ-08" not in skip:
        share_grey = float((grey & ok).sum()) / n_ok
        if 1 - share_grey >= th.colour_majority and share_grey > 0:
            flag(
                "R-VQ-08",
                grey,
                f"{{n}} in greyscale in a set that is {1 - share_grey:.0%} colour",
            )
        elif share_grey >= th.colour_majority and share_grey < 1:
            flag("R-VQ-08", ~grey, f"{{n}} in colour in a set that is {share_grey:.0%} greyscale")

    # R-VQ-09 / 10: duplicates; R-VQ-11 leaks; R-VQ-12 two labels
    sha = [s if good else None for s, good in zip(col["pixel_sha"], ok, strict=True)]
    phash = [p if good else None for p, good in zip(col["phash"], ok, strict=True)]
    dhash = [d if good else None for d, good in zip(col["dhash"], ok, strict=True)]
    want_near = any(c not in skip for c in ("R-VQ-10", "R-VQ-11", "R-VQ-12"))
    groups = (
        duplicate_groups(
            sha, phash, dhash, max_phash=th.near_phash, max_dhash=th.near_dhash, near=want_near
        )
        if "R-VQ-09" not in skip or want_near
        else []
    )
    if "R-VQ-09" not in skip:
        copies = exact_copies(sha)
        if copies:
            extra = [r for rows in copies for r in rows[1:]]
            rows = sorted(r for rows in copies for r in rows)
            add(
                "R-VQ-09",
                rows,
                f"{_plural(len(extra), 'exact copy', 'exact copies')} in "
                f"{_plural(len(copies), 'group')}: the first image of each group is kept",
                count=len(extra),
                exclude=extra,
                kept=[rows_[0] for rows_ in copies],
                dropped=extra,
            )
    near_groups = [g for g in groups if g.kind == "near"]
    if near_groups and common("R-VQ-10", sum(len(g.rows) for g in near_groups)):
        groups = duplicate_groups(sha, phash, dhash, near=False)
        near_groups = []
    if "R-VQ-10" not in skip and near_groups:
        rows = sorted(r for g in near_groups for r in g.rows)
        add(
            "R-VQ-10",
            rows,
            f"{_plural(len(rows), 'near duplicate')} in {_plural(len(near_groups), 'group')}, to "
            "review side by side",
            groups=[g.group for g in near_groups],
        )
    split = list(col["split"])
    found_leaks = leaks(groups, split) if "R-VQ-11" not in skip else []
    if found_leaks:
        eval_rows = sorted(r for lk in found_leaks for r in lk.eval_rows)
        add(
            "R-VQ-11",
            eval_rows,
            f"{_plural(len(eval_rows), 'test image')} {'has' if len(eval_rows) == 1 else 'have'} "
            "a near-copy in train: the test score will be optimistic",
            groups=[lk.group for lk in found_leaks],
        )
    labels = list(col["label"])
    if "R-VQ-12" not in skip and (conflicting := label_conflicts(groups, labels)):
        rows = sorted(r for g in conflicting for r in g.rows)
        add(
            "R-VQ-12",
            rows,
            f"{_plural(len(conflicting), 'group')} of copies with different labels: same image, "
            "two labels",
            count=len(conflicting),
            groups=[g.group for g in conflicting],
        )

    # R-VQ-14 then R-VQ-13: classes
    counts = Counter(
        str(labels[r]) for r in range(n) if labels[r] is not None and ok[r] and not excluded[r]
    )
    recommended, class_weight = "accuracy", None
    if counts and "R-VQ-14" not in skip:
        rare = {c: k for c, k in counts.items() if k < th.class_warn}
        if rare:
            dropped = {c for c, k in rare.items() if k < th.class_exclude}
            names = ", ".join(f"{c} {k}" for c, k in sorted(rare.items(), key=lambda kv: kv[1]))
            rows_rare = [
                r for r in range(n) if labels[r] is not None and str(labels[r]) in rare and ok[r]
            ]
            message = (
                f"{_plural(len(rare), 'class', 'classes')} under {th.class_warn} images ({names})"
            )
            if dropped:
                message += (
                    f"; {_plural(len(dropped), 'class', 'classes')} under {th.class_exclude} "
                    "left out of training"
                )
            add(
                "R-VQ-14",
                rows_rare,
                message,
                count=len(rare),
                exclude=[r for r in rows_rare if str(labels[r]) in dropped],
                classes=rare,
                dropped=sorted(dropped),
            )
            for c in dropped:
                counts.pop(c)
    if len(counts) >= 2 and "R-VQ-13" not in skip:
        (major, high), (minor, low) = counts.most_common()[0], counts.most_common()[-1]
        ratio = high / max(1, low)
        if ratio > th.imbalance_ratio:
            recommended, class_weight = "balanced_accuracy", "balanced"
            add(
                "R-VQ-13",
                [],
                f"Classes are imbalanced ({major} {high:,}, {minor} {low:,}: ratio {ratio:.1f}): "
                "the main metric is balanced accuracy, with class weights",
                count=len(counts),
                ratio=ratio,
                counts=dict(counts),
                metric="balanced_accuracy",
                class_weight="balanced",
            )

    # R-VQ-15: unlike the others
    if "R-VQ-15" not in skip and relative:
        unusual = np.zeros(n, dtype=bool)
        why: dict[str, int] = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            features = {
                "width": np.log(width),
                "height": np.log(height),
                "brightness": _num(col["brightness"]),
                "colour": _num(col["colourfulness"]),
            }
        for name, values in features.items():
            z = np.full(n, np.nan)
            z[ok] = robust_z(values[ok])
            with np.errstate(invalid="ignore"):
                hit = np.abs(z) > th.outlier_z
            if hit.any():
                why[name] = int(hit.sum())
                unusual |= hit
        if descriptors is not None:
            vectors = np.asarray(descriptors, dtype=np.float64)
            usable = ok & np.isfinite(vectors).all(axis=1)
            if usable.sum() >= max(KNN_MIN_IMAGES, th.knn_k + 1):
                rows = np.flatnonzero(usable)
                distance = knn_distances(vectors[rows], th.knn_k)
                cut = float(np.quantile(distance, th.knn_quantile))
                z = robust_z(np.log(distance + 1e-12))
                hit_rows = rows[(distance > cut) & (z > th.outlier_z)]
                if len(hit_rows):
                    why["content"] = len(hit_rows)
                    unusual[hit_rows] = True
            else:
                notes.append(f"The content part of R-VQ-15 needs {KNN_MIN_IMAGES} decoded images.")
        if why:
            reasons = ", ".join(f"{k} {v:,}" for k, v in why.items())
            flag("R-VQ-15", unusual, f"{{n}} unlike the others ({reasons})", by=why)

    order = {code: i for i, code in enumerate(RULES)}
    findings.sort(key=lambda f: order[f.code])
    return QualityReport(
        findings=findings,
        issues=[sorted(s, key=lambda c: order.get(c, 99)) for s in issues],
        excluded=[bool(v) for v in excluded],
        groups=groups,
        group_of=group_ids(n, groups),
        leaks=found_leaks,
        thresholds=th,
        notes=notes,
        recommended_metric=recommended,
        class_weight=class_weight,
    )


def possibly_mislabelled(
    labels: Sequence[Any],
    proba: np.ndarray,
    classes: Sequence[str],
    rows: Sequence[int],
    *,
    given: float = MISLABEL_GIVEN,
    other: float = MISLABEL_OTHER,
    model: str = "",
    calibrated: bool = False,
    scope: str = "",
) -> QualityFinding:
    """R-VQ-16 on out-of-fold probabilities (`proba[i]` for table row `rows[i]`, columns in the
    order of `classes`). The rows are sorted by the other class's probability (the surest first);
    `details` carries, per row, the label given, the class suggested and both probabilities."""
    p = np.asarray(proba, dtype=float)
    index = {str(c): k for k, c in enumerate(classes)}
    found = []
    for i, label in enumerate(labels):
        k = index.get(str(label))
        if k is None or not np.isfinite(p[i]).all():
            continue
        rest = p[i].copy()
        rest[k] = -1.0
        best = int(rest.argmax())
        if p[i, k] < given and p[i, best] > other:
            found.append((float(p[i, best]), int(rows[i]), str(label), str(classes[best]), p[i, k]))
    found.sort(key=lambda f: (-f[0], f[1]))
    rule = MODEL_RULES["R-VQ-16"]
    where = f" by {model}" if model else ""
    if found:
        message = (
            f"{_plural(len(found), 'image')} whose label gets less than {given:.0%} from a model "
            f"that never saw {'it' if len(found) == 1 else 'them'}{where}, while another class "
            f"gets more than {other:.0%}: check {'its' if len(found) == 1 else 'their'} label."
        )
    else:
        message = f"No image looks mislabelled{where} (R-VQ-16)."
    notes = []
    if calibrated:
        notes.append("Probabilities read after the temperature scaling of R-VM-06.")
    if scope:
        notes.append(scope)
    return QualityFinding(
        code="R-VQ-16",
        title=rule.title,
        message=message,
        reason=rule.reason,
        effect="flagged",
        count=len(found),
        rows=tuple(f[1] for f in found),
        parameters={"given": given, "other": other},
        details={
            "model": model,
            "checked": len(labels),
            "notes": notes,
            "images": [
                {
                    "row": r,
                    "label": lab,
                    "suggested": sug,
                    "suggested_proba": round(q, 4),
                    "label_proba": round(float(g), 4),
                }
                for q, r, lab, sug, g in found
            ],
        },
    )


def semantic_duplicates(
    vectors: np.ndarray,
    known: Sequence[DuplicateGroup] = (),
    labels: Sequence[Any] | None = None,
    split: Sequence[Any] | None = None,
    *,
    cosine: float = SEMANTIC_COSINE,
    network: str = "",
    common_share: float = QualityThresholds.common_share,
    min_images: int = QualityThresholds.min_relative,
) -> QualityFinding:
    """R-VQ-17: groups of images whose frozen-network vectors have a cosine of at least `cosine`,
    left out when R-VQ-09 / 10 already grouped them (`known`). `details.groups` lists each group
    with its smallest cosine, whether its labels differ and whether it spans train and test.

    The set's style is not a defect (as for R-VQ-10): when the groups would hold more than
    `common_share` of the compared images (MVTec `bottle`: one object shot the same way, median
    cosine 0.97), nothing is flagged and `details.style` says so."""
    x = np.asarray(vectors, dtype=np.float32)
    n = len(x)
    pairs = semantic_pairs(x, cosine)
    groups = semantic_groups(n, pairs, known)
    computed = int(np.isfinite(x).all(axis=1).sum()) if n else 0
    grouped = sum(len(g.rows) for g in groups)
    style = None
    if computed >= min_images and grouped > common_share * computed:
        style = (
            f"{network or 'The network'} sees {grouped:,} of {computed:,} images as near-copies "
            f"(cosine at least {cosine:g}): they show the same subject the same way. R-VQ-17 "
            "flags nothing: read as this set's style."
        )
        groups = []
    lowest: dict[int, float] = {}
    group_of = {r: g.group for g in groups for r in g.rows}
    for i, j, c in pairs:
        g = group_of.get(i)
        if g is not None and group_of.get(j) == g:
            lowest[g] = min(lowest.get(g, 1.0), c)
    described = []
    for g in groups:
        names = {str(labels[r]) for r in g.rows if labels is not None and labels[r] is not None}
        sides = {str(split[r]) for r in g.rows if split is not None and split[r] is not None}
        described.append(
            {
                "group": g.group,
                "kind": "semantic",
                "rows": list(g.rows),
                "keep": None,
                "cosine": round(lowest.get(g.group, cosine), 4),
                "two_labels": len(names) > 1,
                "leak": "train" in sides and bool(sides & {"test", "val"}),
            }
        )
    rows = tuple(r for g in groups for r in g.rows)
    rule = MODEL_RULES["R-VQ-17"]
    by = f" by {network}" if network else ""
    if style:
        message = style
    elif groups:
        message = (
            f"{_plural(len(groups), 'group')} of near-copies seen{by} that the hashes of "
            f"R-VQ-10 missed ({_plural(len(rows), 'image')}, cosine at least {cosine:g})."
        )
    else:
        message = f"No semantic near-copy beyond R-VQ-10{by} (cosine at least {cosine:g})."
    return QualityFinding(
        code="R-VQ-17",
        title=rule.title,
        message=message,
        reason=rule.reason,
        effect="flagged",
        count=len(rows),
        rows=rows,
        parameters={"cosine": cosine},
        details={
            "groups": described,
            "pairs": len(pairs),
            "compared": computed,
            "leaks": sum(1 for d in described if d["leak"]),
            "conflicts": sum(1 for d in described if d["two_labels"]),
            "style": style,
        },
    )


def thresholds_from(values: Mapping[str, Any] | None) -> QualityThresholds:
    """Thresholds from a mapping (the API's JSON); unknown names are refused by name."""
    if not values:
        return QualityThresholds()
    known = {f.name: f.type for f in fields(QualityThresholds)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"Unknown quality thresholds: {', '.join(unknown)}.")
    base = QualityThresholds()
    cast = {
        k: (int(v) if isinstance(getattr(base, k), int) else float(v)) for k, v in values.items()
    }
    for k, v in cast.items():
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"Quality threshold {k} must be a positive number.")
    return replace(base, **cast)


__all__ = [
    "MISLABEL_GIVEN",
    "MISLABEL_OTHER",
    "MODEL_RULES",
    "RULES",
    "QualityFinding",
    "QualityReport",
    "QualityRule",
    "QualityThresholds",
    "assess",
    "knn_distances",
    "possibly_mislabelled",
    "robust_z",
    "semantic_duplicates",
    "thresholds_from",
]
