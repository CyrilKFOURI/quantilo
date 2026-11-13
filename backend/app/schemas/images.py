"""Image sets (106): the import preview, the import request and the stored report."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ImageSetOptions(BaseModel):
    """What the import reads: a labels file, its label column, the rules undone by code
    (`R-VIS-LAY-01` = ignore folder labels, `R-VQ-08`...), quality thresholds (105)."""

    labels: str | None = None
    label_column: str | None = None
    ignore_rules: list[str] = Field(default_factory=list, max_length=64)
    thresholds: dict[str, float] = Field(default_factory=dict)


class ImagePreviewRequest(BaseModel):
    path: str
    options: ImageSetOptions = Field(default_factory=ImageSetOptions)


class ImageImportRequest(BaseModel):
    path: str
    name: str | None = None
    options: ImageSetOptions = Field(default_factory=ImageSetOptions)
    # 087, R-PRJ-05: put the new dataset in this project
    project_id: str | None = None


class RuleNoteOut(BaseModel):
    """A reading rule that fired: the sentence shown, why it exists, how many files."""

    code: str
    message: str
    reason: str
    count: int | None = None


class SkippedOut(BaseModel):
    rel_path: str
    code: str
    reason: str


class ImagePreviewOut(BaseModel):
    source_kind: Literal["image", "folder", "zip", "parquet"]
    suggested_name: str
    sentence: str
    n_images: int
    bytes: int
    formats: dict[str, int]
    labels: dict[str, int]
    splits: dict[str, int]
    normal: int | None
    masks: int
    rules: list[RuleNoteOut]
    skipped: list[SkippedOut]
    skipped_count: int
    conflicts: int
    sample: list[str]
    label_file: str | None


class QualityFindingOut(BaseModel):
    """A quality rule that fired (R-VQ, 105), with its thresholds and the rows concerned."""

    code: str
    title: str
    message: str
    reason: str
    effect: Literal["excluded", "flagged", "dataset"]
    count: int
    rows: list[int]
    parameters: dict[str, float]
    details: dict[str, Any] = Field(default_factory=dict)


class QualityGroupOut(BaseModel):
    group: int
    kind: Literal["exact", "near"]
    rows: list[int]
    keep: int | None


class QualityLeakOut(BaseModel):
    group: int
    train: list[int]
    eval: list[int]


class QualityOut(BaseModel):
    sentence: str
    findings: list[QualityFindingOut]
    groups: list[QualityGroupOut]
    leaks: list[QualityLeakOut]
    thresholds: dict[str, float]
    notes: list[str]
    recommended_metric: str
    class_weight: str | None
    excluded: int
    flagged: int


class LabelConflictOut(BaseModel):
    rel_path: str
    folder_label: str
    file_label: str


class ImageReportOut(BaseModel):
    sentence: str
    source_kind: Literal["image", "folder", "zip", "parquet"]
    n_images: int
    n_decoded: int
    labels: dict[str, int]
    splits: dict[str, int]
    rules: list[RuleNoteOut]
    skipped: list[SkippedOut]
    skipped_count: int
    conflicts: list[LabelConflictOut]
    conflicts_count: int
    quality: QualityOut | None


class QualityRuleOut(BaseModel):
    code: str
    title: str
    reason: str
    effect: Literal["excluded", "flagged", "dataset"]
    parameters: dict[str, float]


class ImageIndexOut(BaseModel):
    """107: the gallery's index, one list per column in the table's row order (the quality
    report's `rows` are positions in it). No disk path, no raw hash, no coordinate (R-107-04)."""

    count: int
    image_id: list[str]
    rel_path: list[str]
    file_name: list[str | None]
    label: list[str | None]
    split: list[str | None]
    format: list[str | None]
    color_mode: list[str | None]
    camera: list[str | None]
    dominant_colour: list[str | None]
    decode_error: list[str | None]
    decision: list[Literal["exclude", "keep"] | None]
    label_original: list[str | None]  # c3, R-VED-02: the import's label, once a label was edited
    width: list[int | None]
    height: list[int | None]
    file_bytes: list[int | None]
    bit_depth: list[int | None]
    frames: list[int | None]
    duplicate_group: list[int | None]
    megapixels: list[float | None]
    aspect_ratio: list[float | None]
    brightness: list[float | None]
    luma_p05: list[float | None]
    luma_p95: list[float | None]
    contrast: list[float | None]
    entropy: list[float | None]
    saturation: list[float | None]
    colourfulness: list[float | None]
    sharpness: list[float | None]
    clipped_dark: list[float | None]
    clipped_light: list[float | None]
    is_normal: list[bool | None]
    is_grayscale: list[bool | None]
    has_location: list[bool | None]
    decode_ok: list[bool | None]
    excluded: list[bool | None]
    rule_excluded: list[bool | None]
    pinned: list[bool | None]  # c3, R-VED-03
    has_mask: list[bool]
    issues: list[list[str]]
    date_taken: list[str | None]


class ImageDecisionsIn(BaseModel):
    """108 and c3: edits to merge into the draft, by the image's `rel_path` (exact copies share
    their `image_id`). Decisions: None leaves an image to its rules again. Labels (R-VED-01): None
    gives it back the import's label. Pins (R-VED-03): False or None unpins. Thresholds
    (R-VQT-02): the whole set of thresholds the rules will use, replaced."""

    decisions: dict[str, Literal["exclude", "keep"] | None] = Field(
        default_factory=dict, max_length=100_000
    )
    labels: dict[str, str | None] = Field(default_factory=dict, max_length=100_000)
    pins: dict[str, bool | None] = Field(default_factory=dict, max_length=100_000)
    thresholds: dict[str, float] | None = None


class PendingOut(BaseModel):
    decisions: int
    labels: int
    pins: int
    thresholds: int


class ImageDecisionsOut(BaseModel):
    draft: dict[str, Literal["exclude", "keep"]]
    applied: dict[str, Literal["exclude", "keep"]]
    labels_draft: dict[str, str] = Field(default_factory=dict)
    labels_applied: dict[str, str] = Field(default_factory=dict)
    pins_draft: dict[str, bool] = Field(default_factory=dict)
    pins_applied: dict[str, bool] = Field(default_factory=dict)
    thresholds_draft: dict[str, float] = Field(default_factory=dict)
    thresholds_applied: dict[str, float] = Field(default_factory=dict)
    pending: int
    pending_by: PendingOut
    version: int | None


class ImageDecisionsApplied(BaseModel):
    version: int | None
    excluded: int
    decided: int
    relabelled: int = 0
    pinned: int = 0
    reassessed: bool = False  # the R-VQ rules ran again (labels or thresholds changed)


class QualityPreviewIn(BaseModel):
    thresholds: dict[str, float] = Field(default_factory=dict)


class QualityPreviewOut(BaseModel):
    """R-VQT-01: what the R-VQ rules would say with these thresholds, nothing written; `counts`
    (images by rule) next to `current_counts` (what the saved report says)."""

    quality: QualityOut
    counts: dict[str, int]
    current_counts: dict[str, int]
    current_flagged: int
    current_excluded: int
