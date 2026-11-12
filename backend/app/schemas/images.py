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
