"""The Map (109) and Prepare (110) tabs of an image set."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ImageMapMethodOut(BaseModel):
    """A way of describing images: classic descriptors, or a frozen network (114)."""

    key: str
    title: str
    detail: str
    level: Literal["first", "advanced"]
    dims: int
    available: bool  # can be chosen here (the extra and the weights are there)
    downloaded: bool
    size: str | None  # of the weights, shown before downloading (R-VDP-02)
    licence: str | None
    note: str | None
    reason: str | None  # why it is greyed


class ImageWeightsDownloadOut(BaseModel):
    state: Literal["running", "failed", "done"]
    received: int
    total: int
    error: str | None = None


class ImageMapProjectionOut(BaseModel):
    """c3, R-VMAP-12: a projection of Advanced > More (PaCMAP, TriMAP) and whether it runs here."""

    key: Literal["pacmap", "trimap"]
    title: str
    licence: str
    available: bool
    reason: str | None  # R-XAI-10, when it is not
    install: str | None


class ImageMapMethodsOut(BaseModel):
    installed: bool  # the optional extra `vision`
    reason: str | None
    install: str
    methods: list[ImageMapMethodOut]
    downloads: dict[str, ImageWeightsDownloadOut]
    projections: list[ImageMapProjectionOut] = Field(default_factory=list)


class ImageMapBuildIn(BaseModel):
    method: str = "classic"
    confirm_long: bool = False  # the user agreed to a run estimated over 15 minutes


class ImageMapKScoreOut(BaseModel):
    k: int
    silhouette: float | None


class ImageMapReadingOut(BaseModel):
    """R-VMAP-05: groups against labels, and how often neighbours share a label."""

    ari: float | None
    neighbours_same: float | None
    chance: float | None
    labels: int


class ImageMapOut(BaseModel):
    method: str
    title: str
    projection: Literal["pca", "tsne", "pacmap", "trimap"]
    clustering: Literal["kmeans", "hdbscan"]
    rows: list[int]  # table row of each drawn point
    x: list[float]
    y: list[float]
    clusters: list[int]  # group of every row of the table (-1: none)
    k: int
    k_forced: bool
    scores: list[ImageMapKScoreOut]
    silhouette: float | None
    explained: list[float] | None
    reading: ImageMapReadingOut | None
    sentence: str
    notes: list[str]
    points: int
    of: int


class ImageMapStateOut(BaseModel):
    """`missing`: never built; `running`: being built (progress); `ready`: the map;
    `failed`: why."""

    state: Literal["missing", "running", "ready", "failed"]
    method: str
    map: ImageMapOut | None = None
    dims: int | None = None
    done: int | None = None
    total: int | None = None
    message: str | None = None
    error: str | None = None
    code: str | None = None
    seconds: float | None = None


class ImagePrepBordersOut(BaseModel):
    """R-VPR-06: share of images with a uniform frame, on a few thumbnails."""

    share: float
    looked_at: int
    suggest: bool


class ImagePrepOut(BaseModel):
    recipe: dict[str, Any]
    default: dict[str, Any]
    saved: bool
    task: Literal["classification", "anomaly", "clusters"]
    borders: ImagePrepBordersOut


class ImagePrepIn(BaseModel):
    recipe: dict[str, Any]


class ImagePrepPreviewIn(BaseModel):
    image_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    step: str
    recipe: dict[str, Any]
    seed: int = Field(default=0, ge=0, le=1_000_000)


class ImagePrepPreviewOut(BaseModel):
    step: str
    image_id: str
    enabled: bool  # the step is on in the recipe (else previewed with its settings anyway)
    geometric: bool  # R-VPR-08: no pixel difference
    before: str  # data:image/png;base64,...
    after: str
    before_size: list[int]
    after_size: list[int]
    before_measures: dict[str, float]
    after_measures: dict[str, float]
    notes: list[str]


class ImageSemanticGroupOut(BaseModel):
    """A group of R-VQ-17: table rows, the smallest cosine inside, two labels, train and test."""

    group: int
    kind: Literal["semantic"]
    rows: list[int]
    keep: int | None = None
    cosine: float
    two_labels: bool
    leak: bool


class ImageSemanticOut(BaseModel):
    """R-VQ-17 (Quality > Advanced): never computed before the user asks. `available` is false
    with its reason when the extra `vision` or the DINOv2 weights are missing (R-VDP-01, 02);
    `missing`: not asked yet; `running`: vectors being computed; `ready`: the groups."""

    state: Literal["missing", "running", "ready", "failed"]
    available: bool
    reason: str | None = None
    network: str
    cosine: float
    cached: int  # images whose vectors are already on disk (a run or an earlier search)
    images: int
    message: str | None = None
    done: int | None = None
    total: int | None = None
    error: str | None = None
    long_run: str | None = None  # R-VDP-05: the estimate, when the user must confirm
    seconds: float | None = None
    sentence: str | None = None
    groups: list[ImageSemanticGroupOut] = []
    pairs: int | None = None
    compared: int | None = None
    leaks: int | None = None
    conflicts: int | None = None
    style: str | None = None  # the set's style: most images alike, nothing flagged


class ImageSemanticIn(BaseModel):
    confirm_long: bool = False
