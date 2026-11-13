"""Explaining the Advanced > More models of a run, and TCAV (118).

- ProtoPNet-lite: what the run kept (`protopnet.json`): every prototype (the training image and
  the patch it was pushed onto) and, for each scored image, the three prototypes that add the
  most to the predicted class (R-VPP-06). Read as kept: nothing is computed again.
- Concept bottleneck: the concepts predicted for a scored image, their true values when known
  and the class head (`concept_bottleneck.json`): the page recomputes the class when the user
  corrects a concept (R-VCB-05).
- Fine-tuning: the network's own vectors and head (`arrays/finetune_f<fold>_*.npy`): the nearest
  examples of "Why?" are read in the fine-tuned space, the probabilities from the kept head.
- TCAV (R-VTC-01 to 05), on demand, on the run's linear probe on DINOv2 (frozen backbone + the
  probe's head refitted as in 117); cached next to the run.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from app.core.errors import AppError
from app.modules.vision.concepts import ConceptOut, concept_list, concept_rows
from app.modules.vision.explain import ExplainService, WhyImage, _Run

MORE = ("finetune", "protopnet", "concept_bottleneck")
TCAV_POOL = 600  # random images decoded at most (R-VTC-05)
TCAV_CLASS = 100  # R-VTC-04
DECODE = 256  # TCAV decodes small: the network reads 224 px
NOT_A_MAP = (
    "Maps explain the linear probe: open it in the model list. This model's own evidence is "
    "shown above (its prototypes, its concepts or its nearest examples)."
)


# --- API shapes ------------------------------------------------------------------------------


class WhyLook(BaseModel):
    prototype: int
    label: str  # the prototype's class
    similarity: float
    weight: float
    points: float  # similarity x weight: what it adds to the class
    box: tuple[float, float, float, float]  # the patch of this image, relative
    source: WhyImage  # the training image the prototype was pushed onto
    source_box: tuple[float, float, float, float]


class WhyLooksLike(BaseModel):
    """R-VPP-06: "this looks like that"."""

    label: str | None  # the class explained (the predicted one)
    probability: float | None
    looks: list[WhyLook]
    total: float | None  # the points of all prototypes for that class (its score)
    sentence: str
    tested: bool
    notes: list[str] = []


class WhyConcept(BaseModel):
    key: str
    name: str
    kind: str  # "binary" or "numeric"
    predicted: float  # a probability (binary) or a value (numeric)
    truth: float | None  # the image's own value, when the table knows it
    positive: str | None = None
    note: str = ""


class WhyConceptHead(BaseModel):
    classes: list[str]
    mean: list[float]
    scale: list[float]
    weight: list[list[float]]  # (k, c) or (1, c) for two classes
    bias: list[float]
    setting: str


class WhyConcepts(BaseModel):
    """R-VCB-05: the concepts the bottleneck read in this image, and its class head."""

    space: str
    concepts: list[WhyConcept]
    head: WhyConceptHead
    tested: bool
    sentence: str
    notes: list[str] = []


class ProtoRef(BaseModel):
    prototype: int
    label: str
    weight: float
    image: WhyImage
    box: tuple[float, float, float, float]
    duplicate_of: int | None = None


class ProtoClass(BaseModel):
    label: str
    prototypes: list[ProtoRef]


class ProtoOverview(BaseModel):
    classes: list[ProtoClass]
    grid: tuple[int, int]
    distinct: int
    sentence: str
    notes: list[str] = []


class TcavOptions(BaseModel):
    available: bool
    reason: str | None
    concepts: list[ConceptOut]
    layers: list[str]
    layer: str | None  # the default (R-VTC-02)
    cost: str


class WhyTcavClass(BaseModel):
    label: str
    images: int
    score: float
    spread: float
    random: float
    random_spread: float
    p_value: float | None
    significant: bool
    sentence: str


class WhyTcav(BaseModel):
    concept: str
    concept_key: str
    layer: str
    layers: list[str]
    images: int
    random_images: int
    cav_accuracy: float
    random_accuracy: float
    classes: list[WhyTcavClass]
    sentence: str
    seconds: float
    cached: bool = False
    notes: list[str] = []


# --- the models' kept evidence ---------------------------------------------------------------


def _json(run: _Run, name: str) -> dict[str, Any]:
    path: Path = run.folder / name
    if not path.is_file():
        raise AppError(
            "This run did not keep what explains this model: train it again.",
            code="not_found",
            status_code=404,
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _array(run: _Run, name: str) -> np.ndarray:
    path: Path = run.folder / "arrays" / name
    if not path.is_file():
        raise AppError(
            "This run did not keep the fine-tuned network's vectors: train it again.",
            code="not_found",
            status_code=404,
        )
    return np.load(path, allow_pickle=False)


def fit(service: ExplainService, run: _Run, model: dict[str, Any], fold: int) -> dict[str, Any]:
    """`ExplainService._fit` for the More models: the space their evidence is read in."""
    from quantilo.vision import classic_models as cm

    name = model["key"]
    train, _test = run.folds[fold]
    if name == "finetune":
        rows = _array(run, f"finetune_f{fold}_rows.npy").astype(int)
        vectors = _array(run, f"finetune_f{fold}_vectors.npy").astype(np.float64)
        weight = _array(run, f"finetune_f{fold}_weight.npy")
        bias = _array(run, f"finetune_f{fold}_bias.npy")
        where = {int(r): i for i, r in enumerate(rows.tolist())}
        bank = np.array([r for r in train.tolist() if r in where], dtype=int)

        def proba(v: np.ndarray) -> np.ndarray:
            s = np.asarray(v, dtype=np.float64) @ weight.T + bias
            s -= s.max(axis=1, keepdims=True)
            e = np.exp(s)
            return e / e.sum(axis=1, keepdims=True)

        def kept(rows_: np.ndarray) -> np.ndarray:
            missing = [int(r) for r in rows_.tolist() if int(r) not in where]
            if missing:
                raise AppError(
                    "The fine-tuned vectors are kept for the images of this fold only.",
                    code="not_found",
                    status_code=404,
                )
            return vectors[[where[int(r)] for r in rows_.tolist()]]

        return {
            "bank": bank,
            "bank_vectors": kept(bank),
            "vectors": kept,
            "proba": proba,
            "metric": "cosine",
            "space": "the fine-tuned network's own vectors (cosine), kept with the run",
            "notes": [],
        }
    deep = bool(run.result.get("deep", {}).get("used"))
    if deep:
        x = service._deep(run, train)
        good = np.isfinite(x).all(axis=1)
        bank, bank_vectors = train[good], x[good]

        def vectors_of(rows_: np.ndarray) -> np.ndarray:
            return np.nan_to_num(service._deep(run, rows_))

        space = "the frozen DINOv2 ViT-S/14 vectors (cosine)"
    else:
        x = service._classic(run, train)
        good = np.isfinite(x).all(axis=1)
        bank = train[good]
        transform = cm.descriptor_space(x[good], int(run.result.get("seed", 0)))
        bank_vectors = transform.transform(x[good])

        def vectors_of(rows_: np.ndarray) -> np.ndarray:
            return transform.transform(np.nan_to_num(service._classic(run, rows_)))

        space = "the PCA components of the classic descriptors (cosine)"
    own = "its prototypes" if name == "protopnet" else "its concepts"

    def row_proba(row: int) -> np.ndarray | None:
        record = run.predictions.get(row) or {}
        values = record.get(f"{name}_probs")
        return None if values is None else np.asarray(values, dtype=np.float64)

    return {
        "bank": bank,
        "bank_vectors": bank_vectors,
        "vectors": vectors_of,
        "proba": None,
        "row_proba": row_proba,
        "metric": "cosine",
        "space": f"{space}, to compare with; the model's own evidence is {own}, above",
        "notes": [],
    }


def _fold_of_row(run: _Run, row: int) -> tuple[int, bool]:
    return run.fold_of(row)


def looks_like(service: ExplainService, run: _Run, row: int) -> WhyLooksLike:
    data = _json(run, "protopnet.json")
    fold, tested = _fold_of_row(run, row)
    kept = next((f for f in data["folds"] if f["fold"] == fold), None)
    record = run.predictions.get(row) or {}
    label = record.get("protopnet_pred")
    probability = record.get("protopnet_proba")
    if kept is None or str(row) not in kept["images"]:
        return WhyLooksLike(
            label=None,
            probability=None,
            looks=[],
            total=None,
            tested=False,
            sentence=(
                "The prototype network learnt from this image: its 'this looks like that' is kept "
                "for the scored images only. Open an image of the test."
            ),
        )
    prototypes = {int(p["index"]): p for p in kept["prototypes"]}
    looks = []
    for c in kept["images"][str(row)]:
        proto = prototypes[int(c["prototype"])]
        looks.append(
            WhyLook(
                prototype=int(c["prototype"]),
                label=str(proto["label"]),
                similarity=float(c["similarity"]),
                weight=float(c["weight"]),
                points=float(c["points"]),
                box=tuple(c["box"]),  # type: ignore[arg-type]
                source=service._ref(run, int(proto["row"])),
                source_box=tuple(proto["box"]),  # type: ignore[arg-type]
            )
        )
    total = float(sum(look.points for look in looks)) if label is not None else None
    first = looks[0] if looks else None
    sentence = (
        f"This patch looks like that prototype of {first.label} (from a training image): "
        f"similarity {first.similarity:.2f} x weight {first.weight:.2f} = {first.points:.2f} "
        f"points towards {label}; the three largest add {total:.2f} (R-VPP-06). This is how the "
        "score was computed, not a post-hoc map."
        if first is not None and total is not None
        else "No prototype adds to this class."
    )
    return WhyLooksLike(
        label=label,
        probability=probability,
        looks=looks,
        total=total,
        tested=tested,
        sentence=sentence,
        notes=list(kept.get("notes") or []),
    )


def concepts_of(service: ExplainService, run: _Run, row: int) -> WhyConcepts:
    data = _json(run, "concept_bottleneck.json")
    fold, tested = _fold_of_row(run, row)
    kept = next((f for f in data["folds"] if f["fold"] == fold), None)
    if kept is None or row not in kept["rows"]:
        raise AppError(
            "The bottleneck's concepts are kept for the scored images only: open an image of "
            "the test.",
            code="not_found",
            status_code=404,
        )
    i = kept["rows"].index(row)
    predicted = kept["predicted"][i]
    truth = data["truth"].get(str(row)) or [None] * len(data["concepts"])
    concepts = [
        WhyConcept(
            key=c["key"],
            name=c["name"],
            kind=c["kind"],
            predicted=float(predicted[j]),
            truth=None if truth[j] is None else float(truth[j] or 0.0),
            positive=c.get("positive"),
            note=c.get("note") or "",
        )
        for j, c in enumerate(data["concepts"])
    ]
    head = WhyConceptHead(**kept["head"])
    named = ", ".join(
        f"{c.name} {c.predicted:.2f}" if c.kind == "binary" else f"{c.name} {c.predicted:.3g}"
        for c in concepts[:4]
    )
    return WhyConcepts(
        space=str(data.get("space", "")),
        concepts=concepts,
        head=head,
        tested=tested,
        sentence=(
            f"The bottleneck read {named}{' …' if len(concepts) > 4 else ''} in this image, "
            "then the class from those concepts only. Change a concept to see the prediction "
            "follow (R-VCB-05)."
        ),
        notes=list(kept.get("notes") or []),
    )


def prototypes_overview(service: ExplainService, run: _Run) -> ProtoOverview | None:
    if not (run.folder / "protopnet.json").is_file():
        return None
    data = _json(run, "protopnet.json")
    kept = data["folds"][0]
    by_class: dict[str, list[ProtoRef]] = {c: [] for c in data["classes"]}
    for p in kept["prototypes"]:
        by_class.setdefault(str(p["label"]), []).append(
            ProtoRef(
                prototype=int(p["index"]),
                label=str(p["label"]),
                weight=float(p["weight"]),
                image=service._ref(run, int(p["row"])),
                box=tuple(p["box"]),  # type: ignore[arg-type]
                duplicate_of=p.get("duplicate_of"),
            )
        )
    distinct = len({(p["row"], p["cell"]) for p in kept["prototypes"]})
    grid = data["grid"]
    return ProtoOverview(
        classes=[
            ProtoClass(label=c, prototypes=sorted(v, key=lambda r: -r.weight))
            for c, v in by_class.items()
        ],
        grid=(int(grid[0]), int(grid[1])),
        distinct=distinct,
        sentence=(
            f"{data.get('per_class', 10)} prototypes per class, each a patch of a real training "
            f"image ({distinct} distinct patches on a {grid[0]} x {grid[1]} grid): an image is "
            "scored by how much its patches look like them (R-VPP-02, R-VPP-04)."
        ),
        notes=list(kept.get("notes") or []),
    )


# --- TCAV ------------------------------------------------------------------------------------


def _probe(run: _Run) -> dict[str, Any] | None:
    return next((m for m in run.models if m["key"] == "deep_linear"), None)


def tcav_options(service: ExplainService, run: _Run) -> TcavOptions:
    from quantilo.vision import deep

    record = service.vision._record(run.dataset_id)
    found = [c for c in concept_list(service.vision, record, run.table) if c.tcav]
    reason = None
    layers: list[str] = []
    layer = None
    if run.task != "classification":
        reason = "TCAV tests a classifier: this run detects anomalies."
    elif _probe(run) is None:
        reason = (
            "Needs a network: train with the DINOv2 vectors (the linear probe), not the classic "
            "descriptors only."
        )
    else:
        backbone = deep.status().backbone("dinov2_s14")
        if not backbone.available:
            reason = backbone.reason
    if reason is None:
        depth = 12  # DINOv2 ViT-S/14
        layers = [f"block {i + 1}" for i in range(depth)]
        layer = layers[max(0, round(0.75 * depth) - 1)]
    if reason is None and not found:
        reason = (
            "No concept with 20 images or more: save a group of images as a concept from the "
            "gallery's selection, or add a true/false column (R-VTC-01)."
        )
    return TcavOptions(
        available=reason is None,
        reason=reason,
        concepts=found,
        layers=layers,
        layer=layer,
        cost="about 20 s on a GPU (several minutes on the CPU) the first time, then kept",
    )


def tcav(service: ExplainService, run: _Run, key: str, layer: str | None) -> WhyTcav:
    options = tcav_options(service, run)
    if not options.available:
        raise AppError(
            options.reason or "TCAV is not available.", code="not_supported", status_code=409
        )
    if layer is not None and layer not in options.layers:
        raise AppError(f"No layer {layer!r}.", code="not_found", status_code=404)
    record = service.vision._record(run.dataset_id)
    name, rows = concept_rows(service.vision, record, run.table, key)
    digest = hashlib.sha1(",".join(map(str, rows)).encode()).hexdigest()[:12]
    layer = layer or options.layer
    return service._cached(
        run,
        WhyTcav,
        ("tcav", "118-2", key, digest, layer),
        lambda: _tcav(service, run, key, name, rows, layer),
    )


def _tcav(
    service: ExplainService,
    run: _Run,
    key: str,
    name: str,
    rows: list[int],
    layer: str | None,
) -> WhyTcav:
    from quantilo.vision.deep import tcav as tc

    seed = int(run.result.get("seed", 0))
    rng = np.random.default_rng(seed)
    usable = [
        i for i in range(len(run.image_ids)) if run.sources[i] is not None and not _excluded(run, i)
    ]
    inside = set(rows)
    concept_rows_ = [r for r in rows if r in set(usable)]
    if len(concept_rows_) > tc.MAX_IMAGES:
        concept_rows_ = sorted(rng.choice(concept_rows_, tc.MAX_IMAGES, replace=False).tolist())
    outside = [i for i in usable if i not in inside]
    if len(outside) > TCAV_POOL:
        outside = sorted(rng.choice(outside, TCAV_POOL, replace=False).tolist())
    model = _probe(run)
    assert model is not None
    fitted = service._fitted(run, model, 0)
    clf = service._classifier(fitted)
    scored = [int(r) for f in run.folds for r in f[1].tolist()]
    by_class: dict[str, Sequence[Any]] = {}
    for label in run.classes:
        mine = [r for r in scored if run.labels[r] is not None and str(run.labels[r]) == label]
        if len(mine) > TCAV_CLASS:
            mine = sorted(rng.choice(mine, TCAV_CLASS, replace=False).tolist())
        by_class[label] = [service._decode(run, r, DECODE) for r in mine]
    notes = []
    if len(run.folds) > 1:
        notes.append("The probe of the first fold is the one tested.")
    try:
        found = tc.tcav(
            clf,
            [service._decode(run, r, DECODE) for r in concept_rows_],
            [service._decode(run, r, DECODE) for r in outside],
            by_class,
            name=name,
            layer=layer,
            seed=seed,
        )
    except ValueError as error:
        raise AppError(str(error), code="not_supported", status_code=422) from error
    finally:
        from quantilo.vision.deep.anomaly import free_gpu

        free_gpu()
    return WhyTcav(
        concept=found.concept,
        concept_key=key,
        layer=found.layer,
        layers=found.layers,
        images=found.images,
        random_images=found.random_images,
        cav_accuracy=found.cav_accuracy,
        random_accuracy=found.random_accuracy,
        classes=[WhyTcavClass(**vars(c)) for c in found.classes],
        sentence=found.sentence,
        seconds=found.seconds,
        notes=[*found.notes, *notes],
    )


def _excluded(run: _Run, row: int) -> bool:
    from quantilo.vision.task import column

    decoded = column(run.table, "decode_ok")[row]
    excluded = column(run.table, "excluded")[row]
    return not decoded or bool(excluded)


__all__ = [
    "MORE",
    "NOT_A_MAP",
    "ProtoOverview",
    "TcavOptions",
    "WhyConcepts",
    "WhyLooksLike",
    "WhyTcav",
    "concepts_of",
    "fit",
    "looks_like",
    "prototypes_overview",
    "tcav",
    "tcav_options",
]
