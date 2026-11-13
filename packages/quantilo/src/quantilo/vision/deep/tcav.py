"""TCAV: does a concept the user named push the network towards a class? (118; study § 8.)

Kim, Wattenberg, Gilmer, Cai, Wexler, Viégas and Sayres (2018), *Interpretability Beyond
Feature Attribution: Quantitative Testing with Concept Activation Vectors*; course 3 of the Duke
XAI specialisation (§ 13.4). Not a map: one number per class and concept, with its test.

- R-VTC-01 the concept: at least 20 images (a saved group, the images where a true/false column
  is true, or the images of a class); at most 200 are used (drawn with the seed).
- R-VTC-02 the layer: the output of a block three quarters of the way up (DINOv2 ViT-S/14: block
  9 of 12; a ResNet: its third stage), read as the average over its tokens (or cells). The
  network is the run's: the frozen backbone and the probe's head.
- R-VTC-03 the CAV: a logistic regression separates the concept's activations from those of as
  many random images of the set (not in the concept); the CAV is the unit normal pointing to the
  concept; its accuracy on a quarter held out is said (a CAV that does not separate is no
  concept the layer knows).
- R-VTC-04 the TCAV score of class k: the share of the scored images of class k (at most 100)
  whose class score (logit) rises when the layer's activation moves along the CAV: the
  directional derivative, the gradient of the logit with respect to every token (or cell) of
  the layer, summed, dotted with the CAV (the activation moved by the same amount everywhere).
- R-VTC-05 significance: 10 CAVs of the concept against 10 random sets, and 10 CAVs of a random
  set against the same 10; a two-sided Welch t-test of the two lists of TCAV scores; "concept
  found" when p < 0.05, "not significant" otherwise (the score is then not reported as
  meaningful). Random images come from the set itself, not from ImageNet as in the paper: they
  share the set's look, so the test asks "this concept rather than any images of this set".

torch is imported inside the functions only (extra `vision`, R-VDP-01).
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

MIN_IMAGES = 20  # R-VTC-01
MAX_IMAGES = 200
LAYER_SHARE = 0.75  # R-VTC-02
RANDOM_SETS = 10  # R-VTC-05
ALPHA = 0.05
MAX_CLASS_IMAGES = 100  # R-VTC-04
HELD_OUT = 0.25  # R-VTC-03
BATCH = 16

Progress = Callable[[float, str], None]


def layers(module: Any, kind: str) -> list[tuple[str, Any]]:
    """The blocks a TCAV can read, from the input up (a ViT's blocks, a ResNet's stages)."""
    if kind == "vit" and hasattr(module, "blocks"):
        return [(f"block {i + 1}", b) for i, b in enumerate(module.blocks)]
    names = [n for n in ("layer1", "layer2", "layer3", "layer4") if hasattr(module, n)]
    return [(n.replace("layer", "stage "), getattr(module, n)) for n in names]


def default_layer(module: Any, kind: str) -> str:
    """R-VTC-02: three quarters of the way up."""
    found = layers(module, kind)
    if not found:
        raise ValueError("This network has no blocks to read a concept from.")
    index = max(0, min(len(found) - 1, round(LAYER_SHARE * len(found)) - 1))
    return found[index][0]


def _layer(module: Any, kind: str, name: str) -> Any:
    for label, block in layers(module, kind):
        if label == name:
            return block
    raise ValueError(f"No layer {name!r}: one of {', '.join(n for n, _ in layers(module, kind))}.")


def _pool(output: Any, kind: str) -> Any:
    """(n, tokens, d) -> mean over tokens; (n, c, h, w) -> mean over cells."""
    if output.ndim == 3:
        return output.mean(dim=1)
    return output.mean(dim=(2, 3))


def _sum(output: Any) -> Any:
    if output.ndim == 3:
        return output.sum(dim=1)
    return output.sum(dim=(2, 3))


def activations(clf: Any, images: Sequence[Image.Image], layer: str) -> np.ndarray:
    """R-VTC-02: `(n, d)` mean activation of the layer for each image."""
    import torch

    block = _layer(clf.module, clf.spec.kind, layer)
    found: list[Any] = []

    def hook(_module: Any, _args: Any, output: Any) -> None:
        found.append(_pool(output, clf.spec.kind).float().cpu())

    handle = block.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, len(images), BATCH):
                x = torch.cat([clf.tensor(i) for i in images[start : start + BATCH]])
                clf.module.forward_features(x)
    finally:
        handle.remove()
    if not found:
        return np.zeros((0, 0), dtype=np.float64)
    return torch.cat(found).double().numpy()


def gradients(
    clf: Any, images: Sequence[Image.Image], layer: str, targets: Sequence[int]
) -> dict[int, np.ndarray]:
    """R-VTC-04: for each target class, `(n, d)`: the gradient of its logit with respect to the
    layer's output, summed over tokens (or cells): its dot product with a CAV is the directional
    derivative along it."""
    import torch

    block = _layer(clf.module, clf.spec.kind, layer)
    held: dict[str, Any] = {}

    def hook(_module: Any, _args: Any, output: Any) -> Any:
        output.retain_grad()
        held["out"] = output
        return output

    out: dict[int, list[np.ndarray]] = {int(k): [] for k in targets}
    handle = block.register_forward_hook(hook)
    try:
        for start in range(0, len(images), BATCH):
            x = torch.cat([clf.tensor(i) for i in images[start : start + BATCH]])
            with torch.enable_grad():
                # the weights stay frozen: the input carries the graph to the layer
                x = x.detach().clone().requires_grad_(True)
                scores = clf.scores(x)
                layer_out = held["out"]
                for k in out:
                    if layer_out.grad is not None:
                        layer_out.grad = None
                    clf.module.zero_grad(set_to_none=True)
                    scores[:, k].sum().backward(retain_graph=True)
                    assert layer_out.grad is not None
                    out[k].append(_sum(layer_out.grad).double().cpu().numpy())
    finally:
        handle.remove()
    return {k: np.concatenate(v) if v else np.zeros((0, 0)) for k, v in out.items()}


def learn_cav(
    positive: np.ndarray, negative: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, float]:
    """R-VTC-03: (unit CAV pointing to `positive`, held-out accuracy)."""
    from sklearn.linear_model import LogisticRegression

    x = np.vstack([positive, negative])
    y = np.r_[np.ones(len(positive)), np.zeros(len(negative))].astype(int)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    cut = max(2, round(HELD_OUT * len(x)))
    test, train = order[:cut], order[cut:]
    mean = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    scale[scale < 1e-8] = 1.0
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit((x[train] - mean) / scale, y[train])
    accuracy = float(np.mean(model.predict((x[test] - mean) / scale) == y[test]))
    model.fit((x - mean) / scale, y)
    # back to the activation space: a step along w / scale raises the concept's logit most
    direction = model.coef_[0] / scale
    norm = float(np.linalg.norm(direction)) or 1.0
    return direction / norm, accuracy


def tcav_score(grads: np.ndarray, cav: np.ndarray) -> float:
    """R-VTC-04: share of the images whose directional derivative is positive."""
    if len(grads) == 0:
        return float("nan")
    return float(np.mean(grads @ cav > 0))


@dataclass
class ClassTcav:
    label: str
    images: int
    score: float  # mean TCAV score over the concept's CAVs
    spread: float  # their standard deviation
    random: float  # mean TCAV score of the random CAVs
    random_spread: float
    p_value: float | None
    significant: bool
    sentence: str


@dataclass
class TcavResult:
    concept: str
    layer: str
    layers: list[str]
    images: int  # concept images used
    random_images: int  # size of each random set
    cav_accuracy: float  # mean held-out accuracy of the concept's CAVs
    random_accuracy: float
    classes: list[ClassTcav]
    sentence: str
    seconds: float
    notes: list[str] = field(default_factory=list)


def tcav(
    clf: Any,
    concept: Sequence[Image.Image],
    pool: Sequence[Image.Image],
    by_class: dict[str, Sequence[Image.Image]],
    *,
    name: str = "the concept",
    layer: str | None = None,
    seed: int = 0,
    random_sets: int = RANDOM_SETS,
    progress: Progress | None = None,
) -> TcavResult:
    """R-VTC-01 to 05. `clf` is a `vision.explain.Classifier` (backbone + linear head); `pool`
    holds images of the set outside the concept, the random sets are drawn from it."""
    from scipy import stats

    started = time.perf_counter()
    notes: list[str] = []
    if len(concept) < MIN_IMAGES:
        raise ValueError(
            f"A concept needs at least {MIN_IMAGES} images (R-VTC-01); {name!r} has {len(concept)}."
        )
    rng = np.random.default_rng(seed)
    concept = list(concept)
    if len(concept) > MAX_IMAGES:
        concept = [concept[i] for i in sorted(rng.choice(len(concept), MAX_IMAGES, replace=False))]
        notes.append(f"{MAX_IMAGES} of the concept's images drawn (seed {seed}, R-VTC-01).")
    size = min(len(concept), len(pool))
    if size < MIN_IMAGES:
        raise ValueError(
            f"Too few images outside the concept to draw random sets: {len(pool)} (R-VTC-05)."
        )
    if size < len(concept):
        notes.append(
            f"Random sets of {size} images, fewer than the concept's {len(concept)}: the set has "
            "no more images outside it."
        )
    module, kind = clf.module, clf.spec.kind
    names = [n for n, _ in layers(module, kind)]
    layer = layer or default_layer(module, kind)
    sets = [rng.choice(len(pool), size, replace=False) for _ in range(random_sets + 1)]
    used = sorted({int(i) for s in sets for i in s})
    if progress is not None:
        progress(0.05, "Activations of the concept")
    concept_acts = activations(clf, concept, layer)
    if progress is not None:
        progress(0.25, "Activations of the random images")
    pool_acts_used = activations(clf, [pool[i] for i in used], layer)
    where = {i: j for j, i in enumerate(used)}
    random_acts = [pool_acts_used[[where[int(i)] for i in s]] for s in sets]
    concept_cavs, concept_acc = [], []
    random_cavs, random_acc = [], []
    for i in range(1, random_sets + 1):
        cav, acc = learn_cav(concept_acts, random_acts[i], seed + i)
        concept_cavs.append(cav)
        concept_acc.append(acc)
        cav, acc = learn_cav(random_acts[0], random_acts[i], seed + 100 + i)
        random_cavs.append(cav)
        random_acc.append(acc)
    labels = list(clf.head.classes)
    results = []
    for n_done, (label, images) in enumerate(by_class.items()):
        if label not in labels or not images:
            continue
        images = list(images)
        if len(images) > MAX_CLASS_IMAGES:
            images = [
                images[i] for i in sorted(rng.choice(len(images), MAX_CLASS_IMAGES, replace=False))
            ]
        if progress is not None:
            progress(0.45 + 0.5 * n_done / max(1, len(by_class)), f"Gradients of {label}")
        k = labels.index(label)
        grads = gradients(clf, images, layer, [k])[k]
        scores = np.array([tcav_score(grads, c) for c in concept_cavs])
        nulls = np.array([tcav_score(grads, c) for c in random_cavs])
        p_value: float | None
        if (
            np.allclose(scores, scores[0])
            and np.allclose(nulls, nulls[0])
            and scores[0] == nulls[0]
        ):
            p_value = 1.0
        else:
            raw = float(np.asarray(stats.ttest_ind(scores, nulls, equal_var=False)[1]))
            p_value = raw if np.isfinite(raw) else None
        significant = p_value is not None and p_value < ALPHA
        mean = float(scores.mean())
        if significant:
            sentence = (
                f"Concept found: moving towards {name} raises the score of {label} for "
                f"{mean:.0%} of its images (random directions: {float(nulls.mean()):.0%}; "
                f"p = {p_value:.3g})."
                if mean >= float(nulls.mean())
                else f"Concept found, against {label}: moving towards {name} lowers its score for "
                f"{1 - mean:.0%} of its images (random directions: {1 - float(nulls.mean()):.0%}; "
                f"p = {p_value:.3g})."
            )
        else:
            sentence = (
                f"Not significant for {label}: {name} does no better than random directions "
                f"({mean:.0%} against {float(nulls.mean()):.0%}"
                + (f"; p = {p_value:.2g})." if p_value is not None else ").")
            )
        results.append(
            ClassTcav(
                label=label,
                images=len(images),
                score=round(mean, 4),
                spread=round(float(scores.std()), 4),
                random=round(float(nulls.mean()), 4),
                random_spread=round(float(nulls.std()), 4),
                p_value=None if p_value is None else round(p_value, 6),
                significant=bool(significant),
                sentence=sentence,
            )
        )
    accuracy = float(np.mean(concept_acc))
    if accuracy < 0.7:
        notes.append(
            f"The concept's CAVs separate it from random images only {accuracy:.0%} of the time "
            f"on held-out images: {layer} hardly tells it apart (R-VTC-03)."
        )
    raises = [r.label for r in results if r.significant and r.score >= r.random]
    lowers = [r.label for r in results if r.significant and r.score < r.random]
    parts = ([f"raises {', '.join(raises)}"] if raises else []) + (
        [f"lowers {', '.join(lowers)}"] if lowers else []
    )
    sentence = (
        f"The concept {name!r} {' and '.join(parts)} (p < {ALPHA}) at {layer}; its CAV separates "
        f"it from random images {accuracy:.0%} of the time (R-VTC-05)."
        if parts
        else f"The concept {name!r} is not significant for any class at {layer} (R-VTC-05)."
    )
    return TcavResult(
        concept=name,
        layer=layer,
        layers=names,
        images=len(concept),
        random_images=size,
        cav_accuracy=round(accuracy, 4),
        random_accuracy=round(float(np.mean(random_acc)), 4),
        classes=results,
        sentence=sentence,
        seconds=round(time.perf_counter() - started, 2),
        notes=notes,
    )


__all__ = [
    "ClassTcav",
    "TcavResult",
    "activations",
    "default_layer",
    "gradients",
    "layers",
    "learn_cav",
    "tcav",
    "tcav_score",
]
