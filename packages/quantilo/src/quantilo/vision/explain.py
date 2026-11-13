"""Why the model said what it said about one image (117; study 2026-11-11 §§ 8, 9.2, 13.4).

The model's own evidence comes first and lives in `quantilo.interpret.prototypes` (nearest
examples, prototypes, counterfactual example) and in the anomaly detectors (the patch PCA's
error map, PatchCore's nearest normal patch). This module adds the post-hoc maps of the Duke
XAI specialisation (course 3, "deep networks": saliency, attention) and the test that says
whether such a map depends on the model at all:

- R-VEX-05 occlusion (Zeiler and Fergus, 2014): a grid of 8 x 8 cells, each blurred in turn
  (Gaussian, radius = half a cell); value = drop of the class's log-odds, log(p / (1 - p)), p
  clipped to [1e-15, 1 - 1e-15] (a probability shown as 1.00 still moves in its last digits:
  measured on Beans, where 1e-9 hid every change).
  Any model, the classic ones and the browser version included (64 + 1 predictions).
- R-VEX-06 Grad-CAM (Selvaraju et al., 2017): gradients of the class score (the logit, before
  the softmax, which saturates) averaged over the last feature map (a CNN's last stage, a ViT's
  patch tokens laid out on their grid), ReLU of the weighted sum. Every gradient method reads
  that score; a binary head's score is +-z / 2, so its softmax is the model's sigmoid.
- R-VEX-07 Integrated Gradients (Sundararajan et al., 2017): 32 steps (trapezoids) from a
  blurred copy of the image (radius 16 px at 224 px) to the image; the map is the sum of |a|
  over the colour channels; the completeness gap (sum of attributions against the change of
  score) is said.
- R-VEX-08 vanilla saliency (Simonyan et al., 2014) and SmoothGrad (Smilkov et al., 2017):
  |gradient|, largest over the channels; SmoothGrad averages 25 copies with Gaussian noise of
  15 % of the input's range.
- R-VEX-09 attention rollout (Abnar and Zuidema, 2020) for a ViT: each block's attention,
  averaged over heads, mixed half and half with the identity (the residual), rows normalised,
  multiplied from the first block; the CLS token's row over the patches. The same for every
  class, and "attention is not necessarily explanation" (Jain and Wallace, 2019).
- R-VEX-10 sanity test (Adebayo et al., 2018): the map is computed again after randomising the
  head, then the head and the last block, then the last two, the last three (cascading); rank
  correlation (Spearman) with the trained model's map; a map whose correlation stays >= 0.5
  once the head and the last block are random is flagged "do not trust it"; a map that goes
  flat counts as changed (correlation 0, said).
- R-VEX-11 every post-hoc map carries the warning of `POST_HOC_WARNING`.

Maps are non-negative, in their own units (`unit`), on the grid they were computed on; `box`
says which part of the image they cover (relative coordinates: a network reads a centre crop).
The page draws them (`png_gray`: 8-bit grey, 0 = `low`, 255 = `high`) and does colour, opacity,
threshold and contours itself, without another call.

The occlusion and the rollout arithmetic are numpy only; torch is imported inside the gradient
functions only (extra `vision`, R-VDP-01).
"""

import io
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

GRID = 8  # R-VEX-05
IG_STEPS = 32  # R-VEX-07
IG_BLUR = 16.0  # px at 224
SMOOTH_SAMPLES = 25  # R-VEX-08
SMOOTH_NOISE = 0.15
SANITY_BLOCKS = 3  # R-VEX-10: head, then up to the last 3 blocks
SANITY_FLAG = 0.5
BATCH = 8

POST_HOC_WARNING = (
    "This heatmap is a post-hoc explanation: it shows where the model's output is sensitive, "
    "not the reasoning it used. Maps can look plausible even for a model that has learned "
    "nothing (Adebayo et al., 2018). For decisions that matter, prefer the nearest examples and "
    "prototypes, which are the model's own evidence (Rudin, 2019)."
)
ATTENTION_WARNING = (
    "Attention is not necessarily explanation (Jain and Wallace, 2019): the rollout shows where "
    "the network's class token gathered information, the same for every class."
)
OCCLUSION_WARNING = "Shows which regions change the prediction, not why."


@dataclass
class ExplanationMap:
    method: str  # occlusion, grad_cam, integrated_gradients, saliency, smoothgrad, rollout...
    title: str
    values: np.ndarray  # (rows, columns) float32, >= 0, in `unit`
    unit: str
    sentence: str
    warning: str  # "" for the model's own evidence
    own: bool = False  # the model's own computation (patch PCA, PatchCore), not post-hoc
    blocky: bool = False  # draw cells as squares (occlusion), not smoothed
    box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # covered part of the image
    target: str | None = None
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def low(self) -> float:
        return float(np.nanmin(self.values)) if self.values.size else 0.0

    @property
    def high(self) -> float:
        return float(np.nanmax(self.values)) if self.values.size else 0.0


# ---------------------------------------------------------------------------------------------
# Maps as pictures, and comparing maps


def to_uint8(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Values scaled to 0-255 between their minimum and maximum (a flat map is all 0)."""
    v = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    low, high = float(v.min()) if v.size else 0.0, float(v.max()) if v.size else 0.0
    if high - low <= 1e-12:
        return np.zeros(v.shape, dtype=np.uint8), low, high
    scaled = np.round((v - low) / (high - low) * 255.0)
    return scaled.astype(np.uint8), low, high


def png_gray(values: np.ndarray) -> bytes:
    """The map as an 8-bit grey PNG (0 = its minimum, 255 = its maximum)."""
    pixels, _, _ = to_uint8(values)
    out = io.BytesIO()
    Image.fromarray(pixels, "L").save(out, format="PNG", optimize=True)
    return out.getvalue()


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    # average ties
    sorted_values = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + i - 1) / 2.0
            start = i
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation of two maps of the same shape (NaN when one of them is flat)."""
    x, y = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    if x.shape != y.shape:
        raise ValueError("Both maps need the same shape.")
    rx, ry = _rank(x), _rank(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return math.nan
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def pool(values: np.ndarray, side: int) -> np.ndarray:
    """A map averaged down to `side` x `side` cells (maps of different methods compared, or a
    224 x 224 map sent as a lighter grid)."""
    v = np.asarray(values, dtype=np.float32)
    h, w = v.shape
    if h <= side and w <= side:
        return v
    image = Image.fromarray(v, mode="F").resize((side, side), Image.Resampling.BOX)
    return np.asarray(image, dtype=np.float32)


# ---------------------------------------------------------------------------------------------
# R-VEX-05: occlusion, any model


def log_odds(p: Any) -> Any:
    """log(p / (1 - p)), p clipped to [1e-15, 1 - 1e-15] (R-VEX-05)."""
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return np.log(q) - np.log1p(-q)


def blurred(image: Image.Image, radius: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=max(0.5, radius)))


def occlusion(
    image: Image.Image,
    predict: Callable[[Sequence[Image.Image]], np.ndarray],
    target: int,
    *,
    classes: Sequence[str] | None = None,
    grid: int = GRID,
    batch: int = 16,
    cancel: Callable[[], bool] | None = None,
) -> ExplanationMap:
    """R-VEX-05: `predict(images)` returns `(n, classes)` probabilities; the map is the drop of
    class `target`'s probability when each cell is blurred (cells that raise it count 0; the
    largest rise is said)."""
    started = time.perf_counter()
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    cell_w, cell_h = w / grid, h / grid
    soft = blurred(image, max(cell_w, cell_h) / 2.0)
    variants: list[Image.Image] = []
    for gy in range(grid):
        for gx in range(grid):
            box = (
                round(gx * cell_w),
                round(gy * cell_h),
                round((gx + 1) * cell_w),
                round((gy + 1) * cell_h),
            )
            copy = image.copy()
            copy.paste(soft.crop(box), box[:2])
            variants.append(copy)
    base = float(np.asarray(predict([image]))[0, target])
    probabilities = np.zeros(grid * grid, dtype=np.float64)
    for start in range(0, len(variants), batch):
        if cancel is not None and cancel():
            raise RuntimeError("Stopped.")
        p = np.asarray(predict(variants[start : start + batch]), dtype=np.float64)
        probabilities[start : start + len(p)] = p[:, target]
    values = (log_odds(base) - log_odds(probabilities)).reshape(grid, grid)
    lowest = float(probabilities.min())
    name = classes[target] if classes is not None else f"class {target}"
    top = np.unravel_index(int(np.argmax(values)), values.shape)
    at_top = float(probabilities[int(np.argmax(values.ravel()))])
    rise = float(-values.min())
    sentence = (
        f"Blurring the cell at row {top[0] + 1}, column {top[1] + 1} lowers the evidence for "
        f"{name} the most: {values.max():.2f} in log-odds (its probability goes from {base:.2f} "
        f"to {at_top:.2f}; R-VEX-05)."
        if values.max() > 1e-3
        else f"No blurred cell lowers the probability of {name} ({base:.2f}) (R-VEX-05)."
    )
    notes = []
    if rise > 0.05:
        notes.append(
            f"Blurring some cells raises it instead, by up to {rise:.2f} in log-odds: shown as 0."
        )
    if values.max() > 1e-3 and lowest > 0.5 and base > 0.5:
        notes.append(
            f"No single cell changes the answer: the probability stays at {lowest:.2f} or more."
        )
    return ExplanationMap(
        method="occlusion",
        title="Occlusion",
        values=np.clip(values, 0.0, None).astype(np.float32),
        unit="drop in log-odds",
        sentence=sentence,
        warning=f"{OCCLUSION_WARNING} {POST_HOC_WARNING}",
        blocky=True,
        target=name,
        seconds=round(time.perf_counter() - started, 2),
        notes=notes,
        extra={
            "probability": base,
            "lowest": lowest,
            "largest_rise": rise,
            "passes": len(variants) + 1,
        },
    )


# ---------------------------------------------------------------------------------------------
# R-VEX-09: attention rollout, the arithmetic (numpy)


def rollout(attentions: Sequence[np.ndarray], prefix: int, grid: tuple[int, int]) -> np.ndarray:
    """`attentions[l]` is block l's `(heads, tokens, tokens)` attention; returns the CLS row of
    the rolled-out attention over the patch tokens, on the patch grid."""
    if not attentions:
        raise ValueError("No attention to roll out.")
    tokens = attentions[0].shape[-1]
    joint = np.eye(tokens)
    for a in attentions:
        mean = np.asarray(a, dtype=np.float64).mean(axis=0)
        mixed = 0.5 * mean + 0.5 * np.eye(tokens)
        mixed /= mixed.sum(axis=-1, keepdims=True)
        joint = mixed @ joint
    patches = joint[0, prefix:]
    rows, cols = grid
    if patches.size != rows * cols:
        raise ValueError(f"{patches.size} patch tokens do not fill a {rows} x {cols} grid.")
    return patches.reshape(rows, cols).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# A differentiable classifier: the frozen network and the run's linear head


@dataclass
class LinearHead:
    """A standardisation and a multinomial (or binary) logistic head: probabilities of `classes`
    from a vector, exactly as the scikit-learn pipeline computes them."""

    weight: np.ndarray  # (k, d); k = 1 for a binary logistic
    bias: np.ndarray  # (k,)
    mean: np.ndarray  # (d,)
    scale: np.ndarray  # (d,)
    classes: list[str]

    def proba(self, vectors: np.ndarray) -> np.ndarray:
        z = (np.asarray(vectors, dtype=np.float64) - self.mean) / self.scale
        logits = z @ self.weight.T + self.bias
        if self.weight.shape[0] == 1:
            p = 1.0 / (1.0 + np.exp(-logits[:, 0]))
            return np.stack([1.0 - p, p], axis=1)
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)


def head_from_pipeline(model: Any, classes: Sequence[str]) -> LinearHead:
    """The `LinearHead` of a fitted scikit-learn pipeline `[StandardScaler,] LogisticRegression`
    (`classic_models.fit_logistic`), its rows put in the order of `classes` (a class the model
    never saw gets a very negative bias)."""
    steps = list(getattr(model, "steps", [("model", model)]))
    logistic = steps[-1][1]
    scaler = steps[0][1] if len(steps) > 1 else None
    coef = np.asarray(logistic.coef_, dtype=np.float64)
    intercept = np.asarray(logistic.intercept_, dtype=np.float64)
    d = coef.shape[1]
    mean = np.zeros(d) if scaler is None else np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.ones(d) if scaler is None else np.asarray(scaler.scale_, dtype=np.float64)
    seen = [str(c) for c in logistic.classes_]
    if coef.shape[0] == 1:  # binary: p(seen[1]) = sigmoid
        if list(classes) != seen:
            raise ValueError("A binary head needs its two classes in the model's order.")
        return LinearHead(coef, intercept, mean, scale, list(classes))
    weight = np.zeros((len(classes), d))
    bias = np.full(len(classes), -1e4)
    for j, c in enumerate(seen):
        i = list(classes).index(c)
        weight[i], bias[i] = coef[j], intercept[j]
    return LinearHead(weight, bias, mean, scale, list(classes))


class Classifier:
    """`Backbone` (vision.deep.backbones) + `LinearHead`, differentiable end to end (float32)."""

    def __init__(self, net: Any, head: LinearHead, device: Any = None) -> None:
        import torch

        self.net = net
        self.head = head
        self.device = device or torch.device("cpu")
        self.module = net.module.to(self.device).float().eval()
        self._w = torch.tensor(head.weight, dtype=torch.float32, device=self.device)
        self._b = torch.tensor(head.bias, dtype=torch.float32, device=self.device)
        self._mean = torch.tensor(head.mean, dtype=torch.float32, device=self.device)
        self._scale = torch.tensor(head.scale, dtype=torch.float32, device=self.device)

    @property
    def spec(self) -> Any:
        return self.net.spec

    def vectors(self, tokens: Any) -> Any:
        """The backbone's pooled vector from `forward_features`'s output (as `Backbone`)."""
        import torch

        if self.spec.kind == "vit":
            prefix = int(getattr(self.module, "num_prefix_tokens", 1))
            return torch.cat([tokens[:, 0], tokens[:, prefix:].mean(dim=1)], dim=1)
        return tokens.mean(dim=(2, 3))

    def scores_from_vectors(self, v: Any) -> Any:
        """Class scores (logits) whose softmax is the head's probabilities (R-VEX-06)."""
        import torch

        z = (v - self._mean) / self._scale
        logits = z @ self._w.T + self._b
        if self._w.shape[0] == 1:
            half = logits[:, 0] / 2.0
            return torch.stack([-half, half], dim=1)
        return logits

    def proba_from_vectors(self, v: Any) -> Any:
        import torch

        return torch.softmax(self.scores_from_vectors(v), dim=1)

    def scores(self, x: Any) -> Any:
        return self.scores_from_vectors(self.vectors(self.module.forward_features(x)))

    def proba(self, x: Any) -> Any:
        """`(n, 3, side, side)` normalised -> `(n, classes)` probabilities."""
        return self.proba_from_vectors(self.vectors(self.module.forward_features(x)))

    def tensor(self, image: Image.Image) -> Any:
        import torch

        from quantilo.vision.deep.embed import preprocess

        return torch.from_numpy(preprocess(image, self.spec)[None]).to(self.device)

    def predict(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Probabilities of PIL images (occlusion with a deep model)."""
        import torch

        from quantilo.vision.deep.embed import preprocess

        out = []
        with torch.no_grad():
            for start in range(0, len(images), BATCH):
                chunk = [preprocess(i, self.spec) for i in images[start : start + BATCH]]
                x = torch.from_numpy(np.stack(chunk)).to(self.device)
                out.append(self.scores(x).double().cpu().numpy())
        scores = np.concatenate(out)
        scores -= scores.max(axis=1, keepdims=True)
        e = np.exp(scores)  # in float64: a float32 softmax rounds 1 - 1e-9 to 1
        return e / e.sum(axis=1, keepdims=True)


def crop_box(size: tuple[int, int], spec: Any) -> tuple[float, float, float, float]:
    """The part of an image of `size` (width, height) a network reads after `embed.preprocess`
    (shorter side to side / crop, then the centre side x side), in relative coordinates."""
    w, h = size
    short = round(spec.side / spec.crop)
    scale = short / min(w, h)
    rw, rh = max(spec.side, round(w * scale)), max(spec.side, round(h * scale))
    left, top = (rw - spec.side) // 2, (rh - spec.side) // 2
    return (left / rw, top / rh, (left + spec.side) / rw, (top + spec.side) / rh)


def _target_name(clf: Classifier, target: int) -> str:
    return clf.head.classes[target]


def _post(
    method: str,
    title: str,
    values: np.ndarray,
    unit: str,
    sentence: str,
    clf: Classifier,
    image: Image.Image,
    target: int,
    started: float,
    **extra: Any,
) -> ExplanationMap:
    return ExplanationMap(
        method=method,
        title=title,
        values=np.asarray(values, dtype=np.float32),
        unit=unit,
        sentence=sentence,
        warning=POST_HOC_WARNING,
        box=crop_box(image.size, clf.spec),
        target=_target_name(clf, target),
        seconds=round(time.perf_counter() - started, 2),
        extra=extra,
    )


def _patch_grid(clf: Classifier) -> tuple[int, int]:
    grid = getattr(getattr(clf.module, "patch_embed", None), "grid_size", None)
    if grid is not None:
        return int(grid[0]), int(grid[1])
    side = int(clf.spec.side // 14)
    return side, side


def grad_cam_values(clf: Classifier, x: Any, target: int) -> np.ndarray:
    """R-VEX-06 on a normalised input `(1, 3, side, side)`."""
    import torch

    with torch.enable_grad():
        x = x.detach().clone().requires_grad_(True)  # the network's weights stay frozen
        tokens = clf.module.forward_features(x)
        tokens.retain_grad()
        score = clf.scores_from_vectors(clf.vectors(tokens))
        clf.module.zero_grad(set_to_none=True)
        score[0, target].backward()
        grad, act = tokens.grad, tokens.detach()
    assert grad is not None
    if clf.spec.kind == "vit":
        prefix = int(getattr(clf.module, "num_prefix_tokens", 1))
        rows, cols = _patch_grid(clf)
        a = act[0, prefix:].reshape(rows, cols, -1).permute(2, 0, 1)
        g = grad[0, prefix:].reshape(rows, cols, -1).permute(2, 0, 1)
    else:
        a, g = act[0], grad[0]
    weights = g.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * a).sum(dim=0))
    return cam.float().cpu().numpy()


def grad_cam(clf: Classifier, image: Image.Image, target: int) -> ExplanationMap:
    started = time.perf_counter()
    values = grad_cam_values(clf, clf.tensor(image), target)
    where = "patch tokens" if clf.spec.kind == "vit" else "last feature map"
    rows, cols = values.shape
    sentence = (
        f"Grad-CAM for {_target_name(clf, target)} on the {where} of {clf.spec.title} "
        f"({rows} x {cols} cells, R-VEX-06)."
    )
    return _post(
        "grad_cam", "Grad-CAM", values, "weighted activation", sentence, clf, image, target, started
    )


def _gradients(clf: Classifier, x: Any, target: int) -> tuple[Any, Any]:
    """Gradient of the target score (logit) w.r.t. each input of the batch `x`, and the
    scores."""
    import torch

    with torch.enable_grad():
        x = x.detach().clone().requires_grad_(True)
        score = clf.scores(x)
        clf.module.zero_grad(set_to_none=True)
        score[:, target].sum().backward()
    assert x.grad is not None
    return x.grad.detach(), score.detach()


def saliency_values(clf: Classifier, x: Any, target: int) -> np.ndarray:
    grad, _ = _gradients(clf, x, target)
    return grad[0].abs().amax(dim=0).float().cpu().numpy()


def saliency(clf: Classifier, image: Image.Image, target: int) -> ExplanationMap:
    started = time.perf_counter()
    values = saliency_values(clf, clf.tensor(image), target)
    sentence = (
        f"Vanilla saliency for {_target_name(clf, target)}: how much each pixel's change moves "
        "the probability, largest over the colours (R-VEX-08)."
    )
    return _post(
        "saliency", "Saliency", values, "|gradient|", sentence, clf, image, target, started
    )


def smoothgrad(
    clf: Classifier,
    image: Image.Image,
    target: int,
    *,
    samples: int = SMOOTH_SAMPLES,
    noise: float = SMOOTH_NOISE,
    seed: int = 0,
) -> ExplanationMap:
    import torch

    started = time.perf_counter()
    x = clf.tensor(image)
    sigma = float(noise * (x.max() - x.min()))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total = torch.zeros_like(x[0])
    done = 0
    while done < samples:
        n = min(BATCH, samples - done)
        eps = torch.randn((n, *x.shape[1:]), generator=generator).to(x.device) * sigma
        grad, _ = _gradients(clf, x.repeat(n, 1, 1, 1) + eps, target)
        total += grad.abs().sum(dim=0)
        done += n
    values = (total / samples).amax(dim=0).float().cpu().numpy()
    sentence = (
        f"SmoothGrad for {_target_name(clf, target)}: saliency averaged over {samples} noisy "
        f"copies (noise {noise:.0%} of the range, R-VEX-08)."
    )
    return _post(
        "smoothgrad",
        "SmoothGrad",
        values,
        "|gradient|",
        sentence,
        clf,
        image,
        target,
        started,
        samples=samples,
    )


def baseline_of(clf: Classifier, image: Image.Image, radius: float = IG_BLUR) -> Any:
    """R-VEX-07: the blurred reference, blurred at the network's own scale."""
    import torch

    from quantilo.vision.deep.embed import preprocess

    spec = clf.spec
    w, h = image.size
    factor = min(w, h) / (spec.side / spec.crop)
    soft = blurred(image.convert("RGB"), radius * factor)
    return torch.from_numpy(preprocess(soft, spec)[None]).to(clf.device)


def integrated_gradients_values(
    clf: Classifier, x: Any, baseline: Any, target: int, steps: int = IG_STEPS
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Attributions summed |.| over channels, their sum, and the score and probability of the
    image and of the baseline (`score`, `base_score`, `probability`, `base_probability`)."""
    import torch

    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
    weights = torch.ones(steps + 1, device=x.device)
    weights[0] = weights[-1] = 0.5  # trapezoids
    total = torch.zeros_like(x[0])
    ends: dict[str, float] = {}
    for start in range(0, steps + 1, BATCH):
        a = alphas[start : start + BATCH]
        path = baseline + a.view(-1, 1, 1, 1) * (x - baseline)
        grad, score = _gradients(clf, path, target)
        total += (grad * weights[start : start + len(a)].view(-1, 1, 1, 1)).sum(dim=0)
        p = torch.softmax(score, dim=1)
        for j, alpha in enumerate(a.tolist()):
            if alpha in (0.0, 1.0):
                prefix = "" if alpha == 1.0 else "base_"
                ends[f"{prefix}score"] = float(score[j, target])
                ends[f"{prefix}probability"] = float(p[j, target])
    attributions = (x[0] - baseline[0]) * total / steps
    values = attributions.abs().sum(dim=0).float().cpu().numpy()
    return values, float(attributions.sum()), ends


def integrated_gradients(
    clf: Classifier, image: Image.Image, target: int, *, steps: int = IG_STEPS
) -> ExplanationMap:
    started = time.perf_counter()
    x = clf.tensor(image)
    values, total, ends = integrated_gradients_values(
        clf, x, baseline_of(clf, image), target, steps
    )
    p_image, p_base = ends["probability"], ends["base_probability"]
    change = ends["score"] - ends["base_score"]
    gap = abs(total - change) / max(abs(change), 1e-6)
    sentence = (
        f"Integrated Gradients for {_target_name(clf, target)}: from a blurred copy "
        f"(probability {p_base:.2f}) to the image ({p_image:.2f}) in {steps} steps (R-VEX-07)."
    )
    notes = [
        f"Completeness: the attributions add up to {total:.2f} for a change of score of "
        f"{change:.2f} ({gap:.0%} apart; more steps bring them closer)."
    ]
    out = _post(
        "integrated_gradients",
        "Integrated Gradients",
        values,
        "|attribution|",
        sentence,
        clf,
        image,
        target,
        started,
        completeness_gap=gap,
        probability=p_image,
        baseline_probability=p_base,
    )
    out.notes = notes
    return out


def attentions_of(clf: Classifier, x: Any) -> list[np.ndarray]:
    """Each block's `(heads, tokens, tokens)` attention for one input, recomputed from the
    block's input (timm's fused attention does not keep it)."""
    import torch

    blocks = getattr(clf.module, "blocks", None)
    if blocks is None or clf.spec.kind != "vit":
        raise ValueError("Attention rollout needs a vision transformer (DINOv2, ViT).")
    found: list[np.ndarray] = []

    def hook(module: Any, args: tuple[Any, ...], _output: Any) -> None:
        inp = args[0]
        b, n, _c = inp.shape
        qkv = module.qkv(inp).reshape(b, n, 3, module.num_heads, module.head_dim)
        q, k, _v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)
        attn = (q * module.scale) @ k.transpose(-2, -1)
        found.append(attn.softmax(dim=-1)[0].float().cpu().numpy())

    handles = [block.attn.register_forward_hook(hook) for block in blocks]
    try:
        with torch.no_grad():
            clf.module.forward_features(x)
    finally:
        for handle in handles:
            handle.remove()
    return found


def attention_rollout(clf: Classifier, image: Image.Image) -> ExplanationMap:
    started = time.perf_counter()
    found = attentions_of(clf, clf.tensor(image))
    prefix = int(getattr(clf.module, "num_prefix_tokens", 1))
    values = rollout(found, prefix, _patch_grid(clf))
    sentence = (
        f"Attention rollout over the {len(found)} blocks of {clf.spec.title}: where the class "
        "token gathered information (R-VEX-09)."
    )
    return ExplanationMap(
        method="rollout",
        title="Attention rollout",
        values=values,
        unit="rolled-out attention",
        sentence=sentence,
        warning=f"{ATTENTION_WARNING} {POST_HOC_WARNING}",
        box=crop_box(image.size, clf.spec),
        target=None,
        seconds=round(time.perf_counter() - started, 2),
    )


# ---------------------------------------------------------------------------------------------
# R-VEX-10: sanity test by cascading randomisation


@dataclass
class SanityStage:
    label: str
    values: np.ndarray
    correlation: float  # Spearman with the trained model's map (1 for the trained stage)


@dataclass
class SanityResult:
    method: str
    title: str
    stages: list[SanityStage]
    passes: bool
    sentence: str
    seconds: float
    box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    target: str | None = None
    notes: list[str] = field(default_factory=list)


SANITY_METHODS = ("grad_cam", "saliency", "integrated_gradients")


def _top_blocks(module: Any, kind: str) -> list[tuple[str, Any]]:
    """The network's blocks from the output down (a ViT's `blocks`, a ResNet's stages)."""
    if kind == "vit" and hasattr(module, "blocks"):
        blocks = list(module.blocks)
        return [(f"block {i + 1}", blocks[i]) for i in range(len(blocks) - 1, -1, -1)]
    names = [n for n in ("layer4", "layer3", "layer2", "layer1") if hasattr(module, n)]
    return [(n.replace("layer", "stage "), getattr(module, n)) for n in names]


def _randomise(module: Any, generator: Any) -> None:
    """Weights (2-D and more) drawn again with their own standard deviation; norms and biases
    kept (Adebayo et al. re-initialise the layer)."""
    import torch

    with torch.no_grad():
        for p in module.parameters():
            if p.ndim >= 2:
                std = float(p.float().std()) or 0.02
                p.copy_(torch.randn(p.shape, generator=generator).to(p) * std)


def _map_for(method: str, clf: Classifier, image: Image.Image, target: int) -> np.ndarray:
    x = clf.tensor(image)
    if method == "grad_cam":
        return grad_cam_values(clf, x, target)
    if method == "saliency":
        return pool(saliency_values(clf, x, target), 28)
    if method == "integrated_gradients":
        values, *_ = integrated_gradients_values(clf, x, baseline_of(clf, image), target, 16)
        return pool(values, 28)
    raise ValueError(f"No sanity test for {method!r}: one of {', '.join(SANITY_METHODS)}.")


def sanity_check(
    clf: Classifier,
    image: Image.Image,
    target: int,
    method: str = "grad_cam",
    *,
    blocks: int = SANITY_BLOCKS,
    seed: int = 0,
) -> SanityResult:
    """R-VEX-10 on a copy of the network (the classifier given is never changed)."""
    import copy

    import torch

    started = time.perf_counter()
    trained = _map_for(method, clf, image, target)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = copy.deepcopy(clf.head)
    head.weight = rng.normal(0.0, float(np.std(head.weight)) or 0.02, head.weight.shape)
    net = copy.copy(clf.net)
    net.module = copy.deepcopy(clf.net.module)
    shaken = Classifier(net, head, clf.device)
    stages = [SanityStage("trained model", trained, 1.0)]
    values = _map_for(method, shaken, image, target)
    stages.append(SanityStage("random head", values, spearman(trained, values)))
    for label, block in _top_blocks(shaken.module, clf.spec.kind)[:blocks]:
        _randomise(block, generator)
        values = _map_for(method, shaken, image, target)
        stages.append(SanityStage(f"+ random {label}", values, spearman(trained, values)))
    del shaken, net
    from quantilo.vision.deep.anomaly import free_gpu

    free_gpu()
    notes = []
    for stage in stages[1:]:
        if not math.isfinite(stage.correlation):
            stage.correlation = 0.0
            notes.append(f"With {stage.label.removeprefix('+ ')}, the map is flat: it changed.")
    check = stages[2] if len(stages) > 2 else stages[1]
    rho = check.correlation
    passes = rho < SANITY_FLAG
    title = {
        "grad_cam": "Grad-CAM",
        "saliency": "Saliency",
        "integrated_gradients": "Integrated Gradients",
    }[method]
    if passes:
        sentence = (
            f"{title} passes: with a random head and a random last block the map changes "
            f"(rank correlation {rho:.2f} < {SANITY_FLAG}), so it depends on what the model "
            "learnt (R-VEX-10)."
        )
    else:
        sentence = (
            f"{title} fails: with a random head and a random last block the map barely changes "
            f"(rank correlation {rho:.2f} >= {SANITY_FLAG}). If the map barely changes with "
            "random weights, do not trust it (R-VEX-10)."
        )
    return SanityResult(
        method=method,
        title=title,
        stages=stages,
        passes=passes,
        sentence=sentence,
        seconds=round(time.perf_counter() - started, 2),
        box=crop_box(image.size, clf.spec),
        target=_target_name(clf, target),
        notes=notes,
    )


GRADIENT_METHODS = {
    "grad_cam": grad_cam,
    "saliency": saliency,
    "smoothgrad": smoothgrad,
    "integrated_gradients": integrated_gradients,
}


__all__ = [
    "ATTENTION_WARNING",
    "GRADIENT_METHODS",
    "OCCLUSION_WARNING",
    "POST_HOC_WARNING",
    "SANITY_METHODS",
    "Classifier",
    "ExplanationMap",
    "LinearHead",
    "SanityResult",
    "SanityStage",
    "attention_rollout",
    "blurred",
    "crop_box",
    "grad_cam",
    "head_from_pipeline",
    "integrated_gradients",
    "occlusion",
    "png_gray",
    "pool",
    "rollout",
    "saliency",
    "sanity_check",
    "smoothgrad",
    "spearman",
    "to_uint8",
]
