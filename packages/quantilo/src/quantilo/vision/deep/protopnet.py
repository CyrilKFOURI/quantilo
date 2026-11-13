"""ProtoPNet-lite: "this looks like that" on a frozen backbone (118; study §§ 6.4, 13.2).

Chen, Li, Tao, Barnett, Su and Rudin (2019), *This Looks Like That: Deep Learning for
Interpretable Image Recognition* (Duke, Rudin's lab): a model interpretable by construction. Each
class owns prototypes, each a patch of a real training image; an image is scored by how much
its patches look like each prototype, and the class score is a sum of those similarities. The
explanation is the computation: "this patch looks like that prototype (from image X) of class
Y, similarity s, worth w x s points".

Lite, because the backbone stays frozen (the study's choice for a 6 GB card) and only the
prototype part learns, on patch features computed once:

- R-VPP-01 patches: the patch tokens of DINOv2 ViT-S/14 (16 x 16 cells of 14 px over the centre
  square the probe reads), or a CNN's last feature map. Above `FEATURE_BUDGET_MB` the grid is
  pooled 2 x 2 (said).
- R-VPP-02 the head (Chen et al. 2019): two 1 x 1 layers (ReLU, then sigmoid) to a 128-dim
  latent; 10 prototypes per class; squared L2 distance of every patch to every prototype, its
  minimum over the patches; similarity log((d + 1) / (d + 1e-4)); the class scores are a
  linear layer on the similarities, started at +1 for a class's own prototypes and -0.5 for the
  others.
- R-VPP-03 stage 1, the prototypes learn (the last layer fixed): cross-entropy + 0.8 x cluster
  cost (each image's distance to the nearest prototype of its class, pulled down) - 0.08 x
  separation cost (the distance to the nearest prototype of another class, pushed up); Adam
  3e-3; early stopping on 15 % of the training images held out (grouped by copies), patience 4,
  at most 30 epochs.
- R-VPP-04 push: each prototype is replaced by the latent of the nearest patch of a training
  image of its own class; from then on it *is* that patch, and the image and the place are
  kept to be shown. Prototypes that land on the same patch are counted (said).
- R-VPP-05 stage 2, the last layer only (convex): cross-entropy + 1e-4 x L1 on the connections
  of each prototype to the other classes, so a class leans on its own prototypes; 40 epochs.
- R-VPP-06 explanation of an image: for the predicted class, each prototype's similarity x its
  weight, the three largest; the cell of the image closest to the prototype and the cell of the
  training image it was pushed onto, as boxes in each image.

torch is imported inside the functions only (extra `vision`, R-VDP-01).
"""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quantilo.vision.deep import finetune as ft
from quantilo.vision.deep.backbones import Backbone, BackboneSpec

PER_CLASS = 10  # R-VPP-02
LATENT = 128
EPSILON = 1e-4
CLUSTER = 0.8  # R-VPP-03
SEPARATION = 0.08
LR = 3e-3
MAX_EPOCHS = 30
PATIENCE = 4
L1 = 1e-4  # R-VPP-05
LAST_EPOCHS = 40
LAST_LR = 1e-2
BATCH = 128  # images of patch features per step (they are small)
FEATURE_BUDGET_MB = 1_024  # R-VPP-01
TOP = 3  # R-VPP-06

Progress = Callable[[float, str], None]
Cancel = Callable[[], bool]


@dataclass
class PatchFeatures:
    features: np.ndarray  # (n, cells, d) float16, rows of `rows`
    rows: np.ndarray  # table rows
    grid: tuple[int, int]  # cells per column, per row
    pooled: bool  # R-VPP-01: 2 x 2 pooled to fit the budget
    seconds: float
    device: str

    def take(self, rows: Sequence[int] | np.ndarray) -> np.ndarray:
        where = {int(r): i for i, r in enumerate(self.rows.tolist())}
        return self.features[[where[int(r)] for r in rows]]


def patch_features(
    net: Backbone,
    bank: ft.ImageBank,
    rows: Sequence[int] | np.ndarray,
    *,
    allow_gpu: bool = True,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
) -> PatchFeatures:
    """R-VPP-01: the frozen network's patch features of `rows` (the probe's own view)."""
    import torch

    from quantilo.synthetic.deep.device import pick

    spec = net.spec
    module = net.module
    device = pick("auto", allow_gpu)
    on_gpu = device.device.type == "cuda"
    size = 64 if on_gpu else 16
    rows_arr = np.asarray(rows, dtype=int)
    started = time.perf_counter()
    module.to(device.device).float().eval()
    parts: list[np.ndarray] = []
    grid: tuple[int, int] = (0, 0)
    try:
        with torch.inference_mode():
            for start in range(0, len(rows_arr), size):
                if cancel is not None and cancel():
                    raise ft.Cancelled("Stopped: nothing was saved.")
                chunk = rows_arr[start : start + size]
                x = ft.normalise(
                    ft.eval_view(ft.to_input(bank.pixels(chunk), spec, device.device), spec), spec
                )
                with torch.autocast(device.device.type, dtype=torch.float16, enabled=on_gpu):
                    tokens = module.forward_features(x)
                cells, grid = _cells(module, spec, tokens)
                parts.append(cells.half().cpu().numpy())
                if progress is not None:
                    done = min(len(rows_arr), start + size)
                    progress(done / max(1, len(rows_arr)), f"Patches: {done:,} / {len(rows_arr):,}")
    finally:
        module.to("cpu").float()
        if on_gpu:
            torch.cuda.empty_cache()
    features = np.concatenate(parts) if parts else np.zeros((0, 1, spec.dims // 2 or 1), np.float16)
    pooled = False
    if features.nbytes > FEATURE_BUDGET_MB * 1024 * 1024 and grid[0] >= 4 and grid[1] >= 4:
        features, grid = _pool2(features, grid)
        pooled = True
    return PatchFeatures(
        features, rows_arr, grid, pooled, round(time.perf_counter() - started, 2), device.label
    )


def _cells(module: Any, spec: BackboneSpec, tokens: Any) -> tuple[Any, tuple[int, int]]:
    """`(n, cells, d)` and the grid, from `forward_features`."""
    if spec.kind == "vit":
        prefix = int(getattr(module, "num_prefix_tokens", 1))
        patches = tokens[:, prefix:]
        grid = getattr(getattr(module, "patch_embed", None), "grid_size", None)
        if grid is None:
            side = round(math.sqrt(patches.shape[1]))
            grid = (side, side)
        return patches, (int(grid[0]), int(grid[1]))
    _, _, h, w = tokens.shape
    return tokens.flatten(2).transpose(1, 2), (int(h), int(w))


def _pool2(features: np.ndarray, grid: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    n, _, d = features.shape
    h, w = grid
    h2, w2 = h // 2, w // 2
    x = features[:, : h * w].reshape(n, h, w, d)[:, : 2 * h2, : 2 * w2]
    x = x.reshape(n, h2, 2, w2, 2, d).astype(np.float32).mean(axis=(2, 4))
    return x.reshape(n, h2 * w2, d).astype(np.float16), (h2, w2)


def _build(d: int, n_classes: int, per_class: int, seed: int) -> Any:
    import torch
    from torch import nn

    class Head(nn.Module):
        """R-VPP-02."""

        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(seed)
            m = n_classes * per_class
            self.add_on = nn.Sequential(
                nn.Linear(d, LATENT), nn.ReLU(), nn.Linear(LATENT, LATENT), nn.Sigmoid()
            )
            self.prototypes = nn.Parameter(torch.rand(m, LATENT))
            identity = torch.zeros(m, n_classes)
            identity[torch.arange(m), torch.arange(m) // per_class] = 1.0
            self.register_buffer("identity", identity)
            self.last = nn.Linear(m, n_classes, bias=False)
            with torch.no_grad():
                self.last.weight.copy_(identity.T * 1.0 + (1 - identity.T) * -0.5)

        def distances(self, feats: Any) -> Any:
            """`(b, cells, d)` -> squared L2 `(b, cells, m)` in the latent."""
            z = self.add_on(feats)
            p = self.prototypes
            d2 = (z * z).sum(-1, keepdim=True) - 2 * z @ p.T + (p * p).sum(-1)[None, None, :]
            return d2.clamp_min(0.0)

        def similarities(self, distances: Any) -> Any:
            return torch.log((distances + 1.0) / (distances + EPSILON))

        def forward(self, feats: Any) -> tuple[Any, Any]:
            d = self.distances(feats)
            nearest = d.min(dim=1).values  # (b, m)
            return self.last(self.similarities(nearest)), nearest

    return Head()


@dataclass
class Prototype:
    index: int
    label: str
    row: int  # the training image it was pushed onto (table row)
    cell: int  # its cell in that image's grid
    distance: float  # latent distance before the push (0 after)
    weight: float  # its connection to its own class after stage 2
    duplicate_of: int | None = None


@dataclass
class Contribution:
    prototype: int
    similarity: float
    weight: float
    points: float  # similarity x weight
    cell: int  # the cell of the explained image closest to the prototype


@dataclass
class ProtoResult:
    classes: list[str]
    grid: tuple[int, int]
    prototypes: list[Prototype]
    log: ft.TrainingLog
    notes: list[str] = field(default_factory=list)


class ProtoPNetLite:
    """The fitted head: probabilities and "this looks like that" for patch features."""

    def __init__(self, head: Any, classes: list[str], grid: tuple[int, int], device: Any) -> None:
        self.head = head
        self.classes = classes
        self.grid = grid
        self.device = device

    def _run(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(scores (n, k), similarities (n, m), nearest cell (n, m))."""
        import torch

        scores, sims, cells = [], [], []
        with torch.inference_mode():
            for start in range(0, len(feats), BATCH):
                x = torch.from_numpy(feats[start : start + BATCH].astype(np.float32)).to(
                    self.device
                )
                d = self.head.distances(x)
                nearest, where = d.min(dim=1)
                s = self.head.similarities(nearest)
                scores.append(self.head.last(s).double().cpu().numpy())
                sims.append(s.double().cpu().numpy())
                cells.append(where.cpu().numpy())
        k, m = len(self.classes), int(self.head.prototypes.shape[0])
        if not scores:
            return np.zeros((0, k)), np.zeros((0, m)), np.zeros((0, m), dtype=int)
        return np.concatenate(scores), np.concatenate(sims), np.concatenate(cells)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        scores, _, _ = self._run(feats)
        scores -= scores.max(axis=1, keepdims=True)
        e = np.exp(scores)
        return e / e.sum(axis=1, keepdims=True)

    def explain(
        self, feats: np.ndarray, target: Sequence[int] | None = None, top: int = TOP
    ) -> tuple[np.ndarray, list[list[Contribution]]]:
        """R-VPP-06: probabilities, and for each image the `top` prototypes that add the most
        to the class `target[i]` (default: the predicted one)."""
        scores, sims, cells = self._run(feats)
        weights = self.head.last.weight.detach().double().cpu().numpy()  # (k, m)
        z = scores - scores.max(axis=1, keepdims=True)
        proba = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        out = []
        for i in range(len(feats)):
            k = int(np.argmax(proba[i])) if target is None else int(target[i])
            points = sims[i] * weights[k]
            order = np.argsort(-points, kind="mergesort")[:top]
            out.append(
                [
                    Contribution(
                        int(j),
                        round(float(sims[i, j]), 4),
                        round(float(weights[k, j]), 4),
                        round(float(points[j]), 4),
                        int(cells[i, j]),
                    )
                    for j in order
                ]
            )
        return proba, out

    def release(self) -> None:
        import torch

        self.head.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def fit_protopnet(
    features: PatchFeatures,
    train_rows: Sequence[int] | np.ndarray,
    labels: Sequence[str] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    classes: Sequence[str],
    *,
    allow_gpu: bool = True,
    seed: int = 0,
    per_class: int = PER_CLASS,
    max_epochs: int = MAX_EPOCHS,
    last_epochs: int = LAST_EPOCHS,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
) -> tuple[ProtoPNetLite, ProtoResult]:
    """R-VPP-02 to 05 on the training rows' patch features (`labels` aligned with them)."""
    import torch

    from quantilo.synthetic.deep.device import pick

    classes = [str(c) for c in classes]
    index = {c: i for i, c in enumerate(classes)}
    rows = np.asarray(train_rows, dtype=int)
    codes = np.asarray([index[str(v)] for v in labels], dtype=int)
    group_arr = np.asarray([str(g) for g in groups], dtype=object)
    feats = features.take(rows)
    k = len(classes)
    if len(np.unique(codes)) < 2:
        raise ValueError("ProtoPNet-lite needs at least 2 classes among the training images.")
    device = pick("auto", allow_gpu)
    head = _build(feats.shape[2], k, per_class, seed).to(device.device)
    identity = head.identity  # (m, k)
    held = ft.validation_rows(codes, group_arr, seed)
    learn = np.setdiff1d(np.arange(len(rows)), held)
    notes: list[str] = []
    if len(held):
        notes.append(
            f"{len(held):,} of the {len(rows):,} training images held out for early stopping, "
            "grouped by copies (R-VPP-03)."
        )
    generator = np.random.default_rng(seed)
    started = time.perf_counter()
    history: list[ft.EpochLog] = []

    def tensor(i: np.ndarray) -> Any:
        return torch.from_numpy(feats[i].astype(np.float32)).to(device.device)

    def costs(logits: Any, nearest: Any, target: Any) -> Any:
        own = identity[:, target].T  # (b, m): 1 for the prototypes of the image's class
        big = nearest.detach().max() + 1.0
        cluster = (nearest + (1 - own) * big).min(dim=1).values.mean()
        separation = (nearest + own * big).min(dim=1).values.mean()
        ce = torch.nn.functional.cross_entropy(logits, target)
        return ce + CLUSTER * cluster - SEPARATION * separation, ce

    # R-VPP-03: the prototypes and the add-on learn, the last layer is fixed
    head.last.weight.requires_grad_(False)
    optimiser = torch.optim.Adam(
        [{"params": head.add_on.parameters()}, {"params": [head.prototypes]}], lr=LR
    )
    best, best_epoch, waited = math.inf, 0, 0
    best_state = {name: v.detach().clone() for name, v in head.state_dict().items()}
    stopped = ""
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        if cancel is not None and cancel():
            raise ft.Cancelled("Stopped: nothing was saved.")
        t0 = time.perf_counter()
        head.train()
        order = generator.permutation(learn)
        total = 0.0
        for start in range(0, len(order), BATCH):
            chunk = order[start : start + BATCH]
            target = torch.from_numpy(codes[chunk]).to(device.device)
            logits, nearest = head(tensor(chunk))
            loss, _ = costs(logits, nearest, target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(chunk)
        val_loss = val_bacc = None
        if len(held):
            head.eval()
            with torch.no_grad():
                logits, _ = head(tensor(held))
                target = torch.from_numpy(codes[held]).to(device.device)
                val_loss = float(torch.nn.functional.cross_entropy(logits, target))
                val_bacc = ft._balanced(codes[held], logits.argmax(dim=1).cpu().numpy())
        history.append(
            ft.EpochLog(
                epoch,
                round(total / max(1, len(order)), 5),
                None if val_loss is None else round(val_loss, 5),
                None if val_bacc is None else round(val_bacc, 5),
                round(time.perf_counter() - t0, 3),
            )
        )
        if progress is not None:
            progress(0.7 * epoch / max_epochs, f"Prototypes: epoch {epoch} / {max_epochs}")
        if val_loss is None:
            continue
        if val_loss < best - 1e-4:
            best, best_epoch, waited = val_loss, epoch, 0
            best_state = {name: v.detach().clone() for name, v in head.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                stopped = (
                    f"Stage 1 stopped after epoch {epoch}: the held-out loss had not improved for "
                    f"{PATIENCE} epochs; epoch {best_epoch} kept (R-VPP-03)."
                )
                break
    if len(held):
        head.load_state_dict(best_state)
    else:
        best_epoch = epoch
    if not stopped:
        stopped = f"Stage 1 ran {epoch} epochs; epoch {best_epoch} kept (R-VPP-03)."

    # R-VPP-04: push every prototype onto the nearest patch of a training image of its class
    head.eval()
    m = int(head.prototypes.shape[0])
    found_d = np.full(m, np.inf)
    found_row = np.zeros(m, dtype=int)
    found_cell = np.zeros(m, dtype=int)
    found_z = np.zeros((m, LATENT), dtype=np.float32)
    proto_class = np.arange(m) // per_class
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = np.arange(start, min(len(rows), start + BATCH))
            z = head.add_on(tensor(chunk))  # (b, cells, L)
            p = head.prototypes
            d2 = ((z * z).sum(-1, keepdim=True) - 2 * z @ p.T + (p * p).sum(-1)).clamp_min(0)
            d2 = d2.cpu().numpy()  # (b, cells, m)
            zc = z.cpu().numpy()
            for j in range(m):
                mine = np.flatnonzero(codes[chunk] == proto_class[j])
                if not len(mine):
                    continue
                block = d2[mine, :, j]
                flat = int(np.argmin(block))
                b, cell = divmod(flat, block.shape[1])
                if block[b, cell] < found_d[j]:
                    found_d[j] = float(block[b, cell])
                    found_row[j] = int(rows[chunk[mine[b]]])
                    found_cell[j] = cell
                    found_z[j] = zc[mine[b], cell]
        head.prototypes.copy_(torch.from_numpy(found_z).to(device.device))

    # R-VPP-05: the last layer alone, on similarities that no longer change
    head.last.weight.requires_grad_(True)
    with torch.no_grad():
        sims = []
        for start in range(0, len(rows), BATCH):
            chunk = np.arange(start, min(len(rows), start + BATCH))
            _, nearest = head(tensor(chunk))
            sims.append(head.similarities(nearest))
        s_all = torch.cat(sims)
    target_all = torch.from_numpy(codes).to(device.device)
    last_opt = torch.optim.Adam([head.last.weight], lr=LAST_LR)
    mask = (1 - identity.T).to(device.device)  # (k, m): connections to other classes
    for epoch2 in range(1, last_epochs + 1):
        if cancel is not None and cancel():
            raise ft.Cancelled("Stopped: nothing was saved.")
        order = generator.permutation(len(rows))
        for start in range(0, len(order), BATCH):
            chunk = torch.from_numpy(order[start : start + BATCH]).to(device.device)
            logits = head.last(s_all[chunk])
            loss = torch.nn.functional.cross_entropy(logits, target_all[chunk])
            loss = loss + L1 * (head.last.weight * mask).abs().sum()
            last_opt.zero_grad(set_to_none=True)
            loss.backward()
            last_opt.step()
        if progress is not None:
            progress(
                0.7 + 0.3 * epoch2 / last_epochs, f"Last layer: epoch {epoch2} / {last_epochs}"
            )
    head.last.weight.requires_grad_(False)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    weights = head.last.weight.detach().double().cpu().numpy()
    seen: dict[tuple[int, int], int] = {}
    prototypes = []
    for j in range(m):
        key = (int(found_row[j]), int(found_cell[j]))
        first = seen.setdefault(key, j)
        prototypes.append(
            Prototype(
                index=j,
                label=classes[int(proto_class[j])],
                row=int(found_row[j]),
                cell=int(found_cell[j]),
                distance=round(float(found_d[j]), 5),
                weight=round(float(weights[int(proto_class[j]), j]), 4),
                duplicate_of=None if first == j else first,
            )
        )
    duplicates = sum(p.duplicate_of is not None for p in prototypes)
    if duplicates:
        notes.append(
            f"{duplicates} {'prototype' if duplicates == 1 else 'prototypes'} landed on the same "
            "patch as another one after the push: "
            f"{m - duplicates} distinct patches (R-VPP-04)."
        )
    log = ft.TrainingLog(
        device=device.label,
        batch=BATCH,
        precision="fp32 (patch features fp16)",
        epochs=len(history),
        best_epoch=best_epoch,
        stopped=stopped,
        history=history,
        seconds=round(time.perf_counter() - started, 2),
        trained=(
            f"{per_class} prototypes per class ({m}) on the frozen patch features, a "
            f"{LATENT}-dim latent, then the last layer alone (R-VPP-02 to 05)"
        ),
        notes=list(notes),
    )
    model = ProtoPNetLite(head, classes, features.grid, device.device)
    return model, ProtoResult(classes, features.grid, prototypes, log, notes)


def cell_box(
    cell: int, grid: tuple[int, int], size: tuple[int, int], spec: BackboneSpec
) -> tuple[float, float, float, float]:
    """R-VPP-06: a cell of the grid as a box of the original image (relative x0, y0, x1, y1):
    the network reads the centre square (side / crop), the grid covers its centre `crop`."""
    w, h = size
    s = min(w, h)
    keep = s * spec.crop
    left = (w - keep) / 2
    top = (h - keep) / 2
    rows, cols = grid
    r, c = divmod(int(cell), cols)
    x0 = left + keep * c / cols
    y0 = top + keep * r / rows
    return (x0 / w, y0 / h, (x0 + keep / cols) / w, (y0 + keep / rows) / h)


__all__ = [
    "Contribution",
    "PatchFeatures",
    "ProtoPNetLite",
    "ProtoResult",
    "Prototype",
    "cell_box",
    "fit_protopnet",
    "patch_features",
]
