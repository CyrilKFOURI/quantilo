"""Shared pieces of the deep anomaly detectors (115, 116; study §§ 6.3, 7.2, 7.3, 8, 10).

- R-VAD-03: the whole image is decoded safely (103) and brought to `side x side` (256 by
  default), without a crop, so the anomaly map covers the whole picture; `full_map(i)` gives it
  back at the image's own size.
- `AnomalyResult`: one score per image (higher = more unusual) and one map per image, smoothed by
  a Gaussian (sigma = 4 px), plus failures, notes, device and time.
- `Runner`: a frozen network on its budget (R-VDP-04, R-VDP-05: GPU in fp16 or CPU, batch halved
  on out-of-memory, estimate on the first batch, confirmation beyond 15 minutes).
- R-VAD-01, R-VAD-02 (`split_image_set`): which images of an imported set fit the model, which
  are scored, which normals are held out to set the threshold.
- `anomaly_image_set`: fit on the normal training images, score the test images, every metric
  of § 7.2 through `vision.metrics` (111).

torch is imported inside the functions only (R-VDP-01)."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from PIL import Image

from quantilo.vision import metrics as vm
from quantilo.vision.decode import DecodedImage, DecodeFailure, try_decode
from quantilo.vision.deep import budget as budgets
from quantilo.vision.deep.backbones import IMAGENET_MEAN, IMAGENET_STD, Backbone, BackboneSpec
from quantilo.vision.deep.backbones import load as load_backbone

if TYPE_CHECKING:
    from quantilo.vision.ingest import ImageSet

Source = str | Path | bytes | Image.Image | DecodedImage
Method = Literal["patchcore", "padim", "autoencoder", "vae"]

SIDE = 256  # R-VAD-03
SIGMA = 4.0  # Gaussian smoothing of the maps, in pixels of the 256 px image (§ 6.3)
HOLDOUT_SHARE = 0.10  # R-VAD-02 = R-VTH-02: normals kept out of the model to set the threshold
TEST_SHARE = 0.20  # R-VAD-01 without the producer's split
SEED = 0
NORMAL = "good"  # the defect type of a normal image in `auroc_by_type`
TRAIN_SPLITS = ("train", "val", "validation", "dev")
TEST_SPLITS = ("test",)


class Cancelled(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# Images


def decode_rgb(source: Source, side: int = SIDE) -> tuple[np.ndarray, tuple[int, int]] | str:
    """`(side, side, 3)` uint8 pixels and the image's own `(width, height)`, or the reason it was
    not decoded (R-VAD-03: the whole image, squeezed to a square, bilinear)."""
    if isinstance(source, DecodedImage):
        image, size = source.image, (source.width, source.height)
    elif isinstance(source, Image.Image):
        image, size = source, source.size
    else:
        decoded = try_decode(source, reduce_to=side)
        if isinstance(decoded, DecodeFailure):
            return f"{decoded.code}: {decoded.reason}"
        image, size = decoded.image, (decoded.width, decoded.height)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (side, side):
        image = image.resize((side, side), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8), size


def load_mask(source: str | Path | bytes | Image.Image | None, side: int = SIDE) -> np.ndarray:
    """A defect mask at `side x side` (nearest neighbour), True where any value is above 0;
    all False without a mask (a normal image)."""
    if source is None:
        return np.zeros((side, side), dtype=bool)
    if isinstance(source, Image.Image):
        image = source
    else:
        decoded = try_decode(source)
        if isinstance(decoded, DecodeFailure):
            raise ValueError(f"Mask not decoded: {decoded.reason}")
        image = decoded.image
    grey = image.convert("L").resize((side, side), Image.Resampling.NEAREST)
    return np.asarray(grey) > 0


def normalise(pixels: Sequence[np.ndarray], mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """uint8 `(side, side, 3)` images -> `(n, 3, side, side)` float32 normalised as ImageNet."""
    x = np.stack(pixels).astype(np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def smooth(maps: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    """Gaussian blur of each map (the last two axes), as PatchCore and PaDiM do."""
    if sigma <= 0:
        return maps.astype(np.float32)
    from scipy.ndimage import gaussian_filter

    sigmas = (0,) * (maps.ndim - 2) + (sigma, sigma)
    return gaussian_filter(maps.astype(np.float32), sigma=sigmas, mode="reflect").astype(np.float32)


def resize_map(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A float map brought to `(width, height)`, bilinear."""
    image = Image.fromarray(np.ascontiguousarray(values, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def scale_box(
    box: tuple[float, float, float, float], side: int, size: tuple[int, int] | None
) -> tuple[int, int, int, int]:
    """A box of the `side x side` image in the pixels of the image's own size."""
    w, h = size or (side, side)
    x0, y0, x1, y1 = box
    return (
        round(x0 * w / side),
        round(y0 * h / side),
        round(x1 * w / side),
        round(y1 * h / side),
    )


def png(image: Image.Image | np.ndarray) -> bytes:
    """PNG bytes without metadata (for the before -> after previews of 110)."""
    import io

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------------------------
# Result


@dataclass
class AnomalyResult:
    """Scores and maps of the images given to `score`, in their order."""

    method: str
    title: str
    scores: np.ndarray  # (n,) float64, higher = more unusual; NaN when not decoded
    maps: np.ndarray  # (n, side, side) float32, smoothed; NaN when not decoded
    ok: np.ndarray  # (n,) bool
    sizes: list[tuple[int, int] | None]  # each image's own (width, height)
    side: int
    device: str
    seconds: float
    model_key: str
    estimate: str | None = None
    notes: list[str] = field(default_factory=list)
    failures: dict[int, str] = field(default_factory=dict)

    def full_map(self, i: int) -> np.ndarray:
        """The map of image `i` at the image's own size (bilinear)."""
        size = self.sizes[i]
        return self.maps[i] if size is None else resize_map(self.maps[i], size)

    @property
    def sentence(self) -> str:
        n = int(self.ok.sum())
        text = (
            f"{n:,} images scored by {self.title} on {self.device} in "
            f"{budgets.duration_text(self.seconds)}."
        )
        if self.failures:
            text += f" {len(self.failures):,} left out (not decoded)."
        return text

    def subset(self, rows: Sequence[int]) -> "AnomalyResult":
        """The same result restricted to `rows` (in this order)."""
        from copy import copy

        index = np.asarray(rows, dtype=int)
        out = copy(self)
        out.scores = self.scores[index]
        out.maps = self.maps[index]
        out.ok = self.ok[index]
        out.sizes = [self.sizes[i] for i in index]
        position = {int(r): j for j, r in enumerate(index)}
        out.failures = {position[i]: r for i, r in self.failures.items() if i in position}
        return out


# ---------------------------------------------------------------------------------------------
# Running a frozen network on its budget


class Runner:
    """A frozen network placed on its budget (R-VDP-04): fp16 on a GPU, fp32 on the CPU; on a GPU
    out-of-memory error the batch is halved, and at a batch of 1 the run moves to the CPU, said
    in the notes. Back on the CPU in fp32 at the end. The first batch is timed (R-VDP-05)."""

    def __init__(
        self,
        net: Backbone,
        device: budgets.Choice = "auto",
        allow_gpu: bool = True,
        batch: int | None = None,
        *,
        confirm_long: bool = False,
    ) -> None:
        self.net = net
        self.plan = budgets.plan(net.spec, device, allow_gpu, batch)
        self.confirm_long = confirm_long
        self.estimate: budgets.Estimate | None = None
        self._threads: int | None = None

    def __enter__(self) -> "Runner":
        import torch

        self._threads = torch.get_num_threads()
        self.place()
        return self

    def __exit__(self, *exc: object) -> None:
        import torch

        self.net.module.to("cpu").float()
        if self._threads is not None:
            torch.set_num_threads(self._threads)
        free_gpu()

    def place(self) -> None:
        import torch

        module = self.net.module
        module.to(torch.device(self.plan.device))
        if self.plan.half:
            module.half()
        else:
            module.float()
        if self.plan.threads:
            torch.set_num_threads(self.plan.threads)

    def tensor(self, chunk: Sequence[np.ndarray]) -> Any:
        """Normalised `(3, side, side)` arrays -> a batch on the device, in its precision."""
        import torch

        x = torch.from_numpy(np.stack(chunk)).to(self.plan.device)
        return x.half() if self.plan.half else x.float()

    def run[T](
        self, fn: Callable[[Sequence[np.ndarray]], T], items: Sequence[np.ndarray]
    ) -> list[T]:
        try:
            return budgets.run_in_batches(fn, items, self.plan)
        except budgets.GpuExhausted:
            self.plan = budgets.cpu_budget(
                self.net.spec, "GPU ran out of memory with a batch of 1: finished on the CPU."
            )
            self.place()
            return budgets.run_in_batches(fn, items, self.plan)

    def timed(self, seconds: float, done: int, total: int) -> None:
        """R-VDP-05 after the first batch: the estimate, and `LongRun` beyond 15 minutes."""
        if self.estimate is not None:
            return
        self.estimate = budgets.estimate(seconds, done, total, self.plan.label)
        if self.estimate.needs_confirmation and not self.confirm_long:
            raise budgets.LongRun(self.estimate)

    @property
    def notes(self) -> list[str]:
        return [f"Ran on {self.plan.sentence}.", *self.net.notes, *self.plan.notes]


def free_gpu() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def batches(
    sources: Sequence[Source],
    side: int,
    size: Callable[[], int],
    failures: dict[int, str],
    sizes: list[tuple[int, int] | None],
    cancel: Callable[[], bool] | None = None,
) -> Iterator[tuple[list[int], list[np.ndarray]]]:
    """Decoded images, `size()` at a time (the batch may shrink on the way): (rows, pixels).
    Images that cannot be decoded go to `failures`."""
    start = 0
    n = len(sources)
    while start < n:
        if cancel is not None and cancel():
            raise Cancelled("Cancelled.")
        rows = list(range(start, min(n, start + max(1, size()))))
        start = rows[-1] + 1
        kept: list[int] = []
        pixels: list[np.ndarray] = []
        for row in rows:
            decoded = decode_rgb(sources[row], side)
            if isinstance(decoded, str):
                failures[row] = decoded
                continue
            pixels.append(decoded[0])
            sizes[row] = decoded[1]
            kept.append(row)
        yield kept, pixels


def resolve(backbone: "str | BackboneSpec | Backbone", specs: dict[str, BackboneSpec]) -> Backbone:
    """A ready network: given, or loaded from verified weights (never downloads, R-VDP-02)."""
    if isinstance(backbone, Backbone):
        return backbone
    if isinstance(backbone, str):
        if backbone not in specs:
            raise KeyError(f"Unknown backbone {backbone!r}: one of {', '.join(specs)}.")
        backbone = specs[backbone]
    return load_backbone(backbone)


def intermediates(net: Backbone, x: Any, layers: Sequence[int]) -> list[Any]:
    """Feature maps of the given stages (timm: 1 = layer1 ... 4 = layer4), stopping early."""
    return net.module.forward_intermediates(
        x, indices=list(layers), intermediates_only=True, stop_early=True
    )


# ---------------------------------------------------------------------------------------------
# Imported image sets (R-VAD-01, R-VAD-02)


@dataclass
class AnomalySplit:
    fit: list[int]  # rows the model learns from (normal images)
    test: list[int]  # rows scored and evaluated
    holdout: list[int]  # normal rows kept out of the model to set the threshold (R-VTH-02)
    code: str
    reason: str
    notes: list[str] = field(default_factory=list)


def split_rows(
    splits: Sequence[Any] | None,
    normal: Sequence[Any],
    usable: Sequence[bool],
    seed: int = SEED,
) -> AnomalySplit:
    """R-VAD-01: the producer's split (normal `train` images fit, `test` images are scored);
    otherwise 80 % of the normal images fit and the other 20 % join the defects in the test, seed
    0; without any normal flag, every image is assumed normal (R-VML-03). R-VAD-02: when the test
    holds no labelled defect, 10 % of the fitting normals are held out for the threshold."""
    n = len(normal)
    rows = [i for i in range(n) if usable[i]]
    flags = [normal[i] for i in range(n)]
    known = any(flags[i] is not None for i in rows)
    s = (
        [None if v is None else str(v).lower() for v in splits]
        if splits is not None
        else [None] * n
    )
    notes: list[str] = []
    rng = np.random.default_rng(seed)
    train_rows = [i for i in rows if s[i] in TRAIN_SPLITS]
    test_rows = [i for i in rows if s[i] in TEST_SPLITS]
    if train_rows and test_rows:
        fit = [i for i in train_rows if flags[i] is not False]
        dropped = len(train_rows) - len(fit)
        if dropped:
            notes.append(
                f"{dropped:,} training images marked as defects are left out of the model: it "
                "learns from normal images only (R-VAD-01)."
            )
        test = test_rows
        if not known:
            notes.append(
                "No image is marked normal: the training images are assumed normal (R-VML-03)."
            )
        code, reason = (
            "R-VAD-01",
            f"The producer's split: the model learns from {len(fit):,} normal training images and "
            f"scores the {len(test):,} test images (R-VAD-01).",
        )
    elif known:
        normals = [i for i in rows if flags[i] is True]
        defects = [i for i in rows if flags[i] is False]
        shuffled = [normals[j] for j in rng.permutation(len(normals))]
        cut = max(1, round(TEST_SHARE * len(shuffled))) if len(shuffled) >= 2 else 0
        fit = sorted(shuffled[cut:])
        test = sorted(shuffled[:cut] + defects)
        code, reason = (
            "R-VAD-01",
            f"No split from the producer: {len(fit):,} normal images (80 %, seed {seed}) fit the "
            f"model; the other {cut:,} normal images and the {len(defects):,} defects are scored "
            "(R-VAD-01).",
        )
    else:
        fit = train_rows or rows
        test = test_rows or rows
        code, reason = (
            "R-VML-03",
            "No image is marked normal or defective: every image is assumed normal, so the model "
            "assumes most images are normal (R-VML-03).",
        )
    if not fit:
        raise ValueError("No normal image to learn from (R-VAD-01).")
    has_defect = any(flags[i] is False for i in test)
    holdout: list[int] = []
    if not has_defect and len(fit) >= 10:
        picked = rng.permutation(len(fit))[: max(1, round(HOLDOUT_SHARE * len(fit)))]
        holdout = sorted(fit[j] for j in picked)
        kept = set(holdout)
        fit = [i for i in fit if i not in kept]
        notes.append(
            f"No labelled defect among the scored images: {len(holdout):,} normal images (10 %) "
            "are kept out of the model to set the threshold (R-VAD-02, R-VTH-02)."
        )
    return AnomalySplit(fit, test, holdout, code, reason, notes)


def split_image_set(images: "ImageSet", seed: int = SEED) -> AnomalySplit:
    table = images.table
    decoded = table["decode_ok"].to_pylist()
    excluded = table["excluded"].to_pylist()
    stored = table["stored_path"].to_pylist()
    usable = [
        bool(decoded[i]) and not excluded[i] and stored[i] is not None for i in range(len(stored))
    ]
    return split_rows(table["split"].to_pylist(), table["is_normal"].to_pylist(), usable, seed)


@dataclass
class AnomalyRun:
    """A detector fitted on an imported image set and scored on its test images."""

    method: str
    split: AnomalySplit
    result: AnomalyResult  # the test rows, in `split.test` order
    rows: list[int]  # table rows of `result`
    image_ids: list[str]
    report: vm.AnomalyReport | None  # None when the test has no labelled defect
    threshold: vm.ThresholdChoice | None
    model: Any  # the fitted detector (PatchCore, PaDiM or ConvAutoencoder)
    notes: list[str] = field(default_factory=list)

    @property
    def sentence(self) -> str:
        if self.report is None:
            base = f"{self.result.title}: no labelled defect to evaluate on."
        else:
            r = self.report
            base = f"{self.result.title}: image AUROC {r.auroc:.3f}"
            if r.auroc_interval is not None:
                base += f" [{r.auroc_interval.low:.3f}; {r.auroc_interval.high:.3f}]"
            if r.pixel is not None:
                base += f", pixel AUROC {r.pixel.auroc:.3f}, AUPRO {r.pixel.aupro:.3f}"
            base += "."
        if self.threshold is not None:
            base += f" {self.threshold.message}"
        return base


def detector(method: Method, **options: Any) -> Any:
    """A detector by name, with its options."""
    if method == "patchcore":
        from quantilo.vision.deep.patchcore import PatchCore

        return PatchCore(**options)
    if method == "padim":
        from quantilo.vision.deep.padim import PaDiM

        return PaDiM(**options)
    if method in ("autoencoder", "vae"):
        from quantilo.vision.deep.autoencoder import ConvAutoencoder

        return ConvAutoencoder(variational=method == "vae", **options)
    raise ValueError(f"Unknown method {method!r}: patchcore, padim, autoencoder or vae.")


def anomaly_image_set(
    images: "ImageSet",
    method: Method = "patchcore",
    *,
    pixel_metrics: bool = True,
    user_threshold: float | None = None,
    bootstrap: bool = True,
    seed: int = SEED,
    model: Any = None,
    **options: Any,
) -> AnomalyRun:
    """Fit `method` on the normal training images of an imported set and score its test images
    (R-VAD-01, R-VAD-02); with labelled defects, every metric of § 7.2 (image AUROC first, the
    threshold by R-VTH, and with masks the pixel AUROC and AUPRO at the model's side). `model`
    replaces the detector built from `method` and `options` (tests, custom networks)."""
    split = split_image_set(images, seed)
    table = images.table
    stored = table["stored_path"].to_pylist()
    path = images.store.path
    fit_sources: list[Source] = [path(stored[i]) for i in split.fit]
    scored = split.test + split.holdout
    sources: list[Source] = [path(stored[i]) for i in scored]
    det = model if model is not None else detector(method, seed=seed, **options)
    det.fit(fit_sources)
    both = det.score(sources)
    n_test = len(split.test)
    result = both.subset(range(n_test))
    if hasattr(result, "fit_rows"):  # PatchCore: name the normal patches by their table rows
        result.fit_rows = list(split.fit)
        result.top = [result.explain(i) if result.ok[i] else None for i in range(n_test)]
    holdout = both.scores[n_test:]
    holdout = holdout[np.isfinite(holdout)]
    ids = [str(v) for v in table["image_id"].to_pylist()]
    flags = table["is_normal"].to_pylist()
    labels = table["label"].to_pylist()
    masks_col = table["mask_path"].to_pylist()
    ok = result.ok
    test_rows = [split.test[j] for j in range(n_test) if ok[j]]
    is_defect = np.array([flags[i] is False for i in test_rows])
    scores = result.scores[ok]
    notes = list(split.notes)
    report: vm.AnomalyReport | None = None
    threshold: vm.ThresholdChoice | None = None
    if is_defect.any() and (~is_defect).any():
        pairs = list(zip(test_rows, is_defect.tolist(), strict=True))
        types = [str(labels[i] or "defect") if d else NORMAL for i, d in pairs]
        masks = maps = None
        defect_rows = [i for i, d in pairs if d]
        if pixel_metrics and any(masks_col[i] for i in defect_rows):
            if all(masks_col[i] for i in defect_rows):
                masks = np.stack(
                    [
                        load_mask(path(masks_col[i]) if masks_col[i] else None, result.side)
                        for i in test_rows
                    ]
                )
                maps = result.maps[ok]
                notes.append(
                    f"Pixel metrics on the {result.side} x {result.side} maps, masks resized to "
                    "that size (nearest neighbour)."
                )
            else:
                notes.append("Some defects have no mask: pixel metrics left out (R-VML-04).")
        groups = [table["pixel_sha"][i].as_py() or f"row{i}" for i in test_rows]
        report = vm.anomaly_report(
            is_defect,
            scores,
            defect_types=types,
            holdout_normal_scores=holdout if holdout.size else None,
            user_threshold=user_threshold,
            masks=masks,
            maps=maps,
            groups=groups,
            bootstrap=bootstrap,
            seed=seed,
        )
        threshold = report.threshold
    else:
        try:
            threshold = vm.choose_threshold(
                None, None, holdout if holdout.size else None, user_threshold, seed
            )
        except ValueError as error:
            notes.append(str(error))
    return AnomalyRun(
        method if model is None else getattr(det, "method", method),
        split,
        result,
        list(split.test),
        [ids[i] for i in split.test],
        report,
        threshold,
        det,
        notes,
    )
