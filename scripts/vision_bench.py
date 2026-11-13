"""Vision bench (120): validate the test sets, then score models with the protocol of the study
(docs/etudes/2026-11-11-vision-et-xai.md, section 12) through quantilo.vision.metrics (111).

Until the models exist (112, 114, 115), the bench scores trivial models only, to check the
chain end to end: data -> scores -> metrics -> thresholds -> report. Their numbers are the floor
every real model must beat:
- Beans (classification): majority class, random probabilities, and a logistic regression on the
  JPEG file size (a model that reads no pixel at all);
- MVTec AD bottle (anomalies): random scores and maps, and the distance of the PNG file size to
  the training normals.
Expected: random ~0.5 AUROC, random AUPRO ~0.15 (= 0.3 / 2: the PRO curve of random maps is the
diagonal).

With `--models classic` (or `deep`), the bench also imports each set (104, in a temporary folder
deleted at the end) and scores the first-plan models of 112 (`vision.classic_models`): logistic
regression and HGB on the classic descriptors for Beans, patch PCA, nearest normals and their
average rank for MVTec bottle; `deep` adds the DINOv2-S/14 probe and kNN when the extra `vision`
and its weights are installed (R-VCL-06), and the real models of 114 to 118 (b2): the fine-tuned
DINOv2 (Advanced > More) on Beans, and PatchCore, PaDiM and the convolutional autoencoder on MVTec
bottle (one run per detector on the same import, so the vectors are computed once). The
temperature scaling applied out of fold (R-VM-06) is reported (ECE before and after), and the
images R-VQ-16 suggests are counted.

Usage (from packages/quantilo, so that quantilo is importable):
    uv run --no-sync python ../../scripts/vision_bench.py [--data DIR] [--only beans]
        [--draws 1000] [--seed 0] [--models classic|deep] [--json results.json]
Download the sets first with scripts/vision_datasets.py. Images are written only to that
temporary folder (`--models`).
"""

import argparse
import hashlib
import json
import math
import struct
import sys
import tempfile
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "quantilo" / "src"))

from quantilo.vision import metrics as vm  # noqa: E402

DEFAULT_DATA = ROOT.parent / "data-corpus" / "vision"
BEANS_SPLITS = {"train": 1034, "validation": 133, "test": 128}  # published split sizes
BEANS_LABELS = ["angular_leaf_spot", "bean_rust", "healthy"]
BOTTLE_TEST = {"good": 20, "broken_large": 20, "broken_small": 22, "contamination": 21}
BOTTLE_TRAIN = 209
MASK_STRIDE = 4  # pixel metrics on every 4th pixel (900 -> 225): same curves, 16x less memory
HOLDOUT_SHARE = 0.10  # R-VTH-02: normals held out of the training images


class BenchError(RuntimeError):
    pass


def check(condition: bool, message: str, problems: list[str]) -> None:
    if not condition:
        problems.append(message)


# ---------------------------------------------------------------------------------------------
# Minimal PNG reader (standard library + numpy), for the masks and image sizes. Pillow arrives
# with 103; the bench does not need to wait for it.

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def png_header(data: bytes) -> tuple[int, int, int, int, int]:
    if not data.startswith(PNG_SIGNATURE) or data[12:16] != b"IHDR":
        raise BenchError("not a PNG file")
    width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", data[16:29])
    return width, height, depth, colour, interlace


def png_decode(data: bytes) -> np.ndarray:
    """8-bit, non-interlaced grey / RGB / grey+alpha / RGBA PNG to an (H, W[, C]) uint8 array."""
    width, height, depth, colour, interlace = png_header(data)
    if depth != 8 or interlace or colour not in CHANNELS:
        raise BenchError(f"PNG depth {depth}, colour type {colour}, interlace {interlace}")
    pos, idat = 8, []
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        if kind == b"IDAT":
            idat.append(data[pos + 8 : pos + 8 + length])
        pos += 12 + length
    bpp = CHANNELS[colour]
    stride = width * bpp
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), dtype=np.uint8)
    raw = raw.reshape(height, stride + 1)
    out = np.zeros((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.int64)
    for y in range(height):
        kind, line = raw[y, 0], raw[y, 1:].astype(np.int64)
        if kind == 0:
            row = line
        elif kind == 1:  # Sub: a running sum per channel
            row = np.cumsum(line.reshape(width, bpp), axis=0).ravel() % 256
        elif kind == 2:  # Up
            row = (line + previous) % 256
        else:  # Average, Paeth: sequential along the line
            row = np.zeros(stride, dtype=np.int64)
            for x in range(stride):
                a = row[x - bpp] if x >= bpp else 0
                b = previous[x]
                if kind == 3:
                    row[x] = (line[x] + (a + b) // 2) % 256
                else:
                    c = previous[x - bpp] if x >= bpp else 0
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                    row[x] = (line[x] + pred) % 256
        out[y] = row
        previous = row
    return out.reshape(height, width) if bpp == 1 else out.reshape(height, width, bpp)


# ---------------------------------------------------------------------------------------------
# Beans


def _beans_split(folder: Path, split: str) -> tuple[np.ndarray, list[bytes], list[str]]:
    import pyarrow.parquet as pq

    path = folder / "data" / f"{split}-00000-of-00001.parquet"
    if not path.is_file():
        raise BenchError(f"{path} missing: run scripts/vision_datasets.py get beans")
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    names = BEANS_LABELS
    if b"huggingface" in meta:
        info = json.loads(meta[b"huggingface"])
        names = info["info"]["features"]["labels"]["names"]
    images = table.column("image").to_pylist()
    return (
        np.asarray(table.column("labels").to_pylist(), dtype=int),
        [img["bytes"] for img in images],
        names,
    )


def bench_beans(folder: Path, draws: int, seed: int) -> dict:
    started = time.perf_counter()
    problems: list[str] = []
    splits = {name: _beans_split(folder, name) for name in BEANS_SPLITS}
    names = splits["train"][2]
    check(names == BEANS_LABELS, f"labels {names}", problems)
    hashes: dict[str, set[str]] = {}
    for split, (labels, blobs, _) in splits.items():
        check(len(labels) == BEANS_SPLITS[split], f"{split}: {len(labels)} rows", problems)
        check(
            set(labels.tolist()) <= {0, 1, 2}, f"{split}: labels {set(labels.tolist())}", problems
        )
        check(all(b[:3] == b"\xff\xd8\xff" for b in blobs), f"{split}: a non-JPEG image", problems)
        hashes[split] = {hashlib.sha256(b).hexdigest() for b in blobs}
    leaks = {
        f"{a}/{b}": len(hashes[a] & hashes[b])
        for a, b in (("train", "test"), ("train", "validation"), ("validation", "test"))
    }
    y_train, blobs_train, _ = splits["train"]
    y_test, blobs_test, _ = splits["test"]
    label_names = np.array(names)
    t_train, t_test = label_names[y_train], label_names[y_test]
    rng = np.random.default_rng(seed)

    models: dict[str, dict] = {}

    majority = Counter(t_train.tolist()).most_common(1)[0][0]
    models["majority"] = _classification(
        t_test, np.full(len(t_test), majority), None, names, t_train, draws, seed
    )
    random_proba = rng.dirichlet(np.ones(3), size=len(t_test))
    models["random"] = _classification(t_test, None, random_proba, names, t_train, draws, seed)

    from sklearn.linear_model import LogisticRegression

    def size_feature(blobs: list[bytes]) -> np.ndarray:
        return np.log([[len(b)] for b in blobs])

    fitted = LogisticRegression(max_iter=1000).fit(size_feature(blobs_train), t_train)
    order = list(fitted.classes_)
    proba = fitted.predict_proba(size_feature(blobs_test))[:, [order.index(n) for n in names]]
    models["jpeg_size_logistic"] = _classification(t_test, None, proba, names, t_train, draws, seed)

    return {
        "dataset": "beans",
        "rows": {s: len(v[0]) for s, v in splits.items()},
        "classes": {s: dict(Counter(label_names[v[0]].tolist())) for s, v in splits.items()},
        "exact_duplicates_across_splits": leaks,
        "problems": problems,
        "models": models,
        "seconds": round(time.perf_counter() - started, 1),
    }


def _classification(t, pred, proba, names, t_train, draws, seed) -> dict:
    report = vm.classification_report(
        t, y_pred=pred, proba=proba, labels=names, y_train=t_train, draws=draws, seed=seed
    )
    main = report.main.metric
    out = {
        "main": f"{main} ({report.main.reason})",
        "accuracy": round(report.accuracy, 4),
        "balanced_accuracy": round(report.balanced_accuracy, 4),
        "f1_macro": round(report.f1_macro, 4),
        "kappa": round(report.kappa, 4),
        "baseline_accuracy": round(report.baseline.accuracy, 4),
        "intervals": {k: [round(v.low, 4), round(v.high, 4)] for k, v in report.intervals.items()},
    }
    if report.roc_auc is not None:
        assert report.calibration is not None
        out |= {
            "roc_auc": round(report.roc_auc, 4),
            "pr_auc": round(report.pr_auc or math.nan, 4),
            "log_loss": round(report.log_loss or math.nan, 4),
            "brier": round(report.brier or math.nan, 4),
            "ece": round(report.calibration.reliability.ece, 4),
            "temperature": report.calibration.temperature,
        }
    return out


# ---------------------------------------------------------------------------------------------
# MVTec AD, bottle


def bench_bottle(folder: Path, draws: int, seed: int) -> dict:
    started = time.perf_counter()
    problems: list[str] = []
    base = folder / "bottle"
    if not base.is_dir():
        raise BenchError(f"{base} missing: run scripts/vision_datasets.py get mvtec-bottle")
    train = sorted((base / "train" / "good").glob("*.png"))
    check(len(train) == BOTTLE_TRAIN, f"train/good: {len(train)} images", problems)
    test: list[tuple[str, Path]] = []
    for kind, expected in BOTTLE_TEST.items():
        found = sorted((base / "test" / kind).glob("*.png"))
        check(len(found) == expected, f"test/{kind}: {len(found)} images", problems)
        test += [(kind, p) for p in found]
    kinds = np.array([k for k, _ in test])
    is_defect = kinds != "good"
    shape = None
    masks = []
    for kind, path in test:
        data = path.read_bytes()
        width, height, *_ = png_header(data)
        shape = shape or (height, width)
        check((height, width) == shape, f"{path.name}: {width}x{height}", problems)
        if kind == "good":
            masks.append(np.zeros((height, width), dtype=bool)[::MASK_STRIDE, ::MASK_STRIDE])
            continue
        mask_path = base / "ground_truth" / kind / f"{path.stem}_mask.png"
        if not mask_path.is_file():
            problems.append(f"no mask for {kind}/{path.name}")
            masks.append(np.zeros((height, width), dtype=bool)[::MASK_STRIDE, ::MASK_STRIDE])
            continue
        mask = png_decode(mask_path.read_bytes())
        check(mask.shape == (height, width), f"{mask_path.name}: shape {mask.shape}", problems)
        check(set(np.unique(mask).tolist()) <= {0, 255}, f"{mask_path.name}: not binary", problems)
        check(bool(mask.any()), f"{mask_path.name}: empty", problems)
        masks.append(mask[::MASK_STRIDE, ::MASK_STRIDE] > 0)
    gt = np.stack(masks)
    train_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in train}
    test_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for _, p in test}
    rng = np.random.default_rng(seed)
    held = rng.permutation(len(train))[: max(1, round(HOLDOUT_SHARE * len(train)))]
    fit = np.setdiff1d(np.arange(len(train)), held)

    models: dict[str, dict] = {}
    # random scores and maps
    models["random"] = _anomaly(
        is_defect,
        rng.random(len(test)),
        kinds,
        rng.random(len(held)),
        gt,
        rng.random(gt.shape),
        draws,
        seed,
    )
    # distance of the PNG size to the training normals (reads no pixel)
    train_log = np.log([p.stat().st_size for p in train])
    centre, spread = np.median(train_log[fit]), np.std(train_log[fit]) or 1.0

    def size_score(paths: list[Path]) -> np.ndarray:
        return np.abs(np.log([p.stat().st_size for p in paths]) - centre) / spread

    models["png_size_distance"] = _anomaly(
        is_defect,
        size_score([p for _, p in test]),
        kinds,
        size_score([train[i] for i in held]),
        None,
        None,
        draws,
        seed,
    )
    return {
        "dataset": "mvtec-bottle",
        "train": len(train),
        "test": dict(Counter(kinds.tolist())),
        "image_size": shape,
        "mask_pixels_scored": f"every {MASK_STRIDE}th pixel: {gt.shape}",
        "defect_regions": vm.pro_curve(gt, gt.astype(float))[2],
        "exact_duplicates_train_test": len(train_hashes & test_hashes),
        "problems": problems,
        "models": models,
        "seconds": round(time.perf_counter() - started, 1),
    }


def _anomaly(is_defect, scores, kinds, held_scores, gt, maps, draws, seed) -> dict:
    report = vm.anomaly_report(
        is_defect,
        scores,
        defect_types=kinds,
        masks=gt,
        maps=maps,
        pixel_threshold=None if maps is None else float(np.quantile(maps, 0.99)),
        draws=draws,
        seed=seed,
    )
    assert report.threshold is not None and report.at_threshold is not None
    normal_rule = vm.choose_threshold(holdout_normal_scores=held_scores)
    normal_counts = vm.at_threshold(is_defect, scores, normal_rule.threshold)
    out = {
        "auroc": round(report.auroc, 4),
        "auroc_interval": None
        if report.auroc_interval is None
        else [round(report.auroc_interval.low, 4), round(report.auroc_interval.high, 4)],
        "ap": round(report.ap, 4),
        "auroc_by_type": {k: round(v, 4) for k, v in report.auroc_by_type.items()},
        "threshold": report.threshold.message,
        "at_threshold": {
            "f1": round(report.at_threshold.f1, 4),
            "precision": round(report.at_threshold.precision, 4),
            "recall": round(report.at_threshold.recall, 4),
            "false_alarm_rate": round(report.at_threshold.false_alarm_rate, 4),
        },
        "R-VTH-02_on_test": {
            "message": normal_rule.message,
            "false_alarm_rate": round(normal_counts.false_alarm_rate, 4),
            "recall": round(normal_counts.recall, 4),
        },
    }
    if report.pixel is not None:
        out["pixel"] = {
            "auroc": round(report.pixel.auroc, 4),
            "aupro_30": round(report.pixel.aupro, 4),
            "f1": round(report.pixel.f1 or 0, 4),
            "iou": round(report.pixel.iou or 0, 4),
        }
    return out


# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# Models of 112 (imported sets, temporary folder)


def _model_summary(model) -> dict:
    out: dict = {"title": model.title, "setting": model.setting, "default": model.default}
    if model.classification is not None:
        c = model.classification
        out |= {
            "main": c.main.metric,
            "accuracy": round(c.accuracy, 4),
            "balanced_accuracy": round(c.balanced_accuracy, 4),
            "f1_macro": round(c.f1_macro, 4),
            "kappa": round(c.kappa, 4),
            "roc_auc": None if c.roc_auc is None else round(c.roc_auc, 4),
            "log_loss": None if c.log_loss is None else round(c.log_loss, 4),
            "ece": None if c.calibration is None else round(c.calibration.reliability.ece, 4),
            "ece_after": None
            if c.calibration is None or not c.calibration.applied or c.calibration.ece_after is None
            else round(c.calibration.ece_after, 4),
            "temperature": None
            if c.calibration is None or c.calibration.temperature is None
            else round(c.calibration.temperature, 3),
            "log_loss_after": None
            if c.calibration is None or c.calibration.log_loss_after is None
            else round(c.calibration.log_loss_after, 4),
            "intervals": {k: [round(v.low, 4), round(v.high, 4)] for k, v in c.intervals.items()},
            "confusion": c.confusion,
            "inner_cv": None if model.selection is None else round(model.selection, 4),
        }
    if model.anomaly is not None:
        a = model.anomaly
        out |= {
            "auroc": round(a.auroc, 4),
            "auroc_interval": None
            if a.auroc_interval is None
            else [round(a.auroc_interval.low, 4), round(a.auroc_interval.high, 4)],
            "ap": round(a.ap, 4),
            "auroc_by_type": {k: round(v, 4) for k, v in a.auroc_by_type.items()},
            "threshold": None if a.threshold is None else a.threshold.message,
        }
        if a.at_threshold is not None:
            out["at_threshold"] = {
                "f1": round(a.at_threshold.f1, 4),
                "precision": round(a.at_threshold.precision, 4),
                "recall": round(a.at_threshold.recall, 4),
                "false_alarm_rate": round(a.at_threshold.false_alarm_rate, 4),
            }
        if a.pixel is not None:
            out["pixel"] = {
                "auroc": round(a.pixel.auroc, 4),
                "aupro_30": round(a.pixel.aupro, 4),
                "f1": None if a.pixel.f1 is None else round(a.pixel.f1, 4),
                "iou": None if a.pixel.iou is None else round(a.pixel.iou, 4),
            }
    out["seconds"] = model.seconds
    return out


def _run_models(
    source: Path, workspace: Path, name: str, deep: bool, seed: int, runs: list[dict]
) -> dict:
    """Import once, then one `evaluate` per entry of `runs` (its options); a model already scored
    by an earlier run (the classic ones) is kept from the first."""
    from quantilo.vision import image_set
    from quantilo.vision.classic_models import evaluate_image_set

    started = time.perf_counter()
    images = image_set(source, workspace=workspace, dataset_id=name)
    imported = time.perf_counter() - started
    out: dict = {"dataset": name, "import_seconds": round(imported, 1), "models": {}, "runs": []}
    for options in runs if deep else [{}]:
        began = time.perf_counter()
        outcome = evaluate_image_set(images, deep=None if deep else False, seed=seed, **options)
        result = outcome.result
        out.setdefault("task", result.task.message)
        out.setdefault("split", result.split.sentence)
        out.setdefault("deep", result.deep.reason or "used")
        for m in result.models:
            out["models"].setdefault(m.key, _model_summary(m))
        found = result.mislabelled
        out["runs"].append(
            {
                "options": options,
                "default": result.default,
                "sentence": result.sentence,
                "features": {
                    f.key: {"computed": f.computed, "seconds": f.seconds} for f in result.features
                },
                "notes": result.notes,
                "mislabelled": None
                if found is None
                else {
                    "count": found.count,
                    "checked": found.details["checked"],
                    "images": found.details["images"][:10],
                },
                "seconds": round(time.perf_counter() - began, 1),
            }
        )
    return out


def models_beans(folder: Path, seed: int, deep: bool) -> dict:
    """The three Parquet files written as train / validation / test folders of JPEG files (the
    producer's split, R-VIS-LAY-02), then imported and scored; `deep` adds the fine-tuned DINOv2
    of 118 (Advanced > More)."""
    with tempfile.TemporaryDirectory(prefix="quantilo-bench-") as tmp:
        root = Path(tmp) / "beans"
        for split in BEANS_SPLITS:
            labels, blobs, names = _beans_split(folder, split)
            for i, (label, blob) in enumerate(zip(labels, blobs, strict=True)):
                target = root / split / names[label]
                target.mkdir(parents=True, exist_ok=True)
                (target / f"{i:04d}.jpg").write_bytes(blob)
        runs = [{"more": ["finetune"], "confirm_long": True}]
        return _run_models(root, Path(tmp) / "workspace", "beans", deep, seed, runs)


def models_bottle(folder: Path, seed: int, deep: bool) -> dict:
    """`deep`: PatchCore, PaDiM (115) and the convolutional autoencoder (116), one run each."""
    base = folder / "bottle"
    if not base.is_dir():
        raise BenchError(f"{base} missing: run scripts/vision_datasets.py get mvtec-bottle")
    with tempfile.TemporaryDirectory(prefix="quantilo-bench-") as tmp:
        runs = [
            {"detector": d, "confirm_long": True} for d in ("patchcore", "padim", "autoencoder")
        ]
        return _run_models(base, Path(tmp) / "workspace", "bottle", deep, seed, runs)


BENCHES = {"beans": ("beans", bench_beans), "mvtec-bottle": ("mvtec-bottle", bench_bottle)}
MODELS = {"beans": models_beans, "mvtec-bottle": models_bottle}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--only", choices=list(BENCHES), action="append")
    parser.add_argument("--draws", type=int, default=vm.BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=vm.SEED)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--models", choices=["classic", "deep"], help="also score 112's models")
    args = parser.parse_args(argv)
    results, failed = [], False
    for name in args.only or list(BENCHES):
        folder_name, bench = BENCHES[name]
        try:
            result = bench(args.data / folder_name, args.draws, args.seed)
        except BenchError as error:
            print(f"{name}: skipped, {error}", file=sys.stderr)
            failed = True
            continue
        failed |= bool(result["problems"])
        if args.models:
            try:
                result["models_112"] = MODELS[name](
                    args.data / folder_name, args.seed, args.models == "deep"
                )
            except BenchError as error:
                print(f"{name}: models skipped, {error}", file=sys.stderr)
                failed = True
        results.append(result)
        print(json.dumps(result, indent=1, default=str))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
