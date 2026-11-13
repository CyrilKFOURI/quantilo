"""Bench of the deep anomaly detectors (115, 116) on MVTec AD `bottle`.

Imports the category as an image set (104) in a temporary workspace, fits one detector on the
209 normal training images and scores the 83 test images through `anomaly_image_set`: image
AUROC (with its bootstrap interval), AP, pixel AUROC and AUPRO up to 30 % FPR (masks resized to
the maps' 256 px), F1 at the R-VTH-01 threshold, time and peak GPU memory.

MVTec AD is CC BY-NC-SA 4.0: local, non-commercial use; nothing is written to the repository.
Needs the extra `vision` and, for PatchCore and PaDiM, the weights (downloaded only with
`--download`, which is the user's confirmation, R-VDP-02).

Usage (one detector per run, to keep the memory low):
    python scripts/vision_bench_anomaly.py --method patchcore [--backbone wide_resnet50_2]
        [--device gpu|cpu] [--epochs 100] [--download] [--data DIR] [--json out.json]
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "quantilo" / "src"))

from quantilo.vision.deep import weights  # noqa: E402
from quantilo.vision.deep.anomaly import anomaly_image_set  # noqa: E402
from quantilo.vision.ingest import image_set  # noqa: E402

DEFAULT_DATA = ROOT.parent / "data-corpus" / "vision" / "mvtec-bottle" / "bottle"
NEEDS = {"patchcore": "wide_resnet50_2", "padim": "resnet18"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="patchcore")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    import torch

    key = args.backbone or NEEDS.get(args.method)
    if key is not None and not weights.downloaded(key):
        if not args.download:
            raise SystemExit(f"{weights.missing_reason(weights.MANIFEST[key])} (use --download)")
        weights.download(key, confirm=True)
    options: dict = {"device": args.device, "confirm_long": True}
    if args.method in ("patchcore", "padim") and args.backbone:
        options["backbone"] = args.backbone
    if args.method in ("autoencoder", "vae"):
        options["epochs"] = args.epochs
    with tempfile.TemporaryDirectory() as workspace:
        started = time.perf_counter()
        images = image_set(args.data, workspace=workspace, thumbnails=False, quality=False)
        imported = time.perf_counter() - started
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        run = anomaly_image_set(images, args.method, **options)
        seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    r = run.report
    assert r is not None
    out = {
        "method": args.method,
        "title": run.result.title,
        "device": run.result.device,
        "fit_images": len(run.split.fit),
        "test_images": len(run.rows),
        "image_auroc": r.auroc,
        "image_auroc_interval": None
        if r.auroc_interval is None
        else [r.auroc_interval.low, r.auroc_interval.high],
        "ap": r.ap,
        "pixel_auroc": None if r.pixel is None else r.pixel.auroc,
        "aupro_30": None if r.pixel is None else r.pixel.aupro,
        "auroc_by_type": r.auroc_by_type,
        "threshold": None if r.threshold is None else r.threshold.message,
        "f1_at_threshold": None if r.at_threshold is None else r.at_threshold.f1,
        "seconds_fit_and_score": round(seconds, 1),
        "seconds_import": round(imported, 1),
        "peak_gpu_mb": round(peak),
        "notes": run.result.notes + run.notes,
    }
    if hasattr(run.model, "losses") and run.model.losses:
        out["loss_first_last"] = [run.model.losses[0], run.model.losses[-1]]
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
