"""Bench of the c3 extensions (docs/features/c3-vision-suite): MVTec AD `metal_nut` and a CIFAR-10
subset, after `bottle` and Beans (120).

- metal_nut (anomalies, masks): the import read in one process, then with the processes of
  R-VIS-ING-07 (same table, the time of each); the classic models (patch PCA, nearest normals,
  their average rank), PatchCore and the ensemble PatchCore + autoencoder (R-VAN-07): image AUROC
  and AP, pixel AUROC and AUPRO 30 %, time, peak GPU memory.
- cifar10-600 (600 test images, 32 x 32, 10 classes): the quality rules (R-VQ-02 on tiny images),
  the classic classifiers and, with the extra `vision` and its weights, the DINOv2 probe; the map
  drawn with PCA and with PaCMAP (R-VMAP-12) when the extra `interpret` is installed.
- quality (the imagette of 120 T8, made by `tests/vision_fixtures.py`, nothing downloaded): 78
  synthetic images, a case for each R-VQ rule the pixels or the files can trigger among clean
  photos, imported from a folder and from a ZIP with traps; per rule the images expected, found,
  missed and extra (all must be exact), and per trap the rule that skipped it.

Every set is imported in a temporary folder deleted at the end: nothing is written to the
repository; MVTec AD is CC BY-NC-SA 4.0 (local, non-commercial use). Download the sets first:
    python scripts/vision_datasets.py get mvtec-metal_nut --accept-noncommercial
    python scripts/vision_datasets.py get cifar10-600
Usage (from the repository; one set at a time keeps the memory low):
    python scripts/vision_bench_c3.py --only metal_nut [--detectors patchcore_ae]
    python scripts/vision_bench_c3.py --only cifar10 [--json out.json]
    python scripts/vision_bench_c3.py --only quality       # a few seconds, exit code 1 if inexact
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "quantilo" / "src"))
# tests.vision_fixtures makes the quality imagette
sys.path.insert(1, str(ROOT / "packages" / "quantilo"))

from quantilo.vision.classic_models import evaluate_image_set  # noqa: E402
from quantilo.vision.ingest import image_set  # noqa: E402

DEFAULT_DATA = ROOT.parent / "data-corpus" / "vision"


def _peak_reset() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _peak() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20)
    except ImportError:
        pass
    return 0


def _anomaly(model: Any) -> dict[str, Any]:
    a = model.anomaly
    out: dict[str, Any] = {"title": model.title, "seconds": model.seconds}
    if a is not None:
        out |= {
            "image_auroc": round(a.auroc, 4),
            "ap": round(a.ap, 4),
            "pixel_auroc": None if a.pixel is None else round(a.pixel.auroc, 4),
            "aupro_30": None if a.pixel is None else round(a.pixel.aupro, 4),
            "f1_at_threshold": None
            if a.at_threshold is None
            else round(a.at_threshold.f1, 4),
        }
    return out


def bench_metal_nut(folder: Path, detectors: list[str]) -> dict[str, Any]:
    source = folder / "metal_nut"
    out: dict[str, Any] = {
        "set": "MVTec AD metal_nut (CC BY-NC-SA 4.0, local test use)"
    }
    with tempfile.TemporaryDirectory() as workspace:
        timings = {}
        tables = {}
        for workers in (1, None):
            started = time.perf_counter()
            images = image_set(
                source,
                workspace=workspace,
                dataset_id=f"nut{workers or 'p'}",
                workers=workers,
            )
            timings["1 process" if workers == 1 else "R-VIS-ING-07"] = round(
                time.perf_counter() - started, 1
            )
            tables[workers] = images
        parallel = next(
            (r.message for r in tables[None].rules if r.code == "R-VIS-ING-07"), None
        )
        same = (
            tables[1]
            .table.drop_columns(["stored_path", "thumb_path", "mask_path"])
            .equals(
                tables[None].table.drop_columns(
                    ["stored_path", "thumb_path", "mask_path"]
                )
            )
        )
        images = tables[None]
        out["import"] = {
            "images": images.n_images,
            "seconds": timings,
            "parallel": parallel,
            "same_table": same,
            "quality": images.quality.sentence if images.quality else None,
        }
        runs: dict[str, Any] = {}
        for detector in ["none", *detectors]:
            _peak_reset()
            started = time.perf_counter()
            deep = (
                False if detector == "none" else None
            )  # a deep detector needs deep allowed
            found = evaluate_image_set(
                images, detector=detector, deep=deep, confirm_long=True
            )
            seconds = round(time.perf_counter() - started, 1)
            result = found.result
            keys = [m.key for m in result.models]
            wanted = keys if detector == "none" else [k for k in keys if k not in runs]
            for key in wanted:
                runs[key] = _anomaly(result.model(key))
            runs[f"run:{detector}"] = {
                "seconds": seconds,
                "peak_gpu_mb": _peak(),
                "default": result.default,
                "sentence": result.sentence,
            }
        out["models"] = runs
    return out


def bench_cifar(folder: Path, deep: bool) -> dict[str, Any]:
    source = folder / "subset.parquet"
    out: dict[str, Any] = {"set": "CIFAR-10, 600 test images (60 per class)"}
    with tempfile.TemporaryDirectory() as workspace:
        started = time.perf_counter()
        images = image_set(source, workspace=workspace, dataset_id="cifar")
        quality = images.quality
        out["import"] = {
            "images": images.n_images,
            "labels": len(images.labels),
            "seconds": round(time.perf_counter() - started, 1),
            "quality": quality.sentence if quality else None,
            "notes": quality.notes if quality else [],
        }
        _peak_reset()
        started = time.perf_counter()
        found = evaluate_image_set(images, deep=None if deep else False)
        result = found.result
        out["classification"] = {
            "split": result.split.code,
            "default": result.default,
            "seconds": round(time.perf_counter() - started, 1),
            "peak_gpu_mb": _peak(),
            "models": {
                m.key: {
                    "balanced_accuracy": round(m.classification.balanced_accuracy, 4),
                    "accuracy": round(m.classification.accuracy, 4),
                    "majority": round(m.classification.baseline.accuracy, 4),
                }
                for m in result.models
                if m.classification is not None
            },
            "sentence": result.sentence,
        }
        from quantilo.interpret import extras
        from quantilo.vision import embed_classic as ec

        space = ec.classic_space(ec.image_set_sources(images))
        labels = images.table["label"].to_pylist()
        maps: dict[str, Any] = {}
        for projection in ("pca", "pacmap"):
            if projection == "pacmap" and not extras.check("pacmap").available:
                maps[projection] = extras.check("pacmap").reason
                continue
            started = time.perf_counter()
            drawn = ec.image_map(space, labels, projection=projection)
            maps[projection] = {
                "seconds": round(time.perf_counter() - started, 1),
                "sentence": drawn.sentence,
                "neighbours_same": drawn.reading.neighbours_same
                if drawn.reading
                else None,
            }
        out["map"] = maps
    return out


def bench_quality() -> dict[str, Any]:
    """The quality imagette (120 T8): every rule must find exactly its cases."""
    from quantilo.vision.quality import RULES
    from tests.vision_fixtures import (
        IMAGETTE_RULES,
        IMAGETTE_ZIP_SKIPPED,
        imagette_folder,
        imagette_zip,
        quality_imagette,
        rule_confusion,
    )

    started = time.perf_counter()
    imagette = quality_imagette()
    out: dict[str, Any] = {
        "set": "quality imagette (synthetic, tests/vision_fixtures.py)",
        "files": len(imagette.files),
        "bytes": sum(len(b) for b in imagette.files.values()),
        "build_seconds": round(time.perf_counter() - started, 1),
    }
    with tempfile.TemporaryDirectory() as workspace:
        root = Path(workspace)
        sources = {
            "folder": imagette_folder(root / "set", imagette),
            "zip": imagette_zip(root / "imagette.zip", imagette),
        }
        for kind, source in sources.items():
            started = time.perf_counter()
            images = image_set(
                source,
                workspace=root / f"ws-{kind}",
                dataset_id=kind,
                thumbnails=False,
                workers=1,
            )
            seconds = round(time.perf_counter() - started, 1)
            table = images.table
            found = dict(
                zip(
                    table["rel_path"].to_pylist(),
                    table["issues"].to_pylist(),
                    strict=True,
                )
            )
            expected = {p: c for p, c in imagette.expected.items() if p in found}
            rows = rule_confusion(expected, found, IMAGETTE_RULES)
            report = images.quality
            dataset = (
                {f.code for f in report.findings if not f.rows} if report else set()
            )
            for row in rows:
                row["title"] = RULES[str(row["rule"])].title
                if row["rule"] in imagette.dataset:  # set-level: fired or not
                    row["expected"], row["found"] = 1, int(row["rule"] in dataset)
            skipped = {s.rel_path: s.code for s in images.skipped}
            out[kind] = {
                "images": images.n_images,
                "seconds": seconds,
                "sentence": report.sentence if report else None,
                "notes": report.notes if report else [],
                "rules": rows,
                "exact": all(not r["missed"] and not r["extra"] for r in rows)
                and all(r["expected"] == r["found"] for r in rows)
                and (kind == "folder" or skipped == IMAGETTE_ZIP_SKIPPED),
                "skipped": skipped,
            }
    return out


def print_quality(result: dict[str, Any]) -> None:
    """The per-rule table, for tasks.md."""
    for kind in ("folder", "zip"):
        part = result[kind]
        print()
        print(
            f"{kind}: {part['images']} images, {part['seconds']} s, exact: {part['exact']}"
        )
        print(
            f"{'rule':<8} {'title':<28} {'expected':>8} {'found':>6} {'missed':>6} "
            f"{'extra':>6}"
        )
        for row in part["rules"]:
            print(
                f"{row['rule']:<8} {row['title']:<28} {row['expected']:>8} {row['found']:>6} "
                f"{len(row['missed']):>6} {len(row['extra']):>6}"
            )
        for member, code in part["skipped"].items():
            print(f"  skipped {member}: {code}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--only", choices=["metal_nut", "cifar10", "quality"], action="append"
    )
    parser.add_argument("--detectors", default="patchcore_ae")
    parser.add_argument("--classic", action="store_true", help="no DINOv2 on CIFAR-10")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    chosen = args.only or ["metal_nut", "cifar10"]
    results: dict[str, Any] = {}
    if "metal_nut" in chosen:
        detectors = [d for d in args.detectors.split(",") if d]
        results["metal_nut"] = bench_metal_nut(args.data / "mvtec-metal_nut", detectors)
    if "cifar10" in chosen:
        results["cifar10"] = bench_cifar(
            args.data / "cifar10-600", deep=not args.classic
        )
    exact = True
    if "quality" in chosen:
        results["quality"] = bench_quality()
        exact = (
            results["quality"]["folder"]["exact"] and results["quality"]["zip"]["exact"]
        )
    text = json.dumps(results, indent=2, ensure_ascii=False, default=str)
    print(text)
    if "quality" in chosen:
        print_quality(results["quality"])
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
