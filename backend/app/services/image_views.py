"""The Map (109) and Prepare (110) tabs of an image set (study 2026-11-11 vision-et-xai §§ 4.4, 5).

- **Map**: the space of each method (classic descriptors, or a frozen network of 114) is built in
  the background, right after the import for the classic one (R-VMAP-10), and kept as
  `workspace/images/<id>/embeddings/<method>.parquet` (rel_path + 50 components). A map (2D
  coordinates, groups, the reading of the labels) is computed from it on demand and kept in memory.
  A reimport deletes the set's folder, so the space is built again.
- **Weights** of the deep methods are downloaded only after the page asked the user (size and
  licence shown, R-VDP-02), in a background thread.
- **Semantic near-copies** (R-VQ-17, Quality > Advanced, b2): only when the user asks, the DINOv2
  vectors of every image (the per-image cache of the models, `embeddings/dinov2_s14.parquet`: a
  trained model fills it, and this fills it for the models), then the pairs at a cosine of 0.95 or
  more that the hashes of R-VQ-10 did not group. Kept in `datasets/<id>/images/semantic.json` with
  the table's stamp.
- **Prepare**: previews of one step on one image (`quantilo.vision.transforms`), PNG as `data:` URLs
  with the measures of both sides, kept in a cache of 64; the recipe is a JSON file of the dataset
  (`datasets/<id>/images/prepare.json`), settings only (R-VPR-03).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from quantilo.errors import NotFoundError

from app.core.errors import AppError
from app.core.storage import Workspace
from app.services.datasets.service import DatasetService
from app.services.image_sets import FORMAT, ImageSetService

logger = logging.getLogger(__name__)

CLASSIC = "classic"
MAP_CACHE = 16
PREVIEW_CACHE = 64  # § 9.2 of the study
DISPLAY_SIDE = 640  # a preview larger than this is sent scaled down (geometry kept)
BORDER_SAMPLE = 24  # R-VPR-06: thumbnails looked at for uniform borders
MEASURED = ("brightness", "contrast", "sharpness", "saturation", "colourfulness", "entropy")
BuildState = Literal["missing", "running", "ready", "failed"]
SEMANTIC = "semantic"  # R-VQ-17: its build is kept beside the maps' (one per dataset)
SEMANTIC_NETWORK = "dinov2_s14"


@dataclass
class Build:
    state: BuildState = "running"
    done: int = 0
    total: int = 0
    message: str = "Starting"
    error: str | None = None
    code: str | None = None
    started: float = field(default_factory=time.time)
    stop: bool = False
    long_run: str | None = None  # R-VDP-05: the estimate when the user must confirm


@dataclass
class Download:
    state: Literal["running", "failed", "done"] = "running"
    received: int = 0
    total: int = 0
    error: str | None = None


class _Lru(OrderedDict[Any, Any]):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def keep(self, key: Any, value: Any) -> Any:
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.size:
            self.popitem(last=False)
        return value

    def find(self, key: Any) -> Any:
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None


def _stamp(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _png(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=False, compress_level=3)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _display(image: Any) -> Any:
    from PIL import Image

    if max(image.size) <= DISPLAY_SIDE:
        return image
    copy = image.copy()
    copy.thumbnail((DISPLAY_SIDE, DISPLAY_SIDE), Image.Resampling.LANCZOS)
    return copy


def _measures(image: Any) -> dict[str, float]:
    from quantilo.vision.measures import measure

    found = measure(image)
    return {name: round(float(getattr(found, name)), 4) for name in MEASURED}


class ImageViewService:
    def __init__(
        self, workspace: Workspace, datasets: DatasetService, image_sets: ImageSetService
    ) -> None:
        self.workspace = workspace
        self.datasets = datasets
        self.image_sets = image_sets
        self._lock = threading.Lock()
        self._builds: dict[tuple[str, str], Build] = {}
        self._downloads: dict[str, Download] = {}
        self._maps = _Lru(MAP_CACHE)
        self._previews = _Lru(PREVIEW_CACHE)
        self._borders = _Lru(MAP_CACHE)
        # Settings > Compute "Use the GPU when available" (set by the vision module's setup)
        self.allow_gpu: Callable[[], bool] = lambda: True
        datasets.after_import.append(self.after_import)

    # --- paths and table -----------------------------------------------------------------------

    def space_path(self, dataset_id: str, method: str) -> Path:
        # `map-`: `embeddings/<method>.parquet` is the per-image vector cache of the models (112);
        # sharing its name, a trained model overwrote the map's space (found in the browser, 119)
        return self.image_sets.images_dir(dataset_id) / "embeddings" / f"map-{method}.parquet"

    def recipe_path(self, dataset_id: str) -> Path:
        return self.workspace.dataset_dir(dataset_id) / "images" / "prepare.json"

    def _table(self, dataset_id: str, columns: list[str], *, ready: bool = True) -> Any:
        """The asked columns of the image table (`ready=False`: during the import's last steps,
        when the table is written but the dataset not marked ready yet)."""
        import pyarrow.parquet as pq

        record = self.image_sets._ready(dataset_id) if ready else self.datasets.get(dataset_id)
        names = set(pq.read_schema(record.storage_path).names)
        return pq.read_table(record.storage_path, columns=[c for c in columns if c in names])

    # --- methods and weights -------------------------------------------------------------------

    def methods(self) -> dict[str, Any]:
        from quantilo.vision import deep

        found = deep.status()
        rows = [
            {
                "key": CLASSIC,
                "title": "Classic descriptors",
                "detail": "Colour (HSV histogram), shape (HOG) and texture (LBP): 2,072 numbers, "
                "no weights, the same numbers in the browser.",
                "level": "first",
                "dims": 2072,
                "available": True,
                "downloaded": True,
                "size": None,
                "licence": None,
                "note": None,
                "reason": None,
            }
        ]
        for backbone in found.backbones:
            rows.append(
                {
                    "key": backbone.key,
                    "title": backbone.title,
                    "detail": f"A frozen network: {backbone.dims} numbers per image.",
                    "level": backbone.level,
                    "dims": backbone.dims,
                    "available": backbone.available,
                    "downloaded": backbone.downloaded,
                    "size": backbone.size,
                    "licence": backbone.licence,
                    "note": backbone.note or None,
                    "reason": backbone.reason,
                }
            )
        downloads = {key: asdict(value) for key, value in self._downloads.items()}
        return {
            "installed": found.installed,
            "reason": found.reason,
            "install": deep.INSTALL,
            "methods": rows,
            "downloads": downloads,
            "projections": self.projections(),
        }

    @staticmethod
    def projections() -> list[dict[str, Any]]:
        """c3, R-VMAP-12: PaCMAP and TriMAP, greyed with their reason when the extra `interpret`
        cannot run here (R-XAI-10); nothing is imported to know it."""
        from quantilo.interpret import extras

        out = []
        for key in ("pacmap", "trimap"):
            found = extras.check(key)
            out.append(
                {
                    "key": key,
                    "title": found.label,
                    "licence": extras.METHODS[key].license,
                    "available": found.available,
                    "reason": found.reason,
                    "install": found.install,
                }
            )
        return out

    def download(self, key: str) -> dict[str, Any]:
        """R-VDP-02: the page asked the user first (size and licence shown)."""
        from quantilo.vision import deep
        from quantilo.vision.deep import weights

        if key not in weights.MANIFEST:
            raise NotFoundError("Weights")
        if not deep.installed():
            raise AppError(deep.status().reason or "", code="vision_extra", status_code=409)
        with self._lock:
            current = self._downloads.get(key)
            if current is None or current.state != "running":
                spec = weights.MANIFEST[key]
                state = Download(total=spec.size)
                self._downloads[key] = state

                def progress(have: int, total: int) -> None:
                    state.received, state.total = have, total

                def run() -> None:
                    try:
                        weights.download(key, confirm=True, progress=progress)
                        state.state, state.received = "done", state.total
                    except Exception as error:  # shown as is: network, size or SHA-256
                        state.state, state.error = "failed", str(error)

                threading.Thread(target=run, daemon=True, name=f"weights-{key}").start()
        return self.methods()

    # --- map -----------------------------------------------------------------------------------

    def after_import(self, dataset_id: str, _path: str) -> None:
        """R-VMAP-10: the classic space is built right after an import, in the background."""
        try:
            record = self.datasets.get(dataset_id)
        except Exception:
            return
        if record.format != FORMAT:
            return
        if not self.space_path(dataset_id, CLASSIC).is_file():
            self._start(dataset_id, CLASSIC, confirm_long=False)

    def _state(self, dataset_id: str, method: str) -> Build | None:
        build = self._builds.get((dataset_id, method))
        if build is not None and build.state in ("running", "failed"):
            return build
        return None

    def build(self, dataset_id: str, method: str, confirm_long: bool = False) -> dict[str, Any]:
        self.image_sets._ready(dataset_id)
        self._check_method(method)
        if self.space_path(dataset_id, method).is_file():
            return self.map(dataset_id, method)
        self._start(dataset_id, method, confirm_long=confirm_long)
        return self.map(dataset_id, method)

    def cancel(self, dataset_id: str, method: str) -> dict[str, Any]:
        build = self._builds.get((dataset_id, method))
        if build is not None and build.state == "running":
            build.stop = True
        return self.map(dataset_id, method)

    def _check_method(self, method: str) -> None:
        if method == CLASSIC:
            return
        from quantilo.vision import deep
        from quantilo.vision.deep.backbones import BACKBONES

        if method not in BACKBONES:
            raise AppError(
                f"Unknown map method {method!r}.", code="unknown_method", status_code=422
            )
        row = deep.status().backbone(method)
        if not row.available:
            code = "vision_extra" if not deep.installed() else "weights_missing"
            raise AppError(row.reason or "", code=code, status_code=409)

    def _start(self, dataset_id: str, method: str, *, confirm_long: bool) -> None:
        with self._lock:
            current = self._builds.get((dataset_id, method))
            if current is not None and current.state == "running":
                return
            build = Build()
            self._builds[(dataset_id, method)] = build
        self.datasets.jobs.submit(
            lambda: self._run(dataset_id, method, build, confirm_long=confirm_long)
        )

    def _run(self, dataset_id: str, method: str, build: Build, *, confirm_long: bool) -> None:
        from quantilo.vision import embed_classic as ec

        try:
            table = self._table(dataset_id, ["rel_path", "stored_path", "decode_ok"], ready=False)
            store = self.image_sets.images_dir(dataset_id)
            keys = [str(v) for v in table["rel_path"].to_pylist()]
            stored = table["stored_path"].to_pylist()
            decoded = table["decode_ok"].to_pylist()
            sources: list[Path | None] = [
                (store / s) if ok and s else None for s, ok in zip(stored, decoded, strict=True)
            ]
            build.total = len(sources)

            def progress(done: int, total: int) -> None:
                build.done, build.total = done, total
                build.message = f"Describing images: {done:,} of {total:,}"

            def stopped() -> bool:
                return build.stop

            if method == CLASSIC:
                space = ec.classic_space(sources, progress=progress, cancel=stopped)
            else:
                space = self._deep_space(method, sources, progress, stopped, confirm_long)
            build.message = "Saving the map"
            ec.save_space(self.space_path(dataset_id, method), keys, space)
            build.state = "ready"
        except Exception as error:  # every failure is said on the tab
            build.state = "failed"
            build.code = type(error).__name__
            build.error = str(error) or type(error).__name__
            if build.code not in ("Cancelled", "LongRun", "NotFoundError", "AppError"):
                logger.exception("Map build failed for %s (%s)", dataset_id, method)

    def _deep_space(
        self,
        method: str,
        sources: list[Path | None],
        progress: Callable[[int, int], None],
        stopped: Callable[[], bool],
        confirm_long: bool,
    ) -> Any:
        import numpy as np
        from quantilo.vision import embed_classic as ec
        from quantilo.vision.deep import embed

        present = [i for i, s in enumerate(sources) if s is not None]
        found = embed.embed_images(
            [sources[i] for i in present],  # type: ignore[misc]
            method,
            confirm_long=confirm_long,
            progress=progress,
            cancel=stopped,
        )
        vectors = np.full((len(sources), found.dims), np.nan, dtype=np.float32)
        vectors[present] = found.vectors
        return ec.reduce(vectors, method=method, title=found.title, families=None)

    def map(
        self,
        dataset_id: str,
        method: str = CLASSIC,
        projection: str = "pca",
        clustering: str = "kmeans",
        k: int | None = None,
    ) -> dict[str, Any]:
        from quantilo.vision import embed_classic as ec

        record = self.image_sets._ready(dataset_id)
        if projection not in ("pca", "tsne", "pacmap", "trimap") or clustering not in (
            "kmeans",
            "hdbscan",
        ):
            raise AppError("Unknown projection or clustering.", code="invalid_map", status_code=422)
        if projection in ("pacmap", "trimap"):  # R-VMAP-12, R-XAI-10
            from quantilo.interpret import extras

            found = extras.check(projection)
            if not found.available:
                raise AppError(found.reason or "", code="interpret_extra", status_code=409)
        if k is not None and not 2 <= k <= 30:
            raise AppError("k must be between 2 and 30.", code="invalid_map", status_code=422)
        path = self.space_path(dataset_id, method)
        build = self._state(dataset_id, method)
        if not path.is_file():
            if build is None:
                return {"state": "missing", "method": method}
            return {
                "state": build.state,
                "method": method,
                "done": build.done,
                "total": build.total,
                "message": build.message,
                "error": build.error,
                "code": build.code,
                "seconds": round(time.time() - build.started, 1),
            }
        key = (dataset_id, method, projection, clustering, k, _stamp(path),
               _stamp(Path(record.storage_path)))  # fmt: skip
        cached = self._maps.find(key)
        if cached is not None:
            return cached
        table = self._table(dataset_id, ["rel_path", "label"])
        keys = [str(v) for v in table["rel_path"].to_pylist()]
        labels = table["label"].to_pylist() if "label" in table.column_names else None
        space = ec.load_space(path, keys)
        found = ec.image_map(
            space,
            labels,
            projection=projection,  # type: ignore[arg-type]
            clustering=clustering,  # type: ignore[arg-type]
            k=k,
        )
        body = {"state": "ready", "method": method, "map": asdict(found), "dims": space.dims}
        return self._maps.keep(key, body)

    # --- semantic near-copies (R-VQ-17) ---------------------------------------------------------

    def semantic_path(self, dataset_id: str) -> Path:
        return self.workspace.dataset_dir(dataset_id) / "images" / "semantic.json"

    def _semantic_base(self, dataset_id: str, storage: Path) -> dict[str, Any]:
        import pyarrow.parquet as pq
        from quantilo.vision import deep
        from quantilo.vision.duplicates import SEMANTIC_COSINE

        row = deep.status().backbone(SEMANTIC_NETWORK)
        cache = (
            self.image_sets.images_dir(dataset_id) / "embeddings" / f"{SEMANTIC_NETWORK}.parquet"
        )
        cached = pq.read_metadata(cache).num_rows if cache.is_file() else 0
        return {
            "available": row.available,
            "reason": row.reason,
            "network": row.title,
            "cosine": SEMANTIC_COSINE,
            "cached": cached,
            "images": pq.read_metadata(storage).num_rows,
        }

    def semantic(self, dataset_id: str) -> dict[str, Any]:
        """R-VQ-17: where the search stands, its groups once done. Reads files only."""
        record = self.image_sets._ready(dataset_id)
        storage = Path(record.storage_path)
        base = self._semantic_base(dataset_id, storage)
        build = self._builds.get((dataset_id, SEMANTIC))
        if build is not None and build.state == "running":
            return base | {
                "state": "running",
                "done": build.done,
                "total": build.total,
                "message": build.message,
                "seconds": round(time.time() - build.started, 1),
            }
        path = self.semantic_path(dataset_id)
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("stamp") == _stamp(storage):
                return base | {"state": "ready"} | data["result"]
        if build is not None and build.state == "failed":
            return base | {"state": "failed", "error": build.error, "long_run": build.long_run}
        return base | {"state": "missing"}

    def find_semantic(self, dataset_id: str, confirm_long: bool = False) -> dict[str, Any]:
        """Start the search (the user asked); a second call while it runs does nothing."""
        from quantilo.vision import deep

        self.image_sets._ready(dataset_id)
        row = deep.status().backbone(SEMANTIC_NETWORK)
        if not row.available:
            code = "vision_extra" if not deep.installed() else "weights_missing"
            raise AppError(row.reason or "", code=code, status_code=409)
        with self._lock:
            current = self._builds.get((dataset_id, SEMANTIC))
            if current is None or current.state != "running":
                build = Build()
                self._builds[(dataset_id, SEMANTIC)] = build
                self.datasets.jobs.submit(
                    lambda: self._semantic_run(dataset_id, build, confirm_long=confirm_long)
                )
        return self.semantic(dataset_id)

    def cancel_semantic(self, dataset_id: str) -> dict[str, Any]:
        build = self._builds.get((dataset_id, SEMANTIC))
        if build is not None and build.state == "running":
            build.stop = True
        return self.semantic(dataset_id)

    def _semantic_run(self, dataset_id: str, build: Build, *, confirm_long: bool) -> None:
        import numpy as np
        from quantilo.vision.classic_models import Cancelled, deep_features
        from quantilo.vision.deep.backbones import BACKBONES
        from quantilo.vision.deep.budget import LongRun
        from quantilo.vision.duplicates import DuplicateGroup
        from quantilo.vision.quality import semantic_duplicates

        started = time.perf_counter()
        try:
            record = self.image_sets._ready(dataset_id)
            storage = Path(record.storage_path)
            stamp = _stamp(storage)
            columns = ["image_id", "stored_path", "decode_ok", "label", "split", "duplicate_group"]
            table = self._table(dataset_id, columns)
            n = table.num_rows

            def values(name: str) -> list[Any]:
                return table[name].to_pylist() if name in table.column_names else [None] * n

            store = self.image_sets.images_dir(dataset_id)
            sources: list[Path | None] = [
                (store / s) if ok and s else None
                for s, ok in zip(values("stored_path"), values("decode_ok"), strict=True)
            ]
            build.total = n

            def progress(share: float, message: str) -> None:
                build.done = round(share * n)
                build.message = message

            vectors, _info = deep_features(
                sources,
                [str(v) for v in values("image_id")],
                store / "embeddings" / f"{SEMANTIC_NETWORK}.parquet",
                allow_gpu=self.allow_gpu(),
                confirm_long=confirm_long,
                progress=progress,
                cancel=lambda: build.stop,
            )
            build.message = "Comparing the vectors"
            by_group: dict[int, list[int]] = {}
            for r, g in enumerate(values("duplicate_group")):
                if g is not None:
                    by_group.setdefault(int(g), []).append(r)
            known = [DuplicateGroup(g, "near", tuple(rows), None) for g, rows in by_group.items()]
            found = semantic_duplicates(
                np.asarray(vectors),
                known,
                values("label"),
                values("split"),
                network=BACKBONES[SEMANTIC_NETWORK].title,
            )
            result = {
                "sentence": found.message,
                "groups": found.details["groups"],
                "pairs": found.details["pairs"],
                "compared": found.details["compared"],
                "leaks": found.details["leaks"],
                "conflicts": found.details["conflicts"],
                "style": found.details["style"],
                "seconds": round(time.perf_counter() - started, 1),
            }
            path = self.semantic_path(dataset_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".partial")
            partial.write_text(json.dumps({"stamp": stamp, "result": result}), encoding="utf-8")
            partial.replace(path)
            build.state = "ready"
        except LongRun as error:
            build.state, build.error = "failed", str(error)
            build.long_run = error.estimate.sentence
        except Cancelled:
            build.state, build.error = "failed", "Stopped: nothing was kept."
        except Exception as error:  # said on the tab as it is
            build.state, build.error = "failed", str(error) or type(error).__name__
            if not isinstance(error, AppError):
                logger.exception("Semantic near-copies failed for %s", dataset_id)

    # --- prepare -------------------------------------------------------------------------------

    def _task(self, dataset_id: str) -> str:
        """R-VPR-07: labels -> classification; normal images only -> anomalies; nothing ->
        clusters (the task rule of 112 refines this)."""
        table = self._table(dataset_id, ["label", "is_normal"])
        labels = (
            {v for v in table["label"].to_pylist() if v} if "label" in table.column_names else set()
        )
        normal = table["is_normal"].to_pylist() if "is_normal" in table.column_names else []
        if normal and all(v is True for v in normal):
            return "anomaly"
        if any(v is not None for v in normal) and labels:
            return "anomaly" if len(labels) <= 1 else "classification"
        return "classification" if len(labels) >= 2 else "clusters"

    def recipe(self, dataset_id: str) -> dict[str, Any]:
        from quantilo.vision.transforms import Recipe, RecipeError

        self.image_sets._ready(dataset_id)
        task = self._task(dataset_id)
        path = self.recipe_path(dataset_id)
        saved = None
        if path.is_file():
            try:
                saved = Recipe.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (RecipeError, ValueError):
                saved = None
        recipe = saved or Recipe.default(task)
        return {
            "recipe": recipe.to_dict(),
            "default": Recipe.default(task).to_dict(),
            "saved": saved is not None,
            "task": task,
            "borders": self._border_share(dataset_id),
        }

    def save_recipe(self, dataset_id: str, data: dict[str, Any]) -> dict[str, Any]:
        from quantilo.vision.transforms import Recipe, RecipeError

        self.image_sets._ready(dataset_id)
        try:
            recipe = Recipe.from_dict(data)
        except RecipeError as error:
            raise AppError(str(error), code="invalid_recipe", status_code=422) from error
        path = self.recipe_path(dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        partial.write_text(json.dumps(recipe.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        partial.replace(path)
        return self.recipe(dataset_id)

    def _border_share(self, dataset_id: str) -> dict[str, Any]:
        """R-VPR-06: the share of images with a uniform frame, on up to 24 thumbnails (seeded)."""
        import numpy as np
        from PIL import Image
        from quantilo.vision.transforms import has_uniform_border

        record = self.image_sets._ready(dataset_id)
        key = (dataset_id, _stamp(Path(record.storage_path)))
        cached = self._borders.find(key)
        if cached is not None:
            return cached
        table = self._table(dataset_id, ["thumb_path", "decode_ok"])
        paths = table["thumb_path"].to_pylist()
        decoded = table["decode_ok"].to_pylist()
        thumbs = [path for path, ok in zip(paths, decoded, strict=True) if ok and path]
        rng = np.random.default_rng(110)
        picked = sorted(rng.choice(len(thumbs), min(BORDER_SAMPLE, len(thumbs)), replace=False))
        root = self.image_sets.images_dir(dataset_id)
        framed = 0
        for i in picked:
            try:
                with Image.open(root / thumbs[int(i)]) as opened:
                    framed += has_uniform_border(opened.convert("RGB"))
            except OSError:
                continue
        share = framed / len(picked) if picked else 0.0
        body = {"share": round(share, 4), "looked_at": len(picked), "suggest": share >= 0.2}
        return self._borders.keep(key, body)

    def preview(
        self, dataset_id: str, image_id: str, recipe_data: dict[str, Any], step: str, seed: int
    ) -> dict[str, Any]:
        from quantilo.vision.decode import DecodeFailure, try_decode
        from quantilo.vision.transforms import GEOMETRIC, Recipe, RecipeError, preview

        record = self.image_sets._ready(dataset_id)
        try:
            recipe = Recipe.from_dict(recipe_data)
        except RecipeError as error:
            raise AppError(str(error), code="invalid_recipe", status_code=422) from error
        key = (dataset_id, image_id, json.dumps(recipe.to_dict(), sort_keys=True), step, seed,
               _stamp(Path(record.storage_path)))  # fmt: skip
        cached = self._previews.find(key)
        if cached is not None:
            return cached
        path, _ = self.image_sets.file(dataset_id, image_id, "file")
        decoded = try_decode(path)
        if isinstance(decoded, DecodeFailure):
            raise AppError(f"{decoded.code}: {decoded.reason}", code="not_decoded", status_code=422)
        try:
            shown = preview(decoded.image, recipe, step, seed=seed)
        except RecipeError as error:
            raise AppError(str(error), code="invalid_step", status_code=422) from error
        before, after = shown.before, shown.after
        notes = list(decoded.notes) if step == "decode" else []
        notes += shown.notes
        body = {
            "step": step,
            "image_id": image_id,
            "enabled": shown.enabled,
            "geometric": step in GEOMETRIC,
            "before": _png(_display(before)),
            "after": _png(_display(after)),
            "before_size": list(before.size),
            "after_size": list(after.size),
            "before_measures": _measures(before),
            "after_measures": _measures(after),
            "notes": notes,
        }
        return self._previews.keep(key, body)


__all__ = ["CLASSIC", "ImageViewService"]
