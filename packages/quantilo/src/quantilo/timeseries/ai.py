"""AI forecasting models (028 phase B): what they need, whether this machine has it, and their
weights (the optional extra `forecast-ai`: `pip install "quantilo[forecast-ai]"`).

Two kinds of model:
- pretrained (Chronos-Bolt Tiny / Small / Base, Chronos-2, TimesFM 2.5): the code (PyTorch and
  `chronos-forecasting` or `timesfm`) and the weights, a folder downloaded once from the model's
  official Hugging Face repository, at a revision pinned here, every file checked by size and
  SHA-256 (`hub.fetch`). Nothing is downloaded without `download(key, confirm=True)`, never in
  "Never download" mode (`DATA_EXPLORER_OFFLINE=1`, or Settings › Models), never in the
  browser version. The models are then loaded from the local folder only: transformers and
  huggingface_hub are told not to connect (`HF_HUB_OFFLINE`).
- trained (N-HiTS, N-BEATS, `neural.py`): PyTorch only, no weights.

Standard library only at import time: `status()` works without the extra and never imports torch.
"""

import importlib.util
import os
import shutil
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from quantilo import hub

INSTALL = 'pip install "quantilo[forecast-ai]"'
USER_AGENT = "quantilo-forecast-weights/1"
OFFLINE_FILE = "NEVER_DOWNLOAD"  # in the models folder: "Never download" set in Settings
LOADED = 2  # pretrained models kept in memory (Chronos-2 ~0.5 GB, TimesFM ~1 GB)

Family = Literal["chronos_bolt", "chronos_2", "timesfm", "neural"]
State = Literal["ready", "needs_download", "offline", "needs_extra", "browser"]


@dataclass(frozen=True)
class FileSpec:
    name: str
    size: int  # bytes
    sha256: str


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    family: Family
    package: str  # the module imported besides torch
    licence: str
    parameters: str  # "9 M"
    note: str  # one line shown with the model
    repo: str = ""  # Hugging Face model id (pretrained models)
    revision: str = ""  # pinned commit
    files: tuple[FileSpec, ...] = ()
    drivers: bool = False  # takes known-in-advance covariates (Chronos-2)
    gpu: bool = False  # large enough for the GPU to help (when allowed)

    @property
    def pretrained(self) -> bool:
        return bool(self.files)

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def size_text(self) -> str:
        return hub.size_text(self.size) if self.files else ""

    def url(self, name: str) -> str:
        return f"{hub.HF}/{self.repo}/resolve/{self.revision}/{name}"

    @property
    def source(self) -> str:
        return f"{hub.HF}/{self.repo}/tree/{self.revision}" if self.repo else ""


def _bolt(key: str, size: str, revision: str, config: FileSpec, weights: FileSpec, params: str):
    return ModelSpec(
        key,
        f"Chronos-Bolt {size}",
        "chronos_bolt",
        "chronos",
        "Apache-2.0",
        params,
        "Amazon's pretrained model, zero-shot: no training, quantile intervals.",
        f"amazon/chronos-bolt-{size.lower()}",
        revision,
        (config, weights),
        gpu=size == "Base",
    )


# Revisions and digests read from the Hugging Face API on 2026-11-13 (`config.json` hashed from
# its bytes, `model.safetensors` = its Git LFS object id).
MANIFEST: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        _bolt(
            "chronos_bolt_tiny",
            "Tiny",
            "a0e552de83495b5c28c14c71c374f3e33280b340",
            FileSpec(
                "config.json",
                1_120,
                "278f0086733031635fb1c861cb01c1bad6477420c7fcb19381a2993e335785e0",
            ),
            FileSpec(
                "model.safetensors",
                34_622_352,
                "75068728d376d2bec670379eeef4bfb4d24c0cfe24d957451f8d19b447030a32",
            ),
            "9 M",
        ),
        _bolt(
            "chronos_bolt_small",
            "Small",
            "772f3d25d38aec6d914c8949dab4462e2d46f5d8",
            FileSpec(
                "config.json",
                1_121,
                "9ca0ebbe049ccebacb0546361e8aa728f0cf104cd18d1838d78d296aa317bd21",
            ),
            FileSpec(
                "model.safetensors",
                190_888_824,
                "06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd",
            ),
            "48 M",
        ),
        _bolt(
            "chronos_bolt_base",
            "Base",
            "5d9f166d69f47aef3401367a7b842e78fe97b121",
            FileSpec(
                "config.json",
                1_123,
                "46f7e7e695569f67b06178faa4f236c281392945aa2d1bdf844442e57155530b",
            ),
            FileSpec(
                "model.safetensors",
                821_203_576,
                "31f875483a3215bc6880a0837ea608a13ce55f88ad90538c3cdd0b29aeb60b36",
            ),
            "205 M",
        ),
        ModelSpec(
            "chronos_2",
            "Chronos-2",
            "chronos_2",
            "chronos",
            "Apache-2.0",
            "120 M",
            "Amazon's pretrained model with drivers (known-in-advance covariates) and several "
            "series at once.",
            "amazon/chronos-2",
            "29ec3766d36d6f73f0696f85560a422f50e8498c",
            (
                FileSpec(
                    "config.json",
                    1_067,
                    "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031",
                ),
                FileSpec(
                    "model.safetensors",
                    477_930_472,
                    "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42",
                ),
            ),
            drivers=True,
            gpu=True,
        ),
        ModelSpec(
            "timesfm_2p5",
            "TimesFM 2.5",
            "timesfm",
            "timesfm",
            "Apache-2.0",
            "200 M",
            "Google's pretrained model, zero-shot, quantile intervals (PyTorch, without "
            "torch.compile).",
            "google/timesfm-2.5-200m-pytorch",
            "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
            (
                FileSpec(
                    "config.json",
                    475,
                    "cd3315b760d5cc7e278d7afdf41b897031ced888fc6c115bd9b3ac0ea2c47408",
                ),
                FileSpec(
                    "model.safetensors",
                    925_181_104,
                    "2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8",
                ),
            ),
            gpu=True,
        ),
        ModelSpec(
            "nhits",
            "N-HiTS",
            "neural",
            "torch",
            "MIT (this code)",
            "~0.3 M",
            "A small network trained on this series' windows (CPU), coarse to fine.",
        ),
        ModelSpec(
            "nbeats",
            "N-BEATS",
            "neural",
            "torch",
            "MIT (this code)",
            "~0.4 M",
            "A small stack of fully connected blocks trained on this series' windows (CPU).",
        ),
    )
}
PRETRAINED = tuple(key for key, spec in MANIFEST.items() if spec.pretrained)
NEURAL = ("nhits", "nbeats")
KEYS = tuple(MANIFEST)
BOLT = ("chronos_bolt_tiny", "chronos_bolt_small", "chronos_bolt_base")


class AiError(RuntimeError):
    """A model that cannot run here: the message is the reason, shown as it is."""


class ConfirmationRequired(AiError):
    """Nothing is downloaded before the user agrees (name, size, source shown first)."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        super().__init__(
            f"Downloading {spec.title} needs your confirmation: {spec.size_text} "
            f"({spec.licence}) from {spec.source}."
        )


def spec_of(key: str) -> ModelSpec:
    try:
        return MANIFEST[key]
    except KeyError:
        raise KeyError(f"Unknown AI model {key!r}: one of {', '.join(MANIFEST)}.") from None


# --- Where the weights live, and whether the network may be used ---------------------------------


def models_root() -> Path:
    """`QUANTILO_MODELS_DIR` (the app sets it to `<home>/models`), else next to the app's
    embeddings cache, else the user's cache folder. Never the global Hugging Face cache: removing
    the app removes its models."""
    configured = os.environ.get("QUANTILO_MODELS_DIR")
    if configured:
        return Path(configured)
    embeddings = os.environ.get("QUANTILO_EMBEDDINGS_CACHE")
    if embeddings:
        return Path(embeddings).parent.parent / "models"
    return Path.home() / ".cache" / "quantilo" / "models"


def folder(key: str) -> Path:
    """`<models>/forecast/<key>`: the files of one model, under their own names."""
    return models_root() / "forecast" / key


def offline_forced() -> bool:
    """`DATA_EXPLORER_OFFLINE=1` (or `QUANTILO_OFFLINE`): set outside the app, not changeable."""
    for name in ("QUANTILO_OFFLINE", "DATA_EXPLORER_OFFLINE"):
        if os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on"):
            return True
    return False


def offline() -> bool:
    """ "Never download": forced by the environment, or chosen in Settings › Models (a marker file
    in the models folder, so it survives restarts)."""
    return offline_forced() or (models_root() / OFFLINE_FILE).is_file()


def set_offline(value: bool) -> None:
    marker = models_root() / OFFLINE_FILE
    if value:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("Set in Settings › Models: the app never downloads a model.\n")
    else:
        marker.unlink(missing_ok=True)


def in_browser() -> bool:
    return sys.platform == "emscripten"


def _installed(package: str) -> bool:
    if package in sys.modules:
        return True
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def missing_packages(key: str) -> list[str]:
    """Packages a model needs that are not installed (never imported here)."""
    spec = spec_of(key)
    needed = ["torch"] if spec.package == "torch" else ["torch", spec.package]
    names = {"chronos": "chronos-forecasting", "timesfm": "timesfm", "torch": "torch"}
    return [names[p] for p in needed if not _installed(p)]


def downloaded(key: str) -> bool:
    """Every file there with the right size (cheap; the SHA-256 was checked when downloaded)."""
    spec = spec_of(key)
    if not spec.pretrained:
        return True
    where = folder(key)
    return all(
        (where / f.name).is_file() and (where / f.name).stat().st_size == f.size for f in spec.files
    )


@dataclass
class ModelStatus:
    key: str
    title: str
    family: Family
    pretrained: bool
    state: State
    reason: str | None  # why it cannot run now (English; the app shows its own texts by state)
    size: int  # bytes to download (0: trained here)
    size_text: str
    licence: str
    parameters: str
    note: str
    source: str  # Hugging Face repository at the pinned revision
    folder: str
    drivers: bool
    missing: list[str]  # packages to install
    install: str | None


def status(key: str) -> ModelStatus:
    spec = spec_of(key)
    missing = [] if in_browser() else missing_packages(key)
    state: State
    if in_browser():
        state = "browser"
        reason = f"{spec.title} needs PyTorch, which the browser version cannot run: use the app."
    elif missing:
        state = "needs_extra"
        reason = f"Needs the AI forecasting extra ({', '.join(missing)}): {INSTALL}."
    elif downloaded(key):
        state, reason = "ready", None
    elif offline():
        state = "offline"
        reason = (
            f"Never download is on: copy the {spec.title} folder (config.json and "
            f"model.safetensors from {spec.repo}) into {folder(key)}."
        )
    else:
        state = "needs_download"
        reason = f"Download it first: {spec.title}, {spec.size_text}, once, from {spec.source}."
    return ModelStatus(
        key,
        spec.title,
        spec.family,
        spec.pretrained,
        state,
        reason,
        spec.size,
        spec.size_text,
        spec.licence,
        spec.parameters,
        spec.note,
        spec.source,
        str(folder(key)) if spec.pretrained else "",
        spec.drivers,
        missing,
        INSTALL if missing else None,
    )


def statuses() -> list[ModelStatus]:
    return [status(key) for key in MANIFEST]


def require(key: str) -> ModelSpec:
    """The spec, or `AiError` with the reason the model cannot run (shown in its row)."""
    found = status(key)
    if found.state != "ready":
        raise AiError(found.reason or f"{found.title} is not available.")
    return spec_of(key)


# --- Coded rules: which AI model suits the series (028 "Quand proposer quoi") --------------------

SHORT_PERIODS = 30  # fewer periods (or under two cycles): too short to train models
LONG_PERIODS = 500  # from there a network has enough windows to learn from (neural.MIN_PERIODS)
BOLT_PREFERENCE = ("chronos_bolt_small", "chronos_bolt_tiny", "chronos_bolt_base")


@dataclass
class ModelSuggestion:
    kind: Literal["short_series", "long_series"]
    models: list[str]  # ticked by the suggestion's button
    periods: int  # in the series
    threshold: int  # the rule's limit
    reason: str  # English; the app writes its own sentence from the numbers


def refusal(key: str, n: int) -> str | None:
    """Why an AI model is not run on a series of n periods (None: it is, or not an AI model):
    the extra or the weights missing, or a trained network on a short series."""
    if key not in MANIFEST:
        return None
    try:
        require(key)
    except AiError as error:
        return str(error)
    if key not in NEURAL:
        return None
    from quantilo.timeseries import neural

    return neural.refusal(n)


def suggestions(n: int, season: int | None, chosen: list[str]) -> list[ModelSuggestion]:
    """Short series (under 30 periods or two cycles): a pretrained model forecasts it without
    training (the smallest Chronos-Bolt already downloaded, else Small). Long series (500
    periods or more): N-HiTS. Nothing when such a model is already chosen."""
    found: list[ModelSuggestion] = []
    needed = max(SHORT_PERIODS, 2 * season) if season else SHORT_PERIODS
    if n < needed and not any(key in chosen for key in PRETRAINED):
        ready = [key for key in BOLT_PREFERENCE if downloaded(key)]
        pick = ready[0] if ready else BOLT_PREFERENCE[0]
        found.append(
            ModelSuggestion(
                "short_series",
                [pick],
                n,
                needed,
                f"Only {n} periods: too short to train models (they need {needed}). A pretrained "
                f"model forecasts it without training: try {MANIFEST[pick].title}.",
            )
        )
    if n >= LONG_PERIODS and not any(key in chosen for key in NEURAL):
        found.append(
            ModelSuggestion(
                "long_series",
                ["nhits"],
                n,
                LONG_PERIODS,
                f"{n} periods: enough windows to train a network. N-HiTS learns this series' "
                "patterns on the CPU.",
            )
        )
    return found


# --- Download and delete -------------------------------------------------------------------------


def download(
    key: str,
    *,
    confirm: bool = False,
    progress: Callable[[int, int], None] | None = None,
    opener: hub.Opener | None = None,
    cancelled: Callable[[], bool] | None = None,
    pause: float = 2.0,
) -> Path:
    """Fetch a pretrained model's files once into `folder(key)` and return it. `confirm=True` is
    the user's agreement after seeing the size and source; already there: no connection. A
    partial file left by a stopped or interrupted download is resumed. `progress(have, total)`
    counts bytes over every file."""
    spec = spec_of(key)
    if not spec.pretrained:
        raise AiError(f"{spec.title} is trained on your series: there is nothing to download.")
    if in_browser():
        raise AiError("Models cannot be downloaded in the browser version.")
    if downloaded(key):
        return folder(key)
    if offline():
        raise AiError(status(key).reason or "Never download is on.")
    if not confirm:
        raise ConfirmationRequired(spec)
    where = folder(key)
    done = 0
    for file in spec.files:
        target = where / file.name
        if target.is_file() and target.stat().st_size == file.size:
            done += file.size
            continue

        def counted(have: int, _total: int, base: int = done) -> None:
            if progress is not None:
                progress(base + have, spec.size)

        try:
            hub.fetch(
                spec.url(file.name),
                file.size,
                file.sha256,
                target,
                title=f"{spec.title} ({file.name})",
                user_agent=USER_AGENT,
                progress=counted,
                opener=opener,
                pause=pause,
                resume=True,
                cancelled=cancelled,
            )
        except hub.DownloadError as error:
            raise AiError(str(error)) from error
        done += file.size
    return where


def delete(key: str) -> bool:
    """Remove a downloaded model's folder (and what it holds in memory). True when there was one."""
    spec = spec_of(key)
    if not spec.pretrained:
        return False
    with _LOCK:
        _PIPELINES.pop(key, None)
    where = folder(key)
    if not where.exists():
        return False
    shutil.rmtree(where)
    return True


# --- Device and loading --------------------------------------------------------------------------

_allow_gpu: Callable[[], bool] = lambda: False  # noqa: E731 - the library stays on the CPU


def set_gpu_policy(allow: Callable[[], bool]) -> None:
    """The app's Settings › Compute › "Use the GPU" (023b): asked at each load."""
    global _allow_gpu
    _allow_gpu = allow


def device(key: str) -> str:
    """CPU for the small models (Tiny, Small, the trained networks: faster there than copying to
    a card); "cuda" for the large ones when a card is usable and the GPU is allowed."""
    if not spec_of(key).gpu or not _allow_gpu():
        return "cpu"
    try:
        import torch  # pyright: ignore[reportMissingImports]

        return "cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu"
    except Exception:
        return "cpu"


_PIPELINES: OrderedDict[str, Any] = OrderedDict()
_LOCK = threading.Lock()


def _no_network() -> None:
    """The local folder only: neither transformers nor huggingface_hub may connect."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def pipeline(key: str, loader: Callable[[ModelSpec, Path, str], Any] | None = None) -> Any:
    """The loaded model of a pretrained key, kept for the next calls (at most `LOADED`).
    `loader(spec, folder, device)` replaces the real loading (tests)."""
    spec = require(key)
    where = folder(key)
    with _LOCK:
        if key in _PIPELINES:
            _PIPELINES.move_to_end(key)
            return _PIPELINES[key]
        _no_network()
        found = (loader or _load)(spec, where, device(key))
        _PIPELINES[key] = found
        while len(_PIPELINES) > LOADED:
            _PIPELINES.popitem(last=False)
        return found


def _load(spec: ModelSpec, where: Path, on: str) -> Any:
    import torch  # pyright: ignore[reportMissingImports]

    if spec.family == "timesfm":
        import timesfm  # pyright: ignore[reportMissingImports]

        model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
        model.model.device = torch.device(on)
        model.model.device_count = 1
        model.load_checkpoint(str(where / "model.safetensors"), torch_compile=False)
        model.model.to(on).eval()
        return model
    from chronos import BaseChronosPipeline  # pyright: ignore[reportMissingImports]
    from transformers.utils import logging  # pyright: ignore[reportMissingImports]

    logging.set_verbosity_error()
    logging.disable_progress_bar()
    return BaseChronosPipeline.from_pretrained(
        str(where), device_map=on, dtype=torch.float32, local_files_only=True
    )


def forget() -> None:
    """Drop the models held in memory (tests, Settings › Models › Delete)."""
    with _LOCK:
        _PIPELINES.clear()
