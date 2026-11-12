# ruff: noqa: RUF001, RUF003  (›, × are the app's notation)
# pyright: reportAttributeAccessIssue=false
"""Sentence embeddings for free text (091, optional extra `embed`).

TF-IDF + SVD (052) stays the default. Embeddings are an Advanced choice per text column, offered
only when the extra is installed (`pip install "quantilo[embed]"`: model2vec, a static
embedding model, CPU, no torch) and greyed with the reason otherwise. The model is
`minishlab/potion-multilingual-128M` (MIT, about 0.5 GB on disk, 101 languages); its weights are
downloaded once, after the user confirms, never during a training.

- R-FUS-01: texts are encoded in batches, through the cache (embed_cache, R-EMB), then reduced to
  32 numbers by a PCA learnt on the training rows.
- R-FUS-02: embeddings are kept only if, in a 3-fold cross-validation on the training rows, a
  linear model on them beats the same model on TF-IDF by more than one standard deviation of the
  folds' differences; otherwise the column falls back to TF-IDF, and the app says so (Shi et al.,
  2021: on text-heavy tables, stronger text encoders do not always pay).
- R-EMB-05: a saved model keeps its `model_key`; predicting with it refuses, and says why, when
  that model is not on this machine.

Nothing here imports model2vec until a column asks for it (lazy): the engine and the browser
version run without it."""

import importlib.util
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from quantilo.ml import text as textual
from quantilo.ml.embed_cache import EmbeddingCache

MODEL = "minishlab/potion-multilingual-128M"
MODEL_SIZE = "about 0.5 GB"
MODEL_LICENSE = "MIT"
DIMENSIONS = 32  # R-FUS-01: PCA width
FOLDS = 3  # R-FUS-02
COMPARE_ROWS = 20_000  # R-FUS-02: training rows the comparison runs on at most
BATCH = 1024  # texts per encoder call
SEED = 42
INSTALL = 'pip install "quantilo[embed]"'

# Tests and embedders other than model2vec: name → (model_key, encode). Never filled by the app.
_ENCODERS: dict[str, tuple[str, Callable[[list[str]], np.ndarray]]] = {}
_LOADED: dict[str, Any] = {}


def register(name: str, model_key: str, encode: Callable[[list[str]], np.ndarray]) -> None:
    """Make `name` usable without model2vec (tests, or a local encoder of the user's)."""
    _ENCODERS[name] = (model_key, encode)


def cache_root() -> Path:
    """R-EMB-03: `QUANTILO_EMBEDDINGS_CACHE` (the app points it to its workspace), else the
    user's cache folder."""
    configured = os.environ.get("QUANTILO_EMBEDDINGS_CACHE")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "quantilo" / "embeddings"


def weights_folder() -> Path:
    return cache_root().parent / "models" / MODEL.replace("/", "--")


@dataclass
class Status:
    available: bool  # the extra is installed and the weights are on this machine
    installed: bool
    downloaded: bool
    reason: str | None  # why the option is greyed (None when available)
    model: str = MODEL
    size: str = MODEL_SIZE
    license: str = MODEL_LICENSE


def installed() -> bool:
    return importlib.util.find_spec("model2vec") is not None


def downloaded() -> bool:
    return (weights_folder() / "config.json").exists()


def status(name: str | None = None) -> Status:
    """Whether "Sentence embeddings" can be chosen here, and if not, why."""
    name = name or MODEL
    if name in _ENCODERS:
        return Status(True, True, True, None, model=name)
    if sys.platform == "emscripten":
        return Status(
            False, False, False, "Not available in the browser version: it needs the desktop app."
        )
    if not installed():
        return Status(
            False,
            False,
            False,
            f"Needs the optional extra `embed` ({INSTALL}), then the model ({MODEL_SIZE}).",
        )
    if not downloaded():
        return Status(
            False,
            True,
            False,
            f"Download the model first ({MODEL}, {MODEL_SIZE}): Machine learning › Advanced.",
        )
    return Status(True, True, True, None)


def download() -> Path:
    """Fetch the weights once (the caller has asked the user first, size shown)."""
    if not installed():
        raise RuntimeError(f"The embed extra is not installed: {INSTALL}.")
    from model2vec import StaticModel  # type: ignore[import-not-found]

    model = StaticModel.from_pretrained(MODEL)
    folder = weights_folder()
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(folder))
    return folder


def _model2vec(name: str) -> tuple[str, Callable[[list[str]], np.ndarray]]:
    if name not in _LOADED:
        if not downloaded():
            raise RuntimeError(
                f"The text model {name} is not on this machine (R-EMB-05): download it in "
                "Machine learning › Advanced, or train again with TF-IDF."
            )
        from model2vec import StaticModel  # type: ignore[import-not-found]

        _LOADED[name] = StaticModel.from_pretrained(str(weights_folder()))
    model = _LOADED[name]
    folder = weights_folder()
    weights = folder / "model.safetensors"
    stamp = weights.stat().st_size if weights.exists() else 0
    dims = int(getattr(model, "dim", 0) or 0)
    key = f"{name}|{stamp}|{dims}|normalize=True|max_length=512|float32"  # R-EMB-02

    def encode(items: list[str]) -> np.ndarray:
        return np.asarray(model.encode(items, max_length=512), dtype=np.float32)

    return key, encode


def encoder(name: str | None = None) -> tuple[str, Callable[[list[str]], np.ndarray]]:
    """(model_key, encode) of a model, loaded on first use."""
    name = name or MODEL
    if name in _ENCODERS:
        return _ENCODERS[name]
    if not installed():
        raise RuntimeError(f"Sentence embeddings need the embed extra ({INSTALL}).")
    return _model2vec(name)


def encode(values: Any, name: str | None = None) -> tuple[np.ndarray, str]:
    """One vector per text, in batches, through the cache (R-FUS-01, R-EMB-01)."""
    model_key, function = encoder(name)
    items = [str(t) for t in textual.texts(values)]

    def batched(chunk: list[str]) -> np.ndarray:
        parts = [function(chunk[i : i + BATCH]) for i in range(0, len(chunk), BATCH)]
        return np.vstack(parts) if parts else np.zeros((0, 1), dtype=np.float32)

    vectors = EmbeddingCache(cache_root(), model_key).encode(items, batched)
    return vectors, model_key


class EmbeddingFeatures(BaseEstimator, TransformerMixin):
    """One text column → sentence embeddings → PCA (R-FUS-01), kept only if they beat TF-IDF
    (R-FUS-02); otherwise the fitted TF-IDF step (`fallback_`) is used instead."""

    def __init__(
        self,
        analyzer: textual.Analyzer = "words",
        model: str | None = None,  # None: MODEL
        components: int = DIMENSIONS,
        compare: bool = True,
        seed: int = SEED,
    ) -> None:
        self.analyzer: textual.Analyzer = analyzer
        self.model = model
        self.components = components
        self.compare = compare
        self.seed = seed

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = 1
        self.model_name_ = self.model or MODEL
        vectors, self.model_key_ = encode(X, self.model_name_)
        k = max(1, min(self.components, vectors.shape[1], len(vectors) - 1))
        self.pca_ = PCA(k, random_state=self.seed).fit(vectors)
        self.fallback_ = None
        self.kept_ = "embeddings"
        self.scores_: dict[str, float] = {}
        name = str(getattr(self, "feature_names_in_", ["text"])[0])
        self.outcome_ = (
            f"{name}: sentence embeddings used ({self.model_name_}, {k} numbers, R-FUS-01)."
        )
        if self.compare and y is not None:
            self._compare(X, np.asarray(y), self.pca_.transform(vectors), name)
        self.width_ = (
            int(self.fallback_.width_)
            if self.fallback_ is not None
            else int(self.pca_.n_components_)
        )
        return self

    def _compare(self, X, y: np.ndarray, embedded: np.ndarray, name: str) -> None:
        """R-FUS-02: linear model, 3 folds, on the same rows; embeddings must win by > 1 sd."""
        rows = np.arange(len(y))
        if len(rows) > COMPARE_ROWS:
            rows = np.sort(np.random.default_rng(self.seed).choice(len(y), COMPARE_ROWS, False))
        target = y[rows]
        classify = target.dtype.kind not in "fc" or len(np.unique(target)) <= 20
        if classify:
            _, counts = np.unique(target, return_counts=True)
            if counts.min() < FOLDS:
                return
            folds: Any = StratifiedKFold(FOLDS, shuffle=True, random_state=self.seed)
            proxy: Any = LogisticRegression(max_iter=500)
            scoring = "balanced_accuracy"
        else:
            folds = KFold(FOLDS, shuffle=True, random_state=self.seed)
            proxy = Ridge()
            scoring = "r2"
        tfidf = textual.TextFeatures(self.analyzer).fit(textual.texts(X)[rows])
        sparse = tfidf.transform(textual.texts(X)[rows])
        a = cross_val_score(proxy, embedded[rows], target, cv=folds, scoring=scoring)
        b = cross_val_score(proxy, sparse, target, cv=folds, scoring=scoring)
        gain, spread = float(np.mean(a - b)), float(np.std(a - b))
        self.scores_ = {"embeddings": float(np.mean(a)), "tfidf": float(np.mean(b))}
        shown = f"{scoring.replace('_', ' ')} {np.mean(a):.3f} vs {np.mean(b):.3f}"
        if gain > spread:
            self.outcome_ = (
                f"{name}: sentence embeddings kept, they beat TF-IDF in {FOLDS}-fold "
                f"cross-validation ({shown}, R-FUS-02)."
            )
            return
        self.kept_ = "tfidf"
        self.fallback_ = textual.TextFeatures(self.analyzer).fit(X)
        self.outcome_ = (
            f"{name}: sentence embeddings did not beat TF-IDF by more than the folds' spread "
            f"({shown}): TF-IDF used instead (R-FUS-02)."
        )

    @property
    def vocabulary(self) -> list[str]:
        return self.fallback_.vocabulary if self.fallback_ is not None else []

    def transform(self, X):
        if self.fallback_ is not None:
            return self.fallback_.transform(X)
        vectors, key = encode(X, self.model_name_)
        if key != self.model_key_:
            raise RuntimeError(
                f"This model was trained with {self.model_key_}, the text model here is {key} "
                "(R-EMB-05): train it again."
            )
        return np.asarray(self.pca_.transform(vectors), dtype=float)

    def get_feature_names_out(self, input_features=None):
        prefix = (
            str(input_features[0])
            if input_features is not None and len(input_features)
            else str(getattr(self, "feature_names_in_", ["text"])[0])
        )
        tag = "text" if self.fallback_ is not None else "embed"
        return np.array([f"{prefix}_{tag}{k + 1}" for k in range(self.width_)], dtype=object)


def outcomes(pipeline: Any) -> list[str]:
    """What R-FUS-02 decided for each embedded column of a fitted pipeline (said to the user)."""
    found: list[str] = []
    prepare = getattr(pipeline, "named_steps", {}).get("prepare")
    for name, step, _ in getattr(prepare, "transformers_", []):
        if name == "remainder" or not hasattr(step, "named_steps"):
            continue
        features = step.named_steps.get("text")
        if isinstance(features, EmbeddingFeatures):
            found.append(features.outcome_)
    return found
