# ruff: noqa: RUF002, RUF003  (×, −, – are the usual notation)
# pyright: reportAttributeAccessIssue=false
# (fitted scikit-learn objects: the stubs cannot follow them)
"""Free text as a model input (052): TF-IDF on words (1–2) or characters (3–5), then a truncated
SVD to at most 50 numbers, every model alike. The vocabulary and the SVD are learnt on at most
50 000 training rows (bounded memory, the same code runs in Pyodide: scikit-learn only), then
applied to every row. The step also measures, on its training rows, the share of terms its
vocabulary does not know: the reference the check of new data compares with (037)."""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

Analyzer = Literal["words", "chars"]
SEED = 42
TERMS = 5_000  # vocabulary kept at most (R-TEXT-03)
COMPONENTS = 50  # SVD components at most (R-TEXT-04)
FIT_ROWS = 50_000  # training rows the vocabulary and the SVD are learnt on
MIN_ROWS = 2  # a term seen in fewer rows teaches nothing
MIN_SHARE = 0.0005  # … nor one seen in under 0.05 % of the rows
MAX_SHARE = 0.95  # a term in almost every row distinguishes nothing
MIN_TEXTS = 20  # fewer non-empty texts: no vocabulary emerges (R-SEL-03)
SHORT_WORDS = 3  # a median of at most 3 words: a short label, cut into characters (R-TEXT-02)
PROFILE_ROWS = 5_000


def settings(analyzer: Analyzer) -> dict[str, Any]:
    """How the text is cut: the same settings when fitting and when checking new data."""
    return {
        "analyzer": "char_wb" if analyzer == "chars" else "word",
        "ngram_range": (3, 5) if analyzer == "chars" else (1, 2),
        "lowercase": True,
        "strip_accents": "unicode",
    }


def cutter(analyzer: Analyzer) -> Callable[[str], list[str]]:
    """The function that cuts one text into the terms the vocabulary is made of."""
    return TfidfVectorizer(**settings(analyzer)).build_analyzer()  # type: ignore[return-value]


def texts(values: Any) -> np.ndarray:
    """One text per row: empty values become an empty text (R-TEXT-01)."""
    series = pd.Series(np.asarray(values, dtype=object).ravel())
    return series.where(series.notna(), "").astype(str).to_numpy(dtype=object)


@dataclass
class TextProfile:
    texts: int  # non-empty texts
    median_words: float


def profile(series: pd.Series) -> TextProfile:
    """Non-empty texts and their median number of words (on 5 000 of them at most)."""
    present = series.dropna().astype(str)
    present = present[present.str.strip() != ""]
    count = len(present)
    if count > PROFILE_ROWS:
        present = present.sample(PROFILE_ROWS, random_state=SEED)
    words = present.str.split().str.len()
    return TextProfile(count, float(words.median()) if count else 0.0)


def analyzer_for(text: TextProfile, role: str | None) -> Analyzer:
    """R-TEXT-02: characters for a short label, words otherwise."""
    return "chars" if role == "label" or text.median_words <= SHORT_WORDS else "words"


@dataclass
class Gap:
    unknown: float  # share of the terms the vocabulary does not know
    unmatched: float  # share of the non-empty texts with no known term at all
    rows: np.ndarray  # per text: non-empty and no known term


def gap(values: Iterable[str], cut: Callable[[str], list[str]], known: set[str]) -> Gap:
    """How much of these texts the vocabulary cannot read."""
    total = unknown = filled = unmatched = 0
    flags = []
    for value in values:
        terms = cut(value) if value else []
        hits = sum(term in known for term in terms)
        total += len(terms)
        unknown += len(terms) - hits
        empty = bool(terms) and hits == 0
        filled += bool(terms)
        unmatched += empty
        flags.append(empty)
    return Gap(
        unknown / total if total else 0.0,
        unmatched / filled if filled else 0.0,
        np.asarray(flags, dtype=bool),
    )


def _sample(n: int, limit: int, seed: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))


class TextFeatures(BaseEstimator, TransformerMixin):
    """One text column → TF-IDF → truncated SVD (R-TEXT-03 and R-TEXT-04)."""

    def __init__(
        self,
        analyzer: Analyzer = "words",
        terms: int = TERMS,
        components: int = COMPONENTS,
        fit_rows: int = FIT_ROWS,
        seed: int = SEED,
    ) -> None:
        self.analyzer: Analyzer = analyzer
        self.terms = terms
        self.components = components
        self.fit_rows = fit_rows
        self.seed = seed

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = 1
        values = texts(X)
        picked = values[_sample(len(values), self.fit_rows, self.seed)]
        self.fit_rows_ = len(picked)
        self.vectorizer_, matrix = self._vocabulary(picked)
        size = len(self.vocabulary)
        k = min(self.components, size - 1, len(picked) - 1)
        self.svd_ = None
        if matrix is not None and k >= 1:
            self.svd_ = TruncatedSVD(k, algorithm="randomized", n_iter=5, random_state=self.seed)
            self.svd_.fit(matrix)
        self.width_ = k if self.svd_ is not None else max(1, size)
        found = gap(picked, cutter(self.analyzer), set(self.vocabulary))
        self.unknown_share_, self.unmatched_share_ = found.unknown, found.unmatched
        return self

    def _vocabulary(self, picked: np.ndarray) -> tuple[TfidfVectorizer | None, Any]:
        """TF-IDF on the sample; with too few texts for the row limits, every term is kept."""
        least = max(MIN_ROWS, math.ceil(MIN_SHARE * len(picked)))
        for low, high in ((least, MAX_SHARE), (1, 1.0)):
            vectorizer = TfidfVectorizer(
                **settings(self.analyzer),
                max_features=self.terms,
                min_df=low,
                max_df=high,
                sublinear_tf=True,
                dtype=cast(Any, np.float32),
            )
            try:
                matrix = vectorizer.fit_transform(picked)
            except ValueError:  # no term left (empty texts, or every term pruned)
                continue
            if hasattr(vectorizer, "stop_words_"):  # the pruned terms: large, introspection only
                vectorizer.stop_words_ = None
            return vectorizer, matrix
        return None, None

    @property
    def vocabulary(self) -> list[str]:
        """The terms the model knows, in column order of the TF-IDF matrix."""
        vectorizer = getattr(self, "vectorizer_", None)
        if vectorizer is None:
            return []
        return [str(t) for t in vectorizer.get_feature_names_out()]

    def tfidf(self, X) -> Any:
        """The sparse TF-IDF matrix of these texts (None without a vocabulary)."""
        if self.vectorizer_ is None:
            return None
        return self.vectorizer_.transform(texts(X))

    def transform(self, X):
        matrix = self.tfidf(X)
        if matrix is None:
            return np.zeros((len(texts(X)), 1))
        if self.svd_ is None:
            return np.asarray(matrix.toarray(), dtype=float)
        return np.asarray(self.svd_.transform(matrix), dtype=float)

    def terms_matrix(self) -> np.ndarray:
        """(model inputs × terms): how each output reads each term (identity without SVD)."""
        size = len(self.vocabulary)
        if self.svd_ is None:
            return np.eye(self.width_, max(size, 1))
        return np.asarray(self.svd_.components_, dtype=float)

    def get_feature_names_out(self, input_features=None):
        prefix = (
            str(input_features[0])
            if input_features is not None and len(input_features)
            else str(getattr(self, "feature_names_in_", ["text"])[0])
        )
        return np.array([f"{prefix}_text{k + 1}" for k in range(self.width_)], dtype=object)


# --- Finding the text steps of a fitted pipeline --------------------------------------------------


@dataclass
class TextStep:
    column: str
    features: TextFeatures
    scale: Any  # the scaler after the SVD, or "passthrough"


def steps_of(pipeline: Any) -> dict[str, TextStep]:
    """The fitted text steps of a pipeline, per column; for an ensemble (038), those of its first
    member that has one per column."""
    found: dict[str, TextStep] = {}
    named = getattr(pipeline, "named_steps", {})
    prepare = named.get("prepare")
    for name, step, columns in getattr(prepare, "transformers_", []):
        if name == "remainder" or not hasattr(step, "named_steps"):
            continue
        features = step.named_steps.get("text")
        if isinstance(features, TextFeatures):
            found[str(columns[0])] = TextStep(
                str(columns[0]), features, step.named_steps.get("scale", "passthrough")
            )
    model = named.get("model")
    for member in getattr(model, "estimators_", []) or []:
        for column, step in steps_of(member).items():
            found.setdefault(column, step)
    return found


def slopes(scale: Any, width: int) -> np.ndarray:
    """How much each output of the scaler after the SVD moves per unit of its input."""
    factor = getattr(scale, "scale_", None)
    if factor is None:  # "passthrough"
        return np.ones(width)
    factor = np.asarray(factor, dtype=float)
    if isinstance(scale, MinMaxScaler):  # x · scale_ + min_
        return factor
    return np.divide(1.0, factor, out=np.zeros_like(factor), where=factor != 0)  # (x − c) / s
