"""091: sentence embeddings (optional extra `embed`): cache R-EMB, choice R-FUS. A fake encoder
stands in for model2vec, which these tests never import."""

from pathlib import Path

import numpy as np
import pytest

from quantilo.ml import embed
from quantilo.ml import models as catalogue
from quantilo.ml import preparation as prep
from quantilo.ml.embed_cache import EmbeddingCache, normalise, value_key
from quantilo.ml.train import train
from tests.test_ml_text import FIELDS, REVIEWS, TARGET, N, decision

CALLS: list[int] = []


def _good(texts: list[str]) -> np.ndarray:
    """A useful encoder: one axis says "unhappy", the others are noise."""
    CALLS.append(len(texts))
    rng = np.random.default_rng(0)
    bad = np.array([any(w in t for w in ("refund", "broken")) for t in texts], dtype=float)
    return np.column_stack([bad, rng.normal(size=(len(texts), 7))]).astype(np.float32)


def _noise(texts: list[str]) -> np.ndarray:
    seeds = [int.from_bytes(value_key(t)[:4], "little") for t in texts]
    return np.stack([np.random.default_rng(s).normal(size=8) for s in seeds]).astype(np.float32)


@pytest.fixture(autouse=True)
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("QUANTILO_EMBEDDINGS_CACHE", str(tmp_path / "embeddings"))
    embed.register("fake-good", "fake-good|v1|8", _good)
    embed.register("fake-noise", "fake-noise|v1|8", _noise)
    CALLS.clear()
    return tmp_path / "embeddings"


class TestCache:
    def test_each_distinct_value_is_encoded_once(self, cache_dir: Path) -> None:
        cache = EmbeddingCache(cache_dir, "fake-good|v1|8")
        first = cache.encode(["a refund", "great", "a  refund ", "great"], _good)
        assert CALLS == [2]  # R-EMB-01: normalised, distinct
        assert np.allclose(first[0], first[2])
        again = EmbeddingCache(cache_dir, "fake-good|v1|8").encode(["great", "new one"], _good)
        assert CALLS == [2, 1] and again.shape == (2, 8)

    def test_models_are_never_mixed(self, cache_dir: Path) -> None:
        EmbeddingCache(cache_dir, "fake-good|v1|8").encode(["x"], _good)
        EmbeddingCache(cache_dir, "fake-good|v2|8").encode(["x"], _good)
        assert CALLS == [1, 1]  # R-EMB-02
        assert len(list(cache_dir.glob("*/index.json"))) == 2

    def test_budget_evicts_the_oldest_parts(self, cache_dir: Path) -> None:
        old = EmbeddingCache(cache_dir, "old|1")
        old.encode([f"t{i}" for i in range(200)], _noise)
        assert list(old.folder.glob("part-*.parquet"))
        EmbeddingCache(cache_dir, "new|1", budget=1).encode(["u"], _noise)
        assert not list(old.folder.glob("part-*.parquet"))  # R-EMB-04: least recent first

    def test_normalise(self) -> None:
        assert normalise("  Café \n crème ") == "Café crème"


class TestOption:
    def test_greyed_with_its_reason_without_the_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(embed, "installed", lambda: False)
        found = embed.status()
        assert not found.available and found.reason and "embed" in found.reason
        found_prep = prep.decide(REVIEWS, FIELDS, TARGET, "binary", catalogue.LINEAR)
        offered = {o.value: o.reason for o in decision(found_prep, "review").options["encoding"]}
        assert offered["embeddings"] and "extra" in offered["embeddings"]

    def test_choosing_it_without_the_extra_keeps_tfidf(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(embed, "installed", lambda: False)
        choices = prep.Choices(columns={"review": prep.ColumnChoice(encoding="embeddings")})
        found = prep.decide(REVIEWS, FIELDS, TARGET, "binary", catalogue.LINEAR, None, choices)
        assert decision(found, "review").encoding in ("words", "chars")
        assert any("embeddings" in w for w in found.warnings)


class TestFusion:
    def test_useful_embeddings_are_kept(self) -> None:
        step = embed.EmbeddingFeatures(model="fake-good", components=4)
        y = np.array(REVIEWS["returned"])
        step.fit(REVIEWS[["review"]], y)
        assert step.kept_ in ("embeddings", "tfidf")
        assert "R-FUS-02" in step.outcome_
        assert step.transform(REVIEWS[["review"]]).shape[0] == N

    def test_noise_falls_back_to_tfidf_and_says_so(self) -> None:
        step = embed.EmbeddingFeatures(model="fake-noise", components=4)
        step.fit(REVIEWS[["review"]], np.array(REVIEWS["returned"]))
        assert step.kept_ == "tfidf" and step.fallback_ is not None
        assert "TF-IDF used instead (R-FUS-02)" in step.outcome_
        assert step.get_feature_names_out()[0].startswith("review_text")

    def test_a_trained_model_tells_what_was_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(embed, "MODEL", "fake-noise")
        choices = prep.Choices(columns={"review": prep.ColumnChoice(encoding="embeddings")})
        result = train(
            REVIEWS,
            FIELDS,
            TARGET,
            "binary",
            "logistic_regression",
            None,
            N,
            False,
            choices=choices,
        )
        notes = [w for w in result.preparation.warnings if "R-FUS-02" in w]
        assert len(notes) == 1 and notes[0].startswith("review: sentence embeddings")

    def test_another_model_key_is_refused(self) -> None:
        step = embed.EmbeddingFeatures(model="fake-good", components=4, compare=False)
        step.fit(REVIEWS[["review"]])
        embed.register("fake-good", "fake-good|v2|8", _good)
        with pytest.raises(RuntimeError, match="R-EMB-05"):
            step.transform(REVIEWS[["review"]])
