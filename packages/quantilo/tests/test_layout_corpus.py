"""The study's thresholds on the external corpus (080, § 8); skipped when the corpus is absent."""

import os
from pathlib import Path

import pytest

from tests.layout_corpus.evaluate import DEFAULT_CORPUS, evaluate

CORPUS = Path(os.environ.get("QUANTILO_LAYOUT_CORPUS", DEFAULT_CORPUS))


@pytest.mark.skipif(not CORPUS.is_dir(), reason="the excel-corpus folder is not on this machine")
def test_corpus_thresholds() -> None:
    total, families, _, _ = evaluate(CORPUS.resolve())
    ratios = total.ratios()
    assert ratios["recall_eob2"] >= 0.9 and ratios["precision_eob2"] >= 0.9
    assert ratios["headers_exact"] >= 0.85
    assert families["CTRL"].ratios()["recall_eob0"] == 1.0  # control files: no change
