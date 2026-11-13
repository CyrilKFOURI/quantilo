"""Cleaning precision bench (N01): the per-rule table of the Clean suggestions, judged against the
truth, per entry of the catalogue of automatic rules (N02, `quantilo/cleaning/catalogue.py`).

Usage (from packages/quantilo, so that quantilo is importable):
    uv run --no-sync python ../../scripts/clean_bench.py [--only CASE ...] [--json table.json]
    uv run --no-sync python ../../scripts/clean_bench.py --real [--only LABEL ...]

Without `--real`: the committed corpus (`packages/quantilo/tests/clean_corpus/cases.py`, synthetic
files written in code, a few seconds; the same run as `pytest -m clean_bench` in CI). With `--real`:
the audit's 31 real files, read from `data-corpus`, `kaggle-data` and `excel-corpus` next to the
repository and judged by `tests/clean_corpus/truth/*.clean-truth.json` (about a minute, one file at
a time; large files are sampled as in the audit). Exits with 1 when an automatic entry is under
99 %.
"""

import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "packages" / "quantilo"
for path in (PACKAGE / "src", PACKAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tests.clean_corpus.bench import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
