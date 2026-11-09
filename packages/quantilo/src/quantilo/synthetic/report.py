"""The full evaluation of a synthetic set: four headline figures with their verdict, then every
detail (SDMetrics' structure: properties from 0 to 100 %, each the mean of its details)."""

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from quantilo.synthetic import compare, detection, diagnostic, privacy, quality
from quantilo.synthetic.encode import Learned
from quantilo.synthetic.methods import Cancelled, Fitted, StopCheck
from quantilo.synthetic.privacy import Disclosure
from quantilo.synthetic.sequences import SequenceReport
from quantilo.synthetic.utility import Utility

EVALUATION_ROWS = 20_000


@dataclass
class SyntheticScores:
    diagnostic: float
    quality: float
    shapes: float
    trends: float | None
    detection: float
    privacy: float
    utility: float | None = None  # TSTR / TRTR, once computed


@dataclass
class Verdicts:
    diagnostic: str
    quality: str
    detection: str
    privacy: str


@dataclass
class LeftOut:
    column: str
    reason: str


@dataclass
class Header:
    real_rows: int  # training rows the generator learned from
    holdout_rows: int
    synthetic_rows: int
    real_duplicates: int
    synthetic_duplicates: int
    types: dict[str, int] = field(default_factory=dict)  # modelled columns per type
    left_out: list[LeftOut] = field(default_factory=list)


@dataclass
class Report:
    scores: SyntheticScores
    verdicts: Verdicts
    header: Header
    diagnostic: diagnostic.Diagnostic
    quality: quality.Quality
    detection: detection.Detection
    privacy: privacy.Privacy
    columns: list[compare.ColumnLine]
    links: compare.Links
    overview: compare.SyntheticOverview
    fitted: Fitted
    learned: list[Learned]
    utility: Utility | None = None
    disclosure: Disclosure | None = None
    sequences: SequenceReport | None = None  # TimeGAN (023c)


def sample(frame: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    return frame.sample(rows, random_state=seed) if len(frame) > rows else frame


def _diagnostic_verdict(found: diagnostic.Diagnostic) -> str:
    if found.score >= 0.9999:
        return "All values valid"
    return f"{len(found.failures)} check(s) failed: a generator defect"


def evaluate(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    learned: list[Learned],
    fitted: Fitted,
    unique: list[str],
    bounds: dict[str, tuple[float | None, float | None]],
    left_out: list[LeftOut],
    seed: int = 42,
    should_stop: StopCheck = lambda: False,
) -> Report:
    expected = list(synthetic.columns)
    real = sample(train, EVALUATION_ROWS, seed)
    fake = sample(synthetic, EVALUATION_ROWS, seed)
    checks = diagnostic.diagnose(synthetic, learned, unique, bounds, expected)
    if should_stop():
        raise Cancelled
    resemblance = quality.quality(real, fake, learned)
    if should_stop():
        raise Cancelled
    detected = detection.detect(real, fake, learned, seed)
    if should_stop():
        raise Cancelled
    protected = privacy.privacy(train, holdout, synthetic, learned, seed)
    if should_stop():
        raise Cancelled
    names = [i.name for i in learned]
    header = Header(
        len(train),
        len(holdout),
        len(synthetic),
        int(train[names].duplicated().sum()) if names else 0,
        int(synthetic[names].duplicated().sum()) if names else 0,
        dict(Counter(i.type for i in learned)),
        left_out,
    )
    return Report(
        SyntheticScores(
            checks.score,
            resemblance.score,
            resemblance.shapes,
            resemblance.trends,
            detected.score,
            protected.score,
        ),
        Verdicts(
            _diagnostic_verdict(checks),
            quality.verdict(resemblance.score),
            detection.verdict(detected.score),
            "No unique row copied" if not protected.warnings else protected.warnings[0],
        ),
        header,
        checks,
        resemblance,
        detected,
        protected,
        compare.summaries(real, fake, learned, resemblance),
        compare.links(real, fake, learned),
        compare.overview(real, fake, learned, seed),
        fitted,
        learned,
    )
