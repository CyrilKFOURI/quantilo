"""Segments and unusual rows for the library: the app's Segments page as methods.

found = ds.segments().fit()        # columns, preparation, method and k chosen by the app's rules
found.segments[0].name, found.stability.ari
found.to_pandas()                  # the profile table: one row per column, one column per segment
found.to_dataset()                 # the rows with a `segment` column, as a new dataset
dx.plot(found)                     # the map (or "scores", "stability", "dendrogram")
"""

from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from quantilo.filters.model import Condition
from quantilo.ml import preparation as prep
from quantilo.segments.engine import (
    Fitted,
    MethodChoice,
    Scored,
    Segmentation,
    SegmentLabels,
    SegmentPlan,
    SegmentRequest,
    SegmentsEngine,
    SegmentsError,
    UnusualRequest,
    UnusualRows,
)
from quantilo.segments.projection import SegmentMap
from quantilo.segments.unusual import UnusualRow

if TYPE_CHECKING:
    from quantilo.dataset import Dataset

SEGMENT_COLUMN = "segment"


def profile_table(result: Segmentation) -> pd.DataFrame:
    """One row per column, "overall" then one column per segment (mean or most frequent)."""
    names = [s.name for s in result.segments]
    table = pd.DataFrame(
        [[row.overall.value, *(c.value for c in row.cells)] for row in result.profiles],
        index=pd.Index([row.column for row in result.profiles], name="column"),
        columns=["overall", *names],
    )
    return table


def free_name(columns: list[str], wanted: str) -> str:
    """`wanted`, or `wanted_2`, `wanted_3`… when the dataset already has a column of that name."""
    taken = set(columns)
    if wanted not in taken:
        return wanted
    i = 2
    while f"{wanted}_{i}" in taken:
        i += 1
    return f"{wanted}_{i}"


class FittedSegments:
    """A segmentation: scores, stability, the segments with their portraits, the profile table
    and the map. Every field of the result is an attribute (`found.k`, `found.segments`)."""

    def __init__(
        self, source: "Dataset", engine: SegmentsEngine, fitted: Fitted, request: SegmentRequest
    ) -> None:
        self._source = source
        self._engine = engine
        self._fitted = fitted
        self.request = request
        self.result = fitted.result

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.result, name)

    def __repr__(self) -> str:
        stability = f"{self.result.stability.ari:.2f}" if self.result.stability else "n/a"
        return (
            f"<FittedSegments {self.result.method_label}: {self.result.k} segments, "
            f"stability {stability}>"
        )

    @property
    def labels(self) -> pd.DataFrame:
        """The segment of every row: `row` (row number in the file) and `segment` (−1: not
        grouped)."""
        return self._fitted.labels.copy()

    def names(self) -> dict[int, str]:
        return {s.id: s.name for s in self.result.segments}

    def rename(self, names: dict[int, str]) -> "FittedSegments":
        known = {s.id for s in self.result.segments}
        unknown = [str(k) for k in names if k not in known]
        if unknown:
            raise SegmentsError(f"Unknown segment(s): {', '.join(unknown)}.", "unknown_segment")
        for segment in self.result.segments:
            if segment.id in names and names[segment.id].strip():
                segment.name = names[segment.id].strip()[:80]
        return self

    def to_pandas(self) -> pd.DataFrame:
        return profile_table(self.result)

    def as_labels(self) -> SegmentLabels:
        """This segmentation for the "distance to its segment" method of unusual rows."""
        return SegmentLabels(self._fitted.labels, self.names(), self.request)

    def tsne(self) -> SegmentMap:
        """The map drawn with t-SNE (at most 5,000 rows) instead of the PCA."""
        if self.result.tsne is None:
            self.result.tsne = self._engine.tsne(self.result, self._fitted.drawn)
        return self.result.tsne

    def to_dataset(self, name: str | None = None) -> "Dataset":
        """The (filtered) rows with a `segment` column (the segments' names), as a new
        dataset; the source is not changed."""
        from quantilo.dataset import from_pandas

        frame = self._source.to_pandas()
        labels = self._fitted.labels.sort_values("row")
        names = self.names()
        column = free_name(list(frame.columns), SEGMENT_COLUMN)
        frame[column] = [names.get(int(v), str(v)) for v in labels["segment"].to_numpy()]
        made = from_pandas(frame, name or f"{self._source.name} (segments)")
        types = {c.name: c for c in self._source.types}
        for found in made.types:
            original = types.get(found.name)
            if original and original.type != found.type:
                made.retype(found.name, original.type)
        if made.column(column).type != "categorical_nominal":
            made.retype(column, "categorical_nominal")
        return made


class Segments:
    """Groups of similar rows in a dataset (the active filters apply)."""

    def __init__(
        self,
        source: "Dataset",
        engine: SegmentsEngine,
        dataset_id: str,
        conditions: list[Condition],
        request: SegmentRequest,
    ) -> None:
        self._source = source
        self._engine = engine
        self._id = dataset_id
        self._conditions = conditions
        self.request = request

    def plan(self) -> SegmentPlan:
        """The columns used, each with its rule and reason, and the method Auto takes."""
        return self._engine.plan(self._id, self._conditions, self.request)

    def fit(self) -> FittedSegments:
        """Try k = 2 to 10 (unless k is set), keep the k the scores agree on, check stability
        on 10 subsamples, describe each segment."""
        fitted = self._engine.segment(self._id, self._conditions, self.request)
        return FittedSegments(self._source, self._engine, fitted, self.request)


def request(
    columns: list[str] | None,
    method: MethodChoice,
    k: int | str | None,
    preparation: prep.Choices | dict[str, Any] | None,
    seed: int,
) -> SegmentRequest:
    try:
        chosen = None if k in (None, "auto") else int(cast(int, k))
        choices = (
            prep.Choices.model_validate(preparation)
            if isinstance(preparation, dict)
            else preparation
        )
        return SegmentRequest(
            columns=columns, method=method, k=chosen, preparation=choices, seed=seed
        )
    except (ValueError, TypeError) as error:
        raise SegmentsError(f"Invalid segmentation: {error}") from None


class FoundUnusualRows:
    """Unusual rows: the score of every row, the proposed threshold, and the most unusual rows
    with their reasons. Every field of the result is an attribute (`found.threshold`)."""

    def __init__(self, scored: Scored) -> None:
        self.result = scored.result
        self._scores = scored.scores

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.result, name)

    def __repr__(self) -> str:
        return (
            f"<FoundUnusualRows {self.result.method_label}: {self.result.threshold.rows} of "
            f"{self.result.rows} rows above {self.result.threshold.value:.4g}>"
        )

    @property
    def scores(self) -> pd.DataFrame:
        """The score of every row: `row` (row number in the file) and `score`."""
        return self._scores.copy()

    def above(self, threshold: float | None = None) -> list[UnusualRow]:
        """The listed rows at or above a threshold (the proposed one by default)."""
        cut = self.result.threshold.value if threshold is None else threshold
        return [r for r in self.result.listed if r.score >= cut]

    def to_pandas(self) -> pd.DataFrame:
        return unusual_table(self.result)


def unusual_table(result: UnusualRows) -> pd.DataFrame:
    """The listed rows: rank, row, score, reasons, then every column."""
    records = []
    for row in result.listed:
        reasons = [row.combination] if row.combination else []
        reasons += [r.text for r in row.reasons]
        records.append(
            {
                "rank": row.rank,
                "row": row.row,
                "score": row.score,
                "reasons": "; ".join(reasons),
                **row.values,
            }
        )
    return pd.DataFrame(records)


def unusual_request(
    columns: list[str] | None,
    method: str,
    preparation: prep.Choices | dict[str, Any] | None,
    seed: int,
) -> UnusualRequest:
    try:
        choices = (
            prep.Choices.model_validate(preparation)
            if isinstance(preparation, dict)
            else preparation
        )
        return UnusualRequest.model_validate(
            {"columns": columns, "method": method, "preparation": choices, "seed": seed}
        )
    except ValueError as error:
        raise SegmentsError(f"Invalid request: {error}") from None
