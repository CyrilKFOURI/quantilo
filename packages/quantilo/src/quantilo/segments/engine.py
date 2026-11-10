"""Segments on a catalog's dataset (docs/pages/14-segments.md): the automatic column choices, then
the segmentation with its scores, stability, portraits and map. Synchronous: the app adds jobs and
saved runs around it."""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic import Field as PydanticField

from quantilo.catalog import Catalog
from quantilo.errors import ExplorerError
from quantilo.filters.model import Condition
from quantilo.filters.scope import scoped_source, where_for
from quantilo.ingestion.sql import ident, lit
from quantilo.ml import preparation as prep
from quantilo.ml.data import Field
from quantilo.segments import methods, portrait, projection
from quantilo.segments import prepare as preparing
from quantilo.segments import unusual as scoring
from quantilo.segments.methods import (
    Cancelled,
    Method,
    SegmentKChoice,
    SegmentScore,
    SegmentStability,
    Space,
)
from quantilo.segments.portrait import (
    SegmentProfileRow,
    SegmentRules,
    SegmentTrait,
)
from quantilo.segments.projection import SegmentDendrogram, SegmentMap
from quantilo.segments.unusual import (
    UnusualCurve,
    UnusualMethod,
    UnusualRow,
    UnusualThreshold,
)

AUTO_CATEGORICAL = 0.3  # k-prototypes from this share of category columns on
MethodChoice = Literal["auto", "kmeans", "kprototypes", "gmm", "ward", "hdbscan"]
Stage = Literal["loading", "preparing", "choosing", "stability", "assigning", "describing"]


class SegmentsError(ExplorerError):
    def __init__(self, message: str, code: str = "segments", status_code: int = 422) -> None:
        super().__init__(message, code=code, status_code=status_code)


class SegmentRequest(BaseModel):
    """Everything not given follows the app's rules."""

    columns: list[str] | None = PydanticField(default=None, max_length=2_000)  # None: automatic
    method: MethodChoice = "auto"
    k: int | None = PydanticField(default=None, ge=2, le=20)  # None: chosen from the scores
    preparation: prep.Choices | None = None
    seed: int = PydanticField(default=42, ge=0, le=2**31 - 1)


class UnusualRequest(BaseModel):
    """Same columns and preparation as Segments; "segment" needs a segmentation."""

    columns: list[str] | None = PydanticField(default=None, max_length=2_000)
    method: Literal["auto", "isolation_forest", "mahalanobis", "lof", "segment"] = "auto"
    preparation: prep.Choices | None = None
    seed: int = PydanticField(default=42, ge=0, le=2**31 - 1)


@dataclass
class SegmentLabels:
    """A segmentation the "distance to its segment" method measures from."""

    labels: pd.DataFrame  # row, segment
    names: dict[int, str]
    request: SegmentRequest


@dataclass
class UnusualRows:
    method: UnusualMethod
    method_label: str
    method_why: str
    rows: int  # rows scored (after the filters)
    fit_rows: int
    sampled: bool
    threshold: UnusualThreshold  # proposed; the page moves it without scoring again
    curve: UnusualCurve
    listed: list[UnusualRow]  # the most unusual rows, with their reasons
    columns: list[str]  # every column shown for a row
    used: list[str]  # the columns the scores read
    preparation: prep.Preparation
    hidden_expected: int  # rows marked "Expected" left out of the list
    seconds: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class Scored:
    """Unusual rows with the score of every row (kept by the app beside the result)."""

    result: UnusualRows
    scores: pd.DataFrame  # row, score


@dataclass
class SegmentMethodOption:
    key: Method
    label: str
    available: bool
    reason: str | None  # why not, or when Auto takes it


@dataclass
class SegmentPlan:
    """What a segmentation with these choices would use, before running it."""

    preparation: prep.Preparation
    rows: int
    fit_rows: int
    sampled: bool
    method: Method
    method_why: str
    categorical_share: float
    methods: list[SegmentMethodOption]


@dataclass
class SegmentPortrait:
    id: int  # 0 is the largest; −1 gathers the rows HDBSCAN did not group
    name: str  # proposed from the two strongest traits
    size: int
    share: float
    traits: list[SegmentTrait]
    rules: SegmentRules | None
    membership: float | None = None  # Gaussian mixture: mean probability of belonging
    grouped: bool = True


@dataclass
class Segmentation:
    method: Method
    method_label: str
    method_why: str
    k: int
    choice: SegmentKChoice
    scores: list[SegmentScore]  # one per k tried
    chosen: SegmentScore
    stability: SegmentStability | None
    segments: list[SegmentPortrait]
    profiles: list[SegmentProfileRow]
    map: SegmentMap
    dendrogram: SegmentDendrogram | None
    preparation: prep.Preparation
    inputs: int  # prepared columns the distances use
    categorical_share: float
    rows: int  # rows segmented (after the filters)
    fit_rows: int
    sampled: bool
    seconds: float
    warnings: list[str] = field(default_factory=list)
    tsne: SegmentMap | None = None


@dataclass
class Fitted:
    """A segmentation with what the app keeps beside it: the segment of every row, and the drawn
    rows prepared (for t-SNE)."""

    result: Segmentation
    labels: pd.DataFrame  # row (in the file), segment
    drawn: np.ndarray  # prepared values of the map's points


Progress = Callable[[str, float | None, str | None], None]  # stage, share, message


def auto_method(preparation: prep.Preparation, rows: int) -> tuple[Method, str]:
    share = preparing.categorical_share(preparation)
    if share >= AUTO_CATEGORICAL:
        return "kprototypes", (
            f"k-prototypes: {share:.0%} of the columns are categories (from "
            f"{AUTO_CATEGORICAL:.0%} on), compared as they are instead of one-hot."
        )
    if rows > methods.MINIBATCH_ROWS:
        return "kmeans", (
            f"MiniBatch k-means: mostly numbers, and {rows:,} rows (more than "
            f"{methods.MINIBATCH_ROWS:,})."
        )
    return "kmeans", "k-means: mostly numbers."


def method_options(rows: int, auto: Method) -> list[SegmentMethodOption]:
    ward = rows <= methods.WARD_MAX
    return [
        SegmentMethodOption(
            "kmeans", "k-means", True, "Auto with mostly numbers." if auto == "kmeans" else None
        ),
        SegmentMethodOption(
            "kprototypes",
            "k-prototypes",
            True,
            f"Auto from {AUTO_CATEGORICAL:.0%} of category columns.",
        ),
        SegmentMethodOption(
            "gmm", "Gaussian mixture", True, "Groups of different shapes; chooses k by BIC."
        ),
        SegmentMethodOption(
            "ward",
            "Hierarchical (Ward)",
            ward,
            "Shows a dendrogram."
            if ward
            else f"Up to {methods.WARD_MAX:,} rows: filter the data or pick another method.",
        ),
        SegmentMethodOption(
            "hdbscan", "HDBSCAN", True, "Finds the number of groups; leaves noise ungrouped."
        ),
    ]


class SegmentsEngine:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns

    def _fields(self, dataset_id: str, conditions: list[Condition]) -> tuple[list[Field], str]:
        dataset = self.columns.ready(dataset_id)
        views = [v for v in self.columns.columns(dataset_id) if not v.excluded]
        fields = [Field(v.name, v.type, v.parser, v.role, v.order) for v in views]
        source = scoped_source(dataset.storage_path, where_for(views, conditions), numbered=True)
        return fields, source

    def _rows(
        self, fields: list[Field], source: str, request: SegmentRequest | UnusualRequest
    ) -> tuple[preparing.Rows, prep.Preparation]:
        self.check(fields, request)
        connection = duckdb.connect()
        try:
            rows = preparing.load(connection, source, fields)
        finally:
            connection.close()
        if rows.rows < preparing.MIN_ROWS:
            raise SegmentsError(
                f"{rows.rows} rows: segments need at least {preparing.MIN_ROWS}.", "too_few_rows"
            )
        preparation = preparing.decide(rows.data, fields, request.columns, request.preparation)
        return rows, preparation

    @staticmethod
    def check(fields: list[Field], request: SegmentRequest | UnusualRequest) -> None:
        """Every column named in the request exists and is not ignored."""
        names = {f.name for f in fields}
        unknown = [c for c in request.columns or [] if c not in names]
        unknown += [
            c
            for c in (request.preparation.columns if request.preparation else {})
            if c not in names
        ]
        if unknown:
            raise SegmentsError(
                f"Unknown or ignored column(s): {', '.join(dict.fromkeys(unknown))}.",
                "unknown_column",
            )

    def plan(
        self, dataset_id: str, conditions: list[Condition], request: SegmentRequest | None = None
    ) -> SegmentPlan:
        """The columns used (each with its rule and reason) and the method Auto would take."""
        request = request or SegmentRequest()
        fields, source = self._fields(dataset_id, conditions)
        rows, preparation = self._rows(fields, source, request)
        method, why = auto_method(preparation, rows.rows)
        return SegmentPlan(
            preparation,
            rows.rows,
            len(rows.data),
            rows.sampled,
            method,
            why,
            preparing.categorical_share(preparation),
            method_options(rows.rows, method),
        )

    def segment(
        self,
        dataset_id: str,
        conditions: list[Condition],
        request: SegmentRequest,
        should_stop: Callable[[], bool] = lambda: False,
        progress: Progress = lambda _stage, _share, _message: None,
    ) -> Fitted:
        """Prepare, try k = 2 to 10, keep the k the scores agree on, check its stability, give
        every row its segment, describe each one. Raises `Cancelled` when `should_stop()`."""
        started = time.perf_counter()
        fields, source = self._fields(dataset_id, conditions)
        progress("loading", None, None)
        rows, preparation = self._rows(fields, source, request)
        if not preparation.used:
            raise SegmentsError(
                "No column can be used: pick at least one number or category.", "no_columns"
            )
        method, why = auto_method(preparation, rows.rows)
        if request.method != "auto":
            method = request.method
            why = f"{methods.METHODS[method]}: your choice."
        if method == "ward" and rows.rows > methods.WARD_MAX:
            raise SegmentsError(
                f"Ward needs at most {methods.WARD_MAX:,} rows ({rows.rows:,} here): filter "
                "the data or pick another method.",
                "method_unavailable",
            )
        progress("preparing", None, None)
        data = rows.data.reset_index(drop=True)
        try:
            prepared = preparing.matrix(preparation, data)
        except ValueError as error:
            raise SegmentsError(str(error)) from None
        space = Space(prepared.x, prepared.numbers, prepared.codes)
        preparation.inputs = prepared.x.shape[1]
        if should_stop():
            raise Cancelled
        found = self._choose(space, method, request, should_stop, progress)
        scores, choice, fitted, tree = found
        k = choice.k
        chosen = (
            next(s for s in scores if s.k == k)
            if any(s.k == k for s in scores)
            else (methods.score(space.x, fitted.labels, k, request.seed, fitted.bic))
        )
        mapping = methods.relabel(fitted.labels)
        labels = methods.apply(mapping, fitted.labels)
        warnings = list(preparation.warnings)
        stability = None
        if method != "hdbscan" or len(np.unique(labels[labels >= 0])) >= 2:
            progress("stability", 0.0, None)
            stability = methods.stability(
                space,
                method,
                k,
                labels,
                request.seed,
                should_stop,
                lambda done: progress("stability", done / methods.STABILITY_RUNS, None),
            )
            if stability.warning:
                warnings.append(stability.warning)
        if chosen.silhouette is not None and chosen.silhouette < methods.WEAK_SILHOUETTE:
            warnings.append(
                f"The segments overlap (silhouette {chosen.silhouette:.2f}, below "
                f"{methods.WEAK_SILHOUETTE}): read them as tendencies, not as separate groups."
            )
        every = self._assign(
            fields, source, rows, prepared, fitted, mapping, labels, should_stop, progress
        )
        progress("describing", None, None)
        segments = self._describe(data, labels, every, preparation.used, fitted, request.seed)
        if method == "hdbscan" and len([s for s in segments if s.grouped]) < 2:
            warnings.append("HDBSCAN found no clear groups in these rows.")
        row_numbers = data["__row"].to_numpy(dtype=np.int64)
        drawn = projection.sample(len(data), request.seed)
        result = Segmentation(
            method,
            methods.METHODS[method],
            why,
            k if method != "hdbscan" else len([s for s in segments if s.grouped]),
            choice,
            scores,
            chosen,
            stability,
            segments,
            portrait.profiles(data, labels, [s.id for s in segments], preparation.used),
            projection.pca_map(space.x, labels, row_numbers, request.seed),
            projection.dendrogram(tree.linkage, k) if tree is not None else None,
            preparation,
            prepared.x.shape[1],
            preparing.categorical_share(preparation),
            rows.rows,
            len(data),
            rows.sampled,
            0.0,
            warnings,
        )
        result.seconds = round(time.perf_counter() - started, 2)
        return Fitted(result, every, space.x[drawn])

    def _choose(
        self,
        space: Space,
        method: Method,
        request: SegmentRequest,
        should_stop: Callable[[], bool],
        progress: Progress,
    ) -> tuple[list[SegmentScore], SegmentKChoice, methods.Fitted, methods.WardTree | None]:
        tree = methods.ward_tree(space, request.seed) if method == "ward" else None
        if method == "hdbscan":
            fitted = methods.fit(space, method, 0, request.seed)
            groups = len(np.unique(fitted.labels[fitted.labels >= 0]))
            found = methods.score(space.x, fitted.labels, groups, request.seed)
            noise = float(np.mean(fitted.labels < 0))
            choice = SegmentKChoice(
                groups,
                False,
                f"{groups} segment{'s' if groups != 1 else ''}: found by HDBSCAN, "
                f"{portrait.percent(noise)} of the rows not grouped.",
            )
            return [found], choice, fitted, None
        ks = [k for k in methods.K_RANGE if k < len(space)]
        if request.k is not None and request.k not in ks:
            ks = sorted({*ks, request.k})
        # beyond SEARCH_ROWS, k is chosen on a sample (Ward's tree already is one)
        searched = (
            space.take(methods.sample(len(space), methods.SEARCH_ROWS, request.seed))
            if len(space) > methods.SEARCH_ROWS and method != "ward"
            else space
        )
        scores: list[SegmentScore] = []
        fits: dict[int, methods.Fitted] = {}
        for i, k in enumerate(ks):
            if should_stop():
                raise Cancelled
            progress("choosing", i / len(ks), f"k = {k}")
            fitted = methods.fit(searched, method, k, request.seed, tree)
            fits[k] = fitted
            scores.append(methods.score(searched.x, fitted.labels, k, request.seed, fitted.bic))
        choice = methods.choose_k(scores, method, request.k)
        if searched is not space:
            progress("choosing", 1.0, f"k = {choice.k} on every row")
            return scores, choice, methods.fit(space, method, choice.k, request.seed), tree
        return scores, choice, fits[choice.k], tree

    def _assign(
        self,
        fields: list[Field],
        source: str,
        rows: preparing.Rows,
        prepared: preparing.Matrix,
        fitted: methods.Fitted,
        mapping: dict[int, int],
        labels: np.ndarray,
        should_stop: Callable[[], bool],
        progress: Progress,
    ) -> pd.DataFrame:
        """The segment of every row: the fit rows keep theirs, the others (beyond the sample)
        go to the nearest group."""
        if not rows.sampled:
            return pd.DataFrame(
                {"row": rows.data["__row"].to_numpy(dtype=np.int64), "segment": labels}
            )
        parts = []
        done = 0
        connection = duckdb.connect()
        try:
            for chunk in preparing.every_row(connection, source, fields):
                if should_stop():
                    raise Cancelled
                numbers, codes = prepared.mixed(chunk)
                space = Space(prepared.transform(chunk), numbers, codes)
                found = methods.apply(mapping, fitted.predict(space))
                parts.append(
                    pd.DataFrame({"row": chunk["__row"].to_numpy(dtype=np.int64), "segment": found})
                )
                done += len(chunk)
                progress("assigning", done / rows.rows, f"{done:,} / {rows.rows:,} rows")
        finally:
            connection.close()
        every = pd.concat(parts, ignore_index=True)
        fit = pd.Series(labels, index=rows.data["__row"].to_numpy(dtype=np.int64))
        known = every["row"].isin(fit.index.to_list())
        every.loc[known, "segment"] = fit.loc[every.loc[known, "row"]].to_numpy()
        return every

    @staticmethod
    def _describe(
        data: pd.DataFrame,
        labels: np.ndarray,
        every: pd.DataFrame,
        used: list[prep.Decision],
        fitted: methods.Fitted,
        seed: int,
    ) -> list[SegmentPortrait]:
        sizes = every["segment"].value_counts()
        total = len(every)
        ids = sorted(int(g) for g in np.unique(labels) if g >= 0)
        if (labels < 0).any():
            ids.append(-1)
        segments = []
        for g in ids:
            found = portrait.traits(data, labels, g, used) if g >= 0 else []
            membership = (
                float(np.mean(fitted.membership[labels == g]))
                if fitted.membership is not None
                else None
            )
            size = int(sizes.get(g) or 0)
            segments.append(
                SegmentPortrait(
                    g,
                    portrait.name(found) if g >= 0 else portrait.NOT_GROUPED,
                    size,
                    size / total if total else 0.0,
                    found,
                    portrait.rules(data, labels, g, used, seed) if g >= 0 else None,
                    membership,
                    g >= 0,
                )
            )
        names = portrait.unique_names([s.name for s in segments])
        for segment, unique in zip(segments, names, strict=True):
            segment.name = unique
        return segments

    def tsne(self, fitted: Segmentation, drawn: np.ndarray, seed: int = 42) -> SegmentMap:
        """t-SNE of the map's points (refused in the browser version: too slow there)."""
        if preparing.IN_BROWSER:
            raise SegmentsError(
                "t-SNE is too slow in the browser version: install the full app.", "locked"
            )
        shown = fitted.map
        return projection.tsne_map(
            drawn,
            np.asarray(shown.segments),
            np.asarray(shown.rows),
            shown.of,
            seed,
        )

    # --- Unusual rows (033) --------------------------------------------------------------------

    def unusual(
        self,
        dataset_id: str,
        conditions: list[Condition],
        request: UnusualRequest,
        segments: SegmentLabels | None = None,
        hidden: set[str] | None = None,
        should_stop: Callable[[], bool] = lambda: False,
        progress: Progress = lambda _stage, _share, _message: None,
    ) -> Scored:
        """Score every row, propose a threshold, list the most unusual rows with their reasons.
        `hidden`: keys of rows marked "Expected", left out of the list."""
        started = time.perf_counter()
        hidden = hidden or set()
        if request.method == "segment" and segments is None:
            raise SegmentsError(
                "Distance to its segment needs a segmentation: run Segments first.",
                "no_segmentation",
            )
        fields, source = self._fields(dataset_id, conditions)
        progress("loading", None, None)
        columns, choices = request.columns, request.preparation
        if request.method == "segment" and segments is not None:
            columns, choices = segments.request.columns, segments.request.preparation
        shared = SegmentRequest(columns=columns, preparation=choices, seed=request.seed)
        rows, preparation = self._rows(fields, source, shared)
        left_out = scoring.numbers_only(preparation.used) if request.method == "mahalanobis" else []
        used = preparation.used
        if not used:
            raise SegmentsError(
                "No column can be used: pick at least one number or category.", "no_columns"
            )
        progress("preparing", None, None)
        data = rows.data.reset_index(drop=True)
        try:
            prepared = preparing.matrix(preparation, data)
        except ValueError as error:
            raise SegmentsError(str(error)) from None
        x = prepared.x
        preparation.inputs = x.shape[1]
        groups = None
        if request.method == "segment" and segments is not None:
            groups = segment_of(segments.labels, data["__row"].to_numpy(dtype=np.int64))
        method, why = scoring.auto_method(used, x)
        if request.method != "auto":
            method = request.method
            why = f"{scoring.METHODS[method]}: your choice."
        if left_out:
            why = (
                f"{scoring.METHODS[method]}: your choice, on the {len(used)} numbers; "
                f"{', '.join(left_out)} left out (it reads numbers only)."
            )
        if method == "mahalanobis" and not all(d.type in NUMERIC_TYPES for d in used):
            raise SegmentsError(
                "The robust Mahalanobis distance reads numbers only: with categories, it needs "
                f"at least {scoring.MCD_MIN_NUMBERS} numbers among the columns. Leave the "
                "categories out or pick another method.",
                "method_unavailable",
            )
        if should_stop():
            raise Cancelled
        progress("scoring", None, None)
        try:
            scorer = scoring.fit(method, x, groups, request.seed)
        except ValueError as error:
            raise SegmentsError(f"{scoring.METHODS[method]}: {error}") from None
        every = self._every_score(
            fields, source, rows, prepared, scorer, segments, should_stop, progress
        )
        progress("explaining", None, None)
        threshold, shape, listed, hidden_count = self._listing(
            dataset_id,
            data,
            every,
            prepared,
            scorer,
            groups,
            segments,
            used,
            hidden,
            source,
            fields,
        )
        found = UnusualRows(
            method,
            scoring.METHODS[method],
            why,
            rows.rows,
            len(data),
            rows.sampled,
            threshold,
            shape,
            listed,
            [f.name for f in fields],
            [d.column for d in used],
            preparation,
            hidden_count,
            round(time.perf_counter() - started, 2),
            list(preparation.warnings),
        )
        return Scored(found, every)

    def _every_score(
        self,
        fields: list[Field],
        source: str,
        rows: preparing.Rows,
        prepared: preparing.Matrix,
        scorer: scoring.Scorer,
        segments: SegmentLabels | None,
        should_stop: Callable[[], bool],
        progress: Progress,
    ) -> pd.DataFrame:
        """The score of every row: the fit rows have theirs, the others are scored by chunks."""
        fit_rows = rows.data["__row"].to_numpy(dtype=np.int64)
        if not rows.sampled:
            return pd.DataFrame({"row": fit_rows, "score": scorer.fit_scores})
        parts = []
        done = 0
        connection = duckdb.connect()
        try:
            for chunk in preparing.every_row(connection, source, fields):
                if should_stop():
                    raise Cancelled
                numbers = chunk["__row"].to_numpy(dtype=np.int64)
                own = segment_of(segments.labels, numbers) if segments is not None else None
                found = scorer.score(prepared.transform(chunk), own)
                parts.append(pd.DataFrame({"row": numbers, "score": found}))
                done += len(chunk)
                progress("scoring", done / rows.rows, f"{done:,} / {rows.rows:,} rows")
        finally:
            connection.close()
        every = pd.concat(parts, ignore_index=True)
        fit = pd.Series(scorer.fit_scores, index=fit_rows)
        known = every["row"].isin(fit.index.to_list())
        every.loc[known, "score"] = fit.loc[every.loc[known, "row"]].to_numpy()
        return every

    def _listing(
        self,
        dataset_id: str,
        data: pd.DataFrame,
        every: pd.DataFrame,
        prepared: preparing.Matrix,
        scorer: scoring.Scorer,
        groups: np.ndarray | None,
        segments: SegmentLabels | None,
        used: list[prep.Decision],
        hidden: set[str],
        source: str,
        fields: list[Field],
    ) -> tuple[UnusualThreshold, UnusualCurve, list[UnusualRow], int]:
        ordered = every.sort_values(["score", "row"], ascending=[False, True])
        candidates = ordered.head(scoring.LIST_ROWS + len(hidden))
        keys = self.keys(dataset_id, candidates["row"].tolist())
        hidden_rows = {row for row, key in keys.items() if key in hidden}
        hidden_list = sorted(hidden_rows)
        kept = cast(
            pd.DataFrame, candidates[~candidates["row"].isin(hidden_list)].head(scoring.LIST_ROWS)
        )
        scores = np.asarray(ordered.loc[~ordered["row"].isin(hidden_list), "score"], dtype=float)
        threshold = scoring.propose(scores)
        values = self._values(source, fields, kept["row"].tolist())
        top = preparing.typed(values.drop(columns=["__row"]).copy(), fields)
        top_groups = None
        names: dict[int, str] = {}
        if segments is not None and groups is not None:
            top_groups = segment_of(segments.labels, kept["row"].to_numpy(dtype=np.int64))
            names = segments.names
        found = scoring.reasons(
            top,
            kept["score"].to_numpy(dtype=float),
            top_groups,
            used,
            data,
            groups,
            names,
            prepared.transform,
            scorer.score,
        )
        listed = []
        pairs = zip(kept["row"].tolist(), kept["score"].tolist(), strict=True)
        for rank, (row, score) in enumerate(pairs, start=1):
            items, combination = found[rank - 1]
            record = values.iloc[rank - 1]
            listed.append(
                UnusualRow(
                    int(row),
                    keys[int(row)],
                    rank,
                    round(float(score), 6),
                    items,
                    combination,
                    {f.name: plain(record[f.name]) for f in fields},
                    names.get(int(top_groups[rank - 1])) if top_groups is not None else None,
                )
            )
        return threshold, scoring.curve(scores), listed, len(hidden_rows)

    @staticmethod
    def _values(source: str, fields: list[Field], rows: list[int]) -> pd.DataFrame:
        """The typed values of the given rows, in that order."""
        if not rows:
            return pd.DataFrame(columns=[*(f.name for f in fields), "__row"])
        selected = ", ".join(
            f"{preparing.ml_data._expression(f)} AS c{i}" for i, f in enumerate(fields)
        )
        wanted = ", ".join(str(int(r)) for r in rows)
        connection = duckdb.connect()
        try:
            frame = connection.execute(
                f"SELECT {selected}, file_row_number AS __row FROM {source} "
                f"WHERE file_row_number IN ({wanted})"
            ).df()
        finally:
            connection.close()
        frame.columns = [*(f.name for f in fields), "__row"]
        order = {r: i for i, r in enumerate(rows)}
        frame["__order"] = [order[int(r)] for r in frame["__row"]]
        return frame.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    def keys(self, dataset_id: str, rows: list[int]) -> dict[int, str]:
        """What a mark is kept under: "<identifier column>=<value>" when the dataset has an
        identifier, else "row:" + a fingerprint of every stored value of the row (so a mark
        survives a re-import that keeps the rows, even in another order)."""
        if not rows:
            return {}
        dataset = self.columns.ready(dataset_id)
        views = self.columns.columns(dataset_id)
        identifier = next((v for v in views if v.type == "identifier" and not v.excluded), None)
        wanted = ", ".join(str(int(r)) for r in rows)
        if identifier is not None:
            value = f"coalesce(CAST({ident(identifier.name)} AS VARCHAR), '')"
            expression = f"{lit(identifier.name + '=')} || {value}"
        else:
            parts = ", ".join(f"coalesce(CAST({ident(v.name)} AS VARCHAR), chr(0))" for v in views)
            expression = f"'row:' || md5(concat_ws(chr(31), {parts}))"
        connection = duckdb.connect()
        try:
            found = connection.execute(
                f"SELECT file_row_number, {expression} FROM "
                f"read_parquet({lit(dataset.storage_path)}, file_row_number = true) "
                f"WHERE file_row_number IN ({wanted})"
            ).fetchall()
        finally:
            connection.close()
        return {int(row): str(key) for row, key in found}

    def row_card(self, dataset_id: str, conditions: list[Condition], row: int) -> list["RowValue"]:
        """Each value of a row placed on its column's distribution (after the filters)."""
        fields, source = self._fields(dataset_id, conditions)
        values = self._values(source, fields, [row])
        if values.empty:
            raise SegmentsError(f"Row {row} is not in these rows.", "unknown_row", 404)
        record = values.iloc[0]
        found = []
        connection = duckdb.connect()
        try:
            for field_ in fields:
                value = plain(record[field_.name])
                expression = preparing.ml_data._expression(field_)
                if field_.type in NUMERIC_TYPES and isinstance(value, int | float):
                    below, total, median = connection.execute(
                        f"SELECT count(*) FILTER (WHERE v <= {float(value)!r}), count(v), "
                        f"median(v) FROM (SELECT TRY_CAST({expression} AS DOUBLE) AS v "
                        f"FROM {source})"
                    ).fetchone() or (0, 0, None)
                    percentile = below / total if total else None
                    found.append(
                        RowValue(field_.name, field_.type, value, percentile, None, plain(median))
                    )
                elif field_.type in CATEGORY_TYPES and value is not None:
                    same, total = connection.execute(
                        f"SELECT count(*) FILTER (WHERE CAST(v AS VARCHAR) = {lit(str(value))}), "
                        f"count(*) FROM (SELECT {expression} AS v FROM {source})"
                    ).fetchone() or (0, 0)
                    share = same / total if total else None
                    found.append(RowValue(field_.name, field_.type, value, None, share, None))
                else:
                    found.append(RowValue(field_.name, field_.type, value, None, None, None))
        finally:
            connection.close()
        return found


NUMERIC_TYPES = ("numeric_continuous", "numeric_discrete")
CATEGORY_TYPES = ("categorical_nominal", "categorical_ordinal", "boolean")


@dataclass
class RowValue:
    """One value of a row, placed on its column (the row's card)."""

    column: str
    type: str
    value: Any
    percentile: float | None  # numbers: share of rows at or below it
    share: float | None  # categories: share of rows with this value
    median: Any  # numbers: the column's median


def segment_of(labels: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """The segment of the given rows (−1 when a row has none)."""
    found = pd.Series(labels["segment"].to_numpy(), index=labels["row"].to_numpy())
    return found.reindex(rows).fillna(-1).to_numpy(dtype=np.int64)


def plain(value: Any) -> Any:
    """A cell as JSON carries it: numbers, text, ISO dates, or None."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, bool | int | str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)
