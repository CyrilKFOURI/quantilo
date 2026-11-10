"""Experiments on a catalog's dataset: the columns that can describe one, the analysis through the
active filters (results cached in memory), a metric's current value for planning."""

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import duckdb
import numpy as np
import pandas as pd

from quantilo.catalog import Catalog, ColumnView
from quantilo.compare.analyze import label_sql, number_label_sql, number_sql
from quantilo.errors import ColumnNotFound
from quantilo.experiments import decisions as dec
from quantilo.experiments import policy_offer as po
from quantilo.experiments.analyze import (
    MAX_LEVELS,
    MIN_LEVELS,
    SEGMENT,
    TIME,
    UNIT,
    VARIANT,
    ExperimentDesign,
    ExperimentError,
    ExperimentMetric,
    ExperimentResult,
    MetricInput,
    MetricKind,
    analyze,
    guess_control,
)
from quantilo.experiments.bandit import BanditSimulation, simulate
from quantilo.experiments.plan import SampleSize, SampleSizeRequest, sample_size
from quantilo.filters.model import Condition, filters_key
from quantilo.filters.scope import scoped_source, where_for
from quantilo.ml import data as ml_data
from quantilo.ml.data import CATEGORICAL, NUMERIC
from quantilo.ml.data import Field as DataField
from quantilo.ml.target import DESCRIPTOR_ROLES
from quantilo.typing.parsers import typed_sql
from quantilo.typing.types import KEY_ROLES, NUMERIC_TYPES

if TYPE_CHECKING:  # scikit-learn loads on first use, not with `import quantilo`
    from quantilo.experiments import offpolicy as op
    from quantilo.experiments import policy as pl

CACHE_SIZE = 16
MAX_SEGMENT_LEVELS = 20
VARIANT_HINT = re.compile(
    r"variant|group|arm|treatment|bucket|experiment|cohort|version|^ab$|_ab$|^ab_|test", re.I
)
UNIT_HINT = re.compile(
    r"(^|_)(id|user|customer|client|visitor|session|account|member)(_|$)|id$", re.I
)
LEVEL_TYPES = ("categorical_nominal", "categorical_ordinal", "boolean", "numeric_discrete", "text")
UNIT_TYPES = ("identifier", "numeric_discrete", "categorical_nominal", "text")
COVARIATE_TYPES = NUMERIC | CATEGORICAL
PROPENSITY_HINT = re.compile(r"propens|pscore|prob|p_action|logging", re.I)
PLANNED_BALANCE = 1.05  # R-EXP-01: the largest level at most this many times the smallest


@dataclass
class PolicyModel:
    """A saved ML model (037) evaluated as a policy: its name, fitted pipeline and model card."""

    label: str
    pipeline: Any
    card: Any  # quantilo.ml.predict.ModelCard


@dataclass
class ExperimentVariantCandidate:
    column: str
    levels: list[str]
    counts: list[int]
    control: str
    planned: bool = True  # R-EXP-01: looks like an assigned variant, not an observed category
    why: str = ""


@dataclass
class ExperimentMetricCandidate:
    column: str
    kind: MetricKind


@dataclass
class ExperimentUnitCandidate:
    column: str
    rows_per_value: float


@dataclass
class ExperimentCandidates:
    variants: list[ExperimentVariantCandidate]
    metrics: list[ExperimentMetricCandidate]
    units: list[ExperimentUnitCandidate]
    times: list[str]
    segments: list[str]
    suggested: ExperimentDesign | None  # what the page opens on
    note: str | None = None  # R-EXP-01: no column looks like a variant


@dataclass
class MetricBaseline:
    column: str
    kind: Literal["rate", "mean"]
    value: float | None
    sd: float | None
    rows: int


def planned_variant(view: ColumnView, counts: list[int]) -> tuple[bool, str]:
    """R-EXP-01 (094, EXP-1): a column of 2 to 5 levels is an experiment's variant when its name
    says so (group, variant, arm, treatment, test…) or when its levels are balanced like a
    planned split, without describing the units (gender, region…). Otherwise it is an observed
    category: comparing its levels is an observational study, and a difference is not caused
    by the category."""
    hint = VARIANT_HINT.search(view.name)
    if hint:
        return True, f"Its name says a variant ({hint.group(0).lower()})."
    total = sum(counts) or 1
    shares = " / ".join(f"{100 * c / total:.0f} %" for c in counts)
    if view.role in DESCRIPTOR_ROLES:
        role = view.role.replace("_", " ")
        return False, f"Describes the units ({role}): an observed trait, not an assigned variant."
    if min(counts) and max(counts) <= PLANNED_BALANCE * min(counts):
        return True, f"Its levels are balanced ({shares}), as in a planned split."
    return False, (
        f"Not named like a variant and its levels are uneven ({shares}): an observed category, "
        "so a difference would not be caused by it."
    )


def observational_note(variants: list[ExperimentVariantCandidate]) -> str | None:
    """What the page says when no column looks like a variant (R-EXP-01)."""
    if not variants or any(v.planned for v in variants):
        return None
    names = ", ".join(v.column for v in variants[:4])
    return (
        f"No column looks like an experiment's variant (R-EXP-01): {names} "
        f"{'is an observed category' if len(variants) == 1 else 'are observed categories'}. "
        "You can still compare their levels, as an observational study: a difference would "
        "not prove a cause."
    )


def _distinct(view: ColumnView) -> int | None:
    value = (view.summary or {}).get("distinct_count")
    return value if isinstance(value, int) else None


def level_sql(view: ColumnView) -> str:
    """A value as a level: "1", not "1.0", for numbers; "True"/"False" for yes/no."""
    return number_label_sql(view) if view.type in NUMERIC_TYPES else label_sql(view)


def value_sql(view: ColumnView) -> str:
    """A metric as a number: yes/no as 1/0, infinities as missing."""
    if view.type == "boolean":
        typed = typed_sql(view.name, view.parser)
        return f"(CASE WHEN {typed} THEN 1.0 WHEN NOT {typed} THEN 0.0 END)"
    if view.type not in NUMERIC_TYPES:
        raise ExperimentError(
            f"{view.name} is {view.type.replace('_', ' ')}: a metric must be a number or yes/no.",
            "metrics",
        )
    number = number_sql(view)
    return f"(CASE WHEN isfinite({number}) THEN {number} END)"


def repeated_units(views: list[ColumnView], taken: set[str]) -> list[ExperimentUnitCandidate]:
    """Columns that name a unit (an identifier, a user_id…) and repeat over the rows."""
    found = []
    for view in views:
        if view.name in taken or view.type not in UNIT_TYPES:
            continue
        if (
            view.type != "identifier"
            and view.role not in KEY_ROLES
            and not UNIT_HINT.search(view.name)
        ):
            continue
        summary = view.summary or {}
        distinct = summary.get("distinct_count")
        valid = (summary.get("rows") or 0) - (summary.get("null_count") or 0)
        if isinstance(distinct, int) and 0 < distinct < valid:
            found.append(ExperimentUnitCandidate(view.name, valid / distinct))
    return sorted(found, key=lambda unit: -unit.rows_per_value)


class ExperimentEngine:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns
        self._cache: OrderedDict[tuple[Any, ...], ExperimentResult] = OrderedDict()
        self._policies: OrderedDict[tuple[Any, ...], tuple[pl.PolicyScores, pl.PolicyLearning]] = (
            OrderedDict()
        )

    def clear(self) -> None:
        self._cache.clear()
        self._policies.clear()

    def _views(self, dataset_id: str) -> dict[str, ColumnView]:
        return {v.name: v for v in self.columns.columns(dataset_id) if not v.excluded}

    @staticmethod
    def _view(views: dict[str, ColumnView], name: str) -> ColumnView:
        view = views.get(name)
        if view is None:
            raise ColumnNotFound(name, list(views))
        return view

    # --- Describe ------------------------------------------------------------------------------

    def candidates(self, dataset_id: str, conditions: list[Condition]) -> ExperimentCandidates:
        dataset = self.columns.ready(dataset_id)
        views = self._views(dataset_id)
        ordered = sorted(views.values(), key=lambda v: v.position)
        level_views = [
            v
            for v in ordered
            if v.type in LEVEL_TYPES and MIN_LEVELS <= (_distinct(v) or 0) <= MAX_LEVELS
        ]
        level_views.sort(key=lambda v: 0 if VARIANT_HINT.search(v.name) else 1)
        numbers = [
            v
            for v in ordered
            if (v.type == "boolean" or v.type in NUMERIC_TYPES) and v.role not in KEY_ROLES
        ]
        connection = duckdb.connect()
        try:
            source = scoped_source(
                dataset.storage_path, where_for(list(views.values()), conditions)
            )
            variants = []
            for view in level_views[:12]:
                rows = connection.execute(
                    f"SELECT l, count(*) FROM (SELECT {level_sql(view)} AS l FROM {source}) "
                    "WHERE l IS NOT NULL GROUP BY l ORDER BY l"
                ).fetchall()
                if not MIN_LEVELS <= len(rows) <= MAX_LEVELS:
                    continue
                levels = [str(level) for level, _ in rows]
                counts = [int(count) for _, count in rows]
                planned, why = planned_variant(view, counts)
                variants.append(
                    ExperimentVariantCandidate(
                        view.name, levels, counts, guess_control(levels, counts), planned, why
                    )
                )
            metrics = self._metric_kinds(connection, source, numbers)
        finally:
            connection.close()
        variants.sort(key=lambda v: not v.planned)  # stable: the named ones stay first
        variant_names = {v.column for v in variants}
        units = repeated_units(ordered, variant_names)
        times = [v.name for v in ordered if v.type == "datetime"]
        segments = [
            v.name
            for v in ordered
            if v.type in ("categorical_nominal", "categorical_ordinal", "boolean")
            and 2 <= (_distinct(v) or 0) <= MAX_SEGMENT_LEVELS
        ]
        suggested = None
        variant = variants[0].column if variants and variants[0].planned else None
        usable = [m for m in metrics if m.column != variant]
        if variant and usable:
            chosen = next((m for m in usable if m.kind == "rate"), usable[0])
            suggested = ExperimentDesign(
                variant=variant,
                metrics=[ExperimentMetric(column=chosen.column)],
                unit=units[0].column if units else None,
            )
        return ExperimentCandidates(
            variants,
            metrics,
            units,
            times,
            [s for s in segments if s != variant],
            suggested,
            observational_note(variants),
        )

    @staticmethod
    def _metric_kinds(
        connection: duckdb.DuckDBPyConnection, source: str, views: list[ColumnView]
    ) -> list[ExperimentMetricCandidate]:
        if not views:
            return []
        parts = []
        for i, view in enumerate(views):
            value = value_sql(view)
            parts.append(
                f"bool_and({value} IN (0, 1)) AS b{i}, "
                f"bool_and({value} >= 0 AND {value} = floor({value})) AS w{i}"
            )
        row = connection.execute(f"SELECT {', '.join(parts)} FROM {source}").fetchone() or ()
        found = []
        for i, view in enumerate(views):
            binary, whole = row[2 * i], row[2 * i + 1]
            kind: MetricKind
            if view.type == "boolean" or binary:
                kind = "rate"
            elif view.type == "numeric_discrete" and whole:
                kind = "count"
            else:
                kind = "mean"
            found.append(ExperimentMetricCandidate(view.name, kind))
        return found

    # --- Analyze -------------------------------------------------------------------------------

    def analyze(
        self, dataset_id: str, design: ExperimentDesign, conditions: list[Condition]
    ) -> ExperimentResult:
        dataset = self.columns.ready(dataset_id)
        views = self._views(dataset_id)
        named = [design.variant, design.unit, design.segment, design.time]
        for metric in design.metrics:
            named += [metric.column, metric.denominator, metric.pre]
        used = {name: self._view(views, name) for name in named if name}
        key = (
            dataset_id,
            dataset.imported_at,
            json.dumps(design.model_dump(mode="json"), sort_keys=True),
            tuple((v.name, v.type, v.role, v.parser.model_dump_json()) for v in used.values()),
            filters_key(conditions),
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        selected = [f"{level_sql(used[design.variant])} AS {VARIANT}"]
        if design.unit:
            selected.append(f"{level_sql(used[design.unit])} AS {UNIT}")
        if design.segment:
            selected.append(f"{level_sql(used[design.segment])} AS {SEGMENT}")
        if design.time:
            time = used[design.time]
            if time.type != "datetime":
                raise ExperimentError(f"{time.name} is not a date column.", "time")
            selected.append(f"CAST({typed_sql(time.name, time.parser)} AS TIMESTAMP) AS {TIME}")
        inputs = []
        for i, metric in enumerate(design.metrics):
            view = used[metric.column]
            selected.append(f"{value_sql(view)} AS m{i}")
            if metric.denominator:
                selected.append(f"{value_sql(used[metric.denominator])} AS d{i}")
            if metric.pre:
                selected.append(f"{value_sql(used[metric.pre])} AS p{i}")
            inputs.append(
                MetricInput(
                    metric,
                    f"m{i}",
                    f"d{i}" if metric.denominator else None,
                    f"p{i}" if metric.pre else None,
                    view.type == "boolean",
                    view.type == "numeric_discrete",
                )
            )
        connection = duckdb.connect()
        try:
            source = scoped_source(
                dataset.storage_path, where_for(list(views.values()), conditions), numbered=True
            )
            frame = connection.execute(
                f"SELECT * FROM (SELECT {', '.join(selected)}, file_row_number FROM {source}) "
                f"WHERE {VARIANT} IS NOT NULL ORDER BY file_row_number"
            ).df()
        finally:
            connection.close()
        taken = {name for name in named if name}
        repeated = (
            [(u.column, u.rows_per_value) for u in repeated_units(list(views.values()), taken)]
            if not design.unit
            else None
        )
        result = analyze(frame, design, inputs, repeated)
        result.policy_offer = self._policy_offer(views, result, taken)
        self._cache[key] = result
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    # --- Plan ----------------------------------------------------------------------------------

    def baseline(self, dataset_id: str, column: str, conditions: list[Condition]) -> MetricBaseline:
        """A metric's current value (rate or mean, with its spread) to start a plan from."""
        dataset = self.columns.ready(dataset_id)
        views = self._views(dataset_id)
        view = self._view(views, column)
        value = value_sql(view)
        connection = duckdb.connect()
        try:
            source = scoped_source(
                dataset.storage_path, where_for(list(views.values()), conditions)
            )
            mean, sd, rows, binary = connection.execute(
                f"SELECT avg({value}), stddev_samp({value}), count({value}), "
                f"bool_and({value} IN (0, 1)) FROM {source}"
            ).fetchone() or (None, None, 0, None)
        finally:
            connection.close()
        kind: Literal["rate", "mean"] = "rate" if view.type == "boolean" or binary else "mean"
        return MetricBaseline(
            column,
            kind,
            float(mean) if mean is not None else None,
            float(sd) if sd is not None else None,
            int(rows),
        )

    @staticmethod
    def sample_size(request: SampleSizeRequest) -> SampleSize:
        return sample_size(request)

    @staticmethod
    def bandit_simulation(request: SampleSizeRequest) -> BanditSimulation:
        return simulate(request)

    # --- Who should get which variant (055 § B) -----------------------------------------------

    @staticmethod
    def _covariate_candidates(views: list[ColumnView], taken: set[str]) -> list[str]:
        """Columns that may describe a unit: numbers, categories and yes/no, not a key, with at
        least two values (the preparation rules decide once the data is read)."""
        return [
            v.name
            for v in sorted(views, key=lambda v: v.position)
            if v.name not in taken
            and v.type in COVARIATE_TYPES
            and v.role not in KEY_ROLES
            and (_distinct(v) or 2) >= 2
        ]

    def _policy_offer(
        self, views: dict[str, ColumnView], result: ExperimentResult, taken: set[str]
    ) -> po.PolicyOffer:
        primary = next(m for m in result.metrics if m.primary)
        candidates = self._covariate_candidates(list(views.values()), taken)
        units = {v.label: v.units for v in result.variants}
        return po.offer(primary.kind, candidates, units)

    def policy(
        self,
        dataset_id: str,
        design: ExperimentDesign,
        covariates: list[str] | None,
        conditions: list[Condition],
        qini: bool = False,
    ) -> "pl.PolicyLearning":
        """Who should get which variant (R-PL1 to R-PL8), on the experiment as analyzed: the
        same units, the same primary metric. `qini`: add the Qini curve ("Advanced › More"),
        from the scores kept in memory, without a new cross-fitting."""
        from quantilo.experiments import policy as pl

        result = self.analyze(dataset_id, design, conditions)
        offer = result.policy_offer
        if offer is None or not offer.available:
            raise ExperimentError((offer.reason if offer else None) or "Not available.", "policy")
        resolved = result.design
        chosen = list(covariates) if covariates is not None else offer.covariates
        unknown = [name for name in chosen if name not in offer.covariates]
        if unknown:
            raise ExperimentError(
                f"{', '.join(unknown)} cannot describe the units here (a key, a date, a column "
                "of the experiment itself, or a type that is not used).",
                "covariates",
            )
        dataset = self.columns.ready(dataset_id)
        key = (
            dataset_id,
            dataset.imported_at,
            json.dumps(resolved.model_dump(mode="json"), sort_keys=True),
            tuple(chosen),
            filters_key(conditions),
        )
        if key in self._policies:
            self._policies.move_to_end(key)
            return self._with_qini(key, qini)
        views = self._views(dataset_id)
        metric = next(m for m in resolved.metrics if m.column == resolved.primary)
        fields = [self._field(self._view(views, name)) for name in chosen]
        selected = [
            f"{level_sql(self._view(views, resolved.variant))} AS {VARIANT}",
            f"{value_sql(self._view(views, metric.column))} AS {po.VALUE}",
            *(f"{ml_data._expression(f)} AS c{i}" for i, f in enumerate(fields)),
        ]
        if resolved.unit:
            selected.append(f"{level_sql(self._view(views, resolved.unit))} AS {UNIT}")
        connection = duckdb.connect()
        try:
            source = scoped_source(
                dataset.storage_path, where_for(list(views.values()), conditions), numbered=True
            )
            frame = connection.execute(
                f"SELECT * FROM (SELECT {', '.join(selected)}, file_row_number FROM {source}) "
                f"WHERE {VARIANT} IS NOT NULL ORDER BY file_row_number"
            ).df()
        finally:
            connection.close()
        frame = _typed(
            frame.rename(columns={f"c{i}": f.name for i, f in enumerate(fields)}), fields
        )
        if resolved.unit:
            frame = _units(frame, fields, metric.kind == "rate")
        frame, sampled = _sample_by_variant(frame)
        order = [v.label for v in result.variants]
        kind: Literal["rate", "mean", "count"] = (
            "mean" if metric.kind not in ("rate", "count") else metric.kind
        )
        found = pl.scores(frame, fields, kind, order)
        if isinstance(found, pl.PolicyUnavailable):
            raise ExperimentError(found.reason, "policy")
        learnt = pl.learn_scores(found, metric.column, sampled, resolved.alpha)
        self._policies[key] = (found, learnt)
        if len(self._policies) > CACHE_SIZE:
            self._policies.popitem(last=False)
        return self._with_qini(key, qini)

    def _with_qini(self, key: tuple[Any, ...], qini: bool) -> "pl.PolicyLearning":
        from quantilo.experiments import policy as pl

        found, learnt = self._policies[key]
        if not qini:
            return learnt
        return replace(learnt, qini=pl.qini_curves(found))

    @staticmethod
    def _field(view: ColumnView) -> DataField:
        return DataField(view.name, view.type, view.parser, view.role, view.order)

    # --- Evaluate a policy from logged decisions (055 § C) ------------------------------------

    def decisions(self, dataset_id: str, conditions: list[Condition]) -> dec.LoggedDecisions:
        """R-PO1: does the dataset look like a log of decisions (an action, its reward,
        covariates, optionally the propensity)?"""
        dataset = self.columns.ready(dataset_id)
        views = self._views(dataset_id)
        ordered = sorted(views.values(), key=lambda v: v.position)
        levels = {v.name: _distinct(v) or 0 for v in ordered}
        maybe = [
            v
            for v in ordered
            if v.type in NUMERIC_TYPES
            and (v.role == dec.PROPENSITY_ROLE or PROPENSITY_HINT.search(v.name))
        ]
        valid: set[str] = set()
        connection = duckdb.connect()
        try:
            source = scoped_source(
                dataset.storage_path, where_for(list(views.values()), conditions)
            )
            for view in maybe:
                number = number_sql(view)
                low, high = connection.execute(
                    f"SELECT min({number}), max({number}) FROM {source}"
                ).fetchone() or (None, None)
                if low is not None and high is not None and low > 0 and high <= 1:
                    valid.add(view.name)
            found = dec.detect(ordered, levels, valid)
            if found.action:
                view = views[found.action]
                rows = connection.execute(
                    f"SELECT DISTINCT l FROM (SELECT {level_sql(view)} AS l FROM {source}) "
                    "WHERE l IS NOT NULL ORDER BY l LIMIT 11"
                ).fetchall()
                found.actions = [str(level) for (level,) in rows]
        finally:
            connection.close()
        return found

    def evaluate_policy(
        self,
        dataset_id: str,
        request: dec.DecisionsRequest,
        conditions: list[Condition],
        models: list[PolicyModel] | None = None,
    ) -> "op.PolicyEvaluation":
        """R-PO2 to R-PO6: diagnostics, then DM, IPS, SNIPS and DR for the logged decisions,
        every constant policy, the learnt rule and each saved model."""
        from quantilo.experiments import offpolicy as op

        dataset = self.columns.ready(dataset_id)
        views = self._views(dataset_id)
        action = self._view(views, request.action)
        reward = self._view(views, request.reward)
        if request.action == request.reward:
            raise ExperimentError("The action and the reward must be two columns.", "reward")
        taken = {name for name in (request.action, request.reward, request.propensity) if name}
        candidates = self._covariate_candidates(list(views.values()), taken)
        chosen = list(request.covariates) if request.covariates is not None else candidates
        unknown = [name for name in chosen if name not in candidates]
        if unknown:
            raise ExperimentError(
                f"{', '.join(unknown)} cannot describe the units here.", "covariates"
            )
        fields = [self._field(self._view(views, name)) for name in chosen]
        selected = [
            f"{level_sql(action)} AS {dec.ACTION}",
            f"{value_sql(reward)} AS {dec.REWARD}",
            *(f"{ml_data._expression(f)} AS c{i}" for i, f in enumerate(fields)),
        ]
        if request.propensity:
            propensity = self._view(views, request.propensity)
            if propensity.type not in NUMERIC_TYPES:
                raise ExperimentError(f"{propensity.name} is not a number.", "propensity")
            selected.append(f"{number_sql(propensity)} AS {dec.PROPENSITY}")
        connection = duckdb.connect()
        try:
            source = scoped_source(
                dataset.storage_path, where_for(list(views.values()), conditions), numbered=True
            )
            frame = connection.execute(
                f"SELECT * FROM (SELECT {', '.join(selected)}, file_row_number AS __row "
                f"FROM {source}) WHERE {dec.ACTION} IS NOT NULL AND {dec.REWARD} IS NOT NULL "
                "ORDER BY __row"
            ).df()
            frame = _typed(
                frame.rename(columns={f"c{i}": f.name for i, f in enumerate(fields)}), fields
            )
            dropped = 0
            if request.propensity:
                p: Any = pd.Series(pd.to_numeric(frame[dec.PROPENSITY], errors="coerce"))
                ok = p.gt(0) & p.le(1)
                dropped = int((~ok).sum())
                frame = frame[ok].reset_index(drop=True)
            frame, sampled = _sample(frame)
            extra, skipped = _model_policies(
                connection, source, list(views.values()), frame, request, models or []
            )
        finally:
            connection.close()
        evaluated = op.evaluate(
            frame,
            fields,
            request.action,
            request.reward,
            bool(request.propensity),
            sampled,
            extra,
            skipped,
            dropped,
            request.alpha,
            request.learn,
        )
        if isinstance(evaluated, op.EvaluationUnavailable):
            raise ExperimentError(evaluated.reason, "action")
        return evaluated


def _model_policies(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    views: list[ColumnView],
    frame: Any,
    request: dec.DecisionsRequest,
    models: list[PolicyModel],
) -> "tuple[list[op.CandidatePolicy], list[str]]":
    """Each saved model as a policy on the rows of the log (R-PO5), or why it cannot be one."""
    if not models:
        return [], []
    from quantilo.experiments import offpolicy as op
    from quantilo.ml import predict as use

    actions = sorted(frame[dec.ACTION].dropna().astype(str).unique())
    names = {v.name for v in views}
    connection.register("__policy_rows", pd.DataFrame({"r": frame["__row"].to_numpy()}))
    extra: list[op.CandidatePolicy] = []
    skipped: list[str] = []
    try:
        for i, model in enumerate(models):
            card = model.card
            inputs = [c.name for c in card.columns]
            missing = [name for name in inputs if name not in names]
            if missing:
                skipped.append(f"{model.label}: the log has no {', '.join(missing[:3])}.")
                continue
            rows = use.read_rows(
                connection,
                source,
                card,
                views,
                where="WHERE file_row_number IN (SELECT r FROM __policy_rows)",
            )
            rows = rows.set_index("__row").reindex(frame["__row"].to_numpy()).reset_index(drop=True)
            numeric = {c.name for c in card.columns if c.type in NUMERIC}

            def predict(
                data: pd.DataFrame,
                card: Any = card,
                pipeline: Any = model.pipeline,
                numeric: set[str] = numeric,
            ) -> Any:
                x = data[[c.name for c in card.columns]].copy()
                for name in numeric:
                    x[name] = pd.Series(pd.to_numeric(x[name], errors="coerce"), dtype=float)
                if card.task == "regression":
                    return pipeline.predict(x)
                return pipeline.predict_proba(x)

            try:
                probabilities = op.model_policy(
                    predict,
                    card.task,
                    card.target.name,
                    [str(label) for label in card.labels],
                    card.positive,
                    inputs,
                    rows,
                    request.action,
                    actions,
                    request.reward,
                )
            except ValueError as error:
                skipped.append(f"{model.label}: {error}")
                continue
            how = (
                f"the action {model.label} predicts"
                if card.target.name == request.action
                else f"the action that maximizes {model.label}'s predicted {request.reward}"
            )
            extra.append(
                op.CandidatePolicy(
                    f"model:{i}", model.label, f"For each unit, {how}.", probabilities
                )
            )
    finally:
        connection.unregister("__policy_rows")
    return extra, skipped


def _typed(frame: Any, fields: list[DataField]) -> Any:
    """Numbers as floats, as `ml_data.load` reads them."""
    for f in fields:
        if f.type in NUMERIC and f.name in frame:
            frame[f.name] = pd.Series(pd.to_numeric(frame[f.name], errors="coerce"), dtype=float)
    return frame


def _units(frame: Any, fields: list[DataField], rate: bool) -> Any:
    """One row per unit, as the analysis (R-PL3): a unit seen in two variants is left out; the
    metric is 1 if any row is (a rate), else summed; each covariate keeps the first value."""
    frame = frame[frame[UNIT].notna()]
    mixed = frame.groupby(UNIT, observed=True)[VARIANT].nunique()
    frame = frame[~frame[UNIT].isin(mixed[mixed > 1].index)]
    rules: dict[str, str] = {VARIANT: "first", po.VALUE: "max" if rate else "sum"}
    rules.update({f.name: "first" for f in fields})
    grouped = frame.groupby(UNIT, sort=False, observed=True)
    out = grouped.agg(rules)
    out.loc[grouped[po.VALUE].count() == 0, po.VALUE] = np.nan
    return out.reset_index(drop=True)


def _sample(frame: Any) -> tuple[Any, bool]:
    """At most MAX_UNITS rows, drawn reproducibly (R-PC6), kept in file order."""
    from quantilo.experiments import policy_core as core

    if len(frame) <= core.MAX_UNITS:
        return frame.reset_index(drop=True), False
    rng = np.random.default_rng(core.SEED)
    keep = np.sort(rng.choice(len(frame), core.MAX_UNITS, replace=False))
    return frame.iloc[keep].reset_index(drop=True), True


def _sample_by_variant(frame: Any) -> tuple[Any, bool]:
    """At most MAX_UNITS units of a randomized test, small variants kept whole (R-PC6): each
    variant gets an equal share of the room, a smaller one gives what it does not use to the
    others. A random part of each variant keeps the units' profile, so the observed shares stay
    the propensities."""
    from quantilo.experiments import policy_core as core

    if len(frame) <= core.MAX_UNITS:
        return frame.reset_index(drop=True), False
    sizes = frame[VARIANT].value_counts().sort_values()
    room, left = core.MAX_UNITS, len(sizes)
    quota: dict[Any, int] = {}
    for label, size in sizes.items():
        quota[label] = min(int(size), room // left)
        room -= quota[label]
        left -= 1
    rng = np.random.default_rng(core.SEED)
    keep: list[int] = []
    labels = frame[VARIANT].to_numpy()
    for label, count in quota.items():
        rows = np.flatnonzero(labels == label)
        keep += (
            rng.choice(rows, count, replace=False).tolist() if count < len(rows) else rows.tolist()
        )
    return frame.iloc[np.sort(np.array(keep))].reset_index(drop=True), True
