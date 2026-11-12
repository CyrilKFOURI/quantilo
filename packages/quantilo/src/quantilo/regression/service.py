"""Regression of a dataset over the filtered rows (067): read the columns once through DuckDB (a
reproducible sample beyond 200 000 rows), recognise the structure of the rows, fit, and keep the
fit so that the Advanced sections reuse it."""

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import duckdb
import numpy as np
import pandas as pd

from quantilo.catalog import Catalog, ColumnView
from quantilo.compare.analyze import label_sql, number_sql
from quantilo.correlations.matrix import Matrix
from quantilo.correlations.service import CorrelationService
from quantilo.errors import ColumnNotFound, ExplorerError
from quantilo.filters.model import Condition, filters_key
from quantilo.filters.scope import Where, scoped_source, where_for
from quantilo.ingestion.sql import ident
from quantilo.regression import analyze as an
from quantilo.regression import iv as ivs
from quantilo.regression import panel as panels
from quantilo.regression import resampling as rs
from quantilo.regression import robust as rb
from quantilo.regression import selection as sl
from quantilo.regression import stability as st
from quantilo.regression import terms as tm
from quantilo.regression.design import DesignError, LeftOutTerm, Variable, codes
from quantilo.regression.families import EXPOSURE_ROLE, MAX_CLASSES, FamilyRequest
from quantilo.regression.panel import PanelTable
from quantilo.regression.rules import (
    REGRESSOR_TYPES,
    TARGET_TYPES,
    ColumnInfo,
    RegressionPick,
    RegressionSuggestion,
    RowStructure,
    SeRequest,
    pick_columns,
    pick_target,
    structure,
)
from quantilo.typing.parsers import typed_sql
from quantilo.variable.source import SEED

MAX_ROWS = 200_000
FIT_CACHE = 3  # fits keep their arrays: a few only
RESULT_CACHE = 16
MAX_CANDIDATES = 5
NUMERIC = ("numeric_continuous", "numeric_discrete")
CATEGORY = ("categorical_nominal", "categorical_ordinal", "boolean")
REFUSED = {
    "datetime": "a date: add it as a trend, a month or a day of the week in Advanced › Terms "
    "(R-REG-TERM-08)",
    "text": "free text has no coefficient",
    "identifier": "an identifier: one level per row, no effect to estimate",
}
WEIGHT_ROLE = "weight"
YES = ("true", "1", "yes", "oui", "y")
Section = Literal[
    "standard_errors",
    "influence",
    "terms",
    "more",
    "panel",
    "families",
    "quantiles",
    "stability",
    "resampling",
    "selection",
]


class RegressionError(ExplorerError):
    def __init__(self, message: str, code: str = "regression") -> None:
        super().__init__(message, code=code, status_code=422)


@dataclass(frozen=True)
class Request:
    target: str | None = None
    columns: tuple[str, ...] | None = None  # None: the suggestion (R-REG-21)
    weights: str | None = None  # None: a column of role weight if any; "": none
    cluster: str | None = None
    se: SeRequest = "auto"
    focus: str | None = None
    family: FamilyRequest = "auto"  # 069: "auto" follows the rules (R-REG-01…07)
    exposure: str | None = None  # 069: None: a column of role exposure if any; "": none
    terms: tuple[str, ...] = ()  # 070: `spline:age`, `interaction:a|b`, `log:x`, …
    center: bool = False  # 070: numbers centred at their mean (R-REG-TERM-05)
    coding: tm.Coding = "reference"  # 070: categories against a reference or their average


class RegressionService:
    def __init__(self, columns: Catalog, correlations: CorrelationService | None = None) -> None:
        self.columns = columns
        self.correlations = correlations or CorrelationService(columns)
        self._fits: OrderedDict[tuple[Any, ...], an.Fitted] = OrderedDict()
        self._results: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

    # --- public ---------------------------------------------------------------------------------

    def suggestion(
        self, dataset_id: str, target: str | None = None, conditions: list[Condition] | None = None
    ) -> RegressionSuggestion:
        """R-REG-20 and R-REG-21: the target and up to 5 columns, from the correlations."""
        conditions = conditions or []
        views = [v for v in self.columns.columns(dataset_id) if not v.excluded]
        infos = [ColumnInfo(v.name, v.type, v.role) for v in views]
        candidates = [v.name for v in views if v.type in TARGET_TYPES and v.role != WEIGHT_ROLE]
        regressors = [v.name for v in views if v.type in REGRESSOR_TYPES and v.role != WEIGHT_ROLE]
        matrix = self._matrix(dataset_id, conditions)
        if target is not None:
            self._view(views, target)
            chosen = RegressionPick(target, "Chosen in the settings.")
        else:
            usable = [i for i in infos if i.name in candidates]
            chosen = pick_target(usable, matrix)
        picks = (
            pick_columns(chosen.column, matrix, set(regressors) - {chosen.column}) if chosen else []
        )
        return RegressionSuggestion(chosen, picks, candidates, regressors)

    def regression(
        self, dataset_id: str, request: Request, conditions: list[Condition] | None = None
    ) -> an.RegressionReport:
        return self._fit(dataset_id, request, conditions or []).result

    def section(
        self,
        dataset_id: str,
        name: Section,
        request: Request,
        conditions: list[Condition] | None = None,
    ) -> (
        an.StandardErrorTable
        | an.InfluenceTable
        | an.TermsTable
        | an.MoreChecks
        | PanelTable
        | an.FamilyAlternatives
        | rb.QuantileProcess
        | st.StabilityTable
        | rs.ResamplingTable
        | sl.SelectionTable
    ):
        conditions = conditions or []
        key = (*self._key(dataset_id, request, conditions), name)
        if key in self._results:
            self._results.move_to_end(key)
            return self._results[key]
        fit = self._fit(dataset_id, request, conditions)
        builders = {
            "standard_errors": an.standard_errors,
            "influence": an.influence_table,
            "terms": an.terms_table,
            "more": an.more,
            "families": an.family_alternatives,
            "panel": panels.section,  # 071
            "quantiles": rb.process,  # 072: Advanced › More
            "stability": st.section,  # 073
            "resampling": rs.section,  # 077
            "selection": sl.section,  # 077: Advanced › More
        }
        if name not in builders:
            raise RegressionError(f"Unknown section {name!r}.", "unknown_section")
        found = an.finite(builders[name](fit))
        self._results[key] = found
        if len(self._results) > RESULT_CACHE:
            self._results.popitem(last=False)
        return found

    def instruments(
        self,
        dataset_id: str,
        request: Request,
        endogenous: tuple[str, ...],
        instruments: tuple[str, ...],
        conditions: list[Condition] | None = None,
    ) -> ivs.IvTable:
        """071: two-stage least squares, only with the endogenous columns and instruments the
        user names (R-REG-IV-01)."""
        conditions = conditions or []
        request = self._resolve(dataset_id, request, conditions)
        assert request.target is not None and request.columns is not None
        endogenous = tuple(dict.fromkeys(endogenous))
        instruments = tuple(dict.fromkeys(instruments))
        if not endogenous or not instruments:
            raise RegressionError(
                "Name the endogenous column and at least one instrument: IV is never guessed "
                "(R-REG-IV-01).",
                "iv_names",
            )
        for name in endogenous:
            if name not in request.columns:
                raise RegressionError(
                    f"`{name}` is not among the explanatory columns: the endogenous column must "
                    "be one of them.",
                    "iv_endogenous",
                )
        for name in instruments:
            if name == request.target or name in request.columns:
                raise RegressionError(
                    f"`{name}` is already in the model: an instrument stays out of it, it may "
                    "reach the target only through the endogenous column.",
                    "iv_instrument",
                )
        key = (*self._key(dataset_id, request, conditions), "iv", endogenous, instruments)
        if key in self._results:
            self._results.move_to_end(key)
            return self._results[key]
        augmented = replace(request, columns=(*request.columns, *instruments))
        inputs = self._load(dataset_id, augmented, conditions)
        try:
            found = an.finite(ivs.estimate(inputs, list(endogenous), list(instruments)))
        except (DesignError, ivs.IvError) as error:
            raise RegressionError(str(error), "iv") from error
        self._results[key] = found
        if len(self._results) > RESULT_CACHE:
            self._results.popitem(last=False)
        return found

    def robust(
        self,
        dataset_id: str,
        request: Request,
        quantile: float = 0.5,
        conditions: list[Condition] | None = None,
    ) -> rb.RobustTable:
        """072: Huber, Tukey and one quantile side by side with least squares."""
        conditions = conditions or []
        key = (*self._key(dataset_id, request, conditions), "robust", quantile)
        if key in self._results:
            self._results.move_to_end(key)
            return self._results[key]
        fit = self._fit(dataset_id, request, conditions)
        try:
            found = an.finite(rb.table(fit, quantile))
        except rb.RobustError as error:
            raise RegressionError(str(error), "quantile") from error
        self._results[key] = found
        if len(self._results) > RESULT_CACHE:
            self._results.popitem(last=False)
        return found

    def partial(
        self,
        dataset_id: str,
        request: Request,
        column: str,
        conditions: list[Condition] | None = None,
    ) -> tm.PartialPlot:
        """070: the added-variable and component + residual plots of one column, asked for only
        once it is chosen (R-REG-TERM-11)."""
        conditions = conditions or []
        key = (*self._key(dataset_id, request, conditions), "partial", column)
        if key in self._results:
            self._results.move_to_end(key)
            return self._results[key]
        found = an.finite(an.partial_plot(self._fit(dataset_id, request, conditions), column))
        self._results[key] = found
        if len(self._results) > RESULT_CACHE:
            self._results.popitem(last=False)
        return found

    def clear(self) -> None:
        self._fits.clear()
        self._results.clear()

    # --- internals ------------------------------------------------------------------------------

    def _matrix(self, dataset_id: str, conditions: list[Condition]) -> Matrix | None:
        try:
            return self.correlations.correlations(dataset_id, "spearman", None, conditions).matrix
        except ExplorerError:
            return None

    @staticmethod
    def _view(views: list[ColumnView], name: str) -> ColumnView:
        for view in views:
            if view.name == name:
                return view
        raise ColumnNotFound(name, [v.name for v in views])

    def _key(
        self, dataset_id: str, request: Request, conditions: list[Condition]
    ) -> tuple[Any, ...]:
        dataset = self.columns.ready(dataset_id)
        views = self.columns.columns(dataset_id)
        return (
            dataset_id,
            dataset.imported_at,
            dataset.types_version,
            tuple((v.name, v.type, v.role, v.excluded) for v in views),
            request,
            filters_key(conditions),
        )

    def _resolve(self, dataset_id: str, request: Request, conditions: list[Condition]) -> Request:
        if request.target is not None and request.columns is not None:
            return request
        found = self.suggestion(dataset_id, request.target, conditions)
        if found.target is None:
            raise RegressionError("No numeric or yes/no column to explain.", "no_target")
        columns = request.columns
        if columns is None:
            columns = tuple(p.column for p in found.columns)
        return replace(request, target=found.target.column, columns=columns)

    def _fit(self, dataset_id: str, request: Request, conditions: list[Condition]) -> an.Fitted:
        request = self._resolve(dataset_id, request, conditions)
        key = self._key(dataset_id, request, conditions)
        if key in self._fits:
            self._fits.move_to_end(key)
            return self._fits[key]
        inputs = self._load(dataset_id, request, conditions)
        try:
            fitted = an.run(inputs)
        except DesignError as error:
            raise RegressionError(str(error), "design") from error
        proposed = panels.proposal(fitted)  # 071: R-REG-14
        if proposed is not None:
            fitted.result.checks.append(proposed)
        for rule in (st.proposal, sl.proposal):  # 073: R-REG-15; 077: R-REG-16
            found = rule(fitted)
            if found is not None:
                fitted.result.checks.append(found)
        self._fits[key] = fitted
        if len(self._fits) > FIT_CACHE:
            self._fits.popitem(last=False)
        return fitted

    def _load(self, dataset_id: str, request: Request, conditions: list[Condition]) -> an.Inputs:
        assert request.target is not None and request.columns is not None
        dataset = self.columns.ready(dataset_id)
        all_views = self.columns.columns(dataset_id)
        views = [v for v in all_views if not v.excluded]
        target = self._view(views, request.target)
        if target.type not in (*NUMERIC, *CATEGORY):
            raise RegressionError(
                f"`{target.name}` is {target.type.replace('_', ' ')}: pick a number or a "
                "yes/no column to explain."
            )
        weights = self._weights(views, request)
        exposure = self._exposure(views, request, target.name)
        try:
            plan = tm.plan_of(request.terms, request.center, request.coding, request.columns)
        except tm.TermError as error:
            raise RegressionError(str(error), "term") from error
        dated: list[tuple[ColumnView, str]] = []
        for spec in plan.dates():
            view = self._view(views, spec.columns[0])
            if view.type != "datetime":
                raise RegressionError(
                    f"`{view.name}` is not a date: no {tm.DATE_LABELS[spec.kind].lower()} to take "
                    "from it (R-REG-TERM-08).",
                    "term",
                )
            dated.append((view, spec.kind))
        for name in plan.columns():  # a curve or an interaction on a date: its trend
            view = self._view(views, name)
            if view.type == "datetime" and (view, "trend") not in dated:
                dated.append((view, "trend"))
        with_terms = {view.name for view, _ in dated}
        refused: list[LeftOutTerm] = []
        chosen: list[ColumnView] = []
        # 070: a term reads its columns even when they are not among the explanatory ones
        for name in dict.fromkeys([*request.columns, *plan.columns()]):
            if name == target.name:
                continue
            view = self._view(views, name)
            if weights is not None and name == weights[0].name:
                continue
            if view.type in REFUSED:
                if name in request.columns and name not in with_terms:
                    refused.append(LeftOutTerm(name, name, REFUSED[view.type]))
            else:
                chosen.append(view)
        cluster = self._view(views, request.cluster) if request.cluster else None
        dates, periods, entities = self._candidates(views, target.name)
        loaded = self._read(
            dataset.storage_path,
            where_for(all_views, conditions),
            target,
            chosen,
            weights[0] if weights else None,
            cluster,
            dates,
            sorted({*periods, *entities} - set(dates), key=[v.name for v in views].index),
            {v.name: v for v in views},
            exposure[0] if exposure else None,
            list({view.name: view for view, _ in dated}.values()),
        )
        frame, total = loaded
        rows = len(frame)
        shape = self._structure(frame, dates, periods, entities, rows)
        if shape.kind == "series" and shape.date is not None:
            frame = frame.sort_values(shape.date, kind="stable").reset_index(drop=True)
        y, binary, outcomes = self._target(target, cast(pd.Series, frame["__y"]))
        variables = [
            self._variable(view, cast(pd.Series, frame[f"__x{i}"])) for i, view in enumerate(chosen)
        ]
        variables += self._date_variables(frame, dated)
        inputs = an.Inputs(
            target.name,
            y,
            binary,
            variables,
            refused,
            frame["__row"].to_numpy(dtype=np.int64),
            rows,
            total,
            total > rows,
            shape,
            requested=request.se,
            focus=request.focus,
            plan=plan,
        )
        inputs.target_role, inputs.outcomes, inputs.family = target.role, outcomes, request.family
        if exposure is not None:
            inputs.exposure, inputs.exposure_reason = exposure[0].name, exposure[1]
            inputs.exposure_values = frame["__e"].to_numpy(dtype=float, na_value=np.nan)
        if weights is not None:
            inputs.weights, inputs.weights_reason = weights[0].name, weights[1]
            inputs.weight_values = frame["__w"].to_numpy(dtype=float, na_value=np.nan)
        if shape.kind == "series" and shape.date is not None:  # 073: the dates of the rows
            inputs.dates = _date_labels(cast(pd.Series, frame[shape.date]))
        if cluster is not None:
            inputs.cluster = cluster.name
            inputs.cluster_codes = _codes(cast(pd.Series, frame["__g"]))
        if shape.kind == "panel" and shape.entity and shape.period:
            inputs.entity_codes = _codes(cast(pd.Series, frame[shape.entity]))
            inputs.period_codes = _codes(cast(pd.Series, frame[shape.period]))
            inputs.period_order = _period_order(frame, shape.period)
        return inputs

    def _weights(self, views: list[ColumnView], request: Request) -> tuple[ColumnView, str] | None:
        if request.weights == "":
            return None
        if request.weights is not None:
            view = self._view(views, request.weights)
            if view.type not in NUMERIC:
                raise RegressionError(f"`{view.name}` is not a number: it cannot weight the rows.")
            return view, f"Rows weighted by `{view.name}`, chosen in the settings (WLS)."
        for view in views:
            if view.role == WEIGHT_ROLE and view.type in NUMERIC:
                return view, (
                    f"R-REG-SE-04: `{view.name}` has the role weight: a row that stands for 10 "
                    "counts 10 times (WLS)."
                )
        return None

    def _exposure(
        self, views: list[ColumnView], request: Request, target: str
    ) -> tuple[ColumnView, str] | None:
        """R-REG-02: the exposure of a count (a duration, a population), its log the offset."""
        if request.exposure == "":
            return None
        if request.exposure is not None:
            view = self._view(views, request.exposure)
            if view.type not in NUMERIC:
                raise RegressionError(f"`{view.name}` is not a number: it cannot be an exposure.")
            return view, f"`{view.name}` is the exposure, chosen in the settings."
        for view in views:
            if view.role == EXPOSURE_ROLE and view.type in NUMERIC and view.name != target:
                return view, (
                    f"R-REG-02: `{view.name}` has the role exposure: counts are compared per unit "
                    "of it (its log is the offset)."
                )
        return None

    @staticmethod
    def _candidates(views: list[ColumnView], target: str) -> tuple[list[str], list[str], list[str]]:
        """Dates (a series), periods and entities (a panel) worth checking."""
        from quantilo.ml import panel as panels

        types = {v.name: v.type for v in views}
        others = [v for v in views if v.name != target]
        dates = [v.name for v in others if v.type == "datetime"][:MAX_CANDIDATES]
        periods = panels.rank_period_candidates(
            [
                v.name
                for v in others
                if v.type == "datetime"
                or (
                    v.type in ("numeric_discrete", "categorical_ordinal", "categorical_nominal")
                    and panels._hinted(v.name, panels.PERIOD_HINTS)
                )
            ],
            types,
        )[:MAX_CANDIDATES]
        entities = panels.rank_entity_candidates(
            [
                v.name
                for v in others
                if v.type
                in ("identifier", "categorical_nominal", "categorical_ordinal", "numeric_discrete")
                and (
                    v.role in ("entity", "customer_key", "identifier")
                    or panels._hinted(v.name, panels.ENTITY_HINTS)
                )
            ],
            types,
        )[:MAX_CANDIDATES]
        return dates, periods, entities

    @staticmethod
    def _read(
        storage: str,
        where: Where,
        target: ColumnView,
        chosen: list[ColumnView],
        weights: ColumnView | None,
        cluster: ColumnView | None,
        dates: list[str],
        groups: list[str],
        by_name: dict[str, ColumnView],
        exposure: ColumnView | None = None,
        term_dates: list[ColumnView] | None = None,
    ) -> tuple[pd.DataFrame, int]:
        source = scoped_source(storage, where, numbered=True)

        def value(view: ColumnView) -> str:
            return number_sql(view) if view.type in NUMERIC else label_sql(view)

        expressions = ["file_row_number AS __row", f"{value(target)} AS __y"]
        expressions += [f"{value(v)} AS __x{i}" for i, v in enumerate(chosen)]
        if weights is not None:
            expressions.append(f"{number_sql(weights)} AS __w")
        if exposure is not None:
            expressions.append(f"{number_sql(exposure)} AS __e")
        if cluster is not None:
            dated = cluster.type == "datetime"
            group = typed_sql(cluster.name, cluster.parser) if dated else label_sql(cluster)
            expressions.append(f"{group} AS __g")
        for name in dates:
            expressions.append(f"{typed_sql(name, by_name[name].parser)} AS {ident(name)}")
        for i, view in enumerate(term_dates or []):
            expressions.append(f"{typed_sql(view.name, view.parser)} AS __t{i}")
        for name in groups:
            view = by_name[name]
            expression = number_sql(view) if view.type in NUMERIC else label_sql(view)
            expressions.append(f"{expression} AS {ident(name)}")
        connection = duckdb.connect()
        try:
            (total,) = connection.execute(f"SELECT count(*) FROM {source}").fetchone() or (0,)
            query = f"SELECT {', '.join(expressions)} FROM {source}"
            if total > MAX_ROWS:
                picked = (
                    f"SELECT file_row_number FROM {source} "
                    f"ORDER BY hash(file_row_number + {SEED}) LIMIT {MAX_ROWS}"
                )
                query += f" WHERE file_row_number IN ({picked})"
            frame = connection.execute(query + " ORDER BY file_row_number").df()
        finally:
            connection.close()
        return frame, int(total)

    @staticmethod
    def _structure(
        frame: pd.DataFrame, dates: list[str], periods: list[str], entities: list[str], rows: int
    ) -> RowStructure:
        from quantilo.ml import panel as panels

        present = [c for c in {*dates, *periods, *entities} if c in frame.columns]
        distinct = {c: int(cast(pd.Series, frame[c]).nunique()) for c in present}
        periods = [p for p in periods if p in frame.columns and distinct[p] >= panels.MIN_PERIODS]
        entities = [e for e in entities if e in frame.columns and 2 <= distinct[e] < rows]
        return structure(
            frame, [d for d in dates if d in frame.columns], periods, entities, rows, distinct
        )

    @staticmethod
    def _target(
        view: ColumnView, values: pd.Series
    ) -> tuple[np.ndarray, tuple[str, str] | None, list[str]]:
        """The target as numbers: 0/1 for yes/no, codes (most frequent first) for 3 to 10
        classes (R-REG-07), and the labels of either."""
        if view.type in NUMERIC:
            numbers = values.to_numpy(dtype=float, na_value=np.nan)
            distinct = np.unique(numbers[np.isfinite(numbers)])
            if len(distinct) == 2:
                low, high = (float(d) for d in distinct)
                labels = (an.fmt(low), an.fmt(high))
                return (
                    np.where(np.isfinite(numbers), (numbers == high).astype(float), np.nan),
                    labels,
                    [],
                )
            if len(distinct) < 2:
                raise RegressionError(f"`{view.name}` takes a single value: nothing to explain.")
            return numbers, None, []
        labels = [None if pd.isna(v) else str(v) for v in values.tolist()]
        found, levels = codes(labels)
        if len(levels) > MAX_CLASSES:
            raise RegressionError(
                f"`{view.name}` has {len(levels)} levels: at most {MAX_CLASSES} for a multinomial "
                "model (R-REG-07). Merge levels, or pick a number or a yes/no column."
            )
        if len(levels) < 2:
            raise RegressionError(f"`{view.name}` takes a single value: nothing to explain.")
        if len(levels) > 2:
            return np.where(found >= 0, found.astype(float), np.nan), None, levels
        positive = next((i for i, level in enumerate(levels) if level.lower() in YES), 1)
        y = np.where(found >= 0, (found == positive).astype(float), np.nan)
        return y, (levels[1 - positive], levels[positive]), []

    @staticmethod
    def _date_variables(frame: pd.DataFrame, dated: list[tuple[ColumnView, str]]) -> list[Variable]:
        """070 (R-REG-TERM-08): a date as a trend in years, a month or a day of the week."""
        columns = list(dict.fromkeys(view.name for view, _ in dated))
        found: list[Variable] = []
        for view, kind in dated:
            values = cast(pd.Series, frame[f"__t{columns.index(view.name)}"])
            numbers, labels, _ = tm.date_values(values, kind)
            name = tm.date_name(view.name, kind)
            if labels is None:
                found.append(Variable(name, "num", numbers, source=view.name))
            else:
                found.append(Variable(name, "cat", numbers, labels, source=view.name))
        return found

    @staticmethod
    def _variable(view: ColumnView, values: pd.Series) -> Variable:
        if view.type in NUMERIC:
            return Variable(view.name, "num", values.to_numpy(dtype=float, na_value=np.nan))
        labels = [None if pd.isna(v) else str(v) for v in values.tolist()]
        found, levels = codes(labels)
        return Variable(view.name, "cat", found, levels)


def _codes(values: pd.Series) -> np.ndarray:
    labels = values.astype(str).where(values.notna())
    found, _ = pd.factorize(labels, use_na_sentinel=True)
    return np.asarray(found, dtype=np.int64)


def _date_labels(values: pd.Series) -> np.ndarray:
    """073: dates as text, the time left out when every one is at midnight."""
    parsed = pd.to_datetime(values, errors="coerce")
    midnight = bool((parsed.dropna() == parsed.dropna().dt.normalize()).all())
    text = parsed.dt.strftime("%Y-%m-%d" if midnight else "%Y-%m-%d %H:%M")
    return text.fillna("").to_numpy(dtype=object)


def _period_order(frame: pd.DataFrame, column: str) -> np.ndarray:
    """071: each row's rank among the periods, in time (dates) or value order."""
    from quantilo.ml import panel as panels

    return panels.period_ranks(frame, column)[0]
