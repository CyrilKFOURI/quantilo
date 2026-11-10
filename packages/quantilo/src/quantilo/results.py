"""Results as plain data: `to_dict()` for any result (JSON-safe), `to_pandas()` for the tabular
ones. The results themselves are the app's dataclasses, unchanged."""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from quantilo.catalog import ColumnView

if TYPE_CHECKING:
    from quantilo.correlations.matrix import Matrix
    from quantilo.experiments.analyze import ExperimentResult
    from quantilo.kpi.compute import KpiResult
    from quantilo.profiling.overview import Overview
    from quantilo.rows.view import RowsPage


def to_dict(result: Any) -> Any:
    """A result as dicts, lists and plain numbers (NaN and infinities become None): ready for
    `json.dumps`."""
    if type(result).__name__ in ("TrainedModel", "FittedSegments", "FoundUnusualRows"):
        return to_dict(result.result)  # its run, without the pipeline or the labels
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return {
            f.name: to_dict(getattr(result, f.name))
            for f in dataclasses.fields(result)
            if f.name not in ("pipeline", "held_out")  # held_out: every held-out row (036)
        }
    if isinstance(result, BaseModel):
        return to_dict(result.model_dump())
    if isinstance(result, dict):
        return {str(key): to_dict(value) for key, value in result.items()}
    if isinstance(result, list | tuple | set):
        return [to_dict(value) for value in result]
    if isinstance(result, np.ndarray):
        return to_dict(result.tolist())
    if isinstance(result, np.generic):
        return to_dict(result.item())
    if isinstance(result, float):
        return result if math.isfinite(result) else None
    if isinstance(result, enum.Enum):
        return result.value
    if isinstance(result, dt.date | dt.datetime | dt.time):
        return result.isoformat()
    if isinstance(result, Path):
        return str(result)
    if isinstance(result, pd.DataFrame):
        return to_dict(result.to_dict(orient="records"))
    return result


def _rows(items: list[Any]) -> pd.DataFrame:
    return pd.DataFrame([to_dict(item) for item in items])


def _matrix(matrix: Matrix) -> pd.DataFrame:
    names = [member.name for member in matrix.columns]
    return pd.DataFrame(matrix.values, index=names, columns=names, dtype=float)


def _structure(structure: Any) -> pd.DataFrame:
    """One row per column, its loading on each component shown (078)."""
    names = [f"PC{i + 1}" for i in range(structure.shown)]
    return pd.DataFrame(structure.loadings, index=structure.columns, columns=names, dtype=float)


def _factors(factors: Any) -> pd.DataFrame:
    """One row per column: its loading on each factor, communality, uniqueness and KMO (078)."""
    names = [f"F{i + 1}" for i in range(factors.k)]
    frame = pd.DataFrame(factors.loadings, index=factors.columns, columns=names, dtype=float)
    if factors.columns:
        frame["communality"] = factors.communalities
        frame["uniqueness"] = factors.uniqueness
        frame["kmo"] = factors.msa
    return frame


def _overview(overview: Overview) -> pd.DataFrame:
    frame = pd.DataFrame({name: to_dict(stats) for name, stats in overview.column_stats.items()})
    return frame.T.rename_axis("column").reset_index()


def _page(page: RowsPage) -> pd.DataFrame:
    return pd.DataFrame(page.rows, columns=["row", *(c.name for c in page.columns)])


def _kpi(result: KpiResult) -> pd.DataFrame:
    labels = [p.label for p in result.periods]
    frames = []
    for line in result.series:
        if len(line.values) != len(labels):
            continue
        frames.append(
            pd.DataFrame(
                {
                    "measure": line.measure,
                    "split": line.g1,
                    "split2": line.g2,
                    "period": labels,
                    "value": line.values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _shifts(changes: Any) -> pd.DataFrame:
    keep = ["column", "type", "kind", "psi", "drift", "wasserstein", "ks", "ks_p", "chi2", "chi2_p"]
    frame = _rows(changes.distributions)
    return frame.loc[:, [c for c in keep if c in frame.columns]] if len(frame) else frame


def _experiment(result: ExperimentResult) -> pd.DataFrame:
    """One row per metric and variant against the control."""
    rows = []
    for metric in result.metrics:
        for comparison in metric.comparisons:
            row = to_dict(comparison)
            row.pop("checks", None)
            rows.append({"kind": metric.kind, "primary": metric.primary, **row})
    return pd.DataFrame(rows)


def _bandit_simulation(simulation: Any) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "units": simulation.cumulative_units,
            "regret_fixed": simulation.regret_fixed,
            "regret_bandit": simulation.regret_bandit,
            "regret_bandit_low": simulation.regret_bandit_low,
            "regret_bandit_high": simulation.regret_bandit_high,
        }
    )


def _tables() -> list[tuple[type, Callable[[Any], pd.DataFrame], str]]:
    """Each tabular result and its table (imported here: machine learning loads on demand)."""
    from quantilo.changes.compare import Changes
    from quantilo.changes.rows import RowChanges
    from quantilo.compare.analyze import Comparison
    from quantilo.correlations.matrix import Matrix
    from quantilo.correlations.service import Correlations
    from quantilo.correlations.structure import Factors, Structure
    from quantilo.experiments.analyze import ExperimentResult
    from quantilo.experiments.bandit import BanditSimulation, ExperimentBandit
    from quantilo.experiments.plan import SampleSize
    from quantilo.kpi.compute import KpiResult
    from quantilo.ml.engine import Catalogue, Targets
    from quantilo.ml.error_analysis import ErrorAnalysis, ErrorComparison, Fairness
    from quantilo.ml.explain import Explanation
    from quantilo.ml.train import PreparationPreview
    from quantilo.profiling.overview import Overview
    from quantilo.rows.view import RowsPage
    from quantilo.segments.engine import Segmentation, SegmentPlan, UnusualRows
    from quantilo.segments.facade import (
        FittedSegments,
        FoundUnusualRows,
        profile_table,
        unusual_table,
    )
    from quantilo.synthetic.plan import Plan
    from quantilo.timeseries.forecasting import SeriesForecast
    from quantilo.variable.fitting import Fit
    from quantilo.variable.outliers import Outliers
    from quantilo.variable.tails import Tail

    return [
        (RowsPage, _page, "the rows"),
        (Correlations, lambda r: _matrix(r.matrix), "the association matrix"),
        (Matrix, _matrix, "the association matrix"),
        (Structure, _structure, "the loadings of each column on the components"),
        (Factors, _factors, "the loadings after rotation, communalities and KMO"),
        (Overview, _overview, "per-column statistics"),
        (
            Fit,
            lambda r: _rows(r.laws).drop(columns=["density", "quantiles"], errors="ignore"),
            "laws",
        ),
        (Outliers, lambda r: _rows(r.rules), "the rule of each method"),
        (
            Tail,
            lambda r: _rows(r.return_levels),
            "the return levels (1 in 100, 1,000, 10,000 rows)",
        ),
        (Comparison, lambda r: _rows(r.tests), "the tests"),
        (Targets, lambda r: _rows(r.candidates), "every column as a target"),
        (
            Catalogue,
            lambda r: _rows(r.models).loc[:, ["key", "label", "family", "installed"]],
            "models",
        ),
        (
            Explanation,
            lambda r: _rows(r.permutation.items) if r.permutation else pd.DataFrame(),
            "permutation importance",
        ),
        (
            PreparationPreview,
            lambda r: pd.DataFrame(r.rows, columns=r.columns, index=r.row_numbers),
            "the prepared training rows",
        ),
        (ErrorAnalysis, lambda r: r.to_pandas(), "the subgroups where the model is wrong"),
        (
            ErrorComparison,
            lambda r: pd.concat(
                [_rows(r.better).assign(change="better"), _rows(r.worse).assign(change="worse")],
                ignore_index=True,
            ),
            "the subgroups where the second run changes",
        ),
        (
            Fairness,
            lambda r: pd.DataFrame(
                [{"group": g.group, "rows": g.rows, **g.values} for g in r.groups]
            ),
            "metrics per group",
        ),
        (SeriesForecast, lambda r: _rows(r.results), "every model's scores"),
        (Plan, lambda r: _rows(r.columns), "the plan of each column"),
        (KpiResult, _kpi, "values per measure, split and period"),
        (Changes, _shifts, "the shift of each column"),
        (RowChanges, lambda r: _rows(r.examples), "examples of modified values"),
        (ExperimentResult, _experiment, "each metric and variant against the control"),
        (SampleSize, lambda r: _rows(r.curve), "units per group and the detectable lift"),
        (ExperimentBandit, lambda r: _rows(r.arms), "P(best) and expected loss per variant"),
        (BanditSimulation, _bandit_simulation, "the cumulative regret of each policy"),
        (FittedSegments, lambda r: profile_table(r.result), "the profile of each segment"),
        (Segmentation, profile_table, "the profile of each segment"),
        (FoundUnusualRows, lambda r: unusual_table(r.result), "the most unusual rows"),
        (UnusualRows, unusual_table, "the most unusual rows"),
        (
            SegmentPlan,
            lambda r: _rows(r.preparation.decisions).loc[:, ["column", "type", "use", "why"]],
            "the columns used and why",
        ),
    ]


def to_pandas(result: Any) -> pd.DataFrame:
    """The table of a tabular result: rows, association matrix, per-column statistics, laws,
    tests, targets, models, permutation importance, forecasts, a KPI's values, an experiment's
    comparisons, a sample size's curve, the types."""
    if isinstance(result, list) and all(isinstance(item, ColumnView) for item in result):
        return _rows(result).drop(columns=["summary"], errors="ignore")
    tables = _tables()
    for kind, table, _what in tables:
        if isinstance(result, kind):
            return table(result)
    known = ", ".join(f"{kind.__name__} ({what})" for kind, _table, what in tables)
    raise TypeError(f"No table for {type(result).__name__}: to_pandas() knows {known}.")
