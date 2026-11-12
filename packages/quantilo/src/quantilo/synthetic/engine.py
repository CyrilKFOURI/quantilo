"""Synthetic data from a catalog's dataset: the automatic plan, then generation and its report.
Synchronous: the app adds jobs, saved runs and new datasets around it."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import duckdb
import pandas as pd
from pydantic import BaseModel, Field

from quantilo.catalog import Catalog, ColumnView
from quantilo.errors import ExplorerError
from quantilo.filters.scope import scoped_source, where_for
from quantilo.ml.data import Field as TypedField
from quantilo.ml.data import _expression
from quantilo.synthetic import deep, encode, plan
from quantilo.synthetic import sequences as seq
from quantilo.synthetic.deep.training import Training
from quantilo.synthetic.generate import Spec, group_like, synthesize
from quantilo.synthetic.methods import Cancelled, Options, SequenceSettings
from quantilo.synthetic.plan import Method, Plan, PlanColumn, Treatment
from quantilo.synthetic.report import LeftOut, Report, evaluate

HOLDOUT_SHARE = 5  # 1 row in 5 is held out
MAX_HOLDOUT = 20_000
SEQUENCE_ROWS = 200_000  # TimeGAN reads whole histories, in file order


class SyntheticError(ExplorerError):
    def __init__(self, message: str, code: str = "synthetic", status_code: int = 422) -> None:
        super().__init__(message, code=code, status_code=status_code)


class Bound(BaseModel):
    minimum: float | None = None
    maximum: float | None = None


class SequenceRequest(BaseModel):
    """TimeGAN: every field not given takes the plan's choice."""

    time: str | None = None
    entity: str | None = None
    no_entity: bool = False  # one long series even if an entity was found
    length: int | None = Field(default=None, ge=seq.MIN_LENGTH, le=seq.MAX_LENGTH)
    sequences: int | None = Field(default=None, ge=1, le=seq.MAX_SEQUENCES)
    steps: int | None = Field(default=None, ge=100, le=20_000)


class RunRequest(BaseModel):
    """Everything not given takes the plan's automatic choice."""

    method: Method = "cart"
    rows: int | None = Field(default=None, ge=plan.MIN_GENERATED, le=plan.MAX_GENERATED)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    smooth: bool = True  # CART: keeps synthetic numbers off the real values (fewer copies)
    drop_copies: bool = True  # replace synthetic rows identical to a real one
    treatments: dict[str, Treatment] = Field(default_factory=dict)
    bounds: dict[str, Bound] = Field(default_factory=dict)
    epochs: int | None = Field(default=None, ge=10, le=1_000)  # deep methods: None = automatic
    device: Literal["auto", "cpu", "gpu"] = "auto"  # deep methods; "auto" follows the setting
    sequence: SequenceRequest | None = None  # TimeGAN
    # deep methods, Advanced > More (066): optimizer, schedule, clipping, EMA; None = the paper's
    training: Training | None = None


@dataclass
class RunColumn:
    name: str
    type: str
    treatment: Treatment


@dataclass
class Generation:
    """Synthetic rows with the real rows they were learned from and judged against."""

    method: Method
    rows: int
    seconds: float
    columns: list[RunColumn]
    train: pd.DataFrame  # real rows the method learned from (typed)
    holdout: pd.DataFrame  # real rows kept aside, for the report
    data: pd.DataFrame  # the synthetic rows
    report: Report


Prepared = tuple[list[Spec], list[LeftOut], int, seq.SequencePlan | None]


def export_frame(data: pd.DataFrame, columns: list[RunColumn]) -> pd.DataFrame:
    """Typed values as a dataset stores them: booleans and whole numbers as such."""
    out = data.copy()
    for column in columns:
        if column.treatment != "model":
            continue
        if column.type == "boolean":
            out[column.name] = (
                out[column.name].map({"True": True, "False": False}.get).astype("boolean")
            )
        elif column.type == "numeric_discrete":
            out[column.name] = out[column.name].round().astype("Int64")
    return out


class SyntheticEngine:
    def __init__(
        self,
        columns: Catalog,
        compute: Callable[[], tuple[str | None, bool]] = lambda: (None, False),
    ) -> None:
        """`compute()` gives the ready card PyTorch sees (or None) and whether the GPU may be
        used."""
        self.columns = columns
        self.compute = compute

    # --- Plan ----------------------------------------------------------------------------------

    def _views(self, dataset_id: str) -> tuple[str, list[ColumnView]]:
        dataset = self.columns.ready(dataset_id)
        return dataset.storage_path, self.columns.columns(dataset_id)

    def plan(self, dataset_id: str, request: "SequenceRequest | None" = None) -> Plan:
        """`request`: TimeGAN's choices of time, entity and length, if any."""
        storage, views = self._views(dataset_id)
        source = scoped_source(storage, where_for(views, []))
        fields = [TypedField(v.name, v.type, v.parser, v.role, v.order) for v in views]
        parts = ", ".join(
            f"count({_expression(f)}), count(DISTINCT {_expression(f)})" for f in fields
        )
        connection = duckdb.connect()
        try:
            row = connection.execute(f"SELECT count(*), {parts} FROM {source}").fetchone() or ()
        finally:
            connection.close()
        rows = int(row[0]) if row else 0
        columns = [
            PlanColumn(
                v.name, v.type, v.role, rows, int(row[1 + 2 * i]), int(row[2 + 2 * i]), v.excluded
            )
            for i, v in enumerate(views)
        ]
        gpu, allowed = self.compute()
        sequence = self._sequence_plan(storage, views, columns, request)
        return plan.make_plan(columns, rows, deep.installed(), gpu if allowed else None, sequence)

    def _sequence_plan(
        self,
        storage: str,
        views: list[ColumnView],
        columns: list[PlanColumn],
        request: SequenceRequest | None,
    ) -> seq.SequencePlan:
        """Time, entity and length for TimeGAN (rules R-SEQ), from the dates and the columns
        that may name a history."""
        treated = {
            c.name: c for c in plan.make_plan(columns, columns[0].rows if columns else 0).columns
        }
        times = [
            c.name for c in columns if c.type == "datetime" and treated[c.name].treatment == "model"
        ]
        entities = [
            c.name
            for c in columns
            if c.type in seq.ENTITY_TYPES
            and not c.excluded
            and 2 <= c.distinct <= c.rows // seq.MIN_LENGTH
        ]
        if request and request.time:
            times = [request.time, *[t for t in times if t != request.time]]
        if request and request.entity:
            entities = [request.entity]
        if request and request.no_entity:
            entities = []
        if not times:
            return seq.plan_sequences(pd.DataFrame(), [], [])
        wanted = list(dict.fromkeys([*times[:1], *entities]))
        by_name = {v.name: v for v in views}
        fields = [
            TypedField(n, by_name[n].type, by_name[n].parser, by_name[n].role, by_name[n].order)
            for n in wanted
        ]
        source = scoped_source(storage, where_for(views, []), numbered=True)
        selected = ", ".join(f"{_expression(f)} AS c{i}" for i, f in enumerate(fields))
        connection = duckdb.connect()
        try:
            frame = connection.execute(
                f"SELECT {selected} FROM {source} ORDER BY file_row_number LIMIT {SEQUENCE_ROWS}"
            ).df()
        finally:
            connection.close()
        frame.columns = wanted
        found = seq.plan_sequences(frame, times[:1], entities, request.length if request else None)
        if request and request.sequences and not found.error:
            found.sequences = request.sequences
        return found

    def _specs(
        self, dataset_id: str, request: RunRequest
    ) -> tuple[Plan, list[Spec], list[LeftOut]]:
        found = self.plan(dataset_id, request.sequence if request.method == "timegan" else None)
        if found.error:
            raise SyntheticError(found.error, "cannot_generate")
        names = {c.name for c in found.columns}
        unknown = [n for n in [*request.treatments, *request.bounds] if n not in names]
        if unknown:
            raise SyntheticError(f"Unknown column(s): {', '.join(unknown)}.", "unknown_column")
        _, views = self._views(dataset_id)
        orders = {v.name: v.order for v in views}
        specs, left_out = [], []
        if request.method == "timegan" and found.sequence and found.sequence.error:
            raise SyntheticError(f"TimeGAN: {found.sequence.error}", "method_unavailable")
        for column in found.columns:
            treatment = request.treatments.get(column.name, column.treatment)
            if (
                found.sequence
                and request.method == "timegan"
                and column.name == found.sequence.entity
            ):
                treatment = "unique"  # new entities, one per synthetic history
            entity = (
                request.method == "timegan"
                and found.sequence
                and column.name == found.sequence.entity
            )
            if treatment != column.treatment and treatment not in column.options and not entity:
                raise SyntheticError(
                    f"{column.name!r} cannot be set to {treatment!r} ({column.reason})",
                    "bad_treatment",
                )
            if treatment == "leave":
                left_out.append(LeftOut(column.name, column.reason))
                continue
            bound = request.bounds.get(column.name, Bound())
            specs.append(
                Spec(
                    column.name,
                    column.type,
                    treatment,
                    orders.get(column.name),
                    column.grouped,
                    bound.minimum,
                    bound.maximum,
                )
            )
        if not any(s.treatment == "model" for s in specs):
            raise SyntheticError("Model at least one column.", "nothing_to_model")
        return found, specs, left_out

    def _prepare(self, dataset_id: str, request: "RunRequest") -> Prepared:
        """Columns to generate, columns left out, rows to generate, TimeGAN histories."""
        found, specs, left_out = self._specs(dataset_id, request)
        option = next(m for m in found.methods if m.key == request.method)
        if not option.available:
            raise SyntheticError(f"{option.label}: {option.unavailable}", "method_unavailable")
        rows = request.rows or found.default_rows
        if request.method == "timegan" and found.sequence:
            rows = found.sequence.sequences * found.sequence.length
        sequence = found.sequence if request.method == "timegan" else None
        return specs, left_out, rows, sequence

    def _load(
        self,
        dataset_id: str,
        specs: list[Spec],
        seed: int,
        histories: seq.SequencePlan | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
        """Training and held-out real rows (typed), and every real value of the columns that are
        regenerated or faked (so none is reused)."""
        storage, views = self._views(dataset_id)
        by_name = {v.name: v for v in views}
        fields = [
            TypedField(
                s.name,
                by_name[s.name].type,
                by_name[s.name].parser,
                by_name[s.name].role,
                by_name[s.name].order,
            )
            for s in specs
        ]
        source = scoped_source(storage, where_for(views, []), numbered=True)
        selected = ", ".join(f"{_expression(f)} AS c{i}" for i, f in enumerate(fields))
        relation = f"(SELECT {selected}, hash(file_row_number + {int(seed)}) AS __h FROM {source})"
        connection = duckdb.connect()
        try:
            if histories is not None:
                # whole histories: held-out rows are entire entities (or the end of the series)
                every = connection.execute(
                    f"SELECT * EXCLUDE (__h) FROM {relation} LIMIT {SEQUENCE_ROWS}"
                ).df()
                every.columns = [s.name for s in specs]
                assert histories.time is not None
                train, holdout = seq.split_histories(
                    self._typed(every, specs), histories.time, histories.entity, seed
                )
                train.columns = holdout.columns = [f"c{i}" for i in range(len(specs))]
            else:
                train = connection.execute(
                    f"SELECT * EXCLUDE (__h) FROM {relation} WHERE __h % {HOLDOUT_SHARE} <> 0 "
                    f"ORDER BY __h LIMIT {plan.MAX_TRAIN_ROWS}"
                ).df()
                holdout = connection.execute(
                    f"SELECT * EXCLUDE (__h) FROM {relation} WHERE __h % {HOLDOUT_SHARE} = 0 "
                    f"ORDER BY __h LIMIT {MAX_HOLDOUT}"
                ).df()
            existing = {}
            for i, spec in enumerate(specs):
                if spec.treatment in ("unique", "fake"):
                    existing[spec.name] = (
                        connection.execute(
                            f"SELECT DISTINCT {_expression(fields[i])} FROM {source}"
                        )
                        .df()
                        .iloc[:, 0]
                    )
        finally:
            connection.close()
        names = [s.name for s in specs]
        train.columns, holdout.columns = names, names
        return self._typed(train, specs), self._typed(holdout, specs), existing

    @staticmethod
    def _typed(frame: pd.DataFrame, specs: list[Spec]) -> pd.DataFrame:
        for spec in specs:
            if spec.treatment == "model" and spec.type in encode.NUMERIC:
                frame[spec.name] = cast(
                    pd.Series, pd.to_numeric(frame[spec.name], errors="coerce")
                ).astype(float)
        return frame

    def _generate(
        self,
        dataset_id: str,
        request: "RunRequest",
        prepared: Prepared,
        stopped: Callable[[], bool] = lambda: False,
        progress: Callable[[float, str], None] = lambda _share, _message: None,
        stage: Callable[[str], None] = lambda _stage: None,
    ) -> Generation:
        """Learn from the real rows, generate, and evaluate (Diagnostic, Quality, Detection,
        Privacy; sequences for TimeGAN). Raises `Cancelled` when `stopped()` turns true."""
        started = time.perf_counter()
        specs, left_out, rows, histories = prepared
        train, holdout, existing = self._load(dataset_id, specs, request.seed, histories)
        if len(train) < plan.MIN_ROWS // 2:
            raise SyntheticError("Too few rows to learn from.")
        if stopped():
            raise Cancelled
        stage("generating")
        _, allowed = self.compute()

        options = Options(
            rows=rows,
            seed=request.seed,
            smooth=request.smooth,
            drop_copies=request.drop_copies,
            epochs=request.epochs,
            training=request.training,
            device=request.device,
            allow_gpu=allowed,
            progress=progress,
            sequence=SequenceSettings(
                histories.time,
                histories.entity,
                histories.length,
                histories.sequences,
                request.sequence.steps if request.sequence else None,
                histories.step_seconds,
            )
            if histories
            else None,
        )
        synthesis = synthesize(train, specs, request.method, options, existing, stopped)
        if stopped():
            raise Cancelled
        stage("evaluating")
        train, holdout = synthesis.train, group_like(holdout, synthesis.learned)
        entity = histories.entity if histories else None
        unique = [s.name for s in specs if s.treatment == "unique" and s.name != entity]
        report = evaluate(
            train,
            holdout,
            synthesis.frame,
            synthesis.learned,
            synthesis.fitted,
            unique,
            synthesis.bounds,
            left_out,
            request.seed,
            stopped,
        )
        if histories is not None and histories.time is not None:
            from quantilo.synthetic.deep.sequence_report import evaluate_sequences

            report.sequences = evaluate_sequences(
                train,
                holdout,
                synthesis.frame,
                synthesis.learned,
                histories.time,
                histories.entity,
                histories.length,
                histories.step,
                request.seed,
                allowed,
            )
        if stopped():
            raise Cancelled
        return Generation(
            request.method,
            rows,
            round(time.perf_counter() - started, 2),
            [RunColumn(s.name, s.type, s.treatment) for s in specs],
            train,
            holdout,
            synthesis.frame,
            report,
        )
