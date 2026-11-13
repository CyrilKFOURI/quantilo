# pyright: reportAttributeAccessIssue=false
# (fitted scikit-learn objects: the stubs cannot follow them)
"""Use a trained model on new data (037): the model card (what the model expects: columns and
their types, the categories it knows, the training ranges, its validation scores, the versions it
was trained with), the check of a file before predicting (missing columns, types, unknown
categories, values out of the training range, unknown vocabulary of a free text: Ready, Check
or Blocked), predictions in blocks with
a reliability flag per row and a conformal interval for a number, the scores when the file holds
the answer, the explanation of one predicted row, and the export of a model as a zip.

A `joblib` file is a pickle: loading one runs code. The app only loads the pipelines it wrote in
its own workspace; the library's `load_model` (`ml/saved.py`) reads the card first and refuses a
zip without one."""

import hashlib
import importlib.metadata
import io
import json
import math
import platform
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, cast

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
from pydantic import Field as PydanticField

from quantilo.compare.analyze import label_sql, number_label_sql
from quantilo.filters.model import Condition
from quantilo.ml import classes as binning
from quantilo.ml import conformal as intervals
from quantilo.ml import counterfactual as whatif
from quantilo.ml import data as ml_data
from quantilo.ml import evaluate
from quantilo.ml import explain as explainer
from quantilo.ml import models as catalogue
from quantilo.ml import text as textual
from quantilo.ml import train as trainer
from quantilo.ml.data import NUMERIC, Field
from quantilo.ml.engine import period_of
from quantilo.ml.target import Task
from quantilo.typing.parsers import present_sql, typed_sql
from quantilo.typing.types import Parser
from quantilo.variable.source import SEED, clean
from quantilo.version import __version__

CARD_FORMAT = "quantilo/model-card"
LEGACY_CARD_FORMAT = "cyfouri-data-explorer/model-card"  # cards exported before the rename (043)
CARD_VERSION = 1
BLOCK_ROWS = 100_000  # rows predicted at a time
INTERVAL = 0.9  # coverage of the conformal interval of a number
MIN_ERRORS = 8  # validation errors needed for an interval
NOTABLE = 0.01  # unknown categories or values out of range on 1 % of rows or more: Check
UNREADABLE = 0.5  # half the present values of a column unreadable in its type: Blocked
IMPORTANT = 0.05  # share of the model's importance that makes a column important
TOP_IMPORTANT = 3  # the 3 most important columns always are
METRIC_ROWS = 1_000_000  # rows with an answer scored at most (the first ones of the file)
QUANTILES = 101  # validation predictions kept as percentiles (no rows in the card)
BINS = 30
SHOWN_UNKNOWN = 5
EXPLAIN_ROWS = 2_000
TEXT_ROWS = 20_000  # texts of the new data whose vocabulary is checked
TEXT_GAP = 0.15  # unknown terms this far above the training share: Check (052)
WATERFALL = 10
SECURITY = (
    "model.joblib is a Python pickle: loading it runs code. Only load a model you exported "
    "yourself or got from someone you trust. The Quantilo app never imports model files."
)
OUTPUTS = {"row", "predicted", "low", "high", "less_reliable", "why", "actual"}

Level = Literal["ready", "check", "blocked"]


class Readable(Protocol):
    """A column of the new data as the catalog describes it (`ColumnView`)."""

    name: str
    type: Any
    parser: Parser


# --- The model card ------------------------------------------------------------------------------


class TextVocabulary(BaseModel):
    """How the model reads a free-text column (052): the cut, the terms it knows (words or
    pieces of words, no row), and how much of its training texts the vocabulary missed."""

    analyzer: Literal["words", "chars"]
    terms: list[str]
    components: int
    unknown_share: float  # share of the training terms outside the vocabulary (the reference)
    unmatched_share: float = 0.0  # training texts with no known term


class ExpectedColumn(BaseModel):
    """A column the model reads: how it was read in training, how much it matters, the
    categories it knows and the range it was trained on."""

    name: str
    type: str
    parser: Parser = PydanticField(default_factory=Parser)
    role: str | None = None
    order: list[str] | None = None
    importance: float | None = None  # share of the model's importance (None: not known)
    important: bool = True  # an unknown or out-of-range value here makes a row less reliable
    categories: list[str] | None = None  # categories the model knows
    unknown_rule: str | None = None  # what an unknown category becomes
    minimum: float | None = None  # numbers; dates as seconds since 1970
    maximum: float | None = None
    range: str | None = None  # the training range, as read
    text: TextVocabulary | None = None  # a free text: its vocabulary (052)


class Interval(BaseModel):
    """A number's prediction interval from the validation errors (split conformal, or CQR: the
    quantile band kept on the pipeline, widened by `half_width`) (037, 057)."""

    level: float
    half_width: float
    errors: int
    method: Literal["split", "cqr"] = "split"
    coverage: float | None = None  # share of checked held-out rows inside their interval
    check_rows: int = 0


class ValidationSummary(BaseModel):
    method: str
    primary: str
    score: float | None
    scores: dict[str, float | None] = PydanticField(default_factory=dict)
    rows: int = 0
    predicted_quantiles: list[float] = PydanticField(default_factory=list)  # a number
    predicted_shares: dict[str, float] = PydanticField(default_factory=dict)  # classes


class ModelCard(BaseModel):
    """Everything needed to use the model elsewhere, without a single row of data."""

    format: Literal["quantilo/model-card"] = CARD_FORMAT
    version: int = CARD_VERSION

    @field_validator("format", mode="before")
    @classmethod
    def _renamed(cls, value: object) -> object:
        return CARD_FORMAT if value == LEGACY_CARD_FORMAT else value

    model: str
    model_label: str
    task: Task
    target: ExpectedColumn
    classes: binning.Classes | None = None  # a number predicted as classes (018)
    labels: list[str] = PydanticField(default_factory=list)  # the classes, as the model orders them
    positive: str | None = None
    threshold: float | None = None
    columns: list[ExpectedColumn]
    validation: ValidationSummary
    interval: Interval | None = None
    versions: dict[str, str] = PydanticField(default_factory=dict)
    trained_at: str | None = None
    created_at: str
    source: str = ""  # the dataset it was trained on (its name)
    pipeline_sha256: str | None = None  # of model.joblib, in an exported zip
    security: str = SECURITY


def _field(column: ExpectedColumn) -> Field:
    return Field(column.name, cast(Any, column.type), column.parser, column.role, column.order)


def versions(model: str) -> dict[str, str]:
    """Python and the libraries the model needs, as installed now."""
    found = {"python": platform.python_version(), "quantilo": __version__}
    names = ["scikit-learn", "numpy", "pandas", "scipy", "joblib"]
    spec = catalogue.CATALOGUE.get(model)
    if spec is not None and spec.package:
        names.append(spec.package)
    for name in names:
        for distribution in (name, f"{name}-cpu"):
            try:
                found[name] = importlib.metadata.version(distribution)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    return found


def version_warnings(card: ModelCard) -> list[str]:
    """One sentence per library whose installed version differs from the training one."""
    now = versions(card.model)
    found = []
    for name, trained in card.versions.items():
        installed = now.get(name)
        if name == "python" or installed is None or installed == trained:
            continue
        found.append(f"{name} {trained} trained this model; {installed} is installed.")
    return found


def _unknown_rule(step: Any, categories: list[str]) -> str | None:
    """What the preparation does with a category it never saw; None: the next step decides."""
    name = type(step).__name__
    if name == "RareGrouper":  # 063: rare categories grouped before codes or a target encoding
        return "grouped with the rare categories (Other)" if step.other_ else None
    if name == "FrequencyEncoder":
        return "counted as never seen (frequency 0)"
    if name == "OneHotEncoder":
        if (
            getattr(step, "infrequent_categories_", None)
            and step.infrequent_categories_[0] is not None
        ):
            return "grouped with the rare categories (Other)"
        if getattr(step, "drop_idx_", None) is not None and categories:
            return f"read as {categories[0]} (the reference category)"
        return "left with no category (every indicator at 0)"
    if name == "OrdinalEncoder":
        return "coded below every known level"
    if name == "TargetEncoder":
        return "replaced by the average of the target"
    return "handled as the preparation handles them"


def _encoders(pipeline: Any) -> dict[str, tuple[list[str], str]]:
    """Per column: the categories its encoder learned, and the rule for an unknown one."""
    prepare = pipeline.named_steps.get("prepare") if hasattr(pipeline, "named_steps") else None
    found: dict[str, tuple[list[str], str]] = {}
    for name, step, columns in getattr(prepare, "transformers_", []):
        if name == "remainder" or step == "drop" or not hasattr(step, "steps"):
            continue
        categories: list[str] | None = None
        for _, inner in step.steps:
            learned = getattr(inner, "categories_", None)
            if learned is None or not len(learned):
                continue
            if categories is None:  # the categories of the first step that learnt some
                categories = [
                    str(c) for c in learned[0] if not (isinstance(c, float) and math.isnan(c))
                ]
            rule = _unknown_rule(inner, categories)
            if rule is not None:
                found[str(columns[0])] = (categories, rule)
                break
    return found


def _importance(pipeline: Any, trained: Any) -> dict[str, float]:
    """Share of the model's importance per dataset column: the model's own importances, else the
    size of its coefficients; empty when the model gives neither."""
    try:
        owners = explainer.origins(pipeline)
        names = [str(n) for n in pipeline[:-1].get_feature_names_out()]
    except (AttributeError, ValueError, KeyError):
        return {}
    owner = dict(zip(names, owners, strict=False))
    weights: dict[str, float] = {}
    if trained.importances:
        for item in trained.importances:
            column = str(owner.get(item.feature, item.feature))
            weights[column] = weights.get(column, 0.0) + float(item.importance)
    elif trained.coefficients:
        for item in trained.coefficients:
            if item.feature not in owner:
                continue  # the intercept
            column = owner[item.feature]
            weights[column] = weights.get(column, 0.0) + abs(float(item.estimate))
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()} if total > 0 else {}


def _seconds(values: pd.Series) -> np.ndarray:
    stamps = pd.to_datetime(values, errors="coerce")
    seconds = (stamps - pd.Timestamp("1970-01-01")).dt.total_seconds()
    return seconds.to_numpy(dtype=float)


def _range_label(kind: str, low: float, high: float) -> str:
    if kind == "datetime":
        start = pd.Timestamp("1970-01-01") + pd.Timedelta(seconds=low)
        end = pd.Timestamp("1970-01-01") + pd.Timedelta(seconds=high)
        return f"{start:%Y-%m-%d} – {end:%Y-%m-%d}"  # noqa: RUF001
    return f"{low:,.6g} – {high:,.6g}"  # noqa: RUF001


def build_card(
    pipeline: Any,
    trained: Any,
    fields: list[Field],
    frame: pd.DataFrame,
    source: str = "",
    trained_at: str | None = None,
) -> ModelCard:
    """The card of a trained model. `trained`: a saved run or a `TrainedModel` (task, model,
    target, classes, classification, regression, primary, score, scores, validation, importances,
    coefficients, test_rows); `frame`: the rows it was trained and validated on (ranges and, when
    the encoder does not say, categories)."""
    by_name = {f.name: f for f in fields}
    inputs = [str(c) for c in pipeline.feature_names_in_]
    encoders = _encoders(pipeline)
    texts = textual.steps_of(pipeline)
    shares = _importance(pipeline, trained)
    ranked = sorted(shares, key=lambda c: -shares[c])[:TOP_IMPORTANT]
    columns = []
    for name in inputs:
        found = by_name[name]
        column = ExpectedColumn(
            name=name,
            type=found.type,
            parser=found.parser,
            role=found.role,
            order=found.order,
            importance=clean(shares[name]) if name in shares else None,
            important=not shares or shares.get(name, 0.0) >= IMPORTANT or name in ranked,
        )
        values = cast(pd.Series, frame[name]) if name in frame.columns else pd.Series(dtype=object)
        if found.type in NUMERIC or found.type == "datetime":
            numbers = _seconds(values) if found.type == "datetime" else values.to_numpy(dtype=float)
            numbers = numbers[np.isfinite(numbers)]
            if len(numbers):
                column.minimum, column.maximum = float(numbers.min()), float(numbers.max())
                column.range = _range_label(found.type, column.minimum, column.maximum)
        elif found.type == "text":
            step = texts.get(name)
            if step is not None:
                features = step.features
                column.text = TextVocabulary(
                    analyzer=features.analyzer,
                    terms=features.vocabulary,
                    components=int(features.width_),
                    unknown_share=float(features.unknown_share_),
                    unmatched_share=float(features.unmatched_share_),
                )
        else:
            known, rule = encoders.get(name, ([], ""))
            column.categories = known or sorted({str(v) for v in values.dropna()})
            column.unknown_rule = rule or None
        columns.append(column)
    target = by_name[trained.target]
    classification = trained.classification
    summary = ValidationSummary(
        method=trained.validation.method,
        primary=trained.primary,
        score=trained.score,
        scores=dict(trained.scores),
        rows=int(trained.test_rows),
    )
    interval = None
    if trained.regression is not None:
        predicted = np.asarray(trained.regression.predicted, dtype=float)
        if len(predicted):
            summary.predicted_quantiles = [
                float(v) for v in np.quantile(predicted, np.linspace(0, 1, QUANTILES))
            ]
        found = getattr(trained, "interval", None)
        if found is not None:
            interval = Interval(
                level=found.level,
                half_width=found.half_width,
                errors=found.rows,
                method=found.method if hasattr(pipeline, intervals.BANDS_ATTRIBUTE) else "split",
                coverage=found.coverage,
                check_rows=found.check_rows,
            )
        else:  # a run saved before 057: from the sample of validation predictions
            interval = conformal(np.asarray(trained.regression.actual, dtype=float), predicted)
    elif classification is not None:
        totals = np.asarray(classification.confusion, dtype=float).sum(axis=0)
        if totals.sum() > 0:
            summary.predicted_shares = {
                label: float(totals[i] / totals.sum())
                for i, label in enumerate(classification.labels)
            }
    labels = [str(c) for c in getattr(pipeline, "classes_", [])]
    return ModelCard(
        model=trained.model,
        model_label=spec.label
        if (spec := catalogue.CATALOGUE.get(trained.model))
        else trained.model,
        task=trained.task,
        target=ExpectedColumn(
            name=target.name,
            type=target.type,
            parser=target.parser,
            role=target.role,
            order=target.order,
        ),
        classes=trained.classes,
        labels=labels,
        positive=classification.positive if classification else None,
        threshold=classification.threshold if classification else None,
        columns=columns,
        validation=summary,
        interval=interval,
        versions=versions(trained.model),
        trained_at=trained_at,
        created_at=datetime.now(UTC).isoformat(),
        source=source,
    )


def card_from(
    engine: Any,
    dataset_id: str,
    pipeline: Any,
    trained: Any,
    conditions: list[Condition],
    trained_at: str | None = None,
) -> ModelCard:
    """The card of a model trained on a catalog's dataset (`engine`: an `MlEngine`), from the rows
    it was trained and validated on (same filters, same sample)."""
    fields, target, source = engine._request(dataset_id, trained.target, None, conditions)
    inputs = {str(c) for c in pipeline.feature_names_in_}
    missing = sorted(inputs - {f.name for f in fields})
    if missing:
        raise ValueError(
            f"The data changed since this run (no {', '.join(missing)}): train it again."
        )
    chosen = [f for f in fields if f.name in inputs or f.name == target.name]
    connection = duckdb.connect()
    try:
        frame = ml_data.load(
            connection,
            source,
            chosen,
            target.name,
            trained.validation.max_rows,
            period_of(trained.validation, fields),
        )
    finally:
        connection.close()
    name = engine.columns.ready(dataset_id).name
    return build_card(pipeline, trained, fields, frame.data, name, trained_at)


def conformal(
    actual: np.ndarray, predicted: np.ndarray, level: float = INTERVAL
) -> Interval | None:
    """Split conformal: the validation errors' quantile at ⌈(n + 1) level⌉ / n, taken as the
    half-width of every interval (as 028 does for forecasts). None with fewer than 8 errors."""
    errors = np.abs(actual - predicted)
    errors = errors[np.isfinite(errors)]
    n = len(errors)
    if n < MIN_ERRORS:
        return None
    quantile = min(1.0, math.ceil((n + 1) * level) / n)
    return Interval(
        level=level, half_width=float(np.quantile(errors, quantile, method="higher")), errors=n
    )


# --- Reading the new data ------------------------------------------------------------------------


def _family(analysis_type: str) -> str:
    if analysis_type in NUMERIC:
        return "number"
    if analysis_type == "datetime":
        return "date"
    return "category"


def reader(column: ExpectedColumn, found: Readable | None) -> tuple[str, str, bool]:
    """SQL reading the new data's column as the model expects it, the SQL that is non-NULL when a
    value is there, and whether it is read by conversion from another type. A column of the same
    kind is read with the new file's own parser (a comma decimal there is still a number)."""
    if found is None:
        kind = {"number": "DOUBLE", "date": "TIMESTAMP"}.get(_family(column.type), "VARCHAR")
        return f"CAST(NULL AS {kind})", "NULL", False
    own = Field(found.name, found.type, found.parser)
    same = _family(found.type) == _family(column.type)
    parser = found.parser if same else column.parser
    typed = typed_sql(found.name, parser)
    if column.type in NUMERIC:
        expression = f"TRY_CAST({typed} AS DOUBLE)"
    elif column.type == "datetime":
        expression = f"TRY_CAST({typed} AS TIMESTAMP)"
    elif column.type == "boolean" or found.type == "boolean":
        expression = label_sql(Field(found.name, "boolean", parser))  # type: ignore[arg-type]
    elif found.type in NUMERIC:
        expression = number_label_sql(own)  # type: ignore[arg-type]
    else:
        expression = f"CAST({typed} AS VARCHAR)"
    return expression, present_sql(found.name, found.parser), not same


def _views(views: list[Any]) -> dict[str, Readable]:
    return {v.name: v for v in views}


def read_rows(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    card: ModelCard,
    views: list[Any],
    keep: list[str] | None = None,
    where: str = "",
    limit: str = "",
    order: str = "ORDER BY file_row_number",
) -> pd.DataFrame:
    """The model's inputs read from the new data (numbers as floats, as in training), the kept
    columns as the new data types them, the answer when the file has it, and `__row`."""
    found = _views(views)
    parts = []
    for i, column in enumerate(card.columns):
        parts.append(f"{reader(column, found.get(column.name))[0]} AS i{i}")
    kept = [name for name in keep or [] if name in found and name not in OUTPUTS]
    for i, name in enumerate(kept):
        parts.append(f"{typed_sql(name, found[name].parser)} AS k{i}")
    has_target = card.target.name in found
    if has_target:
        parts.append(f"{reader(card.target, found[card.target.name])[0]} AS t0")
    parts.append("file_row_number AS __row")
    data = connection.execute(
        f"SELECT {', '.join(parts)} FROM {source} {where} {order} {limit}"
    ).df()
    data.columns = [
        *(c.name for c in card.columns),
        *(f"keep:{name}" for name in kept),
        *(["target:"] if has_target else []),
        "__row",
    ]
    for column in card.columns:
        if column.type in NUMERIC:  # as ml_data.load reads them
            data[column.name] = pd.Series(
                pd.to_numeric(data[column.name], errors="coerce"), index=data.index, dtype=float
            )
    return data


# --- The check -----------------------------------------------------------------------------------


@dataclass
class ColumnCheck:
    column: str
    expected: str  # the type the model reads
    found: str | None  # its type in the new data; None: missing
    importance: float | None
    important: bool
    converted: bool  # read by conversion from another type
    unreadable: float  # share of the present values that could not be read in the expected type
    missing: float  # share of empty values
    unknown_rows: int  # rows with a category the model never saw
    unknown_share: float
    unknown_values: list[str]  # the most frequent of them
    unknown_rule: str | None
    out_of_range: float  # share of rows outside the training range
    range: str | None
    note: str | None  # one sentence when something deserves a look
    # a free text (052): share of its terms the vocabulary does not know, here and in training,
    # and share of its texts with no known term
    unknown_terms: float | None = None
    training_unknown_terms: float | None = None
    unmatched_texts: float | None = None


@dataclass
class ModelCheck:
    level: Level  # ready · check · blocked
    message: str  # the sentence that explains it
    rows: int
    columns: list[ColumnCheck]  # the model's columns, the most important first
    missing: list[str]  # columns the model needs that the data lacks
    ignored: list[str]  # columns of the data the model does not read
    target_present: bool  # the file holds the answer: it will be scored
    problems: list[str] = field(default_factory=list)  # one sentence each, the most serious first
    # Drift against the training data (031's comparison, the model's columns only, the most
    # important first): None when it could not be compared (037 M4).
    drift: "ModelDrift | None" = None


@dataclass
class DriftColumn:
    column: str
    psi: float | None
    drift: Literal["stable", "moderate", "strong"] | None
    long_tail: bool  # many distinct values: new ones are expected (031)
    important: bool


@dataclass
class ModelDrift:
    level: Literal["none", "check", "breaking"]  # 031's level for the whole file
    headline: str  # 031's one-line summary
    columns: list[DriftColumn]
    shifted: list[str]  # the model's columns with a strong shift (PSI > 0.25)


def with_drift(checked: ModelCheck, changes: Any) -> ModelCheck:
    """Add 031's comparison of the new data with the training data (`changes`, a Changes): a
    strong shift on a column the model reads turns *Ready* into *Check* (037 M4)."""
    order = {c.column: i for i, c in enumerate(checked.columns)}
    important = {c.column for c in checked.columns if c.important}
    shifts = sorted(
        (s for s in changes.distributions if s.column in order), key=lambda s: order[s.column]
    )
    columns = [
        DriftColumn(s.column, s.psi, s.drift, s.long_tail, s.column in important) for s in shifts
    ]
    shifted = [c.column for c in columns if c.drift == "strong" and not c.long_tail]
    checked.drift = ModelDrift(changes.level, changes.headline, columns, shifted)
    if shifted:
        names = ", ".join(shifted[:3]) + (
            f" and {len(shifted) - 3} more" if len(shifted) > 3 else ""
        )
        checked.problems.append(
            f"{names}: distributed differently from the training data (PSI > 0.25): "
            "the scores of the validation may not hold."
        )
        if checked.level == "ready":
            checked.level = "check"
            checked.message = (
                "Check: the data the model reads has shifted since training. Predictions are "
                "made, but compare them with care."
            )
    return checked


def _pct(share: float) -> str:
    return f"{share:.0%}" if share >= 0.01 or share == 0 else "< 1%"


def check(
    connection: duckdb.DuckDBPyConnection, source: str, card: ModelCard, views: list[Any]
) -> ModelCheck:
    """Is this data fit for the model? Missing columns and unreadable types block; unknown
    categories, values out of the training range and partial conversions ask for a look."""
    found = _views(views)
    readers = {c.name: reader(c, found.get(c.name)) for c in card.columns}
    parts = ["count(*)"]
    for column in card.columns:
        expression, present, _ = readers[column.name]
        parts += [f"count({present})", f"count({expression})"]
        if column.minimum is not None and column.maximum is not None:
            value = f"epoch({expression})" if column.type == "datetime" else expression
            parts.append(
                f"count(*) FILTER (WHERE {value} < {column.minimum!r} "
                f"OR {value} > {column.maximum!r})"
            )
        else:
            parts.append("0")
    row = connection.execute(f"SELECT {', '.join(parts)} FROM {source}").fetchone() or ()
    rows = int(row[0]) if row else 0
    checks: list[ColumnCheck] = []
    for i, column in enumerate(card.columns):
        present, read, outside = (int(v or 0) for v in row[1 + 3 * i : 4 + 3 * i])
        view = found.get(column.name)
        converted = readers[column.name][2]
        unknown_rows, unknown_values = 0, []
        if view is not None and column.categories is not None:
            known = set(column.categories)
            counted = connection.execute(
                f"SELECT {readers[column.name][0]} AS v, count(*) AS n FROM {source} "
                "WHERE v IS NOT NULL GROUP BY v ORDER BY n DESC, v"
            ).fetchall()
            unknown = [(str(v), int(n)) for v, n in counted if str(v) not in known]
            unknown_rows = sum(n for _, n in unknown)
            unknown_values = [v for v, _ in unknown[:SHOWN_UNKNOWN]]
        checks.append(
            ColumnCheck(
                column.name,
                column.type,
                str(view.type) if view is not None else None,
                column.importance,
                column.important,
                converted,
                (present - read) / present if present else 0.0,
                1 - read / rows if rows and view is not None else 0.0,
                unknown_rows,
                unknown_rows / rows if rows else 0.0,
                unknown_values,
                column.unknown_rule,
                outside / rows if rows else 0.0,
                column.range,
                None,
            )
        )
    for item, column in zip(checks, card.columns, strict=True):
        if column.text is not None and item.found is not None:
            texts = _sample_texts(connection, source, readers[column.name][0])
            found_gap = textual.gap(
                texts, textual.cutter(column.text.analyzer), set(column.text.terms)
            )
            item.unknown_terms = found_gap.unknown
            item.unmatched_texts = found_gap.unmatched
            item.training_unknown_terms = column.text.unknown_share
    for item in checks:
        item.note = _note(item)
    checks.sort(key=lambda c: (c.found is not None, -(c.importance or 0)))
    used = {c.name for c in card.columns}
    missing = [c.column for c in checks if c.found is None]
    ignored = [v.name for v in views if v.name not in used and v.name != card.target.name]
    level, message, problems = _verdict(checks, missing, rows)
    return ModelCheck(
        level, message, rows, checks, missing, ignored, card.target.name in found, problems
    )


def _sample_texts(connection: duckdb.DuckDBPyConnection, source: str, expression: str) -> list[str]:
    """Up to 20 000 non-empty texts of the new data, drawn with the seed."""
    found = connection.execute(
        f"SELECT v FROM (SELECT {expression} AS v, file_row_number AS r FROM {source}) "
        f"WHERE v IS NOT NULL AND trim(v) <> '' ORDER BY hash(r + {SEED}) LIMIT {TEXT_ROWS}"
    ).fetchall()
    return [str(v) for (v,) in found]


def _text_shift(item: ColumnCheck) -> bool:
    """Many more unknown terms than in training (052)."""
    return (
        item.unknown_terms is not None
        and item.unknown_terms - (item.training_unknown_terms or 0.0) >= TEXT_GAP
    )


def _note(item: ColumnCheck) -> str | None:
    if item.found is None:
        return "Missing: the model needs this column."
    sentences = []
    if item.converted:
        read = f"{_pct(1 - item.unreadable)} of its values" if item.unreadable else "every value"
        sentences.append(f"{item.found} here, read as {item.expected}: {read} could be read.")
    elif item.unreadable:
        sentences.append(
            f"{_pct(item.unreadable)} of its values could not be read: they count as empty."
        )
    if item.unknown_rows:
        shown = ", ".join(item.unknown_values)
        rule = f" They are {item.unknown_rule}." if item.unknown_rule else ""
        sentences.append(
            f"{item.unknown_rows:,} rows ({_pct(item.unknown_share)}) have a category the model "
            f"never saw ({shown}).{rule}"
        )
    if item.out_of_range:
        sentences.append(
            f"{_pct(item.out_of_range)} of rows are outside the training range ({item.range})."
        )
    if _text_shift(item):
        sentences.append(
            f"{_pct(item.unknown_terms or 0.0)} of its terms are unknown to the model "
            f"({_pct(item.training_unknown_terms or 0.0)} in training): they are ignored."
        )
    if item.unmatched_texts and item.unmatched_texts >= NOTABLE:
        sentences.append(
            f"{_pct(item.unmatched_texts)} of its texts have no known term: the model reads them "
            "as empty."
        )
    return " ".join(sentences) or None


def _verdict(
    checks: list[ColumnCheck], missing: list[str], rows: int
) -> tuple[Level, str, list[str]]:
    problems: list[str] = []
    if missing:
        problems.append(
            f"Missing column{'s' if len(missing) > 1 else ''} the model needs: "
            f"{', '.join(missing)}."
        )
    unreadable = [c.column for c in checks if c.found is not None and c.unreadable >= UNREADABLE]
    if unreadable:
        problems.append(f"Values that cannot be read in the model's type: {', '.join(unreadable)}.")
    if rows == 0:
        problems.append("No rows to predict.")
    if problems:
        return "blocked", f"Blocked: {problems[0]}", problems
    notes = []
    for c in checks:
        if c.unknown_share >= NOTABLE:
            notes.append(f"{c.column}: {_pct(c.unknown_share)} of rows have an unknown category.")
        if c.out_of_range >= NOTABLE:
            notes.append(
                f"{c.column}: {_pct(c.out_of_range)} of rows are outside the training range."
            )
        if 0 < c.unreadable < UNREADABLE:
            notes.append(f"{c.column}: {_pct(c.unreadable)} of its values could not be read.")
        if _text_shift(c):
            notes.append(
                f"{c.column}: {_pct(c.unknown_terms or 0.0)} of its terms are unknown to the "
                f"model ({_pct(c.training_unknown_terms or 0.0)} in training)."
            )
    if notes:
        return (
            "check",
            f"Check: the model can predict, but {notes[0][0].lower()}{notes[0][1:]}",
            notes,
        )
    return (
        "ready",
        f"Ready: the {rows:,} rows have the columns, types and values the model was trained on.",
        [],
    )


# --- Predicting ----------------------------------------------------------------------------------


def _labels_of(card: ModelCard, chances: np.ndarray) -> np.ndarray:
    labels = card.labels
    predicted = np.asarray(labels, dtype=object)[chances.argmax(axis=1)]
    if card.threshold is not None and card.positive in labels and len(labels) == 2:
        positive = labels.index(card.positive)
        other = labels[1 - positive]
        predicted = np.where(chances[:, positive] >= card.threshold, card.positive, other)
    return predicted.astype(str)


def _reliability(card: ModelCard, data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Rows with an important value the model never saw (unknown category, out of its range, a
    text without a single known term)."""
    reasons: list[list[str]] = [[] for _ in range(len(data))]
    flagged = np.zeros(len(data), dtype=bool)
    for column in card.columns:
        if not column.important or column.name not in data.columns:
            continue
        values = cast(pd.Series, data[column.name])
        if column.text is not None:
            hit = textual.gap(
                textual.texts(values),
                textual.cutter(column.text.analyzer),
                set(column.text.terms),
            ).rows
            what = "no known word"
        elif column.categories is not None:
            text = values.astype(object)
            hit = (text.notna() & ~text.astype(str).isin(set(column.categories))).to_numpy()
            what = "unknown category"
        elif column.minimum is not None and column.maximum is not None:
            numbers = (
                _seconds(values) if column.type == "datetime" else values.to_numpy(dtype=float)
            )
            with np.errstate(invalid="ignore"):
                hit = np.isfinite(numbers) & (
                    (numbers < column.minimum) | (numbers > column.maximum)
                )
            what = "outside the training range"
        else:
            continue
        flagged |= hit
        for i in np.flatnonzero(hit):
            reasons[i].append(f"{column.name}: {what}")
    return flagged, ["; ".join(r) for r in reasons]


def predict_frame(pipeline: Any, card: ModelCard, data: pd.DataFrame) -> pd.DataFrame:
    """Predictions for rows read by `read_rows`: the kept columns, `predicted` (and `p(<class>)`
    for a classifier, `low` / `high` for a number), `less_reliable` with `why`, and `actual`
    when the file has the answer."""
    x = cast(pd.DataFrame, data[[c.name for c in card.columns]])
    out = pd.DataFrame(index=data.index)
    for name in data.columns:
        if name.startswith("keep:"):
            out[name[5:]] = data[name]
    if card.task == "regression":
        predicted = np.asarray(pipeline.predict(x), dtype=float)
        out["predicted"] = predicted
        bands = getattr(pipeline, intervals.BANDS_ATTRIBUTE, None)
        if card.interval is not None and card.interval.method == "cqr" and bands is not None:
            out["low"], out["high"] = bands.interval(pipeline[:-1].transform(x))
        elif card.interval is not None:
            out["low"] = predicted - card.interval.half_width
            out["high"] = predicted + card.interval.half_width
    else:
        chances = np.asarray(pipeline.predict_proba(x), dtype=float)
        out["predicted"] = _labels_of(card, chances)
        for j, label in enumerate(card.labels):
            out[f"p({label})"] = chances[:, j]
    flagged, why = _reliability(card, data)
    out["less_reliable"] = flagged
    out["why"] = why
    if "target:" in data.columns:
        out["actual"] = _actual(card, cast(pd.Series, data["target:"]))
    return out


def _actual(card: ModelCard, values: pd.Series) -> pd.Series:
    if card.classes is not None:
        return binning.apply(values, card.classes)
    if card.task == "regression":
        return pd.Series(pd.to_numeric(values, errors="coerce"), index=values.index, dtype=float)
    labels = values.astype(object).where(values.notna())
    if trainer.OTHER in card.labels:  # beyond 20 classes the rest was grouped as Other
        known = set(card.labels)
        labels = labels.where(labels.isna() | labels.astype(str).isin(known), trainer.OTHER)
    return labels


@dataclass
class PredictionDistribution:
    """The predictions on this data against those on the validation rows."""

    kind: Literal["number", "classes"]
    edges: list[float]  # a number: the bins
    labels: list[str]  # classes
    new: list[float]  # share of the rows per bin or class
    validation: list[float]


@dataclass
class PredictionReport:
    rows: int
    less_reliable: int
    columns: list[str]  # the result's columns, in order
    distribution: PredictionDistribution
    scores: dict[str, float | None] | None  # on this data, when it has the answer
    scored_rows: int
    validation_scores: dict[str, float | None]
    primary: str
    interval: Interval | None


class Accumulator:
    """What a prediction keeps from block to block for its summary."""

    def __init__(self, card: ModelCard) -> None:
        self.card = card
        self.rows = 0
        self.flagged = 0
        self.values: list[np.ndarray] = []  # a number's predictions
        self.counts: dict[str, int] = {}  # classes predicted
        self.actual: list[np.ndarray] = []
        self.predicted: list[np.ndarray] = []
        self.chances: list[np.ndarray] = []
        self.scored = 0
        self.columns: list[str] = []

    def add(self, block: pd.DataFrame) -> None:
        card = self.card
        self.rows += len(block)
        self.flagged += int(block["less_reliable"].to_numpy(dtype=bool).sum())
        self.columns = [str(c) for c in block.columns]
        if card.task == "regression":
            self.values.append(block["predicted"].to_numpy(dtype=float))
        else:
            for label, n in block["predicted"].value_counts().items():
                self.counts[str(label)] = self.counts.get(str(label), 0) + int(n)
        if "actual" not in block.columns or self.scored >= METRIC_ROWS:
            return
        known = block[block["actual"].notna()].iloc[: METRIC_ROWS - self.scored]
        if known.empty:
            return
        self.scored += len(known)
        if card.task == "regression":
            self.actual.append(known["actual"].to_numpy(dtype=float))
            self.predicted.append(known["predicted"].to_numpy(dtype=float))
        else:
            self.actual.append(known["actual"].astype(str).to_numpy())
            self.predicted.append(known["predicted"].astype(str).to_numpy())
            self.chances.append(
                known[[f"p({label})" for label in card.labels]].to_numpy(dtype=float)
            )

    def summary(self) -> PredictionReport:
        card = self.card
        scores = None
        if self.scored:
            actual = np.concatenate(self.actual)
            predicted = np.concatenate(self.predicted)
            if card.task == "regression":
                scores = evaluate.regression_metrics(actual, predicted)
            else:
                scores = evaluate.classification_metrics(
                    actual, predicted, np.concatenate(self.chances), card.labels, card.positive
                )
        return PredictionReport(
            self.rows,
            self.flagged,
            self.columns,
            self._distribution(),
            scores,
            self.scored,
            card.validation.scores,
            card.validation.primary,
            card.interval,
        )

    def _distribution(self) -> PredictionDistribution:
        card = self.card
        if card.task != "regression":
            labels = list(dict.fromkeys([*card.labels, *self.counts]))
            total = sum(self.counts.values()) or 1
            return PredictionDistribution(
                "classes",
                [],
                labels,
                [self.counts.get(label, 0) / total for label in labels],
                [card.validation.predicted_shares.get(label, 0.0) for label in labels],
            )
        values = np.concatenate(self.values) if self.values else np.zeros(0)
        values = values[np.isfinite(values)]
        quantiles = np.asarray(card.validation.predicted_quantiles, dtype=float)
        ends = [float(v) for v in quantiles[[0, -1]]] if len(quantiles) else []
        if len(values):
            ends += [float(values.min()), float(values.max())]
        if not ends:
            return PredictionDistribution("number", [], [], [], [])
        low, high = min(ends), max(ends)
        if high <= low:
            high = low + 1.0
        edges = np.linspace(low, high, BINS + 1)
        counts, _ = np.histogram(values, bins=edges)
        new = counts / counts.sum() if counts.sum() else counts.astype(float)
        validation = np.zeros(BINS)
        if len(quantiles):
            levels = np.linspace(0, 1, len(quantiles))
            cdf = np.interp(edges, quantiles, levels, left=0.0, right=1.0)
            validation = np.diff(cdf)
        return PredictionDistribution(
            "number",
            [float(v) for v in edges],
            [],
            [float(v) for v in new],
            [float(v) for v in validation],
        )


def predict_blocks(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    pipeline: Any,
    card: ModelCard,
    views: list[Any],
    keep: list[str] | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    block_rows: int = BLOCK_ROWS,
) -> Iterator[pd.DataFrame]:
    """The predictions, block after block in the file's order, each with `__row` (0-based file
    row). `should_stop` is asked between blocks."""
    last = -1
    while not should_stop():
        data = read_rows(
            connection,
            source,
            card,
            views,
            keep,
            where=f"WHERE file_row_number > {last}",
            limit=f"LIMIT {block_rows}",
        )
        if data.empty:
            return
        block = predict_frame(pipeline, card, data)
        block.insert(0, "__row", data["__row"].to_numpy(dtype=int))
        last = int(data["__row"].iloc[-1])
        yield block.reset_index(drop=True)
        if len(data) < block_rows:
            return


# --- Explaining one predicted row ----------------------------------------------------------------


@dataclass
class Contribution:
    label: str  # "region = North"
    value: float


@dataclass
class RowExplanation:
    row: int
    method: str | None
    scale: str | None
    base: float | None  # the average prediction of the rows it is compared with
    positive: str | None
    steps: list[Contribution]  # the strongest contributions, then the others together
    unavailable: str | None  # why there is no explanation


def explain_row(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    pipeline: Any,
    card: ModelCard,
    views: list[Any],
    row: int,
) -> RowExplanation:
    """SHAP contributions of one row of the new data, against up to 2 000 of its rows drawn with
    a seed (the average they start from)."""
    data = read_rows(
        connection,
        source,
        card,
        views,
        order=f"ORDER BY (file_row_number = {int(row)}) DESC, hash(file_row_number + {SEED})",
        limit=f"LIMIT {EXPLAIN_ROWS}",
    )
    if data.empty or int(data["__row"].iloc[0]) != row:
        return RowExplanation(row, None, None, None, None, [], "This row is not in the data.")
    x = cast(pd.DataFrame, data[[c.name for c in card.columns]])
    rows = data["__row"].to_numpy(dtype=int)
    shap, unavailable = explainer.shap_values(pipeline, x, card.task, rows, [row], card.positive)
    if shap is None:
        return RowExplanation(row, None, None, None, None, [], unavailable)
    at = shap.rows.index(row)
    values, labels = shap.matrix[at], shap.labels[at]
    order = sorted(range(len(values)), key=lambda j: -abs(values[j]))
    steps = [Contribution(f"{shap.columns[j]} = {labels[j]}", values[j]) for j in order[:WATERFALL]]
    rest = order[WATERFALL:]
    if rest:
        steps.append(Contribution(f"{len(rest)} other features", sum(values[j] for j in rest)))
    return RowExplanation(row, shap.method, shap.scale, shap.base, shap.positive, steps, None)


def _first_and_sample(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    card: ModelCard,
    views: list[Any],
    row: int,
) -> pd.DataFrame:
    """The row first, then up to 2 000 rows of the same data drawn with a seed."""
    return read_rows(
        connection,
        source,
        card,
        views,
        order=f"ORDER BY (file_row_number = {int(row)}) DESC, hash(file_row_number + {SEED})",
        limit=f"LIMIT {EXPLAIN_ROWS}",
    )


def counterfactual_row(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    pipeline: Any,
    card: ModelCard,
    views: list[Any],
    row: int,
    editable: list[str] | None = None,
    target_class: str | None = None,
    target_value: float | None = None,
    max_changes: int = whatif.MAX_CHANGES,
) -> whatif.Counterfactual:
    """R-XAI-21 (125 bis): what would change the prediction of one row of the new data. The
    values tried and the nearest real row come from up to 2 000 rows of that same data (drawn
    with a seed), not from the training rows: a change is judged plausible against the data the
    model is used on."""
    data = _first_and_sample(connection, source, card, views, row)
    if data.empty or int(data["__row"].iloc[0]) != row:
        raise whatif.CounterfactualError("This row is not in the data.")
    x = cast(pd.DataFrame, data[[c.name for c in card.columns]])
    rows = data["__row"].to_numpy(dtype=int)
    found = whatif.counterfactual(
        pipeline,
        x,
        rows,
        card.task,
        row,
        [_field(c) for c in card.columns],
        set(textual.steps_of(pipeline)),
        editable,
        target_class,
        target_value,
        max_changes,
        card.threshold,
        card.positive,
    )
    found.plausibility = (
        f"{found.plausibility} Compared with {len(x):,} rows of this data (R-XAI-21)."
    ).strip()
    return found


def one_row(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    card: ModelCard,
    views: list[Any],
    row: int,
) -> pd.DataFrame:
    """One row of the new data, read as the model reads it (R-XAI-18)."""
    data = read_rows(
        connection,
        source,
        card,
        views,
        where=f"WHERE file_row_number = {int(row)}",
        limit="LIMIT 1",
    )
    return data.reset_index(drop=True)


# --- Exporting the model -------------------------------------------------------------------------

PREDICT_SCRIPT = '''"""Predictions of a model exported by Quantilo.

    pip install "quantilo[ml]"     # the versions in model_card.json
    python predict.py new_rows.csv predictions.csv [{zip}]

The model zip must sit next to this script (or be given as the third argument). Loading it runs
the pickled pipeline: only use a model you trust (see model_card.json).
"""

import sys
from pathlib import Path

import quantilo as dx

MODEL = Path(__file__).with_name("{zip}")


def main(arguments: list[str]) -> int:
    if len(arguments) < 2:
        print(__doc__)
        return 2
    model = dx.load_model(arguments[2] if len(arguments) > 2 else MODEL)
    data = dx.open(arguments[0])
    check = model.check(data)
    print(check.message)
    if check.level == "blocked":
        return 1
    model.predict(data).to_csv(arguments[1], index=False)
    print(f"{{len(data):,}} rows predicted: {{arguments[1]}}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''


def export_zip(card: ModelCard, pipeline_bytes: bytes, name: str) -> bytes:
    """The model as a zip: `model.joblib` (the pipeline), `model_card.json` (what it expects, its
    validation, the versions, the pipeline's SHA-256) and `predict.py`. No data rows."""
    sealed = card.model_copy(update={"pipeline_sha256": hashlib.sha256(pipeline_bytes).hexdigest()})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model.joblib", pipeline_bytes)
        archive.writestr("model_card.json", json.dumps(sealed.model_dump(mode="json"), indent=2))
        archive.writestr(
            "predict.py",
            PREDICT_SCRIPT.replace("{zip}", name).replace("{{", "{").replace("}}", "}"),
        )
    return buffer.getvalue()


def zip_name(card: ModelCard) -> str:
    """churn_lightgbm.zip: the target and the model, safe for every file system."""
    stem = f"{card.target.name}_{card.model}".lower()
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem).strip("_")
    return f"{safe or 'model'}.zip"
