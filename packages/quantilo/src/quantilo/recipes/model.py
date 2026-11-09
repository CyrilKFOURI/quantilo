"""What a recipe is: the work kept on a dataset (corrections, filters, KPIs, time series, models,
outputs), as names of columns and settings, never rows. Versioned JSON schema (`schema: 1`).

An imported recipe is validated against this schema: an unknown field or schema version is
refused with the field named. Nothing in a recipe is ever run as code.
"""

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from quantilo.errors import ExplorerError
from quantilo.filters.model import MAX_CONDITIONS, Condition
from quantilo.kpi.definition import Definition
from quantilo.ml import classes as binning
from quantilo.ml import evaluate
from quantilo.ml import preparation as prep
from quantilo.timeseries.series import Aggregate, Align, Gaps, Period
from quantilo.typing.types import AnalysisType

FORMAT = "quantilo/recipe"
LEGACY_FORMAT = "cyfouri-data-explorer/recipe"  # recipes exported before the rename (043)
SCHEMA_VERSION = 1
MAX_STEPS = 500
MAX_COLUMNS = 5_000
MAX_CATEGORIES = 50  # known values kept for a category read by a step (more: not checked)
MAX_BYTES = 2_000_000  # an imported .recipe.json

Kind = Literal["column", "filter", "kpi", "timeseries", "ml", "output"]
KIND_ORDER: tuple[Kind, ...] = ("column", "filter", "kpi", "timeseries", "ml", "output")
Name = Annotated[str, Field(min_length=1, max_length=500)]


class RecipeError(ExplorerError):
    def __init__(
        self,
        message: str,
        code: str = "invalid_recipe",
        status_code: int = 422,
        field: str | None = None,
    ) -> None:
        super().__init__(message, code=code, status_code=status_code)
        self.field = field
        if field is not None:
            self.details["field"] = field  # the field to fix, named in the error body


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecipeColumn(Strict):
    """A column of the dataset the recipe was made on: its name and detected type (what a new
    file is matched on), and for a category read by a step, its values (to flag new ones)."""

    name: Name
    type: AnalysisType
    categories: list[Annotated[str, Field(max_length=500)]] | None = Field(
        default=None, max_length=MAX_CATEGORIES
    )


class StepBase(Strict):
    id: str = Field(pattern=r"^s[0-9]{1,4}$")
    enabled: bool = True


class ColumnStep(StepBase):
    """A correction of one column (`None` fields are left as detected)."""

    kind: Literal["column"] = "column"
    column: Name
    type: AnalysisType | None = None
    role: str | None = Field(default=None, max_length=80)  # "" = no role
    order: list[Annotated[str, Field(max_length=500)]] | None = Field(default=None, max_length=1000)
    excluded: bool | None = None

    @model_validator(mode="after")
    def _something(self) -> "ColumnStep":
        if all(v is None for v in (self.type, self.role, self.order, self.excluded)):
            raise ValueError("A column step changes the type, role, order or exclusion.")
        return self


class FilterStep(StepBase):
    """A saved filter; the active one gives its rows to the steps that name it (`filter`)."""

    kind: Literal["filter"] = "filter"
    name: Name
    conditions: list[Condition] = Field(min_length=1, max_length=MAX_CONDITIONS)
    active: bool = False


class KpiStep(StepBase):
    kind: Literal["kpi"] = "kpi"
    name: Name
    definition: Definition
    filter: str | None = Field(default=None, pattern=r"^s[0-9]{1,4}$")  # a filter step's id


class RecipeDriver(Strict):
    """A forecast driver (028 A4), as `timeseries.drivers.Driver`."""

    column: str = Field(max_length=500)
    kind: Literal["column", "flag", "calendar"] = "column"
    known: bool = False  # known in advance; past only otherwise (lagged by the horizon)
    future: Literal["file", "constant", "last"] = "file"
    value: float | None = None  # the constant, when future = "constant"


class SeriesConfig(Strict):
    """A time series analysis as the Time series page keeps it (Save analysis)."""

    time: str | None = Field(default=None, max_length=500)
    value: str | None = Field(default=None, max_length=500)  # None: count rows per period
    group: str | None = Field(default=None, max_length=500)
    series: str | None = Field(default=None, max_length=500)  # the group analysed
    period: Period | Literal["auto"] = "auto"
    aggregate: Aggregate | Literal["auto"] = "auto"
    window: int | None = Field(default=None, ge=2, le=365)
    align: Align = "trailing"
    gaps: Gaps = "leave"
    season: int | None = Field(default=None, ge=2, le=400)
    models: list[Annotated[str, Field(max_length=40)]] | None = Field(default=None, max_length=40)
    horizon: int | None = Field(default=None, ge=1, le=500)  # None: the page's default
    drivers: list[RecipeDriver] | None = Field(default=None, max_length=6)  # 028 A4
    order: tuple[int, int, int] = (1, 1, 1)  # SARIMA's orders, when it is among the models
    seasonal_order: tuple[int, int, int] = (0, 0, 0)

    @model_validator(mode="after")
    def _orders(self) -> "SeriesConfig":
        for orders in (self.order, self.seasonal_order):
            if not all(0 <= v <= 3 for v in orders):
                raise ValueError("SARIMA orders go from 0 to 3.")
        return self


class TimeSeriesStep(StepBase):
    kind: Literal["timeseries"] = "timeseries"
    name: Name
    analysis: SeriesConfig
    forecast: bool = True  # run the forecast (its next value is a key figure)
    filter: str | None = Field(default=None, pattern=r"^s[0-9]{1,4}$")


class MlStep(StepBase):
    """A model trained again on the new rows (not reused: that is predicting, 037)."""

    kind: Literal["ml"] = "ml"
    target: Name
    model: str | None = Field(default=None, max_length=60)
    settings: dict[str, JsonValue] = Field(default_factory=dict, max_length=200)
    preparation: prep.Choices | None = None
    classes: binning.Classes | None = None
    validation: evaluate.Validation = Field(default_factory=evaluate.Validation)
    conditions: list[Condition] = Field(default_factory=list, max_length=MAX_CONDITIONS)
    inputs: list[Annotated[str, Field(max_length=500)]] = Field(
        default_factory=list, max_length=MAX_COLUMNS
    )  # the columns the model used (missing ones are a warning, not a reason to skip)


class OutputStep(StepBase):
    kind: Literal["output"] = "output"
    report: bool = False  # the HTML report
    csv: bool = False  # the (filtered) rows as CSV
    filter: str | None = Field(default=None, pattern=r"^s[0-9]{1,4}$")

    @model_validator(mode="after")
    def _something(self) -> "OutputStep":
        if not (self.report or self.csv):
            raise ValueError("An output step writes the report, the CSV or both.")
        return self


Step = Annotated[
    ColumnStep | FilterStep | KpiStep | TimeSeriesStep | MlStep | OutputStep,
    Field(discriminator="kind"),
]


class Recipe(Strict):
    """A recipe: `Recipe.from_dataset(ds)`, `.run(other)`, `.to_json()`, `.to_python()`."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    format: Literal["quantilo/recipe"] = FORMAT
    schema_version: Literal[1] = Field(default=SCHEMA_VERSION, alias="schema")
    name: Name
    version: int = Field(default=1, ge=1, le=1_000_000)
    created_at: str | None = Field(default=None, max_length=40)
    source: str | None = Field(default=None, max_length=500)  # the dataset it was made on
    columns: list[RecipeColumn] = Field(default_factory=list, max_length=MAX_COLUMNS)
    steps: list[Step] = Field(default_factory=list, max_length=MAX_STEPS)

    @model_validator(mode="after")
    def _consistent(self) -> "Recipe":
        ids = [step.id for step in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("Two steps share the same id.")
        filters = {step.id for step in self.steps if isinstance(step, FilterStep)}
        for step in self.steps:
            used = getattr(step, "filter", None)
            if used is not None and used not in filters:
                raise ValueError(f"Step {step.id} uses an unknown filter step {used}.")
        if sum(1 for s in self.steps if isinstance(s, FilterStep) and s.active) > 1:
            raise ValueError("Only one filter can be active.")
        names = [column.name for column in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("Two expected columns share the same name.")
        return self

    # --- Library ---------------------------------------------------------------------------

    @classmethod
    def from_dataset(cls, ds: Any, name: str, **include: Any) -> "Recipe":
        """A recipe from a library `Dataset`: its column corrections, its filters (as the active
        filter), and the KPIs, time series analyses, trained models and outputs given. See
        `quantilo.recipes.library.from_dataset`."""
        from quantilo.recipes.library import from_dataset

        return from_dataset(ds, name, **include)

    @classmethod
    def from_json(cls, text: str | bytes) -> "Recipe":
        return parse(text)

    def to_json(self, indent: int | None = 2) -> str:
        return self.model_dump_json(by_alias=True, indent=indent)

    def to_python(self, path: str = "data.csv") -> str:
        """A readable script doing the same with the library; it prints the key figures."""
        from quantilo.recipes.script import to_python

        return to_python(self, path)

    def match(self, ds: Any) -> Any:
        """How the columns of `ds` pair with the recipe's (same, matched, proposed, missing)."""
        from quantilo.recipes.matching import match_columns

        return match_columns(self.columns, [(c.name, c.detected_type) for c in ds.types])

    def run(
        self,
        ds: Any,
        mapping: dict[str, str | None] | None = None,
        *,
        accept_proposals: bool = False,
        output_dir: str | None = None,
    ) -> Any:
        """Replay the recipe on a library `Dataset` (changed in place: its corrections); returns
        the report with the key figures. `mapping`: recipe column → column of `ds` (None:
        missing), over the automatic matching."""
        from quantilo.recipes.library import run

        return run(self, ds, mapping, accept_proposals=accept_proposals, output_dir=output_dir)


def _unknown(raw: Any, known: Any, path: str) -> str | None:
    """The first field of `raw` the validated model does not have (nested models that would
    otherwise ignore extra fields included)."""
    if isinstance(raw, dict) and isinstance(known, dict):
        for key, value in raw.items():
            where = f"{path}.{key}" if path else str(key)
            if key not in known:
                return where
            found = _unknown(value, known[key], where)
            if found:
                return found
    elif isinstance(raw, list) and isinstance(known, list | tuple):
        for index, (value, kept) in enumerate(zip(raw, known, strict=False)):
            found = _unknown(value, kept, f"{path}[{index}]")
            if found:
                return found
    return None


def _where(location: tuple[Any, ...]) -> str:
    """steps.2.kpi.definition.measures.0 → steps[2].definition.measures[0] (the union's tag
    left out)."""
    text = ""
    for index, part in enumerate(location):
        tag = index >= 2 and location[index - 2] == "steps" and isinstance(location[index - 1], int)
        if isinstance(part, int):
            text += f"[{part}]"
        elif tag and part in KIND_ORDER:
            continue
        else:
            text += f".{part}" if text else str(part)
    return text or "recipe"


def parse(text: str | bytes | dict[str, Any]) -> Recipe:
    """A recipe read from its JSON, refused with a clear message when it is not one."""
    if isinstance(text, dict):
        raw = text
    else:
        if len(text) > MAX_BYTES:
            raise RecipeError(f"This file is too large for a recipe ({len(text):,} bytes).")
        try:
            raw = json.loads(text)
        except (ValueError, UnicodeDecodeError) as error:
            raise RecipeError(f"This is not a recipe file: invalid JSON ({error}).") from None
    if not isinstance(raw, dict):
        raise RecipeError("This is not a recipe file: expected a JSON object.")
    if raw.get("format") == LEGACY_FORMAT:
        raw = {**raw, "format": FORMAT}
    if raw.get("format") != FORMAT:
        raise RecipeError(
            f"This is not a recipe file: field `format` must be {FORMAT!r}.", field="format"
        )
    if raw.get("schema") != SCHEMA_VERSION:
        raise RecipeError(
            f"Unknown recipe schema version {raw.get('schema')!r} in field `schema`: this "
            f"version of the app reads version {SCHEMA_VERSION}.",
            field="schema",
        )
    try:
        recipe = Recipe.model_validate(raw)
    except ValidationError as error:
        first = error.errors()[0]
        where = _where(tuple(first["loc"]))
        if first["type"] == "extra_forbidden":
            raise RecipeError(f"Unknown field `{where}` in the recipe.", field=where) from None
        raise RecipeError(
            f"Invalid recipe, field `{where}`: {first['msg']}.", field=where
        ) from None
    unknown = _unknown(raw, recipe.model_dump(mode="json", by_alias=True), "")
    if unknown:
        raise RecipeError(f"Unknown field `{unknown}` in the recipe.", field=unknown)
    return recipe


def ordered(steps: list[Any]) -> list[Any]:
    """Steps in execution order: columns, filters, KPIs, time series, models, outputs."""
    return sorted(steps, key=lambda step: KIND_ORDER.index(step.kind))
