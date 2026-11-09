"""What a KPI is: measures, an optional time axis, up to two splits, and how it is shown.

"auto" (or None) always means "let the app choose": the choice is resolved at compute time and
returned with its rule and reason, so a saved KPI keeps following the data.
"""

from datetime import date
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

Agg = Literal["count", "sum", "mean", "median", "min", "max", "distinct", "true_share"]
Grain = Literal["day", "week", "month", "quarter", "year"]
Chart = Literal[
    "bars",
    "grouped_bars",
    "stacked_bars",
    "line",
    "area",
    "stacked_area",
    "hbar",
    "hbar_stacked",
    "heatmap",
    "none",
]
ShowAs = Literal[
    "value",
    "share_total",
    "share_row",
    "share_column",
    "change",
    "change_pct",
    "change_yoy",
    "running",
]
Layout = Literal["periods_rows", "periods_columns", "groups_rows", "crosstab", "single"]
Switch = Literal["auto", "on", "off"]

AGGS: tuple[Agg, ...] = get_args(Agg)
GRAINS: tuple[Grain, ...] = get_args(Grain)
CHARTS: tuple[Chart, ...] = get_args(Chart)
# Rows and sums add up: the value of a whole is the sum of its parts. Nothing else does
# (a customer bought in two countries counts once in the total).
ADDITIVE: frozenset[Agg] = frozenset({"count", "sum"})
NUMERIC_AGGS: frozenset[Agg] = frozenset({"sum", "mean", "median", "min", "max"})
MAX_MEASURES = 4
MAX_GROUPS = 2
RATIO_ID = "ratio"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Format(Strict):
    decimals: int | Literal["auto"] = Field(default="auto")
    compact: Switch = "auto"
    prefix: str = Field(default="", max_length=8)
    suffix: str = Field(default="", max_length=8)

    @model_validator(mode="after")
    def _decimals(self) -> "Format":
        if isinstance(self.decimals, int) and not 0 <= self.decimals <= 4:
            raise ValueError("Decimals go from 0 to 4.")
        return self


class Measure(Strict):
    id: str = Field(pattern=r"^m[0-9]{1,2}$")
    agg: Agg
    column: str | None = None  # None for "count"
    label: str | None = Field(default=None, max_length=80)
    format: Format = Field(default_factory=Format)

    @model_validator(mode="after")
    def _column(self) -> "Measure":
        if self.agg == "count":
            self.column = None
        elif not self.column:
            raise ValueError(f"A {self.agg} needs a column.")
        return self


class Ratio(Strict):
    numerator: str
    denominator: str
    label: str | None = Field(default=None, max_length=80)
    percent: bool = False
    format: Format = Field(default_factory=Format)


class Period(Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["all", "last", "ytd", "range"] = "all"
    n: int = Field(default=12, ge=1, le=2_000)
    start: date | None = Field(default=None, alias="from")
    end: date | None = Field(default=None, alias="to")

    @model_validator(mode="after")
    def _range(self) -> "Period":
        if self.kind == "range":
            if self.start is None and self.end is None:
                raise ValueError("A custom range needs a start or an end.")
            if self.start and self.end and self.start > self.end:
                raise ValueError("The range starts after it ends.")
        return self


class Time(Strict):
    column: str
    grain: Grain | Literal["auto"] = "auto"
    period: Period = Field(default_factory=Period)
    page_filters: bool = True


class Group(Strict):
    column: str


class TableView(Strict):
    layout: Layout | Literal["auto"] = "auto"
    totals: Switch = "auto"
    shading: Switch = "auto"
    compare: list[Literal["previous", "last_year"]] = []
    sort: str | None = None


class View(Strict):
    chart: Chart | Literal["auto"] = "auto"
    share: bool = False
    labels: Switch = "auto"
    table: TableView = Field(default_factory=TableView)
    show_as: ShowAs = "value"


class Definition(Strict):
    measures: list[Measure] = Field(min_length=1, max_length=MAX_MEASURES)
    ratio: Ratio | None = None
    time: Time | None = None
    groups: list[Group] = Field(default_factory=list, max_length=MAX_GROUPS)
    top: int = Field(default=8, ge=1, le=20)
    other: Literal["show", "hide"] = "show"
    group_order: Literal["value", "alpha"] = "value"
    view: View = Field(default_factory=View)

    @model_validator(mode="after")
    def _consistent(self) -> "Definition":
        ids = [m.id for m in self.measures]
        if len(set(ids)) != len(ids):
            raise ValueError("Two measures share the same id.")
        if self.ratio is not None:
            for side in (self.ratio.numerator, self.ratio.denominator):
                if side not in ids:
                    raise ValueError(f"The ratio uses an unknown measure: {side}.")
            if self.ratio.numerator == self.ratio.denominator:
                raise ValueError("A ratio needs two different measures.")
        columns = [g.column for g in self.groups]
        if len(set(columns)) != len(columns):
            raise ValueError("Split by two different columns.")
        return self

    def additive(self, measure_id: str) -> bool:
        if measure_id == RATIO_ID:
            return False
        return next(m.agg for m in self.measures if m.id == measure_id) in ADDITIVE
