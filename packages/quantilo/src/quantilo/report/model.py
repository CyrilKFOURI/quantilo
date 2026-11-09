"""What a dataset report holds (spec 025): plain data, rendered to HTML by `html.render`."""

from dataclasses import dataclass, field
from typing import Literal

from quantilo.charts.prepare import Chart
from quantilo.compare.analyze import Comparison
from quantilo.correlations.matrix import Matrix
from quantilo.variable.analyze import VariableProfile
from quantilo.variable.outliers import OutlierRule

Severity = Literal["warning", "notice", "link"]


@dataclass
class Fact:
    label: str
    value: str


@dataclass
class Finding:
    """One thing worth a look (R-REP-14), with its proof and where to see it."""

    severity: Severity
    title: str
    text: str
    columns: list[str] = field(default_factory=list)
    anchor: str | None = None  # card the finding points to
    score: float = 0.0  # importance, for the key findings


@dataclass
class Related:
    name: str
    anchor: str | None
    measure: str  # "Pearson r", "η"…
    value: float


@dataclass
class Share:
    label: str
    count: int
    share: float  # 0..1


@dataclass
class VariableCard:
    """One column (R-REP-05): figures, charts, what stands out."""

    name: str
    anchor: str
    type: str
    type_label: str
    role: str | None
    missing_pct: float
    distinct: int
    facts: list[Fact] = field(default_factory=list)
    quantiles: list[Fact] = field(default_factory=list)  # numbers
    top: list[Share] = field(default_factory=list)  # categories
    findings: list[Finding] = field(default_factory=list)
    related: list[Related] = field(default_factory=list)
    chart: Chart | None = None  # main chart; None for identifiers and free text
    extra: list[tuple[str, Chart]] = field(default_factory=list)  # dates: month, weekday
    profile: VariableProfile | None = None  # numbers: box plot, density, QQ-plot
    outliers: OutlierRule | None = None


@dataclass
class PairView:
    """A strongly linked pair, drawn as Compare draws it (R-REP-15)."""

    a: str
    b: str
    measure: str
    value: float
    evidence: str
    comparison: Comparison
    verdict: str  # recommended test, in words


@dataclass
class PairRow:
    a: str
    b: str
    metric: str
    value: float
    q_value: float | None
    evidence: str


@dataclass
class Missing:
    name: str
    pct: float  # 0..100


@dataclass
class Report:
    title: str
    source: str
    generated_at: str
    version: str
    rows: int
    total_rows: int
    columns: int
    filters: list[str]
    summary: list[Fact]
    types: list[tuple[str, int]]  # (type label, columns)
    key_findings: list[Finding]
    findings_total: int
    variables: list[VariableCard]
    excluded: list[str]
    pairs_views: list[PairView]
    matrix: Matrix | None
    pairs: list[PairRow]
    insights: list[Finding]
    missing: list[Missing]
    complete: int
    methods: list[Fact]
