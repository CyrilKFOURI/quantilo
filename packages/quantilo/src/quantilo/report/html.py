"""Render a report as one self-contained HTML page (R-REP-01): inline CSS, no script, no link out,
light and dark themes from the system, a print style sheet."""

from html import escape

from quantilo.report import svg
from quantilo.report.model import Fact, Finding, PairView, Report, VariableCard

# The app's tokens (frontend/src/styles/tokens.css), light then dark.
CSS = """
:root {
  color-scheme: light dark;
  --surface-0: #f7f5f2; --surface-1: #fffdfa; --surface-2: #efebe5;
  --ink-1: #24201b; --ink-2: #5c544a; --ink-3: #71695f; --line: #e2dcd3;
  --accent: #ad4a27; --accent-soft: #f6e3da; --warn: #8f5b00; --warn-soft: #f7ecd6;
  --chart-1: #2a78d6; --chart-1-soft: #dfeaf8; --chart-2: #eb6834; --chart-8: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-0: #14120f; --surface-1: #1d1a16; --surface-2: #27231e;
    --ink-1: #ede8e1; --ink-2: #b5aca0; --ink-3: #988f83; --line: #332e28;
    --accent: #e07a50; --accent-soft: #3a2519; --warn: #e0a23a; --warn-soft: #33270f;
    --chart-1: #3987e5; --chart-1-soft: #1a2a3f; --chart-2: #d95926; --chart-8: #e66767;
  }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 64px; }
body {
  margin: 0; background: var(--surface-0); color: var(--ink-1);
  font: 14px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
main { max-width: 1240px; margin: 0 auto; padding: 0 28px 80px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { font-size: 34px; line-height: 1.15; margin: 4px 0 6px; letter-spacing: -0.02em; }
h2 { font-size: 22px; letter-spacing: -0.01em; margin: 0 0 4px; }
h3 { font-size: 15px; margin: 0; }
p { margin: 0 0 8px; }
.eyebrow {
  font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3);
  margin: 0;
}
.muted { color: var(--ink-3); }
.small { font-size: 12.5px; }
.lede { color: var(--ink-2); max-width: 72ch; margin-bottom: 18px; }
section { padding-top: 44px; }
section + section { border-top: 1px solid var(--line); margin-top: 44px; }

.masthead { padding: 40px 0 8px; }
.meta { display: flex; flex-wrap: wrap; gap: 4px 16px; color: var(--ink-3); font-size: 13px; }
.filters {
  display: inline-block; margin-top: 10px; padding: 4px 10px; border-radius: 999px;
  background: var(--accent-soft); color: var(--ink-1); font-size: 13px;
}
nav.toc {
  position: sticky; top: 0; z-index: 5; margin: 20px -28px 0; padding: 10px 28px;
  background: color-mix(in oklab, var(--surface-0) 88%, transparent);
  backdrop-filter: blur(8px); border-bottom: 1px solid var(--line);
  display: flex; flex-wrap: wrap; gap: 4px 20px; font-size: 13px;
}
nav.toc a { color: var(--ink-2); }
nav.toc a:hover { color: var(--accent); text-decoration: none; }

.band {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); margin: 24px 0 0;
  border-block: 1px solid var(--line);
}
.band div { padding: 14px 18px; }
.band div + div { border-left: 1px solid var(--line); }
.band dt { font-size: 12px; color: var(--ink-3); }
.band dd {
  margin: 2px 0 0; font-size: 24px; font-weight: 600; font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}

.findings {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr));
  gap: 14px; margin-top: 16px;
}
.finding {
  position: relative; background: var(--surface-1); border: 1px solid var(--line);
  border-radius: 10px; padding: 14px 16px 14px 20px; break-inside: avoid;
}
.finding::before {
  content: ""; position: absolute; left: 0; top: 12px; bottom: 12px; width: 4px;
  border-radius: 0 4px 4px 0; background: var(--ink-3);
}
.finding.warning::before { background: var(--warn); }
.finding.link::before { background: var(--chart-1); }
.finding .kind {
  font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-3);
}
.finding.warning .kind { color: var(--warn); }
.finding.link .kind { color: var(--chart-1); }
.finding h3 { margin: 2px 0 4px; }
.finding p { color: var(--ink-2); margin: 0; }
.finding .where { margin-top: 8px; font-size: 12.5px; }

.typebar {
  display: flex; height: 10px; border-radius: 999px; overflow: hidden; margin: 18px 0 8px;
  background: var(--surface-2);
}
.typebar span { display: block; height: 100%; }
.typebar span + span { border-left: 2px solid var(--surface-0); }
.legend {
  display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 12.5px; color: var(--ink-2);
  margin-bottom: 16px;
}
.legend i {
  display: inline-block; width: 9px; height: 9px; border-radius: 3px; margin-right: 6px;
}
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td {
  text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line);
  vertical-align: middle;
}
th { font-size: 12px; font-weight: 500; color: var(--ink-3); }
td.num, th.num { text-align: right; }
.glance td:first-child a { font-weight: 600; color: var(--ink-1); }
.meter {
  display: inline-block; width: 70px; height: 6px; border-radius: 999px;
  background: var(--surface-2); vertical-align: middle; margin-right: 8px; overflow: hidden;
}
.meter b { display: block; height: 100%; background: var(--warn); }
.pill {
  display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px;
  background: var(--surface-2); color: var(--ink-2); white-space: nowrap;
}
.pill.warning { background: var(--warn-soft); color: var(--warn); }
.pill.strong { background: var(--chart-1-soft); color: var(--chart-1); }
.scroll { overflow-x: auto; }

.variable {
  background: var(--surface-1); border: 1px solid var(--line); border-radius: 14px;
  padding: 22px 24px; margin-top: 18px; break-inside: avoid;
}
.variable > header {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 12px;
  padding-bottom: 14px; margin-bottom: 18px; border-bottom: 1px solid var(--line);
}
.variable > header h3 { font-size: 20px; letter-spacing: -0.01em; overflow-wrap: anywhere; }
.variable > header .stats { margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }
.vgrid {
  display: grid; grid-template-columns: 220px minmax(0, 1fr) 270px; gap: 28px;
  align-items: start;
}
.vgrid .center .chart { margin: 0 auto; }
.secondary {
  display: flex; flex-wrap: wrap; justify-content: center; gap: 18px; margin-top: 14px;
}
.secondary figure { margin: 0; flex: 1 1 260px; max-width: 320px; }
figcaption { font-size: 12px; color: var(--ink-3); margin-bottom: 4px; }
dl.facts { display: grid; grid-template-columns: 1fr auto; gap: 6px 12px; margin: 0; }
dl.facts dt { color: var(--ink-3); font-size: 13px; }
dl.facts dd {
  margin: 0; text-align: right; font-variant-numeric: tabular-nums; overflow-wrap: anywhere;
}
.mini { margin-top: 18px; }
.mini h4, .stand h4 { font-size: 12px; font-weight: 500; color: var(--ink-3); margin: 0 0 6px; }
.mini table td { padding: 4px 0; font-size: 13px; }
.mini table td + td { padding-left: 10px; }
.note {
  border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; background: var(--surface-2);
  font-size: 13px; color: var(--ink-2);
}
.note b { display: block; color: var(--ink-1); font-weight: 600; margin-bottom: 2px; }
.note.warning { background: var(--warn-soft); }
.note.warning b { color: var(--warn); }
.note.link b { color: var(--chart-1); }
.calm { color: var(--ink-3); font-size: 13px; }
ul.related { list-style: none; padding: 0; margin: 0; font-size: 13px; }
ul.related li {
  display: flex; justify-content: space-between; gap: 8px; padding: 4px 0;
  border-bottom: 1px dashed var(--line);
}

.pairs {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 560px), 1fr));
  gap: 18px; margin-top: 16px;
}
.pair {
  background: var(--surface-1); border: 1px solid var(--line); border-radius: 14px;
  padding: 18px 20px; break-inside: avoid;
}
.pair header {
  display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px;
  align-items: baseline; margin-bottom: 10px;
}
.pair .verdict { margin-top: 8px; font-size: 13px; color: var(--ink-2); }
.crosstab td { text-align: right; font-size: 13px; }
.crosstab th[scope=row] { text-align: left; color: var(--ink-1); font-weight: 500; }

.chart { display: block; width: 100%; height: auto; overflow: visible; }
.thumb { display: block; width: 120px; height: 28px; }
.chart text { font-size: 12px; fill: var(--ink-2); }
.chart text.tick { font-size: 11px; fill: var(--ink-3); }
.chart text.value { fill: var(--ink-3); }
.chart text.axis-title { font-size: 11.5px; fill: var(--ink-2); }
.chart text.cell { fill: var(--ink-1); font-size: 10.5px; }
.chart text.cell.strong { fill: #fff; }
.chart text.mark-mean { fill: var(--chart-2); font-size: 11px; }
.chart text.mark-median { fill: var(--ink-1); font-size: 11px; }
.chart .bar, .thumb .bar { fill: var(--chart-1); }
.chart .bar.others { fill: var(--ink-3); }
.chart .bar.warn { fill: var(--warn); }
.chart .track { fill: var(--surface-2); }
.chart .axis { stroke: var(--ink-3); stroke-width: 1; }
.chart .grid { stroke: var(--line); stroke-width: 1; }
.chart .curve { fill: none; stroke: var(--ink-1); stroke-width: 2; }
.chart .mark-mean { stroke: var(--chart-2); stroke-width: 2; stroke-dasharray: 5 4; }
.chart .mark-median { stroke: var(--ink-1); stroke-width: 2; stroke-dasharray: 2 3; }
.chart .box { fill: var(--chart-1-soft); stroke: var(--chart-1); stroke-width: 1.5; }
.chart .whisker { stroke: var(--chart-1); stroke-width: 1.5; }
.chart .median { stroke: var(--ink-1); stroke-width: 2.5; }
.chart .mean-dot { fill: var(--chart-2); }
.chart .outlier { fill: none; stroke: var(--chart-8); stroke-width: 1.5; }
.chart .dot { fill: var(--chart-1); fill-opacity: 0.45; }
.chart .fit { stroke: var(--chart-2); stroke-width: 2.5; }

dl.methods { display: grid; grid-template-columns: 220px 1fr; gap: 10px 24px; margin-top: 12px; }
dl.methods dt { font-weight: 600; }
dl.methods dd { margin: 0; color: var(--ink-2); }
footer {
  margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--ink-3);
  font-size: 12.5px;
}

@media (max-width: 1100px) {
  .vgrid { grid-template-columns: 200px minmax(0, 1fr); }
  .vgrid .stand { grid-column: 1 / -1; }
}
@media (max-width: 720px) {
  .vgrid { grid-template-columns: 1fr; }
  dl.methods { grid-template-columns: 1fr; }
}
@media print {
  :root {
    --surface-0: #fff; --surface-1: #fff; --surface-2: #f2efea;
    --ink-1: #24201b; --ink-2: #5c544a; --ink-3: #71695f; --line: #e2dcd3;
    --accent: #ad4a27; --accent-soft: #f6e3da; --warn: #8f5b00; --warn-soft: #f7ecd6;
    --chart-1: #2a78d6; --chart-1-soft: #dfeaf8; --chart-2: #eb6834; --chart-8: #e34948;
  }
  main { max-width: none; padding: 0; }
  nav.toc { display: none; }
  .vgrid { grid-template-columns: 190px minmax(0, 1fr) 210px; gap: 18px; }
}
"""

TYPE_COLORS = {
    "Continuous": "#3b6ea8",
    "Discrete": "#5b8ec8",
    "Category": "#8a5a9e",
    "Ordered category": "#a57bb8",
    "Boolean": "#8c6d1f",
    "Date / time": "#2f7d6d",
    "Free text": "#6b6259",
    "Identifier": "#77706a",
}
KIND_LABELS = {"warning": "Worth fixing", "notice": "Worth knowing", "link": "Relationship"}


def _facts(facts: list[Fact], css: str) -> str:
    items = "".join(
        f"<div><dt>{escape(f.label)}</dt><dd>{escape(f.value)}</dd></div>" for f in facts
    )
    return f'<dl class="{css}">{items}</dl>'


def _plain_facts(facts: list[Fact]) -> str:
    items = "".join(f"<dt>{escape(f.label)}</dt><dd>{escape(f.value)}</dd>" for f in facts)
    return f'<dl class="facts">{items}</dl>'


def _finding_card(finding: Finding) -> str:
    where = ""
    if finding.columns:
        names = ", ".join(escape(c) for c in finding.columns[:3])
        where = (
            f'<p class="where"><a href="#{finding.anchor}">→ {names}</a></p>'
            if finding.anchor
            else f'<p class="where muted">{names}</p>'
        )
    return (
        f'<article class="finding {finding.severity}">'
        f'<span class="kind">{KIND_LABELS[finding.severity]}</span>'
        f"<h3>{escape(finding.title)}</h3><p>{escape(finding.text)}</p>{where}</article>"
    )


def _types(report: Report) -> str:
    total = sum(count for _, count in report.types) or 1
    bar = "".join(
        f'<span style="width:{100 * count / total:.2f}%;'
        f'background:{TYPE_COLORS.get(label, "#77706a")}" title="{escape(label)}: {count}"></span>'
        for label, count in report.types
    )
    legend = "".join(
        f'<span><i style="background:{TYPE_COLORS.get(label, "#77706a")}"></i>'
        f"{escape(label)} · {count}</span>"
        for label, count in report.types
    )
    return f'<div class="typebar">{bar}</div><div class="legend">{legend}</div>'


def _glance(report: Report) -> str:
    rows = []
    for v in report.variables:
        warnings = sum(1 for f in v.findings if f.severity == "warning")
        notes = len(v.findings)
        flag = (
            f'<span class="pill warning">{warnings} to fix</span>'
            if warnings
            else (f'<span class="pill">{notes} to know</span>' if notes else "")
        )
        rows.append(
            f'<tr><td><a href="#{v.anchor}">{escape(v.name)}</a></td>'
            f'<td><span class="pill">{escape(v.type_label)}</span></td>'
            f"<td>{svg.thumbnail(v.chart)}</td>"
            f'<td><span class="meter"><b style="width:{min(v.missing_pct, 100):.1f}%"></b></span>'
            f"{v.missing_pct:.1f} %</td>"
            f'<td class="num">{v.distinct:,}</td><td>{flag}</td></tr>'
        )
    return (
        '<div class="scroll"><table class="glance"><thead><tr><th>Variable</th><th>Type</th>'
        '<th>Distribution</th><th>Missing</th><th class="num">Distinct</th><th></th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _left(card: VariableCard) -> str:
    parts = [_plain_facts(card.facts)]
    if card.quantiles:
        rows = "".join(
            f'<tr><td>{escape(q.label)}</td><td class="num">{escape(q.value)}</td></tr>'
            for q in card.quantiles
        )
        parts.append(f'<div class="mini"><h4>Percentiles</h4><table>{rows}</table></div>')
    if card.top:
        rows = "".join(
            f"<tr><td>{escape(svg.short(s.label, 22))}</td>"
            f'<td class="num">{s.count:,}</td><td class="num muted">{s.share * 100:.1f} %</td></tr>'
            for s in card.top
        )
        parts.append(f'<div class="mini"><h4>Most frequent</h4><table>{rows}</table></div>')
    return "".join(parts)


def _center(card: VariableCard) -> str:
    parts = []
    if card.chart is not None:
        parts.append(svg.chart_svg(card.chart, f"Distribution of {card.name}", card.profile))
    secondary = []
    if card.profile is not None:
        if box := svg.box_plot(card.profile, card.outliers):
            secondary.append(f"<figure><figcaption>Box plot</figcaption>{box}</figure>")
        if qq := svg.qq_plot(card.profile):
            secondary.append(f"<figure><figcaption>QQ-plot (normal)</figcaption>{qq}</figure>")
    for title, chart in card.extra:
        periods = [(b["label"], b["count"]) for b in chart.bars]
        drawn = svg.columns(periods, f"{card.name} by {title}", width=300, height=200)
        if drawn:
            secondary.append(f"<figure><figcaption>By {escape(title)}</figcaption>{drawn}</figure>")
    if secondary:
        parts.append(f'<div class="secondary">{"".join(secondary)}</div>')
    if not parts:
        parts.append('<p class="calm">No chart for this type: its values are not summarised.</p>')
    return "".join(parts)


def _right(card: VariableCard) -> str:
    parts = ["<h4>What stands out</h4>"]
    if card.findings:
        parts += [
            f'<div class="note {f.severity}"><b>{escape(f.title)}</b>{escape(f.text)}</div>'
            for f in card.findings
        ]
    else:
        parts.append(
            '<p class="calm">Nothing unusual: no missing values worth noting, no extreme skew '
            "or outliers.</p>"
        )
    if card.related:
        items = "".join(
            "<li>"
            + (f'<a href="#{r.anchor}">{escape(r.name)}</a>' if r.anchor else escape(r.name))
            + f'<span class="muted">{escape(r.measure)} {r.value:+.2f}</span></li>'
            for r in card.related
        )
        parts.append(
            f'<h4 style="margin-top:16px">Most related</h4><ul class="related">{items}</ul>'
        )
    return "".join(parts)


def _variable(card: VariableCard) -> str:
    same = card.role and card.role.lower() == card.type_label.lower()
    role = f'<span class="pill">{escape(card.role)}</span>' if card.role and not same else ""
    missing_css = "pill warning" if card.missing_pct >= 20 else "pill"
    header = (
        f"<header><h3>{escape(card.name)}</h3>"
        f'<span class="pill strong">{escape(card.type_label)}</span>{role}'
        f'<span class="stats"><span class="{missing_css}">{card.missing_pct:.1f} % missing</span>'
        f'<span class="pill">{card.distinct:,} distinct</span></span></header>'
    )
    return (
        f'<article class="variable" id="{card.anchor}">{header}<div class="vgrid">'
        f'<div>{_left(card)}</div><div class="center">{_center(card)}</div>'
        f'<div class="stand">{_right(card)}</div></div></article>'
    )


def _crosstab(view: PairView) -> str:
    table = view.comparison.contingency
    if table is None or not table.rows:
        return ""
    head = "".join(f'<th class="num">{escape(svg.short(c, 14))}</th>' for c in table.columns)
    body = []
    for label, counts in zip(table.rows, table.counts, strict=False):
        total = sum(counts) or 1
        cells = "".join(
            f'<td style="background: color-mix(in oklab, var(--chart-1) {60 * c / total:.0f}%, '
            f'transparent)" title="{c:,} rows">{100 * c / total:.0f} %</td>'
            for c in counts
        )
        body.append(
            f'<tr><th scope="row">{escape(svg.short(label, 18))}</th>{cells}'
            f'<td class="muted">{total:,}</td></tr>'
        )
    x, y = view.comparison.x, view.comparison.y
    return (
        f'<p class="small muted">Share of each {escape(y)} row, by {escape(x)}.</p>'
        f'<div class="scroll"><table class="crosstab"><thead><tr><th></th>{head}'
        f'<th class="num">n</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )


def _pair(view: PairView) -> str:
    c = view.comparison
    drawn = ""
    if c.kind == "scatter":
        if c.points is not None:
            drawn = svg.scatter(c.points, c.regression, c.x, c.y)
        elif c.density is not None:
            drawn = svg.density_map(c.density, c.x, c.y)
    elif c.kind == "groups":
        number, category = (c.y, c.x) if c.group_axis == "x" else (c.x, c.y)
        drawn = svg.group_boxes(c.groups, number, category)
    elif c.kind == "contingency":
        drawn = _crosstab(view)
    evidence = "pill strong" if view.evidence in ("strong", "moderate") else "pill"
    sampled = (
        f'<p class="small muted">Drawn on {c.sample:,} of {c.rows:,} rows.</p>'
        if c.sample and c.sample < c.rows
        else ""
    )
    return (
        f'<article class="pair"><header><h3>{escape(view.a)} × {escape(view.b)}</h3>'
        f'<span><span class="pill">{escape(view.measure)} = {view.value:.2f}</span> '
        f'<span class="{evidence}">{escape(view.evidence)} evidence</span></span></header>'
        f"{drawn or '<p class=calm>No chart for this pair.</p>'}{sampled}"
        f'<p class="verdict">{escape(view.verdict)}</p></article>'
    )


def _correlations(report: Report) -> str:
    if report.matrix is None:
        return '<p class="muted">Fewer than two columns can be compared.</p>'
    matrix = report.matrix
    note = f"Computed on {matrix.rows:,} rows"
    if matrix.sampled:
        note += f" (a reproducible sample of the {matrix.total:,})"
    if matrix.truncated:
        note += f"; {matrix.truncated} more columns left out"
    parts = [
        '<p class="lede">One measure per pair, chosen by the types: Pearson r between numbers, '
        "Cramér's V between categories, η between a number and a category. Blue: positive or "
        "strong link, red: negative. Columns are grouped so that related ones sit together.</p>",
        f'<p class="small muted">{escape(note)}.</p>',
        f'<div class="scroll">{svg.heatmap(matrix)}</div>',
    ]
    if report.pairs:
        rows = "".join(
            f"<tr><td>{escape(p.a)} × {escape(p.b)}</td><td>{escape(p.metric)}</td>"
            f'<td class="num">{p.value:.3f}</td>'
            f'<td class="num">{"—" if p.q_value is None else f"{p.q_value:.2g}"}</td>'
            f'<td><span class="{"pill strong" if p.evidence == "strong" else "pill"}">'
            f"{escape(p.evidence.capitalize())}</span></td></tr>"
            for p in report.pairs
        )
        parts.append(
            "<h3 style='margin-top:28px'>Strongest links</h3>"
            '<div class="scroll"><table><thead><tr><th>Pair</th><th>Measure</th>'
            '<th class="num">Value</th><th class="num">q-value</th><th>Evidence</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>"
        )
    if report.insights:
        parts.append(
            "<h3 style='margin-top:28px'>Insights</h3>"
            f'<div class="findings">{"".join(_finding_card(f) for f in report.insights)}</div>'
        )
    return "".join(parts)


SECTIONS = (
    ("findings", "Key findings"),
    ("glance", "At a glance"),
    ("variables", "Variables"),
    ("relationships", "Relationships"),
    ("correlations", "Correlations"),
    ("missing", "Missing values"),
    ("methods", "Methods"),
)


def render(report: Report) -> str:
    title = escape(report.title)
    filters = (
        '<p class="filters">Filtered: ' + escape(" · ".join(report.filters)) + "</p>"
        if report.filters
        else ""
    )
    toc = "".join(f'<a href="#{key}">{label}</a>' for key, label in SECTIONS)
    findings = (
        f'<div class="findings">{"".join(_finding_card(f) for f in report.key_findings)}</div>'
        if report.key_findings
        else '<p class="calm">Nothing stands out: no quality issue and no strong link.</p>'
    )
    excluded = (
        '<p class="small muted">Left out of the analysis: '
        f"{escape(', '.join(report.excluded))}.</p>"
        if report.excluded
        else ""
    )
    pairs = (
        f'<div class="pairs">{"".join(_pair(v) for v in report.pairs_views)}</div>'
        if report.pairs_views
        else '<p class="calm">No pair of columns is clearly linked.</p>'
    )
    if report.missing:
        missing = svg.missing_bars(report.missing)
        if report.complete:
            missing += (
                f'<p class="small muted">{report.complete:,} other columns have no missing '
                "value.</p>"
            )
    else:
        missing = '<p class="calm">No column has missing values.</p>'
    methods = "".join(
        f"<dt>{escape(m.label)}</dt><dd>{escape(m.value)}</dd>" for m in report.methods
    )
    after = " (after the filters)" if report.filters else ""
    meta = (
        f"<span>{escape(report.source)}</span><span>Generated {escape(report.generated_at)}</span>"
        f"<span>Quantilo {escape(report.version)}</span>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Data report</title>
<style>{CSS}</style>
</head>
<body>
<main>
<header class="masthead">
<p class="eyebrow">Data report</p>
<h1>{title}</h1>
<p class="meta">{meta}</p>
{filters}
{_facts(report.summary, "band")}
</header>
<nav class="toc">{toc}</nav>

<section id="findings">
<p class="eyebrow">{report.findings_total:,} findings in all</p>
<h2>Key findings</h2>
<p class="lede">What deserves attention first, ranked by how much it can change an analysis.
Each card links to the column it is about.</p>
{findings}
</section>

<section id="glance">
<h2>At a glance</h2>
<p class="lede">Every analysed column, with its shape, its missing values and how many distinct
values it holds.</p>
{_types(report)}
{_glance(report)}
{excluded}
</section>

<section id="variables">
<h2>Variables</h2>
<p class="lede">One card per column: its figures on the left, its distribution in the middle,
what stands out on the right.</p>
{"".join(_variable(c) for c in report.variables)}
</section>

<section id="relationships">
<h2>Relationships</h2>
<p class="lede">The most strongly linked pairs, drawn as the Compare page draws them, with the
recommended test and its effect size.</p>
{pairs}
</section>

<section id="correlations">
<h2>Correlations</h2>
{_correlations(report)}
</section>

<section id="missing">
<h2>Missing values</h2>
<p class="lede">Share of empty or unreadable values per column, most first.</p>
{missing}
</section>

<section id="methods">
<h2>Methods</h2>
<dl class="methods">{methods}</dl>
</section>

<footer>Every figure is computed on the {report.rows:,} rows above{after}. The report holds
counts, statistics and the most frequent values, never whole rows.</footer>
</main>
</body>
</html>
"""
