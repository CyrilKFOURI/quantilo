"""SVG charts for the report (R-REP-06): no script, colours from the page's CSS tokens, graduated
axes, one native tooltip (<title>) per mark. Every text from the data is escaped (R-REP-12).

Each chart is drawn at its display size (viewBox = pixels) and shrinks with the page."""

import math
from html import escape

from quantilo.charts.prepare import Chart
from quantilo.compare.analyze import Density, Group, Points, Regression
from quantilo.correlations.matrix import Matrix
from quantilo.report.model import Missing
from quantilo.variable.analyze import VariableProfile
from quantilo.variable.outliers import OutlierRule

GAP = 2  # surface gap between adjacent fills
MAX_SCATTER = 1_500


def _n(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _svg(width: float, height: float, body: list[str], label: str, css: str = "chart") -> str:
    return (
        f'<svg class="{css}" viewBox="0 0 {_n(width)} {_n(height)}" '
        f'style="max-width:{_n(width)}px" role="img" aria-label="{escape(label)}">'
        + "".join(body)
        + "</svg>"
    )


def _rect(css: str, x: float, y: float, w: float, h: float, tip: str | None, rx: int = 0) -> str:
    title = f"<title>{escape(tip)}</title>" if tip else ""
    corner = f' rx="{rx}"' if rx else ""
    return (
        f'<rect class="{css}" x="{_n(x)}" y="{_n(y)}" width="{_n(max(w, 0))}" '
        f'height="{_n(max(h, 0))}"{corner}>{title}</rect>'
    )


def _line(css: str, x1: float, y1: float, x2: float, y2: float, tip: str | None = None) -> str:
    title = f"<title>{escape(tip)}</title>" if tip else ""
    return (
        f'<line class="{css}" x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}">'
        f"{title}</line>"
    )


def _text(x: float, y: float, text: str, anchor: str = "start", css: str = "") -> str:
    cls = f' class="{css}"' if css else ""
    return f'<text x="{_n(x)}" y="{_n(y)}" text-anchor="{anchor}"{cls}>{escape(text)}</text>'


def tick(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e6 or abs(value) < 1e-3:
        return f"{value:.2g}"
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.3g}" if abs(value) >= 1 else f"{value:.2g}"


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round tick values covering [low, high] (1, 2, 2.5, 5 × 10ⁿ steps)."""
    if not math.isfinite(low) or not math.isfinite(high):
        return []
    if high <= low:
        return [low]
    raw = (high - low) / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw)
    start = math.ceil(low / step - 1e-9) * step
    ticks, value = [], start
    while value <= high + step * 1e-9:
        ticks.append(round(value, 12))
        value += step
    return ticks


class Frame:
    """A plot area with linear scales and graduated axes."""

    def __init__(
        self,
        width: float,
        height: float,
        x: tuple[float, float],
        y: tuple[float, float],
        left: float = 44,
        bottom: float = 26,
        top: float = 10,
        right: float = 12,
    ) -> None:
        self.width, self.height = width, height
        self.left, self.right, self.top, self.bottom = left, right, top, bottom
        self.x0, self.x1 = x if x[1] > x[0] else (x[0] - 0.5, x[0] + 0.5)
        self.y0, self.y1 = y if y[1] > y[0] else (y[0] - 0.5, y[0] + 0.5)

    @property
    def plot_w(self) -> float:
        return self.width - self.left - self.right

    @property
    def plot_h(self) -> float:
        return self.height - self.top - self.bottom

    def px(self, value: float) -> float:
        return self.left + (value - self.x0) / (self.x1 - self.x0) * self.plot_w

    def py(self, value: float) -> float:
        return self.top + (1 - (value - self.y0) / (self.y1 - self.y0)) * self.plot_h

    def fit_line(self, slope: float, intercept: float, tip: str | None = None) -> str:
        """The line y = intercept + slope · x, cut to the plot area."""
        xs = [self.x0, self.x1]
        if slope:
            xs += [(self.y0 - intercept) / slope, (self.y1 - intercept) / slope]
        inside = sorted(
            x
            for x in xs
            if self.x0 - 1e-9 <= x <= self.x1 + 1e-9
            and self.y0 - 1e-9 <= intercept + slope * x <= self.y1 + 1e-9
        )
        if len(inside) < 2:
            return ""
        a, b = inside[0], inside[-1]
        return _line(
            "fit",
            self.px(a),
            self.py(intercept + slope * a),
            self.px(b),
            self.py(intercept + slope * b),
            tip,
        )

    def axes(self, x_ticks: bool = True, y_ticks: bool = True) -> list[str]:
        body = []
        base = self.top + self.plot_h
        if y_ticks:
            for value in nice_ticks(self.y0, self.y1, 5):
                y = self.py(value)
                body.append(
                    f'<line class="grid" x1="{_n(self.left)}" x2="{_n(self.width - self.right)}" '
                    f'y1="{_n(y)}" y2="{_n(y)}"/>'
                )
                body.append(_text(self.left - 6, y + 4, tick(value), "end", "tick"))
        if x_ticks:
            for value in nice_ticks(self.x0, self.x1, 6):
                x = self.px(value)
                body.append(_text(x, base + 16, tick(value), "middle", "tick"))
        body.append(
            f'<line class="axis" x1="{_n(self.left)}" x2="{_n(self.width - self.right)}" '
            f'y1="{_n(base)}" y2="{_n(base)}"/>'
        )
        return body


# --- One variable ---------------------------------------------------------------------------


def histogram(chart: Chart, label: str, profile: VariableProfile | None = None) -> str:
    """Bins, with the density curve and the mean and median marked (numbers)."""
    bins = chart.bins
    if not bins:
        return ""
    width, height = 640, 270
    top_count = max(b["count"] for b in bins) or 1
    frame = Frame(width, height, (bins[0]["start"], bins[-1]["end"]), (0, top_count * 1.05), top=36)
    body = frame.axes()
    for b in bins:
        x0, x1 = frame.px(b["start"]), frame.px(b["end"])
        y = frame.py(b["count"])
        tip = f"{tick(b['start'])} – {tick(b['end'])}: {b['count']:,} rows"
        body.append(_rect("bar", x0 + GAP / 2, y, x1 - x0 - GAP, frame.py(0) - y, tip, rx=2))
    numeric = profile.numeric if profile else None
    if numeric and numeric.kde and numeric.kde.x:
        width_bin = (bins[-1]["end"] - bins[0]["start"]) / len(bins)
        total = sum(b["count"] for b in bins)
        scale = total * width_bin
        points = " ".join(
            f"{_n(frame.px(x))},{_n(frame.py(min(y * scale, frame.y1)))}"
            for x, y in zip(numeric.kde.x, numeric.kde.y, strict=False)
            if frame.x0 <= x <= frame.x1
        )
        body.append(f'<polyline class="curve" points="{points}"><title>Density</title></polyline>')
    if numeric:
        d = numeric.describe
        marks = [("mean", d.mean, "mark-mean"), ("median", d.median, "mark-median")]
        for i, (name, value, css) in enumerate(marks):
            if value is None or not frame.x0 <= value <= frame.x1:
                continue
            x = frame.px(value)
            body.append(
                f'<line class="{css}" x1="{_n(x)}" x2="{_n(x)}" y1="{_n(16 + 14 * i)}" '
                f'y2="{_n(frame.py(0))}"><title>{name}: {escape(tick(value))}</title></line>'
            )
            # mean above, median below it, each on the side away from the other
            right = (d.mean or 0) >= (d.median or 0) if i == 0 else (d.median or 0) > (d.mean or 0)
            anchor, shift = ("start", 4) if right else ("end", -4)
            body.append(_text(x + shift, 12 + 14 * i, f"{name} {tick(value)}", anchor, css))
    return _svg(width, height, body, label)


def bars(chart: Chart, label: str, width: int = 640) -> str:
    """Categories as horizontal bars, most frequent first (or in their order), "Others" last."""
    rows: list[tuple[str, int, bool]] = [(b["label"], b["count"], False) for b in chart.bars]
    if chart.others:
        rows.append((f"Others ({chart.others['categories']:,})", chart.others["count"], True))
    if not rows:
        return ""
    total = sum(count for _, count, _ in rows) or 1
    top = max(count for _, count, _ in rows) or 1
    line = 26
    label_width, value_width = 150, 90
    room = width - label_width - value_width
    body = []
    for i, (name, count, others) in enumerate(rows):
        y = i * line
        w = room * count / top
        share = 100 * count / total
        tip = f"{name}: {count:,} rows ({share:.1f} %)"
        body.append(_text(label_width - 8, y + 17, short(name, 22), "end"))
        css = "bar others" if others else "bar"
        body.append(_rect(css, label_width, y + 4, max(w, 1), line - 8, tip, rx=3))
        body.append(_text(label_width + w + 8, y + 17, f"{count:,} · {share:.0f} %", css="value"))
    return _svg(width, line * len(rows), body, label)


def columns(periods: list[tuple[str, int]], label: str, width: int = 640, height: int = 240) -> str:
    """Rows per period (timeline, month, weekday) as columns, labels thinned to stay readable."""
    if not periods:
        return ""
    top = max(count for _, count in periods) or 1
    frame = Frame(width, height, (0, len(periods)), (0, top * 1.08), bottom=30)
    body = frame.axes(x_ticks=False)
    step = frame.plot_w / len(periods)
    gap = GAP if step > 5 else 0
    every = max(1, math.ceil(len(periods) / (frame.plot_w / 70)))
    for i, (name, count) in enumerate(periods):
        x = frame.left + i * step
        y = frame.py(count)
        body.append(
            _rect(
                "bar",
                x + gap / 2,
                y,
                max(step - gap, 0.6),
                frame.py(0) - y,
                f"{name}: {count:,} rows",
                rx=2 if step > 8 else 0,
            )
        )
        if i % every == 0:
            body.append(
                _text(
                    x + step / 2, frame.top + frame.plot_h + 18, short(name, 12), "middle", "tick"
                )
            )
    return _svg(width, height, body, label)


def chart_svg(chart: Chart, label: str, profile: VariableProfile | None = None) -> str:
    if chart.kind == "histogram":
        return histogram(chart, label, profile)
    if chart.kind == "timeline" or chart.view in ("month", "day_of_week", "hour"):
        return columns([(b["label"], b["count"]) for b in chart.bars], label)
    if chart.kind == "bars":
        return bars(chart, label)
    return ""


def box_plot(profile: VariableProfile, outliers: OutlierRule | None = None) -> str:
    """Q1, median, Q3, whiskers at Tukey's fences (clipped to the data), mean as a dot."""
    numeric = profile.numeric
    if numeric is None:
        return ""
    d = numeric.describe
    if None in (d.q1, d.median, d.q3, d.min, d.max):
        return ""
    assert d.q1 is not None and d.median is not None and d.q3 is not None
    assert d.min is not None and d.max is not None
    spread = d.q3 - d.q1
    low = max(d.min, d.q1 - 1.5 * spread)
    high = min(d.max, d.q3 + 1.5 * spread)
    width, height = 300, 120
    frame = Frame(width, height, (d.min, d.max), (0, 1), left=14, right=14, top=16, bottom=26)
    body = frame.axes(y_ticks=False)
    mid = frame.top + frame.plot_h / 2
    box_h = 34
    body.append(_line("whisker", frame.px(low), mid, frame.px(d.q1), mid))
    body.append(_line("whisker", frame.px(d.q3), mid, frame.px(high), mid))
    for end in (low, high):
        x = frame.px(end)
        body.append(_line("whisker", x, mid - 8, x, mid + 8))
    tip = f"Q1 {tick(d.q1)} · median {tick(d.median)} · Q3 {tick(d.q3)}"
    body.append(
        _rect(
            "box",
            frame.px(d.q1),
            mid - box_h / 2,
            frame.px(d.q3) - frame.px(d.q1),
            box_h,
            tip,
            rx=3,
        )
    )
    x = frame.px(d.median)
    body.append(_line("median", x, mid - box_h / 2, x, mid + box_h / 2))
    if d.mean is not None:
        body.append(
            f'<circle class="mean-dot" cx="{_n(frame.px(d.mean))}" cy="{_n(mid)}" r="3.5">'
            f"<title>mean {escape(tick(d.mean))}</title></circle>"
        )
    for side, end in (("low", d.min), ("high", d.max)):
        beyond = end < low if side == "low" else end > high
        if beyond:
            body.append(
                f'<circle class="outlier" cx="{_n(frame.px(end))}" cy="{_n(mid)}" r="3">'
                f"<title>{'min' if side == 'low' else 'max'} {escape(tick(end))}</title></circle>"
            )
    if outliers and outliers.count:
        body.append(_text(width - 14, 11, f"{outliers.count:,} outliers", "end", "tick"))
    return _svg(width, height, body, "Box plot")


def qq_plot(profile: VariableProfile) -> str:
    numeric = profile.numeric
    if numeric is None or numeric.qq is None or not numeric.qq.sample:
        return ""
    qq = numeric.qq
    width, height = 300, 230
    xs, ys = qq.theoretical, qq.sample
    frame = Frame(width, height, (min(xs), max(xs)), (min(ys), max(ys)), left=48, top=12)
    body = frame.axes()
    step = max(1, len(xs) // 300)
    for x, y in list(zip(xs, ys, strict=False))[::step]:
        body.append(f'<circle class="dot" cx="{_n(frame.px(x))}" cy="{_n(frame.py(y))}" r="2"/>')
    body.append(frame.fit_line(qq.slope, qq.intercept))
    body.append(_text(width - 12, height - 30, "normal quantiles →", "end", "tick"))
    return _svg(width, height, body, "QQ-plot against a normal distribution")


def thumbnail(chart: Chart | None) -> str:
    """A tiny distribution for the at-a-glance table."""
    if chart is None:
        return ""
    if chart.kind == "histogram":
        values = [b["count"] for b in chart.bins]
    elif chart.kind in ("bars", "timeline"):
        values = [b["count"] for b in chart.bars][:24]
    else:
        return ""
    if not values:
        return ""
    width, height = 120, 28
    top = max(values) or 1
    step = width / len(values)
    gap = 1 if step > 3 else 0
    body = [
        f'<rect class="bar" x="{_n(i * step + gap / 2)}" y="{_n(height - height * v / top)}" '
        f'width="{_n(max(step - gap, 0.5))}" height="{_n(height * v / top)}"/>'
        for i, v in enumerate(values)
    ]
    return _svg(width, height, body, "Distribution", css="thumb")


# --- Two variables (R-REP-15) ---------------------------------------------------------------


def scatter(points: Points, regression: Regression | None, x_name: str, y_name: str) -> str:
    if not points.x:
        return ""
    width, height = 520, 320
    step = max(1, len(points.x) // MAX_SCATTER)
    xs, ys = points.x[::step], points.y[::step]
    frame = Frame(width, height, (min(xs), max(xs)), (min(ys), max(ys)), left=52, bottom=40)
    body = frame.axes()
    for x, y in zip(xs, ys, strict=False):
        body.append(f'<circle class="dot" cx="{_n(frame.px(x))}" cy="{_n(frame.py(y))}" r="2.4"/>')
    if regression is not None:
        tip = f"y = {tick(regression.slope)} x + {tick(regression.intercept)}"
        if regression.r2 is not None:
            tip += f" · R² = {regression.r2:.2f}"
        body.append(frame.fit_line(regression.slope, regression.intercept, tip))
    body.append(_text(width - 12, height - 4, short(x_name, 40) + " →", "end", "axis-title"))
    body.append(_text(4, 10, "↑ " + short(y_name, 40), "start", "axis-title"))
    return _svg(width, height, body, f"{y_name} against {x_name}")


def density_map(density: Density, x_name: str, y_name: str) -> str:
    if not density.counts:
        return ""
    width, height = 520, 320
    frame = Frame(
        width,
        height,
        (density.x_edges[0], density.x_edges[-1]),
        (density.y_edges[0], density.y_edges[-1]),
        left=52,
        bottom=40,
    )
    body = frame.axes()
    top = max(max(row) for row in density.counts) or 1
    for j, row in enumerate(density.counts):
        for i, count in enumerate(row):
            if not count:
                continue
            x0, x1 = frame.px(density.x_edges[i]), frame.px(density.x_edges[i + 1])
            y0, y1 = frame.py(density.y_edges[j + 1]), frame.py(density.y_edges[j])
            share = 15 + 85 * math.sqrt(count / top)
            body.append(
                f'<rect x="{_n(x0)}" y="{_n(y0)}" width="{_n(x1 - x0)}" height="{_n(y1 - y0)}" '
                f'style="fill: color-mix(in oklab, var(--chart-1) {share:.0f}%, var(--surface-1))">'
                f"<title>{count:,} rows</title></rect>"
            )
    body.append(_text(width - 12, height - 4, short(x_name, 40) + " →", "end", "axis-title"))
    body.append(_text(4, 10, "↑ " + short(y_name, 40), "start", "axis-title"))
    return _svg(width, height, body, f"{y_name} against {x_name}")


def group_boxes(groups: list[Group], number: str, category: str) -> str:
    """One box per group (categories along the bottom, the number upwards)."""
    shown = [g for g in groups if g.q1 is not None and g.q3 is not None and g.median is not None]
    if not shown:
        return ""
    lows = [g.low if g.low is not None else g.q1 for g in shown]
    highs = [g.high if g.high is not None else g.q3 for g in shown]
    width, height = 520, 320
    frame = Frame(width, height, (0, len(shown)), (min(lows), max(highs)), left=56, bottom=44)  # type: ignore[type-var]
    body = frame.axes(x_ticks=False)
    slot = frame.plot_w / len(shown)
    box_w = min(56, slot * 0.6)
    for i, g in enumerate(shown):
        assert g.q1 is not None and g.q3 is not None and g.median is not None
        cx = frame.left + (i + 0.5) * slot
        low = g.low if g.low is not None else g.q1
        high = g.high if g.high is not None else g.q3
        body.append(_line("whisker", cx, frame.py(high), cx, frame.py(low)))
        tip = (
            f"{g.label}: n = {g.n:,} · median {tick(g.median)} · Q1 {tick(g.q1)} · Q3 {tick(g.q3)}"
            + (f" · mean {tick(g.mean)}" if g.mean is not None else "")
        )
        body.append(
            _rect(
                "box",
                cx - box_w / 2,
                frame.py(g.q3),
                box_w,
                frame.py(g.q1) - frame.py(g.q3),
                tip,
                rx=3,
            )
        )
        y = frame.py(g.median)
        body.append(_line("median", cx - box_w / 2, y, cx + box_w / 2, y))
        if g.mean is not None:
            body.append(
                f'<circle class="mean-dot" cx="{_n(cx)}" cy="{_n(frame.py(g.mean))}" r="3.5"/>'
            )
        body.append(_text(cx, frame.top + frame.plot_h + 16, short(g.label, 14), "middle", "tick"))
        body.append(_text(cx, frame.top + frame.plot_h + 30, f"n = {g.n:,}", "middle", "tick"))
    body.append(_text(4, 10, "↑ " + short(number, 40), "start", "axis-title"))
    body.append(_text(width - 12, height - 2, short(category, 40), "end", "axis-title"))
    return _svg(width, height, body, f"{number} by {category}")


# --- Correlations and missing values --------------------------------------------------------


def _cell_style(value: float | None) -> str:
    """Diverging: −1 red (chart-8), 0 the line colour, +1 blue (chart-1); 0..1 measures go blue."""
    if value is None:
        return "fill: var(--surface-2)"
    share = min(abs(value), 1.0) * 100
    pole = "var(--chart-1)" if value >= 0 else "var(--chart-8)"
    return f"fill: color-mix(in oklab, {pole} {share:.0f}%, var(--line))"


def heatmap(matrix: Matrix) -> str:
    order = matrix.clustered or list(range(len(matrix.columns)))
    names = [matrix.columns[i].name for i in order]
    n = len(names)
    cell = max(16, min(40, 640 // max(n, 1)))
    left = top = min(160, 12 + 7 * max(len(short(name, 20)) for name in names))
    size = cell * n
    body = []
    for row, i in enumerate(order):
        y = top + row * cell
        body.append(_text(left - 8, y + cell / 2 + 4, short(names[row], 20), "end"))
        for col, j in enumerate(order):
            value = matrix.values[i][j]
            metric = matrix.metrics[i][j]
            x = left + col * cell
            tip = f"{names[row]} × {names[col]}"
            if value is not None:
                tip += f": {value:.2f}" + (f" ({metric})" if metric else "")
            body.append(
                f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" rx="2" '
                f'style="{_cell_style(value)}"><title>{escape(tip)}</title></rect>'
            )
            if value is not None and cell >= 28 and i != j:
                css = "cell strong" if abs(value) >= 0.6 else "cell"
                body.append(
                    _text(x + cell / 2 - 1, y + cell / 2 + 3, f"{value:.2f}", "middle", css)
                )
    for col in range(n):
        x = left + col * cell + cell / 2
        body.append(
            f'<text x="{_n(x)}" y="{top - 8}" transform="rotate(-50 {_n(x)} {top - 8})">'
            f"{escape(short(names[col], 20))}</text>"
        )
    return _svg(left + size + 8, top + size + 8, body, "Association matrix")


def missing_bars(missing: list[Missing]) -> str:
    if not missing:
        return ""
    line = 24
    width, label_width, value_width = 720, 180, 70
    room = width - label_width - value_width
    body = []
    for i, m in enumerate(missing):
        y = i * line
        w = room * m.pct / 100
        body.append(_text(label_width - 8, y + 16, short(m.name, 26), "end"))
        body.append(_rect("track", label_width, y + 4, room, line - 8, None, rx=3))
        tip = f"{m.name}: {m.pct:.2f} % missing"
        body.append(_rect("bar warn", label_width, y + 4, max(w, 1), line - 8, tip, rx=3))
        body.append(_text(label_width + room + 8, y + 16, f"{m.pct:.1f} %", css="value"))
    return _svg(width, line * len(missing), body, "Missing values by column")
