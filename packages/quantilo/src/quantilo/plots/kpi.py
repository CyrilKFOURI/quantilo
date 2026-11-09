"""The KPI builder's chart (frontend/src/modules/kpi: figures.ts, pivot.ts, format.ts): one
measure, in the chart the rules chose (or another), shown as values, shares or changes."""

from typing import Any

from quantilo.plots.theme import Tokens, with_alpha

Built = dict[str, Any]
OTHER = "Other"
MARKERS_MAX = 60  # R-VIS-34
BAR_LABELS_MAX = 12  # R-VIS-35
DIRECT_LABELS_MAX = 4
SHARE_MODES = ("share_total", "share_row", "share_column")
PERCENT_MODES = (*SHARE_MODES, "change_pct", "change_yoy")
DIVERGING_MODES = ("change", "change_pct", "change_yoy")
SIGNED_MODES = DIVERGING_MODES
ADDITIVE_MODES = (*SHARE_MODES, "running")
TIME_MODES = ("change", "change_pct", "change_yoy", "running")
MINUS = "−"


def series_index(result: dict[str, Any]):
    found = {(s["measure"], s["g1"], s["g2"]): s for s in result["series"]}

    def find(measure: str, g1: str | None = None, g2: str | None = None) -> dict[str, Any] | None:
        return found.get((measure, g1, g2))

    return find


def _ratio(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None or b == 0 else a / b


def _change(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None or b == 0 else (a - b) / abs(b)


def effective_mode(mode: str, additive: bool, time: bool) -> str:
    """A share or running total of a mean falls back to the value, as in Customize."""
    if mode in ADDITIVE_MODES and not additive:
        return "value"
    if mode in TIME_MODES and not time:
        return "value"
    return mode


def shown_values(result: dict[str, Any], s: dict[str, Any], mode: str) -> list[float | None]:
    if mode in SHARE_MODES:
        total = (series_index(result)(s["measure"]) or {}).get("values", [])
        return [_ratio(v, total[i] if i < len(total) else None) for i, v in enumerate(s["values"])]
    if mode == "change":
        return [
            None if v is None or p is None else v - p
            for v, p in zip(s["values"], s["previous"], strict=False)
        ]
    if mode == "change_pct":
        return [_change(v, p) for v, p in zip(s["values"], s["previous"], strict=False)]
    if mode == "change_yoy":
        return [_change(v, p) for v, p in zip(s["values"], s["last_year"], strict=False)]
    if mode == "running":
        total, out = 0.0, []
        for v in s["values"]:
            total += v or 0
            out.append(total)
        return out
    return s["values"]


def shown_overall(result: dict[str, Any], s: dict[str, Any], mode: str) -> float | None:
    if mode in SHARE_MODES:
        return _ratio(s["overall"], (series_index(result)(s["measure"]) or {}).get("overall"))
    if mode in DIVERGING_MODES:
        return None
    return s["overall"]


def _compact(value: float) -> str:
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= size:
            return f"{value / size:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{value:.1f}".rstrip("0").rstrip(".")


def format_value(
    value: float | None, fmt: dict[str, Any], compact: bool, signed: bool = False
) -> str:
    if value is None:
        return ""
    scaled = value * fmt.get("scale", 1)
    body = (
        _compact(abs(scaled))
        if compact and abs(scaled) >= 10_000
        else f"{abs(scaled):,.{fmt.get('decimals', 0)}f}"
    )
    sign = MINUS if scaled < 0 else "+" if signed and scaled > 0 else ""
    return f"{sign}{fmt.get('prefix', '')}{body}{fmt.get('suffix', '')}"


def format_shown(value: float | None, mode: str, fmt: dict[str, Any], compact: bool = True) -> str:
    if value is None:
        return ""
    if mode in SHARE_MODES:
        return f"{value * 100:.1f} %"
    if mode in ("change_pct", "change_yoy"):
        percent = value * 100
        sign = MINUS if percent < 0 else "+" if percent > 0 else ""
        return f"{sign}{abs(percent):.{1 if abs(percent) < 10 else 0}f} %"
    return format_value(value, fmt, compact, mode in SIGNED_MODES)


def level_color(levels: list[str], label: str, ordinal: bool, tokens: Tokens) -> str:
    """A label's colour by its position in the split; "Other" grey; ordinal: one hue."""
    if label == OTHER:
        return tokens.ink3
    real = [level for level in levels if level != OTHER]
    index = real.index(label) if label in real else 0
    if ordinal:
        step = index / (len(real) - 1) if len(real) > 1 else 1
        return with_alpha(tokens.palette[0], 0.5 + 0.5 * step)
    return tokens.palette[index % 8]


class _Input:
    def __init__(
        self,
        result: dict[str, Any],
        measure: dict[str, Any],
        chart: str,
        mode: str,
        labels: str,
        hide_other: bool,
    ) -> None:
        self.result = result
        self.measure = measure["id"]
        self.format = measure["format"]
        self.chart = chart
        self.mode = mode
        self.labels = labels
        self.hide_other = hide_other

    @property
    def factor(self) -> float:
        return 100 if self.mode in PERCENT_MODES else self.format.get("scale", 1)

    def affix(self) -> tuple[str, str]:
        if self.mode in PERCENT_MODES:
            return "", " %"
        return self.format.get("prefix", ""), self.format.get("suffix", "")

    def axis(self, values: list[float | None], percent: bool = False) -> dict[str, Any]:
        prefix, suffix = self.affix()
        largest = max([0.0, *(abs(v or 0) for v in values)])
        decimals = 0 if percent else min(self.format.get("decimals", 0), 2)
        return {
            "tickprefix": prefix,
            "ticksuffix": suffix,
            "tickformat": "~s" if largest >= 10_000 else f",.{decimals}f",
        }

    def scaled(self, values: list[float | None]) -> list[float | None]:
        return [None if v is None else v * self.factor for v in values]

    def text(self, values: list[float | None]) -> list[str]:
        return [format_shown(v, self.mode, self.format) for v in values]

    def visible(self, levels: list[str]) -> list[str]:
        return [level for level in levels if level != OTHER] if self.hide_other else levels


def _legend(tokens: Tokens, count: int) -> dict[str, Any]:
    if count >= 2:
        return {
            "showlegend": True,
            "legend": {
                "orientation": "h",
                "x": 0,
                "y": 1.12,
                "font": {"size": 11, "color": tokens.ink2},
            },
        }
    return {"showlegend": False}


def _heatmap_grid(
    data: _Input, tokens: Tokens, x: list[str], y: list[str], z: list[list[float | None]]
) -> Built:
    """Sequential one hue, or two hues around a grey 0 for changes (R-TAB-09)."""
    diverging = data.mode in DIVERGING_MODES
    blue, orange = tokens.palette[0], tokens.palette[1]
    scale = (
        [[0, orange], [0.5, tokens.line], [1, blue]]
        if diverging
        else [[0, with_alpha(blue, 0.06)], [1, blue]]
    )
    extent = max([0.0, *(abs(v or 0) for row in z for v in row)])
    trace: dict[str, Any] = {
        "type": "heatmap",
        "x": x,
        "y": y,
        "z": [data.scaled(row) for row in z],
        "customdata": [data.text(row) for row in z],
        "hovertemplate": "%{y} · %{x}: %{customdata}<extra></extra>",
        "colorscale": scale,
        "showscale": False,
        "xgap": 2,
        "ygap": 2,
    }
    if diverging:
        trace.update(zmid=0, zmin=-extent * data.factor, zmax=extent * data.factor)
    return {
        "data": [trace],
        "layout": {
            "yaxis": {"autorange": "reversed", "automargin": True, "showgrid": False},
            "xaxis": {"type": "category", "showgrid": False},
        },
    }


def kpi_figure(
    result: dict[str, Any],
    tokens: Tokens,
    measure: str | None = None,
    chart: str | None = None,
    mode: str = "value",
    labels: str = "auto",
    hide_other: bool = False,
    panel: str | None = None,
    any_chart: bool = False,
) -> Built:
    """One measure's chart (one panel of small multiples when `panel` is set). `chart`: the
    rules' choice by default; `mode`: value, share_total, change, change_pct, change_yoy,
    running."""
    measures = {m["id"]: m for m in result["measures"]}
    chosen = measures[measure] if measure else result["measures"][0]
    allowed = {o["value"] for o in chosen["chart"]["options"] if o.get("allowed", True)}
    drawn = chart if chart and (any_chart or chart in allowed) else chosen["chart"]["value"]
    if drawn == "none":
        return {"data": []}
    effective = effective_mode(mode, chosen["additive"], result["resolved"]["time"])
    data = _Input(result, chosen, drawn, effective, labels, hide_other)
    find = series_index(result)
    periods = result["periods"]
    x = [p["label"] for p in periods]
    splits = result["splits"]
    m = data.measure

    if drawn in ("hbar", "hbar_stacked") or (drawn == "heatmap" and not result["resolved"]["time"]):
        first = splits[0] if splits else None
        rows = data.visible(first["levels"] if first else [])
        if drawn == "hbar" or len(splits) < 2:
            values = []
            for g in rows:
                s = find(m, g)
                values.append(shown_overall(result, s, effective) if s else None)
            show = labels != "off" and (labels == "on" or len(rows) <= BAR_LABELS_MAX)
            return {
                "data": [
                    {
                        "type": "bar",
                        "orientation": "h",
                        "y": rows,
                        "x": data.scaled(values),
                        "marker": {
                            "color": [
                                tokens.ink3 if g == OTHER else tokens.palette[0] for g in rows
                            ],
                            "cornerradius": 4,
                        },
                        **({"text": data.text(values)} if show else {}),
                        "textposition": "outside",
                        "cliponaxis": False,
                        "customdata": data.text(values),
                        "hovertemplate": "%{y}: %{customdata}<extra></extra>",
                    }
                ],
                "layout": {
                    "yaxis": {"autorange": "reversed", "automargin": True},
                    "xaxis": {**data.axis(values), "showgrid": True, "rangemode": "tozero"},
                    "margin": {"l": 8, "r": 40, "t": 4, "b": 8},
                },
            }
        second = splits[1]
        columns = data.visible(second["levels"])
        if drawn == "heatmap":
            z = []
            for g1 in rows:
                row = []
                for g2 in columns:
                    s = find(m, g1, g2)
                    row.append(shown_overall(result, s, effective) if s else None)
                z.append(row)
            return _heatmap_grid(data, tokens, columns, rows, z)
        traces = []
        for g2 in columns:
            values = []
            for g1 in rows:
                s = find(m, g1, g2)
                values.append(shown_overall(result, s, effective) if s else None)
            traces.append(
                {
                    "type": "bar",
                    "orientation": "h",
                    "name": g2,
                    "y": rows,
                    "x": data.scaled(values),
                    "marker": {
                        "color": level_color(second["levels"], g2, second["ordinal"], tokens),
                        "line": {"color": tokens.surface, "width": 2},
                    },
                    "customdata": data.text(values),
                    "hovertemplate": f"{g2}: %{{customdata}}<extra>%{{y}}</extra>",
                }
            )
        return {
            "data": traces,
            "layout": {
                "barmode": "stack",
                "yaxis": {"autorange": "reversed", "automargin": True},
                "xaxis": {**data.axis([]), "showgrid": True},
                **_legend(tokens, len(columns)),
            },
        }

    # Over time: one trace per label of the split drawn (the second one inside a panel).
    split = splits[1] if panel is not None and len(splits) > 1 else (splits[0] if splits else None)
    lines: list[dict[str, Any]] = []
    if split:
        for label in data.visible(split["levels"]):
            s = find(m, panel, label) if panel else find(m, label)
            lines.append(
                {
                    "name": label,
                    "color": level_color(split["levels"], label, split["ordinal"], tokens),
                    "values": shown_values(result, s, effective) if s else [],
                }
            )
    else:
        s = find(m, panel)
        lines.append(
            {
                "name": "",
                "color": tokens.palette[0],
                "values": shown_values(result, s, effective) if s else [],
            }
        )

    if drawn == "heatmap":
        return _heatmap_grid(
            data, tokens, x, [line["name"] for line in lines], [line["values"] for line in lines]
        )

    every = [v for line in lines for v in line["values"]]
    partial = [p["partial"] for p in periods]
    many = len(lines) > 1
    bar_labels = labels == "on" or (
        labels == "auto" and not many and len(periods) <= BAR_LABELS_MAX
    )
    traces = []
    for i, line in enumerate(lines):
        text = data.text(line["values"])
        common: dict[str, Any] = {
            **({"name": line["name"]} if line["name"] else {}),
            "x": x,
            "y": data.scaled(line["values"]),
            "customdata": text,
            "hovertemplate": f"{line['name']}: %{{customdata}}<extra></extra>"
            if many
            else "%{customdata}<extra></extra>",
        }
        if drawn in ("bars", "stacked_bars", "grouped_bars"):
            stacked = drawn == "stacked_bars"
            traces.append(
                {
                    **common,
                    "type": "bar",
                    "marker": {
                        "color": line["color"],
                        "cornerradius": 0 if stacked else 4,
                        "line": {"color": tokens.surface, "width": 1 if stacked else 0},
                        # incomplete periods hatched (R-TIME-05)
                        "pattern": {
                            "shape": ["/" if p else "" for p in partial],
                            "fgcolor": tokens.surface,
                            "solidity": 0.4,
                        },
                    },
                    **({"text": text} if bar_labels else {}),
                    "textposition": "inside" if stacked else "outside",
                    "cliponaxis": False,
                }
            )
            continue
        area = drawn in ("area", "stacked_area")
        last = max((j for j, v in enumerate(line["values"]) if v is not None), default=-1)
        direct = labels != "off" and len(lines) <= DIRECT_LABELS_MAX and last >= 0
        trace: dict[str, Any] = {
            **common,
            "type": "scatter",
            "mode": "lines+markers" if len(periods) <= MARKERS_MAX else "lines",
            "line": {"color": line["color"], "width": 2},
            "marker": {
                "size": 6,
                "color": line["color"],
                "symbol": ["circle-open" if p else "circle" for p in partial],
            },
            "fill": ("tozeroy" if i == 0 else "tonexty")
            if drawn == "stacked_area"
            else ("tozeroy" if area else "none"),
            "textposition": "middle right",
            "textfont": {"color": tokens.ink2, "size": 11},
        }
        if area:
            trace["fillcolor"] = with_alpha(
                line["color"], 0.55 if drawn == "stacked_area" else 0.18
            )
        if drawn == "stacked_area":
            trace["stackgroup"] = "one"
        if direct:
            trace["text"] = [text[j] if j == last else "" for j in range(len(line["values"]))]
        traces.append(trace)
    zero = drawn != "line"  # R-VIS-31: bars and areas start at 0, lines fit the data
    layout: dict[str, Any] = {
        "hovermode": "x unified",
        "xaxis": {"type": "category", "showgrid": False, "automargin": True},
        "yaxis": {
            **data.axis(every, effective in PERCENT_MODES),
            "rangemode": "tozero" if zero and effective not in DIVERGING_MODES else "normal",
            "zeroline": effective in DIVERGING_MODES,
        },
        "margin": {"l": 8, "r": 56 if len(lines) <= DIRECT_LABELS_MAX else 8, "t": 4, "b": 8},
        **_legend(tokens, len([line for line in lines if line["name"]])),
    }
    if drawn == "stacked_bars":
        layout["barmode"] = "stack"
    elif drawn == "grouped_bars":
        layout["barmode"] = "group"
    return {"data": traces, "layout": layout}


def share_figure(result: dict[str, Any], tokens: Tokens, measure: str | None = None) -> Built:
    """The "% of total" chart drawn under the main one, same x axis (R-VIS-22)."""
    chart = "stacked_area" if len(result["periods"]) > 24 else "stacked_bars"
    return kpi_figure(result, tokens, measure, chart, "share_total", any_chart=True)
