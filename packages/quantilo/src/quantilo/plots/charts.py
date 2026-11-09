"""A column's chart, as on the Overview cards (frontend/src/charts/figures.ts): Top N bars,
pie, histogram, box, timeline. Each function takes the chart as plain data (`to_dict`)."""

from typing import Any

from quantilo.plots.theme import Tokens, count, format_number, with_alpha

HORIZONTAL_LABEL_LENGTH = 12
HORIZONTAL_BAR_COUNT = 8
MAX_LABEL = 22

Built = dict[str, Any]


def series_color(analysis_type: str, tokens: Tokens) -> str:
    if analysis_type in ("numeric_continuous", "numeric_discrete"):
        return tokens.numeric
    if analysis_type in ("categorical_nominal", "categorical_ordinal"):
        return tokens.categorical
    if analysis_type == "datetime":
        return tokens.date
    if analysis_type == "boolean":
        return tokens.bool
    return tokens.text


def _truncate(label: str) -> str:
    return f"{label[: MAX_LABEL - 1]}…" if len(label) > MAX_LABEL else label


def _share(value: int, total: int) -> str:
    if not total:
        return "0%"
    percent = 100 * value / total
    return "<1%" if 0 < percent < 1 else f"{round(percent)}%"


def category_bars(chart: dict[str, Any], include_missing: bool = True) -> list[dict[str, Any]]:
    """Top N bars plus "Others" and "(missing)", drawn muted and never mixed."""
    bars = [{"label": bar["label"], "count": bar["count"], "muted": False} for bar in chart["bars"]]
    others = chart.get("others")
    if others and others["count"] > 0:
        label = f"Others ({count(others['categories'])})"
        bars.append({"label": label, "count": others["count"], "muted": True})
    if include_missing and chart.get("missing", 0) > 0:
        bars.append({"label": "(missing)", "count": chart["missing"], "muted": True})
    return bars


def is_horizontal(bars: list[dict[str, Any]], ordered: bool) -> bool:
    long_label = any(len(bar["label"]) > HORIZONTAL_LABEL_LENGTH for bar in bars)
    if ordered and len(bars) <= 31:
        return long_label
    return len(bars) > HORIZONTAL_BAR_COUNT or long_label


def bars_figure(chart: dict[str, Any], analysis_type: str, tokens: Tokens) -> Built:
    bars = category_bars(chart)
    horizontal = is_horizontal(bars, chart.get("ordered", False))
    color = series_color(analysis_type, tokens)
    labels = [_truncate(bar["label"]) for bar in bars]
    counts = [bar["count"] for bar in bars]
    value = "x" if horizontal else "y"
    trace = {
        "type": "bar",
        "orientation": "h" if horizontal else "v",
        "x": counts if horizontal else labels,
        "y": labels if horizontal else counts,
        "customdata": [[bar["label"], _share(bar["count"], chart["rows"])] for bar in bars],
        "marker": {
            "color": [tokens.ink3 if bar["muted"] else color for bar in bars],
            "opacity": [0.55 if bar["muted"] else 1 for bar in bars],
        },
        "hovertemplate": f"%{{customdata[0]}}<br>%{{{value}:,}} rows · %{{customdata[1]}}"
        "<extra></extra>",
    }
    layout: dict[str, Any] = (
        {
            "yaxis": {"autorange": "reversed", "showgrid": False, "type": "category"},
            "xaxis": {"showgrid": True},
            "barcornerradius": 3,
        }
        if horizontal
        else {"xaxis": {"type": "category"}, "barcornerradius": 3}
    )
    return {"data": [trace], "layout": layout}


def pie_figure(chart: dict[str, Any], tokens: Tokens) -> Built:
    bars = category_bars(chart)
    return {
        "data": [
            {
                "type": "pie",
                "labels": [bar["label"] for bar in bars],
                "values": [bar["count"] for bar in bars],
                "hole": 0.55,
                "sort": False,
                "direction": "clockwise",
                "marker": {
                    "colors": [
                        tokens.ink3 if bar["muted"] else tokens.palette[i % 8]
                        for i, bar in enumerate(bars)
                    ],
                    "line": {"color": tokens.surface, "width": 2},
                },
                "textinfo": "label+percent",
                "textposition": "outside",
                "textfont": {"color": tokens.ink2, "size": 11},
                "hovertemplate": "%{label}<br>%{value:,} rows · %{percent}<extra></extra>",
            }
        ],
        "layout": {"margin": {"l": 24, "r": 24, "t": 16, "b": 16}},
    }


def histogram_figure(
    chart: dict[str, Any], analysis_type: str, tokens: Tokens, unit: str = "rows"
) -> Built:
    bins = chart["bins"]
    return {
        "data": [
            {
                "type": "bar",
                "x": [(b["start"] + b["end"]) / 2 for b in bins],
                "y": [b["count"] for b in bins],
                "width": [b["end"] - b["start"] for b in bins],
                "customdata": [[format_number(b["start"]), format_number(b["end"])] for b in bins],
                "marker": {
                    "color": series_color(analysis_type, tokens),
                    "line": {"color": tokens.surface, "width": 1},
                },
                "hovertemplate": f"%{{customdata[0]}} – %{{customdata[1]}}<br>%{{y:,}} {unit}"
                "<extra></extra>",
            }
        ],
        "layout": {"bargap": 0, "xaxis": {"showgrid": False}},
    }


def box_figure(chart: dict[str, Any], analysis_type: str, tokens: Tokens) -> Built:
    box = chart.get("box")
    if not box:
        return {"data": []}
    color = series_color(analysis_type, tokens)
    outliers = box.get("outlier_values", [])
    return {
        "data": [
            {
                "type": "box",
                "orientation": "h",
                "y": ["values"],
                "q1": [box["q1"]],
                "median": [box["median"]],
                "q3": [box["q3"]],
                "lowerfence": [box["lowerfence"]],
                "upperfence": [box["upperfence"]],
                "fillcolor": tokens.surface,
                "line": {"color": color, "width": 2},
                "hoverinfo": "x",
            },
            {
                "type": "scatter",
                "mode": "markers",
                "orientation": "h",
                "x": outliers,
                "y": ["values"] * len(outliers),
                "marker": {"color": color, "size": 6, "opacity": 0.6},
                "hovertemplate": "%{x:,}<extra>outlier</extra>",
            },
        ],
        "layout": {"yaxis": {"visible": False}, "xaxis": {"showgrid": True}},
    }


def timeline_figure(chart: dict[str, Any], tokens: Tokens) -> Built:
    points = chart["bars"]
    x = [p["label"] for p in points]
    y = [p["count"] for p in points]
    trace = (
        {
            "type": "scatter",
            "mode": "lines",
            "x": x,
            "y": y,
            "line": {"color": tokens.date, "width": 2},
            "fill": "tozeroy",
            "fillcolor": with_alpha(tokens.date, 0.15),
            "hovertemplate": "%{x}<br>%{y:,} rows<extra></extra>",
        }
        if len(points) > 60
        else {
            "type": "bar",
            "x": x,
            "y": y,
            "marker": {"color": tokens.date},
            "hovertemplate": "%{x}<br>%{y:,} rows<extra></extra>",
        }
    )
    return {
        "data": [trace],
        "layout": {"xaxis": {"type": "date", "nticks": 6, "tickangle": 0}, "barcornerradius": 2},
    }


def chart_figure(chart: dict[str, Any], analysis_type: str, tokens: Tokens) -> Built | None:
    """The card's figure for the chart's kind and view (as the Overview's ChartView)."""
    kind, view = chart["kind"], chart.get("view")
    if kind == "histogram":
        if view == "box":
            return box_figure(chart, analysis_type, tokens)
        lengths = view == "lengths"
        return histogram_figure(
            chart, "text" if lengths else analysis_type, tokens, "texts" if lengths else "rows"
        )
    if kind == "timeline":
        return timeline_figure(chart, tokens)
    if kind == "bars":
        if view == "pie":
            return pie_figure(chart, tokens)
        return bars_figure(chart, analysis_type, tokens)
    return None  # identifiers and free text: no chart, as in the app
