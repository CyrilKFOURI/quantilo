"""The Segments page's figures (frontend/src/modules/segments/figures.ts): the map, the scores that
chose k (small multiples, never two y-scales), stability, Ward's tree, and the unusual-rows score
curve. Segments take the categorical palette in their fixed order; "Not grouped" is muted grey."""

from typing import Any

from quantilo.plots.theme import LEGEND, Tokens, axis_title, with_alpha

Built = dict[str, Any]
SCORES = [
    ("silhouette", "Silhouette (higher is better)"),
    ("calinski_harabasz", "Calinski–Harabasz (higher is better)"),
    ("davies_bouldin", "Davies–Bouldin (lower is better)"),
    ("inertia", "Inertia (look for the elbow)"),
]


def segment_color(segment: int, tokens: Tokens) -> str:
    if segment < 0:
        return tokens.ink3
    return tokens.palette[segment % len(tokens.palette)]


def map_figure(shown: dict[str, Any], names: dict[int, str], tokens: Tokens) -> Built:
    """One trace per segment (legend + direct labels at the centres, so colour is never alone)."""
    traces: list[Built] = []
    segments = sorted({int(s) for s in shown["segments"]}, key=lambda s: (s < 0, s))
    for segment in segments:
        index = [i for i, s in enumerate(shown["segments"]) if s == segment]
        traces.append(
            {
                "type": "scattergl",
                "mode": "markers",
                "name": names.get(segment, str(segment)),
                "x": [shown["x"][i] for i in index],
                "y": [shown["y"][i] for i in index],
                "customdata": [shown["rows"][i] for i in index],
                "marker": {
                    "color": segment_color(segment, tokens),
                    "size": 6,
                    "opacity": 0.35 if segment < 0 else 0.6,
                },
                "hovertemplate": f"{names.get(segment, segment)}<br>row %{{customdata}}"
                "<extra></extra>",
            }
        )
    centres = [c for c in shown["centres"] if c["segment"] >= 0]
    traces.append(
        {
            "type": "scatter",
            "mode": "markers+text",
            "name": "Centres",
            "showlegend": False,
            "x": [c["x"] for c in centres],
            "y": [c["y"] for c in centres],
            "text": [names.get(c["segment"], str(c["segment"])) for c in centres],
            "textposition": "top center",
            "textfont": {"color": tokens.ink1, "size": 12},
            "marker": {
                "color": [segment_color(c["segment"], tokens) for c in centres],
                "size": 12,
                "symbol": "diamond",
                "line": {"color": tokens.surface, "width": 2},
            },
            "hoverinfo": "skip",
        }
    )
    explained = shown.get("explained")
    if shown["projection"] == "pca" and explained:

        def share(i: int) -> str:
            return f"{round((explained[i] if i < len(explained) else 0) * 100)} %"

        x_title, y_title = f"Component 1 ({share(0)})", f"Component 2 ({share(1)})"
    else:
        x_title, y_title = "t-SNE 1", "t-SNE 2"
    return {
        "data": traces,
        "layout": {
            **LEGEND,
            "xaxis": {**axis_title(x_title), "zeroline": False},
            "yaxis": {**axis_title(y_title), "zeroline": False},
        },
    }


def scores_figure(scores: list[dict[str, Any]], chosen: int, tokens: Tokens) -> Built:
    """Four small charts side by side, k on each x axis, the chosen k marked on each."""
    shown = [(key, label) for key, label in SCORES if any(s.get(key) is not None for s in scores)]
    if any(s.get("bic") is not None for s in scores):
        shown.append(("bic", "BIC (lower is better)"))
    width = 1 / max(1, len(shown))
    traces: list[Built] = []
    layout: Built = {"showlegend": False, "annotations": [], "shapes": []}
    for i, (key, label) in enumerate(shown):
        axis = "" if i == 0 else str(i + 1)
        points = [(s["k"], s[key]) for s in scores if s.get(key) is not None]
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": [k for k, _ in points],
                "y": [v for _, v in points],
                "xaxis": f"x{axis}",
                "yaxis": f"y{axis}",
                "line": {"color": tokens.palette[0], "width": 2},
                "marker": {
                    "size": [10 if k == chosen else 7 for k, _ in points],
                    "color": [
                        tokens.accent if k == chosen else tokens.palette[0] for k, _ in points
                    ],
                    "line": {"color": tokens.surface, "width": 2},
                },
                "hovertemplate": f"k = %{{x}}<br>{label.split(' (')[0]} %{{y:.4g}}<extra></extra>",
            }
        )
        domain = [i * width + 0.04, (i + 1) * width - 0.04]
        layout[f"xaxis{axis}"] = {"domain": domain, "dtick": 1, "anchor": f"y{axis}"}
        layout[f"yaxis{axis}"] = {"anchor": f"x{axis}", "showgrid": True}
        layout["annotations"].append(
            {
                "text": label,
                "x": (domain[0] + domain[1]) / 2,
                "y": 1.12,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 11, "color": tokens.ink2},
            }
        )
        layout["shapes"].append(
            {
                "type": "line",
                "xref": f"x{axis}",
                "yref": f"y{axis} domain",
                "x0": chosen,
                "x1": chosen,
                "y0": 0,
                "y1": 1,
                "line": {"color": tokens.ink3, "width": 1, "dash": "dot"},
            }
        )
    layout["margin"] = {"t": 28}
    return {"data": traces, "layout": layout}


def stability_figure(stability: dict[str, Any], tokens: Tokens) -> Built:
    runs = stability["runs"]
    return {
        "data": [
            {
                "type": "bar",
                "x": [f"#{i + 1}" for i in range(len(runs))],
                "y": runs,
                "marker": {"color": with_alpha(tokens.palette[0], 0.8)},
                "hovertemplate": "subsample %{x}<br>adjusted Rand %{y:.2f}<extra></extra>",
            }
        ],
        "layout": {
            "yaxis": {**axis_title("Agreement (adjusted Rand)"), "range": [0, 1.05]},
            "xaxis": axis_title("Subsample (80 % of the rows)"),
            "barcornerradius": 3,
            "shapes": [
                {
                    "type": "line",
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "y0": 0.6,
                    "y1": 0.6,
                    "line": {"color": tokens.ink3, "width": 1, "dash": "dash"},
                }
            ],
        },
    }


def dendrogram_figure(tree: dict[str, Any], tokens: Tokens) -> Built:
    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": ys,
            "line": {"color": tokens.ink2, "width": 1.5},
            "hoverinfo": "skip",
        }
        for xs, ys in zip(tree["icoord"], tree["dcoord"], strict=True)
    ]
    ticks = [5 + 10 * i for i in range(len(tree["leaves"]))]
    return {
        "data": traces,
        "layout": {
            "xaxis": {
                "tickvals": ticks,
                "ticktext": tree["leaves"],
                "tickangle": -60,
                "tickfont": {"size": 9},
            },
            "yaxis": axis_title("Merge height (Ward)"),
            "shapes": [
                {
                    "type": "line",
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "y0": tree["cut"],
                    "y1": tree["cut"],
                    "line": {"color": tokens.accent, "width": 1.5, "dash": "dash"},
                }
            ],
        },
    }


def unusual_figure(found: dict[str, Any], threshold: float | None, tokens: Tokens) -> Built:
    """Scores sorted from the most unusual row down; the threshold as a line, the rows kept
    above it in the accent."""
    curve = found["curve"]
    cut = found["threshold"]["value"] if threshold is None else threshold
    ranks, scores = curve["ranks"], curve["scores"]
    above = [i for i, s in enumerate(scores) if s >= cut]
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": ranks,
                "y": scores,
                "line": {"color": tokens.palette[0], "width": 2},
                "hovertemplate": "rank %{x:,}<br>score %{y:.4g}<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "markers",
                "x": [ranks[i] for i in above],
                "y": [scores[i] for i in above],
                "marker": {"color": tokens.accent, "size": 7},
                "hoverinfo": "skip",
            },
        ],
        "layout": {
            "showlegend": False,
            "xaxis": {**axis_title("Rows, most unusual first"), "type": "log"},
            "yaxis": axis_title("Unusual score"),
            "shapes": [
                {
                    "type": "line",
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "y0": cut,
                    "y1": cut,
                    "line": {"color": tokens.accent, "width": 1.5, "dash": "dash"},
                }
            ],
        },
    }
