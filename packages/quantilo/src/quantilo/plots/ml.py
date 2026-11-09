"""The Machine learning page's figures (frontend/src/modules/ml/figures.ts and Results.tsx)."""

from typing import Any

from quantilo.plots.theme import Tokens, axis_title, with_alpha

Built = dict[str, Any]


def predicted_figure(regression: dict[str, Any], target: str, tokens: Tokens) -> Built:
    """Predicted against actual, with the line of perfect predictions."""
    values = [*regression["actual"], *regression["predicted"]]
    low, high = (min(values), max(values)) if values else (0, 1)
    return {
        "data": [
            {
                "type": "scattergl",
                "mode": "markers",
                "x": regression["actual"],
                "y": regression["predicted"],
                "marker": {"color": tokens.palette[0], "size": 5, "opacity": 0.55},
                "hovertemplate": "actual %{x:,.4g}<br>predicted %{y:,.4g}<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": [low, high],
                "y": [low, high],
                "line": {"color": tokens.ink3, "dash": "dash", "width": 1},
                "hoverinfo": "skip",
            },
        ],
        "layout": {
            "showlegend": False,
            "xaxis": axis_title(f"Actual {target}"),
            "yaxis": axis_title(f"Predicted {target}"),
        },
    }


def residuals_figure(regression: dict[str, Any], tokens: Tokens) -> Built:
    """Residuals against predicted values: a funnel means the error grows with the value."""
    actual, predicted = regression["actual"], regression["predicted"]
    return {
        "data": [
            {
                "type": "scattergl",
                "mode": "markers",
                "x": predicted,
                "y": [a - p for a, p in zip(actual, predicted, strict=False)],
                "marker": {"color": tokens.palette[0], "size": 5, "opacity": 0.55},
                "hovertemplate": "predicted %{x:,.4g}<br>residual %{y:,.4g}<extra></extra>",
            }
        ],
        "layout": {
            "showlegend": False,
            "xaxis": axis_title("Predicted"),
            "yaxis": {
                **axis_title("Actual − predicted"),
                "zeroline": True,
                "zerolinecolor": tokens.ink3,
            },
        },
    }


def residual_histogram(regression: dict[str, Any], tokens: Tokens) -> Built:
    bins = regression["residual_bins"]
    return {
        "data": [
            {
                "type": "bar",
                "x": [((b.get("start") or 0) + (b.get("end") or 0)) / 2 for b in bins],
                "y": [b["count"] for b in bins],
                "width": [(b.get("end") or 0) - (b.get("start") or 0) for b in bins],
                "marker": {
                    "color": with_alpha(tokens.ink3, 0.5),
                    "line": {"color": tokens.surface, "width": 1},
                },
                "hovertemplate": "%{x:,.4g}: %{y} rows<extra></extra>",
            }
        ],
        "layout": {
            "showlegend": False,
            "bargap": 0,
            "xaxis": axis_title("Residual"),
            "yaxis": axis_title("Rows"),
        },
    }


def confusion_figure(
    classification: dict[str, Any], tokens: Tokens, percent: bool = False
) -> Built:
    """Confusion matrix: counts, or % of each actual class (row)."""
    labels, matrix = classification["labels"], classification["confusion"]
    values = []
    for row in matrix:
        total = sum(row) or 1
        values.append([v / total * 100 if percent else v for v in row])
    return {
        "data": [
            {
                "type": "heatmap",
                "x": labels,
                "y": labels,
                "z": values,
                "text": [[f"{v:.0f}%" if percent else str(v) for v in row] for row in values],
                "texttemplate": "%{text}",
                "colorscale": [[0, tokens.surface], [1, tokens.palette[0]]],
                "showscale": False,
                "hovertemplate": "actual %{y}<br>predicted %{x}<br>%{text}<extra></extra>",
            }
        ],
        "layout": {
            "xaxis": {**axis_title("Predicted"), "type": "category"},
            "yaxis": {**axis_title("Actual"), "type": "category", "autorange": "reversed"},
        },
    }


def curve_figure(
    x: list[float],
    y: list[float],
    x_title: str,
    y_title: str,
    reference: str | None,
    tokens: Tokens,
    flat: float = 0,
) -> Built:
    data: list[Built] = [
        {
            "type": "scatter",
            "mode": "lines+markers",
            "x": x,
            "y": y,
            "line": {"color": tokens.palette[0], "width": 2},
            "marker": {"size": 3},
            "hovertemplate": f"{x_title} %{{x:.3f}}<br>{y_title} %{{y:.3f}}<extra></extra>",
        }
    ]
    if reference:
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": [0, 1],
                "y": [0, 1] if reference == "diagonal" else [flat, flat],
                "line": {"color": tokens.ink3, "dash": "dash", "width": 1},
                "hoverinfo": "skip",
            }
        )
    return {
        "data": data,
        "layout": {
            "showlegend": False,
            "xaxis": {**axis_title(x_title), "range": [0, 1]},
            "yaxis": {**axis_title(y_title), "range": [0, 1.02]},
        },
    }


def class_curves_figure(
    curves: list[dict[str, Any]], x_title: str, y_title: str, diagonal: bool, tokens: Tokens
) -> Built:
    """One curve per class against the others (multiclass ROC, precision-recall, calibration)."""
    data: list[Built] = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": curve["name"],
            "x": curve["x"],
            "y": curve["y"],
            "line": {"color": tokens.palette[i % 8], "width": 2},
            "hovertemplate": f"{curve['name']}<br>{x_title} %{{x:.3f}}<br>{y_title} %{{y:.3f}}"
            "<extra></extra>",
        }
        for i, curve in enumerate(curves)
    ]
    if diagonal:
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": [0, 1],
                "y": [0, 1],
                "line": {"color": tokens.ink3, "dash": "dash", "width": 1},
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )
    return {
        "data": data,
        "layout": {
            "showlegend": True,
            "legend": {
                "orientation": "h",
                "x": 0,
                "y": -0.28,
                "font": {"size": 11, "color": tokens.ink2},
            },
            "margin": {"l": 8, "r": 8, "t": 4, "b": 8},
            "xaxis": {**axis_title(x_title), "range": [0, 1]},
            "yaxis": {**axis_title(y_title), "range": [0, 1.02]},
        },
    }


def roc_figure(classification: dict[str, Any], tokens: Tokens) -> Built:
    curves = classification.get("class_curves") or []
    if curves:
        named = [
            {
                "name": k["label"] if k["auc"] is None else f"{k['label']} · AUC {k['auc']:.2f}",
                "x": k["roc"]["x"],
                "y": k["roc"]["y"],
            }
            for k in curves
        ]
        return class_curves_figure(named, "False positive rate", "True positive rate", True, tokens)
    roc = classification.get("roc")
    if not roc:
        return {"data": []}
    return curve_figure(
        roc["x"], roc["y"], "False positive rate", "True positive rate", "diagonal", tokens
    )


def pr_figure(classification: dict[str, Any], tokens: Tokens) -> Built:
    curves = classification.get("class_curves") or []
    if curves:
        named = [
            {
                "name": f"{k['label']} · {k['share'] * 100:.0f} % of rows",
                "x": k["pr"]["x"],
                "y": k["pr"]["y"],
            }
            for k in curves
        ]
        return class_curves_figure(named, "Recall", "Precision", False, tokens)
    pr = classification.get("pr")
    if not pr:
        return {"data": []}
    truth = classification.get("truth") or []
    chance = sum(1 for v in truth if v == 1) / (len(truth) or 1)
    return curve_figure(pr["x"], pr["y"], "Recall", "Precision", "flat", tokens, chance)


def calibration_figure(classification: dict[str, Any], tokens: Tokens) -> Built:
    curves = [k for k in classification.get("class_curves") or [] if k.get("calibration")]
    if curves:
        named = [
            {"name": k["label"], "x": k["calibration"]["x"], "y": k["calibration"]["y"]}
            for k in curves
        ]
        return class_curves_figure(named, "Predicted probability", "Observed share", True, tokens)
    calibration = classification.get("calibration")
    if not calibration:
        return {"data": []}
    return curve_figure(
        calibration["x"],
        calibration["y"],
        "Predicted probability",
        "Observed share",
        "diagonal",
        tokens,
    )


def coefficients_figure(coefficients: list[dict[str, Any]], tokens: Tokens) -> Built:
    """The largest coefficients, with their 95 % interval when there is one."""
    top = list(reversed(coefficients[:15]))
    intervals = top and all(c.get("low") is not None and c.get("high") is not None for c in top)
    return {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "y": [c["feature"] for c in top],
                "x": [c["estimate"] for c in top],
                **(
                    {
                        "error_x": {
                            "type": "data",
                            "symmetric": False,
                            "array": [c["high"] - c["estimate"] for c in top],
                            "arrayminus": [c["estimate"] - c["low"] for c in top],
                            "color": tokens.ink3,
                        }
                    }
                    if intervals
                    else {}
                ),
                "marker": {
                    "color": [
                        tokens.palette[0] if c["estimate"] >= 0 else tokens.palette[1] for c in top
                    ]
                },
                "hovertemplate": "%{y}: %{x:,.4g}<extra></extra>",
            }
        ],
        "layout": {
            "showlegend": False,
            "margin": {"l": 160},
            "xaxis": {**axis_title("Coefficient"), "zeroline": True, "zerolinecolor": tokens.ink3},
        },
    }


def importances_figure(importances: list[dict[str, Any]], tokens: Tokens) -> Built:
    top = list(reversed(importances[:15]))
    return {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "y": [i["feature"] for i in top],
                "x": [i["importance"] for i in top],
                "marker": {"color": tokens.palette[0]},
                "hovertemplate": "%{y}: %{x:.1%}<extra></extra>",
            }
        ],
        "layout": {
            "showlegend": False,
            "margin": {"l": 160},
            "xaxis": {**axis_title("Share of importance"), "tickformat": ".0%"},
        },
    }


def learning_curve_figure(curve: dict[str, Any], tokens: Tokens) -> Built:
    """Training and held-out scores iteration after iteration; the iteration kept is marked."""
    data: list[Built] = []
    if curve.get("train"):
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": "training",
                "x": curve["iterations"],
                "y": curve["train"],
                "line": {"color": tokens.palette[0], "width": 2},
            }
        )
    if curve.get("validation"):
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": "held out",
                "x": curve["iterations"],
                "y": curve["validation"],
                "line": {"color": tokens.palette[1], "width": 2},
            }
        )
    shapes = (
        [
            {
                "type": "line",
                "x0": curve["best"],
                "x1": curve["best"],
                "yref": "paper",
                "y0": 0,
                "y1": 1,
                "line": {"color": tokens.ink3, "dash": "dot", "width": 1},
            }
        ]
        if curve.get("best") is not None
        else []
    )
    return {
        "data": data,
        "layout": {
            "showlegend": len(data) > 1,
            "shapes": shapes,
            "xaxis": axis_title("Iteration"),
            "yaxis": axis_title(curve["label"]),
        },
    }


# --- Panel data (035) ---------------------------------------------------------------------------


def _good(panel: dict[str, Any], value: float | None) -> bool:
    summary = panel["summary"]
    threshold = summary.get("threshold")
    if value is None or threshold is None:
        return True
    return value < threshold if summary.get("lower_is_better") else value > threshold


def period_scores_figure(panel: dict[str, Any], tokens: Tokens) -> Built:
    """The score of every test period; periods on the wrong side of the threshold are muted."""
    periods = panel["periods"]
    summary = panel["summary"]
    shapes = (
        [
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "y0": summary["threshold"],
                "y1": summary["threshold"],
                "line": {"color": tokens.ink3, "dash": "dash", "width": 1},
            }
        ]
        if summary.get("threshold") is not None
        else []
    )
    return {
        "data": [
            {
                "type": "bar",
                "x": [p["period"] for p in periods],
                "y": [p["score"] for p in periods],
                "customdata": [p["fold"] for p in periods],
                "marker": {
                    "color": [
                        tokens.palette[0]
                        if _good(panel, p["score"])
                        else with_alpha(tokens.ink3, 0.55)
                        for p in periods
                    ]
                },
                "hovertemplate": "%{x}: %{y:,.4g} (fold %{customdata})<extra></extra>",
            }
        ],
        "layout": {
            "showlegend": False,
            "bargap": 0.15,
            "shapes": shapes,
            "xaxis": {**axis_title(panel["period_column"]), "type": "category"},
            "yaxis": axis_title(summary["metric"]),
        },
    }


def cumulative_figure(panel: dict[str, Any], tokens: Tokens) -> Built:
    """The scores of the periods added up in time: a steady climb is a regular model."""
    periods = panel["periods"]
    total, running = 0.0, []
    for p in periods:
        total += p["score"] or 0.0
        running.append(total)
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": [p["period"] for p in periods],
                "y": running,
                "line": {"color": tokens.palette[0], "width": 2},
                "hovertemplate": "%{x}: %{y:,.4g}<extra></extra>",
            }
        ],
        "layout": {
            "showlegend": False,
            "xaxis": {**axis_title(panel["period_column"]), "type": "category"},
            "yaxis": {**axis_title(f"Cumulative {panel['summary']['metric']}"), "zeroline": True},
        },
    }


def folds_figure(panel: dict[str, Any], tokens: Tokens) -> Built:
    """Each fold on the time axis: the periods it trained on, the embargo, the periods tested."""
    folds = panel["folds"]
    names = [f"Fold {f['fold']}" for f in folds]
    parts = [
        ("trained on", tokens.palette[0], "train_start", "train_end", "train"),
        ("embargo", with_alpha(tokens.ink3, 0.35), None, None, "gap"),
        ("tested on", tokens.palette[1], "test_start", "test_end", "test"),
    ]
    data = []
    for label, colour, start, end, kind in parts:
        if kind == "gap":
            bases = [f["train_end"] + 1 for f in folds]
            widths = [f["test_start"] - f["train_end"] - 1 for f in folds]
            hover = [f"{w} period{'s' if w != 1 else ''} left out" for w in widths]
        else:
            bases = [f[str(start)] for f in folds]
            widths = [f[str(end)] - f[str(start)] + 1 for f in folds]
            hover = [
                f"{f[kind + '_from']} to {f[kind + '_to']} · {f[kind + '_periods']} periods, "
                f"{f[kind + '_rows']:,} rows"
                for f in folds
            ]
        data.append(
            {
                "type": "bar",
                "orientation": "h",
                "name": label,
                "y": names,
                "x": widths,
                "base": bases,
                "marker": {"color": colour, "line": {"color": tokens.surface, "width": 2}},
                "text": hover,
                "hovertemplate": "%{y} · " + label + ": %{text}<extra></extra>",
                "textposition": "none",
            }
        )
    ticks = {p["rank"]: p["period"] for p in panel["periods"]}
    for f in folds:
        ticks.setdefault(f["train_start"], f["train_from"])
    chosen = sorted(ticks)[:: max(1, len(ticks) // 6)]
    return {
        "data": data,
        "layout": {
            "barmode": "overlay",
            "showlegend": True,
            "legend": {
                "orientation": "h",
                "x": 0,
                "y": 1.02,
                "yanchor": "bottom",
                "entrywidth": 96,
                "font": {"size": 11, "color": tokens.ink2},
            },
            "yaxis": {"autorange": "reversed"},
            "xaxis": {
                **axis_title(panel["period_column"]),
                "tickvals": chosen,
                "ticktext": [ticks[r] for r in chosen],
            },
        },
    }


# --- Explanations --------------------------------------------------------------------------------


def permutation_figure(permutation: dict[str, Any], tokens: Tokens) -> Built:
    """How much the score drops when each column is shuffled (mean ± sd)."""
    top = list(reversed(permutation["items"][:15]))
    return {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "y": [i["column"] for i in top],
                "x": [i["mean"] for i in top],
                "error_x": {"type": "data", "array": [i["std"] for i in top], "color": tokens.ink3},
                "marker": {
                    "color": [tokens.palette[0] if i["mean"] > 0 else tokens.ink3 for i in top]
                },
                "hovertemplate": "%{y}: %{x:.4g}<extra></extra>",
            }
        ],
        "layout": {
            "showlegend": False,
            "margin": {"l": 150},
            "xaxis": {
                **axis_title(f"Drop in {permutation['metric']} when shuffled"),
                "zeroline": True,
                "zerolinecolor": tokens.ink3,
            },
        },
    }


def beeswarm_figure(shap: dict[str, Any], tokens: Tokens) -> Built:
    """One line per column, one point per row: its SHAP value, coloured by the column's value."""
    features = list(reversed(shap["features"]))
    low, high = tokens.palette[0], tokens.palette[1]
    data = []
    for index, feature in enumerate(features):
        # a fixed jitter, the same as the app's
        jitter = [
            (((i * 2654435761) % 2**32) % 1000 / 1000 - 0.5) * 0.6
            for i in range(len(feature["values"]))
        ]
        numeric = any(c is not None for c in feature["colors"])
        marker: dict[str, Any] = (
            {
                "size": 4,
                "opacity": 0.75,
                "color": [0.5 if c is None else c for c in feature["colors"]],
                "colorscale": [[0, low], [0.5, tokens.ink3], [1, high]],
                "cmin": 0,
                "cmax": 1,
                "showscale": index == len(features) - 1,
                "colorbar": {
                    "title": {"text": "value", "side": "right", "font": {"size": 10}},
                    "tickvals": [0, 1],
                    "ticktext": ["low", "high"],
                    "thickness": 8,
                    "len": 0.5,
                },
            }
            if numeric
            else {"size": 4, "opacity": 0.6, "color": tokens.ink3}
        )
        data.append(
            {
                "type": "scattergl",
                "mode": "markers",
                "name": feature["column"],
                "x": feature["values"],
                "y": [index + j for j in jitter],
                "marker": marker,
                "hovertemplate": f"{feature['column']}: %{{x:.4g}}<extra></extra>",
            }
        )
    return {
        "data": data,
        "layout": {
            "showlegend": False,
            "margin": {"l": 160},
            "xaxis": {
                **axis_title(f"SHAP value ({shap['scale']})"),
                "zeroline": True,
                "zerolinecolor": tokens.ink3,
            },
            "yaxis": {
                "tickvals": list(range(len(features))),
                "ticktext": [f["column"] for f in features],
                "range": [-0.6, len(features) - 0.4],
            },
        },
    }


def dependence_figure(found: dict[str, Any], tokens: Tokens) -> Built:
    """The average prediction as one column varies, with the rows at each value below."""
    number = found["kind"] == "number"
    curves = list(found["curves"].items())
    data: list[Built] = [
        {
            "type": "scatter" if number else "bar",
            **({"mode": "lines+markers"} if number else {}),
            "name": name,
            "x": found["grid"],
            "y": values,
            "marker": {"color": tokens.palette[index % 8], "size": 4}
            if number
            else {"color": tokens.palette[index % 8]},
            **({"line": {"color": tokens.palette[index % 8], "width": 2}} if number else {}),
            "hovertemplate": f"{name}: %{{y:.4g}}<extra>%{{x}}</extra>",
        }
        for index, (name, values) in enumerate(curves)
    ]
    data.append(
        {
            "type": "bar",
            "x": found["grid"],
            "y": found["counts"],
            "yaxis": "y2",
            "marker": {"color": tokens.line},
            "hovertemplate": "%{y} rows<extra>%{x}</extra>",
            "showlegend": False,
        }
    )
    return {
        "data": data,
        "layout": {
            "showlegend": len(curves) > 1,
            "barmode": "group",
            "xaxis": {**axis_title(found["column"]), "type": "linear" if number else "category"},
            "yaxis": {**axis_title(found["output"]), "domain": [0.3, 1]},
            "yaxis2": {
                "domain": [0, 0.2],
                "showticklabels": False,
                "title": {"text": "rows", "font": {"size": 10}},
            },
        },
    }


def trials_figure(progress: dict[str, Any], tokens: Tokens) -> Built:
    """Score of every trial and the best so far."""
    done = [t for t in progress["trials"] if t["value"] is not None]
    higher = progress.get("greater_is_better", True)
    best: float | None = None
    running = []
    for trial in done:
        value = trial["value"]
        best = value if best is None else (max(best, value) if higher else min(best, value))
        running.append(best)
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "name": "trial",
                "x": [t["number"] for t in done],
                "y": [t["value"] for t in done],
                "marker": {
                    "color": [
                        tokens.palette[0] if t["state"] == "complete" else tokens.ink3 for t in done
                    ],
                    "size": 6,
                },
                "hovertemplate": "trial %{x}: %{y:.4g}<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "best so far",
                "x": [t["number"] for t in done],
                "y": running,
                "line": {"color": tokens.palette[1], "width": 2, "shape": "hv"},
                "hoverinfo": "skip",
            },
        ],
        "layout": {
            "showlegend": True,
            "xaxis": axis_title("Trial"),
            "yaxis": axis_title(progress["metric"]),
        },
    }


def tuning_importance_figure(importances: dict[str, float], tokens: Tokens) -> Built:
    """How much each setting mattered in the search (fANOVA)."""
    entries = sorted(importances.items(), key=lambda item: item[1])
    return {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "y": [k for k, _ in entries],
                "x": [v for _, v in entries],
                "marker": {"color": tokens.palette[0]},
                "hovertemplate": "%{y}: %{x:.0%}<extra></extra>",
            }
        ],
        "layout": {
            "showlegend": False,
            "margin": {"l": 160},
            "xaxis": {**axis_title("Share of the score's variation"), "tickformat": ".0%"},
        },
    }


def overlay_predicted_figure(runs: list[tuple[str, dict[str, Any]]], tokens: Tokens) -> Built:
    """Predicted against actual for several models, one colour each."""
    values = [
        v
        for _, run in runs
        for v in [*run["regression"]["actual"], *run["regression"]["predicted"]]
    ]
    low, high = (min(values), max(values)) if values else (0, 1)
    return {
        "data": [
            *[
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": name,
                    "x": run["regression"]["actual"],
                    "y": run["regression"]["predicted"],
                    "marker": {"color": tokens.palette[i % 8], "size": 4, "opacity": 0.5},
                }
                for i, (name, run) in enumerate(runs)
            ],
            {
                "type": "scatter",
                "mode": "lines",
                "x": [low, high],
                "y": [low, high],
                "line": {"color": tokens.ink3, "dash": "dash", "width": 1},
                "hoverinfo": "skip",
                "showlegend": False,
            },
        ],
        "layout": {
            "showlegend": True,
            "xaxis": axis_title("Actual"),
            "yaxis": axis_title("Predicted"),
        },
    }


def overlay_roc_figure(runs: list[tuple[str, dict[str, Any]]], tokens: Tokens) -> Built:
    """ROC curves of several binary models."""
    traces = []
    for i, (name, run) in enumerate(runs):
        found = run.get("classification") or {}
        auc = found.get("roc_auc")
        roc = found.get("roc") or {"x": [], "y": []}
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"{name} (AUC {auc:.3f})" if auc is not None else f"{name} (AUC —)",
                "x": roc["x"],
                "y": roc["y"],
                "line": {"color": tokens.palette[i % 8], "width": 2},
            }
        )
    traces.append(
        {
            "type": "scatter",
            "mode": "lines",
            "x": [0, 1],
            "y": [0, 1],
            "line": {"color": tokens.ink3, "dash": "dash", "width": 1},
            "hoverinfo": "skip",
            "showlegend": False,
        }
    )
    return {
        "data": traces,
        "layout": {
            "showlegend": True,
            "xaxis": axis_title("False positive rate"),
            "yaxis": axis_title("True positive rate"),
        },
    }


def waterfall_figure(shap: dict[str, Any], row: int, tokens: Tokens) -> Built:
    """One row: from the average prediction to this row's, the ten largest contributions first
    (`row`: the file row, numbered from 0 as in `shap.rows`)."""
    if row not in shap["rows"]:
        raise ValueError(f"Row {row} is not among the explained rows (shap['rows']).")
    index = shap["rows"].index(row)
    values, labels = shap["matrix"][index], shap["labels"][index]
    order = sorted(range(len(values)), key=lambda j: -abs(values[j]))
    top, rest = order[:10], order[10:]
    steps = [{"label": f"{shap['columns'][j]} = {labels[j]}", "value": values[j]} for j in top]
    if rest:
        steps.append({"label": f"{len(rest)} other columns", "value": sum(values[j] for j in rest)})
    scale = shap["scale"]
    return {
        "data": [
            {
                "type": "waterfall",
                "orientation": "h",
                "measure": ["absolute", *["relative"] * len(steps), "total"],
                "y": ["average", *[s["label"] for s in steps], "this row"],
                "x": [shap["base"], *[s["value"] for s in steps], 0],
                "increasing": {"marker": {"color": tokens.palette[1]}},
                "decreasing": {"marker": {"color": tokens.palette[0]}},
                "totals": {"marker": {"color": tokens.ink3}},
                "connector": {"line": {"color": tokens.line}},
                "hovertemplate": "%{y}: %{x:+.4g}<extra></extra>",
            }
        ],
        "layout": {
            "showlegend": False,
            "margin": {"l": 220},
            "yaxis": {"autorange": "reversed"},
            "xaxis": axis_title("Prediction" if scale == "prediction" else f"Prediction ({scale})"),
        },
    }


# --- Where the model is wrong (036, Errors.tsx and errorFigures.ts) ------------------------------

ERROR_BINS = 12  # bars of one small multiple: the most populated categories beyond


def _shown_bins(column: dict[str, Any]) -> list[dict[str, Any]]:
    """Every quintile; for categories, the 12 most populated (in their order)."""
    bins = [b for b in column["bins"] if b["error"] is not None]
    if column["kind"] != "category" or len(bins) <= ERROR_BINS:
        return bins
    kept = {id(b) for b in sorted(bins, key=lambda b: -b["rows"])[:ERROR_BINS]}
    return [b for b in bins if id(b) in kept]


def error_by_column_figure(
    column: dict[str, Any], overall: float, label: str, tokens: Tokens, few: int = 30
) -> Built:
    """Mean error per category or quintile of one column, against the average (dashed); bins
    under 30 rows are muted (their mean is unsure)."""
    bins = _shown_bins(column)
    return {
        "data": [
            {
                "type": "bar",
                "x": [b["label"] for b in bins],
                "y": [b["error"] for b in bins],
                "customdata": [b["rows"] for b in bins],
                "marker": {
                    "color": [
                        tokens.palette[0] if b["rows"] >= few else with_alpha(tokens.ink3, 0.45)
                        for b in bins
                    ]
                },
                "hovertemplate": "%{x}: %{y:,.4g} (%{customdata:,} rows)<extra></extra>",
            }
        ],
        "layout": {
            "showlegend": False,
            "bargap": 0.2,
            "shapes": [
                {
                    "type": "line",
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "y0": overall,
                    "y1": overall,
                    "line": {"color": tokens.ink3, "dash": "dash", "width": 1},
                }
            ],
            "annotations": [
                {
                    "xref": "paper",
                    "x": 1,
                    "y": overall,
                    "xanchor": "right",
                    "yanchor": "bottom",
                    "text": "average",
                    "showarrow": False,
                    "font": {"size": 10, "color": tokens.ink2},
                }
            ],
            "xaxis": {"type": "category", "automargin": True},
            "yaxis": {**axis_title(label), "rangemode": "tozero"},
        },
    }
