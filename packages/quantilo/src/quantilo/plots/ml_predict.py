"""The figures of "Predict on new data" (037; frontend/src/modules/ml/predictFigures.ts)."""

import itertools
from typing import Any

from quantilo.plots.theme import Tokens, axis_title, with_alpha

Built = dict[str, Any]


def distribution_figure(distribution: dict[str, Any], tokens: Tokens) -> Built:
    """The predictions on the new data (bars) against those on the validation rows (a neutral
    step line for a number, pale bars beside for classes): a shift shows at a glance."""
    if distribution["kind"] == "classes":
        labels = distribution["labels"]
        return {
            "data": [
                {
                    "type": "bar",
                    "name": "This data",
                    "x": labels,
                    "y": distribution["new"],
                    "marker": {"color": tokens.palette[0]},
                    "hovertemplate": "%{x}: %{y:.1%} of rows<extra>This data</extra>",
                },
                {
                    "type": "bar",
                    "name": "Validation",
                    "x": labels,
                    "y": distribution["validation"],
                    "marker": {"color": with_alpha(tokens.ink3, 0.45)},
                    "hovertemplate": "%{x}: %{y:.1%} of rows<extra>Validation</extra>",
                },
            ],
            "layout": {
                "barmode": "group",
                "bargap": 0.3,
                "bargroupgap": 0.08,
                "legend": {"orientation": "h", "y": 1.12, "x": 0},
                "xaxis": axis_title("Predicted class"),
                "yaxis": {**axis_title("Share of rows"), "tickformat": ".0%"},
            },
        }
    edges = distribution["edges"]
    middles = [(a + b) / 2 for a, b in itertools.pairwise(edges)]
    widths = [b - a for a, b in itertools.pairwise(edges)]
    return {
        "data": [
            {
                "type": "bar",
                "name": "This data",
                "x": middles,
                "y": distribution["new"],
                "width": widths,
                "marker": {
                    "color": tokens.palette[0],
                    "line": {"color": tokens.surface, "width": 1},
                },
                "hovertemplate": "%{x:,.4g}: %{y:.1%} of rows<extra>This data</extra>",
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Validation",
                "x": middles,
                "y": distribution["validation"],
                "line": {"color": tokens.ink2, "width": 2, "shape": "hvh"},
                "hovertemplate": "%{x:,.4g}: %{y:.1%} of rows<extra>Validation</extra>",
            },
        ],
        "layout": {
            "bargap": 0,
            "legend": {"orientation": "h", "y": 1.12, "x": 0},
            "xaxis": axis_title("Predicted value"),
            "yaxis": {**axis_title("Share of rows"), "tickformat": ".0%"},
        },
    }
