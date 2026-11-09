import type { Data } from "plotly.js-dist-min";

import type { MlPredictionDistribution } from "../../api/client";
import type { Figure } from "../../charts/Plot";
import { withAlpha } from "../../charts/figures";

const title = (text: string) => ({ title: { text, font: { size: 11 } } });

/** The predictions on the new data (bars) against those on the validation rows (a neutral step
 *  line for a number, pale bars beside for classes): a shift shows at a glance (037). Twin:
 *  quantilo/plots/ml_predict.py. */
export function distributionFigure(distribution: MlPredictionDistribution): Figure {
  return (tokens) => {
    const legend = { orientation: "h" as const, y: 1.12, x: 0 };
    if (distribution.kind === "classes") {
      return {
        data: [
          {
            type: "bar",
            name: "This data",
            x: distribution.labels,
            y: distribution.new,
            marker: { color: tokens.palette[0] ?? tokens.numeric },
            hovertemplate: "%{x}: %{y:.1%} of rows<extra>This data</extra>",
          } as Data,
          {
            type: "bar",
            name: "Validation",
            x: distribution.labels,
            y: distribution.validation,
            marker: { color: withAlpha(tokens.ink3, 0.45) },
            hovertemplate: "%{x}: %{y:.1%} of rows<extra>Validation</extra>",
          } as Data,
        ],
        layout: {
          barmode: "group",
          bargap: 0.3,
          bargroupgap: 0.08,
          legend,
          xaxis: title("Predicted class"),
          yaxis: { ...title("Share of rows"), tickformat: ".0%" },
        },
      };
    }
    const edges = distribution.edges;
    const middles = edges.slice(1).map((end, i) => ((edges[i] ?? end) + end) / 2);
    const widths = edges.slice(1).map((end, i) => end - (edges[i] ?? end));
    return {
      data: [
        {
          type: "bar",
          name: "This data",
          x: middles,
          y: distribution.new,
          width: widths,
          marker: { color: tokens.palette[0] ?? tokens.numeric, line: { color: tokens.surface, width: 1 } },
          hovertemplate: "%{x:,.4g}: %{y:.1%} of rows<extra>This data</extra>",
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          name: "Validation",
          x: middles,
          y: distribution.validation,
          line: { color: tokens.ink2, width: 2, shape: "hvh" },
          hovertemplate: "%{x:,.4g}: %{y:.1%} of rows<extra>Validation</extra>",
        } as Data,
      ],
      layout: {
        bargap: 0,
        legend,
        xaxis: title("Predicted value"),
        yaxis: { ...title("Share of rows"), tickformat: ".0%" },
      },
    };
  };
}
