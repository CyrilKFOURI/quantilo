import type { Data, Layout } from "plotly.js-dist-min";

import type { AssociationMatrix } from "../../api/client";
import type { ChartTokens } from "../../charts/theme";
import { t } from "../../lib/i18n";

type Built = { data: Data[]; layout?: Partial<Layout> };

export const METRIC_LABELS: Record<string, string> = {
  pearson: "Pearson r",
  spearman: "Spearman ρ",
  cramers_v: "Cramér's V",
  get eta() {
    return t("fig.corr.eta");
  },
};

/** Diverging scale: red for negative, neutral grey at 0, blue for positive (validated pair). */
export function divergingScale(tokens: ChartTokens): [number, string][] {
  return [
    [0, tokens.palette[7] ?? tokens.ink3],
    [0.5, tokens.line],
    [1, tokens.palette[0] ?? tokens.ink1],
  ];
}

/** Heatmap of the matrix in the given column order; text values when it stays readable. */
export function matrixFigure(matrix: AssociationMatrix, order: number[]): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const names = order.map((i) => matrix.columns[i]?.name ?? "");
    const z = order.map((i) => order.map((j) => matrix.values[i]?.[j] ?? null));
    const custom = order.map((i) =>
      order.map((j) => [METRIC_LABELS[matrix.metrics[i]?.[j] ?? ""] ?? "", (matrix.n[i]?.[j] ?? 0).toLocaleString("en-US")]),
    );
    const showText = names.length <= 16;
    return {
      data: [
        {
          type: "heatmap",
          x: names,
          y: names,
          z,
          zmin: -1,
          zmax: 1,
          colorscale: divergingScale(tokens),
          customdata: custom,
          text: z.map((row) => row.map((v) => (v === null ? "" : v.toFixed(2)))),
          texttemplate: showText ? "%{text}" : "",
          textfont: { size: 10, color: tokens.ink1 },
          xgap: 1,
          ygap: 1,
          colorbar: { thickness: 10, outlinewidth: 0, tickfont: { size: 10, color: tokens.ink3 }, len: 0.8 },
          hovertemplate: "%{y} × %{x}<br>%{customdata[0]} = %{z:.3f}<br>n = %{customdata[1]}<extra></extra>",
          hoverongaps: false,
        } as unknown as Data,
      ],
      layout: {
        margin: { l: 8, r: 8, t: 8, b: 8 },
        xaxis: { type: "category", showgrid: false, tickangle: -45, side: "bottom" },
        yaxis: { type: "category", showgrid: false, autorange: "reversed" },
      },
    };
  };
}
