import type { Data } from "plotly.js-dist-min";

import type { MlColumnErrors } from "../../api/client";
import type { Figure } from "../../charts/Plot";
import { withAlpha } from "../../charts/figures";

/** Bars of one small multiple: the most populated categories beyond (quintiles: all five). */
export const ERROR_BINS = 12;
/** Bins under 30 rows are muted: their mean error is unsure. */
const FEW_ROWS = 30;

export function shownBins(column: MlColumnErrors): MlColumnErrors["bins"] {
  const bins = column.bins.filter((b) => b.error !== null);
  if (column.kind !== "category" || bins.length <= ERROR_BINS) return bins;
  const kept = new Set([...bins].sort((a, b) => b.rows - a.rows).slice(0, ERROR_BINS));
  return bins.filter((b) => kept.has(b));
}

/** Mean error per category or quintile of one column, against the average (dashed). */
export function errorByColumnFigure(column: MlColumnErrors, overall: number, label: string): Figure {
  return (tokens) => {
    const bins = shownBins(column);
    return {
      data: [
        {
          type: "bar",
          x: bins.map((b) => b.label),
          y: bins.map((b) => b.error),
          customdata: bins.map((b) => b.rows),
          marker: { color: bins.map((b) => (b.rows >= FEW_ROWS ? (tokens.palette[0] ?? tokens.numeric) : withAlpha(tokens.ink3, 0.45))) },
          hovertemplate: "%{x}: %{y:,.4g} (%{customdata:,} rows)<extra></extra>",
        } as Data,
      ],
      layout: {
        showlegend: false,
        bargap: 0.2,
        shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: overall, y1: overall, line: { color: tokens.ink3, dash: "dash", width: 1 } }],
        annotations: [
          { xref: "paper", x: 1, y: overall, xanchor: "right", yanchor: "bottom", text: "average", showarrow: false, font: { size: 10, color: tokens.ink2 } },
        ],
        xaxis: { type: "category", automargin: true },
        yaxis: { title: { text: label, font: { size: 11 } }, rangemode: "tozero" },
      },
    };
  };
}
