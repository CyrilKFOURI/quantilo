import type { Data, Layout } from "plotly.js-dist-min";

import type { FigureHistory } from "../../api/recipes";
import { formatNumber } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";

/** One key figure over the runs of a recipe: a thin line, a marker per run, the last value
 *  labelled on the chart (one series per chart: its title names it, no legend). */
export function historyFigure(line: FigureHistory) {
  return (tokens: ChartTokens): { data: Data[]; layout: Partial<Layout> } => {
    const points = line.points.filter((point) => point.value !== null);
    // One position per run, named by its file (runs can be seconds or months apart).
    const x = points.map((point, index) => `${index + 1}. ${point.dataset_name}`);
    const y = points.map((point) => point.value as number);
    const color = tokens.palette[0] ?? tokens.ink2;
    const last = points.at(-1);
    return {
      data: [
        {
          type: "scatter",
          mode: "lines+markers",
          x,
          y,
          customdata: points.map((point) => [point.created_at.slice(0, 16).replace("T", " "), formatNumber(point.value)]),
          line: { color, width: 2 },
          marker: { color, size: 8, line: { color: tokens.surface, width: 2 } },
          hovertemplate: "%{x}<br>%{customdata[0]}<br><b>%{customdata[1]}</b><extra></extra>",
          hoverlabel: { namelength: 0 },
        } as Data,
      ],
      layout: {
        margin: { l: 8, r: 56, t: 8, b: 4 },
        xaxis: { type: "category", tickangle: 0, showgrid: false, showticklabels: false },
        yaxis: { nticks: 4 },
        hovermode: "closest",
        annotations: last
          ? [
              {
                x: x.at(-1),
                y: last.value as number,
                text: formatNumber(last.value),
                showarrow: false,
                xanchor: "left",
                xshift: 8,
                font: { size: 11, color: tokens.ink1 },
              },
            ]
          : [],
      },
    };
  };
}
