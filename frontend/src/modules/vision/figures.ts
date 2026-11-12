import type { Data, Layout } from "plotly.js-dist-min";

import type { Figure } from "../../charts/Plot";
import { formatNumber, withAlpha } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";
import { OTHER, type Bins, type ImageRow } from "./model";

const LOG_TICKS = [0, 1, 10, 100, 1000, 10000, 100000];

/** The palette slot of a label series: fixed by its place in the sorted label list (R-107-09). */
export function seriesColour(tokens: ChartTokens, series: readonly string[], name: string): string {
  if (name === OTHER) return tokens.ink3;
  const slot = series.indexOf(name);
  return tokens.palette[(slot < 0 ? 0 : slot) % tokens.palette.length] ?? tokens.numeric;
}

/** A histogram of one measure (24 classes); stacked by label when `series` is given. Each bar
 *  carries its [low, high] in customdata, so a click filters the gallery on it. */
export function histogramFigure(bins: Bins, options: { unit: string; series: readonly string[] | null }): Figure {
  return (tokens) => {
    const position = (value: number) => (bins.log ? Math.log10(1 + Math.max(value, 0)) : value);
    const centers: number[] = [];
    const widths: number[] = [];
    const ranges: [number, number, string, string][] = [];
    for (let k = 0; k < bins.counts.length; k += 1) {
      const low = bins.edges[k] ?? 0;
      const high = bins.edges[k + 1] ?? low;
      centers.push((position(low) + position(high)) / 2);
      widths.push(position(high) - position(low));
      ranges.push([low, high, formatNumber(low), formatNumber(high)]);
    }
    const hover = `%{customdata[2]} to %{customdata[3]}<br>%{y:,} ${options.unit}<extra>%{fullData.name}</extra>`;
    const line = { color: tokens.surface, width: 1 };
    const data: Data[] =
      options.series && bins.byLabel
        ? options.series.map(
            (name) =>
              ({
                type: "bar",
                name,
                x: centers,
                y: bins.byLabel?.get(name) ?? [],
                width: widths,
                customdata: ranges,
                marker: { color: seriesColour(tokens, options.series ?? [], name), line },
                hovertemplate: hover,
              }) as Data,
          )
        : [
            {
              type: "bar",
              name: "",
              x: centers,
              y: bins.counts,
              width: widths,
              customdata: ranges,
              marker: { color: tokens.numeric, line },
              hovertemplate: `%{customdata[2]} to %{customdata[3]}<br>%{y:,} ${options.unit}<extra></extra>`,
            } as Data,
          ];
    const layout: Partial<Layout> = {
      bargap: 0,
      barmode: "stack",
      showlegend: Boolean(options.series && options.series.length > 1),
      legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { size: 11, color: tokens.ink2 } },
      xaxis: bins.log
        ? {
            showgrid: false,
            tickvals: LOG_TICKS.map((v) => Math.log10(1 + v)),
            ticktext: LOG_TICKS.map((v) => formatNumber(v)),
          }
        : { showgrid: false },
    };
    return { data, layout };
  };
}

const RATIOS: [number, string][] = [
  [1, "1:1"],
  [4 / 3, "4:3"],
  [16 / 9, "16:9"],
  [3 / 4, "3:4"],
  [9 / 16, "9:16"],
];
const SCATTER_LIMIT = 5000;

/** Width × height on log axes, with the usual ratios drawn and named (R-107-10). Beyond 5,000
 *  images an even sample is drawn (said under the chart). */
export function sizeFigure(rows: readonly ImageRow[]): Figure {
  return (tokens) => {
    const sized = rows.filter((row) => row.width && row.height);
    const step = Math.max(1, Math.ceil(sized.length / SCATTER_LIMIT));
    const shown = sized.filter((_, k) => k % step === 0);
    const widths = shown.map((row) => row.width ?? 1);
    const heights = shown.map((row) => row.height ?? 1);
    const low = Math.max(1, Math.min(...widths, ...heights) / 1.6);
    const high = Math.max(...widths, ...heights) * 1.6;
    const lines: Data[] = RATIOS.map(([ratio]) => ({
      type: "scatter",
      mode: "lines",
      x: [low, high],
      y: [low / ratio, high / ratio],
      line: { color: tokens.line, width: 1, dash: ratio === 1 ? "solid" : "dot" },
      hoverinfo: "skip",
    }));
    const points: Data = {
      type: shown.length > 2000 ? "scattergl" : "scatter",
      mode: "markers",
      x: widths,
      y: heights,
      customdata: shown.map((row) => row.name),
      marker: { color: withAlpha(tokens.numeric, 0.55), size: 7, line: { color: tokens.surface, width: 1 } },
      hovertemplate: "%{customdata}<br>%{x:,} × %{y:,} px<extra></extra>",
    } as Data;
    const annotations = RATIOS.map(([ratio, name]) => {
      const x = ratio >= 1 ? high / 1.25 : (high / 1.25) * ratio;
      return {
        x: Math.log10(x),
        y: Math.log10(x / ratio),
        text: name,
        showarrow: false,
        font: { size: 10, color: tokens.ink3 },
        xanchor: "left" as const,
        yanchor: "bottom" as const,
      };
    });
    return {
      data: [...lines, points],
      layout: {
        xaxis: { type: "log", range: [Math.log10(low), Math.log10(high)], showgrid: false, title: { text: "width (px)", font: { size: 11 } } },
        yaxis: { type: "log", range: [Math.log10(low), Math.log10(high)], title: { text: "height (px)", font: { size: 11 } } },
        annotations,
      },
    };
  };
}

/** Images per label, largest first, coloured by the label's fixed slot, counts written at the bar. */
export function labelsFigure(counts: readonly [string, number][], series: readonly string[]): Figure {
  return (tokens) => {
    const ordered = [...counts].reverse(); // plotly draws the first category at the bottom
    return {
      data: [
        {
          type: "bar",
          orientation: "h",
          x: ordered.map(([, count]) => count),
          y: ordered.map(([label]) => label),
          customdata: ordered.map(([label]) => label),
          text: ordered.map(([, count]) => formatNumber(count)),
          textposition: "outside",
          cliponaxis: false,
          textfont: { size: 11, color: tokens.ink2 },
          marker: {
            color: ordered.map(([label]) => seriesColour(tokens, series, series.includes(label) ? label : OTHER)),
            line: { color: tokens.surface, width: 1 },
          },
          hovertemplate: "%{y}<br>%{x:,} images<extra></extra>",
        } as Data,
      ],
      layout: {
        bargap: 0.25,
        xaxis: { showgrid: true, showticklabels: false },
        yaxis: { showgrid: false, automargin: true, tickfont: { size: 11, color: tokens.ink2, family: "IBM Plex Mono, monospace" } },
        margin: { l: 8, r: 36, t: 4, b: 8 },
      },
    };
  };
}
