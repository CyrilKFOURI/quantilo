import type { Data } from "plotly.js-dist-min";

import type { Structure } from "../../api/client";
import { withAlpha } from "../../charts/figures";
import type { Figure } from "../../charts/Plot";
import { divergingScale } from "./figures";

const title = (text: string) => ({ title: { text, font: { size: 11 } } });
const percent = (share: number) => `${Math.round(share * 100)}%`;

/** Scree (078, R-STR-6): eigenvalues as bars (kept ones in the first palette colour), the
 * parallel-analysis threshold dashed, the cumulative share on a second axis. */
export function screeFigure(structure: Structure): Figure {
  return (tokens) => {
    const names = structure.components.map((c) => `PC${c.index}`);
    const kept = tokens.palette[0] ?? tokens.accent;
    return {
      data: [
        {
          type: "bar",
          name: "Eigenvalue",
          x: names,
          y: structure.components.map((c) => c.eigenvalue),
          marker: { color: structure.components.map((c) => (c.retained ? kept : withAlpha(tokens.ink3, 0.45))) },
          customdata: structure.components.map((c) => [percent(c.explained), c.retained ? "kept" : "not kept"]),
          hovertemplate: "%{x}: eigenvalue %{y:.2f}<br>%{customdata[0]} of the variance · %{customdata[1]}<extra></extra>",
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          name: "Parallel analysis (95th percentile of shuffled data)",
          x: names,
          y: structure.components.map((c) => c.threshold),
          line: { color: tokens.ink2, dash: "dash", width: 1.5 },
          hovertemplate: "%{x}: chance level %{y:.2f}<extra></extra>",
        } as Data,
        {
          type: "scatter",
          mode: "lines+markers",
          name: "Cumulative share",
          x: names,
          y: structure.components.map((c) => c.cumulative * 100),
          yaxis: "y2",
          line: { color: tokens.palette[1] ?? tokens.ink1, width: 1.5 },
          marker: { size: 5 },
          hovertemplate: "%{x}: %{y:.0f}% of the variance so far<extra></extra>",
        } as Data,
      ],
      layout: {
        showlegend: true,
        legend: { orientation: "h", x: 0, y: -0.18, font: { size: 10 } },
        xaxis: { type: "category" },
        yaxis: { ...title("Eigenvalue"), rangemode: "tozero" },
        yaxis2: {
          overlaying: "y",
          side: "right",
          range: [0, 100],
          ticksuffix: "%",
          showgrid: false,
          tickfont: { size: 10, color: tokens.ink3 },
          fixedrange: true,
          automargin: true,
        },
      },
    };
  };
}

/** Loadings (correlation between each column and each component), the matrix's diverging scale. */
export function loadingsFigure(structure: Structure): Figure {
  return (tokens) => {
    const components = Array.from({ length: structure.shown }, (_, i) => `PC${i + 1}`);
    const showText = structure.columns.length <= 16;
    return {
      data: [
        {
          type: "heatmap",
          x: components,
          y: structure.columns,
          z: structure.loadings,
          zmin: -1,
          zmax: 1,
          colorscale: divergingScale(tokens),
          text: structure.loadings.map((row) => row.map((v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2))),
          texttemplate: showText ? "%{text}" : "",
          textfont: { size: 10, color: tokens.ink1 },
          xgap: 1,
          ygap: 1,
          colorbar: { thickness: 10, outlinewidth: 0, tickfont: { size: 10, color: tokens.ink3 }, len: 0.8 },
          hovertemplate: "%{y} on %{x}: %{z:.2f}<extra></extra>",
        } as unknown as Data,
      ],
      layout: {
        xaxis: { type: "category", showgrid: false, side: "top" },
        yaxis: { type: "category", showgrid: false, autorange: "reversed" },
      },
    };
  };
}

/** Biplot: the drawn rows on PC1 × PC2 and an arrow per column, scaled to the points' spread. */
export function biplotFigure(structure: Structure): Figure {
  return (tokens) => {
    const biplot = structure.biplot;
    if (!biplot) return { data: [] };
    const reach = Math.max(...biplot.x.map(Math.abs), ...biplot.y.map(Math.abs), 1);
    const longest = Math.max(...biplot.arrows.map((a) => Math.hypot(a.x, a.y)), 1e-9);
    const scale = (0.85 * reach) / longest;
    const arrowColor = tokens.palette[7] ?? tokens.ink1;
    const share = (i: number) => percent(structure.components[i]?.explained ?? 0);
    const lines: Data = {
      type: "scatter",
      mode: "lines",
      x: biplot.arrows.flatMap((a) => [0, a.x * scale, null]),
      y: biplot.arrows.flatMap((a) => [0, a.y * scale, null]),
      line: { color: arrowColor, width: 1.5 },
      hoverinfo: "skip",
    } as Data;
    const tips: Data = {
      type: "scatter",
      mode: "text+markers",
      x: biplot.arrows.map((a) => a.x * scale),
      y: biplot.arrows.map((a) => a.y * scale),
      text: biplot.arrows.map((a) => a.column),
      textposition: "top center",
      textfont: { size: 10, color: tokens.ink1 },
      marker: { color: arrowColor, size: 5 },
      customdata: biplot.arrows.map((a) => [a.x, a.y]),
      hovertemplate: "%{text}<br>loading %{customdata[0]:.2f} on PC1, %{customdata[1]:.2f} on PC2<extra></extra>",
    } as Data;
    return {
      data: [
        {
          type: "scatter", // 2,000 points draw fine in SVG, and need no WebGL
          mode: "markers",
          x: biplot.x,
          y: biplot.y,
          marker: { color: tokens.palette[0] ?? tokens.ink2, size: 4, opacity: 0.35 },
          hoverinfo: "skip",
        } as Data,
        lines,
        tips,
      ],
      layout: {
        xaxis: { ...title(`PC1 (${share(0)})`), zeroline: true },
        yaxis: { ...title(`PC2 (${share(1)})`), zeroline: true },
      },
    };
  };
}
