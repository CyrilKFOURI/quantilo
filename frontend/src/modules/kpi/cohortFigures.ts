import type { Data, Layout } from "plotly.js-dist-min";

import type { CohortResult, FunnelResult } from "../../api/client";
import { withAlpha } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";
import { formatShare } from "./format";

type Built = { data: Data[]; layout?: Partial<Layout> };

/** Offsets as column headers: "+0", "+1 month", "+2"… */
export function offsetLabels(result: CohortResult): string[] {
  const width = Math.max(0, ...result.cohorts.map((c) => c.retained.length));
  return Array.from({ length: width }, (_, k) => (k === 0 ? `+0` : `+${k}`));
}

/** The triangular matrix (060, P7): cohorts down, offsets across, empty where not observed. */
export function cohortMatrix(result: CohortResult): { x: string[]; y: string[]; z: (number | null)[][] } {
  const x = offsetLabels(result);
  return {
    x,
    y: result.cohorts.map((c) => `${c.label} · ${c.size.toLocaleString("en-US")}`),
    z: result.cohorts.map((c) => x.map((_, k) => c.retained[k] ?? null)),
  };
}

/** One hue from pale to full: retention is a share, never diverging (R-TAB-09). */
export function cohortFigure(result: CohortResult): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const { x, y, z } = cohortMatrix(result);
    const blue = tokens.palette[0] ?? tokens.ink2;
    return {
      data: [
        {
          type: "heatmap",
          x,
          y,
          z: z.map((row) => row.map((v) => (v === null ? null : v * 100))),
          zmin: 0,
          zmax: 100,
          colorscale: [
            [0, withAlpha(blue, 0.06)],
            [1, blue],
          ],
          text: z.map((row) => row.map((v) => (v === null ? "" : formatShare(v)))),
          texttemplate: "%{text}",
          textfont: { size: 10, color: tokens.ink1 },
          hovertemplate: `%{y} · %{x} ${result.grain === "month" ? "months" : "weeks"}: %{text}<extra></extra>`,
          showscale: false,
          xgap: 2,
          ygap: 2,
        } as Data,
      ],
      layout: {
        yaxis: { autorange: "reversed", automargin: true, showgrid: false, type: "category" },
        xaxis: { type: "category", showgrid: false, side: "top" },
      },
    };
  };
}

/** New and returning customers per period, stacked; an incomplete period is paler. */
export function newReturningFigure(result: CohortResult): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const x = result.periods.map((p) => p.label);
    const [first = tokens.ink2, second = tokens.ink3] = tokens.palette;
    const opacity = result.periods.map((p) => (p.partial ? 0.45 : 1));
    return {
      data: [
        { type: "bar", name: "New", x, y: result.periods.map((p) => p.new), marker: { color: first, opacity }, hovertemplate: "%{x}: %{y:,} new<extra></extra>" } as Data,
        { type: "bar", name: "Returning", x, y: result.periods.map((p) => p.returning), marker: { color: second, opacity }, hovertemplate: "%{x}: %{y:,} returning<extra></extra>" } as Data,
      ],
      layout: {
        barmode: "stack",
        showlegend: true,
        legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11, color: tokens.ink2 } },
        xaxis: { type: "category", showgrid: false },
      },
    };
  };
}

/** Each stage's distinct keys (or rows), labelled with the share of the first stage. */
export function funnelFigure(result: FunnelResult): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const y = result.stages.map((s) => s.label);
    const blue = tokens.palette[0] ?? tokens.ink2;
    return {
      data: [
        {
          type: "bar",
          orientation: "h",
          y,
          x: result.stages.map((s) => s.reached),
          marker: { color: blue },
          text: result.stages.map((s) => `${s.reached.toLocaleString("en-US")} · ${formatShare(s.of_first ?? 0)}`),
          textposition: "outside",
          cliponaxis: false,
          hovertemplate: "%{y}: %{text}<extra></extra>",
        } as Data,
      ],
      layout: { yaxis: { autorange: "reversed", automargin: true, type: "category" }, xaxis: { showgrid: true }, margin: { l: 8, r: 72, t: 4, b: 8 } },
    };
  };
}

/** "Qualified → Proposal: 52 %" lines under the funnel, the largest drop first in the summary. */
export function conversions(result: FunnelResult): { from: string; to: string; share: number }[] {
  return result.stages.slice(1).flatMap((stage, i) => {
    const from = result.stages[i];
    return from && stage.conversion !== null ? [{ from: from.label, to: stage.label, share: stage.conversion }] : [];
  });
}
