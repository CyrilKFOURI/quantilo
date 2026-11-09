import type { Data, Layout } from "plotly.js-dist-min";

import type { DateChange, Shift } from "../../api/client";
import type { Figure } from "../../charts/Plot";
import { withAlpha } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";

/** One fixed look per version on the whole page: before = solid grey, after = accent outline,
 *  so the two never depend on colour alone (twin: quantilo/plots/changes.py). */
export function looks(tokens: ChartTokens) {
  return {
    before: { name: "Before", color: tokens.ink2, fill: withAlpha(tokens.ink2, 0.45) },
    after: { name: "After", color: tokens.accent, fill: withAlpha(tokens.accent, 0.12) },
  };
}

const legend: Partial<Layout> = {
  showlegend: true,
  legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11 } },
};
const percentAxis = { tickformat: ".0%", rangemode: "tozero" as const };

function pair(
  tokens: ChartTokens,
  common: Record<string, unknown>,
  before: number[],
  after: number[],
  axis: "x" | "y",
): Data[] {
  const look = looks(tokens);
  const other = axis === "x" ? "y" : "x";
  return [
    {
      ...common,
      name: look.before.name,
      [other]: before,
      marker: { color: look.before.fill, line: { width: 0 } },
      hovertemplate: `%{${axis}}<br>before %{${other}:.1%}<extra></extra>`,
    } as Data,
    {
      ...common,
      name: look.after.name,
      [other]: after,
      marker: { color: look.after.fill, line: { color: look.after.color, width: 1.5 } },
      hovertemplate: `%{${axis}}<br>after %{${other}:.1%}<extra></extra>`,
    } as Data,
  ];
}

/** Both versions on the same intervals (numbers) or values (categories), as shares. */
export function shiftFigure(shift: Shift): Figure {
  return (tokens) => {
    const histogram = shift.histogram;
    const bars = shift.kind === "numeric" && shift.levels.length > 0;
    if (histogram && !bars) {
      const edges = histogram.edges;
      const centres = edges.slice(0, -1).map((e, i) => (e + (edges[i + 1] ?? e)) / 2);
      const widths = edges.slice(0, -1).map((e, i) => (edges[i + 1] ?? e) - e);
      return {
        data: pair(tokens, { type: "bar", x: centres, width: widths }, histogram.before, histogram.after, "x"),
        layout: {
          ...legend,
          barmode: "overlay",
          bargap: 0,
          yaxis: { ...percentAxis, title: { text: "Share of values", font: { size: 11 } } },
        },
      };
    }
    const rows = [...shift.levels].reverse();
    return {
      data: pair(
        tokens,
        { type: "bar", orientation: "h", y: rows.map((r) => r.value) },
        rows.map((r) => r.before),
        rows.map((r) => r.after),
        "y",
      ),
      layout: {
        ...legend,
        barmode: "group",
        bargap: 0.25,
        xaxis: { ...percentAxis, showgrid: true },
        yaxis: { showgrid: false, type: "category" },
      },
    };
  };
}

/** Rows per period in both versions: a hole or a new last period shows at once. */
export function periodsFigure(dates: DateChange): Figure {
  return (tokens) => {
    const look = looks(tokens);
    return {
      data: [
        {
          type: "bar",
          name: look.before.name,
          x: dates.periods,
          y: dates.before_counts,
          marker: { color: look.before.fill },
          hovertemplate: "%{x}<br>before %{y:,} rows<extra></extra>",
        } as Data,
        {
          type: "bar",
          name: look.after.name,
          x: dates.periods,
          y: dates.after_counts,
          marker: { color: look.after.fill, line: { color: look.after.color, width: 1.5 } },
          hovertemplate: "%{x}<br>after %{y:,} rows<extra></extra>",
        } as Data,
      ],
      layout: {
        ...legend,
        barmode: "group",
        bargap: 0.2,
        xaxis: { type: "category" },
        yaxis: { title: { text: "Rows", font: { size: 11 } }, rangemode: "tozero" },
      },
    };
  };
}

/** Height of a shift's chart: one bar pair per category row, never below 240 px. */
export function shiftHeight(shift: Shift): number {
  if (shift.kind === "categorical" || (shift.levels.length > 0 && !shift.histogram)) {
    return Math.max(240, 60 + 26 * shift.levels.length);
  }
  return 300;
}
