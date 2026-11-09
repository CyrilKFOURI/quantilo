import type { Data, Layout } from "plotly.js-dist-min";

import type { KpiNumberFormat, KpiResult } from "../../api/client";
import type { ChartTokens } from "../../charts/theme";
import { withAlpha } from "../../charts/figures";
import { paletteById } from "../../lib/palettes";
import type { Format, ShowAs } from "./definition";
import { formatShown, PERCENT_MODES } from "./format";
import { DIVERGING_MODES, seriesIndex, shownOverall, shownValues } from "./pivot";

const OTHER = "Other";
const MARKERS_MAX = 60; // R-VIS-34
const BAR_LABELS_MAX = 12; // R-VIS-35
const DIRECT_LABELS_MAX = 4;
const BARS = new Set(["bars", "stacked_bars", "grouped_bars"]);
/** Room between the periods (045): one bar, or a group of bars side by side. */
const SPACING = { narrow: 0.12, normal: 0.3, wide: 0.5 } as const;
const GROUP_SPACING = { narrow: 0.2, normal: 0.38, wide: 0.55 } as const;
const STACKED = new Set(["stacked_bars", "stacked_area", "hbar_stacked"]);

export type ChartInput = {
  result: KpiResult;
  measure: string;
  chart: string; // the type drawn (resolved by the server or chosen in Customize)
  mode: ShowAs;
  format: KpiNumberFormat;
  custom: Format;
  labels: "auto" | "on" | "off";
  hideOther: boolean;
  /** 045: a line over bars over time, one per category when the bars are split. */
  line?: boolean;
  /** 045: room between the periods' bars. */
  spacing?: "narrow" | "normal" | "wide";
  /** 047: this KPI's palette (null or unknown: the app's, already in the tokens). */
  palette?: string | null;
  panel?: string | null; // small multiples: the first split's label shown in this panel
};

type Built = { data: Data[]; layout?: Partial<Layout> };

/** The colour of a split's label: fixed categorical order by position in the split, so a label
 *  keeps its colour across views; "Other" grey; an ordinal split one hue light → dark (R-VIS-33).
 *  Two labels (true / false, yes / no) always get two different hues, never two blues (045). */
export function levelColor(levels: string[], label: string, ordinal: boolean, tokens: ChartTokens): string {
  if (label === OTHER) return tokens.ink3;
  const real = levels.filter((l) => l !== OTHER);
  const index = Math.max(0, real.indexOf(label));
  if (ordinal && real.length > 2) {
    const step = real.length > 1 ? index / (real.length - 1) : 1;
    return withAlpha(tokens.palette[0] ?? tokens.ink2, 0.5 + 0.5 * step);
  }
  return tokens.palette[index % tokens.palette.length] ?? tokens.ink2;
}

/** The plotted number: percentages are drawn × 100 with a "%" suffix on the axis. */
function scaleOf(input: ChartInput): { factor: number; prefix: string; suffix: string } {
  if (PERCENT_MODES.has(input.mode)) return { factor: 100, prefix: "", suffix: " %" };
  return {
    factor: input.format.scale,
    prefix: input.custom.prefix || input.format.prefix,
    suffix: input.custom.suffix || input.format.suffix,
  };
}

function axis(input: ChartInput, values: (number | null)[], percent = false) {
  const { prefix, suffix } = scaleOf(input);
  const largest = Math.max(0, ...values.map((v) => Math.abs(v ?? 0)));
  const decimals = typeof input.custom.decimals === "number" ? input.custom.decimals : input.format.decimals;
  return {
    tickprefix: prefix,
    ticksuffix: suffix,
    tickformat: largest >= 10_000 && input.custom.compact !== "off" ? "~s" : `,.${percent ? 0 : Math.min(decimals, 2)}f`,
  };
}

const text = (input: ChartInput, values: (number | null)[]) =>
  values.map((v) => formatShown(v, input.mode, input.format, input.custom, "compact"));

const scaled = (input: ChartInput, values: (number | null)[]) => {
  const { factor } = scaleOf(input);
  return values.map((v) => (v === null ? null : v * factor));
};

function legend(tokens: ChartTokens, count: number): Partial<Layout> {
  return count >= 2
    ? { showlegend: true, legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11, color: tokens.ink2 } } }
    : { showlegend: false };
}

/** One measure's chart (one panel of small multiples when `panel` is set). */
export function kpiFigure(input: ChartInput): (tokens: ChartTokens) => Built {
  return (appTokens) => {
    const own = paletteById(input.palette);
    const tokens = own ? { ...appTokens, palette: [...own.colors] } : appTokens;
    const { result, measure, chart, mode } = input;
    const find = seriesIndex(result);
    const periods = result.periods;
    const x = periods.map((p) => p.label);
    const splits = result.splits;
    const visible = (levels: string[]) => (input.hideOther ? levels.filter((l) => l !== OTHER) : levels);

    if (chart === "hbar" || chart === "hbar_stacked" || (chart === "heatmap" && !result.resolved.time)) {
      const first = splits[0];
      const rows = visible(first?.levels ?? []);
      if (chart === "hbar" || splits.length < 2) {
        const values = rows.map((g) => {
          const s = find(measure, g);
          return s ? shownOverall(result, s, mode) : null;
        });
        const show = input.labels !== "off" && (input.labels === "on" || rows.length <= BAR_LABELS_MAX);
        return {
          data: [
            {
              type: "bar",
              orientation: "h",
              y: rows,
              x: scaled(input, values),
              marker: { color: rows.map((g) => (g === OTHER ? tokens.ink3 : (tokens.palette[0] ?? tokens.ink2))) },
              text: show ? text(input, values) : undefined,
              textposition: "outside",
              cliponaxis: false,
              hovertemplate: `%{y}: %{customdata}<extra></extra>`,
              customdata: text(input, values),
            } as Data,
          ],
          layout: {
            yaxis: { autorange: "reversed", automargin: true },
            xaxis: { ...axis(input, values), showgrid: true, rangemode: "tozero" },
            margin: { l: 8, r: 40, t: 4, b: 8 },
          },
        };
      }
      const second = splits[1];
      if (!second) return { data: [] };
      const columns = visible(second.levels);
      if (chart === "heatmap") return heatmap(input, tokens, columns, rows, (g2, g1) => find(measure, g1, g2));
      return {
        data: columns.map((g2) => {
          const values = rows.map((g1) => {
            const s = find(measure, g1, g2);
            return s ? shownOverall(result, s, mode) : null;
          });
          return {
            type: "bar",
            orientation: "h",
            name: g2,
            y: rows,
            x: scaled(input, values),
            marker: { color: levelColor(second.levels, g2, second.ordinal, tokens), line: { color: tokens.surface, width: 2 } },
            customdata: text(input, values),
            hovertemplate: `${g2}: %{customdata}<extra>%{y}</extra>`,
          } as Data;
        }),
        layout: {
          barmode: "stack",
          yaxis: { autorange: "reversed", automargin: true },
          xaxis: { ...axis(input, []), showgrid: true },
          ...legend(tokens, columns.length),
        },
      };
    }

    // Over time: one trace per label of the split drawn (the second one inside a panel).
    const drawn = input.panel !== undefined && input.panel !== null ? splits[1] : splits[0];
    const traces: { name: string; color: string; values: (number | null)[] }[] = drawn
      ? visible(drawn.levels).map((label) => {
          const s = input.panel ? find(measure, input.panel, label) : find(measure, label);
          return {
            name: label,
            color: levelColor(drawn.levels, label, drawn.ordinal, tokens),
            values: s ? shownValues(result, s, mode) : [],
          };
        })
      : [{ name: "", color: tokens.palette[0] ?? tokens.ink2, values: (() => {
          const s = find(measure, input.panel ?? null);
          return s ? shownValues(result, s, mode) : [];
        })() }];

    if (chart === "heatmap") {
      const rows = traces.map((t) => t.name);
      const z = traces.map((t) => t.values);
      return heatmapGrid(input, tokens, x, rows, z);
    }

    const all = traces.flatMap((t) => t.values);
    const partial = periods.map((p) => p.partial);
    const many = traces.length > 1;
    const barLabels =
      input.labels === "on" || (input.labels === "auto" && !many && periods.length <= BAR_LABELS_MAX);
    const data: Data[] = traces.map((trace, i) => {
      const common = {
        legendgroup: many ? trace.name : undefined,
        name: trace.name || undefined,
        x,
        y: scaled(input, trace.values),
        customdata: text(input, trace.values),
        hovertemplate: many ? `${trace.name}: %{customdata}<extra></extra>` : `%{customdata}<extra></extra>`,
      };
      if (chart === "bars" || chart === "stacked_bars" || chart === "grouped_bars") {
        return {
          ...common,
          type: "bar",
          marker: {
            color: trace.color,
            line: { color: tokens.surface, width: chart === "stacked_bars" ? 1 : 0 },
            // incomplete periods hatched (R-TIME-05)
            pattern: { shape: partial.map((p) => (p ? "/" : "")), fgcolor: tokens.surface, solidity: 0.4 },
          },
          offsetgroup: chart === "grouped_bars" ? String(i) : undefined,
          text: barLabels ? common.customdata : undefined,
          textposition: chart === "stacked_bars" ? "inside" : "outside",
          cliponaxis: false,
        } as Data;
      }
      const area = chart === "area" || chart === "stacked_area";
      const last = trace.values.findLastIndex((v) => v !== null);
      const direct = input.labels !== "off" && traces.length <= DIRECT_LABELS_MAX && last >= 0;
      return {
        ...common,
        type: "scatter",
        mode: periods.length <= MARKERS_MAX ? "lines+markers" : "lines",
        line: { color: trace.color, width: 2 },
        marker: {
          size: 6,
          color: trace.color,
          symbol: partial.map((p) => (p ? "circle-open" : "circle")),
        },
        fill: chart === "stacked_area" ? (i === 0 ? "tozeroy" : "tonexty") : area ? "tozeroy" : "none",
        fillcolor: area ? withAlpha(trace.color, chart === "stacked_area" ? 0.55 : 0.18) : undefined,
        stackgroup: chart === "stacked_area" ? "one" : undefined,
        text: direct ? trace.values.map((_, j) => (j === last ? (common.customdata[j] ?? "") : "")) : undefined,
        textposition: "middle right",
        textfont: { color: tokens.ink2, size: 11 },
      } as Data;
    });
    // 045: a line over the bars, one per category through its own bars.
    if (input.line && BARS.has(chart)) data.push(...linesOver(input, tokens, traces, x, chart));
    const zero = chart !== "line"; // R-VIS-31: bars and areas start at 0, lines fit the data
    const spacing = input.spacing ?? "normal";
    return {
      data,
      layout: {
        barmode: chart === "stacked_bars" ? "stack" : chart === "grouped_bars" ? "group" : undefined,
        bargap: chart === "grouped_bars" ? GROUP_SPACING[spacing] : SPACING[spacing],
        bargroupgap: chart === "grouped_bars" ? 0.06 : undefined,
        // Lines over grouped bars sit on their own bar of the group, not between the groups.
        scattermode: chart === "grouped_bars" ? "group" : undefined,
        scattergap: chart === "grouped_bars" ? GROUP_SPACING[spacing] : undefined,
        hovermode: "x unified",
        xaxis: { type: "category", showgrid: false, automargin: true },
        yaxis: {
          ...axis(input, all, PERCENT_MODES.has(mode)),
          rangemode: zero && !DIVERGING_MODES.has(mode) ? "tozero" : "normal",
          zeroline: DIVERGING_MODES.has(mode),
        },
        margin: { l: 8, r: traces.length <= DIRECT_LABELS_MAX ? 56 : 8, t: 4, b: 8 },
        ...legend(tokens, traces.filter((t) => t.name).length),
      },
    };
  };
}

/** One line per category, in its bars' colour, along the top of its own bars: its bar in a group,
 *  its segment in a stack (047). A plain line, no points, no second hover (the bars carry it). */
function linesOver(
  input: ChartInput,
  tokens: ChartTokens,
  traces: { name: string; color: string; values: (number | null)[] }[],
  x: string[],
  chart: string,
): Data[] {
  const single = traces.length <= 1;
  const grouped = chart === "grouped_bars";
  const tops = chart === "stacked_bars" ? stackTops(traces.map((trace) => trace.values)) : traces.map((trace) => trace.values);
  return traces.map((trace, i) => {
    const color = single ? tokens.ink1 : trace.color;
    return {
      type: "scatter",
      mode: "lines",
      name: trace.name || undefined,
      legendgroup: single ? undefined : trace.name,
      showlegend: false,
      offsetgroup: grouped ? String(i) : undefined,
      x,
      y: scaled(input, tops[i] ?? []),
      hoverinfo: "skip",
      connectgaps: true,
      line: { color, width: 2 },
    } as Data;
  });
}

/** The top of each segment of a stack, in stacking order (positives up from 0, negatives down from 0). */
export function stackTops(series: (number | null)[][]): (number | null)[][] {
  const up: number[] = [];
  const down: number[] = [];
  return series.map((values) =>
    values.map((value, j) => {
      if (value === null || value === undefined) return null;
      const side = value >= 0 ? up : down;
      side[j] = (side[j] ?? 0) + value;
      return side[j];
    }),
  );
}

function heatmap(
  input: ChartInput,
  tokens: ChartTokens,
  columns: string[],
  rows: string[],
  find: (column: string, row: string) => ReturnType<ReturnType<typeof seriesIndex>>,
): Built {
  const z = rows.map((row) =>
    columns.map((column) => {
      const s = find(column, row);
      return s ? shownOverall(input.result, s, input.mode) : null;
    }),
  );
  return heatmapGrid(input, tokens, columns, rows, z);
}

/** Sequential one hue, or two hues around a grey 0 for changes (R-TAB-09). */
function heatmapGrid(input: ChartInput, tokens: ChartTokens, x: string[], y: string[], z: (number | null)[][]): Built {
  const diverging = DIVERGING_MODES.has(input.mode);
  const [blue = tokens.ink2, orange = tokens.ink3] = tokens.palette;
  const colorscale: [number, string][] = diverging
    ? [[0, orange], [0.5, tokens.line], [1, blue]]
    : [[0, withAlpha(blue, 0.06)], [1, blue]];
  const extent = Math.max(0, ...z.flat().map((v) => Math.abs(v ?? 0)));
  return {
    data: [
      {
        type: "heatmap",
        x,
        y,
        z: z.map((row) => scaled(input, row)),
        customdata: z.map((row) => text(input, row)) as unknown as string[],
        hovertemplate: `%{y} · %{x}: %{customdata}<extra></extra>`,
        colorscale,
        zmid: diverging ? 0 : undefined,
        zmin: diverging ? -extent * scaleOf(input).factor : undefined,
        zmax: diverging ? extent * scaleOf(input).factor : undefined,
        showscale: false,
        xgap: 2,
        ygap: 2,
      } as Data,
    ],
    layout: { yaxis: { autorange: "reversed", automargin: true, showgrid: false }, xaxis: { type: "category", showgrid: false } },
  };
}

/** The "% of total" chart drawn under the main one, same x axis (R-VIS-22: never a 2nd y axis). */
export function shareFigure(input: ChartInput): (tokens: ChartTokens) => Built {
  return kpiFigure({ ...input, mode: "share_total", chart: input.result.periods.length > 24 ? "stacked_area" : "stacked_bars" });
}

/** Charts the measure can switch to, and whether the result is stacked. */
export const isStacked = (chart: string) => STACKED.has(chart);
