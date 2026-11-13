import type { Data, Layout } from "plotly.js-dist-min";

import type { AnalysisType, Chart } from "../api/client";
import { formatCount } from "../lib/format";
import type { ChartTokens } from "./theme";
import { t } from "../lib/i18n";

/** Display rules (docs/VISUALS.md). */
export const HORIZONTAL_LABEL_LENGTH = 12;
export const HORIZONTAL_BAR_COUNT = 8;
const MAX_LABEL = 22;

type Built = { data: Data[]; layout?: Partial<Layout> };
type Bar = { label: string; count: number; muted?: boolean };

export function seriesColor(type: AnalysisType, tokens: ChartTokens): string {
  if (type === "numeric_continuous" || type === "numeric_discrete") return tokens.numeric;
  if (type === "categorical_nominal" || type === "categorical_ordinal") return tokens.categorical;
  if (type === "datetime") return tokens.date;
  if (type === "boolean") return tokens.bool;
  return tokens.text;
}

function truncate(label: string): string {
  return label.length > MAX_LABEL ? `${label.slice(0, MAX_LABEL - 1)}…` : label;
}

function share(count: number, total: number): string {
  if (!total) return "0%";
  const value = (100 * count) / total;
  return value > 0 && value < 1 ? "<1%" : `${Math.round(value)}%`;
}

/** Top N bars plus "Others" and "(missing)", which are drawn muted and never mixed. */
export function categoryBars(chart: Chart, options: { includeMissing?: boolean } = {}): Bar[] {
  const bars: Bar[] = chart.bars.map((bar) => ({ label: bar.label, count: bar.count }));
  if (chart.others && chart.others.count > 0) {
    bars.push({ label: `${t("fig.chart.others")} (${formatCount(chart.others.categories)})`, count: chart.others.count, muted: true });
  }
  if ((options.includeMissing ?? true) && chart.missing > 0) {
    bars.push({ label: t("fig.chart.missing"), count: chart.missing, muted: true });
  }
  return bars;
}

export function isHorizontal(bars: Bar[], ordered: boolean): boolean {
  if (ordered && bars.length <= 31) return bars.some((bar) => bar.label.length > HORIZONTAL_LABEL_LENGTH);
  return bars.length > HORIZONTAL_BAR_COUNT || bars.some((bar) => bar.label.length > HORIZONTAL_LABEL_LENGTH);
}

export function barsFigure(chart: Chart, type: AnalysisType): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const bars = categoryBars(chart);
    const total = chart.rows;
    const horizontal = isHorizontal(bars, chart.ordered);
    const color = seriesColor(type, tokens);
    const labels = bars.map((bar) => truncate(bar.label));
    const counts = bars.map((bar) => bar.count);
    const colors = bars.map((bar) => (bar.muted ? tokens.ink3 : color));
    const custom = bars.map((bar) => [bar.label, share(bar.count, total)]);
    const trace = {
      type: "bar",
      orientation: horizontal ? "h" : "v",
      x: horizontal ? counts : labels,
      y: horizontal ? labels : counts,
      customdata: custom,
      marker: { color: colors, opacity: bars.map((bar) => (bar.muted ? 0.55 : 1)) },
      hovertemplate: `%{customdata[0]}<br>%{${horizontal ? "x" : "y"}:,} ${t("fig.rows")} · %{customdata[1]}<extra></extra>`,
    } as Data;
    const layout: Partial<Layout> = horizontal
      ? {
          yaxis: { autorange: "reversed", showgrid: false, type: "category" },
          xaxis: { showgrid: true },
        }
      : { xaxis: { type: "category" } };
    return { data: [trace], layout };
  };
}

export function pieFigure(chart: Chart): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const bars = categoryBars(chart);
    return {
      data: [
        {
          type: "pie",
          labels: bars.map((bar) => bar.label),
          values: bars.map((bar) => bar.count),
          hole: 0.55,
          sort: false,
          direction: "clockwise",
          marker: {
            colors: bars.map((bar, index) => (bar.muted ? tokens.ink3 : tokens.palette[index % 8])),
            line: { color: tokens.surface, width: 2 },
          },
          // Labels always visible: some palette colors are below 3:1 on the light surface.
          textinfo: "label+percent",
          textposition: "outside",
          textfont: { color: tokens.ink2, size: 11 },
          hovertemplate: `%{label}<br>%{value:,} ${t("fig.rows")} · %{percent}<extra></extra>`,
        } as Data,
      ],
      layout: { margin: { l: 24, r: 24, t: 16, b: 16 } },
    };
  };
}

export function histogramFigure(chart: Chart, type: AnalysisType, unit = t("fig.rows")): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const centers = chart.bins.map((bin) => (bin.start + bin.end) / 2);
    const widths = chart.bins.map((bin) => bin.end - bin.start);
    const ranges = chart.bins.map((bin) => [formatNumber(bin.start), formatNumber(bin.end)]);
    return {
      data: [
        {
          type: "bar",
          x: centers,
          y: chart.bins.map((bin) => bin.count),
          width: widths,
          customdata: ranges,
          marker: { color: seriesColor(type, tokens), line: { color: tokens.surface, width: 1 } },
          hovertemplate: `%{customdata[0]} – %{customdata[1]}<br>%{y:,} ${unit}<extra></extra>`,
        } as Data,
      ],
      layout: { bargap: 0, xaxis: { showgrid: false } },
    };
  };
}

export function boxFigure(chart: Chart, type: AnalysisType): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const box = chart.box;
    if (!box) return { data: [] };
    const color = seriesColor(type, tokens);
    return {
      data: [
        {
          type: "box",
          orientation: "h",
          y: ["values"],
          q1: [box.q1],
          median: [box.median],
          q3: [box.q3],
          lowerfence: [box.lowerfence],
          upperfence: [box.upperfence],
          fillcolor: tokens.surface,
          line: { color, width: 2 },
          hoverinfo: "x",
        } as Data,
        {
          type: "scatter",
          mode: "markers",
          orientation: "h",
          x: box.outlier_values,
          y: box.outlier_values.map(() => "values"),
          marker: { color, size: 6, opacity: 0.6 },
          hovertemplate: `%{x:,}<extra>${t("fig.outlier")}</extra>`,
        } as Data,
      ],
      layout: { yaxis: { visible: false }, xaxis: { showgrid: true } },
    };
  };
}

export function timelineFigure(chart: Chart): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const asLine = chart.bars.length > 60;
    const x = chart.bars.map((point) => point.label);
    const y = chart.bars.map((point) => point.count);
    const trace = asLine
      ? ({
          type: "scatter",
          mode: "lines",
          x,
          y,
          line: { color: tokens.date, width: 2 },
          fill: "tozeroy",
          fillcolor: withAlpha(tokens.date, 0.15),
          hovertemplate: `%{x}<br>%{y:,} ${t("fig.rows")}<extra></extra>`,
        } as Data)
      : ({
          type: "bar",
          x,
          y,
          marker: { color: tokens.date },
          hovertemplate: `%{x}<br>%{y:,} ${t("fig.rows")}<extra></extra>`,
        } as Data);
    // Period labels ("2024-03", "2024-03-18", "2024") are dates: a date axis spaces its ticks.
    return { data: [trace], layout: { xaxis: { type: "date", nticks: 6, tickangle: 0 } } };
  };
}

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude >= 1e6 || magnitude < 0.01)) return value.toExponential(2);
  return value.toLocaleString("en-US", { maximumFractionDigits: magnitude >= 100 ? 0 : 2 });
}

/** "#rrggbb" + alpha → "rgba(r, g, b, a)" (Plotly does not understand color-mix). */
export function withAlpha(hex: string, alpha: number): string {
  const match = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex.trim());
  if (!match) return hex;
  const [r, g, b] = match.slice(1).map((part) => parseInt(part, 16));
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
