import type { Data, Layout } from "plotly.js-dist-min";

import type { Chart, Law, NumericProfile } from "../../api/client";
import { formatNumber, withAlpha } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";

type Built = { data: Data[]; layout?: Partial<Layout> };
type Bar = { label: string; count: number };

const legend: Partial<Layout> = {
  showlegend: true,
  legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11 } },
};

export function kdeFigure(profile: NumericProfile): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const curve = profile.kde;
    if (!curve) return { data: [] };
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: curve.x,
          y: curve.y,
          line: { color: tokens.numeric, width: 2, shape: "spline" },
          fill: "tozeroy",
          fillcolor: withAlpha(tokens.numeric, 0.15),
          hovertemplate: "%{x:,.4g}<br>density %{y:.3g}<extra></extra>",
        } as Data,
      ],
      layout: { yaxis: { title: { text: "Density", font: { size: 11 } } } },
    };
  };
}

export function ecdfFigure(profile: NumericProfile): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const curve = profile.ecdf;
    if (!curve) return { data: [] };
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: curve.x,
          y: curve.y,
          line: { color: tokens.numeric, width: 2, shape: "hv" },
          hovertemplate: "%{y:.0%} of values ≤ %{x:,.4g}<extra></extra>",
        } as Data,
      ],
      layout: { yaxis: { tickformat: ".0%", range: [0, 1.02] } },
    };
  };
}

/** Sample quantiles against theoretical ones, with the reference line. */
export function qqFigure(
  theoretical: number[],
  sample: number[],
  line: { slope: number; intercept: number },
  axisLabel: string,
): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const ends = [theoretical[0] ?? 0, theoretical.at(-1) ?? 0];
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: ends,
          y: ends.map((x) => line.intercept + line.slope * x),
          line: { color: tokens.ink3, width: 1.5, dash: "dash" },
          hoverinfo: "skip",
        } as Data,
        {
          type: "scatter",
          mode: "markers",
          x: theoretical,
          y: sample,
          marker: { color: tokens.numeric, size: 8, line: { color: tokens.surface, width: 1 } },
          hovertemplate: "theoretical %{x:.3g}<br>observed %{y:,.4g}<extra></extra>",
        } as Data,
      ],
      layout: {
        xaxis: { title: { text: axisLabel, font: { size: 11 } }, showgrid: true },
        yaxis: { title: { text: "Observed quantiles", font: { size: 11 } } },
      },
    };
  };
}

/** Box plot from the exact quartiles; points beyond the whiskers come from the sample. */
export function boxFromProfile(profile: NumericProfile): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const d = profile.describe;
    if (d.q1 === null || d.q3 === null || d.median === null || d.min === null || d.max === null) return { data: [] };
    const spread = d.q3 - d.q1;
    const low = Math.max(d.min, d.q1 - 1.5 * spread);
    const high = Math.min(d.max, d.q3 + 1.5 * spread);
    const outliers = profile.sample.filter((value) => value < low || value > high);
    return {
      data: [
        {
          type: "box",
          orientation: "h",
          y: ["values"],
          q1: [d.q1],
          median: [d.median],
          q3: [d.q3],
          lowerfence: [low],
          upperfence: [high],
          mean: [d.mean ?? d.median],
          fillcolor: tokens.surface,
          line: { color: tokens.numeric, width: 2 },
          hoverinfo: "x",
        } as Data,
        {
          type: "scatter",
          mode: "markers",
          x: outliers,
          y: outliers.map(() => "values"),
          marker: { color: tokens.numeric, size: 6, opacity: 0.5 },
          hovertemplate: "%{x:,}<extra>outlier</extra>",
        } as Data,
      ],
      layout: { yaxis: { visible: false }, xaxis: { showgrid: true } },
    };
  };
}

export function violinFigure(profile: NumericProfile): (tokens: ChartTokens) => Built {
  return (tokens) => ({
    data: [
      {
        type: "violin",
        orientation: "h",
        x: profile.sample,
        y0: "values",
        line: { color: tokens.numeric, width: 2 },
        fillcolor: withAlpha(tokens.numeric, 0.2),
        box: { visible: true, fillcolor: tokens.surface, line: { color: tokens.numeric } },
        meanline: { visible: true, color: tokens.ink2 },
        points: false,
        hoveron: "violins",
      } as Data,
    ],
    layout: { yaxis: { visible: false }, xaxis: { showgrid: true } },
  });
}

/** Histogram with the best laws' densities, scaled to counts per bin. */
export function lawsFigure(chart: Pick<Chart, "bins">, grid: number[], laws: Law[], valueCount: number): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const width = chart.bins[0] ? chart.bins[0].end - chart.bins[0].start : 1;
    const bars: Data = {
      type: "bar",
      name: "Data",
      x: chart.bins.map((bin) => (bin.start + bin.end) / 2),
      y: chart.bins.map((bin) => bin.count),
      width: chart.bins.map((bin) => bin.end - bin.start),
      marker: { color: withAlpha(tokens.ink3, 0.35), line: { color: tokens.surface, width: 1 } },
      customdata: chart.bins.map((bin) => [formatNumber(bin.start), formatNumber(bin.end)]),
      hovertemplate: "%{customdata[0]} – %{customdata[1]}<br>%{y:,} rows<extra></extra>",
    } as Data;
    const curves = laws.map(
      (law, index) =>
        ({
          type: "scatter",
          mode: "lines",
          name: law.label,
          x: grid,
          y: law.density.map((d) => d * valueCount * (law.discrete ? Math.max(1, Math.round(width)) : width)),
          line: { color: tokens.palette[index], width: 2, shape: law.discrete ? "hvh" : "spline" },
          hovertemplate: `${law.label}<br>%{x:,.4g}: %{y:,.0f} expected<extra></extra>`,
        }) as Data,
    );
    return { data: [bars, ...curves], layout: { ...legend, bargap: 0 } };
  };
}

/** Small histogram from {start, end, count} bins (transformation previews). */
export function miniHistogram(bins: { start: number; end: number; count: number }[]): (tokens: ChartTokens) => Built {
  return (tokens) => ({
    data: [
      {
        type: "bar",
        x: bins.map((bin) => (bin.start + bin.end) / 2),
        y: bins.map((bin) => bin.count),
        width: bins.map((bin) => bin.end - bin.start),
        marker: { color: tokens.numeric, line: { color: tokens.surface, width: 1 } },
        hovertemplate: "%{x:.3g}<br>%{y:,}<extra></extra>",
      } as Data,
    ],
    layout: { bargap: 0, xaxis: { showgrid: false, nticks: 4 }, yaxis: { visible: false } },
  });
}

export function horizontalBars(bars: Bar[], color: (tokens: ChartTokens) => string): (tokens: ChartTokens) => Built {
  return (tokens) => ({
    data: [
      {
        type: "bar",
        orientation: "h",
        x: bars.map((bar) => bar.count),
        y: bars.map((bar) => bar.label),
        marker: { color: color(tokens) },
        hovertemplate: "%{y}<br>%{x:,}<extra></extra>",
      } as Data,
    ],
    layout: { yaxis: { autorange: "reversed", type: "category", showgrid: false }, xaxis: { showgrid: true } },
  });
}

/** Rows per day as a GitHub-style calendar: one column per week, Monday on top. */
export function calendarFigure(days: Bar[]): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const first = days[0] ? new Date(`${days[0].label}T00:00:00Z`) : new Date();
    const weekStart = (date: Date) => {
      const copy = new Date(date);
      copy.setUTCDate(copy.getUTCDate() - ((copy.getUTCDay() + 6) % 7));
      return copy;
    };
    const origin = weekStart(first).getTime();
    const x: string[] = [];
    const y: number[] = [];
    const z: number[] = [];
    const text: string[] = [];
    for (const day of days) {
      const date = new Date(`${day.label}T00:00:00Z`);
      const week = Math.floor((weekStart(date).getTime() - origin) / (7 * 86_400_000));
      x.push(new Date(origin + week * 7 * 86_400_000).toISOString().slice(0, 10));
      y.push((date.getUTCDay() + 6) % 7);
      z.push(day.count);
      text.push(day.label);
    }
    return {
      data: [
        {
          type: "heatmap",
          x,
          y,
          z,
          text,
          xgap: 2,
          ygap: 2,
          colorscale: [
            [0, withAlpha(tokens.date, 0.12)],
            [1, tokens.date],
          ],
          showscale: false,
          hovertemplate: "%{text}<br>%{z:,} rows<extra></extra>",
        } as Data,
      ],
      layout: {
        xaxis: { type: "date", showgrid: false, nticks: 8 },
        yaxis: {
          tickvals: [0, 2, 4, 6],
          ticktext: ["Mon", "Wed", "Fri", "Sun"],
          autorange: "reversed",
          showgrid: false,
        },
      },
    };
  };
}
