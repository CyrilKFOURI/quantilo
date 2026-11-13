import type { Data, Layout } from "plotly.js-dist-min";

import type {
  SeasonalProfile,
  SeriesDecomposition,
  SeriesLine,
  SeriesSeasonality,
} from "../../api/client";
import type { Figure } from "../../charts/Plot";
import { withAlpha } from "../../charts/figures";
import { t } from "../../lib/i18n";

const legend: Partial<Layout> = {
  showlegend: true,
  legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11 } },
};
const title = (text: string) => ({ title: { text, font: { size: 11 } } });

/** `bands`: the 95 % bands of the other smoothers and of the periods filled by Kalman (064). */
export type Layers = { values: boolean; mean: boolean; band: boolean; bands?: boolean };

/** A curve of the Smooth row other than the moving average (064), with its 95 % band. */
export type SmoothTrace = {
  name: string;
  values: (number | null)[];
  low: (number | null)[];
  high: (number | null)[];
  /** Index in the palette. */
  color: number;
};

/** The analysed series with its rolling mean and ± 1σ band; other groups faintly behind. */
export function overviewFigure(
  periods: string[],
  analyzed: SeriesLine,
  others: SeriesLine[],
  layers: Layers,
  what: string,
  rolled: string,
  extra: { window: number; mean: (number | null)[]; name?: string }[] = [],
  smoothed: SmoothTrace[] = [],
  filledName = t("fig.ts.filledByKalman"),
): Figure {
  return (tokens) => {
    const color = tokens.palette[0] ?? tokens.date;
    const data: Data[] = others.map(
      (line, i) =>
        ({
          type: "scatter",
          mode: "lines",
          name: line.label ?? "",
          x: periods,
          y: line.values,
          line: {
            color: withAlpha(tokens.palette[(i + 1) % 8] ?? tokens.ink3, 0.6),
            width: 1,
          },
          hovertemplate: `${line.label}<br>%{x}: %{y:,.4g}<extra></extra>`,
        }) as Data,
    );
    const rolling = analyzed.rolling_mean.length > 0;
    if (rolling && layers.band) {
      const edge = (sign: number) =>
        analyzed.rolling_mean.map((m, i) => {
          const s = analyzed.rolling_std[i];
          return m === null || s === null || s === undefined
            ? null
            : m + sign * s;
        });
      data.push(
        {
          type: "scatter",
          mode: "lines",
          x: periods,
          y: edge(-1),
          line: { width: 0 },
          hoverinfo: "skip",
          showlegend: false,
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          name: t("fig.band1Sigma"),
          x: periods,
          y: edge(1),
          fill: "tonexty",
          fillcolor: withAlpha(color, 0.15),
          line: { width: 0 },
          hoverinfo: "skip",
        } as Data,
      );
    }
    if (layers.values || !rolling) {
      data.push({
        type: "scatter",
        mode: periods.length > 60 ? "lines" : "lines+markers",
        name: analyzed.label ? `${analyzed.label} · ${what}` : what,
        x: periods,
        y: analyzed.values,
        customdata: analyzed.counts,
        line: {
          color: rolling && layers.mean ? withAlpha(color, 0.55) : color,
          width: 1.25,
        },
        marker: { size: 4 },
        hovertemplate: `%{x}<br>${what}: %{y:,.4g}<br>%{customdata:,} ${t("fig.rows")}<extra></extra>`,
      } as Data);
    }
    if (rolling && layers.mean) {
      data.push({
        type: "scatter",
        mode: "lines",
        name: rolled,
        x: periods,
        y: analyzed.rolling_mean,
        line: { color, width: 2.5 },
        hovertemplate: `%{x}<br>${rolled}: %{y:,.4g}<extra></extra>`,
      } as Data);
    }
    // the extra means follow the "Rolling means" layer, like the first one
    (layers.mean ? extra : []).forEach((line, i) => {
      const name = line.name ?? t("fig.ts.periodMean", { window: String(line.window) });
      data.push({
        type: "scatter",
        mode: "lines",
        name,
        x: periods,
        y: line.mean,
        line: { color: tokens.palette[(i + 2) % 8] ?? tokens.ink2, width: 2 },
        hovertemplate: `%{x}<br>${name}: %{y:,.4g}<extra></extra>`,
      } as Data);
    });
    const bands = layers.bands ?? true;
    for (const curve of smoothed) {
      const tint = tokens.palette[curve.color % 8] ?? tokens.ink2;
      if (bands && curve.low.some((v) => v !== null)) {
        data.push(
          {
            type: "scatter",
            mode: "lines",
            x: periods,
            y: curve.low,
            line: { width: 0 },
            hoverinfo: "skip",
            showlegend: false,
          } as Data,
          {
            type: "scatter",
            mode: "lines",
            name: `${curve.name} · 95 %`,
            x: periods,
            y: curve.high,
            fill: "tonexty",
            fillcolor: withAlpha(tint, 0.12),
            line: { width: 0 },
            hoverinfo: "skip",
            showlegend: false,
          } as Data,
        );
      }
      data.push({
        type: "scatter",
        mode: "lines",
        name: curve.name,
        x: periods,
        y: curve.values,
        line: { color: tint, width: 2 },
        hovertemplate: `%{x}<br>${curve.name}: %{y:,.4g}<extra></extra>`,
      } as Data);
    }
    // Periods filled by Kalman: hollow points with their 95 % band as error bars.
    const filled = (analyzed.filled_low ?? []).flatMap((low, i) => {
      const high = analyzed.filled_high?.[i];
      const value = analyzed.values[i];
      return low !== null && high !== null && high !== undefined && value !== null && value !== undefined
        ? [{ i, low, high, value }]
        : [];
    });
    if (filled.length) {
      data.push({
        type: "scatter",
        mode: "markers",
        name: filledName,
        x: filled.map((p) => periods[p.i]),
        y: filled.map((p) => p.value),
        marker: { size: 7, color: tokens.surface, line: { color, width: 1.5 } },
        error_y: bands
          ? {
              type: "data",
              symmetric: false,
              array: filled.map((p) => p.high - p.value),
              arrayminus: filled.map((p) => p.value - p.low),
              color: withAlpha(color, 0.6),
              thickness: 1,
              width: 3,
            }
          : { visible: false },
        hovertemplate: `%{x}<br>${filledName}: %{y:,.4g}<extra></extra>`,
      } as Data);
    }
    return {
      data,
      layout: {
        ...legend,
        hovermode: "x unified",
        xaxis: {
          type: "date",
          rangeslider: { visible: periods.length > 30, thickness: 0.08 },
        },
        yaxis: title(what),
      },
    };
  };
}

/** Box per calendar label (Mon…Sun, Jan…Dec), the strongest label highlighted. */
export function profileFigure(profile: SeasonalProfile, what: string): Figure {
  return (tokens) => ({
    data: profile.labels.map(
      (label, i) =>
        ({
          type: "box",
          name: label,
          y: profile.values[i],
          boxpoints: false,
          marker: {
            color:
              label === profile.strongest
                ? (tokens.palette[0] ?? tokens.numeric)
                : tokens.ink3,
          },
          line: { width: 1.25 },
          hovertemplate: `${label}<br>%{y:,.4g}<br>${
            profile.deviation[i] === null
              ? ""
              : t("fig.ts.vsAverage", { value: `${(profile.deviation[i] ?? 0) >= 0 ? "+" : ""}${(profile.deviation[i] ?? 0).toFixed(1)}` })
          }<extra></extra>`,
        }) as Data,
    ),
    layout: {
      showlegend: false,
      xaxis: { type: "category" },
      yaxis: title(what),
    },
  });
}

/** One line per year, the latest in the accent color. */
export function overlayFigure(
  overlay: NonNullable<SeriesSeasonality["overlay"]>,
  what: string,
): Figure {
  return (tokens) => {
    const latest = overlay.years.at(-1)?.year;
    return {
      data: overlay.years.map(
        (line) =>
          ({
            type: "scatter",
            mode: "lines",
            name: String(line.year),
            x: line.x,
            y: line.y,
            line: {
              color:
                line.year === latest
                  ? (tokens.palette[0] ?? tokens.date)
                  : withAlpha(tokens.ink3, 0.55),
              width: line.year === latest ? 2.5 : 1.25,
            },
            hovertemplate: `${line.year}<br>${overlay.unit} %{x}: %{y:,.4g}<extra></extra>`,
          }) as Data,
      ),
      layout: { ...legend, xaxis: title(overlay.unit), yaxis: title(what) },
    };
  };
}

/** Observed, trend, seasonal component(s) and remainder, stacked on one time axis. */
export function decompositionFigure(
  periods: string[],
  observed: (number | null)[],
  parts: NonNullable<SeriesDecomposition["result"]>,
): Figure {
  return (tokens) => {
    const rows: {
      name: string;
      y: (number | null)[];
      color: string;
      bars?: boolean;
    }[] = [
      { name: t("fig.tail.observed"), y: observed, color: tokens.ink2 },
      {
        name: t("fig.ts.trend"),
        y: parts.trend,
        color: tokens.palette[0] ?? tokens.date,
      },
      ...parts.seasonal.map((y, i) => ({
        name:
          parts.seasonal.length > 1
            ? t("fig.ts.seasonalOf", { season: String(parts.seasons[i]) })
            : t("fig.ts.seasonal"),
        y,
        color: tokens.palette[1 + i] ?? tokens.numeric,
      })),
      { name: t("fig.ts.remainder"), y: parts.resid, color: tokens.ink3, bars: true },
    ];
    const gap = 0.04;
    const size = (1 - gap * (rows.length - 1)) / rows.length;
    const layout: Partial<Layout> & Record<string, unknown> = {
      showlegend: false,
      hovermode: "x unified",
    };
    const data: Data[] = rows.map((row, i) => {
      const axis = i === 0 ? "" : String(i + 1);
      const top = 1 - i * (size + gap);
      layout[`yaxis${axis}`] = {
        domain: [top - size, top],
        ...title(row.name),
        zeroline: row.bars ?? false,
      };
      return {
        type: row.bars ? "bar" : "scatter",
        mode: "lines",
        name: row.name,
        x: periods,
        y: row.y,
        xaxis: "x",
        yaxis: `y${axis}` as "y",
        line: { color: row.color, width: 1.5 },
        marker: { color: row.color },
        hovertemplate: `${row.name}: %{y:,.4g}<extra></extra>`,
      } as Data;
    });
    layout.xaxis = { type: "date", anchor: `y${rows.length}` as "y" };
    return { data, layout };
  };
}

/** One component of a structural fit, as the components chart draws it (028 A6 story 4). */
export type ComponentRow = {
  name: string;
  kind: "trend" | "season" | "holidays" | "driver";
  values: number[];
};

/** Trend, each season, the holidays and each driver of a Prophet-like fit, stacked on one time
 * axis over the fitted history and the forecast; a dotted line marks the forecast's start.
 * Holidays are bars (one spike per holiday), the rest lines. */
export function componentsFigure(
  periods: string[],
  future: string[],
  rows: ComponentRow[],
): Figure {
  return (tokens) => {
    const x = [...periods, ...future];
    const gap = 0.05;
    const size = (1 - gap * (rows.length - 1)) / rows.length;
    const layout: Partial<Layout> & Record<string, unknown> = {
      showlegend: false,
      hovermode: "x unified",
      bargap: 0,
    };
    const data: Data[] = rows.map((row, i) => {
      const axis = i === 0 ? "" : String(i + 1);
      const top = 1 - i * (size + gap);
      const color =
        row.kind === "trend"
          ? (tokens.palette[0] ?? tokens.date)
          : (tokens.palette[i % 8] ?? tokens.numeric);
      layout[`yaxis${axis}`] = {
        domain: [Math.max(0, top - size), top],
        ...title(row.name),
        zeroline: row.kind !== "trend",
      };
      const bars = row.kind === "holidays";
      return {
        type: bars ? "bar" : "scatter",
        mode: "lines",
        name: row.name,
        x,
        y: row.values,
        xaxis: "x",
        yaxis: `y${axis}` as "y",
        line: { color, width: 1.5 },
        marker: { color },
        hovertemplate: `${row.name}: %{y:,.4g}<extra></extra>`,
      } as Data;
    });
    layout.xaxis = { type: "date", anchor: `y${rows.length}` as "y" };
    const start = future[0];
    if (start)
      layout.shapes = [
        {
          type: "line",
          xref: "x",
          yref: "paper",
          x0: start,
          x1: start,
          y0: 0,
          y1: 1,
          line: { color: tokens.ink3, width: 1, dash: "dot" },
        },
      ];
    return { data, layout };
  };
}
