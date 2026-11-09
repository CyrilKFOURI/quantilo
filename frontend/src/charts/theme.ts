import type { Config, Layout } from "plotly.js-dist-min";

import { paletteById } from "../lib/palettes";

/** Colors read from the CSS tokens (docs/UI_UX.md), so charts follow the light/dark theme. */
export type ChartTokens = {
  ink1: string;
  ink2: string;
  ink3: string;
  line: string;
  surface: string;
  accent: string;
  numeric: string;
  categorical: string;
  date: string;
  bool: string;
  text: string;
  id: string;
  font: string;
  /** Categorical palette, fixed order (--chart-1…8, or the whole palette chosen in Settings, 047). */
  palette: string[];
};

const TOKEN_NAMES: Record<Exclude<keyof ChartTokens, "palette">, string> = {
  ink1: "--ink-1",
  ink2: "--ink-2",
  ink3: "--ink-3",
  line: "--line",
  surface: "--surface-1",
  accent: "--accent",
  numeric: "--type-numeric",
  categorical: "--type-categorical",
  date: "--type-date",
  bool: "--type-bool",
  text: "--type-text",
  id: "--type-id",
  font: "--font-ui",
};

export function readTokens(root: HTMLElement = document.documentElement): ChartTokens {
  const style = getComputedStyle(root);
  const read = (name: string) => style.getPropertyValue(name).trim();
  const named = Object.fromEntries(
    Object.entries(TOKEN_NAMES).map(([key, name]) => [key, read(name)]),
  ) as Omit<ChartTokens, "palette">;
  const chosen = paletteById(root.dataset.palette);
  return { ...named, palette: chosen ? [...chosen.colors] : Array.from({ length: 8 }, (_, i) => read(`--chart-${i + 1}`)) };
}

export function baseLayout(tokens: ChartTokens): Partial<Layout> {
  const axis = {
    color: tokens.ink3,
    gridcolor: tokens.line,
    linecolor: tokens.line,
    zerolinecolor: tokens.line,
    tickfont: { size: 11, color: tokens.ink3 },
    automargin: true,
    fixedrange: true,
  };
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: tokens.font, size: 12, color: tokens.ink2 },
    margin: { l: 8, r: 8, t: 4, b: 8 },
    showlegend: false,
    bargap: 0.12,
    hoverlabel: {
      bgcolor: tokens.surface,
      bordercolor: tokens.line,
      font: { family: tokens.font, size: 12, color: tokens.ink1 },
    },
    xaxis: { ...axis, showgrid: false },
    yaxis: { ...axis, showgrid: true },
  };
}

export const plotConfig: Partial<Config> = {
  displaylogo: false,
  responsive: true,
  displayModeBar: false,
};
