import type { Data, Layout } from "plotly.js-dist-min";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

import { PHONE, useMedia } from "../lib/useMedia";
import { baseLayout, plotConfig, readTokens, type ChartTokens } from "./theme";

type PlotlyModule = typeof import("plotly.js-dist-min");
let plotlyPromise: Promise<PlotlyModule> | null = null;

/** plotly.js is large: load it once, never in the initial bundle (main.tsx preloads it when idle). */
export function loadPlotly(): Promise<PlotlyModule> {
  plotlyPromise ??= import("plotly.js-dist-min").then(
    (module) => (module as unknown as { default?: PlotlyModule }).default ?? module,
  );
  return plotlyPromise;
}

/* Charts are drawn one at a time, those on screen first, with a pause between two so the page
   stays responsive. None waits for a scroll: the ones further down follow right after. */
type Draw = { element: Element; run: () => Promise<unknown> };
const draws: Draw[] = [];
const onScreen = new WeakSet<Element>();
let drawing = false;
let watcher: IntersectionObserver | null = null;

function watch(element: Element): () => void {
  if (typeof IntersectionObserver === "undefined") return () => {};
  watcher ??= new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) onScreen.add(entry.target);
        else onScreen.delete(entry.target);
      }
    },
    { rootMargin: "100px" },
  );
  watcher.observe(element);
  return () => {
    watcher?.unobserve(element);
    onScreen.delete(element);
  };
}

function schedule(draw: Draw) {
  const queued = draws.findIndex((d) => d.element === draw.element);
  if (queued >= 0) draws.splice(queued, 1);
  draws.push(draw);
  void drain();
}

async function drain() {
  if (drawing) return;
  drawing = true;
  while (draws.length > 0) {
    const visible = draws.findIndex((d) => onScreen.has(d.element));
    const [draw] = draws.splice(Math.max(visible, 0), 1);
    await draw?.run().catch(() => undefined);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  drawing = false;
}

/** Bumps when the theme changes (data-theme attribute or system preference). */
function useThemeVersion(): number {
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const bump = () => setVersion((v) => v + 1);
    const observer = new MutationObserver(bump);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", bump);
    return () => {
      observer.disconnect();
      media.removeEventListener("change", bump);
    };
  }, []);
  return version;
}

export type Figure = (tokens: ChartTokens) => { data: Data[]; layout?: Partial<Layout> };

export type PlotHandle = { toImage: (filename: string) => Promise<void> };

/** The clicked point: its index in its trace, its x / y, and where the click happened. */
export type PointClick = { index: number; x: unknown; y: unknown; customdata: unknown; clientX: number; clientY: number };

type PlotProps = {
  figure: Figure;
  height: number;
  label: string;
  onPointClick?: (point: PointClick) => void;
};

type PlotlyClick = {
  points: { pointIndex?: number; pointNumber?: number; x: unknown; y: unknown; customdata: unknown }[];
  event: MouseEvent;
};
type ClickableElement = HTMLDivElement & {
  on?: (event: string, handler: (data: PlotlyClick) => void) => void;
  removeAllListeners?: (event: string) => void;
};

/** A Plotly chart themed with the design tokens. `figure` builds traces from the tokens. */
export const Plot = forwardRef<PlotHandle, PlotProps>(function Plot({ figure, height: wanted, label, onPointClick }, ref) {
  // 041: an ordinary chart is 240 px tall at most on a phone; a chart sized by its data
  // (a heatmap, a long ranking: taller than 420 px) keeps its height.
  const phone = useMedia(PHONE);
  const height = phone && wanted <= 420 ? Math.min(wanted, 240) : wanted;
  const container = useRef<HTMLDivElement>(null);
  const themeVersion = useThemeVersion();
  const clickHandler = useRef(onPointClick);
  useEffect(() => {
    clickHandler.current = onPointClick;
  });
  const clickable = Boolean(onPointClick);

  useImperativeHandle(ref, () => ({
    toImage: async (filename) => {
      const Plotly = await loadPlotly();
      if (container.current) {
        await Plotly.downloadImage(container.current, { format: "png", filename, width: 1200, height: 700 });
      }
    },
  }));

  useEffect(() => {
    let cancelled = false;
    const element = container.current;
    if (!element) return;
    const draw = async () => {
      const Plotly = await loadPlotly();
      if (cancelled) return;
      const tokens = readTokens();
      const { data, layout } = figure(tokens);
      const base = baseLayout(tokens);
      await Plotly.react(
        element,
        data,
        {
          ...base,
          ...layout,
          xaxis: { ...base.xaxis, ...layout?.xaxis },
          yaxis: { ...base.yaxis, ...layout?.yaxis },
          height,
        },
        plotConfig,
      ).then(() => {
        const plot = element as ClickableElement;
        plot.removeAllListeners?.("plotly_click");
        if (!clickable) return;
        plot.on?.("plotly_click", (data) => {
          const point = data.points[0];
          if (!point) return;
          clickHandler.current?.({
            index: point.pointIndex ?? point.pointNumber ?? 0,
            x: point.x,
            y: point.y,
            customdata: point.customdata,
            clientX: data.event.clientX,
            clientY: data.event.clientY,
          });
        });
      });
    };
    schedule({ element, run: draw });
    return () => {
      cancelled = true;
    };
  }, [figure, height, themeVersion, clickable]);

  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const unwatch = watch(element);
    return () => {
      unwatch();
      void loadPlotly().then((Plotly) => Plotly.purge(element));
    };
  }, []);

  return (
    <div
      ref={container}
      role="img"
      aria-label={label}
      style={{ height }}
      className={`w-full ${clickable ? "[&_.cursor-crosshair]:cursor-pointer [&_.nsewdrag]:cursor-pointer" : ""}`}
    />
  );
});
