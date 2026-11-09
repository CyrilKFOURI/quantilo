import { Maximize2 } from "lucide-react";
import type { Config, Data, Layout } from "plotly.js-dist-min";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

import { t } from "../lib/i18n";
import { PHONE, TOUCH, useMedia } from "../lib/useMedia";
import { Dialog } from "../ui/Dialog";
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

/** Bumps when the theme or the palette changes (data-theme / data-palette, or system preference). */
function useThemeVersion(): number {
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const bump = () => setVersion((v) => v + 1);
    const observer = new MutationObserver(bump);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "data-palette"] });
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
  /** On a touch screen (044). `button` (default): still, with a button that opens it full screen.
   *  `none`: still, without the button (its card already opens full screen). `explore`: already
   *  full screen, so it pans and zooms in place. */
  zoom?: "button" | "none" | "explore";
};

/** How a chart answers a finger (044). `fixed`: inside a page, it lets the page scroll (no drag,
 *  no zoom; a tap still shows the value). `explore`: full screen, one finger pans, the +/- buttons
 *  zoom. A 3D chart always turns under the finger, as on a laptop. */
type TouchMode = "none" | "fixed" | "explore";

const THREE_D = new Set(["scatter3d", "surface", "mesh3d", "cone", "streamtube", "volume", "isosurface"]);

export function isThreeD(data: Data[]): boolean {
  return data.some((trace) => THREE_D.has(String((trace as { type?: string }).type)));
}

function touchLayout(mode: TouchMode): Partial<Layout> {
  if (mode === "fixed") return { dragmode: false };
  if (mode === "explore") return { dragmode: "pan" };
  return {};
}

function touchConfig(mode: TouchMode): Partial<Config> {
  if (mode !== "explore") return plotConfig;
  return {
    ...plotConfig,
    displayModeBar: true,
    modeBarButtons: [["zoomIn2d", "zoomOut2d", "autoScale2d"]],
    showTips: false,
  };
}

type PlotlyClick = {
  points: { pointIndex?: number; pointNumber?: number; x: unknown; y: unknown; customdata: unknown }[];
  event: MouseEvent;
};
type ClickableElement = HTMLDivElement & {
  on?: (event: string, handler: (data: PlotlyClick) => void) => void;
  removeAllListeners?: (event: string) => void;
};

/** A Plotly chart themed with the design tokens. `figure` builds traces from the tokens. On a touch
 *  screen (044) a 2D chart stays still so the page scrolls, with a button that opens it full screen
 *  to pan and zoom. */
export const Plot = forwardRef<PlotHandle, PlotProps>(function Plot(props, ref) {
  const touch = useMedia(TOUCH);
  const [expanded, setExpanded] = useState(false);
  const [threeD, setThreeD] = useState(false);
  const zoom = props.zoom ?? "button";
  const mode: TouchMode = !touch || threeD ? "none" : zoom === "explore" ? "explore" : "fixed";
  return (
    <div className="relative">
      <Chart {...props} ref={ref} touchMode={mode} onDrawn={setThreeD} />
      {mode === "fixed" && zoom === "button" && (
        <button
          type="button"
          onClick={() => setExpanded(true)}
          aria-label={t("chart.expand")}
          title={t("chart.expand")}
          className="absolute top-0 right-0 z-[1] inline-flex size-7 items-center justify-center rounded-md border border-line bg-surface-1/80 text-ink-3 backdrop-blur"
        >
          <Maximize2 size={14} strokeWidth={1.5} aria-hidden />
        </button>
      )}
      {expanded && (
        <Dialog open onOpenChange={(open) => !open && setExpanded(false)} title={props.label} size="lg">
          <Chart
            figure={props.figure}
            label={props.label}
            height={Math.max(320, Math.round(window.innerHeight * 0.7))}
            onPointClick={props.onPointClick}
            touchMode="explore"
            full
          />
          <p className="mt-2 text-xs text-ink-3">{t("chart.exploreHint")}</p>
        </Dialog>
      )}
    </div>
  );
});

type ChartProps = PlotProps & {
  touchMode: TouchMode;
  /** Full screen: the height is kept as asked (no phone cap). */
  full?: boolean;
  /** After each drawing: whether the figure is 3D. */
  onDrawn?: (threeD: boolean) => void;
};

const Chart = forwardRef<PlotHandle, ChartProps>(function Chart(
  { figure, height: wanted, label, onPointClick, touchMode, full = false, onDrawn },
  ref,
) {
  // 041: an ordinary chart is 240 px tall at most on a phone; a chart sized by its data
  // (a heatmap, a long ranking: taller than 420 px) keeps its height.
  const phone = useMedia(PHONE);
  const height = phone && !full && wanted <= 420 ? Math.min(wanted, 240) : wanted;
  const drawn = useRef(onDrawn);
  useEffect(() => {
    drawn.current = onDrawn;
  });
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
      const threeD = isThreeD(data);
      const mode = threeD ? "none" : touchMode;
      const still = mode === "fixed" ? { fixedrange: true } : {};
      await Plotly.react(
        element,
        data,
        {
          ...base,
          ...layout,
          ...touchLayout(mode),
          xaxis: { ...base.xaxis, ...layout?.xaxis, ...still },
          yaxis: { ...base.yaxis, ...layout?.yaxis, ...still },
          height,
        },
        touchConfig(mode),
      ).then(() => {
        drawn.current?.(threeD);
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
  }, [figure, height, themeVersion, clickable, touchMode]);

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
      className={`w-full ${touchMode === "fixed" ? "touch-pan-y" : ""} ${clickable ? "[&_.cursor-crosshair]:cursor-pointer [&_.nsewdrag]:cursor-pointer" : ""}`}
    />
  );
});
