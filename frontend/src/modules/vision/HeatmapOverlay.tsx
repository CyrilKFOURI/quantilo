import { useQuery } from "@tanstack/react-query";
import { useEffect, useId, useMemo, useRef } from "react";

import { readTokens } from "../../charts/theme";
import { t } from "../../lib/i18n";
import { rampGradient } from "../../ui/colorRamp";
import { SegmentedControl } from "../../ui/SegmentedControl";
import type { WhyMap } from "./explainApi";
import { colourise, contourLevels, contourSegments, decodeMap, formatValue, heatStops, rampOf, valueAt, type MapLevels } from "./heatmap";

/**
 * 117 (study § 9.2): a map laid over its image. The server's 8-bit grey picture is coloured here,
 * with its opacity and threshold, or drawn as isolines that leave the image visible; changing any
 * of them asks nothing of the server. `HeatmapLayer` is the drawing (placed by its parent over the
 * part of the image the map covers), `HeatmapControls` the opacity, threshold, mode and legend.
 */

export type HeatMode = "fill" | "spotlight" | "contours" | "off";
export type HeatSettings = { opacity: number; threshold: number; mode: HeatMode };
export const DEFAULT_HEAT: HeatSettings = { opacity: 0.6, threshold: 0.15, mode: "fill" };

export function useMapLevels(png: string | null) {
  return useQuery({
    queryKey: ["vision-map-levels", png],
    queryFn: () => (png ? decodeMap(png) : Promise.resolve(null)),
    enabled: png !== null,
    staleTime: Infinity,
    gcTime: 60_000,
  });
}

/** Where the map sits in its image, as CSS percentages of the image. */
export function boxStyle(box: readonly number[]) {
  const [x0 = 0, y0 = 0, x1 = 1, y1 = 1] = box;
  return { left: `${x0 * 100}%`, top: `${y0 * 100}%`, width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%` };
}

function paint(canvas: HTMLCanvasElement, map: MapLevels, opacity: number, threshold: number, spotlight: boolean) {
  canvas.width = map.width;
  canvas.height = map.height;
  const context = canvas.getContext("2d");
  if (!context) return;
  const image = context.createImageData(map.width, map.height);
  image.data.set(colourise(map, rampOf(readTokens()), opacity, threshold, spotlight));
  context.putImageData(image, 0, 0);
}

/** The map itself, filling its parent (which the caller places with `boxStyle(map.box)`). */
export function HeatmapLayer({ map, settings }: { map: WhyMap; settings: HeatSettings }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const levels = useMapLevels(settings.mode === "off" ? null : map.png);
  const data = levels.data ?? null;
  useEffect(() => {
    if (canvas.current && data && (settings.mode === "fill" || settings.mode === "spotlight")) {
      paint(canvas.current, data, settings.opacity, settings.threshold, settings.mode === "spotlight");
    }
  }, [data, settings.mode, settings.opacity, settings.threshold]);
  const lines = useMemo(
    () => (data && settings.mode === "contours" ? contourLevels(settings.threshold).map((level) => contourSegments(data, level)) : []),
    [data, settings.mode, settings.threshold],
  );
  if (settings.mode === "off" || !data) return null;
  return settings.mode === "fill" || settings.mode === "spotlight" ? (
    <canvas
      ref={canvas}
      aria-hidden
      className="pointer-events-none absolute inset-0 size-full"
      // a grid of occlusion cells stays square-edged; the other maps are smoothed when enlarged
      style={{ imageRendering: map.blocky ? "pixelated" : "auto" }}
    />
  ) : (
    <svg
      aria-hidden
      viewBox={`0 0 ${data.width} ${data.height}`}
      preserveAspectRatio="none"
      className="pointer-events-none absolute inset-0 size-full overflow-visible"
    >
      {lines.map((segments, index) => (
        <g key={index}>
          {/* a pale halo under each line keeps it readable on dark and light parts of the photo */}
          <path d={pathOf(segments)} fill="none" stroke="rgb(255 255 255 / 0.85)" strokeWidth={3.5} vectorEffect="non-scaling-stroke" strokeLinecap="round" />
          <path d={pathOf(segments)} fill="none" stroke={contourColour(index)} strokeWidth={1.75} vectorEffect="non-scaling-stroke" strokeLinecap="round" />
        </g>
      ))}
    </svg>
  );
}

function pathOf(segments: [number, number, number, number][]): string {
  return segments.map(([x1, y1, x2, y2]) => `M${x1.toFixed(2)} ${y1.toFixed(2)}L${x2.toFixed(2)} ${y2.toFixed(2)}`).join("");
}

/** The three isolines take the tint, the hue and the shade of the ramp, low to high. */
function contourColour(index: number): string {
  const stops = heatStops(readTokens());
  const stop = stops[Math.min(stops.length - 1, index)] ?? stops[0] ?? [0, 0, 0];
  return `rgb(${stop.join(" ")})`;
}

type ControlsProps = {
  map: WhyMap;
  settings: HeatSettings;
  onChange: (settings: HeatSettings) => void;
  className?: string;
};

/** Opacity (`[` and `]`), threshold and mode, and the legend in the map's own unit. */
export function HeatmapControls({ map, settings, onChange, className = "" }: ControlsProps) {
  const id = useId();
  const stops = heatStops(readTokens());
  const threshold = valueAt(map.low, map.high, settings.threshold);
  return (
    <div className={`grid min-w-0 grid-cols-1 gap-x-5 gap-y-3 sm:grid-cols-[auto_minmax(0,1fr)_minmax(0,1fr)] sm:items-end ${className}`}>
      <div className="space-y-1">
        <span className="block text-xs text-ink-2">{t("vision.why.heat.mode")}</span>
        <SegmentedControl<HeatMode>
          label={t("vision.why.heat.mode")}
          value={settings.mode}
          onChange={(mode) => onChange({ ...settings, mode })}
          options={[
            { value: "fill", label: t("vision.why.heat.fill") },
            { value: "spotlight", label: t("vision.why.heat.spotlight") },
            { value: "contours", label: t("vision.why.heat.contours") },
            { value: "off", label: t("vision.why.heat.off") },
          ]}
        />
      </div>
      <label className="min-w-0 space-y-1" htmlFor={`${id}-opacity`}>
        <span className="flex items-baseline justify-between gap-2 text-xs text-ink-2">
          <span>
            {t("vision.why.heat.opacity")} <kbd className="ml-1 font-mono text-[10px] text-ink-3 pointer-coarse:hidden">[ ]</kbd>
          </span>
          <span className="text-ink-1 tabular-nums">{Math.round(settings.opacity * 100)} %</span>
        </span>
        <input
          id={`${id}-opacity`}
          type="range"
          min={0}
          max={100}
          step={5}
          value={Math.round(settings.opacity * 100)}
          disabled={settings.mode !== "fill" && settings.mode !== "spotlight"}
          onChange={(event) => onChange({ ...settings, opacity: Number(event.target.value) / 100 })}
          className="h-5 w-full accent-(--accent) disabled:opacity-50 pointer-coarse:h-11"
        />
      </label>
      <label className="min-w-0 space-y-1" htmlFor={`${id}-threshold`}>
        <span className="flex items-baseline justify-between gap-2 text-xs text-ink-2">
          <span>{t("vision.why.heat.threshold")}</span>
          <span className="text-ink-1 tabular-nums">
            {settings.threshold <= 0 ? t("vision.why.heat.everything") : `≥ ${formatValue(threshold)}`}
          </span>
        </span>
        <input
          id={`${id}-threshold`}
          type="range"
          min={0}
          max={95}
          step={5}
          value={Math.round(settings.threshold * 100)}
          disabled={settings.mode === "off"}
          onChange={(event) => onChange({ ...settings, threshold: Number(event.target.value) / 100 })}
          className="h-5 w-full accent-(--accent) disabled:opacity-50 pointer-coarse:h-11"
        />
      </label>
      <div className="min-w-0 sm:col-span-3">
        <div className="flex items-center gap-2 text-[11px] text-ink-3 tabular-nums">
          <span>{formatValue(map.low)}</span>
          <span className="relative h-2 min-w-16 flex-1 rounded-[2px]" style={{ background: rampGradient(stops) }} aria-hidden>
            {settings.threshold > 0 && (
              <span className="absolute inset-y-[-3px] w-px bg-ink-1" style={{ left: `${settings.threshold * 100}%` }} />
            )}
          </span>
          <span>{formatValue(map.high)}</span>
          <span className="truncate text-ink-2">{map.unit}</span>
        </div>
      </div>
    </div>
  );
}

/** `[` and `]` change the opacity while the panel holding the map is open (not while typing). */
export function useOpacityKeys(settings: HeatSettings, onChange: (settings: HeatSettings) => void, active: boolean) {
  const latest = useRef({ settings, onChange });
  useEffect(() => {
    latest.current = { settings, onChange };
  });
  useEffect(() => {
    if (!active) return;
    const onKey = (event: KeyboardEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest("input:not([type=range]), textarea, select, [contenteditable=true]")) return;
      if (event.key !== "[" && event.key !== "]") return;
      const { settings: now, onChange: change } = latest.current;
      const next = Math.min(1, Math.max(0, now.opacity + (event.key === "]" ? 0.1 : -0.1)));
      change({ ...now, opacity: Math.round(next * 20) / 20, mode: now.mode === "off" ? "fill" : now.mode });
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active]);
}

/** An image with its map over it, without zoom (the stages of the sanity test, the Explain tab). */
export function HeatmapOverlay({ src, alt, map, settings = DEFAULT_HEAT, className = "" }: { src: string; alt: string; map: WhyMap | null; settings?: HeatSettings; className?: string }) {
  return (
    <span className={`relative block overflow-hidden rounded-md bg-surface-2 ${className}`}>
      <span className="relative block size-full">
        <img src={src} alt={alt} decoding="async" className="block size-full object-fill" />
        {map && (
          <span className="absolute" style={boxStyle(map.box)}>
            <HeatmapLayer map={map} settings={settings} />
          </span>
        )}
      </span>
      <span aria-hidden className="pointer-events-none absolute inset-0 rounded-md ring-1 ring-ink-1/8 ring-inset" />
    </span>
  );
}
