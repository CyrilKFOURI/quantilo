import { useEffect, useRef, useState } from "react";

import { rampTable, type Rgb } from "../../ui/colorRamp";

/** Hex `#rrggbb` → [r, g, b]. */
function rgb(hex: string): [number, number, number] {
  const value = hex.replace("#", "");
  const full = value.length === 3 ? value.replace(/(.)/g, "$1$1") : value;
  const n = Number.parseInt(full.slice(0, 6), 16);
  return Number.isFinite(n) ? [(n >> 16) & 255, (n >> 8) & 255, n & 255] : [235, 104, 52];
}

type MapOverlayProps = {
  /** The run's 8-bit grey map (R-113-10). */
  src: string;
  /** 0–1: how much of the map shows over the image. */
  opacity: number;
  colour: string;
  /** Grey levels under this (0–255) stay transparent: only the unusual areas are tinted. */
  floor?: number;
  className?: string;
  onState?: (state: "loading" | "ready" | "missing") => void;
};

/**
 * 113: an anomaly map laid over its image, in a `<canvas>` (R-113-10). The server sends the map
 * once as grey levels; the colour (one hue of the theme's palette, transparent → opaque with the
 * level), the opacity and the floor are applied here, without another request. 117 replaces it
 * with the full `HeatmapOverlay` (contours, threshold on the map).
 */
export function MapOverlay({ src, opacity, colour, floor = 24, className = "", onState }: MapOverlayProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [grey, setGrey] = useState<{ src: string; data: ImageData } | null>(null);

  useEffect(() => {
    let alive = true;
    const image = new Image();
    image.decoding = "async";
    onState?.("loading");
    image.onload = () => {
      if (!alive) return;
      const scratch = document.createElement("canvas");
      scratch.width = image.naturalWidth;
      scratch.height = image.naturalHeight;
      const context = scratch.getContext("2d");
      if (!context) return;
      context.drawImage(image, 0, 0);
      setGrey({ src, data: context.getImageData(0, 0, scratch.width, scratch.height) });
      onState?.("ready");
    };
    image.onerror = () => alive && onState?.("missing");
    image.src = src;
    return () => {
      alive = false;
    };
    // onState is a callback of the parent: following its identity would reload the map
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src]);

  useEffect(() => {
    const element = canvas.current;
    if (!element || !grey || grey.src !== src) return;
    const { data } = grey;
    element.width = data.width;
    element.height = data.height;
    const context = element.getContext("2d");
    if (!context) return;
    const out = context.createImageData(data.width, data.height);
    const [r, g, b] = rgb(colour);
    for (let i = 0; i < data.data.length; i += 4) {
      const level = data.data[i] ?? 0;
      const shown = level <= floor ? 0 : (level - floor) / (255 - floor);
      out.data[i] = r;
      out.data[i + 1] = g;
      out.data[i + 2] = b;
      out.data[i + 3] = Math.round(255 * opacity * Math.pow(shown, 0.8));
    }
    context.putImageData(out, 0, 0);
  }, [grey, src, opacity, colour, floor]);

  return <canvas ref={canvas} aria-hidden className={`pointer-events-none absolute inset-0 h-full w-full [image-rendering:auto] ${className}`} />;
}

/**
 * 113: a kept map as a picture of its own, on the theme's sequential scale (as the difference of
 * `BeforeAfter`, 110): the `after` panel of `image → map`, or the error of `image → reconstruction
 * → error`. A data URL made once per map and theme; null while it loads or when it is missing.
 */
export function useRampedMap(src: string | null, stops: readonly Rgb[]): string | null {
  const [made, setMade] = useState<{ key: string; url: string } | null>(null);
  const key = `${src ?? ""}|${stops.map((stop) => stop.join(",")).join(";")}`;
  useEffect(() => {
    if (!src) return;
    let alive = true;
    const image = new Image();
    image.decoding = "async";
    image.onload = () => {
      if (!alive) return;
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.drawImage(image, 0, 0);
      const data = context.getImageData(0, 0, canvas.width, canvas.height);
      const table = rampTable(stops);
      for (let i = 0; i < data.data.length; i += 4) {
        const level = data.data[i] ?? 0;
        data.data[i] = table[level * 3] ?? 0;
        data.data[i + 1] = table[level * 3 + 1] ?? 0;
        data.data[i + 2] = table[level * 3 + 2] ?? 0;
        data.data[i + 3] = 255;
      }
      context.putImageData(data, 0, 0);
      setMade({ key, url: canvas.toDataURL("image/png") });
    };
    image.src = src;
    return () => {
      alive = false;
    };
    // `key` stands for `src` and `stops`
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return made && made.key === key ? made.url : null;
}
