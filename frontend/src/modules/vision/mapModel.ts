import type { ImageMap } from "./api";
import { LOG_MEASURES, MEASURES, OTHER, type ImageRow, type Measure } from "./model";

/** 109: what colours the map (R-VMAP-11). Categories take the theme's palette by name (8 slots,
 *  then "Other"); a measure takes the sequential scale. */
export const COLOUR_BY = ["label", "cluster", "quality", "split", ...MEASURES] as const;
export type ColourBy = (typeof COLOUR_BY)[number];

export const NONE = "__none__";

export type LegendEntry = {
  key: string;
  /** Palette slot 0 to 7; -1: the neutral colour (Other, none). */
  slot: number;
  count: number;
};

export type Colouring =
  | { kind: "categorical"; entries: LegendEntry[]; keyOf: (string | null)[] /* per drawn point */ }
  | { kind: "sequential"; measure: Measure; min: number; max: number; log: boolean; level: (number | null)[] /* 0-1 per point */ };

function categoryOf(row: ImageRow | undefined, cluster: number | undefined, by: ColourBy): string {
  if (!row) return NONE;
  switch (by) {
    case "label":
      return row.label ?? NONE;
    case "split":
      return row.split ?? NONE;
    case "cluster":
      return cluster === undefined || cluster < 0 ? NONE : String(cluster);
    case "quality":
      return row.issues[0] ?? NONE;
    default:
      return NONE;
  }
}

/** Categories sorted by name (clusters by number), the first 8 coloured, the rest "Other";
 *  "none" is always neutral. Colour follows the category, never its rank or count. */
export function colouring(rows: readonly ImageRow[], map: ImageMap, by: ColourBy): Colouring {
  if ((MEASURES as readonly string[]).includes(by)) {
    const measure = by as Measure;
    const log = LOG_MEASURES.has(measure);
    const values = map.rows.map((row) => {
      const value = rows[row]?.[measure];
      return value === null || value === undefined || !Number.isFinite(value) ? null : log ? Math.log10(1 + Math.max(0, value)) : value;
    });
    const finite = values.filter((v): v is number => v !== null);
    const min = finite.length ? Math.min(...finite) : 0;
    const max = finite.length ? Math.max(...finite) : 1;
    const span = max - min || 1;
    return {
      kind: "sequential",
      measure,
      log,
      min: log ? 10 ** min - 1 : min,
      max: log ? 10 ** max - 1 : max,
      level: values.map((v) => (v === null ? null : (v - min) / span)),
    };
  }
  const raw = map.rows.map((row) => categoryOf(rows[row], map.clusters[row], by));
  const counts = new Map<string, number>();
  for (const key of raw) counts.set(key, (counts.get(key) ?? 0) + 1);
  const named = [...counts.keys()].filter((key) => key !== NONE);
  named.sort((a, b) => (by === "cluster" ? Number(a) - Number(b) : a.localeCompare(b)));
  const coloured = named.length <= 8 ? named : named.slice(0, 7);
  const slot = new Map(coloured.map((key, i) => [key, i]));
  const keyOf = raw.map((key) => (key === NONE ? NONE : slot.has(key) ? key : OTHER));
  const entries: LegendEntry[] = coloured.map((key, i) => ({ key, slot: i, count: counts.get(key) ?? 0 }));
  const others = named.length - coloured.length;
  if (others > 0) {
    entries.push({ key: OTHER, slot: -1, count: named.slice(7).reduce((sum, key) => sum + (counts.get(key) ?? 0), 0) });
  }
  if (counts.has(NONE)) entries.push({ key: NONE, slot: -1, count: counts.get(NONE) ?? 0 });
  return { kind: "categorical", entries, keyOf };
}

/** Every table row of a legend entry (clusters cover undrawn rows too, R-VMAP-08). */
export function rowsOfEntry(rows: readonly ImageRow[], map: ImageMap, by: ColourBy, entry: string, colouringKeys: readonly string[]): number[] {
  const named = new Set(colouringKeys.filter((key) => key !== OTHER && key !== NONE));
  const out: number[] = [];
  rows.forEach((row, i) => {
    const key = categoryOf(row, map.clusters[i], by);
    if (by === "cluster" && (map.clusters[i] ?? -1) < 0 && key !== NONE) return;
    const shown = key === NONE ? NONE : named.has(key) ? key : OTHER;
    if (shown === entry) out.push(i);
  });
  return out;
}

// --- geometry ------------------------------------------------------------------------------------

export type Bounds = { minX: number; maxX: number; minY: number; maxY: number };

export function boundsOf(xs: readonly number[], ys: readonly number[]): Bounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < xs.length; i += 1) {
    const x = xs[i] ?? 0;
    const y = ys[i] ?? 0;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (!Number.isFinite(minX)) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  if (maxX === minX) maxX = minX + 1;
  if (maxY === minY) maxY = minY + 1;
  return { minX, maxX, minY, maxY };
}

/** Zoom and pan of the map: screen = base(world) * scale + offset. */
export type MapView = { scale: number; x: number; y: number };
export const START: MapView = { scale: 1, x: 0, y: 0 };
export const MAX_MAP_ZOOM = 24;

/** World to screen, before the view: the points fill the box with a margin; y grows upwards. */
export function baseTransform(bounds: Bounds, width: number, height: number, margin = 24) {
  const w = Math.max(1, width - 2 * margin);
  const h = Math.max(1, height - 2 * margin);
  const sx = w / (bounds.maxX - bounds.minX);
  const sy = h / (bounds.maxY - bounds.minY);
  return {
    x: (value: number) => margin + (value - bounds.minX) * sx,
    y: (value: number) => margin + (bounds.maxY - value) * sy,
  };
}

/** The view after zooming by `factor` around the screen point (px, py), kept within limits. */
export function zoomView(view: MapView, factor: number, px: number, py: number): MapView {
  const scale = Math.min(MAX_MAP_ZOOM, Math.max(1, view.scale * factor));
  const ratio = scale / view.scale;
  if (scale === 1) return START;
  return { scale, x: px - (px - view.x) * ratio, y: py - (py - view.y) * ratio };
}

/** R-VMAP-07: the side of a drawn thumbnail at this zoom; thumbnails replace points from 28 px. */
export const THUMB_FROM = 28;
export const THUMBS_MAX = 400;
export function thumbSide(scale: number, forced: boolean): number {
  const side = Math.min(72, 12 * scale);
  return forced ? Math.max(THUMB_FROM, side) : side;
}

/** Ray casting: whether (x, y) lies inside the polygon. */
export function insidePolygon(x: number, y: number, polygon: readonly { x: number; y: number }[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i];
    const b = polygon[j];
    if (!a || !b) continue;
    if (a.y > y !== b.y > y && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y || 1e-12) + a.x) inside = !inside;
  }
  return inside;
}

/** The nearest drawn point to (x, y) within `radius` px: its index in the map's lists, or -1. */
export function nearest(xs: Float32Array, ys: Float32Array, x: number, y: number, radius: number, keep?: (i: number) => boolean): number {
  let best = -1;
  let bestDistance = radius * radius;
  for (let i = 0; i < xs.length; i += 1) {
    if (keep && !keep(i)) continue;
    const dx = (xs[i] ?? 0) - x;
    const dy = (ys[i] ?? 0) - y;
    const d = dx * dx + dy * dy;
    if (d <= bestDistance) {
      bestDistance = d;
      best = i;
    }
  }
  return best;
}
