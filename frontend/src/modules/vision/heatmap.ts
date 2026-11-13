import type { ChartTokens } from "../../charts/theme";
import { parseColor, rampTable, type Rgb } from "../../ui/colorRamp";

/**
 * 117: the arithmetic of a heatmap laid over a photograph (study § 9.2, `HeatmapOverlay`). The
 * server sends an 8-bit grey picture of the map (0 = its minimum, 255 = its maximum); colour,
 * opacity, threshold and contours are all done here, without another call.
 *
 * Colour (dataviz: magnitude = one hue, light to dark): the theme's first series hue, from a light
 * tint to a deep shade. Over a photograph the "background" is the image, not the theme's surface,
 * so the ramp is the same in light and dark themes and its lightness only ever falls; the value is
 * also carried by the opacity (a weak cell lets the image through), so it never rests on hue alone.
 * A pale start washed out over foliage (Beans): the tint keeps three quarters of the hue. The
 * spotlight mode uses no colour at all: it dims what the map leaves out. The accent stays reserved
 * for actions.
 */

export type MapLevels = { levels: Uint8Array; width: number; height: number };

const mix = (a: Rgb, b: Rgb, share: number): Rgb => a.map((value, i) => Math.round(value + ((b[i] ?? 0) - value) * share)) as Rgb;

/** The overlay's ramp: tint -> hue -> shade of the first series colour. */
export function heatStops(tokens: Pick<ChartTokens, "palette" | "ink1">): Rgb[] {
  const hue = parseColor(tokens.palette[0] ?? tokens.ink1);
  return [mix(hue, [255, 255, 255], 0.25), hue, mix(hue, [0, 0, 0], 0.4)];
}

/** Relative luminance (WCAG) of a colour: the ramp's lightness check. */
export function luminance([r, g, b]: Rgb): number {
  const channel = (value: number) => {
    const c = value / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

/** Opacity of a cell: 0 below the threshold, then from a third to all of `opacity`. */
export function alphaOf(level: number, opacity: number, threshold: number): number {
  const v = level / 255;
  if (v < threshold || v <= 0) return 0;
  return Math.round(255 * opacity * (0.35 + 0.65 * v));
}

/** Spotlight: how much a cell is dimmed (black), all of `opacity` below the threshold, none at
 *  the map's maximum. */
export function dimOf(level: number, opacity: number, threshold: number): number {
  const v = level / 255;
  if (v < threshold) return Math.round(255 * opacity);
  return Math.round(255 * opacity * (1 - v));
}

/** RGBA pixels of the map: colour from the ramp, opacity and threshold as above; or, as a
 *  spotlight, black that dims the image where the map is low. */
export function colourise(
  map: MapLevels,
  ramp: Uint8ClampedArray,
  opacity: number,
  threshold: number,
  spotlight = false,
): Uint8ClampedArray {
  const n = map.width * map.height;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i += 1) {
    const level = map.levels[i] ?? 0;
    if (spotlight) {
      out[i * 4 + 3] = dimOf(level, opacity, threshold);
      continue;
    }
    out[i * 4] = ramp[level * 3] ?? 0;
    out[i * 4 + 1] = ramp[level * 3 + 1] ?? 0;
    out[i * 4 + 2] = ramp[level * 3 + 2] ?? 0;
    out[i * 4 + 3] = alphaOf(level, opacity, threshold);
  }
  return out;
}

export function rampOf(tokens: Pick<ChartTokens, "palette" | "ink1">): Uint8ClampedArray {
  return rampTable(heatStops(tokens));
}

/** Grey levels of a decoded picture's RGBA pixels (the red channel of a grey PNG). */
export function levelsFromRgba(pixels: Uint8ClampedArray, width: number, height: number): MapLevels {
  const levels = new Uint8Array(width * height);
  for (let i = 0; i < levels.length; i += 1) levels[i] = pixels[i * 4] ?? 0;
  return { levels, width, height };
}

/** Segments of the isoline at `level` (0-1 of the map's range) by marching squares, in the map's
 *  own cells (a cell's centre is at i + 0.5): `[x1, y1, x2, y2]` each. */
export function contourSegments(map: MapLevels, level: number): [number, number, number, number][] {
  const { width: w, height: h, levels } = map;
  const at = (x: number, y: number) => (levels[y * w + x] ?? 0) / 255;
  const out: [number, number, number, number][] = [];
  const lerp = (a: number, b: number) => (a === b ? 0.5 : (level - a) / (b - a));
  for (let y = 0; y < h - 1; y += 1) {
    for (let x = 0; x < w - 1; x += 1) {
      const tl = at(x, y);
      const tr = at(x + 1, y);
      const br = at(x + 1, y + 1);
      const bl = at(x, y + 1);
      const code = (tl >= level ? 8 : 0) | (tr >= level ? 4 : 0) | (br >= level ? 2 : 0) | (bl >= level ? 1 : 0);
      if (code === 0 || code === 15) continue;
      const cx = x + 0.5;
      const cy = y + 0.5;
      const top: [number, number] = [cx + lerp(tl, tr), cy];
      const right: [number, number] = [cx + 1, cy + lerp(tr, br)];
      const bottom: [number, number] = [cx + lerp(bl, br), cy + 1];
      const left: [number, number] = [cx, cy + lerp(tl, bl)];
      const pairs: [number, number][][] = {
        1: [[left, bottom]],
        2: [[bottom, right]],
        3: [[left, right]],
        4: [[top, right]],
        5: [
          [left, top],
          [bottom, right],
        ],
        6: [[top, bottom]],
        7: [[left, top]],
        8: [[left, top]],
        9: [[top, bottom]],
        10: [
          [left, bottom],
          [top, right],
        ],
        11: [[top, right]],
        12: [[left, right]],
        13: [[bottom, right]],
        14: [[left, bottom]],
      }[code] as [number, number][][];
      for (const [a, b] of pairs) {
        if (a && b) out.push([a[0], a[1], b[0], b[1]]);
      }
    }
  }
  return out;
}

/** The isolines drawn in contour mode: three levels above the threshold. */
export function contourLevels(threshold: number): number[] {
  const start = Math.max(0.2, threshold);
  const step = (1 - start) / 3;
  return [start + step * 0.5, start + step * 1.5, start + step * 2.5].map((value) => Math.round(value * 1000) / 1000);
}

/** A value of the map's unit for the legend: 3 significant digits, no exponent below 1e6. */
export function formatValue(value: number): string {
  if (!Number.isFinite(value)) return "-";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1000) return Math.round(value).toLocaleString("en-US");
  if (magnitude >= 1) return Number(value.toPrecision(3)).toString();
  return Number(value.toPrecision(2)).toString();
}

/** The map value at a share of its range (the threshold, said in the map's own unit). */
export function valueAt(low: number, high: number, share: number): number {
  return low + (high - low) * share;
}

/** Decode the map's PNG into its grey levels (null where canvas is missing: tests). */
export async function decodeMap(src: string): Promise<MapLevels | null> {
  const image = await new Promise<HTMLImageElement>((resolve, reject) => {
    const element = new Image();
    element.onload = () => resolve(element);
    element.onerror = reject;
    element.src = src;
  });
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  if (!width || !height) return null;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return null;
  context.drawImage(image, 0, 0);
  return levelsFromRgba(context.getImageData(0, 0, width, height).data, width, height);
}
