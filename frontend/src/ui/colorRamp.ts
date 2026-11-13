import type { ChartTokens } from "../charts/theme";

/** A colour as [r, g, b], 0-255. */
export type Rgb = [number, number, number];

/** "#rgb", "#rrggbb" or "rgb(r, g, b)" to [r, g, b]; black when it cannot be read. */
export function parseColor(value: string): Rgb {
  const text = value.trim();
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(text);
  if (hex) {
    const digits = hex[1] ?? "";
    const full = digits.length === 3 ? [...digits].map((d) => d + d).join("") : digits;
    return [0, 2, 4].map((at) => parseInt(full.slice(at, at + 2), 16)) as Rgb;
  }
  const rgb = /^rgba?\(\s*(\d+)[\s,]+(\d+)[\s,]+(\d+)/i.exec(text);
  if (rgb) return [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])];
  return [0, 0, 0];
}

/** 256 colours along the stops (linear in sRGB between neighbours), as a flat RGB table. */
export function rampTable(stops: readonly Rgb[]): Uint8ClampedArray {
  const table = new Uint8ClampedArray(256 * 3);
  const segments = Math.max(1, stops.length - 1);
  for (let i = 0; i < 256; i += 1) {
    const position = (i / 255) * segments;
    const at = Math.min(segments - 1, Math.floor(position));
    const f = position - at;
    const a = stops[at] ?? [0, 0, 0];
    const b = stops[at + 1] ?? a;
    for (let c = 0; c < 3; c += 1) table[i * 3 + c] = Math.round((a[c] ?? 0) + ((b[c] ?? 0) - (a[c] ?? 0)) * f);
  }
  return table;
}

/** The theme's sequential scale (047 palette, one hue): from the surface towards the first series
 *  colour, then the ink, so "more" is always further from the background, in light and dark themes
 *  alike. `floor` starts the scale that far towards the series colour (0: the surface itself, for a
 *  difference image where "no change" is the background; 0.3 for marks that must stay visible). */
export function sequentialStops(tokens: Pick<ChartTokens, "surface" | "palette" | "ink1">, floor = 0): Rgb[] {
  const surface = parseColor(tokens.surface);
  const hue = parseColor(tokens.palette[0] ?? tokens.ink1);
  const start = surface.map((value, i) => Math.round(value + ((hue[i] ?? 0) - value) * floor)) as Rgb;
  return [start, hue, parseColor(tokens.ink1)];
}

/** CSS gradient of a ramp, for its legend bar. */
export function rampGradient(stops: readonly Rgb[]): string {
  const parts = stops.map((stop, i) => `rgb(${stop.join(" ")}) ${Math.round((i / Math.max(1, stops.length - 1)) * 100)}%`);
  return `linear-gradient(to right, ${parts.join(", ")})`;
}
