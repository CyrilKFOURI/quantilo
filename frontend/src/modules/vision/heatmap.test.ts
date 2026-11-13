import { describe, expect, it } from "vitest";

import { rampTable } from "../../ui/colorRamp";
import { alphaOf, colourise, contourLevels, contourSegments, dimOf, formatValue, heatStops, levelsFromRgba, luminance, valueAt, type MapLevels } from "./heatmap";

const tokens = { palette: ["#2a78d6", "#eb6834"], ink1: "#24201b" };

const map = (width: number, height: number, values: number[]): MapLevels => ({ width, height, levels: Uint8Array.from(values) });

describe("heatmap (117)", () => {
  it("ramps one hue from a tint to a shade, its lightness only falling", () => {
    const stops = heatStops(tokens);
    expect(stops).toHaveLength(3);
    const [tint, hue, shade] = stops.map(luminance) as [number, number, number];
    expect(tint).toBeGreaterThan(hue);
    expect(hue).toBeGreaterThan(shade);
    const table = rampTable(stops);
    const lum = Array.from({ length: 256 }, (_, i) => luminance([table[i * 3] ?? 0, table[i * 3 + 1] ?? 0, table[i * 3 + 2] ?? 0]));
    for (let i = 1; i < 256; i += 1) expect(lum[i]).toBeLessThanOrEqual((lum[i - 1] ?? 0) + 1e-9);
  });

  it("hides what is under the threshold and scales the rest with the opacity", () => {
    expect(alphaOf(0, 1, 0)).toBe(0);
    expect(alphaOf(20, 1, 0.5)).toBe(0);
    expect(alphaOf(255, 1, 0.5)).toBe(255);
    expect(alphaOf(255, 0.5, 0)).toBe(128);
    expect(alphaOf(128, 1, 0)).toBeGreaterThan(alphaOf(64, 1, 0));
  });

  it("spotlight dims the low cells and leaves the maximum clear", () => {
    expect(dimOf(255, 0.8, 0)).toBe(0);
    expect(dimOf(10, 0.8, 0.5)).toBe(204);
    expect(dimOf(128, 1, 0)).toBeGreaterThan(dimOf(200, 1, 0));
  });

  it("colours every cell from the ramp", () => {
    const ramp = rampTable(heatStops(tokens));
    const out = colourise(map(2, 1, [0, 255]), ramp, 1, 0);
    expect(Array.from(out.slice(0, 4))).toEqual([ramp[0], ramp[1], ramp[2], 0]);
    expect(Array.from(out.slice(4, 8))).toEqual([ramp[765], ramp[766], ramp[767], 255]);
    const dim = colourise(map(2, 1, [0, 255]), ramp, 1, 0, true);
    expect(Array.from(dim)).toEqual([0, 0, 0, 255, 0, 0, 0, 0]);
  });

  it("draws an isoline around a bright cell (marching squares)", () => {
    const peak = map(3, 3, [0, 0, 0, 0, 255, 0, 0, 0, 0]);
    const segments = contourSegments(peak, 0.5);
    expect(segments).toHaveLength(4);
    for (const [x1, y1, x2, y2] of segments) {
      for (const [x, y] of [
        [x1, y1],
        [x2, y2],
      ] as const) {
        // every point lies halfway between the centre cell (1.5) and a neighbour
        expect(Math.abs(x - 1.5) + Math.abs(y - 1.5)).toBeCloseTo(0.5, 5);
      }
    }
    expect(contourSegments(map(2, 2, [0, 0, 0, 0]), 0.5)).toEqual([]);
    expect(contourSegments(map(2, 2, [255, 255, 255, 255]), 0.5)).toEqual([]);
  });

  it("splits contour levels above the threshold", () => {
    const levels = contourLevels(0.4);
    expect(levels).toHaveLength(3);
    expect(levels[0]).toBeGreaterThan(0.4);
    expect(levels[2]).toBeLessThan(1);
    expect(contourLevels(0)[0]).toBeGreaterThan(0.2);
  });

  it("reads grey levels from RGBA and says values in the map's unit", () => {
    const levels = levelsFromRgba(Uint8ClampedArray.from([10, 10, 10, 255, 200, 200, 200, 255]), 2, 1);
    expect(Array.from(levels.levels)).toEqual([10, 200]);
    expect(formatValue(0)).toBe("0");
    expect(formatValue(20.589)).toBe("20.6");
    expect(formatValue(0.06547)).toBe("0.065");
    expect(formatValue(1234.5)).toBe("1,235");
    expect(formatValue(Number.NaN)).toBe("-");
    expect(valueAt(0, 20, 0.25)).toBe(5);
  });
});
