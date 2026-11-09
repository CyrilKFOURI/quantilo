import { describe, expect, it } from "vitest";

import type { DateChange, Shift } from "../../api/client";
import type { ChartTokens } from "../../charts/theme";
import { periodsFigure, shiftFigure, shiftHeight } from "./figures";

const tokens = {
  ink1: "#111111",
  ink2: "#555555",
  ink3: "#777777",
  line: "#dddddd",
  surface: "#ffffff",
  accent: "#ad4a27",
  numeric: "#3b6ea8",
  categorical: "#8a5a9e",
  date: "#2f7d6d",
  bool: "#8c6d1f",
  text: "#6b6259",
  id: "#77706a",
  font: "IBM Plex Sans",
  palette: [],
} satisfies ChartTokens;

const numeric = {
  column: "prix",
  kind: "numeric",
  levels: [],
  histogram: { edges: [0, 10, 20], before: [0.6, 0.4], after: [0.2, 0.8] },
} as unknown as Shift;

describe("before / after figures", () => {
  it("draws numbers as two overlaid histograms on the same intervals", () => {
    const built = shiftFigure(numeric)(tokens);
    expect(built.data.map((trace) => (trace as { name: string }).name)).toEqual(["Before", "After"]);
    const [before, after] = built.data as { x: number[]; width: number[]; y: number[]; marker: { line: { width: number } } }[];
    expect(before?.x).toEqual([5, 15]);
    expect(before?.width).toEqual([10, 10]);
    expect(after?.y).toEqual([0.2, 0.8]);
    expect(after?.marker.line.width).toBe(1.5); // the new version is outlined, not only coloured
    expect(built.layout?.barmode).toBe("overlay");
    expect(shiftHeight(numeric)).toBe(300);
  });

  it("draws categories as horizontal pairs, biggest on top", () => {
    const shift = {
      column: "region",
      kind: "categorical",
      histogram: null,
      levels: [
        { value: "North", before: 0.5, after: 0.4 },
        { value: "Outre-mer", before: 0, after: 0.1 },
      ],
    } as unknown as Shift;
    const built = shiftFigure(shift)(tokens);
    const [before] = built.data as { y: string[]; x: number[]; orientation: string }[];
    expect(before?.orientation).toBe("h");
    expect(before?.y).toEqual(["Outre-mer", "North"]);
    expect(before?.x).toEqual([0, 0.5]);
    expect(shiftHeight(shift)).toBe(240);
  });

  it("draws rows per period for a date", () => {
    const dates = { periods: ["2026-09", "2026-10"], before_counts: [10, 12], after_counts: [10, 0] } as unknown as DateChange;
    const built = periodsFigure(dates)(tokens);
    expect((built.data[1] as { y: number[] }).y).toEqual([10, 0]);
    expect(built.layout?.barmode).toBe("group");
  });
});
