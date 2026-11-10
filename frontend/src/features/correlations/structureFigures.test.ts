import { describe, expect, it } from "vitest";

import type { Structure } from "../../api/client";
import type { ChartTokens } from "../../charts/theme";
import { biplotFigure, loadingsFigure, screeFigure } from "./structureFigures";

const tokens: ChartTokens = {
  ink1: "#111111",
  ink2: "#222222",
  ink3: "#333333",
  line: "#444444",
  surface: "#ffffff",
  accent: "#ad4a27",
  numeric: "#3b6ea8",
  categorical: "#8a5a9e",
  date: "#2f7d6d",
  bool: "#8c6d1f",
  text: "#6b6259",
  id: "#77706a",
  font: "sans-serif",
  palette: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
};

const structure = {
  columns: ["a", "b", "c"],
  components: [
    { index: 1, eigenvalue: 2, explained: 0.66, cumulative: 0.66, threshold: 1.2, retained: true, label: "a, b" },
    { index: 2, eigenvalue: 0.7, explained: 0.24, cumulative: 0.9, threshold: 1.05, retained: false, label: "c" },
    { index: 3, eigenvalue: 0.3, explained: 0.1, cumulative: 1, threshold: 0.9, retained: false, label: "c" },
  ],
  retained: 1,
  loadings: [
    [0.9, 0.1],
    [0.85, -0.2],
    [0.2, 0.95],
  ],
  shown: 2,
  biplot: {
    x: [2, -1],
    y: [0.5, -0.5],
    points: 2,
    of: 2,
    arrows: [
      { column: "a", x: 0.9, y: 0.1 },
      { column: "c", x: 0.2, y: 0.95 },
    ],
  },
} as unknown as Structure;

describe("structure figures (078)", () => {
  it("colours the kept components and draws the chance line and the cumulative share", () => {
    const { data } = screeFigure(structure)(tokens);
    const bars = data[0] as { marker: { color: string[] } };
    expect(bars.marker.color[0]).toBe("#2a78d6");
    expect(bars.marker.color[1]).not.toBe("#2a78d6");
    expect((data[1] as { y: number[] }).y).toEqual([1.2, 1.05, 0.9]);
    expect((data[2] as { y: number[]; yaxis: string }).y).toEqual([66, 90, 100]);
  });

  it("draws loadings on the diverging scale, one column per shown component", () => {
    const { data } = loadingsFigure(structure)(tokens);
    const heatmap = data[0] as unknown as { x: string[]; y: string[]; zmin: number; zmax: number };
    expect(heatmap.x).toEqual(["PC1", "PC2"]);
    expect(heatmap.y).toEqual(["a", "b", "c"]);
    expect([heatmap.zmin, heatmap.zmax]).toEqual([-1, 1]);
  });

  it("scales the arrows to the points' spread", () => {
    const { data } = biplotFigure(structure)(tokens);
    const tips = data[2] as { x: number[]; y: number[]; text: string[] };
    expect(tips.text).toEqual(["a", "c"]);
    const longest = Math.max(...tips.x.map((x, i) => Math.hypot(x, tips.y[i] ?? 0)));
    expect(longest).toBeCloseTo(0.85 * 2, 5);
  });
});
