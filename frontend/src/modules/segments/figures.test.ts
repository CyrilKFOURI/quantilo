import { describe, expect, it } from "vitest";

import type { ChartTokens } from "../../charts/theme";
import { run, unusualRun } from "../../test/segments";
import type { SegmentRun, UnusualRun } from "./api";
import { mapFigure, namesOf, rowsAbove, scoreFigure, segmentColor, shownScores, unusualFigure } from "./figures";

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
const segmentation = (run as unknown as SegmentRun).result;
const unusual = (unusualRun as unknown as UnusualRun).result;

describe("segments figures", () => {
  it("draws each score as its own small chart, the chosen k marked", () => {
    expect(shownScores(segmentation.scores).map((s) => s.key)).toEqual(["silhouette", "calinski_harabasz", "davies_bouldin", "inertia"]);
    const { data, layout } = scoreFigure(segmentation.scores, "silhouette", 3)(tokens);
    expect(data).toHaveLength(1);
    const marker = (data[0] as { marker: { color: string[] } }).marker;
    expect(marker.color[1]).toBe(tokens.accent); // k = 3 is marked
    expect(layout?.shapes?.[0]).toMatchObject({ x0: 3, x1: 3 });
  });

  it("gives each segment its palette colour in fixed order, not grouped in grey", () => {
    expect(segmentColor(0, tokens)).toBe("#2a78d6");
    expect(segmentColor(9, tokens)).toBe("#eb6834");
    expect(segmentColor(-1, tokens)).toBe(tokens.ink3);
    const { data } = mapFigure(segmentation.map, namesOf(segmentation))(tokens);
    expect(data.slice(0, 3).map((d) => d.name)).toEqual(segmentation.segments.map((s) => s.name));
  });

  it("counts the listed rows above a threshold and colours them", () => {
    expect(rowsAbove(unusual, 0.7)).toBe(2);
    expect(rowsAbove(unusual, 0.46)).toBe(3);
    const { data } = unusualFigure(unusual, 0.5)(tokens);
    expect((data[1] as { x: number[] }).x).toEqual([1, 2, 3]);
  });
});
