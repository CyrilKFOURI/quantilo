import { describe, expect, it } from "vitest";

import type { ChartTokens } from "../../charts/theme";
import { levelColor, lineAllowed } from "./figures";

const tokens = {
  ink2: "#555",
  ink3: "#999",
  palette: ["#2a78d6", "#eb6834", "#1baf7a"],
} as unknown as ChartTokens;

describe("KPI chart colours and lines (045)", () => {
  it("draws true and false in two different hues, never two blues", () => {
    const levels = ["true", "false"];
    expect(levelColor(levels, "true", true, tokens)).toBe("#2a78d6");
    expect(levelColor(levels, "false", true, tokens)).toBe("#eb6834");
  });

  it("keeps one hue light to dark for an ordered split of three or more", () => {
    const levels = ["low", "mid", "high"];
    const colors = levels.map((l) => levelColor(levels, l, true, tokens));
    expect(new Set(colors).size).toBe(3);
    expect(colors.every((c) => c.startsWith("rgba(42, 120, 214"))).toBe(true);
  });

  it("allows a line over one series, and a total line only where a total means something", () => {
    expect(lineAllowed("share_total", 1)).toBe(true);
    expect(lineAllowed("value", 3)).toBe(true);
    expect(lineAllowed("running", 3)).toBe(true);
    expect(lineAllowed("share_total", 3)).toBe(false);
    expect(lineAllowed("change_pct", 2)).toBe(false);
  });
});
