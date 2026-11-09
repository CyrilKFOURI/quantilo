import { afterEach, describe, expect, it } from "vitest";

import { readTokens } from "../charts/theme";
import { DEFAULT_PALETTE, PALETTES, paletteById, paletteLabel, palettePreference, paletteStyles } from "./palettes";
import { savePreference } from "./preferences";

afterEach(() => {
  delete document.documentElement.dataset.palette;
  localStorage.clear();
});

describe("Chart palettes (047)", () => {
  it("offers palettes of 7 to 20 colours, each a valid hex list without repeats", () => {
    const sizes = new Set(PALETTES.map((p) => p.colors.length));
    for (const size of [7, 8, 10, 12, 20]) expect(sizes.has(size)).toBe(true);
    for (const palette of PALETTES) {
      expect(palette.colors.every((c) => /^#[0-9a-f]{6}$/.test(c))).toBe(true);
      expect(new Set(palette.colors).size).toBe(palette.colors.length);
    }
    expect(new Set(PALETTES.map((p) => p.id)).size).toBe(PALETTES.length);
  });

  it("names a palette with its number of colours", () => {
    expect(paletteLabel(paletteById("tableau20"), "")).toBe("Tableau 20 (20)");
    expect(paletteLabel(undefined, "Quantilo")).toBe("Quantilo (8)");
  });

  it("gives the chosen palette to the charts and to the --chart tokens", () => {
    expect(paletteStyles()).toContain('html:root[data-palette="okabe-ito"] { --chart-1: #e69f00;');
    savePreference(palettePreference, "tableau20");
    expect(document.documentElement.dataset.palette).toBe("tableau20");
    expect(readTokens().palette).toHaveLength(20);
    savePreference(palettePreference, DEFAULT_PALETTE);
    expect(document.documentElement.dataset.palette).toBeUndefined();
    expect(readTokens().palette).toHaveLength(8);
  });
});
