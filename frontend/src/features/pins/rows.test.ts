import { describe, expect, it } from "vitest";

import type { Pin } from "./api";
import { colorVar, labelsOf, nextColor, parseRows, pinsByRow } from "./rows";

const pin = (row: number, label: string, found = true): Pin => ({ key: `row:${row}`, row, label, color: 1, created_at: "", found, inherited: false });

describe("pinned rows helpers", () => {
  it("reads row numbers as shown (from 1) into file rows (from 0)", () => {
    expect(parseRows("12, 40-42")).toEqual([11, 39, 40, 41]);
    expect(parseRows(" 3 ;3  5 ")).toEqual([2, 4]);
    expect(parseRows("")).toEqual([]);
  });

  it("refuses what is not a row list", () => {
    expect(parseRows("abc")).toBeNull();
    expect(parseRows("0")).toBeNull();
    expect(parseRows("9-3")).toBeNull();
    expect(parseRows("1-501")).toBeNull();
  });

  it("gives the first free colour, then goes round", () => {
    expect(nextColor([])).toBe(1);
    expect(nextColor([{ label: "a", color: 1, rows: 1 }, { label: "b", color: 3, rows: 1 }])).toBe(2);
    const all = [1, 2, 3, 4, 5, 6, 7, 8].map((color) => ({ label: String(color), color, rows: 1 }));
    expect(nextColor(all)).toBe(1);
    expect(colorVar(4)).toBe("var(--chart-4)");
  });

  it("keeps only the pins found, for the table and for the step", () => {
    const pins = [pin(3, "big"), pin(9, "test"), pin(20, "big", false)];
    expect([...pinsByRow(pins).keys()]).toEqual([3, 9]);
    expect(labelsOf(pins)).toEqual({ "3": "big", "9": "test" });
  });
});
