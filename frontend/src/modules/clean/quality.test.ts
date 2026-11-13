import { describe, expect, it } from "vitest";

import type { Step, Suggestion } from "./api";
import { qualityBars, shareOf, type ColumnQuality } from "./quality";

const measured = (column: string, rows: number, missing: number, unreadable: number | null): ColumnQuality => ({
  column,
  rows,
  missing,
  unreadable,
  unreadable_values: unreadable ? [{ value: "12 kg", rows: unreadable }] : [],
  reading: unreadable === 0 ? null : "number",
});

const fix = (id: string, column: string, cells: number): Suggestion =>
  ({ id, columns: [column], cells, rows: 0, step: { id: `s${id.length}`, op: "map_values", column, map: { a: "b" }, enabled: true } }) as unknown as Suggestion;

const ticked = (suggestion: string, enabled = true): Step => ({ id: "s9", op: "map_values", column: "x", map: { a: "b" }, enabled, suggestion }) as unknown as Step;

describe("quality bars (N37)", () => {
  it("splits the rows into four parts that add up, a cell counted once", () => {
    const [bar] = qualityBars([measured("amount", 100, 10, 5)], [fix("B-07:amount", "amount", 20)], []);
    expect(bar!.parts).toEqual({ valid: 65, toFix: 20, missing: 10, unreadable: 5 });
    expect(bar!.pending.map((s) => s.id)).toEqual(["B-07:amount"]);
  });

  it("caps what is to fix at the cells left", () => {
    const [bar] = qualityBars([measured("amount", 10, 6, 3)], [fix("B-04:amount", "amount", 8)], []);
    expect(bar!.parts).toEqual({ valid: 0, toFix: 1, missing: 6, unreadable: 3 });
  });

  it("follows the draft: a ticked fix leaves 'to fix', a turned-off step does not", () => {
    const found = [fix("B-01:city", "city", 4), fix("B-03:city", "city", 6)];
    const quality = [measured("city", 50, 0, 0)];
    const [on] = qualityBars(quality, found, [ticked("B-01:city")]);
    expect(on!.parts.toFix).toBe(6);
    expect(on!.fixed).toBe(4);
    const [off] = qualityBars(quality, found, [ticked("B-01:city", false)]);
    expect(off!.parts.toFix).toBe(10);
  });

  it("says when the unreadable values were not counted", () => {
    const [bar] = qualityBars([measured("notes", 30, 2, null)], [], []);
    expect(bar!.uncounted).toBe(true);
    expect(bar!.parts.unreadable).toBe(0);
  });

  it("never reads a part above 0 as 0 %", () => {
    expect(shareOf(1, 1000)).toBe("< 1%");
    expect(shareOf(999, 1000)).toBe("> 99%");
    expect(shareOf(50, 100)).toBe("50%");
  });
});
