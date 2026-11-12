import { describe, expect, it } from "vitest";

import type { Suggestion } from "./api";
import { byLayer, groupOf, groupSuggestions, tickedIn } from "./groups";

const fix = (id: string, order: number, more: Partial<Suggestion> = {}): Suggestion =>
  ({ id, rule: id.split(":")[0], layer: "basic", mode: "A", impact: "format", risk: "safe", order, title: id, reason: "why", columns: [], cells: 1, rows: 0, examples: [], step: null, ...more }) as Suggestion;

describe("Clean groups (096)", () => {
  it("orders the groups from the lossless to the decisions, without empty ones", () => {
    const found = [
      fix("B-19:x", 170, { impact: "columns", risk: "review", mode: "S" }),
      fix("B-07:p", 100, { impact: "types" }),
      fix("B-01:a", 30),
      fix("B-15:empty", 10, { impact: "rows" }),
      fix("B-02:a", 20),
    ];
    const groups = groupSuggestions(found);
    expect(groups.map((group) => group.id)).toEqual(["format", "types", "rows", "decision"]);
    expect(groups[0]!.items.map((item) => item.id)).toEqual(["B-02:a", "B-01:a"]); // the steps' order
  });

  it("reads an older server's suggestion from its step and mode", () => {
    const step = { id: "s1", op: "drop_duplicates", keys: null, keep: "first", enabled: true, origin: "auto", rule: null, suggestion: null } as Suggestion["step"];
    const old = { ...fix("B-14:rows", 160, { step, mode: "A" }), impact: undefined, risk: undefined } as unknown as Suggestion;
    expect(groupOf(old)).toBe("rows");
    expect(groupOf({ ...old, mode: "S" })).toBe("decision");
  });

  it("splits the layers and counts what is ticked", () => {
    const found = [fix("B-01:a", 1), fix("A-01:a", 2, { layer: "advanced" }), fix("E-01:a", 3, { layer: "expert" })];
    const layers = byLayer(found);
    expect([layers.basic.length, layers.advanced.length, layers.expert.length]).toEqual([1, 1, 1]);
    const steps = [{ suggestion: "B-01:a" }, { suggestion: null }] as never;
    expect(tickedIn(found, steps)).toBe(1);
  });
});
