import { describe, expect, it } from "vitest";

import type { Suggestion } from "./api";
import { autoFacts, batchKey, byLayer, groupByColumn, groupOf, groupSuggestions, linesOf, replaceColumn, tickedIn, totals } from "./groups";

const fix = (id: string, order: number, more: Partial<Suggestion> = {}): Suggestion =>
  ({ id, rule: id.split(":")[0], layer: "basic", mode: "A", impact: "format", risk: "safe", order, title: id, reason: "why", columns: [], cells: 1, rows: 0, examples: [], step: null, catalogue: null, auto: false, measured: null, clusters: [], ...more }) as Suggestion;

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

describe("Applied automatically (N03, N04)", () => {
  it("puts the reliable fixes first when the mode applies them, in their effect group otherwise", () => {
    const found = [fix("B-04:x", 90, { impact: "values", auto: true, cells: 4 }), fix("B-01:a", 30, { auto: true, cells: 10 }), fix("B-03:b", 40, { impact: "values" })];
    expect(groupSuggestions(found, true).map((group) => [group.id, group.items.map((item) => item.id)])).toEqual([
      ["auto", ["B-01:a", "B-04:x"]],
      ["values", ["B-03:b"]],
    ]);
    expect(groupSuggestions(found).map((group) => group.id)).toEqual(["format", "values"]);
    const steps = [{ suggestion: "B-01:a" }] as never;
    expect(autoFacts(groupSuggestions(found, true)[0]!.items, steps)).toEqual({ applied: 1, cells: 14, rows: 0 });
  });
});

describe("Clean groups by column (C1)", () => {
  it("gathers a column's fixes in the order of its first one, the whole table last", () => {
    const found = [
      fix("B-14:rows", 160, { impact: "rows", columns: [] }),
      fix("B-03:city", 40, { columns: ["city"] }),
      fix("B-01:name", 30, { columns: ["name"] }),
      fix("B-04:city", 90, { columns: ["city"], impact: "values" }),
    ];
    const groups = groupByColumn(found);
    expect(groups.map((group) => group.column)).toEqual(["name", "city", null]);
    expect(groups[1]!.items.map((item) => item.id)).toEqual(["B-03:city", "B-04:city"]);
  });
});

describe("Repetitive suggestions as one line (N39)", () => {
  const trimOf = (column: string, order: number, more: Partial<Suggestion> = {}) =>
    fix(`B-01:${column}`, order, {
      columns: [column],
      title: `Trim extra spaces in ${column}`,
      cells: 2,
      step: { id: `s${order}`, op: "normalize_spaces", columns: [column], enabled: true, origin: "suggested", rule: "B-01", suggestion: `B-01:${column}` } as Suggestion["step"],
      ...more,
    });
  const sentinel = (column: string, value: string, order: number) =>
    fix(`B-09:${column}:${value}:repeated`, order, {
      columns: [column],
      title: `Read ${value} as missing in ${column}`,
      step: { id: `s${order}`, op: "null_values", column, values: [value], match: "exact", enabled: true, origin: "suggested", rule: "B-09", suggestion: `B-09:${column}:${value}:repeated` } as Suggestion["step"],
    });

  it("replaces a column only where it is a whole word", () => {
    expect(replaceColumn("Trim extra spaces in city", "city", "3 columns")).toBe("Trim extra spaces in 3 columns");
    expect(replaceColumn("Trim velocity", "city", "x")).toBeNull();
    expect(replaceColumn("Store a as numbers", "a", "3 columns")).toBe("Store 3 columns as numbers");
  });

  it("gathers the same fix on 3 columns or more at the place of the first, the others as they are", () => {
    const found = [trimOf("a", 1), trimOf("b", 2), fix("B-02:a", 3, { columns: ["a"] }), trimOf("c", 4), trimOf("d", 5), sentinel("x", "-200", 6), sentinel("y", "-200", 7)];
    const lines = linesOf(found);
    expect(lines.map((line) => (line.kind === "batch" ? line.items.map((item) => item.id) : line.suggestion.id))).toEqual([
      ["B-01:a", "B-01:b", "B-01:c", "B-01:d"],
      "B-02:a",
      "B-09:x:-200:repeated", // two columns: not gathered
      "B-09:y:-200:repeated",
    ]);
    expect(totals(found.slice(0, 2))).toEqual({ cells: 4, rows: 0 });
  });

  it("keeps apart what differs: the value, the kind, a merge, a step-less information", () => {
    expect(batchKey(sentinel("x", "-200", 1))).not.toBe(batchKey(sentinel("y", "-999", 2)));
    expect(batchKey(sentinel("x", "-200", 1))).toBe(batchKey(sentinel("y", "-200", 2)));
    const empty = (column: string) => ({ ...trimOf(column, 1), id: `B-19:${column}`, rule: "B-19", title: `Remove the empty column ${column}` });
    const constant = (column: string) => ({ ...empty(column), title: `Remove the constant column ${column}` });
    expect(batchKey(empty("a"))).toBe(batchKey(empty("b")));
    expect(batchKey(empty("a"))).not.toBe(batchKey(constant("b")));
    expect(batchKey({ ...trimOf("a", 1), rule: "A-01" })).toBeNull(); // reviewed value by value (N38)
    expect(batchKey({ ...trimOf("a", 1), step: null })).toBeNull();
    expect(batchKey(trimOf("a", 1, { auto: true }))).not.toBe(batchKey(trimOf("b", 2)));
  });
});
