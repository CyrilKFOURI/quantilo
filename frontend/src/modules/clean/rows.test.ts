import { describe, expect, it } from "vitest";

import type { Step } from "./api";
import { filterOf } from "./rows";

const base = { id: "s1", enabled: true, origin: "suggested", rule: "X", suggestion: "X:a" };
const step = (more: Record<string, unknown>) => ({ ...base, ...more }) as unknown as Step;
const types = { city: "categorical_nominal", amount: "numeric_continuous", code: "numeric_discrete", note: "text" } as const;

describe("The rows of a fix as a Data filter (N41)", () => {
  it("writes the values a merge or a token changes as 'is any of', or '=' where the type has no list", () => {
    expect(filterOf(step({ op: "map_values", column: "city", map: { PARIS: "Paris", "paris ": "Paris" } }), types)).toEqual([{ column: "city", op: "in", values: ["PARIS", "paris "], negate: false }]);
    expect(filterOf(step({ op: "null_values", column: "amount", values: ["-200"], match: "exact" }), types)).toEqual([{ column: "amount", op: "eq", value: -200, negate: false }]);
    expect(filterOf(step({ op: "null_values", column: "code", values: ["-200", "-999"], match: "exact" }), types)).toEqual([{ column: "code", op: "in", values: ["-200", "-999"], negate: false }]);
    expect(filterOf(step({ op: "null_values", column: "city", values: ["n/a"], match: "token" }), types)).toBeNull(); // a token also matches 'N/A ': not an exact filter
    expect(filterOf(step({ op: "null_values", column: "amount", values: ["-200", "-999"], match: "exact" }), types)).toBeNull(); // no list on a continuous number
  });

  it("writes the conditions of a step when every one has its filter", () => {
    const conditions = [{ column: "amount", op: "lt", value: 0, values: null }, { column: "note", op: "empty", value: null, values: null }];
    expect(filterOf(step({ op: "filter_rows", conditions, combine: "all", action: "remove" }), types)).toEqual([
      { column: "amount", op: "lt", value: 0, negate: false },
      { column: "note", op: "is_null", negate: false },
    ]);
    expect(filterOf(step({ op: "filter_rows", conditions, combine: "any", action: "remove" }), types)).toBeNull(); // OR: not a Data filter
    expect(filterOf(step({ op: "filter_rows", conditions, combine: "all", action: "keep" }), types)).toBeNull();
    const cases = [{ conditions: [{ column: "city", op: "ne", value: "Paris", values: null }], combine: "all", when: null, value: "x", formula: null }];
    expect(filterOf(step({ op: "set_where", column: "city", cases }), types)).toEqual([{ column: "city", op: "eq", value: "Paris", negate: true }]);
    expect(filterOf(step({ op: "set_where", column: "city", cases: [{ ...cases[0], when: "amount > 1" }] }), types)).toBeNull();
  });

  it("leaves out the steps that name rows by number", () => {
    expect(filterOf(step({ op: "drop_rows", rows: [4, 9] }), types)).toBeNull();
    expect(filterOf(step({ op: "normalize_spaces", columns: ["city"] }), types)).toBeNull();
  });
});
