import { describe, expect, it } from "vitest";

import type { CleanCheck, Step, Suggestion } from "./api";
import { blankCheck, checkFilter, checkOfSuggestion, describeCheck, nextCheckId, sameCheck } from "./checkRules";

const suggestion = (rule: string, columns: string[], step: unknown = null) => ({ id: `${rule}:x`, rule, columns, step }) as unknown as Suggestion;

describe("declared checks (N31)", () => {
  it("makes the check a rule's finding stands for", () => {
    expect(checkOfSuggestion(suggestion("A-10", ["city", "country"]), "c1")).toMatchObject({ kind: "determines", columns: ["city", "country"], origin: "rule", rule: "A-10" });
    expect(checkOfSuggestion(suggestion("A-11", ["created", "closed"]), "c1")).toMatchObject({ kind: "compare", op: "le" });
    const dedupe = { id: "s1", op: "drop_duplicates", keys: ["order_id"], keep: "first" } as unknown as Step;
    expect(checkOfSuggestion(suggestion("B-23", ["order_id"], dedupe), "c4")).toMatchObject({ id: "c4", kind: "unique", columns: ["order_id"], rule: "B-23" });
    expect(checkOfSuggestion(suggestion("B-01", ["city"], { id: "s1", op: "normalize_spaces", columns: ["city"] }), "c1")).toBeNull();
  });

  it("says a check and writes its failing rows as a Data filter when it can", () => {
    const types = { city: "categorical_nominal", amount: "numeric_continuous" } as const;
    const notEmpty = blankCheck("not_empty", "c1", ["city"]);
    expect(describeCheck(notEmpty)).toBe("city is never empty");
    expect(checkFilter(notEmpty, types)).toEqual([{ column: "city", op: "is_null", negate: false }]);
    const bounds: CleanCheck = { ...blankCheck("between", "c2", ["amount"]), min: "0", max: "100" };
    expect(describeCheck(bounds)).toBe("amount is between 0 and 100");
    expect(checkFilter(bounds, types)).toEqual([{ column: "amount", op: "between", low: 0, high: 100, negate: true }]);
    const listed: CleanCheck = { ...blankCheck("in_list", "c3", ["city"]), values: ["Paris", "Lyon"] };
    expect(checkFilter(listed, types)).toEqual([{ column: "city", op: "in", values: ["Paris", "Lyon"], negate: true }]);
    expect(checkFilter(blankCheck("compare", "c4", ["amount", "city"]), types)).toBeNull();
    expect(nextCheckId([notEmpty, bounds])).toBe("c3");
    expect(sameCheck(notEmpty, { ...notEmpty, id: "c9", origin: "rule" })).toBe(true);
  });
});
