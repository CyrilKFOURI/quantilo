import { describe, expect, it } from "vitest";

import type { DuplicateGroup, Step, Suggestion } from "./api";
import { choiceOf, isApart, isDuplicateFix, isReviewed, mergedCells, removedNow, rowOfChoice, ruleStep, stepsForGroups, survivorOf, withApart, withSurvivor, type DuplicateStep } from "./duplicates";

const step = { id: "s2", op: "drop_duplicates", keys: ["id"], keep: "latest", order_by: "updated", groups: null, survivors: null, not_duplicates: null, merge: false, enabled: true, origin: "suggested", rule: "B-23", suggestion: "B-23:id" } as DuplicateStep;
// rows 3, 4, 5 of one id: 4 is the latest (the rule's), 5 the most complete, 3 the first
const group: DuplicateGroup = {
  rows: [
    { row: 3, filled: 2, cells: ["C3", "", null] },
    { row: 4, filled: 3, cells: ["C3", "Nice", "2024-05-01"] },
    { row: 5, filled: 4, cells: ["C3", "Nice ", "2024-04-01"] },
  ],
  size: 3,
  order: [4, 5, 3],
  first: 3,
  most_complete: 5,
  latest: { updated: 4 },
  differing: [1, 2],
};

describe("Review of duplicates by group (N35)", () => {
  it("knows the fixes that remove duplicates", () => {
    expect(isDuplicateFix({ rule: "B-23", step } as unknown as Suggestion)).toBe(true);
    expect(isDuplicateFix({ rule: "B-23", step: { ...step, op: "conditional_column" } } as unknown as Suggestion)).toBe(false);
    expect(isDuplicateFix({ rule: "B-19", step } as unknown as Suggestion)).toBe(false);
  });

  it("reads the kept row: the rule's, else the one chosen; the choice that names it", () => {
    expect(survivorOf(group, step)).toBe(4);
    expect(choiceOf(group, 4, ["updated"])).toBe("rule");
    expect(choiceOf(group, 5, ["updated"])).toBe("most_complete");
    expect(choiceOf(group, 3, ["updated"])).toBe("first");
    expect(rowOfChoice(group, "latest:updated")).toBe(4);
    const chosen = withSurvivor(step, group, 5);
    expect(chosen.survivors).toEqual([5]);
    expect(survivorOf(group, chosen)).toBe(5);
    expect(withSurvivor(chosen, group, 4).survivors).toBeNull(); // back to the rule's: nothing written
    expect(choiceOf({ ...group, first: 9 }, 3, [])).toBe("pick");
  });

  it("keeps a group whole, counts the rows removed, and gives the group back", () => {
    const apart = withApart(step, group, true);
    expect(apart.not_duplicates).toEqual([[3, 4, 5]]);
    expect(isApart(group, apart)).toBe(true);
    expect(removedNow({ removed: 7, groups: [group] }, apart)).toBe(5);
    expect(removedNow({ removed: 7, groups: [{ ...group, size: 40 }] }, apart)).toBe(0); // a large group: its size counts
    expect(removedNow({ removed: 7, groups: [] }, apart)).toBe(5); // not shown: the rows listed
    expect(withApart(apart, group, false).not_duplicates).toBeNull();
    expect(isReviewed(step)).toBe(false);
    expect(isReviewed(apart)).toBe(true);
    expect(isReviewed({ ...step, merge: true })).toBe(true);
  });

  it("fills the kept row's empty cells from its group, in the rule's order", () => {
    const filled = mergedCells(group, 3);
    expect([...filled.entries()]).toEqual([
      [1, { value: "Nice", from: 4 }],
      [2, { value: "2024-05-01", from: 4 }],
    ]);
    expect(mergedCells(group, 5).size).toBe(0);
  });

  it("asks for the groups of the rule: the review cleared, the steps after it left out", () => {
    const trim = { id: "s1", op: "normalize_spaces", columns: ["city"], enabled: true } as Step;
    const after = { id: "s3", op: "drop_columns", columns: ["x"], enabled: true } as Step;
    const reviewed = { ...withApart(step, group, true), merge: true, enabled: false } as Step;
    expect(stepsForGroups([trim, reviewed, after], "s2")).toEqual([trim, { ...ruleStep(reviewed as DuplicateStep), enabled: true }]);
  });
});
