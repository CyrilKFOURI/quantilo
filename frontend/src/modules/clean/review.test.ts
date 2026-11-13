import { describe, expect, it } from "vitest";

import type { Step, Suggestion } from "./api";
import { isReviewable, readReview, reviewFacts, reviewGroups, reviewTick, writeReview } from "./review";

const map = { Pariss: "Paris", " Pariss": "Paris", "PARIS.": "Paris", Tolouse: "Toulouse" };
const step = { id: "s1", op: "map_values", column: "city", map, enabled: true, origin: "suggested", rule: "A-03", suggestion: "A-03:city" } as Step;
const clusters = [
  { target: "Paris", rows: 600, values: [{ value: "Pariss", rows: 4, spellings: [" Pariss", "Pariss"] }, { value: "PARIS.", rows: 2, spellings: ["PARIS."] }] },
  { target: "Toulouse", rows: 200, values: [{ value: "Tolouse", rows: 3, spellings: ["Tolouse"] }] },
];
const merge = { id: "A-03:city", rule: "A-03", layer: "advanced", columns: ["city"], step, clusters } as unknown as Suggestion;

describe("Review of a merge, value by value (N38)", () => {
  it("reads the groups of the engine, or of an older server's map by target", () => {
    expect(reviewGroups(merge)).toHaveLength(2);
    const old = reviewGroups({ ...merge, clusters: [] });
    expect(old.map((group) => [group.target, group.values.map((value) => value.value), group.rows])).toEqual([
      ["Paris", ["Pariss", " Pariss", "PARIS."], null],
      ["Toulouse", ["Tolouse"], null],
    ]);
    expect(isReviewable(merge)).toBe(true);
    expect(isReviewable({ ...merge, rule: "B-03", clusters: [] })).toBe(false);
  });

  it("reads the ticks from the draft: everything when the whole step is there, nothing when it is not", () => {
    const groups = reviewGroups(merge);
    expect(readReview(groups, step)).toEqual({ ticked: [[true, true], [true]], targets: ["Paris", "Toulouse"] });
    expect(reviewTick(readReview(groups, undefined))).toBe(false);
    expect(reviewTick(readReview(groups, step))).toBe(true);
  });

  it("writes exactly what remains ticked, a new target for the whole group", () => {
    const groups = reviewGroups(merge);
    const state = { ticked: [[true, false], [true]], targets: ["Paris", "Toulouse"] };
    expect(writeReview(groups, state)).toEqual({ Pariss: "Paris", " Pariss": "Paris", Tolouse: "Toulouse" });
    expect(reviewFacts(groups, state)).toEqual({ groups: 2, values: 2, total: 3, cells: 7 });
    const renamed = { ticked: [[true, true], [false]], targets: ["PARIS", "Toulouse"] };
    const written = writeReview(groups, renamed);
    expect(written).toEqual({ Pariss: "PARIS", " Pariss": "PARIS", "PARIS.": "PARIS", Paris: "PARIS" });
    expect(reviewFacts(groups, renamed).cells).toBe(606); // the target's own rows change too
    // read back from the step it wrote: the same ticks and the new target
    expect(readReview(groups, { ...step, map: written } as Step)).toEqual({ ticked: [[true, true], [false]], targets: ["PARIS", "Toulouse"] });
    expect(reviewTick(readReview(groups, { ...step, map: written } as Step))).toBe("mixed");
    expect(writeReview(groups, { ticked: [[false, false], [false]], targets: ["Paris", "Toulouse"] })).toEqual({});
    expect(writeReview(groups, { ticked: [[true, true], [true]], targets: [" ", "Toulouse"] })).toEqual({ Tolouse: "Toulouse" }); // an empty target: not merged
  });
});
