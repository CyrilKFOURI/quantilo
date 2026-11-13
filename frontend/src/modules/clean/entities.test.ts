import { describe, expect, it } from "vitest";

import type { EntityClusters, Step, Suggestion } from "./api";
import { entitiesNow, isApart, isEntityFix, isEntityReviewed, ruleEntityStep, stepsForClusters, takenOut, withAllApart, withApart, withTaken, type EntityStep } from "./entities";

const record = (name: string) => ({ anchor: "10007", name });
const step: EntityStep = {
  id: "s2",
  op: "entity_column",
  column: "owner",
  anchor: "zip",
  anchor_kind: "postal_code",
  to: "owner_entity",
  groups: [[record("a"), record("a."), record("a,")], [record("b"), record("b.")]],
  not_entities: null,
  split: null,
  enabled: false,
  origin: "suggested",
  rule: "A-20",
  suggestion: "A-20:owner:entity",
};
const clusters = [0, 1].map((index) => ({ index, rows: 3, differing: [], records: step.groups[index]!.map((r, at) => ({ index: at, ...r, spelling: r.name, rows: 1, first: at, cells: [] })) }));
const found = { total_clusters: 2, clusters } as unknown as EntityClusters;

describe("entities (N36)", () => {
  it("knows an entity fix and clears its review for the clusters", () => {
    expect(isEntityFix({ rule: "A-20", step } as unknown as Suggestion)).toBe(true);
    expect(isEntityFix({ rule: "A-16", step } as unknown as Suggestion)).toBe(false);
    const trim = { id: "s1", op: "normalize_spaces", columns: ["owner"] } as unknown as Step;
    const after = { id: "s3", op: "drop_columns", columns: ["zip"] } as unknown as Step;
    const reviewed = { ...step, not_entities: [1] };
    expect(stepsForClusters([trim, reviewed, after], "s2")).toEqual([trim, { ...ruleEntityStep(reviewed), enabled: true }]);
  });

  it("says a cluster apart, and back", () => {
    const apart = withApart(step, clusters[1]!, true);
    expect(apart.not_entities).toEqual([1]);
    expect(isApart(clusters[1]!, apart)).toBe(true);
    expect(withApart(apart, clusters[1]!, false).not_entities).toBeNull();
    expect(withAllApart(step, clusters, true).not_entities).toEqual([0, 1]);
  });

  it("splits spellings off a cluster; all of them is the cluster as it was", () => {
    const one = withTaken(step, clusters[0]!, 2, true);
    expect(one.split).toEqual({ "0": [2] });
    expect([...takenOut(clusters[0]!, one)]).toEqual([2]);
    expect(withTaken(one, clusters[0]!, 2, false).split).toBeNull();
    const two = withTaken(one, clusters[0]!, 1, true);
    expect(two.split).toEqual({ "0": [1, 2] });
    expect(withTaken(two, clusters[0]!, 0, true).split).toBeNull();
    expect(isEntityReviewed(two)).toBe(true);
    expect(isEntityReviewed(step)).toBe(false);
  });

  it("counts the entity numbers the clusters give now", () => {
    expect(entitiesNow(found, null)).toBe(2);
    expect(entitiesNow(found, withTaken(step, clusters[0]!, 2, true))).toBe(3);
    expect(entitiesNow(found, withApart(step, clusters[0]!, true))).toBe(4);
    // a cluster apart and split counts as apart
    expect(entitiesNow(found, withApart(withTaken(step, clusters[0]!, 2, true), clusters[0]!, true))).toBe(4);
  });
});
