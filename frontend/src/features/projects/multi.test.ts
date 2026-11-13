import { describe, expect, it } from "vitest";

import type { FamilyRow } from "../../modules/clean/lineage";
import { moveLocally, projectsOf, type Project } from "./api";
import { groupByProject } from "./grouping";

const project = (id: string, ids: string[]): Project => ({ id, name: id, created_at: "", updated_at: "", dataset_ids: ids });
const row = (id: string) => ({ dataset: { id }, version: null, children: [] }) as unknown as FamilyRow;

describe("a dataset in several projects (C1)", () => {
  it("is listed under each of its projects", () => {
    const groups = groupByProject([row("d1"), row("d2")], [project("a", ["d1"]), project("b", ["d1"])]);
    expect(groups.map((group) => [group.project?.id ?? "none", group.rows.map((r) => r.dataset.id)])).toEqual([
      ["a", ["d1"]],
      ["b", ["d1"]],
      ["none", ["d2"]],
    ]);
    expect(projectsOf("d1", [project("a", ["d1"]), project("b", ["d1"])]).map((p) => p.id)).toEqual(["a", "b"]);
  });

  it("moves out of one project only when told where from", () => {
    const before = [project("a", ["d1"]), project("b", ["d1"]), project("c", [])];
    expect(moveLocally(before, ["d1"], "c", "a").map((p) => p.dataset_ids)).toEqual([[], ["d1"], ["d1"]]);
    expect(moveLocally(before, ["d1"], "c").map((p) => p.dataset_ids)).toEqual([[], [], ["d1"]]);
  });
});
