import { describe, expect, it } from "vitest";

import { dataset } from "../../test/mockApi";
import type { Dataset } from "../../api/client";
import type { LineageItem, Step, Suggestion } from "./api";
import { familyOf, groupFamilies, samePageOf } from "./lineage";
import { describeStep, historyOf, move, newStep, nextId, push, redo, undo, withPinnedLabels, withSuggestion, withoutSuggestion } from "./steps";

const item = (dataset_id: string, parent_id: string, seq: number): LineageItem => ({
  dataset_id,
  parent_id,
  seq,
  origin: seq === 1 ? "auto" : "user",
  steps: 3,
  created_at: "2026-11-10T10:00:00Z",
  out_of_date: false,
  source_id: null,
  source_seq: null,
});

const suggestion = (id: string, order: number, mode: "A" | "S", step: Step): Suggestion =>
  ({ id, order, mode, impact: "values", risk: mode === "A" ? "safe" : "review", rule: id.slice(0, 4), layer: "basic", title: id, reason: "why", columns: [], cells: 1, rows: 0, examples: [], step: { ...step, suggestion: id } }) as Suggestion;

describe("home families (065)", () => {
  const datasets = [dataset({ id: "v2" }), dataset({ id: "other" }), dataset({ id: "orig" }), dataset({ id: "v1" })] as unknown as Dataset[];

  it("puts each original first, its versions under it (v1 first), the latest marked", () => {
    const rows = groupFamilies(datasets, [item("v2", "orig", 2), item("v1", "orig", 1)]);
    expect(rows.map((row) => row.dataset.id)).toEqual(["other", "orig", "v1", "v2"]);
    expect(rows[2]!.version).toMatchObject({ seq: 1, origin: "auto", latest: false });
    expect(rows[3]!.version).toMatchObject({ seq: 2, latest: true, parentId: "orig" });
    expect(rows[1]!.children.map((child) => child.id)).toEqual(["v1", "v2"]);
  });

  it("leaves a version whose original is gone as an ordinary dataset", () => {
    const rows = groupFamilies(datasets, [item("v1", "deleted", 1)]);
    expect(rows.map((row) => row.dataset.id)).toEqual(["v2", "other", "orig", "v1"]);
    expect(rows.every((row) => !row.version)).toBe(true);
  });

  it("gives the top bar the family of an original or of a version", () => {
    const lineage = [item("v1", "orig", 1), item("v2", "orig", 2)];
    expect(familyOf("v2", lineage)).toEqual([
      { id: "orig", seq: null },
      { id: "v1", seq: 1 },
      { id: "v2", seq: 2 },
    ]);
    expect(familyOf("other", lineage)).toEqual([]);
    expect(samePageOf("/d/orig/columns", "v1")).toBe("/d/v1/columns");
    expect(samePageOf("/d/orig", "v1")).toBe("/d/v1");
  });
});

describe("draft steps (065)", () => {
  const spaces = newStep("normalize_spaces", "s2", "city");
  const empty = newStep("drop_empty_rows", "s1", "city");
  const header = newStep("drop_header_rows", "s3", "city");
  const all = [suggestion("B-15:empty", 10, "A", empty), suggestion("B-01:city", 30, "A", spaces), suggestion("B-15:header", 20, "S", header)];

  it("inserts a suggestion at its place among the suggested steps, manual steps kept", () => {
    const manual = newStep("rename_column", "s3", "city"); // the id of a suggestion not yet ticked
    const draft = [all[0]!.step!, all[1]!.step!, manual];
    const next = withSuggestion(draft, all[2]!, all);
    expect(next.map((step) => step.suggestion ?? step.op)).toEqual(["B-15:empty", "B-15:header", "B-01:city", "rename_column"]);
    expect(new Set(next.map((step) => step.id)).size).toBe(4);
    expect(withSuggestion(next, all[2]!, all)).toBe(next); // already there
    expect(withoutSuggestion(next, "B-15:empty").map((step) => step.id)).not.toContain("s1");
  });

  it("numbers, moves and describes steps", () => {
    expect(nextId([empty, spaces, header])).toBe("s4");
    expect(move([empty, spaces], "s2", -1).map((step) => step.id)).toEqual(["s2", "s1"]);
    expect(move([empty, spaces], "s1", -1).map((step) => step.id)).toEqual(["s1", "s2"]);
    expect(describeStep(spaces)).toBe("Trim extra spaces · city");
    expect(describeStep(newStep("rename_column", "s9", "city"))).toBe("Rename city → city 2");
  });

  it("undoes and redoes the draft's changes (C-11)", () => {
    let history = historyOf([empty]);
    history = push(history, [empty, spaces]);
    history = push(history, [spaces]);
    expect(undo(history).present).toEqual([empty, spaces]);
    expect(undo(undo(history)).present).toEqual([empty]);
    expect(redo(undo(history)).present).toEqual([spaces]);
    expect(undo(historyOf([empty]))).toEqual(historyOf([empty]));
    expect(push(undo(history), [header]).future).toEqual([]);
  });
});

describe("labels of pinned rows as a column (099)", () => {
  it("adds one label_rows step at the end, under a free name, then updates it", () => {
    const first = newStep("rename_column", "s1", "city");
    const added = withPinnedLabels([first], { "3": "big" }, ["city", "pin_label"]);
    expect(added).toHaveLength(2);
    const step = added[1] as Extract<Step, { op: "label_rows" }>;
    expect(step).toMatchObject({ id: "s2", op: "label_rows", to: "pin_label_2", labels: { "3": "big" }, other: null });
    expect(describeStep(step)).toBe("Labels of pinned rows → pin_label_2 · 1 rows");

    const updated = withPinnedLabels(added, { "3": "big", "8": "test" }, ["city"]);
    expect(updated).toHaveLength(2);
    expect((updated[1] as typeof step).labels).toEqual({ "3": "big", "8": "test" });
  });
});
