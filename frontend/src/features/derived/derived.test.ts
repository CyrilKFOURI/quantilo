import { describe, expect, it } from "vitest";

import type { Column } from "../../api/client";
import { derivedNote, isDerived, withDerived } from "./derived";

const column = (name: string, position: number, extra: Partial<Column> = {}): Column =>
  ({
    name,
    position,
    storage_dtype: "VARCHAR",
    type: "text",
    role: null,
    role_label: null,
    parser: { kind: "text" },
    order: null,
    detected_type: "text",
    detected_role: null,
    confidence: "high",
    reasons: [],
    overridden: false,
    excluded: false,
    summary: null,
    ...extra,
  }) as Column;

const derived = (source: string, part: string, position: number): Column =>
  column(`${source} › ${part}`, position, {
    type: "numeric_discrete",
    parser: {
      kind: "derived",
      decimal: ".",
      formats: [],
      true_values: [],
      false_values: [],
      derived: { source, part: part.toLowerCase(), option: null, topics: [] },
    },
    reasons: [`derived from '${source}': number of characters (R-DER-01)`],
  });

describe("withDerived", () => {
  it("lists each derived column right after its source (R-DER-08)", () => {
    const columns = [column("review", 0), column("rating", 1), column("title", 2)];
    const found = withDerived(columns, [derived("title", "Length", 2), derived("review", "Length", 0), derived("review", "Words", 0)]);
    expect(found.map((c) => c.name)).toEqual([
      "review",
      "review › Length",
      "review › Words",
      "rating",
      "title",
      "title › Length",
    ]);
  });

  it("keeps the columns as they are without derived columns", () => {
    const columns = [column("a", 0)];
    expect(withDerived(columns, undefined)).toBe(columns);
  });
});

describe("derivedNote", () => {
  it("says where the column comes from, with its rule, and asks to check", () => {
    const note = derivedNote(derived("review", "Length", 0));
    expect(note).toBe("Derived from 'review': number of characters (R-DER-01). Check that your data was read correctly.");
    expect(derivedNote(column("review", 0))).toBeNull();
    expect(isDerived(derived("review", "Length", 0))).toBe(true);
  });
});
