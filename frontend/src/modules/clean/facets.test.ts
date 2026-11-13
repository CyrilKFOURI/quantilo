import { describe, expect, it } from "vitest";

import type { Step } from "./api";
import { actionsOf, inRange, NOTHING, rangeOf, rowsOf, stepOf, toggled, type Facet } from "./facets";

const facet = (extra: Partial<Facet> = {}): Facet =>
  ({
    column: "content_rating",
    kind: "text",
    dtype: "VARCHAR",
    rows: 100,
    distinct: 3,
    blank: 4,
    unreadable: null,
    unreadable_values: [],
    reading: null,
    values: [
      { value: "Not Rated", rows: 30 },
      { value: "Unrated", rows: 62 },
      { value: "PG", rows: 4 },
    ],
    listed_from: 3,
    low: null,
    high: null,
    bins: [],
    matched: null,
    shown: [],
    sample: [],
    ...extra,
  }) as Facet;

const draft = [{ id: "s3" }] as unknown as Step[];

describe("facets (N40)", () => {
  it("offers the gestures a selection allows", () => {
    expect(actionsOf(NOTHING, "text")).toEqual([]);
    expect(actionsOf({ ...NOTHING, values: ["Unrated"] }, "text")).toEqual(["merge", "missing", "remove", "keep"]);
    expect(actionsOf({ ...NOTHING, blank: true }, "text")).toEqual(["remove", "keep"]);
    expect(actionsOf({ ...NOTHING, low: 1, high: 2 }, "number")).toEqual(["remove", "keep", "outside"]);
    expect(actionsOf({ ...NOTHING, low: 1, high: 2 }, "date")).toEqual(["remove", "keep"]);
  });

  it("writes Unrated merged into Not Rated as a map_values step added by hand", () => {
    const selection = toggled(NOTHING, "Unrated");
    const step = stepOf("merge", { ...selection, values: ["Unrated", "Not Rated"] }, facet(), draft, "Not Rated");
    expect(step).toMatchObject({ id: "s4", op: "map_values", column: "content_rating", map: { Unrated: "Not Rated" }, origin: "manual", suggestion: null });
    expect(rowsOf(selection, facet())).toBe(62);
  });

  it("reads values and unreadable cells as missing, and filters the rows of any of them", () => {
    const withUnreadable = facet({ unreadable: 5, unreadable_values: [{ value: "12 kg", rows: 5 }] });
    const selection = { ...NOTHING, values: ["PG"], unreadable: true };
    expect(stepOf("missing", selection, withUnreadable, draft)).toMatchObject({ op: "null_values", values: ["PG", "12 kg"], match: "exact" });
    expect(stepOf("remove", { ...selection, blank: true }, withUnreadable, draft)).toMatchObject({
      op: "filter_rows",
      combine: "any",
      action: "remove",
      conditions: [
        { column: "content_rating", op: "in", values: ["PG", "12 kg"] },
        { column: "content_rating", op: "empty" },
      ],
    });
  });

  it("turns a range of bins into bounds, written as days for a date", () => {
    const bins = [
      { low: 0, high: 10, rows: 1 },
      { low: 10, high: 20, rows: 2 },
      { low: 20, high: 30, rows: 3 },
    ];
    const range = rangeOf(bins, 2, 1);
    expect(range).toMatchObject({ low: 10, high: 30 });
    expect(bins.map((bin) => inRange(bin, range))).toEqual([false, true, true]);
    const number = facet({ kind: "number", dtype: "DOUBLE" });
    expect(stepOf("keep", range, number, draft)).toMatchObject({ op: "filter_rows", combine: "all", action: "keep", conditions: [{ op: "ge", value: "10" }, { op: "le", value: "30" }] });
    expect(stepOf("outside", range, number, draft)).toMatchObject({ op: "null_outside", min: 10, max: 30 });
    const day = facet({ kind: "date", dtype: "DATE" });
    expect(stepOf("remove", { ...NOTHING, low: 86_400 * 3, high: null }, day, draft)).toMatchObject({ conditions: [{ op: "ge", value: "1970-01-04" }] });
  });
});
