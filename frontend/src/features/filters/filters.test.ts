import { describe as group, expect, it } from "vitest";

import type { Chart } from "../../api/client";
import { validateFilterSearch } from "../../api/filters";
import { choiceOf, choicesFor, choicesWith, daysBefore, describe, pastedValues, upgraded } from "./conditions";
import { actionsFor } from "./fromChart";

const click = (index: number, customdata: unknown = null) => ({ index, x: null, y: null, customdata, clientX: 0, clientY: 0 });

function chart(overrides: Partial<Chart>): Chart {
  return {
    kind: "bars",
    view: "bars",
    views: ["bars"],
    rows: 100,
    missing: 0,
    distinct: 3,
    bars: [],
    others: null,
    ordered: false,
    bins: [],
    stats: null,
    box: null,
    step: null,
    steps: [],
    examples: [],
    duplicates: 0,
    message: null,
    ...overrides,
  } as Chart;
}

group("filter conditions", () => {
  it("describes conditions in short readable form", () => {
    expect(describe({ column: "country", op: "in", values: ["FR", "BE"], negate: false })).toBe("country ∈ FR, BE");
    expect(describe({ column: "country", op: "in", values: ["FR"], negate: true })).toBe("country ∉ FR");
    expect(describe({ column: "age", op: "gte", value: 30, negate: false })).toBe("age ≥ 30");
    expect(describe({ column: "age", op: "between", low: 1, high: 9, negate: true })).toBe("age outside [1 – 9]");
    expect(describe({ column: "day", op: "part_in", part: "day_of_week", values: ["1", "7"], negate: false })).toBe(
      "day on Mon, Sun",
    );
    expect(describe({ column: "a", op: "in", values: ["1", "2", "3", "4", "5"], negate: false })).toBe("a ∈ 1, 2, 3 +2");
  });

  it("offers the choices that fit the column type", () => {
    const ids = (type: Parameters<typeof choicesFor>[0]) => choicesFor(type).map((choice) => choice.id);

    // R-FLT-01 to 06: one set per type
    expect(ids("categorical_nominal").slice(0, 4)).toEqual(["is", "is_not", "in", "not_in"]);
    expect(ids("categorical_nominal")).not.toContain("eq");
    expect(ids("categorical_nominal")).not.toContain("contains");
    expect(ids("boolean")[0]).toBe("is");
    expect(ids("numeric_continuous")).not.toContain("in");
    expect(ids("numeric_discrete")[0]).toBe("in");
    expect(ids("text").slice(0, 3)).toEqual(["contains", "not_contains", "starts_with"]);
    expect(ids("text")).not.toContain("eq");
    expect(ids("identifier").slice(0, 3)).toEqual(["eq", "in", "not_in"]);
    expect(choicesFor("identifier").find((c) => c.id === "in")?.input).toBe("paste");
    expect(ids("datetime")).toEqual(expect.arrayContaining(["lt", "gt", "between", "last_days", "part_in", "is_null"]));
    expect(ids("ignored")).toEqual([]);
    expect(choiceOf({ column: "a", op: "between", negate: true }, "numeric_continuous")?.id).toBe("outside");
  });

  it("reads older conditions without changing their rows (R-FLT-07)", () => {
    const eq = { column: "device", op: "eq" as const, value: "mobile", negate: false };
    expect(upgraded(eq, "categorical_nominal")).toEqual({ column: "device", op: "in", values: ["mobile"], negate: false });
    expect(upgraded(eq, "identifier")).toBe(eq);
    expect(choiceOf(eq, "categorical_nominal")?.id).toBe("is");
    expect(choiceOf({ column: "d", op: "in", values: ["a", "b"], negate: true }, "categorical_nominal")?.id).toBe("not_in");
    expect(choiceOf({ column: "d", op: "in", values: ["a"], negate: true }, "categorical_nominal")?.id).toBe("is_not");

    const contains = { column: "device", op: "contains" as const, value: "mob", negate: false };
    expect(choiceOf(contains, "categorical_nominal")).toMatchObject({ legacy: true, label: "contains (older condition)", input: "value" });
    expect(choicesWith("categorical_nominal", contains)[0]?.legacy).toBe(true);
    expect(choicesWith("categorical_nominal", undefined).some((c) => c.legacy)).toBe(false);
    expect(choiceOf({ column: "c", op: "in", values: ["x"], negate: false }, "text")?.input).toBe("paste");
    expect(choiceOf({ column: "c", op: "eq", value: "x", negate: true }, "text")?.label).toBe("doesn't equal (older condition)");
  });

  it("reads pasted lists and counts days back", () => {
    expect(pastedValues("A1\r\nA2, A3;\tA1\n\n")).toEqual(["A1", "A2", "A3"]);
    expect(daysBefore("2024-03-01", 1)).toBe("2024-02-29");
    expect(daysBefore("2024-12-30", 6)).toBe("2024-12-24");
    expect(describe({ column: "date", op: "last_days", value: 30, high: "2024-12-30", negate: false })).toBe(
      "date · last 30 days (to 2024-12-30)",
    );
    expect(describe({ column: "c", op: "is_null", negate: true })).toBe("c is not missing");
  });

  it("keeps only well-formed filters from the URL", () => {
    expect(validateFilterSearch({})).toEqual({});
    expect(validateFilterSearch({ filters: "junk" })).toEqual({});
    const parsed = validateFilterSearch({
      filters: { dataset: "d1", conditions: [{ column: "a", op: "is_null" }, { column: "b", op: "drop" }] },
    });
    expect(parsed.filters?.conditions).toEqual([{ column: "a", op: "is_null", negate: false }]);
  });
});

group("filtering from a chart", () => {
  it("keeps or excludes a clicked category", () => {
    const actions = actionsFor("country", chart({ bars: [{ label: "FR", count: 5 }] }), click(0, ["FR", "5%"]));

    expect(actions.map((action) => action.condition)).toEqual([
      { column: "country", op: "in", values: ["FR"], negate: false },
      { column: "country", op: "in", values: ["FR"], negate: true },
    ]);
  });

  it("filters empty values from the (missing) bar and ignores Others", () => {
    const missing = actionsFor("c", chart({}), click(0, ["(missing)", "1%"]));
    const others = actionsFor("c", chart({}), click(0, ["Others (4)", "9%"]));

    expect(missing[0]?.condition).toEqual({ column: "c", op: "is_null", negate: false });
    expect(others).toEqual([]);
  });

  it("keeps a histogram bin's range", () => {
    const histogram = chart({ kind: "histogram", view: "histogram", bins: [{ start: 0, end: 10, count: 3 }] });

    expect(actionsFor("price", histogram, click(0))[0]?.condition).toEqual({
      column: "price",
      op: "between",
      low: 0,
      high: 10,
      negate: false,
    });
  });

  it("turns a weekday bar into a date part filter", () => {
    const weekdays = chart({ view: "day_of_week", ordered: true, bars: [{ label: "Mon", count: 2 }] });

    expect(actionsFor("signup", weekdays, click(0, ["Mon", "2%"]))[0]?.condition).toEqual({
      column: "signup",
      op: "part_in",
      part: "day_of_week",
      values: ["1"],
      negate: false,
    });
  });
});

group("unique values and duplicates", () => {
  it("are offered for every column type but ignored", () => {
    expect(choicesFor("identifier").map((c) => c.id)).toEqual(expect.arrayContaining(["unique", "duplicated", "not_duplicated"]));
    expect(choicesFor("numeric_continuous").some((c) => c.op === "unique")).toBe(true);
    expect(choicesFor("ignored")).toEqual([]);
  });

  it("read as short chips", () => {
    expect(describe({ column: "vehicle_id", op: "unique", negate: false, keep: "last", order_by: "date" })).toBe(
      "vehicle_id · unique (last by date)",
    );
    expect(describe({ column: "vehicle_id", op: "unique", negate: false, columns: ["month"] })).toBe(
      "vehicle_id + month · unique (first)",
    );
    expect(describe({ column: "email", op: "duplicated", negate: false })).toBe("email · duplicates only");
    expect(describe({ column: "email", op: "duplicated", negate: true })).toBe("email · appearing once");
  });

  it("survive the URL", () => {
    const conditions = [{ column: "vehicle_id", op: "unique", keep: "last" }];
    expect(validateFilterSearch({ filters: { dataset: "d1", conditions } }).filters?.conditions[0]).toMatchObject({
      op: "unique",
      keep: "last",
    });
  });
});
