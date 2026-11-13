import { describe, expect, it } from "vitest";

import type { Column, KpiResult, KpiSeries } from "../../api/client";
import { columnsFor, defaultMeasure, keyColumns, levelsOf, rankedDates, weightColumns } from "./definition";
import { effectiveMode, rollingMean, rollingWindow, shownValues } from "./pivot";

const column = (name: string, type: string, role: string | null = null, levels: string[] = []): Column =>
  ({
    name,
    type,
    role,
    excluded: false,
    summary: { histogram: levels.map((label) => ({ label, count: 1 })) },
  }) as unknown as Column;

const columns = [
  column("Price", "numeric_continuous", "unit_price"),
  column("Quantity", "numeric_discrete", "quantity"),
  column("Revenue", "numeric_continuous", "amount"),
  column("Status", "categorical_nominal", "status", ["paid", "cancelled"]),
  column("Order ID", "categorical_nominal", "order_key"),
  column("Churn", "boolean"),
];

describe("the Builder's date list (048 bis, R-SUG-17)", () => {
  it("follows the server's order and keeps unranked dates last", () => {
    const dates = ["birth_date", "Ship Date", "Order Date", "derived"].map((n) => column(n, "datetime"));
    const order = [{ name: "Order Date" }, { name: "Ship Date" }, { name: "birth_date" }];
    expect(rankedDates(dates, order).map((c) => c.name)).toEqual(["Order Date", "Ship Date", "birth_date", "derived"]);
    expect(rankedDates(dates, undefined)).toBe(dates);
  });
});

describe("business measures (053)", () => {
  it("weights a unit price by the quantity, never a flow (R-MES-01)", () => {
    const price = columns[0] as Column;
    expect(defaultMeasure(price, columns)).toEqual({ agg: "weighted_mean", weight: "Quantity" });
    expect(defaultMeasure(price, columns.filter((c) => c.name !== "Quantity"))).toEqual({ agg: "mean" });
    expect(defaultMeasure(columns[2] as Column, columns)).toEqual({ agg: "sum" });
  });

  it("offers categories to a share, quantities first as weights, keys by role first", () => {
    expect(columnsFor("share", columns).map((c) => c.name)).toEqual(["Status", "Order ID", "Churn"]);
    expect(weightColumns(columns, "Price")[0]?.name).toBe("Quantity");
    expect(keyColumns(columns)[0]?.name).toBe("Order ID");
    expect(levelsOf(columns[3])).toEqual(["paid", "cancelled"]);
  });
});

const periods = (n: number, partialLast = false) =>
  Array.from({ length: n }, (_, i) => ({ start: `2025-${String(i + 1).padStart(2, "0")}-01`, label: `M${i + 1}`, partial: partialLast && i === n - 1 }));

function result(values: (number | null)[], grain: string, partialLast = false, trend: (number | null)[] | null = null): { result: KpiResult; line: KpiSeries } {
  const line = { measure: "m1", g1: null, g2: null, values, previous: values.map(() => null), last_year: values.map(() => null), rows: values.map(() => 1), overall: 0, trend } as KpiSeries;
  const decision = { value: grain, rule: "R", reason: "", overridable: true, options: [] };
  const out = {
    periods: periods(values.length, partialLast),
    series: [line],
    resolved: { grain: decision, layout: decision, shading: decision, time: true, why: [], trend: null },
  } as unknown as KpiResult;
  return { result: out, line };
}

describe("smoothed Show as (R-SHOW-01, R-SHOW-02)", () => {
  it("rolls a trailing mean over the window of the grain", () => {
    const { result: monthly, line } = result([3, 6, 9, 12, 15], "month");
    expect(rollingWindow(monthly)).toBe(3);
    expect(shownValues(monthly, line, "rolling")).toEqual([null, null, 6, 9, 12]);
    expect(rollingWindow(result([1], "day").result)).toBe(7);
    expect(rollingWindow(result([1], "week").result)).toBe(4);
  });

  it("leaves an incomplete period and its share of a window out", () => {
    expect(rollingMean([3, 6, 9, 30], [false, false, false, true], 3)).toEqual([null, null, 6, null]);
    expect(rollingMean([3, null, 9, 12], [false, false, false, false], 3)).toEqual([null, null, 6, 10.5]);
  });

  it("reads the trend the engine computed, empty before it comes", () => {
    const { result: withTrend, line } = result([1, 2, 3, 4], "month", false, [1.1, 2, 2.9, 4]);
    expect(shownValues(withTrend, line, "trend")).toEqual([1.1, 2, 2.9, 4]);
    const { result: before, line: plain } = result([1, 2], "month");
    expect(shownValues(before, plain, "trend")).toEqual([null, null]);
  });

  it("needs a time axis, but not an additive measure", () => {
    expect(effectiveMode("rolling", false, true)).toBe("rolling");
    expect(effectiveMode("trend", false, false)).toBe("value");
  });
});
