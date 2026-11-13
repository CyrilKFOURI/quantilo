import { describe, expect, it } from "vitest";

import type { KpiResult, KpiSeries } from "../../api/client";
import { dataPart, sameData, withFormat, withLabel, withView } from "./definition";
import { formatChange, formatShown, formatValue } from "./format";
import { buildTable, effectiveMode, shownValues, toDelimited, type TableOptions } from "./pivot";
import { byScope, signalMarks } from "./signals";
import type { ChartTokens } from "../../charts/theme";

const decision = (value: string, options: { value: string; allowed: boolean; reason: string | null }[] = []) => ({
  value,
  rule: "R",
  reason: "because",
  overridable: true,
  options,
});
const format = { decimals: 0, scale: 1, prefix: "", suffix: "", percent: false, rule: "R-FMT-01", reason: "Whole numbers." };

function series(g1: string | null, values: number[], previous: (number | null)[], overall: number): KpiSeries {
  return { measure: "m1", g1, g2: null, values, previous, last_year: values.map(() => null), rows: values.map(() => 1), overall, trend: null };
}

/** Revenue by month by country: A and B, three months, the last one incomplete. */
const result: KpiResult = {
  name: "Total revenue by month by country",
  rows: 30,
  periods: [
    { start: "2025-01-01", label: "Jan 2025", partial: false },
    { start: "2025-02-01", label: "Feb 2025", partial: false },
    { start: "2025-03-01", label: "Mar 2025", partial: true },
  ],
  splits: [{ column: "country", levels: ["A", "B"], others: 0, ordinal: false }],
  measures: [
    {
      id: "m1",
      label: "Total revenue",
      agg: "sum",
      column: "revenue",
      additive: true,
      approximate: false,
      format,
      chart: decision("stacked_bars"),
      share: { value: "share", allowed: true, reason: null },
    },
  ],
  series: [
    series(null, [100, 150, 50], [null, 100, 150], 300),
    series("A", [60, 90, 20], [null, 60, 90], 170),
    series("B", [40, 60, 30], [null, 40, 60], 130),
  ],
  tiles: [],
  resolved: {
    grain: decision("month"),
    layout: decision("periods_columns", [
      { value: "periods_rows", allowed: true, reason: null },
      { value: "periods_columns", allowed: true, reason: null },
    ]),
    shading: decision("on"),
    time: true,
    why: [],
    trend: null,
  },
  warnings: [],
  signals: [],
  warning_details: [],
};

const options = (patch: Partial<TableOptions> = {}): TableOptions => ({
  layout: "periods_columns",
  showAs: "value",
  totals: true,
  shading: false,
  compare: [],
  sort: null,
  hideOther: false,
  labels: {},
  ...patch,
});

describe("show as", () => {
  it("turns values into shares, changes and running totals", () => {
    const a = result.series[1] as KpiSeries;
    expect(shownValues(result, a, "share_total")).toEqual([0.6, 0.6, 0.4]);
    expect(shownValues(result, a, "change")).toEqual([null, 30, -70]);
    expect(shownValues(result, a, "change_pct")).toEqual([null, 0.5, -70 / 90]);
    expect(shownValues(result, a, "running")).toEqual([60, 150, 170]);
  });

  it("falls back to the value where a mode means nothing (R-TAB-11)", () => {
    expect(effectiveMode("share_total", false, true)).toBe("value");
    expect(effectiveMode("running", true, false)).toBe("value");
    expect(effectiveMode("change_pct", false, true)).toBe("change_pct");
  });
});

describe("pivot table", () => {
  it("lays periods across with a total column and a total row", () => {
    const table = buildTable(result, options());
    expect(table.head[0]?.map((h) => h.label)).toEqual(["Jan 2025", "Feb 2025", "Mar 2025 †", "Total"]);
    expect(table.rows.map((r) => r.label)).toEqual(["A", "B", "Total"]);
    expect(table.rows[0]?.cells.map((c) => c.value)).toEqual([60, 90, 20, 170]);
    expect(table.rows[2]?.cells.map((c) => c.value)).toEqual([100, 150, 50, 300]);
    expect(table.rows[0]?.cells[2]?.partial).toBe(true);
  });

  it("swaps to periods in rows", () => {
    const table = buildTable(result, options({ layout: "periods_rows" }));
    expect(table.head.at(-1)?.map((h) => h.label)).toEqual(["A", "B", "Total"]);
    expect(table.rows.map((r) => r.label)).toEqual(["Jan 2025", "Feb 2025", "Mar 2025 †", "Total"]);
    expect(table.rows[1]?.cells.map((c) => c.value)).toEqual([90, 60, 150]);
  });

  it("divides by the row or the column total", () => {
    const row = buildTable(result, options({ showAs: "share_row" }));
    expect(row.rows[0]?.cells[0]?.value).toBeCloseTo(60 / 170);
    const column = buildTable(result, options({ showAs: "share_column" }));
    expect(column.rows[0]?.cells[0]?.value).toBeCloseTo(0.6);
  });

  it("adds compare columns for the last complete period", () => {
    const table = buildTable(result, options({ compare: ["previous"] }));
    expect(table.head[0]?.at(-1)?.label).toBe("Δ vs previous (Feb 2025)");
    expect(table.rows[0]?.cells.at(-1)?.value).toBe(30);
  });

  it("sorts, shades and exports", () => {
    const sorted = buildTable(result, options({ layout: "periods_rows", sort: "0:asc", shading: true }));
    expect(sorted.rows.map((r) => r.label)).toEqual(["Mar 2025 †", "Jan 2025", "Feb 2025", "Total"]);
    expect(sorted.rows.find((r) => r.label === "Feb 2025")?.cells[0]?.shade).toBe(1);
    expect(sorted.rows.at(-1)?.cells[0]?.shade).toBeNull(); // totals are not shaded
    const csv = toDelimited(buildTable(result, options()), ",");
    expect(csv.split("\n")[0]).toBe("country,Jan 2025,Feb 2025,Mar 2025 †,Total");
    expect(csv.split("\n")[1]).toBe("A,60,90,20,170");
  });
});

describe("format", () => {
  it("applies the automatic format and the user's overrides", () => {
    expect(formatValue(1234567, format)).toBe("1,234,567");
    expect(formatValue(1234567, format, {}, "compact")).toBe("1.2M");
    expect(formatValue(12.5, format, { decimals: 2, prefix: "€" })).toBe("€12.50");
    expect(formatValue(-3, format)).toBe("−3");
    expect(formatValue(0.253, { ...format, decimals: 1, scale: 100, suffix: " %" })).toBe("25.3 %");
    expect(formatChange(0.042)).toBe("+4.2 %");
    expect(formatShown(0.34, "share_total", format)).toBe("34.0 %");
    expect(formatShown(30, "change", format)).toBe("+30");
  });
});

describe("definition", () => {
  it("tells data changes from display changes", () => {
    const base = { measures: [{ id: "m1", agg: "sum" as const, column: "revenue" }] };
    expect(sameData(base, withView(base, { chart: "line" }))).toBe(true);
    expect(sameData(base, withLabel(base, "m1", "Revenue"))).toBe(true);
    expect(sameData(base, withFormat(base, "m1", { decimals: 2 }))).toBe(true);
    expect(sameData(base, { ...base, other: "hide" })).toBe(true);
    expect(sameData(base, { ...base, top: 3 })).toBe(false);
    expect(dataPart(withView(base, { share: true }))).toBe(dataPart(base));
  });
});

describe("signals (054)", () => {
  const flagged: KpiResult = {
    ...result,
    signals: [
      {
        kind: "low_n",
        severity: "warning",
        scope: "cells",
        rule: "R-SIG-03",
        measure: "m1",
        message: "1 cell of Total revenue rests on fewer than 30 values: read them with care (R-SIG-03).",
        cells: [{ period: "2025-02-01", g1: "B", g2: null, value: 12 }],
        score: 1,
      },
      {
        kind: "thin_period",
        severity: "warning",
        scope: "cells",
        rule: "R-SIG-04",
        measure: null,
        message: "Jan 2025 has under 20 % of the median period's rows.",
        cells: [{ period: "2025-01-01", g1: null, g2: null, value: 3 }],
        score: 0.5,
      },
      {
        kind: "anomaly",
        severity: "info",
        scope: "series",
        rule: "R-SIG-01",
        measure: "m1",
        message: "Feb 2025 unusually high (robust z 4.1, R-SIG-01)",
        cells: [{ period: "2025-02-01", g1: null, g2: null, value: 4.1 }],
        score: 4.1,
      },
    ],
  };

  it("marks the cells a signal names, with the cell's n, and every cell of a thin period", () => {
    const table = buildTable(flagged, options({ layout: "periods_rows" }));
    const [jan, feb] = table.rows;
    // columns: A, B, Total
    expect(feb?.cells[1]?.flag?.message).toBe("Low n: 12 values, under 30 (R-SIG-03)");
    expect(feb?.cells[0]?.flag).toBeNull();
    expect(feb?.cells[2]?.flag?.kind).toBe("anomaly");
    expect(jan?.cells.every((c) => c.flag?.kind === "thin_period")).toBe(true);
    expect(byScope(flagged, "series")).toHaveLength(1);
  });

  it("puts a marker over an unusual period of the measure's chart only", () => {
    const tokens = { ink2: "#555", ink3: "#999" } as unknown as ChartTokens;
    const marks = signalMarks(flagged, "m1", tokens);
    expect(marks?.annotations?.[0]).toMatchObject({ x: 1, text: "▲", hovertext: expect.stringContaining("unusually high") });
    expect(signalMarks(flagged, "m2", tokens)).toBeNull();
  });
});
