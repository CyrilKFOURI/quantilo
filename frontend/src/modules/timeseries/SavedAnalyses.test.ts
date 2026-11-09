import { describe, expect, it } from "vitest";

import type { SeriesSetup } from "../../api/client";
import type { SavedAnalysis } from "../../api/recipes";
import { configOf, searchOf } from "./SavedAnalyses";

const setup = {
  time: "date",
  value: "montant",
  group: null,
  series: null,
  period: "month",
  aggregate: "sum",
} as unknown as SeriesSetup;

describe("saved analyses", () => {
  it("keep the resolved columns and the choices the user made", () => {
    const config = configOf({ period: "month", gaps: "zero" }, setup, ["naive", "ets_auto"], 3);

    expect(config).toMatchObject({
      time: "date",
      value: "montant",
      period: "month",
      aggregate: "auto", // left automatic on the page: stays automatic on a new file
      gaps: "zero",
      models: ["naive", "ets_auto"],
      horizon: 3,
    });
  });

  it("keep the forecast drivers, and none when there are none", () => {
    const price = { column: "price", kind: "column", known: true, future: "file", value: null } as const;

    expect(configOf({}, setup, ["sarimax"], 3, [price]).drivers).toEqual([price]);
    expect(configOf({}, setup, ["naive"], 3).drivers).toBeNull();
  });

  it("reopen the page in the same setup", () => {
    const config = configOf({ period: "month", season: 12 }, setup, ["naive"], null);

    expect(searchOf(config as SavedAnalysis["config"])).toEqual({
      time: "date",
      value: "montant",
      group: undefined,
      series: undefined,
      period: "month",
      agg: undefined,
      window: undefined,
      align: undefined,
      gaps: undefined,
      season: 12,
    });
  });
});
