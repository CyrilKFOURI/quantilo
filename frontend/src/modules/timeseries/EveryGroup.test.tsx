import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { GroupForecast } from "../../api/client";
import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { groupMethods, groupVerdict } from "./EveryGroup";
import { changeableDrivers } from "./Scenarios";
import { parseScenarios } from "./search";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

const periods = Array.from({ length: 12 }, (_, i) => `2024-${String(i + 1).padStart(2, "0")}`);
const future = ["2025-01", "2025-02", "2025-03"];
const values = periods.map((_, i) => 100 + 10 * i);
const setup = {
  time: "day",
  value: "units",
  group: "store",
  series: "S01",
  dates: ["day"],
  numbers: ["units", "price"],
  categories: ["store", "region"],
  groups: ["S01", "S02"],
  period: "month",
  aggregate: "sum",
  period_auto: true,
  aggregate_auto: true,
  coarsened: false,
  first_partial: false,
  last_partial: false,
  gaps: "leave",
  season: 12,
  periods: 12,
  rows: 360,
  total: 360,
};
const overview = {
  setup,
  periods,
  analyzed: { label: "S01", values, counts: periods.map(() => 30), rolling_mean: [], rolling_std: [] },
  lines: [],
  figures: { start: periods[0], end: periods[11], periods: 12, missing: 0, change: 10 },
  window: 3,
  align: "trailing",
  methods: [],
  message: null,
};
const groups = {
  setup,
  group: "store",
  groups: 2,
  limit: 200,
  max_limit: 1000,
  summable: true,
  candidates: [
    { column: "region", values: 1, parent: false },
    { column: "country", values: 1, parent: false },
  ],
  levels: [],
  browser: false,
};
const row = (label: string, level: string, parent: string | null, best: string, volume: number) => ({
  label,
  level,
  parent,
  volume,
  demand: null,
  best,
  info: null,
  mase: 0.8,
  wape: 12.5,
  total: 300,
  change: -4.2,
  short_history: label === "S02",
  other: false,
  history: values,
  forecast: [100, 100, 100],
  low80: [90, 90, 90],
  high80: [110, 110, 110],
  low95: [80, 80, 80],
  high95: [120, 120, 120],
  base: [101, 99, 100],
  global_mase: null,
  error: null,
});
const result: GroupForecast = {
  group: "store",
  horizon: 3,
  periods,
  future,
  models: ["seasonal_naive", "ets_auto", "combination"],
  rows: [
    row("S01", "store", "Total", "ets_auto", 900),
    row("S02", "store", "Total", "seasonal_naive", 300),
    row("Total", "Total", null, "ets_auto", 1200),
  ],
  groups: 2,
  kept: 2,
  pooled: 0,
  limit: 200,
  short: 1,
  best_models: [
    { model: "ets_auto", series: 1, share: 50 },
    { model: "seasonal_naive", series: 1, share: 50 },
  ],
  global_model: { library: "lightgbm", info: "Global LightGBM", wins: 1, series: 2 },
  hierarchy: {
    levels: ["Total", "store"],
    method: "mint_shrink",
    method_auto: true,
    gap: 1e-12,
    errors: [
      { level: "Total", nodes: 1, mase_before: 0.9, mase_after: 0.7 },
      { level: "store", nodes: 2, mase_before: 1.1, mase_after: 1.0 },
    ],
    shrinkage: 0.4,
    note: null,
  },
  notes: [],
  workers: 4,
  seconds: 2,
};
const job = { id: "g1", status: "done", done: 3, total: 3, stage: "reconcile", current: null, result, error: null };
const model = (key: string) => ({
  key,
  info: null,
  scores: { mae: 1, rmse: 1, mape: 1, smape: 1, wape: 1, mase: 0.7, rmsse: 1, bias: 0, coverage80: 0.8, coverage95: 0.9, winkler95: 1 },
  vs_reference: 10,
  forecast: [1, 2, 3],
  low80: [0, 1, 2],
  high80: [2, 3, 4],
  low95: [0, 1, 2],
  high95: [2, 3, 4],
  residual_check: null,
  seconds: 0.1,
  error: null,
  backtests: [],
  uses_drivers: key === "tslm",
  mase_without_drivers: null,
  drivers: [],
});
const forecastJob = {
  id: "job1",
  status: "done",
  done: 2,
  total: 2,
  current: null,
  error: null,
  result: {
    horizon: 3,
    periods,
    history: values,
    future,
    reference: "seasonal_naive",
    origins: 3,
    results: [model("tslm"), model("seasonal_naive")],
    best: "tslm",
    demand: null,
    drivers: [],
    driver_notes: [],
  },
};
const drivers = {
  setup,
  horizon: 3,
  candidates: [{ column: "price", flag: false, correlation: 0.1, suggested: false, future_periods: 3 }],
  calendar: [],
};
const scenario = (forecast: number[]) => ({
  model: "tslm",
  info: null,
  future,
  forecast,
  low80: forecast,
  high80: forecast,
  low95: forecast,
  high95: forecast,
  drivers: [],
  notes: [],
});

function setupApi() {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [
      column({ name: "day", type: "datetime", detected_type: "datetime", role: null }),
      column({ name: "units", position: 1, role: "quantity" }),
    ],
    "GET /column-roles": roles,
    "GET /modules/timeseries/datasets/d1/overview": overview,
    "GET /modules/timeseries/datasets/d1/groups": groups,
    "GET /modules/timeseries/datasets/d1/drivers": drivers,
    "POST /modules/timeseries/datasets/d1/forecast-groups": { ...job, status: "running", done: 0, result: null },
    "GET /modules/timeseries/group-forecast-jobs/g1": job,
    "POST /modules/timeseries/datasets/d1/forecast": { ...forecastJob, status: "running", result: null },
    "GET /modules/timeseries/forecast-jobs/job1": forecastJob,
    "POST /modules/timeseries/datasets/d1/scenario": (body: unknown) =>
      (body as { changes: unknown[] }).changes.length ? scenario([110, 120, 130]) : scenario([100, 100, 100]),
  });
}

beforeEach(() => {
  localStorage.setItem("de.ts.open", JSON.stringify(["trend", "seasonality", "decomposition", "stationarity", "autocorrelation", "volatility", "residuals", "breaks", "forecast"]));
});

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("Every group (039)", () => {
  it("leaves the slow ARIMA family out unless asked for", () => {
    expect(groupMethods(["naive", "arima_auto", "sarimax", "ets_auto"], false)).toEqual(["naive", "ets_auto"]);
    expect(groupMethods(["naive"], true)).toEqual(["naive", "arima_auto"]);
  });

  it("says which model wins most often, and for how many series the global one does", () => {
    expect(groupVerdict(result)).toBe(
      "ETS (automatic) is best for 50 % of store values. The global model wins for 1 of 2.",
    );
  });

  it("forecasts every group, then shows the table, the small multiples and the levels", async () => {
    const calls = setupApi();
    renderAt("/d/d1/timeseries?time=day&value=units&group=store&series=S01");
    const section = await screen.findByRole("region", { name: "Forecast" });
    expect(await within(section).findByText("2 store values")).toBeInTheDocument();
    expect(within(section).getByText("Levels")).toBeInTheDocument(); // auto reconciliation

    fireEvent.click(within(section).getByRole("button", { name: "Forecast every group" }));

    expect(
      await within(section).findByText(/ETS \(automatic\) is best for 50 % of store values/),
    ).toBeInTheDocument();
    const sent = calls.find((c) => c.key.endsWith("/forecast-groups"));
    expect(sent?.params.get("reconcile")).toBe("auto");
    expect(sent?.params.get("group")).toBe("store");
    expect(sent?.params.get("methods")).not.toContain("arima_auto");
    expect(within(section).getByText("Totals reconciled: MinT (shrunk covariance)")).toBeInTheDocument();
    expect(within(section).getAllByText("short history").length).toBe(1);
    expect(within(section).getByRole("img", { name: "S01" })).toBeInTheDocument();
    expect(within(section).getAllByRole("cell", { name: "−4.2 %" }).length).toBe(3);

    expect(within(section).getByRole("cell", { name: "S01" })).toBeInTheDocument();
    fireEvent.change(within(section).getByLabelText("Search a series"), { target: { value: "S02" } });
    expect(within(section).queryByRole("cell", { name: "S01" })).not.toBeInTheDocument();

    fireEvent.click(within(section).getByRole("button", { name: "S01" }));
    await waitFor(() =>
      expect(calls.some((c) => c.key.endsWith("/overview") && c.params.get("series") === "S01")).toBe(true),
    );
  });

  it("keeps scenarios of a known-in-advance driver in the URL and compares them with the base", async () => {
    const calls = setupApi();
    renderAt("/d/d1/timeseries?time=day&value=units&group=store");
    const section = await screen.findByRole("region", { name: "Forecast" });
    fireEvent.click(within(section).getByRole("button", { name: /Drivers/ }));
    fireEvent.change(await within(section).findByLabelText("Add a column…"), { target: { value: "price" } });
    fireEvent.click(within(section).getByRole("button", { name: "Run forecast" }));

    expect(await within(section).findByText("Scenarios")).toBeInTheDocument();
    fireEvent.click(within(section).getByRole("button", { name: "Add scenario" }));

    expect(await within(section).findByRole("cell", { name: "Scenario 1" })).toBeInTheDocument();
    expect(within(section).getByRole("cell", { name: "+60" })).toBeInTheDocument();
    expect(within(section).getByRole("cell", { name: "+20.0 %" })).toBeInTheDocument();
    const sent = calls.filter((c) => c.key.endsWith("/scenario")).at(-1);
    expect(sent?.body).toMatchObject({
      changes: [{ column: "price", how: "percent", amount: -10, start: null, end: null }],
    });
    // the scenarios live in the URL only (no local state): the row is drawn from it
    expect(within(section).getByLabelText("Scenario")).toHaveValue("");
  });

  it("reads scenarios back from the URL and only changes known drivers", () => {
    expect(parseScenarios([{ name: "Cut", column: "price", how: "percent", amount: "-10", start: "2025-12" }])).toEqual([
      { name: "Cut", column: "price", how: "percent", amount: -10, start: "2025-12", end: undefined },
    ]);
    expect(parseScenarios("nope")).toBeUndefined();
    expect(parseScenarios([{ column: 3 }, {}, {}, {}])).toBeUndefined();
    expect(
      changeableDrivers([
        { column: "price", kind: "column", known: true, future: "file", value: null },
        { column: "visits", kind: "column", known: false, future: "file", value: null },
        { column: "month", kind: "calendar", known: true, future: "file", value: null },
        { column: "holiday", kind: "flag", known: true, future: "file", value: null },
      ]).map((d) => d.column),
    ).toEqual(["price", "holiday"]);
  });
});
