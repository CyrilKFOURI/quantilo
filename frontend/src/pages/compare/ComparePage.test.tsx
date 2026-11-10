import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { BivariateTest } from "../../api/client";
import { verdict } from "../../features/compare/TestsPanel";
import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { experimentOf } from "./ComparePage";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot(
    { label },
    ref,
  ) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

const columns = [
  column({ name: "age", position: 0 }),
  column({ name: "income", position: 1 }),
  column({
    name: "segment",
    position: 2,
    type: "categorical_nominal",
    detected_type: "categorical_nominal",
  }),
  column({ name: "comment", position: 3, type: "text", detected_type: "text" }),
  column({
    name: "day",
    position: 4,
    type: "datetime",
    detected_type: "datetime",
  }),
];

const test = (overrides: Partial<BivariateTest>): BivariateTest => ({
  key: "pearson",
  statistic: 0.5,
  p_value: 0.001,
  n: 600,
  effect: { name: "r", value: 0.5, magnitude: "large" },
  recommended: false,
  note: null,
  ...overrides,
});

const scatter = {
  kind: "scatter",
  x: "age",
  y: "income",
  z: null,
  color: null,
  roles: { age: "num", income: "num" },
  rows: 551,
  total: 600,
  sample: null,
  group_axis: null,
  points: { x: [1, 2, 3], y: [2, 4, 6], z: null, color: null },
  regression: { slope: 2, intercept: 0, r2: 0.25 },
  density: null,
  groups: [],
  contingency: null,
  series: [],
  step: null,
  tests: [
    test({}),
    test({
      key: "spearman",
      p_value: 0.2,
      effect: { name: "rho", value: 0.05, magnitude: "negligible" },
    }),
  ],
  message: null,
  intervals: null,
  differences: [],
  multiple: null,
};

const periods = Array.from(
  { length: 10 },
  (_, i) => `2024-01-${String(i + 1).padStart(2, "0")}`,
);
const timeline = {
  ...scatter,
  kind: "timeline",
  x: "day",
  y: "income",
  roles: { day: "date", income: "num" },
  points: null,
  regression: null,
  step: "day",
  series: [
    {
      label: null,
      periods,
      values: periods.map((_, i) => 100 + 10 * i),
      rolling_mean: periods.map((_, i) => (i < 2 ? null : 90 + 10 * i)),
      rolling_std: periods.map((_, i) => (i < 2 ? null : 5)),
      counts: periods.map(() => 3),
    },
  ],
  time: {
    period: "day",
    aggregate: "sum",
    window: 4,
    align: "trailing",
    periods: 10,
    period_auto: true,
    aggregate_auto: true,
    coarsened: false,
  },
  trend: { slope: 10, slope_low: 9, slope_high: 11, per_year: 2.5, lag1: 0.2 },
  tests: [
    test({
      key: "mann_kendall",
      statistic: 3.9,
      p_value: 0.0001,
      n: 10,
      effect: { name: "tau", value: 1, magnitude: "large" },
      recommended: true,
    }),
  ],
};

function setup(result: object = scatter) {
  return mockApi({
    "GET /health": health,
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": columns,
    "GET /column-roles": roles,
    "GET /datasets/d1/compare": result,
  });
}

afterEach(() => vi.unstubAllGlobals());

const summaryWith = (distinct: number) => ({ ...column().summary, distinct_count: distinct });

const coloredGroups = {
  ...scatter,
  kind: "groups",
  x: "pclass",
  y: "age",
  color: "segment",
  roles: { pclass: "cat", age: "num", segment: "cat" },
  group_axis: "x",
  points: null,
  regression: null,
  groups: ["1", "2"].flatMap((label) =>
    ["low", "high"].map((color) => ({ label, color, n: 40, mean: 30, std: 5, q1: 25, median: 30, q3: 35, low: 20, high: 40, ci_low: 28, ci_high: 32, sample: [] })),
  ),
  color_levels: ["low", "high"],
  categories: [{ column: "pclass", values: 3 }],
  message: "Tests compare the pclass groups, all segment values together.",
};

const groupsWithDifferences = {
  ...coloredGroups,
  color: null,
  color_levels: [],
  groups: ["1", "2", "3"].map((label) => ({
    label,
    color: null,
    n: label === "3" ? 12 : 40,
    mean: 30,
    std: 5,
    q1: 25,
    median: 30,
    q3: 35,
    low: 20,
    high: 40,
    ci_low: 28,
    ci_high: 32,
    sample: [],
    ci_method: label === "3" ? "bootstrap" : "t",
    ci_reason: label === "3" ? "12 values: too few for the t interval" : "40 values, skewness within ±1",
    median_low: 29,
    median_high: 31,
    median_reason: "bootstrap: a median has no simple formula",
  })),
  intervals: "Mean intervals: bootstrap for 1 of 3 groups (fewer than 30 values or skewness beyond ±1), Student t for the others.",
  differences: [
    { a: "1", b: "2", n_a: 40, n_b: 40, diff: 1.5, low: 0.2, high: 2.8, interval: "welch", reason: "Welch interval", p_value: 0.03, p_adjusted: 0.09 },
    { a: "1", b: "3", n_a: 40, n_b: 12, diff: 4, low: 2, high: 6, interval: "bootstrap", reason: "bootstrap, 3: 12 values", p_value: 0.0001, p_adjusted: 0.0003 },
    { a: "2", b: "3", n_a: 40, n_b: 12, diff: 0.5, low: -1, high: 2, interval: "bootstrap", reason: "bootstrap, 3: 12 values", p_value: 0.6, p_adjusted: 0.6 },
  ],
  multiple: {
    method: "holm",
    label: "Holm",
    tests: 3,
    significant: 2,
    kept: 1,
    alpha: 0.05,
    auto: true,
    reason: "3 tests: Holm (automatic below 10 tests).",
    sentence: "1 of the 2 significant differences stays significant after correcting for 3 tests (Holm).",
  },
};

describe("Compare page", () => {
  it("lists each group's intervals with their method, and the pairwise differences after correction", async () => {
    mockApi({
      "GET /health": health,
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [
        column({ name: "pclass", position: 0, type: "numeric_discrete", detected_type: "numeric_discrete", summary: summaryWith(3) }),
        column({ name: "age", position: 1 }),
      ],
      "GET /column-roles": roles,
      "GET /datasets/d1/compare": groupsWithDifferences,
    });
    renderAt("/d/d1/compare?x=pclass&y=age");

    expect(await screen.findByText(/bootstrap for 1 of 3 groups/)).toBeInTheDocument();
    expect(screen.getByText("12 values: too few for the t interval")).toBeInTheDocument();
    expect(screen.getByText("Pairwise differences (3)")).toBeInTheDocument();
    expect(screen.getByText(/1 of the 2 significant differences stays significant/)).toBeInTheDocument();
    // 2 − 1: p = 0.03 alone, 0.09 once the three pairs are counted
    expect(screen.getAllByText("no longer holds")).toHaveLength(1);
  });

  it("reads a whole number with few values as categories, and colours its groups", async () => {
    const calls = mockApi({
      "GET /health": health,
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [
        column({ name: "pclass", position: 0, type: "numeric_discrete", detected_type: "numeric_discrete", summary: summaryWith(3) }),
        column({ name: "age", position: 1 }),
        column({ name: "segment", position: 2, type: "categorical_nominal", detected_type: "categorical_nominal", summary: summaryWith(2) }),
        column({ name: "city", position: 3, type: "categorical_nominal", detected_type: "categorical_nominal", summary: summaryWith(9) }),
      ],
      "GET /column-roles": roles,
      "GET /datasets/d1/compare": coloredGroups,
    });
    renderAt("/d/d1/compare?x=pclass&y=age&color=segment");

    expect(await screen.findByText(/3 values, read as categories/)).toBeInTheDocument();
    const colorPicker = screen.getByRole("combobox", { name: "Color" });
    expect(colorPicker).toBeEnabled();
    // any colour works: the server lays the categories out within 15 boxes
    expect(within(colorPicker).getByRole("option", { name: "city" })).toBeEnabled();
    await waitFor(() => {
      const request = calls.find((c) => c.key === "GET /datasets/d1/compare");
      expect(request?.params.get("color")).toBe("segment");
      expect(request?.params.get("numbers")).toBeNull();
    });
    fireEvent.click(screen.getByRole("button", { name: "As numbers" }));
    await waitFor(() => {
      const last = calls.filter((c) => c.key === "GET /datasets/d1/compare").at(-1);
      expect(last?.params.get("numbers")).toBe("pclass");
    });
  });

  it("asks for two columns first and hides non-comparable ones", async () => {
    setup();
    renderAt("/d/d1/compare");

    expect(
      await screen.findByText("Pick an X and a Y column to compare them."),
    ).toBeInTheDocument();
    const x = screen.getByLabelText("X");
    expect(
      within(x).queryByRole("option", { name: "comment" }),
    ).not.toBeInTheDocument();
    expect(
      within(x).getByRole("option", { name: "segment" }),
    ).toBeInTheDocument();
  });

  it("shows the chart and the tests for the chosen pair", async () => {
    const calls = setup();
    renderAt("/d/d1/compare?x=age&y=income");

    expect(
      await screen.findByRole("img", { name: "income against age" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("551 of 600 rows have every value"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Significant correlation · large effect"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("No evidence of a correlation"),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/Linear fit \(R² = 0.250\)/)).toBeChecked();
    const call = calls.find((c) => c.key === "GET /datasets/d1/compare");
    expect(call?.params.get("x")).toBe("age");
    expect(call?.params.get("y")).toBe("income");
  });

  it("draws the LOWESS curve by the rule and says why (064)", async () => {
    const reason = "LOWESS explains 31 % of Y's variance, the line 12 %: the relation bends.";
    setup({
      ...scatter,
      lowess: { x: [1, 2, 3], y: [2, 3, 6], r2: 0.31, line_r2: 0.12, suggested: true, reason },
    });
    renderAt("/d/d1/compare?x=age&y=income");

    const box = await screen.findByLabelText("LOWESS curve");
    expect(box).toBeChecked();
    expect(screen.getByText(reason)).toBeInTheDocument();
    fireEvent.click(box);
    expect(box).not.toBeChecked();
  });

  it("swaps X and Y in the URL", async () => {
    setup();
    const router = renderAt("/d/d1/compare?x=age&y=income");
    await screen.findByRole("img", { name: "income against age" });

    fireEvent.click(screen.getByRole("button", { name: "Swap X and Y" }));

    await waitFor(() =>
      expect(router.state.location.search).toMatchObject({
        x: "income",
        y: "age",
      }),
    );
  });

  it("turns a date and a number into a time series with a trend", async () => {
    setup(timeline);
    renderAt("/d/d1/compare?x=day&y=income");

    expect(
      await screen.findByText("Daily sum of income · 4-day rolling mean ± 1σ"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("income rises by about 10 per day."),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "About +250.0 % per year relative to its average level.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Significant trend · large effect"),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("± 1σ band")).toBeChecked();
    expect(screen.getByText("Data used (10 periods)")).toBeInTheDocument();
  });

  it("opens the date and the number in Time series, prefilled (098)", async () => {
    setup(timeline);
    renderAt("/d/d1/compare?x=day&y=income&period=week");

    const link = await screen.findByRole("link", { name: "Open in Time series →" });
    const href = decodeURIComponent(link.getAttribute("href") ?? "");
    expect(href).toContain("/d/d1/timeseries?");
    expect(href).toContain("time=day");
    expect(href).toContain("value=income");
    expect(href).toContain("period=week");
  });

  it("keeps time series choices in the URL and sends them", async () => {
    const calls = setup(timeline);
    const router = renderAt(
      "/d/d1/compare?x=day&y=income&period=week&agg=mean&window=6",
    );
    await screen.findByText("Daily sum of income · 4-day rolling mean ± 1σ");

    const call = calls.find((c) => c.key === "GET /datasets/d1/compare");
    expect(call?.params.get("period")).toBe("week");
    expect(call?.params.get("agg")).toBe("mean");
    expect(call?.params.get("window")).toBe("6");

    fireEvent.click(screen.getByLabelText("± 1σ band"));
    await waitFor(() =>
      expect(router.state.location.search).toMatchObject({
        show: "values,mean",
      }),
    );
  });

  it("hides Z and Color when the pair cannot use them, and says why (R-CMP-10/11)", async () => {
    setup();
    renderAt("/d/d1/compare?x=age&y=segment");

    expect(await screen.findByLabelText("Color")).toBeEnabled();
    expect(screen.queryByLabelText("Z (3D)")).toBeNull();
    expect(screen.getByRole("tooltip")).toHaveTextContent("Z (3D) appears when X and Y are both numbers");
    expect(screen.getByRole("tooltip")).not.toHaveTextContent("Color appears");
  });

  it("hides Z for two numbers when no third number is left", async () => {
    setup();
    renderAt("/d/d1/compare?x=age&y=income");

    expect(await screen.findByLabelText("Color")).toBeInTheDocument();
    expect(screen.queryByLabelText("Z (3D)")).toBeNull();
  });

  it("shows Z for two numbers with a third one to pick", async () => {
    mockApi({
      "GET /health": health,
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [...columns, column({ name: "spend", position: 5 })],
      "GET /column-roles": roles,
      "GET /datasets/d1/compare": scatter,
    });
    renderAt("/d/d1/compare?x=age&y=income");

    const z = await screen.findByLabelText("Z (3D)");
    expect(within(z).getAllByRole("option").map((o) => o.textContent)).toEqual(["None", "spend"]);
    expect(screen.queryByRole("tooltip")).toBeNull();
  });

  it("hides both for two categories", async () => {
    mockApi({
      "GET /health": health,
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [
        ...columns,
        column({ name: "device", position: 5, type: "categorical_nominal", detected_type: "categorical_nominal" }),
      ],
      "GET /column-roles": roles,
      "GET /datasets/d1/compare": scatter,
    });
    renderAt("/d/d1/compare?x=segment&y=device");

    expect(await screen.findByRole("tooltip")).toHaveTextContent("Color appears");
    expect(screen.queryByLabelText("Z (3D)")).toBeNull();
    expect(screen.queryByLabelText("Color")).toBeNull();
  });
});

describe("test verdicts", () => {
  it("separates significance from importance", () => {
    expect(
      verdict(
        test({ effect: { name: "r", value: 0.02, magnitude: "negligible" } }),
      ).text,
    ).toBe("Significant correlation, but the effect is negligible");
    expect(verdict(test({ key: "anova", p_value: 0.3 })).text).toBe(
      "No evidence that the groups differ",
    );
    expect(
      verdict(
        test({
          key: "chi2",
          effect: { name: "V", value: 0.2, magnitude: "medium" },
        }),
      ).text,
    ).toBe("The variables are associated · medium effect");
    expect(verdict(test({ p_value: null, note: "Too few rows." })).text).toBe(
      "Not run: Too few rows.",
    );
  });
});

describe("Analyze as an experiment (034)", () => {
  const levels = (name: string, type: string, distinct: number) =>
    column({ name, type, detected_type: type, summary: { rows: 100, null_count: 0, invalid_count: 0, distinct_count: distinct, examples: [], histogram: [] } }) as never;

  const groups = {
    ...scatter,
    kind: "groups",
    x: "arm",
    y: "income",
    roles: { arm: "cat", income: "num" },
    points: null,
    regression: null,
    groups: [],
  };
  const candidatesWith = (planned: boolean) => ({
    variants: [{ column: "arm", levels: ["A", "B"], counts: [50, 50], control: "A", planned, why: planned ? "Its name says a variant (arm)." : "Describes the units (region): an observed trait." }],
    metrics: [{ column: "income", kind: "mean" }],
    units: [],
    times: [],
    segments: [],
    suggested: null,
    note: null,
  });
  const routes = (planned: boolean) => ({
    "GET /health": health,
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [levels("arm", "categorical_nominal", 2), column({ name: "income", position: 1 })],
    "GET /column-roles": roles,
    "GET /datasets/d1/compare": groups,
    "GET /modules/experiments/datasets/d1/candidates": candidatesWith(planned),
  });

  it("offers Experiments only on a variant by R-EXP-01, with the rule's reason (098)", async () => {
    mockApi(routes(true));
    renderAt("/d/d1/compare?x=arm&y=income");

    const link = await screen.findByRole("link", { name: "Open in Experiments →" });
    expect(decodeURIComponent(link.getAttribute("href") ?? "")).toContain("/d/d1/experiments?variant=arm&metric=income");
    expect(screen.getByRole("button", { name: "Why it looks like an experiment" })).toBeInTheDocument();
  });

  it("says nothing of an experiment on an observed category (098)", async () => {
    const calls = mockApi(routes(false));
    renderAt("/d/d1/compare?x=arm&y=income");

    await waitFor(() => expect(calls.some((c) => c.key === "GET /modules/experiments/datasets/d1/candidates")).toBe(true));
    await screen.findByRole("img");
    expect(screen.queryByRole("link", { name: "Open in Experiments →" })).not.toBeInTheDocument();
  });

  it("links a column of 2 to 5 values with a number or a yes/no", () => {
    const group = levels("group", "categorical_nominal", 3);
    const converted = levels("converted", "boolean", 2);
    const revenue = levels("revenue", "numeric_continuous", 90);

    expect(experimentOf(group, revenue)).toEqual({ variant: "group", metric: "revenue" });
    expect(experimentOf(revenue, group)).toEqual({ variant: "group", metric: "revenue" });
    expect(experimentOf(converted, group)).toEqual({ variant: "group", metric: "converted" });
    expect(experimentOf(levels("city", "categorical_nominal", 9), revenue)).toBeNull();
    expect(experimentOf(group, levels("segment", "categorical_nominal", 3))).toBeNull();
  });
});
