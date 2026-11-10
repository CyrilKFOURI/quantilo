import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

afterEach(() => vi.unstubAllGlobals());

const columns = [
  column({ name: "price", position: 0 }),
  column({ name: "area", position: 1, role: null }),
  column({ name: "city", position: 2, type: "categorical_nominal", detected_type: "categorical_nominal", role: null }),
];

const term = (overrides: Record<string, unknown>) => ({
  term: "area",
  column: "area",
  level: null,
  estimate: 2980.5,
  std_error: 40.2,
  ci_low: 2901.7,
  ci_high: 3059.3,
  statistic: 74.1,
  p_value: 1e-40,
  q_value: 2e-40,
  evidence: "strong",
  standardized: 0.91,
  std_low: 0.89,
  std_high: 0.93,
  odds_ratio: null,
  marginal: null,
  marginal_low: null,
  marginal_high: null,
  sentence: "Each extra unit of `area` adds 2,980 to `price` [2,900 ; 3,060], other columns held fixed.",
  ...overrides,
});

const regression = {
  target: "price",
  positive: null,
  model: "ols",
  model_label: "Least squares (OLS)",
  model_reason: "A number to explain: least squares, in the units of the data.",
  columns: ["area", "city"],
  focus: null,
  terms: [
    term({ term: "(Intercept)", column: "(Intercept)", estimate: 50000, evidence: "none", q_value: null, standardized: null, sentence: "With every column at 0…" }),
    term({}),
    term({ term: "city = Lyon", column: "city", level: "Lyon", estimate: -60000, evidence: "moderate", standardized: -0.2, sentence: "With `city` = Lyon, `price` is 60,000 lower." }),
  ],
  se: {
    kind: "hc3",
    label: "HC3",
    rule: "R-REG-SE-05",
    reason: "HC3 always: choosing the standard errors from a pre-test distorts the intervals.",
    lag: null,
    groups: [],
    clusters: null,
    warning: null,
  },
  weights: null,
  quality: {
    rows: 1200,
    loaded: 1200,
    total: 1200,
    sampled: false,
    missing: [],
    r2: 0.97,
    adj_r2: 0.969,
    oos_r2: 0.968,
    oos_how: "5 random folds",
    aic: 28000,
    bic: 28020,
    f: 3000,
    f_p: 0,
    df_model: 2,
    df_resid: 1197,
    pseudo_r2: null,
  },
  multiple: {
    method: "holm",
    label: "Holm",
    tests: 2,
    significant: 2,
    kept: 2,
    alpha: 0.05,
    auto: true,
    reason: "2 tests: Holm (automatic below 10 tests).",
    sentence: "All 2 significant effects stay significant after correcting for 2 tests (Holm).",
  },
  sentence: "Each extra unit of `area` adds 2,980 to `price` [2,900 ; 3,060], other columns held fixed. All 2 significant effects stay significant after correcting for 2 tests (Holm).",
  checks: [
    { key: "spread", status: "ok", title: "Spread of the residuals", sentence: "The spread changes (Breusch–Pagan p < 0.001). The HC3 standard errors already allow for it.", rule: "R-REG-12", action: null, column: null },
    { key: "specification", status: "warn", title: "Straight lines", sentence: "The effects may not be straight lines: `area` bends the most.", rule: "R-REG-10", action: "compare", column: "area" },
    { key: "influence", status: "warn", title: "Influential rows", sentence: "38 rows (3.2%) weigh heavily on the coefficients.", rule: "R-REG-11", action: "influence", column: null },
  ],
  references: [{ column: "city", level: "Paris", levels: 2, others: 0, reason: "most frequent level (50% of the rows)" }],
  left_out: [],
  structure: { kind: "none", date: null, entity: null, period: null, periods: 0, entities: 0, message: null },
  hints: [],
  plots: {
    fitted: [1, 2],
    residuals: [0.1, -0.1],
    rows: [0, 1],
    leverage: [0.01, 0.02],
    cook: [0.001, 0.01],
    cook_threshold: 0.0033,
    qq_theoretical: [-1, 1],
    qq_sample: [-1.1, 0.9],
    points: 2,
    of: 2,
  },
};

const suggestion = {
  target: { column: "price", reason: "R-REG-20: a numeric column with the role amount." },
  columns: [
    { column: "area", reason: "R-REG-21: ρ = 0.95 with `price`, strong." },
    { column: "city", reason: "R-REG-21: η = 0.40 with `price`, strong." },
  ],
  candidates: ["price", "area"],
  regressors: ["area", "city"],
};

const influence = {
  terms: ["(Intercept)", "area", "city = Lyon"],
  rows: [{ row: 41, leverage: 0.02, cook: 0.05, studentized: 3.2, bonferroni: 0.4, dfbetas: [0.1, -0.3, 0.05] }],
  influential: 38,
  outliers: 0,
  cook_threshold: 0.0033,
  sentence: "38 rows of 1,200 have a Cook's distance above 4/n.",
};

const standardErrors = {
  terms: ["(Intercept)", "area", "city = Lyon"],
  estimates: [50000, 2980.5, -60000],
  kinds: [
    { kind: "classic", label: "Classic", available: true, reason: "assumes equal variances", chosen: false, std_errors: [900, 30, 3000], p_values: [0, 0, 0] },
    { kind: "hc3", label: "HC3", available: true, reason: "unequal variances", chosen: true, std_errors: [1000, 40.2, 3500], p_values: [0, 0, 0] },
    { kind: "hac", label: "HAC (Newey–West)", available: false, reason: "needs rows in time order: one row per date", chosen: false, std_errors: [], p_values: [] },
  ],
  sentence: "HC3 standard errors are 1.30 × the classic ones.",
};

const more = {
  white: { name: "White", statistic: 40, p_value: 0.001, df: 5 },
  white_sentence: "With every square and cross-product, the spread changes (White p = 0.001).",
  removed: 38,
  shifts: [
    { term: "(Intercept)", estimate: 50000, without: 49000, change: -0.02 },
    { term: "area", estimate: 2980.5, without: 3010, change: 0.01 },
  ],
  shift_sentence: "Without the 38 influential rows, no coefficient moves by more than 20 %.",
};

function routes() {
  return {
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": columns,
    "GET /column-roles": roles,
    "GET /datasets/d1/regression/suggestion": suggestion,
    "GET /datasets/d1/regression": regression,
    "GET /datasets/d1/regression/influence": influence,
    "GET /datasets/d1/regression/standard-errors": standardErrors,
    "GET /datasets/d1/regression/more": more,
  };
}

describe("Regression tab (068)", () => {
  it("starts from the suggestion and shows the first plane, nothing of Advanced", async () => {
    const calls = mockApi(routes());
    renderAt("/d/d1/regression");

    expect(await screen.findByText(/All 2 significant effects stay significant/)).toBeInTheDocument();
    const fit = calls.find((call) => call.key === "GET /datasets/d1/regression");
    expect(fit?.params.get("target")).toBe("price");
    expect(fit?.params.get("columns")).toBe("area,city");

    const table = screen.getByRole("table");
    expect(within(table).getByText("city = Lyon")).toBeInTheDocument();
    expect(within(table).getAllByText("Strong").length).toBeGreaterThan(0);
    expect(screen.getByText("Out-of-sample R²")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Residuals against fitted values" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "See the curve →" })).toHaveAttribute("href", expect.stringContaining("/d/d1/compare"));

    const tabs = screen.getByRole("navigation", { name: "Sections of this page" });
    expect(within(tabs).getByRole("link", { name: "Regression" })).toHaveAttribute("aria-current", "page");
    // the onion rule: no section is asked for before it is opened
    expect(calls.some((call) => /regression\/(standard-errors|influence|terms|more)/.test(call.key))).toBe(false);
  });

  it("computes each Advanced section when it is opened, More last", async () => {
    const calls = mockApi(routes());
    renderAt("/d/d1/regression?target=price&columns=area%2Ccity");
    await screen.findByText(/All 2 significant effects/);
    expect(calls.some((call) => call.key === "GET /datasets/d1/regression/suggestion")).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: /^Advanced/ }));
    expect(calls.some((call) => /regression\/(standard-errors|influence|terms|more)/.test(call.key))).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: /^Standard errors/ }));
    expect(await screen.findByText("HC3 standard errors are 1.30 × the classic ones.")).toBeInTheDocument();
    expect(screen.getByText(/needs rows in time order/)).toBeInTheDocument();
    expect(calls.some((call) => call.key === "GET /datasets/d1/regression/more")).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: /^More/ }));
    expect(await screen.findByText(/no coefficient moves by more than 20 %/)).toBeInTheDocument();
  });

  it("opens the section a check points to", async () => {
    const calls = mockApi(routes());
    renderAt("/d/d1/regression?target=price&columns=area%2Ccity");
    await screen.findByText(/All 2 significant effects/);
    window.HTMLElement.prototype.scrollIntoView = vi.fn();

    fireEvent.click(screen.getByRole("button", { name: "Try…" }));
    expect(await screen.findByText(/38 rows of 1,200 have a Cook's distance/)).toBeInTheDocument();
    await waitFor(() => expect(calls.some((call) => call.key === "GET /datasets/d1/regression/influence")).toBe(true));
  });

  it("offers the target's other links as controls after Two variables", async () => {
    const calls = mockApi(routes());
    renderAt("/d/d1/regression?target=price&columns=area&focus=area");
    expect(await screen.findByText(/as controls/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Add controls" }));
    await waitFor(() =>
      expect(calls.some((call) => call.key === "GET /datasets/d1/regression" && call.params.get("columns") === "area,city")).toBe(true),
    );
  });
});
