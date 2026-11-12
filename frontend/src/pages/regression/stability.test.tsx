import { fireEvent, screen } from "@testing-library/react";
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
  column({ name: "Ret", position: 0 }),
  column({ name: "Mkt", position: 1, role: null }),
  column({ name: "Date", position: 2, type: "datetime", detected_type: "datetime", role: null }),
];

const coefficient = (term: string, estimate: number) => ({
  term,
  column: term,
  level: null,
  estimate,
  std_error: 0.1,
  ci_low: estimate - 0.2,
  ci_high: estimate + 0.2,
  statistic: 20,
  p_value: 1e-6,
  q_value: term === "(Intercept)" ? null : 1e-6,
  evidence: term === "(Intercept)" ? "none" : "strong",
  standardized: null,
  std_low: null,
  std_high: null,
  odds_ratio: null,
  marginal: null,
  marginal_low: null,
  marginal_high: null,
  sentence: `Each extra unit of \`${term}\` adds ${estimate}.`,
});

const regression = {
  target: "Ret",
  positive: null,
  model: "ols",
  model_label: "Least squares (OLS)",
  model_reason: "A number to explain.",
  columns: ["Mkt"],
  focus: null,
  terms: [coefficient("(Intercept)", 0.002), coefficient("Mkt", 1.1)],
  se: { kind: "hac", label: "HAC (Newey–West)", rule: "R-REG-SE-02", reason: "one row per date.", lag: 4, groups: [], clusters: null, warning: null },
  weights: null,
  quality: { rows: 240, loaded: 240, total: 240, sampled: false, missing: [], r2: 0.8, adj_r2: 0.8, oos_r2: 0.79, oos_how: "5 folds in time", aic: 1, bic: 1, f: 10, f_p: 0, df_model: 1, df_resid: 238, pseudo_r2: null },
  multiple: { method: "none", label: "None", tests: 1, significant: 1, kept: 1, alpha: 0.05, auto: true, reason: "", sentence: "One test." },
  sentence: "Each extra unit of `Mkt` adds 1.1.",
  checks: [
    {
      key: "stability",
      status: "warn",
      title: "Stability over time",
      sentence: "The coefficients before and after 2011-01-31 differ (Chow p < 0.001).",
      rule: "R-REG-15",
      action: "stability",
      column: null,
    },
  ],
  references: [],
  left_out: [],
  structure: { kind: "series", date: "Date", entity: null, period: null, periods: 240, entities: 0, message: "One row per `Date`." },
  hints: [],
  plots: null,
};

const path = { dates: ["2001-01-31", "2020-12-31"], values: [0, 3], lower: [-1, -2], upper: [1, 2], crossed: true, first: "2015-01-31" };
const stability = {
  available: true,
  reason: "",
  date: "Date",
  rows: 240,
  cusum: path,
  cusum_squares: { ...path, crossed: false, first: null },
  cusum_p: 0.2,
  chow: { at: "2011-01-31", index: 120, f: 50, p_value: 1e-9, before: [0, 0.8], after: [0, 1.5], searched: true, sentence: "Before and after 2011-01-31, the coefficients differ (Chow F test, p < 0.001)." },
  window: 48,
  window_reason: "R-REG-STAB-02: windows of 48 rows.",
  rolling: [{ term: "Mkt", dates: ["2004-12-31", "2020-12-31"], estimates: [0.8, 1.5], low: [0.7, 1.4], high: [0.9, 1.6], full: 1.1, smallest: 0.8, largest: 1.5, sentence: "Over windows of 48 rows, the effect of `Mkt` goes from 0.8 to 1.5." }],
  rolled: 1,
  factor: null,
  stable: false,
  sentence: "The CUSUM of the recursive residuals leaves its 5 % band at 2015-01-31 (R-REG-15).",
  notes: [],
};

const resampling = {
  available: true,
  reason: "",
  scheme: "block",
  scheme_reason: "R-REG-RS-01: blocks of 6 consecutive rows (n^⅓) are drawn.",
  draws: 999,
  rows: 240,
  resampled: 240,
  terms: [
    { term: "(Intercept)", estimate: 0.002, low: 0.001, high: 0.003, p_value: 0.01, boot_se: 0.0005, boot_low: 0.001, boot_high: 0.003, wild_p: null },
    { term: "Mkt", estimate: 1.1, low: 1, high: 1.2, p_value: 1e-6, boot_se: 0.05, boot_low: 0.98, boot_high: 1.22, wild_p: null },
  ],
  wild: false,
  wild_reason: "The wild cluster bootstrap is for clustered standard errors.",
  weights: null,
  clusters: null,
  permutation: null,
  permutation_reason: "Permutation assumes rows that can be exchanged: not with clusters or dates.",
  sentence: "999 bootstrap draws agree with the first plane (R-REG-RS-05).",
  notes: [],
};

function routes() {
  return {
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": columns,
    "GET /column-roles": roles,
    "GET /datasets/d1/regression": regression,
    "GET /datasets/d1/regression/stability": stability,
    "GET /datasets/d1/regression/resampling": resampling,
  };
}

describe("Regression: stability over time (073) and resampling (077)", () => {
  it("opens Stability over time from the R-REG-15 check, asked for only then", async () => {
    const calls = mockApi(routes());
    renderAt("/d/d1/regression?target=Ret&columns=Mkt");
    await screen.findByText(/before and after 2011-01-31 differ/);
    expect(calls.some((call) => call.key === "GET /datasets/d1/regression/stability")).toBe(false);
    window.HTMLElement.prototype.scrollIntoView = vi.fn();

    fireEvent.click(screen.getByRole("button", { name: "Try…" }));
    expect(await screen.findByText(/leaves its 5 % band at 2015-01-31/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Rolling effect of Mkt" })).toBeInTheDocument();
    expect(screen.getByText(/goes from 0.8 to 1.5/)).toBeInTheDocument();
    expect(screen.getByText(/R-REG-STAB-02/)).toBeInTheDocument();
    expect(calls.some((call) => call.key === "GET /datasets/d1/regression/resampling")).toBe(false);
  });

  it("shows bootstrap intervals beside the formula ones when Resampling opens", async () => {
    const calls = mockApi(routes());
    renderAt("/d/d1/regression?target=Ret&columns=Mkt");
    await screen.findByText(/before and after 2011-01-31 differ/);
    fireEvent.click(screen.getByRole("button", { name: /^Advanced/ }));
    fireEvent.click(screen.getByRole("button", { name: /^Resampling/ }));
    expect(await screen.findByText(/bootstrap draws agree/)).toBeInTheDocument();
    expect(calls.some((call) => call.key === "GET /datasets/d1/regression/resampling")).toBe(true);
    expect(screen.getByRole("columnheader", { name: "Bootstrap 95 % CI" })).toBeInTheDocument();
    expect(screen.getByText(/not with clusters or dates/)).toBeInTheDocument();
  });
});
